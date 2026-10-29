#!/usr/bin/env python3
"""Validate generated transcripts against complete validation references."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from robohead_baseline.track1.local.finetuning.common import normalize, sha256, write_json, write_jsonl
from robohead_baseline.track1.local.finetuning.data import read_audio_record


class GenerativeValCER:
    """Evaluate the current Trainer weights with the official batched ASR interface."""

    def __init__(
        self, processor, rows, batch_size=4, num_workers=4, max_new_tokens=256, reference_file=None, tcp_collar=5.0
    ):
        if not rows or batch_size < 1 or num_workers < 0 or max_new_tokens < 1:
            raise ValueError('Invalid Val generation configuration')
        self.processor, self.rows = processor, rows
        self.batch_size, self.num_workers, self.max_new_tokens = batch_size, num_workers, max_new_tokens
        self.reference_file, self.tcp_collar = reference_file, tcp_collar

    def __call__(self, trainer):
        import torch.distributed as dist
        from qwen_asr import Qwen3ASRModel

        rank, world = trainer.args.process_index, trainer.args.world_size
        model = trainer.accelerator.unwrap_model(trainer.model)
        with _generation_mode(model):
            asr = Qwen3ASRModel(
                backend='transformers',
                model=model,
                processor=self.processor,
                max_inference_batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
            )
            predictions = predict_rows(asr, self.rows, self.batch_size, self.num_workers, rank, world)
        parts = [predictions]
        if world > 1:
            parts = [None] * world
            dist.all_gather_object(parts, predictions)
        merged = merge_predictions(parts, len(self.rows))
        if self.reference_file:
            from robohead_baseline.track1.local.finetuning.scoring import meeting_scores

            scores = meeting_scores(merged, self.reference_file, self.tcp_collar) if rank == 0 else None
            if world > 1:
                shared = [scores]
                dist.broadcast_object_list(shared, src=0)
                scores = shared[0]
            if rank == 0:
                output = Path(trainer.args.output_dir) / 'validation' / f'step{trainer.state.global_step:06d}'
                write_jsonl(output / 'predictions.jsonl', merged)
                write_json(
                    output / 'summary.json',
                    dict(
                        step=trainer.state.global_step,
                        metrics=scores,
                        reference_sha256=sha256(self.reference_file),
                        scope='Full Val predicted-GSS cpCER/tcpCER',
                    ),
                )
            return dict(eval_cpcer=scores['cpcer_pct'], eval_tcpcer=scores['tcpcer_pct'])
        groups = cer_metrics(merged)
        if groups['all']['cer_pct'] is None:
            raise ValueError('Val CER requires nonempty reference characters')
        if rank == 0:
            output = Path(trainer.args.output_dir) / 'validation' / f'step{trainer.state.global_step:06d}'
            write_jsonl(output / 'predictions.jsonl', merged)
            write_json(
                output / 'summary.json',
                dict(
                    step=trainer.state.global_step,
                    metrics=groups,
                    inference=dict(
                        batch_size=self.batch_size,
                        world_size=world,
                        max_new_tokens=self.max_new_tokens,
                        language='Chinese',
                    ),
                    scope='Full Val generated utterance CER (%), not meeting cpCER/tcpCER',
                ),
            )
        return {
            f'eval_{"" if name == "all" else name + "_"}cer': values['cer_pct']
            for name, values in groups.items()
            if values['cer_pct'] is not None
        }


def predict_rows(asr, rows, batch_size, num_workers, rank=0, world=1):
    import torch

    selected = list(enumerate(rows))[rank::world]
    predictions = []
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for start in range(0, len(selected), batch_size):
            batch = selected[start : start + batch_size]
            arrays = list(pool.map(read_audio_record, [row for _, row in batch]))
            with torch.inference_mode():
                results = asr.transcribe(
                    audio=[(samples, 16000) for samples in arrays],
                    language='Chinese',
                    context=[row.get('prompt', '') for _, row in batch],
                )
            if len(results) != len(batch):
                raise ValueError('ASR batch output count mismatch')
            for (index, row), result in zip(batch, results):
                reference = row.get('reference')
                if reference is None:
                    reference = row['text'].split('<asr_text>', 1)[-1]
                predictions.append(
                    dict(
                        index=index,
                        id=row.get('id', str(index)),
                        audio=row['audio'],
                        reference=reference,
                        prediction=result.text,
                        frontend=row.get('frontend', 'all'),
                    )
                )
                predictions[-1].update(
                    {k: row[k] for k in ('session_id', 'speaker', 'start_time', 'end_time') if k in row}
                )
            if start == 0 or len(predictions) % (batch_size * 50) == 0 or len(predictions) == len(selected):
                print(f'Transcribed {len(predictions)}/{len(selected)}', flush=True)
    return predictions


def merge_predictions(parts, expected):
    merged = sorted((row for part in parts for row in part), key=lambda row: row['index'])
    if [row['index'] for row in merged] != list(range(expected)):
        raise ValueError('Incomplete or duplicate predictions')
    return merged


def cer_metrics(rows):
    groups = {}
    for name in ['all', *sorted({row.get('frontend', 'all') for row in rows} - {'all'})]:
        selected = rows if name == 'all' else [row for row in rows if row.get('frontend') == name]
        count = sum(len(normalize(row['reference'])) for row in selected)
        errors = sum(edit_distance(normalize(row['reference']), normalize(row['prediction'])) for row in selected)
        groups[name] = dict(
            samples=len(selected),
            reference_characters=count,
            errors=errors,
            cer_pct=100 * errors / count if count else None,
        )
    return groups


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


@contextmanager
def _generation_mode(model):
    import torch

    training, saved = model.training, []
    for module in (model, getattr(model, 'thinker', None)):
        for name in ('config', 'generation_config'):
            config = getattr(module, name, None)
            if config is None:
                continue
            for key, value in [('use_cache', True), *([('do_sample', False)] if name == 'generation_config' else [])]:
                existed = hasattr(config, key)
                saved.append((config, key, existed, getattr(config, key, None)))
                setattr(config, key, value)
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        for config, key, existed, value in reversed(saved):
            if existed:
                setattr(config, key, value)
            else:
                delattr(config, key)
        model.train(training)
