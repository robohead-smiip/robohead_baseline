"""Evaluate a submitted Model locally with identical fast and real-time audio packets."""
import argparse
import csv
import importlib.util
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

from robohead_baseline.utils.audio import read_audio, write_json
from robohead_baseline.utils.protocol import load_csv, validate_snapshot

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600


def decode(model, audio, realtime):
    if audio.ndim != 2 or not len(audio):
        raise ValueError('Expected nonempty audio [samples, channels]')
    if hasattr(model, 'set_profiling'):
        model.set_profiling(not realtime)
    incoming, ready, failed = queue.Queue(), threading.Event(), threading.Event()
    sender_done = threading.Event()
    events, packets, errors, result = [], [], [], []
    timing = {}
    duration = len(audio) / SAMPLE_RATE
    accepting = False
    decoder_id = sender_id = None

    def record(text):
        stamp = time.monotonic()
        if not accepting or threading.get_ident() != decoder_id:
            raise ValueError('Results must be delivered synchronously by the decoder thread')
        if not isinstance(text, str):
            raise ValueError('Partial callback requires the current full transcript string')
        events.append(dict(event='partial_callback', time=stamp, text=text))

    def process():
        nonlocal accepting, decoder_id
        decoder_id = threading.get_ident()
        try:
            model.reset()
            model.set_partial_callback(record)
            ready.set()
            accepting = True
            while True:
                try:
                    chunk = incoming.get(timeout=0.1)
                except queue.Empty:
                    if sender_done.is_set() and incoming.empty():
                        break
                    continue
                if chunk is None:
                    break
                model.accept_chunk(chunk)
            result.append(model.input_finished())
        except BaseException as error:
            errors.append(error)
            failed.set()
        finally:
            accepting = False
            ready.set()

    def send():
        nonlocal sender_id
        sender_id = threading.get_ident()
        ready.wait()
        origin = time.monotonic()
        try:
            for first in range(0, len(audio), CHUNK_SIZE):
                end = min(first + CHUNK_SIZE, len(audio))
                failed.wait(max(0.0, origin + first / SAMPLE_RATE - time.monotonic()))
                if failed.is_set():
                    break
                chunk = audio[first:end].copy()
                stamp = time.monotonic()
                if first == 0:
                    origin = stamp
                    timing['audio_send_start_time'] = origin
                incoming.put(chunk)
                packets.append(dict(end_sample=end, sent_time=stamp))
            if not failed.is_set():
                deadline = origin + len(packets) * CHUNK_SIZE / SAMPLE_RATE
                failed.wait(max(0.0, deadline - time.monotonic()))
        except BaseException as error:
            errors.append(error)
            failed.set()
        finally:
            timing['input_finished_time'] = time.monotonic()
            incoming.put(None)
            sender_done.set()

    if realtime:
        worker = threading.Thread(target=process, name='ModelDecoder', daemon=True)
        sender = threading.Thread(target=send, name='AudioSender', daemon=True)
        worker.start()
        sender.start()
        sender.join()
        worker.join()
    else:
        sender_id = decoder_id = threading.get_ident()
        model.reset()
        model.set_partial_callback(lambda text: None)
        for first in range(0, len(audio), CHUNK_SIZE):
            chunk = audio[first : first + CHUNK_SIZE].copy()
            stamp = time.monotonic()
            if first == 0:
                timing['audio_send_start_time'] = stamp
            packets.append(dict(end_sample=first + len(chunk), sent_time=stamp))
            model.accept_chunk(chunk)
        timing['input_finished_time'] = time.monotonic()
        result.append(model.input_finished())
    visible = time.monotonic()
    if errors:
        raise errors[0]
    frozen = json.loads(json.dumps(result[0], ensure_ascii=False, allow_nan=False))
    validate_snapshot(frozen, duration)
    events.append(dict(event='final_visible', time=visible, text=frozen['text'], segments=frozen['segments']))
    timing.update(
        audio_end_oracle_time=timing['audio_send_start_time'] + duration,
        final_visible_time=visible,
        clock='evaluator_monotonic',
    )
    return (
        frozen,
        dict(
            events=events,
            timing=timing,
            capture=dict(
                realtime=realtime,
                sample_rate=SAMPLE_RATE,
                channels=audio.shape[1],
                packet_schedule='immediate_first_100ms',
                transport='in_process_queue' if realtime else 'direct',
                evaluator_pid=os.getpid(),
                model_pid=os.getpid(),
                sender_thread_id=sender_id,
                decoder_thread_id=decoder_id,
                audio_packets=packets,
            ),
        ),
    )


def run(args):
    rows = load_csv(args.manifest_csv)
    if args.config:
        os.environ['ROBOHEAD_CONFIG'] = str(args.config.resolve())
    if any(row.get('evaluation_unit') != 'session' for row in rows.values()):
        raise ValueError('This protocol requires evaluation_unit=session in the frozen manifest')
    sys.path.insert(0, str(args.submission_dir.resolve()))
    spec = importlib.util.spec_from_file_location('submission_model', args.submission_dir / 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Model()
    try:
        _run_passes(args, rows, model)
    finally:
        if hasattr(model, 'close'):
            model.close()


def _run_passes(args, rows, model):
    if args.warmup_manifest_csv:
        warmup = load_csv(args.warmup_manifest_csv)
        if set(warmup) & set(rows):
            raise ValueError('Warm-up IDs overlap evaluation IDs')
        for row in warmup.values():
            decode(model, read_audio(row, args.data_root), False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'predict.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'raw_hypos', 'segments', 'processing_time_sec'])
        writer.writeheader()
        for index, (uid, row) in enumerate(rows.items(), 1):
            final, trace = decode(model, read_audio(row, args.data_root), False)
            elapsed = trace['timing']['final_visible_time'] - trace['timing']['audio_send_start_time']
            writer.writerow(
                dict(
                    id=uid,
                    raw_hypos=final['text'],
                    segments=json.dumps(final['segments'], ensure_ascii=False),
                    processing_time_sec=elapsed,
                )
            )
            stream.flush()
            print(f'[fast] {index}/{len(rows)} {uid}', flush=True)
    results = {}
    for index, (uid, row) in enumerate(rows.items(), 1):
        _, trace = decode(model, read_audio(row, args.data_root), True)
        results[uid] = trace
        write_json(args.output_dir / 'partial_results.json', results)
        print(f'[streaming] {index}/{len(rows)} {uid}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--manifest-csv', type=Path, required=True)
    parser.add_argument('--warmup-manifest-csv', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
