"""Check label masking and validation-based checkpoint selection."""

import gc
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('qwen_asr'), 'Qwen training dependencies are required')
class FinetuningTests(unittest.TestCase):
    def test_validation_matches_submission_timing(self):
        from robohead_baseline.track1.local.finetuning.scoring import meeting_scores
        from robohead_baseline.utils.score import score

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = [dict(session_id='a', speaker='p', start_time=0.01, end_time=0.11, words='好')]
            hypothesis = [dict(session_id='a', speaker='p', start_time=5.01, end_time=5.21, words='好')]
            (root / 'reference.json').write_text(json.dumps(reference))
            (root / 'hypothesis.json').write_text(json.dumps(hypothesis))
            (root / 'sessions.csv').write_text('id,audio_duration_sec\na,10\n')
            expected = score(SimpleNamespace(
                manifest_csv=root / 'sessions.csv', reference_json=root / 'reference.json',
                hypothesis_json=root / 'hypothesis.json', predict_csv=None, partial_json=None,
                scoring_uem=None, collar=5.0, meeteval_wer='meeteval-wer',
            ))
            actual = meeting_scores(
                [dict(row, prediction=row['words']) for row in hypothesis], root / 'reference.json',
            )
            self.assertEqual(actual['tcpcer_pct'], expected['error_rate'] * 100)

    def test_prefix_masking_with_left_padding(self):
        import torch
        from robohead_baseline.track1.local.finetuning.qwen3_asr_sft import DataCollatorForQwen3ASRFinetuning

        class Processor:
            tokenizer = SimpleNamespace(eos_token='!', pad_token_id=0)

            def __call__(self, text, **kwargs):
                ids = [[0, 0, 1, 2], [0, 3, 4, 5]] if text == ['a', 'b'] else [[0, 1, 2, 6, 7], [3, 4, 5, 8, 9]]
                tokens = torch.tensor(ids)
                return dict(input_ids=tokens, attention_mask=(tokens != 0).long())

        features = [dict(audio='x.wav', prefix_text=p, target='t') for p in ['a', 'b']]
        with patch('robohead_baseline.track1.local.finetuning.qwen3_asr_sft.load_audio', return_value=[]):
            result = DataCollatorForQwen3ASRFinetuning(Processor())(features)
        self.assertEqual(result['labels'].tolist(), [[-100, -100, -100, 6, 7], [-100, -100, -100, 8, 9]])

    def test_best_checkpoint_uses_generated_validation_score(self):
        import torch
        from transformers import TrainingArguments, default_data_collator
        from robohead_baseline.track1.local.finetuning.qwen3_asr_sft import CastFloatInputsTrainer

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, input_ids, labels):
                prediction = input_ids * self.weight
                return dict(loss=((prediction - labels) ** 2).mean(), logits=prediction)

        class Validation:
            reference_file = 'full_val.json'
            rows = [dict(id='val')]

            def __call__(self, trainer):
                return dict(eval_tcpcer=10.0 * trainer.state.global_step)

        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(
                output_dir=directory,
                use_cpu=True,
                max_steps=2,
                learning_rate=0.01,
                per_device_train_batch_size=1,
                eval_strategy='steps',
                eval_steps=1,
                save_steps=1,
                load_best_model_at_end=True,
                metric_for_best_model='eval_tcpcer',
                greater_is_better=False,
                report_to=[],
                disable_tqdm=True,
            )
            rows = [dict(input_ids=[1.0], labels=[0.0])]
            trainer = CastFloatInputsTrainer(
                model=Model(), args=args, train_dataset=rows, eval_dataset=rows, data_collator=default_data_collator
            )
            trainer.val_cer = Validation()
            trainer.train()
            self.assertEqual(trainer.state.best_metric, 10.0)
            self.assertEqual(Path(trainer.state.best_model_checkpoint).name, 'checkpoint-1')
            from safetensors.torch import load_file

            best = load_file(str(Path(directory) / 'checkpoint-1/model.safetensors'))
            torch.testing.assert_close(trainer.model.weight, best['weight'])
            del best, trainer
            gc.collect()


if __name__ == '__main__':
    unittest.main()
