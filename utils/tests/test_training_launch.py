"""Check distributed launch and rank-local training setup."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from robohead_baseline.track1.local import finetune


class TrainingLaunchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name, rows in (
            ('train.jsonl', [dict(session_id='train', reference='你好')]),
            ('val.jsonl', [dict(session_id='val')]),
        ):
            (self.root / name).write_text('\n'.join(json.dumps(row) for row in rows))
        (self.root / 'val_reference.json').write_text(json.dumps([dict(session_id='val')]))
        self.config = self.root / 'config.yaml'
        self.config.write_text(yaml.safe_dump(dict(
            data_dir=str(self.root), gpu_ids='0,1,2,3,4,5,6,7', num_workers=0,
            output_dir=str(self.root / 'exp'),
        )))
        self.args = SimpleNamespace(config=self.config, data_dir=None, check_only=False)
        self.worker = Mock()
        self.addCleanup(patch.stopall)
        patch.dict('sys.modules', {
            'robohead_baseline.track1.local.finetuning.qwen3_asr_sft': SimpleNamespace(main=self.worker)
        }).start()

    def test_parent_launches_eight_workers_without_training(self):
        with patch.dict(os.environ, {}, clear=True), patch('subprocess.run') as run:
            finetune.train(self.args)
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertIn('torch.distributed.run', command)
        self.assertEqual(command[command.index('--nproc_per_node') + 1], '8')
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
        self.worker.assert_not_called()

    def test_worker_does_not_relaunch_or_write_shared_config(self):
        with patch.dict(os.environ, {'LOCAL_RANK': '1', 'RANK': '1'}), patch('subprocess.run') as run:
            finetune.train(self.args)
        run.assert_not_called()
        self.worker.assert_called_once()
        self.assertFalse((self.root / 'exp/training_config.json').exists())

    def test_check_only_needs_no_workers(self):
        self.args.check_only = True
        with patch('subprocess.run') as run:
            finetune.train(self.args)
        run.assert_not_called()
        self.worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
