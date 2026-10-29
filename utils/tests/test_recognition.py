"""Check Qwen inference model selection from YAML and command-line overrides."""

import json
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from robohead_baseline.utils.audio import write_json


BASELINE = Path(__file__).resolve().parents[2]


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / 'model.yaml'
        self.config.write_text('asr_model: models/configured-qwen\n')
        write_json(self.root / 'segments.json', [])
        (self.root / 'sessions.csv').write_text('id,audio_duration_sec\na,1.0\n')
        self.loader = Mock(return_value=SimpleNamespace(model=object()))
        self.modules = {
            'torch': SimpleNamespace(bfloat16='bfloat16'),
            'qwen_asr': SimpleNamespace(Qwen3ASRModel=SimpleNamespace(from_pretrained=self.loader)),
            'robohead_baseline.track1.local.finetuning.validation': SimpleNamespace(
                predict_rows=lambda *args, **kwargs: [
                    dict(session_id='a', speaker='s', start_time=0, end_time=0.5, prediction='你好')
                ],
                _generation_mode=lambda model: nullcontext(),
            ),
        }

    def test_yaml_model_is_used_without_override(self):
        self.run_recognizer(['--config', str(self.config)])
        self.assertEqual(self.loader.call_args.args[0], 'models/configured-qwen')

    def test_default_config_is_read(self):
        config = self.root / 'conf/model.yaml'
        config.parent.mkdir()
        config.write_text(yaml.safe_dump({'asr_model': 'models/default-qwen'}))
        cwd = Path.cwd()
        try:
            os.chdir(self.root)
            self.run_recognizer([])
        finally:
            os.chdir(cwd)
        self.assertEqual(self.loader.call_args.args[0], 'models/default-qwen')

    def test_explicit_model_overrides_yaml(self):
        self.run_recognizer(['--config', str(self.config), '--asr-model', 'models/override'])
        self.assertEqual(self.loader.call_args.args[0], 'models/override')

    def test_existing_model_path_option_still_works(self):
        self.run_recognizer(['--model-path', 'models/existing'])
        self.assertEqual(self.loader.call_args.args[0], 'models/existing')

    def run_recognizer(self, options):
        script = BASELINE / 'track1/local/recognize.py'
        arguments = [
            str(script),
            '--segments-json', str(self.root / 'segments.json'),
            '--manifest-csv', str(self.root / 'sessions.csv'),
            '--output-dir', str(self.root / 'output'),
            *options,
        ]
        with patch.dict(sys.modules, self.modules), patch.object(sys, 'argv', arguments):
            runpy.run_path(str(script), run_name='__main__')
        rows = json.loads((self.root / 'output/hypothesis.seglst.json').read_text())
        self.assertEqual(rows[0]['words'], '你好')


if __name__ == '__main__':
    unittest.main()
