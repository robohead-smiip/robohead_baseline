"""Check shared evaluation stages and track entry points."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from robohead_baseline.utils.evaluate import evaluate

BASELINE = Path(__file__).resolve().parents[2]


class EvaluationTests(unittest.TestCase):
    def test_invalid_stage_does_not_report_success(self):
        for arguments in (
            ['--track', '2', '--stage', '3'],
            ['--track', '1', '--stage', '2'],
            ['--track', '2', '--stage', '2', '--stop-stage', '1'],
        ):
            with self.subTest(arguments=arguments):
                result = self.run_script('../robohead_evaluate_v1.sh', arguments)
                self.assertNotEqual(result.returncode, 0)

    def test_track1_inference_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / 'python'
            python.write_text(
                f'#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n'
                f'Path({str(root / "command.json")!r}).write_text('
                'json.dumps([sys.argv[1:], os.environ["CUDA_VISIBLE_DEVICES"]]))\n'
            )
            python.chmod(0o755)
            result = subprocess.run(
                [
                    'bash',
                    'run_track1.sh',
                    '--stage',
                    '6',
                    '--gpu-id',
                    '7',
                    '--asr-model',
                    str(root / 'model with spaces'),
                    '--config',
                    str(root / 'config with spaces.yaml'),
                    '--frontend-dir',
                    str(root / 'frontend'),
                    '--output-dir',
                    str(root / 'output'),
                ],
                cwd=BASELINE / 'track1',
                env=dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH']),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            command, device = json.loads((root / 'command.json').read_text())
            self.assertEqual(command[0], '../track1/local/recognize.py')
            self.assertEqual(command[command.index('--model-path') + 1], str(root / 'model with spaces'))
            self.assertEqual(command[command.index('--config') + 1], str(root / 'config with spaces.yaml'))
            self.assertEqual(command[command.index('--segments-json') + 1], str(root / 'frontend/val/segments.json'))
            self.assertEqual(device, '7')

    def test_track1_scripts_defer_model_selection_to_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / 'python'
            python.write_text(
                f'#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n'
                f'Path({str(root / "command.json")!r}).write_text(json.dumps(sys.argv[1:]))\n'
            )
            python.chmod(0o755)
            for script, arguments in (
                ('run_track1.sh', ['--stage', '6']),
                ('../robohead_evaluate_v1.sh', ['--track', '1', '--stage', '0']),
            ):
                for config in ([], ['--config', str(root / 'custom model.yaml')]):
                    with self.subTest(script=script, config=config):
                        result = subprocess.run(
                            ['bash', script, *arguments, *config],
                            cwd=BASELINE / 'track1',
                            env=dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH']),
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        command = json.loads((root / 'command.json').read_text())
                        self.assertEqual(
                            command[command.index('--config') + 1], config[-1] if config else 'conf/model.yaml'
                        )
                        self.assertNotIn('--model-path', command)
                        self.assertNotIn('--asr-model', command)

    def test_track2_separate_scoring_and_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / 'audio.wav', np.zeros((3200, 6)), 16000, subtype='PCM_16')
            (root / 'model.py').write_text(
                'from robohead_baseline.utils.tests.test_protocol import ExampleModel as Model\n'
            )
            manifest = root / 'sessions.csv'
            manifest.write_text(
                'id,audio_filepath,audio_duration_sec,mfa_speech_start,evaluation_unit\n'
                'a,audio.wav,0.2,0.02,session\n'
            )
            reference = root / 'reference.seglst.json'
            reference.write_text(
                json.dumps([dict(session_id='a', speaker='spk', start_time=0, end_time=0.1, words='你好')])
            )
            output = root / 'output'
            common = ['--data-root', str(root), '--output-dir', str(output)]
            result = self.run_script('run_track2.sh', ['--stage', '2', '--submission-dir', str(root), *common])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual({p.name for p in output.iterdir()}, {'predict.csv', 'partial_results.json'})
            stamps = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
            entry = '../robohead_evaluate_v1.sh'
            result = self.run_script(entry, ['--track', '2', '--stage', '1', *common])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            scores = json.loads((output / 'scores.json').read_text())
            self.assertEqual(scores['status'], 'accuracy_only')
            self.assertNotIn('ed_p95_s', scores)
            result = self.run_script(entry, ['--track', '2', '--stage', '2', *common])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            staged = json.loads((output / 'scores.json').read_text())
            self.assertEqual(staged['status'], 'pass')
            evaluate(
                SimpleNamespace(
                    track=2,
                    manifest_csv=manifest,
                    reference_json=reference,
                    hypothesis_json=None,
                    scoring_uem=None,
                    output_dir=output,
                    report_only=False,
                )
            )
            self.assertEqual(staged, json.loads((output / 'scores.json').read_text()))
            result = self.run_script('run_track2.sh', ['--stage', '3', *common])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(stamps, {name: (output / name).stat().st_mtime_ns for name in stamps})
            events = json.loads((output / 'partial_results.json').read_text())
            events['a']['events'][-1]['text'] = '您好'
            events['a']['events'][-1]['segments'][0]['words'] = '您好'
            (output / 'partial_results.json').write_text(json.dumps(events))
            result = self.run_script('run_track2.sh', ['--stage', '3', *common])
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads((output / 'scores.json').read_text())['status'], 'reject')
            result = self.run_script('run_track2.sh', ['--stage', '3', '--report-only', 'true', *common])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def run_script(self, script, arguments):
        return subprocess.run(
            ['bash', script, *arguments],
            cwd=BASELINE / 'track2',
            env=dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']),
            capture_output=True,
            text=True,
        )


if __name__ == '__main__':
    unittest.main()
