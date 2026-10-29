"""Test result-only ZIP scoring and archive validation."""

import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robohead_baseline.utils.submission import evaluate, unpack_submission
from robohead_baseline.utils.score_submission import score_submission


class SubmissionTests(unittest.TestCase):
    def test_rejected_submission_does_not_keep_previous_leaderboard_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / 'scores.json'
            previous.write_text('{"tcpcer_pct": 0}')
            args = SimpleNamespace(track=2, input_dir=root / 'input', output_dir=root)
            with patch('robohead_baseline.utils.score_submission.evaluate', return_value=1):
                with self.assertRaises(SystemExit):
                    score_submission(args)
            self.assertFalse(previous.exists())

    def test_track_archives(self):
        for track, required in ((1, ['hypothesis.seglst.json']), (2, ['predict.csv', 'partial_results.json'])):
            with self.subTest(track=track), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / 'submission.zip'
                with zipfile.ZipFile(archive, 'w') as stream:
                    for name in required:
                        stream.writestr(name, 'example')
                    stream.writestr('conf/model.yaml', 'language: zh-CN')
                unpack_submission(archive, root / 'unpacked', track)
                self.assertEqual((root / 'unpacked' / required[0]).read_text(), 'example')
                self.assertTrue((root / 'unpacked/conf/model.yaml').is_file())

    def test_unsafe_names_and_symlinks(self):
        for name in ('../outside', '/outside', 'local/../../outside', 'local\\outside', 'C:/outside', 'link'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / 'submission.zip'
                with zipfile.ZipFile(archive, 'w') as stream:
                    stream.writestr('predict.csv', 'example')
                    stream.writestr('partial_results.json', '{}')
                    member = zipfile.ZipInfo(name)
                    if name == 'link':
                        member.create_system = 3
                        member.external_attr = (stat.S_IFLNK | 0o777) << 16
                    stream.writestr(member, '/outside')
                with self.assertRaises(ValueError):
                    unpack_submission(archive, root / 'unpacked', 2)
                self.assertFalse((root / 'unpacked').exists())

    def test_missing_root_and_duplicate_paths(self):
        for names in (('submission/predict.csv',), ('predict.csv', './predict.csv'), ('model.py',)):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / 'submission.zip'
                with zipfile.ZipFile(archive, 'w') as stream:
                    for name in names:
                        stream.writestr(name, 'example')
                with self.assertRaises(ValueError):
                    unpack_submission(archive, root / 'unpacked', 2)

    def test_result_only_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('predict.csv', 'partial_results.json'):
                (root / name).write_text('example')
            args = SimpleNamespace(track=2, data=root, output=root / 'scores', scoring_uem=None)
            with patch('subprocess.run') as run:
                run.return_value.returncode = 7
                self.assertEqual(evaluate(args, root), 7)
                command = run.call_args.args[0]
                self.assertEqual(command[0], 'bash')
                self.assertEqual(Path(command[1]).name, 'robohead_evaluate_v1.sh')
                self.assertEqual(command[command.index('--stage') + 1], '1')
                self.assertEqual(command[command.index('--stop-stage') + 1], '2')
                self.assertNotIn('--submission-dir', command)
                self.assertEqual((args.output / 'predict.csv').read_text(), 'example')
            args.track = 1
            with patch('subprocess.run') as run:
                evaluate(args, root)
                self.assertIn(str(root / 'hypothesis.seglst.json'), run.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
