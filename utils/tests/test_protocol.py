"""Check streaming delivery, latency calculation and submission validation."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from robohead_baseline.track1.local.decode import run as run_offline
from robohead_baseline.track2.local.decode import decode, run as run_streaming
from robohead_baseline.track2.local.validate import validate
from robohead_baseline.utils.annotations import read_textgrid
from robohead_baseline.utils.audio import read_audio, write_json
from robohead_baseline.utils.evaluate import evaluate
from robohead_baseline.utils.mfa.align import export as export_alignment
from robohead_baseline.utils.mfa.align import prepare as prepare_alignment
from robohead_baseline.utils.protocol import _percentile, _stable_partial_time, compute_latency, file_hash, load_csv
from robohead_baseline.utils.score import score


class ExampleModel:
    def reset(self):
        self.samples = 0
        self.final_calls = 0

    def set_partial_callback(self, callback):
        self.callback = callback

    def accept_chunk(self, chunk):
        self.samples += len(chunk)
        self.callback('你好')
        return '你好'

    def input_finished(self):
        self.final_calls += 1
        return dict(text='你好', segments=[dict(speaker='spk', start_time=0.0, end_time=0.1, words='你好')])


class ProtocolTest(unittest.TestCase):
    def test_missing_alignment_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'sessions.csv'
            manifest.write_text('id,audio_duration_sec,mfa_speech_start\na,1.0,\n')
            write_json(root / 'utterances.json', {'a_P001_000000': {}})
            before = manifest.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Missing MFA alignment: a_P001_000000'):
                export_alignment(SimpleNamespace(data_dir=root, work_dir=root))
            self.assertEqual(manifest.read_bytes(), before)

    def test_g2p_command_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / 'audio.wav', np.zeros((16000, 6)), 16000, subtype='PCM_16')
            (root / 'sessions.csv').write_text('id,audio_filepath,audio_duration_sec\na,audio.wav,1\n')
            write_json(
                root / 'reference.seglst.json',
                [dict(session_id='a', speaker='P001', start_time=0.1, end_time=0.8, words='你好')],
            )
            (root / 'dictionary.dict').write_text('你 n i\n')
            stub = root / 'mfa'
            stub.write_text(
                f'#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n'
                f'with Path({str(root / "scratch.jsonl")!r}).open("a") as stream:\n'
                '    stream.write(json.dumps([os.environ["MFA_ROOT_DIR"], os.environ["TMPDIR"]]) + "\\n")\n'
                "if sys.argv[1] == 'align':\n"
                f'    Path({str(root / "align_args.json")!r}).write_text(json.dumps(sys.argv[1:]))\n'
                '    raise SystemExit(17)\n'
                f'Path({str(root / "args.json")!r}).write_text(json.dumps(sys.argv[1:]))\n'
                "Path(sys.argv[4]).write_text('好 h ao\\n')\n"
            )
            stub.chmod(0o755)
            env = dict(
                os.environ,
                MFA_BIN=str(stub),
                MFA_G2P_MODEL=str(root / 'g2p.zip'),
                MFA_ROOT_DIR=str(root),
                TMPDIR=str(root),
            )
            env.pop('MFA_BEAM', None)
            env.pop('MFA_RETRY_BEAM', None)
            script = Path(__file__).resolve().parents[1] / 'mfa/run.sh'
            result = subprocess.run(
                [
                    'bash',
                    str(script),
                    str(root),
                    str(root / 'dictionary.dict'),
                    str(root / 'acoustic.zip'),
                    str(root / 'work'),
                ],
                env=env,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 17, result.stderr.decode())
            arguments = json.loads((root / 'args.json').read_text())
            self.assertEqual(arguments[:3], ['g2p', str(root / 'work/oov.txt'), str(root / 'g2p.zip')])
            alignment = json.loads((root / 'align_args.json').read_text())
            self.assertEqual(alignment[alignment.index('--beam') + 1], '100')
            self.assertEqual(alignment[alignment.index('--retry_beam') + 1], '400')
            scratch = [json.loads(line) for line in (root / 'scratch.jsonl').read_text().splitlines()]
            self.assertEqual(len(scratch), 2)
            self.assertEqual(scratch[0], scratch[1])
            self.assertEqual(scratch[0][0], scratch[0][1])
            self.assertNotEqual(scratch[0][0], str(root))
            self.assertEqual(Path(scratch[0][0]).parent, Path('/tmp'))
            self.assertFalse(Path(scratch[0][0]).exists())

    def test_alignment_prepare_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'near.wav'
            sf.write(audio, np.zeros(19200), 16000, subtype='PCM_16')
            write_json(root / 'alignment_sources.json', {'test/a': str(audio)})
            write_json(
                root / 'reference.seglst.json',
                [
                    dict(session_id='test', speaker='a', start_time=0.5, end_time=0.9, words='你好'),
                    dict(session_id='test', speaker='a', start_time=0.95, end_time=1.1, words='好'),
                ],
            )
            (root / 'sessions.csv').write_text('id,audio_duration_sec,mfa_speech_start\ntest,1.2,\n')
            dictionary = root / 'source.dict'
            dictionary.write_text('你 n i\n好 h ao\n')
            args = SimpleNamespace(data_dir=root, work_dir=root / 'work', dictionary=dictionary)
            for folder in ['corpus', 'aligned']:
                (args.work_dir / folder).mkdir(parents=True)
                (args.work_dir / folder / 'previous.TextGrid').write_text('stale alignment')
            prepare_alignment(args)
            self.assertFalse((args.work_dir / 'corpus/previous.TextGrid').exists())
            self.assertFalse((args.work_dir / 'aligned/previous.TextGrid').exists())
            self.assertEqual(len(json.loads((args.work_dir / 'utterances.json').read_text())), 2)
            self.assertEqual((args.work_dir / 'corpus/a/test_a_000000.lab').read_text(), '你 好\n')
            aligned = args.work_dir / 'aligned'
            aligned.mkdir()
            (aligned / 'test_a_000000.TextGrid').write_text(
                'item [1]:\nclass = "IntervalTier"\nname = "words"\nintervals: size = 2\n'
                'intervals [1]:\nxmin = 0.2\nxmax = 0.4\ntext = "你"\n'
                'intervals [2]:\nxmin = 0.4\nxmax = 0.6\ntext = "好"\n'
                'item [2]:\nclass = "IntervalTier"\nname = "phones"\nintervals: size = 3\n'
                'intervals [1]:\nxmin = 0\nxmax = 0.2\ntext = "sil"\n'
                'intervals [2]:\nxmin = 0.2\nxmax = 0.4\ntext = "n"\n'
                'intervals [3]:\nxmin = 0.4\nxmax = 0.6\ntext = "ao"\n'
            )
            (aligned / 'test_a_000001.TextGrid').write_text(
                'item [1]:\nclass = "IntervalTier"\nname = "words"\nintervals: size = 1\n'
                'intervals [1]:\nxmin = 0.2\nxmax = 0.35\ntext = "好"\n'
                'item [2]:\nclass = "IntervalTier"\nname = "phones"\nintervals: size = 1\n'
                'intervals [1]:\nxmin = 0.2\nxmax = 0.35\ntext = "ao"\n'
            )
            reference_before = (root / 'reference.seglst.json').read_bytes()
            export_alignment(args)
            self.assertEqual((root / 'reference.seglst.json').read_bytes(), reference_before)
            self.assertEqual(float(load_csv(root / 'sessions.csv')['test']['mfa_speech_start']), 0.5)
            self.assertAlmostEqual(float(load_csv(root / 'sessions.csv')['test']['mfa_speech_end']), 1.1)

    def test_track1_result_only_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'sessions.csv'
            manifest.write_text('id,audio_duration_sec\ntest,10.0\n')
            reference = root / 'reference.json'
            hypothesis = root / 'hypothesis.seglst.json'
            rows = [dict(session_id='test', speaker='a', start_time=0.0, end_time=0.1, words='你好')]
            write_json(reference, rows)
            args = SimpleNamespace(
                manifest_csv=manifest,
                reference_json=reference,
                hypothesis_json=hypothesis,
                predict_csv=None,
                partial_json=None,
                scoring_uem=None,
                output_dir=root / 'scoring',
                collar=5.0,
                meeteval_wer=str(Path(sys.executable).with_name('meeteval-wer')),
            )
            for start, text, expected in ((0.0, '你好', 0.0), (8.0, '你好', 2.0), (0.0, '', 1.0)):
                write_json(hypothesis, [dict(rows[0], start_time=start, end_time=start + 0.1, words=text)])
                result = score(args)
                self.assertEqual(result['error_rate'], expected)
                self.assertFalse(args.output_dir.exists())
            write_json(hypothesis, [])
            with self.assertRaises(ValueError):
                score(args)

    def test_scoring_uses_segment_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'sessions.csv'
            manifest.write_text('id,audio_duration_sec\ntest,20.0\n')
            reference = root / 'reference.json'
            hypothesis = root / 'hypothesis.json'
            row = dict(session_id='test', speaker='a', start_time=0.0, end_time=20.0, words='你好')
            write_json(reference, [dict(row, timestamps=[dict(char='你好', start=18.0, end=19.0)])])
            write_json(hypothesis, [row])
            args = SimpleNamespace(
                manifest_csv=manifest,
                reference_json=reference,
                hypothesis_json=hypothesis,
                predict_csv=None,
                partial_json=None,
                scoring_uem=None,
                output_dir=root / 'scoring',
                collar=5.0,
                meeteval_wer=str(Path(sys.executable).with_name('meeteval-wer')),
            )
            result = score(args)
            self.assertEqual(result['error_rate'], 0.0)
            provenance = result['provenance']
            self.assertEqual(provenance['reference_timing'], 'character_based')

    def test_audio_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'audio.wav'
            row = dict(audio_filepath=path.name, audio_duration_sec=0.1)
            sf.write(path, np.zeros((1600, 1)), 16000, subtype='PCM_16')
            with self.assertRaises(ValueError):
                read_audio(row, root)
            sf.write(path, np.zeros((1600, 6)), 16000, subtype='FLOAT')
            with self.assertRaises(ValueError):
                read_audio(row, root)
            sf.write(path, np.zeros((1600, 6)), 16000, subtype='PCM_16')
            self.assertEqual(read_audio(row, root).shape, (1600, 6))

    def test_offline_interface_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / 'audio.wav', np.zeros((1600, 6)), 16000, subtype='PCM_16')
            (root / 'model.py').write_text(
                'class Model:\n'
                '    def predict(self, wav_path):\n'
                "        return dict(text='你好', segments=[dict(speaker='a', start_time=0, "
                "end_time=0.1, words='你好')])\n"
            )
            manifest = root / 'sessions.csv'
            manifest.write_text('id,audio_filepath,audio_duration_sec\ntest,audio.wav,0.1\n')
            args = SimpleNamespace(
                manifest_csv=manifest, output_dir=root / 'output', config=None, submission_dir=root, data_root=root
            )
            run_offline(args)
            run_offline(args)
            self.assertEqual({p.name for p in args.output_dir.iterdir()}, {'hypothesis.seglst.json'})
            segments = json.loads((args.output_dir / 'hypothesis.seglst.json').read_text())
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0]['session_id'], 'test')
            reference = root / 'reference.json'
            write_json(reference, segments)
            evaluate(
                SimpleNamespace(
                    track=1,
                    manifest_csv=manifest,
                    reference_json=reference,
                    hypothesis_json=None,
                    scoring_uem=None,
                    output_dir=args.output_dir,
                    report_only=False,
                )
            )
            self.assertEqual({p.name for p in args.output_dir.iterdir()}, {'hypothesis.seglst.json', 'scores.json'})
            self.assertEqual(json.loads((args.output_dir / 'scores.json').read_text())['tcpcer_pct'], 0)

    def test_minimal_streaming_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / 'audio.wav', np.zeros((3200, 6)), 16000, subtype='PCM_16')
            (root / 'model.py').write_text(
                'class Model:\n'
                '    def reset(self):\n'
                '        pass\n'
                '    def set_partial_callback(self, callback):\n'
                '        self.callback = callback\n'
                '    def accept_chunk(self, audio):\n'
                "        self.callback('你好')\n"
                "        return '你好'\n"
                '    def input_finished(self):\n'
                "        return dict(text='你好', segments=[dict(speaker='a', start_time=0, "
                "end_time=0.1, words='你好')])\n"
            )
            manifest = root / 'sessions.csv'
            header = 'id,audio_filepath,audio_duration_sec,mfa_speech_start,evaluation_unit\n'
            manifest.write_text(header + 'a,audio.wav,0.2,0.02,session\nb,audio.wav,0.2,0.02,session\n')
            reference = root / 'reference.json'
            write_json(
                reference,
                [dict(session_id=uid, speaker='a', start_time=0, end_time=0.1, words='你好') for uid in ('a', 'b')],
            )
            args = SimpleNamespace(
                manifest_csv=manifest,
                output_dir=root / 'output',
                submission_dir=root,
                data_root=root,
                config=None,
                warmup_manifest_csv=None,
                track=2,
                reference_json=reference,
                hypothesis_json=None,
                scoring_uem=None,
                report_only=False,
            )
            run_streaming(args)
            expected = {'predict.csv', 'partial_results.json'}
            self.assertEqual({p.name for p in args.output_dir.iterdir()}, expected)
            predictions = load_csv(args.output_dir / 'predict.csv')
            self.assertEqual(set(predictions), {'a', 'b'})
            self.assertTrue(all(float(row['processing_time_sec']) > 0 for row in predictions.values()))
            partials = json.loads((args.output_dir / 'partial_results.json').read_text())
            self.assertEqual(set(partials), {'a', 'b'})
            evaluate(args)
            self.assertEqual({p.name for p in args.output_dir.iterdir()}, expected | {'scores.json'})
            result = json.loads((args.output_dir / 'scores.json').read_text())
            self.assertEqual(result['status'], 'pass')
            self.assertEqual(result['tcpcer_pct'], 0)
            self.assertEqual((result['accuracy_files'], result['latency_files']), (2, 2))
            self.assertGreater(result['rtf'], 0)
            final = partials['b']['events'][-1]
            final['text'] = '您好'
            final['segments'][0]['words'] = '您好'
            write_json(args.output_dir / 'partial_results.json', partials)
            with self.assertRaises(SystemExit):
                evaluate(args)
            result = json.loads((args.output_dir / 'scores.json').read_text())
            self.assertEqual(result['transcript_mismatches'], ['b'])
            self.assertEqual({p.name for p in args.output_dir.iterdir()}, expected | {'scores.json'})

    def test_stable_character_and_empty_fallback(self):
        events = [dict(text=text, time=i) for i, text in enumerate(['你', '我', '你好', '你很好'])]
        self.assertEqual(_stable_partial_time(events), 2)
        self.assertEqual(_stable_partial_time([dict(text='', time=3)]), 3)

    def test_latency_definitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'sessions.csv'
            manifest.write_text(
                'id,audio_duration_sec,mfa_speech_start,mfa_speech_end,evaluation_unit\n' 'a,10,0.5,8,session\n'
            )
            events = root / 'events.json'
            write_json(
                events,
                {
                    'a': {
                        'events': [
                            dict(event='partial_callback', time=102, text='你好'),
                            dict(event='final_visible', time=113, text='你好'),
                        ],
                        'timing': dict(audio_send_start_time=100, audio_end_oracle_time=110, final_visible_time=113),
                    }
                },
            )
            result = compute_latency(events, manifest)
            self.assertEqual(result['ed_sec']['p95'], 3)
            self.assertEqual(result['ttfst_sec']['p95'], 1.5)

    def test_percentile(self):
        self.assertAlmostEqual(_percentile(list(range(1, 35)), 95), 32.35)
        self.assertIsNone(_percentile([], 95))

    def test_long_csv_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['id', 'raw_hypos'])
                writer.writerow(['test', 'a' * 150_000])
            self.assertEqual(len(load_csv(path)['test']['raw_hypos']), 150_000)

    def test_textgrid_quotes_and_empty_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.TextGrid'
            path.write_text(
                '''item [1]:
class = "IntervalTier"
name = "001"
intervals: size = 2
intervals [1]:
xmin = 0
xmax = 1
text = ""
intervals [2]:
xmin = 1
xmax = 2
text = "say ""hello"""
'''
            )
            self.assertEqual(read_textgrid(path)['001'], [(0.0, 1.0, ''), (1.0, 2.0, 'say "hello"')])

    def test_fast_streaming_and_rejection(self):
        model = ExampleModel()
        audio = np.zeros((3200, 6), dtype=np.float32)
        fast, _ = decode(model, audio, False)
        streamed, capture = decode(model, audio, True)
        self.assertEqual(fast, streamed)
        self.assertEqual(model.final_calls, 1)
        self.assertEqual(model.samples, 3200)
        packets = capture['capture']['audio_packets']
        origin = capture['timing']['audio_send_start_time']
        self.assertEqual(packets[0]['sent_time'], origin)
        self.assertGreaterEqual(packets[1]['sent_time'] - origin, 0.099)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'sessions.csv'
            manifest.write_text('id,audio_duration_sec,mfa_speech_start,evaluation_unit\ntest,0.2,0.02,session\n')
            events = root / 'events.json'
            write_json(events, {'test': capture})
            result = compute_latency(events, manifest)
            self.assertEqual(result['status'], 'complete')
            self.assertAlmostEqual(result['ed_sec']['p95'], capture['timing']['final_visible_time'] - origin - 0.2)
            predictions = root / 'fast.csv'
            with predictions.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['id', 'raw_hypos', 'segments', 'processing_time_sec'])
                writer.writerow(['test', fast['text'], json.dumps(fast['segments']), '0.02'])
            args = SimpleNamespace(manifest_csv=manifest, fast_csv=predictions, events_json=events,)
            accuracy = dict(
                error_rate=0.0,
                provenance=dict(manifest_sha256=file_hash(manifest), predict_csv_sha256=file_hash(predictions)),
            )
            for rate, expected in [(0.0, 'pass'), (2.1, 'pass')]:
                accuracy['error_rate'] = rate
                self.assertEqual(validate(args, accuracy)['status'], expected)
            accuracy['error_rate'] = 0.0
            for ed, ttfst, expected in [(1.5, 4.5, 'pass'), (1.5001, 4.5, 'reject'), (1.5, 4.5001, 'reject')]:
                measured = dict(result, ed_sec={'p95': ed}, ttfst_sec={'p95': ttfst})
                with patch('robohead_baseline.track2.local.validate.compute_latency', return_value=measured):
                    self.assertEqual(validate(args, accuracy)['status'], expected)
            with predictions.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['id', 'raw_hypos', 'segments', 'processing_time_sec'])
                writer.writerow(['test', '您好', json.dumps([dict(fast['segments'][0], words='您好')]), '0.02'])
            checked = validate(args, accuracy)
            self.assertEqual(checked['transcript_mismatches'], ['test'])
            self.assertIn('Accuracy was not scored from the complete fast-pass results', checked['reasons'])
            capture['capture']['audio_packets'][0]['sent_time'] += 0.1
            write_json(events, {'test': capture})
            self.assertEqual(compute_latency(events, manifest)['status'], 'invalid_or_incomplete')

    def test_replay_requires_all_accuracy_sessions(self):
        final, capture = decode(ExampleModel(), np.zeros((3200, 6), dtype=np.float32), True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = 'id,audio_duration_sec,mfa_speech_start,evaluation_unit\n'
            manifest = root / 'sessions.csv'
            manifest.write_text(header + 'a,0.2,0.02,session\nb,0.2,0.02,session\n')
            events, predictions = root / 'events.json', root / 'fast.csv'
            write_json(events, {'b': capture})
            with predictions.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['id', 'raw_hypos', 'segments', 'processing_time_sec'])
                for uid in ('a', 'b'):
                    writer.writerow([uid, final['text'], json.dumps(final['segments']), '0.02'])
            accuracy = dict(
                error_rate=0.1,
                provenance=dict(manifest_sha256=file_hash(manifest), predict_csv_sha256=file_hash(predictions)),
            )
            args = SimpleNamespace(manifest_csv=manifest, events_json=events, fast_csv=predictions)
            result = validate(args, accuracy)
            self.assertEqual(result['status'], 'reject')
            self.assertEqual((result['accuracy_files'], result['latency_files']), (2, 2))
            self.assertAlmostEqual(result['rtf'], 0.1)
            self.assertAlmostEqual(result['rtfx'], 10.0)
            write_json(events, {'a': capture, 'b': capture})
            self.assertEqual(validate(args, accuracy)['status'], 'pass')


if __name__ == '__main__':
    unittest.main()
