"""Check fixed partitions, trimmed sessions and training target isolation."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from robohead_baseline.track1.local.prepare_finetune import training_rows
from robohead_baseline.utils.prepare_data import prepare
from robohead_baseline.utils.prepare_track2_local import prepare as prepare_track2
from robohead_baseline.utils.protocol import load_csv


class TrainValTests(unittest.TestCase):
    def test_fixed_partition_and_speaker_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'train'
            (source / 'Audio_AEC').mkdir(parents=True)
            (source / 'DOC_AEC').mkdir()
            for uid, speaker in [('S001', 'P001'), ('S002', 'P002')]:
                sf.write(source / 'Audio_AEC' / f'{uid}.wav', np.zeros((32000, 6)), 16000, subtype='PCM_16')
                (source / 'DOC_AEC' / f'{uid}.TextGrid').write_text(
                    f'item [1]:\nclass = "IntervalTier"\nname = "{speaker}"\nintervals: size = 1\n'
                    'intervals [1]:\nxmin = 0.2\nxmax = 1.8\ntext = "你好<高兴>"\n'
                )
            partition = root / 'partition.tsv'
            partition.write_text('session_id\tsubset\nS001\ttrain\nS002\tval\n')
            args = SimpleNamespace(split_dir=source, output_dir=root / 'prepared', partition_file=partition)
            prepare(args)
            self.assertEqual(set(load_csv(args.output_dir / 'train/sessions.csv')), {'S001'})
            self.assertEqual(set(load_csv(args.output_dir / 'val/sessions.csv')), {'S002'})
            path = source / 'DOC_AEC/S002.TextGrid'
            path.write_text(path.read_text().replace('P002', 'P001'))
            with self.assertRaisesRegex(ValueError, 'speakers overlap'):
                prepare(args)
            partition.write_text('session_id\tsubset\nS001\ttrain\nS003\tval\n')
            with self.assertRaisesRegex(ValueError, 'fixed train/val partition'):
                prepare(args)

    def test_trim_preserves_overlaps_and_all_reference_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / 'audio.wav', np.zeros((12 * 16000, 6)), 16000, subtype='PCM_16')
            (root / 'sessions.csv').write_text('id,audio_filepath,audio_duration_sec\na,audio.wav,12\n')
            reference = [
                dict(session_id='a', speaker=spk, start_time=start, end_time=end, words=text)
                for spk, start, end, text in [('p', 1.0, 4.0, '你好'), ('q', 3.0, 5.0, '世界'), ('p', 7.0, 10.0, '再见')]
            ]
            (root / 'reference.seglst.json').write_text(json.dumps(reference))
            output = root / 'trimmed'
            prepare_track2(
                SimpleNamespace(split_dir=root, output_dir=output, leading_seconds=0.5, trailing_seconds=1.0)
            )
            manifest = load_csv(output / 'sessions.csv')
            self.assertEqual(set(manifest), {'a'})
            self.assertEqual(float(manifest['a']['source_start']), 0.5)
            self.assertEqual(float(manifest['a']['source_end']), 11.0)
            self.assertEqual(float(manifest['a']['audio_duration_sec']), 10.5)
            restored = []
            for row in json.loads((output / 'reference.seglst.json').read_text()):
                shift = float(manifest[row['session_id']]['source_start'])
                restored.append(
                    dict(row, session_id='a', start_time=row['start_time'] + shift, end_time=row['end_time'] + shift)
                )
            self.assertEqual(restored, reference)
            self.assertEqual(sorted(int(r['clip_speakers']) for r in manifest.values()), [2])

    def test_overlap_targets_only_use_separated_audio(self):
        references = [
            dict(session_id='s', speaker=speaker, start_time=start, end_time=end, words='你好')
            for speaker, start, end in [('p', 0.0, 3.0), ('q', 2.0, 4.0), ('p', 6.0, 8.0)]
        ]
        audio = [
            dict(
                session_id='s',
                raw='raw.wav',
                wpe_offline='offline.wav',
                wpe_online='online.wav',
                gss_gt={'p': 'p.wav', 'q': 'q.wav'},
            )
        ]
        rows = training_rows(references, audio)
        early = [r for r in rows if r['audio_interval_samples'][0] < 6 * 16000]
        self.assertEqual({r['frontend'] for r in early}, {'gss_gt'})
        late = [r for r in rows if r['audio_interval_samples'][0] == 6 * 16000]
        self.assertEqual({r['frontend'] for r in late}, {'raw', 'wpe_offline', 'wpe_online', 'gss_gt'})
        self.assertTrue(all(r['reference'] == '你好' for r in rows))

    def test_fixed_crop_shifts_references_and_excludes_context_speech(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = np.tile(np.arange(12 * 16000, dtype=np.float32)[:, None] / 200000, (1, 6))
            sf.write(root / 'audio.wav', audio, 16000, subtype='PCM_16')
            (root / 'sessions.csv').write_text('id,audio_filepath,audio_duration_sec\na,audio.wav,12\n')
            turns = [('p', 1, 2.8, '一二三'), ('p', 3, 7, '你好'), ('q', 4, 8, '世界'), ('p', 8.4, 9, '停录')]
            references = [
                dict(session_id='a', speaker=spk, start_time=start, end_time=end, words=text)
                for spk, start, end, text in turns
            ]
            (root / 'reference.seglst.json').write_text(json.dumps(references))
            crop = root / 'crop.tsv'
            crop.write_text('session_id\tspeech_start\tspeech_end\na\t3\t8\n')
            output = root / 'trimmed'
            args = SimpleNamespace(split_dir=root, output_dir=output, crop_file=crop)
            prepare_track2(args)
            row = load_csv(output / 'sessions.csv')['a']
            self.assertAlmostEqual(float(row['source_start']), 2.8)
            self.assertAlmostEqual(float(row['source_end']), 8.4)
            self.assertEqual(row['clip_speakers'], '2')
            self.assertEqual(row['mfa_speech_start'], '')
            kept = json.loads((output / 'reference.seglst.json').read_text())
            self.assertEqual([r['words'] for r in kept], ['你好', '世界'])
            self.assertAlmostEqual(kept[0]['start_time'], 0.2)
            self.assertAlmostEqual(kept[1]['end_time'], 5.2)
            expected, _ = sf.read(root / 'audio.wav', start=44800, stop=134400, dtype='float32')
            actual, _ = sf.read(output / 'audio/a.wav', dtype='float32')
            np.testing.assert_array_equal(actual, expected)
            for content, message in [
                ('a\t3.5\t8\n', 'cuts an annotated turn'),
                ('a\t3\t7.5\n', 'cuts an annotated turn'),
                ('a\t9.2\t10\n', 'no reference speech'),
                ('a\t3\t13\n', 'invalid crop bounds'),
                ('a\tnan\t8\n', 'invalid crop bounds'),
                ('b\t3\t8\n', 'session IDs'),
                ('a\t3\t8\na\t3\t8\n', 'duplicate session'),
            ]:
                with self.subTest(content=content):
                    crop.write_text('session_id\tspeech_start\tspeech_end\n' + content)
                    with self.assertRaisesRegex(ValueError, message):
                        prepare_track2(args)

    def test_prediction_guided_gss_does_not_require_labels(self):
        from robohead_baseline.track1.local.frontend import prepare as prepare_frontend

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / 'audio.wav', np.zeros((3200, 6)), 16000, subtype='PCM_16')
            (root / 'sessions.csv').write_text('id,audio_filepath,audio_duration_sec\na,audio.wav,0.2\n')
            (root / 'model.yaml').write_text('gss: {}\nwindows: {}\n')
            separation = Mock(return_value={})
            modules = {
                'cupy': SimpleNamespace(random=SimpleNamespace(seed=lambda _: None)),
                'robohead_baseline.frontend.gss': SimpleNamespace(
                    read_rttm=lambda path, duration: [], separate=separation
                ),
            }
            with patch.dict('sys.modules', modules):
                prepare_frontend(
                    SimpleNamespace(
                        stage='gss', config=root / 'model.yaml', data_dir=root, output_dir=root / 'out', num_files=0
                    )
                )
            separation.assert_called_once_with(root / 'audio.wav', [], {})
            self.assertFalse((root / 'reference.seglst.json').exists())
            self.assertEqual(json.loads((root / 'out/segments.json').read_text()), [])


if __name__ == '__main__':
    unittest.main()
