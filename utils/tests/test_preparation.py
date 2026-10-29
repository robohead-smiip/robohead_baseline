"""Check six-channel preparation and session boundary audits."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from robohead_baseline.utils.audio import read_audio
from robohead_baseline.utils.latency_local import prepare as audit_boundaries
from robohead_baseline.utils.prepare_data import prepare
from robohead_baseline.utils.prepare_manifest import prepare as prepare_manifest
from robohead_baseline.utils.prepare_track2_local import prepare as prepare_track2
from robohead_baseline.utils.protocol import load_csv


class PreparationTests(unittest.TestCase):
    def test_training_duplicate_is_retained_in_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, expected in [('train', {'S050'}), ('dev', {'S050', 'S161'})]:
                source = root / split
                (source / 'Audio_AEC').mkdir(parents=True)
                (source / 'DOC_AEC').mkdir()
                for uid in ['S050', 'S161']:
                    sf.write(source / 'Audio_AEC' / f'{uid}.wav', np.zeros((16000, 6)), 16000, subtype='PCM_16')
                    (source / 'DOC_AEC' / f'{uid}.TextGrid').write_text(
                        'item [1]:\nclass = "IntervalTier"\nname = "P001"\nintervals: size = 1\n'
                        'intervals [1]:\nxmin = 0.2\nxmax = 0.8\ntext = "你好"\n'
                    )
                for track, function in [('track1', prepare), ('track2', prepare_track2)]:
                    with self.subTest(split=split, track=track):
                        output = root / f'{split}_{track}'
                        function(SimpleNamespace(split_dir=source, output_dir=output))
                        self.assertEqual(set(load_csv(output / 'sessions.csv')), expected)
                        references = json.loads((output / 'reference.seglst.json').read_text())
                        self.assertEqual({row['session_id'] for row in references}, expected)

    def test_release_audio_and_human_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'train', root / 'prepared'
            (source / 'Audio_AEC').mkdir(parents=True)
            (source / 'DOC_AEC').mkdir()
            (source / 'Audio').mkdir()
            (source / 'DOC').mkdir()
            sf.write(source / 'Audio/S001.wav', np.zeros((16000, 8)), 16000, subtype='PCM_16')
            (source / 'DOC/S001.TextGrid').write_text('This raw annotation must not be read.')
            audio = np.tile(np.arange(6, dtype=np.float32) / 10, (16000, 1))
            sf.write(source / 'Audio_AEC/S001.wav', audio, 16000, subtype='PCM_16')
            tiers = [
                ('P001', ['你好<意图：社交闲聊><高兴-低>', '<unknow>', '。', '嗯。']),
                ('P002', ['小量小量。']),
                ('audio_server_raw_audio_data', ['这是机器人。']),
                ('notes', ['不要识别这段注释']),
            ]
            blocks = []
            for index, (name, texts) in enumerate(tiers, 1):
                block = f'item [{index}]:\nclass = "IntervalTier"\nname = "{name}"\n'
                block += f'intervals: size = {len(texts)}\n'
                for number, text in enumerate(texts, 1):
                    block += (
                        f'intervals [{number}]:\nxmin = {number / 10}\n'
                        f'xmax = {(number + 1) / 10}\ntext = "{text}"\n'
                    )
                blocks.append(block)
            (source / 'DOC_AEC/S001.TextGrid').write_text('\n'.join(blocks))
            prepare(SimpleNamespace(split_dir=source, output_dir=output))
            row = load_csv(output / 'sessions.csv')['S001']
            np.testing.assert_allclose(read_audio(row, output), audio[:, :6], atol=1 / 32768)
            references = json.loads((output / 'reference.seglst.json').read_text())
            self.assertEqual({r['speaker'] for r in references}, {'P001', 'P002'})
            self.assertEqual({r['words'] for r in references}, {'你好', '嗯。', '小量小量。'})
            imported = root / 'track2'
            prepare_track2(SimpleNamespace(split_dir=source, output_dir=imported))
            imported_row = load_csv(imported / 'sessions.csv')['S001']
            self.assertEqual(imported_row['clip_speakers'], '2')
            self.assertEqual(imported_row['mfa_speech_start'], '')
            self.assertEqual(json.loads((imported / 'reference.seglst.json').read_text()), references)

    def test_eight_channel_audio_is_not_aec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / 'a.wav', np.zeros((160, 8)), 16000, subtype='PCM_16')
            row = dict(audio_filepath='a.wav', audio_duration_sec=0.01)
            with self.assertRaises(ValueError):
                read_audio(row, root)

    def test_no_raw_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Audio').mkdir()
            sf.write(root / 'Audio/S001.wav', np.zeros((160, 8)), 16000, subtype='PCM_16')
            with self.assertRaises(ValueError):
                prepare(SimpleNamespace(split_dir=root, output_dir=root / 'prepared'))

    def test_microphone_layouts_in_wav_scp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = np.tile(np.arange(6, dtype=np.float32) / 10, (160, 1))
            sf.write(root / 'release.wav', audio, 16000, subtype='PCM_16')
            for channel in range(6):
                sf.write(root / f'mic{channel}.wav', audio[:, channel], 16000, subtype='PCM_16')
            wav_scp = root / 'wav.scp'
            wav_scp.write_text('release release.wav\nseparate ' + ' '.join(f'mic{i}.wav' for i in range(6)) + '\n')
            prepare_manifest(
                SimpleNamespace(wav_scp=wav_scp, data_root=root, onset_csv=None, output_csv=root / 'sessions.csv')
            )
            rows = load_csv(root / 'sessions.csv')
            np.testing.assert_array_equal(read_audio(rows['release'], root), read_audio(rows['separate'], root))

    def test_single_six_channel_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'train', root / 'prepared'
            (source / 'Audio_AEC').mkdir(parents=True)
            (source / 'DOC_AEC').mkdir()
            audio = np.zeros((16000, 6), dtype=np.float32)
            audio[8000] = np.arange(6) / 10
            sf.write(source / 'Audio_AEC/S001.wav', audio, 16000, subtype='PCM_16')
            (source / 'DOC_AEC/S001.TextGrid').write_text(
                'item [1]:\nclass = "IntervalTier"\nname = "P001"\nintervals: size = 1\n'
                'intervals [1]:\nxmin = 0.2\nxmax = 0.8\ntext = "你好"\n'
            )
            prepare(SimpleNamespace(split_dir=source, output_dir=output))
            row = load_csv(output / 'sessions.csv')['S001']
            self.assertFalse(row.get('audio_channels'))
            np.testing.assert_allclose(read_audio(row, output), audio, atol=1 / 32768)
            imported = root / 'track2'
            prepare_track2(SimpleNamespace(split_dir=output, output_dir=imported))
            imported_row = load_csv(imported / 'sessions.csv')['S001']
            self.assertFalse(imported_row.get('audio_channels'))
            self.assertEqual(imported_row['clip_speakers'], '1')
            np.testing.assert_array_equal(read_audio(row, output), read_audio(imported_row, imported))
            self.assertEqual(
                json.loads((output / 'reference.seglst.json').read_text()),
                json.loads((imported / 'reference.seglst.json').read_text()),
            )

    def test_boundary_audit_retains_long_sessions_and_vad_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'sessions.csv'
            manifest.write_text(
                'id,audio_filepath,audio_duration_sec,mfa_speech_start,mfa_speech_end\n'
                'a,a.wav,600,0.5,599\nb,b.wav,700,0.5,699\n'
            )
            before = manifest.read_bytes()
            args = SimpleNamespace(manifest_csv=manifest, data_root=root, output_csv=root / 'boundaries.csv')
            with patch('robohead_baseline.utils.latency_local.read_audio', return_value=np.zeros((1600, 6))), patch(
                'robohead_baseline.utils.latency_local._vad_bounds', side_effect=[(0.9, 598.5), None]
            ):
                audit_boundaries(args)
            audited = load_csv(args.output_csv)
            self.assertEqual(set(audited), {'a', 'b'})
            self.assertEqual(float(audited['a']['boundary_error_s']), 0.5)
            self.assertEqual(audited['b']['vad_status'], 'no_speech')
            self.assertEqual(manifest.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
