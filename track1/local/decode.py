"""Run an offline submission and save meeting-level predictions."""

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import soundfile as sf

from robohead_baseline.utils.audio import read_audio, write_json
from robohead_baseline.utils.protocol import load_csv, validate_snapshot


def run(args):
    manifest = load_csv(args.manifest_csv)
    if args.config:
        os.environ['ROBOHEAD_CONFIG'] = str(args.config.resolve())
    sys.path.insert(0, str(args.submission_dir.resolve()))
    spec = importlib.util.spec_from_file_location('submitted_model', args.submission_dir / 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Model()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    try:
        for uid, row in manifest.items():
            audio = read_audio(row, args.data_root)
            with tempfile.TemporaryDirectory(prefix='robohead_input_') as temporary:
                wav = Path(temporary) / 'session.wav'
                sf.write(wav, audio, 16000, subtype='PCM_16')
                result = model.predict(str(wav))
            result = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
            validate_snapshot(result, float(row['audio_duration_sec']))
            rows = result['segments'] or [dict(speaker='empty', start_time=0.0, end_time=1 / 16000, words='')]
            segments.extend(dict(segment, session_id=uid) for segment in rows)
            write_json(args.output_dir / 'hypothesis.seglst.json', segments)
            print(f'[offline] {uid}', flush=True)
    finally:
        if hasattr(model, 'close'):
            model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--manifest-csv', required=True)
    parser.add_argument('--data-root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
