"""Prepare complete validation sessions using fixed conversation boundaries."""

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf

from robohead_baseline.utils.annotations import clean_transcript
from robohead_baseline.utils.audio import read_audio, write_json
from robohead_baseline.utils.prepare_data import prepare as prepare_sessions
from robohead_baseline.utils.protocol import load_csv
from robohead_baseline.utils.score import _validate_references


def prepare(args):
    source = args.split_dir.resolve()
    output = args.output_dir.resolve()
    leading = getattr(args, 'leading_seconds', 0.5)
    trailing = getattr(args, 'trailing_seconds', 1.0)
    if leading < 0 or trailing < 0:
        raise ValueError('Context durations must be nonnegative')
    if not (source / 'sessions.csv').is_file():
        prepare_sessions(SimpleNamespace(split_dir=source, output_dir=output))
        source = output
    manifest = load_csv(source / 'sessions.csv')
    crop_file = getattr(args, 'crop_file', None)
    crops = _load_crops(crop_file, manifest) if crop_file else {}
    references = json.loads((source / 'reference.seglst.json').read_text())
    references = [dict(row, words=clean_transcript(row['words'])) for row in references]
    references = [row for row in references if row['words']]
    _validate_references(references, manifest)
    (output / 'audio').mkdir(parents=True, exist_ok=True)
    rows, refs = [], []
    for uid, row in manifest.items():
        audio = read_audio(row, source)
        turns = [r for r in references if r['session_id'] == uid]
        start, end = crops.get(uid, (min(r['start_time'] for r in turns), max(r['end_time'] for r in turns)))
        if not 0 <= start < end <= len(audio) / 16000:
            raise ValueError(f'{uid}: invalid crop bounds')
        kept = [r for r in turns if r['end_time'] > start and r['start_time'] < end]
        if not kept:
            raise ValueError(f'{uid}: no reference speech inside crop')
        if any(r['start_time'] < start - 1e-7 or r['end_time'] > end + 1e-7 for r in kept):
            raise ValueError(f'{uid}: crop cuts an annotated turn')
        before = max((r['end_time'] for r in turns if r['end_time'] <= start), default=0)
        after = min((r['start_time'] for r in turns if r['start_time'] >= end), default=len(audio) / 16000)
        first = round(max(0, start - leading, before) * 16000)
        last = round(min(len(audio) / 16000, end + trailing, after) * 16000)
        sf.write(output / 'audio' / f'{uid}.wav', audio[first:last], 16000, subtype='PCM_16')
        rows.append(
            dict(
                id=uid,
                audio_filepath=f'audio/{uid}.wav',
                audio_channels='',
                audio_duration_sec=(last - first) / 16000,
                mfa_speech_start='',
                mfa_speech_end='',
                evaluation_unit='session',
                source_id=uid,
                source_start=first / 16000,
                source_end=last / 16000,
                clip_speakers=len({r['speaker'] for r in kept}),
            )
        )
        refs.extend(
            dict(r, start_time=r['start_time'] - first / 16000, end_time=r['end_time'] - first / 16000) for r in kept
        )
    _validate_references(refs, {r['id']: r for r in rows})
    with (output / 'sessions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'reference.seglst.json', refs)
    print(f'{len(rows)} trimmed sessions; {sum(r["audio_duration_sec"] for r in rows) / 3600:.3f} h')


def _load_crops(path, manifest):
    crops = {}
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            uid = row['session_id']
            if not uid or uid in crops:
                raise ValueError(f'Empty or duplicate session in crop file: {uid}')
            bounds = float(row['speech_start']), float(row['speech_end'])
            if not all(math.isfinite(value) for value in bounds):
                raise ValueError(f'{uid}: invalid crop bounds')
            crops[uid] = bounds
    if set(crops) != set(manifest):
        raise ValueError('Crop file session IDs must match the input manifest')
    return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--crop-file', type=Path)
    parser.add_argument('--leading-seconds', type=float, default=0.5)
    parser.add_argument('--trailing-seconds', type=float, default=1.0)
    prepare(parser.parse_args())
