"""Audit aligned speech boundaries for every validation session without filtering."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from robohead_baseline.utils.audio import read_audio
from robohead_baseline.utils.protocol import load_csv


def prepare(args):
    import webrtcvad

    rows = load_csv(args.manifest_csv)
    audit = []
    for uid, original in rows.items():
        row = dict(original)
        seconds = float(row['audio_duration_sec'])
        start = float(row['mfa_speech_start'])
        end = float(row['mfa_speech_end'])
        if not all(math.isfinite(x) for x in (seconds, start, end)) or not 0 <= start < end <= seconds:
            raise ValueError(f'{uid}: invalid aligned speech bounds')
        pcm = read_audio(row, args.data_root)[:, 0]
        pcm = np.clip(np.rint(pcm * 32768), -32768, 32767).astype('<i2')
        bounds = _vad_bounds(pcm, webrtcvad.Vad(3))
        row['vad_status'] = 'ok' if bounds else 'no_speech'
        if bounds:
            row.update(
                vad_speech_start=bounds[0],
                vad_speech_end=bounds[1],
                vad_start_error_s=bounds[0] - start,
                vad_end_error_s=bounds[1] - end,
                boundary_error_s=max(abs(bounds[0] - start), abs(bounds[1] - end)),
                leading_s=bounds[0],
                tail_s=seconds - bounds[1],
            )
        audit.append(row)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_csv, audit)
    print(f'{len(audit)} sessions audited; all sessions retained for both inference passes')


def _vad_bounds(pcm, vad):
    flags = [vad.is_speech(pcm[i : i + 480].tobytes(), 16000) for i in range(0, len(pcm) - 479, 480)]
    intervals, start = [], None
    for index, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if intervals and start - intervals[-1][1] <= 10:
                intervals[-1] = intervals[-1][0], index
            else:
                intervals.append((start, index))
            start = None
    kept = [(a, b) for a, b in intervals if b - a >= 8] or intervals
    return (kept[0][0] * 0.03, kept[-1][1] * 0.03) if kept else None


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-csv', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, required=True)
    prepare(parser.parse_args())
