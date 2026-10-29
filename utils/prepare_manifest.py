"""Create a session manifest from microphone paths and optional latency onsets."""

import argparse
import csv
import json
import math
import os
import re
import shlex
from pathlib import Path

import soundfile as sf

from robohead_baseline.utils.protocol import load_csv


def prepare(args):
    onsets = load_csv(args.onset_csv) if args.onset_csv else {}
    rows, seen = [], set()
    for line in args.wav_scp.read_text().splitlines():
        if not line.strip():
            continue
        uid, *paths = shlex.split(line)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', uid) or uid in seen or not paths:
            raise ValueError('Invalid or duplicate recording ID')
        seen.add(uid)
        paths = [str((args.data_root / path).resolve()) for path in paths]
        info = [sf.info(path) for path in paths]
        if any(i.samplerate != 16000 or i.subtype != 'PCM_16' or not i.frames for i in info):
            raise ValueError('Expected nonempty 16 kHz PCM16 WAV')
        channels_ok = (len(info) == 1 and info[0].channels == 6) or (
            len(info) == 6 and all(i.channels == 1 for i in info)
        )
        if len({i.frames for i in info}) != 1 or not channels_ok:
            raise ValueError('Expected one six-channel WAV or six synchronized mono WAVs')
        duration = info[0].frames / 16000
        onset = float(onsets[uid]['mfa_speech_start']) if onsets else ''
        if onsets and (not math.isfinite(onset) or not 0 <= onset < duration):
            raise ValueError(f'Invalid speech onset for {uid}')
        paths = [os.path.relpath(path, args.data_root.resolve()) for path in paths]
        rows.append(
            dict(
                id=uid,
                audio_filepath=paths[0],
                audio_channels=json.dumps(paths) if len(paths) > 1 else '',
                audio_duration_sec=duration,
                mfa_speech_start=onset,
                mfa_speech_end=onsets[uid].get('mfa_speech_end', '') if onsets else '',
                evaluation_unit='session',
            )
        )
    if not rows or args.onset_csv and seen != set(onsets):
        raise ValueError('Manifest and reference IDs must agree')
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav-scp', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('.'))
    parser.add_argument('--onset-csv', type=Path)
    parser.add_argument('--output-csv', type=Path, required=True)
    prepare(parser.parse_args())
