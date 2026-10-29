"""Prepare the fixed training and validation partition from released AEC recordings."""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from robohead_baseline.utils.annotations import clean_transcript, read_textgrid
from robohead_baseline.utils.audio import write_json


def prepare(args):
    root = args.split_dir.resolve()
    output = args.output_dir.resolve()
    microphones = defaultdict(dict)
    recordings = {}
    for path in sorted((root / 'Audio_AEC').glob('*.wav')):
        match = re.fullmatch(r'(S\d+)(?:_([0-5]))?', path.stem)
        if not match:
            raise ValueError(f'Unexpected recording name: {path.name}')
        uid, channel = match.groups()
        if (root.name == 'train' or getattr(args, 'partition_file', None)) and uid == 'S161':
            continue
        if channel is None:
            recordings[uid] = [path]
        else:
            microphones[uid][int(channel)] = path
    for uid, paths in microphones.items():
        if uid in recordings:
            continue
        if set(paths) != set(range(6)):
            raise ValueError(f'{uid}: expected microphone channels 0 to 5')
        recordings[uid] = [paths[i] for i in range(6)]
    if not recordings:
        raise ValueError(f'No AEC recordings found in {root / "Audio_AEC"}')
    durations = {}
    for uid, paths in recordings.items():
        info = [sf.info(path) for path in paths]
        channels_ok = (len(info) == 1 and info[0].channels == 6) or (
            len(info) == 6 and all(x.channels == 1 for x in info)
        )
        if not channels_ok or any(x.samplerate != 16000 or x.subtype != 'PCM_16' for x in info):
            raise ValueError(f'{uid}: expected six-channel WAV or six mono WAVs, 16 kHz PCM16')
        if not info[0].frames or len({x.frames for x in info}) != 1:
            raise ValueError(f'{uid}: empty or unsynchronized microphones')
        durations[uid] = info[0].duration

    references = []
    exclusions = json.loads((Path(__file__).with_name('splits') / 'annotation_exclusions.json').read_text())
    for uid in sorted(recordings):
        path = root / 'DOC_AEC' / f'{uid}.TextGrid'
        for speaker, intervals in read_textgrid(path).items():
            if re.fullmatch(r'P\d+', speaker):
                references.extend(_segments(uid, speaker, intervals, durations[uid], path))
    references = [
        row
        for row in references
        if not any(
            row['session_id'] == r['session_id']
            and row['speaker'] == r['speaker']
            and abs(row['start_time'] - r['start']) < 1 / 16000
            and abs(row['end_time'] - r['end']) < 1 / 16000
            for r in exclusions
        )
    ]
    if {row['session_id'] for row in references} != set(recordings):
        raise ValueError('Every session needs nonempty human annotations')
    references.sort(key=lambda row: (row['session_id'], row['start_time'], row['speaker']))
    partition = getattr(args, 'partition_file', None)
    if partition:
        with Path(partition).open() as stream:
            groups = {r['session_id']: r['subset'] for r in csv.DictReader(stream, delimiter='\t')}
        if set(groups) != set(recordings) or set(groups.values()) != {'train', 'val'}:
            raise ValueError('The released recordings must match the fixed train/val partition')
        speakers = {
            split: {r['speaker'] for r in references if groups[r['session_id']] == split} for split in ('train', 'val')
        }
        if speakers['train'] & speakers['val']:
            raise ValueError('Train and val speakers overlap')
        for split in ('train', 'val'):
            selected = {uid: paths for uid, paths in recordings.items() if groups[uid] == split}
            _write_split(
                output / split, selected, durations, [r for r in references if groups[r['session_id']] == split]
            )
    else:
        _write_split(output, recordings, durations, references)


def _write_split(output, recordings, durations, references):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'reference.seglst.json', references)
    rows = []
    for uid, channels in sorted(recordings.items()):
        paths = [os.path.relpath(path, output) for path in channels]
        rows.append(
            dict(
                id=uid,
                audio_filepath=paths[0],
                audio_channels=json.dumps(paths) if len(paths) > 1 else '',
                audio_duration_sec=durations[uid],
                mfa_speech_start='',
                mfa_speech_end='',
                evaluation_unit='session',
                clip_speakers=len({row['speaker'] for row in references if row['session_id'] == uid}),
            )
        )
    with (output / 'sessions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{output.name}: {len(rows)} sessions, {sum(durations[uid] for uid in recordings) / 3600:.3f} h')


def _segments(uid, speaker, intervals, duration, path):
    rows = []
    for start, end, words in intervals:
        words = clean_transcript(words)
        if not words:
            continue
        if not 0 <= start < end <= duration + 1 / 16000:
            raise ValueError(f'{path}: annotation exceeds audio bounds')
        rows.append(dict(session_id=uid, speaker=speaker, start_time=start, end_time=min(end, duration), words=words))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--partition-file', type=Path)
    prepare(parser.parse_args())
