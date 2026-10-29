"""Build four-stream training examples and prediction-guided validation windows."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from robohead_baseline.track1.local.finetuning.common import write_jsonl
from robohead_baseline.utils.audio import write_json
from robohead_baseline.utils.protocol import normalize_text


def training_rows(references, audio, minimum=8.0, maximum=20.0, gap=1.5):
    sources = {r['session_id']: r for r in audio}
    grouped, activity = defaultdict(list), defaultdict(list)
    for row in references:
        if normalize_text(row['words']):
            grouped[row['session_id'], row['speaker']].append(row)
            activity[row['session_id']].append(row)
    rows = []
    for (uid, speaker), members in sorted(grouped.items()):
        group = []
        for row in sorted(members, key=lambda r: r['start_time']):
            if group and (
                row['start_time'] < group[-1]['end_time']
                or row['start_time'] - group[-1]['end_time'] > gap
                or row['end_time'] - group[0]['start_time'] > maximum
            ):
                rows.extend(_long(group, sources[uid], activity[uid], minimum))
                group = []
            if 0.04 <= row['end_time'] - row['start_time'] <= maximum:
                rows.extend(_variants([row], sources[uid], activity[uid]))
                group.append(row)
            else:
                rows.extend(_long(group, sources[uid], activity[uid], minimum))
                group = []
        rows.extend(_long(group, sources[uid], activity[uid], minimum))
    random.Random(42).shuffle(rows)
    return rows


def prepare(args):
    train_refs = json.loads((args.data_dir / 'train/reference.seglst.json').read_text())
    val_refs = json.loads((args.data_dir / 'val/reference.seglst.json').read_text())
    if {r['session_id'] for r in train_refs} & {r['session_id'] for r in val_refs}:
        raise ValueError('Training and validation sessions overlap')
    audio = json.loads((args.frontend_dir / 'train/audio.json').read_text())
    val = json.loads((args.frontend_dir / 'val/segments.json').read_text())
    if {r['session_id'] for r in audio} != {r['session_id'] for r in train_refs}:
        raise ValueError('Training enhancements are incomplete')
    train = training_rows(train_refs, audio)
    if not train or not val:
        raise ValueError('Empty training or validation examples')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / 'train.jsonl', train)
    write_jsonl(args.output_dir / 'val.jsonl', val)
    write_json(args.output_dir / 'val_reference.json', val_refs)
    print(f'Train: {len(train)}; val: {len(val)}; selection: complete-val tcpCER')


def _long(group, source, activity, minimum):
    if len(group) < 2 or group[-1]['end_time'] - group[0]['start_time'] < minimum:
        return []
    if sum(r['end_time'] - r['start_time'] for r in group) < 0.6 * (group[-1]['end_time'] - group[0]['start_time']):
        return []
    rows = []
    for row in _variants(group, source, activity):
        for speed in (0.9, 1.0, 1.1):
            rows.append(dict(row, id=row['id'] + f'_sp{speed:g}', speed=speed, duration=row['duration'] / speed))
    return rows


def _variants(group, source, activity):
    first, last = group[0], group[-1]
    start, end = first['start_time'], last['end_time']
    speaker, uid = first['speaker'], first['session_id']
    words = ''.join(normalize_text(r['words']) for r in group)
    key = f'{uid}_{speaker}_{round(start * 16000):09d}_{round(end * 16000):09d}'
    base = dict(
        session_id=uid,
        subset='train',
        speaker=speaker,
        audio_channel=0,
        audio_interval_samples=[round(start * 16000), round(end * 16000)],
        duration=end - start,
        reference=words,
        text='language Chinese<asr_text>' + words,
    )
    variants = [('gss_gt', source['gss_gt'][speaker])]
    overlap = any(r['speaker'] != speaker and r['start_time'] < end and r['end_time'] > start for r in activity)
    if not overlap:
        variants += [(name, source[name]) for name in ('raw', 'wpe_offline', 'wpe_online')]
    return [dict(base, id=key + '_' + name, frontend=name, audio=path) for name, path in variants]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--frontend-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args())
