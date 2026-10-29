"""Score full-meeting character transcripts with MeetEval tcpWER."""
import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from robohead_baseline.utils.audio import write_json
from robohead_baseline.utils.protocol import file_hash, load_csv, normalize_text, validate_snapshot


def score(args):
    manifest = load_csv(args.manifest_csv)
    references = json.loads(args.reference_json.read_text(encoding='utf-8'))
    hypotheses = _read_predictions(args, manifest)
    _validate_references(references, manifest)
    regions = _read_regions(args.scoring_uem, manifest) if args.scoring_uem else None
    if regions:
        references = restrict_regions(references, regions)
        hypotheses = restrict_regions(hypotheses, regions)

    with tempfile.TemporaryDirectory(prefix='robohead_score_') as directory:
        work = SimpleNamespace(**vars(args))
        work.output_dir = Path(directory)
        _write_segments(work.output_dir / 'ref.seglst.json', references, manifest)
        _write_segments(work.output_dir / 'hyp.seglst.json', hypotheses, manifest)
        _run_meeteval(work)
        per_recording = json.loads((work.output_dir / 'per_recording.json').read_text())
        result = json.loads((work.output_dir / 'tcpcer.json').read_text())
    if set(per_recording) != set(manifest):
        raise ValueError('MeetEval returned an incomplete recording set')
    provenance = {
        'metric': 'tcpCER',
        'collar': args.collar,
        'reference_timing': 'character_based',
        'hypothesis_timing': 'character_based_points',
        'manifest_sha256': file_hash(args.manifest_csv),
        'reference_sha256': file_hash(args.reference_json),
        'partial_json_sha256': file_hash(args.partial_json) if args.partial_json else None,
        'predict_csv_sha256': file_hash(args.predict_csv) if args.predict_csv else None,
        'hypothesis_json_sha256': file_hash(args.hypothesis_json) if getattr(args, 'hypothesis_json', None) else None,
        'scorer_sha256': file_hash(__file__),
    }
    if args.scoring_uem:
        provenance['asr_uem_sha256'] = file_hash(args.scoring_uem)
    result['provenance'] = provenance
    return result


def restrict_regions(rows, regions):
    result = []
    for row in rows:
        start, end = float(row['start_time']), float(row['end_time'])
        intervals = regions[row['session_id']]
        if any(a <= start and end <= b for a, b in intervals):
            result.append(row)
            continue
        tokens = normalize_text(row['words'])
        if not tokens or end <= start:
            continue
        width = (end - start) / len(tokens)
        for a, b in intervals:
            selected = [token for i, token in enumerate(tokens) if a <= start + (i + 0.5) * width < b]
            if selected:
                result.append(dict(row, start_time=max(start, a), end_time=min(end, b), words=''.join(selected)))
    return result


def _read_predictions(args, manifest):
    if getattr(args, 'hypothesis_json', None):
        rows = json.loads(args.hypothesis_json.read_text(encoding='utf-8'))
        if not isinstance(rows, list):
            raise ValueError('Expected a SegLST JSON array')
        predictions = {}
        for row in rows:
            uid = row['session_id']
            entry = predictions.setdefault(uid, {'text': '', 'segments': []})
            segment = {key: row[key] for key in ('speaker', 'start_time', 'end_time', 'words')}
            entry['segments'].append(segment)
            entry['text'] += segment['words']
    elif args.partial_json:
        records = json.loads(args.partial_json.read_text(encoding='utf-8'))
        predictions = {}
        for uid, record in records.items():
            events = record['events']
            finals = [event for event in events if event['event'] == 'final_visible']
            if len(finals) != 1 or events[-1]['event'] != 'final_visible':
                raise ValueError(f'{uid}: expected one terminal final result')
            predictions[uid] = finals[0]
    else:
        predictions = {
            uid: {'text': row['raw_hypos'], 'segments': json.loads(row['segments'])}
            for uid, row in load_csv(args.predict_csv).items()
        }
    if set(predictions) != set(manifest):
        raise ValueError('Prediction IDs differ from the manifest')
    hypotheses = []
    for uid, result in predictions.items():
        validate_snapshot(result, float(manifest[uid]['audio_duration_sec']))
        hypotheses.extend(dict(row, session_id=uid) for row in result['segments'])
    return hypotheses


def _validate_references(rows, manifest):
    if not isinstance(rows, list) or {row['session_id'] for row in rows} != set(manifest):
        raise ValueError('Reference IDs differ from the manifest')
    for row in rows:
        uid, speaker = row['session_id'], row['speaker']
        if not isinstance(speaker, str) or not speaker or any(c.isspace() for c in uid + speaker):
            raise ValueError('Invalid reference recording or speaker ID')
        start, end = float(row['start_time']), float(row['end_time'])
        duration = float(manifest[uid]['audio_duration_sec'])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration + 1 / 16000:
            raise ValueError(f'{uid}: invalid reference segment bounds')
        if not isinstance(row['words'], str):
            raise ValueError('Reference words must be a string')


def _read_regions(path, manifest):
    regions = {uid: [] for uid in manifest}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        uid, channel, start, end = line.split()
        if uid not in regions or channel != '1':
            raise ValueError('UEM recording/channel differs from the manifest')
        start, end = float(start), float(end)
        if not 0 <= start < end <= float(manifest[uid]['audio_duration_sec']):
            raise ValueError('UEM interval exceeds audio bounds')
        regions[uid].append((start, end))
    for uid, intervals in regions.items():
        intervals.sort()
        if not intervals or any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError(f'{uid}: missing or overlapping UEM intervals')
    return regions


def _write_segments(path, rows, manifest):
    segments, found = [], set()
    for row in rows:
        uid = row['session_id']
        words = ' '.join(normalize_text(row['words']))
        start, end = float(row['start_time']), float(row['end_time'])
        if not 0 <= start <= end <= float(manifest[uid]['audio_duration_sec']) + 1 / 16000:
            raise ValueError(f'{uid}: invalid scored segment bounds')
        if words:
            segments.append(
                {'session_id': uid, 'speaker': row['speaker'], 'start_time': start, 'end_time': end, 'words': words,}
            )
            found.add(uid)
    for uid in sorted(set(manifest) - found):
        segments.append({'session_id': uid, 'speaker': 'empty', 'start_time': 0.0, 'end_time': 1 / 16000, 'words': ''})
    write_json(path, segments)


def _run_meeteval(args):
    command = [
        args.meeteval_wer,
        'tcpwer',
        '-r',
        str(args.output_dir / 'ref.seglst.json'),
        '-h',
        str(args.output_dir / 'hyp.seglst.json'),
        '--collar',
        f'{float(args.collar):g}',
        '--ref-pseudo-word-timing',
        'character_based',
        '--hyp-pseudo-word-timing',
        'character_based_points',
        '--average-out',
        str(args.output_dir / 'tcpcer.json'),
        '--per-reco-out',
        str(args.output_dir / 'per_recording.json'),
    ]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-csv', required=True)
    parser.add_argument('--reference-json', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--predict-csv')
    source.add_argument('--hypothesis-json', type=Path, help='Score a Track 1 SegLST result file')
    source.add_argument('--partial-json', type=Path, help='Score final transcripts from the paced capture')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--collar', type=float, default=5.0)
    parser.add_argument('--meeteval-wer', default='meeteval-wer')
    parser.add_argument('--scoring-uem', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.collar) or args.collar < 0:
        parser.error('--collar must be finite and nonnegative')
    accuracy = score(args)
    write_json(
        args.output_dir / 'scores.json',
        dict(status='complete', tcpcer_pct=accuracy['error_rate'] * 100, accuracy=accuracy),
    )
