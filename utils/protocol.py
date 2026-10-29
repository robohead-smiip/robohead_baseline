"""Compute session-level TTFST and ED from evaluator-recorded streaming results."""
import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

PROTOCOL = 'roboheard_session_baseline_v1'


def compute_latency(events_path, manifest_csv):
    records = json.loads(Path(events_path).read_text(encoding='utf-8'))
    manifest = load_csv(manifest_csv)
    if not isinstance(records, dict):
        raise ValueError('Expected events keyed by session ID')

    issues = [f'{uid}: unexpected session' for uid in sorted(set(records) - set(manifest))]
    sessions = []
    for uid, reference in manifest.items():
        try:
            result = _session_latency(records[uid], reference)
            sessions.append({'id': uid, 'status': 'ok', **result})
        except (KeyError, TypeError, ValueError, IndexError) as error:
            issues.append(f'{uid}: {error}')
            sessions.append({'id': uid, 'status': 'invalid', 'error': str(error)})

    valid = [row for row in sessions if row['status'] == 'ok']
    complete = len(valid) == len(manifest) and not issues
    ed = [row['ed_s'] for row in valid] if complete else []
    ttfst = [row['ttfst_s'] for row in valid] if complete else []
    return {
        'protocol': PROTOCOL,
        'status': 'complete' if complete else 'invalid_or_incomplete',
        'n_files_total': len(manifest),
        'n_files_valid': len(valid),
        'n_capture_verified': sum(row['capture_verified'] for row in valid),
        'ed_sec': _summary(ed),
        'ttfst_sec': _summary(ttfst),
        'issues': issues,
        'files': sessions,
        'provenance': {
            'manifest_sha256': file_hash(manifest_csv),
            'partial_json_sha256': file_hash(events_path),
            'scorer_sha256': file_hash(__file__),
        },
    }


def normalize_text(text):
    text = re.sub('<[^>]*>', '', unicodedata.normalize('NFKC', text).lower())
    return ''.join(char for char in text if unicodedata.category(char)[0] in 'LN')


def load_csv(path):
    csv.field_size_limit(100_000_000)
    rows = {}
    with Path(path).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            uid = row['id']
            if not uid or any(char.isspace() for char in uid) or uid in rows:
                raise ValueError('Invalid or duplicate CSV ID')
            rows[uid] = row
    if not rows:
        raise ValueError('Empty CSV')
    return rows


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_snapshot(result, duration=None):
    if not isinstance(result, dict) or not isinstance(result.get('text'), str):
        raise ValueError('Expected a snapshot with text and segments')
    if not isinstance(result.get('segments'), list):
        raise ValueError('Expected a segment list')
    previous = -1.0
    for row in result['segments']:
        if not isinstance(row, dict) or not isinstance(row.get('speaker'), str) or (not row['speaker']):
            raise ValueError('Expected a nonempty speaker ID')
        if any(char.isspace() for char in row['speaker']) or not isinstance(row.get('words'), str):
            raise ValueError('Invalid speaker ID or segment text')
        start, end = _number(row['start_time']), _number(row['end_time'])
        if not 0 <= start < end or start < previous:
            raise ValueError('Expected chronological segments with positive duration')
        if duration is not None and end > duration + 1 / 16000:
            raise ValueError('Segment exceeds audio duration')
        previous = start
    if normalize_text(result['text']) != normalize_text(''.join(row['words'] for row in result['segments'])):
        raise ValueError('Snapshot text and segments disagree')


def _session_latency(record, reference):
    duration = _number(reference['audio_duration_sec'])
    speech_start = _number(reference['mfa_speech_start'])
    if reference.get('evaluation_unit') != 'session' or not 0 <= speech_start < duration:
        raise ValueError('Expected a session with a valid aligned speech onset')

    timing = record['timing']
    origin = _number(timing['audio_send_start_time'])
    events = _events(record, origin)
    visible = _number(timing['final_visible_time'])
    audio_end = origin + duration
    if abs(visible - events[-1]['time']) > 1e-6:
        raise ValueError('Final timestamp differs from the receiver event')
    if abs(_number(timing['audio_end_oracle_time']) - audio_end) > 1 / 16000:
        raise ValueError('Audio end differs from the manifest duration')
    if visible < audio_end:
        raise ValueError('Final result precedes end of input')
    if 'capture' in record:
        _check_capture(record, origin, duration, visible)

    stable_time = _stable_partial_time(events)
    return {
        'ed_s': visible - audio_end,
        'ttfst_s': stable_time - (origin + speech_start),
        'final_visible_time': visible,
        'first_stable_partial_time': stable_time,
        'audio_end_oracle_time': audio_end,
        'reference_speech_start_time': origin + speech_start,
        'capture_verified': 'capture' in record,
    }


def _number(value):
    if isinstance(value, bool):
        raise ValueError('Expected finite number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Expected finite number')
    return value


def _events(record, origin):
    events, previous = [], origin
    for event in record['events']:
        stamp = _number(event['time'])
        if stamp < previous:
            raise ValueError('Events are not ordered on the evaluator clock')
        if event['event'] not in ('partial_callback', 'partial_return', 'final_visible'):
            raise ValueError('Unknown transcript event')
        if not isinstance(event['text'], str):
            raise ValueError('Transcript must be a string')
        events.append(dict(event, time=stamp))
        previous = stamp
    if (
        not events
        or events[-1]['event'] != 'final_visible'
        or sum(event['event'] == 'final_visible' for event in events) != 1
    ):
        raise ValueError('Require exactly one terminal final_visible event')
    return events


def _check_capture(record, origin, duration, visible):
    capture, timing = record['capture'], record['timing']
    if capture['realtime'] is not True or capture['sample_rate'] != 16000:
        raise ValueError('Expected 1x PCM capture at 16 kHz')
    if timing.get('clock') != 'evaluator_monotonic':
        raise ValueError('Expected evaluator monotonic clock')
    if capture.get('packet_schedule') != 'immediate_first_100ms':
        raise ValueError('Expected immediate-first 100 ms packet schedule')
    previous_sample, previous_time = 0, origin
    total = round(duration * 16000)
    for packet in capture['audio_packets']:
        end, stamp = packet['end_sample'], _number(packet['sent_time'])
        if isinstance(end, bool) or not isinstance(end, int) or end != min(previous_sample + 1600, total):
            raise ValueError('Invalid packet size or sample order')
        if previous_sample == 0 and abs(stamp - origin) > 1e-06:
            raise ValueError('Input origin must be the first packet send time')
        if end > total or stamp < previous_time or stamp + 1e-06 < origin + previous_sample / 16000:
            raise ValueError('Packet sent early, out of order or past EOF')
        previous_sample, previous_time = end, stamp
    eof = _number(timing['input_finished_time'])
    scheduled_eof = origin + math.ceil(total / 1600) * 0.1
    if previous_sample != total or not previous_time <= eof <= visible or eof + 1e-06 < scheduled_eof:
        raise ValueError('Incomplete input or invalid EOF/final order')


def _stable_partial_time(events):
    final = events[-1]
    first_character = normalize_text(final['text'])[:1]
    stable_time = final['time']
    if not first_character:
        return stable_time
    for event in reversed(events[:-1]):
        if normalize_text(event['text'])[:1] != first_character:
            break
        stable_time = event['time']
    return stable_time


def _percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    rank = q / 100 * (len(values) - 1)
    lo, hi = int(rank), min(int(rank) + 1, len(values) - 1)
    fraction = rank - lo
    return values[lo] * (1 - fraction) + values[hi] * fraction


def _summary(values):
    return dict(
        count=len(values), p50=_percentile(values, 50), p90=_percentile(values, 90), p95=_percentile(values, 95)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial-json', required=True)
    parser.add_argument('--manifest-csv', required=True)
    parser.add_argument('--out-json', type=Path)
    args = parser.parse_args()
    result = compute_latency(args.partial_json, args.manifest_csv)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
