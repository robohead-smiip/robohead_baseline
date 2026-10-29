"""Check full-session timing, transcript consistency and latency limits."""

import argparse
import json
import math
from pathlib import Path

from robohead_baseline.utils.audio import write_json
from robohead_baseline.utils.protocol import compute_latency, file_hash, load_csv, normalize_text, validate_snapshot


def validate(args, accuracy):
    manifest = load_csv(args.manifest_csv)
    fast = load_csv(args.fast_csv)
    captured = json.loads(args.events_json.read_text())
    latency = compute_latency(args.events_json, args.manifest_csv)
    reasons = []
    if set(fast) != set(manifest) or set(captured) != set(manifest):
        reasons.append('Prediction IDs differ from the manifest')
    mismatches = []
    for uid in sorted(set(manifest) & set(fast) & set(captured)):
        duration = float(manifest[uid]['audio_duration_sec'])
        try:
            snapshot = dict(text=fast[uid]['raw_hypos'], segments=json.loads(fast[uid]['segments']))
            validate_snapshot(snapshot, duration)
            final = captured[uid]['events'][-1]
            validate_snapshot(final, duration)
            if normalize_text(snapshot['text']) != normalize_text(final['text']):
                mismatches.append(uid)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            reasons.append(f'{uid}: {error}')
    if mismatches:
        reasons.append('Fast and real-time final transcripts differ')
    if latency['status'] != 'complete' or latency['n_capture_verified'] != len(manifest):
        reasons.append('Missing or invalid evaluator timing records')
    provenance = accuracy['provenance']
    expected = dict(manifest_sha256=file_hash(args.manifest_csv), predict_csv_sha256=file_hash(args.fast_csv),)
    if any(provenance.get(key) != value for key, value in expected.items()):
        reasons.append('Accuracy was not scored from the complete fast-pass results')
    rtf = rtfx = None
    try:
        elapsed = [float(fast[uid]['processing_time_sec']) for uid in manifest]
        if any(not math.isfinite(value) or value <= 0 for value in elapsed):
            raise ValueError('Invalid processing time')
        duration = sum(float(row['audio_duration_sec']) for row in manifest.values())
        rtf, rtfx = sum(elapsed) / duration, duration / sum(elapsed)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        reasons.append('Missing or invalid fast-pass processing times')
    error_rate = float(accuracy['error_rate'])
    ed = latency['ed_sec']['p95']
    ttfst = latency['ttfst_sec']['p95']
    if not math.isfinite(error_rate) or error_rate < 0:
        reasons.append('Invalid tcpCER')
    for name, value, limit in [('ED', ed, 1.5), ('TTFST', ttfst, 4.5)]:
        if value is None or not math.isfinite(value):
            reasons.append(f'Invalid {name} P95')
        elif value > limit:
            reasons.append(f'{name} P95 exceeds {limit} s')
    result = dict(
        status='reject' if reasons else 'pass',
        ranking_metric='tcpCER',
        ranking_eligible=not reasons,
        ed_limit_s=1.5,
        ttfst_limit_s=4.5,
        tcpcer_pct=100 * error_rate if math.isfinite(error_rate) else None,
        ed_p95_s=ed,
        ttfst_p95_s=ttfst,
        rtf=rtf,
        rtfx=rtfx,
        transcript_mismatches=mismatches,
        accuracy_files=len(manifest),
        latency_files=len(manifest),
        reasons=reasons,
        latency_issues=latency['issues'],
    )
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-csv', required=True)
    parser.add_argument('--predict-csv', dest='fast_csv', required=True)
    parser.add_argument('--partial-json', dest='events_json', type=Path, required=True)
    parser.add_argument('--scores-json', type=Path, required=True)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    saved = json.loads(args.scores_json.read_text())
    result = validate(args, saved['accuracy'])
    saved.update(result)
    write_json(args.scores_json, saved)
    print(json.dumps(result, indent=2))
    if result['status'] != 'pass' and not args.report_only:
        raise SystemExit(1)
