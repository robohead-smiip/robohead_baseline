"""Score submissions with the fixed accuracy and timing protocol."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from robohead_baseline.track2.local.validate import validate
from robohead_baseline.utils.audio import write_json
from robohead_baseline.utils.score import score


def evaluate(args):
    phase = getattr(args, 'phase', 'all')
    if phase == 'latency':
        if args.track != 2:
            raise ValueError('Latency scoring requires Track 2')
        saved = json.loads((args.output_dir / 'scores.json').read_text())
        accuracy = saved['accuracy']
    else:
        accuracy = score(
            SimpleNamespace(
                manifest_csv=args.manifest_csv,
                reference_json=args.reference_json,
                predict_csv=args.output_dir / 'predict.csv' if args.track == 2 else None,
                hypothesis_json=(args.hypothesis_json or args.output_dir / 'hypothesis.seglst.json')
                if args.track == 1
                else None,
                partial_json=None,
                collar=5.0,
                meeteval_wer='meeteval-wer',
                scoring_uem=args.scoring_uem,
            )
        )
    if args.track == 1 or phase == 'accuracy':
        summary = dict(
            status='complete' if args.track == 1 else 'accuracy_only', tcpcer_pct=accuracy['error_rate'] * 100,
        )
    else:
        summary = validate(
            SimpleNamespace(
                manifest_csv=args.manifest_csv,
                fast_csv=args.output_dir / 'predict.csv',
                events_json=args.output_dir / 'partial_results.json',
            ),
            accuracy,
        )
    summary['accuracy'] = accuracy
    write_json(args.output_dir / 'scores.json', summary)
    columns, keys = ['tcpCER (%)'], ['tcpcer_pct']
    if args.track == 2 and phase != 'accuracy':
        columns += ['ED P95 (s)', 'TTFST P95 (s)', 'RTF', 'RTFx']
        keys += ['ed_p95_s', 'ttfst_p95_s', 'rtf', 'rtfx']
    values = ['—' if summary.get(k) is None else f'{summary[k]:.4f}' for k in keys]
    table = '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---:'] * len(keys)) + ' |\n'
    table += '| ' + ' | '.join(values) + ' |\n'
    print(table)
    if summary['status'] == 'reject' and not args.report_only:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', type=int, choices=(1, 2), required=True)
    parser.add_argument('--phase', choices=('accuracy', 'latency', 'all'), default='all')
    parser.add_argument('--manifest-csv', required=True)
    parser.add_argument('--reference-json', type=Path, required=True)
    parser.add_argument('--hypothesis-json', type=Path, help='Track 1 submitted SegLST file')
    parser.add_argument('--scoring-uem', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report-only', action='store_true')
    evaluate(parser.parse_args())
