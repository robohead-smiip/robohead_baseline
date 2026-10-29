"""Score unpacked submission results against reference files."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from robohead_baseline.utils.submission import evaluate
from robohead_baseline.utils.audio import write_json


def score_submission(args):
    args.input_dir, args.output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    (args.output_dir / 'scores.json').unlink(missing_ok=True)
    work = args.output_dir / 'details'
    status = evaluate(
        SimpleNamespace(track=args.track, data=args.input_dir / 'ref', output=work, scoring_uem=None),
        args.input_dir / 'res',
    )
    if status:
        raise SystemExit(status)
    result = json.loads((work / 'scores.json').read_text())
    keys = ['tcpcer_pct'] if args.track == 1 else ['tcpcer_pct', 'ed_p95_s', 'ttfst_p95_s', 'rtf', 'rtfx']
    write_json(args.output_dir / 'scores.json', {key: result[key] for key in keys})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', type=int, choices=(1, 2), required=True)
    parser.add_argument('--input-dir', type=Path, default=Path('/app/input'))
    parser.add_argument('--output-dir', type=Path, default=Path('/app/output'))
    score_submission(parser.parse_args())
