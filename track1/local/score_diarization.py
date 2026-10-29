"""Score predicted validation RTTMs against human reference activity."""

import argparse
import json
from pathlib import Path
import tempfile

import meeteval

from robohead_baseline.utils.audio import write_json


def score(args):
    references = json.loads(args.reference_json.read_text())
    lines = [
        f"SPEAKER {r['session_id']} 1 {r['start_time']:.6f} "
        f"{r['end_time'] - r['start_time']:.6f} <NA> <NA> {r['speaker']} <NA> <NA>\n"
        for r in references
    ]
    with tempfile.TemporaryDirectory(prefix='robohead_der_') as directory:
        reference = Path(directory) / 'reference.rttm'
        reference.write_text(''.join(lines))
        scores = {}
        for collar in (0.0, 0.25):
            results = meeteval.der.md_eval_22(reference, args.predicted_rttm, collar=collar)
            errors = sum(
                r.falarm_speaker_time + r.missed_speaker_time + r.speaker_error_time for r in results.values()
            )
            length = sum(r.scored_speaker_time for r in results.values())
            scores[f'DER@{collar:g}'] = 100 * float(errors / length)
    write_json(args.output_dir / 'der.json', scores)
    print(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-json', type=Path, required=True)
    parser.add_argument('--predicted-rttm', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    score(parser.parse_args())
