"""Score predicted-GSS Val transcripts against complete meeting references."""

import json
from decimal import Decimal
from pathlib import Path

from robohead_baseline.track1.local.finetuning.common import normalize


def meeting_scores(predictions, reference_file, collar=5.0):
    import meeteval

    reference = json.loads(Path(reference_file).read_text())
    reference = [dict(r, words=' '.join(normalize(r['words']))) for r in reference]
    ids = {r['session_id'] for r in reference}
    hypothesis = [
        dict(
            session_id=r['session_id'],
            speaker=r['speaker'],
            start_time=r['start_time'],
            end_time=r['end_time'],
            words=' '.join(normalize(r['prediction'])),
        )
        for r in predictions
    ]
    if {r['session_id'] for r in hypothesis} - ids:
        raise ValueError('Prediction contains an unknown Val session')
    for uid in sorted(ids - {r['session_id'] for r in hypothesis}):
        hypothesis.append(dict(session_id=uid, speaker='empty', start_time=0.0, end_time=0.0, words=''))
    reference = meeteval.io.SegLST.parse(json.dumps(reference))
    hypothesis = meeteval.io.SegLST.parse(json.dumps(hypothesis))
    cp = meeteval.wer.cp_word_error_rate_multifile(reference, hypothesis)
    tcp = meeteval.wer.tcp_word_error_rate_multifile(
        reference,
        hypothesis,
        collar=Decimal(str(collar)),
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
    )
    length = sum(r.length for r in tcp.values())
    if not length or set(tcp) != ids:
        raise ValueError('Incomplete Val reference scoring')
    return dict(
        reference_characters=length,
        sessions=len(ids),
        cpcer_pct=100 * sum(r.errors for r in cp.values()) / length,
        tcpcer_pct=100 * sum(r.errors for r in tcp.values()) / length,
        tcp_collar_s=collar,
        per_session={
            uid: dict(tcpcer_pct=100 * row.error_rate, errors=row.errors, length=row.length)
            for uid, row in tcp.items()
        },
    )
