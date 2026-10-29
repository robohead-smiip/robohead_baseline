"""Build bounded ASR windows from speaker activity."""


def make_windows(intervals, duration, max_seconds=20.0, merge_gap=0.3, padding=0.15):
    if max_seconds <= 0 or merge_gap < 0 or padding < 0:
        raise ValueError('Invalid ASR window settings')
    merged = _merge(intervals, merge_gap)
    merged = _merge([(max(0.0, a - padding), min(duration, b + padding)) for a, b in merged], 0.0)
    result = []
    for start, end in merged:
        while end - start > max_seconds:
            result.append((start, start + max_seconds))
            start += max_seconds
        if end > start:
            result.append((start, end))
    return result


def _merge(intervals, gap):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1] + gap:
            result[-1] = result[-1][0], max(result[-1][1], b)
        else:
            result.append((a, b))
    return result
