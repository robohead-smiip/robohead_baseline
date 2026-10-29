"""Read the long TextGrid format used by RoboHEARD annotations and MFA exports."""

import re
from pathlib import Path

from robohead_baseline.utils.protocol import normalize_text


def clean_transcript(text):
    text = ' '.join(re.sub(r'<[^>]*>', '', text).split())
    return text if normalize_text(text) else ''


def read_textgrid(path):
    source = Path(path).read_text(encoding='utf-8-sig')
    tiers = {}
    pattern = r'intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*"((?:[^"]|"")*)"'
    for block in re.split(r'item\s*\[\d+\]:', source)[1:]:
        name = re.search(r'name\s*=\s*"((?:[^"]|"")*)"', block)
        if '"IntervalTier"' not in block or name is None:
            continue
        intervals = [(float(a), float(b), text.replace('""', '"')) for a, b, text in re.findall(pattern, block)]
        count = re.search(r'intervals:\s*size\s*=\s*(\d+)', block)
        if count is None or len(intervals) != int(count[1]):
            raise ValueError(f'Invalid interval count in {path}')
        tiers[name[1]] = intervals
    if not tiers:
        raise ValueError(f'Expected a long-format IntervalTier TextGrid: {path}')
    return tiers
