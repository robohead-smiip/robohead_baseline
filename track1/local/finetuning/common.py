"""Shared files and interval utilities for offline multi-speaker ASR."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def output_path(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = output_path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_jsonl(path, rows):
    path = output_path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8'
    )
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for data in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def model_identity(path):
    path = Path(path).resolve(strict=True)
    files = (
        [path] if path.is_file() else sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith('.'))
    )
    return {'path': str(path), 'files': {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in files}}


def normalize(text, language='zh'):
    text = re.sub(r'<[^>]*>', '', unicodedata.normalize('NFKC', text or '')).lower()
    if language == 'zh':
        return ''.join(c for c in text if unicodedata.category(c)[:1] in {'L', 'N'})
    return ' '.join(''.join(c if unicodedata.category(c)[:1] in {'L', 'N'} else ' ' for c in text).split())
