"""Resolve downloaded checkpoints and the installed NeMo source tree."""

from pathlib import Path


def nemo_checkpoint(source):
    path = Path(source).expanduser()
    return str(path if path.suffix == '.nemo' else next(path.glob('*.nemo')))


def qwen_checkpoint(source):
    return str(Path(source).expanduser())


def nemo_root():
    return Path(__file__).resolve().parents[2]
