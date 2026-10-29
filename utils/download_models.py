"""Download inference checkpoints into each track's model directory."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download(track, finetuned=False):
    baseline = Path(__file__).resolve().parents[1]
    for number in ('1', '2') if track == 'all' else (track,):
        models = {'Nemotron-3-Diarization': ('nvidia/Nemotron-3-Diarization', ['*.nemo'])}
        if number == '1':
            if finetuned:
                models['Qwen3-ASR-FT-robohead'] = 'SMIIP-lab/Qwen3-ASR-FT-robohead', None
            else:
                models['Qwen3-ASR-1.7B'] = 'Qwen/Qwen3-ASR-1.7B', None
        else:
            models['Paraformer-zh-streaming'] = 'funasr/paraformer-zh-streaming', None
        for name, (repo, patterns) in models.items():
            snapshot_download(
                repo,
                local_dir=baseline / f'track{number}' / 'models' / name,
                allow_patterns=patterns,
                ignore_patterns=['*.gguf'] if patterns is None else None,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', choices=('1', '2', 'all'), default='all')
    parser.add_argument('--finetuned', action='store_true')
    args = parser.parse_args()
    download(args.track, args.finetuned)
