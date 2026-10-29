"""Check baseline imports and optional GPU execution."""

import argparse
import importlib
import json
from importlib.metadata import version


def check(gpu=False):
    from robohead_baseline.utils.assets import nemo_root

    print(f'NeMo Speech: {nemo_root()}')
    modules = [
        'torch',
        'torchaudio',
        'transformers',
        'qwen_asr',
        'datasets',
        'tensorboard',
        'nara_wpe',
        'funasr',
        'webrtcvad',
        'meeteval',
        'cupy',
        'gss.core.enhancer',
        'nemo.collections.asr.models',
        'robohead_baseline.track1.model',
        'robohead_baseline.track2.model',
    ]
    for name in modules:
        importlib.import_module(name)
        print(f'{name}: OK')
    if gpu:
        import cupy as cp
        import torch

        assert torch.cuda.is_available()
        assert (torch.arange(4, device='cuda') ** 2).sum().item() == 14
        assert float((cp.arange(4) ** 2).sum().get()) == 14.0
    versions = {
        name: version(name)
        for name in (
            'nemo-toolkit',
            'torch',
            'torchaudio',
            'transformers',
            'qwen-asr',
            'funasr',
            'lhotse',
            'meeteval',
        )
    }
    versions['cupy'] = importlib.import_module('cupy').__version__
    print(json.dumps(versions, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    check(parser.parse_args().gpu)
