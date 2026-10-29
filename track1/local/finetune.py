"""yakunet/Fine-tune Qwen3-ASR using independent audio manifests and validation references."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import yaml

from robohead_baseline.track1.local.finetuning.common import read_jsonl, write_json


def train(args):
    config = yaml.safe_load(args.config.read_text())
    data = getattr(args, 'data_dir', None) or Path(config['data_dir'])
    config.pop('data_dir')
    train_file, val_file = data / 'train.jsonl', data / 'val.jsonl'
    training, validation = read_jsonl(train_file), read_jsonl(val_file)
    reference = data / 'val_reference.json'
    refs = json.loads(reference.read_text())
    if not training or not validation or not refs:
        raise ValueError('Empty training or validation data')
    if {r['session_id'] for r in training} & {r['session_id'] for r in refs}:
        raise ValueError('Training and validation sessions overlap')
    if any(not r['reference'] for r in training):
        raise ValueError('Training targets must not be empty')
    if {r['session_id'] for r in validation} - {r['session_id'] for r in refs}:
        raise ValueError('Unexpected validation sessions')
    if args.check_only:
        print(f'Train: {len(training)}; val: {len(validation)}')
        return
    gpu_ids = str(config.pop('gpu_ids', '0'))
    if 'LOCAL_RANK' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        devices = len(gpu_ids.split(','))
        if devices > 1:
            subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'torch.distributed.run',
                    '--standalone',
                    '--nproc_per_node',
                    str(devices),
                    str(Path(__file__).resolve()),
                    '--config',
                    str(args.config.resolve()),
                    '--data-dir',
                    str(data.resolve()),
                ],
                env=os.environ.copy(),
                check=True,
            )
            return
    options = dict(
        train_file=str(train_file),
        eval_file=str(val_file),
        eval_reference_file=str(reference),
        sr=16000,
        seed=42,
        tcp_collar=5.0,
        pin_memory=1,
        persistent_workers=1,
        prefetch_factor=2,
        save_strategy='steps',
        resume=0,
        lr_scheduler_type='linear',
        warmup_ratio=0.02,
    )
    options.update(config)
    if options['num_workers'] == 0:
        options['persistent_workers'] = 0
    if int(os.environ.get('RANK', '0')) == 0:
        write_json(Path(options['output_dir']) / 'training_config.json', dict(options, gpu_ids=gpu_ids))
    with tempfile.TemporaryDirectory(prefix='robohead_qwen_', dir='/tmp') as cache:
        os.environ['HF_DATASETS_CACHE'] = cache
        from robohead_baseline.track1.local.finetuning.qwen3_asr_sft import main

        main(SimpleNamespace(**options))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('conf/finetune.yaml'))
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    train(parser.parse_args())
