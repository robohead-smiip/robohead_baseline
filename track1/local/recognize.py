"""yakunet/Transcribe prepared predicted-GSS windows and write the submission SegLST."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from qwen_asr import Qwen3ASRModel

from robohead_baseline.track1.local.finetuning.validation import predict_rows, _generation_mode
from robohead_baseline.utils.audio import write_json
from robohead_baseline.utils.protocol import load_csv


def recognize(args):
    rows = json.loads(args.segments_json.read_text())
    manifest = load_csv(args.manifest_csv)
    model_path = args.model_path or yaml.safe_load(args.config.read_text())['asr_model']
    model = Qwen3ASRModel.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map='cuda:0',
        max_inference_batch_size=args.batch_size,
        max_new_tokens=512,
        local_files_only=True,
    )
    with _generation_mode(model.model):
        predictions = predict_rows(model, rows, args.batch_size, num_workers=2)
    result = [
        dict(
            session_id=r['session_id'],
            speaker=r['speaker'],
            start_time=r['start_time'],
            end_time=r['end_time'],
            words=r['prediction'],
        )
        for r in predictions
        if r['prediction'].strip()
    ]
    found = {r['session_id'] for r in result}
    result.extend(
        dict(session_id=uid, speaker='empty', start_time=0.0, end_time=1 / 16000, words='')
        for uid in manifest
        if uid not in found
    )
    result.sort(key=lambda r: (r['session_id'], r['start_time'], r['speaker']))
    write_json(args.output_dir / 'hypothesis.seglst.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments-json', type=Path, required=True)
    parser.add_argument('--manifest-csv', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('conf/model.yaml'))
    parser.add_argument('--asr-model', '--model-path', dest='model_path', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    recognize(parser.parse_args())
