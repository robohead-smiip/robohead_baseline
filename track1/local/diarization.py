"""Run the NeMo offline diarization entry point."""

import json
import subprocess
import sys

from robohead_baseline.utils.assets import nemo_checkpoint, nemo_root


def diarize(root, duration, config):
    manifest = root / 'manifest.jsonl'
    manifest.write_text(
        json.dumps(
            dict(
                audio_filepath=str(root / 'session.wav'),
                duration=duration,
                offset=0,
                uniq_id='session',
                label='infer',
                text='-',
            )
        )
        + '\n'
    )
    entry = nemo_root() / 'examples/speaker_tasks/diarization/neural_diarizer/e2e_diarize_speech.py'
    command = [
        sys.executable,
        str(entry),
        f"model_path={nemo_checkpoint(config['diarization_model'])}",
        f'dataset_manifest={manifest}',
        f'out_rttm_dir={root / "pred_rttms"}',
        'out_preds_tensors=null',
        'cuda=0',
        'precision=bf16',
        'batch_size=1',
        'num_workers=0',
        'collar=0.0',
        'ignore_overlap=false',
        'bypass_postprocessing=true',
        'postprocessing_yaml=null',
        'spkcache_len=264',
        'fifo_len=40',
        'chunk_len=340',
        'chunk_left_context=0',
        'chunk_right_context=40',
        'spkcache_update_period=300',
        'random_seed=42',
        'compile_encoder=false',
        'profile_inference=false',
        f'hydra.run.dir={root}',
        'hydra.output_subdir=null',
        'hydra/job_logging=disabled',
        'hydra/hydra_logging=disabled',
    ]
    subprocess.run(command, check=True)
