"""Transcribe a session using enhanced diarization, GPU GSS and Qwen3-ASR."""

import os
import tempfile
from pathlib import Path

import cupy as cp
import numpy as np
import soundfile as sf
import torch
import yaml

from robohead_baseline.frontend.gss import read_rttm, separate
from robohead_baseline.frontend.odas import OdasPipe
from robohead_baseline.frontend.wpe_offline import multichannel_wpe, process_blocks
from robohead_baseline.utils.assets import nemo_checkpoint, nemo_root, qwen_checkpoint
from robohead_baseline.utils.audio import limit_peaks, transcript
from robohead_baseline.track1.local.windows import make_windows


class Model:
    def __init__(self):
        self.nemo_root = nemo_root()
        path = Path(os.environ.get('ROBOHEAD_CONFIG', Path(__file__).with_name('conf') / 'model.yaml'))
        self.config = yaml.safe_load(path.read_text())
        self.diar_path = nemo_checkpoint(self.config['diarization_model'])
        self.asr_path = qwen_checkpoint(self.config['asr_model'])
        self.asr = None

    def predict(self, wav_path):
        audio, rate = sf.read(wav_path, dtype='float32', always_2d=True)
        if (
            rate != 16000
            or audio.shape[1] != 6
            or not len(audio)
            or not np.isfinite(audio).all()
            or np.max(np.abs(audio)) > 1
        ):
            raise ValueError('Expected normalized six-channel 16 kHz audio')
        np.random.seed(42)
        cp.random.seed(42)
        torch.manual_seed(42)
        if self.asr is not None:
            self.asr.model.to('cpu')
            torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(prefix='robohead_offline_') as directory:
            root = Path(directory)
            enhanced = process_blocks(audio.T, self.config['wpe_offline'], multichannel_wpe)
            enhanced = limit_peaks(enhanced).T
            sf.write(root / 'wpe_offline.wav', enhanced, 16000, subtype='PCM_16')
            enhanced, _ = sf.read(root / 'wpe_offline.wav', dtype='float32', always_2d=True)
            hop, delay = self.config['odas']['hop_size'], self.config['odas_delay_samples']
            feed = np.pad(enhanced, ((0, (-len(audio)) % hop + delay), (0, 0)))
            odas = OdasPipe(self.config['odas'], root / 'odas')
            try:
                separated, _ = odas.push(feed)
            finally:
                odas.close()
            summed = separated[delay : delay + len(audio)].sum(axis=1)
            sf.write(root / 'session.wav', summed, 16000, subtype='FLOAT')
            cp.get_default_memory_pool().free_all_blocks()
            self._diarize(root, len(audio) / 16000)
            activity = read_rttm(root / 'pred_rttms/session.rttm', len(audio) / 16000)
            if not activity:
                return dict(text='', segments=[])
            if self.config['gss']['enabled']:
                waveforms = separate(wav_path, activity, self.config['gss'])
            else:
                waveforms = {row['speaker']: summed for row in activity}
            if self.asr is None:
                from qwen_asr import Qwen3ASRModel

                self.asr = Qwen3ASRModel.from_pretrained(
                    self.asr_path,
                    dtype=torch.bfloat16,
                    device_map='cuda:0',
                    max_inference_batch_size=1,
                    max_new_tokens=self.config['max_new_tokens'],
                    local_files_only=True,
                )
            else:
                self.asr.model.to('cuda:0')
            rows = []
            for speaker, waveform in sorted(waveforms.items()):
                intervals = [(r['start'], r['end']) for r in activity if r['speaker'] == speaker]
                for start, end in make_windows(intervals, len(audio) / 16000, **self.config['windows']):
                    clip = waveform[round(start * 16000) : round(end * 16000)]
                    with torch.inference_mode():
                        text = self.asr.transcribe(audio=(clip, 16000), language=self.config['language'])[0].text
                    if text.strip():
                        rows.append(dict(speaker=speaker, start_time=start, end_time=end, words=text))
            rows.sort(key=lambda row: (row['start_time'], row['speaker']))
            return dict(text=transcript(rows), segments=rows)

    def _diarize(self, root, duration):
        from robohead_baseline.track1.local.diarization import diarize

        diarize(root, duration, self.config)
