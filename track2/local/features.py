"""yakunet/Extract causal features and feed the NeMo streaming buffer."""

import math

import numpy as np


class IncrementalFrontend:
    def __init__(self, buffer):
        self.buffer = buffer
        cfg = buffer.model._cfg.preprocessor
        if cfg.get('normalize') not in (None, 'NA', 'None') or cfg.get('exact_pad', False):
            raise ValueError('Live adapter requires the current unnormalized, centered STFT frontend')
        if cfg.get('frame_splicing', 1) != 1:
            raise ValueError('Unexpected feature splicing')
        self.rate = int(cfg.sample_rate)
        self.hop = round(float(cfg.window_stride) * self.rate)
        self.fft = int(cfg.get('n_fft') or 2 ** math.ceil(math.log2(float(cfg.window_size) * self.rate)))
        self.left_frames = math.ceil((self.fft // 2 + 1) / self.hop) + 1
        self.pcm = np.empty(0, dtype=np.float32)
        self.origin = 0
        self.samples = 0
        self.frames = 0
        self.finished = False

    def accept(self, pcm, eof=False):
        if self.finished:
            raise ValueError('PCM after EOF')
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError('Expected finite mono PCM')
        self.pcm = np.concatenate((self.pcm, pcm))
        self.samples += len(pcm)
        safe = self.samples // self.hop + 1 if eof else max(0, (self.samples - self.fft // 2 - 1) // self.hop + 1)
        self.finished = eof
        if safe <= self.frames:
            return None
        if len(self.pcm) <= self.fft // 2:
            raise ValueError('Audio too short for the checkpoint frontend')
        features, _ = self.buffer.preprocess_audio(self.pcm)
        start = self.frames - self.origin // self.hop
        count = safe - self.frames
        result = features[:, :, start : start + count].clone()
        if result.shape[-1] != count:
            raise ValueError('Incremental frontend frame count mismatch')
        self.frames = safe
        keep = max(0, (self.frames - self.left_frames) * self.hop)
        drop = keep - self.origin
        self.pcm = self.pcm[drop:].copy()
        self.origin = keep
        return result
