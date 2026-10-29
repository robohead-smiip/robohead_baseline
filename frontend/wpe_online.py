"""Stateful STFT, recursive WPE and overlap-add synthesis."""

import numpy as np
from nara_wpe.wpe import OnlineWPE


class StreamingWPE:
    def __init__(self, channels, size=512, hop=128, taps=5, delay=3, alpha=0.9999, bypass=False):
        if channels < 1 or size < 4 or hop < 1 or size % hop or hop >= size:
            raise ValueError('Require channels >= 1 and FFT size divisible by hop, with hop < size')
        if taps < 1 or delay < 1 or not 0 < alpha <= 1:
            raise ValueError('Require taps/delay >= 1 and 0 < alpha <= 1')
        self.size, self.hop = size, hop
        self.window = np.sin(np.pi * (np.arange(size) + 0.5) / size)
        self.history = np.zeros((size, channels))
        self.overlap = np.zeros_like(self.history)
        self.weight = np.zeros(size)
        self.pending = np.empty((0, channels))
        self.skip = size - hop
        self.received = 0
        self.emitted = 0
        self.closed = False
        self.wpe = (
            None
            if bypass
            else OnlineWPE(taps=taps, delay=delay, alpha=alpha, channel=channels, frequency_bins=size // 2 + 1)
        )

    def _step(self, block):
        self.history[: -self.hop] = self.history[self.hop :]
        self.history[-self.hop :] = block
        spectrum = np.fft.rfft(self.history * self.window[:, None], axis=0)
        if self.wpe is not None:
            spectrum = self.wpe.step_frame(spectrum)
        frame = np.fft.irfft(spectrum, n=self.size, axis=0)
        self.overlap += frame * self.window[:, None]
        self.weight += self.window ** 2
        result = self.overlap[: self.hop] / self.weight[: self.hop, None]
        self.overlap[: -self.hop] = self.overlap[self.hop :]
        self.overlap[-self.hop :] = 0
        self.weight[: -self.hop] = self.weight[self.hop :]
        self.weight[-self.hop :] = 0
        if self.skip:
            count = min(self.skip, len(result))
            result = result[count:]
            self.skip -= count
        return result

    def push(self, block):
        if self.closed:
            raise ValueError('The stream is already closed')
        self.received += len(block)
        self.pending = np.concatenate((self.pending, block))
        parts = []
        while len(self.pending) >= self.hop:
            parts.append(self._step(self.pending[: self.hop]))
            self.pending = self.pending[self.hop :]
        result = np.concatenate(parts) if parts else self.pending[:0].copy()
        self.emitted += len(result)
        return result

    def finish(self):
        if self.closed:
            raise ValueError('The stream is already closed')
        self.closed = True
        parts = []
        if len(self.pending):
            parts.append(self._step(np.pad(self.pending, ((0, self.hop - len(self.pending)), (0, 0)))))
        for _ in range(self.size // self.hop - 1):
            parts.append(self._step(np.zeros((self.hop, self.history.shape[1]))))
        result = np.concatenate(parts)[: self.received - self.emitted]
        self.emitted += len(result)
        assert self.emitted == self.received
        return result
