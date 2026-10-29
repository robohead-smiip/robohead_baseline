"""Transcribe ODAS tracks using NeMo diarization and streaming Paraformer."""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from funasr import AutoModel

from robohead_baseline.frontend.odas import OnlineEnhancement
from robohead_baseline.track2.local.diarization_local import NeMoDiarization
from robohead_baseline.track2.local.paraformer_local import SpeakerTurn
from robohead_baseline.track2.local.routing import OnlineRouter
from robohead_baseline.utils.audio import transcript


class Model:
    def __init__(self):
        path = Path(os.environ.get('ROBOHEAD_CONFIG', Path(__file__).with_name('conf') / 'model.yaml'))
        self.options = yaml.safe_load(path.read_text())
        torch.manual_seed(self.options['streaming']['random_seed'])
        self.diar = NeMoDiarization(self.options)
        self.asr = AutoModel(
            model=str(Path(self.options['asr_model']).expanduser()),
            hub='ms',
            device='cuda:0',
            ncpu=1,
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
            frontend_conf=dict(dither=0.0),
        )
        self.callback = None
        self.directory = self.frontend = None
        self.finished = True

    def set_partial_callback(self, callback):
        self.callback = callback

    def reset(self):
        self.close()
        torch.manual_seed(self.options['streaming']['random_seed'])
        self.diar.reset()
        self.directory = tempfile.TemporaryDirectory(prefix='robohead_stream_')
        self.frontend = OnlineEnhancement(self.options['odas'], self.directory.name)
        self.router = OnlineRouter(self.frontend)
        self.turns, self.active = [], {}
        self.audio = None
        self.samples = self.consumed = 0
        self.last_text = ''
        self.frontend_ready = False
        self.finished = False

    def accept_chunk(self, audio_chunk):
        if self.finished:
            raise ValueError('Call reset() before accepting audio')
        if audio_chunk.ndim != 2 or audio_chunk.shape[1] != 6 or not len(audio_chunk):
            raise ValueError('Expected six-channel PCM')
        if not np.isfinite(audio_chunk).all() or np.max(np.abs(audio_chunk)) > 1:
            raise ValueError('Expected normalized finite PCM')
        self.samples += len(audio_chunk)
        pcm = np.clip(np.rint(audio_chunk * 32768), -32768, 32767).astype('<i2').astype(np.float32) / 32768
        self._process(self.frontend.push(pcm))
        self._publish()
        return self.last_text

    def input_finished(self):
        if self.finished:
            raise ValueError('input_finished() must be called once after reset()')
        try:
            self._process(self.frontend.push(np.empty((0, 6), np.float32), eof=True), eof=True)
            for turn in self.active.values():
                turn.finish()
            self.active.clear()
            self._publish()
            if self.consumed != self.samples:
                raise ValueError('Diarization did not consume all input samples')
            torch.cuda.synchronize()
            return dict(text=self.last_text, segments=self._segments())
        finally:
            self.finished = True
            self.close()

    def close(self):
        if self.frontend is not None:
            self.frontend.close()
            self.frontend = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None

    @torch.inference_mode()
    def _process(self, enhanced, eof=False):
        if self.options['streaming']['startup_fallback'] and not self.frontend_ready:
            nonzero = np.flatnonzero(np.any(enhanced[:, 2:] != 0, axis=1))
            stop = int(nonzero[0]) if len(nonzero) else len(enhanced)
            enhanced[:stop, 0] = enhanced[:stop, 1]
            self.frontend_ready = bool(len(nonzero))
        self.audio = enhanced.copy() if self.audio is None else np.concatenate((self.audio, enhanced))
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.diar.bf16):
            predictions = self.diar.push(enhanced[:, 0], eof)
        self.router.update(predictions, self.consumed // 1280)
        for probabilities in predictions:
            size = min(1280, len(self.audio))
            if size == 0:
                break
            first, end = self.consumed / 16000, (self.consumed + size) / 16000
            for speaker, probability in enumerate(probabilities):
                speaking = probability > self.options['streaming']['speech_threshold']
                turn = self.active.get(speaker)
                if speaking and turn is None:
                    turn = SpeakerTurn(self, speaker, first)
                    self.active[speaker] = turn
                    self.turns.append(turn)
                if turn is None:
                    continue
                audio = self.audio[:size, 0].copy()
                if speaking and self.options['streaming']['routing'] == 'sum_continuous':
                    slots = self.router.slots(speaker, first + np.arange(size) / 16000, continuous=True)
                    chosen = slots >= 0
                    audio[chosen] = self.audio[np.flatnonzero(chosen), slots[chosen] + 2]
                turn.push(audio if speaking else np.zeros_like(audio), end if speaking else None)
                if not speaking and end - turn.end >= self.options['streaming']['silence_seconds'] - 1e-8:
                    turn.finish()
                    del self.active[speaker]
            self.consumed += size
            self.audio = self.audio[size:]
            self._publish()

    def _segments(self):
        return sorted(
            (turn.segment() for turn in self.turns if turn.text.strip()),
            key=lambda row: (row['start_time'], row['speaker']),
        )

    def _publish(self):
        text = transcript(self._segments())
        if text != self.last_text:
            self.last_text = text
            if self.callback is not None:
                self.callback(text)
