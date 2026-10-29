"""Decode each predicted speaker turn with an independent Paraformer cache."""

import numpy as np
import torch


class SpeakerTurn:
    def __init__(self, model, speaker, start):
        self.model = model
        self.speaker = f'speaker_{speaker}'
        self.start = self.end = start
        self.cache = {}
        self.pending = np.empty(0, np.float32)
        self.text = ''
        self.done = False

    def push(self, audio, speech_end=None):
        if speech_end is not None:
            self.end = speech_end
        self.pending = np.concatenate((self.pending, audio))
        size = self.model.options['asr']['chunk_size'][1] * 960
        while len(self.pending) >= size:
            audio, self.pending = self.pending[:size], self.pending[size:]
            self._decode(audio, False)

    def finish(self):
        if not self.done:
            self._decode(self.pending, True)
            self.pending = np.empty(0, np.float32)
            self.done = True

    def segment(self):
        return dict(speaker=self.speaker, start_time=self.start, end_time=self.end, words=self.text)

    @torch.inference_mode()
    def _decode(self, audio, final):
        options = self.model.options['asr']
        result = self.model.asr.generate(
            input=audio,
            cache=self.cache,
            is_final=final,
            chunk_size=options['chunk_size'],
            encoder_chunk_look_back=options['encoder_chunk_look_back'],
            decoder_chunk_look_back=options['decoder_chunk_look_back'],
            batch_size=1,
        )
        text = result[0].get('text', '').replace('</s>', '').replace('<s>', '').strip() if result else ''
        if text:
            self.text += text
            self.model._publish()
