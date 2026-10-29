"""Check Paraformer cache updates when a speaker turn ends at EOF."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from robohead_baseline.track2.local.paraformer_local import SpeakerTurn


class ParaformerTests(unittest.TestCase):
    def test_final_flush_updates_inference_cache(self):
        def generate(input, cache, is_final, **kwargs):
            if 'features' not in cache:
                cache['features'] = torch.ones(2)
            else:
                cache['features'].mul_(2)
            return [{'text': 'world' if is_final else 'hello'}]

        asr = Mock()
        asr.generate.side_effect = generate
        model = SimpleNamespace(
            asr=asr,
            options={
                'asr': {
                    'chunk_size': [0, 10, 5],
                    'encoder_chunk_look_back': 4,
                    'decoder_chunk_look_back': 1,
                }
            },
            _publish=Mock(),
        )
        turn = SpeakerTurn(model, 0, 0.0)
        with torch.inference_mode():
            turn.push(np.zeros(9700, np.float32), 0.60625)
        self.assertTrue(torch.is_inference(turn.cache['features']))
        turn.finish()
        turn.finish()
        self.assertEqual(asr.generate.call_count, 2)
        self.assertTrue(asr.generate.call_args.kwargs['is_final'])
        self.assertEqual(len(asr.generate.call_args.kwargs['input']), 100)
        self.assertEqual(turn.text, 'helloworld')
        self.assertEqual(turn.cache['features'].tolist(), [2.0, 2.0])
        self.assertTrue(turn.done)
        self.assertEqual(len(turn.pending), 0)


if __name__ == '__main__':
    unittest.main()
