"""Apply offline multichannel WPE with contextual blocks and overlap crossfading."""

import cupy as cp
import numpy as np
from gss.core.stft_module import istft, stft
from gss.wpe import wpe


def multichannel_wpe(audio, params):
    size, hop = params['fft_size'], params['hop_size']
    spectrum = stft(cp.asarray(audio), size=size, shift=hop)
    if float(cp.max(cp.abs(spectrum)).get()) == 0:
        return audio.copy()
    cleaned = wpe(
        spectrum.transpose(2, 0, 1),
        taps=params['taps'],
        delay=params['delay'],
        iterations=params['iterations'],
        statistics_mode='full',
    ).transpose(1, 2, 0)
    return istft(cleaned, size=size, shift=hop)[..., : audio.shape[-1]]


def process_blocks(audio, params, operation):
    length = audio.shape[-1]
    block = round(params['block_seconds'] * 16000)
    context = round(params['context_seconds'] * 16000)
    fade = round(params['crossfade_seconds'] * 16000)
    output = np.zeros(audio.shape, dtype=np.float64)
    total_weight = np.zeros(length, dtype=np.float64)
    for start in range(0, length, block):
        end = min(length, start + block)
        left, right = max(0, start - fade), min(length, end + fade)
        first, last = max(0, left - context), min(length, right + context)
        result = operation(audio[:, first:last], params)[..., left - first : right - first]
        weight = np.ones(right - left)
        if left < start:
            weight[: start - left] = np.linspace(0, 1, start - left, endpoint=False)
        if right > end:
            weight[end - left :] = np.linspace(1, 0, right - end, endpoint=False)
        output[..., left:right] += cp.asnumpy(result) * weight
        total_weight[left:right] += weight
    assert np.all(total_weight > 0)
    return output / total_weight
