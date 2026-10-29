"""Read synchronized session audio and write normalized prediction records."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf


def read_audio(row, root):
    paths = json.loads(row['audio_channels']) if row.get('audio_channels') else [row['audio_filepath']]
    arrays = []
    for path in paths:
        name = Path(root) / path
        if sf.info(name).subtype != 'PCM_16':
            raise ValueError(f'Expected PCM16 audio: {path}')
        audio, rate = sf.read(name, dtype='float32', always_2d=True)
        if rate != 16000 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f'Expected finite, nonempty 16 kHz audio: {path}')
        arrays.append(audio)
    if not arrays or len({len(x) for x in arrays}) != 1:
        raise ValueError('Microphone channels must have equal lengths')
    audio = np.concatenate(arrays, axis=1)
    if audio.shape[1] != 6:
        raise ValueError('Expected six synchronized microphone channels')
    if abs(len(audio) / 16000 - float(row['audio_duration_sec'])) > 1 / 16000:
        raise ValueError('Manifest duration differs from audio')
    if np.max(np.abs(audio)) > 1:
        raise ValueError('Audio amplitude exceeds normalized PCM range')
    return audio


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def transcript(segments):
    return ''.join(row['words'] for row in sorted(segments, key=lambda row: float(row['start_time'])))


def limit_peaks(audio, ceiling=0.98):
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

    peak = np.max(np.abs(audio), axis=0) if audio.ndim == 2 else np.abs(audio)
    envelope = maximum_filter1d(peak, size=321, mode='nearest')
    gain = np.minimum(1.0, ceiling / np.maximum(envelope, 1e-15))
    gain = np.minimum(gain, uniform_filter1d(minimum_filter1d(gain, size=321), size=321))
    return audio * gain
