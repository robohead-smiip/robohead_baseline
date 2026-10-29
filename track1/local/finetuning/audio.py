"""Read training intervals and apply optional speed perturbation."""

from fractions import Fraction
import hashlib
import math

import numpy as np
import soundfile as sf


def read_training_audio(row, key='audio', verify=False, augment=True):
    interval = row.get(f'{key}_interval_samples')
    parts = row.get(f'{key}_parts')
    if parts:
        pieces = []
        for part in parts:
            gap = part.get('gap_samples', 0)
            if type(gap) is not int or gap < 0 or gap > 16000 * 2:
                raise ValueError(f'Invalid gap: {gap}')
            if gap:
                pieces.append(np.zeros(gap, dtype='float32'))
            pieces.append(read_training_audio(part, verify=verify, augment=False))
        samples = np.concatenate(pieces)
        first, last = _interval(interval)
        if len(samples) != last - first:
            raise ValueError('Composite audio length differs from its source interval')
    else:
        samples = _read_interval(row, key, interval)
    digest = row.get(f'{key}_samples_sha256')
    if verify and digest and hashlib.sha256(samples.astype('<f4').tobytes()).hexdigest() != digest:
        raise ValueError(f'Audio changed: {row[key]}')
    speed = float(1 if row.get('speed') is None else row['speed'])
    if not math.isfinite(speed) or not 0.5 <= speed <= 2:
        raise ValueError(f'Invalid speed: {speed}')
    if augment and speed != 1:
        from scipy.signal import resample_poly

        ratio = Fraction(str(speed)).limit_denominator(1000)
        samples = resample_poly(samples, ratio.denominator, ratio.numerator).astype('float32')
    return samples


def _read_interval(row, key, interval):
    with sf.SoundFile(row[key]) as source:
        channel = row.get(f'{key}_channel')
        if channel is None and source.channels == 1:
            channel = 0
        if source.samplerate != 16000 or type(channel) is not int or not 0 <= channel < source.channels:
            raise ValueError(f'Expected 16 kHz audio with an explicit valid channel: {row[key]}')
        first, last = (0, source.frames) if interval is None else _interval(interval)
        if not 0 <= first < last <= source.frames:
            raise ValueError(f'Invalid audio interval: {row[key]} {first}:{last}')
        source.seek(first)
        samples = np.ascontiguousarray(source.read(last - first, dtype='float32', always_2d=True)[:, channel])
    if len(samples) != last - first or not np.isfinite(samples).all():
        raise ValueError(f'Invalid audio: {row[key]}')
    return samples


def _interval(interval):
    if (
        not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or any(type(value) is not int for value in interval)
    ):
        raise ValueError(f'Invalid audio interval: {interval}')
    return interval
