"""Separate raw microphone audio using predicted speaker activity."""

from collections import defaultdict
from pathlib import Path

import cupy as cp
import numpy as np
from gss.core.enhancer import get_enhancer
from gss.utils.data_utils import GssDataset, create_sampler
from lhotse import CutSet, Recording, SupervisionSegment
from lhotse.cut import MultiCut

from robohead_baseline.utils.audio import limit_peaks


def separate(audio_path, activity, config):
    recording = Recording.from_file(audio_path, recording_id='session')
    channels = recording.channel_ids
    full = recording.to_cut()
    full.supervisions = [
        SupervisionSegment(
            id=f'activity_{i}',
            recording_id='session',
            start=row['start'],
            duration=row['end'] - row['start'],
            channel=channels,
            speaker=row['speaker'],
        )
        for i, row in enumerate(activity)
    ]
    if not activity:
        return {}
    enhancer = get_enhancer(
        CutSet.from_cuts([full]),
        context_duration=config['context_seconds'],
        wpe=True,
        bss_iterations=config['iterations'],
    )
    dataset = GssDataset(activity=enhancer.activity, context_duration=enhancer.context_duration)
    targets, mapping = [], {}
    outputs = defaultdict(lambda: np.zeros(recording.num_samples, np.float32))
    maximum = round(config['max_segment_seconds'] * 16000)
    if maximum <= 0:
        raise ValueError('GSS max_segment_seconds must be positive')
    for i, row in enumerate(activity):
        first, last = round(row['start'] * 16000), round(row['end'] * 16000)
        for start in range(first, last, maximum):
            end = min(start + maximum, last)
            uid = f'{i}_{start}'
            supervision = SupervisionSegment(
                id=uid,
                recording_id='session',
                start=0,
                duration=(end - start) / 16000,
                channel=channels,
                speaker=row['speaker'],
            )
            targets.append(
                MultiCut(
                    id=uid,
                    start=start / 16000,
                    duration=(end - start) / 16000,
                    channel=channels,
                    recording=recording,
                    supervisions=[supervision],
                )
            )
            mapping[uid] = row['speaker'], start, end
    sampler = create_sampler(CutSet.from_cuts(targets), max_duration=config['batch_seconds'], num_buckets=1)
    for cuts in sampler:
        batch = dataset[cuts]
        enhanced = _enhance(enhancer, batch)
        offset = 0
        for cut in batch['orig_cuts']:
            speaker, start, end = mapping[cut.id]
            length = end - start
            part = enhanced[0, offset : offset + length]
            if len(part) != length:
                raise ValueError('GSS output length differs from target')
            outputs[speaker][start:end] = limit_peaks(part)
            offset += length
        if not 0 <= enhanced.shape[-1] - offset < enhancer.stft_shift:
            raise ValueError('Unexpected GSS tail length')
    cp.get_default_memory_pool().free_all_blocks()
    return dict(outputs)


def read_rttm(path, duration):
    rows = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 10 or fields[0] != 'SPEAKER' or fields[1] != 'session':
            raise ValueError('Invalid predicted RTTM')
        start = float(fields[3])
        end = min(duration, start + float(fields[4]))
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= duration:
            raise ValueError('Predicted activity exceeds audio')
        rows.append(dict(speaker=fields[7], start=start, end=end))
    previous = {}
    for row in sorted(rows, key=lambda r: (r['speaker'], r['start'])):
        if row['start'] < previous.get(row['speaker'], 0) - 1 / 16000:
            raise ValueError('Overlapping same-speaker RTTM intervals')
        previous[row['speaker']] = row['end']
    return rows


def _enhance(enhancer, batch):
    options = dict(left_context=batch['left_context'], right_context=batch['right_context'])
    for chunks in (1, 2, 4, 8, 16):
        try:
            try:
                output = enhancer.enhance_batch(
                    batch['audio'], batch['activity'], batch['speaker_idx'], num_chunks=chunks, **options
                )
                if np.isfinite(output).all():
                    return output
            except AssertionError as error:
                import traceback

                if not any(
                    frame.name == 'get_optimal_reference_channel'
                    for frame in traceback.extract_tb(error.__traceback__)
                ):
                    raise
            enhancer.bf_drop_context = False
            try:
                output = enhancer.enhance_batch(
                    batch['audio'], batch['activity'], batch['speaker_idx'], num_chunks=chunks, **options
                )
            finally:
                enhancer.bf_drop_context = True
            if not np.isfinite(output).all():
                raise RuntimeError('Non-finite GSS audio after MVDR context retry')
            print('GSS: MVDR context retry succeeded', flush=True)
            return output
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            if chunks == 16:
                raise
