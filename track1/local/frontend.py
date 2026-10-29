"""Prepare training enhancements and prediction-guided validation GSS."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import yaml

from robohead_baseline.track1.local.windows import make_windows
from robohead_baseline.utils.audio import limit_peaks, read_audio, write_json
from robohead_baseline.utils.protocol import load_csv


def prepare(args):
    import cupy as cp

    np.random.seed(42)
    cp.random.seed(42)
    config = yaml.safe_load(args.config.read_text())
    manifest = load_csv(args.data_dir / 'sessions.csv')
    if args.num_files:
        manifest = dict(list(manifest.items())[: args.num_files])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    references = None
    if args.stage == 'train':
        references = json.loads((args.data_dir / 'reference.seglst.json').read_text())
    records = []
    for uid, row in manifest.items():
        root = (args.output_dir / uid).resolve()
        root.mkdir(parents=True, exist_ok=True)
        audio = read_audio(row, args.data_dir)
        raw = (args.data_dir / row['audio_filepath']).resolve()
        if row.get('audio_channels'):
            raw = root / 'raw.wav'
            sf.write(raw, audio, 16000, subtype='PCM_16')
        if args.stage == 'train':
            records.append(prepare_training(uid, raw, audio, references, root, config))
        elif args.stage == 'diarize':
            from robohead_baseline.track1.local.diarization import diarize

            enhance_for_sd(audio, root, config)
            diarize(root, len(audio) / 16000, config)
            source = root / 'pred_rttms/session.rttm'
            (root / 'pred.rttm').write_text(source.read_text().replace('SPEAKER session ', f'SPEAKER {uid} '))
        else:
            from robohead_baseline.frontend.gss import read_rttm, separate

            activity = read_rttm(root / 'pred_rttms/session.rttm', len(audio) / 16000)
            outputs = separate(raw, activity, config['gss'])
            for speaker, waveform in sorted(outputs.items()):
                path = root / 'gss' / f'{speaker}.wav'
                path.parent.mkdir(exist_ok=True)
                sf.write(path, waveform, 16000, subtype='PCM_16')
                intervals = [(r['start'], r['end']) for r in activity if r['speaker'] == speaker]
                for start, end in make_windows(intervals, len(audio) / 16000, **config['windows']):
                    records.append(
                        dict(
                            id=f'{uid}_{speaker}_{round(start * 16000):09d}',
                            session_id=uid,
                            speaker=speaker,
                            start_time=start,
                            end_time=end,
                            audio=str(path),
                            audio_channel=0,
                            audio_interval_samples=[round(start * 16000), round(end * 16000)],
                            duration=end - start,
                            reference='',
                            text='language Chinese<asr_text>',
                            subset='val',
                            frontend='gss_predicted',
                        )
                    )
        print(f'[{args.stage}] {uid}', flush=True)
    if args.stage == 'diarize':
        (args.output_dir / 'pred.rttm').write_text(
            ''.join((args.output_dir / uid / 'pred.rttm').read_text() for uid in manifest)
        )
    else:
        write_json(args.output_dir / ('audio.json' if args.stage == 'train' else 'segments.json'), records)


def enhance_for_sd(audio, root, config):
    from robohead_baseline.frontend.odas import OdasPipe
    from robohead_baseline.frontend.wpe_offline import multichannel_wpe, process_blocks

    enhanced = limit_peaks(process_blocks(audio.T, config['wpe_offline'], multichannel_wpe)).T
    sf.write(root / 'wpe_offline.wav', enhanced, 16000, subtype='PCM_16')
    enhanced, _ = sf.read(root / 'wpe_offline.wav', dtype='float32', always_2d=True)
    hop, delay = config['odas']['hop_size'], config['odas_delay_samples']
    feed = np.pad(enhanced, ((0, (-len(audio)) % hop + delay), (0, 0)))
    with tempfile.TemporaryDirectory(prefix='odas_', dir=root) as directory:
        odas = OdasPipe(config['odas'], directory)
        try:
            separated, _ = odas.push(feed)
        finally:
            odas.close()
    sf.write(root / 'session.wav', separated[delay : delay + len(audio)].sum(axis=1), 16000, subtype='FLOAT')


def prepare_training(uid, raw, audio, references, root, config):
    from robohead_baseline.frontend.gss import separate
    from robohead_baseline.frontend.wpe_offline import multichannel_wpe, process_blocks
    from robohead_baseline.frontend.wpe_online import StreamingWPE

    enhanced = limit_peaks(process_blocks(audio.T, config['wpe_offline'], multichannel_wpe)).T
    sf.write(root / 'wpe_offline.wav', enhanced, 16000, subtype='PCM_16')
    online = StreamingWPE(6, **config['wpe_online'])
    parts = [online.push(audio[i : i + 16000]) for i in range(0, len(audio), 16000)]
    parts.append(online.finish())
    enhanced = limit_peaks(np.concatenate(parts).T).T
    sf.write(root / 'wpe_online.wav', enhanced, 16000, subtype='PCM_16')
    activity = [
        dict(speaker=r['speaker'], start=r['start_time'], end=r['end_time'])
        for r in references
        if r['session_id'] == uid
    ]
    separated = separate(raw, activity, config['gss'])
    paths = {}
    for speaker, waveform in separated.items():
        path = root / 'gss_gt' / f'{speaker}.wav'
        path.parent.mkdir(exist_ok=True)
        sf.write(path, waveform, 16000, subtype='PCM_16')
        paths[speaker] = str(path)
    return dict(
        session_id=uid,
        raw=str(raw),
        wpe_offline=str(root / 'wpe_offline.wav'),
        wpe_online=str(root / 'wpe_online.wav'),
        gss_gt=paths,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'diarize', 'gss'], required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('conf/model.yaml'))
    parser.add_argument('--num-files', type=int, default=0)
    prepare(parser.parse_args())
