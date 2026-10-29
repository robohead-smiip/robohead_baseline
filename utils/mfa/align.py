"""Use MFA boundary segments to set Track 2 reference speech bounds."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import soundfile as sf

from robohead_baseline.utils.annotations import read_textgrid
from robohead_baseline.utils.audio import write_json
from robohead_baseline.utils.protocol import load_csv, normalize_text

SILENCE_PHONES = {'', 'sil', 'sp', 'spn', 'sil_S', 'sil_E', 'sil_I', 'sil_B'}


def prepare(args):
    references = json.loads((args.data_dir / 'reference.seglst.json').read_text())
    leading, trailing = {}, {}
    for row in sorted(references, key=lambda row: row['start_time']):
        if normalize_text(row['words']):
            key = row['session_id'], row['speaker']
            leading.setdefault(key, row)
            if key not in trailing or row['end_time'] > trailing[key]['end_time']:
                trailing[key] = row
    rows = list(
        {
            (r['session_id'], r['speaker'], r['start_time'], r['end_time']): r
            for r in list(leading.values()) + list(trailing.values())
        }.values()
    )
    source_path = args.data_dir / 'alignment_sources.json'
    sources = json.loads(source_path.read_text()) if source_path.exists() else {}
    manifest = load_csv(args.data_dir / 'sessions.csv')
    corpus = args.work_dir / 'corpus'
    for directory in (corpus, args.work_dir / 'aligned'):
        if directory.exists():
            shutil.rmtree(directory)
    corpus.mkdir(parents=True, exist_ok=True)
    entries, vocabulary = {}, set()
    for index, row in enumerate(rows):
        text = normalize_text(row['words'])
        if not text:
            continue
        uid = f"{row['session_id']}_{row['speaker']}_{index:06d}"
        audio_row = manifest[row['session_id']]
        source = sources.get(f"{row['session_id']}/{row['speaker']}")
        if source is None:
            source = (
                json.loads(audio_row['audio_channels'])[0]
                if audio_row.get('audio_channels')
                else audio_row['audio_filepath']
            )
        source = args.data_dir / source
        info = sf.info(source)
        if info.samplerate != 16000:
            raise ValueError(f'Expected 16 kHz audio: {source}')
        first = max(0, round((row['start_time'] - 0.2) * 16000))
        last = min(info.frames, round((row['end_time'] + 0.2) * 16000))
        if first >= last or row['end_time'] > info.duration + 1 / 16000:
            raise ValueError(f'Annotation exceeds audio: {uid}')
        audio, rate = sf.read(source, start=first, stop=last, dtype='float32', always_2d=True)
        audio = audio[:, 0]
        directory = corpus / row['speaker']
        directory.mkdir(exist_ok=True)
        sf.write(directory / f'{uid}.wav', audio, rate, subtype='PCM_16')
        (directory / f'{uid}.lab').write_text(' '.join(text) + '\n', encoding='utf-8')
        entries[uid] = dict(reference=row, offset=first / 16000)
        vocabulary.update(text)
    dictionary = {line.split()[0] for line in args.dictionary.read_text().splitlines() if line.strip()}
    missing = sorted(vocabulary - dictionary)
    (args.work_dir / 'oov.txt').write_text(''.join(word + '\n' for word in missing))
    shutil.copyfile(args.dictionary, args.work_dir / 'dictionary.dict')
    write_json(args.work_dir / 'utterances.json', entries)
    print(f'{len(entries)} utterances; {len(missing)} OOV characters')


def export(args):
    entries = json.loads((args.work_dir / 'utterances.json').read_text())
    grids = {path.stem: path for path in (args.work_dir / 'aligned').rglob('*.TextGrid')}
    onsets, offsets = {}, {}
    manifest = load_csv(args.data_dir / 'sessions.csv')
    for uid, entry in entries.items():
        if uid not in grids:
            raise ValueError(f'Missing MFA alignment: {uid}; inspect the MFA alignment logs')
        tiers = read_textgrid(grids[uid])
        words = [values for name, values in tiers.items() if name == 'words' or name.endswith(' - words')]
        if len(words) != 1:
            raise ValueError(f'{uid}: expected one MFA words tier')
        stamps = [
            dict(char=normalize_text(text), start=a + entry['offset'], end=b + entry['offset'])
            for a, b, text in words[0]
            if normalize_text(text)
        ]
        row = entry['reference']
        if not stamps or ''.join(stamp['char'] for stamp in stamps) != normalize_text(row['words']):
            raise ValueError(f'{uid}: aligned text differs from the reference')
        phones = [values for name, values in tiers.items() if name == 'phones' or name.endswith(' - phones')]
        if len(phones) != 1:
            raise ValueError(f'{uid}: expected one MFA phones tier')
        speech = [
            (a + entry['offset'], b + entry['offset']) for a, b, label in phones[0] if label not in SILENCE_PHONES
        ]
        duration = float(manifest[row['session_id']]['audio_duration_sec'])
        if not speech or any(not 0 <= a < b <= duration + 1 / 16000 for a, b in speech):
            raise ValueError(f'{uid}: invalid aligned interval')
        start = min(a for a, b in speech)
        sid = row['session_id']
        onsets[sid] = min(onsets.get(sid, start), start)
        end = max(b for a, b in speech)
        offsets[sid] = max(offsets.get(sid, end), end)
    if set(onsets) != set(manifest):
        raise ValueError('Incomplete alignment coverage')
    for uid, row in manifest.items():
        row['mfa_speech_start'] = onsets[uid]
        row['mfa_speech_end'] = offsets[uid]
    with (args.data_dir / 'sessions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(next(iter(manifest.values()))))
        writer.writeheader()
        writer.writerows(manifest.values())
    print(f'Updated speech bounds for {len(manifest)} sessions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'export'))
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--dictionary', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare' and args.dictionary is None:
        parser.error('--dictionary is required for corpus preparation')
    (prepare if args.stage == 'prepare' else export)(args)
