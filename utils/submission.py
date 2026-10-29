"""Unpack prediction files and run the fixed scoring program."""

import argparse
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def unpack_submission(archive, destination, track):
    required = ['hypothesis.seglst.json'] if track == 1 else ['predict.csv', 'partial_results.json']
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        names = set()
        for member in members:
            path = PurePosixPath(member.filename)
            mode = stat.S_IFMT(member.external_attr >> 16)
            if (
                path.is_absolute()
                or '..' in path.parts
                or '\\' in member.filename
                or ':' in member.filename
                or not path.parts
                or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
            ):
                raise ValueError(f'Unsupported ZIP entry: {member.filename}')
            if path in names:
                raise ValueError(f'Duplicate ZIP entry: {member.filename}')
            names.add(path)
        for name in required:
            if PurePosixPath(name) not in names or source.getinfo(name).is_dir():
                raise ValueError(f'{name} must be at the ZIP root')
        for member in members:
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as incoming, target.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                target.chmod(0o755 if (member.external_attr >> 16) & 0o111 else 0o644)


def evaluate(args, submission):
    baseline = Path(__file__).resolve().parents[1]
    options = [
        '--track',
        str(args.track),
        '--stage',
        '1',
        '--stop-stage',
        '2' if args.track == 2 else '1',
        '--manifest-csv',
        str(args.data / 'sessions.csv'),
        '--reference-json',
        str(args.data / 'reference.seglst.json'),
        '--output-dir',
        str(args.output),
    ]
    if args.scoring_uem:
        options += ['--scoring-uem', str(args.scoring_uem)]
    if args.track == 2:
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ('predict.csv', 'partial_results.json'):
            if (submission / name).resolve() != (args.output / name).resolve():
                shutil.copyfile(submission / name, args.output / name)
    else:
        options += ['--hypothesis-json', str(submission / 'hypothesis.seglst.json')]
    return subprocess.run(
        ['bash', str(baseline / 'robohead_evaluate_v1.sh'), *options], cwd=baseline / f'track{args.track}'
    ).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('track', type=int, choices=(1, 2))
    parser.add_argument('submission', type=Path, help='Prediction ZIP file or extracted directory')
    parser.add_argument('data', type=Path, help='Prepared manifests and scoring references')
    parser.add_argument('output', type=Path)
    parser.add_argument('scoring_uem', type=Path, nargs='?')
    args = parser.parse_args()
    for name in ('submission', 'data', 'output', 'scoring_uem'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.submission.is_dir():
        return evaluate(args, args.submission)
    with tempfile.TemporaryDirectory(prefix='robohead_submission_') as directory:
        submission = Path(directory)
        unpack_submission(args.submission, submission, args.track)
        return evaluate(args, submission)


if __name__ == '__main__':
    raise SystemExit(main())
