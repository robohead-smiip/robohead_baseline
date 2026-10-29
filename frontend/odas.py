"""Stream audio and track identities through local ODAS pipes."""

import json
import os
import select
import subprocess
from pathlib import Path

import numpy as np

from robohead_baseline.frontend.odas_config import microphone_positions, render_config


class OdasPipe:
    def __init__(self, config, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.channels = int(config['channels'])
        self.slots = int(config['max_sources'])
        self.hop = int(config['hop_size'])
        self.descriptors = []
        self.process = None
        self.log = (directory / 'odas.log').open('w')
        self.frame = 0
        self.pending = ''
        preload = Path(__file__).with_name('fifo.so')
        binary = config['binary']
        try:
            for name in ('input.raw', 'separated.raw', 'tracks.json'):
                path = directory / name
                os.mkfifo(path, 0o600)
                self.descriptors.append(os.open(path, os.O_RDWR | os.O_NONBLOCK))
            options = dict(config)
            options['positions'] = microphone_positions(
                self.channels,
                diameter_mm=config['array_diameter_mm'],
                start_angle=config.get('start_angle', 0.0),
                direction=config.get('direction', 'clockwise'),
            ).tolist()
            options['template'] = str(Path(__file__).with_name('odas.cfg.template'))
            path = directory / 'odas.cfg'
            path.write_text(render_config(options, directory, directory))
            previous = os.environ.get('LD_PRELOAD', '')
            env = dict(os.environ, LD_PRELOAD=str(preload.resolve()) + (':' + previous if previous else ''))
            self.process = subprocess.Popen(
                [binary, '-c', str(path), '-s'], env=env, stdout=self.log, stderr=subprocess.STDOUT
            )
        except BaseException:
            self.close()
            raise

    def push(self, audio):
        if audio.ndim != 2 or audio.shape[1] != self.channels or len(audio) % self.hop:
            raise ValueError('ODAS input must contain complete multichannel hops')
        if not np.isfinite(audio).all():
            raise ValueError('Non-finite ODAS input')
        pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype('<i2')
        blocks, tracks = [], []
        for block in pcm.reshape(-1, self.hop, self.channels):
            data = block.tobytes()
            if os.write(self.descriptors[0], data) != len(data):
                raise IOError('Incomplete ODAS input hop')
            output = bytearray()
            size = self.hop * self.slots * 2
            while len(output) < size:
                output.extend(self._read(1, size - len(output)))
            blocks.append(np.frombuffer(output, '<i2').reshape(self.hop, self.slots).astype(np.float32) / 32768)
            while True:
                try:
                    row, end = json.JSONDecoder().raw_decode(self.pending.lstrip())
                    self.pending = self.pending.lstrip()[end:]
                    break
                except json.JSONDecodeError:
                    self.pending += self._read(2, 8192).decode()
            self.frame += 1
            if row['timeStamp'] != self.frame or len(row['src']) != self.slots:
                raise ValueError('ODAS audio and track frames differ')
            tracks.append(row)
        return np.concatenate(blocks), tracks

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for descriptor in self.descriptors:
            os.close(descriptor)
        self.descriptors = []
        self.log.close()

    def _read(self, index, size):
        descriptor = self.descriptors[index]
        if not select.select([descriptor], [], [], 30)[0]:
            raise TimeoutError(f'ODAS pipe timeout; exit={self.process.poll()}')
        data = os.read(descriptor, size)
        if not data:
            raise EOFError('ODAS closed its output before completing a hop')
        return data


class OnlineEnhancement:
    def __init__(self, config, directory):
        self.odas = OdasPipe(config, directory)
        self.config = config
        self.pending = np.empty((0, config['channels']), np.float32)
        self.carrier = self.pending.copy()
        self.input_samples = self.output_samples = 0
        self.tracks, self.energy = [], []
        self.finished = False

    def push(self, pcm, eof=False):
        if self.finished:
            raise ValueError('Audio after EOF')
        self.finished = eof
        self.input_samples += len(pcm)
        self.carrier = np.concatenate((self.carrier, pcm))
        self.pending = np.concatenate((self.pending, pcm))
        hop = self.config['hop_size']
        count = len(self.pending) // hop * hop
        if eof:
            count = ((len(self.pending) + hop - 1) // hop) * hop + self.config['frame_size']
        feed = self.pending[:count]
        self.pending = self.pending[count:]
        if eof:
            feed = np.pad(feed, ((0, count - len(feed)), (0, 0)))
        if not len(feed):
            return np.empty((0, self.config['max_sources'] + 2), np.float32)
        separated, tracks = self.odas.push(feed)
        valid = min(len(separated), self.input_samples - self.output_samples)
        separated = separated[:valid]
        summed = separated.sum(axis=1) / self.config['max_sources']
        carrier = self.carrier[:valid, 0]
        self.carrier = self.carrier[valid:]
        for i, row in enumerate(tracks):
            if i * hop >= valid:
                break
            self.tracks.append(row['src'])
            self.energy.append(np.mean(separated[i * hop : min((i + 1) * hop, valid)] ** 2, axis=0))
        self.output_samples += valid
        return np.column_stack((summed, carrier, separated)).astype(np.float32)

    def close(self):
        self.odas.close()
