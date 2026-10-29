"""Render the ODAS microphone and stream configuration."""

import json
from pathlib import Path
from string import Template

import numpy as np


def microphone_positions(count, diameter_mm=None, positions_file=None, start_angle=0.0, direction='clockwise'):
    if (diameter_mm is None) == (positions_file is None):
        raise ValueError('Set exactly one of array-diameter-mm or mic-positions')
    if count < 2:
        raise ValueError('ODAS requires at least two microphone channels')
    if positions_file is not None:
        positions = np.loadtxt(str(positions_file), ndmin=2)
    else:
        if not np.isfinite(diameter_mm) or diameter_mm <= 0:
            raise ValueError('Array diameter must be positive')
        sign = -1 if direction == 'clockwise' else 1
        angles = np.deg2rad(start_angle + sign * np.arange(count) * 360 / count)
        positions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(count))) * diameter_mm / 2000
    if positions.shape != (count, 3) or not np.isfinite(positions).all():
        raise ValueError('mic-positions must contain one finite x y z row per selected channel, in meters')
    if len(np.unique(positions, axis=0)) != count:
        raise ValueError('Microphone positions must be distinct')
    return positions


def render_config(
    options, directory, temporary, input_interface=None, separated_interface=None, postfiltered_interface=None
):
    microphones = []
    for position in options['positions']:
        xyz = ', '.join(f'{x:.12f}' for x in position)
        microphones.append(
            '{ mu = ('
            + xyz
            + '); sigma2 = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0); direction = (0.0,0.0,1.0); angle = (180.0,180.0); }'
        )
    values = dict(
        options,
        channels=len(microphones),
        microphones=',\n'.join(microphones),
        mapping=', '.join(str(i + 1) for i in range(len(microphones))),
        inactive=', '.join('250' for _ in range(options['max_sources'])),
        sound_speed=f"{options['sound_speed']:.6f}",
        gain=f"{options['gain']:.6f}",
        search_angles='90.0, 91.0' if options['search_space'] == 'upper' else '180.0, 180.0',
        input_path=json.dumps(str(temporary / 'input.raw')),
        separated_path=json.dumps(str(temporary / 'separated.raw')),
        tracks_path=json.dumps(str(directory / 'tracks.json')),
        potential_path=json.dumps(str(directory / 'potential.json')),
        postfilter='ss' if options['postfilter'] == 'none' else options['postfilter'],
        postfiltered_interface='type = "blackhole";'
        if options['postfilter'] == 'none'
        else 'type = "file"; path = ' + json.dumps(str(temporary / 'postfiltered.raw')) + ';',
    )
    values['input_interface'] = input_interface or 'type = "file"; path = ' + values['input_path'] + ';'
    values['separated_interface'] = separated_interface or 'type = "file"; path = ' + values['separated_path'] + ';'
    if postfiltered_interface is not None:
        values['postfiltered_interface'] = postfiltered_interface
    return Template(Path(options['template']).read_text()).substitute(values)
