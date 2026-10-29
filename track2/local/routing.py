"""Associate received ODAS tracks with diarization speakers."""

from collections import defaultdict

import numpy as np


class CausalRouter:
    def __init__(self, enhancement):
        self.enhancement = enhancement
        self.votes = defaultdict(lambda: np.zeros(8))
        self.labels = {}
        self.seen = 0
        self.routed = self.fallback = 0

    def update(self, probabilities, first):
        for index, posterior in enumerate(probabilities):
            speakers = np.flatnonzero(np.asarray(posterior) >= 0.5)
            if len(speakers) != 1:
                continue
            for frame in range((first + index) * 10, (first + index + 1) * 10):
                if frame >= len(self.enhancement.tracks):
                    break
                energy = self.enhancement.energy[frame]
                share = energy / max(energy.sum(), 1e-12)
                for slot, track in enumerate(self.enhancement.tracks[frame]):
                    if track['id']:
                        self.votes[track['id']][speakers[0]] += share[slot] * track['activity'] * 0.008
        self.labels = {
            track: int(v.argmax()) for track, v in self.votes.items() if v.sum() >= 0.3 and v.max() / v.sum() >= 0.6
        }

    def select(self, speaker, first_s, last_s):
        start, stop = max(0, int(first_s / 0.008)), min(len(self.enhancement.tracks), int(last_s / 0.008))
        scores = np.zeros(6)
        for frame in range(start, stop):
            for slot, track in enumerate(self.enhancement.tracks[frame]):
                if self.labels.get(track['id']) == speaker:
                    scores[slot] += self.enhancement.energy[frame][slot] * track['activity']
        chosen = int(scores.argmax()) if scores.max() > 0 else None
        if chosen is not None and any(
            (self.labels.get(self.enhancement.tracks[f][chosen]['id']) != speaker for f in range(start, stop))
        ):
            chosen = None
        if chosen is None:
            self.fallback += 1
        else:
            self.routed += 1
        return chosen


class OnlineRouter(CausalRouter):
    def __init__(self, enhancement):
        super().__init__(enhancement)
        self.locked = {}

    def slots(self, speaker, times, continuous=False):
        frames = np.floor(np.asarray(times) / 0.008 + 1e-07).astype(int)
        choices = np.full(len(frames), -1, dtype=int)
        valid = (frames >= 0) & (frames < len(self.enhancement.tracks))
        if not valid.any():
            self.fallback += len(frames)
            return choices
        first, last = int(frames[valid].min()), int(frames[valid].max()) + 1
        candidates = []
        if continuous:
            for slot, row in enumerate(self.enhancement.tracks[first]):
                track = row['id']
                if not track or self.labels.get(track) != speaker:
                    continue
                if any(self.enhancement.tracks[i][slot]['id'] != track for i in range(first, last)):
                    continue
                score = sum(
                    (
                        self.enhancement.energy[i][slot] * self.enhancement.tracks[i][slot]['activity']
                        for i in range(first, last)
                    )
                )
                if score > 0:
                    candidates.append((track == self.locked.get(speaker), score, -slot, track))
            if candidates:
                _, _, negative_slot, track = max(candidates)
                choices[valid] = -negative_slot
                self.locked[speaker] = track
            else:
                self.locked.pop(speaker, None)
        else:
            for position in np.flatnonzero(valid):
                frame = frames[position]
                scores = [
                    self.enhancement.energy[frame][slot] * row['activity']
                    if row['id'] and self.labels.get(row['id']) == speaker
                    else 0.0
                    for slot, row in enumerate(self.enhancement.tracks[frame])
                ]
                if max(scores) > 0:
                    choices[position] = int(np.argmax(scores))
        self.routed += int((choices >= 0).sum())
        self.fallback += int((choices < 0).sum())
        return choices
