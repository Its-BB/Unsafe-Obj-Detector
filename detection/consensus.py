"""Multi-frame voting before we trust a threat."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Set

from detection.types import Detection, DetectionList


class ThreatVote:
    def __init__(self, need_frames: int = 3, window: int = 8):
        self.need_frames = max(1, need_frames)
        self.window = max(self.need_frames, window)
        self.votes: Dict[str, Deque[bool]] = {}

    def reset(self) -> None:
        self.votes.clear()

    def _key(self, det: Detection) -> str:
        tid = det.get('track_id')
        if tid is not None:
            return f'track_{tid}'
        return f"{det.get('class_name', 'x')}_{det['bbox']}"

    def record(self, dets: DetectionList) -> None:
        seen: Set[str] = set()
        for det in dets:
            if not det.get('is_dangerous'):
                continue
            key = self._key(det)
            seen.add(key)
            if key not in self.votes:
                self.votes[key] = deque(maxlen=self.window)
            self.votes[key].append(True)
        for key, q in list(self.votes.items()):
            if key not in seen:
                q.append(False)

    def is_confirmed(self, det: Detection) -> bool:
        key = self._key(det)
        q = self.votes.get(key)
        if not q:
            return False
        true_count = sum(1 for v in q if v)
        return true_count >= self.need_frames

    def apply(self, dets: DetectionList) -> DetectionList:
        self.record(dets)
        out: DetectionList = []
        for det in dets:
            copy = dict(det)
            if copy.get('is_dangerous') and not self.is_confirmed(copy):
                copy['is_dangerous'] = False
                copy['threat_level'] = 'low'
            out.append(copy)
        return out


def need_n_frames_to_alert(vote: ThreatVote, dets: DetectionList) -> DetectionList:
    return vote.apply(dets)
