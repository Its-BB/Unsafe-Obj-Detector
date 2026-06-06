"""Smooth confidence over time per track."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

from detection.types import Detection, DetectionList


class ScoreSmoother:
    def __init__(self, window: int = 5, alpha: float = 0.4):
        self.window = max(1, window)
        self.alpha = alpha
        self.history: Dict[int, Deque[float]] = {}
        self.weapon_history: Dict[int, Deque[float]] = {}

    def reset(self) -> None:
        self.history.clear()
        self.weapon_history.clear()

    def _push(self, store: Dict[int, Deque[float]], key: int, value: float) -> float:
        if key not in store:
            store[key] = deque(maxlen=self.window)
        q = store[key]
        if q:
            smooth = self.alpha * value + (1.0 - self.alpha) * q[-1]
        else:
            smooth = value
        q.append(smooth)
        return smooth

    def smooth_one(self, det: Detection) -> Detection:
        out = dict(det)
        tid = int(det.get('track_id', -1))
        conf = float(det.get('confidence', 0.0))
        if tid >= 0:
            out['smooth_confidence'] = self._push(self.history, tid, conf)
            wscore = float(det.get('weapon_score', conf))
            out['weapon_score'] = self._push(self.weapon_history, tid, wscore)
        else:
            out['smooth_confidence'] = conf
        return out

    def smooth_all(self, dets: DetectionList) -> DetectionList:
        return [self.smooth_one(d) for d in dets]


def smooth_confidence(dets: DetectionList, smoother: ScoreSmoother) -> DetectionList:
    return smoother.smooth_all(dets)


def bump_conf_if_stable(det: Detection, min_hits: int = 3) -> Detection:
    out = dict(det)
    hits = int(out.get('frame_age', 1))
    if hits >= min_hits:
        base = out.get('smooth_confidence', out.get('confidence', 0.0))
        out['confidence'] = min(1.0, float(base) * 1.05)
    return out
