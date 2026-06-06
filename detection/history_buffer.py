"""Keep recent frames and detections."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from detection.types import DetectionList


FrameItem = Dict[str, Any]


class FrameHistory:
    def __init__(self, max_frames: int = 30):
        self.max_frames = max(1, max_frames)
        self.items: Deque[FrameItem] = deque(maxlen=self.max_frames)

    def reset(self) -> None:
        self.items.clear()

    def push(self, frame: np.ndarray, dets: DetectionList, extra: Optional[dict] = None) -> None:
        item: FrameItem = {
            'time': time.time(),
            'frame': frame.copy(),
            'detections': [dict(d) for d in dets],
        }
        if extra:
            item.update(extra)
        self.items.append(item)

    def get_last(self) -> Optional[FrameItem]:
        if not self.items:
            return None
        return self.items[-1]

    def get_prev_frame(self) -> Optional[np.ndarray]:
        if len(self.items) < 2:
            return None
        return self.items[-2]['frame']

    def get_recent_threats(self, seconds: float = 10.0) -> DetectionList:
        now = time.time()
        out: DetectionList = []
        for item in reversed(self.items):
            if now - float(item['time']) > seconds:
                break
            for det in item['detections']:
                if det.get('is_dangerous'):
                    out.append(det)
        return out

    def count_threat_frames(self, seconds: float = 5.0) -> int:
        now = time.time()
        count = 0
        for item in self.items:
            if now - float(item['time']) > seconds:
                continue
            if any(d.get('is_dangerous') for d in item['detections']):
                count += 1
        return count

    def get_clip_frames(self, n: int = 10) -> List[np.ndarray]:
        return [item['frame'] for item in list(self.items)[-n:]]

    def get_timeline(self) -> List[Tuple[float, int]]:
        return [
            (float(item['time']), sum(1 for d in item['detections'] if d.get('is_dangerous')))
            for item in self.items
        ]
