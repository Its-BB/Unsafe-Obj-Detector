"""Running detection stats."""

from __future__ import annotations

import csv
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from detection.threat_level import scene_threat_level
from detection.types import DetectionList


class DetectionStats:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.threat_frames = 0
        self.total_detections = 0
        self.threat_count = 0
        self.class_counts: Counter = Counter()
        self.threat_by_class: Counter = Counter()
        self.level_counts: Counter = Counter()
        self.max_fps = 0.0
        self._last_frame_time = self.start_time

    def reset(self) -> None:
        self.__init__()

    def tick_frame(self) -> float:
        now = time.time()
        self.frame_count += 1
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt <= 0:
            return 0.0
        fps = 1.0 / dt
        self.max_fps = max(self.max_fps, fps)
        return fps

    def get_fps(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.frame_count / elapsed

    def add_detections(self, dets: DetectionList) -> None:
        self.total_detections += len(dets)
        had_threat = False
        for det in dets:
            cls = det.get('class_name', 'unknown')
            self.class_counts[cls] += 1
            if det.get('is_dangerous'):
                self.threat_count += 1
                self.threat_by_class[cls] += 1
                had_threat = True
            level = det.get('threat_level', 'none')
            self.level_counts[level] += 1
        if had_threat:
            self.threat_frames += 1

    def get_summary(self, dets: DetectionList) -> Dict[str, object]:
        return {
            'fps': self.get_fps(),
            'frames': self.frame_count,
            'threat_level': scene_threat_level(dets),
            'track_count': sum(1 for d in dets if d.get('track_id') is not None),
            'detection_count': len(dets),
            'threat_frames': self.threat_frames,
            'uptime_sec': time.time() - self.start_time,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            'frames': self.frame_count,
            'fps_avg': self.get_fps(),
            'fps_max': self.max_fps,
            'total_detections': self.total_detections,
            'threat_count': self.threat_count,
            'threat_frames': self.threat_frames,
            'classes': dict(self.class_counts),
            'threat_classes': dict(self.threat_by_class),
            'levels': dict(self.level_counts),
            'started': datetime.fromtimestamp(self.start_time).isoformat(),
        }

    def export_csv(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key, val in self.to_dict().items():
                if isinstance(val, dict):
                    for k2, v2 in val.items():
                        writer.writerow([f'{key}.{k2}', v2])
                else:
                    writer.writerow([key, val])
        return str(path)
