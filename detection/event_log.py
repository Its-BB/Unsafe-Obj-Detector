"""Simple detection event log."""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

from detection.threat_level import scene_threat_level
from detection.types import DetectionList


class EventLog:
    def __init__(self, max_events: int = 500):
        self.max_events = max_events
        self.events: Deque[dict] = deque(maxlen=max_events)

    def reset(self) -> None:
        self.events.clear()

    def add_frame(self, dets: DetectionList, extra: Optional[dict] = None) -> None:
        event = {
            'time': time.time(),
            'iso': datetime.now().isoformat(),
            'count': len(dets),
            'threat_level': scene_threat_level(dets),
            'threats': [
                {
                    'class': d.get('class_name'),
                    'conf': d.get('confidence'),
                    'track_id': d.get('track_id'),
                    'bbox': d.get('bbox'),
                }
                for d in dets if d.get('is_dangerous')
            ],
        }
        if extra:
            event.update(extra)
        self.events.append(event)

    def get_recent(self, n: int = 20) -> List[dict]:
        return list(self.events)[-n:]

    def get_threat_events(self, seconds: float = 60.0) -> List[dict]:
        now = time.time()
        return [e for e in self.events if now - e['time'] <= seconds and e.get('threats')]

    def count_alerts(self, seconds: float = 300.0) -> int:
        return len(self.get_threat_events(seconds))

    def to_list(self) -> List[dict]:
        return list(self.events)

    def save_json(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(self.to_list(), f, indent=2)
        return str(path)

    def print_last(self) -> None:
        if not self.events:
            return
        last = self.events[-1]
        print(f"[{last['iso']}] level={last['threat_level']} count={last['count']}")
