"""Export session reports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from detection.event_log import EventLog
from detection.count_stats import DetectionStats


def make_text_report(stats: DetectionStats, events: EventLog) -> str:
    data = stats.to_dict()
    lines = [
        'Detection Session Report',
        '========================',
        f"Generated: {datetime.now().isoformat()}",
        f"Frames: {data.get('frames', 0)}",
        f"Avg FPS: {data.get('fps_avg', 0):.2f}",
        f"Total detections: {data.get('total_detections', 0)}",
        f"Threat count: {data.get('threat_count', 0)}",
        f"Threat frames: {data.get('threat_frames', 0)}",
        '',
        'Classes:',
    ]
    for cls, count in sorted((data.get('classes') or {}).items(), key=lambda x: -x[1]):
        lines.append(f'  {cls}: {count}')
    lines.append('')
    lines.append('Recent threat events:')
    for ev in events.get_threat_events(120):
        lines.append(f"  {ev.get('iso')} level={ev.get('threat_level')} n={len(ev.get('threats', []))}")
    return '\n'.join(lines)


def save_text_report(stats: DetectionStats, events: EventLog, folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text(make_text_report(stats, events), encoding='utf-8')
    return str(path)


def save_json_report(stats: DetectionStats, events: EventLog, folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        'stats': stats.to_dict(),
        'events': events.to_list(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return str(path)
