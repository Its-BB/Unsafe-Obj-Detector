"""Save frames, logs, and threat clips."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from detection.draw_overlay import draw_boxes
from detection.types import DetectionList

logger = logging.getLogger(__name__)


def make_output_dir(base: str, sub: str = '') -> Path:
    root = Path(base)
    if sub:
        root = root / sub
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_frame(frame: np.ndarray, folder: Path, prefix: str = 'frame') -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = folder / f'{prefix}_{ts}.jpg'
    cv2.imwrite(str(path), frame)
    return str(path)


def save_json_log(data: dict, folder: Path, prefix: str = 'log') -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = folder / f'{prefix}_{ts}.json'
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    return str(path)


def save_threat_frame(
    frame: np.ndarray,
    dets: DetectionList,
    config: dict,
    report: Optional[dict] = None,
) -> Optional[str]:
    logging_cfg = config.get('logging', {})
    if not logging_cfg.get('save_detections', True):
        return None
    base = logging_cfg.get('output_dir', 'detections')
    folder = make_output_dir(base, 'threats')
    annotated = draw_boxes(frame.copy(), dets, config)
    path = save_frame(annotated, folder, 'threat')
    meta = {
        'path': path,
        'time': datetime.now().isoformat(),
        'count': len(dets),
        'classes': [d.get('class_name') for d in dets],
        'report': report,
    }
    save_json_log(meta, folder, 'threat_meta')
    logger.warning('saved threat frame: %s', path)
    return path


def save_incident_bundle(
    frame: np.ndarray,
    dets: DetectionList,
    config: dict,
    report: dict,
) -> Dict[str, str]:
    base = config.get('logging', {}).get('output_dir', 'detections')
    folder = make_output_dir(base, 'incidents')
    incident_id = report.get('incident_id', f'INC_{int(time.time())}')
    frame_path = folder / f'{incident_id}_frame.jpg'
    report_path = folder / f'{incident_id}_report.json'
    annotated = draw_boxes(frame.copy(), dets, config)
    cv2.imwrite(str(frame_path), annotated)
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    return {'frame': str(frame_path), 'report': str(report_path)}


def rotate_old_files(folder: Path, keep: int = 200) -> int:
    files = sorted(folder.glob('*.jpg'), key=lambda p: p.stat().st_mtime)
    removed = 0
    while len(files) > keep:
        old = files.pop(0)
        try:
            old.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def save_clip_frames(frames: List[np.ndarray], folder: Path, prefix: str = 'clip') -> List[str]:
    paths: List[str] = []
    for i, fr in enumerate(frames):
        p = folder / f'{prefix}_{i:03d}.jpg'
        cv2.imwrite(str(p), fr)
        paths.append(str(p))
    return paths


class EvidenceWriter:
    def __init__(self, config: dict):
        self.config = config
        self.last_save_time = 0.0
        self.min_gap = float(config.get('detection_pipeline', {}).get('save_gap_seconds', 2.0))
        self.output = config.get('logging', {}).get('output_dir', 'detections')

    def can_save_now(self) -> bool:
        return time.time() - self.last_save_time >= self.min_gap

    def save_if_needed(self, frame: np.ndarray, dets: DetectionList, report: Optional[dict] = None) -> Optional[str]:
        threats = [d for d in dets if d.get('is_dangerous')]
        if not threats:
            return None
        if not self.can_save_now():
            return None
        path = save_threat_frame(frame, threats, self.config, report)
        self.last_save_time = time.time()
        return path

    def save_session_summary(self, summary: dict) -> str:
        folder = make_output_dir(self.output, 'sessions')
        return save_json_log(summary, folder, 'session')
