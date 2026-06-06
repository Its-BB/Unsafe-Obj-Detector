"""Draw boxes and HUD on frames."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Optional

from detection.colors import get_box_color, get_box_thickness, get_label_color, get_zone_color
from detection.threat_level import scene_threat_level
from detection.types import Detection, DetectionList


def draw_box(frame: np.ndarray, det: Detection) -> np.ndarray:
    x1, y1, x2, y2 = det['bbox']
    color = get_box_color(det)
    thick = get_box_thickness(det)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    return frame


def make_label(det: Detection, show_conf: bool = True) -> str:
    name = det.get('class_name', 'obj')
    parts = [name]
    if show_conf:
        conf = det.get('smooth_confidence', det.get('confidence', 0.0))
        parts.append(f'{conf:.2f}')
    tid = det.get('track_id')
    if tid is not None:
        parts.append(f'id{tid}')
    if det.get('zone_name'):
        parts.append(det['zone_name'])
    if det.get('person_nearby'):
        parts.append('near_person')
    label = ' '.join(parts)
    if det.get('is_dangerous'):
        label = f'ALERT {label}'
    return label


def draw_label(frame: np.ndarray, det: Detection, font_scale: float = 0.6) -> np.ndarray:
    x1, y1, x2, y2 = det['bbox']
    label = make_label(det, True)
    color = get_box_color(det)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    bg_y1 = max(0, y1 - h - 8)
    cv2.rectangle(frame, (x1, bg_y1), (min(frame.shape[1], x1 + w + 8), y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, get_label_color(det), 1)
    return frame


def draw_zone(frame: np.ndarray, name: str, box: tuple[int, int, int, int]) -> np.ndarray:
    color = get_zone_color(name)
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 1)
    cv2.putText(frame, name, (box[0] + 4, box[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def draw_boxes(frame: np.ndarray, dets: DetectionList, config: dict) -> np.ndarray:
    out = frame.copy()
    display = config.get('display', {})
    font_scale = float(display.get('font_scale', 0.6))
    show_conf = bool(display.get('show_confidence', True))
    for det in dets:
        draw_box(out, det)
        if show_conf or det.get('is_dangerous'):
            draw_label(out, det, font_scale)
    return out


def draw_status_bar(frame: np.ndarray, stats: Dict[str, object], config: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    display = config.get('display', {})
    if display.get('show_fps', True):
        fps = float(stats.get('fps', 0.0))
        cv2.putText(out, f'FPS: {fps:.1f}', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    threat = str(stats.get('threat_level', 'none'))
    cv2.putText(out, f'Threat: {threat}', (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
    tracks = int(stats.get('track_count', 0))
    cv2.putText(out, f'Tracks: {tracks}', (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    title = str(display.get('window_title', 'Detection'))
    cv2.putText(out, title, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    status_color = (0, 255, 0) if threat == 'none' else (0, 0, 255)
    cv2.circle(out, (w - 24, 24), 8, status_color, -1)
    return out


def draw_all(
    frame: np.ndarray,
    dets: DetectionList,
    stats: Dict[str, object],
    config: dict,
    zone_boxes: Optional[list] = None,
) -> np.ndarray:
    out = frame.copy()
    if zone_boxes:
        for name, box in zone_boxes:
            draw_zone(out, name, box)
    out = draw_boxes(out, dets, config)
    out = draw_status_bar(out, stats, config)
    return out
