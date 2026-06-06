"""Run YOLO and turn results into simple dicts."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from detection.types import Detection, DetectionList

logger = logging.getLogger(__name__)


def run_model(model: YOLO, frame: np.ndarray, conf: float, imgsz: int = 640) -> Any:
    return model(frame, conf=conf, imgsz=imgsz, verbose=False)


def get_class_name(model: YOLO, class_id: int) -> str:
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, f'class_{class_id}'))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f'class_{class_id}'


def parse_one_box(box: Any, model: YOLO) -> Optional[Detection]:
    try:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())
        class_name = get_class_name(model, class_id)
        return {
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'confidence': confidence,
            'class_name': class_name,
            'source': 'yolo',
        }
    except Exception as exc:
        logger.debug('parse box failed: %s', exc)
        return None


def parse_yolo_result(result: Any, model: YOLO) -> DetectionList:
    out: DetectionList = []
    boxes = getattr(result, 'boxes', None)
    if boxes is None:
        return out
    for box in boxes:
        det = parse_one_box(box, model)
        if det is not None:
            out.append(det)
    return out


def run_and_parse(model: YOLO, frame: np.ndarray, conf: float, imgsz: int = 640) -> DetectionList:
    try:
        results = run_model(model, frame, conf, imgsz)
        dets: DetectionList = []
        for result in results:
            dets.extend(parse_yolo_result(result, model))
        return dets
    except Exception as exc:
        logger.error('yolo run failed: %s', exc)
        return []


def tag_detection_type(det: Detection) -> Detection:
    name = det.get('class_name', '').lower()
    if name == 'person':
        det['type'] = 'human'
    elif name in ('knife', 'pistol', 'gun', 'rifle'):
        det['type'] = 'weapon'
    else:
        det['type'] = 'object'
    return det


def mark_dangerous(det: Detection, dangerous_list: List[str]) -> Detection:
    name = det.get('class_name', '').lower()
    det['is_dangerous'] = any(d.lower() in name for d in dangerous_list)
    return det


def run_object_pass(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    dangerous_list: List[str],
    imgsz: int = 640,
) -> DetectionList:
    raw = run_and_parse(model, frame, conf, imgsz)
    out: DetectionList = []
    for det in raw:
        tag_detection_type(det)
        mark_dangerous(det, dangerous_list)
        det['detection_method'] = 'object_model'
        out.append(det)
    return out


def run_human_pass(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    imgsz: int = 640,
) -> DetectionList:
    raw = run_and_parse(model, frame, conf, imgsz)
    humans: DetectionList = []
    for det in raw:
        if det.get('class_name', '').lower() == 'person':
            det['type'] = 'human'
            det['is_dangerous'] = False
            det['detection_method'] = 'human_model'
            humans.append(det)
    return humans


def merge_weapon_results(weapon_dets: DetectionList) -> DetectionList:
    out: DetectionList = []
    for det in weapon_dets:
        copy = dict(det)
        copy['type'] = 'weapon'
        copy['is_dangerous'] = copy.get('is_weapon', copy.get('is_dangerous', False))
        if 'detection_method' not in copy:
            copy['detection_method'] = 'weapon_model'
        out.append(copy)
    return out
