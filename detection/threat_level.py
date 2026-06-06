"""Threat scoring with simple names."""

from __future__ import annotations

from typing import List, Optional

from detection.types import Detection, DetectionList


def is_threat(det: Detection) -> bool:
    return bool(det.get('is_dangerous', False))


def has_threat(dets: DetectionList) -> bool:
    return any(is_threat(d) for d in dets)


def count_threats(dets: DetectionList) -> int:
    return sum(1 for d in dets if is_threat(d))


def score_one(det: Detection) -> float:
    if not is_threat(det):
        return 0.0
    conf = float(det.get('smooth_confidence', det.get('confidence', 0.0)))
    wscore = float(det.get('weapon_score', conf))
    base = max(conf, wscore)
    if det.get('person_nearby'):
        base = min(1.0, base + 0.15)
    if det.get('zone_name'):
        base = min(1.0, base + 0.1)
    return base


def get_threat_level(score: float) -> str:
    if score >= 0.85:
        return 'critical'
    if score >= 0.65:
        return 'high'
    if score >= 0.45:
        return 'medium'
    if score > 0.0:
        return 'low'
    return 'none'


def tag_threat_levels(dets: DetectionList) -> DetectionList:
    out: DetectionList = []
    for det in dets:
        copy = dict(det)
        s = score_one(copy)
        copy['threat_level'] = get_threat_level(s)
        if s > 0:
            copy['is_dangerous'] = True
        out.append(copy)
    return out


def pick_worst_threat(dets: DetectionList) -> Optional[Detection]:
    threats = [d for d in dets if is_threat(d)]
    if not threats:
        return None
    return max(threats, key=score_one)


def scene_threat_level(dets: DetectionList) -> str:
    worst = pick_worst_threat(dets)
    if worst is None:
        return 'none'
    return worst.get('threat_level', get_threat_level(score_one(worst)))


def should_alert(dets: DetectionList, min_level: str = 'medium') -> bool:
    order = ['none', 'low', 'medium', 'high', 'critical']
    level = scene_threat_level(dets)
    try:
        return order.index(level) >= order.index(min_level)
    except ValueError:
        return False


def list_class_names(dets: DetectionList) -> List[str]:
    return sorted({d.get('class_name', 'unknown') for d in dets})
