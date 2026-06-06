"""Link weapons to nearby people."""

from __future__ import annotations

from typing import Optional

from detection.box_math import distance_between, get_center
from detection.types import Detection, DetectionList


def get_humans(dets: DetectionList) -> DetectionList:
    return [d for d in dets if d.get('type') == 'human' or d.get('class_name', '').lower() == 'person']


def get_weapons(dets: DetectionList) -> DetectionList:
    return [
        d for d in dets
        if d.get('type') == 'weapon' or d.get('is_weapon', False) or d.get('is_dangerous', False)
    ]


def person_near_weapon(weapon: Detection, humans: DetectionList, max_dist: float) -> bool:
    if not humans:
        return False
    wbox = weapon['bbox']
    for h in humans:
        if distance_between(wbox, h['bbox']) <= max_dist:
            return True
    return False


def find_nearest_person(weapon: Detection, humans: DetectionList) -> Optional[Detection]:
    if not humans:
        return None
    return min(humans, key=lambda h: distance_between(weapon['bbox'], h['bbox']))


def mark_person_nearby(dets: DetectionList, max_dist: float = 200.0) -> DetectionList:
    humans = get_humans(dets)
    out: DetectionList = []
    for det in dets:
        copy = dict(det)
        if copy.get('type') == 'weapon' or copy.get('is_weapon') or copy.get('is_dangerous'):
            copy['person_nearby'] = person_near_weapon(copy, humans, max_dist)
        else:
            copy['person_nearby'] = False
        out.append(copy)
    return out


def boost_weapon_near_person(dets: DetectionList) -> DetectionList:
    out: DetectionList = []
    for det in dets:
        copy = dict(det)
        if copy.get('person_nearby') and copy.get('is_dangerous'):
            conf = float(copy.get('confidence', 0.0))
            copy['confidence'] = min(1.0, conf + 0.1)
            w = float(copy.get('weapon_score', conf))
            copy['weapon_score'] = min(1.0, w + 0.1)
        out.append(copy)
    return out
