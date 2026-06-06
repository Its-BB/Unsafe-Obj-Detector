"""Merge and split detection lists."""

from __future__ import annotations

from typing import Dict, List, Optional

from detection.box_math import get_overlap, merge_two_boxes
from detection.types import Detection, DetectionList


def same_class(a: Detection, b: Detection) -> bool:
    return a.get('class_name', '').lower() == b.get('class_name', '').lower()


def pick_higher_conf(a: Detection, b: Detection) -> Detection:
    if a.get('confidence', 0) >= b.get('confidence', 0):
        return dict(a)
    return dict(b)


def merge_pair(a: Detection, b: Detection) -> Detection:
    out = pick_higher_conf(a, b)
    out['bbox'] = merge_two_boxes(a['bbox'], b['bbox'])
    out['confidence'] = max(a.get('confidence', 0), b.get('confidence', 0))
    out['weapon_score'] = max(a.get('weapon_score', 0), b.get('weapon_score', 0))
    out['is_dangerous'] = a.get('is_dangerous') or b.get('is_dangerous')
    return out


def merge_close_boxes(dets: DetectionList, iou_min: float = 0.5) -> DetectionList:
    if not dets:
        return []
    out: DetectionList = []
    used = [False] * len(dets)
    for i, a in enumerate(dets):
        if used[i]:
            continue
        cur = dict(a)
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            b = dets[j]
            if same_class(cur, b) and get_overlap(cur['bbox'], b['bbox']) >= iou_min:
                cur = merge_pair(cur, b)
                used[j] = True
        out.append(cur)
        used[i] = True
    return out


def group_by_class(dets: DetectionList) -> Dict[str, DetectionList]:
    groups: Dict[str, DetectionList] = {}
    for d in dets:
        key = d.get('class_name', 'unknown')
        groups.setdefault(key, []).append(d)
    return groups


def group_by_track(dets: DetectionList) -> Dict[int, DetectionList]:
    groups: Dict[int, DetectionList] = {}
    for d in dets:
        tid = d.get('track_id')
        if tid is None:
            continue
        groups.setdefault(int(tid), []).append(d)
    return groups


def keep_best_per_class(dets: DetectionList) -> DetectionList:
    out: DetectionList = []
    for _, group in group_by_class(dets).items():
        best = max(group, key=lambda d: d.get('confidence', 0))
        out.append(best)
    return out


def keep_best_per_track(dets: DetectionList) -> DetectionList:
    out: DetectionList = []
    for _, group in group_by_track(dets).items():
        best = max(group, key=lambda d: d.get('confidence', 0))
        out.append(best)
    return out


def split_by_source(dets: DetectionList) -> Dict[str, DetectionList]:
    buckets: Dict[str, DetectionList] = {}
    for d in dets:
        src = d.get('detection_method', d.get('source', 'unknown'))
        buckets.setdefault(str(src), []).append(d)
    return buckets


def combine_weapon_sources(dets: DetectionList) -> DetectionList:
    weapons = [d for d in dets if d.get('type') == 'weapon' or d.get('is_weapon')]
    rest = [d for d in dets if d not in weapons]
    merged_weapons = merge_close_boxes(weapons, 0.45)
    return rest + merged_weapons


def drop_low_weapon_scores(dets: DetectionList, min_score: float = 0.25) -> DetectionList:
    out: DetectionList = []
    for d in dets:
        if d.get('type') == 'weapon' or d.get('is_weapon'):
            if d.get('weapon_score', d.get('confidence', 0)) < min_score:
                continue
        out.append(d)
    return out


def tag_source_priority(dets: DetectionList) -> DetectionList:
    priority = {
        'weapon_model': 3,
        'original_model': 2,
        'object_model': 1,
    }
    out: DetectionList = []
    for d in dets:
        copy = dict(d)
        method = copy.get('detection_method', '')
        for key, val in priority.items():
            if key in method:
                copy['source_priority'] = val
                break
        else:
            copy['source_priority'] = 0
        out.append(copy)
    return out


def resolve_conflicts(dets: DetectionList) -> DetectionList:
    tagged = tag_source_priority(dets)
    by_class = group_by_class(tagged)
    resolved: DetectionList = []
    for _, group in by_class.items():
        best = max(group, key=lambda d: (d.get('source_priority', 0), d.get('confidence', 0)))
        resolved.append(best)
    return resolved
