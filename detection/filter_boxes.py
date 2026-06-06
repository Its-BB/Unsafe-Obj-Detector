"""Filter and merge detection lists."""

from __future__ import annotations

from typing import Callable, List, Optional, Set

from detection.box_math import get_area, get_overlap, is_huge_box, is_tiny_box, sort_by_conf
from detection.types import Detection, DetectionList


def filter_by_conf(dets: DetectionList, min_conf: float) -> DetectionList:
    return [d for d in dets if d.get('confidence', 0.0) >= min_conf]


def filter_by_class(dets: DetectionList, class_names: Set[str]) -> DetectionList:
    wanted = {c.lower() for c in class_names}
    return [d for d in dets if d.get('class_name', '').lower() in wanted]


def filter_out_class(dets: DetectionList, class_names: Set[str]) -> DetectionList:
    blocked = {c.lower() for c in class_names}
    return [d for d in dets if d.get('class_name', '').lower() not in blocked]


def filter_by_type(dets: DetectionList, det_type: str) -> DetectionList:
    return [d for d in dets if d.get('type') == det_type]


def filter_dangerous_only(dets: DetectionList) -> DetectionList:
    return [d for d in dets if d.get('is_dangerous', False)]


def filter_safe_only(dets: DetectionList) -> DetectionList:
    return [d for d in dets if not d.get('is_dangerous', False)]


def filter_by_min_area(dets: DetectionList, min_area: int) -> DetectionList:
    return [d for d in dets if get_area(d['bbox']) >= min_area]


def filter_by_max_area(dets: DetectionList, max_area: int) -> DetectionList:
    return [d for d in dets if get_area(d['bbox']) <= max_area]


def filter_tiny_boxes(dets: DetectionList, min_pixels: int = 12) -> DetectionList:
    return [d for d in dets if not is_tiny_box(d['bbox'], min_pixels)]


def filter_huge_boxes(dets: DetectionList, frame_w: int, frame_h: int, ratio: float = 0.92) -> DetectionList:
    return [d for d in dets if not is_huge_box(d['bbox'], frame_w, frame_h, ratio)]


def apply_nms(dets: DetectionList, iou_limit: float = 0.45) -> DetectionList:
    if not dets:
        return []
    ordered = sort_by_conf(dets)
    kept: DetectionList = []
    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        ordered = [
            d for d in ordered
            if get_overlap(best['bbox'], d['bbox']) < iou_limit
            or d.get('class_name') != best.get('class_name')
        ]
    return kept


def remove_duplicates(dets: DetectionList, iou_limit: float = 0.9) -> DetectionList:
    return apply_nms(dets, iou_limit)


def keep_top_n(dets: DetectionList, n: int) -> DetectionList:
    if n <= 0:
        return []
    return sort_by_conf(dets)[:n]


def run_custom_filter(dets: DetectionList, fn: Callable[[Detection], bool]) -> DetectionList:
    return [d for d in dets if fn(d)]


def merge_lists(*lists: DetectionList) -> DetectionList:
    out: DetectionList = []
    for lst in lists:
        out.extend(lst)
    return out


def split_weapons_and_rest(dets: DetectionList) -> tuple[DetectionList, DetectionList]:
    weapons: DetectionList = []
    rest: DetectionList = []
    for d in dets:
        if d.get('type') == 'weapon' or d.get('is_weapon', False):
            weapons.append(d)
        else:
            rest.append(d)
    return weapons, rest


def filter_pipeline(
    dets: DetectionList,
    frame_w: int,
    frame_h: int,
    min_conf: float,
    nms_iou: float,
    min_box_pixels: int = 10,
) -> DetectionList:
    out = filter_by_conf(dets, min_conf)
    out = filter_tiny_boxes(out, min_box_pixels)
    out = filter_huge_boxes(out, frame_w, frame_h)
    out = apply_nms(out, nms_iou)
    return out


def pick_primary_threat(dets: DetectionList) -> Optional[Detection]:
    threats = filter_dangerous_only(dets)
    if not threats:
        return None
    return sort_by_conf(threats)[0]
