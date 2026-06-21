"""Final passes on a detection list."""

from __future__ import annotations

from typing import Callable, List

from detection.class_rules import apply_class_rules, boost_tools_near_humans, clean_labels
from detection.filter_boxes import apply_nms, filter_by_conf_or_weapon_score, keep_top_n
from detection.merge_rules import combine_weapon_sources, keep_best_per_track, resolve_conflicts
from detection.types import Detection, DetectionList


PassFn = Callable[[DetectionList], DetectionList]


def run_passes(dets: DetectionList, passes: List[PassFn]) -> DetectionList:
    out = dets
    for fn in passes:
        out = fn(out)
    return out


def drop_empty_names(dets: DetectionList) -> DetectionList:
    return [d for d in dets if d.get('class_name', '').strip()]


def drop_zero_conf(dets: DetectionList) -> DetectionList:
    return [d for d in dets if d.get('confidence', 0) > 0.001]


def fill_missing_type(dets: DetectionList) -> DetectionList:
    out: DetectionList = []
    for d in dets:
        copy = dict(d)
        if 'type' not in copy:
            copy['type'] = 'object'
        out.append(copy)
    return out


def fill_missing_method(dets: DetectionList) -> DetectionList:
    out: DetectionList = []
    for d in dets:
        copy = dict(d)
        if 'detection_method' not in copy:
            copy['detection_method'] = 'standard'
        out.append(copy)
    return out


def cap_list_size(dets: DetectionList, max_items: int = 50) -> DetectionList:
    return keep_top_n(dets, max_items)


def full_post_process(
    dets: DetectionList,
    dangerous: List[str],
    min_conf: float,
    nms_iou: float,
    max_items: int = 50,
    min_weapon_score: float = 0.2,
) -> DetectionList:
    steps: List[PassFn] = [
        drop_empty_names,
        drop_zero_conf,
        clean_labels,
        lambda x: apply_class_rules(x, dangerous),
        lambda x: filter_by_conf_or_weapon_score(x, min_conf, min_weapon_score),
        lambda x: apply_nms(x, nms_iou),
        combine_weapon_sources,
        resolve_conflicts,
        boost_tools_near_humans,
        keep_best_per_track,
        fill_missing_type,
        fill_missing_method,
        lambda x: cap_list_size(x, max_items),
    ]
    return run_passes(dets, steps)


def diff_lists(before: DetectionList, after: DetectionList) -> dict:
    return {
        'before': len(before),
        'after': len(after),
        'removed': len(before) - len(after),
        'danger_before': sum(1 for d in before if d.get('is_dangerous')),
        'danger_after': sum(1 for d in after if d.get('is_dangerous')),
    }
