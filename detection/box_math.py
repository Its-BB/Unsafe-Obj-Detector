"""Simple box math helpers."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from detection.types import Box, Detection, DetectionList

BoxList = List[Box]


def get_width(box: Box) -> int:
    return max(0, box[2] - box[0])


def get_height(box: Box) -> int:
    return max(0, box[3] - box[1])


def get_area(box: Box) -> int:
    return get_width(box) * get_height(box)


def get_center(box: Box) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def get_box_from_detection(det: Detection) -> Box:
    return det['bbox']


def make_box(x1: float, y1: float, x2: float, y2: float) -> Box:
    return int(x1), int(y1), int(x2), int(y2)


def clip_box(box: Box, frame_w: int, frame_h: int) -> Box:
    x1 = max(0, min(box[0], frame_w - 1))
    y1 = max(0, min(box[1], frame_h - 1))
    x2 = max(0, min(box[2], frame_w))
    y2 = max(0, min(box[3], frame_h))
    if x2 <= x1:
        x2 = min(frame_w, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_h, y1 + 1)
    return x1, y1, x2, y2


def expand_box(box: Box, pad: int, frame_w: int, frame_h: int) -> Box:
    return clip_box((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), frame_w, frame_h)


def shrink_box(box: Box, pad: int) -> Box:
    x1, y1, x2, y2 = box
    return x1 + pad, y1 + pad, x2 - pad, y2 - pad


def box_inside(inner: Box, outer: Box) -> bool:
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def boxes_touch(a: Box, b: Box) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def get_overlap(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = get_area(a) + get_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def same_box(a: Box, b: Box, tol: int = 8) -> bool:
    return all(abs(a[i] - b[i]) <= tol for i in range(4))


def merge_two_boxes(a: Box, b: Box) -> Box:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def merge_boxes(boxes: BoxList) -> Optional[Box]:
    if not boxes:
        return None
    out = boxes[0]
    for b in boxes[1:]:
        out = merge_two_boxes(out, b)
    return out


def sort_by_area(dets: DetectionList, big_first: bool = True) -> DetectionList:
    return sorted(dets, key=lambda d: get_area(d['bbox']), reverse=big_first)


def sort_by_conf(dets: DetectionList) -> DetectionList:
    return sorted(dets, key=lambda d: d.get('confidence', 0.0), reverse=True)


def pick_biggest(dets: DetectionList) -> Optional[Detection]:
    if not dets:
        return None
    return sort_by_area(dets)[0]


def pick_best_conf(dets: DetectionList) -> Optional[Detection]:
    if not dets:
        return None
    return sort_by_conf(dets)[0]


def distance_between(a: Box, b: Box) -> float:
    ax, ay = get_center(a)
    bx, by = get_center(b)
    return math.hypot(ax - bx, ay - by)


def nearest_box(target: Box, boxes: BoxList) -> Optional[Box]:
    if not boxes:
        return None
    return min(boxes, key=lambda b: distance_between(target, b))


def split_box(box: Box, parts: int = 2) -> BoxList:
    if parts <= 1:
        return [box]
    x1, y1, x2, y2 = box
    w = (x2 - x1) / parts
    out: BoxList = []
    for i in range(parts):
        sx1 = int(x1 + w * i)
        sx2 = int(x1 + w * (i + 1))
        out.append((sx1, y1, sx2, y2))
    return out


def box_to_xywh(box: Box) -> Tuple[int, int, int, int]:
    return box[0], box[1], get_width(box), get_height(box)


def xywh_to_box(x: int, y: int, w: int, h: int) -> Box:
    return x, y, x + w, y + h


def normalize_box(box: Box, frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
    return (
        box[0] / frame_w,
        box[1] / frame_h,
        box[2] / frame_w,
        box[3] / frame_h,
    )


def denormalize_box(norm: Tuple[float, float, float, float], frame_w: int, frame_h: int) -> Box:
    return (
        int(norm[0] * frame_w),
        int(norm[1] * frame_h),
        int(norm[2] * frame_w),
        int(norm[3] * frame_h),
    )


def box_aspect(box: Box) -> float:
    w = get_width(box)
    h = get_height(box)
    if h == 0:
        return 0.0
    return w / h


def is_tiny_box(box: Box, min_pixels: int = 16) -> bool:
    return get_width(box) < min_pixels or get_height(box) < min_pixels


def is_huge_box(box: Box, frame_w: int, frame_h: int, ratio: float = 0.95) -> bool:
    frame_area = frame_w * frame_h
    if frame_area <= 0:
        return False
    return get_area(box) / frame_area >= ratio


def move_box(box: Box, dx: int, dy: int) -> Box:
    return box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy


def scale_box(box: Box, scale: float, frame_w: int, frame_h: int) -> Box:
    cx, cy = get_center(box)
    w = get_width(box) * scale
    h = get_height(box) * scale
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    return clip_box((x1, y1, x2, y2), frame_w, frame_h)


def copy_detection(det: Detection) -> Detection:
    return dict(det)


def set_box_on_detection(det: Detection, box: Box) -> Detection:
    out = copy_detection(det)
    out['bbox'] = box
    return out
