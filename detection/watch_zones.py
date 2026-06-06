"""Watch zones — mark detections inside named regions."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from detection.box_math import get_center
from detection.types import Box, Detection, DetectionList

Zone = Dict[str, object]


def make_zone(name: str, x1: float, y1: float, x2: float, y2: float, enabled: bool = True) -> Zone:
    return {
        'name': name,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'enabled': enabled,
    }


def make_full_frame_zone(name: str = 'full') -> Zone:
    return make_zone(name, 0.0, 0.0, 1.0, 1.0)


def zone_to_pixels(zone: Zone, frame_w: int, frame_h: int) -> Box:
    x1 = int(float(zone['x1']) * frame_w)
    y1 = int(float(zone['y1']) * frame_h)
    x2 = int(float(zone['x2']) * frame_w)
    y2 = int(float(zone['y2']) * frame_h)
    return x1, y1, x2, y2


def point_in_zone(px: float, py: float, zone: Zone, frame_w: int, frame_h: int) -> bool:
    if not zone.get('enabled', True):
        return False
    x1, y1, x2, y2 = zone_to_pixels(zone, frame_w, frame_h)
    return x1 <= px <= x2 and y1 <= py <= y2


def box_in_zone(box: Box, zone: Zone, frame_w: int, frame_h: int) -> bool:
    cx, cy = get_center(box)
    return point_in_zone(cx, cy, zone, frame_w, frame_h)


def find_zone_for_box(box: Box, zones: List[Zone], frame_w: int, frame_h: int) -> Optional[str]:
    for zone in zones:
        if box_in_zone(box, zone, frame_w, frame_h):
            return str(zone.get('name', 'zone'))
    return None


def tag_zones(dets: DetectionList, zones: List[Zone], frame_w: int, frame_h: int) -> DetectionList:
    out: DetectionList = []
    for det in dets:
        copy = dict(det)
        name = find_zone_for_box(det['bbox'], zones, frame_w, frame_h)
        if name:
            copy['zone_name'] = name
        out.append(copy)
    return out


def load_zones_from_config(config: dict) -> List[Zone]:
    pipe = config.get('detection_pipeline', {})
    raw = pipe.get('zones', [])
    zones: List[Zone] = []
    if not raw:
        zones.append(make_full_frame_zone('full'))
        return zones
    for item in raw:
        if not isinstance(item, dict):
            continue
        zones.append(make_zone(
            str(item.get('name', 'zone')),
            float(item.get('x1', 0)),
            float(item.get('y1', 0)),
            float(item.get('x2', 1)),
            float(item.get('y2', 1)),
            bool(item.get('enabled', True)),
        ))
    return zones


class ZoneManager:
    def __init__(self, config: dict):
        self.zones = load_zones_from_config(config)
        self.frame_w = 0
        self.frame_h = 0

    def set_frame_size(self, w: int, h: int) -> None:
        self.frame_w = w
        self.frame_h = h

    def apply(self, dets: DetectionList) -> DetectionList:
        if self.frame_w <= 0 or self.frame_h <= 0:
            return dets
        return tag_zones(dets, self.zones, self.frame_w, self.frame_h)

    def get_zone_boxes(self) -> List[Tuple[str, Box]]:
        out: List[Tuple[str, Box]] = []
        for z in self.zones:
            if not z.get('enabled', True):
                continue
            out.append((str(z['name']), zone_to_pixels(z, self.frame_w, self.frame_h)))
        return out
