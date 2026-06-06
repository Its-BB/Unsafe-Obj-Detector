"""Box colors."""

from __future__ import annotations

from typing import Tuple

from detection.types import Detection

Color = Tuple[int, int, int]

RED: Color = (0, 0, 255)
GREEN: Color = (0, 255, 0)
BLUE: Color = (255, 0, 0)
YELLOW: Color = (0, 255, 255)
ORANGE: Color = (0, 140, 255)
WHITE: Color = (255, 255, 255)
PURPLE: Color = (255, 0, 255)


def get_box_color(det: Detection) -> Color:
    level = det.get('threat_level', 'none')
    if level == 'critical':
        return (0, 0, 220)
    if det.get('is_dangerous'):
        method = det.get('detection_method', '')
        if 'multi_angle' in method:
            return (0, 0, 200)
        if 'multi_scale' in method:
            return (0, 50, 255)
        return RED
    if det.get('type') == 'human':
        return GREEN
    if det.get('type') == 'weapon':
        return ORANGE
    return BLUE


def get_box_thickness(det: Detection) -> int:
    if det.get('threat_level') == 'critical':
        return 4
    if det.get('is_dangerous'):
        return 3
    return 2


def get_label_color(det: Detection) -> Color:
    return WHITE


def get_zone_color(name: str) -> Color:
    if name == 'full':
        return (80, 80, 80)
    return YELLOW
