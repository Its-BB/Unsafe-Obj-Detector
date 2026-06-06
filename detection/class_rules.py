"""Simple rules for class names and danger flags."""

from __future__ import annotations

from typing import List, Set

from detection.types import Detection, DetectionList

WEAPON_WORDS = {
    'knife', 'pistol', 'gun', 'rifle', 'handgun', 'firearm', 'blade',
    'sword', 'dagger', 'machete', 'axe', 'weapon', 'revolver', 'shotgun',
}

HUMAN_WORDS = {'person', 'human'}

TOOL_WORDS = {
    'scissors', 'hammer', 'drill', 'box_cutter', 'razor', 'screwdriver',
}

SUSPICIOUS_WORDS = {
    'remote', 'cell phone', 'bottle', 'umbrella', 'flashlight',
    'spray bottle', 'hair dryer', 'stapler', 'glue gun',
}


def norm_name(name: str) -> str:
    return name.strip().lower().replace('_', ' ')


def is_weapon_name(name: str) -> bool:
    n = norm_name(name)
    return any(w in n for w in WEAPON_WORDS)


def is_human_name(name: str) -> bool:
    return norm_name(name) in HUMAN_WORDS


def is_tool_name(name: str) -> bool:
    n = norm_name(name)
    return any(w in n for w in TOOL_WORDS)


def is_suspicious_name(name: str) -> bool:
    n = norm_name(name)
    return any(w in n for w in SUSPICIOUS_WORDS)


def name_in_list(name: str, items: List[str]) -> bool:
    n = norm_name(name)
    return any(norm_name(item) in n or n in norm_name(item) for item in items)


def set_type_from_name(det: Detection) -> Detection:
    out = dict(det)
    name = out.get('class_name', '')
    if is_human_name(name):
        out['type'] = 'human'
    elif is_weapon_name(name):
        out['type'] = 'weapon'
    elif is_tool_name(name):
        out['type'] = 'tool'
    elif is_suspicious_name(name):
        out['type'] = 'suspicious'
    elif 'type' not in out:
        out['type'] = 'object'
    return out


def set_danger_from_list(det: Detection, dangerous: List[str]) -> Detection:
    out = dict(det)
    name = out.get('class_name', '')
    out['is_dangerous'] = name_in_list(name, dangerous) or bool(out.get('is_weapon'))
    return out


def apply_class_rules(dets: DetectionList, dangerous: List[str]) -> DetectionList:
    out: DetectionList = []
    for det in dets:
        d = set_type_from_name(det)
        d = set_danger_from_name(d, dangerous)
        out.append(d)
    return out


def rename_legacy_labels(det: Detection) -> Detection:
    out = dict(det)
    name = norm_name(out.get('class_name', ''))
    mapping = {
        'cellphone': 'cell phone',
        'hand gun': 'handgun',
        'boxcutter': 'box cutter',
    }
    for old, new in mapping.items():
        if old in name:
            out['class_name'] = new
            break
    return out


def clean_labels(dets: DetectionList) -> DetectionList:
    return [rename_legacy_labels(d) for d in dets]


def count_by_type(dets: DetectionList) -> dict:
    counts = {'human': 0, 'weapon': 0, 'tool': 0, 'suspicious': 0, 'object': 0}
    for d in dets:
        t = d.get('type', 'object')
        counts[t] = counts.get(t, 0) + 1
    return counts


def has_human(dets: DetectionList) -> bool:
    return any(d.get('type') == 'human' for d in dets)


def has_weapon(dets: DetectionList) -> bool:
    return any(d.get('type') == 'weapon' or d.get('is_weapon') for d in dets)


def list_weapon_names(dets: DetectionList) -> List[str]:
    return sorted({d.get('class_name', '') for d in dets if is_weapon_name(d.get('class_name', ''))})


def boost_tools_near_humans(dets: DetectionList) -> DetectionList:
    if not has_human(dets):
        return dets
    out: DetectionList = []
    for d in dets:
        copy = dict(d)
        if copy.get('type') == 'tool':
            copy['person_nearby'] = True
            if copy.get('confidence', 0) > 0.4:
                copy['is_dangerous'] = True
        out.append(copy)
    return out
