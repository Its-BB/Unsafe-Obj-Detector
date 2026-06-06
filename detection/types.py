from typing import Any, Dict, List, Tuple, TypedDict, Optional

Box = Tuple[int, int, int, int]


class Detection(TypedDict, total=False):
    bbox: Box
    confidence: float
    class_name: str
    type: str
    is_dangerous: bool
    is_weapon: bool
    weapon_score: float
    detection_method: str
    track_id: int
    threat_level: str
    zone_name: str
    person_nearby: bool
    smooth_confidence: float
    frame_age: int
    source: str


DetectionList = List[Detection]
ConfigDict = Dict[str, Any]
