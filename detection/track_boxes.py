"""Simple box tracker — gives each detection a track_id."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from detection.box_math import get_center, get_overlap
from detection.types import Detection, DetectionList

TrackState = Dict[str, object]


class ObjectTracker:
    def __init__(self, max_age: float = 2.0, min_hits: int = 2, iou_match: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_match = iou_match
        self.next_id = 1
        self.tracks: Dict[int, TrackState] = {}

    def reset(self) -> None:
        self.next_id = 1
        self.tracks.clear()

    def _new_track(self, det: Detection, now: float) -> int:
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {
            'bbox': det['bbox'],
            'class_name': det.get('class_name', 'unknown'),
            'hits': 1,
            'misses': 0,
            'last_seen': now,
            'first_seen': now,
            'confidence_sum': det.get('confidence', 0.0),
            'danger_hits': 1 if det.get('is_dangerous') else 0,
        }
        return tid

    def _update_track(self, tid: int, det: Detection, now: float) -> None:
        t = self.tracks[tid]
        t['bbox'] = det['bbox']
        t['hits'] = int(t['hits']) + 1
        t['misses'] = 0
        t['last_seen'] = now
        t['confidence_sum'] = float(t['confidence_sum']) + det.get('confidence', 0.0)
        if det.get('is_dangerous'):
            t['danger_hits'] = int(t['danger_hits']) + 1

    def _drop_old(self, now: float) -> None:
        dead = []
        for tid, t in self.tracks.items():
            if now - float(t['last_seen']) > self.max_age:
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

    def match_detections(self, dets: DetectionList) -> List[Tuple[int, Detection]]:
        now = time.time()
        self._drop_old(now)
        if not dets:
            return []

        unmatched_tracks = set(self.tracks.keys())
        pairs: List[Tuple[int, Detection]] = []

        for det in dets:
            best_tid: Optional[int] = None
            best_iou = 0.0
            for tid in list(unmatched_tracks):
                track_box = self.tracks[tid]['bbox']
                iou = get_overlap(det['bbox'], track_box)
                same_class = (
                    self.tracks[tid].get('class_name') == det.get('class_name')
                    or det.get('is_dangerous')
                )
                if iou >= self.iou_match and iou > best_iou and same_class:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                self._update_track(best_tid, det, now)
                unmatched_tracks.discard(best_tid)
                pairs.append((best_tid, det))
            else:
                tid = self._new_track(det, now)
                pairs.append((tid, det))

        for tid in unmatched_tracks:
            self.tracks[tid]['misses'] = int(self.tracks[tid]['misses']) + 1

        return pairs

    def attach_ids(self, dets: DetectionList) -> DetectionList:
        pairs = self.match_detections(dets)
        out: DetectionList = []
        for tid, det in pairs:
            copy = dict(det)
            copy['track_id'] = tid
            track = self.tracks.get(tid, {})
            hits = int(track.get('hits', 1))
            if hits < self.min_hits:
                copy['is_dangerous'] = False
            out.append(copy)
        return out

    def get_track_count(self) -> int:
        return len(self.tracks)

    def get_active_tracks(self) -> Dict[int, TrackState]:
        return dict(self.tracks)

    def get_track_age(self, track_id: int) -> float:
        t = self.tracks.get(track_id)
        if not t:
            return 0.0
        return float(t['last_seen']) - float(t['first_seen'])

    def is_stable_track(self, track_id: int) -> bool:
        t = self.tracks.get(track_id)
        if not t:
            return False
        return int(t['hits']) >= self.min_hits

    def get_avg_conf(self, track_id: int) -> float:
        t = self.tracks.get(track_id)
        if not t:
            return 0.0
        hits = int(t['hits'])
        if hits <= 0:
            return 0.0
        return float(t['confidence_sum']) / hits


def track_objects(dets: DetectionList, tracker: ObjectTracker) -> DetectionList:
    return tracker.attach_ids(dets)
