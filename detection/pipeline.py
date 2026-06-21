"""Main detection pipeline — simple entry points."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from detection.consensus import ThreatVote, need_n_frames_to_alert
from detection.count_stats import DetectionStats
from detection.draw_overlay import draw_all
from detection.event_log import EventLog
from detection.export_report import save_json_report, save_text_report
from detection.filter_boxes import filter_pipeline, merge_lists
from detection.frame_tools import get_frame_size, is_valid_frame, prep_frame
from detection.history_buffer import FrameHistory
from detection.merge_rules import combine_weapon_sources, resolve_conflicts
from detection.post_process import full_post_process
from detection.person_near import boost_weapon_near_person, mark_person_nearby
from detection.quality_check import fix_if_dark, should_skip_frame
from detection.run_yolo import merge_weapon_results, run_object_pass
from detection.save_clips import EvidenceWriter, save_incident_bundle
from detection.smooth_scores import ScoreSmoother, smooth_confidence
from detection.threat_level import should_alert, tag_threat_levels
from detection.track_boxes import ObjectTracker, track_objects
from detection.types import DetectionList
from detection.watch_zones import ZoneManager
from weapon_detector import WeaponDetector

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Runs the full detect → filter → track → score → draw path."""

    def __init__(
        self,
        config: dict,
        object_model: YOLO,
        weapon_detector: WeaponDetector,
        human_model: Optional[YOLO] = None,
    ):
        self.config = config
        self.object_model = object_model
        self.human_model = human_model or object_model
        self.weapon_detector = weapon_detector

        det_cfg = config.get('detection', {})
        pipe_cfg = config.get('detection_pipeline', {})
        self.conf_threshold = float(det_cfg.get('confidence_threshold', 0.5))
        self.nms_iou = float(det_cfg.get('nms_threshold', 0.4))
        self.imgsz = int(det_cfg.get('input_size', 640))
        self.dangerous_list = list(config.get('alerts', {}).get('dangerous_objects', []))

        self.tracker = ObjectTracker(
            max_age=float(pipe_cfg.get('track_max_age', 2.0)),
            min_hits=int(pipe_cfg.get('track_min_hits', 2)),
            iou_match=float(pipe_cfg.get('track_iou', 0.3)),
        )
        self.smoother = ScoreSmoother(
            window=int(pipe_cfg.get('smooth_window', 5)),
            alpha=float(pipe_cfg.get('smooth_alpha', 0.4)),
        )
        self.voter = ThreatVote(
            need_frames=int(pipe_cfg.get('confirm_frames', 3)),
            window=int(pipe_cfg.get('vote_window', 8)),
        )
        self.zones = ZoneManager(config)
        self.history = FrameHistory(max_frames=int(pipe_cfg.get('history_frames', 30)))
        self.stats = DetectionStats()
        self.events = EventLog(max_events=int(pipe_cfg.get('event_log_size', 500)))
        self.evidence = EvidenceWriter(config)
        self._prev_frame: Optional[np.ndarray] = None
        self._skipped_frames = 0

    def reset(self) -> None:
        self.tracker.reset()
        self.smoother.reset()
        self.voter.reset()
        self.history.reset()
        self.stats.reset()
        self.events.reset()
        self._prev_frame = None
        self._skipped_frames = 0

    def run_yolo_pass(self, frame: np.ndarray) -> DetectionList:
        object_dets = run_object_pass(
            self.object_model,
            frame,
            self.conf_threshold,
            self.dangerous_list,
            self.imgsz,
        )
        weapon_raw = self.weapon_detector.detect_weapons(frame)
        weapon_dets = merge_weapon_results(weapon_raw)
        return merge_lists(object_dets, weapon_dets)

    def clean_detections(self, dets: DetectionList, frame_w: int, frame_h: int) -> DetectionList:
        pipe_cfg = self.config.get('detection_pipeline', {})
        min_pixels = int(pipe_cfg.get('min_box_pixels', 10))
        min_weapon_score = float(pipe_cfg.get('min_weapon_score', 0.2))
        return filter_pipeline(
            dets,
            frame_w,
            frame_h,
            self.conf_threshold,
            self.nms_iou,
            min_pixels,
            min_weapon_score,
        )

    def run_track_pass(self, dets: DetectionList) -> DetectionList:
        return track_objects(dets, self.tracker)

    def run_smooth_pass(self, dets: DetectionList) -> DetectionList:
        return smooth_confidence(dets, self.smoother)

    def run_zone_pass(self, dets: DetectionList) -> DetectionList:
        return self.zones.apply(dets)

    def run_person_pass(self, dets: DetectionList) -> DetectionList:
        pipe_cfg = self.config.get('detection_pipeline', {})
        max_dist = float(pipe_cfg.get('person_near_dist', 200.0))
        out = mark_person_nearby(dets, max_dist)
        return boost_weapon_near_person(out)

    def run_threat_pass(self, dets: DetectionList) -> DetectionList:
        out = tag_threat_levels(dets)
        if self.config.get('detection_pipeline', {}).get('use_vote', True):
            out = need_n_frames_to_alert(self.voter, out)
        return out

    def process_frame(self, frame: np.ndarray) -> DetectionList:
        if not is_valid_frame(frame):
            return []

        pipe_cfg = self.config.get('detection_pipeline', {})
        if pipe_cfg.get('skip_bad_frames', False) and should_skip_frame(frame):
            self._skipped_frames += 1
            return []

        ready = fix_if_dark(prep_frame(frame, self.config))
        frame_w, frame_h = get_frame_size(ready)
        self.zones.set_frame_size(frame_w, frame_h)

        raw = self.run_yolo_pass(ready)
        raw = self.clean_detections(raw, frame_w, frame_h)
        raw = self.run_track_pass(raw)
        raw = self.run_smooth_pass(raw)
        raw = self.run_zone_pass(raw)
        raw = self.run_person_pass(raw)
        raw = full_post_process(
            raw,
            self.dangerous_list,
            self.conf_threshold,
            self.nms_iou,
            int(pipe_cfg.get('max_detections', 50)),
            float(pipe_cfg.get('min_weapon_score', 0.2)),
        )
        raw = self.run_threat_pass(raw)

        self.stats.add_detections(raw)
        self.events.add_frame(raw, {'skipped_total': self._skipped_frames})
        self.history.push(ready, raw, {'threat_level': raw and raw[0].get('threat_level')})
        self._prev_frame = ready.copy()
        return raw

    def draw_frame(self, frame: np.ndarray, dets: DetectionList) -> np.ndarray:
        summary = self.stats.get_summary(dets)
        zone_boxes = self.zones.get_zone_boxes()
        return draw_all(frame, dets, summary, self.config, zone_boxes)

    def get_stats(self, dets: DetectionList) -> Dict[str, Any]:
        return self.stats.get_summary(dets)

    def tick_fps(self) -> float:
        return self.stats.tick_frame()

    def get_dangerous(self, dets: DetectionList) -> DetectionList:
        return [d for d in dets if d.get('is_dangerous')]

    def should_fire_alert(self, dets: DetectionList) -> bool:
        pipe_cfg = self.config.get('detection_pipeline', {})
        min_level = str(pipe_cfg.get('alert_min_level', 'medium'))
        return should_alert(dets, min_level)

    def save_threat(self, frame: np.ndarray, dets: DetectionList, report: Optional[dict] = None) -> Optional[str]:
        return self.evidence.save_if_needed(frame, dets, report)

    def save_incident(self, frame: np.ndarray, dets: DetectionList, report: dict) -> Dict[str, str]:
        return save_incident_bundle(frame, dets, self.config, report)

    def export_stats_csv(self, path: str) -> str:
        from pathlib import Path
        return self.stats.export_csv(Path(path))

    def get_session_report(self) -> dict:
        return self.stats.to_dict()

    def export_session(self, output_dir: str) -> Dict[str, str]:
        from pathlib import Path
        folder = Path(output_dir) / 'sessions'
        return {
            'txt': save_text_report(self.stats, self.events, folder),
            'json': save_json_report(self.stats, self.events, folder),
        }
