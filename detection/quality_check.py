"""Check if a frame is good enough to run detection."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

from detection.frame_tools import frame_mean_color, is_bright_frame, is_dark_frame, is_valid_frame


def is_blurry(frame: np.ndarray, limit: float = 80.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < limit


def is_noisy(frame: np.ndarray, limit: float = 45.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(np.std(gray)) > limit


def is_flat(frame: np.ndarray, limit: float = 8.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(np.std(gray)) < limit


def get_blur_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def get_brightness(frame: np.ndarray) -> float:
    b, g, r = frame_mean_color(frame)
    return (b + g + r) / 3.0


def rate_frame(frame: Optional[np.ndarray]) -> Tuple[str, float]:
    if not is_valid_frame(frame):
        return 'bad', 0.0
    score = 100.0
    if is_blurry(frame):
        score -= 35
    if is_dark_frame(frame):
        score -= 25
    if is_bright_frame(frame):
        score -= 15
    if is_flat(frame):
        score -= 20
    if score >= 70:
        return 'good', score
    if score >= 40:
        return 'ok', score
    return 'bad', score


def should_skip_frame(frame: Optional[np.ndarray], min_score: float = 35.0) -> bool:
    label, score = rate_frame(frame)
    return label == 'bad' or score < min_score


def fix_if_dark(frame: np.ndarray) -> np.ndarray:
    if is_dark_frame(frame):
        return cv2.convertScaleAbs(frame, alpha=1.4, beta=25)
    return frame
