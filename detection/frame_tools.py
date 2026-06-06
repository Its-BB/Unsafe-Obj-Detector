"""Frame prep before detection."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

Frame = np.ndarray


def get_frame_size(frame: Frame) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    return w, h


def resize_frame(frame: Frame, width: int, height: int) -> Frame:
    if width <= 0 or height <= 0:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_keep_ratio(frame: Frame, max_side: int) -> Frame:
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = max_side / max(h, w)
    return resize_frame(frame, int(w * scale), int(h * scale))


def to_gray(frame: Frame) -> Frame:
    if len(frame.shape) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def to_bgr(frame: Frame) -> Frame:
    if len(frame.shape) == 3:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def fix_brightness(frame: Frame, target: float = 128.0) -> Frame:
    gray = to_gray(frame)
    mean = float(np.mean(gray))
    if mean <= 1.0:
        return frame
    gain = target / mean
    out = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out


def auto_contrast(frame: Frame, clip_percent: float = 2.0) -> Frame:
    if len(frame.shape) == 2:
        channels = [frame]
    else:
        channels = cv2.split(frame)
    out_channels = []
    for ch in channels:
        hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
        total = hist.sum()
        if total <= 0:
            out_channels.append(ch)
            continue
        low = 0
        high = 255
        cum = 0.0
        cut = total * (clip_percent / 100.0)
        for i in range(256):
            cum += hist[i]
            if cum >= cut:
                low = i
                break
        cum = 0.0
        for i in range(255, -1, -1):
            cum += hist[i]
            if cum >= cut:
                high = i
                break
        if high <= low:
            out_channels.append(ch)
            continue
        lut = np.zeros(256, dtype=np.uint8)
        scale = 255.0 / (high - low)
        for i in range(256):
            v = int((i - low) * scale)
            lut[i] = max(0, min(255, v))
        out_channels.append(cv2.LUT(ch, lut))
    if len(out_channels) == 1:
        return out_channels[0]
    return cv2.merge(out_channels)


def reduce_noise(frame: Frame, strength: int = 5) -> Frame:
    k = max(3, strength | 1)
    return cv2.GaussianBlur(frame, (k, k), 0)


def sharpen_frame(frame: Frame, amount: float = 1.0) -> Frame:
    blur = reduce_noise(frame, 3)
    sharp = cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)
    return sharp


def flip_horizontal(frame: Frame) -> Frame:
    return cv2.flip(frame, 1)


def flip_vertical(frame: Frame) -> Frame:
    return cv2.flip(frame, 0)


def rotate_90(frame: Frame) -> Frame:
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


def rotate_180(frame: Frame) -> Frame:
    return cv2.rotate(frame, cv2.ROTATE_180)


def crop_frame(frame: Frame, x1: int, y1: int, x2: int, y2: int) -> Frame:
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return frame[y1:y2, x1:x2].copy()


def add_border(frame: Frame, pad: int, color: Tuple[int, int, int] = (0, 0, 0)) -> Frame:
    return cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=color)


def blend_frames(a: Frame, b: Frame, alpha: float = 0.5) -> Frame:
    if a.shape != b.shape:
        b = resize_frame(b, a.shape[1], a.shape[0])
    return cv2.addWeighted(a, alpha, b, 1.0 - alpha, 0)


def frame_diff(a: Frame, b: Frame) -> Frame:
    if a.shape != b.shape:
        b = resize_frame(b, a.shape[1], a.shape[0])
    return cv2.absdiff(a, b)


def motion_mask(prev: Optional[Frame], curr: Frame, thresh: int = 25) -> Frame:
    if prev is None:
        return np.zeros(curr.shape[:2], dtype=np.uint8)
    diff = frame_diff(prev, curr)
    gray = to_gray(diff)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return mask


def has_motion(prev: Optional[Frame], curr: Frame, min_pixels: int = 500) -> bool:
    mask = motion_mask(prev, curr)
    return int(np.count_nonzero(mask)) >= min_pixels


def draw_motion_zones(frame: Frame, mask: Frame, color: Tuple[int, int, int] = (0, 255, 255)) -> Frame:
    out = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1)
    return out


def normalize_for_model(frame: Frame, size: int) -> Frame:
    return resize_keep_ratio(frame, size)


def copy_frame(frame: Frame) -> Frame:
    return frame.copy()


def is_valid_frame(frame: Optional[Frame]) -> bool:
    if frame is None:
        return False
    if frame.size == 0:
        return False
    h, w = frame.shape[:2]
    return h > 0 and w > 0


def frame_mean_color(frame: Frame) -> Tuple[float, float, float]:
    if len(frame.shape) == 2:
        m = float(np.mean(frame))
        return m, m, m
    b, g, r = cv2.split(frame)
    return float(np.mean(b)), float(np.mean(g)), float(np.mean(r))


def is_dark_frame(frame: Frame, limit: float = 40.0) -> bool:
    b, g, r = frame_mean_color(frame)
    return (b + g + r) / 3.0 < limit


def is_bright_frame(frame: Frame, limit: float = 220.0) -> bool:
    b, g, r = frame_mean_color(frame)
    return (b + g + r) / 3.0 > limit


def prep_frame(frame: Frame, config: dict) -> Frame:
    """Run configured prep steps in order."""
    pipe = config.get('detection_pipeline', {})
    out = copy_frame(frame)
    if pipe.get('fix_brightness', True) and is_dark_frame(out):
        out = fix_brightness(out)
    if pipe.get('auto_contrast', False):
        out = auto_contrast(out)
    if pipe.get('reduce_noise', False):
        out = reduce_noise(out, pipe.get('noise_strength', 3))
    if pipe.get('sharpen', False):
        out = sharpen_frame(out, pipe.get('sharpen_amount', 0.5))
    max_side = pipe.get('max_frame_side', 0)
    if max_side and max_side > 0:
        out = resize_keep_ratio(out, max_side)
    return out
