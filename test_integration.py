#!/usr/bin/env python3
"""Integration tests: python test_integration.py [--offline]"""

from __future__ import annotations

import argparse
import sys
import time
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_stream_url() -> str:
    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    det = cfg.get("detection") or {}
    return str(det.get("remote_stream_url", "http://YOUR_ESP32_IP:80/stream"))


def test_configuration_offline() -> bool:
    print("\n" + "=" * 60)
    print("TEST: Configuration (offline)")
    print("=" * 60)
    try:
        from paths import (
            has_trained_weapon_model,
            load_config,
            resolve_weapon_model_path,
            weapon_dataset_root,
            weapon_dataset_yaml,
        )

        cfg = load_config()
        det = cfg.get("detection") or {}
        out_dir = Path(__file__).parent / (cfg.get("logging") or {}).get("output_dir", "detections")
        print("[OK] config.yaml loaded")
        print(f"  - use_remote_stream: {det.get('use_remote_stream', False)}")
        print(f"  - remote_stream_url: {det.get('remote_stream_url', 'N/A')}")
        print(f"  - weapon model: {resolve_weapon_model_path(cfg)}")
        print(f"  - trained weights: {has_trained_weapon_model(cfg)}")
        print(f"  - dataset yaml: {weapon_dataset_yaml(cfg)} ({weapon_dataset_yaml(cfg).exists()})")
        print(f"  - detections dir exists: {out_dir.is_dir()} ({out_dir})")
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_configuration_with_system() -> bool:
    print("\n" + "=" * 60)
    print("TEST: AIDetectionSystem init (needs camera stream)")
    print("=" * 60)
    try:
        import os

        os.environ["DRONEAI_SKIP_CAMERA"] = "1"
        from app import AIDetectionSystem

        from paths import resolve_weapon_model_path

        system = AIDetectionSystem()
        config = system.config["detection"]
        print("[OK] AIDetectionSystem loaded (camera skipped)")
        print(f"  - use_remote_stream: {config.get('use_remote_stream', False)}")
        print(f"  - weapon model: {resolve_weapon_model_path(system.config)}")
        print(f"  - custom weapon detector: {system.weapon_detector.using_custom_model}")
        system.cleanup()
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_remote_stream_connection(url: str) -> bool:
    print("\n" + "=" * 60)
    print("TEST: Remote Stream Connection")
    print("=" * 60)
    try:
        from app import RemoteVideoStream

        stream = RemoteVideoStream(url)
        stream.start()
        time.sleep(3)
        ok = stream.isOpened()
        stream.stop()
        if ok:
            print(f"[OK] Stream connected: {url}")
        else:
            print(f"[FAIL] Stream failed: {url}")
        return ok
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_frame_reading(url: str) -> bool:
    print("\n" + "=" * 60)
    print("TEST: Frame Reading")
    print("=" * 60)
    try:
        from app import RemoteVideoStream

        stream = RemoteVideoStream(url)
        stream.start()
        time.sleep(3)
        frame_count = 0
        for _ in range(5):
            ret, frame = stream.read()
            if ret and frame is not None:
                frame_count += 1
            time.sleep(0.5)
        stream.stop()
        print(f"[OK] Read {frame_count}/5 frames")
        return frame_count > 0
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_detection_system_init(url: str) -> bool:
    print("\n" + "=" * 60)
    print("TEST: Detection System Init")
    print("=" * 60)
    try:
        from app import AIDetectionSystem

        system = AIDetectionSystem()
        if not system.camera or not system.camera.isOpened():
            print("[FAIL] Camera/stream not initialized")
            system.cleanup()
            return False
        ret, frame = system.camera.read()
        system.cleanup()
        if ret and frame is not None:
            print(f"[OK] Frame shape: {frame.shape}")
            return True
        print("[FAIL] No frame from camera")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only run config/import tests (no ESP32-CAM required)",
    )
    args = parser.parse_args()
    url = load_stream_url()

    print("\n" + "=" * 60)
    print("ESP32-CAM / droneai Integration Test Suite")
    print("=" * 60)
    print(f"Stream URL from config.yaml: {url}")

    if args.offline:
        results = {
            "Configuration (yaml)": test_configuration_offline(),
            "AIDetectionSystem (no camera)": test_configuration_with_system(),
        }
    else:
        results = {
            "Configuration (yaml)": test_configuration_offline(),
            "Remote Stream Connection": test_remote_stream_connection(url),
            "Frame Reading": test_frame_reading(url),
            "Detection System Init": test_detection_system_init(url),
        }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"{'[PASS]' if ok else '[FAIL]'}: {name}")
    print(f"\nTotal: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
