# PyInstaller spec — standalone Windows folder.
# Build: python -m PyInstaller droneai.spec --noconfirm

from pathlib import Path

block_cipher = None
root = Path(SPECPATH)

datas = [
    (str(root / "config.example.yaml"), "."),
    (str(root / "droneai" / "data" / "config.example.yaml"), "droneai" + "/data"),
]

hiddenimports = [
    "ultralytics",
    "ultralytics.nn",
    "ultralytics.models",
    "ultralytics.utils",
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "yaml",
    "requests",
    "scipy",
    "tqdm",
    "psutil",
    "pygame",
    "PIL",
    "matplotlib",
    "detection",
    "detection.pipeline",
    "detection.track_boxes",
    "detection.threat_level",
    "detection.draw_overlay",
    "detection.filter_boxes",
    "detection.post_process",
    "detection.smooth_scores",
    "detection.consensus",
    "detection.watch_zones",
    "detection.person_near",
    "detection.colors",
    "detection.types",
    "detection.box_math",
    "detection.class_rules",
    "detection.merge_rules",
    "detection.quality_check",
    "detection.run_yolo",
    "detection.frame_tools",
    "detection.history_buffer",
    "detection.event_log",
    "detection.count_stats",
    "detection.save_clips",
    "detection.export_report",
    "app",
    "paths",
    "alert_system",
    "weapon_detector",
    "scene_analyzer",
    "fetch_video",
    "test_integration",
]

a = Analysis(
    [str(root / "droneai" / "cli.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="droneai",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="droneai-security",
)
