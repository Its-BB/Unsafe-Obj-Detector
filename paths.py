import os
import shutil
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

_DEV_ROOT = Path(__file__).resolve().parent


def _frozen_root() -> Path | None:
    """Directory next to a PyInstaller one-file/one-folder executable."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return None


def user_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path.home() / ".local" / "share"
    path = base / "droneai-security"
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_root() -> Path:
    """Writable directory for config, weights, logs, and detection output."""
    frozen = _frozen_root()
    if frozen is not None:
        return frozen

    if (_DEV_ROOT / "config.example.yaml").exists() or (_DEV_ROOT / "config.yaml").exists():
        return _DEV_ROOT

    cwd = Path.cwd()
    if (cwd / "config.yaml").exists() or (cwd / "config.example.yaml").exists():
        return cwd

    return user_data_dir()


def bundled_config_example() -> Path:
    with resources.as_file(
        resources.files("droneai.data").joinpath("config.example.yaml")
    ) as bundled:
        return Path(bundled)


def ensure_config(target_dir: Path | None = None) -> Path:
    """Create config.yaml from the bundled example when missing."""
    root = target_dir or project_root()
    config_path = root / "config.yaml"
    if config_path.exists():
        return config_path

    example = root / "config.example.yaml"
    if not example.exists():
        try:
            shutil.copy2(bundled_config_example(), example)
        except (ModuleNotFoundError, FileNotFoundError):
            dev_example = _DEV_ROOT / "config.example.yaml"
            if dev_example.exists():
                shutil.copy2(dev_example, example)

    if example.exists():
        shutil.copy2(example, config_path)
    else:
        raise FileNotFoundError(
            "config.example.yaml not found. Run from the repo or reinstall droneai-security."
        )

    (root / "detections").mkdir(parents=True, exist_ok=True)
    (root / "weapon_training" / "models").mkdir(parents=True, exist_ok=True)
    return config_path


# Backward-compatible alias used across the codebase.
ROOT = project_root()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else project_root() / "config.yaml"
    if not path.exists():
        path = ensure_config(path.parent)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    return (project_root() / relative).resolve()


def weapon_training_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    return cfg['weapon_training']


def weapon_dataset_root(config: dict[str, Any] | None = None) -> Path:
    wt = weapon_training_config(config)
    return resolve_path(f"weapon_training/dataset/{wt['kaggle']['extract_subdir']}")


def weapon_dataset_yaml(config: dict[str, Any] | None = None) -> Path:
    return weapon_dataset_root(config) / 'data.yaml'


def weapon_model_deployed_path(config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    rel = cfg['paths']['weapon_model_deployed']
    return resolve_path(rel)


def weapon_model_best_path(config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    rel = cfg['paths']['weapon_model_best']
    return resolve_path(rel)


def has_trained_weapon_model(config: dict[str, Any] | None = None) -> bool:
    return weapon_model_deployed_path(config).exists() or weapon_model_best_path(config).exists()


def resolve_weapon_model_path(config: dict[str, Any] | None = None) -> str:
    cfg = config or load_config()
    deployed = weapon_model_deployed_path(cfg)
    if deployed.exists():
        return str(deployed)
    best = weapon_model_best_path(cfg)
    if best.exists():
        return str(best)
    weapon = cfg.get('weapon', {})
    return weapon.get('fallback_primary', cfg['detection'].get('weapon_model', 'yolov8n.pt'))


def resolve_weapon_secondary_path(config: dict[str, Any] | None = None) -> str:
    if has_trained_weapon_model(config):
        return resolve_weapon_model_path(config)
    cfg = config or load_config()
    return cfg.get('weapon', {}).get('fallback_secondary', 'yolov8s.pt')


def weapon_results_dir(config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    return resolve_path(cfg['paths']['weapon_results'])


def deploy_weapon_model(source: Path, config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    source = Path(source)
    deployed = weapon_model_deployed_path(cfg)
    best = weapon_model_best_path(cfg)
    best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, deployed)
    shutil.copy2(source, best)
    return deployed
