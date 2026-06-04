import shutil
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else ROOT / 'config.yaml'
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    return (ROOT / relative).resolve()


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
