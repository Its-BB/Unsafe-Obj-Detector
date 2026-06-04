import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import (
    deploy_weapon_model,
    load_config,
    weapon_dataset_yaml,
    weapon_results_dir,
    weapon_training_config,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRAIN_ARG_KEYS = frozenset({
    'data', 'epochs', 'imgsz', 'batch', 'patience', 'device', 'workers', 'cache',
    'optimizer', 'lr0', 'lrf', 'warmup_epochs', 'cos_lr', 'close_mosaic', 'mosaic',
    'mixup', 'copy_paste', 'degrees', 'translate', 'scale', 'shear', 'perspective',
    'flipud', 'fliplr', 'hsv_h', 'hsv_s', 'hsv_v', 'amp', 'save_period', 'val',
    'plots', 'project', 'name', 'exist_ok', 'pretrained', 'verbose', 'resume',
})

PRESET_TRAIN_KEYS = frozenset(TRAIN_ARG_KEYS - {
    'data', 'project', 'name', 'exist_ok', 'pretrained', 'verbose', 'resume', 'val',
})


def resolve_device(requested, allow_cpu: bool):
    if requested == 'auto':
        if torch.cuda.is_available():
            logger.info('GPU: %s', torch.cuda.get_device_name(0))
            return 0
        if allow_cpu:
            logger.warning('No GPU - using CPU (slow)')
            return 'cpu'
        logger.warning('No GPU - using CPU. Use --cpu to silence.')
        return 'cpu'
    if requested in (0, '0', 'cuda', 'gpu'):
        return 0 if torch.cuda.is_available() else ('cpu' if allow_cpu else 0)
    return requested


def build_kwargs(preset, data_yaml, config, resume=None, allow_cpu=False, batch=None):
    raw = {
        'data': str(data_yaml),
        'epochs': preset['epochs'],
        'imgsz': preset['imgsz'],
        'batch': batch if batch is not None else preset['batch'],
        'patience': preset.get('patience', 10),
        'device': resolve_device(str(preset.get('device', 'auto')), allow_cpu),
        'workers': preset.get('workers', 4),
        'cache': preset.get('cache', 'disk'),
        'amp': preset.get('amp', True),
        'mosaic': preset.get('mosaic', 1.0),
        'mixup': preset.get('mixup', 0.0),
        'copy_paste': preset.get('copy_paste', 0.0),
        'close_mosaic': preset.get('close_mosaic', 5),
        'val': True,
        'plots': preset.get('plots', False),
        'project': str(weapon_results_dir(config)),
        'name': preset['name'],
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
    }
    for key in PRESET_TRAIN_KEYS:
        if key in preset and key not in raw:
            raw[key] = preset[key]
    if resume:
        raw['resume'] = str(resume)
    return {k: v for k, v in raw.items() if k in TRAIN_ARG_KEYS}


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return 'out of memory' in msg or 'cuda error' in msg


def train_with_oom_retry(model, kwargs):
    batch = int(kwargs['batch'])
    while batch >= 4:
        attempt = {**kwargs, 'batch': batch}
        try:
            model.train(**attempt)
            return
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch //= 2
            logger.warning('GPU OOM - retrying with batch=%s', batch)
    raise RuntimeError('GPU OOM at batch=4; try --cpu or close other GPU apps')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None, help='Override epoch count (e.g. 20)')
    parser.add_argument('--batch', type=int, default=None, help='Override batch size (e.g. 12)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    config = load_config()
    data_yaml = weapon_dataset_yaml(config)
    if not data_yaml.exists():
        logger.error('Missing %s - run prepare_dataset.py', data_yaml)
        return 1

    wt = weapon_training_config(config)
    preset = dict(wt['training_fast'])
    if args.epochs is not None:
        preset['epochs'] = args.epochs
    if args.batch is not None:
        preset['batch'] = args.batch

    resume = None
    if args.resume:
        last = weapon_results_dir(config) / preset['name'] / 'weights' / 'last.pt'
        if last.exists():
            resume = last

    kwargs = build_kwargs(preset, data_yaml, config, resume, allow_cpu=args.cpu)
    logger.info(
        'Fast train %s | device=%s | epochs=%s imgsz=%s batch=%s',
        preset['model'], kwargs['device'], kwargs['epochs'], kwargs['imgsz'], kwargs['batch'],
    )

    t0 = time.perf_counter()
    model = YOLO(str(resume) if resume else preset['model'])
    train_with_oom_retry(model, kwargs)
    logger.info('Training done in %.1f min', (time.perf_counter() - t0) / 60)

    best = weapon_results_dir(config) / preset['name'] / 'weights' / 'best.pt'
    if not best.exists():
        return 1
    deploy_weapon_model(best, config)
    val = YOLO(str(best))
    r = val.val(data=str(data_yaml))
    if hasattr(r, 'box'):
        logger.info('mAP50: %.4f  mAP50-95: %.4f', r.box.map50, r.box.map)
    return 0


if __name__ == '__main__':
    sys.exit(main())
