#!/usr/bin/env python3

import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import (
    has_trained_weapon_model,
    load_config,
    resolve_weapon_model_path,
    weapon_dataset_yaml,
)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_model():
    config = load_config()
    if not has_trained_weapon_model(config):
        logger.error('No trained model found')
        return False
    model_path = resolve_weapon_model_path(config)
    logger.info('Loading %s', model_path)
    model = YOLO(model_path)
    logger.info('Classes: %s', model.names)
    data_yaml = weapon_dataset_yaml(config)
    if data_yaml.exists():
        val_results = model.val(data=str(data_yaml))
    else:
        val_results = model.val()
    if hasattr(val_results, 'box'):
        logger.info('mAP50: %.4f', val_results.box.map50)
        logger.info('mAP50-95: %.4f', val_results.box.map)
        if hasattr(val_results.box, 'maps'):
            for i, m in enumerate(val_results.box.maps):
                logger.info('%s mAP50: %.4f', model.names[i], m)
    test_dir = weapon_dataset_yaml(config).parent / 'test' / 'images'
    if test_dir.exists():
        images = list(test_dir.glob('*.jpg'))[:10]
        detected = 0
        for img_path in images:
            results = model(str(img_path))
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                detected += 1
                logger.info('%s: %d objects', img_path.name, len(boxes))
            else:
                logger.info('%s: none', img_path.name)
        logger.info('Detection rate: %d/%d', detected, len(images))
    return True


if __name__ == '__main__':
    print('Testing Custom Weapon Detection Model')
    print('=' * 40)
    ok = test_model()
    print('Done.' if ok else 'Failed.')
    sys.exit(0 if ok else 1)
