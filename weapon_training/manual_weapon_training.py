#!/usr/bin/env python3

import json
import logging
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import deploy_weapon_model, load_config, weapon_training_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


class ManualWeaponTrainer:
    def __init__(self, base_dir='./'):
        self.base_dir = Path(base_dir)
        self.config = load_config()
        self.wt = weapon_training_config(self.config)
        self.class_names = self.wt['dataset']['classes']
        self.dataset_dir = self.base_dir / 'manual_dataset'
        self.models_dir = self.base_dir / 'models'
        self.results_dir = self.base_dir / 'results'
        for d in (self.dataset_dir, self.models_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    def create_sample_dataset(self):
        logger.info('Creating sample dataset structure...')
        yolo_dataset_dir = self.dataset_dir / 'yolo_format'
        yolo_dataset_dir.mkdir(parents=True, exist_ok=True)
        for split in ('train', 'val'):
            (yolo_dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        dataset_config = {
            'path': str(yolo_dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': self.class_names,
            'nc': self.wt['dataset']['nc'],
        }
        config_path = yolo_dataset_dir / 'dataset.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        logger.info('Dataset config: %s', config_path)
        logger.info('Add images to train/images and val/images')
        logger.info('Add YOLO labels to train/labels and val/labels')
        return config_path

    def download_public_dataset(self):
        logger.info('Looking for alternative public datasets...')
        return self.create_sample_dataset()

    def _has_images(self):
        yolo_dir = self.dataset_dir / 'yolo_format'
        for split in ('train', 'val'):
            img_dir = yolo_dir / split / 'images'
            if not img_dir.exists():
                return False
            if not any(img_dir.glob('*.jpg')) and not any(img_dir.glob('*.png')):
                return False
        return True

    def train_minimal_model(self, dataset_config_path, epochs=10):
        logger.info('Training minimal weapon detection model...')
        try:
            if self._has_images():
                model = YOLO(self.config['weapon']['fallback_primary'])
                model.train(
                    data=str(dataset_config_path),
                    epochs=epochs,
                    imgsz=640,
                    batch=8,
                    project=str(self.results_dir),
                    name='weapon_detection_manual',
                    exist_ok=True,
                    device='auto',
                    patience=15,
                    val=True,
                )
                best = self.results_dir / 'weapon_detection_manual' / 'weights' / 'best.pt'
                if best.exists():
                    deploy_weapon_model(best, self.config)
                    logger.info('Trained and deployed: %s', best)
                    return best
            model = YOLO(self.config['weapon']['fallback_primary'])
            logger.info('Model classes available:')
            for i, name in model.names.items():
                logger.info('  %s: %s', i, name)
            custom_model_path = self.models_dir / 'weapon_detection_demo.pt'
            model.save(str(custom_model_path))
            deploy_weapon_model(custom_model_path, self.config)
            logger.info('Demo model saved: %s', custom_model_path)
            return custom_model_path
        except Exception as e:
            logger.error('Error during training: %s', e)
            return None

    def create_weapon_focused_model(self):
        logger.info('Creating weapon-focused detection configuration...')
        model = YOLO(self.config['weapon']['fallback_primary'])
        weapon_related_classes = {
            43: 'knife',
            44: 'spoon',
            45: 'bowl',
            46: 'banana',
            64: 'potted plant',
            67: 'cell phone',
            73: 'laptop',
        }
        model_path = self.models_dir / 'weapon_focused_yolo.pt'
        config = {
            'model_path': str(model_path),
            'focus_classes': weapon_related_classes,
            'detection_threshold': 0.3,
            'weapon_mapping': {
                'knife': ['knife', 'spoon'],
                'pistol': ['cell phone', 'remote'],
                'suspicious': ['laptop', 'potted plant', 'banana'],
            },
        }
        config_path = self.models_dir / 'weapon_focus_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        model.save(str(model_path))
        deploy_weapon_model(model_path, self.config)
        logger.info('Weapon-focused model: %s', model_path)
        logger.info('Configuration: %s', config_path)
        return model_path, config_path

    def run_quick_setup(self):
        logger.info('Running quick weapon detection setup...')
        dataset_config = self.create_sample_dataset()
        model_path, config_path = self.create_weapon_focused_model()
        return {
            'model_path': model_path,
            'config_path': config_path,
            'dataset_config': dataset_config,
        }


def main():
    trainer = ManualWeaponTrainer(SCRIPT_DIR)
    print('Manual Weapon Detection Model Setup')
    print('=====================================')
    print()
    print('Weapon detection setup without Kaggle API.')
    print('1. Quick setup - weapon-focused YOLO model')
    print('2. Sample dataset structure only')
    print('3. Train on manual_dataset (needs images + labels)')
    print('4. Exit')
    print()
    try:
        choice = input('Enter your choice (1-4): ').strip()
    except KeyboardInterrupt:
        print('\nExiting...')
        return
    if choice == '1':
        print('\nRunning quick setup...')
        result = trainer.run_quick_setup()
        print(f"\nQuick setup completed.")
        print(f"Model: {result['model_path']}")
        print(f"Config: {result['config_path']}")
        print(f"Dataset: {result['dataset_config']}")
    elif choice == '2':
        print('\nCreating sample dataset structure...')
        dataset_config = trainer.create_sample_dataset()
        print(f"\nDataset structure created.")
        print(f"Location: {dataset_config}")
    elif choice == '3':
        yaml_path = trainer.dataset_dir / 'yolo_format' / 'dataset.yaml'
        if not yaml_path.exists():
            trainer.create_sample_dataset()
        out = trainer.train_minimal_model(yaml_path, epochs=50)
        if out:
            print(f"\nTraining done: {out}")
        else:
            print('\nTraining failed.')
    elif choice == '4':
        print('\nGoodbye!')
        return
    else:
        print('\nInvalid choice.')
        return
    print('\nNext steps:')
    print('1. Run app.py or detection_system.py')
    print('2. For Kaggle dataset: run_full_training.ps1')
    print('3. Test with real weapon images')


if __name__ == '__main__':
    main()
