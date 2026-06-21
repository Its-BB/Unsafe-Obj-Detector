import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import load_config, weapon_dataset_root, weapon_training_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    root = weapon_dataset_root(config)
    if not root.exists():
        logger.error('Dataset missing. Run download_kaggle_dataset.py first.')
        return 1
    ds = weapon_training_config(config)['dataset']
    data = {
        'path': '.',
        'train': ds['train_split'],
        'val': ds['val_split'],
        'nc': ds['nc'],
        'names': ds['classes'],
    }
    out = root / 'data.yaml'
    with open(out, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.info('Wrote %s', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
