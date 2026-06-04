import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import load_config, weapon_dataset_root, weapon_training_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def setup_kaggle_credentials():
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    Path.home().joinpath('.kaggle').mkdir(exist_ok=True)
    if kaggle_json.exists():
        try:
            with open(kaggle_json, encoding='utf-8') as f:
                data = json.load(f)
            if data.get('username') and data.get('key'):
                if data['username'] != 'your_kaggle_username':
                    return True
        except json.JSONDecodeError:
            pass
    logger.error('Kaggle token missing. Create %s (not in this repo)', kaggle_json)
    return False


def main():
    config = load_config()
    if not setup_kaggle_credentials():
        return 1
    wt = weapon_training_config(config)
    dataset_dir = SCRIPT_DIR / 'dataset'
    expected = weapon_dataset_root(config)
    if not (expected.exists() and (expected / wt['dataset']['train_split']).exists()):
        cmd = [
            'kaggle', 'datasets', 'download',
            wt['kaggle']['dataset_slug'],
            '-p', str(dataset_dir), '--unzip',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error('%s', result.stderr or result.stdout)
            return 1
    if not weapon_dataset_root(config).exists():
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
