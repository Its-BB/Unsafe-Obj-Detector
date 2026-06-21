# AI Security Detection

# NOTE: THIS WAS MADE AND TESTED ON A SYSTEM WITH RTX5070; 32GIGS OF RAM; AND A RYZEN 5 7600X CPU

Real-time camera system that spots knives and pistols, draws boxes on the feed, and raises alerts when something looks dangerous.

![Model predictions on validation images](docs/images/val_predictions.jpg)

## Install (one command)

### From source (recommended until PyPI publish)

Python 3.10+. Installs torch, ultralytics, opencv, and everything else:

```powershell
git clone https://github.com/Its-BB/Unsafe-Obj-Detector.git
cd Droneai
.\scripts\install.ps1
```

Or manually:

```powershell
pip install .
droneai init
droneai test --offline
```

### Windows standalone (no Python required)

Download `droneai-security-windows.zip` from [GitHub Releases](https://github.com/Its-BB/Unsafe-Obj-Detector/releases), extract, then:

```powershell
.\droneai.exe init
.\droneai.exe test --offline
.\droneai.exe
```

Press **Q** in the video window to quit.

> **First run needs internet once** to download YOLO weights (~6 MB) unless you bundled `yolov8n.pt` or `weapon_detection_custom.pt` in the release folder.

## Quick start

```powershell
droneai init               # creates config.yaml (webcam mode by default)
droneai test --offline     # verify models load without a camera
droneai                    # live detection with your webcam
```

For best knife/pistol accuracy, place trained weights at `weapon_detection_custom.pt` next to `droneai.exe` or in your project data folder. Without them, the app uses generic YOLO weights (still runs; less accurate on pistols).

**Project data folder** (config, logs, detections):
- Installed via pip: `%APPDATA%\droneai-security` on Windows, `~/.local/share/droneai-security` on Linux
- Standalone `.exe`: same folder as `droneai.exe`
- Development: repo root (after `droneai init` or copying `config.example.yaml`)

## Demo for reviewers

| What | URL / command |
|------|----------------|
| **Demo (no Python)** | [GitHub Releases](https://github.com/Its-BB/Unsafe-Obj-Detector/releases) → download `droneai-security-windows.zip` → run `droneai.exe` |
| **Source install** | `git clone https://github.com/Its-BB/Unsafe-Obj-Detector.git` → `.\scripts\install.ps1` |
| **PyPI** | `pip install droneai-security` — available after publishing to PyPI |
| **5-minute try** | `droneai init` → `droneai test --offline` → `droneai` with a webcam |

No separate torch/opencv install steps — dependencies are in `pyproject.toml` and bundled in the Windows zip.

## CLI commands

| Command | Description |
|---------|-------------|
| `droneai` | Start live weapon detection |
| `droneai init` | Create `config.yaml` from defaults |
| `droneai test --offline` | Smoke test (no camera/stream) |
| `droneai test` | Full test including ESP32 stream if configured |
| `droneai fetch-video` | ESP32-CAM MJPEG viewer |

## Features

- Detects **knife** and **pistol** with a custom YOLO model (falls back to generic YOLO if weights are missing)
- Works with a **USB webcam** or an **ESP32-CAM** MJPEG stream
- On-screen boxes, FPS, and confidence scores
- Sound and visual alerts for high-confidence threats
- Saves incident frames under `detections/` when logging is on
- Kaggle guns/knives dataset pipeline to train your own weights
- Extended detection pipeline: tracking, smoothing, zones, multi-frame threat voting, stats, evidence saves

## Screenshots

**Train / val / test**

![Train](docs/images/train_labels.jpg)
![Val](docs/images/val_labels.jpg)
![Test](docs/images/test_labels.jpg)

**Validation predictions**

![Val predictions](docs/images/val_predictions.jpg)
![Confusion matrix](docs/images/confusion_matrix.png)

**Training curve** (10-epoch)

![Training metrics](docs/images/training_metrics.png)

## Run with your webcam

Edit `config.yaml` (created by `droneai init`):

```yaml
detection:
  use_remote_stream: false
  camera_index: 0
```

Then:

```powershell
droneai
```

Use `camera_index: 1` if you have more than one camera.

## Run with ESP32-CAM

Firmware source is in `ESP32-CAM-DroneAI/` (see that folder's README). After flash, use the IP from serial monitor.

1. Copy `ESP32-CAM-DroneAI/wifi_config.h.example` to `main/wifi_config.h` and set your WiFi.
2. Build and flash: `idf.py build flash monitor`.
3. Set your board IP in `config.yaml`:

```yaml
detection:
  use_remote_stream: true
  remote_stream_url: "http://YOUR_ESP32_IP:80/stream"
```

4. Test the stream only:

```powershell
droneai fetch-video
```

5. Run the full app:

```powershell
droneai
```

## Train the weapon model

Needs Python 3.10+, a GPU recommended, and Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json`.

```powershell
pip install ".[train]"
cd weapon_training
py download_kaggle_dataset.py
py prepare_dataset.py
py train_max_accuracy.py
```

Or one shot:

```powershell
cd weapon_training
.\run_full_training.ps1
```

Copy `weapon_detection_custom.pt` into your project data folder (or next to `droneai.exe` in the release zip). Validate:

```powershell
cd weapon_training
py test_trained_model.py
```

Default training preset is **15 epochs**, YOLOv8n, batch 16.

## Build standalone Windows release

For maintainers creating the reviewer zip:

```powershell
.\scripts\build_exe.ps1
```

Output: `dist\droneai-security-windows.zip` (~1–2 GB, includes torch).

If you have trained weights locally, the build script copies `weapon_detection_custom.pt` into the zip automatically.

Optional: tag a release so GitHub Actions builds and uploads the zip:

```powershell
git tag v1.0.0
git push origin v1.0.0
```

## How it works

Video frames go through **Ultralytics YOLO**. A model trained on the [guns/knives Kaggle dataset](https://www.kaggle.com/datasets/iqmansingh/guns-knives-object-detection) handles weapon classes; general YOLO weights are only a fallback. `paths.py` picks deployed weights (`weapon_detection_custom.pt`) or the best checkpoint under `weapon_training/models/`. Per-class thresholds in `config.yaml` cut down false alarms before the alert system fires.

## Project layout

| Path | Purpose |
|------|---------|
| `droneai/cli.py` | CLI entry point (`droneai` command) |
| `app.py` | Main live detection app (webcam or ESP32) |
| `detection_system.py` | Alternate entry point with the same idea |
| `weapon_detector.py` | Weapon detection logic |
| `detection/` | Pipeline modules (`pipeline.py`, `track_boxes.py`, `threat_level.py`, etc.) |
| `weapon_training/` | Download, prepare, train, test |
| `config.example.yaml` | Default settings template (`droneai init` copies to `config.yaml`) |
| `pyproject.toml` | PyPI package metadata and dependencies |
| `droneai.spec` | PyInstaller spec for Windows standalone build |
| `LICENSE` | MIT license |

## System requirements

- **Python install**: Python 3.10+, webcam optional for offline test
- **Standalone exe**: Windows 10/11 x64, webcam optional
- **GPU**: Optional; CPU inference works (slower). CUDA torch is picked up automatically if installed.
- **Network**: First run downloads YOLO weights unless bundled in the release folder

## Credits

- Dataset: [iqmansingh/guns-knives-object-detection](https://www.kaggle.com/datasets/iqmansingh/guns-knives-object-detection) on Kaggle
- Detection: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
