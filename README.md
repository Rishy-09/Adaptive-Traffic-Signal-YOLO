# Adaptive-Traffic-Signal-YOLO

> **Short:** An end-to-end computer-vision pipeline that trains a YOLO model to detect vehicles, applies lightweight tracking+counting in video, computes queue length/density, and adapts traffic-signal green time using a simple control rule. This README explains repository layout, how to set up and run the notebooks, the core logic (detection → tracking → queue → adaptive timing), and common troubleshooting tips.

---

## Table of contents

* [Project overview](#project-overview)
* [Features & goals](#features--goals)
* [Repository structure](#repository-structure)
* [Quick setup (3-step)](#quick-setup-3-step)
* [Detailed setup](#detailed-setup)
* [How to run (notebook order)](#how-to-run-notebook-order)
* [Core algorithm & design decisions](#core-algorithm--design-decisions)

  * [YOLOv8 model training (notebook 01)](#yolov8-model-training-notebook-01)
  * [Detection → Tracking → Counting (notebook 02)](#detection--tracking--counting-notebook-02)
  * [Adaptive green-time formula](#adaptive-green-time-formula)
* [Evaluation & visualization](#evaluation--visualization)
* [Where to store large files (models/videos)](#where-to-store-large-files-modelsvideos)
* [Common errors & fixes](#common-errors--fixes)
* [Collaboration notes (for group)](#collaboration-notes-for-group)
* [Contributing and license](#contributing-and-license)

---

## Project overview

This project demonstrates a practical pipeline for traffic monitoring and adaptive signal control using an object detector (YOLOv8) and simple rule-based control logic. The goal is to show how detection outputs can feed a small control law that changes the green-light duration based on measured queue length or vehicle density.

It is intentionally kept reproducible and modular so each team member can work on: dataset preparation / model training / inference & logic / visualization.

---

## Features & goals

* Train a YOLOv8 detector on a traffic dataset (cars, buses, trucks, bikes).
* Run inference on video frames to get bounding boxes and class labels.
* Track objects over frames (ID assignment) and compute counts & queue lengths.
* Compute adaptive green-light duration using a linear control rule.
* Save annotated output video and simple plots (queue length vs time).

---

## Repository structure

```
Adaptive-Traffic-Signal-YOLO/
│
├── 01_Model_Training.ipynb   # Traffic_YOLO.ipynb (training + dataset steps)
|── 02_Traffic_Logic.ipynb    # ITCS.ipynb (inference, tracking, queue logic)
│
├── data/                     # optional: ignored by git if large
│   ├── test_video.mp4
│   └── best.pt                # trained model (optional to store here)
│
├── runs/                   # annotated videos, plots (tracked in git via .gitkeep)
│   └── annotated_output.mp4
│   └── counts_log.csv
│   └── detections.json
│   └── queue_plot.png
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

> Note: keep `data/` and `runs/` in `.gitignore` by default to avoid pushing large files to GitHub.

---

## Quick setup

1. Create & activate a Python venv (or use conda).
2. Install requirements: `pip install -r requirements.txt`.
3. Create a `.env` with your API key(s) and ensure `.env` is in `.gitignore`.

---

## Detailed setup

### 1) Python environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Add `python-dotenv`, `ultralytics`, `roboflow`, `opencv-python`, `matplotlib`, `numpy`, `pandas`, and any tracker libraries you used (e.g., `sort` or `opencv-contrib-python`) to `requirements.txt`.

### 2) .env (API keys)

Create `.env` at repo root containing keys (example for Roboflow):

```
ROBOFLOW_API_KEY=your_key_here
```

Load it in notebooks with `python-dotenv`:

```python
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('ROBOFLOW_API_KEY')
```

Do **not** commit `.env` — add `.env` to `.gitignore`.

### 3) Git LFS (optional — for large models)

If your model file (`best.pt`) > 100MB, enable Git LFS and track `.pt`:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

Alternatively upload the model to Google Drive, Kaggle, or Roboflow and place a download snippet in the notebook.

---

## How to run (notebook order)

1. **01_Model_Training.ipynb** — dataset download & training

   * Download dataset (Roboflow or local). If using Roboflow, ensure `.env` contains the key.
   * Inspect dataset, adjust `data.yaml` if necessary.
   * Run training with YOLOv8 (Ultralytics API). Save `best.pt`.

2. **02_Traffic_Logic.ipynb** — inference, tracking, adaptive logic

   * Place `test_video.mp4` in `data/` (or change path in notebook).
   * Load `best.pt` (or a public weights URL).
   * Run inference frame-by-frame, pass detections to tracker, compute queue metrics, and save annotated video + plots.

There are clearly marked cells: `# CONFIG` where you set paths, weights, and parameters like `BASE_GREEN`, `K` (gain), and `QUEUE_THRESHOLD`.

---

## Core algorithm & design decisions

### YOLOv8 model training (notebook 01)

* **Why YOLOv8?** Fast, production-friendly, and easy to train with Ultralytics API. Fits well for video pipelines where per-frame latency matters.
* **Dataset**: Use Roboflow export or a custom dataset with `train/val/test` splits and `data.yaml` file specifying classes and paths.
* **Hyperparams to watch**: `imgsz`, `batch`, `epochs`, and augmentation. For small datasets, use transfer learning by loading a pretrained backbone.

Tips:

* Start with `epochs=50` and observe mAP; reduce if overfitting.
* Use `val` set for early stopping.

### Detection → Tracking → Counting (notebook 02)

* **Detections**: model returns `[x1,y1,x2,y2,conf,cls]` per object per frame.
* **Tracker**: we use a lightweight online tracker (e.g., SORT, ByteTrack-like logic, or OpenCV’s `cv2.TrackerCSRT_create()` wrappers). The tracker maintains persistent IDs so you can count unique vehicles crossing zones.
* **Counting & queue**:

  * Define a **counting line** or a polygonal **zone** in frame coordinates.
  * When a tracked object crosses the line in the direction of interest, increment counts.
  * Compute queue length as either:

    * number of vehicles inside a pre-defined *queue zone*, or
    * sum of distances of vehicles from stop-line normalized by lane length (for density estimate).

Implementation detail (pseudo):

```python
for frame in video:
    detections = model(frame)
    tracks = tracker.update(detections)
    for t in tracks:
        if crosses_count_line(t): counts += 1
    queue_len = len([t for t in tracks if in_queue_zone(t)])
    green_time = BASE_GREEN + K * queue_len
```

### Adaptive green-time formula

A simple linear control rule used here:
$$
GreenTime = \max(MIN\_GREEN,\ Base + K \cdot QueueLength)
$$


* `Base` is minimum green time (configurable).
* `K` is the tunable gain (seconds per vehicle). Suggested start: `Base=10s, K=2s/vehicle`.
* `MIN_GREEN` and `MAX_GREEN` enforce safety/time limits.

Why this simple rule? It's explainable, robust, and easy to implement for a demo. For production, consider PID or RL-based controllers.

---

## Evaluation & visualization

* Save per-frame queue length to a CSV: `timestamp, queue_len, green_time`.
* Plot `queue_len` and `green_time` over time to validate controller responsiveness.
* Metrics to track: detection `mAP`, false positives/negatives on key frames, average queue length reduction, and safety constraints (e.g., min wait time).

---

## Where to store large files (models/videos)

* **Recommended:** Host model weights on Google Drive / Kaggle dataset / Roboflow and add a download cell in the training notebook. Avoid pushing large `.pt` or `.mp4` to GitHub unless using Git LFS.
* Provide a `data/README.md` with instructions for team members to download model and test video.

---

## Common errors & fixes

* **`ModuleNotFoundError: roboflow`** → `pip install roboflow` and add to `requirements.txt`.
* **KeyError / None when loading env key** → ensure `.env` is in project root and `load_dotenv()` is called BEFORE `os.getenv()`.
* **`RuntimeError: CUDA out of memory`** → lower `imgsz` or `batch` size; or run on CPU for demo (slower).
* **`Git push` rejected (file too large)`** → file >100MB; use Git LFS or remove the file and host externally.
* **Tracker ID switches frequently** → reduce detection noise by increasing detection confidence threshold and enable NMS; consider a stronger tracker.

---

## Collaboration notes (for group)

* Keep `data/` and `runs/` ignored. Store small demo assets in `assets/`.
* Use branches for features: `feature/training`, `feature/inference`, `feature/visuals`.
* Add `CODEOWNERS` or a short `CONTRIBUTING.md` describing the role of each member and code-review process.

Suggested short blurb for GitHub description: `Adaptive-Traffic-Signal-YOLO — YOLOv8-based traffic detection + lightweight tracking for adaptive signal timing. Group project.`

---

## Example config (edit in notebook `# CONFIG` cell)

```python
CONFIG = {
    'weights_path': 'data/best.pt',
    'video_path': 'data/test_video.mp4',
    'output_path': 'output/annotated.mp4',
    'imgsz': 640,
    'conf_thresh': 0.4,
    'iou_thresh': 0.5,
    'BASE_GREEN': 10,    # seconds
    'K': 2,              # seconds per vehicle
    'MIN_GREEN': 7,
    'MAX_GREEN': 45
}
```

---

## How to share this README or copy into your repo

* Use GitHub Desktop: create repo, copy files, commit with message `Initial commit: notebooks + README` and publish.

---

## Contributing

If you'd like to improve this project, here are a few ideas:

* Replace linear rule with a PID controller or RL agent for green-time optimization.
* Add multi-lane support and per-lane queue estimation.
* Replace tracker with a ReID-based tracker for long-term ID stability.

---

## License

This repository uses the MIT license. See `LICENSE` file.

---

## Contact / Authors

Group members: add GitHub handles in `CONTRIBUTORS.md` or below:

* Member 1 — [@Rishy-09](https://github.com/Rishy-09)
* Member 2 — [@MoHiT05os](https://github.com/MoHiT05os)
* Member 3 — [@Teammate2](https://github.com/Teammate2)
* Member 4 — [@Teammate3](https://github.com/Teammate3)

