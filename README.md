# YOLO-DINO: YOLOv8 + DINOv3 for Textile Defect Detection

Industrial textile defect detection by extending a **local Ultralytics fork** with DINOv3 backbone modules.

## Layout

```
yolov8-dinov3/
├── third_party/ultralytics/     # local fork — model changes live here
│   └── ultralytics/
│       ├── nn/modules/dinov3.py # DINOv3Backbone (stub + TODOs)
│       └── cfg/models/v8/
│           ├── yolov8-dinov3.yaml
│           └── yolov8-dinov3-hybrid.yaml
├── configs/                     # data + experiment hypers
├── scripts/                     # train / val / predict / prepare / sanity
├── data/                        # datasets (gitignored)
└── outputs/                     # runs (gitignored)
```

## Variants

| Mode | Model | Notes |
|------|--------|--------|
| `baseline` | `yolov8n.pt` | Official YOLOv8 |
| `full` | `yolov8-dinov3.yaml` | DINOv3 backbone + PAN head (encoder still **stub**) |
| `hybrid` | `yolov8-dinov3-hybrid.yaml` | Placeholder = YOLOv8; TODOs for CSP+DINO |

DINOv3 backbone: local Meta repo + optional `.pth` (see below).

## Setup

```bash
uv sync
uv run bash scripts/setup_dinov3.sh   # clone third_party/dinov3
# download weights → third_party/dinov3_weights/*.pth  (or DINOV3_WEIGHTS=/path/to.pth)
uv run python scripts/sanity_check.py
```

## Train / val / predict

```bash
# data
uv run python scripts/prepare_data.py --raw-dir data/raw/tianchi

# baseline
uv run python scripts/train.py --mode baseline --config configs/exp_baseline.yaml

# DINOv3 full (stub backbone until wired)
uv run python scripts/train.py --mode full --config configs/exp_dinov3_full.yaml

# validate
uv run python scripts/val.py --weights outputs/runs/A0_baseline_yolov8/weights/best.pt

# predict
uv run python scripts/predict.py --weights path/to/best.pt --source path/to/images
```

## Dataset

[Tianchi Fabric Defect Detection](https://tianchi.aliyun.com/dataset/147338) — see `configs/tianchi.yaml`.
