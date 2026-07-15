# YOLO-DINO: YOLOv8 + DINOv3 Backbone Fusion for Textile Defect Detection

Industrial textile defect detection via backbone fusion of YOLOv8 and DINOv3 (Meta's self-supervised ViT foundation model).

## Research Question

Can replacing or augmenting YOLOv8's CSPDarknet backbone with DINOv3's pretrained ViT improve small-defect detection on industrial fabric imagery?

## Architecture

| Variant | Backbone | Neck | Head |
|---------|----------|------|------|
| A0 Baseline | CSPDarknet (YOLOv8) | PAN-FPN | Detect |
| A1 Full | DINOv3 ViT-S/B (frozen) | FusionNeck | Detect |
| A2 Full+LoRA | DINOv3 ViT-S/B (LoRA) | FusionNeck | Detect |
| A3 Hybrid | CSP(P1-P3) + DINOv3(P4-P5) | HybridFusionNeck | Detect |

## Dataset

[Tianchi Fabric Defect Detection](https://tianchi.aliyun.com/dataset/147338) — 15 defect categories, ~7000+ images.

Defect classes: 破洞, 水渍, 油污, 色斑, 纬缩, 经缩, 松经, 紧经, 吊经, 粗经, 粗纬, 浆斑, 整经结, 星跳, 跳花

## Quick Start

```bash
# 1. Install deps
pixi install

# 2. Setup DINOv3 (clone repo)
pixi run setup-dinov3

# 3. Sanity check
pixi run sanity

# 4. Prepare dataset (after downloading raw data)
pixi run python scripts/prepare_data.py --raw-dir data/raw/tianchi

# 5. Train baseline
pixi run python scripts/train.py --mode baseline

# 6. Train DINOv3 full replacement
pixi run python scripts/train.py --mode full --dinov3-size vits

# 7. Train hybrid backbone
pixi run python scripts/train.py --mode hybrid --dinov3-size vits

# 8. Evaluate
pixi run python scripts/eval.py --mode full --weights outputs/runs/A1_dinov3_full_frozen/best.pt

# 9. Predict
pixi run python scripts/predict.py --mode full --weights outputs/runs/A1_dinov3_full_frozen/best.pt --input data/raw/test_images
```

## Project Structure

```
yolo-dino/
├── pixi.toml              # Dependency management
├── pyproject.toml
├── configs/               # Experiment & data configs
├── data/                  # Dataset (gitignored)
├── models/pretrained/     # Checkpoints (gitignored)
├── src/yolo_dino/
│   ├── models/
│   │   ├── backbones/     # DINOv3 wrapper
│   │   ├── necks/         # FusionNeck, HybridFusionNeck
│   │   └── yolo_dinov3.py # YOLODINOv3, YOLODINOv3Hybrid
│   ├── data/              # Tianchi dataset loader & converter
│   └── utils/             # Metrics, helpers
├── scripts/               # Train, eval, predict, setup
├── notebooks/             # Analysis & visualization
└── outputs/               # Training runs (gitignored)
```

## Environment

- Python 3.11, PyTorch 2.5+ (MPS), Ultralytics 8.3+
- Hardware: Apple Silicon (MPS) / CUDA
