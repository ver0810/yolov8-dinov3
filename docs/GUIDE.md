# YOLO-DINO 使用指南

YOLOv8 检测 + DINOv3 骨干，用于工业纺织瑕疵检测。模型改动在本地 Ultralytics fork 中；训练/验证走官方 API。

## 目录

```
yolov8-dinov3/
├── third_party/ultralytics/     # Ultralytics 本地源码（可编辑安装）
│   └── ultralytics/nn/modules/dinov3.py   # DINOv3Backbone
│   └── ultralytics/cfg/models/v8/
│       ├── yolov8-dinov3.yaml           # A1: 全 DINOv3 backbone
│       └── yolov8-dinov3-hybrid.yaml    # A3: hybrid 占位（仍为 YOLOv8）
├── third_party/dinov3/          # Meta DINOv3 源码（setup 脚本 clone）
├── third_party/dinov3_weights/  # 手动放置 .pth 预训练权重
├── configs/                     # 数据与实验配置
├── scripts/                     # 训练 / 验证 / 预测 / 数据准备
├── data/                        # 数据集（gitignore）
└── outputs/                     # 训练输出（gitignore）
```

## 变体

| mode | 模型 | 说明 |
|------|------|------|
| `baseline` | `yolov8n.pt` | 原版 YOLOv8 |
| `full` | `yolov8-dinov3.yaml` | DINOv3 ViT → P3/P4/P5 → PAN + Detect |
| `hybrid` | `yolov8-dinov3-hybrid.yaml` | 规划 CSP early + DINO deep（当前仍是标准 YOLOv8） |

## 环境

```bash
# 安装依赖（含本地 editable ultralytics）
uv sync

# 克隆 Meta DINOv3 源码
uv run bash scripts/setup_dinov3.sh

# 自检（建图 + 前向）
uv run python scripts/sanity_check.py
```

## DINOv3 权重

1. 申请并下载：https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/  
2. 用 `wget` 下载，放到例如：

```text
third_party/dinov3_weights/dinov3_vits16_pretrain_lvd1689m-xxxx.pth
```

3. 或指定路径：

```bash
export DINOV3_WEIGHTS=/path/to/xxx.pth
# 可选
export DINOV3_REPO=/path/to/dinov3          # 默认 third_party/dinov3
export DINOV3_WEIGHTS_DIR=/path/to/weights  # 默认 third_party/dinov3_weights
```

无权重时仍可跑通（`pretrained=False` 随机初始化，会打 warning）。

加载逻辑：`third_party/ultralytics/ultralytics/nn/modules/dinov3.py`

## 数据

```bash
# 天池原始数据 → YOLO 格式
uv run python scripts/prepare_data.py --raw-dir data/raw/tianchi

# 数据配置
# configs/tianchi.yaml
```

## 训练

```bash
# baseline
uv run python scripts/train.py --mode baseline --config configs/exp_baseline.yaml

# DINOv3 full
uv run python scripts/train.py --mode full --config configs/exp_dinov3_full.yaml

# hybrid（占位）
uv run python scripts/train.py --mode hybrid --config configs/exp_dinov3_hybrid.yaml

# 常用覆盖
uv run python scripts/train.py --mode full --epochs 50 --batch 4 --device 0
```

输出默认：`outputs/runs/<exp_name>/`

## 验证 / 预测

```bash
uv run python scripts/val.py \
  --weights outputs/runs/A0_baseline_yolov8/weights/best.pt

uv run python scripts/predict.py \
  --weights outputs/runs/A0_baseline_yolov8/weights/best.pt \
  --source data/raw/test_images
```

## 关键代码位置

| 内容 | 路径 |
|------|------|
| DINOv3 模块 | `third_party/ultralytics/ultralytics/nn/modules/dinov3.py` |
| 模块注册 | `.../nn/modules/__init__.py`、`.../nn/tasks.py` |
| A1 YAML | `.../cfg/models/v8/yolov8-dinov3.yaml` |
| A3 YAML | `.../cfg/models/v8/yolov8-dinov3-hybrid.yaml` |
| 训练入口 | `scripts/train.py` |

## 后续待办（简）

- [ ] 放入真实 DINOv3 `.pth` 后复测 sanity / 小规模 train  
- [ ] Hybrid：CSP early + DINOv3 deep  
- [ ] 可选：ViT 多层特征替代单层 resize 金字塔  
- [ ] 可选：LoRA（`peft`，`uv sync --extra lora`）
