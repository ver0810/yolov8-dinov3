#!/usr/bin/env python3
"""Train baseline YOLOv8 or DINOv3-fusion models via Ultralytics API."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_MAP = {
    "baseline": "yolov8n.pt",
    "full": ROOT / "third_party/ultralytics/ultralytics/cfg/models/v8/yolov8-dinov3.yaml",
    "hybrid": ROOT / "third_party/ultralytics/ultralytics/cfg/models/v8/yolov8-dinov3-hybrid.yaml",
    "yolo_family": None,  # model must be provided via --model or yolo_model in config
}


def load_exp(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO / YOLO-DINOv3")
    p.add_argument("--mode", choices=list(MODEL_MAP), default="yolo_family")
    p.add_argument("--config", type=str, default=None, help="Optional exp YAML (configs/exp_*.yaml)")
    p.add_argument("--model", type=str, default=None, help="Override model pt/yaml path")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="e.g. 0, cpu, mps")
    p.add_argument("--project", type=str, default=str(ROOT / "outputs/runs"))
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--weights", type=str, default=None, help="Init weights (pt) for yaml models")
    p.add_argument("--task", type=str, default=None, choices=["detect", "segment", "classify", "pose", "obb"], help="Force task (e.g. detect for seg weights)")
    return p.parse_args()


def main():
    args = parse_args()
    exp = load_exp(args.config)

    mode = exp.get("mode", args.mode)
    model_path = args.model or exp.get("yolo_model") or MODEL_MAP.get(mode)
    if not model_path:
        raise ValueError("--model or yolo_model in config is required for this mode")
    data = args.data or exp.get("data_config", str(ROOT / "configs/tianchi.yaml"))
    name = args.name or exp.get("exp_name") or f"{mode}"
    train_kw = {
        "data": str(data),
        "epochs": args.epochs if args.epochs is not None else exp.get("epochs", 100),
        "imgsz": args.imgsz if args.imgsz is not None else exp.get("img_size", 640),
        "batch": args.batch if args.batch is not None else exp.get("batch_size", 8),
        "device": args.device if args.device is not None else exp.get("device", None),
        "project": str(args.project or exp.get("output_dir", ROOT / "outputs/runs")),
        "name": name,
        "patience": args.patience if args.patience is not None else exp.get("patience", 50),
        "lr0": args.lr0 if args.lr0 is not None else exp.get("lr", 0.01),
        "exist_ok": True,
        "resume": args.resume,
    }
    # Drop None device so ultralytics auto-selects CUDA/MPS/CPU
    if train_kw["device"] is None:
        train_kw.pop("device")

    # Pass custom augmentation & training params from config
    _FORWARD_KEYS = (
        "use_randaugment", "copy_paste", "close_mosaic",
        "cos_lr", "multi_scale", "cls", "cls_pw", "freeze",
        "mixup", "cutmix", "lrf", "weight_decay", "dropout",
        "mosaic", "erasing", "auto_augment", "degrees",
    )
    for key in _FORWARD_KEYS:
        if key in exp:
            train_kw[key] = exp[key]

    print(f"mode={mode} model={model_path}")
    task = args.task or exp.get("task", None)
    model = YOLO(str(model_path), task=task)
    if args.weights and str(model_path).endswith((".yaml", ".yml")):
        model.load(args.weights)
    elif exp.get("pretrained") and str(model_path).endswith((".yaml", ".yml")):
        model.load(exp["pretrained"])
    model.train(**train_kw)


if __name__ == "__main__":
    main()
