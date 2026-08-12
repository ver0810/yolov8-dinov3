#!/usr/bin/env python3
"""Sanity: build models and run a dummy forward."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "third_party/ultralytics/ultralytics/cfg/models/v8"

CHECKS = [
    ("baseline", "yolov8n.yaml"),
    ("full", "yolov8-dinov3.yaml"),
    ("hybrid", "yolov8-dinov3-hybrid.yaml"),
    ("convnext", "yolov8-dinov3-convnext.yaml"),
    ("vitms", "yolov8-dinov3-vitms.yaml"),
]


def check_one(name: str, cfg: str) -> None:
    path = cfg if cfg.endswith(".pt") else CFG_DIR / cfg
    print(f"\n=== {name}: {path} ===")
    model = YOLO(str(path))
    x = torch.zeros(1, 3, 640, 640)
    model.model.eval()
    with torch.no_grad():
        y = model.model(x)
    if isinstance(y, (list, tuple)):
        print("out types:", [type(t).__name__ for t in y])
        for i, t in enumerate(y):
            if hasattr(t, "shape"):
                print(f"  [{i}] shape={tuple(t.shape)}")
    elif hasattr(y, "shape"):
        print("out shape:", tuple(y.shape))
    else:
        print("out:", type(y))
    print("OK")


def main():
    failed = []
    for name, cfg in CHECKS:
        try:
            check_one(name, cfg)
        except Exception as e:
            print(f"FAIL {name}: {e}")
            failed.append(name)
    if failed:
        print("\nFailed:", failed)
        sys.exit(1)
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
