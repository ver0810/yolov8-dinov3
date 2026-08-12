#!/usr/bin/env python3
"""Train YOLO-World v2 + PSA (e=0.25). Patches C2fAttn for YAML-based stride computation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party/ultralytics"))

import ultralytics.nn.modules.block as block_mod
from ultralytics import YOLO

_orig_forward = block_mod.C2fAttn.forward


def _patched_forward(self, x, guide=None):
    if guide is None:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(y[-1])
        return self.cv2(torch.cat(y, 1))
    return _orig_forward(self, x, guide)


block_mod.C2fAttn.forward = _patched_forward

import torch  # noqa: E402

MODEL_YAML = "third_party/ultralytics/ultralytics/cfg/models/v8/yolov8s-worldv2-psa.yaml"
WEIGHTS = "yolov8s-worldv2.pt"
DATA = "configs/own_dataset_world.yaml"
OUTPUT = str(ROOT / "outputs/runs")
NAME = "015_yoloworlds_psa"

model = YOLO(MODEL_YAML, task="detect")
model.load(WEIGHTS)

model.train(
    data=DATA,
    epochs=100,
    imgsz=640,
    batch=8,
    lr0=0.01,
    patience=100,
    project=OUTPUT,
    name=NAME,
    exist_ok=True,
)
