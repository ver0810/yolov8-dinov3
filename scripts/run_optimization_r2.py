#!/usr/bin/env python3
"""Run optimization round 2 experiments 018–023 (resolution, cls weight, aug, freeze, scale, P2)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

experiments = [
    ("018", "configs/exp_v11s_1280.yaml",       "yolo11s + 1280 + cos_lr"),
    ("019", "configs/exp_v11s_cls_weight.yaml",  "yolo11s + 1280 + cls_weight"),
    ("020", "configs/exp_v11s_strong_aug.yaml",  "yolo11s + 1280 + strong_aug"),
    ("021", "configs/exp_v11s_freeze.yaml",      "yolo11s + 1280 + freeze10"),
    ("022", "configs/exp_v11m_1280.yaml",        "yolo11m + 1280"),
    ("023", "configs/exp_v11s_p2_nose.yaml",     "yolo11s + P2 head (no SE) + 1280"),
]

for num, config, desc in experiments:
    print(f"\n{'='*60}")
    print(f"[{num}] {desc}")
    print(f"{'='*60}")
    cmd = [
        sys.executable,
        str(ROOT / "scripts/train.py"),
        "--config", str(ROOT / config),
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[{num}] FAILED with code {result.returncode}")
        break
    print(f"[{num}] DONE")
