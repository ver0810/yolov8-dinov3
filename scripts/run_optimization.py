#!/usr/bin/env python3
"""Run optimization experiments 008–011 (RandAugment + PSA depth ablation)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

experiments = [
    ("008", "configs/exp_v11s_randaug.yaml", "yolo11s + RandAugment"),
    ("009", "configs/exp_v11s_psa_p3.yaml", "yolo11s + PSA @ P3"),
    ("010", "configs/exp_v11s_psa_p4.yaml", "yolo11s + PSA @ P4"),
    ("011", "configs/exp_v11s_psa_p345.yaml", "yolo11s + PSA @ P3+P4+P5"),
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
