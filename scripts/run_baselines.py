#!/usr/bin/env python3
"""Run all baseline experiments: YOLOv8/v11/v26, nano + small, with recommended params."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

BASELINES = [
    # (config_file, description)
    ("exp_v8n.yaml", "YOLOv8-Nano"),
    ("exp_v8s.yaml", "YOLOv8-Small"),
    ("exp_v11n.yaml", "YOLO11-Nano"),
    ("exp_v11s.yaml", "YOLO11-Small"),
    ("exp_v26n.yaml", "YOLO26-Nano"),
    ("exp_v26s.yaml", "YOLO26-Small"),
]

# WORLD_BASELINES = [("exp_world_s_v2.yaml", "YOLOWorld-Small v2")]  # needs English class names


def run_one(config_name: str, description: str) -> int:
    config_path = CONFIG_DIR / config_name
    print(f"\n{'=' * 60}")
    print(f"  Running: {description}  ({config_name})")
    print(f"{'=' * 60}\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train.py"),
            "--mode", "yolo_family",
            "--config", str(config_path),
        ],
        cwd=str(ROOT),
    )
    return result.returncode


def main():
    total = len(BASELINES)
    for i, (config, desc) in enumerate(BASELINES, 1):
        print(f"\n{'#' * 60}")
        print(f"  [{i}/{total}] Starting: {desc}")
        print(f"{'#' * 60}")
        rc = run_one(config, desc)
        if rc != 0:
            print(f"\n[ERROR] {desc} failed with exit code {rc}")
            sys.exit(rc)
        print(f"\n[{i}/{total}] Completed: {desc}")
    print(f"\n{'=' * 60}")
    print(f"  All {total} baselines completed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
