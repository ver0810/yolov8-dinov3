#!/usr/bin/env python3
"""Run optimization round 3: recall/precision joint improvement.

029: yolo11s + cls_pw=1.0 (clean retry at 640, isolated from 1280 pollution)
030: yolo11s + multi_scale=0.5 (random scale jitter 480~800)
031: yolo11s + weak-class copy-paste augmentation (bd/heidian/wy pasted)
032: 004 two-stage fine-tune on weak-class-only subset

031/032 require dataset prep scripts first (run automatically here).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"  FAILED with code {result.returncode}")
        sys.exit(1)
    print(f"  DONE")


def main():
    # Dataset prep for 031/032
    run([sys.executable, str(ROOT / "scripts/augment_copy_paste.py")],
        "Prepare copy-paste augmented dataset (031)")
    run([sys.executable, str(ROOT / "scripts/prepare_weak_subset.py")],
        "Prepare weak-class subset (032)")

    experiments = [
        # ("029", "configs/exp_v11s_clspw.yaml",        "yolo11s + cls_pw=1.0 @640"),
        # ("030", "configs/exp_v11s_multiscale.yaml",    "yolo11s + multi_scale=0.5")  # 跳过：峰值 960×960@batch16 需 ~9G，超 6G 显存，WSL2 兜底仍崩
        ("031", "configs/exp_v11s_copypaste.yaml",     "yolo11s + copy-paste aug"),
        ("032", "configs/exp_v11s_weakfinetune.yaml",  "004 → weak-class fine-tune"),
    ]

    for num, config, desc in experiments:
        run([sys.executable, str(ROOT / "scripts/train.py"), "--config", str(ROOT / config)],
            f"[{num}] {desc}")


if __name__ == "__main__":
    main()
