#!/usr/bin/env python3
"""Prepare weak-class training subset for two-stage fine-tuning.

Selects all training images containing at least one weak-class instance
(bd=1, heidian=2, wy=3), writes an image-path list (.txt) that ultralytics
can use as the train source, and generates configs/own_dataset_weak.yaml.

Usage:
    uv run python scripts/prepare_weak_subset.py \
        --data /home/ancheng/dataset/dataset_split \
        --classes 1 2 3

Output:
    outputs/weak_subset/weak_train.txt   (image paths for training)
    configs/own_dataset_weak.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WEAK_DEFAULT = [1, 2, 3]


def parse_args():
    p = argparse.ArgumentParser(description="Prepare weak-class training subset")
    p.add_argument("--data", type=str, default="/home/ancheng/dataset/dataset_split")
    p.add_argument("--classes", type=int, nargs="+", default=WEAK_DEFAULT)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data)
    classes = set(args.classes)

    img_dir = data_dir / "train" / "images"
    lbl_dir = data_dir / "train" / "labels"

    selected = []
    n_weak = 0
    for lbl in sorted(lbl_dir.glob("*.txt")):
        with open(lbl) as f:
            cls_ids = [int(line.split()[0]) for line in f if line.strip()]
        if any(c in classes for c in cls_ids):
            img_path = next((img_dir / f"{lbl.stem}{e}" for e in (".png", ".jpg", ".jpeg", ".bmp")
                             if (img_dir / f"{lbl.stem}{e}").exists()), None)
            if img_path is not None:
                selected.append(str(img_path))
                n_weak += sum(1 for c in cls_ids if c in classes)

    out_dir = ROOT / "outputs" / "weak_subset"
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "weak_train.txt"
    txt_path.write_text("\n".join(selected) + "\n")

    # Write data config: train=txt, val/test keep original
    cfg_path = ROOT / "configs" / "own_dataset_weak.yaml"
    cfg = {
        "path": str(data_dir),
        "train": str(txt_path),
        "val": "val/images",
        "test": "test/images",
        "nc": 17,
        "names": [
            "lj", "bd", "heidian", "wy", "zyc", "zmty", "cq", "zj", "bj",
            "jt", "yy", "bmss", "zy", "pd", "HD", "cy", "jiaodai",
        ],
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"Selected {len(selected)} images containing weak classes {sorted(classes)}")
    print(f"  Weak instances in subset: {n_weak}")
    print(f"  Train list: {txt_path}")
    print(f"  Config: {cfg_path}")


if __name__ == "__main__":
    main()
