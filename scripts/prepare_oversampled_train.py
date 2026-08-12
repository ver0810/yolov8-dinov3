#!/usr/bin/env python3
"""Prepare full training set with weak-class oversampling.

Every training image appears once; images containing at least one weak-class
instance (bd=1, heidian=2, wy=3, bmss=11) are repeated `--factor` times total.
Keeps strong classes at full strength while multiplying weak-class exposure —
addresses 033's catastrophic forgetting (only-weak subset) without losing
global capability.

Usage:
    uv run python scripts/prepare_oversampled_train.py \
        --data /home/ancheng/dataset/dataset_split \
        --classes 1 2 3 11 \
        --factor 3

Output:
    outputs/oversampled/oversampled_train.txt
    configs/own_dataset_oversampled.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WEAK_DEFAULT = [1, 2, 3, 11]


def parse_args():
    p = argparse.ArgumentParser(description="Prepare full train set with weak-class oversampling")
    p.add_argument("--data", type=str, default="/home/ancheng/dataset/dataset_split")
    p.add_argument("--classes", type=int, nargs="+", default=WEAK_DEFAULT)
    p.add_argument("--factor", type=int, default=3, help="Total copies of weak-class images (1 = no oversampling)")
    p.add_argument("--out-name", type=str, default="oversampled", help="Output basename (list file + config yaml)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data)
    weak_classes = set(args.classes)
    factor = args.factor

    img_dir = data_dir / "train" / "images"
    lbl_dir = data_dir / "train" / "labels"

    lines = []
    n_weak_copies = 0
    n_total = 0
    for lbl in sorted(lbl_dir.glob("*.txt")):
        with open(lbl) as f:
            cls_ids = [int(line.split()[0]) for line in f if line.strip()]
        img_path = next((img_dir / f"{lbl.stem}{e}" for e in (".png", ".jpg", ".jpeg", ".bmp")
                         if (img_dir / f"{lbl.stem}{e}").exists()), None)
        if img_path is None:
            continue
        n_total += 1
        is_weak = any(c in weak_classes for c in cls_ids)
        copies = factor if is_weak else 1
        lines.extend([str(img_path)] * copies)
        if copies > 1:
            n_weak_copies += copies

    out_dir = ROOT / "outputs" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "oversampled_train.txt"
    txt_path.write_text("\n".join(lines) + "\n")

    cfg_path = ROOT / "configs" / f"own_dataset_{args.out_name}.yaml"
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

    strong = n_total - sum(1 for _ in [])
    n_weak_imgs = n_weak_copies // factor
    print(f"Total train images: {n_total} (weak-containing: {n_weak_imgs}, strong-only: {n_total - n_weak_imgs})")
    print(f"Oversampled weak classes {sorted(weak_classes)} x{factor} → train list lines: {len(lines)}")
    print(f"  Train list: {txt_path}")
    print(f"  Config: {cfg_path}")


if __name__ == "__main__":
    main()