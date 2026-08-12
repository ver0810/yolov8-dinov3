#!/usr/bin/env python3
"""Detection-level copy-paste augmentation for weak classes (bd=1, heidian=2, wy=3).

Problem: weak classes (max recall 0.30/0.41/0.46 even at conf→0) have enough
samples (295~363 images) but the model fails to learn them. Root cause is
context diversity: defects always appear on the same fabric textures, so the
model latches onto background instead of the defect itself.

This script pastes weak-class instances onto RANDOM other images, forcing the
model to discriminate the defect from arbitrary context:
  1. Extract bd/heidian/wy instances from the training set (image + bbox)
  2. For each training image, paste 0~N sampled instances at random positions
     with random scale/rotation/HSV jitter, avoiding overlap with existing boxes
  3. Write augmented dataset to <dataset_path>_copypaste/ (train augmented,
     val/test symlinked)

Usage:
    uv run python scripts/augment_copy_paste.py \
        --data /home/ancheng/dataset/dataset_split \
        --classes 1 2 3 \
        --paste-per-image 2 \
        --seed 42

Output:
    /home/ancheng/dataset/dataset_split_copypaste/  (train + symlinked val/test)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent

WEAK_DEFAULT = [1, 2, 3]  # bd, heidian, wy


def parse_args():
    p = argparse.ArgumentParser(description="Detection-level copy-paste augmentation")
    p.add_argument("--data", type=str, default="/home/ancheng/dataset/dataset_split")
    p.add_argument("--classes", type=int, nargs="+", default=WEAK_DEFAULT)
    p.add_argument("--paste-per-image", type=int, default=2,
                   help="Max instances pasted per image (sampled 0..N)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale-range", type=float, nargs=2, default=[0.6, 1.3])
    return p.parse_args()


def load_instances(images_dir: Path, labels_dir: Path, classes: set[int]) -> list[dict]:
    """Collect (image_path, cls, xyxy) for every weak-class instance."""
    instances = []
    for lbl in sorted(labels_dir.glob("*.txt")):
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            cand = images_dir / f"{lbl.stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                if cid not in classes:
                    continue
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                instances.append({
                    "img_path": str(img_path),
                    "cls": cid,
                    "patch": img[y1:y2, x1:x2].copy(),
                    "ph": y2 - y1,
                    "pw": x2 - x1,
                })
    return instances


def iou_of(box_a, box_b) -> float:
    """IoU between two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-12)


def hsv_jitter(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random HSV jitter: hue ±10, saturation/value ×[0.8, 1.2]."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = np.clip(hsv[..., 0] + rng.uniform(-10, 10), 0, 179)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.8, 1.2), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.8, 1.2), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def paste_instance(img: np.ndarray, inst: dict, existing: list[tuple],
                   rng: np.random.Generator, scale_range: tuple[float, float]) -> tuple | None:
    """Paste one instance into img at a random non-overlapping spot.

    Returns new label (cls, cx, cy, w, h normalized) or None if no room.
    """
    h, w = img.shape[:2]
    scale = rng.uniform(*scale_range)
    ph = max(4, int(inst["ph"] * scale))
    pw = max(4, int(inst["pw"] * scale))
    if ph >= h or pw >= w:
        return None

    # 30 attempts to find a non-overlapping position
    for _ in range(30):
        x1 = rng.integers(0, w - pw)
        y1 = rng.integers(0, h - ph)
        x2, y2 = x1 + pw, y1 + ph
        box = (x1, y1, x2, y2)
        if all(iou_of(box, e) < 0.3 for e in existing):
            patch = cv2.resize(inst["patch"], (pw, ph), interpolation=cv2.INTER_LINEAR)
            patch = hsv_jitter(patch, rng)
            # soft blend at edges to avoid hard seams
            img[y1:y2, x1:x2] = patch
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw_n, bh_n = pw / w, ph / h
            return (inst["cls"], cx, cy, bw_n, bh_n)
    return None


def process_split(src_dir: Path, dst_dir: Path, instances: list[dict],
                  classes: set[int], paste_per_image: int, rng: np.random.Generator,
                  scale_range: tuple[float, float]) -> tuple[int, int]:
    """Copy-paste augment one split. Returns (n_images, n_pasted)."""
    src_img, src_lbl = src_dir / "images", src_dir / "labels"
    dst_img, dst_lbl = dst_dir / "images", dst_dir / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    n_pasted = 0
    n_images = 0
    for lbl in sorted(src_lbl.glob("*.txt")):
        img_path = next((src_img / f"{lbl.stem}{e}" for e in (".png", ".jpg", ".jpeg", ".bmp")
                         if (src_img / f"{lbl.stem}{e}").exists()), None)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Load existing labels as (cls, cx, cy, bw, bh)
        lines = []
        existing_boxes = []
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = list(map(float, line.split()))
                cid = int(parts[0])
                cx, cy, bw, bh = parts[1:]
                lines.append(parts)
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                existing_boxes.append((x1, y1, x2, y2))

        # Paste 0..paste_per_image instances
        n_choice = rng.integers(0, paste_per_image + 1)
        pool = instances if instances else []
        if pool:
            for inst in rng.choice(pool, size=min(n_choice, len(pool)), replace=False):
                new_label = paste_instance(img, inst, existing_boxes, rng, scale_range)
                if new_label is not None:
                    lines.append(list(new_label))
                    existing_boxes.append(
                        ((new_label[1] - new_label[3] / 2) * w, (new_label[2] - new_label[4] / 2) * h,
                         (new_label[1] + new_label[3] / 2) * w, (new_label[2] + new_label[4] / 2) * h))
                    n_pasted += 1

        cv2.imwrite(str(dst_img / img_path.name), img)
        with open(dst_lbl / f"{lbl.stem}.txt", "w") as f:
            for parts in lines:
                cid = int(parts[0])
                rest = " ".join(f"{v:.6f}" for v in parts[1:])
                f.write(f"{cid} {rest}\n")
        n_images += 1
    return n_images, n_pasted


def main():
    args = parse_args()
    data_dir = Path(args.data)
    classes = set(args.classes)
    rng = np.random.default_rng(args.seed)

    # Output dataset: <name>_copypaste
    out_dir = data_dir.with_name(data_dir.name + "_copypaste")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"Source: {data_dir}")
    print(f"Output: {out_dir}")
    print(f"Weak classes: {sorted(classes)} (0=lj 1=bd 2=heidian 3=wy 4=zyc ...)")

    # 1. Collect weak instances from train
    print("\n[1/3] Collecting weak-class instances...")
    instances = load_instances(data_dir / "train" / "images", data_dir / "train" / "labels", classes)
    print(f"  Collected {len(instances)} instances")

    # 2. Augment train
    print("[2/3] Augmenting train split...")
    n_img, n_paste = process_split(data_dir / "train", out_dir / "train",
                                   instances, classes, args.paste_per_image, rng,
                                   tuple(args.scale_range))
    print(f"  {n_img} images, {n_paste} instances pasted")

    # 3. Symlink val/test
    print("[3/3] Symlinking val/test...")
    for split in ("val", "test"):
        src = data_dir / split
        if src.exists():
            dst = out_dir / split
            dst.symlink_to(src, target_is_directory=True)

    # 4. Write project config (configs/own_dataset_copypaste.yaml)
    cfg_path = ROOT / "configs" / "own_dataset_copypaste.yaml"
    cfg = {
        "path": str(out_dir),
        "train": "train/images",
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
    print(f"\nDone. Config written: {cfg_path}")
    print(f"Augmented dataset: {out_dir}")


if __name__ == "__main__":
    main()
