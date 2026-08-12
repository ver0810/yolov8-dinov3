#!/usr/bin/env python3
"""Pseudo-label the v2 test split with a trained model and merge into training.

Strategy:
  1. Predict on v2 test (180 imgs) with a strong checkpoint (e.g. v2_034).
  2. Keep detections with conf >= --conf and IoU-overlap with GT below --iou_gt
     (avoid duplicating boxes that GT already covers; avoids label noise).
  3. Write merged labels (GT + pseudo) to outputs/v2_pseudo/ and a train list
     that combines v2 train images with the pseudo-labeled test images.

Usage:
    uv run python scripts/make_pseudo_labels.py \
        --weights outputs/runs/v2_034_yolo11s_1280_oversample/weights/best.pt \
        --conf 0.5 --iou-gt 0.5
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
syspath = ROOT / "third_party" / "ultralytics"
import sys

sys.path.insert(0, str(syspath))


def xywh2xyxy(b):
    x, y, w, h = b
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    return inter / (a_area(a) + a_area(b) - inter)


def a_area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/own_dataset_v2.yaml")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou-gt", type=float, default=0.5, help="drop pseudo boxes overlapping GT by > this IoU")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out-name", default="v2_pseudo")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    test_img_dir = ROOT / "outputs" / "dataset_v2" / "test" / "images"
    res = model.predict(source=str(test_img_dir), imgsz=args.imgsz, conf=args.conf, verbose=False,
                        max_det=300, device=0)

    test_lbl_dir = ROOT / "outputs" / "dataset_v2" / "test" / "labels"
    train_img_dir = ROOT / "outputs" / "dataset_v2" / "train" / "images"
    out_dir = ROOT / "outputs" / args.out_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)

    n_all, n_gts, n_pseudo, n_drop = 0, 0, 0, 0
    merged_lines = []
    for r in res:
        stem = Path(r.path).stem
        # GT boxes
        gt_file = test_lbl_dir / f"{stem}.txt"
        gts = []
        if gt_file.exists():
            for line in open(gt_file):
                p = line.split()
                gts.append((int(p[0]), xywh2xyxy([float(x) for x in p[1:5]])))
        # pseudo detections
        lines_txt = []
        for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy().astype(int)):
            if conf < args.conf:
                continue
            b_norm = [float(box[0] / r.orig_shape[1]), float(box[1] / r.orig_shape[0]),
                      float(box[2] / r.orig_shape[1]), float(box[3] / r.orig_shape[0])]
            # drop if overlaps GT (duplicate label of existing annotation)
            if any(iou(b_norm, gb) > args.iou_gt for _, gb in gts):
                n_drop += 1
                continue
            cx = (b_norm[0] + b_norm[2]) / 2
            cy = (b_norm[1] + b_norm[3]) / 2
            w = b_norm[2] - b_norm[0]
            h = b_norm[3] - b_norm[1]
            lines_txt.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_pseudo += 1

        # merged = GT + pseudo (GT first)
        merged = []
        for c, gb in gts:  # gb = xyxy
            cx = (gb[0] + gb[2]) / 2
            cy = (gb[1] + gb[3]) / 2
            merged.append(f"{c} {cx:.6f} {cy:.6f} {gb[2]-gb[0]:.6f} {gb[3]-gb[1]:.6f}")
        merged += lines_txt
        n_gts += len(gts)
        if merged:
            (out_dir / "labels" / f"{stem}.txt").write_text("\n".join(merged) + "\n")
            (out_dir / "images" / Path(r.path).name).symlink_to(Path(r.path).resolve())
            n_all += 1

    # training list: all v2 train images + pseudo-labeled test images
    train_list = sorted(str(p) for p in train_img_dir.glob("*"))
    pseudo_list = sorted(str(p) for p in (out_dir / "images").glob("*"))
    combined = train_list + pseudo_list
    list_file = out_dir / "pseudo_train.txt"
    list_file.write_text("\n".join(combined) + "\n")
    print(f"test imgs with merged labels: {n_all} | GT boxes: {n_gts} | pseudo boxes added: {n_pseudo} (dropped {n_drop} GT-overlaps)")
    print(f"train images: {len(train_list)} + pseudo {len(pseudo_list)} = {len(combined)}")
    print(f"list: {list_file}")


if __name__ == "__main__":
    main()