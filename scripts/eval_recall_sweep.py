#!/usr/bin/env python3
"""Inference-side recall limit sweep.

Explores how far recall can be pushed WITHOUT any retraining:
  1. Single-model confidence threshold sweep
  2. Class-specific thresholds (low conf for weak classes)
  3. TTA (multi-scale + horizontal flip)
  4. Model ensemble (union of detections)

Metrics (IoU>=0.5 matching):
  - instance recall: TP / total GT instances
  - image recall: images with >=1 TP / images with GT
  - precision: TP / (TP + FP)

Usage:
    uv run python scripts/eval_recall_sweep.py \
        --data /home/ancheng/dataset/dataset_split \
        --models outputs/runs/004_yolo11s/weights/best.pt \
                 outputs/runs/006_yolo26s/weights/best.pt \
                 outputs/runs/014_yolo11s_psa/weights/best.pt

Output:
    outputs/eval_recall_sweep/report.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "ultralytics"))

from ultralytics import YOLO

IOU_THR = 0.5
CONF_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
WEAK_CLASSES = {1, 2, 3}  # bd, heidian, wy (from own_dataset.yaml order)


def parse_args():
    p = argparse.ArgumentParser(description="Inference-side recall limit sweep")
    p.add_argument("--data", type=str, default="/home/ancheng/dataset/dataset_split")
    p.add_argument("--models", type=str, nargs="+", required=True, help="best.pt paths (>=1)")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tta-scales", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    p.add_argument("--tta-flip", action="store_true", help="Enable horizontal-flip TTA")
    p.add_argument("--nms-iou", type=float, default=None, help="NMS IoU for duplicate merging (e.g. 0.5); None = raw union")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


# ----------------------------------------------------------------------
# GT loading
# ----------------------------------------------------------------------

def load_gt(data_dir: Path, split: str) -> list[dict]:
    """Load ground-truth boxes (absolute xyxy in original image coords)."""
    img_dir = data_dir / split / "images"
    lbl_dir = data_dir / split / "labels"
    samples = []
    for img_path in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        h, w = cv2.imread(str(img_path)).shape[:2]
        gts = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                gts.append((cid, x1, y1, x2, y2))
        if gts:
            samples.append({"img": str(img_path), "gt": gts})
    return samples


# ----------------------------------------------------------------------
# Prediction (low-conf, returns boxes in original image coords)
# ----------------------------------------------------------------------

def predict_boxes(model, img: np.ndarray, imgsz: int, flip: bool = False):
    """Run prediction at conf=0.001; returns (boxes, confs, clss)."""
    img_in = cv2.flip(img, 1) if flip else img
    r = model.predict(img_in, imgsz=imgsz, conf=0.001, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float64)
    confs = r.boxes.conf.cpu().numpy().astype(np.float64)
    clss = r.boxes.cls.cpu().numpy().astype(int)
    if flip:
        W = img.shape[1]
        boxes = np.stack([W - boxes[:, 2], boxes[:, 1], W - boxes[:, 0], boxes[:, 3]], axis=1)
    return boxes, confs, clss


def gather_predictions(models, sample, imgsz, tta_scales, tta_flip, nms_iou=None):
    """Run all models (+TTA variants), NMS-merge duplicates, return detections.

    Args:
        nms_iou: IoU threshold for NMS merging. None = no NMS (raw union).
    """
    img = cv2.imread(sample["img"])
    all_boxes, all_confs, all_clss = [], [], []
    for model in models:
        for scale in tta_scales:
            sz = int(imgsz * scale)
            b, c, s = predict_boxes(model, img, sz, flip=False)
            all_boxes.append(b)
            all_confs.append(c)
            all_clss.append(s)
            if tta_flip:
                bf, cf, sf = predict_boxes(model, img, sz, flip=True)
                all_boxes.append(bf)
                all_confs.append(cf)
                all_clss.append(sf)
    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    boxes = np.concatenate(all_boxes)
    confs = np.concatenate(all_confs)
    clss = np.concatenate(all_clss)
    if nms_iou is not None and len(boxes) > 0:
        boxes, confs, clss = nms_merge(boxes, confs, clss, nms_iou)
    return boxes, confs, clss


def nms_merge(boxes, confs, clss, iou_thr):
    """Class-aware NMS merge. Returns (boxes, confs, clss) with duplicates removed."""
    import torch
    from torchvision.ops import nms

    keep_all = []
    for c in np.unique(clss):
        idx = np.where(clss == c)[0]
        if len(idx) == 0:
            continue
        keep = nms(
            torch.tensor(boxes[idx], dtype=torch.float32),
            torch.tensor(confs[idx], dtype=torch.float32),
            iou_thr,
        )
        keep_all.append(idx[keep.numpy()])
    if not keep_all:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    keep = np.concatenate(keep_all)
    return boxes[keep], confs[keep], clss[keep]


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------

def box_iou(a, b):
    """IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-12)


def match_image(pred_boxes, pred_confs, pred_clss, gts, conf_thr, cls_specific=False):
    """Greedy IoU matching. Returns (tp_gt_indices, n_fp)."""
    keep = pred_confs >= conf_thr
    if cls_specific:
        # weak classes use lower threshold
        weak_keep = pred_confs >= (conf_thr / 10)
        keep = np.where(np.isin(pred_clss, list(WEAK_CLASSES)), weak_keep, keep)

    matched = set()
    tp = set()
    for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
        best_iou = 0.0
        best_pi = -1
        gbox = (gx1, gy1, gx2, gy2)
        for pi in range(len(pred_boxes)):
            if pi in matched or not keep[pi]:
                continue
            iou = box_iou(gbox, pred_boxes[pi])
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        if best_iou >= IOU_THR:
            matched.add(best_pi)
            tp.add(gi)

    n_fp = int(keep.sum()) - len(matched)
    return tp, n_fp


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------

def sweep(samples, models, imgsz, tta_scales, tta_flip, cls_specific=False, nms_iou=None):
    """Run threshold sweep over pre-gathered predictions (cached per sample)."""
    # Gather predictions once per sample (union across models + TTA)
    cached = []
    for s in samples:
        cached.append(gather_predictions(models, s, imgsz, tta_scales, tta_flip, nms_iou))

    rows = []
    for conf_thr in CONF_THRESHOLDS:
        total_gt = 0
        total_tp = 0
        total_fp = 0
        n_img_gt = 0
        n_img_hit = 0
        per_cls_tp = {}
        per_cls_gt = {}
        for s, (pb, pc, ps) in zip(samples, cached):
            tp, fp = match_image(pb, pc, ps, s["gt"], conf_thr, cls_specific)
            total_gt += len(s["gt"])
            total_tp += len(tp)
            total_fp += fp
            n_img_gt += 1
            if len(tp) > 0:
                n_img_hit += 1
            for gi in tp:
                gc = s["gt"][gi][0]
                per_cls_tp[gc] = per_cls_tp.get(gc, 0) + 1
            for gc, *_ in s["gt"]:
                per_cls_gt[gc] = per_cls_gt.get(gc, 0) + 1

        inst_r = total_tp / total_gt if total_gt else 0.0
        img_r = n_img_hit / n_img_gt if n_img_gt else 0.0
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        macro_r = np.mean([per_cls_tp.get(c, 0) / n for c, n in per_cls_gt.items()]) if per_cls_gt else 0.0
        rows.append({
            "conf": conf_thr,
            "instance_recall": inst_r,
            "image_recall": img_r,
            "precision": prec,
            "macro_recall": macro_r,
            "n_tp": total_tp,
            "n_gt": total_gt,
            "n_fp": total_fp,
        })
    return rows


def format_rows(rows):
    lines = []
    lines.append(f"{'conf':>7s} {'inst_R':>7s} {'img_R':>7s} {'macro_R':>7s} {'prec':>7s} {'TP/GT':>10s} {'FP':>6s}")
    lines.append("-" * 60)
    for r in rows:
        lines.append(
            f"{r['conf']:7.3f} {r['instance_recall']:7.4f} {r['image_recall']:7.4f} "
            f"{r['macro_recall']:7.4f} {r['precision']:7.4f} {r['n_tp']:4d}/{r['n_gt']:<4d} {r['n_fp']:6d}"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    data_dir = Path(args.data)
    out_dir = Path(args.output) if args.output else ROOT / "outputs" / "eval_recall_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    models = [YOLO(str(Path(m))) for m in args.models]
    n_models = len(models)

    print(f"Loading GT ({args.split} split)...")
    samples = load_gt(data_dir, args.split)
    print(f"  {len(samples)} images with GT")

    nms = args.nms_iou
    print(f"Gathering predictions (imgsz={args.imgsz}, TTA scales={args.tta_scales}, flip={args.tta_flip}, NMS iou={nms})...")

    report = []
    report.append("=" * 70)
    report.append("INFERENCE-SIDE RECALL LIMIT SWEEP")
    report.append(f"models: {args.models}")
    report.append(f"imgsz={args.imgsz}, TTA scales={args.tta_scales}, flip={args.tta_flip}, NMS iou={nms}")
    report.append(f"images with GT: {len(samples)}, IoU thr: {IOU_THR}")
    report.append("=" * 70)

    # 1. Single model (first model), no TTA
    report.append("\n[1] SINGLE MODEL, NO TTA (conf sweep)")
    rows = sweep(samples, [models[0]], args.imgsz, [1.0], False, nms_iou=nms)
    report.append(format_rows(rows))

    # 2. Single model + class-specific thresholds
    report.append("\n[2] SINGLE MODEL + CLASS-SPECIFIC THRESHOLDS (weak cls conf/10)")
    rows_cs = sweep(samples, [models[0]], args.imgsz, [1.0], False, cls_specific=True, nms_iou=nms)
    report.append(format_rows(rows_cs))

    # 3. Single model + TTA
    report.append(f"\n[3] SINGLE MODEL + TTA (scales={args.tta_scales}, flip={args.tta_flip})")
    rows_tta = sweep(samples, [models[0]], args.imgsz, args.tta_scales, args.tta_flip, nms_iou=nms)
    report.append(format_rows(rows_tta))

    # 4. All models ensemble (no TTA)
    if n_models > 1:
        report.append(f"\n[4] ENSEMBLE ({n_models} models), NO TTA")
        rows_ens = sweep(samples, models, args.imgsz, [1.0], False, nms_iou=nms)
        report.append(format_rows(rows_ens))

        # 5. Ensemble + TTA
        report.append(f"\n[5] ENSEMBLE ({n_models} models) + TTA")
        rows_ens_tta = sweep(samples, models, args.imgsz, args.tta_scales, args.tta_flip, nms_iou=nms)
        report.append(format_rows(rows_ens_tta))

    # Summary of best achievable
    report.append("\n" + "=" * 70)
    report.append("SUMMARY (best instance recall at any conf)")
    report.append("=" * 70)
    bests = []
    for name, r in [("[1] single", rows), ("[2] cls-spec", rows_cs),
                    ("[3] single+TTA", rows_tta),
                    ("[4] ensemble", rows_ens if n_models > 1 else rows),
                    ("[5] ensemble+TTA", rows_ens_tta if n_models > 1 else rows)]:
        best = max(r, key=lambda x: x["instance_recall"])
        bests.append((name, best))
        report.append(f"  {name:16s} conf={best['conf']:.3f} inst_R={best['instance_recall']:.4f} "
                      f"img_R={best['image_recall']:.4f} macro_R={best['macro_recall']:.4f} prec={best['precision']:.4f}")

    # Precision-constrained operating points (scoring items 1+2 must hold together)
    report.append("\n" + "=" * 70)
    report.append("OPERATING POINTS UNDER PRECISION CONSTRAINT (prec >= 0.95, i.e. FP/(TP+FP) <= 5%)")
    report.append("=" * 70)
    for name, r in [("[1] single", rows), ("[2] cls-spec", rows_cs),
                    ("[3] single+TTA", rows_tta),
                    ("[4] ensemble", rows_ens if n_models > 1 else rows),
                    ("[5] ensemble+TTA", rows_ens_tta if n_models > 1 else rows)]:
        ok = [x for x in r if x["precision"] >= 0.95]
        if ok:
            best = max(ok, key=lambda x: x["instance_recall"])
            report.append(f"  {name:16s} conf={best['conf']:.3f} inst_R={best['instance_recall']:.4f} "
                          f"macro_R={best['macro_recall']:.4f} prec={best['precision']:.4f} (FP={best['n_fp']})")
        else:
            report.append(f"  {name:16s} NO operating point with prec >= 0.95 (max prec={max(r, key=lambda x: x['precision'])['precision']:.4f})")

    text = "\n".join(report)
    print("\n" + text)
    (out_dir / "report.txt").write_text(text)
    print(f"\nSaved: {out_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
