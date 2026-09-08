#!/usr/bin/env python3
"""Extract 097 per-class mAP and per-size analysis at best point conf=0.548"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "ultralytics"))

from ultralytics import YOLO

weights = "/tmp/k097/runs/rtdetr_l_097/weights/best.pt"
data = "configs/own_dataset.yaml"
imgsz = 1280

print("Loading 097...")
model = YOLO(weights)
print("Running val conf=0.001 full curves...")
metrics = model.val(data=data, imgsz=imgsz, batch=8, device=0, split="val", conf=0.001, verbose=False, plots=False)

box = metrics.box
names = metrics.names
nt = metrics.nt_per_class  # per class GT count
ap_class_index = box.ap_class_index  # class ids present
# box.all_ap shape: (nc_present, 10) AP at IoU 0.5:0.95
# box.ap50 is AP@0.5? In ultralytics, box.maps? Let's inspect
print(f"names: {names}")
print(f"ap_class_index: {ap_class_index}")
print(f"nt_per_class: {nt}")
print(f"box.all_ap shape: {box.all_ap.shape if hasattr(box.all_ap, 'shape') else len(box.all_ap)}")
# Try to get AP50 and mAP
# In ultralytics 8.4, box.ap is mAP50-95 per class, box.ap50 is AP50?
# Check attributes
print(f"Has ap: {hasattr(box, 'ap')}, ap50: {hasattr(box, 'ap50')}, maps: {hasattr(box, 'maps')}, map: {hasattr(box, 'map')}, map50: {hasattr(box, 'map50')}")
if hasattr(box, 'ap'):
    print(f"box.ap: {box.ap}")
if hasattr(box, 'ap50'):
    print(f"box.ap50: {box.ap50}")
if hasattr(box, 'maps'):
    print(f"box.maps: {box.maps}")
if hasattr(box, 'map'):
    print(f"box.map: {box.map}")
if hasattr(box, 'map50'):
    print(f"box.map50: {box.map50}")
print(f"box.p: {box.p[:3] if hasattr(box.p, '__len__') else box.p}")
print(f"box.r: {box.r[:3] if hasattr(box.r, '__len__') else box.r}")
print(f"box.f1: {box.f1[:3] if hasattr(box.f1, '__len__') else box.f1}")
print(f"box.px shape: {box.px.shape}")
print(f"p_curve shape: {box.p_curve.shape}, r_curve shape: {box.r_curve.shape}")

# Compute macro curves at best-F1 point already known: conf 0.548
px = box.px
p_curve = box.p_curve
r_curve = box.r_curve
f1_curve = box.f1_curve
# macro
active = [ci for ci in range(len(names)) if nt[ci] > 0]
# Find best-F1 global (macro mean)
f1s = [2 * p_curve[i] * r_curve[i] / (p_curve[i] + r_curve[i] + 1e-16) for i in range(len(ap_class_index))]
f1_macro = np.mean(f1s, axis=0)
fidx = int(np.argmax(f1_macro))
print(f"\nGlobal best-F1 @conf={px[fidx]:.3f}: R macro={np.mean([r_curve[i][fidx] for i in range(len(ap_class_index))]):.4f} P macro={np.mean([p_curve[i][fidx] for i in range(len(ap_class_index))]):.4f} F1={f1_macro[fidx]:.4f}")

# Per-class AP details
print("\n=== PER-CLASS mAP ===")
for i, ci in enumerate(ap_class_index):
    cname = names[ci]
    n_gt = int(nt[ci])
    # AP50 is all_ap[i,0], mAP 0.5:0.95 is mean over 10
    ap50 = float(box.all_ap[i, 0]) if hasattr(box.all_ap, '__getitem__') else 0
    mAP = float(box.all_ap[i].mean()) if hasattr(box.all_ap, '__getitem__') else 0
    # best F1 for this class
    f1_c = 2 * p_curve[i] * r_curve[i] / (p_curve[i] + r_curve[i] + 1e-16)
    bf_idx = int(np.argmax(f1_c))
    print(f"{cname:>12s} n={n_gt:3d}  AP50={ap50:.3f}  mAP={mAP:.3f}  bestF1={f1_c[bf_idx]:.3f}@{px[bf_idx]:.3f}  R@best={r_curve[i][bf_idx]:.3f} P@best={p_curve[i][bf_idx]:.3f}  maxR={r_curve[i].max():.3f}  R@0.25={r_curve[i][np.argmin(np.abs(px-0.25))]:.3f} P@0.25={p_curve[i][np.argmin(np.abs(px-0.25))]:.3f}")

# Also save curves for later use
import json, pickle
out = Path("outputs/eval_scoring/097_best_analysis")
out.mkdir(parents=True, exist_ok=True)
import csv
with open(out / "perclass_map.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class_id","class_name","n_gt","AP50","mAP","bestF1","best_conf","R_best","P_best","maxR","R_025","P_025"])
    for i, ci in enumerate(ap_class_index):
        cname = names[ci]
        n_gt = int(nt[ci])
        ap50 = float(box.all_ap[i, 0])
        mAP = float(box.all_ap[i].mean())
        f1_c = 2 * p_curve[i] * r_curve[i] / (p_curve[i] + r_curve[i] + 1e-16)
        bf_idx = int(np.argmax(f1_c))
        w.writerow([ci,cname,n_gt,f"{ap50:.4f}",f"{mAP:.4f}",f"{f1_c[bf_idx]:.4f}",f"{px[bf_idx]:.4f}",f"{r_curve[i][bf_idx]:.4f}",f"{p_curve[i][bf_idx]:.4f}",f"{r_curve[i].max():.4f}",f"{r_curve[i][np.argmin(np.abs(px-0.25))]:.4f}",f"{p_curve[i][np.argmin(np.abs(px-0.25))]:.4f}"])

# Overall
print(f"\nSaved to {out/'perclass_map.csv'}")
# Mean AP
print(f"Overall mAP50: {box.all_ap[:,0].mean():.4f}  mAP50-95: {box.all_ap.mean():.4f}")
