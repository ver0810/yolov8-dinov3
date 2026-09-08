"""Manual conf=0.25 macro R/P: independent matcher (no ultralytics Metrics semantics).
GT: /home/ancheng/dataset/dataset_split/val/labels/*.txt (v1 val, 17 classes, 0-indexed)
Usage: uv run python /tmp/r25_manual.py <weights> [imgsz]
"""
import sys, glob, os
import numpy as np
import torch
from ultralytics import YOLO
def xywh2xyxy_np(b):
    x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    hw, hh = w / 2, h / 2
    return np.stack([x - hw, y - hh, x + hw, y + hh], axis=1)

def iou_np(a, b):
    """(1,4) vs (1,4) normalized xyxy -> iou"""
    ax1, ay1, ax2, ay2 = a[0]
    bx1, by1, bx2, by2 = b[0]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0

w = sys.argv[1]
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
YAML = "configs/own_dataset.yaml"

m = YOLO(w)
# ground truth boxes
gt = {}  # stem -> (cls_xywh list)
label_dir = "/home/ancheng/dataset/dataset_split/val/labels"
files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
for f in files:
    stem = os.path.basename(f)[:-4]
    boxes = []
    for line in open(f):
        t = line.split()
        if len(t) < 5:
            continue
        c, x, y, wd, h = (float(v) for v in t[:5])
        boxes.append((int(c), x, y, wd, h))
    gt[stem] = boxes

# predictions
preds = m.predict(source="/home/ancheng/dataset/dataset_split/val/images",
                  imgsz=imgsz, conf=0.001, iou=0.5, device=0, verbose=False,
                  save=False, max_det=300)
g_count = np.zeros(17)
correct = np.zeros(17)
pred_count = np.zeros(17)
matching = {"pred": 0}

for pred in preds:
    stem = os.path.basename(pred.path)[:-4]
    g = gt.get(stem, [])
    gxywh = np.array([[x, y, wd, h] for (c, x, y, wd, h) in g], dtype=np.float32).reshape(-1, 4) if g else np.zeros((0, 4))
    gcls = np.array([c for (c, x, y, wd, h) in g], dtype=int) if g else np.zeros((0,), dtype=int)
    if len(g):
        gbox_n = np.clip(xywh2xyxy_np(gxywh), 0, 1)  # normalized xyxy
    else:
        gbox_n = np.zeros((0, 4))
    for c in gcls:
        g_count[c] += 1
    # filter preds by conf >= 0.25 (predict was run at 0.001 for full set, so filter here)
    if pred.boxes is None or len(pred.boxes) == 0:
        continue
    pcls = pred.boxes.cls.cpu().numpy().astype(int)
    pconf = pred.boxes.conf.cpu().numpy()
    pbox = pred.boxes.xyxy.cpu().numpy()
    oh, ow = pred.orig_shape if hasattr(pred, "orig_shape") else (imgsz, imgsz)
    pbox_x = pbox / np.array([ow, oh, ow, oh])  # normalize to original image coords
    keep = pconf >= 0.25
    pcls, pconf, pbox_x = pcls[keep], pconf[keep], pbox_x[keep]
    for i in range(len(pcls)):
        matching["pred"] += 1
        pred_count[pcls[i]] += 1
    if len(pcls) == 0 or len(gbox_n) == 0:
        continue
    # match: for each gt, best pred same-class IoU>=0.5
    matched = set()
    for gi in range(len(gbox_n)):
        c = gcls[gi]
        best_iou, best_idx = 0.0, -1
        for pi in range(len(pcls)):
            if pcls[pi] != c or pi in matched:
                continue
            iou = iou_np(gbox_n[gi:gi+1], pbox_x[pi:pi+1])
            if iou > best_iou:
                best_iou, best_idx = iou, pi
        if best_iou >= 0.5:
            matched.add(best_idx)
            correct[c] += 1

R = correct / np.maximum(g_count, 1)
P = correct / np.maximum(pred_count, 1)
macroR = R.mean()
macroP = P.mean() if matching["pred"] else 0.0
f1 = 2 * macroP * macroR / (macroP + macroR) if macroP + macroR > 0 else 0.0
print(f"WEIGHTS={w}")
print(f"TP={int(correct.sum())} / GT={int(g_count.sum())} / PRED(filt)={matching['pred']}")
print(f"macro R@0.25={macroR:.4f}  macro P@0.25={macroP:.4f}  F1@0.25={f1:.4f}")
for c in range(17):
    print(f"  cls{c}: R={R[c]:.4f} P={P[c]:.4f} gt={int(g_count[c])} tp={int(correct[c])}")
