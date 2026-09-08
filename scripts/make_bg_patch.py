#!/usr/bin/env python3
"""Build a background-patch pool from v1 train (2048 originals).

Random 512x512 windows with ZERO overlap with any GT box (IoU==0, so no
cut-defect appears in the patch). Background patches carry empty labels:
they teach the model "this texture is NOT a defect" without touching any
positive-sample gradient.

Output: outputs/dataset_v1_bg_patch/{images,labels} + pool_bg.txt + count stats.
"""
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path("/home/ancheng/Code/yolov8-dinov3")
SRC = Path("/home/ancheng/dataset/dataset_split/train")
OUT = ROOT / "outputs/dataset_v1_bg_patch"
PATCH = 512
IMG = 2048

out_img = OUT / "images"; out_img.mkdir(parents=True, exist_ok=True)
out_lbl = OUT / "labels"; out_lbl.mkdir(parents=True, exist_ok=True)
rng = random.Random(21)

train_imgs = sorted((SRC / "images").glob("*.png"))

def boxes_of(stem):
    boxes = []
    txt = SRC / "labels" / f"{stem}.txt"
    if txt.exists():
        for line in txt.read_text().strip().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            x, y, w, h = float(p[1]) * IMG, float(p[2]) * IMG, float(p[3]) * IMG, float(p[4]) * IMG
            boxes.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return boxes

def overlap_zero(w, boxes):
    wx1, wy1, wx2, wy2 = w
    for x1, y1, x2, y2 in boxes:
        ix1, iy1 = max(wx1, x1), max(wy1, y1)
        ix2, iy2 = min(wx2, x2), min(wy2, y2)
        if ix1 < ix2 and iy1 < iy2:   # any intersection -> reject
            return False
    return True

n_total = n_bg = 0
pool = []
for ti, p in enumerate(train_imgs):
    stem = p.stem
    boxes = boxes_of(stem)
    im = None
    for try_i in range(30):
        x0 = rng.randint(0, IMG - PATCH)
        y0 = rng.randint(0, IMG - PATCH)
        w = (x0, y0, x0 + PATCH, y0 + PATCH)
        n_total += 1
        if not overlap_zero(w, boxes):
            continue
        if im is None:
            im = Image.open(p).convert("RGB")
        name = f"bg_{stem}_{try_i:02d}{x0:04d}{y0:04d}.jpg"
        patch = im.crop((x0, y0, x0 + PATCH, y0 + PATCH))
        patch.save(out_img / name, quality=92)
        (out_lbl / f"{name.rsplit('.', 1)[0]}.txt").write_text("")  # empty label
        pool.append(str(out_img / name))
        n_bg += 1
        if n_bg >= 2000:   # cap pool
            break
    if n_bg >= 2000:
        break
    if (ti + 1) % 200 == 0:
        print(f"  {ti+1} imgs, bg patches={n_bg}", flush=True)

(OUT / "pool_bg.txt").write_text("\n".join(pool) + "\n")
print(f"DONE: {n_bg} bg patches (zero GT overlap) -> {OUT}")