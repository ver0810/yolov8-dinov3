#!/usr/bin/env python3
"""Instance-centered patch dataset for bd/heidian ONLY (size-split strategy).

Full training images stay whole (1445 imgs, mid/large defects untouched).
For every bd/heidian GT instance (classes 1/2), crop a 512x512 window centered
on the box (clamped into image bounds). Every label FULLY inside a window is
kept (bd/heidian always complete; other classes kept when coincidentally whole).
Writes images/*.jpg, labels/*.txt, visuals/*.png (annotated) + pool_bdhd.txt.
"""
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/home/ancheng/Code/yolov8-dinov3")
SRC = Path("/home/ancheng/dataset/dataset_split/train")
OUT = ROOT / "outputs/dataset_v1_patch_bdhd"
PATCH = 512
IMG = 2048
NAMES = ["lj", "bd", "heidian", "wy", "zyc", "zmty", "cq", "zj", "bj", "jt",
         "yy", "bmss", "zy", "pd", "HD", "cy", "jiaodai"]
TARGET = {1, 2}  # bd, heidian
COLORS = {1: (255, 0, 0), 2: (0, 0, 255)}

out_img = OUT / "images"; out_img.mkdir(parents=True, exist_ok=True)
out_lbl = OUT / "labels"; out_lbl.mkdir(parents=True, exist_ok=True)
out_vis = OUT / "visuals"; out_vis.mkdir(parents=True, exist_ok=True)

lbls = sorted((SRC / "labels").glob("*.txt"))
n_inst = defaultdict(int)
n_patches = 0
n_patch_target = 0
n_other_in_patch = 0
pool = []

print(f"{len(lbls)} label files; instance-centered {PATCH}px windows (bd/heidian only)")

for ti, txt in enumerate(lbls):
    stem = txt.stem
    im = Image.open(SRC / "images" / f"{stem}.png").convert("RGB")
    boxes = []
    for line in txt.read_text().strip().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c, x, y, w, h = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
        x1, y1 = (x - w / 2) * IMG, (y - h / 2) * IMG
        x2, y2 = (x + w / 2) * IMG, (y + h / 2) * IMG
        boxes.append((c, x1, y1, x2, y2))
        if c in TARGET:
            n_inst[c] += 1

    # collect unique windows (centered on each bd/heidian instance, clamped)
    wins = {}
    for c, x1, y1, x2, y2 in boxes:
        if c not in TARGET:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x0 = int(min(max(cx - PATCH / 2, 0), IMG - PATCH))
        y0 = int(min(max(cy - PATCH / 2, 0), IMG - PATCH))
        wins.setdefault((x0, y0), []).append((c, x1, y1, x2, y2))

    for (x0, y0), tgt in wins.items():
        name = f"{stem}_{x0:04d}_{y0:04d}.jpg"
        patch = im.crop((x0, y0, x0 + PATCH, y0 + PATCH))
        patch.save(out_img / name, quality=92)
        # keep labels fully inside window
        keep = []
        for c, x1, y1, x2, y2 in boxes:
            if x1 >= x0 and y1 >= y0 and x2 <= x0 + PATCH and y2 <= y0 + PATCH:
                nx = ((x1 + x2) / 2 - x0) / PATCH
                ny = ((y1 + y2) / 2 - y0) / PATCH
                nw = (x2 - x1) / PATCH
                nh = (y2 - y1) / PATCH
                keep.append((c, nx, ny, nw, nh))
        stem_name = name.rsplit(".", 1)[0]
        (out_lbl / f"{stem_name}.txt").write_text(
            "\n".join(f"{c} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}" for c, nx, ny, nw, nh in keep) + "\n")
        # annotated visual
        vis = patch.copy()
        d = ImageDraw.Draw(vis)
        for c, x1, y1, x2, y2 in boxes:
            if x1 >= x0 and y1 >= y0 and x2 <= x0 + PATCH and y2 <= y0 + PATCH:
                d.rectangle([x1 - x0, y1 - y0, x2 - x0, y2 - y0], outline=COLORS.get(c, (0, 200, 0)), width=3)
                d.text((x1 - x0 + 3, max(y1 - y0 - 13, 0)), NAMES[c], fill=COLORS.get(c, (0, 200, 0)))
        vis.save(out_vis / f"{stem_name}.png")
        n_patches += 1
        pool.append(str(out_img / name))
        n_patch_target += len([1 for c, *_ in keep if c in TARGET])
        n_other_in_patch += len([1 for c, *_ in keep if c not in TARGET])

    if (ti + 1) % 300 == 0:
        print(f"  {ti+1}/{len(lbls)} imgs, patches={n_patches}", flush=True)

(OUT / "pool_bdhd.txt").write_text("\n".join(pool) + "\n")
print(f"\nDONE: patches={n_patches}")
print(f"  bd instances={n_inst[1]}  heidian instances={n_inst[2]}  (window dedup -> {n_patches} patches)")
print(f"  patches containing a target instance (label-assigned): {n_patch_target}")
print(f"  other-class labels also kept inside patches: {n_other_in_patch}")
print(f"  visuals: {out_vis}")
