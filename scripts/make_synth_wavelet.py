#!/usr/bin/env python3
"""064 wavelet-enhanced synthetic pool for bd/heidian.

Extends 063's synth pool with wavelet-enhanced defect variants:
per instance: 4 regular variants (brightness/contrast/rotation/scale) +
              2 wavelet variants (2-level Haar, detail soft-threshold + gain 2.0 / 3.0).
Wavelet enhancement is applied to the defect crop BEFORE pasting onto a real
background window (background stays real, no texture amplification).

Output: outputs/dataset_v1_patch_wavelet/{images,labels} + pool_wavelet.txt
"""
import random
import sys
from pathlib import Path

import numpy as np
import pywt
from PIL import Image, ImageEnhance

ROOT = Path("/home/ancheng/Code/yolov8-dinov3")
SRC = Path("/home/ancheng/dataset/dataset_split/train")
OUT = ROOT / "outputs/dataset_v1_patch_wavelet"
PATCH = 512
IMG = 2048
NAMES = ["lj", "bd", "heidian", "wy", "zyc", "zmty", "cq", "zj", "bj", "jt",
         "yy", "bmss", "zy", "pd", "HD", "cy", "jiaodai"]
TARGET = {1, 2}
REGULAR_VARIANTS = 4   # as in 063
# wavelet variant: soft-threshold DENOISE only (gain=1.0). Grid test on real data:
# SNR 1.54->3.33 (2.16x), zero overshoot, dot drift <0.5 gray. gain>1 adds ringing risk.
WAVELET_GAINS = (1.0,)

out_img = OUT / "images"; out_img.mkdir(parents=True, exist_ok=True)
out_lbl = OUT / "labels"; out_lbl.mkdir(parents=True, exist_ok=True)
rng = random.Random(11)

# ---- collect unique bd/heidian instances from ORIGINAL v1 train labels ----
instances = []  # (class, src_image_path, x1, y1, x2, y2) pixels in 2048 img
for t in sorted((SRC / "labels").glob("*.txt")):
    stem = t.stem
    for p in t.read_text().strip().splitlines():
        p = p.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        if c not in TARGET:
            continue
        x, y, w, h = float(p[1]) * IMG, float(p[2]) * IMG, float(p[3]) * IMG, float(p[4]) * IMG
        instances.append((c, SRC / "images" / f"{stem}.png", x - w / 2, y - h / 2, x + w / 2, y + h / 2))
print(f"target instances (unique): {len(instances)}")

train_imgs = sorted((SRC / "images").glob("*.png"))


def rand_background():
    p = rng.choice(train_imgs)
    im = Image.open(p).convert("RGB")
    x0 = rng.randint(0, IMG - PATCH)
    y0 = rng.randint(0, IMG - PATCH)
    return im, x0, y0, p.stem


def keep_boxes(txt_stem, x0, y0):
    keep = []
    txt = SRC / "labels" / f"{txt_stem}.txt"
    if txt.exists():
        for line in txt.read_text().strip().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, x, y, w, h = int(p[0]), float(p[1]) * IMG, float(p[2]) * IMG, float(p[3]) * IMG, float(p[4]) * IMG
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            if x1 >= x0 and y1 >= y0 and x2 <= x0 + PATCH and y2 <= y0 + PATCH:
                keep.append((c, x1 - x0, y1 - y0, x2 - x0, y2 - y0))
    return keep


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


def wavelet_enhance_gray(gray, gain):
    """2-level Haar on grayscale, soft-threshold details (mean*1.5) then scale by `gain`."""
    coeffs = pywt.wavedec2(gray, "haar", level=2)
    for detail in coeffs[1:]:
        for arr in detail:
            th = float(np.abs(arr).mean()) * 1.5
            arr[...] = np.sign(arr) * np.maximum(np.abs(arr) - th, 0) * gain
    recon = pywt.waverec2(coeffs, "haar")
    return recon[: gray.shape[0], : gray.shape[1]]


def paste_defect(patch, defect, px, py):
    patch.paste(defect, (px, py))


n = 0
for ci, (c, imgp, x1, y1, x2, y2) in enumerate(instances):
    src = Image.open(imgp).convert("RGB")
    m = 16
    cx0, cy0 = max(0, int(x1) - m), max(0, int(y1) - m)
    cx1, cy1 = min(IMG, int(x2) + m), min(IMG, int(y2) + m)
    defect = src.crop((cx0, cy0, cx1, cy1))

    variants = []
    # 4 regular variants (same recipe as 063)
    for _ in range(REGULAR_VARIANTS):
        dv = defect
        if rng.random() < 0.8:
            dv = ImageEnhance.Brightness(dv).enhance(rng.uniform(0.85, 1.15))
        if rng.random() < 0.8:
            dv = ImageEnhance.Contrast(dv).enhance(rng.uniform(0.8, 1.3))
        if rng.random() < 0.5:
            dv = dv.rotate(rng.uniform(-15, 15), expand=True, fillcolor=255)
        sc = rng.uniform(0.8, 1.2)
        dv = dv.resize((max(4, int(dv.width * sc)), max(4, int(dv.height * sc))))
        variants.append(dv)
    # 2 wavelet variants (grayscale enhancement, RGB channels set equal)
    for gain in WAVELET_GAINS:
        gray = np.asarray(defect.convert("L")).astype(float)
        enh = wavelet_enhance_gray(gray, gain)
        enh = np.clip(enh, 0, 255).astype(np.uint8)
        dv = Image.fromarray(enh).convert("RGB").resize(
            (max(4, int(defect.width * 0.9)), max(4, int(defect.height * 0.9))))
        variants.append(dv)

    for v, dv in enumerate(variants):
        bg, bx0, by0, stem = rand_background()
        keep = keep_boxes(stem, bx0, by0)
        for _ in range(40):
            px = rng.randint(0, PATCH - dv.width)
            py = rng.randint(0, PATCH - dv.height)
            new_box = (px, py, px + dv.width, py + dv.height)
            if all(iou(new_box, kb) < 0.3 for kb in keep):
                break
        patch = bg.crop((bx0, by0, bx0 + PATCH, by0 + PATCH))
        paste_defect(patch, dv, px, py)
        allb = keep + [(c, px, py, px + dv.width, py + dv.height)]
        name = f"wav_{ci:04d}_{v}.jpg"
        patch.save(out_img / name, quality=92)
        lines = [f"{cb} {((a + d) / 2) / PATCH:.6f} {((b + e) / 2) / PATCH:.6f} {(d - a) / PATCH:.6f} {(e - b) / PATCH:.6f}"
                 for cb, a, b, d, e in allb]
        (out_lbl / f"{name.rsplit('.', 1)[0]}.txt").write_text("\n".join(lines) + "\n")
        n += 1
    if (ci + 1) % 150 == 0:
        print(f"  {ci+1}/{len(instances)} instances, {n} patches", flush=True)

pool = [str(out_img / f"wav_{i:04d}_{v}.jpg") for i in range(len(instances)) for v in range(len(variants))]
(OUT / "pool_wavelet.txt").write_text("\n".join(pool) + "\n")
print(f"DONE: {n} wavelet-synth patches -> {OUT}")