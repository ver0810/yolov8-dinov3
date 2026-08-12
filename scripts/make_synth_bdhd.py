#!/usr/bin/env python3
"""Build synthetic bd/heidian patch pool (062b).

For every bd/heidian instance in the 061 patch pool: crop the defect (with margin),
generate 4 variants (brightness ±15%, contrast ×0.8–1.3, rotation ±15°, scale 0.8–1.2),
paste onto a random 512x512 background window cropped from v1 train images
(original labels inside the window kept; synthetic defect placed with IoU<0.3
against kept boxes; boxes outside window dropped like the instance-center pool).

Output: outputs/dataset_v1_patch_bdhd_synth/{images,labels} + pool_synth.txt
"""
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageOps

ROOT = Path("/home/ancheng/Code/yolov8-dinov3")
SRC = Path("/home/ancheng/dataset/dataset_split/train")
POOL = ROOT / "outputs/dataset_v1_patch_bdhd"
OUT = ROOT / "outputs/dataset_v1_patch_bdhd_synth"
PATCH = 512
IMG = 2048
NAMES = ["lj", "bd", "heidian", "wy", "zyc", "zmty", "cq", "zj", "bj", "jt",
         "yy", "bmss", "zy", "pd", "HD", "cy", "jiaodai"]
TARGET = {1, 2}
VARIANTS = 4

out_img = OUT / "images"; out_img.mkdir(parents=True, exist_ok=True)
out_lbl = OUT / "labels"; out_lbl.mkdir(parents=True, exist_ok=True)
rng = random.Random(11)

# collect target instances from ORIGINAL v1 train labels (unique), pixels in 2048 img
instances = []  # (class, src_image_path, x1, y1, x2, y2)
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
    """Random 512 window from a random train image + labels fully inside."""
    p = rng.choice(train_imgs)
    im = Image.open(p).convert("RGB")
    x0 = rng.randint(0, IMG - PATCH)
    y0 = rng.randint(0, IMG - PATCH)
    return im, x0, y0, p.stem

def keep_boxes(txt_stem, x0, y0):
    """Labels fully inside window, in patch-normalized coords."""
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

n = 0
for ci, (c, imgp, x1, y1, x2, y2) in enumerate(instances):
    src = Image.open(imgp).convert("RGB")
    # crop defect with margin (target is tiny; margin 16px)
    m = 16
    cx0, cy0 = max(0, int(x1) - m), max(0, int(y1) - m)
    cx1, cy1 = min(IMG, int(x2) + m), min(IMG, int(y2) + m)
    defect = src.crop((cx0, cy0, cx1, cy1))
    for v in range(VARIANTS):
        dv = defect
        if rng.random() < 0.8:
            dv = ImageEnhance.Brightness(dv).enhance(rng.uniform(0.85, 1.15))
        if rng.random() < 0.8:
            dv = ImageEnhance.Contrast(dv).enhance(rng.uniform(0.8, 1.3))
        if rng.random() < 0.5:
            dv = dv.rotate(rng.uniform(-15, 15), expand=True, fillcolor=255)
        sc = rng.uniform(0.8, 1.2)
        dw, dh = int(dv.width * sc), int(dv.height * sc)
        dv = dv.resize((max(4, dw), max(4, dh)))
        # background
        bg, bx0, by0, stem = rand_background()
        keep = keep_boxes(stem, bx0, by0)
        # random paste position not colliding
        for _ in range(40):
            px = rng.randint(0, PATCH - dv.width)
            py = rng.randint(0, PATCH - dv.height)
            new_box = (px, py, px + dv.width, py + dv.height)
            if all(iou(new_box, kb) < 0.3 for kb in keep):
                break
        patch = bg.crop((bx0, by0, bx0 + PATCH, by0 + PATCH))
        patch.paste(dv, (px, py))
        # label: original kept + synthetic
        allb = keep + [(c, new_box[0], new_box[1], new_box[2], new_box[3])]
        name = f"syn_{ci:04d}_{v}.jpg"
        patch.save(out_img / name, quality=92)
        lines = [f"{cb} {((a+d)/2)/PATCH:.6f} {((b+e)/2)/PATCH:.6f} {(d-a)/PATCH:.6f} {(e-b)/PATCH:.6f}"
                 for cb, a, b, d, e in allb]
        (out_lbl / f"{name.rsplit('.',1)[0]}.txt").write_text("\n".join(lines) + "\n")
        n += 1
    if (ci + 1) % 150 == 0:
        print(f"  {ci+1}/{len(instances)} instances, {n} synth patches", flush=True)

pool = [str(out_img / f"syn_{i:04d}_{v}.jpg") for i in range(len(instances)) for v in range(VARIANTS)]
(OUT / "pool_synth.txt").write_text("\n".join(pool) + "\n")
print(f"DONE: {n} synthetic patches -> {OUT}")
