#!/usr/bin/env python3
"""Build a patch-based training pool from v1 train (2048x2048 originals).

- Crops 512x512 patches with stride 384 (25% overlap) -> up to 25 patches/image.
- A GT box is kept only if it is FULLY inside the patch (bounding cut boxes are
  dropped; this is what removes large wy etc. from the patch pool).
- Outputs three prioritized pools:
    pool_bdhd.txt  - patches containing bd/heidian boxes (kept always)
    pool_other.txt - patches containing other-class boxes
    pool_bg.txt    - background patches (subsampled by --bg-frac)
  plus full image/label files so any subset can be trained via a .txt list.

Usage:
    uv run python scripts/make_patch_dataset.py \
        --src /home/ancheng/dataset/dataset_split/train \
        --out outputs/dataset_v1_patch --patch 512 --stride 384 --bg-frac 0.2
"""
import argparse
import random
from pathlib import Path

from PIL import Image

ROOT = Path("/home/ancheng/Code/yolov8-dinov3")
NAMES = ["lj", "bd", "heidian", "wy", "zyc", "zmty", "cq", "zj", "bj", "jt",
         "yy", "bmss", "zy", "pd", "HD", "cy", "jiaodai"]
TARGET = {NAMES.index("bd"), NAMES.index("heidian")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/home/ancheng/dataset/dataset_split/train"))
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/dataset_v1_patch")
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--bg-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    img_dir, lbl_dir = args.src / "images", args.src / "labels"
    out_img, out_lbl = args.out / "images", args.out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_txt = sorted(lbl_dir.glob("*.txt"))
    print(f"{len(n_txt)} label files")
    n_bdhd = n_other = n_bg = 0
    n_patch_total = 0
    bdhd_list, other_list, bg_list = [], [], []
    grid = []
    for iy in range(0, 2048 - args.patch + 1, args.stride):
        for ix in range(0, 2048 - args.patch + 1, args.stride):
            grid.append((ix, iy))
    print(f"grid {len(grid)} patches/image")

    for ti, txt in enumerate(n_txt):
        stem = txt.stem
        imgp = img_dir / f"{stem}.png"
        if not imgp.exists():
            for ext in (".jpg", ".jpeg", ".bmp"):
                cand = img_dir / f"{stem}{ext}"
                if cand.exists():
                    imgp = cand
                    break
        im = Image.open(imgp).convert("RGB")
        boxes = []  # (c, x1, y1, x2, y2) in pixels
        for line in txt.read_text().strip().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, x, y, w, h = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
            x1, y1 = (x - w / 2) * 2048, (y - h / 2) * 2048
            x2, y2 = (x + w / 2) * 2048, (y + h / 2) * 2048
            boxes.append((c, x1, y1, x2, y2))
        for gi, (ix, iy) in enumerate(grid):
            keep = []
            has_bdhd = False
            for c, x1, y1, x2, y2 in boxes:
                if x1 >= ix and y1 >= iy and x2 <= ix + args.patch and y2 <= iy + args.patch:
                    nx = ((x1 + x2) / 2 - ix) / args.patch
                    ny = ((y1 + y2) / 2 - iy) / args.patch
                    nw = (x2 - x1) / args.patch
                    nh = (y2 - y1) / args.patch
                    keep.append(f"{c} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
                    if c in TARGET:
                        has_bdhd = True
            patch = im.crop((ix, iy, ix + args.patch, iy + args.patch))
            name = f"{stem}_{gi:02d}.jpg"
            patch.save(out_img / name, quality=92)
            (out_lbl / f"{name.rsplit('.', 1)[0]}.txt").write_text("\n".join(keep) + ("\n" if keep else ""))
            n_patch_total += 1
            if has_bdhd:
                n_bdhd += 1
                bdhd_list.append(str(out_img / name))
            elif keep:
                n_other += 1
                other_list.append(str(out_img / name))
            else:
                n_bg += 1
                bg_list.append(str(out_img / name))
        if (ti + 1) % 200 == 0:
            print(f"  {ti+1}/{len(n_txt)} images, patches: bdhd={n_bdhd} other={n_other} bg={n_bg}", flush=True)

    rng.shuffle(bg_list)
    n_bg_keep = int(n_bg * args.bg_frac)
    bg_keep = bg_list[:n_bg_keep]

    def write_list(path, items):
        Path(path).write_text("\n".join(items) + "\n")

    write_list(args.out / "pool_bdhd.txt", bdhd_list)
    write_list(args.out / "pool_other.txt", other_list)
    write_list(args.out / "pool_bg.txt", bg_keep)

    # full train list = bdhd (x3 oversample) + other + bg
    train_list = bdhd_list * 3 + other_list + bg_keep
    rng.shuffle(train_list)
    write_list(args.out / "train_pool.txt", train_list)

    print(f"\nDONE: total patches {n_patch_total} | bdhd {n_bdhd} | other {n_other} | bg {n_bg} (keep {n_bg_keep})")
    print(f"train pool size: {len(train_list)} (bdhd x3 = {len(bdhd_list)*3} + other {len(other_list)} + bg {n_bg_keep})")


if __name__ == "__main__":
    main()
