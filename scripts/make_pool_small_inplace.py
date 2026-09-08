#!/usr/bin/env python3
"""Build v3-inplace bd/heidian pool: original-size, NO upscaling.

Fix for size-domain mismatch: aug/bd pools scaled bd/heidian instances 3-3.6x
(train w median 0.0195 vs val 0.0063) -> trained on ~20px, evaluated on ~7px.
This pool keeps instances at their ORIGINAL size (w ~0.005-0.01, matching val),
applying only in-place enhancement (contrast/brightness/gray/wavelet) on the
full 2048x2048 image -> bbox coords/scale unchanged.

Sources: v1_train labels (bd=1, heidian=2 instances), 145 images / 596 instances.
Deterministic: fixed seed, labeled img_<stem>_<variant> naming, pool txt output.

Usage:
    uv run python scripts/make_pool_small_inplace.py
Outputs: outputs/dataset_v1_pool_small_inplace/{images,labels,pool_small_inplace.txt}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import random

ROOT = Path("/home/ancheng/Code/yolov8-dinov3")
V1 = Path("/home/ancheng/dataset/dataset_split/train")
OUT = ROOT / "outputs" / "dataset_v1_pool_small_inplace"
RNG = random.Random(7)

VARIANTS = {
    "orig": None,              # 原图（保真基准）
    "c30": "contrast30",       # 对比度 ×1.30（明显但不溢出）
    "clahe": "clahe",          # 灰度 + CLAHE（tile16, clip3 —— 温和局部增强，防纹理二值化）
    "bw": "wavelet",           # 2 层 Haar 软阈值降噪（gain=1.0）
}

# 小波 2 层 Haar 软阈值（与 make_synth_wavelet 同机理，仅灰度通道）
def wavelet_denoise_gray(gray):
    g = np.asarray(gray, dtype=np.float32)
    for _ in range(2):
        n, m = g.shape
        n2, m2 = n // 2 * 2, m // 2 * 2
        g = g[:n2, :m2]
        # 2-level Haar
        a = (g[:n2:2, :m2:2] + g[1::2, :m2:2] + g[:n2:2, 1::2] + g[1::2, 1::2]) / 4
        # 对 subband 软阈值（简化为对细节做 threshold）
        # 这里用简单降噪（盒式均值+细节抑制）——与 064 变体近似
        smooth = np.asarray(Image.fromarray(np.uint8(np.clip(g, 0, 255))).filter(
            __import__("PIL").ImageFilter.MedianFilter(5)), dtype=np.float32)
        g = 0.7 * g + 0.3 * smooth
    return np.uint8(np.clip(g, 0, 255))


def make_variant(img: Image.Image, key: str) -> Image.Image:
    if key == "c30":
        return ImageEnhance.Contrast(img).enhance(1.30)
    if key == "clahe":
        # 温和 CLAHE: tile 16x16 (128px), clip=3.0 —— 局部对比适度增强，避免纹理二值化
        g = np.asarray(img.convert("L"), dtype=np.float32)
        tile = 16
        H, W = g.shape
        th, tw = H // tile, W // tile
        out = np.zeros_like(g)
        clip = 3.0
        for i in range(tile):
            for j in range(tile):
                part = g[i*th:(i+1)*th, j*tw:(j+1)*tw]
                if part.size == 0:
                    continue
                hist, _ = np.histogram(part.ravel(), bins=256, range=(0, 256))
                avg = part.size / 256.0
                hist = np.minimum(hist, clip * avg)  # clip=3*avg
                cdf = hist.cumsum()
                cdf = (cdf - cdf.min()) * 255.0 / (cdf.max() - cdf.min() + 1e-8)
                lk = np.interp(part, np.arange(256), cdf).reshape(part.shape)
                # blend 70% CLAHE + 30% 原始（防过强）
                out[i*th:(i+1)*th, j*tw:(j+1)*tw] = 0.7 * lk + 0.3 * part
        return Image.fromarray(np.uint8(np.clip(out, 0, 255))).convert("RGB")
    if key == "bw":
        # 真 2 层 Haar 软阈值（soft-threshold = mean*1.5 on details）；保持原始尺寸
        g = np.asarray(img.convert("L"), dtype=np.float32)
        H, W = g.shape
        for _ in range(2):
            n, m = g.shape
            n2, m2 = n // 2 * 2, m // 2 * 2
            g = g[:n2, :m2]
            a = (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2]) / 4
            h = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 4
            v = (g[0::2, 0::2] - g[1::2, 0::2] + g[0::2, 1::2] - g[1::2, 1::2]) / 4
            thr = np.abs(h).mean() * 1.5
            h_s = np.sign(h) * np.maximum(np.abs(h) - thr, 0)
            thr2 = np.abs(v).mean() * 1.5
            v_s = np.sign(v) * np.maximum(np.abs(v) - thr2, 0)
            g = a + h_s + v_s
        # 上采样回原始尺寸（最近邻）——仅 bw 需还原（Haar 每次降 2 倍）
        g = np.asarray(Image.fromarray(np.uint8(np.clip(g, 0, 255))).resize((W, H), Image.NEAREST), dtype=np.float32)
        return Image.fromarray(np.uint8(np.clip(g, 0, 255))).convert("RGB")
    if key == "cs":
        g = np.asarray(img.convert("L"), dtype=np.float32)
        lo, hi = np.percentile(g, 2), np.percentile(g, 98)
        g = (g - lo) * 255.0 / (hi - lo + 1e-6)
        return Image.fromarray(np.uint8(np.clip(g, 0, 255))).convert("RGB")
    return img


def main() -> None:
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)

    # 收集含 bd/heidian 的源图
    inst_srcs = []  # (stem, img_path, label_path)
    for lbl in sorted((V1 / "labels").glob("*.txt")):
        has_bd = False
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 1 and int(float(p[0])) in (1, 2):
                has_bd = True
                break
        if has_bd:
            stem = lbl.stem
            img = V1 / "images" / f"{stem}.png"
            if img.exists():
                inst_srcs.append((stem, img, lbl))
    print(f"源图 {len(inst_srcs)} 张")

    lines = []
    total = 0
    for stem, img_path, lbl_path in inst_srcs:
        img = Image.open(img_path).convert("RGB")
        for vname in VARIANTS:
            out_name = f"ins_{stem}_{vname}.jpg"
            make_variant(img, vname).save(OUT / "images" / out_name, quality=92)
            lbl = lbl_path.read_text()
            (OUT / "labels" / (out_name.rsplit(".", 1)[0] + ".txt")).write_text(lbl)
            lines.append(str(OUT / "images" / out_name))
            total += 1
    (OUT / "pool_small_inplace.txt").write_text("\n".join(lines) + "\n")
    print(f"产物 {total} 张 (145 源图 x {len(VARIANTS)} 变体)")


if __name__ == "__main__":
    main()
