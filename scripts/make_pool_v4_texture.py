"""v4 纹理池: 含 bmss 实例的 v1-train 图 -> LAB-L CLAHE 变体(保色).

按类策略(来自 texture probe 结论):
  bmss -> CLAHE(clip 2.0, tile 8x8, 与探针完全一致的参数)
  wy / zmty -> 无变体(探针显示 CLAHE 伤害 wy, zmty 中性)
确定性: 文件排序, 无随机数, 标签逐字节复制(纯像素变换, 几何不动).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path("/home/ancheng/Code/yolov8-dinov3")
TRAIN_IMG = Path("/home/ancheng/dataset/dataset_split/train/images")
TRAIN_LBL = Path("/home/ancheng/dataset/dataset_split/train/labels")
OUT = ROOT / "outputs/dataset_v1_pool_v4"
SUFFIX = "_v4clahe"
IMG_EXTS = (".png", ".jpg", ".jpeg")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    # 1. 选源: 含 >=1 个 bmss(11) 的 v1 train 图, 排序保证确定性
    srcs = []
    for lf in sorted(TRAIN_LBL.glob("*.txt")):
        n_bmss = 0
        for ln in lf.read_text().splitlines():
            t = ln.split()
            if t and int(float(t[0])) == 11:
                n_bmss += 1
        if not n_bmss:
            continue
        cand = [p for p in TRAIN_IMG.glob(lf.stem + ".*") if p.suffix.lower() in IMG_EXTS]
        assert len(cand) == 1, f"图歧义或缺失: {lf.stem} -> {cand}"
        srcs.append((cand[0], lf, n_bmss))
    print(f"源图: {len(srcs)} 张, bmss实例: {sum(s[2] for s in srcs)} 个")

    # 2. 变换: 与探针完全一致的参数
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    manifest = []
    for img_p, lbl_p, n_bmss in srcs:
        img = Image.open(img_p).convert("RGB")
        lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)
        lch, ach, bch = cv2.split(lab)
        enh = Image.fromarray(cv2.cvtColor(cv2.merge([clahe.apply(lch), ach, bch]), cv2.COLOR_LAB2RGB))
        assert enh.size == img.size, f"尺寸变化: {img_p.name}"
        dst_img = OUT / "images" / (img_p.stem + SUFFIX + img_p.suffix)
        enh.save(dst_img)
        dst_lbl = OUT / "labels" / (lbl_p.stem + SUFFIX + ".txt")
        dst_lbl.write_bytes(lbl_p.read_bytes())  # 标签逐字节复制
        manifest.append({
            "src_img": str(img_p), "src_lbl": str(lbl_p),
            "dst_img": str(dst_img), "dst_lbl": str(dst_lbl),
            "n_bmss": n_bmss, "size": list(img.size),
            "sha256_src_img": sha256(img_p), "sha256_dst_img": sha256(dst_img),
            "sha256_label": sha256(dst_lbl),
        })
    (OUT / "pool_v4_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"池构建完毕: {len(manifest)} 张 -> {OUT}")
    print(f"POLICY=bmss-only CLAHE(clip2.0,tile8) color-preserved; wy/zmty=none")


if __name__ == "__main__":
    main()
