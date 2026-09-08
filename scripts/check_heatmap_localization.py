"""Quantify how well DINOv3 heatmaps localize the weak-class GT targets.

For each image: compute ViT CLS-attention and convnext feature heatmaps, take the
top-k activation centroid, and measure distance to the GT target centers.
Outputs a per-class comparison table.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "ultralytics"))

IMGDIR = Path("/home/ancheng/dataset/dataset_split/val/images")
LABDIR = Path("/home/ancheng/dataset/dataset_split/val/labels")
IMG_SIZE = 448

# name -> (image stem, class index)
IMGS = {
    "bd": ("LZW_ZQ_20260124_266", 1),
    "heidian": ("113__0921", 2),
    "wy": ("XBW_20250804_1593", 3),
    "bmss": ("LZW_20260124_362", 9),
}


def gt_centers(lab_path, cls):
    cs = []
    for line in open(lab_path):
        p = line.split()
        if int(p[0]) == cls:
            cs.append((float(p[1]), float(p[2])))  # cx, cy (relative)
    return cs


def load(img_t):
    from PIL import Image
    im = Image.open(IMGDIR / "images" / f"{stem}.png").convert("RGB")
    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(im).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
    t = (t - t.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / t.new_tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)
    return t


def centroid_of_topk(hm, k=0.3):
    """Centroid of activation above top-k percentile, in normalized coords (x, y)."""
    th = np.quantile(hm, 1 - k)
    ys, xs = np.where(hm >= th)
    if len(xs) == 0:
        return None
    w = hm[ys, xs]
    cx = (xs * w).sum() / w.sum() / hm.shape[1]
    cy = (ys * w).sum() / w.sum() / hm.shape[0]
    return cx, cy


if __name__ == "__main__":
    from visualize_dinov3_heatmaps import build_vit, conv_heatmap, load_img, vit_attn_heatmap

    vit, captured = build_vit()
    import timm
    conv = timm.create_model(
        "convnext_tiny.dinov3_lvd1689m", pretrained=True, features_only=True, out_indices=[2, 3]
    ).cuda().eval()

    print(f"{'class':10} {'n_gt':>4} {'GT center':>18} {'ViT ctd':>20} {'dist':>6} {'conv16 ctd':>20} {'dist':>6} {'conv32 ctd':>20} {'dist':>6}")
    for cls, (stem, cidx) in IMGS.items():
        lab = LABDIR / f"{stem}.txt"
        centers = gt_centers(lab, cidx)
        im, t = load_img(stem)
        hm_vit = vit_attn_heatmap(vit, captured, t)
        hm16, hm32 = conv_heatmap(conv, t)
        c_vit = centroid_of_topk(hm_vit)
        c16 = centroid_of_topk(hm16)
        c32 = centroid_of_topk(hm32)
        for cx, cy in centers:
            dv = (c_vit[0] - cx, c_vit[1] - cy)
            d16 = (c16[0] - cx, c16[1] - cy)
            d32 = (c32[0] - cx, c32[1] - cy)
            print(f"{cls:10s} {centers.index((cx,cy))+1:>4}/{(len(centers)):>2} ({cx:.3f},{cy:.3f}) "
                  f"({c_vit[0]:.3f},{c_vit[1]:.3f}) {np.hypot(*dv):.3f} "
                  f"({c16[0]:.3f},{c16[1]:.3f}) {np.hypot(*d16):.3f} "
                  f"({c32[0]:.3f},{c32[1]:.3f}) {np.hypot(*d32):.3f}")
