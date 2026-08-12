#!/usr/bin/env python3
"""Load pretrained yolo11s weights into DINOv3-route models.

`model.load(pt)` (intersect_dicts) transfers nothing because the custom
backbones re-order layer indices. This loader performs an explicit
structural transfer:

1. Hybrid:  src yolo11s backbone layers 0-4  → dst hybrid module's internal
            ``yolo_early`` (same pretrained Conv/C3k2 layers, COCO-tuned).
            dst key ``model.0.yolo_early.<i>.<leaf>`` ← src ``model.<i>.<leaf>``.

2. Head:    PAN head C3k2/Conv/Detect layers are matched between source and
            destination by *leaf-shape overlap* — every yolo11 head layer has
            the same internal suffix structure as the route YAML head, so the
            dst layer whose leaf-key set best overlaps a src layer is paired
            to it, then all shape-compatible leaves are copied.

Usage::

    python scripts/load_pretrained_head.py --model <yaml> --weights <018.pt> --out <out.pt>
    # then train with --weights <out.pt>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def split_idx(key: str) -> tuple[int, str]:
    """('model.13.cv1.conv.weight') → (13, 'cv1.conv.weight')."""
    parts = key.split(".", 2)
    return int(parts[1]), parts[2]


def head_layers(sd: dict[str, torch.Tensor], skip_until: int):
    """Group keys by layer index for idx > skip_until (head region)."""
    layers: dict[int, dict[str, list[int]]] = {}  # idx -> {leaf: shape}
    for k, v in sd.items():
        idx, leaf = split_idx(k)
        if idx <= skip_until:
            continue
        layers.setdefault(idx, {})[leaf] = list(v.shape)
    return layers


def match_head(src_layers, dst_layers) -> dict[int, int]:
    """dst_idx -> src_idx with best leaf-shape overlap."""
    pairs: dict[int, int] = {}
    for d_idx, d_leafs in sorted(dst_layers.items()):
        best_s, best_score = None, 0.0
        for s_idx, s_leafs in sorted(src_layers.items()):
            common = sum(1 for leaf, sh in d_leafs.items() if s_leafs.get(leaf) == sh)
            score = common / max(len(d_leafs), 1)
            if score > best_score:
                best_s, best_score = s_idx, score
        if best_s is not None and best_score >= 0.8:
            pairs[d_idx] = best_s
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="target route yaml")
    ap.add_argument("--weights", required=True, help="source yolo11s best.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src_m = YOLO(args.weights).model
    dst_m = YOLO(str(args.model)).model
    src_sd, dst_sd = src_m.state_dict(), dst_m.state_dict()
    n = 0

    # --- 1. hybrid: yolo_early backbone transfer ---
    for k, v in src_sd.items():
        idx, leaf = split_idx(k)
        if idx <= 4:
            dk = f"model.0.yolo_early.{idx}.{leaf}"
            if dk in dst_sd and dst_sd[dk].shape == v.shape:
                dst_sd[dk].copy_(v)
                n += 1

    # --- 2. head transfer by leaf-shape overlap ---
    src_h = head_layers(src_sd, skip_until=10)  # yolo11s backbone = 0..10
    dst_h = head_layers(dst_sd, skip_until=3)   # route backbone ends at layer 3
    for d_idx, s_idx in match_head(src_h, dst_h).items():
        for leaf, v in dst_h[d_idx].items():
            sk = f"model.{s_idx}.{leaf}"
            if sk in src_sd and tuple(src_sd[sk].shape) == tuple(v):
                dst_sd[f"model.{d_idx}.{leaf}"].copy_(src_sd[sk])
                n += 1

    print(f"Transferred {n} tensors from {args.weights}")
    torch.save({"model": dst_m.float()}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()