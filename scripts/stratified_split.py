#!/usr/bin/env python3
"""Stratified re-split of the 1807-image dataset.

Merges train/val/test, then re-splits stratified by (source-prefix, dominant class)
to fix three issues found in the original split:
  1. Low-frequency classes had tiny val counts (yy: 2 instances, HD: 9) -> noisy scores
  2. Small sources unevenly covered (YF_DHW only in val, zmty/zyc/bj 1 img in test)
  3. val/test at 12.5% is small for 17 classes

Outputs a new directory tree (symlinks) at outputs/dataset_v2/ plus
configs/own_dataset_v2.yaml. The old val (181 imgs) is preserved as an anchor
list (outputs/dataset_v2/anchor_old_val.txt) for cross-checking score stability.

Usage:
    uv run python scripts/stratified_split.py --val 220 --test 220 --seed 42
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = Path("/home/ancheng/dataset/dataset_split")
OUT = ROOT / "outputs" / "dataset_v2"

NAMES = ["lj","bd","heidian","wy","zyc","zmty","cq","zj","bj","jt","yy","bmss","zy","pd","HD","cy","jiaodai"]


def source_of(stem: str) -> str:
    toks = re.split(r"_\d{8}", stem)[0].split("_")
    toks = [t for t in toks if t and not t.isdigit()]
    return "_".join(toks[:2]) if toks else "digits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=220)
    ap.add_argument("--test", type=int, default=220)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1. collect all images + their labels
    all_imgs = sorted(glob.glob(str(DATA / "*" / "images" / "*")))
    entries = []  # (stem, img_path, lbl_path, source, class_counter, n_inst)
    for img in all_imgs:
        stem = Path(img).stem
        lbl = DATA / Path(img).parent.parent.name / "labels" / f"{stem}.txt"
        if not lbl.exists():
            continue
        cc = Counter()
        for line in open(lbl):
            cc[int(line.split()[0])] += 1
        if not cc:
            continue
        entries.append((stem, img, str(lbl), source_of(stem), cc, sum(cc.values())))

    rng = random.Random(args.seed)
    n_total = len(entries)
    n_val, n_test = args.val, args.test
    print(f"total images with labels: {n_total}")

    # 2. stratified assignment. Stratum = (source, dominant class).
    #    Greedy round-robin within strata against GLOBAL quotas (val, test):
    #    process largest strata first, take floor(proportional share) then
    #    top-up from the largest remaining strata. Precisely hits the target
    #    sizes instead of forcing >=1 per stratum (which overflowed).
    strata = defaultdict(list)
    for e in entries:
        stem, img, lbl, src, cc, n = e
        dom = max(cc, key=cc.get)  # dominant class id
        strata[(src, dom)].append(e)

    val_set, test_set = set(), set()
    ordered = sorted(strata.items(), key=lambda kv: -len(kv[1]))
    # first pass: proportional floors
    for key, items in ordered:
        qv = round(len(items) * n_val / n_total)
        qt = round(len(items) * n_test / n_total)
        qv = min(qv, len(items) - 0)  # floor 0; strata with 1 img may go to train
        qt = min(qt, len(items) - qv)
        rng.shuffle(items)
        val_set.update(e[0] for e in items[:qv])
        test_set.update(e[0] for e in items[qv:qv + qt])
    # second pass: top-up to exact quotas from largest untouched strata
    for key, items in ordered:
        if len(val_set) >= n_val and len(test_set) >= n_test:
            break
        for e in items:
            if e[0] in val_set or e[0] in test_set:
                continue
            if len(val_set) < n_val:
                val_set.add(e[0])
            elif len(test_set) < n_test:
                test_set.add(e[0])
            else:
                break

    # 3. write lists + symlink tree
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val", "test"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    train_lines, val_lines, test_lines = [], [], []
    for stem, img, lbl, src, cc, n in entries:
        if stem in val_set:
            split = "val"
        elif stem in test_set:
            split = "test"
        else:
            split = "train"
        (OUT / split / "images" / Path(img).name).symlink_to(Path(img))
        (OUT / split / "labels" / Path(lbl).name).symlink_to(Path(lbl))
        lines = {"train": train_lines, "val": val_lines, "test": test_lines}[split]
        lines.append(str(OUT / split / "images" / Path(img).name))

    # 4. configs
    cfg = {
        "path": str(OUT),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 17,
        "names": NAMES,
    }
    (ROOT / "configs" / "own_dataset_v2.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    # 5. anchor: old val basenames (for cross-checking)
    old_val = sorted(Path(DATA / "val" / "images").glob("*"))
    (OUT / "anchor_old_val.txt").write_text("\n".join(str(p) for p in old_val) + "\n")

    # report
    print(f"split sizes: train={len(train_lines)} val={len(val_lines)} test={len(test_lines)}")
    for split, lines in (("train", train_lines), ("val", val_lines), ("test", test_lines)):
        cc = Counter()
        for l in lines:
            lbl = Path(l).parent.parent / "labels" / (Path(l).stem + ".txt")
            for ln in open(lbl):
                cc[int(ln.split()[0])] += 1
        dist = ", ".join(f"{NAMES[i]}:{cc[i]}" for i in range(17) if cc[i])
        print(f"  {split:5} instances: {sum(cc.values())} | {dist}")
    print(f"new data root: {OUT}")
    print(f"config: configs/own_dataset_v2.yaml")


if __name__ == "__main__":
    main()
