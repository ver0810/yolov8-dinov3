#!/usr/bin/env python3
"""Run 057 (yolo26s v2+cos) on the v2 val split; GT-vs-pred comparison.

Modes:
  default : stats (report.txt, cases.json) + 5 representative overlays in OUT/
  --all   : additionally export per-image overlays for the ENTIRE val into OUT/all/
            (350 images; red=GT, blue=pred+conf)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "outputs/runs/057_yolo26s_v2_cos/weights/best.pt"
IMG_DIR = ROOT / "outputs/dataset_v2/val/images"
LBL_DIR = ROOT / "outputs/dataset_v2/val/labels"
OUT = ROOT / "outputs/pred_gt_compare"
ALL_OUT = OUT / "all"
IMG_SIZE = 640
CONF = 0.05
IOU = 0.5

CLASSES = ["lj","bd","heidian","wy","zyc","zmty","cq","zj","bj","jt","yy","bmss","zy","pd","HD","cy","jiaodai"]

def load_gt(lbl_path: Path, w: int, h: int):
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, bw, bh = float(p[0]), *map(float, p[1:5])
        boxes.append((int(c), (cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
    return boxes

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1)
    inter = iw*ih
    aA = (a[2]-a[0])*(a[3]-a[1]); bA = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aA + bA - inter + 1e-9)

def match(gt, pred):
    hit, miss, fp = [], [], []
    used = set()
    for g in gt:
        best, bi = 0, -1
        for i, p in enumerate(pred):
            if i in used or p[0] != g[0]:
                continue
            v = iou(g[1:], p[1:])
            if v > best:
                best, bi = v, i
        if best >= IOU and bi >= 0:
            hit.append((g, pred[bi])); used.add(bi)
        else:
            miss.append(g)
    fp = [p for i, p in enumerate(pred) if i not in used]
    return hit, miss, fp

def draw_overlay(model, imgname, outdir, dpi=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    imgp = IMG_DIR / imgname
    img = plt.imread(imgp)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    gt = load_gt(LBL_DIR / (imgname.replace(".png", ".txt")), img.shape[1], img.shape[0])
    res = model.predict(str(imgp), imgsz=IMG_SIZE, conf=CONF, iou=0.45, verbose=False)[0]
    pred = [(int(b.cls), *b.xyxy[0].tolist(), float(b.conf)) for b in res.boxes] if res.boxes is not None else []
    hit, miss, fp = match(gt, pred)
    for cc, x1, y1, x2, y2 in gt:
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="red", lw=1.6))
        ax.text(x1, max(0, y1-3), CLASSES[cc], color="red", fontsize=9,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0))
    for cc, x1, y1, x2, y2, confv in pred:
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="blue", lw=1.1))
        ax.text(x1, max(0, y1-3), f"{CLASSES[cc]}:{confv:.2f}", color="blue", fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0))
    ax.set_title(f"{imgname}  GT={len(gt)} pred={len(pred)} hit={len(hit)} miss={len(miss)} fp={len(fp)}", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(outdir / (imgname.replace(".png", "_ov.png")), dpi=dpi)
    plt.close(fig)
    return len(gt), len(hit), len(miss), len(fp)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="overlay every val image into OUT/all/")
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    model = YOLO(str(WEIGHTS))
    imgs = sorted(IMG_DIR.glob("*.png"))
    print(f"val images: {len(imgs)}")

    per_cls = {c: {"gt": 0, "hit": 0, "miss": 0, "fp": 0} for c in CLASSES}
    cases = []
    for path in imgs:
        res = model.predict(str(path), imgsz=IMG_SIZE, conf=CONF, iou=0.45, verbose=False)[0]
        w, h = res.orig_shape[1], res.orig_shape[0]
        gt = load_gt(LBL_DIR / (path.stem + ".txt"), w, h)
        pred = [(int(b.cls), *b.xyxy[0].tolist()) for b in res.boxes] if res.boxes is not None else []
        hit, miss, fp = match(gt, pred)
        for c in gt: per_cls[CLASSES[c[0]]]["gt"] += 1
        for g, _ in hit: per_cls[CLASSES[g[0]]]["hit"] += 1
        for g in miss: per_cls[CLASSES[g[0]]]["miss"] += 1
        for p in fp: per_cls[CLASSES[p[0]]]["fp"] += 1
        cases.append({"img": path.name, "gt": len(gt), "pred": len(pred),
                      "hit": len(hit), "miss": len(miss), "fp": len(fp),
                      "miss_cls": sorted({CLASSES[g[0]] for g in miss}),
                      "fp_cls": sorted({CLASSES[p[0]] for p in fp})})

    lines = ["057 (yolo26s v2+cos) on v2 val - GT vs pred (conf>=0.05, IoU>=0.5)",
             "=" * 78,
             f"{'class':<10}{'GT':>5}{'hit':>5}{'recall':>8}{'fp':>5}"]
    for c in CLASSES:
        s = per_cls[c]
        r = s["hit"] / s["gt"] if s["gt"] else 1.0
        lines.append(f"{c:<10}{s['gt']:>5}{s['hit']:>5}{r:>8.3f}{s['fp']:>5}")
    tot = {k: sum(per_cls[c][k] for c in CLASSES) for k in ("gt","hit","miss","fp")}
    lines.append("-" * 78)
    lines.append(f"{'TOTAL':<10}{tot['gt']:>5}{tot['hit']:>5}{tot['hit']/tot['gt']:>8.3f}{tot['fp']:>5}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    (OUT / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1))

    if args.all:
        ALL_OUT.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(imgs, 1):
            g, hit, miss, fp = draw_overlay(model, path.name, ALL_OUT, dpi=args.dpi)
            if i % 50 == 0:
                print(f"  [{i}/{len(imgs)}] {path.name} gt={g} hit={hit} miss={miss} fp={fp}", flush=True)
        print(f"all overlays -> {ALL_OUT} ({len(imgs)} images)")
        return

    # representative set (default)
    has_miss = [c for c in cases if c["miss"] > 0]
    has_fp = [c for c in cases if c["fp"] > 0]
    clean = [c for c in cases if c["miss"] == 0 and c["fp"] == 0 and c["gt"] > 0]
    pick = sorted(has_miss, key=lambda c: c["miss"], reverse=True)[:2] + has_fp[:1] + clean[:1]
    seen, chosen = set(), []
    for c in pick + clean:
        if c["img"] in seen: continue
        seen.add(c["img"]); chosen.append(c)
        if len(chosen) >= 5: break
    for c in chosen:
        g, hit, miss, fp = draw_overlay(model, c["img"], OUT, dpi=args.dpi)
        print(f"  overlay: {c['img']}  gt={g} hit={hit} miss={miss} fp={fp}")

    print(f"\noutputs -> {OUT}")

if __name__ == "__main__":
    main()
