#!/usr/bin/env python3
"""GT box size distribution for V1 val (2048 base)"""
from pathlib import Path
import glob, os, numpy as np, csv, json

label_dir = Path("/home/ancheng/dataset/dataset_split/val/labels")
img_dir = Path("/home/ancheng/dataset/dataset_split/val/images")

# class names from own_dataset.yaml? Use order from ultralytics 17
names = ["jiaodai","bd","heidian","wy","zyc","zmty","cq","zj","bj","jt","yy","bmss","zy","pd","HD","cy","HD2"] # placeholder, will map via yaml
# Actually read yaml
import yaml
with open("configs/own_dataset.yaml") as f:
    y = yaml.safe_load(f)
names = y["names"] if "names" in y else names
print(f"names: {names}")

files = sorted(glob.glob(str(label_dir / "*.txt")))
# collect sizes
records = []  # (cls, w, h, area_norm, area_px2048, sqrt_area)
per_class_sizes = {i: [] for i in range(len(names))}
all_areas = []

for f in files:
    # image size: assume 2048x2048, but try to read actual via PIL if available
    stem = Path(f).stem
    # find image
    img_path = None
    for ext in [".jpg",".png",".jpeg"]:
        p = img_dir / (stem + ext)
        if p.exists():
            img_path = p
            break
    W = H = 2048
    if img_path:
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                W, H = im.size
        except: pass
    for line in open(f):
        t = line.split()
        if len(t) < 5: continue
        c = int(float(t[0])); x,y,w,h = map(float, t[1:5])
        # w,h are normalized 0-1
        area_norm = w*h
        area_px = w*W * h*H
        sqrt_area = np.sqrt(area_px)
        records.append((c,w,h,area_norm,area_px,sqrt_area,W,H))
        per_class_sizes[c].append((w,h,area_norm,area_px,sqrt_area))
        all_areas.append(area_px)

# Size buckets COCO style on 2048 base
# small <32*32=1024, medium 32-96 (1024-9216), large >96*96
def bucket(area):
    if area < 32*32:
        return "S (<32px)"
    elif area < 96*96:
        return "M (32-96px)"
    else:
        return "L (>96px)"

print(f"Total GT: {len(records)} (377 expected)")
# per class summary
print("\n=== PER-CLASS SIZE SUMMARY ===")
print(f"{'class':>10s} {'n':>3s} {'med_side':>8s} {'med_area':>8s} {'S%':>5s} {'M%':>5s} {'L%':>5s}")
for cid in range(len(names)):
    arr = per_class_sizes[cid]
    if not arr: continue
    sides = [np.sqrt(a[3]) for a in arr]
    areas = [a[3] for a in arr]
    med_side = float(np.median(sides))
    med_area = float(np.median(areas))
    buckets = [bucket(a) for a in areas]
    s = sum(1 for b in buckets if b.startswith("S"))
    m = sum(1 for b in buckets if b.startswith("M"))
    l = sum(1 for b in buckets if b.startswith("L"))
    n = len(arr)
    print(f"{names[cid]:>10s} {n:3d} {med_side:8.1f} {med_area:8.0f} {100*s/n:5.1f} {100*m/n:5.1f} {100*l/n:5.1f}")

# overall
import collections
cnt = collections.Counter(bucket(a) for a in all_areas)
print(f"\nOverall buckets: {cnt}")
print(f"Overall median side: {np.median([np.sqrt(a) for a in all_areas]):.1f} px  median area: {np.median(all_areas):.0f}")

# also normalized area distribution
# Save detailed csv
out = Path("outputs/eval_scoring/097_best_analysis")
out.mkdir(parents=True, exist_ok=True)
with open(out / "gt_size_per_class.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["class_id","class_name","n_gt","median_side_px","median_area_px","small_pct","medium_pct","large_pct"])
    for cid in range(len(names)):
        arr = per_class_sizes[cid]
        if not arr: continue
        sides = [np.sqrt(a[3]) for a in arr]
        areas = [a[3] for a in arr]
        med_side = float(np.median(sides))
        med_area = float(np.median(areas))
        buckets = [bucket(a) for a in areas]
        s = sum(1 for b in buckets if b.startswith("S"))
        m = sum(1 for b in buckets if b.startswith("M"))
        l = sum(1 for b in buckets if b.startswith("L"))
        n = len(arr)
        w.writerow([cid,names[cid],n,f"{med_side:.1f}",f"{med_area:.0f}",f"{100*s/n:.1f}",f"{100*m/n:.1f}",f"{100*l/n:.1f}"])

# Save all boxes for histogram
with open(out / "gt_boxes.json","w") as f:
    json.dump([{"cls":int(r[0]),"cls_name":names[r[0]],"w":r[1],"h":r[2],"area_norm":r[3],"area_px":r[4]} for r in records], f, indent=2)

print(f"\nSaved to {out/'gt_size_per_class.csv'}")
