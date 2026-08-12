#!/usr/bin/env python3
"""Evaluate few-shot finetuned runs under the OFFICIAL metric:
macro recall @ conf=0.25 (self-val = v2 val), best-F1 point as reference.
Reads every runs/fewshot_K*_seed*/weights/best.pt present.
Output: outputs/eval_fewshot_v2/fewshot_report_conf025.csv
"""
import csv
import gc
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/ancheng/Code/yolov8-dinov3")
sys.path.insert(0, str(ROOT / "third_party" / "ultralytics"))
import numpy as np
import torch
from ultralytics import YOLO

RUNS = ROOT / "outputs/eval_fewshot_v2/runs"
out = defaultdict(dict)   # [K][seed] = (R25, P25, Rf1, Pf1)
pat = re.compile(r"fewshot_K(\d+)_seed(\d+)")

for d in sorted(RUNS.iterdir()):
    mch = pat.match(d.name)
    if not mch:
        continue
    wt = d / "weights/best.pt"
    if not wt.exists():
        print(f"{d.name}: no weights", flush=True)
        continue
    k, sd = int(mch.group(1)), int(mch.group(2))
    m = YOLO(str(wt))
    metrics = m.val(data=str(ROOT / "configs/own_dataset_v2.yaml"), imgsz=640, batch=16,
                    split="val", conf=0.001, plots=False, verbose=False)
    box = metrics.box
    cr = box.curves_results
    px, p = cr[2][0], cr[2][1]
    r = cr[3][1]
    apci = box.ap_class_index

    def macro(idx):
        Rs, Ps = [], []
        for i, ci in enumerate(apci):
            Rs.append(float(r[i, idx]))
            Ps.append(float(p[i, idx]))
        return float(np.mean(Rs)), float(np.mean(Ps))

    idx25 = int(np.argmin(np.abs(px - 0.25)))
    R25, P25 = macro(idx25)
    f1 = 2 * p * r / (p + r + 1e-9)
    fi = int(np.argmax(np.mean(f1, axis=0)))
    Rf, Pf = macro(fi)
    out[k][sd] = (R25, P25, Rf, Pf)
    print(f"K={k} seed={sd}: R@0.25={R25:.4f} P@0.25={P25:.4f} F1max={Rf:.4f}/{Pf:.4f}", flush=True)
    del m, metrics
    gc.collect()
    torch.cuda.empty_cache()

# aggregate
rows = []
ks = sorted(out)
for k in ks:
    vals = list(out[k].values())
    r25s = [v[0] for v in vals]
    rows.append([k, f"{float(np.mean(r25s)):.4f}", f"{float(np.std(r25s)):.4f}", len(vals)])
avg = float(np.mean([float(np.mean([v[0] for v in out[k].values()])) for k in ks]))
rows.append(["AVG(1/3/5/10)", f"{avg:.4f}", "", sum(len(out[k]) for k in ks)])
print("\n" + "-" * 60)
for row in rows:
    print(f"K={row[0]:>5}  R@0.25 mean={row[1]} std={row[2]}  n={row[3]}")
print(f"Item3 平均检出率 = {avg:.4f}")

csv_path = ROOT / "outputs/eval_fewshot_v2/fewshot_report_conf025.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["K", "macro_recall_mean@0.25", "std", "n_seeds"])
    w.writerows(rows)
print(f"Saved: {csv_path}")