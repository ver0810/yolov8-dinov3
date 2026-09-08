#!/usr/bin/env python3
"""Evaluate a trained checkpoint against the competition scoring criteria.

OFFICIAL METRIC (adopted 2026-08): self-val (each weight validated on its
OWN training split's val) + macro recall @ conf=0.25 as the primary operating
point; best-F1 point as reference. The legacy conf->0 max-recall fallback and
cross-split validation are DEPRECATED (produced inflated numbers).

Scoring items (from docs/GOAL.md):
  (1) Recall >= 95%                          (30%)
  (2) Over-detection rate <= 5%, FP rate <= 5%  (15%)
  (3) Few-shot avg recall @ K={1,3,5,10}     (15%)
  (4) Inference speed >= 30 fps @1024x1024 @RTX3080  (20%)
  (5) Innovation & report quality             (20%)

This script covers items (1), (2), and (4).
Item (3) requires a separate few-shot protocol (scripts/eval_fewshot.py +
conf=0.25 evaluation, see eval_fewshot_scoring.py).
Item (5) is qualitative.

Usage:
    uv run python scripts/eval_scoring.py \
        --weights outputs/runs/004_yolo11s/weights/best.pt \
        --data configs/own_dataset.yaml \
        --imgsz 640

Output:
    outputs/eval_scoring/<run_name>/
        scoring_report.txt      — human-readable summary
        scoring_report.csv      — per-class table
        rp_curves.png           — per-class R-P curves with R=95% line
        conf_recall_curve.png   — macro recall vs conf threshold
        conf_precision_curve.png— macro precision vs conf threshold
        fps_benchmark.json      — latency stats at various imgsz
"""

from __future__ import annotations

import argparse
import json

# Ensure local ultralytics fork is importable
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "ultralytics"))

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate against competition scoring criteria")
    p.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    p.add_argument("--data", type=str, default=str(ROOT / "configs/own_dataset.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--tta", action="store_true", help="Enable TTA (ultralytics augment=True)")
    p.add_argument("--output", type=str, default=None, help="Output dir (default: outputs/eval_scoring/<run_name>)")
    p.add_argument("--target-recall", type=float, default=0.95, help="Target recall for scoring item 1")
    p.add_argument("--target-fp-rate", type=float, default=0.05, help="Target FP rate for scoring item 2")
    return p.parse_args()


def run_validation(model, data, imgsz, batch, device, split, tta=False):
    """Run ultralytics val with conf=0.001 to get full R-P curves."""
    kw = dict(data=data, imgsz=imgsz, batch=batch, split=split, conf=0.001, plots=False, verbose=False,
              augment=tta)
    if device is not None:
        kw["device"] = device
    metrics = model.val(**kw)
    return metrics


def extract_curves(metrics):
    """Extract per-class R-P curves from DetMetrics.

    Returns:
        px: (1000,) confidence thresholds [0, 1]
        p_curve: (nc, 1000) precision at each conf per class
        r_curve: (nc, 1000) recall at each conf per class
        f1_curve: (nc, 1000) F1 at each conf per class
        names: dict[int, str]
        nt_per_class: (nc,) GT instance counts
        ap_class_index: list of class indices with data
        all_ap: (nc_with_data, 10) AP at 10 IoU thresholds
    """
    box = metrics.box
    return {
        "px": box.px,                    # (1000,)
        "p_curve": box.p_curve,          # (nc, 1000)
        "r_curve": box.r_curve,          # (nc, 1000)
        "f1_curve": box.f1_curve,        # (nc, 1000)
        "names": metrics.names,
        "nt_per_class": metrics.nt_per_class,
        "ap_class_index": box.ap_class_index,
        "all_ap": box.all_ap,
        "p_at_f1": box.p,
        "r_at_f1": box.r,
        "f1_at_f1": box.f1,
    }


def compute_macro_recall_precision(px, p_curve, r_curve, ap_class_index, nc, nt_per_class):
    """Compute macro-averaged recall and precision at each conf threshold.

    Macro average = mean over classes that have GT instances (nt_per_class > 0).
    """
    macro_r = np.zeros((nc, len(px)))
    macro_p = np.zeros((nc, len(px)))
    for i, ci in enumerate(ap_class_index):
        macro_r[ci] = r_curve[i]
        macro_p[ci] = p_curve[i]

    # Only average over classes that have GT instances
    active = [ci for ci in range(nc) if nt_per_class[ci] > 0]
    if not active:
        return np.zeros_like(px), np.zeros_like(px)

    mean_r = macro_r[active].mean(axis=0)
    mean_p = macro_p[active].mean(axis=0)
    return mean_r, mean_p



def compute_weighted_recall_macro_precision(metrics, px):
    """Compute instance-weighted (micro) recall and macro precision.

    Weighted recall = sum(nt_c * r_c) / sum(nt_c) — true micro recall.
    Precision = macro approximation (unweighted mean over classes)
    since per-class prediction counts are not available after process().
    """
    nt = metrics.nt_per_class
    aci = metrics.box.ap_class_index
    r_curve = metrics.box.r_curve
    p_curve = metrics.box.p_curve

    total_gt = nt[aci].sum()
    if total_gt == 0:
        return np.zeros_like(px), np.zeros_like(px)

    # Weighted recall = sum(nt_c * r_c) / sum(nt_c)
    weighted_r = sum(nt[ci] * r_curve[i] for i, ci in enumerate(aci)) / total_gt

    mean_p = p_curve.mean(axis=0)

    return weighted_r, mean_p



def find_conf_for_target_recall(px, mean_r, target_r):
    """Find the highest confidence threshold where macro recall >= target_r.

    px goes from 0 to 1; recall decreases as conf increases.
    """
    valid = np.where(mean_r >= target_r)[0]
    if len(valid) == 0:
        return None, None
    idx = valid[-1]  # highest conf that still meets target
    return float(px[idx]), float(mean_r[idx])

def per_class_analysis(px, p_curve, r_curve, names, ap_class_index, nt_per_class, target_r):
    """Per-class: find conf threshold for target recall, report precision there."""
    rows = []
    for i, ci in enumerate(ap_class_index):
        cname = names.get(ci, f"class_{ci}")
        n_gt = int(nt_per_class[ci])
        r_c = r_curve[i]
        p_c = p_curve[i]

        # Find conf where this class reaches target recall
        valid = np.where(r_c >= target_r)[0]
        if len(valid) > 0:
            idx = valid[-1]
            conf_at_target = float(px[idx])
            prec_at_target = float(p_c[idx])
            recall_at_target = float(r_c[idx])
        else:
            conf_at_target = None
            prec_at_target = None
            recall_at_target = float(r_c.max())

        # Best F1 point
        f1_c = 2 * p_c * r_c / (p_c + r_c + 1e-16)
        best_f1_idx = np.argmax(f1_c)
        best_f1 = float(f1_c[best_f1_idx])
        best_f1_conf = float(px[best_f1_idx])
        best_f1_p = float(p_c[best_f1_idx])
        best_f1_r = float(r_c[best_f1_idx])

        rows.append({
            "class_id": ci,
            "class_name": cname,
            "n_gt": n_gt,
            "max_recall": float(r_c.max()),
            "conf_for_R95": conf_at_target,
            "P_at_R95": prec_at_target,
            "R_at_R95_conf": recall_at_target,
            "best_F1": best_f1,
            "best_F1_conf": best_f1_conf,
            "P_at_best_F1": best_f1_p,
            "R_at_best_F1": best_f1_r,
        })
    return rows


def benchmark_fps(model, imgsz_list, device, n_warmup=10, n_runs=100):
    """Measure inference latency at various image sizes.

    Returns dict: {imgsz: {mean_ms, std_ms, fps, ...}}
    """
    results = {}
    for sz in imgsz_list:
        # Create dummy input
        dummy = torch.randint(0, 256, (1, 3, sz, sz), dtype=torch.uint8)
        if device and "cuda" in str(device):
            dummy = dummy.cuda()
        elif torch.cuda.is_available():
            dummy = dummy.cuda()

        # Warmup
        for _ in range(n_warmup):
            model.predict(dummy, verbose=False, imgsz=sz)

        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.predict(dummy, verbose=False, imgsz=sz)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        times = np.array(times)
        results[sz] = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "fps": float(1000.0 / np.mean(times)),
            "n_runs": n_runs,
        }
    return results


def generate_plots(px, p_curve, r_curve, names, ap_class_index, nt_per_class,
                   mean_r, mean_p, target_r, save_dir):
    """Generate R-P curve plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return

    # --- Per-class R-P curves ---
    nc_data = len(ap_class_index)
    ncols = min(4, nc_data)
    nrows = (nc_data + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, ci in enumerate(ap_class_index):
        i = idx  # curve arrays are indexed by position in ap_class_index
        ax = axes[idx // ncols, idx % ncols]
        cname = names.get(ci, f"class_{ci}")
        n_gt = int(nt_per_class[ci])
        ax.plot(r_curve[i], p_curve[i], linewidth=1.5)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axvline(x=target_r, color="red", linewidth=1, linestyle="--", label=f"R={target_r:.0%}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{cname} (n={n_gt})", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(nc_data, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Per-class Precision-Recall Curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / "rp_curves.png", dpi=150)
    plt.close(fig)

    # --- Macro R/P vs Confidence ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(px, mean_r, linewidth=2, color="blue")
    ax1.axhline(y=target_r, color="red", linewidth=1, linestyle="--", label=f"Target R={target_r:.0%}")
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Macro Recall")
    ax1.set_title("Macro Recall vs Confidence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(px, mean_p, linewidth=2, color="green")
    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("Macro Precision")
    ax2.set_title("Macro Precision vs Confidence")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "conf_recall_precision.png", dpi=150)
    plt.close(fig)

    # --- Per-class recall vs confidence (for identifying weak classes) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, ci in enumerate(ap_class_index):
        cname = names.get(ci, f"class_{ci}")
        ax.plot(px, r_curve[i], linewidth=1, alpha=0.7, label=f"{cname}")
    ax.axhline(y=target_r, color="red", linewidth=2, linestyle="--", label=f"Target R={target_r:.0%}")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Recall")
    ax.set_title("Per-class Recall vs Confidence")
    ax.legend(fontsize=7, ncol=3, loc="center right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "per_class_recall_vs_conf.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {save_dir}/")


def main():
    args = parse_args()

    # Resolve output dir
    weights_path = Path(args.weights)
    run_name = weights_path.parent.parent.name  # e.g. 004_yolo11s
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = ROOT / "outputs" / "eval_scoring" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  Scoring Evaluation: {run_name}")
    print(f"  Weights: {weights_path}")
    print(f"  Data: {args.data}")
    print(f"  imgsz: {args.imgsz}, split: {args.split}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    # Load model
    print("\n[1/4] Loading model...")
    model = YOLO(str(weights_path))

    # Run validation
    print("[2/4] Running validation (conf=0.001 for full curves)...")
    metrics = run_validation(model, args.data, args.imgsz, args.batch, args.device, args.split, args.tta)

    # Extract curves
    curves = extract_curves(metrics)
    px = curves["px"]
    p_curve = curves["p_curve"]
    r_curve = curves["r_curve"]
    names = curves["names"]
    nt_per_class = curves["nt_per_class"]
    ap_class_index = curves["ap_class_index"]
    nc = len(names)

    print(f"  Classes with GT: {len(ap_class_index)}/{nc}")
    print(f"  Total GT instances: {nt_per_class.sum()}")

    # Compute macro R/P curves
    mean_r, mean_p = compute_macro_recall_precision(px, p_curve, r_curve, ap_class_index, nc, nt_per_class)

    # ============ OFFICIAL METRIC (self-val, conf=0.25) ============
    print("\n  OFFICIAL METRIC (self-val + conf=0.25):")
    idx25 = int(np.argmin(np.abs(px - 0.25)))
    R25, P25 = float(mean_r[idx25]), float(mean_p[idx25])
    F25 = 2 * R25 * P25 / (R25 + P25 + 1e-9) if R25 + P25 > 0 else 0.0
    f1s = [2 * p_curve[i] * r_curve[i] / (p_curve[i] + r_curve[i] + 1e-9)
           for i in range(len(ap_class_index))]
    f1_curve = np.mean(f1s, axis=0)
    fidx = int(np.argmax(f1_curve))
    print(f"    R@0.25 = {R25:.4f}  P@0.25 = {P25:.4f}  F1@0.25 = {F25:.4f}")
    print(f"    best-F1 point: R={float(mean_r[fidx]):.4f} P={float(mean_p[fidx]):.4f} @conf={float(px[fidx]):.3f}")
    print(f"    ITEM1: {'PASS' if R25 >= 0.95 else 'FAIL (gap {:.4f})'.format(0.95 - R25)}")
    print(f"    [INFO only, deprecated] maxR(conf->0) = {float(mean_r.max()):.4f} "
          f"(P={float(mean_p[int(np.argmax(mean_r))]):.4f}; 该点 FP 为 TP 的数十倍，不作任何判定)")
    print(f"{'='*70}")

    # ============ Diagnostic (deprecated working-point search; curve outputs only) ============
    # Compute instance-weighted recall (macro precision) — diagnostic
    weighted_r, weighted_p = compute_weighted_recall_macro_precision(metrics, px)
    print(f"  SCORING ITEM 1 (deprecated conf@R95 diagnostic): Recall >= {args.target_recall:.0%}")

    conf_at_target, recall_at_target = find_conf_for_target_recall(px, mean_r, args.target_recall)
    conf_at_target_wr, _ = find_conf_for_target_recall(px, weighted_r, args.target_recall)

    if conf_at_target is not None:
        # Find precision at this conf
        idx = np.argmin(np.abs(px - conf_at_target))
        prec_at_target = float(mean_p[idx])
        print(f"  [MACRO] conf={conf_at_target:.4f} → R={recall_at_target:.4f}, P={prec_at_target:.4f}")
        print(f"  [MACRO] PASS: R >= {args.target_recall:.0%} achievable at conf={conf_at_target:.4f}")
    else:
        max_macro_r = float(mean_r.max())
        print(f"  [MACRO] FAIL: max macro recall = {max_macro_r:.4f} < {args.target_recall:.0%}")
        print(f"  [MACRO] Gap: {args.target_recall - max_macro_r:.4f}")

    if conf_at_target_wr is not None:
        idx_w = np.argmin(np.abs(px - conf_at_target_wr))
        prec_at_target_w = float(weighted_p[idx_w])
        print(f"  [WEIGHTED] conf={conf_at_target_wr:.4f} → R={float(weighted_r[idx_w]):.4f}, P={prec_at_target_w:.4f}")
    else:
        max_weighted_r = float(weighted_r.max())
        print(f"  [WEIGHTED] FAIL: max weighted recall = {max_weighted_r:.4f} < {args.target_recall:.0%}")

    # --- Per-class analysis ---
    print(f"\n{'='*70}")
    print(f"  PER-CLASS ANALYSIS (target R={args.target_recall:.0%})")
    print(f"{'='*70}")

    rows = per_class_analysis(px, p_curve, r_curve, names, ap_class_index, nt_per_class, args.target_recall)

    # Sort by max_recall ascending (worst first)
    rows.sort(key=lambda r: r["max_recall"])

    header = f"{'Class':>12s} {'n_GT':>5s} {'max_R':>6s} {'conf@R95':>9s} {'P@R95':>6s} {'best_F1':>7s} {'conf@F1':>8s}"
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for r in rows:
        conf_str = f"{r['conf_for_R95']:.4f}" if r['conf_for_R95'] is not None else "N/A"
        p_str = f"{r['P_at_R95']:.3f}" if r['P_at_R95'] is not None else "N/A"
        print(f"  {r['class_name']:>12s} {r['n_gt']:5d} {r['max_recall']:.4f} {conf_str:>9s} {p_str:>6s} {r['best_F1']:.4f} {r['best_F1_conf']:.4f}")

    n_pass = sum(1 for r in rows if r["conf_for_R95"] is not None)
    print(f"\n  Classes achieving R>={args.target_recall:.0%}: {n_pass}/{len(rows)}")

    # --- Scoring Item 2: FP rate ---
    print(f"\n{'='*70}")
    print(f"  SCORING ITEM 2: Over-detection / FP rate <= {args.target_fp_rate:.0%}")
    print(f"{'='*70}")
    print("  [NOTE] No clean (defect-free) images in dataset → cannot measure FP on clean images.")
    print("  [NOTE] Reporting FP among all predictions on val set (proxy metric).")

    # Official FP proxy at conf=0.25 (self-val operating point)
    idx = int(np.argmin(np.abs(px - 0.25)))
    total_fp = 0
    total_tp = 0
    for i, ci in enumerate(ap_class_index):
        r_c = float(r_curve[i, idx])
        p_c = float(p_curve[i, idx])
        n_gt_c = int(nt_per_class[ci])
        tp_c = r_c * n_gt_c
        if p_c > 1e-6:
            fp_c = tp_c * (1 - p_c) / p_c
        else:
            fp_c = tp_c * 999  # very high FP when precision ~0
        total_tp += tp_c
        total_fp += fp_c
    total_pred = total_tp + total_fp
    fp_rate = total_fp / total_pred if total_pred > 0 else 0
    print("  At conf=0.25 (official operating point):")
    print(f"    Estimated TP={total_tp:.0f}, FP={total_fp:.0f}, total_pred={total_pred:.0f}")
    print(f"    FP rate (FP/total_pred) = {fp_rate:.4f} ({fp_rate:.1%})")
    if fp_rate <= args.target_fp_rate:
        print(f"    PASS: FP rate <= {args.target_fp_rate:.0%}")
    else:
        print(f"    FAIL: FP rate {fp_rate:.1%} > {args.target_fp_rate:.0%}")

    # --- Scoring Item 4: FPS benchmark ---
    print(f"\n{'='*70}")
    print("  SCORING ITEM 4: Inference Speed >= 30 fps @1024x1024")
    print(f"{'='*70}")

    imgsz_list = [640, 1024, 1280]
    fps_results = benchmark_fps(model, imgsz_list, args.device, n_warmup=10, n_runs=50)

    print(f"  {'imgsz':>6s} {'mean_ms':>8s} {'p50_ms':>8s} {'p95_ms':>8s} {'fps':>7s} {'@3080*':>8s}")
    print(f"  {'-'*50}")
    for sz, r in fps_results.items():
        # RTX 3080 ≈ 2.5x RTX 3060 Laptop (FP16 TFLOPS ratio)
        fps_3080_est = r["fps"] * 2.5
        flag = "OK" if fps_3080_est >= 30 else "SLOW"
        print(f"  {sz:6d} {r['mean_ms']:8.1f} {r['p50_ms']:8.1f} {r['p95_ms']:8.1f} {r['fps']:7.1f} {fps_3080_est:7.1f} {flag}")
    print("  * Estimated RTX 3080 FPS = measured FPS × 2.5 (TFLOPS ratio)")

    # --- Generate plots ---
    print("\n[3/4] Generating plots...")
    generate_plots(px, p_curve, r_curve, names, ap_class_index, nt_per_class,
                   mean_r, mean_p, args.target_recall, out_dir)

    # --- Save reports ---
    print(f"[4/4] Saving reports to {out_dir}/")

    # CSV
    import csv
    csv_path = out_dir / "scoring_report.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    else:
        print("  [WARN] No per-class data to write to CSV")

    # JSON (FPS)
    fps_path = out_dir / "fps_benchmark.json"
    with open(fps_path, "w") as f:
        json.dump(fps_results, f, indent=2)

    # Text summary
    txt_path = out_dir / "scoring_report.txt"
    with open(txt_path, "w") as f:
        f.write("Scoring Evaluation Report\n")
        f.write(f"{'='*70}\n")
        f.write(f"Model: {run_name}\n")
        f.write(f"Weights: {weights_path}\n")
        f.write(f"Data: {args.data}\n")
        f.write(f"imgsz: {args.imgsz}, split: {args.split}\n")
        f.write(f"Target recall: {args.target_recall:.0%}\n")
        f.write(f"Target FP rate: {args.target_fp_rate:.0%}\n\n")

        f.write(f"SCORING ITEM 1: Recall >= {args.target_recall:.0%}\n")
        f.write(f"{'-'*40}\n")
        if conf_at_target is not None:
            f.write(f"  Macro: conf={conf_at_target:.4f} → R={recall_at_target:.4f}, P={prec_at_target:.4f}\n")
            f.write("  Status: PASS (at macro level)\n")
        else:
            f.write(f"  Macro max recall: {float(mean_r.max()):.4f}\n")
            f.write(f"  Status: FAIL (gap = {args.target_recall - float(mean_r.max()):.4f})\n")
        f.write("\n")

        f.write(f"SCORING ITEM 2: FP rate <= {args.target_fp_rate:.0%}\n")
        f.write(f"{'-'*40}\n")
        f.write("  No clean images available for direct measurement.\n")
        if conf_at_target is not None:
            f.write(f"  Proxy FP rate at R>={args.target_recall:.0%} working point: {fp_rate:.1%}\n")
        f.write("\n")

        f.write("SCORING ITEM 4: Speed >= 30 fps @1024x1024 @RTX3080\n")
        f.write(f"{'-'*40}\n")
        for sz, r in fps_results.items():
            fps_3080 = r["fps"] * 2.5
            f.write(f"  {sz}x{sz}: {r['fps']:.1f} fps (measured), ~{fps_3080:.1f} fps (est. @3080)\n")
        f.write("\n")

        f.write("PER-CLASS DETAIL (sorted by max_recall ascending)\n")
        f.write(f"{'-'*70}\n")
        f.write(f"{'Class':>12s} {'n_GT':>5s} {'max_R':>6s} {'conf@R95':>9s} {'P@R95':>6s} {'best_F1':>7s}\n")
        for r in rows:
            conf_str = f"{r['conf_for_R95']:.4f}" if r['conf_for_R95'] is not None else "N/A"
            p_str = f"{r['P_at_R95']:.3f}" if r['P_at_R95'] is not None else "N/A"
            f.write(f"{r['class_name']:>12s} {r['n_gt']:5d} {r['max_recall']:.4f} {conf_str:>9s} {p_str:>6s} {r['best_F1']:.4f}\n")

    print(f"\n{'='*70}")
    print(f"  DONE. Reports saved to: {out_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
