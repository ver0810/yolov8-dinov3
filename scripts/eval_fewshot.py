#!/usr/bin/env python3
"""Few-shot evaluation protocol for competition scoring item (3).

Two modes (set via --weights):
  - True few-shot:  --weights yolo11s.pt        (COCO-pretrained backbone)
  - Continual:      --weights 004 best.pt       (domain-trained backbone)

Protocol:
  - K in {1, 3, 5, 10}: sample K images per class from training set
  - N_SEEDS=3: repeat each K with different random seeds, report mean +/- std
  - Training: freeze backbone (freeze=10), train head 50 epochs
  - Evaluation: full val set, conf=0.001, report per-class max recall + macro recall

Scoring item (3): average recall across K={1,3,5,10} under few-shot setting.

Usage (true few-shot):
    uv run python scripts/eval_fewshot.py \
        --weights yolo11s.pt \
        --data configs/own_dataset.yaml \
        --k-shots 1 3 5 10 --seeds 3 --epochs 50

Usage (continual):
    uv run python scripts/eval_fewshot.py \
        --weights outputs/runs/004_yolo11s/weights/best.pt \
        --data configs/own_dataset.yaml \
        --k-shots 1 3 5 10 --seeds 3 --epochs 50

Output:
    outputs/eval_fewshot/
        fewshot_report.txt / fewshot_report.csv / fewshot_per_class.csv
        fewshot_curves.png / fewshot_heatmap.png
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "ultralytics"))

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Few-shot evaluation protocol")
    p.add_argument("--weights", type=str, required=True, help="Pretrained weights (e.g. 004 best.pt)")
    p.add_argument("--data", type=str, default=str(ROOT / "configs/own_dataset.yaml"))
    p.add_argument("--train-dir", type=str, default="/home/ancheng/dataset/dataset_split/train")
    p.add_argument("--val-dir", type=str, default="/home/ancheng/dataset/dataset_split/val")
    p.add_argument("--k-shots", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--seeds", type=int, default=3, help="Number of random seeds per K")
    p.add_argument("--epochs", type=int, default=50, help="Fine-tuning epochs per run")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--freeze", type=int, default=10, help="Freeze first N layers (10 = full backbone for yolo11s)")
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate for fine-tuning")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--skip-train", action="store_true", help="Skip training, only evaluate (for debugging)")
    return p.parse_args()


def load_class_image_map(train_dir: Path) -> dict[int, list[Path]]:
    """Parse train labels to build class_id → [image_paths] mapping.

    Returns dict mapping class_id (int) to list of absolute image paths.
    """
    labels_dir = train_dir / "labels"
    images_dir = train_dir / "images"
    class_map = defaultdict(list)

    for txt_file in sorted(labels_dir.glob("*.txt")):
        stem = txt_file.stem
        # Find corresponding image
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        with open(txt_file) as f:
            lines = f.readlines()

        classes_in_image = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            classes_in_image.add(cls_id)

        for cls_id in classes_in_image:
            class_map[cls_id].append(img_path)

    return dict(class_map)


def sample_fewshot(class_map: dict[int, list[Path]], k: int, seed: int) -> list[Path]:
    """Sample K images per class. If a class has < K images, take all.

    Returns deduplicated list of image paths.
    """
    rng = random.Random(seed)
    selected = set()
    for cls_id, paths in class_map.items():
        n = min(k, len(paths))
        sampled = rng.sample(paths, n)
        selected.update(sampled)
    return sorted(selected)


def write_shot_txt(images: list[Path], output_path: Path) -> None:
    """Write image paths to a .txt file for ultralytics data loader."""
    with open(output_path, "w") as f:
        for img in images:
            f.write(str(img) + "\n")


def write_temp_data_yaml(original_yaml: str, train_txt: Path, output_path: Path) -> None:
    """Create a temporary data yaml pointing train to the few-shot txt file."""
    with open(original_yaml) as f:
        data = yaml.safe_load(f)
    data["train"] = str(train_txt)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def train_fewshot(weights: str, data_yaml: str, epochs: int, imgsz: int, batch: int,
                  freeze: int, lr: float, device: str | None, project: str, name: str) -> Path:
    """Fine-tune from pretrained weights with frozen backbone.

    Returns path to best.pt.
    """
    model = YOLO(weights)
    kw = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        freeze=freeze,
        lr0=lr,
        project=project,
        name=name,
        exist_ok=True,
        patience=epochs,  # no early stopping for few-shot
        plots=False,
        verbose=False,
    )
    if device is not None:
        kw["device"] = device
    model.train(**kw)
    return Path(project) / name / "weights" / "best.pt"


def evaluate_recall(weights: str, data_yaml: str, imgsz: int, batch: int,
                    device: str | None) -> tuple[np.ndarray, dict[int, str], np.ndarray]:
    """Run validation and return per-class recall at conf=0.001.

    Returns:
        per_class_max_recall: (nc,) array of max recall per class
        names: dict[int, str]
        nt_per_class: (nc,) GT counts
    """
    model = YOLO(weights)
    kw = dict(data=data_yaml, imgsz=imgsz, batch=batch, split="val",
              conf=0.001, plots=False, verbose=False)
    if device is not None:
        kw["device"] = device
    metrics = model.val(**kw)

    box = metrics.box
    r_curve = box.r_curve  # (n_data_classes, 1000)
    ap_class_index = box.ap_class_index
    names = metrics.names
    nt_per_class = metrics.nt_per_class
    nc = len(names)

    per_class_max_recall = np.zeros(nc)
    for i, ci in enumerate(ap_class_index):
        per_class_max_recall[ci] = float(r_curve[i].max())

    return per_class_max_recall, names, nt_per_class


def main():
    args = parse_args()

    # Setup output dir
    if args.output:
        out_dir = Path(args.output).resolve()
    else:
        out_dir = ROOT / "outputs" / "eval_fewshot"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = Path(args.train_dir)
    original_data = args.data

    print(f"{'='*70}")
    print(f"  Few-Shot Evaluation Protocol")
    print(f"{'='*70}")
    print(f"  Pretrained weights: {args.weights}")
    print(f"  K-shots: {args.k_shots}")
    print(f"  Seeds per K: {args.seeds}")
    print(f"  Epochs per run: {args.epochs}")
    print(f"  Freeze: {args.freeze} layers")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    # Build class → image map
    print("\n[1/4] Building class → image mapping from training set...")
    class_map = load_class_image_map(train_dir)
    nc = len(class_map)
    print(f"  Found {nc} classes in training set")
    for cls_id in sorted(class_map.keys()):
        print(f"    class {cls_id:2d}: {len(class_map[cls_id]):4d} images")

    # Check which classes have enough samples for each K
    print(f"\n  Feasibility check:")
    for k in args.k_shots:
        short = [cid for cid, paths in class_map.items() if len(paths) < k]
        if short:
            print(f"    K={k}: {len(short)} classes have < {k} images (will use all available)")
        else:
            print(f"    K={k}: all classes have >= {k} images")

    # Run few-shot experiments
    print(f"\n[2/4] Running few-shot experiments...")

    # Results storage: results[k][seed] = per_class_max_recall array
    results = {}
    all_names = None
    all_nt = None

    total_runs = len(args.k_shots) * args.seeds
    run_idx = 0

    for k in args.k_shots:
        results[k] = {}
        for seed in range(args.seeds):
            run_idx += 1
            run_name = f"fewshot_K{k}_seed{seed}"
            print(f"\n  [{run_idx}/{total_runs}] K={k}, seed={seed}")

            # Sample images
            shot_images = sample_fewshot(class_map, k, seed)
            n_images = len(shot_images)
            print(f"    Sampled {n_images} unique images")

            # Count per-class instances in the shot (for logging)
            shot_class_counts = defaultdict(int)
            for img in shot_images:
                txt = img.parent.parent / "labels" / f"{img.stem}.txt"
                if txt.exists():
                    with open(txt) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                shot_class_counts[int(line.split()[0])] += 1
            print(f"    Shot instances: {dict(shot_class_counts)}")

            # Write shot txt and temp data yaml
            tmp_dir = out_dir / "tmp"
            tmp_dir.mkdir(exist_ok=True)
            shot_txt = tmp_dir / f"shot_K{k}_seed{seed}.txt"
            write_shot_txt(shot_images, shot_txt)

            tmp_yaml = tmp_dir / f"data_K{k}_seed{seed}.yaml"
            write_temp_data_yaml(original_data, shot_txt, tmp_yaml)

            per_class_r = np.zeros(nc) if 'nc' in dir() else None  # fallback
            try:
                if args.skip_train:
                    print(f"    [SKIP-TRAIN] Using pretrained weights directly")
                    best_pt = args.weights
                else:
                    # Train
                    project = str(out_dir / "runs")
                    print(f"    Training {args.epochs} epochs (freeze={args.freeze})...")
                    best_pt = train_fewshot(
                        weights=args.weights,
                        data_yaml=str(tmp_yaml),
                        epochs=args.epochs,
                        imgsz=args.imgsz,
                        batch=args.batch,
                        freeze=args.freeze,
                        lr=args.lr,
                        device=args.device,
                        project=project,
                        name=run_name,
                    )
                    print(f"    Training done: {best_pt}")

                # Evaluate
                print(f"    Evaluating on full val set...")
                per_class_r, names, nt = evaluate_recall(
                    weights=str(best_pt),
                    data_yaml=original_data,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                )
                all_names = names
                all_nt = nt
            except Exception as e:
                print(f"    [FAIL] {run_name}: {type(e).__name__}: {e}")
                per_class_r = np.array([np.nan] * 17)  # fill NaN for this run
                import traceback; traceback.print_exc()

            results[k][seed] = per_class_r
            if per_class_r is not None and not np.isnan(per_class_r[0]):
                macro_r = per_class_r[nt > 0].mean() if (nt > 0).any() else 0.0
                print(f"    Macro recall (conf→0): {macro_r:.4f}")
            else:
                print(f"    Skipping aggregation — run failed")

    # Aggregate results
    print(f"\n[3/4] Aggregating results...")

    if all_names is None or all_nt is None:
        print("  [FAIL] No successful evaluation runs — cannot aggregate.")
        sys.exit(1)

    # Per-K statistics (skip NaN seeds from failed runs)
    k_stats = {}
    for k in args.k_shots:
        seed_recalls = np.array([results[k][s] for s in range(args.seeds)])  # (seeds, nc)
        mean_per_class = np.nanmean(seed_recalls, axis=0)
        std_per_class = np.nanstd(seed_recalls, axis=0)

        # Macro recall = mean over classes with GT, skip NaN seeds
        active = all_nt > 0
        macro_means = []
        for s in range(args.seeds):
            r = results[k][s]
            if not np.all(np.isnan(r)):
                macro_means.append(r[active].mean())
        macro_mean = np.mean(macro_means) if macro_means else float("nan")
        macro_std = np.std(macro_means) if len(macro_means) > 1 else 0.0

        k_stats[k] = {
            "macro_recall_mean": macro_mean,
            "macro_recall_std": macro_std,
            "per_class_mean": mean_per_class,
            "per_class_std": std_per_class,
            "n_success": len(macro_means),
        }

    valid_ks = [k for k in args.k_shots if not np.isnan(k_stats[k]["macro_recall_mean"])]
    if not valid_ks:
        print("  [FAIL] All K-shot experiments failed.")
        sys.exit(1)
    avg_recall = np.nanmean([k_stats[k]["macro_recall_mean"] for k in args.k_shots])
    print(f"\n{'='*70}")
    print(f"  FEW-SHOT RESULTS (macro recall @ conf→0)")
    print(f"{'='*70}")
    print(f"  {'K-shot':>8s} {'mean_R':>8s} {'std_R':>8s}")
    print(f"  {'-'*28}")
    for k in args.k_shots:
        s = k_stats[k]
        print(f"  {k:8d} {s['macro_recall_mean']:8.4f} {s['macro_recall_std']:8.4f}")
    print(f"  {'-'*28}")
    print(f"  {'AVG':>8s} {avg_recall:8.4f}          ← scoring item (3)")

    # Per-class detail at each K
    print(f"\n  Per-class recall (mean over {args.seeds} seeds):")
    header = f"  {'Class':>12s} {'n_GT':>5s}"
    for k in args.k_shots:
        header += f" {'K='+str(k):>8s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for ci in range(len(all_names)):
        if all_nt[ci] == 0:
            continue
        cname = all_names.get(ci, f"c{ci}")
        row = f"  {cname:>12s} {int(all_nt[ci]):5d}"
        for k in args.k_shots:
            row += f" {k_stats[k]['per_class_mean'][ci]:8.4f}"
        print(row)

    # Generate plots
    print(f"\n[4/4] Generating plots and saving reports...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Bar chart: macro recall vs K
        fig, ax = plt.subplots(figsize=(8, 5))
        ks = args.k_shots
        means = [k_stats[k]["macro_recall_mean"] for k in ks]
        stds = [k_stats[k]["macro_recall_std"] for k in ks]
        bars = ax.bar([str(k) for k in ks], means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
        ax.axhline(y=0.95, color="red", linewidth=1.5, linestyle="--", label="Target R=95%")
        ax.set_xlabel("K-shot")
        ax.set_ylabel("Macro Recall (conf→0)")
        ax.set_title(f"Few-Shot Detection Performance\n(avg across K = {avg_recall:.4f})")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "fewshot_curves.png", dpi=150)
        plt.close(fig)

        # Heatmap: per-class recall vs K
        nc_active = sum(1 for ci in range(len(all_names)) if all_nt[ci] > 0)
        fig, ax = plt.subplots(figsize=(6, max(4, nc_active * 0.35)))
        data_matrix = np.zeros((nc_active, len(ks)))
        class_labels = []
        row_idx = 0
        for ci in range(len(all_names)):
            if all_nt[ci] == 0:
                continue
            class_labels.append(all_names.get(ci, f"c{ci}"))
            for j, k in enumerate(ks):
                data_matrix[row_idx, j] = k_stats[k]["per_class_mean"][ci]
            row_idx += 1

        im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"K={k}" for k in ks])
        ax.set_yticks(range(len(class_labels)))
        ax.set_yticklabels(class_labels, fontsize=8)
        ax.set_title("Per-class Recall vs K-shot")
        plt.colorbar(im, ax=ax, label="Recall")
        # Annotate cells
        for i in range(data_matrix.shape[0]):
            for j in range(data_matrix.shape[1]):
                v = data_matrix[i, j]
                color = "white" if v < 0.4 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
        fig.tight_layout()
        fig.savefig(out_dir / "fewshot_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  Plots saved to {out_dir}/")
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plots")

    # Save CSV reports
    # Summary CSV
    csv_path = out_dir / "fewshot_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "macro_recall_mean", "macro_recall_std"])
        for k in args.k_shots:
            s = k_stats[k]
            writer.writerow([k, f"{s['macro_recall_mean']:.6f}", f"{s['macro_recall_std']:.6f}"])
        writer.writerow(["AVG", f"{avg_recall:.6f}", ""])

    # Per-class CSV
    csv_path2 = out_dir / "fewshot_per_class.csv"
    with open(csv_path2, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["class_id", "class_name", "n_GT_val"]
        for k in args.k_shots:
            header.extend([f"K{k}_mean", f"K{k}_std"])
        writer.writerow(header)
        for ci in range(len(all_names)):
            if all_nt[ci] == 0:
                continue
            row = [ci, all_names.get(ci, f"c{ci}"), int(all_nt[ci])]
            for k in args.k_shots:
                row.append(f"{k_stats[k]['per_class_mean'][ci]:.6f}")
                row.append(f"{k_stats[k]['per_class_std'][ci]:.6f}")
            writer.writerow(row)

    # Text report
    txt_path = out_dir / "fewshot_report.txt"
    with open(txt_path, "w") as f:
        f.write("Few-Shot Evaluation Report\n")
        f.write(f"{'='*70}\n")
        f.write(f"Pretrained: {args.weights}\n")
        f.write(f"K-shots: {args.k_shots}\n")
        f.write(f"Seeds per K: {args.seeds}\n")
        f.write(f"Epochs: {args.epochs}, freeze={args.freeze}, lr={args.lr}\n")
        f.write(f"imgsz: {args.imgsz}, batch: {args.batch}\n\n")

        f.write(f"SCORING ITEM (3): Few-shot average recall\n")
        f.write(f"{'-'*40}\n")
        for k in args.k_shots:
            s = k_stats[k]
            f.write(f"  K={k:2d}: R = {s['macro_recall_mean']:.4f} ± {s['macro_recall_std']:.4f}\n")
        f.write(f"  AVG:  R = {avg_recall:.4f}\n\n")

        f.write(f"Per-class detail (mean ± std over {args.seeds} seeds):\n")
        f.write(f"{'-'*70}\n")
        hdr = f"{'Class':>12s} {'n_GT':>5s}"
        for k in args.k_shots:
            hdr += f" {'K='+str(k):>14s}"
        f.write(hdr + "\n")
        for ci in range(len(all_names)):
            if all_nt[ci] == 0:
                continue
            cname = all_names.get(ci, f"c{ci}")
            row = f"{cname:>12s} {int(all_nt[ci]):5d}"
            for k in args.k_shots:
                m = k_stats[k]['per_class_mean'][ci]
                s = k_stats[k]['per_class_std'][ci]
                row += f" {m:6.3f}±{s:.3f}  "
            f.write(row + "\n")

    # Cleanup tmp
    tmp_dir = out_dir / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"\n{'='*70}")
    print(f"  DONE. Reports saved to: {out_dir}/")
    print(f"  Scoring item (3) value: {avg_recall:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
