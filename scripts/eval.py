"""Evaluate trained YOLO-DINOv3 models on the validation/test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_dino.data.tianchi import TianchiFabricDataset, get_val_transforms, TIANCHI_DEFECT_CLASSES
from yolo_dino.models.yolo_dinov3 import YOLODINOv3, YOLODINOv3Hybrid
from yolo_dino.utils.helpers import get_device, set_seed
from yolo_dino.utils.metrics import compute_map, save_metrics


def build_model(args, device: torch.device) -> torch.nn.Module:
    common = dict(
        nc=args.nc,
        dinov3_size=args.dinov3_size,
        patch_size=args.patch_size,
        freeze_backbone=True,
        pretrained=False,
        dinov3_path=args.dinov3_path,
    )
    if args.mode == "full":
        model = YOLODINOv3(**common)
    elif args.mode == "hybrid":
        model = YOLODINOv3Hybrid(**common, yolo_width=args.yolo_width)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def evaluate_baseline(args):
    """Evaluate a vanilla YOLOv8 model via Ultralytics API."""
    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.val(
        data=args.data_config,
        imgsz=args.img_size,
        batch=args.batch_size,
        device="mps" if torch.backends.mps.is_available() else "cpu",
    )
    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }
    print(f"Baseline Results: mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50-95']:.4f}")
    save_metrics(metrics, Path(args.output_dir) / args.exp_name / "metrics.json")
    return metrics


@torch.no_grad()
def evaluate_fusion(args):
    device = get_device()
    set_seed(args.seed)

    model = build_model(args, device)

    val_ds = TianchiFabricDataset(
        args.data_root, split="val", img_size=args.img_size, transforms=get_val_transforms(args.img_size)
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    all_predictions = []
    all_ground_truths = []

    for images, targets, _ in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        preds = model(images)

        if isinstance(preds, (list, tuple)):
            pred_tensor = preds[0] if preds[0].dim() == 3 else preds[1]
        else:
            pred_tensor = preds

        for i in range(images.shape[0]):
            pred_i = pred_tensor[i] if pred_tensor.dim() == 3 else pred_tensor
            scores = pred_i[:, 4] if pred_i.shape[-1] > 4 else pred_i[:, -1]
            conf_mask = scores > args.conf_threshold
            pred_filtered = pred_i[conf_mask]

            if pred_filtered.shape[0] > 0:
                boxes = pred_filtered[:, 0:4].cpu().numpy()
                scores_np = pred_filtered[:, 4].cpu().numpy()
                labels = pred_filtered[:, 5].long().cpu().numpy() if pred_filtered.shape[1] > 5 else np.zeros(len(boxes), dtype=int)
            else:
                boxes = np.zeros((0, 4))
                scores_np = np.zeros(0)
                labels = np.zeros(0, dtype=int)

            all_predictions.append({"boxes": boxes, "scores": scores_np, "labels": labels})

        target_list = targets.tolist() if isinstance(targets, torch.Tensor) else targets
        idx = 0
        for _ in range(images.shape[0]):
            gt_boxes, gt_labels = [], []
            while idx < len(target_list) and target_list[idx][0] == 0:
                gt_labels.append(int(target_list[idx][1]))
                gt_boxes.append(target_list[idx][2:6])
                idx += 1
            if gt_boxes:
                gt_boxes_np = np.array(gt_boxes)
                gt_boxes_xyxy = np.zeros_like(gt_boxes_np)
                gt_boxes_xyxy[:, 0] = (gt_boxes_np[:, 0] - gt_boxes_np[:, 2] / 2) * args.img_size
                gt_boxes_xyxy[:, 1] = (gt_boxes_np[:, 1] - gt_boxes_np[:, 3] / 2) * args.img_size
                gt_boxes_xyxy[:, 2] = (gt_boxes_np[:, 0] + gt_boxes_np[:, 2] / 2) * args.img_size
                gt_boxes_xyxy[:, 3] = (gt_boxes_np[:, 1] + gt_boxes_np[:, 3] / 2) * args.img_size
            else:
                gt_boxes_xyxy = np.zeros((0, 4))
            all_ground_truths.append({"boxes": gt_boxes_xyxy, "labels": np.array(gt_labels, dtype=int)})

    results_50 = compute_map(all_predictions, all_ground_truths, iou_threshold=0.5, num_classes=args.nc)
    results_75 = compute_map(all_predictions, all_ground_truths, iou_threshold=0.75, num_classes=args.nc)

    metrics = {
        "mAP50": results_50["mAP"],
        "mAP75": results_75["mAP"],
        "per_class_AP_50": results_50["per_class_AP"],
        "per_class_AP_75": results_75["per_class_AP"],
        "class_names": TIANCHI_DEFECT_CLASSES,
    }

    print(f"\n{'='*50}")
    print(f"mAP@0.50 : {metrics['mAP50']:.4f}")
    print(f"mAP@0.75 : {metrics['mAP75']:.4f}")
    print(f"{'='*50}")
    for cid, ap in results_50["per_class_AP"].items():
        name = TIANCHI_DEFECT_CLASSES[cid] if cid < len(TIANCHI_DEFECT_CLASSES) else f"class_{cid}"
        print(f"  {name:<10s} : {ap:.4f}")

    save_metrics(metrics, Path(args.output_dir) / args.exp_name / "metrics.json")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO-DINOv3 models")
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "full", "hybrid"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--exp-name", type=str, default="eval")
    parser.add_argument("--data-root", type=str, default="data/yolo_format")
    parser.add_argument("--data-config", type=str, default="configs/tianchi.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis")

    parser.add_argument("--nc", type=int, default=15)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dinov3-size", type=str, default="vits")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dinov3-path", type=str, default="third_party/dinov3")
    parser.add_argument("--yolo-width", type=float, default=1.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "baseline":
        evaluate_baseline(args)
    else:
        evaluate_fusion(args)
