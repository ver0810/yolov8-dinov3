from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes in xyxy format."""
    x1 = np.maximum(box_a[:, 0:1], box_b[:, 0])
    y1 = np.maximum(box_a[:, 1:2], box_b[:, 1])
    x2 = np.minimum(box_a[:, 2:3], box_b[:, 2])
    y2 = np.minimum(box_a[:, 3:4], box_b[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box_a[:, 2:3] - box_a[:, 0:1]) * (box_a[:, 3:4] - box_a[:, 1:2])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-6)


def yolo_to_xyxy(bboxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert YOLO format (cx, cy, w, h) normalized to xyxy pixel coords."""
    cx, cy, bw, bh = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    return np.stack([x1, y1, x2, y2], axis=1)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using 101-point interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    thresholds = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for t in thresholds:
        mask = mrec >= t
        if mask.any():
            ap += np.max(mpre[mask])
    return ap / 101.0


def compute_map(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
    num_classes: int = 15,
) -> dict:
    """Compute mAP across all classes.

    Each prediction dict: {"boxes": np.ndarray (N,4) xyxy, "scores": np.ndarray (N,), "labels": np.ndarray (N,)}
    Each ground_truth dict: {"boxes": np.ndarray (M,4) xyxy, "labels": np.ndarray (M,)}
    """
    aps = {}
    for cls_id in range(num_classes):
        all_scores = []
        all_tp = []
        n_gt_total = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_mask = pred["labels"] == cls_id
            gt_mask = gt["labels"] == cls_id

            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]
            gt_boxes = gt["boxes"][gt_mask]

            n_gt_total += len(gt_boxes)

            if len(pred_boxes) == 0:
                continue

            sort_idx = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]

            matched = np.zeros(len(gt_boxes), dtype=bool)
            for i in range(len(pred_boxes)):
                if len(gt_boxes) == 0:
                    all_tp.append(0)
                    all_scores.append(pred_scores[i])
                    continue
                ious = compute_iou(pred_boxes[i : i + 1], gt_boxes)[0]
                best_idx = np.argmax(ious)
                if ious[best_idx] >= iou_threshold and not matched[best_idx]:
                    all_tp.append(1)
                    matched[best_idx] = True
                else:
                    all_tp.append(0)
                all_scores.append(pred_scores[i])

        if n_gt_total == 0:
            aps[cls_id] = 0.0
            continue

        all_scores = np.array(all_scores)
        all_tp = np.array(all_tp)
        sort_idx = np.argsort(-all_scores)
        all_tp = all_tp[sort_idx]

        tp_cumsum = np.cumsum(all_tp)
        fp_cumsum = np.cumsum(1 - all_tp)
        recalls = tp_cumsum / n_gt_total
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        aps[cls_id] = compute_ap(recalls, precisions)

    mAP = np.mean(list(aps.values())) if aps else 0.0
    return {"mAP": float(mAP), "per_class_AP": {int(k): float(v) for k, v in aps.items()}}


def save_metrics(metrics: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
