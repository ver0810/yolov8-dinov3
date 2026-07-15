"""Run inference on images and visualize/save detection results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_dino.data.tianchi import TIANCHI_DEFECT_CLASSES
from yolo_dino.models.yolo_dinov3 import YOLODINOv3, YOLODINOv3Hybrid
from yolo_dino.utils.helpers import get_device


COLORS = np.random.randint(0, 255, size=(len(TIANCHI_DEFECT_CLASSES), 3), dtype=int)


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


def preprocess(img_path: str, img_size: int, device: torch.device) -> torch.Tensor:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device), img


def draw_detections(img: np.ndarray, detections: list[dict], img_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale_x, scale_y = w / img_size, h / img_size

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
        x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
        label = det["label"]
        score = det["score"]
        color = tuple(int(c) for c in COLORS[label % len(COLORS)])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{TIANCHI_DEFECT_CLASSES[label]} {score:.2f}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


@torch.no_grad()
def predict(args):
    device = get_device()
    model = build_model(args, device)

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))
    else:
        print(f"Input path not found: {input_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        tensor, orig_img = preprocess(str(img_path), args.img_size, device)
        preds = model(tensor)

        if isinstance(preds, (list, tuple)):
            pred_tensor = preds[0] if preds[0].dim() == 3 else preds[1]
        else:
            pred_tensor = preds

        pred_i = pred_tensor[0]
        scores = pred_i[:, 4] if pred_i.shape[-1] > 4 else pred_i[:, -1]
        mask = scores > args.conf_threshold
        pred_filtered = pred_i[mask]

        detections = []
        for det in pred_filtered:
            box = det[:4].cpu().numpy()
            score = float(det[4].cpu())
            label = int(det[5].cpu()) if det.shape[0] > 5 else 0
            detections.append({"box": box, "score": score, "label": label})

        result_img = draw_detections(orig_img.copy(), detections, args.img_size)
        out_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_path), result_img)
        print(f"{img_path.name}: {len(detections)} detections -> {out_path}")


def predict_baseline(args):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.predict(
        source=args.input,
        imgsz=args.img_size,
        conf=args.conf_threshold,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        save=True,
        project=str(args.output_dir),
        name="predict",
    )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with YOLO-DINOv3")
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "full", "hybrid"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image file or directory")
    parser.add_argument("--output-dir", type=str, default="outputs/predict")

    parser.add_argument("--nc", type=int, default=15)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-threshold", type=float, default=0.25)

    parser.add_argument("--dinov3-size", type=str, default="vits")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dinov3-path", type=str, default="third_party/dinov3")
    parser.add_argument("--yolo-width", type=float, default=1.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "baseline":
        predict_baseline(args)
    else:
        predict(args)
