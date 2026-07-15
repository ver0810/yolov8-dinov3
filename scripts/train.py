"""Train YOLOv8 baseline or YOLO-DINOv3 fusion models on Tianchi fabric defect dataset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_dino.data.tianchi import create_dataloaders, generate_data_yaml, TIANCHI_DEFECT_CLASSES
from yolo_dino.models.yolo_dinov3 import YOLODINOv3, YOLODINOv3Hybrid
from yolo_dino.utils.helpers import get_device, load_yaml, model_summary, set_seed


def train_baseline(args):
    """Train vanilla YOLOv8 via Ultralytics API."""
    from ultralytics import YOLO

    data_yaml = Path(args.data_config)
    if not data_yaml.exists():
        generate_data_yaml(Path(args.data_root), data_yaml, nc=args.nc)

    model = YOLO(args.yolo_model or "yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        project=str(args.output_dir),
        name=args.exp_name,
        patience=args.patience,
        optimizer="AdamW",
        lr0=args.lr,
        exist_ok=True,
    )
    return results


def build_model(args, device: torch.device) -> nn.Module:
    common = dict(
        nc=args.nc,
        dinov3_size=args.dinov3_size,
        patch_size=args.patch_size,
        freeze_backbone=not args.finetune_backbone,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        pretrained=True,
        dinov3_path=args.dinov3_path,
    )
    if args.mode == "full":
        model = YOLODINOv3(**common)
    elif args.mode == "hybrid":
        model = YOLODINOv3Hybrid(**common, yolo_width=args.yolo_width)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    model = model.to(device)
    print(model_summary(model))
    return model


def train_fusion(args):
    """Train YOLO-DINOv3 fusion model."""
    device = get_device()
    set_seed(args.seed)

    model = build_model(args, device)

    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, targets, _ in pbar:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)

            batch_dict = {"cls": targets[:, 1], "bboxes": targets[:, 2:6]}
            if hasattr(model, "loss"):
                loss_dict = model.loss(preds, batch_dict)
                loss = loss_dict if isinstance(loss_dict, torch.Tensor) else sum(loss_dict.values())
            else:
                loss = preds if isinstance(preds, torch.Tensor) else preds[0]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"  -> Saved best model (loss={best_loss:.4f})")

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f"epoch_{epoch}.pt")

    torch.save(model.state_dict(), output_dir / "last.pt")
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO-DINOv3 models")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "full", "hybrid"])
    parser.add_argument("--exp-name", type=str, default="exp001")
    parser.add_argument("--data-root", type=str, default="data/yolo_format")
    parser.add_argument("--data-config", type=str, default="configs/tianchi.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/runs")

    parser.add_argument("--nc", type=int, default=15)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)

    parser.add_argument("--dinov3-size", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dinov3-path", type=str, default="third_party/dinov3")
    parser.add_argument("--finetune-backbone", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--yolo-width", type=float, default=1.0)

    parser.add_argument("--yolo-model", type=str, default=None, help="YOLOv8 model name for baseline (e.g. yolov8n.pt)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "baseline":
        train_baseline(args)
    else:
        train_fusion(args)
