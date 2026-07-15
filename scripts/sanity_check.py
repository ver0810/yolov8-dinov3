"""Sanity check: verify environment, imports, model instantiation, and MPS availability."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def check_imports():
    print("[1/6] Checking imports...")
    import torch
    import torchvision
    import ultralytics
    import numpy as np
    import cv2
    import yaml

    print(f"  torch       : {torch.__version__}")
    print(f"  torchvision : {torchvision.__version__}")
    print(f"  ultralytics : {ultralytics.__version__}")
    print(f"  numpy       : {np.__version__}")
    print(f"  cv2         : {cv2.__version__}")
    print("  OK")


def check_device():
    print("\n[2/6] Checking device...")
    import torch

    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available()
    print(f"  CUDA available : {cuda_ok}")
    print(f"  MPS  available : {mps_ok}")

    if mps_ok:
        device = torch.device("mps")
        x = torch.randn(1, 3, 64, 64, device=device)
        y = x * 2
        print(f"  MPS tensor test: shape={y.shape}, device={y.device}")
    print("  OK")


def check_dinov3_backbone():
    print("\n[3/6] Checking DINOv3Backbone instantiation (random weights)...")
    import torch
    from yolo_dino.models.backbones.dinov3_backbone import DINOv3Backbone

    backbone = DINOv3Backbone(model_size="vits", patch_size=16, freeze=True, pretrained=False)
    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        out = backbone(x)
    print(f"  Input  : {x.shape}")
    print(f"  Output : P4={out[0].shape}, P8={out[1].shape}, P16={out[2].shape}")
    print(f"  out_channels : {backbone.out_channels}")
    print("  OK")


def check_fusion_neck():
    print("\n[4/6] Checking FusionNeck...")
    import torch
    from yolo_dino.models.necks.fusion_neck import FusionNeck

    neck = FusionNeck(in_channels=192, out_channels=(256, 256, 256))
    feats = (torch.randn(2, 192, 80, 80), torch.randn(2, 192, 40, 40), torch.randn(2, 192, 20, 20))
    out = neck(feats)
    print(f"  Output : P4={out[0].shape}, P8={out[1].shape}, P16={out[2].shape}")
    print("  OK")


def check_full_model():
    print("\n[5/6] Checking YOLODINOv3 full model...")
    import torch
    from yolo_dino.models.yolo_dinov3 import YOLODINOv3, YOLODINOv3Hybrid

    model = YOLODINOv3(nc=15, dinov3_size="vits", pretrained=False)
    x = torch.randn(1, 3, 640, 640)
    model.eval()
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (list, tuple)):
        print(f"  Output type: list of {len(out)} tensors")
        for i, o in enumerate(out):
            print(f"    [{i}] shape={o.shape}")
    else:
        print(f"  Output shape: {out.shape}")

    print("\n  Checking YOLODINOv3Hybrid...")
    model_h = YOLODINOv3Hybrid(nc=15, dinov3_size="vits", pretrained=False)
    model_h.eval()
    with torch.no_grad():
        out_h = model_h(x)
    if isinstance(out_h, (list, tuple)):
        print(f"  Hybrid output type: list of {len(out_h)} tensors")
    else:
        print(f"  Hybrid output shape: {out_h.shape}")
    print("  OK")


def check_dataset():
    print("\n[6/6] Checking TianchiFabricDataset...")
    from yolo_dino.data.tianchi import TianchiFabricDataset, TIANCHI_DEFECT_CLASSES

    print(f"  Classes ({len(TIANCHI_DEFECT_CLASSES)}): {TIANCHI_DEFECT_CLASSES}")
    print("  (Skipping actual data loading — no dataset downloaded yet)")
    print("  OK")


if __name__ == "__main__":
    print("=" * 60)
    print("YOLO-DINO Sanity Check")
    print("=" * 60)
    check_imports()
    check_device()
    check_dinov3_backbone()
    check_fusion_neck()
    check_full_model()
    check_dataset()
    print("\n" + "=" * 60)
    print("All checks passed!")
    print("=" * 60)
