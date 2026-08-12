#!/usr/bin/env python3
"""Validate a trained checkpoint with Ultralytics."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--data", type=str, default=str(ROOT / "configs/tianchi.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--split", type=str, default="val")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    kw = dict(data=args.data, imgsz=args.imgsz, batch=args.batch, split=args.split)
    if args.device is not None:
        kw["device"] = args.device
    metrics = model.val(**kw)
    print(metrics)


if __name__ == "__main__":
    main()
