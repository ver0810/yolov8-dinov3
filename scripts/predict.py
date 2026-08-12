#!/usr/bin/env python3
"""Run prediction with a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True, help="Image, dir, or glob")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--project", type=str, default=str(ROOT / "outputs/predict"))
    p.add_argument("--name", type=str, default="exp")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    kw = dict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    if args.device is not None:
        kw["device"] = args.device
    model.predict(**kw)


if __name__ == "__main__":
    main()
