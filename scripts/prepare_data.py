"""Prepare Tianchi fabric defect dataset: convert annotations and split into train/val/test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_dino.data.tianchi import (
    convert_json_to_yolo,
    convert_xml_to_yolo,
    generate_data_yaml,
    split_dataset,
    TIANCHI_DEFECT_CLASSES,
)


def prepare_from_xml(raw_dir: Path, output_dir: Path, train_ratio: float, val_ratio: float):
    """Raw data has XML annotations (Pascal VOC format)."""
    xml_dir = raw_dir / "Annotations"
    img_dir = raw_dir / "Images"
    if not xml_dir.exists():
        xml_dir = raw_dir / "annotations"
    if not img_dir.exists():
        img_dir = raw_dir / "images"

    label_out = raw_dir / "labels_yolo"
    label_out.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob("*.xml"))
    print(f"Converting {len(xml_files)} XML annotations to YOLO format...")
    for xml_path in xml_files:
        convert_xml_to_yolo(xml_path, label_out)

    print(f"Splitting dataset (train={train_ratio}, val={val_ratio})...")
    counts = split_dataset(img_dir, label_out, output_dir, train_ratio, val_ratio)
    print(f"Split counts: {counts}")


def prepare_from_json(raw_dir: Path, output_dir: Path, train_ratio: float, val_ratio: float):
    """Raw data has COCO-style JSON annotations."""
    img_dir = raw_dir / "images"
    if not img_dir.exists():
        img_dir = raw_dir / "Images"

    json_files = list(raw_dir.glob("*.json"))
    if not json_files:
        print("No JSON annotation files found in", raw_dir)
        return

    label_out = raw_dir / "labels_yolo"
    label_out.mkdir(parents=True, exist_ok=True)

    for json_path in json_files:
        print(f"Converting {json_path.name} ...")
        convert_json_to_yolo(json_path, label_out)

    print(f"Splitting dataset (train={train_ratio}, val={val_ratio})...")
    counts = split_dataset(img_dir, label_out, output_dir, train_ratio, val_ratio)
    print(f"Split counts: {counts}")


def prepare_from_yolo(raw_dir: Path, output_dir: Path, train_ratio: float, val_ratio: float):
    """Raw data already in YOLO format, just split."""
    img_dir = raw_dir / "images"
    lbl_dir = raw_dir / "labels"
    if not img_dir.exists():
        img_dir = raw_dir
    if not lbl_dir.exists():
        lbl_dir = raw_dir

    print(f"Splitting dataset (train={train_ratio}, val={val_ratio})...")
    counts = split_dataset(img_dir, lbl_dir, output_dir, train_ratio, val_ratio)
    print(f"Split counts: {counts}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Tianchi fabric defect dataset")
    parser.add_argument("--raw-dir", type=str, required=True, help="Path to raw downloaded dataset")
    parser.add_argument("--output-dir", type=str, default="data/yolo_format")
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "xml", "json", "yolo"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--nc", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    fmt = args.format
    if fmt == "auto":
        if list(raw_dir.rglob("*.xml")):
            fmt = "xml"
        elif list(raw_dir.rglob("*.json")):
            fmt = "json"
        else:
            fmt = "yolo"
    print(f"Detected annotation format: {fmt}")

    if fmt == "xml":
        prepare_from_xml(raw_dir, output_dir, args.train_ratio, args.val_ratio)
    elif fmt == "json":
        prepare_from_json(raw_dir, output_dir, args.train_ratio, args.val_ratio)
    else:
        prepare_from_yolo(raw_dir, output_dir, args.train_ratio, args.val_ratio)

    yaml_path = generate_data_yaml(output_dir, Path("configs/tianchi.yaml"), nc=args.nc)
    print(f"\nGenerated data config: {yaml_path}")
    print(f"Classes ({args.nc}): {TIANCHI_DEFECT_CLASSES}")
    print("\nDone! You can now run:")
    print("  pixi run python scripts/train.py --mode baseline --data-root data/yolo_format")
