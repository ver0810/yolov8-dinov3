#!/usr/bin/env python3
"""Prepare Tianchi fabric defect dataset: convert annotations and split train/val/test."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

TIANCHI_DEFECT_CLASSES = [
    "破洞",
    "水渍",
    "油污",
    "色斑",
    "纬缩",
    "经缩",
    "松经",
    "紧经",
    "吊经",
    "粗经",
    "粗纬",
    "浆斑",
    "整经结",
    "星跳",
    "跳花",
]
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(TIANCHI_DEFECT_CLASSES)}


def convert_xml_to_yolo(xml_path: Path, output_dir: Path, class_map: dict[str, int] | None = None):
    if class_map is None:
        class_map = CLASS_NAME_TO_ID
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in class_map:
            continue
        cid = class_map[name]
        box = obj.find("bndbox")
        xmin, ymin = float(box.find("xmin").text), float(box.find("ymin").text)
        xmax, ymax = float(box.find("xmax").text), float(box.find("ymax").text)
        cx, cy = ((xmin + xmax) / 2) / w, ((ymin + ymax) / 2) / h
        bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{xml_path.stem}.txt").write_text("\n".join(lines))


def convert_json_to_yolo(json_path: Path, output_dir: Path, class_map: dict[str, int] | None = None):
    if class_map is None:
        class_map = CLASS_NAME_TO_ID
    data = json.loads(json_path.read_text())
    img_map = {img["id"]: img for img in data["images"]}
    cat_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    anns: dict[int, list] = {}
    for ann in data["annotations"]:
        anns.setdefault(ann["image_id"], []).append(ann)
    output_dir.mkdir(parents=True, exist_ok=True)
    for img_id, info in img_map.items():
        fname = Path(info["file_name"]).stem
        w, h = info["width"], info["height"]
        lines = []
        for ann in anns.get(img_id, []):
            cat_name = cat_map.get(ann["category_id"], "")
            if cat_name not in class_map:
                continue
            cid = class_map[cat_name]
            x, y, bw, bh = ann["bbox"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (output_dir / f"{fname}.txt").write_text("\n".join(lines))


def split_dataset(src_images: Path, src_labels: Path, dst_root: Path, train_ratio=0.7, val_ratio=0.2, seed=42):
    random.seed(seed)
    all_imgs = sorted(list(src_images.glob("*.jpg")) + list(src_images.glob("*.png")))
    random.shuffle(all_imgs)
    n = len(all_imgs)
    n_train, n_val = int(n * train_ratio), int(n * val_ratio)
    splits = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train : n_train + n_val],
        "test": all_imgs[n_train + n_val :],
    }
    for split_name, imgs in splits.items():
        img_dst = dst_root / "images" / split_name
        lbl_dst = dst_root / "labels" / split_name
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            shutil.copy2(img_path, img_dst / img_path.name)
            lbl = src_labels / f"{img_path.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, lbl_dst / lbl.name)
    return {k: len(v) for k, v in splits.items()}


def write_data_yaml(data_root: Path, output_path: Path, nc: int = 15):
    payload = {
        "path": str(data_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": nc,
        "names": TIANCHI_DEFECT_CLASSES,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/yolo_format")
    p.add_argument("--format", choices=["auto", "xml", "json", "yolo"], default="auto")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--data-yaml", type=Path, default=ROOT / "configs/tianchi.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    raw, out = args.raw_dir, args.output_dir
    fmt = args.format
    if fmt == "auto":
        if (raw / "Annotations").exists() or (raw / "annotations").exists():
            fmt = "xml"
        elif list(raw.glob("*.json")):
            fmt = "json"
        else:
            fmt = "yolo"

    label_out = raw / "labels_yolo"
    if fmt == "xml":
        xml_dir = raw / "Annotations" if (raw / "Annotations").exists() else raw / "annotations"
        img_dir = raw / "Images" if (raw / "Images").exists() else raw / "images"
        label_out.mkdir(parents=True, exist_ok=True)
        for xml_path in sorted(xml_dir.glob("*.xml")):
            convert_xml_to_yolo(xml_path, label_out)
        counts = split_dataset(img_dir, label_out, out, args.train_ratio, args.val_ratio)
    elif fmt == "json":
        img_dir = raw / "images" if (raw / "images").exists() else raw / "Images"
        label_out.mkdir(parents=True, exist_ok=True)
        for jp in raw.glob("*.json"):
            convert_json_to_yolo(jp, label_out)
        counts = split_dataset(img_dir, label_out, out, args.train_ratio, args.val_ratio)
    else:
        img_dir = raw / "images" if (raw / "images").exists() else raw
        lbl_dir = raw / "labels" if (raw / "labels").exists() else raw
        counts = split_dataset(img_dir, lbl_dir, out, args.train_ratio, args.val_ratio)

    write_data_yaml(out, args.data_yaml)
    print(f"format={fmt} split={counts}")
    print(f"data yaml -> {args.data_yaml}")


if __name__ == "__main__":
    main()
