from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None


TIANCHI_DEFECT_CLASSES = [
    "破洞", "水渍", "油污", "色斑", "纬缩",
    "经缩", "松经", "紧经", "吊经", "粗经",
    "粗纬", "浆斑", "整经结", "星跳", "跳花",
]

CLASS_NAME_TO_ID = {name: i for i, name in enumerate(TIANCHI_DEFECT_CLASSES)}


def get_train_transforms(img_size: int = 640):
    if A is None:
        return None
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(p=0.2),
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_ids"], min_visibility=0.1),
    )


def get_val_transforms(img_size: int = 640):
    if A is None:
        return None
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_ids"], min_visibility=0.1),
    )


class TianchiFabricDataset(Dataset):
    """Tianchi fabric defect dataset in YOLO format.

    Expects directory structure::

        root/
            images/
                img_001.jpg
                img_002.jpg
            labels/
                img_001.txt   (class_id cx cy w h, normalized)
                img_002.txt
    """

    def __init__(self, root: str | Path, split: str = "train", img_size: int = 640, transforms=None):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split

        if not self.img_dir.exists():
            self.img_dir = self.root / "images"
            self.lbl_dir = self.root / "labels"

        self.img_paths = sorted(self.img_dir.glob("*.jpg")) + sorted(self.img_dir.glob("*.png"))
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def _load_label(self, img_path: Path) -> tuple[list, list]:
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        bboxes, class_ids = [], []
        if not lbl_path.exists():
            return bboxes, class_ids
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                bboxes.append([cx, cy, w, h])
                class_ids.append(cid)
        return bboxes, class_ids

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        bboxes, class_ids = self._load_label(img_path)

        if self.transforms is not None and A is not None:
            augmented = self.transforms(image=img, bboxes=bboxes, class_ids=class_ids)
            img = augmented["image"]
            bboxes = augmented["bboxes"]
            class_ids = augmented["class_ids"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        targets = torch.zeros((len(bboxes), 6))
        if len(bboxes) > 0:
            targets[:, 0] = 0
            targets[:, 1] = torch.tensor(class_ids, dtype=torch.long)
            targets[:, 2:6] = torch.tensor(bboxes, dtype=torch.float32)

        return img, targets, str(img_path)


def collate_fn(batch):
    images, targets, paths = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.cat(targets, dim=0)
    return images, targets, list(paths)


def create_dataloaders(
    data_root: str | Path,
    img_size: int = 640,
    batch_size: int = 8,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TianchiFabricDataset(data_root, split="train", img_size=img_size, transforms=get_train_transforms(img_size))
    val_ds = TianchiFabricDataset(data_root, split="val", img_size=img_size, transforms=get_val_transforms(img_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader


def convert_xml_to_yolo(xml_path: Path, output_dir: Path, class_map: dict[str, int] | None = None):
    """Convert Pascal-VOC-style XML annotation to YOLO txt format."""
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
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        cx = ((xmin + xmax) / 2) / w
        cy = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / (xml_path.stem + ".txt"), "w") as f:
        f.write("\n".join(lines))


def convert_json_to_yolo(json_path: Path, output_dir: Path, class_map: dict[str, int] | None = None):
    """Convert COCO-style JSON annotation to YOLO txt format."""
    if class_map is None:
        class_map = CLASS_NAME_TO_ID

    with open(json_path) as f:
        data = json.load(f)

    img_map = {img["id"]: img for img in data["images"]}
    cat_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    anns_by_image: dict[int, list] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    output_dir.mkdir(parents=True, exist_ok=True)
    for img_id, img_info in img_map.items():
        fname = Path(img_info["file_name"]).stem
        w, h = img_info["width"], img_info["height"]
        lines = []
        for ann in anns_by_image.get(img_id, []):
            cat_name = cat_map.get(ann["category_id"], "")
            if cat_name not in class_map:
                continue
            cid = class_map[cat_name]
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            bw /= w
            bh /= h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(output_dir / (fname + ".txt"), "w") as f:
            f.write("\n".join(lines))


def split_dataset(
    src_images: Path,
    src_labels: Path,
    dst_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """Split images + labels into train/val/test directories."""
    random.seed(seed)
    all_imgs = sorted(list(src_images.glob("*.jpg")) + list(src_images.glob("*.png")))
    random.shuffle(all_imgs)

    n = len(all_imgs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

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
            lbl_path = src_labels / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

    return {k: len(v) for k, v in splits.items()}


def generate_data_yaml(data_root: Path, output_path: Path, nc: int = 15):
    """Generate Ultralytics-compatible data.yaml."""
    import yaml

    data = {
        "path": str(data_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": nc,
        "names": TIANCHI_DEFECT_CLASSES,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return output_path
