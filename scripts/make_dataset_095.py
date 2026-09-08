#!/usr/bin/env python3
"""Build train/val/test dataset from inplace pool + v1_train.

Composition (compliant, zero leakage validated):
- train = v1_train 1445 + dataset_v1_pool_small_inplace 580 = 2025 images
  (580 = 145 source x4 variants: orig/c30/clahe/bw, w median 0.0054 vs val 0.0063)
- val   = v1_val 181 (exact)
- test  = v1_test 181
All sources from v1_train only (pool stems ∩ v1_val =0 verified).

Usage:
    uv run python scripts/make_dataset_095.py
Outputs: outputs/dataset_095_inplace/{train,val,test}/{images,labels}, data.yaml
         outputs/kaggle_staging/dataset-095-inplace/{data.zip, dataset-metadata.json}
"""

from __future__ import annotations
import shutil
import zipfile
import json
from pathlib import Path

ROOT = Path("/home/ancheng/Code/yolov8-dinov3")
V1 = Path("/home/ancheng/dataset/dataset_split")
POOL = ROOT / "outputs/dataset_v1_pool_small_inplace"
OUT = ROOT / "outputs/dataset_095_inplace"
STAGE = ROOT / "outputs/kaggle_staging/dataset-095-inplace"

NAMES = ["lj","bd","heidian","wy","zyc","zmty","cq","zj","bj","jt","yy","bmss","zy","pd","HD","cy","jiaodai"]

def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src)

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for split in ("train","val","test"):
        (OUT/split/"images").mkdir(parents=True, exist_ok=True)
        (OUT/split/"labels").mkdir(parents=True, exist_ok=True)
    for split in ("val","test"):
        for img in (V1/split/"images").glob("*"):
            link_or_copy(img, OUT/split/"images"/img.name)
        for lbl in (V1/split/"labels").glob("*.txt"):
            link_or_copy(lbl, OUT/split/"labels"/lbl.name)
        print(f"{split}: {len(list((OUT/split/'images').glob('*')))} images")
    for img in (V1/"train"/"images").glob("*"):
        link_or_copy(img, OUT/"train"/"images"/img.name)
    for lbl in (V1/"train"/"labels").glob("*.txt"):
        link_or_copy(lbl, OUT/"train"/"labels"/lbl.name)
    print(f"train v1: {len(list((OUT/'train'/'images').glob('*')))}")
    for img in (POOL/"images").glob("*.jpg"):
        shutil.copy(img, OUT/"train"/"images"/img.name)
    for lbl in (POOL/"labels").glob("*.txt"):
        shutil.copy(lbl, OUT/"train"/"labels"/lbl.name)
    print(f"train total: {len(list((OUT/'train'/'images').glob('*')))} images")
    (OUT/"data.yaml").write_text(f"""path: {OUT}
train: train/images
val: val/images
test: test/images
nc: 17
names:
{chr(10).join('  - '+n for n in NAMES)}
""")
    print(f"data.yaml: {OUT/'data.yaml'}")
    train_stems = {p.stem for p in (OUT/"train"/"images").glob("*")}
    v1v = {p.stem for p in (V1/"val"/"images").glob("*.png")}
    print(f"train ∩ v1_val = {len(train_stems & v1v)} (must 0)")
    for split in ("train","val","test"):
        (STAGE/split/"images").mkdir(parents=True, exist_ok=True)
        (STAGE/split/"labels").mkdir(parents=True, exist_ok=True)
    for split in ("train","val","test"):
        for img in (OUT/split/"images").glob("*"):
            src = img.resolve() if img.is_symlink() else img
            shutil.copy(src, STAGE/split/"images"/img.name)
        for lbl in (OUT/split/"labels").glob("*.txt"):
            src = lbl.resolve() if lbl.is_symlink() else lbl
            shutil.copy(src, STAGE/split/"labels"/lbl.name)
    (STAGE/"data.yaml").write_text(f"""train: train/images
val: val/images
test: test/images
nc: 17
names:
{chr(10).join('  - '+n for n in NAMES)}
""")
    (STAGE/"dataset-metadata.json").write_text(json.dumps({
        "title": "v1 train + inplace small pool (size-aligned bd/heidian)",
        "id": "ver0810/dataset-095-inplace",
        "licenses": [{"name": "cc0"}]
    }, indent=2))
    print(f"Kaggle staging: {STAGE}")
    zip_path = STAGE/"data.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for p in STAGE.rglob("*"):
            if p.is_file() and p.name != "data.zip":
                z.write(p, p.relative_to(STAGE))
    print(f"zip: {zip_path} ({zip_path.stat().st_size/1024/1024:.1f} MB)")

if __name__ == "__main__":
    main()
