import json
from pathlib import Path
from PIL import Image

ROOT = Path("outputs/dataset_095_inplace")
OUT = Path("/tmp/coco_095")
OUT.mkdir(parents=True, exist_ok=True)

NAMES = ["lj","bd","heidian","wy","zyc","zmty","cq","zj","bj","jt","yy","bmss","zy","pd","HD","cy","jiaodai"]

def convert(split):
    img_dir = ROOT / split / "images"
    lbl_dir = ROOT / split / "labels"
    imgs = sorted(img_dir.glob("*"))
    images, annotations = [], []
    ann_id = 1
    for img_id, p in enumerate(imgs, 1):
        w, h = Image.open(p).size
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lbl = lbl_dir / f"{p.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0])); cx, cy, bw, bh = map(float, parts[1:5])
            x = (cx - bw/2) * w; y = (cy - bh/2) * h; bw *= w; bh *= h
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": cid+1, "bbox": [x,y,bw,bh], "area": bw*bh, "iscrowd": 0})
            ann_id += 1
    categories = [{"id": i+1, "name": n} for i, n in enumerate(NAMES)]
    data = {"images": images, "annotations": annotations, "categories": categories}
    out = OUT / f"{split}.json"
    out.write_text(json.dumps(data))
    print(f"{split}: {len(images)} images, {len(annotations)} ann -> {out} ({out.stat().st_size/1024:.1f}KB)")

for s in ("train","val","test"):
    convert(s)
