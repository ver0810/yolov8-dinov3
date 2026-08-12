import cv2
import numpy as np
from pathlib import Path
import random
from collections import defaultdict
from ultralytics import YOLO

MODEL_PATH = "outputs/runs/004_yolo11s/weights/best.pt"
TEST_IMG_DIR = "/home/ancheng/dataset/dataset_split/test/images"
TEST_LBL_DIR = "/home/ancheng/dataset/dataset_split/test/labels"
OUT_DIR = Path("outputs/inference_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAME_MAP = {
    0:  "bqc",  1:  "cf",   2:  "chy",  3:  "dmg",
    4:  "hs",   5:  "hw",   6:  "lj",   7:  "mh",
    8:  "sh",   9:  "sw",   10: "tss",  11: "xw",
    12: "yq",   13: "zf",   14: "zmty", 15: "zy",
    16: "zyc",
}

# Select diverse images: one per class, preferring images with single class
selected = [
    # (file_stem, primary_class, desc)
    ("12__0812", 0, "bqc"),          # 不良情况 - poor condition
    ("3461__0724", 1, "cf"),         # 成分 - composition
    ("1465_XBW_20250213", 2, "chy"), # 差异 - difference
    ("133__XBW_20241116", 3, "dmg"), # 损坏 - damage
    ("1106__0921", 4, "hs"),         # 灰色 - grey
    ("1086__XBW_20241116", 5, "hw"), # 花纹 - pattern
    ("1126_DHW_20250103", 6, "lj"),  # 聚集 - clustering
    ("2338_DHW_20250103before", 7, "mh"), # 模糊 - blur
    ("1403_DHW_20250103before", 8, "sh"), # 水痕 - watermark
    ("1821_LZW_20250208", 9, "sw"),  # 缩水 - shrinkage
    ("LZW_20260124_152", 10, "tss"), # 脱色 - fading
    ("1045_DHW_20250103", 11, "xw"), # 纤维 - fiber
    ("1099__20241017", 12, "yq"),    # 油污 - oil stain
    ("1721_DHW_20250103before", 13, "zf"), # 折痕 - crease
    ("1199_DHW_20250103", 14, "zmty"), # 正面套印 - front register
    ("274__0812", 15, "zy"),         # 折印 - fold mark
    ("1086__XBW_20241116", 16, "zyc"), # 折印差异 - fold diff
]

# Remove duplicates (same image used for multiple classes)
seen = set()
unique = []
for stem, cls, abbr in selected:
    if stem not in seen:
        unique.append((stem, cls, abbr))
    seen.add(stem)
print(f"Selected {len(unique)} unique images")

# Load model
print("Loading model...")
model = YOLO(MODEL_PATH)

# Colors per class
colors = {}
for cls in range(17):
    colors[cls] = (
        random.randint(80, 220),
        random.randint(80, 220),
        random.randint(80, 220),
    )
# Fix seed for reproducibility
random.seed(42)
for cls in range(17):
    colors[cls] = (
        random.randint(80, 220),
        random.randint(80, 220),
        random.randint(80, 220),
    )


def draw_yolo_boxes(img, label_path, img_w, img_h, draw_class_id=True):
    """Draw ground truth boxes on image (YOLO normalized format)."""
    if not Path(label_path).exists():
        return img
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            bgr = colors.get(cls_id, (0, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
            label = NAME_MAP.get(cls_id, str(cls_id))
            if draw_class_id:
                label = f"{label}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), bgr, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def draw_pred_boxes(img, results, draw_conf=True):
    """Draw prediction boxes from YOLO results."""
    if results[0].boxes is None:
        return img
    boxes = results[0].boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        bgr = colors.get(cls_id, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
        label = NAME_MAP.get(cls_id, str(cls_id))
        if draw_conf:
            label = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), bgr, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


H, W = 640, 640  # resize for display

for stem, cls_id, abbr in unique:
    img_path = Path(TEST_IMG_DIR) / f"{stem}.png"
    label_path = Path(TEST_LBL_DIR) / f"{stem}.txt"

    if not img_path.exists():
        print(f"  SKIP {stem}: image not found")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img_h, img_w = img.shape[:2]

    # Left: Ground Truth
    img_gt = img.copy()
    img_gt = draw_yolo_boxes(img_gt, str(label_path), img_w, img_h)

    # Right: Model Prediction
    results = model(str(img_path), conf=0.25, iou=0.45, verbose=False)
    img_pred = img.copy()
    img_pred = draw_pred_boxes(img_pred, results)

    # Resize for display
    img_gt = cv2.resize(img_gt, (W, H))
    img_pred = cv2.resize(img_pred, (W, H))

    # Add titles
    side = np.hstack([img_gt, img_pred])

    # Header bar
    header = np.ones((40, W * 2, 3), dtype=np.uint8) * 40
    cv2.putText(header, "Ground Truth", (W // 2 - 80, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(header, "YOLO11s Prediction", (W + W // 2 - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.line(header, (W, 0), (W, 40), (80, 80, 80), 2)

    combined = np.vstack([header, side])

    out_path = OUT_DIR / f"{abbr}_{stem}.png"
    cv2.imwrite(str(out_path), combined)
    print(f"  Saved: {out_path}")

print(f"\nDone. Images saved to {OUT_DIR}/")
