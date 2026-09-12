"""手动推理 / 演示脚本：DEIMv2 引擎 + ultralytics 两套权重统一入口。

用法示例见文件末尾 __main__ 帮助，或：
  # DEIMv2（103/104/105/106/107/108）
  uv run --offline --project . --with omegaconf --with pycocotools python scripts/infer_demo.py \
      --engine deim --config third_party/DEIMv2/configs/deimv2/v1_dinov3_S_v4_68ep.yml \
      --weights outputs/runs/107_deimv2_dinov3_S_v4/best_stg1.pth --limit 8
  # ultralytics（019 yolo11s / 096 RT-DETR-L ...）
  uv run python scripts/infer_demo.py \
      --weights outputs/runs/096_rtdetr_l_kaggle1280/weights/best.pt --limit 8

输出：outputs/inference_demo/<模型名>/<图名>.png —— 左 GT / 右 预测（同尺寸并排）。
"""
import argparse
import glob as glob_mod
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party/DEIMv2"))

# 17 类权威映射（顺序与 /home/ancheng/dataset/dataset_split/classes.md、
# outputs/deim_smoke/coco/val.json 的 categories 一致，勿改）
CLASSES = ["lj", "bd", "heidian", "wy", "zyc", "zmty", "cq", "zj", "bj", "jt",
           "yy", "bmss", "zy", "pd", "HD", "cy", "jiaodai"]
CN = {"lj": "垃圾", "bd": "白点", "heidian": "黑点", "wy": "污印", "zyc": "纸异常",
      "zmty": "脏面条印", "cq": "串气", "zj": "纸接", "bj": "布接", "jt": "接头",
      "yy": "压印", "bmss": "表面损伤", "zy": "皱印", "pd": "破洞", "HD": "横档",
      "cy": "重影", "jiaodai": "胶带"}
VAL_IMAGES = Path("/home/ancheng/dataset/dataset_split/val/images")
VAL_LABELS = Path("/home/ancheng/dataset/dataset_split/val/labels")

# 每类固定颜色（BGR），索引即类 id
COLORS = [(int(255 * r), int(255 * g), int(255 * b)) for r, g, b in
          [(0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
           (0.58, 0.40, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.50, 0.50, 0.50),
           (0.74, 0.74, 0.13), (0.09, 0.75, 0.81), (0.85, 0.33, 0.10), (0.12, 0.24, 0.63),
           (0.65, 0.35, 0.05), (0.35, 0.75, 0.35), (0.75, 0.55, 0.75), (0.20, 0.60, 0.90),
           (0.90, 0.75, 0.20)]]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# DEIMv2 run → 训练用的 config（--config 省略时按此自动解析）
DEIM_CFG_DIR = ROOT / "third_party/DEIMv2/configs/deimv2"
CONFIG_BY_RUN = {
    "103_deimv2_dinov3_L_v1_1280": "v1_dinov3_L_1280_4090.yml",
    "104_deimv2_dinov3_S_v1_1280": "v1_dinov3_S_1280_4090.yml",
    "105_deimv2_dinov3_S_132ep": "v1_dinov3_S_132ep_official.yml",
    "106_deimv2_dinov3_S_originals": "v1_dinov3_S_originals_68ep.yml",
    "107_deimv2_dinov3_S_v4": "v1_dinov3_S_v4_68ep.yml",
    "108_deimv2_dinov3_S_v5": "v1_dinov3_S_v5_68ep.yml",
}


def run_dir_of(weights):
    wp = Path(weights)
    return wp.parent.parent if wp.parent.name == "weights" else wp.parent


def resolve_deim_config(weights):
    """找 DEIMv2 config；显式 --config 优先，否则按 run 名查表。"""
    run = run_dir_of(weights).name
    if run in CONFIG_BY_RUN:
        return DEIM_CFG_DIR / CONFIG_BY_RUN[run]
    raise SystemExit(
        f"无法自动确定 {run} 的 config，请显式传 --config。\n"
        f"已知 run → config 映射：\n  " +
        "\n  ".join(f"{k} → {DEIM_CFG_DIR.name}/{v}" for k, v in CONFIG_BY_RUN.items()))


def check_config_includes(cfg_path: Path):
    """快照文件（outputs/runs/*/config_snapshot.yml）的 __include__ 是相对 DEIMv2 目录写的，
    换位置后会 FileNotFound——提前给出可读报错，而不是让引擎抛 traceback。"""
    txt = cfg_path.read_text()
    m = re.search(r"__include__:\s*\[(.*?)\]", txt, re.S)
    if not m:
        return
    for inc in re.findall(r"'([^']+)'|\"([^\"]+)\"", m.group(1)):
        rel = inc[0] or inc[1]
        if not (cfg_path.parent / rel).resolve().exists():
            raise SystemExit(
                f"config 的 __include__ 解析失败：{cfg_path.parent / rel}\n"
                f"→ 看起来是 outputs/runs/ 下的 config_snapshot.yml（相对路径只在 DEIMv2 目录内有效）。\n"
                f"请改用 {DEIM_CFG_DIR}/ 下的原 config，或省略 --config 让脚本自动解析。")


def parse_args():
    p = argparse.ArgumentParser(description="手动推理（DEIMv2 / ultralytics），输出 GT|PRED 并排图")
    p.add_argument("--engine", choices=["auto", "deim", "ultralytics"], default="auto",
                   help="默认 auto：.pth→deim 引擎，.pt→ultralytics")
    p.add_argument("--weights", required=True, help=".pth（deim）或 .pt（ultralytics）")
    p.add_argument("--config", default=None, help="deim 必填：训练用的 DEIMv2 config yml")
    p.add_argument("--images", nargs="+", default=[str(VAL_IMAGES)],
                   help="图片路径 / 目录 / glob（默认 v1 val images）")
    p.add_argument("--labels", default=None, help="GT 标签目录（默认按 <images>/../labels 推断）")
    p.add_argument("--limit", type=int, default=8, help="最多处理几张大图（0=全部）")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值（正式口径 0.25）")
    p.add_argument("--imgsz", type=int, default=None, help="推理尺寸（deim 会与 config 校验，必须一致）")
    p.add_argument("--zoom", type=float, default=0.0,
                   help=">0 时按预测/GT 框做裁剪放大（值=相对边长的扩边比例，如 0.3）")
    p.add_argument("--out", default=None, help="输出目录（默认 outputs/inference_demo/<权重名>）")
    p.add_argument("--save-txt", action="store_true", help="同时输出 YOLO 格式预测 txt")
    p.add_argument("--ext", choices=["jpg", "png"], default="jpg", help="输出格式（默认 jpg，演示用）")
    p.add_argument("--max-width", type=int, default=2400, help="输出最大宽度（0=不缩放）")
    return p.parse_args()


def collect_images(specs):
    files = []
    for s in specs:
        p = Path(s)
        if p.is_dir():
            files += sorted([q for q in p.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")])
        elif any(ch in s for ch in "*?["):
            files += sorted(Path(q) for q in glob_mod.glob(s))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"找不到图片：{s}")
    return files


class DeimRunner:
    """DEIMv2 引擎推理（手动预处理：BGR->RGB, resize, /255, ImageNet normalize）。"""

    def __init__(self, config, weights, imgsz=None):
        import argparse as _ap
        from engine.misc import dist_utils
        from engine.core import YAMLConfig, yaml_utils
        from engine.solver import TASKS

        cfg_path = Path(config)
        if not cfg_path.is_absolute():
            cfg_path = ROOT / cfg_path
        # 从 config 读训练/评测分辨率，防止 640/1280 误配（AGENTS.md 陷阱）
        cfg_spatial = None
        for ln in cfg_path.read_text().splitlines():
            if ln.strip().startswith("eval_spatial_size:"):
                cfg_spatial = int(ln.split("[")[1].split(",")[0].strip())
                break
        if imgsz is None:
            imgsz = cfg_spatial
        elif cfg_spatial is not None and imgsz != cfg_spatial:
            raise SystemExit(f"--imgsz {imgsz} 与 config 的 eval_spatial_size {cfg_spatial} 不一致——"
                             f"训练-推理尺度必须绑定（AGENTS.md 陷阱）")
        if imgsz is None:
            raise SystemExit("无法确定推理尺寸：请在 config 里写 eval_spatial_size，或显式传 --imgsz")

        args = _ap.Namespace(config=str(cfg_path), resume=None, tuning=None, device="cuda:0",
                             seed=0, use_amp=False, output_dir="/tmp/_infer_demo",
                             summary_dir=None, test_only=True, print_method="builtin",
                             print_rank=0, local_rank=None, update=[])
        dist_utils.setup_distributed(0, "builtin", seed=0)
        ud = yaml_utils.parse_cli(args.update)
        ud.update({k: v for k, v in args.__dict__.items() if k not in ["update"] and v is not None})
        cfg = YAMLConfig(str(cfg_path), **ud)
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.eval()
        self.solver = solver
        self.device = solver.device
        self.imgsz = imgsz

        ck = torch.load(str(ROOT / weights if not Path(weights).is_absolute() else weights),
                        map_location="cpu", weights_only=False)
        sd = ck.get("ema", {}).get("module", ck.get("model", ck))
        # anchors / valid_mask 随 eval_spatial_size 重建，不能从 ckpt 拷
        skip = ("decoder.anchors", "decoder.valid_mask")
        filtered = {k: v for k, v in sd.items() if k not in skip}
        model = solver.ema.module if solver.ema else dist_utils.de_parallel(solver.model)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        bad = [k for k in missing if k not in skip]
        if bad:
            raise SystemExit(f"权重缺键（非 anchors/valid_mask）：{bad[:5]} …")
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, img_bgr, conf):
        h, w = img_bgr.shape[:2]
        x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        t = ((t - torch.as_tensor(MEAN, device=self.device).view(1, 3, 1, 1))
             / torch.as_tensor(STD, device=self.device).view(1, 3, 1, 1))
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.model(t)
        orig = torch.tensor([[w, h]], device=self.device, dtype=torch.float32)
        r = self.solver.postprocessor(out, orig)[0]
        keep = (r["scores"] > conf).nonzero().flatten()
        return np.array([[*r["boxes"][i].tolist(), float(r["scores"][i]), int(r["labels"][i])]
                         for i in keep.tolist()], dtype=object).reshape(-1, 6)


class UltralyticsRunner:
    def __init__(self, weights, imgsz):
        from ultralytics import YOLO
        self.model = YOLO(str(ROOT / weights if not Path(weights).is_absolute() else weights))
        self.imgsz = imgsz or 1280

    def __call__(self, img_bgr, conf):
        r = self.model.predict(img_bgr, imgsz=self.imgsz, conf=conf, iou=0.5,
                               half=True, verbose=False, device=0)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 6), dtype=object)
        return np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]])


def load_gt(img_path, labels_dir):
    h, w = cv2.imread(str(img_path)).shape[:2]
    lp = Path(labels_dir) / (Path(img_path).stem + ".txt") if labels_dir else None
    out = []
    if lp and lp.exists():
        for ln in lp.read_text().splitlines():
            if not ln.strip():
                continue
            c, cx, cy, bw, bh = ln.split()
            x1, y1 = (float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h
            out.append([x1, y1, x1 + float(bw) * w, y1 + float(bh) * h, 1.0, int(c)])
    return out


def put_tag(im, text):
    """左上角标签：白底黑字，保证在深色织物上也清晰。"""
    fs = max(1.0, im.shape[0] / 1200)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
    cv2.rectangle(im, (8, 8), (8 + tw + 16, 8 + th + 18), (255, 255, 255), -1)
    cv2.putText(im, text, (16, 8 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3)
    return im


def compose(left, right, gap=10):
    sep = np.full((left.shape[0], gap, 3), 128, np.uint8)
    return np.concatenate([left, sep, right], axis=1)


def draw(img, dets, with_conf, thick=3):
    im = img.copy()
    for x1, y1, x2, y2, s, c in dets:
        c = int(c)
        col = COLORS[c % len(COLORS)]
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(im, p1, p2, col, thick)
        # 注意：系统无 CJK 字体，cv2.putText 只能画 ASCII → 用拼音类名（与全项目一致）
        text = f"{CLASSES[c]}" + (f" {s:.2f}" if with_conf else "")
        fs = max(0.8, im.shape[0] / 1400)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        ty = p1[1] - 6 if p1[1] - th - 8 > 0 else p2[1] + th + 8
        cv2.rectangle(im, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty + 4), col, -1)
        cv2.putText(im, text, (p1[0] + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)
    return im


def main():
    a = parse_args()
    files = collect_images(a.images)
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit("没有可处理的图片")

    wp = Path(a.weights)
    engine = a.engine
    if engine == "auto":
        engine = "deim" if wp.suffix == ".pth" else "ultralytics"
    if engine == "deim":
        cfg = Path(a.config) if a.config else resolve_deim_config(a.weights)
        if not cfg.is_absolute():
            cfg = ROOT / cfg
        check_config_includes(cfg)
        runner = DeimRunner(cfg, a.weights, a.imgsz)
    else:
        if wp.suffix == ".pth":
            raise SystemExit(f"{wp.name} 是 DEIMv2 权重（.pth），请用 --engine deim 或让它自动判定")
        runner = UltralyticsRunner(a.weights, a.imgsz)

    run_dir = run_dir_of(a.weights)
    name = f"{run_dir.name}_{wp.stem}"
    out_dir = Path(a.out) if a.out else ROOT / "outputs/inference_demo" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.txt").write_text(
        "\n".join(f"{i}\t{c}\t{CN[c]}" for i, c in enumerate(CLASSES)) + "\n")
    labels_dir = Path(a.labels) if a.labels else (files[0].parent.parent / "labels")
    print(f"引擎={engine} 权重={a.weights} 分辨率={getattr(runner, 'imgsz', '?')} "
          f"conf={a.conf} 图片={len(files)} 输出={out_dir}", flush=True)

    tot = 0
    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        if img is None:
            print(f"  [跳过] 读不到 {f}")
            continue
        preds = runner(img, a.conf)
        gts = load_gt(f, labels_dir)
        tot += len(preds)
        left = put_tag(draw(img, gts, with_conf=False), "GT")
        right = put_tag(draw(img, preds, with_conf=True), f"PRED {name}")
        h, w = img.shape[:2]
        full = compose(left, right)
        save_path = out_dir / f"{Path(f).stem}.{a.ext}"
        if a.zoom > 0 and (len(preds) or len(gts)):
            box = np.array([list(d[:4]) for d in list(preds) + list(gts)], dtype=float)
            cx1, cy1, cx2, cy2 = box[:, 0].min(), box[:, 1].min(), box[:, 2].max(), box[:, 3].max()
            pad = a.zoom * max(cx2 - cx1, cy2 - cy1)
            x1, y1 = max(0, int(cx1 - pad)), max(0, int(cy1 - pad))
            x2, y2 = min(w, int(cx2 + pad)), min(h, int(cy2 + pad))
            out_img = compose(left[y1:y2, x1:x2], right[y1:y2, x1:x2])
        else:
            out_img = full
        if a.max_width and out_img.shape[1] > a.max_width:
            sc = a.max_width / out_img.shape[1]
            out_img = cv2.resize(out_img, (a.max_width, int(out_img.shape[0] * sc)),
                                 interpolation=cv2.INTER_AREA)
        if a.ext == "jpg":
            cv2.imwrite(str(save_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            cv2.imwrite(str(save_path), out_img)
        if a.save_txt:
            with open(out_dir / f"{Path(f).stem}.txt", "w") as fh:
                for x1, y1, x2, y2, s, c in preds:
                    fh.write(f"{int(c)} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
                             f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f} {s:.4f}\n")
        print(f"  [{i}/{len(files)}] {Path(f).name}: GT {len(gts)} 框 / PRED {len(preds)} 框", flush=True)
    print(f"完成：{len(files)} 张，平均 {tot/len(files):.1f} 框/图 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
