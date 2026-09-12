"""v4 池校验(本地/远端通用, 仅依赖 PIL+numpy+stdlib):
1 COUNT  2 LABEL_IDENTITY  3 SOURCE_RESTRICTION  4 LEAKAGE(val+test)
5 IMAGE_INTEGRITY  6 MANIFEST  7 TRANSFORM_FIDELITY  8 COMBINED_HASH(供两端对比)
退出码 0=全PASS, 1=任一FAIL. 用法: python verify_pool_v4.py [pool_dir]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/ancheng/Code/yolov8-dinov3")
TRAIN_IMG = Path("/home/ancheng/dataset/dataset_split/train/images")
VAL_IMG = Path("/home/ancheng/dataset/dataset_split/val/images")
TEST_IMG = Path("/home/ancheng/dataset/dataset_split/test/images")
SUFFIX = "_v4clahe"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    pool = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/dataset_v1_pool_v4"
    manifest = json.loads((pool / "pool_v4_manifest.json").read_text())
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
        ok = ok and cond

    imgs = sorted((pool / "images").glob("*"))
    lbls = sorted((pool / "labels").glob("*.txt"))

    # 1. COUNT
    check("COUNT", len(imgs) == len(lbls) == len(manifest), f"img={len(imgs)} lbl={len(lbls)} manifest={len(manifest)}")
    check("COUNT_NONEMPTY", len(imgs) > 0)

    # stem 集合
    def strip(p: Path) -> str:
        n = p.stem
        assert n.endswith(SUFFIX), f"命名异常: {p.name}"
        return n[: -len(SUFFIX)]

    stems = [strip(p) for p in imgs]
    lbl_stems = [strip(p) for p in lbls]
    check("NAME_MATCH", sorted(stems) == sorted(lbl_stems))

    val_stems = {p.stem for p in VAL_IMG.glob("*")}
    test_stems = {p.stem for p in TEST_IMG.glob("*")}

    # 逐项校验
    label_ok = src_ok = leak_ok = img_ok = man_ok = trans_ok = True
    diffs = []
    man_by_dst = {m["dst_img"]: m for m in manifest}
    for p in imgs:
        s = strip(p)
        m = man_by_dst.get(str(p))
        if m is None:
            man_ok = False
            continue
        # 2. LABEL_IDENTITY: v4标签 == 源标签(逐字节)
        src_lbl = Path(m["src_lbl"])
        if sha256(pool / "labels" / (s + SUFFIX + ".txt")) != sha256(src_lbl):
            label_ok = False
        # 3. SOURCE_RESTRICTION: 源必须在 train
        src_img = Path(m["src_img"])
        if not (src_img.exists() and TRAIN_IMG in src_img.parents):
            src_ok = False
        # 4. LEAKAGE: 源 stem 不在 val/test
        if s in val_stems or s in test_stems:
            leak_ok = False
        # 5. IMAGE_INTEGRITY: 可打开/RGB/尺寸一致
        try:
            a = Image.open(p).convert("RGB")
            b = Image.open(src_img).convert("RGB")
            if a.size != b.size:
                img_ok = False
        except Exception:
            img_ok = False
            continue
        # 7. TRANSFORM_FIDELITY: 确实被变换(差异>0)且保留结构(差异<60)
        d = float(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).mean())
        diffs.append(d)
        if not (0 < d < 60):
            trans_ok = False
    check("LABEL_IDENTITY", label_ok, "(v4标签逐字节==源标签)")
    check("SOURCE_RESTRICTION", src_ok, "(源全在train)")
    check("LEAKAGE", leak_ok, "(源stem不在val/test)")
    check("IMAGE_INTEGRITY", img_ok, "(可打开/RGB/尺寸一致)")
    # 6. MANIFEST: 记录的sha与重算一致
    man_sha_ok = all(
        sha256(Path(m["dst_img"])) == m["sha256_dst_img"] for m in manifest
    )
    check("MANIFEST", man_sha_ok and man_ok, "(sha256自洽+全覆盖)")
    if diffs:
        import statistics as st
        check("TRANSFORM_FIDELITY", trans_ok, f"(mean_abs_diff {st.mean(diffs):.2f}, n={len(diffs)})")
    # 8. COMBINED_HASH
    combined = hashlib.sha256("".join(sorted(m["sha256_dst_img"] for m in manifest)).encode()).hexdigest()
    print(f"COMBINED={combined} N={len(manifest)}")

    # bmss 实例守恒: v4标签里的 bmss 数 == 源标签里的
    n_bmss = 0
    for p in lbls:
        for ln in p.read_text().splitlines():
            t = ln.split()
            if t and int(float(t[0])) == 11:
                n_bmss += 1
    print(f"BMSS_INSTANCES={n_bmss}")
    print("ALL_PASS" if ok else "HAS_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
