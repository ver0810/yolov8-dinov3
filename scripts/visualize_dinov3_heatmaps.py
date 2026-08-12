"""Visualize DINOv3 feature maps: ViT CLS-attention heatmaps + convnext feature activation.

Usage:
    uv run python scripts/visualize_dinov3_heatmaps.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "ultralytics"))

import timm
from PIL import Image

OUT = ROOT / "outputs" / "heatmaps"
IMGDIR = Path("/home/ancheng/dataset/dataset_split/val/images")

IMGS = {
    "bd": "LZW_ZQ_20260124_266",
    "heidian": "113__0921",
    "wy": "XBW_20250804_1593",
    "bmss": "LZW_20260124_362",
}
IMG_SIZE = 448  # ViT patch16 -> 28x28 grid


def build_vit():
    m = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True).cuda().eval()
    captured = {}
    attn = m.blocks[-1].attn

    def patched(x, rope=None, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        gate = attn.gate(x).sigmoid() if attn.gate is not None else None
        if attn.qkv is not None:
            if attn.q_bias is None:
                qkv = attn.qkv(x)
            else:
                qkv_bias = torch.cat((attn.q_bias, attn.k_bias, attn.v_bias))
                if attn.qkv_bias_separate:
                    qkv = attn.qkv(x)
                    qkv += qkv_bias
                else:
                    qkv = torch.nn.functional.linear(x, weight=attn.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        if rope is not None:
            from timm.layers import apply_rot_embed_cat
            npt = attn.num_prefix_tokens
            half = getattr(attn, "rotate_half", False)
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
        q = q * attn.scale
        w = q @ k.transpose(-2, -1)
        w = w.softmax(dim=-1)
        w = attn.attn_drop(w)
        captured["attn"] = w  # (B, heads, N, N)
        x = w @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = attn.norm(x)
        if gate is not None:
            x = x * gate
        x = attn.proj(x)
        return attn.proj_drop(x)

    attn.forward = patched
    return m, captured


def vit_attn_heatmap(model, captured, img_t):
    """CLS->patch attention from last layer, head-averaged, registers excluded."""
    with torch.no_grad():
        model(img_t)
    a = captured["attn"][0]  # (heads, N, N)
    n_prefix = 5  # CLS + 4 register tokens
    cls_attn = a[:, 0, n_prefix:].mean(0)  # (N_patch,)
    grid = IMG_SIZE // 16
    hm = cls_attn.reshape(grid, grid).float().cpu().numpy()
    return (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)


def conv_heatmap(model, img_t):
    """convnext feature energy at the layers hybrid actually uses (stride 16/32)."""
    with torch.no_grad():
        feats = model(img_t)
    hms = []
    for f in feats:
        e = f.square().mean(1).squeeze(0).sqrt()  # (H, W) RMS over channels
        e = (e - e.min()) / (e.max() - e.min() + 1e-6)
        hms.append(e.float().cpu().numpy())
    return hms


def load_img(name):
    im = Image.open(IMGDIR / f"{name}.png").convert("RGB")
    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(im).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
    t = (t - t.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / t.new_tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)
    return im, t


def save_grid(cls, im, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(rows) + 1, figsize=(5 * (len(rows) + 1), 5))
    axes[0].imshow(im)
    axes[0].set_title(f"{cls} (original)")
    axes[0].axis("off")
    for ax, (title, hm) in zip(axes[1:], rows):
        ax.imshow(im, alpha=0.55)
        ax.imshow(hm, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"{cls}_dinov3_heatmaps.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    vit, captured = build_vit()
    conv = timm.create_model(
        "convnext_tiny.dinov3_lvd1689m", pretrained=True, features_only=True, out_indices=[2, 3]
    ).cuda().eval()
    for cls, name in IMGS.items():
        im, t = load_img(name)
        hm_vit = vit_attn_heatmap(vit, captured, t)
        hm16, hm32 = conv_heatmap(conv, t)
        print(f"[{cls}] vit_attn max={hm_vit.max():.3f} | conv s16/s32 max={hm16.max():.3f}/{hm32.max():.3f}")
        save_grid(cls, im, [("ViT CLS-attn (last layer)", hm_vit), ("convnext stride16", hm16), ("convnext stride32", hm32)])
    print("saved to", OUT)