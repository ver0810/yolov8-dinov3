from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV3_SIZE_CONFIGS = {
    "vits": {"dim": 384, "depth": 12, "heads": 6},
    "vitb": {"dim": 768, "depth": 12, "heads": 12},
    "vitl": {"dim": 1024, "depth": 24, "heads": 16},
}

VALID_SIZES = tuple(DINOV3_SIZE_CONFIGS.keys())


class DINOv3Backbone(nn.Module):
    """DINOv3 ViT backbone producing multi-scale features for YOLOv8.

    Extracts features at strides 8, 16, 32 by token-to-spatial reshape,
    then uses learned conv layers to build a 3-level feature pyramid.
    """

    def __init__(
        self,
        model_size: str = "vits",
        patch_size: int = 16,
        freeze: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        pretrained: bool = True,
        dinov3_path: str | Path | None = None,
    ):
        super().__init__()
        if model_size not in DINOV3_SIZE_CONFIGS:
            raise ValueError(f"model_size must be one of {VALID_SIZES}, got {model_size}")

        self.model_size = model_size
        self.patch_size = patch_size
        self.embed_dim = DINOV3_SIZE_CONFIGS[model_size]["dim"]

        self.dinov3 = self._load_dinov3(model_size, patch_size, pretrained, dinov3_path)
        self.freeze = freeze
        if freeze:
            for p in self.dinov3.parameters():
                p.requires_grad = False

        if use_lora and not freeze:
            raise ValueError("use_lora=True requires freeze=False; apply LoRA separately")
        self.use_lora = use_lora
        if use_lora:
            self._apply_lora(lora_r, lora_alpha)

        d = self.embed_dim
        self.lateral_p4 = nn.Sequential(
            nn.Conv2d(d, d // 2, 1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.SiLU(inplace=True),
        )
        self.lateral_p8 = nn.Sequential(
            nn.Conv2d(d, d // 2, 1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.SiLU(inplace=True),
        )
        self.lateral_p16 = nn.Sequential(
            nn.Conv2d(d, d // 2, 1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.SiLU(inplace=True),
        )
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(d // 2, d // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.SiLU(inplace=True),
        )
        self.smooth_p8 = nn.Sequential(
            nn.Conv2d(d // 2, d // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.SiLU(inplace=True),
        )
        self.smooth_p16 = nn.Sequential(
            nn.Conv2d(d // 2, d // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.SiLU(inplace=True),
        )

        self.out_channels = (d // 2, d // 2, d // 2)

    def _load_dinov3(self, size: str, patch_size: int, pretrained: bool, path: str | Path | None):
        model_name = f"dinov3_{size}{patch_size}"
        if path is not None and Path(path).exists():
            sys.path.insert(0, str(Path(path).resolve()))
            try:
                from dinov3.hub.backbones import dinov3_vit
                weights = "lvd1689m" if pretrained else "none"
                return dinov3_vit(size, patch_size, weights=weights)
            except ImportError:
                pass
        try:
            weights = "lvd1689m" if pretrained else "none"
            return torch.hub.load("facebookresearch/dinov3", model_name, weights=weights, source="github")
        except Exception:
            return torch.hub.load("facebookresearch/dinov3", model_name, pretrained=pretrained, source="github")

    def _apply_lora(self, r: int, alpha: int):
        try:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["qkv"],
                lora_dropout=0.05,
                bias="none",
            )
            self.dinov3 = get_peft_model(self.dinov3, lora_cfg)
        except ImportError:
            raise ImportError("pip install peft  # required for LoRA fine-tuning")

    @torch.no_grad() if False else lambda fn: fn
    def _maybe_no_grad(self):
        return torch.no_grad() if self.freeze else torch.enable_grad()

    def _tokens_to_spatial(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        ph, pw = h // self.patch_size, w // self.patch_size
        tokens = tokens[:, : ph * pw]
        return tokens.transpose(1, 2).reshape(-1, self.embed_dim, ph, pw)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape

        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            features = self.dinov3.forward_features(x)
            if isinstance(features, dict):
                patch_tokens = features["x_norm_patchtokens"]
            else:
                patch_tokens = features

        feat_full = self._tokens_to_spatial(patch_tokens, H, W)

        f_p4 = F.interpolate(feat_full, scale_factor=0.5, mode="bilinear", align_corners=False)
        f_p8 = feat_full
        f_p16 = F.interpolate(feat_full, scale_factor=0.5, mode="bilinear", align_corners=False)

        f_p4 = self.smooth_p4(self.lateral_p4(f_p4))
        f_p8 = self.smooth_p8(self.lateral_p8(f_p8))
        f_p16 = self.smooth_p16(self.lateral_p16(f_p16))

        return f_p4, f_p8, f_p16
