from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from yolo_dino.models.backbones.dinov3_backbone import DINOv3Backbone
from yolo_dino.models.necks.fusion_neck import FusionNeck


def _make_detect_head(nc: int, ch: tuple[int, int, int]):
    from ultralytics.nn.modules.head import Detect

    head = Detect(nc=nc, ch=list(ch))
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.initialize_biases()
    return head


class YOLODINOv3(nn.Module):
    """Full DINOv3 backbone replacement for YOLOv8.

    Architecture:  DINOv3-ViT  →  FusionNeck (PAN)  →  YOLOv8 Detect head
    """

    def __init__(
        self,
        nc: int = 15,
        dinov3_size: str = "vits",
        patch_size: int = 16,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        pretrained: bool = True,
        dinov3_path: str | Path | None = None,
        head_channels: tuple[int, int, int] = (256, 256, 256),
    ):
        super().__init__()
        self.backbone = DINOv3Backbone(
            model_size=dinov3_size,
            patch_size=patch_size,
            freeze=freeze_backbone,
            use_lora=use_lora,
            lora_r=lora_r,
            pretrained=pretrained,
            dinov3_path=dinov3_path,
        )
        backbone_out = self.backbone.out_channels[0]
        self.neck = FusionNeck(in_channels=backbone_out, out_channels=head_channels)
        self.head = _make_detect_head(nc, head_channels)

    def forward(self, x):
        feats = self.backbone(x)
        neck_out = self.neck(feats)
        return self.head(list(neck_out))

    def loss(self, preds, batch):
        return self.head.loss(preds, batch)


class YOLOv8EarlyStages(nn.Module):
    """YOLOv8 CSPDarknet early stages (stride 2 → 4 → 8)."""

    def __init__(self, width_mult: float = 1.0):
        super().__init__()
        from ultralytics.nn.modules import Conv, C2f

        base = [64, 128, 256]
        c = [max(round(ch * width_mult), 1) for ch in base]

        self.p1 = Conv(3, c[0], k=3, s=2)
        self.p2 = Conv(c[0], c[1], k=3, s=2)
        self.c2f_p2 = C2f(c[1], c[1], n=3, shortcut=True)
        self.p3_down = Conv(c[1], c[2], k=3, s=2)
        self.c2f_p3 = C2f(c[2], c[2], n=6, shortcut=True)

        self.out_channels = c[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.p1(x)
        x = self.p2(x)
        x = self.c2f_p2(x)
        x = self.p3_down(x)
        x = self.c2f_p3(x)
        return x


class HybridFusionNeck(nn.Module):
    """Fuses YOLOv8 early features (stride 8) with DINOv3 deep features (stride 16, 32)."""

    def __init__(
        self,
        yolo_ch: int = 256,
        dino_ch: int = 192,
        out_channels: tuple[int, int, int] = (256, 256, 256),
    ):
        super().__init__()
        c3, c4, c5 = out_channels

        self.yolo_proj = nn.Sequential(
            nn.Conv2d(yolo_ch, c3, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
        )
        self.dino_proj_p8 = nn.Sequential(
            nn.Conv2d(dino_ch, c4, 1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
        )
        self.dino_proj_p16 = nn.Sequential(
            nn.Conv2d(dino_ch, c5, 1, bias=False),
            nn.BatchNorm2d(c5),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        yolo_feat: torch.Tensor,
        dino_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, dino_p8, dino_p16 = dino_feats
        return self.yolo_proj(yolo_feat), self.dino_proj_p8(dino_p8), self.dino_proj_p16(dino_p16)


class YOLODINOv3Hybrid(nn.Module):
    """Hybrid backbone: YOLOv8 early stages (P1–P3) + DINOv3 deep stages (P4–P5).

    YOLOv8 handles fine-grained stride-8 features (good for small defects),
    DINOv3 provides semantically rich stride-16/32 features.
    """

    def __init__(
        self,
        nc: int = 15,
        dinov3_size: str = "vits",
        patch_size: int = 16,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        pretrained: bool = True,
        dinov3_path: str | Path | None = None,
        yolo_width: float = 1.0,
        head_channels: tuple[int, int, int] = (256, 256, 256),
    ):
        super().__init__()
        self.yolo_early = YOLOv8EarlyStages(width_mult=yolo_width)
        self.dinov3 = DINOv3Backbone(
            model_size=dinov3_size,
            patch_size=patch_size,
            freeze=freeze_backbone,
            use_lora=use_lora,
            lora_r=lora_r,
            pretrained=pretrained,
            dinov3_path=dinov3_path,
        )
        dino_ch = self.dinov3.out_channels[0]
        self.neck = HybridFusionNeck(
            yolo_ch=self.yolo_early.out_channels,
            dino_ch=dino_ch,
            out_channels=head_channels,
        )
        self.head = _make_detect_head(nc, head_channels)

    def forward(self, x):
        yolo_feat = self.yolo_early(x)
        dino_feats = self.dinov3(x)
        neck_out = self.neck(yolo_feat, dino_feats)
        return self.head(list(neck_out))

    def loss(self, preds, batch):
        return self.head.loss(preds, batch)
