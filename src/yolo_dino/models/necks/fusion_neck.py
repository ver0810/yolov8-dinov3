from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_silu(c_in: int, c_out: int, k: int, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(inplace=True),
    )


class FusionNeck(nn.Module):
    """Top-down + bottom-up PAN-style neck adapting DINOv3 features to YOLOv8 head.

    Inputs:  (P4, P8, P16) at strides 8, 16, 32  — all with same channel dim C.
    Outputs: (out_p4, out_p8, out_p16) with channels ``out_channels``.
    """

    def __init__(self, in_channels: int = 192, out_channels: tuple[int, int, int] = (256, 256, 256)):
        super().__init__()
        c = in_channels
        c3, c4, c5 = out_channels

        self.reduce_p4 = _conv_bn_silu(c, c, 1)
        self.reduce_p8 = _conv_bn_silu(c, c, 1)
        self.reduce_p16 = _conv_bn_silu(c, c, 1)

        self.td_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_conv8 = _conv_bn_silu(c * 2, c, 3)
        self.td_conv4 = _conv_bn_silu(c * 2, c, 3)

        self.bu_down = _conv_bn_silu(c, c, 3, s=2)
        self.bu_conv8 = _conv_bn_silu(c * 2, c, 3)
        self.bu_conv16 = _conv_bn_silu(c * 2, c, 3)

        self.out_conv4 = _conv_bn_silu(c, c3, 1)
        self.out_conv8 = _conv_bn_silu(c, c4, 1)
        self.out_conv16 = _conv_bn_silu(c, c5, 1)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        p4, p8, p16 = features
        p4 = self.reduce_p4(p4)
        p8 = self.reduce_p8(p8)
        p16 = self.reduce_p16(p16)

        td8 = self.td_conv8(torch.cat([self.td_up(p16), p8], dim=1))
        td4 = self.td_conv4(torch.cat([self.td_up(td8), p4], dim=1))

        bu8 = self.bu_conv8(torch.cat([self.bu_down(td4), td8], dim=1))
        bu16 = self.bu_conv16(torch.cat([self.bu_down(bu8), p16], dim=1))

        return self.out_conv4(td4), self.out_conv8(bu8), self.out_conv16(bu16)
