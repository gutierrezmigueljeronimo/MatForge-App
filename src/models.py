"""
PyTorch model definitions for MatForgeNet.

Contains FPNDecoder, RefineHead, and MatForgeNet — verbatim from the
verified architecture spec in scripts/matforge_app_00_inference_check.py.
Do not modify these classes without re-validating against the checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder that fuses multi-scale encoder features."""

    def __init__(self, in_channels=(64, 128, 320, 512), out_channels=256):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(64,  256, 1, bias=False),
            nn.Conv2d(128, 256, 1, bias=False),
            nn.Conv2d(320, 256, 1, bias=False),
            nn.Conv2d(512, 256, 1, bias=False),
        ])
        self.merge = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            for _ in range(3)
        ])

    def forward(self, features):
        # features: [L1, L2, L3, L4] — shallow to deep
        projected = [proj(f) for proj, f in zip(self.proj, features)]
        x = projected[-1]                                         # L4, deepest
        for i in range(len(self.merge) - 1, -1, -1):
            x = F.interpolate(x, size=projected[i].shape[-2:], mode="nearest")
            x = self.merge[i](torch.cat([x, projected[i]], dim=1))
        return x                                                  # (B, 256, H/4, W/4)


class RefineHead(nn.Module):
    """Upsampling refinement head that produces a single PBR map channel group."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsample layers carry no parameters and are not in the state dict;
        # they live in forward() so the Sequential index positions match the checkpoint.
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(64, out_channels, 1)  # bias confirmed in checkpoint

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block2(x)
        return self.out(x)


class MatForgeNet(nn.Module):
    """Full PBR map prediction network: PVT-v2-B1 encoder + FPN + three RefineHeads."""

    def __init__(self):
        super().__init__()
        self.encoder       = timm.create_model("pvt_v2_b1", pretrained=False, features_only=True)
        self.fpn           = FPNDecoder(in_channels=(64, 128, 320, 512), out_channels=256)
        self.head_normal    = RefineHead(256, 3)
        self.head_roughness = RefineHead(256, 1)
        self.head_metallic  = RefineHead(256, 1)

    def forward(self, x):
        features    = self.encoder(x)
        fpn_out     = self.fpn(features)
        raw_normal    = self.head_normal(fpn_out)
        raw_roughness = self.head_roughness(fpn_out)
        raw_metallic  = self.head_metallic(fpn_out)
        # normal is unit-length in [-1, 1]; roughness is [0, 1];
        # metallic logits are left raw — sigmoid applied at inference
        normal    = F.normalize(torch.tanh(raw_normal), dim=1, eps=1e-6)
        roughness = torch.sigmoid(raw_roughness)
        return {"normal": normal, "roughness": roughness, "metallic": raw_metallic}
