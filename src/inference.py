"""
Model loading and tile-and-merge inference for MatForgeNet.

Tile logic ported verbatim from scripts/matforge_app_00_inference_check.py
(step5_tiled_inference). Do not refactor without re-running that script.
"""

import math
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image

from src.models import MatForgeNet

log = logging.getLogger(__name__)

CHECKPOINT_PATH = Path("checkpoints/matforge/best_gan.pt")
TILE   = 256
STRIDE = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# float16 causes PVT attention overflow on real images; float32 is required
DTYPE  = torch.float32


def _vram_mb() -> float:
    return torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0


def _hann_window(n: int, device, dtype) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=dtype)
    w1d = torch.sin(math.pi * k / (n - 1)) ** 2
    return torch.outer(w1d, w1d)


@st.cache_resource
def load_model() -> MatForgeNet:
    log.info("Loading MatForgeNet from %s", CHECKPOINT_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    model = MatForgeNet()
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE).to(DTYPE)
    log.info("  checkpoint: %s | device: %s | dtype: %s | VRAM: %.1f MB",
             CHECKPOINT_PATH, DEVICE, DTYPE, _vram_mb())
    return model


def run_inference(image: Image.Image) -> dict:
    """
    Run tile-and-merge inference on a PIL RGB image of any size.

    Returns dict with keys:
      normal    — np.ndarray (H, W, 3) float32 in [-1, 1]
      roughness — np.ndarray (H, W, 1) float32 in [0, 1]
      metallic  — np.ndarray (H, W, 1) float32 in [0, 1]
    """
    model = load_model()

    # Preprocess
    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img_np = (img_np - mean) / std
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_t = img_t.to(DEVICE).to(DTYPE)

    _, _, H, W = img_t.shape

    # Reflection-pad so H and W are multiples of STRIDE
    pad_h = (STRIDE - H % STRIDE) % STRIDE
    pad_w = (STRIDE - W % STRIDE) % STRIDE
    img_p = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H_p, W_p = img_p.shape

    hann = _hann_window(TILE, device=DEVICE, dtype=torch.float32)

    acc_n = torch.zeros(1, 3, H_p, W_p, device=DEVICE, dtype=torch.float32)
    acc_r = torch.zeros(1, 1, H_p, W_p, device=DEVICE, dtype=torch.float32)
    acc_m = torch.zeros(1, 1, H_p, W_p, device=DEVICE, dtype=torch.float32)
    acc_w = torch.zeros(1, 1, H_p, W_p, device=DEVICE, dtype=torch.float32)

    autocast_ctx = torch.amp.autocast(DEVICE, enabled=(DTYPE == torch.float16))

    with torch.no_grad(), autocast_ctx:
        for y in range(0, H_p - TILE + 1, STRIDE):
            for x in range(0, W_p - TILE + 1, STRIDE):
                tile = img_p[:, :, y : y + TILE, x : x + TILE]
                out  = model(tile)
                w    = hann.unsqueeze(0).unsqueeze(0)
                acc_n[:, :, y : y + TILE, x : x + TILE] += out["normal"].float()    * w
                acc_r[:, :, y : y + TILE, x : x + TILE] += out["roughness"].float() * w
                acc_m[:, :, y : y + TILE, x : x + TILE] += out["metallic"].float()  * w
                acc_w[:, :, y : y + TILE, x : x + TILE] += w

    # eps prevents 0/0 at tile edges where Hann weight is exactly 0
    denom = acc_w + 1e-8
    normal_t    = F.normalize(acc_n / denom, dim=1, eps=1e-6)
    roughness_t = acc_r / denom
    metallic_t  = torch.sigmoid(acc_m / denom)

    # Crop to original size
    normal_t    = normal_t[:, :, :H, :W]
    roughness_t = roughness_t[:, :, :H, :W]
    metallic_t  = metallic_t[:, :, :H, :W]

    # Convert to numpy (H, W, C)
    normal_np    = normal_t[0].permute(1, 2, 0).cpu().float().numpy()
    roughness_np = roughness_t[0].permute(1, 2, 0).cpu().float().numpy()
    metallic_np  = metallic_t[0].permute(1, 2, 0).cpu().float().numpy()

    log.info("Inference done. VRAM after: %.1f MB", _vram_mb())

    return {
        "normal":    normal_np,
        "roughness": roughness_np,
        "metallic":  metallic_np,
    }
