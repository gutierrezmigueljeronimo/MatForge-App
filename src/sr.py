# SR module — RRDBNet architecture adapted from Real-ESRGAN
# Original work: Copyright (c) 2021, Xintao Wang
# Licensed under the BSD 3-Clause License
# https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE
# Modifications: fine-tuned on MatSynth dataset for PBR material SR.

import gc
import logging
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Model architecture (exact reproduction of RRDBNet ×4, 23 RRDB blocks)
# ----------------------------------------------------------------------

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block as used in RRDBNet."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialisation as in the original repository
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight, a=0.2)
            m.weight.data *= 0.1

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block – three RDBs with residual scaling."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet ×4 super‑resolution network.

    Args:
        num_in_ch: Number of input channels (3 for RGB).
        num_out_ch: Number of output channels (3 for RGB).
        num_feat: Base feature count.
        num_block: Number of RRDB blocks in the trunk.
        num_grow_ch: Growth rate inside each RDB.
        scale: Upscaling factor (fixed to 4 by upsampling logic).
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # Two ×2 nearest‑neighbour upsampling steps (no pixel‑shuffle)
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
CHECKPOINT_PRIMARY = Path("checkpoints/sr/sr_ft_phase1_best_lpips.pt")
CHECKPOINT_FALLBACK = Path("checkpoints/sr/RealESRGAN_x4plus.pth")
TILE = 256
STRIDE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ----------------------------------------------------------------------
# Model loading – cached to avoid reloads within a session
# ----------------------------------------------------------------------

@st.cache_resource(max_entries=1)
def load_sr_model() -> RRDBNet:
    """Load the SR model from the primary or fallback checkpoint.

    The checkpoint state dict may be nested under ``params_ema``, ``params``,
    or stored directly. The function handles all three conventions.

    Returns:
        The evaluation‑mode RRDBNet on the selected device and dtype.

    Raises:
        FileNotFoundError: If neither checkpoint file exists.
    """
    checkpoint_path = None
    if CHECKPOINT_PRIMARY.exists():
        checkpoint_path = CHECKPOINT_PRIMARY
    elif CHECKPOINT_FALLBACK.exists():
        checkpoint_path = CHECKPOINT_FALLBACK
    else:
        raise FileNotFoundError(
            f"Neither primary checkpoint ({CHECKPOINT_PRIMARY}) "
            f"nor fallback checkpoint ({CHECKPOINT_FALLBACK}) found."
        )

    logger.info("Loading SR checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "params_ema" in ckpt:
        state = ckpt["params_ema"]
    elif "params" in ckpt:
        state = ckpt["params"]
    else:
        state = ckpt

    model = RRDBNet(num_block=23)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(DEVICE, dtype=DTYPE)

    logger.info(
        "SR model loaded on %s with dtype %s (checkpoint: %s)",
        DEVICE, str(DTYPE), checkpoint_path.name
    )
    return model


# ----------------------------------------------------------------------
# SR inference with tile‑based processing and Hann window blending
# ----------------------------------------------------------------------

def run_sr(image: Image.Image) -> Image.Image:
    """Apply RRDBNet ×4 super‑resolution to a PIL RGB image.

    Processing is performed on overlapping tiles with a Hann window
    to avoid seam artifacts, following the Real‑ESRGAN inference logic.

    Args:
        image: Input RGB PIL image of any size.

    Returns:
        An RGB PIL image at 4× the original resolution.
    """
    model = load_sr_model()
    orig_h, orig_w = image.height, image.width

    # PIL → float32 numpy [0,1] → tensor (1,3,H,W)
    img_np = np.array(image, dtype=np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    img_t = img_t.to(DEVICE, dtype=DTYPE)

    # Pad so that both dimensions are multiples of STRIDE
    pad_h = (STRIDE - orig_h % STRIDE) % STRIDE
    pad_w = (STRIDE - orig_w % STRIDE) % STRIDE
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
    padded_h, padded_w = img_t.shape[2], img_t.shape[3]

    # Hann window for blending (256×256) → upscaled to output tile (1024×1024)
    hann_1d = torch.sin(
        torch.pi * torch.arange(TILE, dtype=torch.float32, device=DEVICE) / (TILE - 1)
    ) ** 2
    hann_2d = hann_1d[:, None] * hann_1d[None, :]  # (256,256)
    hann_2d_out = F.interpolate(
        hann_2d[None, None, ...], scale_factor=4, mode="bilinear", align_corners=False
    ).squeeze()  # (1024,1024)

    # Accumulators for output and weights
    out_h = padded_h * 4
    out_w = padded_w * 4
    output = torch.zeros(1, 3, out_h, out_w, dtype=torch.float32, device=DEVICE)
    weight = torch.zeros(1, 1, out_h, out_w, dtype=torch.float32, device=DEVICE)

    # Tile loop
    for y in range(0, padded_h - TILE + 1, STRIDE):
        for x in range(0, padded_w - TILE + 1, STRIDE):
            tile = img_t[:, :, y:y + TILE, x:x + TILE]  # (1,3,256,256)
            with torch.no_grad():
                with torch.amp.autocast(device_type=("cuda" if DEVICE == "cuda" else "cpu"),
                                        enabled=(DEVICE == "cuda")):
                    out_tile = model(tile)  # (1,3,1024,1024)

            # Place tile in accumulator weighted by upscaled Hann window
            out_y = y * 4
            out_x = x * 4
            output[:, :, out_y:out_y + TILE * 4, out_x:out_x + TILE * 4] += out_tile * hann_2d_out
            weight[:, :, out_y:out_y + TILE * 4, out_x:out_x + TILE * 4] += hann_2d_out

    # Normalise by weight sum
    output = output / (weight + 1e-8)

    # Crop back to original dimensions (remove padding effect)
    output = output[:, :, :orig_h * 4, :orig_w * 4]

    # Clamp, convert to uint8, back to PIL
    output = output.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H*4,W*4,3)
    output = (output * 255.0).round().astype(np.uint8)

    logger.info("SR processed image from %dx%d to %dx%d px", orig_w, orig_h, orig_w * 4, orig_h * 4)
    return Image.fromarray(output, "RGB")


# ----------------------------------------------------------------------
# Explicit VRAM release
# ----------------------------------------------------------------------

def release_sr_model() -> None:
    """Clear the cached SR model from Streamlit's cache and free GPU memory.

    This is intended to be called immediately after SR inference finishes,
    before loading the main MatForge model, so that VRAM is available.
    """
    load_sr_model.clear()          # remove cached instance
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# Verification block – runs when the module is executed directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ok = False
    for path in (CHECKPOINT_PRIMARY, CHECKPOINT_FALLBACK):
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"Checkpoint found: {path} ({size_mb:.1f} MB)")
            ok = True
        else:
            print(f"Checkpoint missing: {path}")

    if ok:
        print("sr.py: checkpoint check OK")
    else:
        print("sr.py: WARNING – no SR checkpoint found, model loading will fail.")