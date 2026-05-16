# matforge_sr_00_vram_check.py
#
# Measures Real-ESRGAN peak VRAM consumption on the local GTX 1650 Max-Q
# under FP16 inference with tile-and-merge compatible patch sizes.
# Implements RRDB/RRDBNet and SRVGGNetCompact architectures directly,
# avoiding basicsr/realesrgan import issues with torchvision >= 0.17.

import torch
import torch.nn as nn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZES = [(256, 256), (320, 320), (512, 512)]

# ---------------------------------------------------------------------------
# Architecture definitions (mirrors basicsr implementations exactly)
# ---------------------------------------------------------------------------

def make_layer(block, n_layers):
    return nn.Sequential(*[block() for _ in range(n_layers)])


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block — core unit of RRDB."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat,              num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch,   num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4*num_grow_ch, num_feat,    3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        # Residual scaling factor 0.2 matches original ESRGAN implementation
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    Generator network used in ESRGAN / Real-ESRGAN.
    scale=4 uses two pixel-shuffle upsample stages.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            lambda: RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch),
            num_block
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # Upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(
            nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(
            nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    """
    Compact generator used in Real-ESRGANv2 (animevideo variants).
    Lighter than RRDBNet; useful as a lower-cost reference point.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_conv=32, upscale=4, act_type="prelu"):
        super().__init__()
        self.num_in_ch  = num_in_ch
        self.num_out_ch = num_out_ch
        self.upscale    = upscale

        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)]
        act  = nn.PReLU(num_parameters=num_feat) if act_type == "prelu" \
               else nn.LeakyReLU(0.2, inplace=True)
        body.append(act)
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            act = nn.PReLU(num_parameters=num_feat) if act_type == "prelu" \
                  else nn.LeakyReLU(0.2, inplace=True)
            body.append(act)
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        body.append(nn.PixelShuffle(upscale))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


# ---------------------------------------------------------------------------
# VRAM measurement
# ---------------------------------------------------------------------------

def measure_vram(model: nn.Module, patch_h: int, patch_w: int) -> dict:
    """
    Single FP16 forward pass with a zero-filled tensor.
    Returns peak VRAM stats in MB, or OOM flag if allocation fails.
    """
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.empty_cache()

    dummy = torch.zeros(1, 3, patch_h, patch_w,
                        dtype=torch.float16, device=DEVICE)
    try:
        with torch.no_grad():
            _ = model(dummy)
        peak_mb     = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2
        reserved_mb = torch.cuda.max_memory_reserved(DEVICE)  / 1024 ** 2
        return {"peak_allocated_mb": round(peak_mb, 1),
                "peak_reserved_mb":  round(reserved_mb, 1),
                "oom": False}
    except torch.cuda.OutOfMemoryError:
        return {"peak_allocated_mb": None, "peak_reserved_mb": None, "oom": True}
    finally:
        del dummy
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "RRDBNet_x4_23blocks": {
        "model": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                 num_block=23, num_grow_ch=32, scale=4),
        "description": "Real-ESRGAN full generator — 23 RRDB blocks",
    },
    "RRDBNet_x4_6blocks": {
        "model": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                 num_block=6, num_grow_ch=32, scale=4),
        "description": "Real-ESRGAN+ lighter generator — 6 RRDB blocks",
    },
    "SRVGGNet_x4_compact": {
        "model": lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                         num_conv=32, upscale=4, act_type="prelu"),
        "description": "Real-ESRGANv2 compact generator — SRVGGNet",
    },
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if DEVICE == "cpu":
        log.warning("CUDA not available — results are not representative of GTX 1650.")

    log.info(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(DEVICE)
        total_vram_mb = props.total_memory / 1024 ** 2
        log.info(f"GPU: {props.name} | Total VRAM: {total_vram_mb:.0f} MB")

    results = []

    for model_name, cfg in MODELS.items():
        log.info(f"\n{'=' * 60}")
        log.info(f"Model : {model_name}")
        log.info(f"Desc  : {cfg['description']}")

        model = cfg["model"]().to(DEVICE).half().eval()
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        log.info(f"Params: {param_count:.2f}M")

        for ph, pw in PATCH_SIZES:
            stats = measure_vram(model, ph, pw)
            if stats["oom"]:
                log.warning(f"  Patch {ph}x{pw}: OOM")
            else:
                log.info(
                    f"  Patch {ph}x{pw}: "
                    f"allocated={stats['peak_allocated_mb']} MB | "
                    f"reserved={stats['peak_reserved_mb']} MB"
                )
            results.append({"model": model_name, "patch": f"{ph}x{pw}", **stats})

        del model
        torch.cuda.empty_cache()

    # Summary table
    log.info(f"\n{'=' * 60}")
    log.info("SUMMARY — Peak VRAM allocated (MB)")
    log.info(f"{'Model':<30} {'Patch':<12} {'Alloc MB':>10}")
    log.info("-" * 55)
    for r in results:
        alloc = f"{r['peak_allocated_mb']}" if not r["oom"] else "OOM"
        log.info(f"{r['model']:<30} {r['patch']:<12} {alloc:>10}")

    # Hard constraint check
    log.info(f"\n{'=' * 60}")
    log.info("Constraint check: peak_allocated < 3500 MB")
    for r in results:
        if r["oom"]:
            log.info(f"  [OOM ] {r['model']} @ {r['patch']}")
        else:
            status = "PASS" if r["peak_allocated_mb"] < 3500 else "FAIL"
            log.info(f"  [{status}] {r['model']} @ {r['patch']}: "
                     f"{r['peak_allocated_mb']} MB")


if __name__ == "__main__":
    main()