"""
Shared utilities for MatForge — no dependencies on other src/ modules.
"""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

_MIN_DIM = 64
_MAX_DIM = 8192


def load_and_validate_image(source: bytes | str | Path) -> Image.Image:
    """
    Open an image from raw bytes or a file path and return it as RGB.

    Args:
        source: Raw bytes (e.g. from st.file_uploader) or a file path.

    Returns:
        PIL.Image in RGB mode.

    Raises:
        ValueError: If the image dimensions fall outside [64, 8192] on either axis.
    """
    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)

    img = img.convert("RGB")
    w, h = img.size

    if w < _MIN_DIM or h < _MIN_DIM:
        raise ValueError(
            f"Image too small: {w}×{h}. Minimum is {_MIN_DIM}×{_MIN_DIM}."
        )
    if w > _MAX_DIM or h > _MAX_DIM:
        raise ValueError(
            f"Image too large: {w}×{h}. Maximum is {_MAX_DIM}×{_MAX_DIM}."
        )

    return img


def apply_zoom(image: Image.Image, zoom: float) -> Image.Image:
    """
    Resize an image by a zoom factor without modifying the original.

    Args:
        image: Input PIL.Image in RGB mode.
        zoom:  Scale factor in (0.0, 1.0]; 1.0 keeps original size.

    Returns:
        Resized PIL.Image in RGB mode with both dimensions >= 64px.
    """
    w, h = image.size
    new_w = max(_MIN_DIM, int(w * zoom))
    new_h = max(_MIN_DIM, int(h * zoom))
    return image.resize((new_w, new_h), Image.LANCZOS)


def apply_perspective_warp(
    image: Image.Image,
    src_points: list[list[float]],
) -> Image.Image:
    """
    Correct perspective by warping a quadrilateral region to a rectangle.

    Args:
        image:      Input PIL.Image in RGB mode.
        src_points: Four [x, y] coordinates in pixel space ordered
                    top-left, top-right, bottom-right, bottom-left.

    Returns:
        Warped PIL.Image in RGB mode.

    Raises:
        ValueError: If src_points does not contain exactly 4 points.
    """
    if len(src_points) != 4:
        raise ValueError(
            f"src_points must contain exactly 4 points, got {len(src_points)}."
        )

    src = np.array(src_points, dtype=np.float32)

    xs, ys = src[:, 0], src[:, 1]
    out_w = int(xs.max() - xs.min())
    out_h = int(ys.max() - ys.min())

    dst = np.array(
        [[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32
    )

    M = cv2.getPerspectiveTransform(src, dst)
    img_cv = np.array(image)
    warped = cv2.warpPerspective(img_cv, M, (out_w, out_h))
    return Image.fromarray(warped).convert("RGB")


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL RGB image to a float32 array normalised to [0, 1].

    Args:
        image: PIL.Image in RGB mode.

    Returns:
        np.ndarray of shape (H, W, 3), dtype float32, values in [0.0, 1.0].
    """
    return np.array(image, dtype=np.float32) / 255.0


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """
    Convert a numpy array to a PIL image.

    Args:
        array: Shape (H, W, 3) or (H, W, 1). Float32 in [0, 1] or
               uint8 in [0, 255].

    Returns:
        PIL.Image in RGB mode for 3-channel input, L mode for 1-channel.
    """
    arr = array.copy()

    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(axis=2)
        return Image.fromarray(arr, mode="L")

    return Image.fromarray(arr, mode="RGB")


def numpy_to_tensor(
    array: np.ndarray,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert a (H, W, C) float32 numpy array to a (1, C, H, W) tensor.

    Args:
        array:  Shape (H, W, C), dtype float32.
        device: Target device string, e.g. "cpu" or "cuda".
        dtype:  Target torch dtype.

    Returns:
        torch.Tensor of shape (1, C, H, W) on the specified device and dtype.
    """
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, C, H, W) or (C, H, W) tensor to a (H, W, C) float32 array.

    Args:
        tensor: Tensor of shape (1, C, H, W) or (C, H, W).

    Returns:
        np.ndarray of shape (H, W, C), dtype float32, on CPU.
    """
    t = tensor.detach().cpu().float()
    if t.ndim == 4:
        t = t.squeeze(0)
    return t.permute(1, 2, 0).numpy()


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """
    Encode a PIL image to a base64 string for HTML embedding.

    Args:
        image: Source PIL.Image.
        fmt:   Image format string accepted by PIL (e.g. "PNG", "JPEG").

    Returns:
        Base64-encoded string without a data URI prefix.
    """
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def normal_map_to_display(array: np.ndarray) -> np.ndarray:
    """
    Remap a normal map from [-1, 1] to [0, 1] for display.

    Args:
        array: float32 array in [-1, 1], arbitrary shape.

    Returns:
        float32 array of the same shape, values in [0, 1].
    """
    return np.clip((array.astype(np.float32) + 1.0) / 2.0, 0.0, 1.0)


def get_effective_resolution(
    original_size: tuple[int, int],
    zoom: float,
    sr_active: bool,
) -> tuple[int, int]:
    """
    Compute the resolution that will enter MatForge inference.

    Args:
        original_size: (width, height) of the uploaded image in pixels.
        zoom:          Zoom factor applied before inference.
        sr_active:     True when super-resolution (4× upscale) is enabled.

    Returns:
        (effective_width, effective_height) both >= 64px.
    """
    w, h = original_size
    scale = 4 * zoom if sr_active else zoom
    eff_w = max(_MIN_DIM, int(w * scale))
    eff_h = max(_MIN_DIM, int(h * scale))
    return eff_w, eff_h


def init_session_state(defaults: dict) -> None:
    """
    Initialise Streamlit session state keys that are not yet set.

    Args:
        defaults: Mapping of key → default value. Existing keys are untouched.
    """
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def invalidate_session_keys(keys: list[str]) -> None:
    """
    Set session state keys to None to signal that cached results are stale.

    Args:
        keys: List of st.session_state keys to invalidate.
    """
    for key in keys:
        if key in st.session_state:
            st.session_state[key] = None

def estimate_processing_time(
    image: Image.Image,
    zoom: float,
    use_sr: bool,
) -> tuple[int, float]:
    """Estimate tile count and total processing time in seconds.

    Based on benchmarked throughput on GTX 1650 Max-Q with float32.
    Constants are provisional until measured on real hardware.

    Args:
        image:  Input PIL image before zoom is applied.
        zoom:   Zoom factor (0.1–1.0).
        use_sr: Whether the SR module will be active.

    Returns:
        Tuple of (n_tiles, estimated_seconds).
    """
    TILE_SIZE = 256
    STRIDE = 128
    MATFORGE_SPT = 0.12  # measured: ~0.10–0.14 s/tile on GTX 1650 Max-Q, float32
    SR_OVERHEAD_SECONDS = 9.0

    w, h = image.size
    eff_w = max(256, int(round(w * zoom)))
    eff_h = max(256, int(round(h * zoom)))

    tiles_x = max(1, 1 + (eff_w - TILE_SIZE) // STRIDE)
    tiles_y = max(1, 1 + (eff_h - TILE_SIZE) // STRIDE)
    n_tiles = tiles_x * tiles_y

    seconds = n_tiles * MATFORGE_SPT
    if use_sr:
        seconds += SR_OVERHEAD_SECONDS

    return n_tiles, seconds