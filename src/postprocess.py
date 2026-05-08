"""
src/postprocess.py
Post-processing utilities for MatForge PBR maps.
Pure logic, no Streamlit, no GPU, no state.
All functions accept and return float32 numpy arrays.
"""

import numpy as np
from scipy import ndimage  # noqa: F401   # available if needed in future
import cv2

# --- Noise engine imports with fallback --------------------------------------
_noise_generator = None  # cached when first requested

def _get_noise_generator():
    """Return a noise generator instance (pyfastnoiselite preferred, opensimplex fallback).

    The returned object provides a method ``noise2D(x, y)`` where *x* and *y*
    are 2D coordinate arrays of the same shape, returning an array of noise
    values in [-1, 1].
    """
    global _noise_generator
    if _noise_generator is not None:
        return _noise_generator

    # Try pyfastnoiselite first
    try:
        import pyfastnoiselite as pyn

        class FastNoiseWrapper:
            def __init__(self):
                self._fn = pyn.FastNoiseLite()

            def noise2D(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
                # FastNoiseLite only accepts scalars; vectorize with a ufunc
                vfunc = np.vectorize(self._fn.GetNoise, otypes=[np.float32])
                return vfunc(x, y)

            def seed(self, s: int):
                self._fn.SetSeed(s)
                return self

        _noise_generator = FastNoiseWrapper
        return _noise_generator

    except ImportError:
        pass

    # Fallback to opensimplex
    try:
        from opensimplex import OpenSimplex

        class SimplexWrapper:
            def __init__(self, seed: int = 0):
                self._simplex = OpenSimplex(seed)

            def noise2D(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
                return self._simplex.noise2(x, y).astype(np.float32)

            def seed(self, s: int):
                self._simplex = OpenSimplex(s)
                return self

        _noise_generator = SimplexWrapper
        return _noise_generator

    except ImportError:
        _noise_generator = None
        return None


def _generate_fbm(
    gen,
    height: int,
    width: int,
    frequency: float,
    octaves: int,
    persistence: float,
    seed_offset: int = 0,
) -> np.ndarray:
    """Generate fractal Brownian motion noise field.

    Args:
        gen: Noise generator instance with ``noise2D(x, y)`` and ``seed()``.
        height, width: output dimensions.
        frequency: base spatial frequency.
        octaves: number of octaves.
        persistence: amplitude multiplier per octave.
        seed_offset: added to the base seed inside the generator.

    Returns:
        float32 array of shape (height, width), values roughly in [-1, 1].
    """
    # create coordinate grid scaled to [0, 1] range
    xs = np.linspace(0, 1, width, endpoint=False, dtype=np.float32)
    ys = np.linspace(0, 1, height, endpoint=False, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    noise = np.zeros((height, width), dtype=np.float32)
    for o in range(octaves):
        freq = frequency * (2 ** o)
        amp = persistence ** o
        gen.seed(seed_offset + o * 1000)  # decorrelate octaves
        noise += amp * gen.noise2D(X * freq, Y * freq)
    return noise


# ---------------------------------------------------------------------------

def adjust_gain_offset(
    tensor: np.ndarray,
    gain: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Apply gain and offset to a roughness or metallic map.

    Args:
        tensor: Input array (H, W) or (H, W, 1) in [0, 1], float32.
        gain: Multiplicative factor, must be in [0.5, 2.0].
        offset: Additive constant, must be in [-0.5, 0.5].

    Returns:
        Array with same shape and dtype, clipped to [0, 1].

    Raises:
        ValueError: If ``tensor`` contains negative values (likely a normal map)
            or if gain/offset are out of range.
    """
    # Safety check: normal maps live in [-1, 1] and have negative values.
    if np.any(tensor < -0.01):
        raise ValueError(
            "adjust_gain_offset expects a roughness/metallic map in [0, 1], "
            "but input contains negative values. It should not be used on normal maps."
        )
    if not (0.5 <= gain <= 2.0):
        raise ValueError("gain must be in [0.5, 2.0]")
    if not (-0.5 <= offset <= 0.5):
        raise ValueError("offset must be in [-0.5, 0.5]")

    return np.clip(gain * tensor + offset, 0.0, 1.0).astype(tensor.dtype)


# ---------------------------------------------------------------------------

_CALIBRATION_RANGES = {
    "stone_rough":         (0.55, 0.95, 0.0, 0.0),
    "concrete_plaster":    (0.55, 0.95, 0.0, 0.0),
    "brick_terracotta":    (0.55, 0.95, 0.0, 0.0),
    "mixed_ambiguous":     (0.55, 0.95, 0.0, 0.0),
    "wood":                (0.45, 0.80, 0.0, 0.0),
    "ceramic_ground":      (0.05, 0.30, 0.0, 0.0),
    "marble_smooth":       (0.05, 0.30, 0.0, 0.0),
    "metal":               (0.05, 0.75, 0.85, 1.0),
}


def calibrate_by_group(
    roughness: np.ndarray,
    metallic: np.ndarray,
    group: str,
    knn_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull roughness and metallic towards plausible ranges for a given
    functional group, guided by KNN confidence.

    When the classifier is very confident (low *knn_distance*), the maps are
    strongly clamped to the physical ranges. When confidence is low, the
    original predictions are preserved.

    Args:
        roughness: Roughness map, float32, shape (H, W) or (H, W, 1), [0, 1].
        metallic:  Metallic map, same shape and range.
        group:     String identifying the functional group (see table).
        knn_distance: Scalar distance from the KNN classifier; higher values
                    indicate lower confidence.

    Returns:
        Tuple ``(roughness_cal, metallic_cal)`` with the same shapes and
        dtypes as the inputs.
    """
    ranges = _CALIBRATION_RANGES.get(group)
    if ranges is None:
        # Unknown group – return copies.
        return roughness.copy(), metallic.copy()

    r_min, r_max, m_min, m_max = ranges
    alpha = max(0.0, min(1.0, 1.0 - knn_distance * 3.0))

    # Smooth calibration: blend between clamped and original.
    rough_cal = alpha * np.clip(roughness, r_min, r_max) + (1.0 - alpha) * roughness
    metal_cal = alpha * np.clip(metallic, m_min, m_max) + (1.0 - alpha) * metallic

    return rough_cal.astype(roughness.dtype), metal_cal.astype(metallic.dtype)


# ---------------------------------------------------------------------------

def make_tileable_simple(
    image: np.ndarray,
    is_normal_map: bool = False,
) -> np.ndarray:
    """Convert an image to a tileable version using a 50% offset and cross-fade.

    The seam is moved to the centre and a linear gradient mask blends the rolled
    version with the original. For normal maps the blending is performed on
    unpacked vectors and the result is re-normalised.

    Args:
        image: Input array, float32, either (H, W), (H, W, 1) or (H, W, 3).
        is_normal_map: If True, the image is assumed to be a packed normal map
                      ([0, 1] range) that will be unpacked to [-1, 1] before
                      blending and re-packed afterwards.

    Returns:
        Tileable image with the same shape and dtype.
    """
    h, w = image.shape[:2]
    dtype = image.dtype

    # Shift by half the image size to move seams to the centre.
    shift_y, shift_x = h // 2, w // 2
    if image.ndim == 2:
        rolled = np.roll(image, (shift_y, shift_x), axis=(0, 1))
    else:
        rolled = np.roll(image, (shift_y, shift_x, 0), axis=(0, 1))

    # Horizontal and vertical ramp masks over the central 20% of the image.
    # Values from 0 to 1 from 10% left of centre to 10% right of centre.
    t_h = np.clip((np.arange(w, dtype=np.float32) - (w / 2 - 0.1 * w)) / (0.2 * w), 0.0, 1.0)
    t_v = np.clip((np.arange(h, dtype=np.float32) - (h / 2 - 0.1 * h)) / (0.2 * h), 0.0, 1.0)
    h_mask = t_h[np.newaxis, :]      # (1, w)
    v_mask = t_v[:, np.newaxis]      # (h, 1)
    mask = np.maximum(h_mask, v_mask)  # (h, w)

    # If image has colour channels, add a trailing dimension for broadcasting.
    if image.ndim == 3:
        mask = mask[..., np.newaxis]

    if is_normal_map:
        # Unpack from [0,1] to [-1,1], blend, normalise, repack.
        orig_unpack = image * 2.0 - 1.0
        rolled_unpack = rolled * 2.0 - 1.0
        blended_unpack = rolled_unpack * mask + orig_unpack * (1.0 - mask)
        # Renormalise per pixel.
        norms = np.linalg.norm(blended_unpack, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # avoid division by zero
        blended_normalised = blended_unpack / norms
        out = (blended_normalised + 1.0) * 0.5
    else:
        out = rolled * mask + image * (1.0 - mask)

    return out.astype(dtype)


# ---------------------------------------------------------------------------

def blend_normals_rnm(
    n_base: np.ndarray,
    n_detail: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Blend two normal maps using Reoriented Normal Mapping (RNM).

    Follows the technique of Barré-Brisebois & Hill (2012). The base normal is
    reoriented to agree with the detail normal, avoiding common artifacts of
    simple blending.

    Args:
        n_base:   Base normal map, (H, W, 3), float32, unit length.
        n_detail: Detail normal map, (H, W, 3), float32, unit length.
        mask:     Optional blend mask, (H, W, 1), float32, range [0, 1].
                 When provided, the result lerps between *n_base* and the
                 RNM blend using the mask.

    Returns:
        Blended normal map, (H, W, 3), float32, unit length.
    """
    t = n_base + np.array([0.0, 0.0, 1.0], dtype=np.float32)      # (H, W, 3)
    u = n_detail * np.array([-1.0, -1.0, 1.0], dtype=np.float32)  # (H, W, 3)

    dot_tu = np.sum(t * u, axis=-1)      # (H, W)
    t_z = t[..., 2]                      # (H, W)
    safe_t_z = np.maximum(t_z, 1e-6)

    r = t * (dot_tu / safe_t_z)[..., np.newaxis] - u
    norm_r = np.linalg.norm(r, axis=-1, keepdims=True)
    safe_norm_r = np.maximum(norm_r, 1e-6)
    rnm_blend = r / safe_norm_r

    if mask is not None:
        blended = n_base * (1.0 - mask) + rnm_blend * mask
        # Renormalise after linear interpolation.
        norm_blended = np.linalg.norm(blended, axis=-1, keepdims=True)
        norm_blended = np.maximum(norm_blended, 1e-6)
        blended = blended / norm_blended
        return blended.astype(n_base.dtype)

    return rnm_blend.astype(n_base.dtype)


# ---------------------------------------------------------------------------

def generate_variations(
    roughness: np.ndarray,
    metallic: np.ndarray,
    normal: np.ndarray | None = None,
    n_variants: int = 4,
    seed: int = 42,
) -> list[dict[str, np.ndarray]]:
    """Generate procedural variations of the base PBR maps.

    Three techniques are cycled through the requested number of variants:
    *zonal mixing*, *worn edges* (only if a normal map is provided), and
    *random scaling*.

    Args:
        roughness: Roughness map, (H, W) or (H, W, 1), float32, [0, 1].
        metallic:  Metallic map, same shape and range.
        normal:    Normal map, (H, W, 3), float32, range [-1, 1] (not packed).
                   If None, only roughness and metallic variants are created.
        n_variants: Number of variations to produce.
        seed:       Base seed for reproducible randomness.

    Returns:
        List of dictionaries, each with keys ``"roughness"``, ``"metallic"``,
        and ``"normal"`` (if a normal map was supplied). If the noise engine
        is unavailable, the list contains *n_variants* copies of the inputs.
    """
    GenClass = _get_noise_generator()
    if GenClass is None:
        # Fallback – return identical copies.
        variants = []
        for _ in range(n_variants):
            variant = {
                "roughness": roughness.copy(),
                "metallic": metallic.copy(),
            }
            if normal is not None:
                variant["normal"] = normal.copy()
            variants.append(variant)
        return variants

    # Determine available techniques (0: zonal mix, 1: worn edges, 2: scale)
    techniques = ["zonal", "worn", "scale"]
    if normal is None:
        techniques = ["zonal", "scale"]  # worn edges requires a normal map

    variants = []
    for i in range(n_variants):
        rng = np.random.RandomState(seed + i)
        technique = techniques[i % len(techniques)]

        r_var = roughness.copy()
        m_var = metallic.copy()
        n_var = normal.copy() if normal is not None else None

        if technique == "zonal":
            gen = GenClass(seed + i)
            fbm = _generate_fbm(gen, r_var.shape[0], r_var.shape[1],
                                frequency=4.0, octaves=3, persistence=0.5,
                                seed_offset=seed + i)
            mask = 1.0 / (1.0 + np.exp(-fbm * 5.0))
            # Reshape mask to match roughness channels if needed
            if r_var.ndim == 3:
                mask = mask.reshape(r_var.shape)
            shifted = np.clip(roughness + fbm.reshape(r_var.shape) * 0.15, 0.0, 1.0)
            r_var = roughness * (1.0 - mask) + shifted * mask

        elif technique == "worn":
            # Worn edges – only feasible with normal map
            if normal is not None:
                # Compute gradient magnitude of the normal map
                gray = np.sum(normal, axis=-1) / 3.0  # simple luminance proxy
                dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                grad_mag = np.sqrt(dx**2 + dy**2)  # (H, W)
                # Normalise grad_mag to [0,1] quickly
                if grad_mag.max() > 1e-6:
                    grad_mag /= grad_mag.max()
                # FBM low-frequency noise to mask edges
                gen = GenClass(i)  # need not be the same seed, just reproducible
                noise_mask = _generate_fbm(gen, r_var.shape[0], r_var.shape[1],
                                           frequency=2.0, octaves=2, persistence=0.5,
                                           seed_offset=seed + i)
                # Combine gradient and noise
                edge_mask = grad_mag * noise_mask
                edge_mask = np.clip(edge_mask, 0.0, 1.0)
                # Darken roughness in worn areas
                r_var = r_var * (1.0 - edge_mask.reshape(r_var.shape) * 0.3)
                r_var = np.clip(r_var, 0.0, 1.0)

        elif technique == "scale":
            # Random scale factor
            sf = rng.uniform(0.85, 1.15)
            new_h = int(round(r_var.shape[0] * sf))
            new_w = int(round(r_var.shape[1] * sf))

            def _scale_tile_wrap(arr, sf, new_h, new_w):
                # Resize
                resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Bring back to original size with tile-like wrapping
                h, w = arr.shape[:2]
                if new_h >= h and new_w >= w:
                    shift_y = rng.randint(0, new_h - h + 1)
                    shift_x = rng.randint(0, new_w - w + 1)
                    rolled = np.roll(resized, (shift_y, shift_x), axis=(0, 1))
                    return rolled[:h, :w, ...]  # crop central window
                else:
                    pad_top = rng.randint(0, h - new_h + 1)
                    pad_bottom = (h - new_h) - pad_top
                    pad_left = rng.randint(0, w - new_w + 1)
                    pad_right = (w - new_w) - pad_left
                    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom,
                                                pad_left, pad_right,
                                                cv2.BORDER_REFLECT)
                    return padded[:h, :w, ...]

            r_var = _scale_tile_wrap(r_var, sf, new_h, new_w)
            m_var = _scale_tile_wrap(m_var, sf, new_h, new_w)

        variant = {
            "roughness": r_var.astype(roughness.dtype),
            "metallic": m_var.astype(metallic.dtype),
        }
        if normal is not None:
            variant["normal"] = n_var.astype(normal.dtype)
        variants.append(variant)

    return variants


# ---------------------------------------------------------------------------

def blend_materials(
    r_a: np.ndarray, m_a: np.ndarray, n_a: np.ndarray,
    r_b: np.ndarray, m_b: np.ndarray, n_b: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = mask.astype(np.float32)
    # Ensure mask is 2D for broadcasting
    if mask.ndim == 3:
        mask_2d = mask.squeeze(-1)
    else:
        mask_2d = mask.copy()
    # Expand to match target arrays if they carry a channel dimension
    mask_r = mask_2d[..., np.newaxis] if r_a.ndim == 3 else mask_2d
    mask_m = mask_2d[..., np.newaxis] if m_a.ndim == 3 else mask_2d

    r_out = r_a * (1.0 - mask_r) + r_b * mask_r
    m_out = m_a * (1.0 - mask_m) + m_b * mask_m
    n_out = blend_normals_rnm(n_a, n_b, mask if mask.ndim == 3 else mask[..., np.newaxis])

    return (
        r_out.astype(r_a.dtype),
        m_out.astype(m_a.dtype),
        n_out.astype(n_a.dtype),
    )