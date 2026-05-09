"""
app.py – Streamlit entry point for the MatForge application.

Orchestrates the UI, session state, and calls to all business-logic
modules. No inference or image-processing logic lives here.
"""

from __future__ import annotations

import gc
import io

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image

from src.classifier import classify_material
from src.export import export_maps
from src.inference import run_inference
from src.postprocess import adjust_gain_offset
from src.quality import evaluate_normal_quality
from src.sr import release_sr_model, run_sr
from src.ui_components import (
    build_threejs_viewer,
    image_to_base64,
    inject_css,
    normal_map_to_display,
    render_ai_output_notice,
    render_comparison_slider,
    render_map_grid,
)
from src.utils import (
    apply_zoom,
    get_effective_resolution,
    invalidate_session_keys,
    load_and_validate_image,
)

# ---------------------------------------------------------------------------
# Page configuration & design system
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MatForge",
    layout="wide",
    page_icon="🔨",
)
inject_css()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Session state management
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Create session-state keys with safe defaults if they do not exist."""
    defaults = {
        "input_image": None,
        "zoom": 1.0,
        "use_sr": False,
        "maps": None,
        "group_label": None,
        "knn_distance": None,
        "asset_name": "material",
        "engine": "Blender",
        "viewer_geometry": "sphere",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ---------------------------------------------------------------------------
# Time estimation helper (to be moved to utils in a future refactor)
# ---------------------------------------------------------------------------
def estimate_processing_time(
    image: Image.Image,
    zoom: float,
    use_sr: bool,
) -> tuple[int, float]:
    """Estimate tile count and total processing time in seconds.

    Based on benchmarked throughput on GTX 1650 Max-Q with float32.
    Constants are provisional until measured on real hardware.

    Args:
        image: Input PIL image before zoom is applied.
        zoom: Zoom factor (0.1–1.0).
        use_sr: Whether the SR module will be active.

    Returns:
        Tuple of (n_tiles, estimated_seconds).
    """
    TILE_SIZE = 256
    STRIDE = 128
    MATFORGE_SPT = 0.15          # seconds per tile, provisional
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🔨 MatForge")
st.sidebar.caption("PBR Map Prediction")
st.sidebar.divider()

# ----- file uploader -------------------------------------------------------
def _on_file_change() -> None:
    invalidate_session_keys(["maps", "group_label", "knn_distance"])
    st.session_state["input_image"] = None

uploaded_file = st.sidebar.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "webp"],
    on_change=_on_file_change,
)
if uploaded_file is not None:
    # store the PIL image directly, replacing any previous value
    st.session_state["input_image"] = Image.open(uploaded_file).convert("RGB")

# ----- zoom slider ---------------------------------------------------------
zoom = st.sidebar.slider(
    "Zoom",
    min_value=0.1,
    max_value=1.0,
    step=0.05,
    value=st.session_state["zoom"],
    key="zoom",
    help=(
        "Controls how much of the image each 256×256 tile sees before "
        "inference. Rule of thumb: 1K → 1.0 · 2K → 0.5 · 4K → 0.25. "
        "Lower zoom also reduces processing time."
    ),
)
if st.session_state["input_image"] is not None:
    eff_w, eff_h = get_effective_resolution(
        st.session_state["input_image"].size,
        st.session_state["zoom"],
        st.session_state["use_sr"],
    )
    st.sidebar.caption(f"Effective input to MatForge: {eff_w}×{eff_h} px")

# ----- SR toggle -----------------------------------------------------------
use_sr = st.sidebar.checkbox(
    "Super-Resolution (×4)",
    key="use_sr",
)
if use_sr:
    st.sidebar.warning("SR adds ~8–10 s processing time.")

# ----- time estimate -------------------------------------------------------
if st.session_state["input_image"] is not None:
    n_tiles, est_secs = estimate_processing_time(
        st.session_state["input_image"],
        st.session_state["zoom"],
        st.session_state["use_sr"],
    )
    st.sidebar.caption(
        f"Estimated time: ~{int(est_secs)} s · {n_tiles} tiles"
    )

# ----- generate button -----------------------------------------------------
generate = st.sidebar.button(
    "Generate Maps",
    type="primary",
    use_container_width=True,
)

# ----- export section (only when maps are ready) ----------------------------
if st.session_state["maps"] is not None:
    st.sidebar.divider()
    st.sidebar.markdown("### Export")

    asset_name = st.sidebar.text_input("Asset name", key="asset_name")
    engine = st.sidebar.selectbox(
        "Engine",
        options=["Blender", "Unreal Engine 5", "Unity URP",
                 "Unity HDRP", "Godot 4"],
        key="engine",
    )

    _ENGINE_KEY = {
        "Blender": "blender",
        "Unreal Engine 5": "ue5",
        "Unity URP": "unity_urp",
        "Unity HDRP": "unity_hdrp",
        "Godot 4": "godot",
    }

    zip_bytes = export_maps(
        normal=st.session_state["maps"]["normal"],
        roughness=st.session_state["maps"]["roughness"],
        metallic=st.session_state["maps"]["metallic"],
        asset_name=asset_name,
        engines=[_ENGINE_KEY[engine]],
    )
    st.sidebar.download_button(
        label=f"Download {engine} pack",
        data=zip_bytes,
        file_name=f"{asset_name}_{_ENGINE_KEY[engine]}.zip",
        mime="application/zip",
        use_container_width=True,
    )

# ----- tools section (only when maps are ready) ----------------------------
if st.session_state["maps"] is not None:
    st.sidebar.divider()
    st.sidebar.markdown("### Tools")

    # -- Adjust R/M ----------------------------------------------------------
    with st.sidebar.expander("Adjust R/M"):
        r_gain = st.slider("Roughness gain", 0.5, 2.0, 1.0, 0.05, key="r_gain")
        r_offset = st.slider("Roughness offset", -0.5, 0.5, 0.0, 0.05, key="r_offset")
        m_gain = st.slider("Metallic gain", 0.5, 2.0, 1.0, 0.05, key="m_gain")
        m_offset = st.slider("Metallic offset", -0.5, 0.5, 0.0, 0.05, key="m_offset")
        if st.button("Apply R/M adjustments", key="apply_rm"):
            st.session_state["maps"]["roughness"] = adjust_gain_offset(
                st.session_state["maps"]["roughness"], r_gain, r_offset
            )
            st.session_state["maps"]["metallic"] = adjust_gain_offset(
                st.session_state["maps"]["metallic"], m_gain, m_offset
            )

    # -- Normal Map Quality --------------------------------------------------
    with st.sidebar.expander("Normal Map Quality"):
        if st.button("Evaluate quality", key="eval_quality"):
            result = evaluate_normal_quality(st.session_state["maps"]["normal"])
            st.metric("Overall score", f"{result['overall_score']:.2f}")
            if result["warnings"]:
                for w in result["warnings"]:
                    st.warning(w)
            st.image(result["heatmap"], caption="Diagnostic heatmap",
                     use_container_width=True)

    # -- Material Group ------------------------------------------------------
    with st.sidebar.expander("Material Group"):
        if st.session_state["group_label"] is not None:
            st.metric("Group", st.session_state["group_label"])
            st.caption(f"KNN distance: {st.session_state['knn_distance']:.3f}")
        else:
            st.caption("Run Generate Maps first.")


# ===========================================================================
# GENERATE pipeline
# ===========================================================================
if generate:
    if st.session_state["input_image"] is None:
        st.error("Please upload an image first.")
    else:
        try:
            with st.status("Generating PBR maps…", expanded=True) as status_box:
                # 1. Load & zoom
                status_box.write("Loading image…")
                img = load_and_validate_image(st.session_state["input_image"])
                img = apply_zoom(img, st.session_state["zoom"])

                # 2. Optional SR
                if st.session_state["use_sr"]:
                    status_box.write("Running Super-Resolution…")
                    img = run_sr(img)
                    release_sr_model()
                    gc.collect()
                    torch.cuda.empty_cache()

                # 3. Material classification
                status_box.write("Classifying material…")
                label, distance = classify_material(img)
                st.session_state["group_label"] = label
                st.session_state["knn_distance"] = distance

                # 4. MatForge inference
                status_box.write("Predicting PBR maps…")
                maps = run_inference(img)
                st.session_state["maps"] = maps

                status_box.update(label="Done.", state="complete", expanded=False)

        except Exception as exc:
            st.error(str(exc))

# ===========================================================================
# Main area
# ===========================================================================
if st.session_state["maps"] is None:
    st.info(
        "Upload an image in the sidebar and click **Generate Maps** "
        "to predict Normal, Roughness and Metallic maps."
    )
    if st.session_state["input_image"] is not None:
        st.image(
            st.session_state["input_image"],
            caption="Uploaded image",
            use_container_width=False,
        )
else:
    maps = st.session_state["maps"]
    normal = maps["normal"]
    roughness = maps["roughness"]
    metallic = maps["metallic"]

    # 1. Map grid
    render_map_grid(normal, roughness, metallic)

    # 2. AI content notice
    render_ai_output_notice()

    # 3. 3D Preview expander
    with st.expander("3D Preview", expanded=True):
        geo = st.radio(
            "Geometry",
            options=["sphere", "box", "plane"],
            key="viewer_geometry",
            horizontal=True,
        )

        def _cap_texture(arr: np.ndarray) -> np.ndarray:
            if arr.shape[0] > 1024 or arr.shape[1] > 1024:
                pil = Image.fromarray(
                    (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                )
                pil = pil.resize((1024, 1024), Image.LANCZOS)
                return np.array(pil).astype(np.float32) / 255.0
            return arr

        normal_disp = normal_map_to_display(normal)
        normal_disp_capped = _cap_texture(normal_disp)
        rough_capped = _cap_texture(roughness.squeeze())
        metal_capped = _cap_texture(metallic.squeeze())

        n_b64 = image_to_base64(normal_disp_capped)
        r_b64 = image_to_base64(rough_capped)
        m_b64 = image_to_base64(metal_capped)

        viewer_html = build_threejs_viewer(
            normal_b64=n_b64,
            roughness_b64=r_b64,
            metallic_b64=m_b64,
            geometry=st.session_state["viewer_geometry"],
            height=620,
        )
        components.html(viewer_html, height=620)

    # 4. Comparison expander
    with st.expander("Comparison", expanded=False):
        if st.session_state["input_image"] is not None:
            input_b64 = image_to_base64(
                np.array(st.session_state["input_image"]).astype(np.float32) / 255.0
            )
            normal_b64_disp = image_to_base64(normal_disp_capped)
            render_comparison_slider(input_b64, normal_b64_disp, height=400)
        else:
            st.caption("No reference image available for comparison.")