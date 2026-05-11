# -*- coding: utf-8 -*-

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
from src.postprocess import adjust_gain_offset, blend_materials, calibrate_by_group, generate_variations, make_tileable_frequency
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
    estimate_processing_time,
    get_effective_resolution,
    invalidate_session_keys,
    load_and_validate_image,
    apply_perspective_warp,
)
from src.ui_components import TEXT_PRIMARY, TEXT_SECONDARY 

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
        "maps_raw": None,         # unmodified inference output, source of truth for R/M adjust
        "maps_calibrated": None,  # state after calibrate_by_group
        "maps_tileable": None,    # state after make_tileable_frequency
        "maps_adjusted": None,    # state after adjust_gain_offset
        "maps_blended": None,     # state after blend_materials
        "original_image": None,   # pre-SR image, preserved for display metadata
        "sr_was_used": False,     # True if SR was active during last generation
        "maps_variations": None,  # state after generate_variations (selected variant)
        "export_state": "Raw",    # default export state
        "viewer_state": "Raw",    # default 3D viewer state
        "tile_preview_state": "Raw",  # default tiling preview state
        "tile_preview_map": "Normal", # default tiling preview map
        "warp_points": None,      # list[list[int]] — 4 puntos en coords de imagen original
        "warped_image": None,     # PIL.Image — resultado del warp, o None
        "warp_confirmed": False,  # True cuando el usuario hace Apply
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# Apply any pending state redirects before widgets are instantiated
for _pending, _target in [
    ("_pending_viewer_state", "viewer_state"),
    ("_pending_export_state", "export_state"),
    ("_pending_tile_state",   "tile_preview_state"),
]:
    if st.session_state.get(_pending):
        st.session_state[_target] = st.session_state[_pending]
        st.session_state[_pending] = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div style="padding: 16px 8px 8px 8px;">
        <div style="font-family:'Inter',sans-serif; font-size:20px;
                    font-weight:500; color:{TEXT_PRIMARY};
                    letter-spacing:-0.01em;">
            🔨 MatForge
        </div>
        <div style="font-size:11px; color:{TEXT_SECONDARY};
                    margin-top:2px; letter-spacing:0.04em;">
            PBR Map Prediction
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()

# ----- file uploader -------------------------------------------------------
def _on_file_change() -> None:
    invalidate_session_keys([
        "maps", "maps_raw", "maps_calibrated", "maps_tileable",
        "maps_adjusted", "maps_blended", "maps_variations",
        "group_label", "knn_distance",
        "warp_points", "warped_image",
    ])
    st.session_state["input_image"] = None
    st.session_state["original_image"] = None
    st.session_state["sr_was_used"] = False
    st.session_state["warp_confirmed"] = False

uploaded_file = st.sidebar.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "webp"],
    on_change=_on_file_change,
)
if uploaded_file is not None:
    # store the PIL image directly, replacing any previous value
    st.session_state["input_image"] = Image.open(uploaded_file).convert("RGB")
    st.session_state["original_image"] = st.session_state["input_image"]

# ----- zoom slider ---------------------------------------------------------
zoom = st.sidebar.slider(
    "Zoom",
    min_value=0.1,
    max_value=1.0,
    step=0.05,
    key="zoom",
    help=(
        "Controls how much of the image each 256×256 tile sees before "
        "inference. Rule of thumb: 1K → 1.0 · 2K → 0.5 · 4K → 0.25. "
        "Lower zoom also reduces processing time."
    ),
)
if st.session_state["input_image"] is not None:
    _ref_img = st.session_state.get("original_image") or st.session_state["input_image"]
    eff_w, eff_h = get_effective_resolution(
        _ref_img.size,
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
    if st.session_state["input_image"] is not None:
        _ref_img = st.session_state.get("original_image") or st.session_state["input_image"]
        _sr_w = int(_ref_img.width * st.session_state["zoom"]) * 4
        _sr_h = int(_ref_img.height * st.session_state["zoom"]) * 4
        if _sr_w > 1024 or _sr_h > 1024:
            st.sidebar.error(
                f"SR will produce a {_sr_w}×{_sr_h} px image. "
                f"MatForge was trained on 1K patches — results above 1024 px "
                f"are likely to be flat or incoherent. "
                f"Reduce zoom before enabling SR."
            )

# ----- time estimate -------------------------------------------------------
if st.session_state["input_image"] is not None:
    _ref_for_estimate = st.session_state.get("original_image") or st.session_state["input_image"]
    n_tiles, est_secs = estimate_processing_time(
        _ref_for_estimate,
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

# ===========================================================================
# GENERATE pipeline
# ===========================================================================
if generate:
    if st.session_state["input_image"] is None:
        st.error("Please upload an image first.")
    else:
        try:
            with st.status("Generating PBR maps…", expanded=True) as status_box:
                # 0. Perspective warp injection
                img = st.session_state["input_image"]
                if st.session_state["warp_confirmed"] and st.session_state["warped_image"] is not None:
                    img = st.session_state["warped_image"]

                # 1. Load & zoom
                status_box.write("Loading image…")
                # Prevent zoom from reducing the image below one tile dimension.
                _min_zoom = max(0.1, 256 / min(img.width, img.height))
                if st.session_state["zoom"] < _min_zoom:
                    st.warning(
                        f"Zoom {st.session_state['zoom']:.2f} is too low for this image size "
                        f"({img.width}×{img.height} px). "
                        f"Minimum safe zoom is {_min_zoom:.2f}. Clamping automatically."
                    )
                img = apply_zoom(img, max(st.session_state["zoom"], _min_zoom))

                # 2. Optional SR
                st.session_state["sr_was_used"] = st.session_state["use_sr"]
                if st.session_state["use_sr"]:
                    status_box.write("Running Super-Resolution…")
                    img = run_sr(img)
                    release_sr_model()
                    gc.collect()
                    torch.cuda.empty_cache()
                    st.session_state["input_image"] = img

                # 3. Material classification
                status_box.write("Classifying material…")
                label, distance = classify_material(img)
                st.session_state["group_label"] = label
                st.session_state["knn_distance"] = distance

                # 4. MatForge inference
                status_box.write("Predicting PBR maps…")
                maps = run_inference(img)
                st.session_state["maps"] = maps
                st.session_state["maps_raw"] = {
                    "normal": maps["normal"].copy(),
                    "roughness": maps["roughness"].copy(),
                    "metallic": maps["metallic"].copy(),
                }
                # Reset derived states and viewer to Raw on every new generation
                st.session_state["maps_adjusted"]  = None
                st.session_state["maps_calibrated"] = None
                st.session_state["maps_tileable"]   = None
                st.session_state["maps_blended"]    = None
                st.session_state["_variations_list"] = None
                st.session_state["maps_variations"] = None
                st.session_state["viewer_state"]    = "Raw"
                st.session_state["export_state"]    = "Raw"

                status_box.update(label="Done.", state="complete", expanded=False)

        except Exception as exc:
            st.error(str(exc))

# Export and Tools rendered after pipeline so session state is current.
if st.session_state["maps"] is not None:
    st.sidebar.divider()
    st.sidebar.markdown("### Tools")

    with st.sidebar.expander("Adjust R/M"):
        r_gain = st.slider("Roughness gain", 0.5, 2.0, 1.0, 0.05, key="r_gain")
        r_offset = st.slider("Roughness offset", -0.5, 0.5, 0.0, 0.05,
                            key="r_offset")
        m_gain = st.slider("Metallic gain", 0.5, 2.0, 1.0, 0.05, key="m_gain")
        m_offset = st.slider("Metallic offset", -0.5, 0.5, 0.0, 0.05,
                            key="m_offset")
        if st.button("Apply R/M adjustments", key="apply_rm"):
            raw = st.session_state["maps_raw"]
            st.session_state["maps"]["roughness"] = adjust_gain_offset(
                raw["roughness"], r_gain, r_offset
            )
            st.session_state["maps"]["metallic"] = adjust_gain_offset(
                raw["metallic"], m_gain, m_offset
            )
            st.session_state["maps_adjusted"] = {
                "normal":    st.session_state["maps"]["normal"].copy(),
                "roughness": st.session_state["maps"]["roughness"].copy(),
                "metallic":  st.session_state["maps"]["metallic"].copy(),
            }

    with st.sidebar.expander("Normal Map Quality"):
        if st.button("Evaluate quality", key="eval_quality"):
            result = evaluate_normal_quality(st.session_state["maps"]["normal"])
            st.metric("Overall score", f"{result['overall_score']:.2f}")
            if result["warnings"]:
                _hard_edge_groups = {
                    "brick_terracotta", "stone_rough",
                    "concrete_plaster", "mixed_ambiguous"
                }
                _is_hard_edge = (
                    st.session_state["group_label"] in _hard_edge_groups
                )
                for w in result["warnings"]:
                    if _is_hard_edge and "continuity" in w:
                        st.info(
                            f"{w} — expected for hard-edge materials "
                            f"({st.session_state['group_label']})."
                        )
                    elif _is_hard_edge and "blockiness" in w:
                        st.info(
                            f"{w} — may reflect natural pattern repetition "
                            f"in {st.session_state['group_label']}."
                        )
                    else:
                        st.warning(w)
            st.image(result["heatmap"], caption="Diagnostic heatmap",
                     use_container_width=True)

    with st.sidebar.expander("Calibrate by Group"):
        if st.session_state["group_label"] is not None:
            st.caption(
                f"Detected: **{st.session_state['group_label']}** "
                f"(KNN distance: {st.session_state['knn_distance']:.3f})"
            )
            _groups = [
                "stone_rough", "concrete_plaster", "brick_terracotta",
                "mixed_ambiguous", "wood", "ceramic_ground",
                "marble_smooth", "metal",
            ]
            selected_group = st.selectbox(
                "Material group override",
                options=_groups,
                index=_groups.index(st.session_state["group_label"])
                      if st.session_state["group_label"] in _groups else 0,
                key="calibration_group_override",
                help="Override the detected group if the classifier is wrong.",
            )
            # Recalculate alpha — if overridden, use full confidence.
            knn_dist = st.session_state["knn_distance"]
            if selected_group != st.session_state["group_label"]:
                alpha = 1.0
                st.caption("Manual override — calibration confidence: 1.00")
            else:
                alpha = max(0.0, min(1.0, 1.0 - knn_dist * 3.0))
                st.caption(f"Calibration confidence: {alpha:.2f}")
            if alpha < 0.3:
                st.warning(
                    "Low confidence — calibration will have minimal effect. "
                    "Consider adjusting manually via Adjust R/M."
                )
            if st.button("Apply Calibration", key="apply_calibration"):
                r_cal, m_cal = calibrate_by_group(
                    roughness=st.session_state["maps_raw"]["roughness"],
                    metallic=st.session_state["maps_raw"]["metallic"],
                    group=selected_group,
                    knn_distance=0.0 if selected_group !=
                                 st.session_state["group_label"] else knn_dist,
                )
                st.session_state["maps"]["roughness"] = r_cal
                st.session_state["maps"]["metallic"]  = m_cal
                st.session_state["maps_calibrated"] = {
                    "normal":    st.session_state["maps"]["normal"].copy(),
                    "roughness": st.session_state["maps"]["roughness"].copy(),
                    "metallic":  st.session_state["maps"]["metallic"].copy(),
                }
                st.success(f"Calibration applied for group: {selected_group}.")
        else:
            st.caption("Run Generate Maps first.")

    with st.sidebar.expander("Make Tileable (Beta)"):
        st.caption(
            "Removes low-frequency gradients from Roughness and Metallic "
            "that break tileability after tile-and-merge inference. "
            "Enable SR Seam Blend if the image was processed with SR."
        )
        st.caption(
            "Caution: Does not correct large-scale pattern repetition or "
            "directional lighting baked into the source photo."
        )
        # Source info
        _til_active = (
            "Calibrated" if st.session_state["maps_calibrated"] is not None
            else "Adjusted" if st.session_state["maps_adjusted"] is not None
            else "Raw"
        )
        st.caption(f"ℹ Operates on: **{_til_active}** maps.")
        sr_seam = st.checkbox(
            "SR seam blend (use if SR was active)",
            value=st.session_state.get("sr_was_used", False),
            key="tile_sr_seam",
        )
        sigma = st.slider(
            "Low-freq sigma", 32.0, 128.0, 64.0, 8.0,
            key="tile_sigma",
            help="Higher values remove broader gradients.",
        )
        if st.button("Apply Tileable", key="apply_tileable"):
            # Resolve source: use most advanced available state
            _til_source = (
                st.session_state["maps_calibrated"]
                or st.session_state["maps_adjusted"]
                or st.session_state["maps_raw"]
            )
            result = make_tileable_frequency(
                normal=_til_source["normal"],
                roughness=_til_source["roughness"],
                metallic=_til_source["metallic"],
                sr_active=sr_seam,
                sigma=sigma,
            )
            st.session_state["maps"]["normal"]    = result["normal"]
            st.session_state["maps"]["roughness"] = result["roughness"]
            st.session_state["maps"]["metallic"]  = result["metallic"]
            st.session_state["maps_tileable"] = {
                "normal":    result["normal"].copy(),
                "roughness": result["roughness"].copy(),
                "metallic":  result["metallic"].copy(),
            }
            st.success("Tileable maps applied.")

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

    # Build available export states dynamically
    _export_states = {"Raw": "maps_raw"}
    if st.session_state["maps_adjusted"] is not None:
        _export_states["Adjusted"] = "maps_adjusted"
    if st.session_state["maps_blended"] is not None:
        _export_states["Blended"] = "maps_blended"
    if st.session_state["maps_variations"] is not None:
        _export_states["Variation"] = "maps_variations"
    if st.session_state["maps_calibrated"] is not None:
        _export_states["Calibrated"] = "maps_calibrated"
    if st.session_state["maps_tileable"] is not None:
        _export_states["Tileable"] = "maps_tileable"

    export_state_label = st.sidebar.selectbox(
        "Export state",
        options=list(_export_states.keys()),
        key="export_state",
        help="Choose which version of the maps to export.",
    )
    _export_source = st.session_state[_export_states[export_state_label]]

    # Resolve color: use blended color if available, otherwise input image
    _export_color = _export_source.get("color")
    if _export_color is None and st.session_state["input_image"] is not None:
        _export_color = np.array(
            st.session_state["input_image"], dtype=np.float32
        ) / 255.0

    zip_bytes = export_maps(
        normal=_export_source["normal"],
        roughness=_export_source["roughness"],
        metallic=_export_source["metallic"],
        asset_name=asset_name,
        engines=[_ENGINE_KEY[engine]],
        color=_export_color,
    )
    st.sidebar.download_button(
        label=f"Download {engine} pack",
        data=zip_bytes,
        file_name=f"{asset_name}_{_ENGINE_KEY[engine]}.zip",
        mime="application/zip",
        use_container_width=True,
    )

# ===========================================================================
# Main area
# ===========================================================================

# Title and device info
st.markdown(
    f'<h1 style="margin-bottom:0;">🔨 MatForge</h1>'
    f'<p style="color:#9A9890; font-size:14px; margin-top:4px; margin-bottom:24px;">'
    f'PBR Map Prediction · {DEVICE.upper()}</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Perspective Correction (always visible, independent of maps state)
# ---------------------------------------------------------------------------
def _parse_warp_points(raw: str) -> list[list[int]] | None:
    """Parse 'x0,y0,x1,y1,x2,y2,x3,y3' into [[x0,y0], ...]. Returns None if malformed."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        values = [int(float(v)) for v in raw.split(",")]
        if len(values) != 8:
            return None
        return [[values[i * 2], values[i * 2 + 1]] for i in range(4)]
    except (ValueError, IndexError):
        return None


def _render_perspective_canvas(image: Image.Image, canvas_height: int = 460) -> None:
    """Render the interactive 4-point perspective correction canvas."""
    import json
    img_np = np.array(image)
    ih, iw = img_np.shape[:2]

    # Derive canvas height from the image aspect ratio so drawImage
    # fills the canvas without stretching. Cap at canvas_height.
    container_width = 704  # approximate main area width in Streamlit wide layout
    aspect_ratio = ih / iw
    computed_height = min(canvas_height, int(container_width * aspect_ratio))
    canvas_height = max(200, computed_height)

    # Build base64 of the source image for embedding in the canvas.
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = __import__("base64").b64encode(buf.getvalue()).decode()

    margin_x = int(iw * 0.10)
    margin_y = int(ih * 0.10)
    default_pts = [
        [margin_x,        margin_y],
        [iw - margin_x,   margin_y],
        [iw - margin_x,   ih - margin_y],
        [margin_x,        ih - margin_y],
    ]
    pts_init = (
        st.session_state["warp_points"]
        if st.session_state["warp_points"] is not None
        else default_pts
    )
    pts_json = json.dumps(pts_init)

    html = f"""
<style>
  #wc {{ position:relative; width:100%; user-select:none; }}
  #cv {{ width:100%; display:block; border-radius:8px; cursor:crosshair; }}
  #wh {{ margin-top:6px; font-size:12px; color:#9A9890; font-family:sans-serif; }}
</style>
<div id="wc"><canvas id="cv"></canvas>
<p id="wh">Drag the handles to frame the surface, then click <strong>↩ Update preview</strong> below.</p></div>
<script>
(function(){{
  const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
  const IW={iw}, IH={ih};
  const img=new Image(); img.src='data:image/png;base64,{b64}';
  let pts={pts_json}, drag=-1;
  const R=11;
  const i2c=(x,y)=>[x*cv.width/IW, y*cv.height/IH];
  const c2i=(x,y)=>[x*IW/cv.width, y*IH/cv.height];
  function rsz(){{
    const r=cv.getBoundingClientRect();
    cv.width = r.width || 800;
    cv.height = cv.width * (IH / IW);
    draw();
  }}
  function draw(){{
    if(!img.complete) return;
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.drawImage(img,0,0,cv.width,cv.height);
    const [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]=pts.map(p=>i2c(...p));
    ctx.save(); ctx.beginPath();
    ctx.rect(0,0,cv.width,cv.height);
    ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.lineTo(x2,y2); ctx.lineTo(x3,y3); ctx.closePath();
    ctx.fillStyle='rgba(0,0,0,0.52)'; ctx.fill('evenodd'); ctx.restore();
    ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.lineTo(x2,y2); ctx.lineTo(x3,y3); ctx.closePath();
    ctx.strokeStyle='#E8A835'; ctx.lineWidth=2; ctx.stroke();
    [[x0,y0],[x1,y1],[x2,y2],[x3,y3]].forEach(([cx,cy])=>{{
      ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2);
      ctx.fillStyle='#E8A835'; ctx.fill();
      ctx.strokeStyle='#1C1B18'; ctx.lineWidth=2; ctx.stroke();
    }});
  }}
  function hit(cx,cy){{ for(let i=0;i<4;i++){{ const [hx,hy]=i2c(...pts[i]); if(Math.hypot(cx-hx,cy-hy)<=R+5) return i; }} return -1; }}
  function epos(e){{ const r=cv.getBoundingClientRect(); return e.touches?[e.touches[0].clientX-r.left,e.touches[0].clientY-r.top]:[e.clientX-r.left,e.clientY-r.top]; }}
  function push(){{
    const flat=pts.map(p=>[Math.round(p[0]),Math.round(p[1])].join(',')).join(',');
    try{{ const u=new URL(window.parent.location.href); u.searchParams.set('warp',flat); window.parent.history.replaceState(null,'',u.toString()); }}catch(e){{}}
  }}
  cv.addEventListener('mousedown',e=>{{ drag=hit(...epos(e)); }});
  window.addEventListener('mouseup',()=>{{ if(drag>=0) push(); drag=-1; }});
  cv.addEventListener('mousemove',e=>{{ if(drag<0) return; const [cx,cy]=epos(e); pts[drag]=c2i(Math.max(0,Math.min(cv.width,cx)),Math.max(0,Math.min(cv.height,cy))); draw(); }});
  cv.addEventListener('touchstart',e=>{{e.preventDefault(); drag=hit(...epos(e));}},{{passive:false}});
  window.addEventListener('touchend',()=>{{ if(drag>=0) push(); drag=-1; }});
  cv.addEventListener('touchmove',e=>{{ e.preventDefault(); if(drag<0) return; const [cx,cy]=epos(e); pts[drag]=c2i(Math.max(0,Math.min(cv.width,cx)),Math.max(0,Math.min(cv.height,cy))); draw(); }},{{passive:false}});
  img.onload=rsz; window.addEventListener('resize',rsz); if(img.complete) rsz();
}})();
</script>"""
    components.html(html, height=800, scrolling=True)


if st.session_state["input_image"] is not None:
    with st.expander("Perspective Correction", expanded=False):
        st.caption(
            "Drag the **four amber handles** to frame the flat surface. "
            "Click **↩ Update preview** to apply and preview the crop. "
            "Then click **Apply** to use it as pipeline input."
        )

        if st.button("↩ Update preview", use_container_width=True, key="btn_warp_update"):
            warp_param = st.query_params.get("warp", "")
            if warp_param:
                pts = _parse_warp_points(warp_param)
                if pts is not None:
                    st.session_state["warp_points"] = pts
                    try:
                        st.session_state["warped_image"] = apply_perspective_warp(
                            st.session_state["input_image"], pts
                        )
                    except Exception as exc:
                        st.error(f"Warp failed: {exc}")
                        st.session_state["warped_image"] = None
                st.query_params.clear()

        _render_perspective_canvas(st.session_state["input_image"])

        if st.session_state["warped_image"] is not None:
            col_w, col_t = st.columns(2)
            with col_w:
                st.caption("**Corrected crop**")
                st.image(
                    st.session_state["warped_image"],
                    use_container_width=True,
                )
            with col_t:
                st.caption("**Tiling 2×2 preview**")
                _warp_np = np.array(st.session_state["warped_image"])
                _tiled = np.concatenate([
                    np.concatenate([_warp_np, _warp_np], axis=1),
                    np.concatenate([_warp_np, _warp_np], axis=1),
                ], axis=0)
                st.image(_tiled, use_container_width=True)

        col_apply, col_reset = st.columns([2, 1])
        with col_apply:
            if st.button(
                "Apply perspective correction",
                use_container_width=True,
                key="btn_warp_apply",
                disabled=st.session_state["warped_image"] is None,
            ):
                st.session_state["warp_confirmed"] = True
                invalidate_session_keys([
                    "maps", "maps_raw", "maps_calibrated", "maps_tileable",
                    "group_label", "knn_distance",
                ])
                st.rerun()
        with col_reset:
            if st.button("↺ Reset", use_container_width=True, key="btn_warp_reset"):
                st.session_state["warp_points"] = None
                st.session_state["warped_image"] = None
                st.session_state["warp_confirmed"] = False
                invalidate_session_keys([
                    "maps", "maps_raw", "maps_calibrated", "maps_tileable",
                    "group_label", "knn_distance",
                ])
                st.rerun()

        if st.session_state["warp_confirmed"]:
            st.info("**Correction active** — pipeline will use the corrected crop.")

if st.session_state["maps"] is None:
    st.info(
        "Upload an image in the sidebar and click **Generate Maps** "
        "to predict Normal, Roughness and Metallic maps."
    )
    if st.session_state["input_image"] is not None:
            _preview = st.session_state["input_image"].copy()
            if _preview.width > 480:
                _r = 480 / _preview.width
                _preview = _preview.resize(
                    (480, int(_preview.height * _r)), Image.LANCZOS
                )
            st.image(_preview, caption="Uploaded image", use_container_width=False)
else:
    maps = st.session_state["maps"]
    normal = maps["normal"]
    roughness = maps["roughness"]
    metallic = maps["metallic"]

    # 1. Input reference + map grid
    if st.session_state["input_image"] is not None:
        with st.expander("Input Image", expanded=False):
            col_img, col_info = st.columns([2, 1])
            with col_img:
                st.image(
                    st.session_state["input_image"],
                    caption="Source image",
                    use_container_width=True,
                )
            with col_info:
                w, h = st.session_state["input_image"].size
                st.caption(f"**Resolution:** {w}×{h} px")
                st.caption(f"**Zoom applied:** {st.session_state['zoom']:.2f}")
                if st.session_state["group_label"]:
                    st.caption(f"**Material group:** {st.session_state['group_label']}")
                    st.caption(f"**KNN distance:** {st.session_state['knn_distance']:.3f}")
                if st.session_state["use_sr"]:
                    st.caption("**SR:** active (×4)")

    # 2. Map grid
    render_map_grid(normal, roughness, metallic)

    # 4. AI content notice
    render_ai_output_notice()

    # 5. 3D Preview expander
    with st.expander("3D Preview", expanded=True):
        # State selector — mirrors the export state logic
        _viewer_states = {"Raw": "maps_raw"}
        if st.session_state["maps_adjusted"] is not None:
            _viewer_states["Adjusted"] = "maps_adjusted"
        if st.session_state["maps_blended"] is not None:
            _viewer_states["Blended"] = "maps_blended"
        if st.session_state["maps_variations"] is not None:
            _viewer_states["Variation"] = "maps_variations"
        if st.session_state["maps_calibrated"] is not None:
            _viewer_states["Calibrated"] = "maps_calibrated"
        if st.session_state["maps_tileable"] is not None:
            _viewer_states["Tileable"] = "maps_tileable"

        viewer_state_label = st.selectbox(
            "Preview state",
            options=list(_viewer_states.keys()),
            key="viewer_state",
            help="Choose which version of the maps to preview in 3D.",
        )
        _viewer_source = st.session_state[_viewer_states[viewer_state_label]]
        _v_normal   = _viewer_source["normal"]
        _v_roughness = _viewer_source["roughness"]
        _v_metallic  = _viewer_source["metallic"]

        col_geo, col_color = st.columns(2)
        with col_geo:
            geo = st.radio(
                "Geometry",
                options=["sphere", "box", "plane"],
                key="viewer_geometry",
                horizontal=True,
            )
        with col_color:
            show_color = st.checkbox(
                "Show source color",
                value=False,
                key="viewer_show_color",
                help="Uses the original image as albedo. "
                     "Disables neutral grey base color.",
            )
            show_env = st.checkbox(
                "Room environment",
                value=False,
                key="viewer_show_env",
                help="Adds a room environment map for correct metallic "
                     "rendering. Recommended for metals — may look too "
                     "bright on non-metallic materials.",
            )

        def _cap_texture(arr: np.ndarray) -> np.ndarray:
            if arr.shape[0] > 1024 or arr.shape[1] > 1024:
                pil = Image.fromarray(
                    (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                )
                pil = pil.resize((1024, 1024), Image.LANCZOS)
                return np.array(pil).astype(np.float32) / 255.0
            return arr

        normal_disp = normal_map_to_display(_v_normal)
        normal_disp_capped = _cap_texture(normal_disp)
        rough_capped = _cap_texture(_v_roughness.squeeze())
        metal_capped = _cap_texture(_v_metallic.squeeze())

        n_b64 = image_to_base64(normal_disp_capped)
        r_b64 = image_to_base64(rough_capped)
        m_b64 = image_to_base64(metal_capped)

        # Encode source image as albedo if requested.
        color_b64 = None
        if show_color:
            _blended_color = (
                st.session_state["maps_blended"].get("color")
                if st.session_state["maps_blended"] is not None
                else None
            )
            if _blended_color is not None and viewer_state_label == "Blended":
                color_arr = _cap_texture(_blended_color)
                color_b64 = image_to_base64(color_arr)
            elif st.session_state["input_image"] is not None:
                color_arr = np.array(
                    st.session_state["input_image"]
                ).astype(np.float32) / 255.0
                color_arr = _cap_texture(color_arr)
                color_b64 = image_to_base64(color_arr)

        viewer_html = build_threejs_viewer(
            normal_b64=n_b64,
            roughness_b64=r_b64,
            metallic_b64=m_b64,
            geometry=st.session_state["viewer_geometry"],
            height=620,
            color_b64=color_b64,
            show_env=show_env,
        )
        components.html(viewer_html, height=620)

    # 6. Comparison expander
    with st.expander("Comparison", expanded=False):
        if st.session_state["maps_raw"] is None:
            st.caption("Generate maps first to use the comparison tool.")
        else:
            # Build available states for comparison
            comp_states = {"Raw": "maps_raw"}
            if st.session_state["maps_adjusted"] is not None:
                comp_states["Adjusted"] = "maps_adjusted"
            if st.session_state["maps_blended"] is not None:
                comp_states["Blended"] = "maps_blended"
            if st.session_state["maps_variations"] is not None:
                comp_states["Variation"] = "maps_variations"
            if st.session_state["maps_calibrated"] is not None:
                comp_states["Calibrated"] = "maps_calibrated"
            if st.session_state["maps_tileable"] is not None:
                comp_states["Tileable"] = "maps_tileable"

            # Color available if input_image exists or blended color exists
            _comp_has_color = st.session_state["input_image"] is not None
            _comp_map_options = ["Normal", "Roughness", "Metallic"]
            if _comp_has_color:
                _comp_map_options.append("Color")

            selected_map = st.selectbox(
                "Map",
                options=_comp_map_options,
                key="comparison_map",
            )

            col_before, col_after = st.columns(2)
            with col_before:
                before_state = st.selectbox(
                    "Before",
                    options=list(comp_states.keys()),
                    key="before_state",
                )
            with col_after:
                after_state = st.selectbox(
                    "After",
                    options=list(comp_states.keys()),
                    key="after_state",
                )

            def _get_map_arr(state_key: str, map_name: str) -> np.ndarray:
                source = st.session_state[state_key]
                if map_name == "Normal":
                    return normal_map_to_display(source["normal"])
                elif map_name == "Roughness":
                    return source["roughness"].squeeze()
                elif map_name == "Metallic":
                    return source["metallic"].squeeze()
                else:
                    # Color: use blended color if available, else input_image
                    blended_color = source.get("color")
                    if blended_color is not None:
                        return blended_color
                    return np.array(
                        st.session_state["input_image"], dtype=np.float32
                    ) / 255.0

            before_arr = _get_map_arr(comp_states[before_state], selected_map)
            after_arr = _get_map_arr(comp_states[after_state], selected_map)

            before_b64 = image_to_base64(before_arr)
            after_b64 = image_to_base64(after_arr)

            # Labels above the slider
            lbl_col1, lbl_col2 = st.columns(2)
            with lbl_col1:
                st.caption(f"Before: {before_state}")
            with lbl_col2:
                st.caption(f"After: {after_state}")

            render_comparison_slider(before_b64, after_b64, height=400)

    # 7. Material Blender
    with st.expander("Material Blender (RNM)", expanded=False):
        st.caption(
            "Blend the current material (A) with a second material (B) "
            "using Reoriented Normal Mapping. Upload the three maps for "
            "material B and set the blend factor."
        )

        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            b_normal_file = st.file_uploader(
                "Normal map B", type=["png", "jpg", "jpeg"],
                key="blender_normal_b"
            )
        with col_b2:
            b_rough_file = st.file_uploader(
                "Roughness map B", type=["png", "jpg", "jpeg"],
                key="blender_rough_b"
            )
        with col_b3:
            b_metal_file = st.file_uploader(
                "Metallic map B", type=["png", "jpg", "jpeg"],
                key="blender_metal_b"
            )
        with col_b4:
            b_color_file = st.file_uploader(
                "Color map B (optional)", type=["png", "jpg", "jpeg"],
                key="blender_color_b"
            )

        blend_alpha = st.slider(
            "Blend factor (0 = full A, 1 = full B)",
            0.0, 1.0, 0.5, 0.05,
            key="blend_alpha",
        )

        if st.button("Apply Blend", key="apply_blend"):
            if b_normal_file is None or b_rough_file is None or b_metal_file is None:
                st.error("Upload all three maps for material B before blending.")
            else:
                # Load material A from maps_raw (source of truth)
                n_a = st.session_state["maps_raw"]["normal"]
                r_a = st.session_state["maps_raw"]["roughness"]
                m_a = st.session_state["maps_raw"]["metallic"]
                H, W = n_a.shape[:2]

                def _load_map_b(f, channels: int) -> np.ndarray:
                    """Load uploaded map, resize to match material A, return float32."""
                    img_b = Image.open(f).convert("RGB" if channels == 3 else "L")
                    if img_b.size != (W, H):
                        st.warning(
                            f"{f.name}: resized from {img_b.width}×{img_b.height} "
                            f"to {W}×{H} to match material A."
                        )
                        img_b = img_b.resize((W, H), Image.LANCZOS)
                    arr = np.array(img_b, dtype=np.float32) / 255.0
                    if channels == 1:
                        arr = arr[..., np.newaxis]
                    return arr

                n_b_packed = _load_map_b(b_normal_file, 3)
                r_b        = _load_map_b(b_rough_file, 1)
                m_b        = _load_map_b(b_metal_file, 1)

                # Unpack normal B from [0,1] to [-1,1]
                n_b = n_b_packed * 2.0 - 1.0

                # Optional color blend
                color_blended = None
                if b_color_file is not None and st.session_state["input_image"] is not None:
                    c_a = np.array(
                        st.session_state["input_image"].resize((W, H), Image.LANCZOS),
                        dtype=np.float32
                    ) / 255.0
                    c_b_pil = Image.open(b_color_file).convert("RGB")
                    if c_b_pil.size != (W, H):
                        st.warning(
                            f"{b_color_file.name}: resized from "
                            f"{c_b_pil.width}×{c_b_pil.height} to {W}×{H} "
                            f"to match material A."
                        )
                        c_b_pil = c_b_pil.resize((W, H), Image.LANCZOS)
                    c_b = np.array(c_b_pil, dtype=np.float32) / 255.0
                    color_blended = (c_a * (1.0 - blend_alpha) + c_b * blend_alpha).clip(0, 1)

                # Build constant mask from slider
                mask = np.full((H, W, 1), blend_alpha, dtype=np.float32)

                r_out, m_out, n_out = blend_materials(
                    r_a, m_a, n_a,
                    r_b, m_b, n_b,
                    mask,
                )

                st.session_state["maps_blended"] = {
                    "normal":    n_out,
                    "roughness": r_out,
                    "metallic":  m_out,
                    "color":     color_blended,          # None if no color map uploaded
                }
                st.session_state["maps"]["normal"]    = n_out
                st.session_state["maps"]["roughness"] = r_out
                st.session_state["maps"]["metallic"]  = m_out
                st.session_state["_pending_viewer_state"] = "Blended"
                st.session_state["_pending_export_state"] = "Blended"
                st.session_state["_pending_tile_state"]   = "Blended"
                st.rerun()

    # 8. Procedural Variations
    with st.expander("Procedural Variations", expanded=False):
        st.caption(
            "Generate procedural variations of the base material using "
            "noise-based techniques (zonal mixing, worn edges, random scaling). "
        )
        st.caption("ℹ Operates on: **Raw** maps.")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            n_variants = st.slider(
                "Number of variants", 1, 4, 4, 1,
                key="variations_n",
            )
        with col_v2:
            var_seed = st.number_input(
                "Seed", min_value=0, max_value=9999, value=42, step=1,
                key="variations_seed",
            )

        if st.button("Generate Variations", key="btn_generate_variations"):
            _raw = st.session_state["maps_raw"]
            _variants = generate_variations(
                roughness=_raw["roughness"],
                metallic=_raw["metallic"],
                normal=_raw["normal"],
                n_variants=n_variants,
                seed=int(var_seed),
            )
            st.session_state["_variations_list"] = _variants

        if st.session_state.get("_variations_list"):
            _variants = st.session_state["_variations_list"]
            st.caption(f"{len(_variants)} variant(s) generated. Select one to preview and apply.")

            # Preview grid
            _v_cols = st.columns(len(_variants))
            for _i, (_vcol, _v) in enumerate(zip(_v_cols, _variants)):
                with _vcol:
                    _r_disp = (_v["roughness"].squeeze() * 255).clip(0, 255).astype(np.uint8)
                    st.image(_r_disp, caption=f"Variant {_i + 1} — R", use_container_width=True)

            selected_variant = st.selectbox(
                "Select variant to apply",
                options=[f"Variant {i + 1}" for i in range(len(_variants))],
                key="variations_selected",
            )

            if st.button("Apply Variation", key="btn_apply_variation"):
                _idx = int(selected_variant.split()[-1]) - 1
                _v = _variants[_idx]
                _normal = _v.get("normal", st.session_state["maps_raw"]["normal"])
                st.session_state["maps_variations"] = {
                    "normal":    _normal,
                    "roughness": _v["roughness"],
                    "metallic":  _v["metallic"],
                }
                st.session_state["maps"]["normal"]    = _normal
                st.session_state["maps"]["roughness"] = _v["roughness"]
                st.session_state["maps"]["metallic"]  = _v["metallic"]
                st.session_state["_pending_viewer_state"] = "Variation"
                st.session_state["_pending_export_state"] = "Variation"
                st.session_state["_pending_tile_state"]   = "Variation"
                st.rerun()

        if st.session_state["maps_variations"] is not None:
            st.info("**Variation active** — select 'Variation' in viewer and export.")

    # 9. 2x2 Tiling Preview
    with st.expander("Tiling Preview", expanded=False):
        st.caption("2×2 tile preview to verify seamless repetition.")

        # State and map selectors
        _tile_states = {"Raw": "maps_raw"}
        if st.session_state["maps_adjusted"] is not None:
            _tile_states["Adjusted"] = "maps_adjusted"
        if st.session_state["maps_blended"] is not None:
            _tile_states["Blended"] = "maps_blended"
        if st.session_state["maps_variations"] is not None:
            _tile_states["Variation"] = "maps_variations"
        if st.session_state["maps_calibrated"] is not None:
            _tile_states["Calibrated"] = "maps_calibrated"
        if st.session_state["maps_tileable"] is not None:
            _tile_states["Tileable"] = "maps_tileable"

        col_ts, col_tm = st.columns(2)
        with col_ts:
            tile_state_label = st.selectbox(
                "State",
                options=list(_tile_states.keys()),
                key="tile_preview_state",
            )
        with col_tm:
            _tile_map_options = ["Normal", "Roughness", "Metallic"]
            if st.session_state["input_image"] is not None:
                _tile_map_options.append("Color")
            tile_map_label = st.selectbox(
                "Map",
                options=_tile_map_options,
                key="tile_preview_map",
            )
        _tile_source = st.session_state[_tile_states[tile_state_label]]

        if tile_map_label == "Normal":
            _tile_arr = normal_map_to_display(_tile_source["normal"])
        elif tile_map_label == "Roughness":
            _tile_arr = _tile_source["roughness"].squeeze()
        elif tile_map_label == "Metallic":
            _tile_arr = _tile_source["metallic"].squeeze()
        else:
            _blended_c = _tile_source.get("color")
            if _blended_c is not None:
                _tile_arr = _blended_c
            else:
                _tile_arr = np.array(
                    st.session_state["input_image"], dtype=np.float32
                ) / 255.0

        # Cap each tile at 512px before building the 2×2 mosaic (P1)
        _arr_uint8 = (np.clip(_tile_arr, 0, 1) * 255).astype(np.uint8)
        if _arr_uint8.ndim == 2:
            _arr_uint8 = np.stack([_arr_uint8] * 3, axis=-1)
        if _arr_uint8.shape[1] > 512:
            _pil_tile = Image.fromarray(_arr_uint8)
            _scale = 512 / _pil_tile.width
            _pil_tile = _pil_tile.resize(
                (512, int(_pil_tile.height * _scale)), Image.LANCZOS
            )
            _arr_uint8 = np.array(_pil_tile)

        tiled = np.concatenate([
            np.concatenate([_arr_uint8, _arr_uint8], axis=1),
            np.concatenate([_arr_uint8, _arr_uint8], axis=1),
        ], axis=0)
        st.image(
            tiled,
            caption=f"{tile_map_label} ({tile_state_label}) — 2×2 tile",
            use_container_width=True,
        )