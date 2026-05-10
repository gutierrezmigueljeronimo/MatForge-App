"""
src/ui_components.py
Centralised design system and reusable UI components for MatForge.
All visual styling, CSS, and HTML fragments live here – no other module
defines inline styles or CSS.
"""

import base64
import io

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# ----------------------------------------------------------------------
# DESIGN SYSTEM – colour palette and typography
# ----------------------------------------------------------------------

BG_PRIMARY = "#1C1B18"
BG_SECONDARY = "#252420"
BG_TERTIARY = "#2E2D29"
BORDER = "#3A3830"
ACCENT_1 = "#E8A835"
ACCENT_2 = "#C4863A"
TEXT_PRIMARY = "#E8E6DF"
TEXT_SECONDARY = "#9A9890"
SUCCESS = "#4A6741"
WARNING = "#7A5A28"
ERROR = "#7A3030"
INFO = "#2A4A6A"

# Typography: font family set via Google Fonts, sizes and weights defined in CSS.
# H1: 28px weight 500, H2: 20px weight 500, H3: 16px weight 500,
# Body: 14px weight 400, Caption: 12px weight 400, Code: 13px 400 monospace.

# ----------------------------------------------------------------------
# CSS injection (called once at startup)
# ----------------------------------------------------------------------

def inject_css() -> None:
    """Inject the MatForge design-system CSS and the Inter font."""
    google_fonts = (
        '<link href="https://fonts.googleapis.com/css2?'
        'family=Inter:wght@400;500&display=swap" rel="stylesheet">'
    )
    css = f"""
    <style>
    /* ---- Global resets ---- */
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background-color: {BG_PRIMARY};
    }}
    [data-testid="stSidebar"] {{
        background-color: {BG_PRIMARY};
    }}
    html, body, .stApp {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_PRIMARY};
    }}

    /* ---- Headings ---- */
    h1 {{ font-size: 28px; font-weight: 500; color: {TEXT_PRIMARY}; }}
    h2 {{ font-size: 20px; font-weight: 500; color: {TEXT_PRIMARY}; }}
    h3 {{ font-size: 16px; font-weight: 500; color: {TEXT_PRIMARY}; }}

    /* Buttons */
    .stButton > button {{
        background-color: {ACCENT_1};
        color: {BG_PRIMARY};
        border-radius: 6px;
        font-weight: 500;
        border: none;
    }}
    .stButton > button:hover {{
        background-color: {ACCENT_2};
    }}
    /* Secondary button style (applied via a wrapper element) */
    .secondary .stButton > button {{
        background: transparent;
        border: 1px solid {BORDER};
        color: {TEXT_PRIMARY};
    }}

    /* Sliders */
    .stSlider {{
        accent-color: {ACCENT_1};
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
    }}

    /* Captions */
    .stCaption {{
        color: {TEXT_SECONDARY};
        font-size: 12px;
    }}

    /* Metric labels */
    .stMetric label {{
        color: {TEXT_SECONDARY};
    }}

    /* Cards – rendered via render_result_card, no extra CSS needed */
    </style>
    """
    st.markdown(google_fonts + css, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Reusable UI components
# ----------------------------------------------------------------------

def render_result_card(
    title: str,
    content_fn: callable,
    expanded: bool = True
) -> None:
    """Draw a styled card with a title and Streamlit content.

    Args:
        title: Heading displayed at the top of the card.
        content_fn: A zero-argument callable that renders the card's
            interior using Streamlit elements (e.g. ``st.image``,
            ``st.write``).
        expanded: If False, the card starts collapsed (currently unused,
            kept for future compatibility with expandable cards).
    """
    # Card shell
    card_html = f"""
    <div style="
        background: {BG_SECONDARY};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    ">
        <h3 style="color:{TEXT_PRIMARY}; font-size:16px; font-weight:500; margin:0 0 12px 0;">
            {title}
        </h3>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    if expanded:
        content_fn()


def render_status_indicator(label: str, status: str) -> None:
    """Show a coloured dot next to a label.

    Args:
        label: Text label displayed next to the dot.
        status: One of ``"success"``, ``"warning"``, ``"error"``,
            ``"info"``. Controls the dot colour.
    """
    color_map = {
        "success": SUCCESS,
        "warning": WARNING,
        "error": ERROR,
        "info": INFO,
    }
    if status not in color_map:
        raise ValueError(f"Unknown status '{status}'. Valid: {list(color_map.keys())}")

    dot_color = color_map[status]
    html = f"""
    <div style="display:flex; align-items:center; gap:8px;">
        <div style="width:8px; height:8px; border-radius:50%;
                    background:{dot_color};"></div>
        <span style="color:{TEXT_PRIMARY}; font-size:14px;">{label}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_ai_output_notice() -> None:
    """Display a legal notice about AI-generated content (EU AI Act readiness)."""
    html = f"""
    <div style="
        background: {INFO}4D;
        border-left: 3px solid {ACCENT_1};
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        font-size: 13px;
        color: {TEXT_SECONDARY};
        margin-top: 16px;
    ">
        These PBR maps are generated by an AI system (MatForge).
        Under European copyright law, AI-generated content is not
        protected by copyright. The maps may be used freely,
        including for commercial purposes.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def image_to_base64(arr: np.ndarray) -> str:
    """Encode a float32 numpy array as a base64 PNG string.

    Converts float32 [0, 1] to uint8 [0, 255] before encoding.
    Handles both (H, W) greyscale and (H, W, 3) RGB arrays.

    Args:
        arr: float32 numpy array in [0, 1].

    Returns:
        Base64-encoded PNG string (no data URI prefix).
    """
    arr_uint8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr_uint8.ndim == 2:
        pil = Image.fromarray(arr_uint8, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(arr_uint8, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def normal_map_to_display(normal: np.ndarray) -> np.ndarray:
    """Convert a normal map from [-1, 1] to display space [0, 1].

    Applies the standard (n + 1) / 2 remapping used for tangent-space
    normal map visualisation.

    Args:
        normal: (H, W, 3) float32 array with values in [-1, 1].

    Returns:
        (H, W, 3) float32 array with values in [0, 1].
    """
    return np.clip((normal + 1.0) / 2.0, 0.0, 1.0)


def render_map_grid(
    normal: np.ndarray,
    roughness: np.ndarray,
    metallic: np.ndarray,
) -> None:
    """Display predicted PBR maps with a tab selector.

    Args:
        normal: Normal map (H,W,3) float32 in [-1,1].
        roughness: Roughness map (H,W,1) float32 in [0,1].
        metallic: Metallic map (H,W,1) float32 in [0,1].
    """
    tab_all, tab_n, tab_r, tab_m = st.tabs(
        ["All maps", "Normal", "Roughness", "Metallic"]
    )

    normal_disp = np.clip((normal + 1.0) / 2.0, 0.0, 1.0)
    rough_disp  = roughness.squeeze()
    metal_disp  = metallic.squeeze()

    with tab_all:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(normal_disp, caption="Normal Map",
                     use_container_width=True)
        with col2:
            st.image(rough_disp, caption="Roughness",
                     use_container_width=True, clamp=True)
        with col3:
            st.image(metal_disp, caption="Metallic",
                     use_container_width=True, clamp=True)

    with tab_n:
        st.image(normal_disp, caption="Normal Map",
                 use_container_width=True)

    with tab_r:
        st.image(rough_disp, caption="Roughness",
                 use_container_width=True, clamp=True)

    with tab_m:
        st.image(metal_disp, caption="Metallic",
                 use_container_width=True, clamp=True)


def build_threejs_viewer(
    normal_b64: str,
    roughness_b64: str,
    metallic_b64: str,
    geometry: str = "sphere",
    height: int = 600,
    color_b64: str | None = None,
    show_env: bool = False,
) -> str:
    """Create an interactive 3D material preview using Three.js.

    The normal map is expected to be an OpenGL normal map (Y+ up),
    which is the default convention for Three.js MeshStandardMaterial.

    Args:
        normal_b64: Base64‑encoded PNG of the normal map (packed [0,1]).
        roughness_b64: Base64‑encoded PNG of the roughness map.
        metallic_b64: Base64‑encoded PNG of the metallic map.
        geometry: Preview shape – ``"sphere"``, ``"box"``, or ``"plane"``.
        height: Height of the viewer in pixels.

    Returns:
        Full HTML string to be embedded with
        ``st.components.v1.html``.
    """
    # Geometry choices
    geo_map = {
        "sphere": "new THREE.SphereGeometry(1, 64, 64)",
        "box": "new THREE.BoxGeometry(1.5, 1.5, 1.5)",
        "plane": "new THREE.PlaneGeometry(2, 2, 1, 1)",
    }
    if geometry not in geo_map:
        raise ValueError(f"Unknown geometry '{geometry}'. Choose from {list(geo_map.keys())}")
    geo_code = geo_map[geometry]

    # Build the JS snippet for albedo outside the f-string to avoid escaping issues.
    if color_b64:
        albedo_js = (
            f"const colorTex = new THREE.TextureLoader().load("
            f"'data:image/png;base64,{color_b64}');"
            f" colorTex.wrapS = THREE.RepeatWrapping;"
            f" colorTex.wrapT = THREE.RepeatWrapping;"
            f" colorTex.repeat.set(2, 2);"
            f" colorTex.colorSpace = THREE.SRGBColorSpace;"
            f" materialParams.map = colorTex;"
        )
    else:
        albedo_js = "materialParams.color = new THREE.Color(0.6, 0.6, 0.6);"

    if show_env:
        env_js = (
            "const pmremGenerator = new THREE.PMREMGenerator(renderer);"
            " scene.environment = pmremGenerator.fromScene("
            " new RoomEnvironment(), 0.04).texture;"
            " pmremGenerator.dispose();"
        )
    else:
        env_js = ""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ margin: 0; overflow: hidden; background: {BG_PRIMARY}; }}
            canvas {{ display: block; }}
        </style>
    </head>
    <body>
        <script type="importmap">
        {{
            "imports": {{
                "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
            }}
        }}
        </script>

        <script type="module">
            import * as THREE from 'three';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
            import {{ RoomEnvironment }} from 'three/addons/environments/RoomEnvironment.js';

            // Use container width rather than window.innerWidth to avoid
            // distortion caused by the Streamlit sidebar narrowing the iframe.
            const W = document.documentElement.clientWidth;
            const H = {height};

            const scene = new THREE.Scene();
            scene.background = new THREE.Color('{BG_PRIMARY}');

            const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 100);
            camera.position.set(0, 0, 3);

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.setSize(W, H);
            renderer.outputColorSpace = THREE.SRGBColorSpace;
            document.body.appendChild(renderer.domElement);

            // Two directional lights improve PBR readability without an envMap.
            // A fill light from the opposite side prevents fully black shadows.
            const ambient = new THREE.AmbientLight(0xffffff, 0.4);
            scene.add(ambient);
            const light1 = new THREE.DirectionalLight(0xffffff, 2.5);
            light1.position.set(3, 4, 5);
            scene.add(light1);
            const light2 = new THREE.DirectionalLight(0xffeedd, 0.8);
            light2.position.set(-4, -2, -3);
            scene.add(light2);

            {env_js}

            // Helper: load a base64 PNG texture with repeat wrapping for
            // tiling materials. colorSpace=null keeps linear space for
            // non-colour data (normal, roughness, metallic).
            function loadTex(b64, isColorData) {{
                const t = new THREE.TextureLoader().load(
                    'data:image/png;base64,' + b64
                );
                t.wrapS = THREE.RepeatWrapping;
                t.wrapT = THREE.RepeatWrapping;
                t.repeat.set(2, 2);
                // Non-colour data (normal, roughness, metallic) must use
                // linear space — SRGBColorSpace applies gamma that distorts
                // the values and produces incorrect PBR rendering.
                t.colorSpace = isColorData
                    ? THREE.SRGBColorSpace
                    : THREE.NoColorSpace;
                return t;
            }}

            const normalTex    = loadTex('{normal_b64}', false);
            const roughnessTex = loadTex('{roughness_b64}', false);
            const metallicTex  = loadTex('{metallic_b64}', false);

            // Neutral grey albedo makes normal/roughness/metallic detail
            // visible without an envMap. Pure white washes out normal detail.
            const materialParams = {{
                normalMap: normalTex,
                normalScale: new THREE.Vector2(1, 1),
                roughnessMap: roughnessTex,
                roughness: 1.0,
                metalnessMap: metallicTex,
                metalness: 1.0,
            }};
            {albedo_js}
            const material = new THREE.MeshStandardMaterial(materialParams);

            const geom = {geo_code};
            const mesh = new THREE.Mesh(geom, material);
            scene.add(mesh);

            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.autoRotate = true;
            controls.autoRotateSpeed = 0.5;

            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();

            window.addEventListener('resize', () => {{
                const nW = document.documentElement.clientWidth;
                camera.aspect = nW / H;
                camera.updateProjectionMatrix();
                renderer.setSize(nW, H);
            }});
        </script>
    </body>
    </html>
    """
    return html


def render_comparison_slider(
    before_b64: str,
    after_b64: str,
    height: int = 400,
) -> None:
    """Display a before/after comparison with a draggable slider.

    Args:
        before_b64: Base64‑encoded PNG of the original image.
        after_b64: Base64‑encoded PNG of the processed result.
        height: Height of the comparison widget in pixels.
    """
    html = f"""
    <div id="comp-wrap" style="position:relative; width:100%;
                height:{height}px; background:{BG_PRIMARY};
                overflow:hidden; cursor:ew-resize;">
        <img id="after-img" src="data:image/png;base64,{after_b64}"
             style="position:absolute; top:0; left:0; width:100%;
                    height:100%; object-fit:contain;">
        <img id="before-img" src="data:image/png;base64,{before_b64}"
             style="position:absolute; top:0; left:0; width:100%;
                    height:100%; object-fit:contain;
                    clip-path: inset(0 50% 0 0);">

        <!-- Divider line -->
        <div id="divider" style="position:absolute; top:0; left:50%;
                    width:2px; height:100%;
                    background:rgba(255,255,255,0.8);
                    transform:translateX(-50%);
                    pointer-events:none;">
            <!-- Handle -->
            <div style="position:absolute; top:50%; left:50%;
                        transform:translate(-50%,-50%);
                        width:32px; height:32px; border-radius:50%;
                        background:rgba(255,255,255,0.9);
                        display:flex; align-items:center;
                        justify-content:center;
                        font-size:14px; color:#1C1B18;
                        box-shadow:0 2px 8px rgba(0,0,0,0.4);">
                ↔
            </div>
        </div>

        <!-- Labels -->
        <div id="label-before" style="position:absolute; top:8px; left:8px;
                    background:rgba(0,0,0,0.55); color:#E8E6DF;
                    font-size:11px; padding:3px 8px; border-radius:4px;
                    pointer-events:none;">Before</div>
        <div id="label-after" style="position:absolute; top:8px; right:8px;
                    background:rgba(0,0,0,0.55); color:#E8E6DF;
                    font-size:11px; padding:3px 8px; border-radius:4px;
                    pointer-events:none;">After</div>

        <!-- Invisible range input for interaction -->
        <input type="range" id="slider" min="0" max="100" value="50"
               style="position:absolute; top:0; left:0; width:100%;
                      height:100%; opacity:0; cursor:ew-resize; margin:0;">
    </div>
    <script>
        const slider   = document.getElementById('slider');
        const before   = document.getElementById('before-img');
        const divider  = document.getElementById('divider');

        slider.addEventListener('input', (e) => {{
            const v = e.target.value;
            before.style.clipPath  = `inset(0 ${{100 - v}}% 0 0)`;
            divider.style.left     = v + '%';
        }});
    </script>
    """
    components.html(html, height=height)


# ----------------------------------------------------------------------
# Module verification
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("ui_components.py: module structure OK")
    print(f"  Design constants loaded: BG_PRIMARY={BG_PRIMARY}")
    print("  Functions: inject_css, render_result_card,")
    print("             render_status_indicator, render_ai_output_notice,")
    print("             render_map_grid, build_threejs_viewer,")
    print("             render_comparison_slider")