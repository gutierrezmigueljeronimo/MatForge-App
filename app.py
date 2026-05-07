import streamlit as st
from PIL import Image

from src.inference import load_model, run_inference, DEVICE

st.set_page_config(page_title="MatForge", layout="wide")

load_model()

st.title("MatForge — PBR Map Prediction")

uploaded = st.file_uploader("Upload an RGB image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    W, H = image.size
    st.image(image, caption="Input", use_container_width=False)

    try:
        with st.spinner("Running inference..."):
            maps = run_inference(image)

        col_n, col_r, col_m = st.columns(3)

        normal_display = (maps["normal"] + 1.0) / 2.0
        col_n.image(normal_display, caption="Normal", clamp=True)

        col_r.image(maps["roughness"].squeeze(-1), caption="Roughness", clamp=True)
        col_m.image(maps["metallic"].squeeze(-1),  caption="Metallic",  clamp=True)

        st.caption(f"Resolution: {W}×{H} | Device: {DEVICE}")

    except Exception as exc:
        st.error(str(exc))
