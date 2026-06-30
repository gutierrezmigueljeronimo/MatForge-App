# Material classifier — DINOv2 feature extraction
# DINOv2 © 2023 Meta AI, Apache License 2.0
# https://github.com/facebookresearch/dinov2

"""
Material group classifier using DINOv2-small + PCA-50 + KNN.

Loads three pre-computed artifacts from ``artifacts/`` and exposes a
single-function API that takes a PIL RGB image and returns a
(group name, confidence distance) tuple.
"""
import gc
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import timm
import torch
from PIL import Image

from src.utils import pil_to_numpy, numpy_to_tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path("artifacts")
PCA_PATH      = ARTIFACTS_DIR / "pca_model.pkl"
KNN_PATH      = ARTIFACTS_DIR / "knn_classifier.pkl"
ENCODER_PATH  = ARTIFACTS_DIR / "label_encoder.pkl"

MODEL_NAME    = "vit_small_patch14_dinov2.lvd142m"
IMG_SIZE      = 518                    # DINOv2-small native resolution
EMBED_DIM     = 384                    # [CLS] token dimension

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Lazy-loader helpers (streamlit-safe)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_dinov2():
    """
    Load the frozen DINOv2-small encoder.

    Returns:
        timm model in eval mode on ``DEVICE`` with classification head removed.
    """
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0,          # output [CLS] token instead of logits
    )
    model.eval()
    model.to(DEVICE)
    return model


@st.cache_resource
def _load_artifacts():
    """
    Load the three serialised sklearn objects from disk.

    Returns:
        Tuple of (pca, knn, label_encoder).
    """
    pca           = joblib.load(PCA_PATH)
    knn           = joblib.load(KNN_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return pca, knn, label_encoder

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_material(image: Image.Image) -> tuple[str, float]:
    """
    Classify a material image into one of eight functional groups.

    Args:
        image: PIL RGB image at any resolution (resized internally to 518×518).

    Returns:
        (group_name, knn_distance) where *group_name* is one of
        ``['brick_terracotta', 'ceramic_ground', 'concrete_plaster',
        'marble_smooth', 'metal', 'mixed_ambiguous', 'stone_rough',
        'wood']`` and *knn_distance* is the cosine distance to the
        closest training sample in the reduced 50-d PCA space.
        Lower distance = higher confidence.
    """
    model      = _load_dinov2()
    pca, knn, le = _load_artifacts()

    # ---- 1. Preprocess input -------------------------------------------------
    img_resized = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = pil_to_numpy(img_resized)                         # (518,518,3) float32 [0,1]

    # ImageNet normalisation (same as used during relabeling training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    tensor = numpy_to_tensor(arr, device=DEVICE, dtype=torch.float32)  # (1,3,224,224)

    # ---- 2. Extract [CLS] embedding (384-d) ----------------------------------
    with torch.no_grad():
        emb = model(tensor)                    # (1, 384)
    emb_np = emb.cpu().numpy().astype(np.float32)

    # ---- 3. PCA-50 → KNN predict ---------------------------------------------
    emb_pca = pca.transform(emb_np)            # (1, 50)
    pred    = knn.predict(emb_pca)[0]          # int label
    dist    = knn.kneighbors(emb_pca, n_neighbors=1)[0][0, 0]  # cosine distance

    group_name = le.inverse_transform([pred])[0]

    return group_name, float(dist)
