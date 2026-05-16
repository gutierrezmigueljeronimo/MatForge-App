#!/usr/bin/env python3
"""
MatForge Relabeling Pipeline v1.0
===================================
Semantic relabeling of PBR texture dataset using DINOv2 visual embeddings,
dimensionality reduction (PCA + UMAP), and density-based clustering (HDBSCAN).

Produces:
  - relabeling_final.csv       : per-texture functional group assignment
  - sampler_weights.json       : per-group sampling weights for the DataLoader
  - knn_classifier.pkl         : serialized KNN classifier for Streamlit inference
  - pca_model.pkl              : serialized PCA model for inference pipeline
  - relabeling_output/         : UMAP visualizations, cluster summary, metrics

EXECUTION MODES:
  "cluster"  : run full pipeline (embeddings → PCA → UMAP → HDBSCAN → report)
  "validate" : load existing clusters, show metrics and panel without recomputing
  "export"   : train KNN classifier and export inference artifacts

Run in sequence: cluster → (manual review of CSV) → export

DEPENDENCIES:
  pip install torch torchvision timm umap-learn hdbscan scikit-learn
              numpy pandas matplotlib seaborn Pillow tqdm joblib

HARDWARE: CPU only. GPU not required.
Estimated runtime on SSD (3245 textures): ~20-35 min for "cluster" mode.
"""

import os
import json
import warnings
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import timm

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score
import joblib

import umap
import hdbscan

warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

MATFORGE_DIR   = r"D:\PI\matforge_dataset"       # dataset root (contains maps/)
OUTPUT_DIR     = r"D:\PI\relabeling_output"      # all outputs land here
MODO           = "export"                       # "cluster" | "validate" | "export"
RANDOM_STATE   = 42
BATCH_SIZE     = 32                              # images per forward pass (CPU)
IMG_SIZE       = 518                             # DINOv2 input resolution (lvd142m variant requires 518)

# ── PCA ────────────────────────────────────────────────────────────────
PCA_COMPONENTS = 50

# ── UMAP ───────────────────────────────────────────────────────────────
UMAP_N_COMPONENTS   = 15     # clustering space
UMAP_VIZ_COMPONENTS = 2      # visualization space (independent run)
UMAP_N_NEIGHBORS    = 30
UMAP_MIN_DIST_CLUST = 0.0    # tight packing for clustering
UMAP_MIN_DIST_VIZ   = 0.1    # slightly looser for visualization

# ── HDBSCAN ────────────────────────────────────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES      = 5
HDBSCAN_METRIC           = "euclidean"   # applied on UMAP-reduced space
HDBSCAN_SELECTION_METHOD = "eom"

# ── KNN classifier (inference) ─────────────────────────────────────────
KNN_N_NEIGHBORS = 7
KNN_METRIC      = "cosine"
KNN_WEIGHTS     = "distance"

# ── Functional group names (post-relabeling labels) ─────────────────────
# These are assigned during manual review of the UMAP panel.
# The script outputs numeric cluster IDs; this dict maps them to names
# AFTER you inspect the panel. Pre-fill your best guesses here;
# update after inspection and re-run in "export" mode.
GROUP_NAMES = {
    -1: "mixed_ambiguous",

    # ── wood ─────────────────────────────────────────────────────────
    4:  "wood",   # wood 84.0% — largest and purest wood cluster
    6:  "wood",   # wood 48.5% — mixed wood subtype
    8:  "wood",   # wood 29.4% — visually wood-adjacent
    26: "wood",   # wood 61.5%

    # ── stone_rough ───────────────────────────────────────────────────
    2:  "stone_rough",   # stone 70.0%
    12: "stone_rough",   # stone 39.6% — mixed but stone-dominant zone
    13: "stone_rough",   # stone 66.0%
    17: "stone_rough",   # stone 91.8% — very pure
    18: "stone_rough",   # stone 77.6%
    19: "stone_rough",   # stone 95.1% — purest stone cluster
    23: "stone_rough",   # stone 43.3%
    29: "stone_rough",   # stone 36.8%

    # ── brick_terracotta ──────────────────────────────────────────────
    20: "brick_terracotta",   # terracotta 77.7%
    21: "brick_terracotta",   # terracotta 56.8%

    # ── metal ─────────────────────────────────────────────────────────
    3:  "metal",   # metal 75.4%
    25: "metal",   # metal 81.2%
    32: "metal",   # metal 51.1%
    35: "metal",   # metal 37.0% — lowest purity but kept for metallic signal

    # ── marble_smooth ─────────────────────────────────────────────────
    10: "marble_smooth",   # marble 48.1%
    28: "marble_smooth",   # marble 31.8% — low purity; force_assign_marble
                           # will complement these with remaining marble textures

    # ── ceramic_ground ────────────────────────────────────────────────
    7:  "ceramic_ground",   # ceramic 58.1%
    11: "ceramic_ground",   # ceramic 65.1%
    15: "ceramic_ground",   # ceramic 73.2%
    16: "ceramic_ground",   # ceramic 47.1%
    22: "ceramic_ground",   # ceramic 32.9% — borderline; visual review advised
    27: "ceramic_ground",   # ground  70.8%
    30: "ceramic_ground",   # ground  97.6% — purest ground cluster
    31: "ceramic_ground",   # ground  82.1%
    33: "ceramic_ground",   # ceramic 59.3%
    34: "ceramic_ground",   # ceramic 82.6%

    # ── concrete_plaster ──────────────────────────────────────────────
    9:  "concrete_plaster",   # concrete 33.3%
    14: "concrete_plaster",   # plaster  42.6%
    24: "concrete_plaster",   # plaster  46.3%

    # ── mixed_ambiguous ───────────────────────────────────────────────
    0:  "mixed_ambiguous",   # terracotta 40.0% — no clear dominant
    1:  "mixed_ambiguous",   # ceramic    40.6% — mixed
    5:  "mixed_ambiguous",   # ceramic    47.7% — borderline with ceramic_ground
    36: "mixed_ambiguous",   # plaster    21.5% — very impure, 93 textures
}

# Sampling weights per functional group for the DataLoader.
# Groups not listed here get weight 1.0. mixed_ambiguous gets 0.5.
GROUP_WEIGHTS = {
    "mixed_ambiguous": 0.5,
    "marble_smooth":   1.2,   # upweight smallest domain-principal group
    "metal":           1.3,   # upweight: metallic signal is scarce
}

# ╚══════════════════════════════════════════════════════════════════════╝

MAPS_DIR    = os.path.join(MATFORGE_DIR, "maps", "rgb")
EMBEDS_PATH = os.path.join(OUTPUT_DIR,   "embeddings.npy")
NAMES_PATH  = os.path.join(OUTPUT_DIR,   "filenames.npy")
CLUSTERS_PATH   = os.path.join(OUTPUT_DIR, "cluster_labels.npy")
PCA_PATH        = os.path.join(OUTPUT_DIR, "pca_model.pkl")
KNN_PATH        = os.path.join(OUTPUT_DIR, "knn_classifier.pkl")
CSV_FINAL       = os.path.join(OUTPUT_DIR, "relabeling_final.csv")
WEIGHTS_JSON    = os.path.join(OUTPUT_DIR, "sampler_weights.json")
METRICS_JSON    = os.path.join(OUTPUT_DIR, "cluster_metrics.json")

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#264653", "#A8DADC", "#6D6875", "#B5838D", "#FFAFCC",
    "#CDB4DB", "#BDE0FE", "#CAFFBF", "#FDFFB6", "#FFD6FF",
    "#606C38", "#DDA15E", "#BC6C25", "#8338EC", "#3A86FF",
]


# ══════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def setup_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_category(filename: str) -> str:
    """Derive original MatSynth category from filename prefix."""
    return filename.rsplit("_", 1)[0]


def load_image(path: str) -> torch.Tensor:
    """
    Load a single RGB image, resize to IMG_SIZE, and apply
    ImageNet normalization. Returns a [3, H, W] float tensor.
    """
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1))


# ══════════════════════════════════════════════════════════════════════
# STAGE 1 — DINOV2 EMBEDDING EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def extract_embeddings(filenames: list[str]) -> np.ndarray:
    """
    Extract DINOv2-small [CLS] token embeddings for all textures.

    The encoder is fully frozen (eval mode, no_grad). Each image is
    resized to 224×224 and normalized with ImageNet statistics before
    the forward pass. Batching is used to amortize overhead on CPU.

    Returns an array of shape (N, 384).
    """
    print("\n" + "="*60)
    print("  STAGE 1 — DINOv2 embedding extraction")
    print("="*60)

    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,       # remove classification head → outputs [CLS] token
    )
    model.eval()

    n     = len(filenames)
    embeds = np.zeros((n, 384), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, n, BATCH_SIZE), desc="Extracting", unit="batch"):
            batch_files = filenames[start : start + BATCH_SIZE]
            tensors = []
            for fname in batch_files:
                path = os.path.join(MAPS_DIR, fname)
                try:
                    tensors.append(load_image(path))
                except Exception:
                    tensors.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

            batch = torch.stack(tensors)          # [B, 3, 224, 224]
            out   = model(batch)                  # [B, 384]
            embeds[start : start + len(batch_files)] = out.numpy()

    del model
    gc.collect()

    np.save(EMBEDS_PATH, embeds)
    print(f"  Embeddings saved → {EMBEDS_PATH}  shape: {embeds.shape}")
    return embeds


# ══════════════════════════════════════════════════════════════════════
# STAGE 2 — DIMENSIONALITY REDUCTION (PCA + UMAP)
# ══════════════════════════════════════════════════════════════════════

def reduce_dimensions(embeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-step dimensionality reduction pipeline:
      1. PCA 384D → 50D  (linear, fast, preserves global variance)
      2. UMAP 50D → 15D  (non-linear, preserves local manifold structure)

    A second independent UMAP 50D → 2D is computed for visualization only.
    Both UMAP runs share the same PCA projection but are kept separate so
    that the clustering space is not constrained by 2D visualization needs.

    Returns (umap_clust, umap_viz) both as numpy arrays.
    """
    print("\n" + "="*60)
    print("  STAGE 2 — Dimensionality reduction (PCA + UMAP)")
    print("="*60)

    # ── PCA ──────────────────────────────────────────────────────────
    print(f"  PCA: {embeds.shape[1]}D → {PCA_COMPONENTS}D ...", end=" ", flush=True)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    embeds_pca = pca.fit_transform(embeds)
    explained  = pca.explained_variance_ratio_.sum()
    print(f"done. Explained variance: {explained:.1%}")
    joblib.dump(pca, PCA_PATH)

    # ── UMAP clustering space (15D) ───────────────────────────────────
    print(f"  UMAP clustering: {PCA_COMPONENTS}D → {UMAP_N_COMPONENTS}D ...")
    reducer_clust = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST_CLUST,
        metric="cosine",
        random_state=RANDOM_STATE,
        verbose=False,
    )
    umap_clust = reducer_clust.fit_transform(embeds_pca)
    np.save(os.path.join(OUTPUT_DIR, "umap_clust.npy"), umap_clust)

    # ── UMAP visualization space (2D) ─────────────────────────────────
    print(f"  UMAP visualization: {PCA_COMPONENTS}D → {UMAP_VIZ_COMPONENTS}D ...")
    reducer_viz = umap.UMAP(
        n_components=UMAP_VIZ_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST_VIZ,
        metric="cosine",
        random_state=RANDOM_STATE,
        verbose=False,
    )
    umap_viz = reducer_viz.fit_transform(embeds_pca)
    np.save(os.path.join(OUTPUT_DIR, "umap_viz.npy"), umap_viz)

    print("  Reduction complete.")
    return umap_clust, umap_viz


# ══════════════════════════════════════════════════════════════════════
# STAGE 3 — HDBSCAN CLUSTERING
# ══════════════════════════════════════════════════════════════════════

def run_hdbscan(umap_clust: np.ndarray) -> np.ndarray:
    """
    Density-based clustering on the 15D UMAP-reduced embedding space.

    prediction_data=True enables soft cluster membership scores and
    allows classifying new points without re-fitting the clusterer.

    Returns an integer array of cluster labels (-1 = noise/ambiguous).
    """
    print("\n" + "="*60)
    print("  STAGE 3 — HDBSCAN clustering")
    print("="*60)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_SELECTION_METHOD,
        prediction_data=True,
        gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(umap_clust)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    pct_noise  = n_noise / len(labels)

    # DBCV score via the internal attribute name used in this hdbscan version.
    try:
        dbcv = clusterer.relative_validity_
    except AttributeError:
        dbcv = getattr(clusterer, "_relative_validity", float("nan"))

    print(f"  Clusters found    : {n_clusters}")
    print(f"  Noise points      : {n_noise} ({pct_noise:.1%})")
    print(f"  DBCV score        : {dbcv:.4f}  (higher is better, range [-1, 1])")

    np.save(CLUSTERS_PATH, labels)
    joblib.dump(clusterer, os.path.join(OUTPUT_DIR, "hdbscan_model.pkl"))

    # ── Contingency checks ────────────────────────────────────────────
    if pct_noise > 0.20:
        print("\n  ⚠️  WARNING: noise > 20%. Consider reducing min_cluster_size to 20.")
    if n_clusters > 15:
        print(f"\n  ⚠️  WARNING: {n_clusters} clusters found (expected 7-10).")
        print("     Consider increasing min_cluster_size to 40.")
    if n_clusters < 5:
        print(f"\n  ⚠️  WARNING: only {n_clusters} clusters found.")
        print("     Consider reducing min_cluster_size to 20.")

    return labels


# ══════════════════════════════════════════════════════════════════════
# STAGE 4 — METRICS AND CLUSTER REPORT
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(
    filenames: list[str],
    labels: np.ndarray,
    umap_clust: np.ndarray,
) -> dict:
    """
    Compute clustering quality metrics and produce a per-cluster summary.

    Metrics computed:
      - DBCV   : density-based validity (primary metric)
      - % noise: fraction of points assigned to cluster -1
      - NMI    : normalized mutual information between cluster labels
                 and original MatSynth category tags. A value of ~0.5-0.7
                 indicates discovery of structure beyond the original tags.
      - Per-cluster breakdown: size, dominant original category, composition.
    """
    categories = [extract_category(f) for f in filenames]
    le = LabelEncoder()
    cat_encoded = le.fit_transform(categories)

    valid_mask = labels >= 0
    nmi = normalized_mutual_info_score(
        cat_encoded[valid_mask], labels[valid_mask]
    ) if valid_mask.sum() > 0 else 0.0

    _clusterer_loaded = joblib.load(os.path.join(OUTPUT_DIR, "hdbscan_model.pkl"))
    try:
        dbcv = float(_clusterer_loaded.relative_validity_)
    except AttributeError:
        dbcv = float(getattr(_clusterer_loaded, "_relative_validity", float("nan")))
    n_noise   = int((labels == -1).sum())
    pct_noise = n_noise / len(labels)
    n_clust   = len(set(labels)) - (1 if -1 in labels else 0)

    # Per-cluster composition
    df = pd.DataFrame({
        "filename": filenames,
        "category": categories,
        "cluster":  labels,
    })

    cluster_summary = []
    for cid in sorted(df["cluster"].unique()):
        sub   = df[df["cluster"] == cid]
        top   = sub["category"].value_counts()
        entry = {
            "cluster_id":        int(cid),
            "size":              int(len(sub)),
            "dominant_category": str(top.index[0]) if len(top) else "—",
            "dominant_pct":      round(float(top.iloc[0]) / len(sub) * 100, 1) if len(top) else 0,
            "composition":       top.to_dict(),
        }
        cluster_summary.append(entry)

    metrics = {
        "n_clusters":  n_clust,
        "n_noise":     n_noise,
        "pct_noise":   round(pct_noise * 100, 2),
        "dbcv":        round(dbcv, 4),
        "nmi_vs_original_tags": round(nmi, 4),
        "clusters":    cluster_summary,
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*60)
    print("  STAGE 4 — Cluster metrics")
    print("="*60)
    print(f"  DBCV score        : {dbcv:.4f}")
    print(f"  NMI vs orig. tags : {nmi:.4f}")
    print(f"  Noise             : {n_noise} ({pct_noise:.1%})")
    print(f"\n  {'Cluster':>8}  {'Size':>6}  {'Dominant category':>22}  {'Purity':>7}")
    print("  " + "-"*52)
    for s in cluster_summary:
        label = "noise" if s["cluster_id"] == -1 else str(s["cluster_id"])
        print(f"  {label:>8}  {s['size']:>6}  {s['dominant_category']:>22}  {s['dominant_pct']:>6.1f}%")

    return metrics


# ══════════════════════════════════════════════════════════════════════
# STAGE 5 — UMAP VISUALIZATION PANELS
# ══════════════════════════════════════════════════════════════════════

def generate_panels(
    filenames: list[str],
    labels: np.ndarray,
    umap_viz: np.ndarray,
) -> None:
    """
    Generate two UMAP scatter plots saved as high-resolution PNG files:

    Panel A — colored by HDBSCAN cluster ID.
               Use this to assess cluster separation and identify
               which numeric cluster corresponds to which material group.

    Panel B — colored by original MatSynth category tag.
               Use this to see how well original tags align with clusters
               and to identify which groups mixed or split.

    Both panels use the same 2D UMAP projection (independent from clustering).
    """
    print("\n" + "="*60)
    print("  STAGE 5 — UMAP visualization panels")
    print("="*60)

    categories = np.array([extract_category(f) for f in filenames])
    unique_cats = sorted(set(categories))

    x, y = umap_viz[:, 0], umap_viz[:, 1]

    # ── Panel A: clusters ─────────────────────────────────────────────
    unique_labels = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    for i, cid in enumerate(unique_labels):
        mask  = labels == cid
        color = "#555555" if cid == -1 else PALETTE[i % len(PALETTE)]
        name  = "noise / ambiguous" if cid == -1 else f"Cluster {cid}"
        ax.scatter(
            x[mask], y[mask],
            c=color, s=8, alpha=0.6 if cid != -1 else 0.25,
            linewidths=0, label=f"{name} (n={mask.sum()})",
        )

    ax.set_title("MatForge Relabeling — HDBSCAN Clusters (UMAP 2D)",
                 color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    ax.set_xlabel("UMAP dim 1", color="white")
    ax.set_ylabel("UMAP dim 2", color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    legend = ax.legend(
        loc="upper right", fontsize=7, framealpha=0.3,
        labelcolor="white", facecolor="#1a1a2e",
    )
    plt.tight_layout()
    path_a = os.path.join(OUTPUT_DIR, "panel_A_clusters.png")
    plt.savefig(path_a, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Panel A saved → {path_a}")

    # ── Panel B: original categories ──────────────────────────────────
    cat_palette = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(unique_cats)}

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    for cat in unique_cats:
        mask  = categories == cat
        ax.scatter(
            x[mask], y[mask],
            c=cat_palette[cat], s=8, alpha=0.6,
            linewidths=0, label=f"{cat} (n={mask.sum()})",
        )

    ax.set_title("MatForge Relabeling — Original MatSynth Categories (UMAP 2D)",
                 color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    ax.set_xlabel("UMAP dim 1", color="white")
    ax.set_ylabel("UMAP dim 2", color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(
        loc="upper right", fontsize=7, framealpha=0.3,
        labelcolor="white", facecolor="#1a1a2e",
    )
    plt.tight_layout()
    path_b = os.path.join(OUTPUT_DIR, "panel_B_original_cats.png")
    plt.savefig(path_b, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Panel B saved → {path_b}")

    # ── Panel C: cluster composition bar chart ────────────────────────
    df = pd.DataFrame({"cluster": labels, "category": categories})
    df_valid = df[df["cluster"] >= 0].copy()
    ct = pd.crosstab(df_valid["cluster"], df_valid["category"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(10, len(ct_norm) * 1.2), 6))
    ct_norm.plot(kind="bar", stacked=True, ax=ax,
                 color=[cat_palette.get(c, "#AAAAAA") for c in ct_norm.columns],
                 edgecolor="none", width=0.8)
    ax.set_title("Cluster Composition by Original Category", fontsize=13)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Fraction")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    path_c = os.path.join(OUTPUT_DIR, "panel_C_composition.png")
    plt.savefig(path_c, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Panel C saved → {path_c}")

    # ── Panel D: functional groups (post GROUP_NAMES assignment) ─────
    functional_labels = np.array([
        GROUP_NAMES.get(int(lbl), f"cluster_{lbl}") for lbl in labels
    ])
    unique_groups = sorted(set(functional_labels))
    group_palette = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    for group in unique_groups:
        mask  = functional_labels == group
        color = group_palette[group]
        alpha = 0.25 if group == "mixed_ambiguous" else 0.65
        ax.scatter(
            x[mask], y[mask],
            c=color, s=10, alpha=alpha,
            linewidths=0, label=f"{group} (n={mask.sum()})",
        )

    ax.set_title("MatForge Relabeling — Functional Groups (UMAP 2D)",
                 color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    ax.set_xlabel("UMAP dim 1", color="white")
    ax.set_ylabel("UMAP dim 2", color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(
        loc="upper right", fontsize=8, framealpha=0.3,
        labelcolor="white", facecolor="#1a1a2e",
    )
    plt.tight_layout()
    path_d = os.path.join(OUTPUT_DIR, "panel_D_functional_groups.png")
    plt.savefig(path_d, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Panel D saved → {path_d}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 6 — CSV EXPORT AND MANUAL REVIEW PREPARATION
# ══════════════════════════════════════════════════════════════════════

def export_csv(filenames: list[str], labels: np.ndarray) -> None:
    """
    Export relabeling_final.csv with one row per texture.

    Columns:
      filename          : image filename (e.g. stone_0042.png)
      original_category : MatSynth tag extracted from filename
      cluster_id        : HDBSCAN cluster ID (-1 = noise)
      functional_group  : human-readable group name (from GROUP_NAMES dict)
                          Fill GROUP_NAMES in the config section after inspecting
                          the UMAP panels, then re-run in "export" mode.
      needs_review      : True for noise points and ambiguous assignments

    The CSV is the primary artifact for manual review. Open it alongside
    the UMAP panels to assign names to numeric cluster IDs.
    """
    categories = [extract_category(f) for f in filenames]
    functional = [GROUP_NAMES.get(int(lbl), f"cluster_{lbl}") for lbl in labels]
    needs_rev  = [lbl == -1 or GROUP_NAMES.get(int(lbl), "").startswith("cluster_")
                  for lbl in labels]

    df = pd.DataFrame({
        "filename":          filenames,
        "original_category": categories,
        "cluster_id":        labels.astype(int),
        "functional_group":  functional,
        "needs_review":      needs_rev,
    })
    df.to_csv(CSV_FINAL, index=False, encoding="utf-8")
    print(f"\n  CSV exported → {CSV_FINAL}")
    print(f"  Rows needing review: {sum(needs_rev)}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 7 — KNN CLASSIFIER AND INFERENCE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════

def train_and_export_classifier(
    embeds: np.ndarray,
    filenames: list[str],
) -> None:
    """
    Train a KNN classifier on DINOv2 embeddings reduced to 50D (PCA only,
    not UMAP) so that inference on new images requires only PCA + KNN —
    a deterministic, reproducible pipeline with <5ms latency per query.

    Labels are read from relabeling_final.csv (after manual review).
    Textures with needs_review=True are excluded from training.

    Artifacts saved:
      pca_model.pkl        : PCA fitted on training embeddings
      knn_classifier.pkl   : fitted KNN classifier
      sampler_weights.json : per-texture weight for WeightedRandomSampler
      label_encoder.pkl    : maps string group names ↔ integers
    """
    print("\n" + "="*60)
    print("  STAGE 7 — KNN classifier training and artifact export")
    print("="*60)

    if not os.path.exists(CSV_FINAL):
        print(f"  ❌ {CSV_FINAL} not found. Run 'cluster' mode first.")
        return

    df = pd.read_csv(CSV_FINAL, encoding="utf-8")
    df_clean = df[~df["needs_review"].astype(bool)].copy()

    if len(df_clean) < 50:
        print("  ❌ Too few reviewed textures to train classifier.")
        return

    # Align embeddings with the CSV (same order guaranteed by filenames.npy)
    fname_to_idx = {f: i for i, f in enumerate(filenames)}
    idxs = [fname_to_idx[f] for f in df_clean["filename"] if f in fname_to_idx]
    X    = embeds[idxs]
    y    = df_clean["functional_group"].values[:len(idxs)]

    # ── PCA (re-fit on clean subset or load existing) ──────────────────
    pca = joblib.load(PCA_PATH) if os.path.exists(PCA_PATH) else PCA(
        n_components=PCA_COMPONENTS, random_state=RANDOM_STATE
    )
    X_pca = pca.transform(X) if hasattr(pca, "components_") else pca.fit_transform(X)
    joblib.dump(pca, PCA_PATH)

    # ── KNN ───────────────────────────────────────────────────────────
    le  = LabelEncoder()
    y_enc = le.fit_transform(y)
    knn = KNeighborsClassifier(
        n_neighbors=KNN_N_NEIGHBORS,
        metric=KNN_METRIC,
        weights=KNN_WEIGHTS,
        algorithm="brute",
        n_jobs=-1,
    )
    knn.fit(X_pca, y_enc)

    joblib.dump(knn, KNN_PATH)
    joblib.dump(le,  os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
    print(f"  KNN trained on {len(X_pca)} textures, {len(le.classes_)} groups.")
    print(f"  Groups: {list(le.classes_)}")
    print(f"  KNN saved → {KNN_PATH}")

    # ── Sampler weights ───────────────────────────────────────────────
    weights_out = {}
    for _, row in df.iterrows():
        group  = row["functional_group"]
        weight = GROUP_WEIGHTS.get(group, 1.0)
        weights_out[row["filename"]] = weight

    with open(WEIGHTS_JSON, "w", encoding="utf-8") as f:
        json.dump(weights_out, f, indent=2, ensure_ascii=False)
    print(f"  Sampler weights → {WEIGHTS_JSON}")

    # Summary
    group_counts = pd.Series(list(weights_out.values())).value_counts().sort_index()
    print("\n  Sampler weight distribution:")
    for w, c in group_counts.items():
        print(f"    weight {w:.1f} → {c} textures")


# ══════════════════════════════════════════════════════════════════════
# CONTINGENCY: MARBLE FORCE-ASSIGN
# ══════════════════════════════════════════════════════════════════════

def force_assign_marble(csv_path: str) -> None:
    """
    Contingency measure: if marble textures were absorbed into noise or
    a larger cluster, force-assign them to a dedicated 'marble_smooth' group.

    Call this after inspecting the UMAP panels if marble is not a distinct
    cluster. It modifies relabeling_final.csv in place.
    """
    if not os.path.exists(csv_path):
        print("CSV not found. Run cluster mode first.")
        return

    df = pd.read_csv(csv_path, encoding="utf-8")
    mask = df["original_category"] == "marble"
    n_changed = (df.loc[mask, "functional_group"] != "marble_smooth").sum()
    df.loc[mask, "functional_group"] = "marble_smooth"
    df.loc[mask, "needs_review"]     = False
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  force_assign_marble: {n_changed} textures reassigned to 'marble_smooth'.")


# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════

def print_final_summary(filenames: list[str], labels: np.ndarray) -> None:
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"\n  Total textures processed : {len(filenames)}")
    n_clust = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Functional groups found  : {n_clust}")
    print(f"  Noise / ambiguous        : {(labels == -1).sum()}")
    print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("  Files generated:")
    for fname in [
        "embeddings.npy", "cluster_labels.npy",
        "pca_model.pkl", "hdbscan_model.pkl",
        "umap_clust.npy", "umap_viz.npy",
        "relabeling_final.csv", "cluster_metrics.json",
        "panel_A_clusters.png", "panel_B_original_cats.png",
        "panel_C_composition.png",
    ]:
        full = os.path.join(OUTPUT_DIR, fname)
        status = "✅" if os.path.exists(full) else "⚠️  missing"
        print(f"    {status}  {fname}")

    print("\n  NEXT STEPS:")
    print("  1. Open panel_A_clusters.png and panel_B_original_cats.png.")
    print("  2. Fill GROUP_NAMES dict in configuration section.")
    print("  3. If marble is absorbed → call force_assign_marble().")
    print("  4. Update needs_review column in relabeling_final.csv.")
    print("  5. Set MODO = 'export' and re-run to train the KNN classifier.")
    print("="*60)


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    setup_dirs()

    if MODO == "cluster":
        # Collect all RGB filenames
        filenames = sorted([
            f for f in os.listdir(MAPS_DIR) if f.endswith(".png")
        ])
        if not filenames:
            print(f"❌ No PNG files found in {MAPS_DIR}")
            return

        np.save(NAMES_PATH, np.array(filenames))
        print(f"  Dataset: {len(filenames)} textures found.")

        # Stage 1: embeddings
        if os.path.exists(EMBEDS_PATH):
            print(f"\n  Embeddings cache found → loading {EMBEDS_PATH}")
            embeds = np.load(EMBEDS_PATH)
        else:
            embeds = extract_embeddings(filenames)

        # Stage 2: reduction
        umap_clust, umap_viz = reduce_dimensions(embeds)

        # Stage 3: clustering
        labels = run_hdbscan(umap_clust)

        # Stage 4: metrics
        compute_metrics(filenames, labels, umap_clust)

        # Stage 5: panels
        generate_panels(filenames, labels, umap_viz)

        # Stage 6: CSV
        export_csv(filenames, labels)

        print_final_summary(filenames, labels)

    elif MODO == "validate":
        if not all(os.path.exists(p) for p in [EMBEDS_PATH, NAMES_PATH, CLUSTERS_PATH]):
            print("❌ Missing cache files. Run 'cluster' mode first.")
            return
        filenames  = list(np.load(NAMES_PATH, allow_pickle=True))
        labels     = np.load(CLUSTERS_PATH)
        umap_viz   = np.load(os.path.join(OUTPUT_DIR, "umap_viz.npy"))
        umap_clust = np.load(os.path.join(OUTPUT_DIR, "umap_clust.npy"))

        compute_metrics(filenames, labels, umap_clust)
        generate_panels(filenames, labels, umap_viz)
        export_csv(filenames, labels)
        print_final_summary(filenames, labels)

    elif MODO == "export":
        if not all(os.path.exists(p) for p in [EMBEDS_PATH, NAMES_PATH, CSV_FINAL]):
            print("❌ Missing files. Run 'cluster' mode and review CSV first.")
            return
        filenames = list(np.load(NAMES_PATH, allow_pickle=True))
        embeds    = np.load(EMBEDS_PATH)
        train_and_export_classifier(embeds, filenames)

    else:
        print(f"❌ Unknown MODO: '{MODO}'. Use 'cluster', 'validate', or 'export'.")


if __name__ == "__main__":
    main()
