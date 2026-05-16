"""
Qualitative comparison panels — MatForge vs Materialize vs Substance 3D Sampler.

One figure per texture. Layout:
    Rows : GT | MatForge | Materialize | Substance
    Cols : Color | Normal | Roughness | Metallic | Render (if available)

Roughness note: Materialize exports Smoothness (= 1 - Roughness).
                Substance exports SpecularRoughness. Both are labelled
                accordingly in the panel headers.

Output: F:\\PI\\benchmark_results\\output_samples\\Paneles\\panel_<texture>.png
"""

# =============================================================================
# DEPENDENCIES — installed automatically on first run
# =============================================================================
import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '--quiet'])

try:
    import matplotlib
except ImportError:
    _install('matplotlib')

try:
    from PIL import Image
except ImportError:
    _install('Pillow')

# =============================================================================
# IMPORTS
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image

# =============================================================================
# PATHS
# =============================================================================
ROOT    = Path(r'F:\PI\benchmark_results\output_samples')
GT_ROOT = ROOT / 'GT'
MAT_ROOT  = ROOT / 'Materialize'
SUB_ROOT  = ROOT / '3D_Sampler'
MF_ROOT   = ROOT / 'MatForge'
BLD_ROOT  = ROOT / 'Blender'
OUT_DIR   = ROOT / 'Paneles'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TEXTURES AND THEIR GROUPS (for GT metallic path)
# =============================================================================
TEXTURES = {
    'ceramic_0494'  : 'ceramic_ground',
    'concrete_0180' : 'concrete_plaster',
    'metal_0175'    : 'metal',
    'stone_0201'    : 'stone_rough',
    'stone_0480'    : 'stone_rough',
    'terracotta_0166': 'brick_terracotta',
}

# Textures that have Blender renders
BLENDER_TEXTURES = {'ceramic_0494', 'metal_0175', 'terracotta_0166'}

# =============================================================================
# FILE NAME MAPS PER SOURCE
# =============================================================================
def gt_paths(stem, group):
    """Ground truth map paths."""
    metal_path = GT_ROOT / 'metal' / f'{stem}.png'
    return {
        'Color'    : GT_ROOT / 'rgb'      / f'{stem}.png',
        'Normal'   : GT_ROOT / 'normal'   / f'{stem}.png',
        'Roughness': GT_ROOT / 'roughness'/ f'{stem}.png',
        'Metallic' : metal_path if metal_path.exists() else None,
    }

def materialize_paths(stem):
    return {
        'Color'    : MAT_ROOT / stem / f'{stem}_diffuseOriginal.png',
        'Normal'   : MAT_ROOT / stem / f'{stem}_normal.png',
        'Roughness': MAT_ROOT / stem / f'{stem}_smoothness.png',
        'Metallic' : MAT_ROOT / stem / f'{stem}_metallic.png',
    }

def substance_paths(stem):
    return {
        'Color'    : SUB_ROOT / stem / f'{stem}_BaseColor.png',
        'Normal'   : SUB_ROOT / stem / f'{stem}_Normal.png',
        'Roughness': SUB_ROOT / stem / f'{stem}_SpecularRoughness.png',
        'Metallic' : SUB_ROOT / stem / f'{stem}_BaseMetalness.png',
    }

def matforge_paths(stem):
    folder = MF_ROOT / f'{stem}_blender'
    return {
        'Color'    : folder / f'{stem}_color.png',
        'Normal'   : folder / f'{stem}_normal.png',
        'Roughness': folder / f'{stem}_roughness.png',
        'Metallic' : folder / f'{stem}_metallic.png',
    }

def blender_paths(stem):
    folder = BLD_ROOT / stem
    return {
        'MatForge'   : folder / f'{stem}_matforge_render.png',
        'Materialize': folder / f'{stem}_materialize_render.png',
        'Substance'  : folder / f'{stem}_substance_render.png',
    }

# =============================================================================
# IMAGE LOADING
# =============================================================================
DISPLAY_SIZE = 256

def load_img(path):
    """Load image as float32 [0,1] RGB array.
    
    Square images are resized directly. Wide/tall images are centre-cropped
    to square first to preserve aspect ratio before resizing.
    """
    if path is None or not Path(path).exists():
        return None
    img = Image.open(path).convert('RGB')
    w, h = img.size
    # Centre crop to square if aspect ratio is not 1:1
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top  = (h - side) // 2
        img  = img.crop((left, top, left + side, top + side))
    img = img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def placeholder(text='N/A'):
    """Grey placeholder image with centred text label."""
    arr = np.ones((DISPLAY_SIZE, DISPLAY_SIZE, 3), dtype=np.float32) * 0.15
    return arr, text


# =============================================================================
# PANEL GENERATION
# =============================================================================
ROW_LABELS = ['GT', 'MatForge', 'Materialize', 'Substance']

# Column headers per source (roughness label differs)
COL_HEADERS_BASE = ['Color', 'Normal', 'Roughness*', 'Metallic']

def generate_panel(stem, group):
    has_render = stem in BLENDER_TEXTURES
    n_cols     = 5 if has_render else 4
    n_rows     = len(ROW_LABELS)

    # Gather all paths
    paths = {
        'GT'         : gt_paths(stem, group),
        'MatForge'   : matforge_paths(stem),
        'Materialize': materialize_paths(stem),
        'Substance'  : substance_paths(stem),
    }
    bld = blender_paths(stem) if has_render else {}

    # Column headers
    col_headers = COL_HEADERS_BASE.copy()
    if has_render:
        col_headers.append('Render (Blender)')

    fig_w = n_cols * 2.8
    fig_h = n_rows * 2.8 + 0.8   # extra for title

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='#1a1a1a')
    gs  = gridspec.GridSpec(
        n_rows + 1, n_cols,
        figure=fig,
        hspace=0.06, wspace=0.04,
        top=0.93, bottom=0.02, left=0.10, right=0.99,
        height_ratios=[0.18] + [1.0] * n_rows,
    )

    # --- Column headers row ---
    for col_idx, header in enumerate(col_headers):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_facecolor('#1a1a1a')
        ax.text(0.5, 0.5, header, ha='center', va='center',
                fontsize=9, fontweight='bold', color='#e0e0e0',
                transform=ax.transAxes)
        ax.axis('off')

    # --- Image rows ---
    for row_idx, row_label in enumerate(ROW_LABELS):
        source_paths = paths[row_label]

        for col_idx, map_key in enumerate(['Color', 'Normal', 'Roughness*', 'Metallic']):
            # Roughness* key maps to 'Roughness' in the path dicts
            lookup = 'Roughness' if map_key == 'Roughness*' else map_key
            path   = source_paths.get(lookup)
            img    = load_img(path)

            ax = fig.add_subplot(gs[row_idx + 1, col_idx])
            ax.set_facecolor('#1a1a1a')

            if img is not None:
                ax.imshow(img.clip(0, 1))
            else:
                ax.imshow(np.ones((DISPLAY_SIZE, DISPLAY_SIZE, 3)) * 0.15)
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=10, color='#666666', transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Row label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight='bold',
                              color='#e0e0e0', rotation=90,
                              labelpad=6, va='center')

        # --- Render column (col index 4) ---
        if has_render:
            col_idx = 4
            if row_label == 'GT':
                render_path = None
            else:
                render_path = bld.get(row_label)

            img = load_img(render_path)
            ax  = fig.add_subplot(gs[row_idx + 1, col_idx])
            ax.set_facecolor('#1a1a1a')

            if img is not None:
                ax.imshow(img.clip(0, 1))
            else:
                ax.imshow(np.ones((DISPLAY_SIZE, DISPLAY_SIZE, 3)) * 0.15)
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=10, color='#666666', transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Title
    fig.suptitle(
        stem,
        fontsize=13, fontweight='bold', color='#ffffff',
        y=0.97
    )

    # Footnote
    footnote = (
        '* Materialize exports Smoothness (= 1 − Roughness). '
        'Substance exports SpecularRoughness. GT and MatForge export standard Roughness.'
    )
    fig.text(0.01, 0.005, footnote, fontsize=6.5, color='#888888', va='bottom')

    save_path = OUT_DIR / f'panel_{stem}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {save_path}')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    for stem, group in TEXTURES.items():
        print(f'Processing {stem} ...')
        generate_panel(stem, group)
    print(f'\nAll panels saved to {OUT_DIR}')
