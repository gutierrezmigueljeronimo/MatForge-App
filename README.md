# MatForge — PBR Map Prediction from a Single Image

MatForge is a local Streamlit application that predicts physically-based rendering (PBR) maps — Normal, Roughness, and Metallic — from a single RGB image. It runs entirely on-device, requires no internet connection after setup, and is designed for 3D artists and technical artists who need to generate PBR material maps from photographic references.

![Hero shot](docs/assets/hero_shot.png)

---

## Features

- **PBR map prediction** — Normal, Roughness, and Metallic maps from a single RGB input.
- **Material classifier** — automatic material group detection (brick, wood, metal, stone, fabric, and more) using DINOv2 + KNN, with manual override.
- **Optional Super-Resolution** — Real-ESRGAN ×4 upscaling before inference for low-resolution inputs.
- **Perspective correction** — interactive four-point warp with live preview before inference.
- **Roughness / Metallic adjustment** — gain and offset sliders per channel.
- **Calibration by group** — applies group-specific correction curves based on the detected material.
- **Normal map quality evaluation** — heuristic scoring (coherence, continuity, blockiness) with a diagnostic heatmap.
- **Make Tileable** — seamless frequency-domain blending for tileable outputs.
- **Material Blender (RNM)** — blend two PBR material sets using Reoriented Normal Mapping.
- **Procedural Variations** — three noise-based techniques (Zonal Mix, Worn Edges, Scale Shift) with seed control.
- **3D Preview** — real-time Three.js viewer with geometry selector, environment toggle, and color overlay.
- **Multi-engine export** — Blender, Unreal Engine 5, Unity URP, Unity HDRP, and Godot 4, with XMP metadata embedded in every PNG.
- **Batch ZIP processing** — process an entire ZIP of images through the full pipeline and download a single organized archive.

![3D Viewer](docs/assets/viewer_3d.png)

---

## System Requirements

| Component | Minimum |
|---|---|
| OS | Windows 10 / 11 (64-bit) |
| Python | 3.11 |
| GPU | NVIDIA GPU with 4 GB VRAM (CUDA-capable) |
| CUDA | 11.8 |
| NVIDIA Driver | ≥ 452.39 |
| RAM | 8 GB |
| Disk | ~4 GB (models + environment) |

> **CPU fallback**: MatForge runs on CPU if no CUDA-capable GPU is detected. Processing times will be significantly longer. The application displays the active device (CUDA / CPU) in the title bar at startup.

> **Performance note**: processing times were benchmarked on an NVIDIA GTX 1650 Max-Q (4 GB VRAM), CUDA 11.8, Python 3.11, Windows 11. Estimated times shown in the UI are calibrated to this hardware and may differ on other configurations.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/migueljeronimogutierrez/MatForge-App.git
cd MatForge-App
```

### 2. Download model weights

Model weights are distributed as release assets due to their size (tracked via Git LFS). Download them from the [latest release](https://github.com/migueljeronimogutierrez/MatForge-App/releases/latest) and place them in the following locations:

```
checkpoints/
├── matforge/
│   └── best_gan.pt
└── sr/
    ├── sr_ft_phase1_best_lpips.pt
    └── RealESRGAN_x4plus.pth
```

### 3. Run the installer

```bash
install.bat
```

This script creates a virtual environment, installs PyTorch with CUDA 11.8, and installs all remaining dependencies from `requirements.txt`.

### 4. Launch the application

```bash
launch_matforge.bat
```

The application will open automatically in your default browser at `http://localhost:8501`.

---

## Manual Installation

If you prefer to set up the environment manually:

```bash
# Create virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate

# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

Then launch with:

```bash
.venv\Scripts\activate
streamlit run app.py
```

---

## Usage

Upload an RGB image using the sidebar uploader. Adjust zoom and optional Super-Resolution settings, then click **Generate Maps**. Use the tabs in the main area to inspect Normal, Roughness, and Metallic outputs. Export to your target engine using the Export section.

Sample input images are provided in `sample_inputs/` for immediate testing. These images were not used during model training.

For detailed usage instructions, see the [User Manual](docs/USER_MANUAL.md). For information on the technical structure and implementation details, refer to the [Technical Manual](docs/TECHNICAL_MANUAL.md).

![Export](docs/assets/export.png)

---

## Project Structure

```
MatForge-App/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── LICENSE
├── PI/                     # Research documentation (in Spanish)
├── install.bat             # One-time environment setup
├── launch_matforge.bat     # Application launcher
├── launch_matforge.ps1     # PowerShell alternative launcher
├── checkpoints/            # Model weights (download separately)
│   ├── matforge/
│   └── sr/
├── artifacts/              # KNN classifier artifacts
├── sample_inputs/          # CC0 sample images for testing
├── src/                    # Source modules
│   ├── models.py           # MatForgeNet architecture
│   ├── inference.py        # Tile-and-merge inference pipeline
│   ├── classifier.py       # DINOv2 + KNN material classifier
│   ├── postprocess.py      # Adjustments, blending, variations
│   ├── quality.py          # Normal map quality evaluation
│   ├── export.py           # Multi-engine export with XMP metadata
│   ├── sr.py               # Real-ESRGAN super-resolution module
│   └── utils.py            # Shared utilities
└── docs/
    ├── USER_MANUAL.md
    ├── MANUAL_DE_USUARIO.md
    ├── TECHNICAL_MANUAL.md
    ├── MANUAL_TECNICO.md
    └── assets/             # Screenshots used in documentation
```

---

## Models

### MatForge

A custom encoder-decoder architecture trained from scratch for dense PBR map prediction:

- **Encoder**: PVT-v2-B1 (hierarchical vision transformer), pre-trained on ImageNet-1K via timm.
- **Decoder**: custom FPN with skip connections at four scales.
- **Output heads**: Normal (3ch, Tanh + L2 renormalization), Roughness (1ch, Sigmoid), Metallic (1ch, Sigmoid).
- **Training**: 90 supervised epochs on the MatSynth dataset, followed by GAN fine-tuning with a multi-scale PatchGAN discriminator.
- **Final checkpoint performance**: MAE Normal 10.37°, LPIPS 0.0976.

### Super-Resolution Module

Real-ESRGAN (RRDBNet, 23 residual blocks) fine-tuned on MatSynth for domain-specific upscaling. Applied optionally before MatForge inference to improve results on low-resolution inputs. Inference uses tile-and-merge with a Hann window (256×256 tiles, stride 128) for seamless reconstruction.

### Material Classifier

DINOv2-small (ViT-S/14, 518×518 input) with PCA-50 dimensionality reduction and a KNN classifier trained on MatSynth material group labels. Used to select group-specific calibration curves and to contextualize quality evaluation warnings.

---

## Licenses and Attribution

MatForge is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

Third-party components and their licenses:

| Component | License | Reference |
|---|---|---|
| PVT-v2-B1 (via timm) | Apache 2.0 | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| DINOv2-small | Apache 2.0 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| Real-ESRGAN | BSD-3-Clause | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| MatSynth dataset | CC0 / CC-BY 4.0 | [gvecchio/MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth) |
| Three.js | MIT | [threejs.org](https://threejs.org) |

> **Note on PVT-v2-B1 weights**: the pre-trained weights used by this model were trained on ImageNet-1K, which carries a non-commercial research restriction. Use of this application for commercial purposes may require legal review. See the [legal audit document](docs/INFORME_LEGAL.md) for a detailed analysis.

### AI Act (EU Regulation 2024/1689)

MatForge is classified as a minimal-risk AI system under Article 2(6) (scientific research exemption). All generated outputs include XMP provenance metadata identifying them as AI-generated. A transparency notice is displayed in the application interface in accordance with Article 50 requirements effective August 2026.

### Sample Images

Sample images in `sample_inputs/` are sourced from:

- [Poly Haven](https://polyhaven.com) — CC0
- [Pixnio](https://pixnio.com) — CC0
- [PxHere](https://pxhere.com) — CC0

These images were not used during model training.

---

## Academic Context

Developed as a final project for the Postgraduate Programme in Artificial Intelligence and Big Data at [EUSA — Cámara de Comercio de Sevilla](https://www.eusa.es).
