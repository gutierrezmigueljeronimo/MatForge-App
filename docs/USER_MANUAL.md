# MatForge — User Manual

**Version**: 1.0 | **Application**: MatForge App | **Last updated**: May 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Interface Layout](#2-interface-layout)
3. [Loading an Image](#3-loading-an-image)
4. [Perspective Correction](#4-perspective-correction)
5. [Generating PBR Maps](#5-generating-pbr-maps)
6. [Inspecting the Results](#6-inspecting-the-results)
7. [Sidebar Tools](#7-sidebar-tools)
   - 7.1 [Adjust Roughness / Metallic](#71-adjust-roughness--metallic)
   - 7.2 [Normal Map Quality](#72-normal-map-quality)
   - 7.3 [Calibrate by Group](#73-calibrate-by-group)
   - 7.4 [Make Tileable](#74-make-tileable)
8. [Main Area Tools](#8-main-area-tools)
   - 8.1 [Material Blender](#81-material-blender)
   - 8.2 [Procedural Variations](#82-procedural-variations)
   - 8.3 [3D Preview](#83-3d-preview)
   - 8.4 [Comparison](#84-comparison)
   - 8.5 [Tiling Preview](#85-tiling-preview)
   - 8.6 [Batch ZIP](#86-batch-zip)
9. [Exporting Maps](#9-exporting-maps)
10. [Map States Reference](#10-map-states-reference)
11. [Performance Guidelines](#11-performance-guidelines)
12. [Known Limitations](#12-known-limitations)

---

## 1. Overview

MatForge predicts three PBR maps from a single RGB photograph:

- **Normal map** — encodes surface micro-geometry as directional vectors. Used by renderers to simulate lighting detail without additional geometry.
- **Roughness map** — controls how diffuse or specular a surface appears. Values close to 0 produce mirror-like reflections; values close to 1 produce fully diffuse surfaces.
- **Metallic map** — distinguishes metallic from non-metallic surfaces. Most materials are either fully metallic (1.0) or fully non-metallic (0.0).

The application processes one image at a time through a configurable pipeline: optional perspective correction → zoom → optional Super-Resolution → material classification → PBR inference. Post-processing tools (adjustment, calibration, blending, tiling, variations) are applied non-destructively on top of the generated maps.

---

## 2. Interface Layout

The interface is divided into two areas:

**Sidebar (left)** — image upload, pipeline configuration (zoom, Super-Resolution), and per-map post-processing tools (Adjust R/M, Normal Map Quality, Calibrate by Group, Make Tileable).

**Main area (right)** — map tabs, input image inspector, and all interactive tools (Perspective Correction, Material Blender, Procedural Variations, 3D Preview, Comparison, Tiling Preview, Batch ZIP, Export).

![Sidebar_1](assets/sidebar_1.png)

![Sidebar_2](assets/sidebar_2.png)

---

## 3. Loading an Image

Use the **Upload image** file uploader in the sidebar. Accepted formats: JPG, PNG, WEBP.

**Zoom slider** — scales the image before inference. Use this to control the effective resolution passed to MatForge:

| Input size | Recommended zoom |
|---|---|
| Up to 1K (1024 px) | 1.0 |
| Around 2K (2048 px) | 0.5 |
| Around 4K (4096 px) | 0.25 |

Reducing zoom lowers processing time significantly. The sidebar displays the effective resolution after zoom and an estimated processing time before generation.

**Super-Resolution (SR)** — when enabled, Real-ESRGAN upscales the zoomed image ×4 before passing it to MatForge. This improves results for low-resolution inputs but adds approximately 9 seconds of processing time and significantly increases the output resolution. SR is not recommended for inputs already above 512 px after zoom, as it may produce flat or incoherent results beyond the resolution MatForge was trained on.

> The title bar displays **[CUDA]** or **[CPU]** to indicate the active device. Processing on CPU is supported but substantially slower.

---

## 4. Perspective Correction

The **Perspective Correction** tool is available as soon as an image is uploaded, before generating maps.

![Perspective Correction](assets/perspective_correction.png)

1. Open the **Perspective Correction** expander in the main area.
2. Drag the four amber handles to the corners of the surface to correct.
3. Click **↩ Update preview** to see the corrected result and a 2×2 tiling preview.
4. Click **Apply** to confirm. The corrected image becomes the pipeline input.
5. Click **↺ Reset** to discard the correction and return to the original image.

The correction is invalidated automatically when a new image is uploaded.

---

## 5. Generating PBR Maps

Click **Generate Maps** in the sidebar after configuring zoom and SR options. The pipeline runs sequentially:

1. Zoom is applied to the uploaded (or warped) image.
2. If SR is enabled, Real-ESRGAN upscales the result ×4.
3. The material classifier identifies the material group and confidence score.
4. MatForge runs tile-and-merge inference (256×256 tiles, Hann window blending).
5. Raw maps are stored and made available to all post-processing tools.

The sidebar displays the detected material group and KNN distance after generation. All post-processing tools become available once maps are generated.

---

## 6. Inspecting the Results

The main area shows three tabs after generation: **Normal**, **Roughness**, and **Metallic**. Each tab displays the currently active map state (Raw, Adjusted, Calibrated, etc. — see [Section 10](#10-map-states-reference)).

The **Input Image** expander below the tabs shows:

- **Original** — the uploaded image dimensions before any processing.
- **Pre-SR input** — the effective resolution after zoom, before Super-Resolution.
- **SR output** — the final resolution passed to MatForge (only shown when SR was active).
- **Material group** and **KNN distance** — classifier output.

---

## 7. Sidebar Tools

### 7.1 Adjust Roughness / Metallic

Applies gain and offset corrections to the Roughness and Metallic channels of the Raw maps. Results are stored as the **Adjusted** state.

- **Gain** — multiplies the channel values. Values above 1.0 increase the effect; values below 1.0 reduce it.
- **Offset** — adds a constant shift to all values, clamped to [0, 1].

Click **Apply R/M Adjustment** to compute and store the Adjusted maps.

### 7.2 Normal Map Quality

Evaluates the generated Normal map using three heuristic metrics:

- **Coherence** — fraction of pixels with approximately unit-length vectors.
- **Continuity** — smoothness of the normal field; high gradients may indicate seam artifacts.
- **Blockiness** — detection of patch-based grid patterns from tile-and-merge inference.

Click **Evaluate quality** to run the evaluation. An estimated time is shown before running — this operation is CPU-intensive and may take up to several minutes for large maps.

Results include a score per metric, an overall score, contextualised warnings based on the detected material group, and a diagnostic heatmap overlay available in the Normal tab.

### 7.3 Calibrate by Group

Applies group-specific gain and offset corrections derived from the detected material class. Uses the Raw maps as the source of truth and stores results as the **Calibrated** state.

The detected group is shown with its confidence score. Use the **Override group** selector to apply a different group's calibration curves if the automatic detection is incorrect.

Click **Calibrate** to compute and store the Calibrated maps.

### 7.4 Make Tileable

Applies frequency-domain blending to reduce visible seams when the maps are tiled. Operates on Calibrated maps if available, then Adjusted, then Raw. Results are stored as the **Tileable** state.

The **SR seam blend** option applies additional blending at tile boundaries introduced by Super-Resolution. Enable this if SR was active during generation.

This tool is marked **Beta** — results depend heavily on the input material. Verify the output in the [Tiling Preview](#85-tiling-preview) before exporting.

---

## 8. Main Area Tools

### 8.1 Material Blender

Blends two PBR material sets using Reoriented Normal Mapping (RNM) for the Normal channel and linear interpolation for Roughness and Metallic.

![Material Blender](assets/material_blender.png)

1. Upload Normal B, Roughness B, and Metallic B maps using the provided uploaders.
2. Optionally upload a Color B map for color blending.
3. Adjust the **Blend factor** slider (0.0 = 100% material A, 1.0 = 100% material B).
4. Click **Blend Materials** to compute and store the **Blended** state.

If material B maps have a different resolution than the generated maps, they are resized automatically with a warning.

### 8.2 Procedural Variations

Generates up to three variations of the base material using noise-based techniques. Operates on Raw maps.

![Procedural Variations](assets/procedural_variations.png)

- **Zonal Mix** — applies FBM noise to create spatially varying roughness zones.
- **Worn Edges** — uses normal map gradients to identify edges and applies a wear mask to roughness.
- **Scale Shift** — randomly scales and repositions the texture within the original dimensions.

If no Normal map is available, only Zonal Mix and Scale Shift are generated.

Adjust **Number of variants** (1–3) and **Seed** before clicking **Generate Variations**. An estimated processing time is shown — this operation is CPU-intensive. Select a variant from the preview grid and click **Apply Variation** to store it as the **Variation** state.

### 8.3 3D Preview

A real-time Three.js viewer for evaluating the generated material on a 3D surface.

![3D Viewer](assets/viewer_3d.png)

- **Geometry** — Sphere, Box or Plane.
- **Preview state** — selects which map state to display (Raw, Adjusted, Calibrated, Blended, Tileable, Variation).
- **Show source color** — overlays the input image as an albedo map on the geometry.
- **RoomEnvironment** — toggles a procedural environment map for more realistic lighting.
- **Tiling** — applies ×2 UV tiling to evaluate the material at a smaller apparent scale.

Drag to rotate, scroll to zoom.

### 8.4 Comparison

A slider-based side-by-side comparison between two map states or channels. Drag the slider to reveal either side.

Use the **Left** and **Right** selectors to choose the states and channels to compare. The Color channel uses the source input image.

### 8.5 Tiling Preview

Displays a 2×2 tiled mosaic of the selected map to evaluate seamless repetition.

![Tiling Preview](assets/tiling_preview.png)

Select the **State** (Raw, Adjusted, Calibrated, Blended, Tileable, Variation) and **Map** (Normal, Roughness, Metallic, Color) to preview. Use this after applying Make Tileable to verify that seams are not visible at tile boundaries.

### 8.6 Batch ZIP

Processes multiple images in a single run. Upload a ZIP file containing JPG, PNG, or WEBP images.

![Batch ZIP](assets/batch_zip.png)

**Settings:**
- **Engine** — export format applied to all images in the batch.
- **Zoom** — applied uniformly to all images.
- **Super-Resolution** — applied to each image before inference if enabled.

**Pre-flight analysis** — before processing, the tool analyses the ZIP contents and displays:
- Number of valid images found.
- A warning if any image's effective resolution exceeds 1024 px after zoom/SR — results above the training resolution may be flat or incoherent.
- A warning if SR is enabled with more than 3 images.
- A warning if more than 10 images are detected.
- Estimated total processing time with and without SR.

Click **Process Batch** to start. Progress is shown image by image. Failed images are skipped and logged — the batch never aborts on a single error. A summary and download button are shown on completion.

Output ZIP structure:
```
matforge_batch_{engine}.zip
├── image_name_1/
│   ├── (engine-specific map files)
├── image_name_2/
│   └── ...
```

---

## 9. Exporting Maps

The **Export** section is located in the main area, below the map tabs.

![Export](assets/export.png)

1. Enter an **Asset name** (used as the base filename in the exported files).
2. Select the target **Engine**.
3. Select the **Map state** to export (Raw, Adjusted, Calibrated, Blended, Tileable, or Variation).
4. Click **Export** to download a ZIP file with all maps for that engine.

All exported PNG files include embedded XMP metadata identifying them as AI-generated outputs.

**Output files by engine:**

| Engine | Normal | Roughness/Metallic | Color |
|---|---|---|---|
| Blender | `_normal.png` | `_roughness.png`, `_metallic.png` | `_color.png` |
| Unreal Engine 5 | `T_name_N.png` | `T_name_ORM.png` | `T_name_D.png` |
| Unity URP | `_normal.png` | `_MetallicSmoothness.png` | `_Albedo.png` |
| Unity HDRP | `_normal.png` | `_MaskMap.png` | `_Albedo.png` |
| Godot 4 | `_normal.png` | `_orm.png` | `_albedo.png` |

---

## 10. Map States Reference

Post-processing tools produce named states that accumulate non-destructively. Each tool reads from its defined source and writes to its own state — the Raw maps are never overwritten.

| State | Source | Produced by |
|---|---|---|
| Raw | — | Generate Maps |
| Adjusted | Raw | Adjust R/M |
| Calibrated | Raw | Calibrate by Group |
| Blended | Active maps | Material Blender |
| Tileable | Calibrated → Adjusted → Raw | Make Tileable |
| Variation | Raw | Procedural Variations |

All tools that have a state selector (3D Preview, Comparison, Tiling Preview, Export) allow choosing which state to use independently.

---

## 11. Performance Guidelines

| Operation | GTX 1650 Max-Q (4 GB VRAM) |
|---|---|
| Generate Maps — 512×512, zoom 1.0 | ~1 s |
| Generate Maps — 1024×1024, zoom 1.0 | ~6 s |
| Generate Maps — 1920×1920, zoom 0.5 | ~5 s |
| Super-Resolution overhead (any size) | +~9 s |
| Normal Map Quality — 512×512 | ~7 s |
| Normal Map Quality — 1024×1024 | ~100 s |
| Procedural Variations — 352×352 | ~25 s |
| Procedural Variations — 640×640 | ~80 s |

Normal Map Quality evaluation and Procedural Variations are CPU-intensive operations. Estimated times are shown in the interface before running each tool.

---

## 12. Known Limitations

- **Minimum effective resolution**: images smaller than 256 px in any dimension after zoom cause a padding error during inference. Use zoom values that keep the effective resolution at or above 256 px.
- **Training resolution ceiling**: MatForge was trained on patches up to 1024 px. Results at higher effective resolutions (e.g. after SR on a large image) may be flat or lack surface detail.
- **Normal Map Quality evaluation time**: the blockiness metric uses a sliding-window entropy computation that is slow on CPU. Evaluation of maps larger than 512×512 may take over a minute.
- **Procedural Variations — Scale Shift**: visible displacement may appear on non-tileable inputs. The effect is more natural on tileable textures.
- **Make Tileable**: results depend on the frequency content of the input. Highly detailed or non-repetitive textures may still show residual seams.
- **Perspective Correction**: the corrected image region is not automatically used as the input for zoom and time estimation in the sidebar — these values still refer to the original uploaded image.
- **Batch ZIP**: images with effective resolution below 256 px after zoom are skipped with an error. Adjust zoom before processing.
