#!/usr/bin/env python3
"""
MatForge EDA v1.0 — Análisis Exploratorio y Limpieza Semi-Automática
======================================================================
Ejecutar en local (VS Code) sobre el SSD donde reside el dataset.
NO requiere GPU. Tiempo estimado: 15-25 min sobre SSD para ~3.800 texturas.

ORDEN DE EJECUCIÓN RECOMENDADO:
  1. Elimina manualmente las placas base en metal (metal_XXXX.png).
  2. Edita la sección CONFIGURACIÓN (solo MATFORGE_DIR y RESOLUCION_ANALISIS).
  3. Ejecuta: python matforge_eda.py
  4. Revisa el informe HTML generado en: <MATFORGE_DIR>/eda_output/revision_humana.html
  5. Abre candidates_to_discard.csv y marca "si" en la columna "confirmar_descarte".
  6. Vuelve a ejecutar con MODO = "aplicar_descarte" para mover los archivos.

DEPENDENCIAS:
  pip install opencv-python numpy pandas matplotlib seaborn imagehash Pillow tqdm
"""

import os
import json
import shutil
import csv
import math
import hashlib
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Sin pantalla: genera los PNGs sin abrir ventanas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
import imagehash
from tqdm import tqdm


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURACIÓN                                 ║
# ║          ← EDITA SOLO ESTA SECCIÓN ←                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Carpeta raíz del dataset (la que contiene la subcarpeta maps/)
MATFORGE_DIR = r"F:\PI\matforge_dataset"

# Modo de ejecución:
#   "analizar"        → extrae métricas, genera gráficos e informe HTML de sospechosos
#   "aplicar_descarte"→ lee el CSV confirmado y mueve los archivos a descartados/
MODO = "aplicar_descarte"

# Resolución de análisis: trabaja a menor resolución para velocidad.
# Las métricas estadísticas son robustas a 256px. No cambies esto.
RESOLUCION_ANALISIS = (256, 256)

# Umbral de Hamming para near-duplicates con pHash (0=idéntico, 10=muy parecido)
# 6 es el estándar en la literatura para texturas con ligeras variaciones de color.
PHASH_UMBRAL = 6

# ╚══════════════════════════════════════════════════════════════════════╝


# ─── Rutas derivadas ───────────────────────────────────────────────────
MAPS_DIR      = os.path.join(MATFORGE_DIR, "maps")
DIR_RGB       = os.path.join(MAPS_DIR, "rgb")
DIR_NORMAL    = os.path.join(MAPS_DIR, "normal")
DIR_ROUGH     = os.path.join(MAPS_DIR, "roughness")
DIR_METALLIC  = os.path.join(MAPS_DIR, "metallic")
DIR_OUT       = os.path.join(MATFORGE_DIR, "eda_output")
DIR_DESCARTES = os.path.join(MATFORGE_DIR, "descartados")

CSV_METRICAS   = os.path.join(DIR_OUT, "metricas_completas.csv")
CSV_CANDIDATOS = os.path.join(DIR_OUT, "candidates_to_discard.csv")
HTML_REVISION  = os.path.join(DIR_OUT, "revision_humana.html")
DIR_GRAFICOS   = os.path.join(DIR_OUT, "graficos")


# ══════════════════════════════════════════════════════════════════════
# UMBRALES POR CATEGORÍA
# ══════════════════════════════════════════════════════════════════════
#
# Cada categoría tiene umbrales adaptados a sus propiedades físicas reales.
# NO son globales. Esto es lo que el EDA anterior no hacía.
#
# Claves de cada entrada:
#   azul_min     → media canal Z del normal. Un normal OpenGL válido tiene Z>0,
#                  mapeado a [128, 255]. Valores < 160 indican inversión o corrupción.
#   std_normal_max → varianza máxima tolerable del normal. Un valor >80 indica ruido
#                    extremo inducible por corrupción de archivo o convención errónea.
#   std_rgb_min_para_normal_alto → en el filtro "albedo muerto + relieve fuerte",
#                  el "albedo muerto" se define como std_rgb < este umbral.
#   rough_media_min → roughness con media por debajo de este valor es sospechoso
#                     de ser un espejo perfecto (solo si std_rough también es bajo).
#   rough_std_min  → por debajo de este umbral, el roughness es completamente plano.
#                    Se combina con rough_media para evitar falsos positivos en marble.
#   metallic_check → si True, analiza el mapa metallic de esta categoría.
#
# JUSTIFICACIÓN de los valores clave:
#   marble: rough_media_min muy bajo (5) porque el mármol pulido es físicamente
#           correcto con roughness casi negro. El EDA anterior descartó demasiados.
#   metal:  rough_media_min=0 porque los metales pulidos pueden tener roughness=0.
#           std_normal_max más alto porque los metales tienen normales más suaves.
#   wood:   rough_media_min moderado. La madera siempre tiene algo de rugosidad.
#   ground: azul_min más permisivo (150) porque suelos con piedras y raíces pueden
#           tener normales localmente invertidas en zonas pequeñas.

UMBRALES = {
    "concrete":   {"azul_min": 160, "std_normal_max": 80, "std_rgb_dead": 5.0,
                   "rough_media_min": 20, "rough_std_min": 2.5, "metallic_check": False},
    "marble":     {"azul_min": 160, "std_normal_max": 80, "std_rgb_dead": 5.0,
                   "rough_media_min": 5,  "rough_std_min": 1.0, "metallic_check": False},
    "plaster":    {"azul_min": 160, "std_normal_max": 80, "std_rgb_dead": 5.0,
                   "rough_media_min": 15, "rough_std_min": 2.0, "metallic_check": False},
    "stone":      {"azul_min": 160, "std_normal_max": 80, "std_rgb_dead": 5.0,
                   "rough_media_min": 20, "rough_std_min": 2.5, "metallic_check": False},
    "terracotta": {"azul_min": 160, "std_normal_max": 80, "std_rgb_dead": 5.0,
                   "rough_media_min": 20, "rough_std_min": 2.5, "metallic_check": False},
    "wood":       {"azul_min": 160, "std_normal_max": 80, "std_rgb_dead": 5.0,
                   "rough_media_min": 15, "rough_std_min": 2.0, "metallic_check": False},
    "metal":      {"azul_min": 155, "std_normal_max": 90, "std_rgb_dead": 4.0,
                   "rough_media_min": 0,  "rough_std_min": 0.5, "metallic_check": True},
    "ceramic":    {"azul_min": 160, "std_normal_max": 80, "std_rgb_dead": 5.0,
                   "rough_media_min": 10, "rough_std_min": 1.5, "metallic_check": False},
    "ground":     {"azul_min": 150, "std_normal_max": 85, "std_rgb_dead": 5.0,
                   "rough_media_min": 20, "rough_std_min": 2.5, "metallic_check": False},
}

# Fallback para categorías no mapeadas (no debería ocurrir con nuestro dataset)
UMBRAL_DEFAULT = {"azul_min": 160, "std_normal_max": 80, "std_rgb_dead": 5.0,
                  "rough_media_min": 20, "rough_std_min": 2.5, "metallic_check": False}


# ══════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════

def leer_imagen_robusta(ruta, modo=cv2.IMREAD_COLOR):
    """Lee imagen con soporte a rutas con tildes/espacios."""
    try:
        buf = np.fromfile(ruta, dtype=np.uint8)
        img = cv2.imdecode(buf, modo)
        return img
    except Exception:
        return None


def redimensionar(img, size=RESOLUCION_ANALISIS):
    if img is None:
        return None
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def extraer_categoria(nombre_archivo):
    """Extrae la categoría del nombre 'categoria_NNNN.png'."""
    return nombre_archivo.rsplit("_", 1)[0]


def setup_directorios():
    os.makedirs(DIR_OUT, exist_ok=True)
    os.makedirs(DIR_GRAFICOS, exist_ok=True)
    for sub in ["rgb", "normal", "roughness", "metallic"]:
        os.makedirs(os.path.join(DIR_DESCARTES, sub), exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# FASE 1 — EXTRACCIÓN DE MÉTRICAS
# ══════════════════════════════════════════════════════════════════════

def extraer_metricas():
    print("\n" + "="*60)
    print("  FASE 1 — Extracción de métricas")
    print("="*60)

    archivos_rgb = sorted(os.listdir(DIR_RGB))
    registros = []

    for archivo in tqdm(archivos_rgb, desc="Analizando texturas", unit="tex"):
        if not archivo.endswith(".png"):
            continue

        ruta_rgb  = os.path.join(DIR_RGB,      archivo)
        ruta_norm = os.path.join(DIR_NORMAL,   archivo)
        ruta_rug  = os.path.join(DIR_ROUGH,    archivo)
        ruta_met  = os.path.join(DIR_METALLIC, archivo)

        # Integridad obligatoria: los tres mapas base deben existir
        if not (os.path.exists(ruta_norm) and os.path.exists(ruta_rug)):
            registros.append({"archivo": archivo, "categoria": extraer_categoria(archivo),
                               "ERROR": "Mapas incompletos"})
            continue

        categoria = extraer_categoria(archivo)

        # Carga y redimensión para velocidad
        img_bgr  = redimensionar(leer_imagen_robusta(ruta_rgb,  cv2.IMREAD_COLOR))
        img_norm = redimensionar(leer_imagen_robusta(ruta_norm, cv2.IMREAD_COLOR))
        img_rug  = redimensionar(leer_imagen_robusta(ruta_rug,  cv2.IMREAD_GRAYSCALE))

        if img_bgr is None or img_norm is None or img_rug is None:
            registros.append({"archivo": archivo, "categoria": categoria,
                               "ERROR": "No se pudo leer"})
            continue

        img_rgb  = cv2.cvtColor(img_bgr,  cv2.COLOR_BGR2RGB)
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)

        # ─── Métricas RGB ────────────────────────────────────────────
        std_rgb   = float(np.std(img_rgb))
        media_rgb = float(np.mean(img_rgb))

        # ─── Métricas Normal ─────────────────────────────────────────
        std_normal  = float(np.std(img_norm))
        canal_r     = img_norm[:, :, 0].astype(float)
        canal_g     = img_norm[:, :, 1].astype(float)
        canal_b     = img_norm[:, :, 2].astype(float)
        media_azul  = float(np.mean(canal_b))
        media_rojo  = float(np.mean(canal_r))
        media_verde = float(np.mean(canal_g))

        # Cociente R/G: un normal OpenGL bien calibrado tiene R≈G≈128.
        # Desviaciones fuertes indican convención invertida o corrupción.
        cociente_rg = media_rojo / media_verde if media_verde > 0 else 999.0

        # Renormalización implícita: comprobamos si los vectores son unitarios.
        # Mapeamos [0,255] → [-1,1]: v = (pixel/127.5) - 1
        norm_r = canal_r / 127.5 - 1.0
        norm_g = canal_g / 127.5 - 1.0
        norm_b = canal_b / 127.5 - 1.0
        magnitudes = np.sqrt(norm_r**2 + norm_g**2 + norm_b**2)
        # Un normal válido tiene magnitud ~1.0. Desviaciones > 0.3 son sospechosas.
        desviacion_norma = float(np.mean(np.abs(magnitudes - 1.0)))

        # ─── Métricas Roughness ───────────────────────────────────────
        std_rough   = float(np.std(img_rug))
        media_rough = float(np.mean(img_rug))

        # ─── Métricas Metallic (solo si existe el archivo) ───────────
        std_metallic   = None
        media_metallic = None
        if os.path.exists(ruta_met):
            img_met = redimensionar(leer_imagen_robusta(ruta_met, cv2.IMREAD_GRAYSCALE))
            if img_met is not None:
                std_metallic   = float(np.std(img_met))
                media_metallic = float(np.mean(img_met))

        # ─── pHash (sobre rgb a tamaño fijo para velocidad) ──────────
        try:
            pil_img = Image.fromarray(img_rgb)
            ph = str(imagehash.phash(pil_img))
        except Exception:
            ph = ""

        registros.append({
            "archivo":          archivo,
            "categoria":        categoria,
            "std_rgb":          round(std_rgb, 4),
            "media_rgb":        round(media_rgb, 4),
            "std_normal":       round(std_normal, 4),
            "media_azul":       round(media_azul, 4),
            "media_rojo":       round(media_rojo, 4),
            "media_verde":      round(media_verde, 4),
            "cociente_rg":      round(cociente_rg, 4),
            "desviacion_norma": round(desviacion_norma, 4),
            "std_rough":        round(std_rough, 4),
            "media_rough":      round(media_rough, 4),
            "std_metallic":     round(std_metallic, 4) if std_metallic is not None else None,
            "media_metallic":   round(media_metallic, 4) if media_metallic is not None else None,
            "phash":            ph,
            "ERROR":            "",
        })

    df = pd.DataFrame(registros)
    df.to_csv(CSV_METRICAS, index=False, encoding="utf-8")
    print(f"\n✅ Métricas guardadas: {CSV_METRICAS}")
    print(f"   Total analizadas: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════════
# FASE 2 — DETECCIÓN DE NEAR-DUPLICATES
# ══════════════════════════════════════════════════════════════════════

def detectar_duplicados(df):
    """
    Compara pHashes de todos los pares. O(n²) pero con n≈3800 y pHash de 64 bits
    es perfectamente manejable en CPU (~30 segundos).
    Retorna un set de archivos a marcar como near-duplicate.
    """
    print("\n" + "="*60)
    print("  FASE 2 — Detección de near-duplicates (pHash)")
    print("="*60)

    df_valid = df[df["phash"].notna() & (df["phash"] != "")].copy()
    hashes   = {row["archivo"]: imagehash.hex_to_hash(row["phash"])
                for _, row in df_valid.iterrows()}

    archivos   = list(hashes.keys())
    n          = len(archivos)
    duplicados = {}   # archivo → archivo del que es duplicado

    print(f"   Comparando {n} pHashes...")
    for i in tqdm(range(n), desc="pHash sweep", unit="img"):
        a1 = archivos[i]
        if a1 in duplicados:
            continue
        for j in range(i + 1, n):
            a2 = archivos[j]
            if a2 in duplicados:
                continue
            distancia = hashes[a1] - hashes[a2]
            if distancia <= PHASH_UMBRAL:
                # Conservamos el que tenga menor índice numérico (el "original")
                duplicados[a2] = a1

    print(f"   Near-duplicates detectados: {len(duplicados)}")
    return duplicados


# ══════════════════════════════════════════════════════════════════════
# FASE 3 — APLICACIÓN DE FILTROS Y GENERACIÓN DE CANDIDATOS
# ══════════════════════════════════════════════════════════════════════

def aplicar_filtros(df, duplicados):
    print("\n" + "="*60)
    print("  FASE 3 — Aplicación de filtros por categoría")
    print("="*60)

    candidatos = []   # Lista de dicts para el CSV
    df_limpio  = df[df["ERROR"] == ""].copy()

    for _, fila in df_limpio.iterrows():
        archivo   = fila["archivo"]
        categoria = fila["categoria"]
        u         = UMBRALES.get(categoria, UMBRAL_DEFAULT)

        motivos   = []
        score     = 0   # Cuanto más alto, más seguro es el descarte

        std_rgb          = fila["std_rgb"]
        std_normal       = fila["std_normal"]
        media_azul       = fila["media_azul"]
        cociente_rg      = fila["cociente_rg"]
        desviacion_norma = fila["desviacion_norma"]
        std_rough        = fila["std_rough"]
        media_rough      = fila["media_rough"]
        std_metallic     = fila.get("std_metallic")
        media_metallic   = fila.get("media_metallic")

        # ── F1: Albedo muerto + relieve fuerte ──────────────────────────
        # El "efecto gotelé" tiene std_rgb baja Y std_normal baja (ambas sin señal).
        # El fallo real tiene std_rgb baja Y std_normal ALTA (incoherente).
        if std_rgb < u["std_rgb_dead"] and std_normal > 50.0:
            motivos.append(f"Albedo muerto (std_rgb={std_rgb:.1f}) con relieve fuerte (std_normal={std_normal:.1f})")
            score += 3   # Alta confianza

        # ── F2: Normal azul bajo (vector Z medio no apunta a cámara) ───
        if media_azul < u["azul_min"]:
            motivos.append(f"Canal Z del normal bajo (media_azul={media_azul:.1f} < {u['azul_min']})")
            score += 3

        # ── F3: Desequilibrio R/G del normal (convención incorrecta) ───
        if cociente_rg < 0.70 or cociente_rg > 1.40:
            motivos.append(f"Desequilibrio R/G del normal (ratio={cociente_rg:.2f}). Posible convención DirectX o corrupción.")
            score += 2

        # ── F4: Vectores no unitarios (desviación de norma) ─────────────
        if desviacion_norma > 0.30:
            motivos.append(f"Vectores normales no unitarios (desv_norma={desviacion_norma:.3f})")
            score += 1   # Sospechoso pero no definitivo solo

        # ── F5: Ruido extremo en normal ──────────────────────────────────
        if std_normal > u["std_normal_max"]:
            motivos.append(f"Ruido extremo en normal (std={std_normal:.1f} > {u['std_normal_max']})")
            score += 2

        # ── F6: Roughness completamente plano ────────────────────────────
        # Solo flaggea si AMBAS condiciones se cumplen para evitar falsos positivos
        # en marble (roughness oscuro pero válido) y metal (roughness claro pero válido).
        if std_rough < u["rough_std_min"]:
            if media_rough < u["rough_media_min"] or media_rough > 245:
                motivos.append(f"Roughness completamente plano (std={std_rough:.2f}, media={media_rough:.1f})")
                score += 2
            else:
                # Roughness plano pero en rango medio: sospechoso, score bajo
                motivos.append(f"Roughness casi plano, posible válido (std={std_rough:.2f}, media={media_rough:.1f})")
                score += 1

        # ── F7: Metallic todo blanco en categoría no-metal ──────────────
        if u["metallic_check"] and std_metallic is not None and media_metallic is not None:
            if std_metallic < 2.0 and media_metallic > 240:
                motivos.append(f"Metallic completamente blanco en metal (inusual, std={std_metallic:.2f})")
                score += 1

        # ── F8: Near-duplicate ──────────────────────────────────────────
        if archivo in duplicados:
            original = duplicados[archivo]
            motivos.append(f"Near-duplicate de {original} (pHash dist ≤ {PHASH_UMBRAL})")
            score += 2

        # ── F9: Mapas incompletos detectados en métricas ─────────────────
        if fila.get("ERROR", "") != "":
            motivos.append(f"Error en lectura: {fila['ERROR']}")
            score += 5

        if motivos:
            candidatos.append({
                "archivo":              archivo,
                "categoria":            categoria,
                "score":                score,
                "motivos":              " | ".join(motivos),
                "std_rgb":              round(std_rgb, 2),
                "std_normal":           round(std_normal, 2),
                "media_azul":           round(media_azul, 2),
                "cociente_rg":          round(cociente_rg, 3),
                "desviacion_norma":     round(desviacion_norma, 3),
                "std_rough":            round(std_rough, 2),
                "media_rough":          round(media_rough, 2),
                "confirmar_descarte":   "si" if score >= 3 else "revisar",
                # Score ≥ 3 → descarte automático recomendado
                # Score < 3 → requiere revisión visual
            })

    candidatos.sort(key=lambda x: -x["score"])

    df_cand = pd.DataFrame(candidatos)
    df_cand.to_csv(CSV_CANDIDATOS, index=False, encoding="utf-8")

    n_auto    = sum(1 for c in candidatos if c["confirmar_descarte"] == "si")
    n_revisar = sum(1 for c in candidatos if c["confirmar_descarte"] == "revisar")
    print(f"   Candidatos totales     : {len(candidatos)}")
    print(f"   Descarte automático    : {n_auto}  (score ≥ 3)")
    print(f"   Requieren revisión     : {n_revisar}  (score < 3)")
    print(f"   CSV generado           : {CSV_CANDIDATOS}")

    return candidatos


# ══════════════════════════════════════════════════════════════════════
# FASE 4 — GRÁFICOS DE DISTRIBUCIÓN
# ══════════════════════════════════════════════════════════════════════

PALETA_CATEGORIAS = {
    "stone": "#8B7355", "concrete": "#9E9E9E", "plaster": "#D2B48C",
    "marble": "#B0D4E8", "terracotta": "#CD5C5C",
    "wood": "#8B4513", "metal": "#708090", "ceramic": "#E8C4A0", "ground": "#6B8E23",
}

def generar_graficos(df):
    print("\n" + "="*60)
    print("  FASE 4 — Generación de gráficos")
    print("="*60)
    sns.set_theme(style="whitegrid", palette="muted")
    df_ok = df[df["ERROR"] == ""].copy()

    # ── G1: Distribución de texturas por categoría ──────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    counts = df_ok["categoria"].value_counts().sort_values(ascending=False)
    colores = [PALETA_CATEGORIAS.get(c, "#AAAAAA") for c in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colores, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Distribución de texturas por categoría (antes del EDA)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Número de texturas")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, "g1_distribucion_categorias.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"   ✅ {os.path.basename(ruta)}")

    # ── G2: RGB std vs Normal std (scatter por categoría) ───────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for cat, grupo in df_ok.groupby("categoria"):
        ax.scatter(grupo["std_rgb"], grupo["std_normal"],
                   label=cat, alpha=0.5, s=15,
                   color=PALETA_CATEGORIAS.get(cat, "#AAAAAA"))
    ax.axvline(x=5,  color="red",    linestyle="--", linewidth=1.2, label="RGB muerto (< 5)")
    ax.axhline(y=80, color="orange", linestyle="--", linewidth=1.2, label="Normal ruidoso (> 80)")
    ax.axhline(y=50, color="gold",   linestyle=":",  linewidth=1.0, label="Relieve fuerte (> 50)")
    ax.set_title("std_rgb vs std_normal — Coherencia albedo/normal", fontsize=12, fontweight="bold")
    ax.set_xlabel("Desviación estándar RGB (variedad de color)")
    ax.set_ylabel("Desviación estándar Normal (variedad geométrica)")
    ax.legend(markerscale=2, fontsize=8, loc="upper right")
    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, "g2_rgb_vs_normal_std.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"   ✅ {os.path.basename(ruta)}")

    # ── G3: Distribución del canal Z del normal (por categoría) ─────
    fig, ax = plt.subplots(figsize=(11, 5))
    for cat, grupo in df_ok.groupby("categoria"):
        sns.kdeplot(grupo["media_azul"], ax=ax, label=cat, fill=False,
                    color=PALETA_CATEGORIAS.get(cat, "#AAAAAA"), linewidth=1.5)
    ax.axvline(x=160, color="red", linestyle="--", linewidth=1.5, label="Umbral corrupción (< 160)")
    ax.set_title("Distribución del canal Z del normal map (por categoría)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Media del canal Azul (Z del normal OpenGL)")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=8)
    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, "g3_canal_azul_normal.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"   ✅ {os.path.basename(ruta)}")

    # ── G4: Roughness media vs std (scatter por categoría) ──────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for cat, grupo in df_ok.groupby("categoria"):
        ax.scatter(grupo["media_rough"], grupo["std_rough"],
                   label=cat, alpha=0.5, s=15,
                   color=PALETA_CATEGORIAS.get(cat, "#AAAAAA"))
    ax.axhline(y=2.5, color="orange", linestyle="--", linewidth=1.2, label="Roughness plano (std < 2.5)")
    ax.axvline(x=20,  color="red",    linestyle="--", linewidth=1.2, label="Roughness oscuro límite (media < 20)")
    ax.set_title("Roughness: media vs desviación estándar", fontsize=12, fontweight="bold")
    ax.set_xlabel("Media del roughness (0=espejo, 255=mate)")
    ax.set_ylabel("Desviación estándar del roughness (detalle)")
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, "g4_roughness_media_vs_std.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"   ✅ {os.path.basename(ruta)}")

    # ── G5: Boxplot de std_rgb por categoría ─────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    orden = sorted(df_ok["categoria"].unique())
    colores_box = [PALETA_CATEGORIAS.get(c, "#AAAAAA") for c in orden]
    bp = ax.boxplot([df_ok[df_ok["categoria"]==c]["std_rgb"].values for c in orden],
                    labels=orden, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colores_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(y=5, color="red", linestyle="--", linewidth=1.2, label="Albedo muerto (< 5)")
    ax.set_title("Distribución de variedad de color (std_rgb) por categoría", fontsize=12, fontweight="bold")
    ax.set_xlabel("Categoría")
    ax.set_ylabel("std_rgb")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, "g5_boxplot_std_rgb.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"   ✅ {os.path.basename(ruta)}")

    # ── G6: Metallic media y std (solo categoría metal) ─────────────
    df_metal = df_ok[df_ok["categoria"] == "metal"].dropna(subset=["media_metallic"])
    if not df_metal.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df_metal["media_metallic"], bins=30, kde=True,
                     color="#708090", ax=axes[0])
        axes[0].axvline(x=240, color="orange", linestyle="--", label="Todo blanco (> 240)")
        axes[0].axvline(x=15,  color="red",    linestyle="--", label="Todo negro (< 15)")
        axes[0].set_title("Distribución media_metallic (categoría metal)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Media del mapa metallic")
        axes[0].legend()

        sns.histplot(df_metal["std_metallic"], bins=30, kde=True,
                     color="#4682B4", ax=axes[1])
        axes[1].axvline(x=2, color="red", linestyle="--", label="Plano (std < 2)")
        axes[1].set_title("Distribución std_metallic (categoría metal)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Std del mapa metallic (variedad)")
        axes[1].legend()

        plt.tight_layout()
        ruta = os.path.join(DIR_GRAFICOS, "g6_metallic_distribucion.png")
        plt.savefig(ruta, dpi=150)
        plt.close()
        print(f"   ✅ {os.path.basename(ruta)}")

    # ── G7: Cociente R/G del normal (indicador de convención) ────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for cat, grupo in df_ok.groupby("categoria"):
        # Limitamos el cociente para no disparar el eje con outliers extremos
        vals = grupo["cociente_rg"].clip(0, 3)
        sns.kdeplot(vals, ax=ax, label=cat, fill=False,
                    color=PALETA_CATEGORIAS.get(cat, "#AAAAAA"), linewidth=1.5)
    ax.axvline(x=0.70, color="red",  linestyle="--", linewidth=1.2, label="Límite inferior (0.70)")
    ax.axvline(x=1.40, color="red",  linestyle="--", linewidth=1.2, label="Límite superior (1.40)")
    ax.axvline(x=1.00, color="green",linestyle=":",  linewidth=1.0, label="Ideal (1.0)")
    ax.set_xlim(0, 3)
    ax.set_title("Cociente R/G del normal map (indicador de convención/corrupción)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Ratio media_rojo / media_verde")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=8)
    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, "g7_cociente_rg_normal.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"   ✅ {os.path.basename(ruta)}")

    # ── G8: Desviación de la norma de los vectores normales ──────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for cat, grupo in df_ok.groupby("categoria"):
        sns.kdeplot(grupo["desviacion_norma"], ax=ax, label=cat, fill=False,
                    color=PALETA_CATEGORIAS.get(cat, "#AAAAAA"), linewidth=1.5)
    ax.axvline(x=0.30, color="red", linestyle="--", linewidth=1.5,
               label="Umbral sospechoso (> 0.30)")
    ax.set_title("Desviación media de la norma del vector normal (debe ≈ 1.0)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Desviación media de |N| respecto a 1.0")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=8)
    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, "g8_desviacion_norma_vectores.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"   ✅ {os.path.basename(ruta)}")

    print(f"\n   Todos los gráficos guardados en: {DIR_GRAFICOS}")


# ══════════════════════════════════════════════════════════════════════
# FASE 5 — INFORME HTML DE REVISIÓN HUMANA
# ══════════════════════════════════════════════════════════════════════

def generar_html(candidatos, df):
    """
    Genera un archivo HTML navegable con thumbnails de los 3 mapas
    por cada textura candidata a descarte, ordenados por score.
    """
    print("\n" + "="*60)
    print("  FASE 5 — Generación del informe HTML de revisión")
    print("="*60)

    # Generamos thumbnails a 128px para el HTML (muy ligeros)
    THUMB_SIZE = 128
    DIR_THUMBS = os.path.join(DIR_OUT, "thumbs")
    os.makedirs(DIR_THUMBS, exist_ok=True)

    def make_thumb(ruta_src, nombre_dest):
        if not os.path.exists(ruta_src):
            return False
        img = leer_imagen_robusta(ruta_src, cv2.IMREAD_COLOR)
        if img is None:
            return False
        img_small = cv2.resize(img, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
        ruta_dest = os.path.join(DIR_THUMBS, nombre_dest)
        cv2.imwrite(ruta_dest, img_small)
        return True

    print(f"   Generando {len(candidatos)*3} thumbnails...")
    for c in tqdm(candidatos, desc="Thumbnails", unit="tex"):
        arch = c["archivo"]
        base = arch.replace(".png", "")
        make_thumb(os.path.join(DIR_RGB,    arch), f"{base}_rgb.jpg")
        make_thumb(os.path.join(DIR_NORMAL, arch), f"{base}_norm.jpg")
        make_thumb(os.path.join(DIR_ROUGH,  arch), f"{base}_rough.jpg")

    # Estadísticas finales
    total    = len(df[df["ERROR"] == ""])
    n_auto   = sum(1 for c in candidatos if c["confirmar_descarte"] == "si")
    n_rev    = sum(1 for c in candidatos if c["confirmar_descarte"] == "revisar")
    n_ok     = total - len(candidatos)

    # Colores para score
    def score_color(s):
        if s >= 5: return "#FF4444"
        if s >= 3: return "#FF8C00"
        return "#FFD700"

    # Construir HTML
    filas_html = []
    for c in candidatos:
        arch  = c["archivo"]
        base  = arch.replace(".png", "")
        score = c["score"]
        color = score_color(score)

        thumb_rgb   = f"thumbs/{base}_rgb.jpg"
        thumb_norm  = f"thumbs/{base}_norm.jpg"
        thumb_rough = f"thumbs/{base}_rough.jpg"

        estado_badge = (
            f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-weight:bold;">{"AUTO-DESCARTAR" if score >= 3 else "REVISAR"}'
            f' (score={score})</span>'
        )

        motivos_fmt = "<br>".join(
            f'<span style="color:#555">• {m}</span>'
            for m in c["motivos"].split(" | ")
        )

        img_cell = lambda src, label: (
            f'<td style="text-align:center;padding:4px">'
            f'<div style="font-size:10px;color:#666;margin-bottom:2px">{label}</div>'
            f'<img src="{src}" width="{THUMB_SIZE}" height="{THUMB_SIZE}" '
            f'style="border-radius:4px;border:1px solid #ddd" '
            f'onerror="this.style.display=\'none\'">'
            f'</td>'
        )

        filas_html.append(f"""
        <tr style="border-bottom:1px solid #eee">
          <td style="padding:8px;font-family:monospace;font-size:12px">
            <b>{arch}</b><br>
            <span style="color:#888">{c['categoria']}</span>
          </td>
          <td style="padding:8px">{estado_badge}</td>
          {img_cell(thumb_rgb,   'RGB')}
          {img_cell(thumb_norm,  'Normal')}
          {img_cell(thumb_rough, 'Roughness')}
          <td style="padding:8px;font-size:11px;max-width:320px">{motivos_fmt}</td>
          <td style="padding:8px;font-size:11px;color:#777">
            std_rgb={c['std_rgb']}<br>
            std_norm={c['std_normal']}<br>
            azul={c['media_azul']}<br>
            R/G={c['cociente_rg']}<br>
            std_rough={c['std_rough']}<br>
            μ_rough={c['media_rough']}
          </td>
        </tr>""")

    # Enlazar gráficos
    graficos_disponibles = sorted(os.listdir(DIR_GRAFICOS)) if os.path.exists(DIR_GRAFICOS) else []
    graf_html = "".join(
        f'<div style="display:inline-block;margin:8px;text-align:center">'
        f'<img src="graficos/{g}" style="max-width:480px;border-radius:6px;border:1px solid #ddd">'
        f'<br><span style="font-size:11px;color:#666">{g}</span></div>'
        for g in graficos_disponibles if g.endswith(".png")
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>MatForge EDA — Informe de Revisión</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f9f9f9; color: #333; }}
    h1   {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 8px; }}
    h2   {{ color: #34495e; margin-top: 40px; }}
    .stat-box {{ display: inline-block; background: white; border-radius: 8px; padding: 15px 25px;
                 margin: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: center; }}
    .stat-num {{ font-size: 2em; font-weight: bold; }}
    table {{ width: 100%; border-collapse: collapse; background: white;
             border-radius: 8px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
    th {{ background: #2c3e50; color: white; padding: 10px; text-align: left; font-size: 13px; }}
    tr:hover {{ background: #f0f8ff; }}
    .instruction {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 18px;
                    border-radius: 4px; margin: 16px 0; }}
  </style>
</head>
<body>
  <h1>🔍 MatForge EDA — Informe de Revisión Humana</h1>

  <div class="instruction">
    <b>Instrucciones:</b> Las texturas marcadas en <span style="color:#FF4444">rojo (AUTO-DESCARTAR)</span>
    tienen score ≥ 3 y serán descartadas automáticamente en la siguiente ejecución.<br>
    Las marcadas en <span style="color:#FFD700;background:#555;padding:0 3px">amarillo (REVISAR)</span>
    requieren tu confirmación. Abre <code>candidates_to_discard.csv</code>, cambia "revisar" a
    <b>"si"</b> para descartar o <b>"no"</b> para conservar, y ejecuta con <code>MODO = "aplicar_descarte"</code>.
  </div>

  <h2>📊 Estadísticas del dataset</h2>
  <div>
    <div class="stat-box"><div class="stat-num" style="color:#27ae60">{n_ok}</div>texturas limpias</div>
    <div class="stat-box"><div class="stat-num" style="color:#FF4444">{n_auto}</div>auto-descartar</div>
    <div class="stat-box"><div class="stat-num" style="color:#FF8C00">{n_rev}</div>revisar</div>
    <div class="stat-box"><div class="stat-num" style="color:#333">{total}</div>total analizadas</div>
  </div>

  <h2>📈 Gráficos de distribución</h2>
  <div style="overflow-x:auto">{graf_html}</div>

  <h2>⚠️ Candidatos a descarte ({len(candidatos)} texturas)</h2>
  <table>
    <thead>
      <tr>
        <th>Archivo</th><th>Estado</th>
        <th>RGB</th><th>Normal</th><th>Roughness</th>
        <th>Motivos detectados</th><th>Métricas clave</th>
      </tr>
    </thead>
    <tbody>
      {''.join(filas_html)}
    </tbody>
  </table>
</body>
</html>"""

    with open(HTML_REVISION, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"   ✅ Informe HTML: {HTML_REVISION}")
    print(f"   ✅ Thumbnails  : {DIR_THUMBS}")


# ══════════════════════════════════════════════════════════════════════
# FASE 6 — RESUMEN ESTADÍSTICO FINAL
# ══════════════════════════════════════════════════════════════════════

def resumen_final(df, candidatos):
    print("\n" + "="*60)
    print("  RESUMEN FINAL")
    print("="*60)
    df_ok = df[df["ERROR"] == ""]
    total = len(df_ok)
    n_auto = sum(1 for c in candidatos if c["confirmar_descarte"] == "si")
    n_rev  = sum(1 for c in candidatos if c["confirmar_descarte"] == "revisar")

    print(f"\n  Total texturas analizadas : {total}")
    print(f"  Auto-descartar (score≥3)  : {n_auto}")
    print(f"  Revisar (score<3)         : {n_rev}")
    print(f"  Texturas limpias estimadas: {total - n_auto} (sin contar revisiones)")

    print("\n  Por categoría:")
    for cat in sorted(df_ok["categoria"].unique()):
        n_cat  = len(df_ok[df_ok["categoria"] == cat])
        n_flag = sum(1 for c in candidatos if c["categoria"] == cat)
        print(f"    {cat:15s}: {n_cat:4d} total | {n_flag:3d} flaggeadas")

    print(f"\n  📁 Archivos generados en: {DIR_OUT}")
    print(f"     • metricas_completas.csv   — todas las métricas")
    print(f"     • candidates_to_discard.csv — candidatos para revisar/confirmar")
    print(f"     • revision_humana.html      — informe visual navegable")
    print(f"     • graficos/                 — 8 gráficos de distribución")
    print(f"\n  ⏩ SIGUIENTE PASO:")
    print(f"     1. Abre revision_humana.html en tu navegador.")
    print(f"     2. Revisa los casos marcados como 'revisar' en el CSV.")
    print(f"     3. Pon MODO = 'aplicar_descarte' y vuelve a ejecutar.")
    print("="*60)


# ══════════════════════════════════════════════════════════════════════
# MODO: APLICAR DESCARTE
# ══════════════════════════════════════════════════════════════════════

def aplicar_descarte_confirmado():
    print("\n" + "="*60)
    print("  MODO: Aplicar descarte confirmado")
    print("="*60)

    if not os.path.exists(CSV_CANDIDATOS):
        print(f"❌ No se encontró {CSV_CANDIDATOS}. Ejecuta primero en modo 'analizar'.")
        return

    df_cand = pd.read_csv(CSV_CANDIDATOS, encoding="utf-8")
    a_descartar = df_cand[df_cand["confirmar_descarte"].astype(str).str.lower().isin(["si", "sí", "revisar"])]

    movidos = 0
    for _, fila in a_descartar.iterrows():
        archivo = fila["archivo"]
        for subcarpeta, dir_origen in [("rgb", DIR_RGB), ("normal", DIR_NORMAL),
                                        ("roughness", DIR_ROUGH), ("metallic", DIR_METALLIC)]:
            ruta_src = os.path.join(dir_origen, archivo)
            ruta_dst = os.path.join(DIR_DESCARTES, subcarpeta, archivo)
            if os.path.exists(ruta_src):
                shutil.move(ruta_src, ruta_dst)
        movidos += 1

    print(f"\n✅ Archivos movidos a descartados/: {movidos}")

    # Conteo final del dataset limpio
    print("\n📊 Dataset tras la limpieza:")
    for cat in sorted(set(os.path.basename(f).rsplit("_", 1)[0]
                          for f in os.listdir(DIR_RGB) if f.endswith(".png"))):
        n = len([f for f in os.listdir(DIR_RGB)
                 if f.endswith(".png") and f.rsplit("_", 1)[0] == cat])
        print(f"   {cat:15s}: {n:4d}")

    total_final = len([f for f in os.listdir(DIR_RGB) if f.endswith(".png")])
    print(f"\n   TOTAL FINAL: {total_final} texturas listas para el relabeling")
    print("="*60)


# ══════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════

def main():
    setup_directorios()

    if MODO == "analizar":
        df         = extraer_metricas()
        duplicados = detectar_duplicados(df)
        candidatos = aplicar_filtros(df, duplicados)
        generar_graficos(df)
        generar_html(candidatos, df)
        resumen_final(df, candidatos)

    elif MODO == "aplicar_descarte":
        aplicar_descarte_confirmado()

    else:
        print(f"❌ MODO desconocido: '{MODO}'. Usa 'analizar' o 'aplicar_descarte'.")


if __name__ == "__main__":
    main()
