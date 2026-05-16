#!/usr/bin/env python3
"""
MatForge Dataset Downloader v2.0
=================================
Script robusto para descargar texturas PBR de MatSynth (HuggingFace).

Características principales:
  - Reanudable: si se interrumpe, continúa donde lo dejó.
  - Anti-duplicados: nunca sobreescribe archivos ya descargados.
  - Tolerante a fallos: errores en un parquet no abortan la descarga.
  - Limpieza de caché automática: no acumula gigas en disco.
  - Compatible con datasets existentes: detecta lo que ya tienes.
  - Soporte para nuevas categorías (wood, metal, ceramic, ground).
  - Descarga el mapa Metallic (necesario para renders de metal correctos).

ANTES DE USAR — LEE ESTO:
  1. Edita ÚNICAMENTE la sección "CONFIGURACIÓN".
  2. Pon tu token de HuggingFace en HF_TOKEN.
  3. Apunta MATFORGE_DIR a donde quieres guardar (o ya tienes) el dataset.
  4. Si ya tienes texturas descargadas, ponlas dentro de MATFORGE_DIR/maps/
     (subcarpetas: rgb/, normal/, roughness/). El script las detectará.
  5. Si añades categorías nuevas a un dataset existente (wood, metal...),
     pon FORCE_RESCAN = True la PRIMERA vez que ejecutes con esas categorías.
     Después vuelve a ponerlo en False para que las reanudaciones sean rápidas.
  6. Ejecuta: python matforge_downloader.py
"""

import os
import json
import shutil
import glob
import time
import gc
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURACIÓN                                    ║
# ║              ← EDITA SOLO ESTA SECCIÓN ←                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Tu token de HuggingFace (consíguelo en https://huggingface.co/settings/tokens)
# ⚠️  NO compartas este archivo con el token visible. Bórralo antes de subir a GitHub.
HF_TOKEN = "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"

# Carpeta raíz del dataset (ruta absoluta recomendada para evitar confusiones)
# Ejemplo Windows: r"F:\MatSynth\matforge_dataset"
# Ejemplo Linux/Mac: "/home/usuario/datos/matforge_dataset"
MATFORGE_DIR = "F:\PI\matforge_dataset"

# ─── Categorías a descargar y sus límites ──────────────────────────────────
#
# MatSynth usa estos IDs de categoría:
#   ceramic=0, concrete=1, fabric=2, ground=3, leather=4,
#   marble=5,  metal=6,    misc=7,   plaster=8, plastic=9,
#   stone=10,  terracotta=11, wood=12
#
# "limite": cuántas texturas quieres en total de esa categoría.
#           Si ya tienes 278 de concrete y el límite es 300, solo descargará 22 más.
#           Pon 0 para NO descargar esa categoría en absoluto.
#
# "metallic": True = también descarga el mapa Metallic para esa categoría.
#             Solo necesario para la categoría "metal".
#
CATEGORIAS = {
    # ── Dominio pétreo (ya tienes parte descargada) ────────────────────────
    1:  {"nombre": "concrete",   "limite": 300,  "metallic": False},
    5:  {"nombre": "marble",     "limite": 150,  "metallic": False},
    8:  {"nombre": "plaster",    "limite": 300,  "metallic": False},
    10: {"nombre": "stone",      "limite": 800,  "metallic": False},
    11: {"nombre": "terracotta", "limite": 350,  "metallic": False},

    # ── Nuevos dominios (descarga fresca) ─────────────────────────────────
    12: {"nombre": "wood",       "limite": 600,  "metallic": False},
    6:  {"nombre": "metal",      "limite": 600,  "metallic": True},   # ← necesita metallic
    0:  {"nombre": "ceramic",    "limite": 600,  "metallic": False},
    3:  {"nombre": "ground",     "limite": 400,  "metallic": False},
}

# ─── ¿Forzar re-escaneo de parquets ya procesados? ─────────────────────────
#
# False (normal): Salta los parquets que ya procesaste antes. Reanuda rápido.
# True  (primera vez con categorías nuevas): Re-escanea todos los parquets
#       para encontrar wood/metal que antes ignorábamos.
#       Los archivos ya descargados NO se tocan (anti-duplicados activo).
#
# ⚠️  INSTRUCCIÓN IMPORTANTE:
#   Si acabas de añadir categorías nuevas (wood, metal...) a tu configuración,
#   pon FORCE_RESCAN = True, ejecuta el script, y cuando termine vuelve a
#   poner FORCE_RESCAN = False para futuras reanudaciones.
#
FORCE_RESCAN = False   # ← True la primera vez; False para reanudar

# Resolución de guardado. 1024 = 1K (recomendado para este proyecto)
RESOLUCION = (1024, 1024)

# ╚══════════════════════════════════════════════════════════════════════════╝
# FIN DE CONFIGURACIÓN — No toques nada de aquí para abajo salvo que sepas
# exactamente lo que haces.
# ╔══════════════════════════════════════════════════════════════════════════╝


# ─── Rutas internas ────────────────────────────────────────────────────────
MAPS_DIR      = os.path.join(MATFORGE_DIR, "maps")
META_DIR      = os.path.join(MATFORGE_DIR, "metadata")
TEMP_DIR      = os.path.join(MATFORGE_DIR, "_temp_parquet")
PROGRESS_FILE = os.path.join(META_DIR, "progress.json")
COUNTS_FILE   = os.path.join(META_DIR, "category_counts.json")

SUBCARPETAS_BASE = ["rgb", "normal", "roughness"]
TODAS_SUBCARPETAS = ["rgb", "normal", "roughness", "metallic"]


# ══════════════════════════════════════════════════════════════════════════
# FUNCIONES DE UTILIDAD
# ══════════════════════════════════════════════════════════════════════════

def setup_directorios():
    """Crea la estructura de carpetas del dataset si no existe."""
    for sub in TODAS_SUBCARPETAS:
        os.makedirs(os.path.join(MAPS_DIR, sub), exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)


def verificar_espacio_disco(gb_minimo=15):
    """
    Comprueba que hay espacio suficiente antes de empezar.
    Estimación: cada textura ocupa ~7-10 MB (3 mapas a 1K).
    Para ~800 texturas nuevas → ~6-8 GB.
    """
    directorio_check = MATFORGE_DIR if os.path.exists(MATFORGE_DIR) else "."
    total, usado, libre = shutil.disk_usage(directorio_check)
    libre_gb = libre / (1024 ** 3)
    total_gb = total / (1024 ** 3)

    print(f"💾 Espacio en disco: {libre_gb:.1f} GB libres de {total_gb:.1f} GB totales")

    if libre_gb < gb_minimo:
        print(f"\n⚠️  ADVERTENCIA: Solo {libre_gb:.1f} GB libres.")
        print(f"   Se recomiendan al menos {gb_minimo} GB para esta descarga.")
        respuesta = input("   ¿Continuar de todas formas? (s/n): ").strip().lower()
        if respuesta != 's':
            print("Descarga cancelada.")
            raise SystemExit(0)


def cargar_progreso():
    """Carga el JSON de progreso. Si no existe, devuelve estado vacío."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️  Archivo de progreso corrupto. Empezando progreso desde cero.")
    return {"parquets_procesados": [], "ultima_actualizacion": "nunca"}


def guardar_progreso(progreso: dict):
    """Persiste el estado de progreso en disco."""
    progreso["ultima_actualizacion"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progreso, f, indent=2, ensure_ascii=False)


def contar_existentes() -> tuple[dict, int]:
    """
    Cuenta las texturas ya descargadas buscando en maps/rgb/.
    La fuente de verdad es rgb/ porque es el mapa siempre presente.
    Devuelve (dict_contadores_por_categoria, total).
    """
    contadores = {info["nombre"]: 0 for info in CATEGORIAS.values()}
    rgb_dir = os.path.join(MAPS_DIR, "rgb")

    if not os.path.exists(rgb_dir):
        return contadores, 0

    for archivo in glob.glob(os.path.join(rgb_dir, "*.png")):
        nombre_base = os.path.basename(archivo)
        # Nombre formato: "{categoria}_{NNNN}.png"
        # rsplit con maxsplit=1 separa en la última "_"
        partes = nombre_base.rsplit('_', 1)
        if len(partes) == 2:
            categoria = partes[0]
            if categoria in contadores:
                contadores[categoria] += 1

    total = sum(contadores.values())
    return contadores, total


def calcular_proximos_indices(contadores: dict) -> dict:
    """
    Para cada categoría, determina el índice del siguiente archivo a guardar.
    Ejemplo: si ya existen stone_0000 a stone_0599, el próximo es 600.
    Usa los contadores en lugar de buscar en disco para mayor velocidad.
    """
    # Si el contador es correcto, el próximo índice es el contador actual
    # (asume nombres consecutivos desde 0000)
    return {info["nombre"]: contadores.get(info["nombre"], 0)
            for info in CATEGORIAS.values()}


def limpiar_temp():
    """Elimina la carpeta temporal y la vuelve a crear limpia."""
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
    except Exception:
        pass


def guardar_imagen(imagen_pil, carpeta_nombre: str, nombre_archivo: str, grayscale=False):
    """
    Guarda una imagen PIL en la subcarpeta correspondiente de maps/.
    - grayscale=True: convierte a L y luego a RGB (para roughness/metallic).
      Guardamos como RGB para uniformidad con el resto del dataset.
    """
    ruta = os.path.join(MAPS_DIR, carpeta_nombre, nombre_archivo)
    img_resized = imagen_pil.resize(RESOLUCION)
    if grayscale:
        img_resized = img_resized.convert("L").convert("RGB")
    else:
        img_resized = img_resized.convert("RGB")
    img_resized.save(ruta, format="PNG", optimize=False)


def imprimir_barra_progreso(nombre: str, actual: int, limite: int, ancho: int = 20):
    """Imprime una barra de progreso simple para una categoría."""
    if limite == 0:
        return ""
    porcentaje = min(actual / limite, 1.0)
    relleno = int(porcentaje * ancho)
    barra = "█" * relleno + "░" * (ancho - relleno)
    return f"{nombre:15s} [{barra}] {actual:4d}/{limite:4d}"


# ══════════════════════════════════════════════════════════════════════════
# PROGRAMA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("   MatForge Dataset Downloader v2.0")
    print("=" * 65)

    # ── 1. Validaciones previas ────────────────────────────────────────────
    if HF_TOKEN == "TU_TOKEN_AQUÍ":
        print("\n❌ ERROR: No has configurado tu token de HuggingFace.")
        print("   Edita la variable HF_TOKEN en la sección CONFIGURACIÓN.")
        return

    categorias_activas = {k: v for k, v in CATEGORIAS.items() if v["limite"] > 0}
    if not categorias_activas:
        print("\n❌ ERROR: Todas las categorías tienen límite 0. Nada que hacer.")
        return

    # ── 2. Preparar estructura de carpetas ────────────────────────────────
    print("\n📁 Preparando estructura de carpetas...")
    setup_directorios()
    print(f"   Raíz del dataset : {os.path.abspath(MATFORGE_DIR)}")
    print(f"   Mapas en         : {os.path.abspath(MAPS_DIR)}")

    # ── 3. Verificar espacio ───────────────────────────────────────────────
    print()
    verificar_espacio_disco(gb_minimo=15)

    # ── 4. Estado actual del dataset ──────────────────────────────────────
    print("\n📊 Estado actual del dataset:")
    contadores, total_existentes = contar_existentes()
    total_objetivo = sum(v["limite"] for v in categorias_activas.values())

    for cat_id, info in categorias_activas.items():
        print(f"   {imprimir_barra_progreso(info['nombre'], contadores.get(info['nombre'], 0), info['limite'])}")

    print(f"\n   ▶ Total actual : {total_existentes}")
    print(f"   ▶ Objetivo     : {total_objetivo}")
    print(f"   ▶ Pendientes   : {max(0, total_objetivo - total_existentes)}")

    if total_existentes >= total_objetivo:
        print("\n🎉 ¡El dataset ya está completo! No hay nada que descargar.")
        return

    # ── 5. Cargar progreso de parquets ────────────────────────────────────
    progreso = cargar_progreso()

    if FORCE_RESCAN:
        print("\n⚠️  FORCE_RESCAN=True activado.")
        print("   Se re-escanearán todos los parquets del repositorio.")
        print("   Los archivos ya descargados NO serán sobreescritos.")
        progreso["parquets_procesados"] = []
        guardar_progreso(progreso)

    parquets_ya_vistos = set(progreso.get("parquets_procesados", []))
    print(f"\n📂 Parquets ya procesados anteriormente: {len(parquets_ya_vistos)}")

    # ── 6. Obtener lista de parquets del repositorio ───────────────────────
    print("\n🌐 Conectando con HuggingFace para obtener lista de parquets...")
    from huggingface_hub import HfApi

    try:
        api = HfApi()
        todos_archivos = api.list_repo_files(
            repo_id="gvecchio/MatSynth",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        parquets = sorted([
            f for f in todos_archivos
            if f.startswith("data/train") and f.endswith(".parquet")
        ])
        print(f"✅ Encontrados {len(parquets)} parquets en el repositorio.\n")
    except Exception as e:
        print(f"\n❌ No se pudo conectar con HuggingFace: {e}")
        print("   Verifica tu token y tu conexión a internet.")
        return

    # ── 7. Bucle de descarga principal ────────────────────────────────────
    from datasets import load_dataset

    # Índices de escritura: próximo número de archivo por categoría
    indices = calcular_proximos_indices(contadores)

    n_total_parquets = len(parquets)
    parquets_pendientes = [p for p in parquets if p.split('/')[-1] not in parquets_ya_vistos]
    print(f"📦 Parquets a procesar en esta sesión: {len(parquets_pendientes)}\n")

    for i, parquet_file in enumerate(parquets_pendientes):
        nombre_parquet = parquet_file.split('/')[-1]

        # ── Comprobar si ya terminamos ─────────────────────────────────────
        hay_pendientes = any(
            contadores.get(info["nombre"], 0) < info["limite"]
            for info in categorias_activas.values()
        )
        if not hay_pendientes:
            print("\n🎉 Todas las categorías han alcanzado su límite. ¡Descarga completada!")
            break

        print(f"[{i+1}/{len(parquets_pendientes)}] ⏳ {nombre_parquet}")
        guardados_este_parquet = 0

        try:
            # ── 7a. Descargar el parquet ───────────────────────────────────
            # Controlamos el caché para que no se escape a ~/.cache/huggingface
            os.environ["HF_HOME"] = TEMP_DIR
            os.environ["HF_DATASETS_CACHE"] = TEMP_DIR
            os.environ["HUGGINGFACE_HUB_CACHE"] = TEMP_DIR

            ruta_local = hf_hub_download(
                repo_id="gvecchio/MatSynth",
                filename=parquet_file,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=TEMP_DIR,
                local_dir_use_symlinks=False,   # Archivos reales, no symlinks
            )

            # ── 7b. Cargar el parquet con datasets ─────────────────────────
            ds_local = load_dataset(
                "parquet",
                data_files=ruta_local,
                split="train",
                cache_dir=TEMP_DIR,
            )

            # ── 7c. Iterar sobre los ejemplos del parquet ─────────────────
            for ejemplo in ds_local:
                cat_id = ejemplo.get('category')

                # ¿Es una categoría que nos interesa?
                if cat_id not in categorias_activas:
                    continue

                info = categorias_activas[cat_id]
                nombre_cat = info["nombre"]
                limite_cat = info["limite"]
                actual_cat = contadores.get(nombre_cat, 0)

                # ¿Ya llegamos al límite de esta categoría?
                if actual_cat >= limite_cat:
                    continue

                # ¿Tenemos los mapas obligatorios?
                img_rgb = ejemplo.get('basecolor') or ejemplo.get('diffuse')
                img_normal = ejemplo.get('normal')
                img_roughness = ejemplo.get('roughness')

                if img_rgb is None or img_normal is None or img_roughness is None:
                    # Este material no tiene todos los mapas; lo ignoramos
                    continue

                # ¿El mapa normal tiene la forma correcta? (por si hay corrupción)
                # Un normal map válido tiene 3 canales
                try:
                    img_normal.convert("RGB")
                except Exception:
                    continue

                # ── Guardar los mapas ──────────────────────────────────────
                idx = indices[nombre_cat]
                nombre_archivo = f"{nombre_cat}_{idx:04d}.png"

                try:
                    guardar_imagen(img_rgb,       "rgb",       nombre_archivo, grayscale=False)
                    guardar_imagen(img_normal,    "normal",    nombre_archivo, grayscale=False)
                    guardar_imagen(img_roughness, "roughness", nombre_archivo, grayscale=True)

                    # Metallic: solo si la categoría lo requiere y el mapa existe
                    if info["metallic"]:
                        img_metallic = ejemplo.get('metallic')
                        if img_metallic is not None:
                            guardar_imagen(img_metallic, "metallic", nombre_archivo, grayscale=True)
                        # Si no existe el mapa metallic, guardamos uno negro (metallic=0)
                        # NOTA: para metal esto no debería pasar en MatSynth, pero por si acaso:
                        else:
                            from PIL import Image
                            negro = Image.new("L", RESOLUCION, 0)
                            ruta_fallback = os.path.join(MAPS_DIR, "metallic", nombre_archivo)
                            negro.convert("RGB").save(ruta_fallback, format="PNG")

                    # Actualizar contadores e índices
                    indices[nombre_cat] = idx + 1
                    contadores[nombre_cat] = actual_cat + 1
                    guardados_este_parquet += 1

                except Exception as e_guardado:
                    # Error al guardar: limpiamos archivos parciales de este ejemplo
                    print(f"   ⚠️  Error guardando {nombre_archivo}: {e_guardado}")
                    for sub in TODAS_SUBCARPETAS:
                        ruta_parcial = os.path.join(MAPS_DIR, sub, nombre_archivo)
                        if os.path.exists(ruta_parcial):
                            try:
                                os.remove(ruta_parcial)
                            except Exception:
                                pass
                    # Importante: NO incrementamos el índice para reutilizarlo

            # ── 7d. Resumen de este parquet ───────────────────────────────
            total_actual = sum(contadores.values())
            estado_cats = " | ".join(
                f"{info['nombre']}:{contadores.get(info['nombre'], 0)}"
                for info in categorias_activas.values()
                if contadores.get(info['nombre'], 0) > 0
            )
            print(f"   ✅ +{guardados_este_parquet} nuevos | Total: {total_actual}/{total_objetivo}")
            if guardados_este_parquet > 0:
                print(f"   📊 {estado_cats}")

        except KeyboardInterrupt:
            # El usuario ha pulsado Ctrl+C: guardamos progreso y salimos limpiamente
            print("\n\n⛔ Descarga interrumpida por el usuario (Ctrl+C).")
            print("   El progreso ha sido guardado. Puedes reanudar ejecutando el script de nuevo.")
            print("   ⚠️  Recuerda poner FORCE_RESCAN = False para reanudar correctamente.")
            guardar_progreso(progreso)
            return

        except MemoryError:
            # El parquet era demasiado grande para la RAM disponible
            print(f"   ❌ MemoryError: el parquet {nombre_parquet} es demasiado grande.")
            print(f"      Saltando este parquet. Cierra otras aplicaciones si se repite.")
            limpiar_temp()
            # NO marcamos como procesado para poder reintentarlo
            continue

        except Exception as e:
            # Cualquier otro error (timeout, corrupción, etc.)
            print(f"   ❌ Error procesando {nombre_parquet}: {type(e).__name__}: {e}")
            print(f"      Se reintentará en la próxima ejecución.")
            limpiar_temp()
            # NO marcamos como procesado para reintentarlo automáticamente
            continue

        finally:
            # Limpieza siempre, haya error o no
            try:
                del ds_local
            except Exception:
                pass
            gc.collect()
            limpiar_temp()

        # ── Marcar parquet como procesado y guardar progreso ──────────────
        # (Solo llegamos aquí si no hubo excepción grave)
        parquets_ya_vistos.add(nombre_parquet)
        progreso["parquets_procesados"] = list(parquets_ya_vistos)
        guardar_progreso(progreso)

    # ── 8. Resumen final ──────────────────────────────────────────────────
    contadores_finales, total_final = contar_existentes()

    print("\n" + "=" * 65)
    print("   DESCARGA FINALIZADA")
    print("=" * 65)
    print("\n📊 Resumen final:")
    for info in categorias_activas.values():
        nombre = info["nombre"]
        limite = info["limite"]
        actual = contadores_finales.get(nombre, 0)
        estado = "✅ Completo" if actual >= limite else f"⚠️  Incompleto ({limite - actual} pendientes)"
        print(f"   {imprimir_barra_progreso(nombre, actual, limite)}  {estado}")

    print(f"\n   ▶ TOTAL FINAL: {total_final} texturas")

    if total_final < total_objetivo:
        faltantes = total_objetivo - total_final
        print(f"\n⚠️  Faltan {faltantes} texturas para alcanzar el objetivo.")
        print(f"   Esto puede significar que MatSynth no tiene tantas texturas")
        print(f"   de ciertas categorías como el límite que configuraste.")
        print(f"   Es normal. El dataset sigue siendo válido.")

    # Guardar resumen en JSON para el EDA posterior
    with open(COUNTS_FILE, "w", encoding="utf-8") as f:
        json.dump(contadores_finales, f, indent=2)

    print(f"\n📝 Resumen guardado en : {os.path.abspath(COUNTS_FILE)}")
    print(f"📁 Dataset disponible  : {os.path.abspath(MATFORGE_DIR)}")
    print(f"\n{'=' * 65}\n")


# ══════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    main()
