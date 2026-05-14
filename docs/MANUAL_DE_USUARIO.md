# MatForge — Manual de Usuario

**Versión**: 1.0 | **Aplicación**: MatForge App | **Última actualización**: mayo 2026

---

## Índice

1. [Descripción general](#1-descripción-general)
2. [Distribución de la interfaz](#2-distribución-de-la-interfaz)
3. [Cargar una imagen](#3-cargar-una-imagen)
4. [Corrección de perspectiva](#4-corrección-de-perspectiva)
5. [Generar mapas PBR](#5-generar-mapas-pbr)
6. [Inspeccionar los resultados](#6-inspeccionar-los-resultados)
7. [Herramientas del sidebar](#7-herramientas-del-sidebar)
   - 7.1 [Ajuste de Roughness / Metallic](#71-ajuste-de-roughness--metallic)
   - 7.2 [Calidad del Normal Map](#72-calidad-del-normal-map)
   - 7.3 [Calibración por grupo](#73-calibración-por-grupo)
   - 7.4 [Make Tileable](#74-make-tileable)
8. [Herramientas del área principal](#8-herramientas-del-área-principal)
   - 8.1 [Material Blender](#81-material-blender)
   - 8.2 [Variaciones procedurales](#82-variaciones-procedurales)
   - 8.3 [Visor 3D](#83-visor-3d)
   - 8.4 [Comparación](#84-comparación)
   - 8.5 [Tiling Preview](#85-tiling-preview)
   - 8.6 [Batch ZIP](#86-batch-zip)
9. [Exportar mapas](#9-exportar-mapas)
10. [Referencia de estados de mapa](#10-referencia-de-estados-de-mapa)
11. [Guía de rendimiento](#11-guía-de-rendimiento)
12. [Limitaciones conocidas](#12-limitaciones-conocidas)

---

## 1. Descripción general

MatForge predice tres mapas PBR a partir de una única fotografía RGB:

- **Normal map** — codifica la microgeometría de la superficie como vectores direccionales. Los motores de renderizado lo usan para simular detalle de iluminación sin geometría adicional.
- **Roughness map** — controla el aspecto difuso o especular de una superficie. Valores cercanos a 0 producen reflejos de tipo espejo; valores cercanos a 1 producen superficies completamente difusas.
- **Metallic map** — distingue superficies metálicas de no metálicas. La mayoría de materiales son completamente metálicos (1.0) o completamente no metálicos (0.0).

La aplicación procesa una imagen a la vez a través de un pipeline configurable: corrección de perspectiva opcional → zoom → Super-Resolución opcional → clasificación de material → inferencia PBR. Las herramientas de postproceso (ajuste, calibración, mezcla, tileabilidad, variaciones) se aplican de forma no destructiva sobre los mapas generados.

---

## 2. Distribución de la interfaz

La interfaz se divide en dos áreas:

**Sidebar (izquierda)** — carga de imagen, configuración del pipeline (zoom, Super-Resolución) y herramientas de postproceso por mapa (Adjust R/M, Normal Map Quality, Calibrate by Group, Make Tileable).

**Área principal (derecha)** — pestañas de mapas, inspector de imagen de entrada y todas las herramientas interactivas (Perspective Correction, Material Blender, Procedural Variations, Visor 3D, Comparison, Tiling Preview, Batch ZIP, Export).

![Sidebar_1](docs/assets/sidebar_1.png)

![Sidebar_2](docs/assets/sidebar_2.png)

---

## 3. Cargar una imagen

Usa el uploader **Upload image** del sidebar. Formatos aceptados: JPG, PNG, WEBP.

**Slider de zoom** — escala la imagen antes de la inferencia. Úsalo para controlar la resolución efectiva que recibe MatForge:

| Tamaño de entrada | Zoom recomendado |
|---|---|
| Hasta 1K (1024 px) | 1.0 |
| Alrededor de 2K (2048 px) | 0.5 |
| Alrededor de 4K (4096 px) | 0.25 |

Reducir el zoom disminuye significativamente el tiempo de procesado. El sidebar muestra la resolución efectiva tras el zoom y un tiempo de procesado estimado antes de la generación.

**Super-Resolución (SR)** — cuando está activa, Real-ESRGAN escala la imagen con zoom aplicado ×4 antes de pasarla a MatForge. Mejora los resultados en entradas de baja resolución pero añade aproximadamente 9 segundos de procesado y aumenta significativamente la resolución de salida. No se recomienda para entradas ya superiores a 512 px tras el zoom, ya que puede producir resultados planos o incoherentes por encima de la resolución con la que MatForge fue entrenado.

> La barra de título muestra **[CUDA]** o **[CPU]** para indicar el dispositivo activo. El procesado en CPU es compatible pero sustancialmente más lento.

---

## 4. Corrección de perspectiva

La herramienta **Perspective Correction** está disponible en cuanto se carga una imagen, antes de generar los mapas.

![Corrección de perspectiva](docs/assets/perspective_correction.png)

1. Abre el expander **Perspective Correction** en el área principal.
2. Arrastra los cuatro handles ámbar hasta las esquinas de la superficie a corregir.
3. Haz clic en **↩ Update preview** para ver el resultado corregido y una previsualización 2×2.
4. Haz clic en **Apply** para confirmar. La imagen corregida pasa a ser la entrada del pipeline.
5. Haz clic en **↺ Reset** para descartar la corrección y volver a la imagen original.

La corrección se invalida automáticamente al cargar una nueva imagen.

---

## 5. Generar mapas PBR

Haz clic en **Generate Maps** en el sidebar tras configurar el zoom y las opciones de SR. El pipeline se ejecuta de forma secuencial:

1. Se aplica el zoom a la imagen cargada (o corregida en perspectiva).
2. Si SR está activo, Real-ESRGAN escala el resultado ×4.
3. El clasificador de material identifica el grupo de material y la puntuación de confianza.
4. MatForge ejecuta la inferencia tile-and-merge (tiles de 256×256, mezcla con ventana de Hann).
5. Los mapas Raw se almacenan y quedan disponibles para todas las herramientas de postproceso.

El sidebar muestra el grupo de material detectado y la distancia KNN tras la generación. Todas las herramientas de postproceso quedan disponibles una vez generados los mapas.

---

## 6. Inspeccionar los resultados

El área principal muestra tres pestañas tras la generación: **Normal**, **Roughness** y **Metallic**. Cada pestaña muestra el estado de mapa activo en ese momento (Raw, Adjusted, Calibrated, etc. — ver [Sección 10](#10-referencia-de-estados-de-mapa)).

El expander **Input Image** bajo las pestañas muestra:

- **Original** — dimensiones de la imagen cargada antes de cualquier procesado.
- **Pre-SR input** — resolución efectiva tras el zoom, antes de la Super-Resolución.
- **SR output** — resolución final pasada a MatForge (solo se muestra cuando SR estuvo activa).
- **Material group** y **KNN distance** — salida del clasificador.

---

## 7. Herramientas del sidebar

### 7.1 Ajuste de Roughness / Metallic

Aplica correcciones de ganancia y offset a los canales Roughness y Metallic de los mapas Raw. Los resultados se almacenan como estado **Adjusted**.

- **Gain** — multiplica los valores del canal. Valores superiores a 1.0 intensifican el efecto; valores inferiores lo reducen.
- **Offset** — añade un desplazamiento constante a todos los valores, limitado al rango [0, 1].

Haz clic en **Apply R/M Adjustment** para calcular y almacenar los mapas Adjusted.

### 7.2 Calidad del Normal Map

Evalúa el Normal map generado mediante tres métricas heurísticas:

- **Coherence** — fracción de píxeles con vectores de longitud aproximadamente unitaria.
- **Continuity** — suavidad del campo de normales; gradientes elevados pueden indicar artefactos de costura.
- **Blockiness** — detección de patrones de cuadrícula por bloques introducidos por la inferencia tile-and-merge.

Haz clic en **Evaluate quality** para ejecutar la evaluación. Se muestra un tiempo estimado antes de ejecutar — esta operación es intensiva en CPU y puede tardar varios minutos para mapas grandes.

Los resultados incluyen una puntuación por métrica, una puntuación global, advertencias contextualizadas según el grupo de material detectado y un mapa de calor diagnóstico disponible en la pestaña Normal.

### 7.3 Calibración por grupo

Aplica correcciones de ganancia y offset específicas del grupo de material detectado. Usa los mapas Raw como fuente de verdad y almacena los resultados como estado **Calibrated**.

El grupo detectado se muestra con su puntuación de confianza. Usa el selector **Override group** para aplicar las curvas de calibración de un grupo distinto si la detección automática no es correcta.

Haz clic en **Calibrate** para calcular y almacenar los mapas Calibrated.

### 7.4 Make Tileable

Aplica mezcla en dominio de frecuencias para reducir las costuras visibles al teselar los mapas. Opera sobre los mapas Calibrated si están disponibles, después Adjusted, después Raw. Los resultados se almacenan como estado **Tileable**.

La opción **SR seam blend** aplica mezcla adicional en los límites de tile introducidos por la Super-Resolución. Actívala si SR estuvo activa durante la generación.

Esta herramienta está marcada como **Beta** — los resultados dependen en gran medida del material de entrada. Verifica el resultado en el [Tiling Preview](#85-tiling-preview) antes de exportar.

---

## 8. Herramientas del área principal

### 8.1 Material Blender

Mezcla dos conjuntos de materiales PBR usando Reoriented Normal Mapping (RNM) para el canal Normal e interpolación lineal para Roughness y Metallic.

![Material Blender](docs/assets/material_blender.png)

1. Sube los mapas Normal B, Roughness B y Metallic B mediante los uploaders disponibles.
2. Opcionalmente sube un mapa Color B para mezcla de color.
3. Ajusta el slider **Blend factor** (0.0 = 100% material A, 1.0 = 100% material B).
4. Haz clic en **Blend Materials** para calcular y almacenar el estado **Blended**.

Si los mapas del material B tienen una resolución distinta a la de los mapas generados, se redimensionan automáticamente con un aviso.

### 8.2 Variaciones procedurales

Genera hasta tres variaciones del material base mediante técnicas basadas en ruido. Opera sobre los mapas Raw.

![Variaciones procedurales](docs/assets/procedural_variations.png)

- **Zonal Mix** — aplica ruido FBM para crear zonas de roughness espacialmente variables.
- **Worn Edges** — usa los gradientes del normal map para identificar bordes y aplica una máscara de desgaste sobre el roughness.
- **Scale Shift** — escala y reposiciona aleatoriamente la textura dentro de las dimensiones originales.

Si no hay Normal map disponible, solo se generan Zonal Mix y Scale Shift.

Ajusta **Number of variants** (1–3) y **Seed** antes de hacer clic en **Generate Variations**. Se muestra un tiempo estimado antes de ejecutar — esta operación es intensiva en CPU. Selecciona una variante en la cuadrícula de previsualizaciones y haz clic en **Apply Variation** para almacenarla como estado **Variation**.

### 8.3 Visor 3D

Visor Three.js en tiempo real para evaluar el material generado sobre una superficie 3D.

![Visor 3D](docs/assets/viewer_3d.png)

- **Geometry** — Sphere, Box o Plane.
- **Preview state** — selecciona qué estado de mapa mostrar (Raw, Adjusted, Calibrated, Blended, Tileable, Variation).
- **Show source color** — superpone la imagen de entrada como mapa de albedo sobre la geometría.
- **RoomEnvironment** — activa un mapa de entorno procedural para iluminación más realista.
- **Tiling** — aplica teselado UV ×2 para evaluar el material a una escala aparente menor.

Arrastra para rotar, desplaza la rueda del ratón para hacer zoom.

### 8.4 Comparación

Comparación lado a lado entre dos estados o canales de mapa mediante un slider. Arrastra el slider para revelar cada lado.

Usa los selectores **Left** y **Right** para elegir los estados y canales a comparar. El canal Color usa la imagen de entrada original.

### 8.5 Tiling Preview

Muestra un mosaico 2×2 del mapa seleccionado para evaluar la repetición sin costuras.

![Tiling Preview](docs/assets/tiling_preview.png)

Selecciona el **State** (Raw, Adjusted, Calibrated, Blended, Tileable, Variation) y el **Map** (Normal, Roughness, Metallic, Color) a previsualizar. Úsalo tras aplicar Make Tileable para verificar que las costuras no son visibles en los límites de tile.

### 8.6 Batch ZIP

Procesa múltiples imágenes en una sola ejecución. Sube un archivo ZIP con imágenes JPG, PNG o WEBP.

![Batch ZIP](docs/assets/batch_zip.png)

**Configuración:**
- **Engine** — formato de exportación aplicado a todas las imágenes del lote.
- **Zoom** — aplicado uniformemente a todas las imágenes.
- **Super-Resolution** — aplicada a cada imagen antes de la inferencia si está activa.

**Análisis previo** — antes del procesado, la herramienta analiza el contenido del ZIP y muestra:
- Número de imágenes válidas encontradas.
- Aviso si la resolución efectiva de alguna imagen supera 1024 px tras zoom/SR — los resultados por encima de la resolución de entrenamiento pueden ser planos o incoherentes.
- Aviso si SR está activa con más de 3 imágenes.
- Aviso si se detectan más de 10 imágenes.
- Tiempo total estimado con y sin SR.

Haz clic en **Process Batch** para iniciar. El progreso se muestra imagen por imagen. Las imágenes fallidas se omiten y se registran — el proceso nunca se interrumpe por un error individual. Al finalizar se muestra un resumen y un botón de descarga.

Estructura del ZIP de salida:
```
matforge_batch_{motor}.zip
├── nombre_imagen_1/
│   ├── (archivos de mapa específicos del motor)
├── nombre_imagen_2/
│   └── ...
```

---

## 9. Exportar mapas

La sección **Export** se encuentra en el área principal, bajo las pestañas de mapas.

![Exportación](docs/assets/export.png)

1. Introduce un **Asset name** (usado como nombre base de los archivos exportados).
2. Selecciona el **Engine** de destino.
3. Selecciona el **Map state** a exportar (Raw, Adjusted, Calibrated, Blended, Tileable o Variation).
4. Haz clic en **Export** para descargar un ZIP con todos los mapas para ese motor.

Todos los archivos PNG exportados incluyen metadatos XMP embebidos que los identifican como outputs generados por IA.

**Archivos de salida por motor:**

| Motor | Normal | Roughness/Metallic | Color |
|---|---|---|---|
| Blender | `_normal.png` | `_roughness.png`, `_metallic.png` | `_color.png` |
| Unreal Engine 5 | `T_name_N.png` | `T_name_ORM.png` | `T_name_D.png` |
| Unity URP | `_normal.png` | `_MetallicSmoothness.png` | `_Albedo.png` |
| Unity HDRP | `_normal.png` | `_MaskMap.png` | `_Albedo.png` |
| Godot 4 | `_normal.png` | `_orm.png` | `_albedo.png` |

---

## 10. Referencia de estados de mapa

Las herramientas de postproceso producen estados nombrados que se acumulan de forma no destructiva. Cada herramienta lee desde su fuente definida y escribe en su propio estado — los mapas Raw nunca se sobreescriben.

| Estado | Fuente | Producido por |
|---|---|---|
| Raw | — | Generate Maps |
| Adjusted | Raw | Adjust R/M |
| Calibrated | Raw | Calibrate by Group |
| Blended | Mapas activos | Material Blender |
| Tileable | Calibrated → Adjusted → Raw | Make Tileable |
| Variation | Raw | Procedural Variations |

Todas las herramientas con selector de estado (Visor 3D, Comparison, Tiling Preview, Export) permiten elegir qué estado usar de forma independiente.

---

## 11. Guía de rendimiento

| Operación | GTX 1650 Max-Q (4 GB VRAM) |
|---|---|
| Generate Maps — 512×512, zoom 1.0 | ~1 s |
| Generate Maps — 1024×1024, zoom 1.0 | ~6 s |
| Generate Maps — 1920×1920, zoom 0.5 | ~5 s |
| Sobrecarga de Super-Resolución (cualquier tamaño) | +~9 s |
| Normal Map Quality — 512×512 | ~7 s |
| Normal Map Quality — 1024×1024 | ~100 s |
| Variaciones procedurales — 352×352 | ~25 s |
| Variaciones procedurales — 640×640 | ~80 s |

La evaluación de Normal Map Quality y las Variaciones procedurales son operaciones intensivas en CPU. Los tiempos estimados se muestran en la interfaz antes de ejecutar cada herramienta.

---

## 12. Limitaciones conocidas

- **Resolución efectiva mínima**: imágenes inferiores a 256 px en cualquier dimensión tras el zoom producen un error de padding durante la inferencia. Usa valores de zoom que mantengan la resolución efectiva en 256 px o más.
- **Techo de resolución de entrenamiento**: MatForge fue entrenado con patches de hasta 1024 px. Los resultados a resoluciones efectivas superiores (por ejemplo, tras SR sobre una imagen grande) pueden ser planos o carecer de detalle superficial.
- **Tiempo de evaluación de Normal Map Quality**: la métrica de blockiness usa un cálculo de entropía con ventana deslizante que es lento en CPU. La evaluación de mapas superiores a 512×512 puede tardar más de un minuto.
- **Variaciones procedurales — Scale Shift**: puede aparecer un desplazamiento visible en entradas no tileables. El efecto es más natural en texturas tileables.
- **Make Tileable**: los resultados dependen del contenido de frecuencias de la entrada. Texturas muy detalladas o no repetitivas pueden mostrar costuras residuales.
- **Corrección de perspectiva**: la región de imagen corregida no se usa automáticamente como entrada para el zoom y la estimación de tiempo del sidebar — estos valores siguen haciendo referencia a la imagen original cargada.
- **Batch ZIP**: las imágenes con resolución efectiva inferior a 256 px tras el zoom se omiten con un error. Ajusta el zoom antes de procesar.
