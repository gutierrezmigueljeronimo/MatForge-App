# MatForge App — Documento de Arquitectura Permanente

**Versión**: 1.0 | **Creado**: 07/05/2026  
**Estado**: Referencia activa — no modificar sin justificación técnica documentada

---

## Índice

1. [Descripción general del proyecto](#1-descripción-general-del-proyecto)
2. [Estructura de carpetas](#2-estructura-de-carpetas)
3. [Descripción de módulos](#3-descripción-de-módulos)
4. [Flujo de datos](#4-flujo-de-datos)
5. [Sistema de diseño visual](#5-sistema-de-diseño-visual)
6. [Protocolo de gestión de VRAM](#6-protocolo-de-gestión-de-vram)
7. [Decisiones cerradas](#7-decisiones-cerradas)

---

## 1. Descripción general del proyecto

MatForge App es una aplicación Streamlit local que predice mapas de materiales PBR (Normal, Roughness, Metallic) a partir de una única imagen RGB de superficie. Se ejecuta en una sola máquina con una GTX 1650 Max-Q (4 GB VRAM) y está destinada al uso por parte de un artista 3D sin conocimientos de aprendizaje profundo.

La aplicación encadena cuatro componentes de modelo secuenciales: un módulo SR opcional (RRDBNet ×4), un clasificador de material (DINOv2-small + PCA-50 + KNN) y el predictor principal (MatForgeNet: encoder PVT-v2-B1 + decoder FPN + tres refine heads). El módulo SR y MatForgeNet nunca se ejecutan simultáneamente.

---

## 2. Estructura de carpetas

F:\PI\MatForge_App
│
├── app.py                        ← Punto de entrada de Streamlit
├── requirements.txt
├── .gitignore
├── .gitattributes                ← Reglas de seguimiento Git LFS
│
├── checkpoints
│   ├── matforge
│   │   └── best_gan.pt           ← Pesos finales de MatForgeNet (Git LFS)
│   └── sr
│       ├── sr_ft_phase1_best_lpips.pt  ← Checkpoint primario del módulo SR (Git LFS)
│       └── RealESRGAN_x4plus.pth       ← Checkpoint de respaldo del SR (Git LFS)
│
├── artifacts
│   ├── knn_classifier.pkl        ← Clasificador KNN de material
│   ├── pca_model.pkl             ← PCA 384D → 50D
│   └── label_encoder.pkl         ← Codificador de 8 clases
│
├── assets
│   ├── three\                    ← Archivos locales de Three.js (Herramienta 1)
│   │   ├── three.module.js
│   │   └── OrbitControls.js
│   └── env\                      ← Mapa de entorno PMREM mínimo (Herramienta 1)
│       └── env_minimal.hdr
│
├── scripts
│   └── matforge_app_00_inference_check.py  ← Solo diagnóstico, no forma parte de la app
│
└── src
├── models.py                 ← Clases MatForgeNet, FPNDecoder, RefineHead
├── inference.py              ← Pipeline tile-and-merge con ventana Hann
├── sr.py                     ← Cargador y pipeline del módulo SR
├── classifier.py             ← Pipeline DINOv2 + PCA + KNN
├── postprocess.py            ← Ajustes post-predicción (Herramientas 2, 5, 6, 7, 9)
├── export.py                 ← Lógica de exportación multi-motor (Herramienta 3)
├── quality.py                ← Evaluación de calidad del mapa de normales (Herramienta 10)
├── ui_components.py          ← Componentes UI reutilizables e inyección CSS
└── utils.py                  ← Utilidades compartidas (carga de imágenes, base64, logging)

---

## 3. Descripción de módulos

### app.py

Punto de entrada de Streamlit. Define el layout de alto nivel: sidebar para controles, área principal para resultados. Orquesta las llamadas a los módulos de `src/`. Gestiona las claves de `st.session_state`. No contiene lógica de inferencia directamente.

| Recibe | Devuelve | Efectos secundarios |
|---|---|---|
| — | — | Renderiza la interfaz Streamlit completa en cada rerun |

### src/models.py

Define `MatForgeNet`, `FPNDecoder` y `RefineHead`. Son las clases exactas cuyo state dict coincide con `best_gan.pt`. Sin código de entrenamiento ni funciones de pérdida — solo definiciones para inferencia.

| Recibe | Devuelve | Efectos secundarios |
|---|---|---|
| — | Subclases de `nn.Module` instanciadas | Ninguno |

### src/inference.py

Carga MatForgeNet con `@st.cache_resource`. Implementa tile-and-merge con ventana Hann (tile 256×256, stride 128). Aplica normalización ImageNet a la entrada. Post-procesa las salidas: renormalización L2 del mapa de normales, sigmoid sobre metallic, clip de todos los mapas a rangos válidos.

| Recibe | Devuelve | Efectos secundarios |
|---|---|---|
| `PIL.Image` (RGB) | `dict` con claves `normal`, `roughness`, `metallic` como `np.ndarray float32` | Asignación de memoria GPU; lee el checkpoint en la primera llamada |

### src/sr.py

Carga el modelo SR (RRDBNet ×4, 23 bloques RRDB) con `@st.cache_resource(max_entries=1)`. Implementa tile-and-merge para SR (tile 256×256, stride 128, ventana Hann). Ejecuta el protocolo de liberación de VRAM al finalizar. Intenta el checkpoint primario; si no está disponible, usa `RealESRGAN_x4plus.pth`.

| Recibe | Devuelve | Efectos secundarios |
|---|---|---|
| `PIL.Image` (RGB) | `PIL.Image` (RGB, 4× más grande) | Asignación GPU durante SR; liberación explícita de VRAM al completar |

### src/classifier.py

Carga DINOv2-small con `@st.cache_resource`. Carga los artefactos PCA y KNN con `joblib.load`. Extrae el token CLS (384D), aplica PCA (→ 50D) y predice la etiqueta de grupo y la distancia KNN. La distancia es utilizada por la Herramienta 7 para ponderar la confianza de la calibración.

| Recibe | Devuelve | Efectos secundarios |
|---|---|---|
| `PIL.Image` (RGB) | `tuple(str, float)`: etiqueta de grupo + distancia KNN | Asignación GPU para DINOv2 (CPU aceptable como fallback) |

### src/postprocess.py

Funciones puras en NumPy. Sin GPU, sin estado de Streamlit. Implementa: ajuste de gain/offset para R/M (Herramienta 2), mezcla RNM para mapas de normales (Herramienta 5), conversión a tileable (Herramienta 6), calibración de rango PBR por grupo (Herramienta 7), generación de variaciones procedurales (Herramienta 9).

| Recibe | Devuelve | Efectos secundarios |
|---|---|---|
| Arrays `np.ndarray` + parámetros | Arrays `np.ndarray` modificados | Ninguno |

### src/export.py

Convierte y empaqueta los mapas para Blender, UE5, Unity URP, Unity HDRP y Godot 4. Gestiona el empaquetado de canales (ORM), el flip del canal Y para la convención DirectX y la creación del ZIP. Devuelve un objeto `bytes` listo para `st.download_button`.

| Recibe | Devuelve | Efectos secundarios |
|---|---|---|
| `dict` de arrays `np.ndarray` + cadena de motor + nombre de asset | `bytes` (archivo ZIP) | Ninguno |

### src/quality.py

Evalúa la calidad del mapa de normales mediante heurísticas: desviación de la norma L2 por píxel respecto a 1.0, magnitud del gradiente Sobel, entropía local en ventanas de 16×16. Devuelve una puntuación y un array de heatmap para visualización.

| Recibe | Devuelve | Efectos secundarios |
|---|---|---|
| `np.ndarray` mapa de normales (H,W,3) float32 en [-1,1] | `dict`: puntuación (float), heatmap (np.ndarray RGBA) | Ninguno |

### src/ui_components.py

Inyecta todo el CSS mediante `st.markdown(unsafe_allow_html=True)` al cargar la app. Contiene funciones de componentes reutilizables: tarjetas de resultado, indicadores de estado, plantilla HTML del visor Three.js (Herramienta 1) y slider de comparación HTML. Todos los valores visuales constantes (hex, tamaños de fuente) residen aquí exclusivamente.

### src/utils.py

Utilidades compartidas: carga y validación de imágenes, conversión PIL/numpy/tensor, codificación base64 para embeds HTML, aplicación de zoom (resize LANCZOS), corrección de perspectiva (cv2), cálculo de resolución efectiva y helpers de session state.

---

## 4. Flujo de datos

[Usuario sube imagen]
│
▼
[utils.py: carga + validación]
│
▼
[utils.py: apply_zoom (LANCZOS)]
│
├──── SR activo? ────▶ [sr.py: tile-and-merge RRDBNet] ──┐
│                                                         │
│         No ─────────────────────────────────────────────┤
│                                                         │
▼                                                         ▼
[classifier.py: DINOv2 → PCA → KNN]              [Misma imagen, mayor resolución]
│                                                         │
│ etiqueta de grupo + distancia                           │
▼                                                         ▼
[inference.py: tile-and-merge MatForgeNet] ◀─────────────┘
│
│ normal (H,W,3) | roughness (H,W,1) | metallic (H,W,1)
▼
[session_state: almacenar predicciones brutas]
│
├──▶ [postprocess.py: gain/offset, calibración, mezcla, tileable, variaciones]
├──▶ [quality.py: evaluación del mapa de normales]
├──▶ [export.py: empaquetado por motor]
└──▶ [ui_components.py: visor Three.js, slider de comparación, tarjetas de resultado]

---

## 5. Sistema de diseño visual

> Todos los valores de esta sección son definitivos. Cualquier cambio requiere justificación explícita documentada en una nueva versión de este documento.

### 5.1 Justificación del diseño

La aplicación está dirigida a artistas 3D familiarizados con Blender, Substance Painter y Unreal Engine, que utilizan interfaces oscuras con tonos cálidos. El sistema visual sigue la misma lógica: grises oscuros cálidos como fondo, los mapas PBR como protagonistas visuales y la UI como marco neutro. La firma cromática azul-violácea del mapa de normales y la escala de grises del roughness se perciben correctamente sobre fondos oscuros cálidos sin interferencia cromática.

### 5.2 Paleta de colores

| Rol | Hex | Uso |
|---|---|---|
| Fondo primario | `#1C1B18` | Fondo de la app, sidebar |
| Fondo secundario | `#252420` | Tarjetas, paneles, expanders |
| Fondo terciario | `#2E2D29` | Campos de entrada, estados hover |
| Borde | `#3A3830` | Divisores, bordes de tarjeta |
| Acento primario | `#E8A835` | Botones de acción primaria, estados activos, progreso |
| Acento secundario | `#C4863A` | Botones secundarios, destacados, hover sobre acento |
| Texto primario | `#E8E6DF` | Encabezados, etiquetas, contenido principal |
| Texto secundario | `#9A9890` | Captions, textos de ayuda, metadatos |
| Estado: éxito | `#4A6741` | Indicadores PASS, pasos completados |
| Estado: advertencia | `#7A5A28` | Mensajes de precaución, tiempos estimados |
| Estado: error | `#7A3030` | Indicadores FAIL, errores de carga |
| Estado: info | `#2A4A6A` | Cajas informativas, consejos, etiquetas de grupo de material |

### 5.3 Tipografía

| Elemento | Tamaño | Peso | Uso |
|---|---|---|---|
| H1 | 28px | 500 | Título de la app únicamente |
| H2 | 20px | 500 | Encabezados de sección (Resultados, Herramientas) |
| H3 | 16px | 500 | Títulos de tarjeta, nombres de herramienta |
| Cuerpo | 14px | 400 | Contenido general |
| Caption | 12px | 400 | Textos de ayuda, metadatos, st.caption |
| Código | 13px | 400 | Rutas de archivo, valores técnicos |

Fuente: `Inter` cargada mediante inyección de Google Fonts en `ui_components.py`. Fallback: sans-serif del sistema.

### 5.4 Layout

- Ancho del sidebar: 320px (fijo). Contiene todos los controles, configuración y ajustes de herramientas.
- Área principal: ancho restante. Contiene la entrada de imagen, los resultados y el visor Three.js.
- Padding de tarjeta: 16px. Separación entre tarjetas: 12px.
- Los mapas de resultado se muestran en una cuadrícula de 3 columnas (Normal | Roughness | Metallic) con ancho igual.
- Ancho máximo de contenido: 1200px (centrado en pantallas anchas).

### 5.5 Especificación de componentes

| Componente | Especificación |
|---|---|
| Botón primario | Fondo `#E8A835`, texto `#1C1B18`, radio 6px, padding 8px 16px, font-weight 500 |
| Botón secundario | Fondo transparente, borde 1px `#3A3830`, texto `#E8E6DF`, mismo tamaño |
| Tarjeta de resultado | Fondo `#252420`, borde 1px `#3A3830`, radio 8px, padding 16px |
| Indicador de estado | Punto de color (8px) + texto. Colores de la paleta de estados. |
| Slider | Nativo de Streamlit; color acento `#E8A835` vía override CSS |
| Expander | Fondo `#252420`, borde izquierdo 2px `#E8A835` cuando está abierto |

---

## 6. Protocolo de gestión de VRAM

El módulo SR y MatForgeNet nunca ocupan la memoria GPU simultáneamente. La secuencia obligatoria de liberación tras completar el SR:

```python
model.to('cpu')           # mover tensores fuera del allocator CUDA
del model                 # eliminar la referencia Python
gc.collect()              # forzar ciclo de recolección de basura
torch.cuda.empty_cache()  # devolver bloques al driver
```

Ambos modelos usan `@st.cache_resource(max_entries=1)`. MatForgeNet se carga en la primera inferencia y permanece en caché durante toda la sesión. El módulo SR se carga, usa y libera explícitamente cada vez que se invoca.

Detección de dispositivo y dtype al arrancar:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
```

---

## 7. Decisiones cerradas

| Decisión | Valor |
|---|---|
| Versión mínima de Streamlit | 1.50 |
| Decorador de caché de modelos | `@st.cache_resource` — nunca `@st.cache` |
| SR y MatForgeNet simultáneos | No — estrictamente secuencial |
| Tamaño de tile | 256×256, stride 128 (50% de solapamiento) |
| Blending Hann para Normal | Acumular → dividir → renormalizar L2. Nunca renormalizar por tile antes de acumular. |
| Post-procesado de Metallic | Sigmoid aplicado tras el merge Hann, no dentro del forward del modelo |
| Normalización ImageNet | mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] — coincide con el preentrenamiento de PVT-v2-B1 |
| Despliegue de Three.js | Archivos locales en `assets/three/` — sin dependencia de CDN |
| Límite de textura en Three.js | Máximo 1024×1024 px para evitar sobrecarga del WebSocket |
| Convención de mapa de normales | OpenGL (+Y arriba). La exportación a UE5 aplica flip del canal Y. |
| Metallic en grupos no metálicos | La salida del clasificador KNN se usa solo para calibración — no se fuerza a cero. |
| Idioma del código | Inglés en todo el código fuente, comentarios y cadenas de UI |
| Estilo de comentarios | Impersonal y técnico — explica el porqué, no el qué. Sin referencias a modelos de IA. |
| Dirección del diseño visual | Oscuro cálido (Estilo B) — grises de fondo con sesgo cálido, acento ámbar |
| Versión de scikit-learn | 1.8.0 — coincide con la versión usada para serializar los artefactos KNN/PCA |