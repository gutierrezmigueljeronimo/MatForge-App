# Backlog SCRUM — MatForge App
## Proyecto Intermodular — Postgrado en Inteligencia Artificial y Big Data
### EUSA — Cámara de Comercio de Sevilla | Autor: Miguel Jerónimo Gutiérrez Barranco

---

## 1. Definition of Done

Una tarea o historia de usuario se considera **completada** cuando cumple todos los criterios siguientes:

1. **Funcional**: el código implementado produce el resultado esperado sobre al menos una imagen de material real (no sintética).
2. **Integrado**: el módulo o función está integrado en el pipeline de la aplicación y no rompe ninguna funcionalidad preexistente.
3. **Verificado**: existe al menos una evidencia de funcionamiento correcto (captura de pantalla, output de terminal, log de entrenamiento o resultado cuantitativo).
4. **Documentado**: la decisión técnica relevante (si existe) está registrada en la bitácora de desarrollo o en el documento de arquitectura permanente correspondiente.
5. **Versionado**: el artefacto ha sido commiteado al repositorio Git con un mensaje semántico (`feat`, `fix` o `chore`) o, en el caso de checkpoints de modelo, subido al dataset de Kaggle correspondiente.
6. **Sin deuda técnica bloqueante**: los bugs conocidos que impiden el uso normal de la funcionalidad están resueltos. Los bugs de mejora no bloqueantes se documentan en la bitácora como deuda técnica.

---

## 2. Product Backlog

Las historias de usuario se ordenan por prioridad de producto, de mayor a menor. La columna OE indica el objetivo específico al que responde cada historia.

| ID | Historia de usuario | OE | Prioridad |
|---|---|---|---|
| US-01 | Como artista 3D, quiero subir una fotografía de una superficie y obtener los mapas Normal, Roughness y Metallic en menos de 10 segundos en GPU, para integrarlos directamente en mi flujo de trabajo en Blender sin postproceso manual. | OE1, OE7 | Must |
| US-02 | Como artista 3D, quiero que la aplicación funcione completamente en local sin necesidad de conexión a internet durante la inferencia, para poder usarla con activos en desarrollo sin riesgo de exposición. | OE7 | Must |
| US-03 | Como artista 3D, quiero que la aplicación detecte automáticamente si tengo una GPU disponible y la use, o cambie a CPU de forma transparente, para no tener que configurar nada manualmente. | OE7 | Must |
| US-04 | Como artista 3D, quiero que el modelo prediga mapas PBR físicamente coherentes con un MAE angular de normales inferior a 11° y un LPIPS sobre render inferior a 0,10, para obtener resultados comparables o superiores al estado del arte accesible. | OE1 | Must |
| US-05 | Como artista 3D, quiero que la aplicación identifique automáticamente el tipo de material de mi imagen (piedra, madera, metal, cerámica…) sin que yo tenga que especificarlo, para que las calibraciones específicas se apliquen sin intervención manual. | OE3 | Must |
| US-06 | Como artista 3D, quiero poder exportar los mapas PBR generados en el formato correcto para Blender, Unreal Engine 5, Unity URP/HDRP o Godot 4 con un solo clic, para no tener que hacer conversiones manuales de convenciones de normal map o empaquetado de canales. | OE5 | Must |
| US-07 | Como artista 3D, quiero que cada PNG exportado incluya metadatos XMP que identifiquen el contenido como generado con asistencia de IA, para cumplir con las obligaciones de transparencia y poder rastrear el origen de los activos en mi pipeline de producción. | OE5 | Must |
| US-08 | Como artista 3D, quiero poder escalar con ×4 una imagen de baja resolución antes de que MatForge la procese, para mejorar la calidad de los mapas predichos cuando fotografío materiales con un dispositivo móvil o en condiciones de baja resolución. | OE2 | Should |
| US-09 | Como artista 3D, quiero que el módulo de super-resolución mejore la calidad perceptual (LPIPS) al menos un 10% respecto al modelo Real-ESRGAN base sin fine-tuning, para justificar el uso del modelo especializado frente al genérico. | OE2 | Should |
| US-10 | Como artista 3D, quiero poder ajustar los valores de Roughness y Metallic de los mapas predichos mediante sliders de gain y offset, para corregir desviaciones sistemáticas del modelo sin necesidad de editar los mapas en una aplicación externa. | OE4 | Should |
| US-11 | Como artista 3D, quiero poder corregir la perspectiva de una fotografía de material tomada en ángulo antes de procesarla, para que el modelo reciba una imagen ortogonal y produzca mapas geométricamente coherentes. | OE4 | Should |
| US-12 | Como artista 3D, quiero poder convertir los mapas PBR generados en texturas tileables (sin costuras), para usarlos como materiales de superficie repetibles en motores 3D sin artefactos visibles en los bordes. | OE4 | Should |
| US-13 | Como artista 3D, quiero poder mezclar los mapas PBR de dos materiales distintos mediante la fórmula de mezcla RNM (Reoriented Normal Mapping), para crear variaciones de material compuesto sin necesidad de un software de edición de texturas dedicado. | OE4 | Should |
| US-14 | Como artista 3D, quiero poder generar variaciones procedurales del mapa de Roughness mediante ruido FBM (Worn Edges, Zonal Mix, Scale Shift), para obtener versiones envejecidas o desgastadas del material sin assets adicionales. | OE4 | Should |
| US-15 | Como artista 3D, quiero poder visualizar el material PBR completo (Normal + Roughness + Metallic) en un visor 3D interactivo dentro de la propia aplicación, con iluminación PBR real y posibilidad de orbitar la cámara, para evaluar el resultado sin necesidad de importarlo en Blender o Unreal. | OE4 | Should |
| US-16 | Como artista 3D, quiero ver un indicador de calidad del mapa de normales predicho con un mapa de calor visual, para identificar zonas problemáticas (vectores no unitarios, falta de continuidad) antes de exportar. | OE4 | Could |
| US-17 | Como artista 3D, quiero poder procesar un lote de imágenes de materiales en una sola operación y descargar todos los mapas generados en un ZIP organizado por asset y motor de exportación, para agilizar la generación de materiales en proyectos con muchos activos. | OE4 | Could |
| US-18 | Como investigador, quiero disponer de una comparativa cuantitativa de MatForge frente a Pix2Pix, DeepPBR y las herramientas comerciales Materialize y Substance 3D Sampler sobre las métricas MAE angular, LPIPS, SSIM y RMSE, para evaluar objetivamente la contribución técnica del proyecto. | OE6 | Must |
| US-19 | Como investigador, quiero disponer de paneles cualitativos comparativos con renders Blender normalizados de los mapas PBR generados por MatForge, Materialize y Substance 3D Sampler sobre los mismos materiales de referencia, para complementar la comparativa cuantitativa con evidencia visual. | OE6 | Should |

---

## 3. Sprint Backlog

### Sprint 0 — Estrategia e investigación SOTA (22–23/04/2026)

**Objetivo del sprint:** Cerrar todas las decisiones de diseño arquitectónico y estratégico antes de escribir código.

| US | Tarea | Estado | Criterio de aceptación |
|---|---|---|---|
| US-01 | Investigación SOTA: arquitecturas encoder jerárquico viables para T4 | ✅ Completado | Informe con comparativa de planes A/B/C documentado |
| US-01 | Decisión de arquitectura (Plan A: PVT-v2-B1 + FPN + 3 heads) | ✅ Completado | Documento permanente v1.1 cerrado |
| US-01 | Decisión de framework (PyTorch sobre TensorFlow) | ✅ Completado | Justificación documentada en permanente v1.1 |
| US-02 | Diseño del pipeline de descarga de MatSynth (script reanudable) | ✅ Completado | `matforge_downloader.py` v2.0 implementado |
| US-05 | Decisión de estrategia de relabeling (DINOv2+HDBSCAN, no manual) | ✅ Completado | Documentado en permanente v1.1 |

**Artefactos entregados:** `MatForge_Arquitectura_Permanente.md` v1.1, `matforge_downloader.py` v2.0
**Velocidad:** 5 tareas completadas / 2 jornadas

---

### Sprint 1 — Dataset: descarga, EDA y relabeling (24–28/04/2026)

**Objetivo del sprint:** Construir el dataset de entrenamiento limpio y los artefactos de clasificación serializados.

| US | Tarea | Estado | Criterio de aceptación |
|---|---|---|---|
| US-01 | Descarga completa de MatSynth (9 categorías, 4 mapas/textura) | ✅ Completado | 3.814 texturas verificadas en disco local |
| US-01 | EDA semi-automático con 8 filtros por categoría | ✅ Completado | `matforge_eda.py` ejecutado; informe HTML generado |
| US-01 | Revisión humana de casos ambiguos y aplicación del descarte | ✅ Completado | 3.245 texturas limpias confirmadas |
| US-05 | Pipeline de relabeling DINOv2+PCA+UMAP+HDBSCAN | ✅ Completado | 37 clústeres → 8 grupos funcionales; DBCV=0,3279 |
| US-05 | Entrenamiento y serialización del clasificador KNN | ✅ Completado | `knn_classifier.pkl`, `pca_model.pkl`, `label_encoder.pkl` generados |
| US-01 | Subida del dataset a Kaggle | ✅ Completado | Dataset `MatForge PBR Dataset` (privado) accesible en Kaggle |

**Artefactos entregados:** `matforge_eda.py`, `matforge_relabeling.py`, artefactos de relabeling serializados, dataset Kaggle
**Velocidad:** 6 tareas completadas / 5 jornadas

---

### Sprint 2 — Implementación y entrenamiento de MatForgeNet (29/04–03/05/2026)

**Objetivo del sprint:** Implementar MatForgeNet, entrenarla y obtener un checkpoint final con métricas dentro de los objetivos de OE1.

| US | Tarea | Estado | Criterio de aceptación |
|---|---|---|---|
| US-04 | Implementación y validación del renderer Cook-Torrance diferenciable | ✅ Completado | Todos los tests unitarios superados (gradientes, inputs extremos, rendimiento) |
| US-04 | Implementación del DataLoader con MetalGuaranteedSampler | ✅ Completado | Distribución de metal en batch ≥ 2/8; split 85/15 persistido |
| US-04 | Implementación de MatForgeNet (PVT-v2-B1 + FPN + 3 RefineHeads) | ✅ Completado | Dry run sin errores; VRAM dentro del margen T4 |
| US-04 | Entrenamiento supervisado 90 épocas | ✅ Completado | MAE Normal ≤ 11°; S compuesto ≤ 11,0 en época 89 |
| US-04 | GAN fine-tuning 20 épocas (discriminador PatchGAN 2 escalas) | ✅ Completado | LPIPS ≤ 0,10 en best_gan.pt |
| US-18 | Documento permanente v1.5 cerrado con resultados reales | ✅ Completado | Todas las fases documentadas; checkpoint final identificado |

**Artefactos entregados:** `matforge-01-renderer-test.ipynb`, `matforge-02-dataloader-test.ipynb`, `matforge-03-training.ipynb`, `matforge_split.csv`, `best_gan.pt`, `MatForge_Arquitectura_Permanente_v1.5.md`
**Velocidad:** 6 tareas completadas / 5 jornadas

---

### Sprint 3 — Módulo SR (05/05/2026)

**Objetivo del sprint:** Diseñar, implementar y evaluar el módulo de super-resolución especializado.

| US | Tarea | Estado | Criterio de aceptación |
|---|---|---|---|
| US-08 | Benchmark de VRAM de candidatos SR en GTX 1650 Max-Q | ✅ Completado | Todos los candidatos verificados con constraint <3.500 MB |
| US-09 | Fine-tuning Real-ESRGAN (RRDBNet 23 bloques) sobre MatSynth — Fase 1 | ✅ Completado | Mejora val_LPIPS ≥ 10% sobre modelo base en época 24 |
| US-09 | Fase 2 GAN SR (discriminador U-Net PatchGAN) | ⚠️ Parcial | Abortada por colapso del discriminador; Fase 1 adoptada como resultado final |
| US-08 | Generación del informe técnico del módulo SR | ✅ Completado | `MatForge_SR_Informe_Tecnico.md` con 17 referencias IEEE |

**Artefactos entregados:** `matforge_sr_00_vram_check.py`, `matforge-sr-01-training.ipynb`, `sr_ft_phase1_best_lpips.pt`, `MatForge_SR_Informe_Tecnico.md`
**Velocidad:** 3 tareas completadas, 1 parcial / 1 jornada

**Nota:** La Fase 2 del fine-tuning SR se considera parcialmente completada: el objetivo de mejora ≥10% de OE2 se cumple con el checkpoint de Fase 1 (`sr_ft_phase1_best_lpips.pt`, val_LPIPS=0,2380, mejora del 10,9%). El colapso del discriminador en Fase 2 es un resultado técnico documentado, no un fallo de implementación.

---

### Sprint 4 — Integración en Streamlit y herramientas (06–09/05/2026)

**Objetivo del sprint:** Implementar la aplicación Streamlit completa con todas las herramientas de OE4 y OE5.

| US | Tarea | Estado | Criterio de aceptación |
|---|---|---|---|
| US-01 | Infraestructura del repositorio GitHub con Git LFS | ✅ Completado | Repositorio privado con checkpoints accesibles via `git clone` |
| US-05 | `src/classifier.py` (DINOv2+PCA+KNN) | ✅ Completado | Clasificación correcta de imagen sintética como `mixed_ambiguous` |
| US-10 | `src/postprocess.py` (6 funciones de postproceso) | ✅ Completado | Importación correcta sin errores; funciones verificadas |
| US-16 | `src/quality.py` (evaluación heurística normal map) | ✅ Completado | Heatmap generado correctamente para imagen de prueba |
| US-06 | `src/export.py` (5 motores + XMP) | ✅ Completado | ZIP con mapas correctos generado para Blender en prueba local |
| US-08 | `src/sr.py` (RRDBNet autocontenido + fallback) | ✅ Completado | SR funcional en float32; NaN resuelto |
| US-15 | `src/ui_components.py` (CSS warm-dark, Three.js, slider) | ✅ Completado | Visor 3D con RoomEnvironment operativo |
| US-01 | `app.py` integración completa del pipeline | ✅ Completado | Pipeline sin SR: ~2 s en GTX 1650 Max-Q; tres mapas correctos |
| US-07 | Metadatos XMP en todos los PNGs exportados | ✅ Completado | Campos dc:creator, xmp:CreatorTool verificados en archivos exportados |
| US-11 | H8 — Corrección de perspectiva interactiva | ✅ Completado | 4 handles arrastrables; warp aplicado correctamente |
| US-12 | H6 — Make Tileable (normalización de frecuencias) | ✅ Completado | Sin costuras visibles en preview 2×2 para imagen de ladrillos |
| US-13 | H5 — Mezclador RNM con color opcional | ✅ Completado | Mezcla blend=0,5 verificada visualmente en visor 3D |
| US-14 | H9 — Variaciones procedurales FBM | ✅ Completado | 3 variantes nombradas (Zonal Mix, Worn Edges, Scale Shift) generadas |
| US-10 | H7 — Calibración automática por grupo KNN | ✅ Completado | Expander con override manual y confianza α calculada |
| US-17 | H4 — Batch ZIP multi-imagen | ✅ Completado | 5 imágenes procesadas sin errores en test 6; ZIP estructurado por asset |
| US-02 | Testeo pre-release (6 tests / 6 materiales) | ✅ Completado | Sin bugs bloqueantes; versión v1.0 declarada estable |

**Artefactos entregados:** Todos los módulos de `src/`, `app.py`, `MatForge_App_Arquitectura_Permanente.md` v1.3
**Velocidad:** 16 tareas completadas / 4 jornadas (incluyendo iteraciones B y C)

---

### Sprint 5 — Evaluación, benchmarking y documentación (10–15/05/2026)

**Objetivo del sprint:** Evaluar cuantitativamente el sistema completo, comparar con el estado del arte y producir la documentación académica del PI.

| US | Tarea | Estado | Criterio de aceptación |
|---|---|---|---|
| US-03 | Documentación v1.0: README bilingüe, manual de usuario, release notes | ✅ Completado | README en raíz; manuales en `docs/`; release notes publicadas |
| US-18 | Evaluación cuantitativa: Tabla 1 (PBR restringida, 3 modelos) | ✅ Completado | Métricas MatForge, DeepPBR y Pix2Pix sobre split validación SEED=42 |
| US-18 | Evaluación cuantitativa: Tabla 2 (MatForge por grupo funcional) | ✅ Completado | 8 grupos evaluados; mejor marble_smooth (7,56°), peor stone_rough (16,42°) |
| US-18 | Evaluación cuantitativa: Tabla 3 (SR comparativa) | ✅ Completado | Bicúbico, Real-ESRGAN base y MatForge SR evaluados sobre 100 texturas |
| US-19 | Comparativa cualitativa Materialize (6 texturas) | ✅ Completado | Paneles cualitativos generados; Materialize verificado sin ML (GPL-3.0) |
| US-19 | Comparativa cualitativa Substance 3D Sampler (modo AI Powered) | ✅ Completado | Paneles cualitativos generados; modo AI Powered verificado en documentación |
| US-18 | Informe de benchmarking completo | ✅ Completado | `Informe_Benchmarking_MatForge.md` con tablas 1–3, análisis cualitativo y posicionamiento |
| US-18 | Tabla de posicionamiento competitivo (18 dimensiones) | ✅ Completado | MatForge App comparable a Materialize y Substance como sistema completo |

**Artefactos entregados:** `matforge-benchmark.ipynb`, CSVs de resultados (×7), grids y paneles visuales, `Informe_Benchmarking_MatForge.md`, documentación académica PI
**Velocidad:** 8 tareas completadas / 6 jornadas
