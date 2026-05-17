# 1. Introducción

## 1.1 Idea general del proyecto

MatForge App es una aplicación local de predicción automática de materiales PBR (*Physically Based Rendering*) que genera, a partir de una única imagen RGB de una superficie plana, los tres mapas de textura que definen el comportamiento físico de esa superficie bajo cualquier iluminación: el mapa de Normales, el mapa de Rugosidad (*Roughness*) y el mapa Metálico (*Metallic*). El sistema opera completamente en local, sin dependencias de conectividad en inferencia, sobre hardware de consumo convencional, y expone los resultados a través de una interfaz Streamlit accesible a artistas 3D sin formación en inteligencia artificial.

El núcleo predictivo de MatForge App es **MatForgeNet**, una arquitectura encoder-decoder compuesta por un encoder jerárquico PVT-v2-B1 (*Pyramid Vision Transformer v2*, variante B1) preentrenado en ImageNet-1K y un decoder FPN (*Feature Pyramid Network*) con tres cabezas de refinado independientes, una por cada mapa de salida. El modelo fue entrenado durante 90 épocas supervisadas (épocas 0–89) sobre un subconjunto curado de 3.245 texturas del dataset MatSynth [1], seguidas de 20 épocas de ajuste fino adversarial con discriminador PatchGAN multiescala, alcanzando un error angular medio en normales de 10,37°, un MAE de rugosidad de 0,1117 y un LPIPS de 0,0976 sobre el conjunto de validación.

La aplicación integra, además del modelo de predicción, un módulo de super-resolución ×4 basado en RRDBNet con ajuste fino especializado sobre MatSynth, un clasificador semántico de material (DINOv2-small + KNN) capaz de identificar automáticamente el grupo funcional de cualquier textura de entrada, y un conjunto de herramientas de postproceso no destructivo que incluye corrección de perspectiva interactiva, mezclador de materiales por Reoriented Normal Mapping, conversión a textura tileable, generación de variaciones procedurales y exportación directa para cuatro motores de renderizado: Blender, Unreal Engine 5, Unity (URP y HDRP) y Godot 4. Cada archivo PNG exportado incluye metadatos XMP de procedencia incrustados, con los campos `dc:creator`, `dc:rights`, `xmp:CreatorTool` y `xmpRights:Marked`, que identifican el origen de IA del contenido y anticipan las obligaciones de transparencia establecidas por el Reglamento UE 2024/1689 [49].

---

## 1.2 Contexto y justificación

### 1.2.1 El flujo de trabajo PBR y su coste

Los materiales PBR constituyen el estándar de representación de apariencia superficial en los motores de renderizado modernos. Un material PBR completo requiere al menos tres mapas de textura interdependientes: el mapa de Normales, que codifica la microgeometría de la superficie como un campo de vectores unitarios y permite simular detalle geométrico sin coste de vértices adicionales; el mapa de Rugosidad, que controla la dispersión de la luz a nivel superficial en el continuo entre superficie especular y difusa; y el mapa Metálico, que determina si el modelo de reflexión Cook-Torrance [25] trata la superficie como conductor o dieléctrico. La generación de estos mapas con calidad de producción exige, en el flujo de trabajo convencional, fotogrametría con iluminación controlada, autoría manual en software especializado o adquisición de bibliotecas de materiales prefabricados, opciones que resultan costosas en tiempo, en recursos económicos o en ambos [2].

### 1.2.2 Brecha identificada en el estado del arte

El análisis del estado del arte en estimación automática de SVBRDF (*Spatially-Varying Bidirectional Reflectance Distribution Function*) desde imagen única revela una brecha estructural no resuelta por ninguna herramienta disponible en mayo de 2026. Los métodos de investigación con mayor calidad perceptual —ControlMat [9], MatFusion [7] o SuperMat [11]— requieren recursos de cómputo incompatibles con hardware de consumo (entre 5 y 12 GB de VRAM) y carecen de interfaz accesible para usuarios no técnicos. Las herramientas comerciales de referencia, como Adobe Substance 3D Sampler [16], ofrecen la mejor experiencia de usuario pero imponen un modelo de suscripción, autenticación periódica en línea y procesado parcial en servidores remotos que las descalifica para flujos de trabajo con requisitos de privacidad de activos. Las alternativas de código abierto basadas en heurísticas, como Materialize [18] o AwesomeBump [19], son de acceso libre y ejecutan en local, pero producen mapas de calidad insuficiente para producción al carecer de cualquier componente aprendido. Ninguna herramienta identificada satisface simultáneamente los cuatro criterios del caso de uso objetivo: ejecución completamente local, hardware de consumo (4–8 GB VRAM), salida completa de los tres mapas PBR canónicos (Normal, Roughness y Metallic) desde una fotografía única, e interfaz accesible para artistas sin formación técnica en aprendizaje profundo.

### 1.2.3 Evolución respecto al proyecto grupal previo

MatForge App toma como referencia crítica el trabajo grupal previo **DeepPBR**, desarrollado conjuntamente con Adrián Bienvenido Pavón. DeepPBR demostró la viabilidad del problema en hardware modesto mediante una arquitectura ResNet50 con decoders duales y discriminador PatchGAN, pero presentó cuatro limitaciones estructurales que no son corregibles de forma incremental: un encoder de clasificación inadecuado para predicción densa, una función de pérdida sin anclaje físico en el renderizado, activación prematura del discriminador desde la primera época de entrenamiento, y un dataset sesgado a materiales pétreos (~1.500 texturas). MatForge no es una extensión de DeepPBR: es un rediseño completo cuyas decisiones arquitectónicas y de datos parten de cero, motivadas explícitamente por las limitaciones observadas en ese trabajo previo.

---

## 1.3 Definición del problema

El problema tecnológico abordado consiste en estimar automáticamente, a partir de una única imagen RGB de una superficie de material plana y tileable capturada bajo condiciones de iluminación no controladas, los tres mapas PBR canónicos (Normal, Roughness y Metallic) que describen el comportamiento de reflexión física de esa superficie, de forma que los mapas predichos sean coherentes entre sí bajo el modelo de reflexión Cook-Torrance, integrables directamente en los principales motores de renderizado 3D sin correcciones manuales, y generables en un sistema ejecutable en local sobre hardware de consumo convencional con un tiempo de respuesta inferior a 10 segundos en GPU de 4–8 GB VRAM.

---

## 1.4 Objetivo general

Desarrollar MatForge App, una herramienta de predicción automática de materiales PBR ejecutable en local sobre hardware de consumo, que permita a artistas 3D sin formación en inteligencia artificial obtener mapas Normal, Roughness y Metallic físicamente coherentes a partir de una única fotografía de superficie, mediante una arquitectura encoder-decoder con pérdidas ancladas al modelo de reflexión Cook-Torrance y un conjunto de herramientas de refinado y exportación integradas.

---

## 1.5 Objetivos específicos

** *OE1 — Modelo de predicción PBR* **
Desarrollar MatForgeNet, una arquitectura encoder-decoder compuesta por el encoder jerárquico PVT-v2-B1 [3] y un decoder FPN [30] con tres cabezas de refinado independientes, capaz de predecir simultáneamente los mapas Normal, Roughness y Metallic a partir de una única imagen RGB, alcanzando un error angular medio en normales inferior a 11°, un MAE de rugosidad inferior a 0,12 y un LPIPS inferior a 0,10, mediante 90 épocas de entrenamiento supervisado (épocas 0–89) sobre 3.245 texturas curadas de MatSynth [1] seguidas de 20 épocas de ajuste fino adversarial con discriminador PatchGAN multiescala.

**OE2 — Módulo de super-resolución especializado**
Implementar MatForge SR, un módulo de super-resolución ×4 basado en RRDBNet [65] con 23 bloques RRDB, ajustado sobre el dominio de materiales de superficie de MatSynth mediante una función de pérdida compuesta por reconstrucción L1, pérdida perceptual VGG-19 y LPIPS, logrando una mejora de al menos un 10% en LPIPS respecto al modelo Real-ESRGAN base en el dominio de validación del fine-tuning, e integrándose como preprocesado transparente y condicional en el pipeline de la aplicación.

**OE3 — Pipeline de relabeling semántico y clasificador de material**
Desarrollar un pipeline de relabeling semántico no supervisado —DINOv2-small [31] + PCA-50 + UMAP + HDBSCAN [35]— que reorganice las 3.245 texturas del dataset en 8 grupos funcionales visualmente coherentes, y serializar el clasificador resultante (KNN k=7, espacio PCA-50, métrica coseno) para su integración en la aplicación Streamlit, donde permite identificar automáticamente el grupo de material de cualquier textura de entrada y habilitar la calibración diferenciada del postproceso por categoría.

**OE4 — Herramientas de refinado no destructivo**
Integrar en la interfaz Streamlit un conjunto de herramientas de postproceso no destructivas —corrección de perspectiva interactiva, zoom adaptativo, ajuste de ganancia y desplazamiento por canal, calibración por grupo funcional, mezclador por Reoriented Normal Mapping, variaciones procedurales FBM y conversión a textura tileable por dominio de frecuencias— que permitan al artista refinar los mapas generados sin alterar el modelo subyacente ni requerir conocimientos técnicos de aprendizaje profundo.

**OE5 — Exportación multi-motor con metadatos de procedencia**
Implementar un sistema de exportación compatible con los cuatro motores de renderizado principales —Blender, Unreal Engine 5, Unity (URP y HDRP) y Godot 4— con empaquetado correcto de canales según las convenciones de cada motor, que incluya metadatos XMP de procedencia incrustados en cada archivo PNG, con los campos `dc:creator`, `dc:rights`, `xmp:CreatorTool` y `xmpRights:Marked`, identificando el origen de IA del contenido y anticipando las obligaciones del Art. 50 del Reglamento UE 2024/1689 [49].

**OE6 — Evaluación cuantitativa y benchmarking**
Evaluar MatForge App de forma cuantitativa frente a los sistemas de referencia del trabajo grupal previo (Pix2Pix, DeepPBR) y frente a las herramientas representativas del estado del arte (Materialize, Adobe Substance Sampler), mediante las métricas estándar de estimación de materiales PBR (MAE angular, LPIPS, SSIM y RMSE), y documentar la superioridad del sistema propuesto sobre los baselines en todas las dimensiones evaluadas, incluyendo una comparativa de capacidades de sistema frente a Materialize y Adobe Substance 3D Sampler como pipelines completos de producción.

**OE7 — Accesibilidad y despliegue local**
Desplegar MatForge App como herramienta ejecutable en local, sin dependencias de conectividad en inferencia ni modelo de suscripción, sobre hardware de consumo con GPU de 4–8 GB VRAM o CPU únicamente —con detección automática de dispositivo y ajuste de precisión numérica—, de forma que un artista 3D sin formación en inteligencia artificial pueda obtener mapas PBR listos para producción a partir de una fotografía en menos de 10 segundos en GPU.
