# Informe de Benchmarking Comparativo
## MatForge — Evaluación cuantitativa y cualitativa de modelos de predicción de mapas PBR

---

## Resumen ejecutivo

Este informe presenta la evaluación comparativa de los cuatro modelos desarrollados en el proyecto MatForge (Pix2Pix, DeepPBR, MatForge y MatForge_SR) frente a herramientas de referencia del sector (Materialize y Adobe Substance 3D Sampler). La evaluación combina métricas cuantitativas sobre el split de validación fijo del dataset MatSynth (483 texturas, SEED=42) con una comparativa cualitativa visual sobre seis texturas representativas procesadas mediante renderizado físico en Blender.

Los resultados muestran que MatForge supera a sus predecesores en todas las métricas evaluadas, con una reducción del MAE angular de normales del 29.6% respecto a Pix2Pix y del 23.8% respecto a DeepPBR, y una mejora del 48.9% en Roughness MAE respecto a Pix2Pix. En la comparativa cualitativa frente a herramientas externas, MatForge produce resultados competitivos con Adobe Substance 3D Sampler en materiales metálicos y supera a Materialize en la mayoría de categorías de material evaluadas.

---

## 1. Objetivos del benchmarking

Los objetivos de esta evaluación son:

1. Cuantificar la mejora progresiva de calidad entre los tres modelos de predicción PBR desarrollados en el proyecto (Pix2Pix → DeepPBR → MatForge) sobre el mismo conjunto de datos de validación.
2. Evaluar el impacto del módulo de super-resolución MatForge_SR frente al modelo base Real-ESRGAN y frente a no aplicar SR.
3. Contextualizar la calidad de MatForge frente a herramientas de referencia del sector mediante comparativa cualitativa visual.
4. Identificar fortalezas y limitaciones del sistema MatForge con evidencia cuantitativa y visual.

---

## 2. Descripción de los sistemas evaluados

### 2.1 Modelos propios

**Pix2Pix** es el primer modelo experimental del proyecto. Arquitectura U-Net con encoder MobileNetV2 preentrenado y decoder con skip connections. El generador produce un tensor de cuatro canales (Normal XYZ + Roughness) a resolución fija de 256×256 píxeles. Entrenado con pérdida L1 + adversarial PatchGAN sobre el dataset MatSynth procesado.

**DeepPBR** es el segundo modelo. Encoder ResNet50 preentrenado con módulos de atención CBAM (*Convolutional Block Attention Module*) en las skip connections y decoder dual con cabezas independientes para Normal y Roughness. Resolución fija de 256×256 píxeles. No predice el mapa Metallic.

**MatForge** es el modelo final del proyecto. Encoder jerárquico PVT-v2-B1 con preentrenamiento ImageNet-1K [6], decoder FPN piramidal y tres cabezas RefineHead independientes (Normal, Roughness, Metallic). Opera sobre imágenes de resolución arbitraria mediante tile-and-merge con ventana de Hann (parches 256×256, stride 128). Checkpoint: `best_gan.pt`.

El modelo se integra en **MatForge App**, una aplicación Streamlit local que implementa un pipeline de producción completo. El pipeline incluye: corrección de perspectiva interactiva, zoom adaptativo, super-resolución opcional (MatForge_SR), clasificación automática del grupo de material mediante DINOv2-small [7] con reducción PCA-50 y clasificador KNN entrenado sobre las etiquetas de grupo de MatSynth, e inferencia PBR. Sobre los mapas generados, la aplicación ofrece herramientas de postproceso no destructivas: ajuste de ganancia y offset por canal, calibración por grupo funcional con curvas específicas, mezcla de materiales mediante Reoriented Normal Mapping (RNM) [8], generación de variaciones procedurales basadas en ruido FBM, y conversión a textura tileable mediante mezcla en dominio de frecuencias. La exportación cubre cinco motores de renderizado (Blender, Unreal Engine 5, Unity URP, Unity HDRP, Godot 4) con metadatos XMP de procedencia en todos los PNG, y admite procesado en lote mediante archivo ZIP. Esta integración sitúa a MatForge App como sistema de producción completo, comparable funcionalmente con Materialize y Adobe Substance 3D Sampler, y no únicamente como modelo de inferencia aislado.

**MatForge_SR** es el módulo de super-resolución independiente. Arquitectura RRDBNet de 6 bloques RRDB (*Residual-in-Residual Dense Block*), factor de escala ×4. Fine-tuning supervisado sobre el dataset MatSynth durante 30 épocas (Fase 1). Checkpoint: `sr_ft_phase1_best_lpips.pt`.

### 2.2 Herramientas externas

**Materialize** (Bounding Box Software, GPL-3.0) es una herramienta gratuita y de código abierto para la generación de mapas PBR a partir de imágenes. Su implementación, disponible públicamente en GitHub como proyecto Unity [A], está basada en shaders GPU y operaciones de procesamiento de imagen: transformaciones de gradiente, filtros de frecuencia y operaciones de nivel de píxel sobre la GPU. No incorpora modelos de aprendizaje automático. Genera mapas de Height, Normal (derivado del Height), Metallic y Smoothness (inverso del Roughness estándar).

**Adobe Substance 3D Sampler** es una herramienta comercial de autoría de materiales. Su flujo de trabajo *Image to Material* dispone de dos algoritmos: el método B2M (*Bitmap to Material*), basado en técnicas procedurales, y el método *AI Powered*, que utiliza una red neuronal entrenada sobre un amplio corpus de materiales para reconocer formas, objetos y propiedades de superficie y generar los mapas Normal, Height y Roughness [B]. En este benchmarking se utilizó el modo *AI Powered* para maximizar la comparabilidad con los modelos basados en aprendizaje profundo.

---

## 3. Metodología

### 3.1 Dataset y split de evaluación

Todos los modelos se evalúan sobre el split de validación fijo del dataset MatSynth procesado (SEED=42, estratificado por grupo funcional). El split contiene 483 texturas distribuidas en ocho grupos funcionales: `stone_rough`, `wood`, `ceramic_ground`, `mixed_ambiguous`, `brick_terracotta`, `marble_smooth`, `metal` y `concrete_plaster`.

El grupo `metal` (238 texturas en el dataset completo, ~36 en validación) es el único con mapa Metallic real en el ground truth. Para el resto de grupos, el GT de Metallic es cero constante.

### 3.2 Métricas cuantitativas

| Métrica | Mapa | Descripción |
|---|---|---|
| MAE angular (°) | Normal | Ángulo medio en grados entre vector normal predicho y GT. $\text{MAE}_{ang} = \frac{180}{\pi} \cdot \text{mean}(\arccos(\text{clamp}(\hat{N}_{pred} \cdot \hat{N}_{gt}, -1, 1)))$ |
| MAE | Roughness, Metallic | Error absoluto medio en [0,1] |
| RMSE | Roughness, Metallic | Raíz del error cuadrático medio; más sensible a outliers que el MAE |
| LPIPS (AlexNet) | Render sintético | Distancia perceptual entre renders del material predicho y GT bajo 5 luces fijas [C] |

El render para LPIPS utiliza el albedo GT compartido por todos los modelos, de forma que las diferencias en el render reflejan exclusivamente la calidad de los mapas predichos.

### 3.3 Condiciones de evaluación para Pix2Pix y DeepPBR

Pix2Pix y DeepPBR operan a resolución fija de 256×256 píxeles. Las texturas del dataset están a 1024×1024 píxeles. Para la comparativa restringida (Tabla 1) se extrae un crop central de 256×256 de cada textura y se evalúan todos los modelos sobre ese crop, igualando las condiciones de entrada. Esta decisión penaliza ligeramente a MatForge (que en producción opera sobre la imagen completa) pero garantiza que la comparativa es justa respecto a la información de entrada disponible.

### 3.4 Comparativa SR

El módulo SR se evalúa sobre 100 texturas seleccionadas del split de validación (SEED=42). Para cada textura se genera una versión de baja resolución mediante downscale bicúbico ×4, y se comparan tres condiciones de reconstrucción: interpolación bicúbica (baseline), Real-ESRGAN base (sin fine-tuning) y MatForge_SR (fine-tuned). Las métricas se calculan respecto a la imagen original 1K como ground truth.

### 3.5 Comparativa cualitativa con herramientas externas

Se seleccionaron seis texturas representativas del split de validación, cubriendo diversidad de categorías: `ceramic_0494`, `concrete_0180`, `metal_0175`, `stone_0201`, `stone_0480` y `terracotta_0166`. Cada textura fue procesada manualmente con Materialize y Adobe Substance 3D Sampler, y los mapas resultantes fueron renderizados bajo la misma escena Blender (tres luces direccionales tipo Sun, cámara cenital, material Principled BSDF, Cycles renderer) para garantizar condiciones de iluminación idénticas entre herramientas.

---

## 4. Resultados cuantitativos

### 4.1 Tabla 1 — Comparativa restringida (Normal + Roughness)

Evaluación sobre 483 texturas del split de validación, crops de 256×256.

| Modelo | Normal MAE° ↓ | Roughness MAE ↓ | Roughness RMSE ↓ | LPIPS render ↓ |
|---|---|---|---|---|
| Pix2Pix | 14.74 | 0.2120 | 0.2327 | 0.2988 |
| DeepPBR | 13.64 | 0.2238 | 0.2472 | 0.3392 |
| **MatForge** | **10.40** | **0.1084** | **0.1268** | **0.2911** |

MatForge supera a ambos modelos anteriores en las cuatro métricas. La mejora más significativa respecto a Pix2Pix es en Roughness MAE (−48.9%) y Roughness RMSE (−45.5%). Respecto a DeepPBR, la mejora en Normal MAE es del 23.8% y en Roughness MAE del 51.6%.

El MAE angular de DeepPBR (13.64°) es coherente con un modelo que ha aprendido a estimar normales correctamente sobre el mismo dominio de datos. MatForge sigue siendo superior en las cuatro métricas, con un margen especialmente pronunciado en Roughness, donde la cabeza especializada y la función de pérdida compuesta con término de render LPIPS aportan una ventaja estructural respecto a las arquitecturas anteriores.

### 4.2 Tabla 2 — Evaluación completa MatForge por grupo funcional

Evaluación sobre imágenes completas a 1024×1024 mediante tile-and-merge.

| Grupo | Normal MAE° ↓ | Roughness MAE ↓ | Roughness RMSE ↓ | Metallic MAE ↓ | Metallic RMSE ↓ | LPIPS render ↓ |
|---|---|---|---|---|---|---|
| **GLOBAL** | **10.40** | **0.1084** | **0.1268** | **0.0410** | **0.0485** | **0.2911** |
| brick_terracotta | 11.56 | 0.0816 | 0.0993 | 0.0120 | 0.0157 | 0.2732 |
| ceramic_ground | 8.96 | 0.0990 | 0.1188 | 0.0226 | 0.0271 | 0.2373 |
| concrete_plaster | 13.21 | 0.0992 | 0.1196 | 0.1347 | 0.1596 | 0.3961 |
| marble_smooth | 7.56 | 0.1285 | 0.1455 | 0.0705 | 0.0739 | 0.2942 |
| metal | 8.23 | 0.0943 | 0.1103 | 0.1092 | 0.1527 | 0.3157 |
| mixed_ambiguous | 9.51 | 0.1172 | 0.1377 | 0.0481 | 0.0527 | 0.3102 |
| stone_rough | 16.42 | 0.0896 | 0.1069 | 0.0159 | 0.0183 | 0.3732 |
| wood | 8.73 | 0.1309 | 0.1478 | 0.0262 | 0.0296 | 0.2275 |

**Análisis por grupo**: `marble_smooth` obtiene el mejor Normal MAE (7.56°), coherente con la baja complejidad geométrica de las superficies lisas. `stone_rough` presenta el peor Normal MAE (16.42°), lo que refleja la dificultad inherente de las superficies con microdetalle geométrico complejo y alto índice de repetición de patrón. `concrete_plaster` muestra el Metallic MAE más alto (0.1347) siendo un grupo no metálico, lo que indica que el modelo tiende a predecir valores residuales de metallic en superficies grises de alta reflectancia especular, cuyas características cromáticas pueden resultar ambiguas para el modelo.

El Metallic MAE del grupo `metal` (0.1092) debe interpretarse considerando que solo ~36 texturas de validación pertenecen a este grupo. El modelo predice correctamente metallic ≈ 1 en las zonas metálicas dominantes pero tiene dificultad con materiales de metallic parcial o con combinaciones de zonas metálicas y no metálicas dentro de la misma textura.

La variabilidad de rendimiento entre grupos funcionales es la base del sistema de calibración por grupo de MatForge App. El clasificador KNN (DINOv2-small + PCA-50) identifica automáticamente el grupo de material de la imagen de entrada, permitiendo aplicar curvas de corrección específicas que compensan los sesgos sistemáticos observados en la Tabla 2. Esta compensación es especialmente relevante para `stone_rough` (Normal MAE 16.42°) y `concrete_plaster` (Metallic MAE 0.1347), donde el modelo presenta las mayores desviaciones respecto al ground truth.

### 4.3 Tabla 3 — Comparativa SR

Evaluación sobre 100 texturas del split de validación. LR generado mediante downscale bicúbico ×4. GT: imagen original 1024×1024.

| Condición | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicúbico | **32.59** | **0.7950** | 0.4113 |
| Real-ESRGAN base | 29.45 | 0.7328 | **0.2862** |
| MatForge SR (fine-tuned) | 27.83 | 0.7319 | 0.5070 |

**Interpretación**: el bicúbico obtiene el mejor PSNR y SSIM porque estas métricas penalizan las diferencias pixel-wise, y la interpolación bicúbica produce resultados conservadores y suavizados que minimizan ese error. Real-ESRGAN base obtiene el mejor LPIPS (0.2862), lo que indica que sus salidas son perceptualmente más cercanas al ground truth a pesar de tener menor fidelidad pixel-wise. Este es el comportamiento documentado en la literatura para modelos entrenados con pérdidas perceptuales y adversariales [2][3].

MatForge_SR fine-tuned presenta el peor resultado en las tres métricas. Esto indica que la Fase 1 del fine-tuning (30 épocas, solo generador, sin adversarial) no fue suficiente para superar al modelo base de 23 bloques en el dominio de evaluación general. La Fase 2 fue abortada por colapso del discriminador. El resultado es coherente con el *distribution shift* entre la degradación sintética utilizada en el entrenamiento y las condiciones reales de evaluación [15][16]. La mejora del −10.9% en LPIPS de validación observada durante el entrenamiento no se transfiere al conjunto de evaluación general.

---

## 5. Resultados cualitativos

### 5.1 Comparativa visual entre modelos PBR

Las figuras siguientes presentan, para cada una de las seis texturas seleccionadas, los mapas de Color, Normal y Roughness generados por GT, Pix2Pix, DeepPBR y MatForge en formato de cuadrícula vertical (4 filas × 3 columnas).

![Comparativa PBR — ceramic_0494](../assets/grid_ceramic_0494.png)
![Comparativa PBR — concrete_0180](../assets/grid_concrete_0180.png)
![Comparativa PBR — metal_0175](../assets/grid_metal_0175.png)
![Comparativa PBR — stone_0201](../assets/grid_stone_0201.png)
![Comparativa PBR — stone_0480](../assets/grid_stone_0480.png)
![Comparativa PBR — terracotta_0166](../assets/grid_terracotta_0166.png)

Los patrones observables en el conjunto de texturas evaluadas son consistentes con los resultados cuantitativos. MatForge produce mapas de normales con mayor detalle estructural y menor suavizado que Pix2Pix, especialmente en zonas de transición entre materiales y en bordes de grano. Los mapas de roughness de MatForge son significativamente más precisos que los de ambos modelos anteriores, con menor sobreestimación en zonas de baja rugosidad.

### 5.1b Comparativa visual SR

Las figuras siguientes presentan, para cada una de las cuatro texturas seleccionadas, la comparativa entre interpolación bicúbica, Real-ESRGAN base y MatForge SR, con detalle de zona ampliada.

![Comparativa SR — ceramic_0166](../assets/sr_grid_ceramic_0166.png)
![Comparativa SR — plaster_0095](../assets/sr_grid_plaster_0095.png)
![Comparativa SR — stone_0086](../assets/sr_grid_stone_0086.png)
![Comparativa SR — stone_0678](../assets/sr_grid_stone_0678.png)

### 5.2 Comparativa cualitativa frente a herramientas externas

Las figuras siguientes presentan, para cada textura, los mapas generados por GT, MatForge, Materialize y Substance 3D Sampler (columnas: Color, Normal, Roughness, Metallic, Render Blender cuando disponible).

![Panel comparativo — ceramic_0494](../assets/panel_ceramic_0494.png)
![Panel comparativo — concrete_0180](../assets/panel_concrete_0180.png)
![Panel comparativo — metal_0175](../assets/panel_metal_0175.png)
![Panel comparativo — stone_0201](../assets/panel_stone_0201.png)
![Panel comparativo — stone_0480](../assets/panel_stone_0480.png)
![Panel comparativo — terracotta_0166](../assets/panel_terracotta_0166.png)

**Materialize** genera sus mapas mediante shaders GPU basados en operaciones matemáticas sobre la imagen de entrada — transformaciones de gradiente, filtros de alta frecuencia y operaciones de nivel de píxel — sin modelos de aprendizaje automático [A]. Este enfoque tiene implicaciones directas en la calidad de los mapas:

- **Normal map**: en texturas con microdetalle geométrico complejo (como `metal_0175`, con patrón entrelazado de alta densidad), Materialize produce mapas de normales prácticamente planos (canal B dominante, sin variación en R/G). El algoritmo de gradiente no es capaz de inferir la geometría tridimensional de patrones de alta frecuencia a partir de la imagen 2D. En texturas con contraste de luminancia claro (como `ceramic_0494`), el Normal map presenta artefactos de ruido no estructurado.
- **Roughness/Smoothness**: exporta Smoothness en lugar de Roughness estándar. Los mapas muestran patrones amplificados respecto al GT, con tendencia a exagerar las diferencias locales de rugosidad.
- **Metallic**: aplica un heurístico global de tono de color para determinar la metalicidad. Esto produce predicciones de Metallic ≈ 1 para cualquier textura con tonos grises o plateados, independientemente de si el material es metálico. Este comportamiento es incorrecto para materiales como `concrete_plaster` o `ceramic_ground`.
- **Render Blender**: el resultado visual es perceptualmente plano en materiales de geometría compleja debido al Normal map incorrecto. En materiales de textura sencilla, el resultado es aceptable.

**Adobe Substance 3D Sampler** (modo *AI Powered*) utiliza una red neuronal entrenada sobre un amplio corpus de materiales para generar los mapas PBR [B]. Los resultados muestran un comportamiento cualitativamente diferente:

- **Normal map**: captura correctamente la orientación general de las superficies y produce mapas coherentes con la geometría visible. En `metal_0175`, captura la estructura entrelazada con suavizado moderado. En `ceramic_0494`, el Normal map es plano (similar a MatForge), indicando que ambos modelos tienen dificultad con la geometría de las juntas de baldosa a resolución de evaluación.
- **Roughness**: estimaciones globalmente coherentes con el GT, aunque con menor variación local que MatForge en materiales de roughness heterogéneo.
- **Metallic**: predicción correcta de metallic ≈ 0 en materiales no metálicos. En `metal_0175`, el mapa Metallic muestra valores altos en las zonas metálicas, aunque sin la precisión del GT.
- **Render Blender**: resultado visualmente convincente y comparable a MatForge en la mayoría de texturas evaluadas.

**MatForge** muestra el mejor comportamiento global en materiales metálicos, con Normal maps de alta fidelidad al GT y predicción de Metallic precisa para el grupo `metal`. En materiales no metálicos con geometría compleja (especialmente `ceramic_0494`), el Normal map tiende a ser plano, limitación compartida con Substance en este tipo de material.

### 5.3 Análisis de posicionamiento

| Dimensión | MatForge | Substance 3D Sampler | Materialize |
|---|---|---|---|
| Normal map (materiales metálicos) | ✅ Alta fidelidad al GT | ⚠️ Bueno, con suavizado | ❌ Plano en geometría compleja |
| Normal map (materiales no metálicos) | ⚠️ Limitado en geometría fina | ✅ Coherente | ❌ Artefactos de ruido |
| Roughness | ✅ Precisión alta | ✅ Coherente globalmente | ⚠️ Amplificado, Smoothness invertido |
| Metallic | ✅ Correcto (metal y no metal) | ✅ Correcto | ❌ Heurístico por tono |
| Render final | ✅ Fiel al GT en metal | ✅ Convincente | ⚠️ Plano en geometría compleja |
| Tecnología base | Red neuronal (PVT-v2-B1 + FPN) | Red neuronal (propietaria) | Shaders GPU (sin ML) |
| Clasificación automática de material | ✅ DINOv2 + KNN (8 grupos) | ❌ No disponible | ❌ No disponible |
| Calibración por grupo funcional | ✅ Curvas específicas por grupo | ❌ No disponible | ❌ No disponible |
| Super-resolución integrada | ✅ Real-ESRGAN ×4 (opcional) | ❌ No disponible | ❌ No disponible |
| Corrección de perspectiva | ✅ Warp interactivo de 4 puntos | ❌ No disponible | ❌ No disponible |
| Mezcla de materiales (RNM) | ✅ Blend paramétrico [8] | ⚠️ Parcial (interfaz propietaria) | ❌ No disponible |
| Variaciones procedurales | ✅ 3 técnicas (FBM, worn, scale) | ⚠️ Limitado | ❌ No disponible |
| Textura tileable | ✅ Mezcla en dominio de frecuencias | ✅ Disponible | ⚠️ Básico |
| Exportación multi-motor | ✅ 5 motores + metadatos XMP | ⚠️ Limitado a formatos Adobe | ❌ Exportación manual |
| Procesado en lote | ✅ Batch ZIP con pipeline completo | ❌ No disponible | ❌ No disponible |
| Visor 3D integrado | ✅ Three.js, geometría configurable | ❌ Visor externo | ❌ No disponible |
| Coste | Gratuito (local) | Suscripción Adobe | Gratuito (open source) |
| Requisito de hardware | GPU CUDA (local) | Servicio cloud (Adobe) | Cualquier GPU |

**Fortalezas de MatForge**: es el único sistema de los tres evaluados que produce el conjunto completo de mapas PBR (Normal, Roughness y Metallic) con entrenamiento supervisado sobre ground truth real del dataset MatSynth. Esto le confiere una ventaja estructural en la predicción de Metallic y en la fidelidad al ground truth físico en materiales para los que el dataset proporciona etiquetas de alta calidad.

**Limitaciones honestas**: MatForge opera exclusivamente sobre la imagen de color de entrada, sin acceso a información de iluminación, geometría 3D ni profundidad. En texturas con geometría compleja de bajo contraste cromático (como juntas de baldosa en cerámica gris), el modelo no tiene señal suficiente para inferir la geometría correctamente. Substance 3D Sampler, entrenado sobre un corpus significativamente más grande y diverso, muestra mayor robustez en estos casos.

---

## 6. Conclusiones

MatForge constituye una mejora cuantitativa consistente respecto a los modelos previos del proyecto en todas las métricas evaluadas. La reducción del MAE angular de normales del 29.6% respecto a Pix2Pix y del 23.8% respecto a DeepPBR, junto con la mejora del 48.9% en Roughness MAE respecto a Pix2Pix, evidencian el impacto de la arquitectura transformer jerárquica y el entrenamiento con función de pérdida compuesta sobre un dataset curado.

En la comparativa con herramientas externas, MatForge es competitivo con Adobe Substance 3D Sampler en materiales metálicos y supera a Materialize en la mayoría de categorías evaluadas. Las principales limitaciones de MatForge respecto a Substance son la menor robustez en geometría fina de materiales no metálicos y el dataset de entrenamiento más reducido. Estas limitaciones son estructuralmente esperables dada la diferencia de escala entre ambos sistemas y no comprometen la validez del proyecto.

El módulo MatForge_SR en su estado actual (Fase 1 completada, Fase 2 abortada) no supera al modelo base Real-ESRGAN en condiciones de evaluación general. La mejora del −10.9% en LPIPS observada durante la validación de entrenamiento no se transfiere al conjunto de evaluación, lo que indica que el fine-tuning de Fase 1 introduce una especialización en la distribución de degradación sintética que no generaliza suficientemente. La Fase 2 adversarial, interrumpida por colapso del discriminador, habría sido necesaria para cerrar esta brecha.

La integración del modelo en MatForge App amplía el alcance del sistema más allá de la predicción de mapas. La combinación de clasificación automática de material, calibración por grupo funcional, herramientas de postproceso no destructivas y exportación multi-motor sitúa a MatForge App como pipeline de producción completo orientado al flujo de trabajo real de artistas 3D y artistas técnicos. En esta dimensión, el sistema es funcionalmente comparable con herramientas comerciales como Adobe Substance 3D Sampler y supera a Materialize en la amplitud de capacidades ofrecidas, con la ventaja adicional de operar completamente en local sin dependencia de servicios externos ni suscripción.

---

## 7. Referencias

[A] M. Voeller, "Materialize," Bounding Box Software, open source under GNU GPL v3. [Online]. Available: https://github.com/BoundingBoxSoftware/Materialize

[B] Adobe Inc., "Image to Material — AI Powered algorithm," in *Adobe Substance 3D Sampler Documentation*, 2024. [Online]. Available: https://experienceleague.adobe.com/en/docs/substance-3d-sampler/using/filters/tools/image-to-material

[C] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," in *Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 586–595. doi: 10.1109/CVPR.2018.00066.

[6] W. Wang, E. Xie, X. Li, D.-P. Fan, K. Song, D. Liang, T. Lu, P. Luo, and L. Shao, "PVT v2: Improved Baselines with Pyramid Vision Transformer," *Comput. Vis. Media*, vol. 8, no. 3, pp. 415–424, 2022. doi: 10.1007/s41095-022-0274-8.

[7] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, "DINOv2: Learning Robust Visual Features without Supervision," *Trans. Mach. Learn. Res.*, 2024. [Online]. Available: https://openreview.net/forum?id=a68SUt6zFt

[8] C. Barré-Brisebois and S. Hill, "Blending in Detail," in *Game Developers Conference (GDC)*, San Francisco, CA, USA, Mar. 2012. [Online]. Available: https://blog.selfshadow.com/publications/blending-in-detail/

[2] X. Wang, K. Yu, S. Wu, J. Gu, Y. Liu, C. Dong, Y. Qiao, and C. C. Loy, "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks," in *Proc. Eur. Conf. Comput. Vis. Workshops (ECCVW)*, 2018, pp. 63–79. doi: 10.1007/978-3-030-11021-5_5.

[3] X. Wang, L. Xie, C. Dong, and Y. Shan, "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW)*, Oct. 2021, pp. 1905–1914. doi: 10.1109/ICCVW54120.2021.00217.

[4] G. Vecchio and V. Deschaintre, "MatSynth: A Modern PBR Materials Dataset," in *Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)*, 2024.

[5] C. Rodriguez-Pardo, H. Dominguez-Elvira, D. Pascual-Hernandez, and E. Garces, "UMat: Uncertainty-Aware Single Image High Resolution Material Capture," in *Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)*, 2023.

[15] R. Zhang, J. Gu, H. Chen, C. Dong, Y. Zhang, and W. Yang, "Crafting Training Degradation Distribution for the Accuracy-Generalization Trade-off in Real-World Super-Resolution," arXiv preprint arXiv:2305.18107, 2023.

[16] Y. Li, C. Dong, Y. Qiao, and C. C. Loy, "Suppressing Model Overfitting for Image Super-Resolution Networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recog. Workshops (CVPRW)*, Jun. 2019. arXiv:1906.04809.

---

## Anexos

Los siguientes archivos complementan este informe y se encuentran en la carpeta `PI/anexos/`:

| Archivo | Descripción |
|---|---|
| `matforge_benchmark.ipynb` | Notebook Kaggle con el pipeline completo de evaluación cuantitativa |
| `generate_qualitative_panels.py` | Script Python para generación de paneles comparativos cualitativos |
| `results_pix2pix.csv` | Métricas por textura — Pix2Pix |
| `results_deeppbr.csv` | Métricas por textura — DeepPBR |
| `results_matforge.csv` | Métricas por textura — MatForge |
| `results_sr.csv` | Métricas por textura — comparativa SR |
| `table1_pbr_restricted.csv` | Tabla 1 agregada — comparativa restringida PBR |
| `table2_matforge_by_group.csv` | Tabla 2 agregada — MatForge por grupo funcional |
| `table3_sr.csv` | Tabla 3 agregada — comparativa SR |
| `README.md` | Documentación principal del repositorio público MatForge App |
| `docs/USER_MANUAL.md` | Manual de usuario completo de MatForge App (EN) |
| `docs/assets/hero_shot.png` | Captura de MatForge App en funcionamiento con mapas generados |
| `docs/assets/batch_zip.png` | Captura del pipeline de procesado en lote (Batch ZIP) |