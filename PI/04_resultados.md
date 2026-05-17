# 4. Resultados

La presente sección expone los resultados obtenidos por MatForge App en relación con los objetivos específicos del Proyecto Intermodular. La evaluación combina tres tipos de evidencia: métricas internas de validación del modelo, benchmarking cuantitativo frente a los modelos previos del proyecto y validación funcional de la aplicación como herramienta local de producción de materiales PBR.

Para evitar ambigüedades, se distinguen dos niveles de medición. Por un lado, las métricas internas del entrenamiento permiten verificar los umbrales definidos en los objetivos específicos. Por otro, el benchmarking final evalúa el sistema en condiciones comparativas homogéneas frente a Pix2Pix, DeepPBR, Materialize y Adobe Substance 3D Sampler. En los casos en los que ambas fuentes no son directamente comparables, se explicita el alcance de cada resultado.

---

## 4.1. Resultados del modelo MatForge

*Responde a: OE1*

El objetivo OE1 establecía que el modelo final debía predecir mapas PBR con un error angular medio de normales inferior a 11°, un MAE de Roughness inferior a 0,12 y una distancia perceptual LPIPS inferior a 0,10 en el contexto de validación interna del entrenamiento. El checkpoint final adoptado para la aplicación fue `best_gan.pt`.

### 4.1.1. Métricas internas de validación

| Métrica | Checkpoint supervisado `ep89` | Checkpoint final `best_gan.pt` | Criterio OE1 | Cumplimiento |
|---|---:|---:|---:|:---:|
| MAE Normal (°) | 10,45 | **10,37** | < 11° | ✅ |
| Roughness MAE | **0,1087** | 0,1117 | < 0,12 | ✅ |
| LPIPS | 0,1094 | **0,0976** | < 0,10 | ✅ |

El checkpoint final mejora el error angular de normales y la métrica perceptual LPIPS respecto al checkpoint supervisado, aunque presenta una ligera degradación en Roughness MAE. Esta diferencia es aceptable dentro del objetivo definido, ya que el valor final de Roughness se mantiene por debajo del umbral de 0,12. La mejora de LPIPS indica que el ajuste final favorece la coherencia perceptual del render resultante, aun cuando no todas las métricas pixel-wise evolucionen en la misma dirección.

Debe señalarse que el ajuste adversarial no se interpreta como una mejora atribuible exclusivamente al discriminador. La evidencia metodológica previa indica que el discriminador tendió al colapso y que la mejora perceptual procede principalmente de la función de *feature matching*. Por tanto, el resultado se presenta como mejora del checkpoint final, no como validación plena de una presión adversarial estable.

### 4.1.2. Evaluación agregada por grupo funcional

La evaluación completa de MatForge sobre imágenes de 1024×1024 mediante *tile-and-merge* muestra una media global de **10,40°** en Normal MAE, **0,1084** en Roughness MAE y **0,2911** en LPIPS render. Esta LPIPS render no es directamente comparable con el LPIPS interno de entrenamiento del OE1, porque se calcula sobre renders sintéticos bajo un protocolo de benchmarking específico.

| Grupo            |   Normal MAE° ↓ |   Roughness MAE ↓ |   Roughness RMSE ↓ |   Metallic MAE ↓ |   Metallic RMSE ↓ |   LPIPS render ↓ |
|:-----------------|----------------:|------------------:|-------------------:|-----------------:|------------------:|-----------------:|
| GLOBAL           |           10.4  |            0.1084 |             0.1268 |           0.041  |            0.0485 |           0.2911 |
| brick_terracotta |           11.56 |            0.0816 |             0.0993 |           0.012  |            0.0157 |           0.2732 |
| ceramic_ground   |            8.96 |            0.099  |             0.1188 |           0.0226 |            0.0271 |           0.2373 |
| concrete_plaster |           13.21 |            0.0992 |             0.1196 |           0.1347 |            0.1596 |           0.3961 |
| marble_smooth    |            7.56 |            0.1285 |             0.1455 |           0.0705 |            0.0739 |           0.2942 |
| metal            |            8.23 |            0.0943 |             0.1103 |           0.1092 |            0.1527 |           0.3157 |
| mixed_ambiguous  |            9.51 |            0.1172 |             0.1377 |           0.0481 |            0.0527 |           0.3102 |
| stone_rough      |           16.42 |            0.0896 |             0.1069 |           0.0159 |            0.0183 |           0.3732 |
| wood             |            8.73 |            0.1309 |             0.1478 |           0.0262 |            0.0296 |           0.2275 |

El mejor comportamiento en Normal MAE se observa en `marble_smooth` (7,56°), seguido de `metal` (8,23°), `wood` (8,73°) y `ceramic_ground` (8,96°). Estos grupos presentan superficies más regulares o con patrones visuales suficientemente discriminables para el modelo. En cambio, `stone_rough` alcanza el mayor error angular (16,42°), lo que resulta coherente con la dificultad de inferir microgeometría irregular de alta frecuencia a partir de una única imagen RGB.

En términos perceptuales, `wood` obtiene el mejor LPIPS render (0,2275), mientras que `concrete_plaster` muestra el peor valor (0,3961). Este grupo también presenta el mayor Metallic MAE (0,1347), a pesar de tratarse de un grupo no metálico. El resultado sugiere que las superficies grises o de alta reflectancia difusa pueden inducir predicciones residuales de metalicidad. Esta observación justifica la incorporación de calibración por grupo funcional en la aplicación.

### 4.1.3. Evidencia visual

La evidencia visual asociada al OE1 se documenta mediante seis grids PBR. Cada grid compara, para un material representativo, la imagen RGB de entrada, los mapas de referencia y los mapas generados por los modelos evaluados:

![Comparativa PBR — concrete_0180](assets/grid_concrete_0180.png)

![Comparativa PBR — ceramic_0494](assets/grid_ceramic_0494.png)

![Comparativa PBR — metal_0175](assets/grid_metal_0175.png)

![Comparativa PBR — stone_0201](assets/grid_stone_0201.png)

![Comparativa PBR — stone_0480](assets/grid_stone_0480.png)

![Comparativa PBR — terracotta_0166](assets/grid_terracotta_0166.png)

Estas figuras permiten comprobar que MatForge conserva mejor la estructura local de los mapas de normales y reduce la sobreestimación de Roughness observada en modelos anteriores. En materiales metálicos, la predicción del canal Metallic aporta una ventaja adicional, ya que Pix2Pix y DeepPBR no cubrían de forma equivalente el mapa metálico.

---

## 4.2. Resultados del módulo de Super-Resolución

*Responde a: OE2*

El objetivo OE2 establecía la incorporación de un módulo de super-resolución ×4 con una mejora mínima del 10% en LPIPS respecto a Real-ESRGAN base. El módulo implementado se basa en RRDBNet y fue fine-tuneado sobre MatSynth. El checkpoint adoptado fue `sr_ft_phase1_best_lpips.pt`, correspondiente a la época 24 del entrenamiento interno.

### 4.2.1. Progresión interna del fine-tuning

| Época | LPIPS validación | Mejora vs. base |
|---:|---:|---:|
| 0 — base | 0,2672 | — |
| 10 | 0,2510 | −6,1% |
| 20 | 0,2430 | −9,1% |
| **24 — óptimo** | **0,2380** | **−10,9%** |
| 30 | 0,2401 | −10,1% |

Según esta validación interna, el módulo alcanza una mejora del **10,9%** en LPIPS, superando el umbral mínimo definido para el OE2. No obstante, este resultado debe interpretarse de forma restringida al dominio de validación utilizado durante el fine-tuning.

### 4.2.2. Evaluación general frente a Real-ESRGAN base

La evaluación final de SR se realizó sobre 100 texturas del split de validación. Para cada textura se generó una versión de baja resolución mediante *downscale* bicúbico ×4 y se compararon tres condiciones: bicúbico, Real-ESRGAN base y MatForge SR fine-tuned.

| Condition                |   PSNR ↑ |   SSIM ↑ |   LPIPS ↓ |
|:-------------------------|---------:|---------:|----------:|
| Bicubic                  |    32.59 |   0.795  |    0.4113 |
| Real-ESRGAN base         |    29.45 |   0.7328 |    0.2862 |
| MatForge SR (fine-tuned) |    27.83 |   0.7319 |    0.507  |

Los resultados muestran que Real-ESRGAN base obtiene el mejor LPIPS en la evaluación general (0,2862), mientras que MatForge SR fine-tuned alcanza 0,5070. Por tanto, la mejora interna de validación no se transfiere al protocolo general de evaluación. Esta divergencia se atribuye al *distribution shift* entre la degradación sintética empleada durante el entrenamiento y las condiciones reales o más generales de evaluación.

En consecuencia, el cumplimiento del OE2 debe formularse con matiz técnico: **el objetivo se cumple en la validación interna del fine-tuning, pero no queda plenamente confirmado en la evaluación general frente a Real-ESRGAN base**. Esta lectura es más rigurosa que considerar el módulo como éxito absoluto, ya que evita ocultar la limitación observada.

### 4.2.3. Evidencia visual del módulo SR

La comparación visual del módulo SR se documenta mediante cuatro grids específicos:

![Comparativa SR — ceramic_0166](assets/sr_grid_ceramic_0166.png)

![Comparativa SR — plaster_0095](assets/sr_grid_plaster_0095.png)

![Comparativa SR — stone_0086](assets/sr_grid_stone_0086.png)

![Comparativa SR — stone_0678](assets/sr_grid_stone_0678.png)

La evidencia cualitativa confirma que la super-resolución aporta utilidad en entradas de baja resolución, pero también que su integración debe usarse de forma controlada. En imágenes ya suficientemente grandes o alejadas de la distribución de entrenamiento, el módulo puede producir resultados visualmente planos o incoherentes.

---

## 4.3. Resultados del relabeling y del clasificador de materiales

*Responde a: OE3*

El objetivo OE3 requería reorganizar el dataset en ocho grupos funcionales, obtener una calidad mínima de agrupamiento DBCV ≥ 0,30 y disponer de un clasificador KNN serializado operativo para la aplicación.

El proceso de relabeling produjo **37 clusters HDBSCAN**, con **DBCV = 0,3279**, **15,1% de ruido** y **NMI = 0,3604** respecto a las categorías originales. Estas métricas quedan respaldadas por `cluster_metrics.json`. Posteriormente, los clusters se fusionaron manualmente en ocho grupos funcionales, más adecuados para calibración PBR que las categorías semánticas originales; la asignación final textura-grupo se conserva en `relabeling_final.csv`.

| Grupo funcional | Texturas | Peso sampler |
|---|---:|---:|
| `stone_rough` | 479 | 1,0 |
| `wood` | 658 | 1,0 |
| `ceramic_ground` | 503 | 1,0 |
| `mixed_ambiguous` | 775 | 0,5 |
| `brick_terracotta` | 276 | 1,0 |
| `marble_smooth` | 189 | 1,2 |
| `metal` | 238 | 1,3 |
| `concrete_plaster` | 127 | 1,0 |
| **TOTAL** | **3.245** | — |

El clasificador integrado usa **k = 7**, reducción **PCA-50** y distancia coseno. El artefacto serializado se encuentra en `artifacts/knn_classifier.pkl`. La ponderación funcional utilizada durante el entrenamiento queda documentada en `sampler_weights.json`. Su latencia de clasificación es inferior a 5 ms en CPU, por lo que no introduce un cuello de botella relevante dentro del pipeline de inferencia.

La evidencia visual del relabeling se recoge en los paneles UMAP:

![UMAP antes del relabeling — categorías originales](assets/panel_B_original_cats.png)

![UMAP después del relabeling — grupos funcionales](assets/panel_D_functional_groups.png)

El primer panel muestra la dispersión de las categorías originales, que no siempre se alinean con propiedades físicas útiles para PBR. El segundo panel muestra la reorganización en grupos funcionales, más adecuada para calibración de Roughness, Metallic y análisis de sesgos por tipo de material.

---

## 4.4. Funcionalidades implementadas en MatForge App

*Responde a: OE4, OE5, OE7*

La aplicación final no se limita a ejecutar el modelo MatForge, sino que integra un pipeline local de producción de materiales PBR. Las herramientas se aplican de forma no destructiva mediante estados de mapa acumulativos: `Raw`, `Adjusted`, `Calibrated`, `Blended`, `Tileable` y `Variation`.

![Vista principal de MatForge App](../docs/assets/hero_shot.png)

### 4.4.1. Herramientas de refinado no destructivo

**Corrección de perspectiva.** Permite rectificar fotografías tomadas con ángulo mediante cuatro puntos de control antes de ejecutar la inferencia. Esta funcionalidad reduce distorsiones geométricas que afectarían a la predicción PBR.

![Corrección de perspectiva](../docs/assets/perspective_correction.png)

**Zoom adaptativo.** Ajusta la escala efectiva de entrada antes del *tile-and-merge*. Su función no es solo acelerar la inferencia, sino mantener la imagen dentro de una distribución espacial compatible con los parches de entrenamiento.

![Panel de controles 1](../docs/assets/sidebar_1.png)

![Panel de controles 2](../docs/assets/sidebar_2.png)

**Ajuste de Roughness y Metallic.** Permite aplicar ganancia y desplazamiento a los canales Roughness y Metallic sin reinferir el modelo. El resultado se almacena en el estado `Adjusted`.

**Calibración por grupo funcional.** Usa la salida del clasificador KNN para aplicar curvas de corrección específicas del grupo de material detectado. La herramienta permite intervención manual mediante *override* del grupo cuando la clasificación automática no sea adecuada.

**Mezcla de materiales mediante RNM.** El mezclador combina dos conjuntos de mapas PBR preservando la coherencia vectorial del Normal map mediante Reoriented Normal Mapping [47].

![Material Blender](../docs/assets/material_blender.png)

**Variaciones procedurales.** Genera variantes mediante técnicas basadas en ruido FBM, desgaste de bordes y desplazamiento de escala. Su objetivo es reducir repetición visual cuando un mismo material se usa sobre superficies extensas.

![Variaciones procedurales](../docs/assets/procedural_variations.png)

**Conversión tileable y previsualización.** La herramienta `Make Tileable` reduce costuras visibles al teselar los mapas, y la previsualización 2×2 permite verificar el resultado antes de exportar.

![Previsualización tileable](../docs/assets/tiling_preview.png)

**Visor 3D integrado.** El visor Three.js permite comprobar el material sobre geometrías básicas sin abrir herramientas externas. Esta validación visual inmediata acerca la aplicación al flujo real de artistas técnicos.

![Visor 3D integrado](../docs/assets/viewer_3d.png)

### 4.4.2. Exportación multi-motor y metadatos

El sistema implementa exportación en ZIP para cinco configuraciones de motor: Blender, Unreal Engine 5, Unity URP, Unity HDRP y Godot 4. Cada motor recibe los mapas con la convención de nombres y empaquetado adecuada. Además, todos los PNG exportados incluyen metadatos XMP embebidos para identificar su procedencia como outputs generados por IA.

![Panel de exportación](../docs/assets/export.png)

| Motor | Normal | Roughness / Metallic | Color |
|---|---|---|---|
| Blender | `_normal.png` | `_roughness.png`, `_metallic.png` | `_color.png` |
| Unreal Engine 5 | `T_name_N.png` | `T_name_ORM.png` | `T_name_D.png` |
| Unity URP | `_normal.png` | `_MetallicSmoothness.png` | `_Albedo.png` |
| Unity HDRP | `_normal.png` | `_MaskMap.png` | `_Albedo.png` |
| Godot 4 | `_normal.png` | `_orm.png` | `_albedo.png` |

También se implementó procesado por lotes mediante ZIP. Esta funcionalidad analiza previamente el contenido, advierte sobre resoluciones problemáticas, procesa cada imagen de forma independiente y registra errores sin interrumpir el lote completo.

![Procesado en lote](../docs/assets/batch_zip.png)

### 4.4.3. Despliegue local y rendimiento

MatForge App funciona localmente, sin dependencia de red durante la inferencia. La aplicación detecta automáticamente si existe GPU CUDA disponible y adapta el tipo de dato a GPU o CPU. La precisión de cómputo se fija en FP32 para todos los modelos, tanto en GPU como en CPU. El uso de FP16 bajo autocast produjo valores NaN en la GTX 1650 Max-Q en ambos modelos durante el desarrollo, y fue descartado experimentalmente.

| Operación | GTX 1650 Max-Q — 4 GB VRAM |
|---|---:|
| Generate Maps — 512×512, zoom 1.0 | ~1 s |
| Generate Maps — 1024×1024, zoom 1.0 | ~6 s |
| Generate Maps — 1920×1920, zoom 0.5 | ~5 s |
| Sobrecarga de Super-Resolución | +~9 s |
| Normal Map Quality — 512×512 | ~7 s |
| Normal Map Quality — 1024×1024 | ~100 s |
| Variaciones procedurales — 352×352 | ~25 s |
| Variaciones procedurales — 640×640 | ~80 s |

Estos datos permiten considerar cumplido el criterio de inferencia inferior a 10 segundos en GPU de 4–8 GB **para el pipeline PBR sin SR**. Con SR activada, el tiempo puede superar los 10 segundos, por lo que debe considerarse una modalidad opcional de mayor coste computacional.

---

## 4.5. Benchmarking comparativo

*Responde a: OE6*

### 4.5.1. Metodología de evaluación

El benchmarking cuantitativo se realizó sobre el split de validación fijo del dataset MatSynth procesado, con **483 texturas** y `SEED=42`. Para igualar las condiciones de comparación con Pix2Pix y DeepPBR, que operan a resolución fija de 256×256, se evaluaron crops centrales de 256×256. Esta decisión penaliza parcialmente a MatForge, ya que en producción opera sobre imágenes completas mediante *tile-and-merge*, pero garantiza una comparación justa respecto a la información disponible para cada modelo.

Los sistemas evaluados fueron:

- **Pix2Pix**, modelo U-Net con encoder MobileNetV2 y salida Normal + Roughness.
- **DeepPBR**, modelo ResNet50 con atención CBAM y cabezas separadas para Normal y Roughness.
- **MatForge**, modelo PVT-v2-B1 + FPN + tres cabezas independientes para Normal, Roughness y Metallic [3].
- **MatForge_SR**, módulo de super-resolución ×4 basado en RRDBNet y evaluado por separado frente a Real-ESRGAN [37].
- **Materialize**, herramienta gratuita basada en shaders GPU y operaciones de procesamiento de imagen [48].
- **Adobe Substance 3D Sampler**, herramienta comercial con modo `Image to Material` basado en IA [49].

Las métricas principales fueron MAE angular para Normal, MAE/RMSE para Roughness y Metallic, y LPIPS render para evaluar diferencias perceptuales en renders sintéticos [24].

### 4.5.2. Resultados cuantitativos globales — Tabla 1

| Modelo   |   Normal MAE (°) |   Roughness MAE |   Roughness RMSE |   LPIPS render |
|:---------|-----------------:|----------------:|-----------------:|---------------:|
| Pix2Pix  |            14.74 |          0.212  |           0.2327 |         0.2988 |
| DeepPBR  |            13.64 |          0.2238 |           0.2472 |         0.3392 |
| MatForge |            10.4  |          0.1084 |           0.1268 |         0.2911 |

MatForge supera a Pix2Pix y DeepPBR en las cuatro métricas evaluadas. La mejora respecto a Pix2Pix alcanza el **29,5%** en Normal MAE, el **48,9%** en Roughness MAE y el **45,5%** en Roughness RMSE. Respecto a DeepPBR, la mejora es del **23,8%** en Normal MAE y del **51,6%** en Roughness MAE.

La única mejora más estrecha se observa en LPIPS render frente a Pix2Pix: MatForge obtiene 0,2911 frente a 0,2988. Aunque el margen es menor que en las métricas geométricas y de Roughness, sigue siendo favorable a MatForge. Frente a DeepPBR, la diferencia perceptual es más clara: 0,2911 frente a 0,3392.

### 4.5.3. Resultados por grupo funcional — Tabla 2

La Tabla 2, ya presentada en §4.1.2, evidencia que el rendimiento de MatForge no es homogéneo entre materiales. Los grupos `marble_smooth`, `metal`, `wood` y `ceramic_ground` presentan menor error angular, mientras que `stone_rough` y `concrete_plaster` concentran las mayores dificultades.

Esta variación no invalida el modelo, sino que justifica varias decisiones de diseño de la aplicación: clasificación automática de material, calibración por grupo funcional y posibilidad de ajuste manual. En un pipeline PBR real, estas herramientas son relevantes porque permiten compensar sesgos sistemáticos que no desaparecen únicamente con la inferencia del modelo.

### 4.5.4. Resultados de Super-Resolución — Tabla 3

La Tabla 3 muestra que MatForge SR no supera a Real-ESRGAN base en la evaluación general. Este resultado obliga a diferenciar entre éxito de entrenamiento interno y robustez externa. Aunque el fine-tuning alcanzó una mejora del 10,9% en su validación interna, el módulo no mejoró el LPIPS frente al modelo base en el protocolo general.

Desde el punto de vista académico, este resultado es valioso porque delimita con precisión el estado real del módulo: funcional e integrado en la app, pero no superior a Real-ESRGAN base fuera de la distribución de validación interna.

### 4.5.5. Análisis cualitativo frente a Materialize y Substance

La comparación cualitativa se realizó sobre seis texturas representativas: `ceramic_0494`, `concrete_0180`, `metal_0175`, `stone_0201`, `stone_0480` y `terracotta_0166`. Cada textura fue procesada con MatForge, Materialize y Adobe Substance 3D Sampler, y posteriormente renderizada bajo la misma escena Blender.

![Panel comparativo — ceramic_0494](assets/panel_ceramic_0494.png)

![Panel comparativo — concrete_0180](assets/panel_concrete_0180.png)

![Panel comparativo — metal_0175](assets/panel_metal_0175.png)

![Panel comparativo — stone_0201](assets/panel_stone_0201.png)

![Panel comparativo — stone_0480](assets/panel_stone_0480.png)

![Panel comparativo — terracotta_0166](assets/panel_terracotta_0166.png)

En materiales metálicos, MatForge muestra un comportamiento especialmente sólido, con mapas Normal y Metallic próximos al ground truth. Materialize presenta limitaciones más acusadas en este tipo de material, ya que sus normales tienden a ser planas o a depender de gradientes de luminancia. Substance 3D Sampler ofrece resultados visualmente convincentes y, en materiales no metálicos con geometría fina, puede mostrar mayor robustez que MatForge debido a la escala de su corpus propietario.

En materiales no metálicos, MatForge supera a Materialize en coherencia global de los mapas, aunque comparte con Substance algunas dificultades en geometría fina de bajo contraste. Estas limitaciones son esperables en un sistema que infiere propiedades físicas únicamente desde una imagen RGB, sin profundidad, iluminación conocida ni geometría 3D.

### 4.5.6. Posicionamiento como sistema completo

| Dimensión | MatForge App | Substance 3D Sampler | Materialize |
|---|---|---|---|
| Normal map en materiales metálicos | Alta fidelidad al GT | Bueno, con suavizado | Plano en geometría compleja |
| Normal map en materiales no metálicos | Limitado en geometría fina | Coherente | Artefactos de ruido |
| Roughness | Precisión alta | Coherente globalmente | Amplificado / Smoothness invertido |
| Metallic | Correcto en metal y no metal | Correcto | Heurístico por tono |
| Render final | Fiel al GT en metal | Convincente | Plano en geometría compleja |
| Tecnología base | Red neuronal PVT-v2-B1 + FPN | Red neuronal propietaria | Shaders GPU sin ML |
| Clasificación automática de material | DINOv2 + KNN, 8 grupos | No disponible | No disponible |
| Calibración por grupo funcional | Sí | No disponible | No disponible |
| Super-resolución integrada | Real-ESRGAN ×4 opcional | No disponible | No disponible |
| Corrección de perspectiva | Warp interactivo de 4 puntos | No disponible | No disponible |
| Mezcla de materiales | RNM paramétrico | Parcial / propietario | No disponible |
| Variaciones procedurales | 3 técnicas | Limitado | No disponible |
| Textura tileable | Mezcla en dominio de frecuencias | Disponible | Básico |
| Exportación multi-motor | 5 motores + XMP | Limitada a ecosistema Adobe | Manual |
| Procesado en lote | Batch ZIP | No disponible | No disponible |
| Visor 3D integrado | Three.js configurable | Visor externo | No disponible |
| Coste | Gratuito y local | Suscripción | Gratuito |
| Requisito de hardware | GPU CUDA local recomendada | Servicio / ecosistema Adobe | Cualquier GPU |

El principal valor diferencial de MatForge App es que no se limita a producir mapas, sino que integra inferencia, refinado, validación visual, exportación y procesamiento por lotes. En esta dimensión, la aplicación se aproxima a un pipeline completo de producción, con la ventaja de operar localmente y sin suscripción.

### 4.5.7. Síntesis del cumplimiento de OE6

El benchmarking confirma la superioridad cuantitativa de MatForge sobre Pix2Pix y DeepPBR en todas las métricas comunes evaluadas: Normal MAE, Roughness MAE, Roughness RMSE y LPIPS render. Frente a herramientas externas, MatForge se muestra competitivo con Substance 3D Sampler en materiales metálicos y supera a Materialize en la mayoría de categorías cualitativas analizadas. Como sistema completo, MatForge App ofrece una cobertura funcional más amplia que Materialize y más abierta/local que Substance 3D Sampler.

---

## 4.6. Comparación global con los objetivos específicos

| OE | Criterio | Resultado obtenido | Cumplimiento |
|---|---|---|:---:|
| OE1 | MAE Normal < 11° | 10,37° en validación interna; 10,40° en benchmarking global | ✅ |
| OE1 | Roughness MAE < 0,12 | 0,1117 en validación interna; 0,1084 en benchmarking | ✅ |
| OE1 | LPIPS < 0,10 | 0,0976 en validación interna | ✅ |
| OE2 | Mejora LPIPS ≥ 10% vs. Real-ESRGAN base | 10,9% en validación interna; no se transfiere al benchmark general | ⚠️ |
| OE3 | 8 grupos funcionales, DBCV ≥ 0,30, KNN operativo | 8 grupos, DBCV 0,3279, KNN serializado | ✅ |
| OE4 | Herramientas de refinado implementadas | Corrección perspectiva, zoom, ajuste R/M, calibración, RNM, tileable, variaciones, visor, comparación y calidad normal | ✅ |
| OE5 | Exportación multi-motor + XMP | Blender, UE5, Unity URP, Unity HDRP, Godot 4 + XMP en PNG | ✅ |
| OE6 | Superioridad sobre Pix2Pix y DeepPBR; comparativa frente a Materialize y Substance | Superioridad cuantitativa en Tabla 1 y análisis cualitativo/sistema completo | ✅ |
| OE7 | Inferencia < 10 s en GPU 4–8 GB, offline y detección CPU/GPU | ~1–6 s sin SR en GTX 1650 Max-Q; offline; detección CUDA/CPU | ✅ |

El único objetivo que requiere matización es OE2. El módulo SR existe, está integrado y alcanza el umbral en validación interna, pero la evaluación general demuestra que no supera a Real-ESRGAN base fuera de esa distribución. Esta limitación debe trasladarse a las conclusiones para evitar una lectura excesivamente optimista del resultado.

---

## 4.7. Limitaciones identificadas

Los resultados permiten identificar varias limitaciones relevantes:

1. **Dependencia de la información RGB.** MatForge no dispone de profundidad, iluminación conocida ni geometría 3D. Por ello, materiales con geometría fina de bajo contraste pueden producir normales planas o incompletas.

2. **Variabilidad por grupo funcional.** El rendimiento no es homogéneo entre grupos. `stone_rough` concentra el mayor error angular y `concrete_plaster` el mayor error de Metallic. La calibración por grupo mitiga este problema, pero no lo elimina.

3. **Super-Resolución sensible al dominio.** La mejora interna de MatForge SR no se generaliza a la evaluación global frente a Real-ESRGAN base. Esto confirma que la cadena de degradación sintética del entrenamiento no reproduce suficientemente todas las condiciones reales de uso.

4. **Coste de herramientas CPU.** Algunas herramientas no pertenecen al núcleo de inferencia PBR y pueden superar ampliamente los tiempos del modelo, especialmente `Normal Map Quality` y las variaciones procedurales sobre mapas grandes.

5. **Pipeline con SR superior a 10 segundos.** El criterio de OE7 se cumple para generación PBR sin SR, pero la activación de SR añade aproximadamente 9 segundos y puede superar el umbral.

Estas limitaciones no invalidan el proyecto, pero delimitan con precisión su estado técnico actual y orientan futuras mejoras.

# Archivos complementarios referenciados

Los resultados cuantitativos y cualitativos presentados en esta sección se apoyan en los siguientes archivos complementarios ubicados en `PI/anexos/` y `PI/investigaciones/`.

## Archivos en `PI/anexos/`

- `PI/anexos/matforge-benchmark.ipynb` — notebook principal de evaluación cuantitativa del benchmarking.
- `PI/anexos/table1_pbr_restricted.csv` — tabla de resultados globales del benchmarking PBR.
- `PI/anexos/table2_matforge_by_group.csv` — tabla de resultados de MatForge por grupo funcional.
- `PI/anexos/table3_sr.csv` — tabla de evaluación del módulo de super-resolución.
- `PI/anexos/results_pix2pix.csv` — resultados cuantitativos del baseline Pix2Pix.
- `PI/anexos/results_deeppbr.csv` — resultados cuantitativos del baseline DeepPBR.
- `PI/anexos/results_matforge.csv` — resultados cuantitativos de MatForge.
- `PI/anexos/results_sr.csv` — resultados cuantitativos del módulo de super-resolución.
- `PI/anexos/generate_qualitative_panels.py` — script utilizado para generar los paneles cualitativos comparativos.
- `PI/anexos/deeppbr-net.ipynb` — notebook del baseline DeepPBR empleado en la evaluación comparativa.
- `PI/anexos/pix2pix.ipynb` — notebook del baseline Pix2Pix empleado en la evaluación comparativa.
- `PI/anexos/metricas_completas.csv` — métricas completas del análisis exploratorio del dataset.
- `PI/anexos/candidates_to_discard.csv` — listado de texturas candidatas al descarte o revisión durante el EDA.
- `PI/anexos/revision_humana.html` — informe visual usado para la revisión manual de casos ambiguos.

## Archivos en `PI/investigaciones/`

- `PI/investigaciones/Informe_Benchmarking_MatForge.md` — informe técnico de benchmarking que documenta la metodología, los resultados cuantitativos y la comparación cualitativa frente a Materialize y Adobe Substance 3D Sampler.
- `PI/investigaciones/MatForge_App_Informe_Herramientas.md` — informe técnico utilizado para justificar las funcionalidades implementadas en la aplicación.
- `PI/investigaciones/MatForge_Informe_Tecnico.md` — informe técnico del modelo MatForgeNet y de sus métricas internas.
- `PI/investigaciones/MatForge_SR_Informe_Tecnico.md` — informe técnico del módulo de super-resolución y de sus resultados de validación.

## Recursos externos de reproducibilidad

Además de los archivos incluidos en `PI/anexos/` y `PI/investigaciones/`, el proyecto cuenta con recursos externos alojados en Kaggle que respaldan la reproducibilidad del entrenamiento y de los modelos comparativos:

- MatForge PBR Dataset: <https://www.kaggle.com/datasets/mjgut05/matforge-pbr-dataset>
- DeepPBR notebook: <https://www.kaggle.com/code/mjgut05/deeppbr-net>
- Pix2Pix notebook: <https://www.kaggle.com/code/mjgut05/pix2pix>

Estos recursos se referencian como material externo asociado al proyecto y no forman parte de la carpeta `PI/`.
