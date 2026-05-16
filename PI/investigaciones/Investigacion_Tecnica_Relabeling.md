# Informe de Investigación Técnica
## Estrategia de Relabeling Semántico para Dataset PBR Multidominio con DINOv2 + HDBSCAN
**Proyecto**: MatForge — Predicción de mapas PBR (Normal, Roughness, Metallic)
**Fecha**: 28/04/2026
**Fase**: Fase 1b — Relabeling del dataset

---

## 1. Introducción y Alcance

El objetivo de este informe es determinar la configuración técnica óptima para el pipeline de relabeling semántico del dataset MatForge, compuesto por 3.245 texturas PBR a resolución 1K distribuidas en 9 categorías originales de MatSynth (stone, wood, ceramic, terracotta, metal, concrete, ground, plaster, marble). Los tags originales son semánticamente inconsistentes —terracotta contiene principalmente ladrillos, plaster mezcla mortero, mosaicos y paredes heterogéneas— por lo que se requiere un sistema de agrupación basado en similitud visual real en lugar de etiquetas nominales.

El informe cubre cuatro decisiones técnicas diferenciadas:
1. Configuración óptima de DINOv2 (variante y tipo de feature).
2. Configuración óptima de HDBSCAN (parámetros y métrica).
3. Métrica de evaluación del clustering.
4. Clasificador ligero para inferencia en tiempo real en Streamlit.

---

## 2. Selección de la Variante de DINOv2 y Tipo de Feature

### 2.1 La familia DINOv2

DINOv2 es un modelo de aprendizaje auto-supervisado desarrollado por Meta AI y publicado en 2023, entrenado sobre el corpus curado LVD-142M mediante una combinación de DINO, iBOT y SwAV sobre arquitecturas ViT [1]. La característica fundamental que lo hace idóneo para nuestro caso de uso es que produce features de propósito general que no requieren fine-tuning: actúa como extractor de características congelado y genera representaciones competitivas con modelos supervisados en tareas de clasificación, retrieval, segmentación y estimación de profundidad [1].

La arquitectura ViT de DINOv2 produce dos tipos distintos de features en su capa final [2]:

- **Token [CLS]**: un único vector de dimensión fija que representa la imagen completa de forma global. Es el idóneo para clustering y clasificación a nivel de imagen.
- **Patch tokens**: una secuencia de vectores locales, uno por parche de 14×14 píxeles, útiles para tareas densas como segmentación semántica.

Para el relabeling de texturas completas, el **token [CLS]** es la elección correcta, ya que captura la identidad visual global de la textura (tipo de material, patrón dominante, propiedades cromáticas) sin requerir agregación sobre parches [2].

Una nota técnica relevante: investigación reciente ha demostrado que en los modelos más grandes de DINOv2 (ViT-g y ViT-L con register tokens), el [CLS] token y los patch tokens experimentan una desalineación creciente, donde el [CLS] token tiende a absorber información global a expensas de la especificidad local de los patches [3]. Este fenómeno **no afecta a DINOv2-small** (ViT-S/14), que mantiene una buena correspondencia entre representación global y local, lo que lo hace más predecible para nuestro uso.

### 2.2 Variante recomendada: DINOv2-small (ViT-S/14)

La familia DINOv2 incluye cuatro variantes principales: ViT-S/14 (small, 21M params, embedding 384D), ViT-B/14 (base, 86M params, embedding 768D), ViT-L/14 (large, 307M params, embedding 1024D) y ViT-g/14 (giant, 1.1B params, embedding 1536D) [1].

Para nuestras restricciones (CPU local, modelo <1GB, tiempo de extracción <30 min sobre 3.245 imágenes), **DINOv2-small es la única opción viable**:

| Variante | Params | Embedding | Tamaño en disco | Tiempo estimado CPU (3.245 imgs) |
|---|---|---|---|---|
| ViT-S/14 (small) | 21M | 384D | ~85 MB | ~15-20 min |
| ViT-B/14 (base) | 86M | 768D | ~330 MB | ~40-60 min |
| ViT-L/14 (large) | 307M | 1024D | ~1.2 GB | >2h |
| ViT-g/14 (giant) | 1.1B | 1536D | >4 GB | Inviable en CPU |

DINOv2-small supera el límite de 1GB de dependencias y es el único que encaja en el presupuesto de tiempo. La calidad de sus embeddings es suficiente para clustering semántico de materiales: ha sido validado en tareas de segmentación de materiales con resultados competitivos usando únicamente regresión logística sobre sus features congelados [4].

### 2.3 Resolución de entrada y preprocesamiento

DINOv2 fue entrenado con imágenes de 224×224 píxeles y opera con parches de 14×14. La resolución de entrada recomendada es 224×224, lo que da 16×16 = 256 patches por imagen. Redimensionar nuestras texturas de 1K a 224×224 para la extracción de embeddings es correcto y estándar: las propiedades visuales globales de una textura (patrón, color, estructura) se preservan a esa resolución.

El preprocesamiento estándar consiste en normalización con la media y desviación estándar de ImageNet:
```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Extraer únicamente el token `[CLS]` de la última capa del modelo congelado produce un vector de **384 dimensiones** por textura. El coste de almacenamiento de todos los embeddings es 3.245 × 384 × 4 bytes ≈ **5 MB**, completamente manejable en RAM.

---

## 3. Pipeline de Reducción de Dimensionalidad

### 3.1 El problema de la maldición de la dimensionalidad para HDBSCAN

HDBSCAN es un algoritmo basado en densidad. En espacios de alta dimensión, el concepto de "densidad local" se degrada porque las distancias entre puntos tienden a homogeneizarse (maldición de la dimensionalidad) [5]. Trabajar directamente sobre los 384 dimensiones del embedding DINOv2 con HDBSCAN produce resultados subóptimos: la mayoría de los puntos quedan marcados como ruido porque ninguna región del espacio parece más densa que las demás.

La solución es aplicar reducción de dimensionalidad **antes** de HDBSCAN. La documentación oficial del propio paquete UMAP lo describe explícitamente: *"high dimensional data requires more observed samples to produce much density. If we could reduce the dimensionality of the data more we would make the density more evident and make it far easier for HDBSCAN to cluster the data"* [6].

### 3.2 PCA vs UMAP como paso previo a HDBSCAN

Existe evidencia empírica de que para embeddings de transformers, **UMAP supera significativamente a PCA** como paso de reducción previo a HDBSCAN [7]:

- PCA preserva estructura global lineal pero destruye relaciones no lineales locales. En embeddings ViT, la estructura semánticamente relevante es no lineal, por lo que PCA colapsa clusters distintos y provoca que HDBSCAN clasifique casi todo como ruido.
- UMAP preserva simultáneamente estructura local y global del manifold de datos, produciendo clusters separados y cohesionados que HDBSCAN puede detectar con alta fiabilidad [6].

Un experimento documentado con embeddings de 512 dimensiones demuestra que PCA produce resultados "casi sin sentido" para HDBSCAN mientras que UMAP produce clusters robustos y separados [7].

### 3.3 Pipeline de reducción recomendado

La documentación oficial de UMAP especifica el pipeline canónico para este tipo de tarea [6]:

```
DINOv2 embeddings (384D) → PCA (50D) → UMAP (15D) → HDBSCAN
```

**Fase 1 — PCA a 50D**: reducción lineal preliminar para acelerar UMAP. PCA a 50 dimensiones retiene más del 95% de la varianza en embeddings ViT típicos y reduce el tiempo de UMAP de forma significativa. Es el paso recomendado por la documentación de UMAP para datasets con embeddings de dimensionalidad alta [6].

**Fase 2 — UMAP a 15D** (para clustering): UMAP a un espacio de 10-20 dimensiones preserva la estructura de manifold y produce la representación más adecuada para HDBSCAN. El uso de UMAP reduce el tiempo de ejecución de HDBSCAN de >26 minutos a ~5 segundos en datasets similares [8].

**Fase 3 — UMAP a 2D** (solo para visualización): una segunda proyección UMAP independiente, usada únicamente para el panel visual. No se usa para el clustering. Los parámetros de esta proyección pueden ser distintos a los de la proyección de clustering.

**Parámetros de UMAP recomendados para la proyección de clustering** (15D):
```python
umap.UMAP(
    n_components=15,
    n_neighbors=30,       # más alto que el defecto para capturar estructura global
    min_dist=0.0,         # sin distancia mínima: los puntos pueden estar juntos
    metric='cosine',      # coseno es la métrica natural para embeddings ViT
    random_state=42
)
```

La elección de `metric='cosine'` es fundamental: los embeddings de transformers como DINOv2 están optimizados para comparación por similitud coseno, no por distancia euclidiana [1][5].

---

## 4. Configuración de HDBSCAN

### 4.1 Parámetros principales

HDBSCAN tiene dos parámetros con impacto real en el clustering [9][10]:

**`min_cluster_size`**: el parámetro más importante. Define el número mínimo de puntos que debe tener un grupo para ser considerado un cluster. Puntos en grupos más pequeños se clasifican como ruido (-1). Para nuestro dataset de 3.245 texturas con grupos esperados de 141 (marble) a ~700 (stone), un valor de **30** es el apropiado:
- Si se pone demasiado bajo (5-10): se producen clusters espurios de texturas con similitudes accidentales.
- Si se pone demasiado alto (>100): marble podría no formar su propio cluster.
- **Valor recomendado: 30** (el grupo más pequeño esperable tiene 141 texturas, por lo que 30 da margen suficiente sin ser demasiado permisivo).

**`min_samples`**: controla la robustez del clustering. Valores altos producen más ruido pero clusters más puros; valores bajos producen menos ruido pero clusters más heterogéneos [10]. Según la documentación oficial [10]: *"The lower the value, the less noise you'll get"*. Para nuestro caso, un valor de **5** es adecuado: queremos detectar outliers genuinos (texturas ambiguas) sin penalizar texturas en la periferia de grupos válidos.

**`metric`**: la métrica de distancia usada por HDBSCAN. En el espacio UMAP reducido, **`'euclidean'`** es la elección correcta porque UMAP ya transformó el espacio de coseno a uno más euclidiano durante la reducción [6]. Aplicar coseno directamente sobre el espacio UMAP sería una doble transformación no justificada.

**`cluster_selection_method`**: dos opciones, `'eom'` (Excess of Mass, por defecto) y `'leaf'`. EOM produce clusters de tamaño más variable y es más adecuado cuando los grupos no tienen tamaño uniforme, que es exactamente nuestro caso (marble ~141 vs stone_rough ~700) [9].

### 4.2 Configuración completa recomendada

```python
hdbscan.HDBSCAN(
    min_cluster_size=30,
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True   # necesario para clasificación de nuevos puntos
)
```

`prediction_data=True` es imprescindible para poder usar el modelo de clustering para clasificar nuevas texturas en inferencia sin reentrenar.

### 4.3 Estrategia de búsqueda de hiperparámetros

Dado que no conocemos a priori el número de clusters ni su tamaño exacto, se recomienda un barrido de parámetros previo con una muestra del 20% del dataset (≈650 texturas) para identificar el rango óptimo antes de ejecutar en el dataset completo. Los parámetros a barrer son únicamente `min_cluster_size` en el rango [15, 20, 25, 30, 40, 50].

---

## 5. Métrica de Evaluación del Clustering

### 5.1 Por qué el Silhouette Score es inadecuado aquí

El Silhouette Score es la métrica de clustering más conocida, pero tiene una limitación fundamental para nuestro caso: **asume que los clusters son convexos (esféricos) y de densidad uniforme** [11]. HDBSCAN produce clusters de forma arbitraria y densidad variable, por lo que el Silhouette Score puede dar puntuaciones altas a resultados de clustering malos y bajas a resultados buenos, como ha sido demostrado empíricamente comparando KMeans vs HDBSCAN en los mismos datos [11][12].

### 5.2 DBCV: la métrica correcta para clustering basado en densidad

**DBCV (Density-Based Clustering Validation)** es el índice de validación diseñado específicamente para evaluar resultados de algoritmos basados en densidad como HDBSCAN [13]. A diferencia del Silhouette Score, DBCV:

- Computa la densidad relativa dentro de cada cluster y entre clusters, no distancias absolutas.
- Considera los puntos de ruido (-1) como parte integral de la evaluación.
- Produce puntuaciones en el rango [-1, 1] donde valores cercanos a 1 indican clustering excelente [13].

La implementación está disponible como paquete Python independiente (`dbcv` en PyPI) y también es accesible directamente desde el objeto HDBSCAN entrenado mediante `clusterer.relative_validity_`.

### 5.3 Métricas complementarias

| Métrica | Propósito | Cómo computarla |
|---|---|---|
| **DBCV** | Métrica principal de calidad del clustering | `clusterer.relative_validity_` |
| **% de ruido** | Control de outliers excesivos | `(labels == -1).mean()` |
| **Número de clusters** | Verificar que la segmentación es razonable | `len(set(labels)) - (1 if -1 in labels else 0)` |
| **Tamaño mínimo/máximo de cluster** | Detectar clusters degenerad | `pd.Series(labels[labels>=0]).value_counts()` |
| **NMI con categorías originales** | Alineación con los tags de MatSynth | `normalized_mutual_info_score(labels, original_cats)` |

El **NMI (Normalized Mutual Information)** con las categorías originales de MatSynth es especialmente útil: no como criterio de calidad absoluta (los tags originales son inconsistentes, por lo que un NMI perfecto sería una señal de alarma), sino como herramienta diagnóstica. Un NMI de ~0.5-0.7 indica que el clustering ha descubierto estructura real más allá de los tags originales, que es exactamente lo que buscamos.

### 5.4 Umbrales orientativos de calidad

| Indicador | Valor problemático | Valor aceptable | Valor excelente |
|---|---|---|---|
| DBCV | < 0.2 | 0.3 – 0.5 | > 0.5 |
| % de ruido | > 20% | 5% – 15% | < 5% |
| Nº de clusters | < 5 o > 20 | 6 – 12 | 7 – 10 |
| Tamaño del cluster más pequeño | < 50 | 80 – 130 | > 100 |

Si el % de ruido supera el 20%, reducir `min_cluster_size` a 20. Si el número de clusters supera 15, aumentar `min_cluster_size` a 40.

---

## 6. Clasificador Ligero para Inferencia en Streamlit

### 6.1 Arquitectura del pipeline de inferencia

El objetivo es que en la herramienta Streamlit, cuando un usuario suba una imagen RGB, el sistema identifique automáticamente el grupo de material al que pertenece antes de pasarla a MatForge. El pipeline completo es:

```
[Imagen usuario] → [DINOv2-small: extracción CLS token 384D]
                 → [PCA: 384D → 50D]  (PCA ajustado en entrenamiento)
                 → [Clasificador: 50D → etiqueta de grupo]
                 → ["Textura detectada: Mármol / Piedra rugosa / Metal..."]
                 → [MatForge: Normal + Roughness + Metallic]
```

El PCA y el clasificador se ajustan **una sola vez** durante el pipeline de relabeling y se serializan como archivos `.pkl` para cargarse en Streamlit. DINOv2-small ya está congelado.

### 6.2 KNN como clasificador recomendado

Para la tarea de asignar un grupo a una nueva textura dada su representación en el espacio de embeddings DINOv2, **KNN (K-Nearest Neighbors) es la elección óptima** por las siguientes razones:

**a) Sin entrenamiento real**: KNN almacena los embeddings del dataset de entrenamiento y clasifica un nuevo punto por mayoría de sus K vecinos más cercanos [14]. Esto significa que el "modelo" es simplemente el fichero de embeddings serializado, sin parámetros que ajustar.

**b) Interpretabilidad**: en Streamlit se puede mostrar al usuario no solo el grupo predicho sino también las N texturas más similares de la base de datos, lo cual es visualmente útil y técnicamente demostrable en la memoria del PI.

**c) Latencia**: KNN sobre 3.245 embeddings de 50D (post-PCA) en CPU moderna tiene latencia de ~1-5 ms por consulta usando el índice por defecto de scikit-learn (KD-Tree o Ball-Tree según dimensionalidad) [15], muy por debajo del límite de 100 ms requerido.

**d) Adaptabilidad**: si en el futuro se añaden nuevas texturas al dataset, se pueden añadir directamente a la base de embeddings sin reentrenar. Esto es especialmente relevante si se amplía el dominio a madera o metal con más muestras [16].

**Parámetros recomendados**:
```python
sklearn.neighbors.KNeighborsClassifier(
    n_neighbors=7,      # impar para desempate; 7 es estándar para datasets ~3000
    metric='cosine',    # métrica natural para embeddings de transformers
    weights='distance', # vecinos más cercanos tienen más peso en la votación
    algorithm='brute',  # más rápido que KD-Tree para métrica coseno en baja dim
    n_jobs=-1
)
```

La elección de `algorithm='brute'` con `metric='cosine'` es correcta: scikit-learn no puede usar KD-Tree ni Ball-Tree con métrica coseno, por lo que `'brute'` es el único algoritmo disponible para esa métrica. Con n=3.245 y d=50, la búsqueda exhaustiva es O(n·d) ≈ 162.000 operaciones por consulta, lo que es instantáneo en CPU moderna.

### 6.3 Alternativa: SVM con kernel RBF

Si KNN produce demasiados errores en las fronteras de grupos (por ejemplo, texturas de plaster que el KNN clasifica incorrectamente como ceramic), una SVM con kernel RBF es la alternativa más robusta. La SVM aprende fronteras de decisión más suaves y es menos sensible a outliers individuales. El coste es un tiempo de entrenamiento de ~30-60 segundos sobre 3.245 muestras de 50D, y la latencia de inferencia es comparable a KNN.

**Recomendación**: empezar con KNN. Si la validación manual del clasificador muestra más de un 15% de errores en los grupos más difusos, migrar a SVM.

---

## 7. Plan de Contingencia

### 7.1 Si el porcentaje de ruido supera el 20%

**Diagnóstico**: `min_cluster_size` es demasiado alto para la densidad real del dataset.
**Acción**: reducir `min_cluster_size` de 30 a 20. Si el problema persiste, reducir a 15.
**Límite inferior**: nunca bajar de 15, porque por debajo de ese valor se generan clusters espurios de texturas accidentalmente similares.

### 7.2 Si grupos semánticamente distintos se fusionan (ej. wood + terracotta)

**Diagnóstico**: la representación UMAP ha proyectado esos grupos en regiones del espacio con alta densidad compartida.
**Acción**: revisar el cluster fusionado en el panel UMAP y separarlo manualmente usando la categoría original de MatSynth como desempate. Si más del 60% de las texturas de un cluster pertenecen a una categoría original, asignar el cluster a esa categoría y reclasificar manualmente las texturas minoritarias.

### 7.3 Si marble desaparece como cluster independiente

**Diagnóstico**: con solo 141 texturas, HDBSCAN podría absorberlo en stone_rough o en ruido.
**Acción obligatoria**: forzar manualmente la asignación de todas las texturas de la categoría original "marble" a un grupo propio, independientemente del resultado del clustering. marble es el grupo más pequeño pero también el más diferenciado visualmente, por lo que su separación es crítica para el balance del sampler.

### 7.4 Si el número de clusters es excesivo (>15)

**Diagnóstico**: `min_cluster_size` es demasiado bajo, produciendo sub-clusters dentro de dominios homogéneos.
**Acción**: aumentar `min_cluster_size` a 40-50. Alternativamente, ejecutar HDBSCAN con `cluster_selection_epsilon=0.1` para fusionar clusters cercanos en el árbol jerárquico.

### 7.5 Si mixed_ambiguous supera 400 texturas (>12% del total)

**Diagnóstico**: el dataset tiene más heterogeneidad de la esperada, probablemente por las categorías plaster y ground.
**Acción**: dividir mixed_ambiguous en dos subgrupos: `ambiguous_stonish` (los que tienen canal Z de normal >200 y roughness alto) y `ambiguous_smooth` (los que tienen roughness bajo). Asignar a cada uno un peso diferente en el sampler (0.4 y 0.3 respectivamente).

---

## 8. Resumen de Decisiones Técnicas

| Decisión | Elección recomendada | Alternativa |
|---|---|---|
| Variante DINOv2 | ViT-S/14 (small, 384D) | — (única opción viable en CPU) |
| Feature extraída | Token [CLS] de la última capa | — |
| Resolución de entrada | 224×224 | — |
| Reducción previa a UMAP | PCA a 50D | — |
| Proyección para clustering | UMAP a 15D, coseno | — |
| Proyección para visualización | UMAP a 2D (independiente) | — |
| Algoritmo de clustering | HDBSCAN | — |
| min_cluster_size | 30 | 20 (si ruido >20%) |
| min_samples | 5 | — |
| Métrica HDBSCAN | euclidean (sobre espacio UMAP) | — |
| cluster_selection_method | eom | leaf (si clusters son demasiado grandes) |
| Métrica de evaluación primaria | DBCV | — |
| Métrica de evaluación secundaria | % ruido + NMI con cats. originales | — |
| Clasificador de inferencia | KNN (k=7, coseno, distance weights) | SVM RBF |
| Espacio del clasificador | 50D (post-PCA, pre-UMAP) | — |

---

## 9. Referencias en Formato IEEE

[1] M. Oquab *et al.*, "DINOv2: Learning Robust Visual Features without Supervision," *Transactions on Machine Learning Research*, 2024. [En línea]. Disponible: https://arxiv.org/abs/2304.07193.

[2] HuggingFace, "DINOv2 — Model Documentation," HuggingFace Transformers. [En línea]. Disponible: https://huggingface.co/docs/transformers/model_doc/dinov2. [Accedido: 28 abr. 2026].

[3] F. Lux *et al.*, "Register and [CLS] Tokens Induce a Decoupling of Local and Global Features in Large ViTs," *arXiv preprint arXiv:2505.05892*, 2025. [En línea]. Disponible: https://arxiv.org/abs/2505.05892.

[4] P. Docherty *et al.*, "Upsampling DINOv2 Features for Unsupervised Vision Tasks and Weakly Supervised Materials Segmentation," *arXiv preprint arXiv:2410.19836*, 2024. [En línea]. Disponible: https://arxiv.org/abs/2410.19836.

[5] G. Shadecoder, "HDBSCAN: A Comprehensive Guide for 2025," Shadecoder, 2025. [En línea]. Disponible: https://www.shadecoder.com/topics/hdbscan-a-comprehensive-guide-for-2025. [Accedido: 28 abr. 2026].

[6] L. McInnes, J. Healy, y J. Melville, "Using UMAP for Clustering," UMAP Documentation, v0.5.8. [En línea]. Disponible: https://umap-learn.readthedocs.io/en/latest/clustering.html. [Accedido: 28 abr. 2026].

[7] The GDELT Project, "Visualizing An Entire Day of Global News Coverage: Technical Experiments: PCA vs UMAP for HDBSCAN & t-SNE Dimensionality Reduction," GDELT Blog, nov. 2023. [En línea]. Disponible: https://blog.gdeltproject.org/visualizing-an-entire-day-of-global-news-coverage-technical-experiments-pca-vs-umap-for-hdbscan-t-sne-dimensionality-reduction/. [Accedido: 28 abr. 2026].

[8] H. Hamdan *et al.*, "Considerably Improving Clustering Algorithms Using UMAP Dimensionality Reduction Technique: A Comparative Study," en *Advances in Intelligent Systems and Computing*, vol. 1230, Springer, 2021, pp. 317–325. doi: 10.1007/978-3-030-51935-3_34.

[9] L. McInnes, J. Healy, y S. Astels, "Parameter Selection for HDBSCAN*," HDBSCAN Documentation, v0.8.1. [En línea]. Disponible: https://hdbscan.readthedocs.io/en/latest/parameter_selection.html. [Accedido: 28 abr. 2026].

[10] L. McInnes, J. Healy, y S. Astels, "Frequently Asked Questions — hdbscan," HDBSCAN Documentation, v0.8.1. [En línea]. Disponible: https://hdbscan.readthedocs.io/en/latest/faq.html. [Accedido: 28 abr. 2026].

[11] Towards AI, "The Limitation of Silhouette Score Which Is Often Ignored By Many," Daily Dose of Data Science, jul. 2023. [En línea]. Disponible: https://blog.dailydoseofds.com/p/the-limitation-of-silhouette-score. [Accedido: 28 abr. 2026].

[12] P. Jaskowiak, R. J. G. B. Campello, y I. G. Costa, "On the Evaluation of Unsupervised Outlier Detection: Measures, Datasets, and an Empirical Study," *Data Mining and Knowledge Discovery*, vol. 30, no. 4, pp. 891–927, 2016. doi: 10.1007/s10618-015-0444-8.

[13] D. Moulavi, P. A. Jaskowiak, R. J. G. B. Campello, A. Zimek, y J. Sander, "Density-Based Clustering Validation," en *Proc. 2014 SIAM International Conference on Data Mining*, 2014, pp. 839–847. doi: 10.1137/1.9781611973440.96. Implementación Python: https://github.com/christopherjenness/DBCV.

[14] scikit-learn Developers, "1.6. Nearest Neighbors," scikit-learn Documentation, v1.8.0. [En línea]. Disponible: https://scikit-learn.org/stable/modules/neighbors.html. [Accedido: 28 abr. 2026].

[15] S. Doerrich, T. Archut, F. Di Salvo, y C. Ledig, "Integrating kNN with Foundation Models for Adaptable and Privacy-Aware Image Classification," *arXiv preprint arXiv:2402.12500*, 2024. [En línea]. Disponible: https://arxiv.org/abs/2402.12500.

[16] scikit-learn Developers, "sklearn.cluster.HDBSCAN," scikit-learn Documentation, v1.8.0. [En línea]. Disponible: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html. [Accedido: 28 abr. 2026].

[17] R. J. G. B. Campello, D. Moulavi, A. Zimek, y J. Sander, "Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection," *ACM Transactions on Knowledge Discovery from Data*, vol. 10, no. 1, pp. 1–51, jul. 2015. doi: 10.1145/2733381.

[18] L. McInnes, J. Healy, y J. Melville, "Frequently Asked Questions — UMAP," UMAP Documentation, v0.5.8. [En línea]. Disponible: https://umap-learn.readthedocs.io/en/latest/faq.html. [Accedido: 28 abr. 2026].
