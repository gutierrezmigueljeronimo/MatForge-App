# **Informe de Investigación Técnica**

## **MatForge — Fase 1: Dataset y Metodología de Limpieza**

**Fecha**: 24-26 de abril de 2026 **Proyecto**: MatForge — Predicción de mapas PBR mediante aprendizaje profundo **Fuentes consultadas**: HuggingFace, arXiv, CVF Open Access, ScienceDirect, GitHub

---

## **1\. Investigación sobre el Dataset MatSynth**

### **1.1 Descripción general**

MatSynth es el dataset principal utilizado en este proyecto. Fue publicado en el marco de la conferencia CVPR 2024 por Vecchio y Deschaintre \[1\]. Se trata de un repositorio de materiales PBR de alta resolución publicado bajo licencias permisivas, diseñado específicamente para dar soporte a métodos modernos de aprendizaje automático aplicados a la estimación y síntesis de materiales.

El dataset completo comprende más de 4.000 materiales a resolución ultra-alta (hasta 4K por canal), divididos en dos splits: un split de entrenamiento con 3.980 materiales y un split de test con 89 materiales. Cada material está representado por un conjunto común de mapas PBR: Basecolor, Diffuse, Normal, Height, Roughness, Metallic, Specular y, cuando es pertinente, Opacity. El estándar de coordenadas adoptado para el mapa Normal es OpenGL (eje Y apuntando hacia arriba), lo cual es un dato crítico para el diseño de las funciones de pérdida de MatForge.

### **1.2 Estructura de metadatos**

Cada material del dataset incluye metadatos enriquecidos que facilitan la selección y el uso precisos. Los campos disponibles son los siguientes:

* **category**: categoría semántica del material. Los identificadores numéricos utilizados en MatSynth son: ceramic=0, concrete=1, fabric=2, ground=3, leather=4, marble=5, metal=6, misc=7, plaster=8, plastic=9, stone=10, terracotta=11, wood=12.  
* **capture\_method**: método de captura o generación del material (photogrammetry, procedural generation, approximation).  
* **tags**: lista de etiquetas descriptivas del material.  
* **source\_name** y **source\_link**: origen del material (AmbientCG, PolyHaven, etc.).  
* **license**: tipo de licencia (CC0, CC-BY, etc.).  
* **timestamp**: fecha de creación o actualización.  
* **author**: nombre del autor cuando está disponible (presente en 387 materiales).  
* **description**: descripción textual cuando está disponible (presente en 572 materiales).  
* **physical\_size**: tamaño físico real del material en centímetros (presente en 358 materiales).

### **1.3 Acceso mediante HuggingFace**

El dataset está alojado en HuggingFace Hub bajo el identificador `gvecchio/MatSynth`. El acceso programático se realiza mediante la librería `datasets` de HuggingFace o mediante descarga directa de los archivos Parquet que componen el split de entrenamiento. El split de entrenamiento se distribuye en 431 archivos Parquet de entre 0.9 GB y 1.3 GB cada uno. Cada archivo contiene entre 8 y 15 materiales con todos sus mapas embebidos en formato binario.

### **1.4 Limitaciones identificadas en el contexto del proyecto**

El análisis exploratorio reveló varias inconsistencias relevantes entre los tags de categoría y el contenido visual real de los materiales:

* La categoría **terracotta** contiene predominantemente ladrillos, no terracota en sentido estricto.  
* La categoría **plaster** incluye mortero limpio, paredes de papel rasgado, mosaicos y superficies con aspecto de madera o ladrillo visible bajo el enlucido.  
* La categoría **stone** es la más heterogénea: incluye pavimentos con adoquines, superficies con vegetación (musgo, hierba), y algunos materiales que presentan aspecto metálico oxidado sin mapa Metallic asociado.  
* La categoría **metal** incluye cotas de malla, cadenas, placas base electrónicas y superficies con aspecto pétreo, además de metales convencionales.

Estas inconsistencias son la justificación técnica principal del pipeline de relabeling con embeddings visuales planificado para la Fase 2\.

---

## **2\. Investigación para el Diseño del EDA**

### **2.1 Validación de normal maps en espacio OpenGL**

La validación matemática de los normal maps se fundamenta en las propiedades geométricas del espacio de coordenadas OpenGL. En este estándar, cada píxel del normal map representa un vector unitario en el espacio tangente de la superficie, con el eje Z (canal azul) apuntando siempre hacia la cámara. Esto implica que, en un normal map correctamente generado, la media del canal azul debe ser significativamente superior a 128 (el punto neutro en el rango \[0, 255\]). Valores por debajo de 160 indican que una fracción importante de los vectores normales apunta en dirección opuesta a la cámara, lo que es físicamente imposible en una superficie convexa y es síntoma de corrupción o de uso de la convención DirectX (donde el canal verde está invertido) \[2\].

El desequilibrio entre los canales rojo (X) y verde (Y) es un indicador complementario de convención incorrecta. En un normal map OpenGL bien calibrado, ambos canales tienen media aproximadamente centrada en 128, lo que resulta en un cociente R/G cercano a 1.0. Desviaciones fuertes (fuera del rango \[0.70, 1.40\]) sugieren inversión de un canal o corrupción del archivo.

La condición de unitariedad del vector (‖N‖ ≈ 1.0) es la tercera propiedad matemática que se verifica. Al mapear los valores de píxel del rango \[0, 255\] al rango \[-1, 1\] mediante la transformación v \= (pixel / 127.5) \- 1, la norma euclidiana de cada vector debería ser próxima a 1.0. Una desviación media superior a 0.30 indica que una proporción significativa de los vectores no son unitarios, lo cual produce errores en el cálculo de la iluminación durante el renderizado \[3\].

### **2.2 Detección de near-duplicates mediante perceptual hashing**

La detección de imágenes near-duplicate en datasets de gran escala es un problema bien estudiado en visión por computador. Los métodos basados en perceptual hashing ofrecen un equilibrio entre velocidad y robustez adecuado para datasets de tamaño medio como el nuestro.

El algoritmo pHash (Perceptual Hash) opera reduciendo la imagen a una representación compacta de 64 bits capturando su estructura visual global mediante la Transformada Discreta del Coseno. Dos imágenes se consideran near-duplicates cuando la distancia de Hamming entre sus hashes es inferior a un umbral predefinido \[4\]. La distancia de Hamming entre dos hashes de 64 bits mide el número de bits que difieren, de forma que un valor de 0 indica imágenes prácticamente idénticas y valores superiores a 10 indican imágenes visualmente distintas.

Para nuestro caso se adoptó un umbral de Hamming de 6, consistente con los valores reportados en la literatura para datasets de texturas con variaciones de color moderadas \[5\]. Este umbral se aplica entre todos los pares del dataset (incluyendo pares entre categorías distintas), dado que MatSynth incluye materiales procedentes de múltiples fuentes que pueden haber descargado el mismo material bajo categorías distintas.

Un benchmark reciente sobre métodos de deduplicación \[5\] demuestra que los métodos de perceptual hashing clásicos (AHash, DHash, PHash, WHash) son significativamente más rápidos que los embeddings CNN para datasets de este tamaño, con una pérdida de precisión aceptable para el caso de near-duplicates no transformados. Para near-duplicates con transformaciones geométricas severas (rotaciones arbitrarias), los embeddings CNN superan al hashing, pero en nuestro contexto ese caso no es prioritario.

La librería `imagededup` \[6\] proporciona una implementación de referencia de estos algoritmos con soporte para Python 3.9+, incluyendo tanto métodos de hashing como CNN-based embeddings, y fue adoptada como base para la implementación del filtro F8 del EDA.

### **2.3 Filtrado por coherencia inter-mapa**

El problema del filtrado por coherencia entre el albedo y el normal map no tiene una solución estándar en la literatura de datasets PBR, dado que la relación entre ambos mapas es semánticamente compleja. Un material como el mármol puede tener un albedo con gran variedad de color pero un normal map con relieve mínimo, mientras que una pared de gotelé puede tener el comportamiento opuesto. Ambos casos son físicamente válidos.

La estrategia adoptada para el filtro F1 ("albedo muerto con relieve fuerte") se basa en el trabajo de limpieza de datasets de imagen propuesto en CleanPatrick \[7\], que formaliza la detección de muestras problemáticas como una tarea de ranking por score de anomalía. En nuestro caso, el score de anomalía se compone de la contribución ponderada de múltiples filtros, cada uno con un peso asignado en función de su fiabilidad como indicador de problema real.

El criterio específico para F1 (std\_rgb \< 5.0 AND std\_normal \> 50.0) surge de la observación empírica del EDA anterior: en el dataset pétreo original, el umbral std\_rgb \< 5 capturaba exactamente los casos de albedo completamente plano (valor único de píxel en todos los canales), mientras que la condición adicional std\_normal \> 50 limita el descarte a los casos donde esa planitud del albedo es incoherente con un relieve geométrico alto. Sin esa segunda condición, se descartan incorrectamente las paredes de gotelé y las superficies de mármol pulido con albedo homogéneo.

### **2.4 Umbrales por categoría**

La principal innovación metodológica del EDA respecto al trabajo previo es la adopción de umbrales específicos por categoría en lugar de umbrales globales. Esta decisión se fundamenta en las distintas propiedades físicas de los materiales:

El modelo PBR Cook-Torrance \[8\], que es el modelo de renderizado que utilizará MatForge, parametriza la apariencia de un material mediante roughness (rugosidad microscópica de la superficie) y metallic (fracción metálica de la superficie). Bajo este modelo, un mármol pulido es físicamente correcto con roughness cercano a 0 (casi especular), mientras que el mismo valor de roughness en un hormigón o una piedra indica un error de captura. Aplicar el mismo umbral de roughness a ambas categorías sería físicamente incorrecto y provocaría el descarte de texturas válidas de marble, exactamente como ocurrió en el EDA anterior.

---

## **3\. Referencias en Formato IEEE**

\[1\] G. Vecchio y V. Deschaintre, "MatSynth: A Modern PBR Materials Dataset," en *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Seattle, WA, USA, 2024\.

\[2\] LearnOpenGL, "Normal Mapping," learnopengl.com. \[En línea\]. Disponible: https://learnopengl.com/Advanced-Lighting/Normal-Mapping. \[Accedido: 24 abr. 2026\].

\[3\] Khronos Group, "glTF 2.0 Specification — Metallic-Roughness Material Model," Khronos Group, 2017\. \[En línea\]. Disponible: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html. \[Accedido: 24 abr. 2026\].

\[4\] B. Hoyt, "Duplicate image detection with perceptual hashing in Python," benhoyt.com. \[En línea\]. Disponible: https://benhoyt.com/writings/duplicate-image-detection/. \[Accedido: 24 abr. 2026\].

\[5\] C. Oprea, I. Florea, C. Florea, y C. Vertan, "Comparative Evaluation of Perceptual Hashing and Deep Embedding Methods for Robust and Efficient Image Deduplication," *Electronics*, vol. 15, no. 7, art. 1493, 2026\. doi: 10.3390/electronics15071493.

\[6\] T. Jain, C. Lennan, Z. John, y D. Tran, "Imagededup," GitHub, 2019\. \[En línea\]. Disponible: https://github.com/idealo/imagededup. \[Accedido: 24 abr. 2026\].

\[7\] J. Ahrendt *et al.*, "CleanPatrick: A Benchmark for Image Data Cleaning," Zenodo, 2025\. doi: 10.5281/zenodo.15591625.

\[8\] R. L. Cook y K. E. Torrance, "A reflectance model for computer graphics," *ACM Transactions on Graphics*, vol. 1, no. 1, pp. 7–24, ene. 1982\. doi: 10.1145/357290.357293.

\[9\] I. Lopes, F. Pizzati, y R. de Charette, "Material Palette: Extraction of Materials from a Single Image," en *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Seattle, WA, USA, 2024\. \[En línea\]. Disponible: https://astra-vision.github.io/MaterialPalette/.

\[10\] P. Kocsis *et al.*, "IntrinsiX: High-Quality PBR Generation using Image Priors," *arXiv preprint arXiv:2504.01008*, 2025\. \[En línea\]. Disponible: https://arxiv.org/abs/2504.01008.

\[11\] R. Martin *et al.*, "Single-input high-resolution PBR material acquisition from outdoor surfaces," en *Computer Graphics Forum*, vol. 41, no. 2, 2022\. doi: 10.1111/cgf.14479.

\[12\] A. Doğan *et al.*, "Effective near-duplicate image detection using perceptual hashing and deep learning," *Information Processing & Management*, vol. 62, no. 3, art. 103648, 2025\. doi: 10.1016/j.ipm.2025.103648.