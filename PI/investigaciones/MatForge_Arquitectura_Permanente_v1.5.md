# MatForge — Documento de Arquitectura Permanente
**Versión**: 1.5 | **Fase actual**: Fases 2, 3 y GAN completadas / Fase 4 pendiente
**Última actualización**: 03/05/2026
**Propósito**: Documento de referencia técnica que debe adjuntarse al inicio de cada sesión de trabajo para garantizar coherencia total entre fases.

---

## 0. CONTEXTO DEL PROYECTO

**Objetivo**: predecir mapas **Normal** (3 canales, espacio OpenGL, rango [-1,1], norma unitaria por píxel), **Roughness** (1 canal, rango [0,1]) y **Metallic** (1 canal, rango [0,1]) a partir de una sola imagen RGB de un material de superficie tileable.

**Uso final**: pipeline PBR en motores 3D. Los mapas predichos deben ser físicamente coherentes, no solo visualmente correctos.

**Baseline previo**: DeepPBR-Net (ResNet50 encoder + dual decoder + CBAM + PatchGAN). Limitaciones conocidas: encoder de clasificación adaptado a tarea densa, GAN prematuro desde época 1, VGG perceptual sobre mapas crudos (no sobre renders), roughness plano en materiales homogéneos, alucinación geométrica tardía.

**MatForge NO es una mejora de DeepPBR. Es un modelo nuevo construido desde cero.**

---

## 1. STACK TÉCNICO

| Componente | Elección | Motivo |
|---|---|---|
| Framework | **PyTorch** | Soporte nativo de MiT-B1 via timm/transformers, AMP nativo, ecosistema de investigación |
| Backbone | **PVT-v2-B1** (Pyramid Vision Transformer v2) | Jerárquico, preentrenado ImageNet, inductive bias denso, ~13M params. *(v1.5: MiT-B1 no disponible en timm 1.0.25; PVT-v2-B1 es arquitectónicamente equivalente con shapes de feature maps idénticos)* |
| Backbone source | `timm.create_model('pvt_v2_b1', pretrained=True)` | Pesos oficiales disponibles en timm 1.0.25 |
| Decoder | **FPN-style custom** con skip connections en 4 escalas | Fusión multiescala top-down |
| Cabezas de salida | **Tres ramas independientes** con refine heads | Evita task interference entre Normal, Roughness y Metallic |
| Precisión | **AMP (torch.cuda.amp)** | Reduce VRAM ~50%, acelera ~30% en Tensor Cores |
| Optimizador | **AdamW** | Weight decay desacoplado, estándar en transformers |
| Interfaz final | **Streamlit** | Requerimiento del proyecto |
| Hardware objetivo | **Kaggle (T4 16GB / P100 16GB)** | Restricción real del proyecto |

---

## 2. ARQUITECTURA DETALLADA DE MATFORGE

### 2.1 Diagrama de flujo completo

```
[Entrada: RGB 256×256×3]
          │
    ┌─────▼──────┐
    │  PVT-v2-B1 │  ← Encoder jerárquico preentrenado (CONGELADO primeras 2-3 épocas)
    │  Encoder   │
    └─────┬──────┘
          │ produce 4 feature maps:
          ├── L1: 64×64×64    (1/4 resolución)   ─────────────────────────────────┐
          ├── L2: 32×32×128   (1/8 resolución)   ──────────────────────┐          │
          ├── L3: 16×16×320   (1/16 resolución)  ──────────┐           │          │
          └── L4:  8×8×512    (1/32 resolución)  ──┐        │           │          │
                                                    │        │           │          │
    ┌───────────────────────────────────────────────▼────────▼───────────▼──────────▼──┐
    │                         FPN Decoder (top-down)                                    │
    │  8×8 → [conv] → 16×16 + L3 → [conv] → 32×32 + L2 → [conv] → 64×64 + L1        │
    └──────────────────────────────────────────────────────────────────────────────┬───┘
                                                                                   │
                                                                        [64×64 features]
                                                                      /       |        \
                                                               [Normal]  [Roughness]  [Metallic]
                                                                   ↓          ↓            ↓
                                                               refine 128  refine 128  refine 128
                                                                   ↓          ↓            ↓
                                                               refine 256  refine 256  refine 256
                                                                   ↓          ↓            ↓
                                                               3ch+Tanh  1ch+Sigmoid  1ch+Logits
                                                               +renorm L2             →sigmoid en
                                                                                       inferencia
```

### 2.2 Especificaciones de cada bloque

#### Encoder PVT-v2-B1 *(actualizado v1.5: sustituye a MiT-B1)*
- **Parámetros**: ~13M
- **Preentrenamiento**: ImageNet-1K (via timm 1.0.25)
- **Motivo del cambio**: `mit_b1` no está registrado en timm 1.0.25 (0 modelos encontrados). PVT-v2-B1 produce shapes de feature maps idénticos: L1(64×64×64), L2(32×32×128), L3(16×16×320), L4(8×8×512). El FPN decoder y las tres heads no requirieron ningún cambio.
- **Estrategia de fine-tuning**: congelar los primeros 1-2 stages durante las primeras **2-3 épocas**, después descongelar todo.
- **Inductive bias**: ventanas de atención locales + MLP feedforward; jerárquico sin embeddings posicionales rígidos
- **⚠️ RESTRICCIÓN IMPORTANTE**: el tamaño de entrada debe ser múltiplo de 32 (256px y 320px lo cumplen)

#### FPN Decoder
- Fusión top-down: desde L4 (8×8) hacia L1 (64×64)
- En cada nivel: upsample bilinear 2× + concatenación con skip connection + Conv 3×3 + BN + ReLU
- Dimensión de canales de trabajo: 256 en todos los niveles (proyección 1×1 antes de fusionar)
- **Parámetros estimados**: ~4-6M (requiere validación con dry run)

#### Refine Heads (por rama)
- Dos bloques: uno en 128×128 y otro en 256×256
- Cada bloque: [Upsample bilinear 2×] → [Conv 3×3, 128ch] → [BN] → [ReLU] → [Conv 3×3, 64ch] → [BN] → [ReLU]
- Capa final de cada rama:
  - **Normal**: Conv 1×1 → 3 canales → Tanh → **renormalización L2 por píxel** (OBLIGATORIO)
  - **Roughness**: Conv 1×1 → 1 canal → Sigmoid, rango [0,1]
  - **Metallic**: Conv 1×1 → 1 canal → **sin activación final (logits)**. La cabeza emite logits que se pasan a `BCEWithLogitsLoss` durante el entrenamiento; en inferencia se aplica `sigmoid` + clip a [0,1]. *(Actualizado v1.4: cambio de Sigmoid + Charbonnier a logits + WBCE por desbalance 238/3007)*. Para materiales pétreos el GT es siempre 0 (negro uniforme). Para la categoría metal, el GT tiene varianza real y es donde esta cabeza aporta valor. Parámetros adicionales estimados: ~0.5M.
- **Parámetros estimados por rama**: ~1.5-2.5M

#### Tamaño total estimado del modelo
~20-23M parámetros (con la tercera cabeza). **Esta estimación requiere validación con dry run.** Si supera 25M, revisar la anchura del FPN decoder.

---

## 3. FUNCIÓN DE PÉRDIDA

### 3.1 Formulación completa

```
L_total = α·L_normal + β·L_roughness + ζ·L_metallic + γ·L_grad + δ·L_render
```

#### L_normal (pérdida de normales)
```
L_normal = 1.0 · L_coseno(N_pred, N_gt) + 0.25 · L_charbonnier(N_pred, N_gt)
```
- **L_coseno**: `1 - mean(dot(N_pred, N_gt))` donde N_pred y N_gt son vectores renormalizados. Mide desviación angular.
- **L_charbonnier**: `mean(sqrt((N_pred - N_gt)² + ε²))` con ε=0.001. Regularización suave de valores absolutos.

#### L_roughness (pérdida de roughness)
```
L_roughness = 1.0 · L_charbonnier(R_pred, R_gt)
```
Solo Charbonnier para roughness. No coseno (roughness es escalar, no vector).

#### L_metallic (pérdida de metallic)
```
L_metallic = 0.5 · BCEWithLogitsLoss(M_logits_pred, M_gt_binary, pos_weight=8.0)
```
*(Actualizado v1.4: sustituye Charbonnier por BCEWithLogitsLoss con pos_weight=8.0)*

**Justificación del cambio**: con 238 ejemplos positivos (metal) frente a 3.007 negativos, Charbonnier no tiene mecanismo para compensar el desbalance; los miles de negativos triviales dominan y la cabeza colapsa a predecir cero. `BCEWithLogitsLoss` con `pos_weight=w_p` escala la contribución de los positivos en `w_p` veces, corrigiendo estructuralmente el desbalance. Valor de partida: `pos_weight=8.0` (no `12.6` —el ratio exacto de píxeles— porque un peso demasiado alto dispara falsos positivos). Si en las primeras 3-5 épocas la cabeza sigue colapsando a cero, activar `FocalLoss(gamma=2)` como contingencia.

La cabeza emite **logits** (sin sigmoid final). El sigmoid se aplica solo en inferencia.

**Síntomas de colapso a monitorizar desde época 1**:
- Predicción media de la cabeza Metallic en texturas de metal < 0.1
- Recall en grupo `metal` en validación < 0.20

#### L_grad (pérdida de gradiente multiescala)
```
L_grad = 0.20 · mean(|∇N_pred - ∇N_gt| + |∇R_pred - ∇R_gt|)
```
- Calcula gradientes en X e Y con filtros Sobel
- Se aplica en 2 escalas: resolución original y mitad de resolución
- **⚠️ RIESGO**: si el peso es demasiado alto, introduce ruido falso en materiales ambiguos. Empezar con 0.20 y reducir si aparecen artefactos de alta frecuencia.

#### L_render (re-render loss / pérdida física)
```
render_pred = Cook_Torrance(albedo_GT, N_pred, R_pred, luz_aleatoria)
render_gt   = Cook_Torrance(albedo_GT, N_gt,   R_gt,   luz_aleatoria)
L_render = 0.20 · L1(render_pred, render_gt) + 0.05 · LPIPS(render_pred, render_gt)
```
- Se usan 2-4 direcciones de luz aleatorias por batch
- El albedo GT viene del dataset (no se predice)
- **⚠️ RIESGO CRÍTICO**: si el renderer no está calibrado correctamente, los gradientes pueden explotar o ser nulos. Validar el renderer de forma aislada ANTES de integrarlo en el entrenamiento.
- LPIPS se calcula sobre renderizados, NO sobre los mapas PBR crudos.

### 3.2 Pesos de la pérdida compuesta y calendario de activación

*(Actualizado v1.4: activación progresiva del render loss; metallic cambia a WBCE)*

| Componente | Épocas 1-5 | Épocas 6-15 | Épocas 16-90 | Notas |
|---|---|---|---|---|
| L_normal | α = 1.0 | α = 1.0 | α = 1.0 | Ancla principal del entrenamiento |
| L_roughness | β = 0.8 | β = 0.8 | β = 0.8 | Ligeramente inferior para equilibrar escala |
| L_metallic | ζ = 0.5 | ζ = 0.5 | ζ = 0.5 | WBCE con pos_weight=8.0; logits como entrada |
| L_grad | γ = 0.15 | γ = 0.15 | γ = 0.15 | 2 escalas Sobel; reducir si artefactos |
| L_render L1 | δ₁ = **0.00** | δ₁ = **0.10** | δ₁ = **0.15** | Activación progresiva — crítico |
| L_render LPIPS | δ₂ = **0.00** | δ₂ = **0.00** | δ₂ = **0.03** | Solo en fase avanzada |

**Justificación de la activación progresiva del render loss**: si L_render se activa desde época 1, puede dominar sobre L_normal antes de que el modelo haya aprendido orientación angular básica. El diagnóstico correcto es monitorizar el ratio `g_render / g_normal` sobre los parámetros del encoder+FPN cada ~200 steps. **Alerta temprana**: ratio > 0.5 sostenido en épocas 1-5. **Alerta dura**: ratio > 1.0 durante > 10% de los puntos medidos.

**Estos pesos son orientativos y REQUIEREN validación experimental.** Son el punto de partida, no valores fijos.

### 3.3 GAN / Discriminador
**NO se usa en la primera corrida de entrenamiento.** Si se añade en iteraciones futuras: activar solo después de época 50, con peso inicial ≤ 0.1, usando PatchGAN de 70×70. Nunca como componente dominante.

---

## 4. ESTRATEGIA DE ENTRENAMIENTO

### 4.1 Hiperparámetros

| Parámetro | Valor | Justificación |
|---|---|---|
| Optimizador | AdamW | Estándar para transformers |
| LR encoder | 1e-4 | Más bajo porque viene preentrenado |
| LR decoder + heads | 3e-4 | Mayor libertad para adaptarse |
| Weight decay | 1e-2 | Regularización estándar AdamW |
| Batch size | 8-10 con AMP | Límite de VRAM en T4/P100 |
| Gradient accumulation | si batch efectivo < 8 | Simular batches más grandes |
| Scheduler | Cosine decay | Decaimiento suave y estable |
| Warmup | 5 épocas | Estabiliza el inicio del entrenamiento |
| EMA | 0.999 — **activar al descongelar el encoder** (época 3-4) | Activar con encoder congelado tiene poco efecto real; ventana efectiva ~1000 pasos |
| Total épocas | 90 | Estimación; ajustar según señal temprana |
| Freeze encoder inicial | **2-3 épocas** (stages 1-2) | *Actualizado v1.4*: 5 épocas sin respaldo empírico; 2-3 protegen el preentrenamiento y permiten al encoder vivir parte del warmup |
| Semilla | 42 | Reproducibilidad |

### 4.2 Curriculum de resolución

| Fase | Épocas | Resolución de crop | Motivo |
|---|---|---|---|
| Fase principal | 0 → 65 | 256×256 | Velocidad de iteración |
| Fase final | 65 → 90 | 320×320 | Mejora de microdetalle |
| Si T4 se queda sin memoria | — | Quedarse en 256 todo | Seguridad > detalle |
| NO hacer en entrenamiento | — | 512×512 | Fuera de presupuesto de VRAM |

### 4.3 Checkpoints y validación

**Checkpoints automáticos** (no requieren intervención):
- Cada 5 épocas: `checkpoint_ep{N}.pt`
- Mejor overall: `best_overall.pt` (criterio: 0.45·normal + 0.35·roughness + 0.20·render)
- Mejor normal: `best_normal.pt`
- Mejor roughness: `best_roughness.pt`
- Último estado: `last.pt`

**Validación visual automática** (no requiere intervención inmediata):
- Panel fijo de 12 materiales de validación
- Inferencia a resolución completa 1K con Hann blending cada 10 épocas
- Guardado automático de imágenes para revisión posterior

**Revisión humana requerida solo en**:
1. Dry run inicial (épocas 1-5): confirmar que el entrenamiento arranca limpio
2. Época ~20: aplicar criterio de pivot (ver sección 4.4)

### 4.4 Criterio de pivot y parada temprana

*(Actualizado v1.4: criterio cuantitativo añadido)*

**Criterio de pivot (época 20)**: si NO se observa mejora clara en **al menos 2 de estas 3 dimensiones** respecto a DeepPBR:
- Detalle en normales: MAE angular en validación
- Coherencia en roughness: MAE roughness en validación
- Calidad del renderizado: LPIPS sobre renders en validación

→ **Cambiar a Plan B (Restormer dual-head) sin extender Plan A.**

**Criterio de parada temprana cuantitativo** (métrica compuesta `S`):
```
S = MAE_normal_deg + 0.6 · MAE_roughness + 0.2 · LPIPS_render
```
- Guardar el mejor checkpoint por `S`
- Paciencia: 8 validaciones consecutivas sin mejora en ninguno de estos umbrales:
  - 0.25° en MAE angular de normales, **O**
  - 0.005 en LPIPS render, **O**
  - 2% relativo en MAE roughness
- Si ninguno de los tres mejora en 8 validaciones: corrida agotada → revisar pesos de loss o pivotar

---

## 5. MÉTRICAS DE EVALUACIÓN

*(Actualizado v1.4: protocolo cuantitativo completo con tabla comparativa MatForge vs. DeepPBR)*

| Métrica | Mapa | Qué mide | Fórmula |
|---|---|---|---|
| **MAE Angular (grados)** | Normal | Desviación angular media — métrica principal | `mean(arccos(clamp(N̂_pred · N̂_gt, -1+ε, 1-ε))) · 180/π` |
| Cosine Distance | Normal | Alternativa a MAE angular, escala [0,2] | `1 - mean(N̂_pred · N̂_gt)` |
| MAE | Roughness | Error de reconstrucción | `mean(|R_pred - R_gt|)` |
| RMSE | Roughness | Error cuadrático medio | `sqrt(mean((R_pred - R_gt)²))` |
| LPIPS (alex) | Renderizados | Calidad perceptual bajo iluminación | Sobre renders Cook-Torrance con 3-5 luces fijas |
| SSIM | Renderizados | Similitud estructural | Sobre renders Cook-Torrance |
| BCE / Recall | Metallic | Detección de superficie metálica | Recall en grupo `metal` de validación |

**Implementación PyTorch de MAE Angular**:
```python
def mean_angular_error_deg(pred, target, eps_norm=1e-6, eps_acos=1e-7):
    pred_n  = F.normalize(pred,   dim=1, eps=eps_norm)
    tgt_n   = F.normalize(target, dim=1, eps=eps_norm)
    dot = (pred_n * tgt_n).sum(dim=1).clamp(min=-1.0 + eps_acos, max=1.0 - eps_acos)
    return (torch.acos(dot) * (180.0 / math.pi)).mean()
```

**Protocolo de evaluación cuantitativa MatForge vs. DeepPBR**:

Disponemos de pesos entrenados de DeepPBR y de su dataset original. La comparación debe incluir dos tablas:

- **Tabla A** (obligatoria): evaluar MatForge y DeepPBR sobre el 15% de validación de MatForge, con el mismo preprocesado e inferencia.
- **Tabla B** (recomendada): evaluar ambos sobre el dataset de validación original de DeepPBR, para demostrar que MatForge no pierde generalización en el dominio anterior.

Para la comparación de renders entre modelos: usar albedo GT y metallic GT para ambos modelos, comparando solo el efecto de sus Normal/Roughness sobre el render (no penalizar a DeepPBR por no tener cabeza Metallic).

| Modelo | Dataset eval | Normal MAE ↓ | Cos dist ↓ | Roughness MAE ↓ | Roughness RMSE ↓ | Render LPIPS ↓ | Render SSIM ↑ | Metallic Recall ↑ |
|---|---|---|---|---|---|---|---|---|
| DeepPBR | MatForge-val | | | | | | | N/A |
| MatForge | MatForge-val | | | | | | | |
| DeepPBR | DeepPBR-val | | | | | | | N/A |
| MatForge | DeepPBR-val | | | | | | | |

---

## 6. ESTRATEGIA DE DATASET

### 6.1 Estado actual del dataset (post-Fase 1b, 28/04/2026)

```
Dataset local (misma copia subida a Kaggle):
maps/
├── rgb/
│   ├── stone_XXXX.png        (733 texturas)
│   ├── wood_XXXX.png         (548 texturas)
│   ├── ceramic_XXXX.png      (520 texturas)
│   ├── terracotta_XXXX.png   (311 texturas)
│   ├── metal_XXXX.png        (358 texturas)
│   ├── concrete_XXXX.png     (255 texturas)
│   ├── ground_XXXX.png       (207 texturas)
│   ├── plaster_XXXX.png      (172 texturas)
│   └── marble_XXXX.png       (141 texturas)
├── normal/     (misma estructura de nombres)
├── roughness/  (misma estructura de nombres)
└── metallic/   (solo para categoría metal, 358 archivos)
relabeling/
├── relabeling_final.csv
├── sampler_weights.json
├── knn_classifier.pkl
├── pca_model.pkl
└── label_encoder.pkl

TOTAL LIMPIO: 3.245 texturas
```

**Rutas en Kaggle** (dataset: `MatForge PBR Dataset`, privado):
```
/kaggle/input/matforge-pbr-dataset/maps/rgb/
/kaggle/input/matforge-pbr-dataset/maps/normal/
/kaggle/input/matforge-pbr-dataset/maps/roughness/
/kaggle/input/matforge-pbr-dataset/maps/metallic/
/kaggle/input/matforge-pbr-dataset/relabeling/relabeling_final.csv
/kaggle/input/matforge-pbr-dataset/relabeling/sampler_weights.json
/kaggle/input/matforge-pbr-dataset/relabeling/knn_classifier.pkl
/kaggle/input/matforge-pbr-dataset/relabeling/pca_model.pkl
/kaggle/input/matforge-pbr-dataset/relabeling/label_encoder.pkl
```

### 6.2 Historial de construcción del dataset

| Etapa | Texturas | Descripción |
|---|---|---|
| Dataset original (DeepPBR) | ~1.603 | Solo dominio pétreo, tags MatSynth originales |
| Post-descarga v2.0 | 3.814 | Añadidos wood, metal, ceramic, ground. Soporte Metallic |
| Post-limpieza manual pre-EDA | ~3.809 | Eliminadas placas base (PCBs) en categoría metal |
| Post-EDA | 3.245 | Filtrado semi-automático + revisión humana confirmada |
| **Post-relabeling (dataset final)** | **3.245** | Grupos funcionales asignados, pesos calculados, subido a Kaggle |

### 6.3 Resultado del EDA por categoría (tags originales MatSynth)

| Categoría | Pre-EDA | Post-EDA | Descartadas | % descarte |
|---|---|---|---|---|
| stone | 800 | 733 | 67 | 8.4% |
| wood | 600 | 548 | 52 | 8.7% |
| ceramic | 582 | 520 | 62 | 10.7% |
| terracotta | 319 | 311 | 8 | 2.5% |
| metal | 560 | 358 | 202 | 36.1% |
| concrete | 278 | 255 | 23 | 8.3% |
| ground | 263 | 207 | 56 | 21.3% |
| marble | 142 | 141 | 1 | 0.7% |
| plaster | 265 | 172 | 93 | 35.1% |
| **TOTAL** | **3.809** | **3.245** | **564** | **14.8%** |

### 6.4 Resultado del relabeling por grupo funcional (28/04/2026)

Pipeline: DINOv2-small (ViT-S/14, embeddings 384D, resolución entrada 518×518) → PCA (384D→50D, varianza explicada 82.3%) → UMAP clustering (50D→15D) → HDBSCAN (37 clusters brutos, DBCV=0.3279, ruido 15.1%) → fusión manual a 8 grupos funcionales.

| Grupo funcional | Texturas | Peso sampler | Rol en el modelo |
|---|---|---|---|
| `stone_rough` | 479 | 1.0 | Dominio principal |
| `wood` | 658 | 1.0 | Vacuna de dominio |
| `ceramic_ground` | 503 | 1.0 | Vacuna de dominio |
| `mixed_ambiguous` | 775 | 0.5 | Heterogéneo (23.9% del total) |
| `brick_terracotta` | 276 | 1.0 | Dominio principal |
| `marble_smooth` | 189 | 1.2 | Dominio principal (upweight por escasez) |
| `metal` | 238 | 1.3 | Vacuna de dominio (upweight por señal metallic escasa) |
| `concrete_plaster` | 127 | 1.0 | Dominio principal (grupo más pequeño — monitorizar) |
| **TOTAL** | **3.245** | — | — |

**Decisiones críticas del relabeling**:
- El entrenamiento es **no condicionado por etiqueta de grupo**: MiT-B1 infiere el tipo de material de la imagen. Las etiquetas solo se usan para el sampler balanceado.
- `mixed_ambiguous` con 775 texturas (23.9%) refleja heterogeneidad real del dataset, no un fallo del clustering. Si el modelo muestra degradación en grupos específicos, el primer ajuste es bajar el peso de `mixed_ambiguous` a 0.3.
- `force_assign_marble` no fue necesario: el clustering asignó 189 texturas a `marble_smooth`, superando las 141 originales.
- `concrete_plaster` con solo 127 texturas es el grupo de dominio principal más pequeño. Monitorizar sus métricas de validación específicamente.
- El clasificador KNN (k=7, coseno, distance weights) entrenado sobre embeddings DINOv2 post-PCA se integrará en el pipeline Streamlit para identificar automáticamente el grupo de material en inferencia.

### 6.5 Filtros del EDA aplicados

| Filtro | Descripción | Criterio |
|---|---|---|
| F1 | Albedo muerto + relieve fuerte | std_rgb < 5.0 AND std_normal > 50.0 |
| F2 | Canal Z del normal bajo | media_azul < umbral por categoría (150-160) |
| F3 | Desequilibrio R/G del normal | ratio R/G fuera de [0.70, 1.40] |
| F4 | Vectores normales no unitarios | desviación media de la norma > 0.30 |
| F5 | Ruido extremo en normal | std_normal > umbral por categoría (80-90) |
| F6 | Roughness completamente plano | std_rough < umbral Y media fuera de rango válido por categoría |
| F7 | Metallic todo blanco en metal | std_metallic < 2.0 AND media_metallic > 240 |
| F8 | Near-duplicates | distancia Hamming pHash ≤ 6 en todo el dataset |

### 6.6 Problemas conocidos residuales (post-EDA)

- Texturas giradas (no alineadas con los ejes): mitigadas con augmentaciones de rotación 0°/90°/180°/270°.
- Texturas fuera de su categoría semántica original: tratadas por el relabeling con DINOv2.
- Roughness con patrones inusuales pero físicamente válidos (humedad, manchas): conservados.
- Texturas de concrete con barras de hierro oxidado visibles: conservadas (aportan variedad).
- Texturas oxidadas en stone: conservadas (dieléctricos válidos, metallic=0).

### 6.7 Split del dataset
- **Train**: 85% de cada grupo funcional
- **Validación**: 15% de cada grupo funcional (fijo, no cambia entre experimentos)
- **Panel visual fijo**: 12 materiales de validación seleccionados manualmente (2 por grupo principal)

### 6.8 Data augmentation *(Añadido v1.4)*

**Pipeline base de augmentación (obligatorio desde época 1)**:

| Augmentación | Aplicar a | Notas |
|---|---|---|
| RandomCrop(256) | RGB + todos los GTs | Coordenadas idénticas para todos los canales |
| Flip horizontal (p=0.5) | RGB + Normal (transform.) + Roughness + Metallic | Ver tabla de transformación Normal |
| Flip vertical (p=0.5) | RGB + Normal (transform.) + Roughness + Metallic | Ver tabla de transformación Normal |
| Rotaciones 0/90/180/270° (p=0.25 c/u) | RGB + Normal (transform.) + Roughness + Metallic | Ver tabla de transformación Normal |
| Jitter fotométrico muy suave | **Solo RGB de entrada** | Brillo ±8%, contraste ±8%, saturación ±5%, hue ±2° |
| Blur gaussiano ligero (p=0.1) | **Solo RGB de entrada** | No tocar GTs |
| Ruido gaussiano ligero (p=0.1) | **Solo RGB de entrada** | No tocar GTs |

**Transformaciones obligatorias del Normal map en espacio tangente OpenGL (+Z saliente)**:

Con `N = (X, Y, Z)` en rango [-1, 1]:

| Transformación de imagen | Transformación del Normal |
|---|---|
| Flip horizontal | `(-X, Y, Z)` |
| Flip vertical | `(X, -Y, Z)` |
| Rotación 90° CCW | `(-Y, X, Z)` |
| Rotación 180° | `(-X, -Y, Z)` |
| Rotación 270° CCW | `(Y, -X, Z)` |

⚠️ **CRÍTICO**: aplicar la transformación geométrica del tensor Normal Y la transformación de componentes. Si solo se hace la transformación espacial, la orientación del vector queda físicamente incorrecta.

**Lo que NO se usa**:
- Rotaciones arbitrarias (degradación por interpolación + anisotropía en `wood`)
- CutMix / MixUp (mezclas físicamente incoherentes entre mapas PBR)
- Jitter de color sobre los GTs (Normal, Roughness, Metallic nunca se perturban con jitter)

---

## 7. SISTEMA DE INFERENCIA

Sistema heredado de DeepPBR con ajustes mínimos. **Reutilizable directamente.**

1. **Padding por reflexión** en los bordes de la imagen de entrada
2. **División en parches de 256×256** con stride 128 (solapamiento 50%)
3. **Procesamiento independiente** de cada parche por MatForge
4. **Fusión con ventana de Hann**: `w(x,y) = sin²(πx/(N-1)) · sin²(πy/(N-1))`
5. **Acumulación y normalización** por matriz de pesos acumulados

**Resultado**: mapas de Normal, Roughness y Metallic a la resolución original de la imagen de entrada, sin costuras.

**Pipeline de identificación de material en Streamlit**:
```
[Imagen usuario] → [DINOv2-small CLS token 384D]
                 → [PCA serializado: 384D → 50D]
                 → [KNN classifier: 50D → grupo funcional]
                 → ["Textura detectada: marble_smooth / wood / metal..."]
                 → [MatForge: Normal + Roughness + Metallic]
```
Artefactos necesarios: `knn_classifier.pkl`, `pca_model.pkl`, `label_encoder.pkl` (todos en `/kaggle/input/matforge-pbr-dataset/relabeling/`).

**Normalización de entrada** *(Añadido v1.4)*:
- Usar la normalización resuelta por `timm.data.resolve_model_data_config(encoder)`, que coincide con el régimen ImageNet esperado por MiT-B1. **No cambiar a media/std del dataset propio** salvo como ablación posterior — la estabilidad del preentrenamiento vale más que una ganancia marginal no confirmada.

**Orden correcto de Hann blending para Normal maps** *(Añadido v1.4)*:
1. Predecir el parche (Normal sin renormalizar post-blend)
2. **Acumular vectores normales con pesos Hann** (NO renormalizar por parche antes de acumular)
3. Dividir por suma de pesos
4. **Renormalizar L2 por píxel al final** sobre el mapa fusionado completo

⚠️ El promedio ponderado de vectores unitarios NO es unitario. Renormalizar antes de acumular produce un campo incorrecto en las costuras. Para Roughness y Metallic (mapas escalares): acumular con pesos Hann y clip a [0,1] al final, sin renormalización vectorial.

---

## 8. RIESGOS CONOCIDOS Y PLANES DE MITIGACIÓN

### R1 — Roughness plano en materiales homogéneos
- **Probabilidad**: Alta
- **Síntoma**: roughness con desviación estándar mucho menor que GT en validación
- **Mitigación**: L_grad con peso moderado + verificar que L_render tiene suficiente señal diferenciadora
- **Si persiste**: añadir término de varianza en la pérdida de roughness

### R2 — Coste del decoder custom mayor que el estimado
- **Probabilidad**: Media
- **Síntoma**: OOM en Kaggle o tiempos por época > 15 min
- **Mitigación**: dry run de 3-5 épocas ANTES del entrenamiento completo
- **Si ocurre**: reducir canales del FPN de 256 a 128

### R3 — Re-render loss con gradientes inestables *(Actualizado v1.4)*
- **Probabilidad**: Media-alta
- **Síntoma**: pérdida total oscila o explota después de introducir L_render
- **Mitigación**: validar renderer de forma aislada antes de integrarlo; usar gradient clipping (max_norm=1.0); **activación progresiva** (L_render desactivado épocas 1-5, ver sección 3.2); 3 luces aleatorias por batch con cos(θ) en [0.25, 0.92].
- **Cuatro puntos críticos de gradiente en el renderer** (proteger con `clamp`):
  1. Renormalización de normales: `eps=1e-6` en `safe_normalize`
  2. Normalización del half-vector H: `eps=1e-6`
  3. Denominador NDF GGX: `clamp(min=1e-6)` sobre `π·denom²`
  4. Denominador specular `4·NoL·NoV`: `clamp(min=1e-4)`
- **Si persiste**: desactivar L_render y sustituir por SSIM directo sobre mapas

### R4 — Overfitting al dominio pétreo
- **Probabilidad**: Media
- **Síntoma**: buenas métricas en stone/concrete, malas en marble/wood en validación
- **Mitigación**: sampler balanceado por grupo funcional (obligatorio desde época 1)

### R5 — Normal map no renormalizado
- **Probabilidad**: Baja, pero impacto alto
- **Síntoma**: renders incorrectos en motor 3D aunque visualmente parezcan bien
- **Mitigación**: renormalización L2 por píxel OBLIGATORIA en la capa de salida de la rama Normal. Verificar con `assert (norma - 1.0).abs().max() < 0.01` en validación

### R6 — GAN fine-tuning *(actualizado v1.5 con resultados reales)*
- **Estado**: ejecutado. 20 épocas de GAN fine-tuning completadas desde checkpoint ep89.
- **Resultado**: el discriminador colapsó a D(real)≈D(fake)≈0.5 desde época GAN 1 (incapaz de distinguir real de fake), pero la **feature matching loss** (W_FM=10.0) continuó aportando señal perceptual útil incluso con discriminador inútil.
- **Mejoras obtenidas respecto al supervisado ep89**:
  - LPIPS: 0.1094 → **0.0976** (−10.8%) ✅
  - MAE Normal: 10.45° → **10.37°** (−0.08°) ✅
  - Roughness MAE: 0.1087 → 0.1117 (+0.003) ⚠️ leve retroceso
- **Checkpoint final**: `best_gan.pt` (época GAN 11, S=10.457)
- **Causa del colapso del discriminador**: el generador maduro (90 épocas supervisadas) supera inmediatamente al discriminador inicializado desde cero. Spectral normalization + dropout pueden reducir excesivamente la capacidad del discriminador contra un generador competente. No se investigará más dado el tiempo restante.
- **Diagnóstico previo resuelto**: bug de NaN en R1 gradient penalty bajo AMP corregido forzando float32 dentro de `r1_gradient_penalty` con `torch.amp.autocast("cuda", enabled=False)`.
- **Decisión**: aceptar `best_gan.pt` como checkpoint final. La mejora de LPIPS es real y cuantificable.

### R7 — Gradient loss amplificando ruido en materiales ambiguos
- **Probabilidad**: Media
- **Síntoma**: artefactos de alta frecuencia similares a los de DeepPBR en época 158+
- **Mitigación**: peso L_grad máximo 0.20; si aparecen artefactos, bajar a 0.10 antes de tocar otra pérdida

### R8 — Colapso de la cabeza Metallic a predecir cero constante *(Actualizado v1.4)*
- **Probabilidad**: Alta si no se gestiona correctamente
- **Contexto**: 238 texturas metal vs. 3.007 no-metal (ratio ~1:12.6). El sampler con peso x1.3 es **insuficiente** (mueve presencia de metal del 7.3% al 9.3% — corrección cosmética, no estructural).
- **Síntoma**: predicción media de la cabeza Metallic en texturas de metal < 0.1; Recall en grupo `metal` < 0.20 en validación.
- **Mitigación primaria**: sampler garantizado por batch (2 metal + 6 no-metal en batch de 8) + `BCEWithLogitsLoss(pos_weight=8.0)`.
- **Mitigación de contingencia**: si colapso persiste en épocas 3-5, activar `FocalLoss(gamma=2)`.
- **Si persiste tras ambas**: reducir ζ de L_metallic a 0.2 y reportar metallic como resultado secundario en la memoria.

### R9 — mixed_ambiguous degradando grupos específicos *(detectado en relabeling)*
- **Probabilidad**: Media
- **Contexto**: 775 texturas (23.9% del total) en `mixed_ambiguous`. Con peso 0.5, siguen representando ~12% efectivo del muestreo.
- **Síntoma**: métricas de validación peores de lo esperado en grupos de dominio principal, especialmente `concrete_plaster` (127 texturas).
- **Mitigación**: si se detecta en la época 20, reducir peso de `mixed_ambiguous` a 0.3 en `sampler_weights.json` y recargar el DataLoader.

---

## 9. PLAN B (RESTORMER DUAL-HEAD)

Solo activar si Plan A no supera claramente a DeepPBR en época 20.

- **Modelo**: Restormer completo con stem compartido + bifurcación tardía en último tercio del decoder
- **Hiperparámetros**: AdamW, LR 2e-4-3e-4, batch 6-8, mismo curriculum de resolución
- **Misma pila de pérdidas**: para que la comparación A/B sea limpia
- **Tiempo estimado**: 10-16h en P100 / 12-18h en T4
- **Riesgo principal**: más sensible a la calidad del dato que Plan A; overfitting más probable si el dataset no está bien limpio

---

## 10. FASES DEL PROYECTO Y ESTADO ACTUAL

| Fase | Descripción | Estado |
|---|---|---|
| **Fase 0** | Consolidación, diagnóstico y estrategia | ✅ CERRADA — 22/04/2026 |
| **Fase 1** | Descarga del dataset y EDA | ✅ CERRADA — 26/04/2026 |
| **Fase 1b** | Relabeling DINOv2 + split train/val + subida a Kaggle | ✅ CERRADA — 28/04/2026 |
| **Fase 2** | Implementación de arquitectura y renderer | ✅ CERRADA — 01/05/2026 |
| **Fase 3** | Entrenamiento supervisado 90 épocas + GAN fine-tuning 20 épocas | ✅ CERRADA — 02/05/2026 |
| **Fase 4** | Evaluación cuantitativa MatForge vs. DeepPBR | 🔄 SIGUIENTE |
| **Fase 5** | Integración Streamlit y preparación de entrega | ⏳ Pendiente |

---

## 11. DECISIONES CERRADAS (NO REABRIR SIN JUSTIFICACIÓN TÉCNICA)

| Decisión | Valor elegido |
|---|---|
| Framework | PyTorch |
| Encoder | MiT-B1 (timm) |
| Decoder | FPN custom + tres ramas + refine heads |
| Cabezas de salida | Tres: Normal, Roughness, Metallic |
| Modelo separado por familia | NO (modelo unificado) |
| GAN en primera corrida | NO |
| VGG sobre mapas crudos | NO |
| Resolución de entrenamiento base | 256px |
| Resolución máxima de entrenamiento | 320px (fase final) |
| Inferencia | Hann blending por parches 256×256 con stride 128 |
| Nombre del modelo | **MatForge** |
| Umbrales del EDA | Por categoría (no globales) |
| Texturas oxidadas en stone | Conservadas (dieléctricos válidos, metallic=0) |
| Placas base en metal | Eliminadas manualmente (pre-EDA) |
| Near-duplicates | Filtrado con pHash, umbral Hamming ≤ 6, entre todas las categorías |
| Loss de metallic | `BCEWithLogitsLoss(pos_weight=8.0)` — NO Charbonnier *(v1.4)* |
| Sampler metal | Batch garantizado: 2 metal + 6 non-metal (batch=8) — NO solo x1.3 *(v1.4)* |
| Activación L_render | Progresiva: desactivado épocas 1-5, parcial 6-15, completo 16+ *(v1.4)* |
| Freeze encoder | 2-3 épocas (NO 5 épocas) *(v1.4)* |
| EMA inicio | Al descongelar el encoder (época 3-4), NO desde época 1 *(v1.4)* |
| Normalización entrada encoder | Media/std ImageNet via `resolve_model_data_config` — NO del dataset propio *(v1.4)* |
| Hann blending en Normal | Acumular → dividir → renormalizar L2; NUNCA renormalizar por parche antes *(v1.4)* |
| Augmentación geométrica | Flip H/V y rotaciones 0/90/180/270° con transformación coherente del Normal map *(v1.4)* |
| CutMix / MixUp | NO — incoherentes físicamente para mapas PBR *(v1.4)* |
| Rotaciones arbitrarias | NO en corrida principal *(v1.4)* |
| LPIPS backbone | `alex` — mejor coste-rendimiento en Kaggle *(v1.4)* |
| Encoder real utilizado | `pvt_v2_b1` — MiT-B1 no disponible en timm 1.0.25 *(v1.5)* |
| GAN en primera corrida | NO (confirmado; GAN añadido como fine-tuning post-ep89) *(v1.5)* |
| Arquitectura discriminador | PatchGAN condicional multi-escala 2 escalas (D₁ 256/320px + D₂ 128/160px) *(v1.5)* |
| Loss discriminador | LSGAN + feature matching (W_FM=10.0) + R1 lazy (λ=10, cada 16 steps) *(v1.5)* |
| Input discriminador | 8 canales: RGB(3) + Normal(3) + Roughness(1) + Metallic(1) *(v1.5)* |
| R1 bajo AMP | float32 forzado en `r1_gradient_penalty` con `autocast(enabled=False)` *(v1.5)* |
| Checkpoint final | `best_gan.pt` (GAN época 11) — S=10.457, MAE Normal 10.37°, LPIPS 0.0976 *(v1.5)* |
| w_adv progresivo GAN | 0.02 (épocas 0-4) → 0.05 (5-9) → 0.10 (10+) *(v1.5)* |

---

## 12. ARCHIVOS DISPONIBLES DEL PROYECTO

| Archivo | Estado | Cuándo se necesita |
|---|---|---|
| `matforge_01_renderer_test.py` | Completado y validado | Referencia del renderer Cook-Torrance |
| `matforge_02_dataloader_test.py` | Completado y validado | Referencia del pipeline de datos |
| `matforge_03_training.py` | Completado — versión con GAN fine-tuning | Notebook principal de entrenamiento |
| `matforge_split.csv` | Generado con SEED=42 y subido a Kaggle | Split estratificado 85/15 — NO regenerar |
| `Investigación_Discriminador_GAN_MatForge.md` | Generado | Informe técnico del discriminador con 14 referencias IEEE |
| Dataset Kaggle `matforge-checkpoints-ep20` (v2 activa) | `best_overall.pt` ep89 + `best_gan.pt` ep_gan11 + checkpoints intermedios | Checkpoint final del modelo |
| `matforge_downloader.py` v2.0 | Completado | Si se añaden nuevas texturas en el futuro |
| `matforge_relabeling.py` | Completado y ejecutado | Referencia; artefactos ya subidos a Kaggle |
| `eda_output/metricas_completas.csv` | Generado | Referencia histórica del EDA |
| `relabeling_output/relabeling_final.csv` | Generado y subido a Kaggle | DataLoader de entrenamiento |
| `relabeling_output/sampler_weights.json` | Generado y subido a Kaggle | DataLoader de entrenamiento |
| `relabeling_output/knn_classifier.pkl` | Generado y subido a Kaggle | Pipeline de inferencia Streamlit |
| `relabeling_output/pca_model.pkl` | Generado y subido a Kaggle | Pipeline de inferencia Streamlit |
| `relabeling_output/label_encoder.pkl` | Generado y subido a Kaggle | Pipeline de inferencia Streamlit |
| Código DeepPBR (versiones) | Disponible en local | Fase 2: referencia del renderer o dataloader |
| Imágenes de épocas DeepPBR | Disponible en local | Comparación visual en memoria del PI |
| Dataset completo en Kaggle | `MatForge PBR Dataset` (privado) — 3.245 texturas | Fase 2: entrenamiento |

**Instrucción de uso de archivos en el chat**: adjuntar como archivo, nunca pegar código largo. Empezar siempre por la versión más reciente disponible.

---

## 13. HISTORIAL DE CAMBIOS

- **v1.0** (22/04/2026): Documento inicial. Arquitectura dual-head (Normal + Roughness).
- **v1.1** (23/04/2026): Añadida tercera cabeza de salida (Metallic). Estrategia de dataset ampliada a dominios no pétreos. Descargador v2.0 implementado.
- **v1.2** (26/04/2026): Actualización completa post-Fase 1. Dataset final documentado (3.245 texturas). EDA completado con 8 filtros por categoría. Añadido R8.
- **v1.5** (03/05/2026): Actualización post-entrenamiento completo (Fases 2, 3 y GAN). **Cambios críticos**: (C5) encoder cambiado de MiT-B1 a PVT-v2-B1 por incompatibilidad de timm 1.0.25; (C6) resultados reales del entrenamiento documentados; (C7) arquitectura y resultados del GAN fine-tuning documentados. **Añadidos**: sección de resultados del entrenamiento supervisado y GAN, arquitectura completa del discriminador, R6 actualizado con resultados reales, fases 2 y 3 cerradas, checkpoint final identificado (`best_gan.pt`).

---

## 14. RESULTADOS DEL ENTRENAMIENTO *(añadido v1.5)*

### 14.1 Entrenamiento supervisado (épocas 0-89)

| Hito | Época | MAE Normal | Roughness MAE | LPIPS | S compuesto |
|---|---|---|---|---|---|
| Inicio | 0 | 13.49° | 0.1667 | 0.1835 | 13.62 |
| Descongelado encoder | 3 | — | — | — | — |
| Render loss activado (parcial) | 5 | 12.46° | 0.1705 | 0.1287 | 12.59 |
| Render loss completo | 15 | 11.01° | 0.1272 | 0.1189 | 11.11 |
| Curriculum 320px | 65 | 10.52° | 0.1059 | 0.1076 | 10.60 |
| Plateau | 70 | 10.49° | 0.1068 | 0.1069 | 10.57 |
| **Final supervisado** | **89** | **10.45°** | **0.1087** | **0.1094** | **10.53** |

- Tiempo por época: ~4.5-6 min en T4
- Tiempo total: ~8.5h
- Scheduler: cosine warmup 5 épocas + cosine decay. Cosine restart en época 20 con LR_enc=2e-5 / LR_dec=6e-5 por agotamiento del ciclo.
- Curriculum a 320px en época 65: transición suave, sin inestabilidad observable.

### 14.2 GAN fine-tuning (épocas GAN 0-19)

- **Punto de partida**: `best_overall.pt` (época 89, S=10.533)
- **Arquitectura del discriminador**: PatchGAN condicional multi-escala de dos escalas (D₁ a 256/320px, D₂ a 128/160px). Spectral normalization en todas las capas Conv. InstanceNorm. Dropout p=0.1. Input de 8 canales: RGB(3)+Normal(3)+Roughness(1)+Metallic(1). Sin activación final (logits raw).
- **Loss**: LSGAN (D y G) + feature matching loss (W_FM=10.0) + R1 gradient penalty lazy (λ=10, cada 16 steps, en float32).
- **Optimizador D**: AdamW, lr=1e-4, betas=(0.0, 0.99), weight_decay=0.
- **Resultado del discriminador**: colapso a D(real)≈D(fake)≈0.50 desde época GAN 1. La feature matching loss siguió aportando señal útil independientemente.

| Épocas GAN | MAE Normal | Roughness MAE | LPIPS | S compuesto |
|---|---|---|---|---|
| 0 | 10.47° | 0.1086 | 0.1025 | 10.554 |
| 2 | 10.39° | 0.1090 | 0.0991 | 10.470 |
| **11 (mejor)** | **10.37°** | **0.1117** | **0.0976** | **10.457** |
| 19 (final) | 10.47° | 0.1097 | 0.0972 | 10.557 |

- **Checkpoint final**: `best_gan.pt` (época GAN 11)
- **Mejora LPIPS respecto al supervisado**: −10.8%
- **Roughness**: leve retroceso (+0.003) — aceptable dado el contexto

### 14.3 Métricas del checkpoint final para Fase 4

El siguiente paso (Fase 4) es comparar `best_gan.pt` con DeepPBR sobre el split de validación de MatForge (487 texturas) siguiendo el protocolo de la sección 5.

| Modelo | Checkpoint | MAE Normal ↓ | Roughness MAE ↓ | LPIPS ↓ | S ↓ |
|---|---|---|---|---|---|
| **MatForge** | best_gan.pt | **10.37°** | **0.1117** | **0.0976** | **10.457** |
| DeepPBR | — | TBD (Fase 4) | TBD | TBD | TBD |

---

*Este documento debe adjuntarse al inicio de cada nueva sesión de trabajo para garantizar que el asistente técnico (Claude) mantiene coherencia total con las decisiones previas. Actualizar el campo "Fase actual" y "Estado" en la tabla de la sección 10 al cerrar cada fase.*
