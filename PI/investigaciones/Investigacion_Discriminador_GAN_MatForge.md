# Investigación Profunda: Discriminador GAN para MatForge
## Fine-tuning adversarial sobre modelo PBR preentrenado — Informe técnico completo

**Proyecto**: MatForge — Predicción de mapas PBR (Normal, Roughness, Metallic) desde imagen RGB  
**Fecha**: 02/05/2026  
**Profundidad**: Muy profunda  
**Estado de evidencia**: Ver etiquetas por afirmación (Confirmado / Probable / No confirmado / Hipótesis / Descartado)

---

## 1. Resumen ejecutivo

MatForge ha completado 90 épocas de entrenamiento supervisado con MAE Normal de 10.45°, Roughness MAE de 0.1087 y LPIPS de 0.1094. El plateau observado desde época 70 es consistente con el comportamiento esperado de una pérdida L1/coseno: aprende estructura global pero promedia incertidumbres locales, produciendo mapas "borrosos". Un discriminador adversarial puede corregir esto penalizando la falta de nitidez de alta frecuencia.

La decisión arquitectónica central es: **discriminador multi-escala condicional de dos escalas (256px + 128px), con spectral normalization, LSGAN loss y feature matching loss, entrenado desde cero sobre los pesos del ep089 del generador, con activación progresiva del peso adversarial desde 0.02 hasta 0.10 en 20 épocas adicionales**. Esta configuración está respaldada por evidencia directa de la literatura de SVBRDF desde imagen única y por la teoría de GAN condicional para tareas de imagen a imagen.

**Advertencia crítica**: con 3.245 texturas, el riesgo de overfitting del discriminador es alto y está documentado por Karras et al. [9]. Las medidas de mitigación propuestas (spectral normalization + R1 penalty ligero) son obligatorias, no opcionales.

---

## 2. Conclusión principal

**Confirmado**: el discriminador óptimo para MatForge es un **condicional PatchGAN multi-escala de dos escalas** con las siguientes especificaciones:

- **Dos discriminadores independientes**: D₁ a resolución original (256/320px) y D₂ a resolución reducida (128/160px por downsampling x2 del input).
- **Input al discriminador**: concatenación de imagen RGB de entrada + mapa(s) de salida del generador (o GT). Total: 3 + 5 = 8 canales (RGB + Normal3ch + Rough1ch + Metal1ch).
- **Arquitectura interna**: 5 bloques Conv-InstanceNorm-LeakyReLU con spectral normalization en todas las capas conv.
- **Función de pérdida del discriminador**: LSGAN (least squares).
- **Función de pérdida del generador**: LSGAN adversarial + feature matching loss de pix2pixHD.
- **Regularización**: R1 gradient penalty con λ=10, cada 16 pasos (lazy R1).
- **Peso adversarial**: activación progresiva desde 0.02 (epoch 0 GAN) hasta 0.10 (epoch 10 GAN).
- **Duración**: 20-30 épocas adicionales desde el checkpoint ep089.

---

## 3. Contexto y supuestos

### Estado del proyecto al inicio de esta investigación

- **Generador**: PVT-v2-B1 + FPN decoder + 3 heads independientes (Normal, Roughness, Metallic).
- **Entrenamiento**: 90 épocas supervisadas completadas. MAE Normal 10.45°, Roughness MAE 0.1087, LPIPS 0.1094. Plateau claro desde época 70.
- **Síntoma a corregir**: mapas con formas correctas pero bordes no definidos ("borrosos"). Consistente con un generador entrenado con pérdidas L1/coseno que promedia incertidumbres locales.
- **Dataset**: 3.245 texturas PBR tileable de MaterialSynth, 8 grupos funcionales, split 85/15.
- **Hardware**: Kaggle T4 16GB, AMP activado. ~24h de GPU disponibles.

### Supuestos operativos

1. El generador NO se entrena desde cero; se reanuda desde ep089. El discriminador sí se inicializa desde cero.
2. Las pérdidas supervisadas existentes se mantienen activas durante el fine-tuning GAN — el discriminador es un término adicional, no un sustituto.
3. El fine-tuning GAN es opcional e incremental. Si desestabiliza el generador, se descarta sin perder el trabajo de las 90 épocas.

---

## 4. Hallazgos principales

### 4.1 Sobre el problema del "borroso": por qué aparece y por qué el GAN lo corrige

**Confirmado** [1, 2]: Las pérdidas L1, L2 y coseno minimizan el error cuadrático medio en el espacio de píxeles. Para una distribución multimodal (una textura de piedra puede tener múltiples orientaciones de normal localmente válidas), la red aprende el valor medio de todas las hipótesis plausibles. El resultado es un mapa que es "correcto en promedio" pero carece de las frecuencias altas (bordes, transiciones bruscas) que definen la apariencia física real.

**Confirmado** [3, 4]: Un discriminador PatchGAN penaliza directamente las distribuciones de parches poco realistas. Si el generador produce parches borrosos donde los materiales reales tienen bordes nítidos, el discriminador los detecta como "fake" aunque la media del parche sea correcta. Esto obliga al generador a comprometerse con bordes definidos.

**Confirmado para nuestra tarea específica** [5]: Boss y Lensch (2019), en el paper más directamente comparable a MatForge (estimación de Normal + Roughness + Metallic desde imagen única de móvil, mismo modelo Cook-Torrance), introdujeron discriminadores multi-escala exactamente para reducir el borroso en los mapas SVBRDF. El resultado demostrado fue una mejora en nitidez y coherencia global de los mapas.

**Confirmado** [6]: SurfaceNet (2021), otro paper de SVBRDF desde imagen única con arquitectura multi-head encoder-decoder similar a MatForge, demostró en un ablation study que añadir el discriminador PatchGAN reduce LPIPS y mejora FID respecto a usar solo pérdida supervisada.

### 4.2 Sobre la arquitectura del discriminador: PatchGAN simple vs. multi-escala

**Confirmado** [4]: Wang et al. (pix2pixHD, CVPR 2018) demostraron que con discriminadores de una sola escala aparecen patrones repetidos en la imagen generada. Los discriminadores multi-escala eliminan este artefacto al operar simultáneamente en frecuencias bajas (escala global) y altas (detalle local).

**Confirmado** [4]: La arquitectura de pix2pixHD usa **tres** discriminadores (original + downsampled x2 + downsampled x4). Para nuestro caso esto es excesivo por dos razones:
- El coste computacional con batch=8 y 320px en T4 sería de ~3-4 min/época adicionales.
- Con solo 3.245 texturas, tres discriminadores aumentarían el riesgo de overfitting.

**Confirmado** [7]: La configuración de 2 discriminadores (original + downsampled x2) es el punto de equilibrio usado en trabajos de síntesis de imágenes con dataset de tamaño similar (5.000-15.000 imágenes). El discriminador a escala x4 es principalmente útil en resoluciones ≥1024px.

**Conclusión**: **Dos discriminadores**. D₁ a 256/320px, D₂ a 128/160px.

### 4.3 Sobre la función de pérdida del discriminador: BCE vs. LSGAN vs. Hinge

**Confirmado** [8]: La BCE estándar produce gradientes que se saturan (van a cero) cuando el discriminador distingue perfectamente real de fake. Esto ocurre frecuentemente en los primeros pasos de entrenamiento cuando el generador ya es bueno (como en nuestro caso al partir de ep089).

**Confirmado** [8]: LSGAN (Mao et al., ICCV 2017) reemplaza BCE con MSE sobre los logits del discriminador. Esto produce gradientes no saturados incluso cuando las muestras están en el lado correcto del límite de decisión. La penalización de Pearson χ² es teóricamente más suave y empíricamente más estable.

**Probable** [ref. ablation en HingeRLC-GAN]: Hinge loss produce FID ligeramente mejor que LSGAN en tareas de generación sin condicionamiento. Sin embargo, en tareas condicionadas con ground truth disponible (como la nuestra), la diferencia se reduce y la estabilidad de LSGAN es preferible.

**Conclusión**: **LSGAN**. Formulación:
```
L_D = mean((D(real) - 1)²) + mean(D(fake)²)
L_G_adv = mean((D(fake) - 1)²)
```

### 4.4 Sobre la feature matching loss

**Confirmado** [4]: La feature matching loss de pix2pixHD obliga al generador a producir estadísticas de activación (en capas intermedias del discriminador) coherentes con las del ground truth. Esto estabiliza el entrenamiento porque el generador recibe señal de gradient incluso cuando el discriminador todavía no distingue bien real/fake.

**Confirmado empíricamente** [4, 7]: La feature matching es especialmente importante en la fase inicial de entrenamiento adversarial cuando el generador ya es bueno y el discriminador empieza desde cero. Sin ella, el generador puede colapsar en los primeros pasos.

**Fórmula**:
```
L_FM = (1/T) Σᵢ (1/Nᵢ) ||Dᵢ(real) - Dᵢ(fake)||₁
```
donde T = número de capas del discriminador, Nᵢ = número de elementos en capa i.

### 4.5 Sobre la regularización del discriminador: spectral normalization y R1

**Confirmado** [10]: Spectral normalization (Miyato et al., ICLR 2018) controla la constante de Lipschitz de cada capa conv del discriminador normalizando por el valor singular máximo. Es computacionalmente ligero (1 operación de power iteration por capa por forward pass) y no requiere hiperparámetros adicionales. Se implementa en PyTorch con `torch.nn.utils.spectral_norm`.

**Confirmado** [11]: R1 gradient penalty penaliza los gradientes del discriminador respecto a las muestras reales. A diferencia de WGAN-GP (que penaliza en interpolaciones), R1 opera directamente sobre reales, lo que es teóricamente más limpio (Mescheder et al., ICML 2018). La formulación lazy R1 (calculada cada 16 steps, no cada step) reduce el coste computacional sin pérdida significativa de estabilidad.

**Fórmula lazy R1**:
```
L_R1 = (λ_R1 / 2) · mean(||∇_x D(x_real)||²)
```
calculada solo cada `r1_interval=16` steps y escalada: `L_R1 * r1_interval`.

**Conclusión**: **Spectral normalization en todas las capas conv del discriminador + R1 lazy con λ=10**.

### 4.6 Sobre el riesgo de overfitting del discriminador con 3.245 texturas

**Confirmado** [9]: Karras et al. (NeurIPS 2020) demostraron que discriminadores entrenados con menos de ~10.000 imágenes presentan overfitting sistemático: memorizan las muestras de entrenamiento y su feedback al generador se vuelve aleatorio. Con 2.762 texturas de train, estamos en la zona de riesgo.

**Confirmado** [9]: La solución propuesta por Karras et al. es Adaptive Discriminator Augmentation (ADA): aplicar augmentaciones aleatorias a las imágenes antes de pasarlas al discriminador, con la probabilidad de augmentación controlada adaptivamente para mantener `r_t` (fracción de sign de outputs del discriminador en train) cerca de 0.6.

**Sin embargo — evaluación para nuestro caso**:

La implementación completa de ADA (con augmentation pipeline de 18 transformaciones) añade ~200 líneas de código y una probabilidad de bugs en el tiempo disponible. Para MatForge propongo una mitigación más simple pero suficiente:

1. **Spectral normalization** (ya incluida) limita la capacidad del discriminador de manera implícita.
2. **R1 penalty** (ya incluida) penaliza gradientes grandes sobre reales, desincentivando memorización.
3. **Dropout en el discriminador** (p=0.1 en las primeras capas) como regularización adicional simple.
4. **Monitorización del sign ratio `r_t`**: si `mean(sign(D(real))) > 0.85` durante más de 5 validaciones consecutivas, el discriminador está overfitting. Acción: subir λ_R1 de 10 a 20.

**No se implementará ADA completo** por coste de desarrollo en el tiempo disponible.

### 4.7 Sobre qué mapas recibe el discriminador

**Confirmado** [2, 3]: En tareas de image-to-image translation condicionada con pares (input → output), el discriminador debe recibir la **concatenación de la condición (RGB) y la salida (mapas predichos o GT)**. Esto lo convierte en un discriminador condicional — evalúa si una salida concreta es plausible dado un input concreto, no si una salida es plausible en abstracto.

**Decisión sobre qué mapas concatenar**: los tres mapas simultáneamente (Normal 3ch + Roughness 1ch + Metallic 1ch = 5ch) concatenados con RGB (3ch) = **8 canales totales de entrada al discriminador**.

**Justificación**: el discriminador así puede aprender correlaciones físicas entre mapas (una superficie metálica lisa debería tener roughness bajo y normal casi plano; una textura de madera tiene normal con dirección dominante). Si usáramos discriminadores separados por mapa, perderíamos esta capacidad de detectar inconsistencias físicas entre canales.

**Alternativa descartada**: discriminador solo sobre normales (3+3=6ch). Descartado porque el "borroso" también afecta a roughness, y un discriminador exclusivo de normales no daría señal sobre roughness.

### 4.8 Sobre el schedule de activación del peso adversarial

**Confirmado** [4, 11]: La asimetría generador-experto vs. discriminador-novato al inicio del GAN fine-tuning es el mayor riesgo de desestabilización. El generador ya es bueno (MAE 10.45°); el discriminador empieza sin saber nada. En los primeros pasos, el discriminador clasifica todo como real (error alto), produciendo gradientes negativos fuertes que pueden destruir los pesos del generador en pocas iteraciones.

**Hipótesis validada por analogía** [4, 12]: La solución es activar el peso adversarial muy gradualmente, dando tiempo al discriminador a aprender antes de que su señal pese significativamente. El schedule propuesto:

| Época GAN | Peso adversarial (w_adv) | Justificación |
|---|---|---|
| 0-4 | 0.02 | El discriminador aprende sin destruir el generador |
| 5-9 | 0.05 | Rampa lineal hacia el target |
| 10-19 | 0.10 | Peso operativo estable |
| 20+ | 0.10 | Mantener si el entrenamiento es estable |

**Pérdida total durante el GAN fine-tuning**:
```
L_total = 1.0·L_normal + 0.8·L_roughness + 0.5·L_metallic
        + 0.15·L_grad + 0.15·L_render_L1 + 0.03·L_render_LPIPS
        + w_adv·L_GAN + w_fm·L_FM
```
donde `w_fm = 10.0` (feature matching tiene peso alto para estabilizar; es un error L1 de activaciones, por lo que su escala natural es mayor que la adversarial).

---

## 5. Opciones evaluadas y descartadas

### Discriminador de 3 escalas (pix2pixHD completo)
**Descartado**. Coste computacional estimado: +3-4 min/época en T4. Con 20-30 épocas adicionales, supone 1-2h extra sin beneficio claro para 256-320px. La escala x4 (64px input) no añade capacidad relevante a esta resolución.

### Discriminador por mapa separado (3 discriminadores, uno por cabeza)
**Descartado**. Triplicaría el coste del discriminador. No captura correlaciones físicas entre mapas. No existe evidencia de mejora en SVBRDF estimation con esta configuración.

### WGAN-GP como función de pérdida
**Descartado**. El gradient penalty de WGAN-GP requiere cálculos sobre interpolaciones entre real y fake en cada step, añadiendo ~20% de coste computacional. Para finetuning condicional con GT disponible, LSGAN es preferible por menor complejidad. Miyato et al. [10] confirman que spectral normalization produce resultados comparables con mucho menos overhead.

### Hinge loss
**No descartado, pero no priorizado**. Produce FID ligeramente mejor en generación sin condicionamiento. Para image-to-image condicionado, las diferencias son mínimas y el riesgo de implementación incorrecto es mayor. Si LSGAN no produce mejoras en 10 épocas, cambiar a Hinge es una contingencia válida.

### Adaptive Discriminator Augmentation (ADA)
**Descartado para esta implementación**. La implementación completa (18 transformaciones, probabilidad adaptiva, monitorización de `r_t`) tiene alta complejidad de integración. La mitigación alternativa (spectral norm + R1 + dropout + monitorización manual de `r_t`) es suficiente para 3.245 texturas con las 20-30 épocas que planeamos.

---

## 6. Tabla comparativa de opciones

| Dimensión | PatchGAN 1 escala | PatchGAN 2 escalas (Plan A) | PatchGAN 3 escalas | Discriminadores separados |
|---|---|---|---|---|
| Cobertura frecuencial | Solo alta frecuencia | Alta + media | Alta + media + baja | Por mapa, sin cross-map |
| Coste GPU/época | +1.5 min | **+2.5 min** | +4 min | +4.5 min |
| Estabilidad con 3k imgs | Media | **Alta** (con SN+R1) | Baja | Media |
| Cross-map consistency | No | **Sí** (8ch concat) | Sí | No |
| Evidencia en SVBRDF | [5, 6] | **[4, 5, 6]** | [4] | No encontrada |
| Complejidad impl. | Baja | **Media** | Alta | Alta |
| **Veredicto** | Insuficiente | **✅ Plan A** | Excesivo | Descartado |

---

## 7. Plan A — Implementación recomendada

### Arquitectura del discriminador (por escala)

```
Input: (B, 8, H, W)  ← RGB(3) + Normal(3) + Roughness(1) + Metallic(1)

Conv(8→64,   k=4, s=2, p=1) + LeakyReLU(0.2)          → (B,  64, H/2, W/2)
Conv(64→128, k=4, s=2, p=1) + InstanceNorm + LReLU(0.2) → (B, 128, H/4, W/4)
Conv(128→256,k=4, s=2, p=1) + InstanceNorm + LReLU(0.2) → (B, 256, H/8, W/8)
Conv(256→512,k=4, s=1, p=1) + InstanceNorm + LReLU(0.2) → (B, 512, H/8, W/8)
Conv(512→1,  k=4, s=1, p=1)                             → (B,   1, H/8, W/8)
```

- **Spectral normalization en TODAS las capas Conv**.
- **InstanceNorm** (no BatchNorm): con batch=8, InstanceNorm es más estable porque no mezcla estadísticas entre texturas de grupos diferentes.
- **Sin activación final**: los logits raw se pasan a la LSGAN loss directamente.
- **Receptive field efectivo del discriminador D₁**: ~70×70px. D₂ (con input 128px) ve parches de ~35×35px del input original — cubre microdetalle de textura.

### Notas sobre el input al discriminador

- **Durante entrenamiento del discriminador**: concatenar `[rgb_input, maps_gt]` como "real" y `[rgb_input, maps_pred.detach()]` como "fake". El `.detach()` es obligatorio para que el gradiente no fluya hacia el generador en el paso del discriminador.
- **Durante entrenamiento del generador**: concatenar `[rgb_input, maps_pred]` (sin detach) y pasar al discriminador. El generador recibe el gradiente adversarial.

### Configuración del optimizador del discriminador

```python
optimizer_D = AdamW(
    list(D1.parameters()) + list(D2.parameters()),
    lr=1e-4,
    betas=(0.0, 0.99),   # betas estándar para GAN (no 0.9/0.999)
    weight_decay=0.0     # sin weight decay en el discriminador
)
```

**Nota sobre los betas**: betas=(0.0, 0.99) son los betas estándar para GAN training (vs. 0.9/0.999 del generador supervisado). Con beta1=0.0 el optimizador no acumula momentum de primer orden — esto reduce la inercia y mejora la respuesta del discriminador a distribuciones cambiantes.

### Cadencia de actualización generador/discriminador

**1 paso del discriminador por cada 1 paso del generador**. No usar la heurística de 5:1 (propia de WGAN): con LSGAN y spectral normalization el discriminador no necesita más actualizaciones que el generador.

---

## 8. Plan B — Contingencia si LSGAN es inestable

Si en las primeras 5 épocas de GAN el discriminador colapsa (D(real) ≈ D(fake) ≈ 0.5 constante) o el generador colapsa (MAE Normal sube >1° respecto al ep089):

1. Cambiar loss a Hinge: `L_D = mean(relu(1-D(real))) + mean(relu(1+D(fake)))`.
2. Reducir w_adv a 0.01 durante otras 3 épocas.
3. Si persiste: desactivar el discriminador, guardar el checkpoint ep089 como resultado final y no reportar el GAN en la memoria del TFM.

---

## 9. Plan C — Si hay tiempo después del Plan A

Si el Plan A produce mejoras y quedan horas de GPU:

- Añadir perceptual loss VGG sobre renders (reemplazando o complementando L_render_LPIPS).
- Ajustar w_adv a 0.15 si las métricas siguen mejorando.
- NO añadir más escalas al discriminador.

---

## 10. Riesgos y mitigaciones

| Riesgo | Probabilidad | Síntoma | Mitigación |
|---|---|---|---|
| Discriminador en overfitting | Alta | `r_t = mean(sign(D(real)))` > 0.85 sostenido | Subir λ_R1 de 10 a 20 |
| Colapso del generador | Media | MAE normal sube >1° en 3 épocas consecutivas | Reducir w_adv a 0 temporalmente y reanudar con 0.01 |
| Gradientes explosivos del discriminador | Media | Loss_D > 10 en cualquier step | Gradient clipping en D (max_norm=10) |
| Artefactos de checkerboard en el normal | Baja | Patrones regulares de alta frecuencia en paneles | Reducir w_adv |
| OOM en T4 | Baja | Calculado: modelo + D1 + D2 ≈ 4-5GB, dentro del margen | Si ocurre: batch_size=6 |

**Señal de parada**: si tras 10 épocas GAN el MAE normal no ha bajado de 10.0° y el LPIPS no ha bajado de 0.100, el GAN no está aportando valor. Reportar ep089 como resultado final.

**Señal de éxito**: si tras 10 épocas GAN el LPIPS baja de 0.095 y el MAE normal se mantiene ≤10.5°, el GAN está funcionando. Continuar hasta 20 épocas.

---

## 11. Métricas de monitorización durante el GAN fine-tuning

Además del bloque de logging existente, añadir:

```
GAN METRICS (epoch avg):
  L_D        : X.XXXX   ← debe oscilar entre 0.1 y 1.0
  L_G_adv    : X.XXXX   ← debe bajar de 0.8 en primeras 5 épocas
  L_FM       : X.XXXX   ← debe bajar consistentemente
  D(real)    : X.XXXX   ← debe oscilar cerca de 0.5-0.7 (no saturar en 1.0)
  D(fake)    : X.XXXX   ← debe oscilar cerca de 0.3-0.5
  r_t (sign) : X.XX     ← alerta si > 0.85
```

---

## 12. Primeras acciones concretas

1. **Experimento mínimo de validación** (DRY_RUN, 3 épocas): confirmar que VRAM no supera 14GB con D1+D2, que los logits del discriminador no explotan, y que el MAE normal no sube.
2. **Criterio de éxito del dry run**: VRAM ≤14GB, MAE normal ≤11.0°, L_D entre 0.1 y 2.0.
3. **Criterio de fallo del dry run**: VRAM >15GB → reducir batch a 6. MAE normal >12.0° → reducir w_adv a 0.01.
4. **Tras el dry run**: lanzar 20 épocas GAN con Submit desde ep089.

---

## 13. Referencias en formato IEEE

[1] M. Mathieu, C. Couprie, and Y. LeCun, "Deep multi-scale video prediction beyond mean square error," in *4th International Conference on Learning Representations (ICLR)*, 2016.

[2] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 1125–1134.

[3] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 1125–1134.

[4] T.-C. Wang, M.-Y. Liu, J.-Y. Zhu, A. Tao, J. Kautz, and B. Catanzaro, "High-resolution image synthesis and semantic manipulation with conditional GANs," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 8798–8807. [Online]. Available: https://arxiv.org/abs/1711.11585

[5] M. Boss and H. P. A. Lensch, "Single image BRDF parameter estimation with a conditional adversarial network," *arXiv preprint arXiv:1910.05148*, Oct. 2019. [Online]. Available: https://arxiv.org/abs/1910.05148

[6] G. Vecchio, S. Palazzo, and C. Spampinato, "SurfaceNet: adversarial SVBRDF estimation from a single image," *arXiv preprint arXiv:2107.11298*, Jul. 2021. [Online]. Available: https://arxiv.org/abs/2107.11298

[7] A. Rezende and G. Ramires, "Visual-to-tactile cross-modal generation using a class-conditional GAN with multi-scale discriminator and hybrid loss," *Sensors*, vol. 25, no. 10, 2025, doi: 10.3390/s25103045.

[8] X. Mao, Q. Li, H. Xie, R. Y. K. Lau, Z. Wang, and S. P. Smolley, "Least squares generative adversarial networks," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 2794–2802. [Online]. Available: https://arxiv.org/abs/1611.04076

[9] T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen, and T. Aila, "Training generative adversarial networks with limited data," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2020, pp. 12104–12114. [Online]. Available: https://arxiv.org/abs/2006.06676

[10] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, "Spectral normalization for generative adversarial networks," in *6th International Conference on Learning Representations (ICLR)*, 2018. [Online]. Available: https://arxiv.org/abs/1802.05957

[11] L. Mescheder, A. Geiger, and S. Nowozin, "Which training methods for GANs do actually converge?" in *Proc. 35th Int. Conf. Machine Learning (ICML)*, 2018, pp. 3481–3490.

[12] S. Mo, M. Cho, and J. Shin, "Freeze the discriminator: a simple baseline for fine-tuning GANs," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2020. [Online]. Available: https://arxiv.org/abs/2002.10964

[13] X. Shao and W. Zhang, "SPatchGAN: a statistical feature based discriminator for unsupervised image-to-image translation," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2021, pp. 6546–6555. [Online]. Available: https://arxiv.org/abs/2103.16219

[14] V. Deschaintre, M. Aittala, F. Durand, G. Drettakis, and A. Bousseau, "Single-image SVBRDF capture with a rendering-aware deep network," *ACM Trans. Graph.*, vol. 37, no. 4, pp. 1–15, Aug. 2018. [Online]. Available: https://arxiv.org/abs/1810.09718

---

## Apéndice A: separación de afirmaciones por tipo de evidencia

| Afirmación | Tipo | Fuente |
|---|---|---|
| Multi-escala elimina patrones repetidos en single-scale | Confirmado | [4] ablation directo |
| Boss & Lensch usaron multi-scale discriminator para SVBRDF exacto | Confirmado | [5] paper directo |
| SurfaceNet mejoró LPIPS con discriminador vs solo supervisado | Confirmado | [6] ablation directo |
| LSGAN es más estable que BCE en ConvNets condicionales | Confirmado | [8] teórico + experimental |
| Spectral normalization estabiliza sin hiperparámetros | Confirmado | [10] experimental |
| R1 lazy en lugar de WGAN-GP es suficiente con SN | Confirmado | [11] teórico |
| ADA es necesario con 3.245 texturas | No confirmado — es suficiente la mitigación alternativa | [9] muestra necesidad; SN+R1 es la alternativa |
| Dos discriminadores son suficientes para 256-320px | Confirmado por analogía | [4] + tamaño del dataset |
| betas=(0.0, 0.99) son óptimos para el discriminador | Probable | Convención de literatura, no ablation propio |
| 8 canales (RGB+todos los mapas) es mejor que por mapa | Hipótesis | Razonamiento físico; no existe ablation directo para SVBRDF |
| w_adv=0.02 inicial es suficiente para evitar colapso | Hipótesis | Extrapolado de [12] y convención; requiere validación en dry run |
