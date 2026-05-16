# 5. Conclusiones

## 5.1 Conclusiones por objetivo específico

El proyecto alcanza el objetivo general planteado y completa con éxito seis de los siete objetivos específicos en los términos definidos al inicio del PI. El único objetivo que requiere una lectura matizada es el módulo de super-resolución, cuya mejora interna se confirma, pero no se transfiere de forma robusta al benchmarking general frente a Real-ESRGAN base.

### 5.1.1 OE1 — Modelo de predicción PBR

El primer objetivo específico consistía en desarrollar MatForgeNet, una arquitectura encoder-decoder capaz de predecir simultáneamente mapas Normal, Roughness y Metallic a partir de una única imagen RGB, cumpliendo tres umbrales cuantitativos: MAE angular de normales inferior a 11°, MAE de Roughness inferior a 0,12 y LPIPS inferior a 0,10.

El objetivo se considera cumplido. El checkpoint final `best_gan.pt` alcanzó un MAE angular de normales de 10,37°, un MAE de Roughness de 0,1117 y un LPIPS de 0,0976 en validación interna, situándose por debajo de los tres umbrales establecidos. En el benchmarking global, MatForge mantuvo un comportamiento coherente, con 10,40° de MAE angular y 0,1084 de MAE de Roughness sobre el split de validación evaluado mediante *tile-and-merge*.

La principal dificultad técnica fue la inestabilidad de la fase adversarial. El discriminador PatchGAN colapsó desde la primera época GAN, con D(real) y D(fake) convergiendo a valores próximos a 0,50. Por tanto, la mejora perceptual final no debe atribuirse a una presión adversarial estable, sino principalmente a la *feature matching loss*, que continuó aportando una señal útil pese al colapso del discriminador. Esta decisión permitió conservar el mejor checkpoint perceptual sin presentar el entrenamiento GAN como un éxito pleno del discriminador.

### 5.1.2 OE2 — Módulo de super-resolución especializado

El segundo objetivo específico planteaba implementar un módulo de super-resolución ×4 basado en RRDBNet y ajustado al dominio de materiales PBR, con una mejora mínima del 10% en LPIPS respecto a Real-ESRGAN base.

El objetivo se considera cumplido con matiz. En la validación interna del fine-tuning, el checkpoint `sr_ft_phase1_best_lpips.pt`, correspondiente a la época 24, alcanzó un LPIPS de 0,2380 frente a 0,2672 del modelo base, lo que representa una mejora del 10,9%. Este resultado supera el umbral definido y confirma que el entrenamiento especializado aprendió información útil dentro de la distribución de validación empleada durante el ajuste.

Sin embargo, el resultado no se transfirió al benchmarking general. En la evaluación final sobre 100 texturas del split de validación, Real-ESRGAN base obtuvo mejor LPIPS que MatForge SR fine-tuned. Esta divergencia se interpreta como un caso de *distribution shift*: el modelo aprendió a invertir la degradación sintética usada durante el entrenamiento, pero no generalizó con la misma eficacia a condiciones más amplias. La Fase 2 adversarial del módulo SR también colapsó, por lo que fue abortada y se adoptó el checkpoint de Fase 1. Esta limitación queda conectada directamente con las propuestas de trabajo futuro, especialmente la revisión de la cadena de degradación y la posible integración de MUJICA [38].

### 5.1.3 OE3 — Pipeline de relabeling semántico y clasificador de material

El tercer objetivo específico consistía en reorganizar el dataset en grupos funcionales visualmente coherentes mediante relabeling no supervisado y serializar un clasificador de material operativo para la aplicación.

El objetivo se considera cumplido. El pipeline DINOv2-small + PCA-50 + UMAP + HDBSCAN produjo 37 clústeres brutos con DBCV = 0,3279, superando el umbral mínimo de 0,30; las métricas agregadas quedan respaldadas por `cluster_metrics.json`. Posteriormente, los clústeres se fusionaron en ocho grupos funcionales: `stone_rough`, `wood`, `ceramic_ground`, `mixed_ambiguous`, `brick_terracotta`, `marble_smooth`, `metal` y `concrete_plaster`, con la asignación final documentada en `relabeling_final.csv`. La solución final permitió sustituir las categorías nominales originales de MatSynth por agrupaciones más útiles para la calibración PBR.

El clasificador KNN resultante, con k=7, métrica coseno y operación sobre el espacio PCA-50, fue serializado e integrado en la aplicación mediante los artefactos `knn_classifier.pkl`, `pca_model.pkl` y `label_encoder.pkl`. Su latencia inferior a 5 ms en CPU evita que la clasificación actúe como cuello de botella en el flujo de inferencia.

La dificultad principal fue que las categorías originales del dataset no se correspondían siempre con propiedades visuales o físicas homogéneas. La combinación de embeddings auto-supervisados y clustering basado en densidad resolvió esta limitación sin depender de una reclasificación manual textura por textura.

### 5.1.4 OE4 — Herramientas de refinado no destructivo

El cuarto objetivo específico planteaba integrar herramientas de refinado no destructivo que permitieran al artista corregir, ajustar y enriquecer los mapas generados sin modificar el modelo subyacente ni requerir conocimientos técnicos de aprendizaje profundo.

El objetivo se considera cumplido. La aplicación final incorpora diez herramientas funcionales: corrección de perspectiva, zoom adaptativo, ajuste de Roughness y Metallic, calibración por grupo funcional, mezcla de materiales mediante Reoriented Normal Mapping, conversión a textura tileable, variaciones procedurales FBM, visor 3D integrado, evaluación de calidad del mapa de normales y comparación visual de estados. Estas herramientas convierten MatForge App en un pipeline de producción más amplio que una simple interfaz de inferencia.

La decisión metodológica más relevante fue mantener un estado `maps_raw` inmutable como fuente de verdad. A partir de ese estado se derivan versiones ajustadas, calibradas, mezcladas, tileables o procedurales, evitando que una operación destructiva degrade de forma irreversible la predicción original. Esta arquitectura de estados facilita la experimentación del usuario y mantiene trazabilidad funcional dentro de la sesión.

La mayor dificultad técnica apareció en el tratamiento de mapas de normales. Operaciones aparentemente simples, como mezclas, fundidos o conversión tileable, no pueden aplicarse sobre valores RGB empaquetados sin invalidar la norma vectorial. La solución consistió en operar sobre vectores desempacados y renormalizar en L2 siempre que el mapa Normal fuera transformado.

### 5.1.5 OE5 — Exportación multi-motor con metadatos de procedencia

El quinto objetivo específico consistía en implementar exportación compatible con los principales motores de renderizado y añadir metadatos de procedencia en cada PNG exportado.

El objetivo se considera cumplido. MatForge App exporta materiales para cinco configuraciones de motor: Blender, Unreal Engine 5, Unity URP, Unity HDRP y Godot 4. La exportación respeta las convenciones de cada entorno, incluyendo inversión del canal verde del mapa Normal para Unreal Engine 5, empaquetado ORM en motores que lo requieren y conversión Metallic/Smoothness para Unity. Además, cada ZIP exportado incorpora un `README.txt` con las convenciones aplicadas.

La implementación de metadatos se resolvió mediante XMP estándar Adobe/W3C incrustado en los PNG, con los campos `dc:creator`, `dc:rights`, `xmp:CreatorTool` y `xmpRights:Marked`. Esta solución identifica el origen generado con asistencia de IA y anticipa obligaciones de transparencia, aunque no debe describirse como implementación completa de IPTC 2025.1 ni como C2PA firmado. C2PA queda correctamente situado como evolución futura, al requerir manifiestos y firma criptográfica [58].

La dificultad principal fue evitar una formulación legal excesiva. El sistema implementa metadatos útiles y verificables, pero no un estándar completo de procedencia criptográfica. Por ello, la redacción final debe limitarse a los campos realmente escritos por la aplicación.

### 5.1.6 OE6 — Evaluación cuantitativa y benchmarking

El sexto objetivo específico consistía en evaluar MatForge frente a los modelos del trabajo grupal previo y frente a herramientas representativas del flujo de producción PBR, combinando métricas cuantitativas y análisis cualitativo.

El objetivo se considera cumplido. En el benchmarking cuantitativo, MatForge supera a Pix2Pix y DeepPBR en todas las métricas comunes evaluadas. Frente a Pix2Pix, la mejora alcanza el 29,5% en MAE angular de normales, el 48,9% en MAE de Roughness y el 45,5% en RMSE de Roughness. Frente a DeepPBR, la mejora es del 23,8% en MAE angular y del 51,6% en MAE de Roughness. También obtiene mejor LPIPS render que ambos baselines, aunque el margen frente a Pix2Pix es más estrecho.

La comparación cualitativa con Materialize y Adobe Substance 3D Sampler muestra un resultado más matizado. MatForge supera de forma clara a Materialize en coherencia global de mapas y se muestra especialmente competitivo en materiales metálicos. Frente a Substance, la aplicación no alcanza necesariamente la robustez perceptual de un sistema comercial propietario en todos los materiales, pero ofrece una ventaja diferencial: integra inferencia, refinado, clasificación, exportación, visor 3D, procesado por lotes y ejecución local sin suscripción.

La principal dificultad fue diseñar una comparación justa entre sistemas heterogéneos. La solución adoptada fue separar benchmarking cuantitativo frente a modelos entrenados con datos comparables y análisis cualitativo/sistémico frente a herramientas de producción.

### 5.1.7 OE7 — Accesibilidad y despliegue local

El séptimo objetivo específico planteaba desplegar MatForge App como herramienta ejecutable localmente, sin dependencia de red en inferencia, sobre hardware de consumo y con tiempo de generación inferior a 10 segundos en GPU de 4–8 GB.

El objetivo se considera cumplido con una precisión importante. El pipeline PBR sin super-resolución cumple el criterio temporal: aproximadamente 1 segundo para 512×512 y 6 segundos para 1024×1024 en una GTX 1650 Max-Q de 4 GB. La aplicación funciona sin red durante la inferencia, detecta automáticamente CUDA/CPU y puede ejecutarse en hardware de consumo, aunque con diferencias de rendimiento según dispositivo.

El matiz se encuentra en el módulo SR y en algunas herramientas auxiliares. La activación de super-resolución añade aproximadamente 9 segundos, por lo que el pipeline completo con SR puede superar el umbral de 10 segundos. Además, herramientas CPU como Normal Map Quality o variaciones procedurales sobre mapas grandes pueden requerir tiempos muy superiores. Por tanto, el cumplimiento del OE7 se refiere estrictamente al núcleo de generación PBR sin SR.

La dificultad técnica principal fue la aparición de NaN al usar float16/autocast en hardware real. La solución fue fijar `torch.float32` como tipo universal de cómputo para MatForgeNet y SR, sacrificando parte de la eficiencia a cambio de estabilidad reproducible.

## 5.2 Logro del objetivo general

El objetivo general del PI se considera logrado. MatForge App desarrolla una herramienta local de predicción automática de materiales PBR que permite obtener mapas Normal, Roughness y Metallic a partir de una única fotografía RGB, con una arquitectura encoder-decoder basada en PVT-v2-B1 y FPN, pérdidas físicamente fundamentadas mediante renderizado Cook-Torrance y un conjunto integrado de herramientas de refinado y exportación. El sistema no se limita a demostrar una red neuronal aislada: constituye una aplicación funcional con interfaz Streamlit, clasificación automática de material, postproceso no destructivo, visor 3D, exportación multi-motor, metadatos XMP y procesado por lotes.

La consecución del objetivo general se apoya en tres evidencias principales. En primer lugar, el modelo central cumple los umbrales cuantitativos establecidos para OE1 y supera a los baselines Pix2Pix y DeepPBR en el benchmarking. En segundo lugar, la aplicación final opera localmente y mantiene tiempos interactivos para el pipeline PBR sin SR sobre una GPU de consumo de 4 GB, lo que valida la accesibilidad técnica del sistema. En tercer lugar, el proyecto se ha desarrollado con coste real de 0 € en licencias, datos y cómputo, aprovechando software abierto, MatSynth y la cuota gratuita de Kaggle, lo que refuerza su viabilidad como solución académica y reproducible.

La única reserva relevante afecta a la super-resolución. El módulo está implementado e integrado, y cumple su métrica interna, pero no demuestra superioridad general frente a Real-ESRGAN base fuera de la distribución de validación del fine-tuning. Esta limitación no invalida el objetivo general, porque el núcleo de MatForge App —predicción PBR local, refinado y exportación— funciona de forma independiente y cumple los criterios principales del PI.

## 5.3 Aprendizajes significativos

El primer aprendizaje técnico fue que la elección de arquitectura en proyectos de predicción densa debe equilibrar rendimiento esperado, disponibilidad real de implementación y compatibilidad con el entorno de entrenamiento. MiT-B1 era una opción razonable en diseño, pero su ausencia en `timm 1.0.25` obligó a sustituirlo por PVT-v2-B1. La sustitución fue viable porque PVT-v2-B1 mantenía una estructura jerárquica equivalente y producía *feature maps* compatibles con el decoder FPN. El aprendizaje no fue solo arquitectónico, sino metodológico: una decisión de diseño no queda cerrada hasta verificarse contra el entorno real de ejecución.

El segundo aprendizaje fue que el entrenamiento adversarial no debe evaluarse únicamente por la presencia formal de un discriminador. En MatForgeNet, el discriminador colapsó desde el inicio de la fase GAN, pero la *feature matching loss* siguió aportando una mejora perceptual medible. Esto obliga a distinguir entre “fase GAN ejecutada” y “presión adversarial estable”. La mejora final de LPIPS fue real, pero su causa técnica dominante no fue una competición generador-discriminador plenamente equilibrada.

El tercer aprendizaje procede del módulo SR. La mejora del 10,9% en validación interna demuestra que el fine-tuning aprendió el patrón de degradación diseñado, pero el benchmarking general evidenció que ese patrón no representaba suficientemente las condiciones reales o más amplias de uso. En super-resolución, la cadena de degradación es casi tan importante como la arquitectura. Un modelo puede mejorar métricas internas y, aun así, empeorar al cambiar el dominio de evaluación.

El cuarto aprendizaje se relaciona con la ingeniería de datos. El EDA con umbrales por categoría fue más adecuado que una limpieza global, porque las propiedades físicas de los materiales no son homogéneas: un mármol pulido, una piedra rugosa y una superficie metálica no admiten los mismos umbrales de roughness, normal o metallic. Esta decisión evitó descartar ejemplos válidos por aplicar criterios estadísticos desconectados del dominio físico. La trazabilidad de esta fase queda respaldada por `metricas_completas.csv`, `candidates_to_discard.csv` y `revision_humana.html`.

El quinto aprendizaje fue estrictamente de despliegue. El uso de float16, razonable en entrenamiento y habitual en inferencia eficiente, produjo NaN en hardware real tanto en PVT-v2-B1 como en RRDBNet. La decisión de operar en float32 redujo el riesgo operativo y estabilizó la aplicación. En un proyecto orientado a usuario final, la estabilidad del resultado es más importante que una optimización numérica que solo funciona en condiciones controladas.

## 5.4 Dificultades encontradas

- **Incompatibilidad P100/PyTorch.** El acelerador P100 de Kaggle produjo errores de compatibilidad CUDA con la versión de PyTorch empleada. La mitigación consistió en migrar los entrenamientos al acelerador T4, compatible con el entorno, sin impacto relevante en el resultado final.

- **MiT-B1 no disponible en `timm 1.0.25`.** El encoder previsto inicialmente no estaba accesible en el registro de modelos. Se sustituyó por PVT-v2-B1, que ofrecía *feature maps* compatibles con el decoder FPN y permitió mantener la arquitectura prevista sin rehacer las cabezas de salida.

- **Agotamiento prematuro del scheduler cosine.** El primer tramo del entrenamiento agotó el ciclo de aprendizaje antes de completar el plan supervisado. Se aplicó un reinicio conservador del scheduler, permitiendo continuar el entrenamiento hasta el plateau.

- **Colapso del discriminador GAN de MatForgeNet.** El discriminador perdió capacidad discriminativa desde la primera época GAN. Se mantuvo la fase únicamente porque la *feature matching loss* aportaba señal perceptual independiente, y se adoptó el checkpoint con mejor LPIPS.

- **Colapso del discriminador SR.** La Fase 2 del fine-tuning SR mostró D(real)≈D(fake) desde el inicio. Se abortó automáticamente y se adoptó el mejor checkpoint de Fase 1, evitando degradar un resultado ya válido internamente.

- **Artefactos de borde en tile-and-merge.** La ventana Hann generaba pesos casi nulos en los bordes, provocando artefactos al fusionar los mapas. La solución fue añadir padding simétrico de `TILE//2` antes del bucle de tiles y recortar posteriormente el resultado.

- **NaN con float16 en hardware real.** La precisión mixta generó valores NaN en la GTX 1650 Max-Q. Se fijó float32 como precisión universal, priorizando estabilidad sobre velocidad.

- **Distribution shift en super-resolución.** La cadena de degradación sintética del fine-tuning no representó suficientemente el dominio general de evaluación. La limitación se documentó explícitamente y se trasladó a trabajo futuro.

## 5.5 Propuestas de mejora y trabajo futuro

1. **Fase 2 SR con estrategia de entrenamiento revisada.** Se propone retomar el fine-tuning adversarial del módulo SR con una cadena de degradación más representativa de las condiciones reales de uso: desenfoques anisótropos, compresión variable, ruido dependiente de luminancia, reescalados no bicúbicos y capturas móviles reales. El objetivo no sería solo mejorar LPIPS interno, sino reducir el *distribution shift* observado.

2. **Integración de MUJICA.** MUJICA [38] plantea una vía técnicamente más adecuada para materiales PBR, al incorporar atención cruzada entre mapas. Su integración permitiría desplazar la SR desde el RGB de entrada hacia los mapas PBR predichos, preservando mejor la coherencia entre Normal, Roughness y Metallic.

3. **Ampliación selectiva del dataset.** Los grupos `concrete_plaster` y `stone_rough` concentran los errores más relevantes. Se propone ampliar su representación mediante descarga selectiva adicional de MatSynth, augmentación controlada y revisión manual de casos ambiguos, evitando aumentar indiscriminadamente el dataset sin corregir los sesgos detectados.

4. **Evolución de XMP hacia C2PA.** El sistema actual escribe campos XMP útiles, pero no firma criptográficamente la procedencia. Una futura versión pública debería integrar C2PA [58], con manifiestos firmados, certificados y validación externa, especialmente si el proyecto se distribuye más allá del contexto académico.

5. **Exploración de modelos de mayor capacidad.** PVT-v2-B2 o PVT-v2-B3 podrían mejorar la representación de geometría fina y materiales difíciles, siempre que se disponga de una GPU con mayor VRAM. Esta línea debería condicionarse a una evaluación coste-beneficio, porque el valor del proyecto depende en parte de su compatibilidad con hardware de consumo.

6. **Despliegue web con modelo ligero.** Una versión cuantizada o destilada, calibrada para INT8 o FP16 en hardware compatible, permitiría desplegar MatForge como servicio web o aplicación ligera sin exigir GPU dedicada. Esta línea requeriría validar que la cuantización no degrada mapas de normales ni introduce inestabilidad numérica.

# Archivos complementarios referenciados

Las conclusiones del proyecto se apoyan en los resultados, decisiones técnicas y evidencias recogidas en los siguientes archivos complementarios de `PI/anexos/` y `PI/investigaciones/`.

## Archivos en `PI/anexos/`

- `PI/anexos/bitacora_de_desarrollo.md` — fuente cronológica de las decisiones, incidencias y evidencias utilizadas para reconstruir dificultades, aprendizajes y desviaciones del proyecto.
- `PI/anexos/backlog_scrum.md` — backlog formal empleado para vincular los objetivos específicos con las historias de usuario y los sprints de desarrollo.
- `PI/anexos/matforge-03-training.ipynb` — notebook de entrenamiento de MatForgeNet, utilizado como respaldo de las métricas finales del modelo.
- `PI/anexos/matforge-sr-01-training.ipynb` — notebook de fine-tuning del módulo SR, utilizado como respaldo de la mejora interna del 10,9% en LPIPS.
- `PI/anexos/matforge-benchmark.ipynb` — notebook de benchmarking utilizado como respaldo de la evaluación comparativa final.
- `PI/anexos/table1_pbr_restricted.csv` — tabla de benchmarking global frente a Pix2Pix y DeepPBR.
- `PI/anexos/table2_matforge_by_group.csv` — tabla de resultados por grupo funcional.
- `PI/anexos/table3_sr.csv` — tabla de evaluación del módulo SR.
- `PI/anexos/metricas_completas.csv` — métricas completas del análisis exploratorio del dataset.
- `PI/anexos/candidates_to_discard.csv` — listado de texturas candidatas al descarte o revisión durante el EDA.
- `PI/anexos/revision_humana.html` — informe visual usado para la revisión manual de casos ambiguos.
- `PI/anexos/cluster_metrics.json` — métricas agregadas del clustering usado en el relabeling.
- `PI/anexos/relabeling_final.csv` — asignación final de texturas a grupos funcionales.

## Archivos en `PI/investigaciones/`

- `PI/investigaciones/MatForge_Informe_Tecnico.md` — base técnica para las conclusiones relativas al modelo MatForgeNet y al cumplimiento del OE1.
- `PI/investigaciones/MatForge_SR_Informe_Tecnico.md` — base técnica para las conclusiones relativas al módulo de super-resolución y al cumplimiento matizado del OE2.
- `PI/investigaciones/MatForge_App_Informe_Herramientas.md` — base técnica para las conclusiones relativas a herramientas de refinado, exportación y despliegue.
- `PI/investigaciones/MatForge_Arquitectura_Permanente_v1.5.md` — documento permanente que consolida la arquitectura final del modelo y sus decisiones cerradas.
- `PI/investigaciones/MatForge_App_Arquitectura_Permanente.md` — documento permanente que consolida la arquitectura final de la aplicación.
- `PI/investigaciones/Informe_Benchmarking_MatForge.md` — base técnica para las conclusiones del OE6 y para la comparación frente a Pix2Pix, DeepPBR, Materialize y Adobe Substance 3D Sampler.
- `PI/investigaciones/Investigacion_Discriminador_GAN_MatForge.md` — respaldo técnico de la interpretación del colapso del discriminador y de la contribución de la feature matching loss.
- `PI/investigaciones/Investigacion_Tecnica_Dataset_Limpieza.md` — respaldo técnico de las conclusiones sobre EDA, limpieza y filtros por categoría.
- `PI/investigaciones/Investigacion_Tecnica_Relabeling.md` — respaldo técnico de las conclusiones sobre clustering, relabeling y clasificación funcional.
