# Bitácora: experimento de aportes (PCA, kernel PCA y autoencoder)

Notebook: `experimento_aportes.ipynb` · Resultados: `resultados/aportes/` · Figuras: `figuras/aportes_*.png`
Fechas: 5 y 6 de noviembre de 2026

## Resumen

**Pregunta.** En NSL-KDD, ¿cuánto aporta un autoencoder entrenado por descenso del gradiente frente a métodos clásicos de reconstrucción? Se separan tres aportes: la no linealidad, el aprendizaje de la representación y la cantidad de datos.

**Diseño.** Tres métodos con el mismo esquema (comprimir a *k* dimensiones, reconstruir y usar el error L2 como *score*), el mismo split y el mismo preprocesamiento:

| Método | Representación | Cómo se obtiene |
|---|---|---|
| PCA | Lineal | Solución cerrada |
| Kernel PCA (RBF) | No lineal, fija | Solución cerrada |
| AE (3, *w*): `entrada → w → z → w → salida` | No lineal, aprendida | Adam |

**Resultado principal** (Δ AUC en test, promedio sobre *k* = *z* ∈ {2, 4, 6, 8, 12, 16}):

| Aporte | Δ AUC | IC 95 % | *k* ganados |
|---|---|---|---|
| No linealidad fija (KPCA − PCA) | +0.0054 | [+0.0049, +0.0060] | 6 de 6 |
| Representación aprendida (AE − KPCA) | −0.0067 | [−0.0071, −0.0062] | 0 de 6 |
| Más datos, AE | −0.0013 | [−0.0014, −0.0011] | 1 de 6 |
| Más datos, PCA | −0.0005 | [−0.0005, −0.0004] | 0 de 6 |
| Total deep learning (AE − PCA, completo) | −0.0021 | [−0.0028, −0.0013] | 3 de 6 |

Las diferencias son pequeñas: todas por debajo de 0.012 de AUC. Los intervalos solo capturan la variación por la muestra de test; no incluyen la variación entre semillas del AE (desviación de 0.002 a 0.02). La conclusión más sólida es KPCA − PCA, porque no tiene semillas y su signo se mantiene en todos los *k*.

**Hallazgo metodológico.** La AUC de validación está saturada (0.98–0.99) y casi no anticipa la de test. La correlación de Spearman entre ambas es 0.23 (AE), 0.30 (kernel PCA) y −0.02 (PCA). Elegir configuraciones por validación habría invertido algunas conclusiones.

---

## Bitácora de decisiones

### 1. Punto de partida: revisión de fuga de información en PCA
- **Pregunta:** ¿el buen desempeño de PCA en `script.ipynb` se debía a alguna variable que revelara el ataque?
- **Conclusión:** no hay fuga directa. `label`, `difficulty` y `attack` están excluidas, el ajuste usa solo normales del train y el umbral se elige en validación. La AUC de test de PCA (~0.96) es casi igual a la del AE; las diferencias de MCC vienen del umbral.
- **Hallazgo lateral:** el one-hot se ajusta solo con normales (`handle_unknown="ignore"`). Un servicio que ningún normal usó deja el bloque en ceros y es trivialmente anómalo. Eso pasa con el 31 % de los ataques de validación y solo con el 10.5 % de los de test.

### 2. Qué comparar
- **Propuesta inicial descartada:** un diseño 2×2 (lineal / no lineal × superficial / profundo) con autoencoders.
- **Razón:** el AE `entrada → z → salida` y el AE lineal profundo solo servirían como controles; lo segundo equivale a PCA por construcción.
- **Decisión:** comparar PCA, kernel PCA y un AE no lineal.

### 3. Profundidad del AE: AE (3, *w*)
- **Motivo:** kernel PCA equivale a una red con una capa oculta muy ancha de neuronas RBF fijas (una por punto de la submuestra) a cada lado de la latente. Un AE (3, *w*) tiene la misma estructura con neuronas aprendidas.
- **Interpretación:** AE − KPCA no mide la profundidad, sino el aporte de aprender la representación por gradiente frente a fijarla con un kernel.
- **Referencia de mayor profundidad:** el AE (5, 64) del notebook anterior, sin volver a entrenarlo.
- **Equiparar por número de parámetros, descartado:** kernel PCA es no paramétrico y guarda la submuestra completa.

### 4. Control del tamaño de muestra
- **Problema:** kernel PCA no cabe en memoria con los 53 874 normales (la matriz kernel ocuparía ~23 GB).
- **Decisión:** submuestra de *m* = 10 000 normales, la misma para los tres métodos. PCA y el AE se entrenan además con los datos completos.
- **Comparaciones resultantes:** KPCA(m) − PCA(m), AE(m) − KPCA(m), AE(completo) − AE(m) y AE(completo) − PCA(completo).
- **Estratificación:** por `service`, agrupando en "otros" los servicios con menos de 10 registros. No hay clases que estratificar (todos son normales), pero si un servicio normal raro quedara fuera de la submuestra, los modelos lo tratarían como anómalo.

### 5. Presupuesto de entrenamiento del AE
- **Concepto clave:** actualizaciones = épocas × ⌈n / batch⌉. Con el mismo número de épocas, la submuestra recibiría 5 veces menos actualizaciones.
- **Decisión:** igualar el presupuesto en **12 000 actualizaciones** como tope, fijado a partir de las corridas anteriores (sin piloto).
  - Datos completos: 114 épocas.
  - Submuestra: 600 épocas.
- **Fijos en ambas condiciones:** batch de 512, Adam con lr = 1e-4 y weight decay = 1e-5, `shuffle=True`.
- **Early stopping:** paciencia de 5 evaluaciones y restauración de los mejores pesos. Las evaluaciones se alinean cada ~200 actualizaciones (cada 2 épocas con los datos completos y cada 10 con la submuestra), para que la paciencia equivalga a ~1 000 actualizaciones en ambas condiciones.

### 6. Rejillas y semillas
- *k* = *z* ∈ {2, 4, 6, 8, 12, 16}, común a los tres métodos.
- *w* ∈ {32, 64, 128, 256}, elegido con una sola semilla (42) por condición. Salió *w* = 64 en ambas.
- Semillas 42–46 para el AE con el *w* elegido.
- Kernel PCA: γ = γ₀ × {0.1, 0.3, 1, 3, 10}, con γ₀ por la heurística de la mediana, y `alpha` ∈ {0.1, 1} para la pre-imagen.

### 7. Implementación de kernel PCA en TensorFlow (GPU)
- **Motivo:** medir el costo de los tres métodos en el mismo dispositivo.
- **Memoria:** la matriz de ajuste cabe completa (400 MB en float32); validación y test se procesan por bloques de 2 048 filas.
- **Reutilización:** una descomposición espectral por γ, con *k* máx. = 16; cada *k* toma las primeras columnas y cada `alpha` solo cambia la pre-imagen.
- **Precisión:** matriz kernel y descomposición en float32 (en float64 tardaba más de 2 minutos por γ en la GPU). Pre-imagen en float64: en float32 el sistema ridge daba diferencias de 0.6 % frente a sklearn.
- **Validación:** frente a `sklearn.KernelPCA`, la diferencia es de 2e-4 en la proyección y 3e-5 en el error, con la misma AUC.
- **Rendimiento del AE:** el entrenamiento pasó a `tf.data`, ~30 % más rápido por menor costo fijo por época.

### 8. Métricas
- AUC en test como métrica principal; MCC como secundaria, porque depende del traslado del umbral desde validación.
- Bootstrap pareado (1 000 remuestreos, los mismos índices para todos los métodos) en lugar de varias semillas para los métodos deterministas.
- Costo: tiempo de ajuste, inferencia por registro y memoria.

### 9. Después del primer resultado: la validación no sirve para elegir
- **Observación:** con selección por AUC de validación, kernel PCA eligió k = 16 y γ = 10·γ₀ y quedaba por debajo de PCA. A igual *k*, en cambio, lo superaba.
- **Optuna o búsqueda bayesiana, descartadas:** cambian cómo se busca, no qué se optimiza. Afinar más contra una validación poco representativa la sobreajusta.
- **Error de reconstrucción de los normales como criterio, descartado:** no usa etiquetas, pero mide el ajuste a lo normal y no la detección; tiende a premiar la memorización.
- **Quitar de la validación los ataques del atajo, descartado:** se decidiría mirando test, lo que es una fuga indirecta.
- **Decisión:** análisis principal **a igual capacidad**, sin elegir *k*. Kernel PCA con γ = γ₀ y `alpha` = 1 fijados a priori. La selección por validación queda como resultado secundario.

### 10. Brecha entre validación y test
- **Cuantificación:**

| | Validación | Test |
|---|---|---|
| % de ataques | 46.5 | 56.9 |
| Tipos de ataque | 20 | 37 |
| % de ataques de tipos que no están en el train | 0 | 29.2 |
| % de ataques en el atajo del one-hot | 31.2 | 10.5 |

- **Decisión:** reportarla como característica del dataset (`KDDTest+` es más difícil a propósito) y no corregir el preprocesamiento a posteriori, porque eso sería un grado de libertad del investigador después de ver test.

---

## Limitaciones
- El preprocesamiento se ajusta con todos los normales, también en las condiciones con submuestra.
- El AE elige su checkpoint y su early stopping con la AUC de validación (usa etiquetas). *w* = 64 también se eligió con validación, con diferencias ≤ 0.003.
- El umbral del MCC se fija con las etiquetas de validación (no afecta a la AUC).
- Los intervalos del bootstrap no incluyen la variación entre semillas del AE.
- El AE (5, 64) de referencia usa otro protocolo (100 épocas, sin early stopping) y no tiene *scores* guardados para el bootstrap.

## Trabajo futuro
- Análisis de sensibilidad con `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)`, para comprobar si el orden de los métodos se mantiene sin el atajo (requiere volver a entrenar, ~2 h).
- Una validación que simule ataques nuevos, dejando tipos de ataque fuera del train.
- Incluir la variación entre semillas en el bootstrap, remuestreando también las semillas.
