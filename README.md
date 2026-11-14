# Detección de intrusiones en red con autoencoders

Proyecto integrador del curso de Deep Learning (Pregrado en Ciencia de Datos, Universidad Externado de Colombia).

El proyecto tiene dos partes:

1. **Replicación** de Song et al. (2021), *Analysis of Autoencoders for Network Intrusion Detection*, sobre NSL-KDD. Un autoencoder entrenado solo con tráfico normal marca como ataque todo registro cuyo error de reconstrucción supera un umbral. Se compara con una línea base supervisada (XGBoost).
2. **Auditoría del aporte del deep learning.** ¿Cuánto aporta el autoencoder frente a métodos de reconstrucción con solución cerrada? Se comparan PCA, kernel PCA y un autoencoder a igual capacidad, y se separan tres factores: la no linealidad, el aprendizaje de la representación y la cantidad de datos.

## Resultados principales

Diferencia de AUC en test a igual capacidad, como promedio sobre la dimensión latente *k* = *z* ∈ {2, 4, 6, 8, 12, 16}, con IC del 95 % por bootstrap pareado:

| Aporte | Comparación | Δ AUC | IC 95 % |
|---|---|---|---|
| No linealidad fija | Kernel PCA − PCA | +0.0054 | [+0.0049, +0.0060] |
| Representación aprendida | AE − kernel PCA | −0.0067 | [−0.0071, −0.0062] |
| Más datos (AE) | AE completo − AE submuestra | −0.0013 | [−0.0014, −0.0011] |
| Total deep learning | AE − PCA (datos completos) | −0.0021 | [−0.0028, −0.0013] |

- En AUC, el autoencoder no supera a PCA, pierde frente a kernel PCA con los mismos datos y no mejora cuando recibe más datos.
- Su única ventaja aparece con una tasa de falsos positivos muy baja (TPR a FPR = 1 %). Kernel PCA la iguala.
- PCA cuesta órdenes de magnitud menos: se ajusta en unos 0.02 s y puntúa en 0.24 µs por registro. Es el modelo final elegido.
- La AUC de validación está saturada (≈ 0.99) y casi no anticipa la de test (Spearman entre −0.02 y 0.30). Una de las causas es un atajo del one-hot, que vuelve triviales al 31 % de los ataques de validación pero solo al 10 % de los de test.

El análisis completo está en `informe_final.Rmd`, y la bitácora de decisiones en `bitacora_experimento_aportes.md`.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `script.ipynb` | Replicación de Song et al.: preprocesamiento, línea base XGBoost, autoencoders (120 corridas) y PCA |
| `experimento_aportes.ipynb` | Experimento de aportes: PCA, kernel PCA (implementado en TensorFlow) y AE (3, *w*), con bootstrap pareado, costos y UMAP |
| `informe.Rmd` | Informe de la segunda entrega (replicación) |
| `informe_final.Rmd` | Informe de la entrega final (auditoría y dictamen sobre la profundidad) |
| `bitacora_experimento_aportes.md` | Decisiones, limitaciones y trabajo futuro del experimento de aportes |
| `resultados/` | CSV con las métricas de cada corrida; `resultados/aportes/` incluye scores, pesos e historiales del AE |
| `figuras/` | Figuras generadas por los notebooks |
| `preamble.tex`, `referencias.bib`, `apa.csl` | Formato, bibliografía y estilo de citas de los informes |

## Cómo reproducir

### 1. Datos

Los datos no se versionan. Descarga NSL-KDD (`KDDTrain+.txt`, `KDDTest+.txt` y `KDDTrain+.arff`) del [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html) y ubícalos en `data/nslkdd/`.

### 2. Entorno de Python

Se usó Python con TensorFlow 2.21 (GPU recomendada):

```bash
pip install tensorflow==2.21.0 keras==3.15.1 numpy pandas scikit-learn scipy matplotlib seaborn umap-learn xgboost
```

### 3. Notebooks

Ejecuta primero `script.ipynb` y después `experimento_aportes.ipynb`, porque el segundo usa resultados del primero. Las semillas están fijas: 42 para el split y semillas de 42 a 51 (replicación) y de 42 a 46 (aportes) para los autoencoders. El experimento de aportes tarda varias horas. Si defines la variable de entorno `APORTES_RAPIDO=1`, corre una versión reducida para verificar el flujo y guarda sus resultados en carpetas aparte.

### 4. Informes

Necesitas R con `rmarkdown`, `bookdown`, `ggplot2`, `knitr` y `kableExtra`, XeLaTeX y las fuentes Source Serif Pro y Lato:

```bash
Rscript -e 'rmarkdown::render("informe_final.Rmd")'
```

Las tablas y figuras de los informes se generan desde los CSV de `resultados/`.

## Referencia

Song, Y., Hyun, S., & Cheong, Y.-G. (2021). Analysis of Autoencoders for Network Intrusion Detection. *Sensors, 21*(13), 4294. https://doi.org/10.3390/s21134294

## Autoría y uso de IA

Autor: Yeiver Estevan Sanabria Sánchez. Profesor: Wilmer Pineda Ríos.

Se usaron herramientas de IA generativa (Claude) como apoyo en la implementación de algunas gráficas, la redacción y la búsqueda de literatura. Todas las decisiones técnicas fueron revisadas por el autor.
