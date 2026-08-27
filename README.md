# Deep Learning Sequence Models Lab

Este repositorio es un laboratorio práctico centrado en el modelado de datos secuenciales utilizando **Deep Learning (TensorFlow/Keras)**. A través de tres proyectos aplicados a dominios diferentes (Finanzas, NLP y Meteorología), se explora el comportamiento, las ventajas y las limitaciones de arquitecturas como **RNN, LSTM y GRU**.

## Proyectos Incluidos

### 1. Predicción Bursátil Multi-paso (Finanzas)
* **Script:** `01_Finance_GRU/finance_forecasting.py`
* **Descripción:** Extracción de datos históricos en tiempo real mediante la API `yfinance` (Apple Inc.). Implementación de modelado temporal con ventanas deslizantes (*sliding windows*) para predecir **simultáneamente el precio de cierre de los próximos 3 días**, utilizando una red **GRU**.

### 2. Análisis de Sentimiento y Optimización (NLP)
* **Script:** `02_NLP_LSTM/nlp_sentiment_optuna.py`
* **Descripción:** Clasificación binaria de reseñas de películas (dataset IMDb). Compara el rendimiento de arquitecturas mediante capas de `Embedding` conectadas a redes **LSTM**. Además, integra **Optuna** para la búsqueda automatizada y eficiente de los mejores hiperparámetros (tasa de aprendizaje, dropout, neuronas, etc.).

### 3. Predicción Meteorológica Multivariable (Clima)
* **Script:** `03_Weather_Forecasting/weather_forecasting.py`
* **Descripción:** Procesamiento del masivo *Jena Climate Dataset*. Incluye técnicas de *Data Engineering* como *downsampling* temporal (de 10 minutos a medias por hora) e imputación de valores nulos. Entrena una red **GRU** multivariable para predecir la temperatura exacta a **24 horas vista** usando los últimos 3 días de contexto de presión, humedad y viento.

## Insights y Conclusiones Técnicas

Tras el desarrollo de los experimentos, se han extraído las siguientes conclusiones sobre las redes recurrentes:

* **Contexto Bidireccional:** Las LSTM Bidireccionales demostraron mejorar significativamente la precisión en NLP, ya que el sentimiento de una palabra suele depender del final de la oración. Sin embargo, presentan una tendencia drástica al *overfitting*, requiriendo una regularización estricta (Early Stopping y Dropout altos).
* **Correlación Temporal:** En el pronóstico climático, se comprobó matemáticamente cómo el margen de error promedio crece exponencialmente con el tiempo (de ~1.2°C a 1h vista, hasta >3.1°C a 24h vista) debido a la degradación natural de la correlación a largo plazo.
* **El Peligro de Ventanas Inmensas:** Aumentar indiscriminadamente la ventana temporal de entrada no garantiza un mejor modelo. Frecuentemente, solo expone a la red a gradientes desvanecientes y a asimilar "ruido" irrelevante del pasado lejano.

## Estructura del Repositorio

```text
Deep-Learning-Sequence-Models/
├── 01_Finance_GRU/
│   └── finance_forecasting.py
├── 02_NLP_LSTM/
│   └── nlp_sentiment_optuna.py
├── 03_Weather_Forecasting/
│   └── weather_forecasting.py
├── requirements.txt
├── .gitignore
└── README.md
