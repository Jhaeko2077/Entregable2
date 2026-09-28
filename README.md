# Clasificación de exoplanetas con redes neuronales

Análisis exploratorio y modelo de **deep learning** que clasifica el **tipo de planeta** a partir del *NASA Exoplanet Archive*, descargado con `kagglehub`.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)

## Pipeline

1. **Carga y exploración**: descarga del dataset y media y mediana por `planet_type`.
2. **Preprocesamiento**:
   - columnas numéricas normalizadas con `StandardScaler`;
   - columnas categóricas codificadas con `OneHotEncoder`;
   - etiquetas convertidas con `LabelEncoder`.
3. **División** en entrenamiento y prueba (80/20).
4. **Modelo**: red neuronal densa en Keras (120 → 60 → softmax), con optimizador Adam y pérdida `sparse_categorical_crossentropy`, entrenada 100 épocas.
5. **Evaluación**: predicciones sobre el conjunto de prueba y **matriz de confusión**.
6. **Valores atípicos**: detección por desviación estándar (|z| > 3) sobre la masa planetaria.
7. **Visualización**: histograma y diagramas de dispersión (Matplotlib y Seaborn).

## Ejecución

```bash
pip install -r requirements.txt
python entregable.py
```
