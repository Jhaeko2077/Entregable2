import kagglehub
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np

path = kagglehub.dataset_download("kanchana1990/nasa-exoplanet-archive-intelligence")

#Uso "exo" principal termino para mis variables ya que el DataSet trata de ExoPlanetas

#Carga y exploración de un dataset con pandas
files = os.listdir(path)
csv_file = [f for f in files if f.endswith('.csv')][0]
exo = pd.read_csv(os.path.join(path, csv_file))
print("Primeras 5 lineas:\n", exo.head(5))

#Media
print(exo.groupby('planet_type').mean(numeric_only=True))

#Mediana
print(exo.groupby('planet_type').median(numeric_only=True))

#Declaración de Objetos para la normalización (En este caso solo usaremos los siguientes)
scaler=StandardScaler()
nominal_encoder=OneHotEncoder(sparse_output=False)
label_encoder = LabelEncoder()

#Separamos columnas numéricas y categóricas
num_cols = exo.select_dtypes(include=['int64', 'float64']).columns
cat_cols = exo.select_dtypes(include=['object']).columns

#Normalizamos columnas numericas
exo_num = scaler.fit_transform(exo[num_cols])

#Codificar categóricas
exo_cat = nominal_encoder.fit_transform(exo[cat_cols])

#Unimos todo
exo_final = np.hstack([exo_num, exo_cat])
print("Datos normalizados y codificados:\n", exo_final)

#Convertimos etiquetas a números
y = label_encoder.fit_transform(exo['planet_type'])
#Datos para entrenamiento
X = exo_final

#Partimos datos
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(120, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

#Entrenamiento del modelo
model.fit(x_train, y_train, epochs=100, verbose=1)

#Predicción
muestra = x_test[0].reshape(1, X.shape[1])
y_pred = model.predict(muestra)

indice = np.argmax(y_pred, 1)
print("Predicción (clase):", indice)
print("Tipo de planeta:", label_encoder.inverse_transform(indice))

predicciones = model.predict(x_test)

print("Predicciones:")
predicciones = np.argmax(predicciones, 1)
print(predicciones)

print("Targets reales:")
print(y_test)

#MATRIZ DE CONFUSIÓN
print("Matriz de confusión:")
print(confusion_matrix(y_test, predicciones))

#Identificación de valores atípicos usando desviación estándar
col = 'planet_mass_earth'
media = exo[col].mean()
std = exo[col].std()
outliers = exo[abs((exo[col] - media) / std) > 3]
print(outliers[['planet_type', col]].head())

#GRAFICOS
#Histograma
plt.hist(exo[col], bins=30)
plt.title("Histograma")
plt.show()
#Dispersión
col2 = 'planet_mass_earth'
plt.scatter(exo[col], exo[col2])
plt.xlabel(col)
plt.ylabel(col2)
plt.title("Dispersión")
plt.show()
#Seaborn
sns.scatterplot(x=exo[col], y=exo[col2], hue=exo['planet_type'])
plt.show()