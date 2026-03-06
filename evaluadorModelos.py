import pickle

import json
import pandas as pd
import entrenadorModelos as entrenadorModelos
# Asumo que tienes importada tu función load_data y apply_preprocessing

# 1. Cargar el paquete
with open('mejor_modelo_knn.pkl', 'rb') as archivo:
    paquete_cargado = pickle.load(archivo)

modelo_knn = paquete_cargado['modelo_knn']
mis_herramientas = paquete_cargado['herramientas_preproceso']

# 2. Cargar el test secreto
# (Usar load_data para asegurar que la columna objetivo va al final)
data_test = entrenadorModelos.load_data("iris.csv", "Especie") # Pon aquí el nombre de tu columna

with open("config_file.json", 'r') as file:
    config = json.load(file)

# 3. Preprocesamos los datos de testing
data_test_limpio = entrenadorModelos.apply_preprocessing("config_file.json", data_test, None, mis_herramientas)

# --- Separar atributos (X) de la clase real (Y) ---
X_test = data_test_limpio.iloc[:, :-1].values  # Todas las columnas menos la última
y_test_real = data_test_limpio.iloc[:, -1].values # Solo la última columna

# 4. Predecir (¡Solo le pasamos la X al modelo!)
predicciones = modelo_knn.predict(X_test)

#Printear resultados
df_resultados = pd.DataFrame({
    'Clase Real (Ground Truth)': y_test_real,
    'Predicción del Modelo': predicciones
})
print("\n--- RESULTADOS DE LAS PREDICCIONES ---")
print(df_resultados.to_string()) # to_string() fuerza a imprimir todas las filas sin cortarlas