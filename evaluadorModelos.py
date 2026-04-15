import pickle
import sys
import json
import pandas as pd
import entrenadorModelos as entrenadorModelos
# Asumo que tienes importada tu función load_data y apply_preprocessing

if len(sys.argv) < 5 or "-c" not in sys.argv:
    print("Uso: python script.py <fichero> <columna_objetivo> <mejor_modelo> -c <config.json>")
    sys.exit(1)

# Asignamos las variables desde la consola para que sea más fácil de leer
fichero = sys.argv[1]
columna_objetivo = sys.argv[2]
archivo_pickle = sys.argv[3]
indice_c = sys.argv.index("-c")
config = sys.argv[indice_c + 1]


print(f"Cargando modelo desde {archivo_pickle}...")
with open(archivo_pickle, 'rb') as f:
    paquete_cargado = pickle.load(f)

#Cargamos del mejor modelo las herramientas utilizadas
modelo = paquete_cargado['modelo']
mis_herramientas = paquete_cargado['herramientas_preproceso']

# 2. Cargar el test secreto
# (Usar load_data para asegurar que la columna objetivo va al final)
file = open(config, 'r')
config_json = json.load(file)
data_test = entrenadorModelos.load_data(fichero, columna_objetivo, config_json)

# 3. Preprocesamos los datos de testing
data_test_limpio = entrenadorModelos.apply_preprocessing(config, data_test, None, mis_herramientas)

# --- Separar atributos (X) de la clase real (Y) ---
X_test = data_test_limpio.iloc[:, :-1].values  # Todas las columnas menos la última
y_test_real = data_test_limpio.iloc[:, -1].values # Solo la última columna

# 4. Predecir
predicciones = modelo.predict(X_test)

#Printear resultados
df_resultados = pd.DataFrame({
    'Clase Real (Ground Truth)': y_test_real,
    'Predicción del Modelo': predicciones
})

print("\n--- RESULTADOS DE LAS PREDICCIONES ---")
print(df_resultados.to_string()) # to_string() fuerza a imprimir todas las filas sin cortarlas

#Obtener matriz de confusión
print("\n--- MATRIZ DE CONFUSIÓN ---")
print(entrenadorModelos.calculate_confusion_matrix(y_test_real, predicciones))

#Obtener métricas
print("\n--- MÉTRICAS ---")
entrenadorModelos.calculate_metrics(y_test_real, predicciones, config)