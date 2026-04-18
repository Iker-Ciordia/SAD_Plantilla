import pickle
import sys
import json
import pandas as pd
import entrenadorModelos as entrenadorModelos
# Asumo que tienes importada tu función load_data y apply_preprocessing

def imprimir_hiperparametros(modelo):
    """
    Función que detecta el tipo de algoritmo e imprime sus hiperparámetros clave.
    """
    hiperparametros = modelo.get_params()
    nombre_modelo = type(modelo).__name__

    print(f"\n" + "=" * 50)
    print(f"[*] EVALUANDO MODELO: {nombre_modelo}")
    print(f"[-] Hiperparámetros detectados en el .pkl:")

    # Filtramos los parámetros para imprimir solo los más relevantes
    if nombre_modelo == "KNeighborsClassifier":
        print(
            f"    -> k={hiperparametros.get('n_neighbors')}, p={hiperparametros.get('p')}, w={hiperparametros.get('weights')}")

    elif nombre_modelo == "DecisionTreeClassifier":
        print(
            f"    -> max_depth={hiperparametros.get('max_depth')}, min_samples_split={hiperparametros.get('min_samples_split')}, criterion={hiperparametros.get('criterion')}")

    elif nombre_modelo == "RandomForestClassifier":
        print(
            f"    -> n_estimators={hiperparametros.get('n_estimators')}, max_depth={hiperparametros.get('max_depth')}, min_samples_split={hiperparametros.get('min_samples_split')}, criterion={hiperparametros.get('criterion')}")

    elif "NB" in nombre_modelo:  # Cubre MultinomialNB, CategoricalNB, ComplementNB...
        print(f"    -> alpha={hiperparametros.get('alpha', 'No aplica (Gaussian)')}")

    else:
        # Por si pruebas otro algoritmo en el futuro, imprime todo por defecto
        for param, valor in hiperparametros.items():
            print(f"    -> {param}: {valor}")

    print("=" * 50 + "\n")


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

imprimir_hiperparametros(modelo)

#Obtener matriz de confusión
print("\n--- MATRIZ DE CONFUSIÓN ---")
print(entrenadorModelos.calculate_confusion_matrix(y_test_real, predicciones))

#Obtener métricas
print("\n--- MÉTRICAS ---")
entrenadorModelos.calculate_metrics(y_test_real, predicciones, config)