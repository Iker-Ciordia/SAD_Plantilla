import sys
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Forzamos a Python a mirar una carpeta más arriba (la raíz SAD_Plantilla)
ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_raiz not in sys.path:
    sys.path.append(ruta_raiz)
from clasificacion import entrenadorModelos

def K_Means(data, n_clusters=2, n_init=2):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    kmeans.fit(data)
    return kmeans


def extraer_top_palabras_por_clase(X_clustering, data_final, columna_objetivo,
                                   vectorizador, n_palabras=10):

    #Calcula las n mejores palabras de cada tópico
    nombres_palabras = vectorizador.get_feature_names_out()
    if isinstance(X_clustering, np.ndarray):
        X_array = X_clustering
    elif hasattr(X_clustering, "values"):
        X_array = X_clustering.values
    else:
        X_array = X_clustering.toarray()

    # Aseguramos que los índices están alineados
    X_array_reindexed = X_array  # ya viene alineado con data_final

    clusters_unicos = sorted(data_final["cluster_id"].unique())
    clases_unicas   = sorted(data_final[columna_objetivo].dropna().unique())

    resultados = []

    print("\n" + "=" * 50)
    print("  PALABRAS CLAVE POR TÓPICO Y CLASE REAL")
    print("=" * 50)

    for cluster_id in clusters_unicos:
        print(f"\n{'─' * 50}")
        print(f"  TÓPICO #{cluster_id}")
        print(f"{'─' * 50}")

        mascara_cluster = (data_final["cluster_id"] == cluster_id).values

        for clase in clases_unicas:
            mascara_clase   = (data_final[columna_objetivo] == clase).values
            mascara_conjunta = mascara_cluster & mascara_clase

            n_instancias = mascara_conjunta.sum()
            if n_instancias == 0:
                print(f"\n  [{clase}]  →  (sin instancias en este tópico)")
                continue

            # Centroide medio del subgrupo
            subgrupo   = X_array_reindexed[mascara_conjunta]
            centroide  = subgrupo.mean(axis=0)

            indices_top  = centroide.argsort()[::-1][:n_palabras]
            top_palabras = [(nombres_palabras[i], round(float(centroide[i]), 5))
                            for i in indices_top]

            print(f"\n  [{clase}]  ({n_instancias} instancias)")
            for rank, (palabra, peso) in enumerate(top_palabras, start=1):
                print(f"    {rank:>2}. {palabra:<25} {peso:.5f}")

            for rank, (palabra, peso) in enumerate(top_palabras, start=1):
                resultados.append({
                    "cluster_id"  : cluster_id,
                    "clase"       : clase,
                    "n_instancias": n_instancias,
                    "rank"        : rank,
                    "palabra"     : palabra,
                    "peso_medio"  : peso,
                })

    df_resultado = pd.DataFrame(resultados)
    return resultados, df_resultado


if __name__ == "__main__":
    # Pedimos fichero, objetivo y obligatoriamente el JSON
    if len(sys.argv) < 4 or "-c" not in sys.argv:
        print("Uso: python script.py <fichero> <columna_objetivo> -c <config_file.json>")
        sys.exit(1)

    # Asignamos las variables desde la consola para que sea más fácil de leer
    fichero = sys.argv[1]
    columna_objetivo = sys.argv[2]

    # 1. Buscamos el archivo JSON en los argumentos
    config_file = None
    indice_c = sys.argv.index("-c")
    if indice_c + 1 < len(sys.argv):
        config_file = sys.argv[indice_c + 1]

    # 2. Cargamos el JSON de configuración
    algoritmo = "K-Means"  # Valor por defecto
    if config_file:
        with open(config_file, 'r') as file:
            config = json.load(file)
        algoritmo = config.get("algorithm", "K-Means")

    # --- INICIO DEL FLUJO DE CLUSTERING COMÚN ---

    # A. Cargamos los datos
    data = entrenadorModelos.load_data(fichero, columna_objetivo, config)

    # C. Aplicamos el preprocesado pasándole ambos trozos
    if config_file:
        data_pre, _, mis_herramientas = entrenadorModelos.apply_preprocessing(config_file, data, None)  # Preprocesamos los datos (lematizar, tokenizar, eliminar stopwords... para clustering
        X_clustering = data_pre.iloc[:, :-1] #Nos quedamos solo con los comentarios. En clustering nos da igual la clase real

    # Bloque para ver los datos preprocesados
    try:
        os.mkdir("./clustering/datos_preprocesados_clustering")
        print(f"Directorio 'datos_preprocesados_clustering' creado exitosamente.")
    except FileExistsError:
        print(f"Error: El directorio 'datos_preprocesados_clustering' ya existe.")
    data_pre.to_csv("./clustering/datos_preprocesados_clustering/datos_preprocesados.csv", index=False)

    # --- ENRUTADOR DE ALGORITMOS ---

    ##########################
    #  Empieza algoritmo K-Means #
    ##########################
    if algoritmo == "K-Means":
        print("\n[->] Ejecutando modelo: K-Means")

        # Leemos los rangos del JSON (con valores por defecto por si acaso)
        hiper_kmeans = config.get("hyperparametersKMeans", {})
        k_min = hiper_kmeans.get("k_min", 3)
        k_max = hiper_kmeans.get("k_max", 3)
        step = hiper_kmeans.get("step", 1)
        n_inicios = hiper_kmeans.get("n_inicios", 10)

        inercias = []
        ks = range(k_min, k_max + 1, step)
        for k in ks:  # Avanza con lo que el usuario del script crea conveniente
            print(f"\n--------------------------------------------------")
            print(f"--> Evaluando combinación: k={k}")
            modelo = K_Means(X_clustering, n_clusters=k, n_init=n_inicios)
            inercias.append(modelo.inertia_) #La inercia obtenida con este modelo de KMeans la guardamos para el codo.

        # --- GENERAR GRÁFICO DEL CODO ---
        plt.figure(figsize=(8, 5))
        plt.plot(ks, inercias, 'bx-')
        plt.xlabel('Número de Clústeres (K)')
        plt.ylabel('Inercia (Suma de distancias)')
        plt.title('Método del Codo para determinar K óptimo')
        plt.show()

        # --- PROCESO FINAL DE EXTRACCIÓN ---

        # 1. Elegimos el K que nos ha gustado del gráfico del codo
        k_final = int(input("\n[?] Introduce el valor de K óptimo que has visto en el gráfico: "))
        modelo_final = K_Means(X_clustering, n_clusters=k_final, n_init=n_inicios)

        # 2. Asignar cada instancia a su cluster en el DataFrame original
        data_final = data.copy()
        data_final['cluster_id'] = modelo_final.labels_

        # 3. Imprimir pesos de las palabras clave por tópico
        if 'vectorizers' in mis_herramientas:
            col_name     = list(mis_herramientas['vectorizers'].keys())[0]
            vectorizador = mis_herramientas['vectorizers'][col_name]

            # Necesitamos X como array denso con el mismo índice que data_final
            if isinstance(X_clustering, np.ndarray):
                X_array = X_clustering
            elif hasattr(X_clustering, "values"):
                X_array = X_clustering.values
            else:
                X_array = X_clustering.toarray()

            _, df_palabras_clase = extraer_top_palabras_por_clase(
                X_array, data_final, columna_objetivo, vectorizador, n_palabras=10
            )

            # Guardar CSV de palabras clave por tópico y clase
            ruta_palabras_clase = "./clustering/palabras_clave_por_clase.csv"
            df_palabras_clase.to_csv(ruta_palabras_clase, index=False)
            print(f"\n[V] CSV de palabras clave por tópico y clase guardado en: {ruta_palabras_clase}")

        # 4. Guardar el resultado principal con los cluster_id en CSV
        ruta_salida = "./clustering/resultados_agrupados.csv"
        data_final.to_csv(ruta_salida, index=False)
        print(f"\n[V] CSV con clusters guardado en: {ruta_salida}")