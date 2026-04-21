import sys
import json
import os
import matplotlib.pyplot as plt
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
        # Crea la carpeta en el directorio actual
        import os

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
        #Bucle del hiperparámetro K
        ks = range(k_min, k_max + 1, step)
        for k in ks:  # Avanza con lo que el usuario del script crea conveniente
            print(f"\n--------------------------------------------------")
            print(f"--> Evaluando combinación: k={k}")

            # Llamamos a la función
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
        # Usamos .copy() para no tocar el 'data' original por error
        data_final = data.copy()
        data_final['cluster_id'] = modelo_final.labels_

        # 3. Extraer las palabras más importantes de cada tópico
        print("\n" + "=" * 30)
        print("  PALABRAS CLAVE POR TÓPICO")
        print("=" * 30)

        # Necesitamos el vectorizador que tenemos guardado en 'mis_herramientas'
        if 'vectorizers' in mis_herramientas:
            # Cogemos el nombre de la primera columna de texto que se vectorizó
            col_name = list(mis_herramientas['vectorizers'].keys())[0]
            vectorizador = mis_herramientas['vectorizers'][col_name]

            # Obtenemos los nombres de las palabras (columnas)
            nombres_palabras = vectorizador.get_feature_names_out()

            # Obtenemos los centroides (la "esencia" de cada grupo)
            centroides = modelo_final.cluster_centers_

            for i, centroide in enumerate(centroides):
                # Ordenamos los índices del centroide de mayor a menor peso
                # cogemos los 10 primeros
                indices_top = centroide.argsort()[::-1][:10]
                top_palabras = [nombres_palabras[idx] for idx in indices_top]

                print(f"\n[Tópico #{i}]")
                print(f" -> {', '.join(top_palabras)}")

        # 4. Guardar el resultado en un CSV para Tableau
        ruta_salida = "./clustering/resultados_agrupados.csv"
        data_final.to_csv(ruta_salida, index=False)
        print(f"\n[V] CSV con clusters guardado en: {ruta_salida}")