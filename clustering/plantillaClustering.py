import sys
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import gensim                          # Librería principal para LDA
import gensim.corpora                  # Para construir el Dictionary y el corpus BOW
import gensim.models                   # Para LdaModel y CoherenceModel

# Forzamos a Python a mirar una carpeta más arriba (la raíz SAD_Plantilla)
ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_raiz not in sys.path:
    sys.path.append(ruta_raiz)
from clasificacion import entrenadorModelos


def K_Means(data, n_clusters=2, n_init=2):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    kmeans.fit(data)
    return kmeans


def construir_corpus_gensim(X_array, nombres_palabras):
    """
    Convierte la matriz TF-IDF (numpy array) al formato que necesita Gensim para LDA.

    Gensim no trabaja con matrices densas sino con dos estructuras:
      - Dictionary: mapeo de id numérico -> palabra (y viceversa)
      - Corpus BOW : lista de documentos, donde cada documento es una lista
                     de tuplas (word_id, peso) para las palabras con peso > 0

    Al usar los pesos TF-IDF como sustituto del conteo de palabras, LDA puede
    aprender las distribuciones temáticas sobre el mismo espacio de características
    que ya ha preprocesado el pipeline del proyecto.

    Parámetros:
        X_array         : np.ndarray (n_instancias x n_palabras) con pesos TF-IDF
        nombres_palabras: array de strings con el vocabulario del vectorizador

    Retorna:
        dictionary: gensim.corpora.Dictionary construido desde el vocabulario
        corpus    : lista de listas de tuplas (word_id, peso)
    """
    # Construimos el Dictionary de Gensim a partir del vocabulario del vectorizador TF-IDF.
    # El Dictionary es simplemente un mapa bidireccional id <-> palabra.
    id2word = dict(enumerate(nombres_palabras))
    dictionary = gensim.corpora.Dictionary()
    dictionary.id2token = id2word
    dictionary.token2id = {palabra: idx for idx, palabra in id2word.items()}

    # Gensim necesita también algunos contadores internos para que CoherenceModel funcione
    dictionary.num_docs = X_array.shape[0]
    dictionary.num_pos  = int(X_array.sum())           # total de pesos sumados
    dictionary.num_nnz  = int((X_array > 0).sum())     # total de entradas no nulas

    # Construimos el corpus BOW: para cada documento (fila), guardamos solo las
    # palabras con peso > 0 como lista de (word_id, peso). Así evitamos pasar
    # millones de ceros a Gensim, que es ineficiente.
    corpus = []
    for fila in X_array:
        bow = [(int(i), float(peso)) for i, peso in enumerate(fila) if peso > 0]
        corpus.append(bow)

    return dictionary, corpus


def extraer_top_palabras_por_cluster(X_array, data_final, vectorizador, n_palabras=10):
    """
    Para cada cluster calcula las N palabras más importantes usando el centroide
    medio de los vectores TF-IDF de las instancias que pertenecen a ese cluster.

    Como todos los datos que se pasan al script son de la misma clase (positivo,
    negativo o neutro), no tiene sentido desglosar por clase: cada cluster
    representa directamente un tópico temático dentro de esa clase.

    Parámetros:
        X_array      : np.ndarray (n_instancias x n_palabras) con pesos TF-IDF
        data_final   : DataFrame original con la columna 'cluster_id' añadida
        vectorizador : vectorizador TF-IDF ajustado en el preprocesado
        n_palabras   : número de palabras top a extraer por cluster

    Retorna:
        resultados  : lista de dicts con keys cluster_id, rank, palabra, peso_medio
        df_resultado: DataFrame listo para guardar como CSV
    """
    nombres_palabras = vectorizador.get_feature_names_out()
    clusters_unicos  = sorted(data_final["cluster_id"].unique())
    resultados       = []

    print("\n" + "=" * 50)
    print("  PALABRAS CLAVE POR TÓPICO")
    print("=" * 50)

    for cluster_id in clusters_unicos:
        print(f"\n{'─' * 50}")
        print(f"  TÓPICO #{cluster_id}")
        print(f"{'─' * 50}")

        # Seleccionamos solo las instancias que pertenecen a este cluster
        mascara = (data_final["cluster_id"] == cluster_id).values
        n_instancias = mascara.sum()

        # Calculamos el centroide medio: promedio de los vectores TF-IDF del cluster
        subgrupo  = X_array[mascara]
        centroide = subgrupo.mean(axis=0)

        # Ordenamos las palabras por peso descendente y cogemos el top N
        indices_top  = centroide.argsort()[::-1][:n_palabras]
        top_palabras = [(nombres_palabras[i], round(float(centroide[i]), 5))
                        for i in indices_top]

        print(f"  ({n_instancias} instancias)")
        for rank, (palabra, peso) in enumerate(top_palabras, start=1):
            print(f"    {rank:>2}. {palabra:<25} {peso:.5f}")

        for rank, (palabra, peso) in enumerate(top_palabras, start=1):
            resultados.append({
                "cluster_id"  : cluster_id,
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

    # Convertimos X_clustering a numpy array denso una sola vez aquí,
    # ya que ambos algoritmos lo necesitan en ese formato
    if isinstance(X_clustering, np.ndarray):
        X_array = X_clustering
    elif hasattr(X_clustering, "values"):
        X_array = X_clustering.values
    else:
        X_array = X_clustering.toarray()

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
        # --- GUARDAR DATOS PARA TABLEAU ---
        df_codo = pd.DataFrame({
            'numero_clusters_k': list(ks),
            'inercia': inercias
        })
        ruta_codo = "./clustering/kmeans_datos_codo_tableau.csv"
        df_codo.to_csv(ruta_codo, index=False)
        print(f"[V] Datos del gráfico del codo guardados para Tableau en: {ruta_codo}")

        plt.show()

        # --- PROCESO FINAL DE EXTRACCIÓN ---

        # 1. Elegimos el K que nos ha gustado del gráfico del codo
        k_final = int(input("\n[?] Introduce el valor de K óptimo que has visto en el gráfico: "))
        modelo_final = K_Means(X_clustering, n_clusters=k_final, n_init=n_inicios)

        # 2. Asignar cada instancia a su cluster en el DataFrame original
        data_final = data.copy()
        data_final['cluster_id'] = modelo_final.labels_

        # 3. Imprimir y guardar las palabras clave por tópico
        if 'vectorizers' in mis_herramientas:
            col_name     = list(mis_herramientas['vectorizers'].keys())[0]
            vectorizador = mis_herramientas['vectorizers'][col_name]

            _, df_palabras = extraer_top_palabras_por_cluster(
                X_array, data_final, vectorizador, n_palabras=10
            )

            # Guardar CSV de palabras clave por tópico
            ruta_palabras = "./clustering/kmeans_palabras_clave_por_topico.csv"
            df_palabras.to_csv(ruta_palabras, index=False)
            print(f"\n[V] CSV de palabras clave por tópico guardado en: {ruta_palabras}")

        # 4. Guardar el resultado principal con los cluster_id en CSV
        ruta_salida = "./clustering/kmeans_resultados_agrupados.csv"
        data_final.to_csv(ruta_salida, index=False)
        print(f"\n[V] CSV con clusters guardado en: {ruta_salida}")

    ##############################
    #  Empieza algoritmo LDA     #
    ##############################
    elif algoritmo == "LDA":
        print("\n[->] Ejecutando modelo: LDA (Latent Dirichlet Allocation)")

        # Leemos los hiperparámetros del JSON (con valores por defecto por si acaso).
        # El JSON debe tener una sección "hyperparametersLDA" con estos campos:
        #   k_min              : número mínimo de tópicos a explorar
        #   k_max              : número máximo de tópicos a explorar
        #   step               : salto entre valores de k en la exploración
        #   passes             : número de pasadas sobre el corpus en el entrenamiento
        #                        (más pasadas = modelo más ajustado, pero más lento)
        #   alpha              : prior de Dirichlet sobre la distribución documento-tópico.
        #                        'auto' deja que Gensim lo aprenda solo de los datos.
        #   eta                : prior de Dirichlet sobre la distribución tópico-palabra.
        #                        'auto' deja que Gensim lo aprenda solo del corpus.
        #   coherencia_metrica : métrica para evaluar la calidad de los tópicos.
        #                        Opciones:
        #                          "u_mass" → solo necesita el corpus BOW. Más rápida.
        #                                     Valores más cercanos a 0 = mejor.
        #                          "c_uci"  → necesita los textos tokenizados.
        #                                     Valores más altos = mejor.
        #                          "c_v"    → necesita los textos tokenizados. La más
        #                                     usada en la teoría. Valores entre 0 y 1,
        #                                     más alto = mejor.
        hiper_lda = config.get("hyperparametersLDA", {})
        k_min              = hiper_lda.get("k_min", 2)
        k_max              = hiper_lda.get("k_max", 10)
        step               = hiper_lda.get("step", 1)
        passes             = hiper_lda.get("passes", 10)
        alpha              = hiper_lda.get("alpha", "auto")
        eta                = hiper_lda.get("eta", "auto")
        coherencia_metrica = hiper_lda.get("coherencia_metrica", "u_mass")

        # Validamos que la métrica elegida sea una de las tres soportadas
        metricas_validas = {"u_mass", "c_uci", "c_v"}
        if coherencia_metrica not in metricas_validas:
            print(f"[!] Métrica '{coherencia_metrica}' no reconocida. Usando 'u_mass' por defecto.")
            coherencia_metrica = "u_mass"

        if 'vectorizers' not in mis_herramientas:
            print("[!] No se encontró el vectorizador en mis_herramientas. Saliendo.")
            sys.exit(1)

        col_name         = list(mis_herramientas['vectorizers'].keys())[0]
        vectorizador     = mis_herramientas['vectorizers'][col_name]
        nombres_palabras = vectorizador.get_feature_names_out()

        # --- CONSTRUCCIÓN DE DATOS GENSIM ---
        # LDA en Gensim no acepta matrices directamente: necesita un diccionario
        # (vocabulario) y un corpus en formato BOW.
        # Usamos la función auxiliar construir_corpus_gensim para generarlos
        # a partir de la matriz TF-IDF que ya tenemos preprocesada.
        dictionary, corpus = construir_corpus_gensim(X_array, nombres_palabras)

        # --- TEXTOS TOKENIZADOS (solo necesarios para c_uci y c_v) ---
        # u_mass trabaja directamente con el corpus BOW, pero c_uci y c_v
        # necesitan los textos como listas de tokens para calcular las
        # co-ocurrencias de palabras en una ventana deslizante.
        # Los reconstruimos a partir de X_array: para cada documento tomamos
        # los nombres de las columnas (palabras) con peso TF-IDF > 0.
        # Nota: se pierde el orden y la repetición de palabras, pero es suficiente
        # para que CoherenceModel calcule las co-ocurrencias correctamente.
        if coherencia_metrica in {"c_uci", "c_v"}:
            texts = [
                [nombres_palabras[i] for i in np.where(fila > 0)[0]]
                for fila in X_array
            ]
        else:
            texts = None  # u_mass no los necesita

        print(f"\n[i] Métrica de coherencia seleccionada: {coherencia_metrica}")

        # --- BÚSQUEDA DEL NÚMERO ÓPTIMO DE TÓPICOS ---
        # Equivalente al metodo del codo de K-Means, pero usando la métrica de coherencia
        # elegida. La coherencia mide si las palabras más probables de cada tópico
        # tienden a aparecer juntas en los documentos.
        coherencias = []
        ks = range(k_min, k_max + 1, step)
        for k in ks:
            print(f"\n--------------------------------------------------")
            print(f"--> Evaluando combinación: num_topics={k}")

            # Entrenamos un LDA temporal solo para medir la coherencia con este k
            lda_temp = gensim.models.LdaModel(
                corpus=corpus,
                id2word=dictionary.id2token,
                num_topics=k,
                passes=passes,
                alpha=alpha,
                eta=eta,
                random_state=42     # Fijamos la semilla para reproducibilidad
            )

            # Construimos CoherenceModel pasando los argumentos que correspondan
            # según la métrica elegida:
            #   u_mass → solo necesita corpus + dictionary
            #   c_uci  → necesita texts + dictionary (co-ocurrencias en ventana externa)
            #   c_v    → necesita texts + dictionary (la más robusta de las tres)
            if coherencia_metrica == "u_mass":
                coherence_model = gensim.models.CoherenceModel(
                    model=lda_temp,
                    corpus=corpus,
                    dictionary=dictionary,
                    coherence='u_mass'
                )
            else:
                # processes=1 evita el deadlock que causa Gensim en Windows
                # cuando c_uci o c_v intentan lanzar workers en paralelo
                coherence_model = gensim.models.CoherenceModel(
                    model=lda_temp,
                    texts=texts,
                    dictionary=dictionary,
                    coherence=coherencia_metrica,
                    processes=1
                )

            coherencia = coherence_model.get_coherence()
            coherencias.append(coherencia)
            print(f"    Coherencia {coherencia_metrica}: {coherencia:.4f}")

        # --- GENERAR GRÁFICO DE COHERENCIA ---
        # Buscamos el pico más alto de la curva (o el punto donde se estanca).
        # Referencia por métrica:
        #   u_mass → más cercano a 0 (menos negativo) = mejor
        #   c_uci  → valor más alto = mejor
        #   c_v    → valor más alto (entre 0 y 1) = mejor
        plt.figure(figsize=(8, 5))
        plt.plot(ks, coherencias, 'rx-')
        plt.xlabel('Número de Tópicos (K)')
        plt.ylabel(f'Coherencia {coherencia_metrica}')
        plt.title(f'Coherencia {coherencia_metrica} por número de tópicos LDA')

        # --- GUARDAR DATOS PARA TABLEAU ---
        df_coherencia = pd.DataFrame({
            'numero_topicos_k': list(ks),
            f'coherencia_{coherencia_metrica}': coherencias
        })
        ruta_coh = "./clustering/lda_datos_coherencia_tableau.csv"
        df_coherencia.to_csv(ruta_coh, index=False)
        print(f"[V] Datos de coherencia guardados para Tableau en: {ruta_coh}")
        plt.show()

        # --- PROCESO FINAL DE EXTRACCIÓN ---

        # 1. Elegimos el número de tópicos óptimo que hemos visto en el gráfico
        k_final = int(input("\n[?] Introduce el número de tópicos óptimo que has visto en el gráfico: "))

        # Entrenamos el modelo definitivo con el k elegido
        lda_final = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary.id2token,
            num_topics=k_final,
            passes=passes,
            alpha=alpha,
            eta=eta,
            random_state=42
        )

        # 2. Asignar distribución completa (Soft) Y tópico dominante (Hard)
        data_final = data.copy()

        probabilidades_por_topico = {i: [] for i in range(k_final)}
        cluster_ids = []  # <-- Recuperamos la lista para no romper el paso 3

        for bow in corpus:
            distribucion = lda_final.get_document_topics(bow, minimum_probability=0)

            # A. Calculamos el tópico dominante (Hard Clustering)
            topico_dominante = max(distribucion, key=lambda x: x[1])[0]
            cluster_ids.append(topico_dominante)

            # B. Repartimos las probabilidades (Soft Clustering)
            for topico_id, prob in distribucion:
                probabilidades_por_topico[topico_id].append(prob)

        # Añadimos la columna cluster_id (¡Esto arregla tu error!)
        data_final['cluster_id'] = cluster_ids

        # Añadimos las nuevas columnas de probabilidades
        for i in range(k_final):
            nombre_columna = f"prob_topico_{i}"
            data_final[nombre_columna] = probabilidades_por_topico[i]

        # 3. Imprimir y guardar las palabras clave por tópico.
        # En LDA el peso de cada palabra en un tópico es directamente su probabilidad
        # dentro de la distribución tópico-palabra aprendida por el modelo.
        # Esto es diferente a K-Means: no es una media de vectores, sino la
        # probabilidad que el propio modelo asigna a esa palabra para ese tópico.
        print("\n" + "=" * 50)
        print("  PALABRAS CLAVE POR TÓPICO  (probabilidad LDA)")
        print("  Peso = P(palabra | tópico), aprendido por el modelo LDA")
        print("=" * 50)

        resultados_globales = []
        for i in range(k_final):
            # show_topic devuelve las n palabras más probables del tópico i
            # como lista de tuplas (palabra, probabilidad)
            top_palabras = lda_final.show_topic(i, topn=10)

            # Contamos cuántas instancias tienen este tópico como dominante
            n_instancias = (data_final["cluster_id"] == i).sum()

            print(f"\n{'─' * 50}")
            print(f"  TÓPICO #{i}  ({n_instancias} instancias)")
            print(f"{'─' * 50}")
            for rank, (palabra, prob) in enumerate(top_palabras, start=1):
                print(f"    {rank:>2}. {palabra:<25} {prob:.5f}")
                resultados_globales.append({
                    "cluster_id"  : i,
                    "n_instancias": n_instancias,
                    "rank"        : rank,
                    "palabra"     : palabra,
                    "prob_lda"    : round(prob, 5),
                })

        # Guardar CSV con las palabras clave por tópico
        df_global = pd.DataFrame(resultados_globales)
        ruta_palabras = "./clustering/lda_palabras_por_topico.csv"
        df_global.to_csv(ruta_palabras, index=False)
        print(f"\n[V] CSV de palabras clave por tópico guardado en: {ruta_palabras}")

        # 4. Guardar el resultado principal con los tópicos asignados en CSV
        ruta_salida = "./clustering/lda_resultados_agrupados.csv"
        data_final.to_csv(ruta_salida, index=False)
        print(f"\n[V] CSV con tópicos LDA guardado en: {ruta_salida}")