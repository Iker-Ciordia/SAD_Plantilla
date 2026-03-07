# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score


def load_data(file, columna_target):
    """
    Función para cargar los datos de un fichero csv y mover la clase al final
    :param file: Fichero csv
    :param target_column: Nombre de la columna que contiene las clases a predecir
    :return: Datos del fichero con la columna objetivo al final
    """
    data = pd.read_csv(file)
    #print(data)

    # Comprobamos que la columna realmente existe en el CSV
    if columna_target not in data.columns:
        print(f"Error: La columna '{columna_target}' no se ha encontrado en el archivo.")
        sys.exit(1)

    # Extraemos la lista de columnas, quitamos la objetivo y la ponemos al final para que el algoritmo nunca se confunda
    columnas = data.columns.tolist() #Convertir a lista tradicional de Python para poder usar sus comandos
    columnas.remove(columna_target)
    columnas.append(columna_target)

    # Reordenamos el DataFrame
    data = data[columnas]
    #print(data)

    return data


def apply_preprocessing(config_file, data_train, data_dev=None, herramientas_guardadas=None):
    """
    Aplica el preprocesado evitando la fuga de datos (Data Leakage).
    Aprende reglas en train, las aplica ciegamente en dev.
    Es necesario tener dividido en 2 funciones distintas el preprocesado porque los datos se nos darán como train/dev en un CSV y test en otro CSV.
    Esta función devuelve los 2 datasets de train y dev preprocesado y las herramientas usadas en fit_transform para almacenarlas con Pickle.
    De esta forma podemos ejecutar el preprocesado sobre un dataset de test en otra función.
    - Si herramientas_guardadas es None (MODO TRAIN): Aprende las reglas y devuelve los datasets preprocesados y las herramientas.
    - Si pasas las herramientas (MODO TEST): Aplica las reglas aprendidas ciegamente y devuelve el dataset de test preprocesado.
    """

    import json
    import pandas as pd
    import numpy as np

    #Leer el archivo JSON
    file = open(config_file, 'r')
    config = json.load(file)

    # --- EL INTERRUPTOR INTELIGENTE ---
    is_train = herramientas_guardadas is None

    opciones = config.get("preprocessing", {}) #Nos quedamos con el segundo JSON de parámetros de preprocesado dentro del principal
    print(f"Aplicando preprocesado desde {config_file}...")

    if is_train:
        print(f"Aplicando preprocesado (MODO TRAIN) desde {config_file}...")
        herramientas = {}  # Preparamos la mochila vacía
    else:
        print(f"Aplicando preprocesado (MODO TEST) usando herramientas guardadas...")
        herramientas = herramientas_guardadas  # Cargamos la mochila

    #Eliminar atributos innecesarios (aquellos que no queramos usar para el entrenamiento)
    if "drop_features" in opciones and len(opciones["drop_features"]) > 0: #Si existe una llave "drop..." y NO está vacía
        columnas_a_borrar = []
        for col in opciones["drop_features"]: #Para toda columna que se quiera eliminar
            if col in data_train.columns: #Si la columna se encuentra en el DataFrame de entrenamiento
                columnas_a_borrar.append(col) #Añade la columna a la lista que habrá que borrar luego

        data_train = data_train.drop(columns=columnas_a_borrar) #Borra del DataFrame todas las columnas que se hayan indicado en el JSON y no interesan
        if data_dev is not None: #La comprobacion se hace por si estamos preprocesando para el test.
            data_dev = data_dev.drop(columns=columnas_a_borrar)

        if is_train:
            print(f" -> Columnas eliminadas: {columnas_a_borrar}")

    #Separamos temporalmente las columnas de atributos de la clase objetivo para no alterarla
    columnas_x = data_train.columns[:-1] #Desde la primera a la penúltima
    columna_y = data_train.columns[-1] #La última (previamente hemos ordenado para que la objetivo siempre esté al final

    #Tratar valores faltantes
    if opciones.get("missing_values") == "impute": #Si la clave "missing..." dice que hay que imputar valores
        estrategia = opciones.get("impute_strategy", "mean")  #Cogemos el valor que se indique en la estrategía.
                                                              #Si está vacío, se coge la media por defecto (de ahí el segundo param.)
        if is_train:
            print(f" -> Imputando valores faltantes usando la estrategia: {estrategia}")

        # Seleccionamos solo las columnas numéricas para imputar (evita errores con texto)
        num_cols = data_train[columnas_x].select_dtypes(include=[np.number]).columns #Del DataFrame nos quedamos con las filas de las columnas que no son la columna a predecir.
                                                                               #Nos quedamos con las filas de aquellas columnas que sean numéricas (include=[np.number]).
                                                                               #.columns no da de ese DataFrame final sin columnas categóricas da los nombres de las columnas.
                                                                               #El objetivo es sacar los nombres de las columnas numéricas.

        if len(num_cols) > 0: #Si hay al menos una columna numérica
            from sklearn.impute import SimpleImputer
            if is_train: #Si estamos entrenando
                imputer = SimpleImputer(strategy=estrategia) #Prepara la herramienta de imputación de valores
                # ¡ATENCIÓN! Train hace fit_transform, Dev solo transform
                #La diferencia radica en que el valor que vamos a calcular imputar en Machine Learning real solo se debe calcular sobre el conjunto de Train.
                #Es decir, si en el train la moda es 2000, se imputará con 2000 tanto en el train, como el dev como el dev.
                #De esta forma, aunque la moda en el dev sea 1000, se pondrá un 2000. Esto se hace porque si no estaríamos permitiendo que
                #la distribución de datos del conjunto de dev influya en el preprocesado, lo que se considera "hacer trampas"
                data_train[num_cols] = imputer.fit_transform(data_train[num_cols]) #Imputamos los valores faltantes con la estrategía extraída previamente.
                if data_dev is not None:
                    data_dev[num_cols] = imputer.transform(data_dev[num_cols])

                # Guardamos la herramienta en la mochila
                herramientas['imputer'] = imputer
                herramientas['imputer_cols'] = num_cols
            else:
                # MODO TEST: Rescatamos la herramienta y SOLO transformamos
                imputer = herramientas['imputer']
                cols = [c for c in herramientas['imputer_cols'] if c in data_train.columns]
                data_train[cols] = imputer.transform(data_train[cols]) #Aunque la variable se llama "data_train" realmente sería la de test.


    # Preprocesamiento de texto (TF-IDF, BoW/frecuency o Binario/one-hot)
    metodo_texto = opciones.get("text_preprocess")
    if metodo_texto in ["tf-idf", "frequency", "one-hot"]:
        # 1. Leemos la lista exacta de columnas que queremos convertir a TF-IDF/BOW desde el JSON
        columnas_json = opciones.get("categorical_features_convert", [])

        # 2. Comprobamos que esas columnas realmente existan en nuestro dataset (por seguridad)
        text_cols = [col for col in columnas_json if col in data_train.columns]

        if len(text_cols) > 0:
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

            if is_train:
                herramientas['vectorizers'] = {} #Preparamos la mochila para aceptar una nueva herramienta
                # --- Selección de estrategia ---
                if metodo_texto == "tf-idf":
                    vectorizer = TfidfVectorizer()
                    prefijo = "tfidf"
                elif metodo_texto == "frequency":
                    vectorizer = CountVectorizer()  # Cuenta frecuencias: 1, 2, 3...
                    prefijo = "frequency"
                elif metodo_texto == "one-hot":
                    vectorizer = CountVectorizer(binary=True)  # One-Hot: 0 o 1
                    prefijo = "onehot"

                herramientas['prefijo_texto'] = prefijo
                print(f" -> Aplicando {metodo_texto} a las columnas: {list(text_cols)}")
            else: #Rama para preproceso de testing
                prefijo = herramientas['prefijo_texto'] #Cargamos con qué tipo se preprocesó el train

            for col in text_cols:
                if is_train:
                    # Transformamos el texto (asegurando string para evitar errores con NaNs)
                    # ¡ATENCIÓN! Train aprende el diccionario, Dev se adapta. Aplica el mismo criterio de no contaminación que el escalado
                    matrix_train = vectorizer.fit_transform(data_train[col].astype(str))
                    if data_dev is not None:
                        matrix_dev = vectorizer.transform(data_dev[col].astype(str))
                    herramientas['vectorizers'][col] = vectorizer # Guardamos el diccionario de palabras
                else: #Rama de preproceso de testeo
                    vectorizer = herramientas['vectorizers'][col]
                    # Test se adapta usando solo transform
                    matrix_train = vectorizer.transform(data_train[col].astype(str)) #Aunque se llama "train" sería la de "test". Reaprovecha la misma variable

                # --- Nombres de columnas con las palabras reales ---
                palabras = vectorizer.get_feature_names_out()
                nombres_cols = [f"{col}_{prefijo}_{w}" for w in palabras]

                # Convertimos a DataFrames manteniendo el índice original
                df_train = pd.DataFrame(matrix_train.toarray(), columns=nombres_cols, index=data_train.index)
                data_train = data_train.drop(columns=[col]).join(df_train) # Eliminamos la original y unimos las nuevas


                if data_dev is not None:
                    df_dev = pd.DataFrame(matrix_dev.toarray(), columns=nombres_cols, index=data_dev.index)
                    data_dev = data_dev.drop(columns=[col]).join(df_dev) # Eliminamos la original y unimos las nuevas


    # Escalado de valores
    metodo_escalado = opciones.get("scaling") #Cogemos el valor de escalado del JSON
    if metodo_escalado in ["max-min", "max", "z-score", "standard"]:
        if is_train:
            from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

            # 1. Seleccionamos el escalador según el JSON
            if metodo_escalado == "max-min":
                scaler = MinMaxScaler()
            elif metodo_escalado == "max":
                scaler = MaxAbsScaler()
            else:  # z-score o standard es lo mismo
                scaler = StandardScaler()

            print(f" -> Aplicando escalado tipo: {metodo_escalado}")

            # 2. Identificamos qué columnas escalar (todas menos la columna objetivo)
            columnas_a_escalar = data_train.columns.drop(columna_y).tolist()

            # 3. Aplicamos la transformación
            # ¡ATENCIÓN! Train da las medias/máximos, Dev solo se ajusta a ellos
            data_train[columnas_a_escalar] = scaler.fit_transform(data_train[columnas_a_escalar])
            if data_dev is not None:
                data_dev[columnas_a_escalar] = scaler.transform(data_dev[columnas_a_escalar])
            print(f" -> {len(columnas_a_escalar)} columnas escaladas correctamente.")

            # Guardamos el escalador
            herramientas['scaler'] = scaler
            herramientas['scaler_cols'] = columnas_a_escalar
        else:
            scaler = herramientas['scaler']
            cols = [c for c in herramientas['scaler_cols'] if c in data_train.columns]
            data_train[cols] = scaler.transform(data_train[cols]) #Aunque se llame "train" realmente sería el dataset de test.

    # Volvemos a asegurar que la columna objetivo (y) esté al final tras las posibles modificaciones
    # Reordenar por seguridad (Objetivo siempre al final)
    def reordenar(df):
        cols = df.columns.tolist()
        if columna_y in cols:
            cols.remove(columna_y)
            cols.append(columna_y)
        return df[cols]

    # --- RETORNO DINÁMICO ---
    if is_train:
        return reordenar(data_train), reordenar(data_dev), herramientas
    else:
        return reordenar(data_train)  # En modo test, solo devolvemos el test limpio


def calculate_metrics(y_dev, y_pred):
    """
    Función para calcular el F-score
    :param y_dev: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    from sklearn.metrics import f1_score        #Importamos todas las librerias de métricas
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score

    # Leer el archivo JSON
    file = open(config_file, 'r')
    config = json.load(file)
    tipo_metrica = config["metric_to_evaluate"]

    print("\nPrecision:")
    precision = precision_score(y_dev, y_pred, average=tipo_metrica)
    print(str(tipo_metrica) + ": " + str(precision))

    print("\nRecall:")
    recall = recall_score(y_dev, y_pred, average=tipo_metrica)
    print(str(tipo_metrica) + ": " + str(recall))

    print("\nF-score:")
    fscore = f1_score(y_dev, y_pred, average=tipo_metrica)
    print(str(tipo_metrica) + ": " + str(fscore))

    return fscore #Se devuelve el F1 para poder decidir qué modelo es el mejor.



def calculate_confusion_matrix(y_dev, y_pred): # TODO las métricas no sé si también hay que permitir elegir cuál usar. Supongo que sí
    """
    Función para calcular la matriz de confusión
    :param y_dev: Valores reales
    :param y_pred: Valores predichos
    :return: Matriz de confusión
    """
    from sklearn.metrics import confusion_matrix #Importamos el modulo para hacer la matriz de confusión
    import pandas as pd
    import numpy as np

    cm = confusion_matrix(y_dev, y_pred)
    #print(y_dev)

    #Extraemos las etiquetas únicas y ordenadas de las clases reales de dev
    # (las sacamos de dev y no de la prediccion por si hay alguna clase que no ha predicho)
    etiquetas = np.unique(y_dev)

    #Creamos los nombres para las filas (Realidad) y columnas (Predicción)
    nombres_filas = [f"Realidad: {e}" for e in etiquetas]
    nombres_columnas = [f"Predicción: {e}" for e in etiquetas]

    #Juntamos ambos en un DataFrame de pandas para que se imprima bonito
    matriz_bonita = pd.DataFrame(cm, index=nombres_filas, columns=nombres_columnas)
    return matriz_bonita

def kNN(data_train, data_dev, k, weights, p):
    """
    Función para implementar el algoritmo kNN con datos ya preprocesados y divididos
    """

    # Seleccionamos las características y la clase del conjunto de datos de entrenamiento.
    # El .values se usa para convertirlo de DataFrame a matriz normal, que es lo que usa Skicit.
    X_train = data_train.iloc[:, :-1].values # Todas las columnas menos la última (atributos que se van a usar para entrenar)
    y_train = data_train.iloc[:, -1].values # Última columna (atributo a predecir). Sí o sí está en la última columna

    # Seleccionamos las características y la clase del conjunto de datos de dev.
    X_dev = data_dev.iloc[:, :-1].values
    y_dev = data_dev.iloc[:, -1].values

    # Entrenamos el modelo
    from sklearn.neighbors import KNeighborsClassifier #Importamos el algoritmo KNN
    classifier = KNeighborsClassifier(n_neighbors = k, weights = weights, p = p) #Creamos el modelo con sus hiperparámetros concretos
    classifier.fit(X_train, y_train) #Entrenamos el modelo con los datasets de training
                                     # X_train son las instancias con atributos de entrenamiento
                                     # y_train es la clase real de dicha instancia
    
    # Predecimos los resultados
    y_pred = classifier.predict(X_dev) #Probamos el modelo con el dataset de dev (sin darle la clase real)
    
    return y_dev, y_pred, classifier #Devolvemos el classifier (el modelo) para poder quedarnos con aquel que sea el mejor

def decisionTree(data_train, data_dev, max_depth, min_samples_split, min_samples_leaf, criterion): #TODO Realizar el algoritmo en si de DecisionTree
    """
    Función para implementar el algoritmo DecisionTree con datos ya preprocesados y divididos
    """

def calcular_impureza(y, criterion): #TODO hay que calcular la impureza con entropia o Gini depende de lo que se le pase
    if len(y) == 0: return 0
    probs = y.value_counts(normalize=True)

    if criterion.lower() == "gini":
        return 1 - np.sum(probs ** 2)
    else:
        return -np.sum(probs * np.log2(probs + 1e-9))

def guardar_resultados_csv(k, p, weights, y_dev, y_pred):
    """Guarda las métricas en una fila del archivo CSV."""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    import csv
    import os

    # Leer el archivo JSON
    file = open(config_file, 'r')
    config = json.load(file)
    tipo_metrica = config["metric_to_evaluate"]

    # Calculamos las métricas
    acc = accuracy_score(y_dev, y_pred)
    prec = precision_score(y_dev, y_pred, average=tipo_metrica, zero_division=0)
    rec = recall_score(y_dev, y_pred, average=tipo_metrica, zero_division=0)
    f1 = f1_score(y_dev, y_pred, average=tipo_metrica, zero_division=0)

    combinacion = f"k={k}, p={p}, {weights}"
    archivo_csv = 'resultados.csv'

    # Si el archivo no existe, creamos la cabecera primero
    cabecera = not os.path.exists(archivo_csv)

    with open(archivo_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if cabecera:
            writer.writerow(['Combinacion', 'Accuracy', 'Precision', 'Recall', 'F_score'])
        # Guardamos los valores redondeados a 4 decimales
        writer.writerow([combinacion, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

if __name__ == "__main__":
    import sys
    import json
    from sklearn.model_selection import train_test_split

    # Pedimos fichero, objetivo y obligatoriamente el JSON
    if len(sys.argv) < 4 or "-c" not in sys.argv:
        print("Uso: python script.py <fichero> <columna_objetivo> -c <config.json>")
        sys.exit(1)

    # Asignamos las variables desde la consola para que sea más fácil de leer
    fichero = sys.argv[1]
    columna_objetivo = sys.argv[2]

    # 1. Buscamos el archivo JSON en los argumentos
    config_file = None
    indice_c = sys.argv.index("-c")
    if indice_c + 1 < len(sys.argv):
        config_file = sys.argv[indice_c + 1]

    # 2. Leemos qué algoritmo quiere el usuario desde el JSON
    algoritmo = "KNN"  # Valor por defecto
    if config_file:
        with open(config_file, 'r') as file:
            config = json.load(file)
        algoritmo = config.get("algorithm", "KNN")

    # --- INICIO DEL FLUJO DE MACHINE LEARNING COMÚN ---

    # A. Cargamos los datos
    data = load_data(fichero, columna_objetivo)

    # B. División del conjunto de train con el de dev. Evitamos Data Leakage para CUALQUIER algoritmo
    data_train, data_dev = train_test_split(data, test_size=0.20, random_state=42)

    # C. Aplicamos el preprocesado pasándole ambos trozos
    if config_file:
        data_train, data_dev, mis_herramientas = apply_preprocessing(config_file, data_train, data_dev) #Preprocesamos train y dev y obtenemos las herramientas usadas pa cuando toque con test

    # --- ENRUTADOR DE ALGORITMOS ---
    y_dev, y_pred = None, None


    ##########################
    #  Empieza algoritmo KNN #
    ##########################
    if algoritmo == "KNN":
        print("\n[->] Ejecutando modelo: kNN")

        # Leemos los rangos del JSON (con valores por defecto por si acaso)
        hiper_knn = config.get("hyperparametersKNN", {})
        k_min = hiper_knn.get("k_min", 3)
        k_max = hiper_knn.get("k_max", 3)
        p_min = hiper_knn.get("p_min", 2)
        p_max = hiper_knn.get("p_max", 2)
        pesos_lista = hiper_knn.get("w", ["uniform"])
        step = hiper_knn.get("step", 2)

        # Por seguridad: si pesos_lista es un solo string, lo convertimos a lista
        if isinstance(pesos_lista, str):
            pesos_lista = [pesos_lista]

        # Borramos el CSV antiguo si existe para empezar limpios
        import os
        if os.path.exists('resultados.csv'):
            os.remove('resultados.csv')

        mejor_f1 = -1.0
        mejor_modelo = None
        mejores_hiperparametros = ""
        # Bucle interno de hiperparámetros (Súper rápido porque el preprocesado ya está hecho)
        for k in range(k_min, k_max + 1, step):  # Avanza con lo que el usuario del script crea conveniente
            for p in range(p_min, p_max + 1): #Para los 2 tipos de distancias posibles
                for weights in pesos_lista:
                    print(f"\n--------------------------------------------------")
                    print(f"--> Evaluando combinación: k={k}, p={p}, w={weights}")

                    # Llamamos a la función
                    y_dev, y_pred, modelo_entrenado = kNN(data_train, data_dev, k, weights, p)

                    # Mostramos y guardamos resultados de ESTA combinación
                    print(calculate_confusion_matrix(y_dev, y_pred))
                    # Calculas las métricas. Hacemos que devuelva el F1 para poder usarlo como decisor del mejor modelo.
                    f1_actual = calculate_metrics(y_dev, y_pred)

                    if f1_actual > mejor_f1:
                        mejor_f1 = f1_actual
                        mejor_modelo = modelo_entrenado
                        mejores_hiperparametros = f"k={k}, p={p}, w={weights}"
                        print(f"    [!] ¡Nuevo mejor modelo encontrado! F1: {mejor_f1:.4f}")

                    guardar_resultados_csv(k, p, weights, y_dev, y_pred)
        print(f"\n==================================================")
        print(f"EL GANADOR ES: {mejores_hiperparametros} con F1={mejor_f1:.4f}")

        import pickle

        # Creamos un diccionario (el paquete final) con el modelo a usar para test y las herramientas de preprocesado utilizadas con train y dev
        paquete_final = {
            'modelo': mejor_modelo,
            'herramientas_preproceso': mis_herramientas
        }
        nombre_archivo = 'mejor_modelo_knn.pkl'

        archivo = open(nombre_archivo, 'wb')
        pickle.dump(paquete_final, archivo)
        archivo.close()

    ###################################
    #  Empieza algoritmo DecisionTree #
    ###################################
    elif algoritmo == "DecisionTree": #TODO Mirar que este bien hecho
        print("\n[->] Ejecutando modelo: Árbol de Decisión")
        # Leemos los rangos del JSON (con valores por defecto por si acaso)
        hiper_DecisionTree = config.get("hyperparametersDecisionTree", {})
        min_depth = hiper_DecisionTree.get("min_depth", 1)
        max_depth = hiper_DecisionTree.get("max_depth", 10)
        min_samples_split = hiper_DecisionTree.get("min_samples_split", 5)
        min_samples_leaf = hiper_DecisionTree.get("min_samples_leaf", 5)
        criterion_lista = hiper_DecisionTree.get("criterion", "Gini")

        # Por seguridad: si criterion_lista es un solo string, lo convertimos a lista
        if isinstance(criterion_lista, str):
            pesos_lista = [criterion_lista]

        # Borramos el CSV antiguo si existe para empezar limpios
        import os
        if os.path.exists('resultados.csv'):
            os.remove('resultados.csv')

        mejor_f1 = -1.0
        mejor_modelo = None
        mejores_hiperparametros = ""
        # Bucle interno de hiperparámetros (Súper rápido porque el preprocesado ya está hecho)
        # 2. Bucle interno de hiperparámetros (Grid Search)
        for depth in range(min_depth, max_depth + 1):
            for crit in criterion_lista:
                print(f"\n--------------------------------------------------")
                print(
                    f"--> Evaluando combinación: max_depth={depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, criterion={crit}")

                # Llamamos a la función con los parámetros de esta iteración
                y_dev, y_pred, modelo_entrenado = decisionTree(data_train,data_dev, max_depth=depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=crit)

                # Mostramos y guardamos resultados de ESTA combinación
                print(calculate_confusion_matrix(y_dev, y_pred))

                # Calculas las métricas. Devuelve el F1 para poder usarlo como decisor.
                f1_actual = calculate_metrics(y_dev, y_pred)

                if f1_actual > mejor_f1:
                    mejor_f1 = f1_actual
                    mejor_modelo = modelo_entrenado
                    mejores_hiperparametros = f"depth={depth}, split={min_samples_split}, leaf={min_samples_leaf}, crit={crit}"
                    print(f"    [!] ¡Nuevo mejor modelo encontrado! F1: {mejor_f1:.4f}")

                # CUIDADO AQUÍ: Hay que adaptar que la función guardar_resultados_csv acepte estos nuevos parámetros
                guardar_resultados_csv(depth, min_samples_split, min_samples_leaf, crit, y_dev, y_pred)
        print(f"\n==================================================")
        print(f"EL GANADOR ES: {mejores_hiperparametros} con F1={mejor_f1:.4f}")

    elif algoritmo == "RandomForest":
        print("\n[->] Ejecutando modelo: Random Forest")
        # TODO: Leer hyperparametersRandomForest del JSON
        pass

    elif algoritmo == "NaiveBayes":
        print("\n[->] Ejecutando modelo: Naive Bayes")
        # TODO: Leer hyperparametersNaiveBayes del JSON
        pass
    else:
        print(f"Error: Algoritmo '{algoritmo}' no reconocido en el JSON.")
        sys.exit(1)
