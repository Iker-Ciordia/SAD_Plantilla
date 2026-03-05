# -*- coding: utf-8 -*-

import pandas as pd

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


def apply_preprocessing(data_train, data_test, config_file): #TODO Revisar lo que hace la función en profundidad
    """
    Aplica el preprocesado evitando la fuga de datos (Data Leakage).
    Aprende reglas en train, las aplica ciegamente en test.
    """

    import json
    import pandas as pd
    import numpy as np

    #Leer el archivo JSON
    file = open(config_file, 'r')
    config = json.load(file)

    opciones = config.get("preprocessing", {}) #Nos quedamos con el segundo JSON de parámetros de preprocesado dentro del principal
    print(f"Aplicando preprocesado desde {config_file}...")

    #Eliminar atributos innecesarios (aquellos que no queramos usar para el entrenamiento)
    if "drop_features" in opciones and len(opciones["drop_features"]) > 0: #Si existe una llave "drop..." y NO está vacía
        columnas_a_borrar = []
        for col in opciones["drop_features"]: #Para toda columna que se quiera eliminar
            if col in data_train.columns: #Si la columna se encuentra en el DataFrame de entrenamiento
                columnas_a_borrar.append(col) #Añade la columna a la lista que habrá que borrar luego

        data_train = data_train.drop(columns=columnas_a_borrar) #Borra del DataFrame todas las columnas que se hayan indicado en el JSON y no interesan
        data_test = data_test.drop(columns=columnas_a_borrar)
        print(f" -> Columnas eliminadas: {columnas_a_borrar}")

    #Separamos temporalmente las columnas de atributos de la clase objetivo para no alterarla
    columnas_x = data_train.columns[:-1] #Desde la primera a la penúltima
    columna_y = data_train.columns[-1] #La última (previamente hemos ordenado para que la objetivo siempre esté al final

    #Tratar valores faltantes
    if opciones.get("missing_values") == "impute": #Si la clave "missing..." dice que hay que imputar valores
        estrategia = opciones.get("impute_strategy", "mean")  #Cogemos el valor que se indique en la estrategía.
                                                              #Si está vacío, se coge la media por defecto (de ahí el segundo param.)
        print(f" -> Imputando valores faltantes usando la estrategia: {estrategia}")

        # Seleccionamos solo las columnas numéricas para imputar (evita errores con texto)
        num_cols = data_train[columnas_x].select_dtypes(include=[np.number]).columns #Del DataFrame nos quedamos con las filas de las columnas que no son la columna a predecir.
                                                                               #Nos quedamos con las filas de aquellas columnas que sean numéricas (include=[np.number]).
                                                                               #.columns no da de ese DataFrame final sin columnas categóricas da los nombres de las columnas.
                                                                               #El objetivo es sacar los nombres de las columnas numéricas.

        if len(num_cols) > 0: #Si hay al menos una columna numérica
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy=estrategia) #Prepara la herramienta de imputación de valores
            # ¡ATENCIÓN! Train hace fit_transform, Test solo transform
            #La diferencia radica en que el valor que vamos a calcular imputar en Machine Learning real solo se debe calcular sobre el conjunto de Train.
            #Es decir, si en el train la moda es 2000, se imputará con 2000 tanto en el train, como el dev como el test.
            #De esta forma, aunque la moda en el test sea 1000, se pondrá un 2000. Esto se hace porque si no estaríamos permitiendo que
            #la distribución de datos del conjunto de test influya en el preprocesado, lo que se considera "hacer trampas"
            data_train[num_cols] = imputer.fit_transform(data_train[num_cols]) #Imputamos los valores faltantes con la estrategía extraída previamente.
            data_test[num_cols] = imputer.transform(data_test[num_cols])


    # Preprocesamiento de texto (TF-IDF, BoW/frecuency o Binario/one-hot)
    metodo_texto = opciones.get("text_preprocess")
    if metodo_texto in ["tf-idf", "frequency", "one-hot"]:
        # 1. Leemos la lista exacta de columnas que queremos convertir a TF-IDF/BOW desde el JSON
        columnas_json = opciones.get("categorical_features_convert", [])

        # 2. Comprobamos que esas columnas realmente existan en nuestro dataset (por seguridad)
        text_cols = [col for col in columnas_json if col in data_train.columns]

        if len(text_cols) > 0:
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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

            print(f" -> Aplicando {metodo_texto} a las columnas: {list(text_cols)}")

            for col in text_cols:
                # Transformamos el texto (asegurando string para evitar errores con NaNs)
                # ¡ATENCIÓN! Train aprende el diccionario, Test se adapta. Aplica el mismo criterio de no contaminación que el escalado
                matrix_train = vectorizer.fit_transform(data_train[col].astype(str))
                matrix_test = vectorizer.transform(data_test[col].astype(str))

                # --- Nombres de columnas con las palabras reales ---
                palabras = vectorizer.get_feature_names_out()
                nombres_cols = [f"{col}_{prefijo}_{w}" for w in palabras]

                # Convertimos a DataFrames manteniendo el índice original
                df_train = pd.DataFrame(matrix_train.toarray(), columns=nombres_cols, index=data_train.index)
                df_test = pd.DataFrame(matrix_test.toarray(), columns=nombres_cols, index=data_test.index)

                # Eliminamos la original y unimos las nuevas
                data_train = data_train.drop(columns=[col]).join(df_train)
                data_test = data_test.drop(columns=[col]).join(df_test)


    # Escalado de valores
    metodo_escalado = opciones.get("scaling") #Cogemos el valor de escalado del JSON
    if metodo_escalado in ["max-min", "max", "z-score", "standard"]:
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
        # ¡ATENCIÓN! Train da las medias/máximos, Test solo se ajusta a ellos
        data_train[columnas_a_escalar] = scaler.fit_transform(data_train[columnas_a_escalar])
        data_test[columnas_a_escalar] = scaler.transform(data_test[columnas_a_escalar])
        print(f" -> {len(columnas_a_escalar)} columnas escaladas correctamente.")

    # Volvemos a asegurar que la columna objetivo (y) esté al final tras las posibles modificaciones
    # Reordenar por seguridad (Objetivo siempre al final)
    def reordenar(df):
        cols = df.columns.tolist()
        if columna_y in cols:
            cols.remove(columna_y)
            cols.append(columna_y)
        return df[cols]

    return reordenar(data_train), reordenar(data_test)


def calculate_metrics(y_test, y_pred): #TODO Habría que diseñarlo de tal forma que en el JSON se pueda elegir qué métricas evaluar
    """
    Función para calcular el F-score
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    from sklearn.metrics import f1_score        #Importamos todas las librerias de métricas
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score

    print("\nPrecision:")
    precision_micro = precision_score(y_test, y_pred, average='micro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    print("Micro: " + str(precision_micro), "Macro: " + str(precision_macro))

    print("\nRecall:")
    recall_micro = recall_score(y_test, y_pred, average='micro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    print("Micro: " + str(recall_micro), "Macro: " + str(recall_macro))

    print("\nF-score:")
    fscore_micro = f1_score(y_test, y_pred, average='micro')
    fscore_macro = f1_score(y_test, y_pred, average='macro')
    print("Micro: " + str(fscore_micro), "Macro: " + str(fscore_macro))



def calculate_confusion_matrix(y_test, y_pred): # TODO las métricas no sé si también hay que permitir elegir cuál usar. Supongo que sí
    """
    Función para calcular la matriz de confusión
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Matriz de confusión
    """
    from sklearn.metrics import confusion_matrix #Importamos el modulo para hacer la matriz de confusión
    import pandas as pd
    import numpy as np

    cm = confusion_matrix(y_test, y_pred)
    #print(y_test)

    #Extraemos las etiquetas únicas y ordenadas de las clases reales de test
    # (las sacamos de test y no de la prediccion por si hay alguna clase que no ha predicho)
    etiquetas = np.unique(y_test)

    #Creamos los nombres para las filas (Realidad) y columnas (Predicción)
    nombres_filas = [f"Realidad: {e}" for e in etiquetas]
    nombres_columnas = [f"Predicción: {e}" for e in etiquetas]

    #Juntamos ambos en un DataFrame de pandas para que se imprima bonito
    matriz_bonita = pd.DataFrame(cm, index=nombres_filas, columns=nombres_columnas)
    return matriz_bonita

def kNN(data_train, data_test, k, weights, p):
    """
    Función para implementar el algoritmo kNN con datos ya preprocesados y divididos
    """

    # Seleccionamos las características y la clase del conjunto de datos de entrenamiento.
    # El .values se usa para convertirlo de DataFrame a matriz normal, que es lo que usa Skicit.
    X_train = data_train.iloc[:, :-1].values # Todas las columnas menos la última (atributos que se van a usar para entrenar)
    y_train = data_train.iloc[:, -1].values # Última columna (atributo a predecir). Sí o sí está en la última columna

    # Seleccionamos las características y la clase del conjunto de datos de testeo.
    X_test = data_test.iloc[:, :-1].values
    y_test = data_test.iloc[:, -1].values

    # Entrenamos el modelo
    from sklearn.neighbors import KNeighborsClassifier #Importamos el algoritmo KNN
    classifier = KNeighborsClassifier(n_neighbors = k, weights = weights, p = p) #Creamos el modelo con sus hiperparámetros concretos
    classifier.fit(X_train, y_train) #Entrenamos el modelo con los datasets de training
                                     # X_train son las instancias con atributos de entrenamiento
                                     # y_train es la clase real de dicha instancia
    
    # Predecimos los resultados
    y_pred = classifier.predict(X_test) #Probamos el modelo con el dataset de testeo (sin darle la clase real)
    
    return y_test, y_pred

def guardar_resultados_csv(k, p, weights, y_test, y_pred):
    """Guarda las métricas en una fila del archivo CSV."""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    import csv
    import os

    # Calculamos las métricas (usamos macro como ejemplo, cambiarlo a None si es binario)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

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

if __name__ == "__main__": #TODO Falta por probar que funcione bien el tema del preprocesado (no me ha dado tiempo a probarlo)
    import sys
    import json
    from sklearn.model_selection import train_test_split

    # Pedimos fichero, objetivo y obligatoriamente el JSON
    if len(sys.argv) < 4 or "-c" not in sys.argv:
        print("Uso: python script.py <fichero> <columna_objetivo> -c <config.json>")
        print("Opcional (para lanzador KNN): python script.py <fich> <obj> <k> <w> <p> -c <config.json>")
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

    # B. División del conjunto de train con el de test. Evitamos Data Leakage para CUALQUIER algoritmo
    data_train, data_test = train_test_split(data, test_size=0.20, random_state=42)

    # C. Aplicamos el preprocesado pasándole ambos trozos
    if config_file:
        data_train, data_test = apply_preprocessing(data_train, data_test, config_file)

    # --- ENRUTADOR DE ALGORITMOS ---
    y_test, y_pred = None, None

    if algoritmo == "KNN":
        print("\n[->] Ejecutando modelo: kNN")

        # Leemos los rangos del JSON (con valores por defecto por si acaso)
        hiper_knn = config.get("hyperparametersKNN", {})
        k_min = hiper_knn.get("k_min", 3)
        k_max = hiper_knn.get("k_max", 3)
        p_min = hiper_knn.get("p_min", 2)
        p_max = hiper_knn.get("p_max", 2)
        pesos_lista = hiper_knn.get("w", ["uniform"])

        # Por seguridad: si pesos_lista es un solo string, lo convertimos a lista
        if isinstance(pesos_lista, str):
            pesos_lista = [pesos_lista]

        # Borramos el CSV antiguo si existe para empezar limpios
        import os
        if os.path.exists('resultados.csv'):
            os.remove('resultados.csv')

        # Bucle interno de hiperparámetros (Súper rápido porque el preprocesado ya está hecho)
        for k in range(k_min, k_max + 1, 2):  # Avanza de 2 en 2 para k impares
            for p in range(p_min, p_max + 1): #Para los 2 tipos de distancias posibles
                for weights in pesos_lista:
                    print(f"\n--------------------------------------------------")
                    print(f"--> Evaluando combinación: k={k}, p={p}, w={weights}")

                    # Llamamos a la función
                    y_test, y_pred = kNN(data_train, data_test, k, weights, p)

                    # Mostramos y guardamos resultados de ESTA combinación
                    print(calculate_confusion_matrix(y_test, y_pred))
                    calculate_metrics(y_test, y_pred)
                    guardar_resultados_csv(k, p, weights, y_test, y_pred)

    elif algoritmo == "DecisionTree":
        print("\n[->] Ejecutando modelo: Árbol de Decisión")
        # TODO: Leer hyperparametersDecisionTree del JSON
        # TODO: y_test, y_pred = decisionTree(data_train, data_test, max_depth, ...)
        pass

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
