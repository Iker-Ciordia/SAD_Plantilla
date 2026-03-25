# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)


def load_data(file, columna_target, config):
    """
    Función para cargar los datos de un fichero csv y mover la clase al final
    :param file: Fichero csv
    :param columna_target: Nombre de la columna que contiene las clases a predecir
    :param config: Fichero JSON usado para obtener el separador de columnas en el CSV.
    :return: Datos del fichero con la columna objetivo al final
    """

    separador = config.get("preprocessing").get("separator", ",")
    #data = pd.read_csv(file, sep=None, engine='python') #Interpreta él solo cuál el separador de columnas en el CSV
    data = pd.read_csv(file, sep=separador) #Si el de arriba no funciona introducimos manualmente el separador
    print(data)

    # Comprobamos que la columna realmente existe en el CSV
    if columna_target not in data.columns:
        import sys
        print(f"Error: La columna '{columna_target}' no se ha encontrado en el archivo. Comprueba el separador.")
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
    is_train = herramientas_guardadas is None #Si la variable herramientas... tiene un valor, is_train será True, si es None, será False

    opciones = config.get("preprocessing", {}) #Nos quedamos con el segundo JSON de parámetros de preprocesado dentro del principal
    print(f"Aplicando preprocesado desde {config_file}...")

    if is_train:
        print(f"Aplicando preprocesado (MODO TRAIN) desde {config_file}...")
        herramientas = {}  # Preparamos la mochila vacía
    else:
        print(f"Aplicando preprocesado (MODO TEST) usando herramientas guardadas...")
        herramientas = herramientas_guardadas  # Cargamos la mochila


    ##########################################
    #  Eliminación de atributos innecesarios #
    ##########################################
    # (aquellos que no queramos usar para el entrenamiento)
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

    ####################################
    #  Imputación de valores faltantes #
    ####################################
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

    #######################################################
    #        Discretización de variables continuas        # TODO Probar que realmente funciona bien
    #######################################################
    columnas_a_discretizar = opciones.get("continuous_features_discretize", [])
    cant_rangos = opciones.get("discretize_bins", 5)

    # Comprobamos que existan en el DataFrame
    cols_discretize = [col for col in columnas_a_discretizar if col in data_train.columns]

    #Si existe alguna columna a discretizar
    if len(cols_discretize) > 0:
        from sklearn.preprocessing import KBinsDiscretizer

        if is_train: #Si estamos entrenando el modelo y no testeandolo
            print(f" -> Discretizando variables continuas: {cols_discretize}")
            # encode='onehot-dense' devuelve 0s y 1s normales. n_bins divide en tantos grupos como los metidos en el JSON.
            discretizador = KBinsDiscretizer(n_bins=cant_rangos, encode='ordinal', strategy='uniform') #Le decimos que para cada intervalo le asocie un número y ya
                                                                                                       #no es necesario que cada intervalo sea una columna porque el
                                                                                                       #CategoricalNB de Naive Bayes entiende que no hay relación entre un 0-1 o 0-2

            # Ajustamos y transformamos en TRAIN
            matriz_train_disc = discretizador.fit_transform(data_train[cols_discretize])
            if data_dev is not None:
                matriz_dev_disc = discretizador.transform(data_dev[cols_discretize])

            herramientas['discretizer'] = discretizador
        else:
            # En TEST, solo transformamos usando la herramienta guardada
            discretizador = herramientas['discretizer']
            matriz_train_disc = discretizador.transform(data_train[cols_discretize])

        # --- Generar los nuevos nombres de las columnas ---
        # Como es ordinal (categorias dentro de la propia columna, no una columna por categoria), devuelve exactamente 1 columna por cada original
        nombres_nuevos = []
        for col in cols_discretize:
            nombres_nuevos.append(f"{col}_disc")

        # --- Reemplazar las columnas originales por las discretizadas ---
        df_train_disc = pd.DataFrame(matriz_train_disc, columns=nombres_nuevos, index=data_train.index)
        data_train = data_train.drop(columns=cols_discretize).join(df_train_disc)

        if data_dev is not None:
            df_dev_disc = pd.DataFrame(matriz_dev_disc, columns=nombres_nuevos, index=data_dev.index)
            data_dev = data_dev.drop(columns=cols_discretize).join(df_dev_disc)




    ########################################################################
    #  Preprocesamiento de texto (TF-IDF, BoW/frecuency o Binario/one-hot) # TODO Preguntar si hay que usar One-Hot para el cat2num en KNN y Label-Encoding en Decision Tree o no hace falta diferenciarlos
    ########################################################################
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
                    vectorizer_class = TfidfVectorizer
                    prefijo = "tfidf"
                elif metodo_texto == "frequency":
                    vectorizer_class = CountVectorizer  # Cuenta frecuencias: 1, 2, 3...
                    prefijo = "frequency"
                elif metodo_texto == "one-hot":
                    vectorizer_class = lambda: CountVectorizer(binary=True)  # One-Hot: 0 o 1
                    prefijo = "onehot"

                herramientas['prefijo_texto'] = prefijo
                print(f" -> Aplicando {metodo_texto} a las columnas: {list(text_cols)}")
            else: #Rama para preproceso de testing
                prefijo = herramientas['prefijo_texto'] #Cargamos con qué tipo se preprocesó el train

            for col in text_cols:
                # --- IMPUTACIÓN PARA COLUMNAS DE TEXTO ---
                # Rellenamos los huecos (NaN) de las columnas con texto con la palabra "desconocido" antes de vectorizar
                data_train[col] = data_train[col].apply(lambda x: limpiar_texto(x, config['dataset_language']))
                if data_dev is not None:
                    data_dev[col] = data_dev[col].apply(lambda x: limpiar_texto(x, config['dataset_language']))
                # ----------------------------------------


                if is_train:
                    vectorizer = vectorizer_class()
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

    #######################
    # Escalado de valores # TODO Según Aitziber es buena idea permitir no escalar todas las columnas y que quede a elección del usuario.
    #######################
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
            columnas_indicadas = opciones.get("features_scale", []) # Leemos la lista de columnas a escalar del JSON

            if len(columnas_indicadas) > 0:
                # Comprobamos que existan en el DataFrame (por seguridad)
                columnas_a_escalar = [col for col in columnas_indicadas if col in data_train.columns]
                print(f"    [i] Escalando SOLO las columnas indicadas en JSON: {columnas_a_escalar}")
            else:
                # Comportamiento: Si el JSON no dice nada, escalamos todas menos la 'y'
                columnas_a_escalar = data_train.columns.drop(columna_y).tolist()
                print("    [i] No se especificaron 'features_scale'. Escalando todas las columnas menos la que hay que predecir.")

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

    #######################
    #  Balanceo de datos  #
    #######################
    metodo_balanceo = opciones.get("sampling")
    ratio_balanceo = opciones.get("balance", "auto") #Si no hay nada por defecto es auto, que es 50-50

    if is_train and metodo_balanceo in ["undersampling", "oversampling"]:
        print(f" -> Aplicando balanceo de clases tipo: {metodo_balanceo}")

        # SMOTE necesita las predicciones (X) y la clase objetivo (Y) separadas
        X_train_temp = data_train.drop(columns=[columna_y])
        y_train_temp = data_train[columna_y]

        # Elegimos la herramienta
        if metodo_balanceo == "oversampling":
            from imblearn.over_sampling import SMOTE
            balanceador = SMOTE(sampling_strategy=ratio_balanceo)
        elif metodo_balanceo == "undersampling":
            from imblearn.under_sampling import RandomUnderSampler
            balanceador = RandomUnderSampler(sampling_strategy=ratio_balanceo)
        else:
            print("[!] Método de balanceo no reconocido. Saltando paso.")

        if balanceador is not None:
            # Generamos los nuevos datos balanceados
            X_bal, y_bal = balanceador.fit_resample(X_train_temp, y_train_temp)

            # Volvemos a fusionarlos en el DataFrame data_train original
            data_train = pd.DataFrame(X_bal, columns=X_train_temp.columns)
            data_train[columna_y] = y_bal

            print(f"[+] Distribución final de clases en TRAIN:\n{data_train[columna_y].value_counts().to_string()}")

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


def limpiar_texto(texto, idioma='english'):
    """Limpia, tokeniza, quita stopwords y lematiza un texto."""
    import pandas as pd
    # 1. Tratar nulos
    if pd.isna(texto) or str(texto).strip() == "": #Si la casilla está vacía o si solo tiene espacios en blanco
        return "desconocido"

    # 2. Minúsculas
    texto = str(texto).lower()

    # 3. Tokenizar
    tokens = word_tokenize(texto)

    # 4. Eliminar Stopwords y signos de puntuación (.isalnum() filtra comas, puntos...)
    stop_words = set(stopwords.words(idioma))
    tokens_limpios = []
    for w in tokens:
        # Si la palabra NO es una stopword Y solo contiene letras/números (no es un signo de puntuación)
        if w not in stop_words and w.isalnum():
            tokens_limpios.append(w)  # La guardamos

    tokens_lematizados = []
    # 5. Lematizar (Convertir verbos/plurales a su raíz)
    if idioma == 'english':
        lemmatizer = WordNetLemmatizer()
        for w in tokens_limpios:
            raiz = lemmatizer.lemmatize(w)  # Calculamos su raíz
            tokens_lematizados.append(raiz)  # La guardamos

    else: #Para el resto de idiomas que no sean inglés no tenemos diccionario, usamos un truqui truqui que consiste en recortar las palabras
        from nltk.stem import SnowballStemmer
        stemmer = SnowballStemmer(idioma)
        for w in tokens_limpios:
            raiz = stemmer.stem(w) #Corta los sufijos ("corriendo" -> "corr")
            tokens_lematizados.append(raiz)


    # 6. Devolver texto limpio y unido por espacios (o 'desconocido' si se quedó vacío)
    resultado = " ".join(tokens_lematizados)
    return resultado if resultado != "" else "desconocido"


def calculate_metrics(y_dev, y_pred, config_file):
    """
    Función para calcular el F-score
    :param y_dev: Valores reales
    :param y_pred: Valores predichos
    :param config_file: String con la ruta al fichero JSON de configuración
    :return: F-score (micro), F-score (macro)
    """
    from sklearn.metrics import f1_score        #Importamos todas las librerias de métricas
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    import json

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



def calculate_confusion_matrix(y_dev, y_pred): #
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

def decisionTree(data_train, data_dev, max_depth, min_samples_split, min_samples_leaf, criterion):
    """
    Función para implementar el algoritmo DecisionTree con datos ya preprocesados y divididos
    """
    # Seleccionamos las características y la clase del conjunto de datos de entrenamiento.
    # El .values se usa para convertirlo de DataFrame a matriz normal, que es lo que usa Scikit.
    X_train = data_train.iloc[:, :-1].values  # Todas las columnas menos la última (atributos que se van a usar para entrenar)
    y_train = data_train.iloc[:, -1].values  # Última columna (atributo a predecir). Sí o sí está en la última columna

    # Seleccionamos las características y la clase del conjunto de datos de dev.
    X_dev = data_dev.iloc[:, :-1].values
    y_dev = data_dev.iloc[:, -1].values

    # 1. Instanciamos el modelo pasándole los hiperparámetros
    modelo_arbol = DecisionTreeClassifier(
        criterion=criterion.lower(),  # 'gini' o 'entropy'
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    modelo_arbol.fit(X_train, y_train)  # Entrenamos el modelo con los datasets de training
                                        # X_train son las instancias con atributos de entrenamiento
                                        # y_train es la clase real de dicha instancia

    # Predecimos los resultados
    y_pred = modelo_arbol.predict(X_dev)  # Probamos el modelo con el dataset de dev (sin darle la clase real)

    return y_dev, y_pred, modelo_arbol  # Devolvemos el classifier (el modelo) para poder quedarnos con aquel que sea el mejor

def randomForest(data_train, data_dev, n_estimators, max_depth, min_samples_split, min_samples_leaf, criterion):
    """
        Función para implementar el algoritmo Random Forest con datos ya preprocesados
    """
    # Separar atributos y clase
    X_train = data_train.iloc[:, :-1].values
    y_train = data_train.iloc[:, -1].values
    X_dev = data_dev.iloc[:, :-1].values
    y_dev = data_dev.iloc[:, -1].values

    # Instanciar el modelo con sus hiperparámetros
    from sklearn.ensemble import RandomForestClassifier
    modelo_forest = RandomForestClassifier(
        n_estimators=n_estimators,  # <-- ¡El nuevo parámetro estrella!
        criterion=criterion.lower(),
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    modelo_forest.fit(X_train, y_train) # Entrenamos el modelo con los datasets de training
                                        # X_train son las instancias con atributos de entrenamiento
                                        # y_train es la clase real de dicha instancia

    # Predecimos los resultados
    y_pred = modelo_forest.predict(X_dev)

    return y_dev, y_pred, modelo_forest

def naiveBayes(data_train, data_dev, alpha=None, tipo="multinomial"):
    # Separar atributos y clase
    X_train = data_train.iloc[:, :-1].values
    y_train = data_train.iloc[:, -1].values
    X_dev = data_dev.iloc[:, :-1].values
    y_dev = data_dev.iloc[:, -1].values

    # 2. Instanciar el modelo con el hiperparámetro alpha
    # CategoricalNB es la versión de Naïve Bayes diseñada para contar frecuencias de palabras. Es decir, para columnas no continuas
    # La idea es que si todas las columnas son categóricas (hay que pasarlas a números con frequency) o convertimos a categóricas las continuas, se use CategoricalNB.
    # Si la mayoría de columnas son continuas, aunque haya categóricas (aquí también hay que pasarlas a números con frequency) usamos GaussianNB.
    # Esto se indica con un parámetro en el JSON, porque el usuario tiene que analizar previamente el dataset.
    # AVISO: CON EL CATEGORICALNB NUNCA USAR TF-IDF PORQUE ENTENDERÍA CADA NÚMERO DECIMAL COMO UNA CATEGORÍA Y SE LIA GORDA.

    if tipo == "categorical":
        from sklearn.naive_bayes import CategoricalNB
        modelo_naive_bayes = CategoricalNB(alpha=alpha)
    elif tipo == "multinomial":
        from sklearn.naive_bayes import MultinomialNB
        modelo_naive_bayes = MultinomialNB(alpha=alpha)
    else:
        from sklearn.naive_bayes import GaussianNB
        modelo_naive_bayes = GaussianNB()

    # 3. Entrenar el modelo
    modelo_naive_bayes.fit(X_train, y_train)

    # 4. Realizar las predicciones sobre el conjunto de desarrollo (dev)
    y_pred = modelo_naive_bayes.predict(X_dev)

    return y_dev, y_pred, modelo_naive_bayes

def guardar_resultados_csv(combinacion_Params, y_dev, y_pred):
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

    archivo_csv = 'resultados_train.csv'

    # Si el archivo no existe, creamos la cabecera primero
    cabecera = not os.path.exists(archivo_csv)

    with open(archivo_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if cabecera:
            writer.writerow(['Combinacion', 'Accuracy', 'Precision', 'Recall', 'F_score'])
        # Guardamos los valores redondeados a 4 decimales
        writer.writerow([combinacion_Params, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

if __name__ == "__main__":
    import sys
    import json
    from sklearn.model_selection import train_test_split

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

    # 2. Leemos qué algoritmo quiere el usuario desde el JSON
    algoritmo = "KNN"  # Valor por defecto
    if config_file:
        with open(config_file, 'r') as file:
            config = json.load(file)
        algoritmo = config.get("algorithm", "KNN")

    # --- INICIO DEL FLUJO DE MACHINE LEARNING COMÚN ---

    # A. Cargamos los datos
    data = load_data(fichero, columna_objetivo, config)

    # B. División del conjunto de train con el de dev. Evitamos Data Leakage para CUALQUIER algoritmo
    data_train, data_dev = train_test_split(data, test_size=0.20, random_state=42, stratify=data[columna_objetivo])

    # C. Aplicamos el preprocesado pasándole ambos trozos
    if config_file:
        data_train, data_dev, mis_herramientas = apply_preprocessing(config_file, data_train, data_dev) #Preprocesamos train y dev y obtenemos las herramientas usadas pa cuando toque con test

    #Bloque para ver los datos preprocesados
    try:
        # Crea la carpeta en el directorio actual
        import os
        os.mkdir("datos_preprocesados")
        print(f"Directorio 'datos_preprocesados' creado exitosamente.")
    except FileExistsError:
        print(f"Error: El directorio 'datos_preprocesados' ya existe.")
    data_train.to_csv("datos_preprocesados/train_preprocesado.csv", index=False)
    if data_dev is not None:
        data_dev.to_csv("datos_preprocesados/dev_preprocesado.csv", index=False)

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
                    f1_actual = calculate_metrics(y_dev, y_pred, config_file)

                    if f1_actual > mejor_f1:
                        mejor_f1 = f1_actual
                        mejor_modelo = modelo_entrenado
                        mejores_hiperparametros = f"k={k}, p={p}, w={weights}"
                        print(f"    [!] ¡Nuevo mejor modelo encontrado! F1: {mejor_f1:.4f}")

                    combinacion_Params = f"k={k}, p={p}, {weights}"
                    guardar_resultados_csv(combinacion_Params, y_dev, y_pred)
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
                f1_actual = calculate_metrics(y_dev, y_pred, config_file)

                if f1_actual > mejor_f1:
                    mejor_f1 = f1_actual
                    mejor_modelo = modelo_entrenado
                    mejores_hiperparametros = f"depth={depth}, split={min_samples_split}, leaf={min_samples_leaf}, crit={crit}"
                    print(f"    [!] ¡Nuevo mejor modelo encontrado! F1: {mejor_f1:.4f}")

                combinacion_Params = f"depth={depth}, split={min_samples_split}, leaf={min_samples_leaf}, crit={crit}"
                guardar_resultados_csv(combinacion_Params, y_dev, y_pred)
        print(f"\n==================================================")
        print(f"EL GANADOR ES: {mejores_hiperparametros} con F1={mejor_f1:.4f}")

        import pickle

        # Creamos un diccionario (el paquete final) con el modelo a usar para test y las herramientas de preprocesado utilizadas con train y dev
        paquete_final = {
            'modelo': mejor_modelo,
            'herramientas_preproceso': mis_herramientas
        }
        nombre_archivo = 'mejor_modelo_decision_tree.pkl'

        archivo = open(nombre_archivo, 'wb')
        pickle.dump(paquete_final, archivo)
        archivo.close()

    ###################################
    #  Empieza algoritmo RandomForest #
    ###################################
    elif algoritmo == "RandomForest":
        print("\n[->] Ejecutando modelo: Random Forest")
        # Leemos los rangos del JSON (con valores por defecto por si acaso)
        hiper_RandomForest = config.get("hyperparametersDecisionTree", {})
        n_estimators = hiper_RandomForest.get("n_estimators", 100)
        min_depth = hiper_RandomForest.get("min_depth", 1)
        max_depth = hiper_RandomForest.get("max_depth", 10)
        min_samples_split = hiper_RandomForest.get("min_samples_split", 5)
        min_samples_leaf = hiper_RandomForest.get("min_samples_leaf", 5)
        criterion_lista = hiper_RandomForest.get("criterion", "Gini")

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
        for depth in range(min_depth, max_depth + 1):
            for crit in criterion_lista:
                print(f"\n--------------------------------------------------")
                print(
                    f"--> Evaluando combinación: max_depth={depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, criterion={crit}")

                # Llamamos a la función con los parámetros de esta iteración
                y_dev, y_pred, modelo_entrenado = randomForest(data_train, data_dev, n_estimators=n_estimators, max_depth=depth,
                                                               min_samples_split=min_samples_split,
                                                               min_samples_leaf=min_samples_leaf, criterion=crit)

                # Mostramos y guardamos resultados de ESTA combinación
                print(calculate_confusion_matrix(y_dev, y_pred))

                # Calculas las métricas. Devuelve el F1 para poder usarlo como decisor.
                f1_actual = calculate_metrics(y_dev, y_pred, config_file)

                if f1_actual > mejor_f1:
                    mejor_f1 = f1_actual
                    mejor_modelo = modelo_entrenado
                    mejores_hiperparametros = f"depth={depth}, split={min_samples_split}, leaf={min_samples_leaf}, crit={crit}"
                    print(f"    [!] ¡Nuevo mejor modelo encontrado! F1: {mejor_f1:.4f}")

                combinacion_Params = f"depth={depth}, split={min_samples_split}, leaf={min_samples_leaf}, crit={crit}"
                guardar_resultados_csv(combinacion_Params, y_dev, y_pred)
        print(f"\n==================================================")
        print(f"EL GANADOR ES: {mejores_hiperparametros} con F1={mejor_f1:.4f}")

        import pickle

        # Creamos un diccionario (el paquete final) con el modelo a usar para test y las herramientas de preprocesado utilizadas con train y dev
        paquete_final = {
            'modelo': mejor_modelo,
            'herramientas_preproceso': mis_herramientas
        }
        nombre_archivo = 'mejor_modelo_random_forest.pkl'

        archivo = open(nombre_archivo, 'wb')
        pickle.dump(paquete_final, archivo)
        archivo.close()

    elif algoritmo == "NaiveBayes": #TODO probar que funciona el algoritmo de NaiveBayes
        print("\n[->] Ejecutando modelo: Naive Bayes")
        # Leemos los rangos del JSON (con valores por defecto por si acaso)
        hiper_NaiveBayes = config.get("hyperparametersNaiveBayes", {})
        min_alpha = hiper_NaiveBayes.get("min_alpha", 0.01)
        max_alpha = hiper_NaiveBayes.get("max_alpha", 1)
        step = hiper_NaiveBayes.get("step_alpha", 0.01)
        tipo_nb = hiper_NaiveBayes.get("type", "multinomial")

        # Borramos el CSV antiguo si existe para empezar limpios
        import os

        if os.path.exists('resultados.csv'):
            os.remove('resultados.csv')

        mejor_f1 = -1.0
        mejor_modelo = None
        mejores_hiperparametros = ""
        if tipo_nb in ["categorical", "multinomial"]:
            # Bucle interno de hiperparámetros (Súper rápido porque el preprocesado ya está hecho)
            # Usamos arange para definir el salto exacto (step)
            # Sumamos un pequeño margen para incluir el max_alpha
            lista_alphas = np.arange(min_alpha, max_alpha + (step / 2), step)
            for alpha in lista_alphas:
                alpha = round(float(alpha), 4)
                print(f"\n--------------------------------------------------")
                print(
                    f"--> Evaluando combinación: alpha:{alpha}")

                # Llamamos a la función con los parámetros de esta iteración
                y_dev, y_pred, modelo_entrenado = naiveBayes(data_train=data_train, data_dev=data_dev, alpha = alpha, tipo=tipo_nb)

                # Mostramos y guardamos resultados de ESTA combinación
                print(calculate_confusion_matrix(y_dev, y_pred))

                # Calculas las métricas. Devuelve el F1 para poder usarlo como decisor.
                f1_actual = calculate_metrics(y_dev, y_pred, config_file)

                if f1_actual > mejor_f1:
                    mejor_f1 = f1_actual
                    mejor_modelo = modelo_entrenado
                    mejores_hiperparametros = f"alpha:{alpha}"
                    print(f"    [!] ¡Nuevo mejor modelo encontrado! F1: {mejor_f1:.4f}")

                combinacion_Params = f"alpha:{alpha}"
                guardar_resultados_csv(combinacion_Params, y_dev, y_pred)
        else:
            # Llamamos a la función con los parámetros
            y_dev, y_pred, modelo_entrenado = naiveBayes(data_train=data_train, data_dev=data_dev, tipo=tipo_nb)

            # Mostramos y guardamos resultados
            print(calculate_confusion_matrix(y_dev, y_pred))

            # Calculas las métricas. Devuelve el F1 para poder usarlo como decisor.
            f1_actual = calculate_metrics(y_dev, y_pred, config_file)

            mejor_f1 = f1_actual
            mejor_modelo = modelo_entrenado
            mejores_hiperparametros = "Gaussian (Sin hiperparámetros)"
            print(f"    [!] ¡Nuevo mejor modelo encontrado! F1: {mejor_f1:.4f}")

            combinacion_Params = f"No tiene alpha: {tipo_nb}"
            guardar_resultados_csv(combinacion_Params, y_dev, y_pred)
        print(f"\n==================================================")
        print(f"EL GANADOR ES: {mejores_hiperparametros} con F1={mejor_f1:.4f}")

        import pickle

        # Creamos un diccionario (el paquete final) con el modelo a usar para test y las herramientas de preprocesado utilizadas con train y dev
        paquete_final = {
            'modelo': mejor_modelo,
            'herramientas_preproceso': mis_herramientas
        }
        nombre_archivo = 'mejor_modelo_naive_bayes.pkl'

        archivo = open(nombre_archivo, 'wb')
        pickle.dump(paquete_final, archivo)
        archivo.close()
    else:
        print(f"Error: Algoritmo '{algoritmo}' no reconocido en el JSON.")
        sys.exit(1)
