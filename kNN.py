# -*- coding: utf-8 -*-
"""
Autor: Xabier Gabiña Barañano
Script para la implementación del algoritmo kNN
Recoge los datos de un fichero csv y los clasifica en función de los k vecinos más cercanos
"""

import sys

import json
import sklearn as sk
import numpy as np
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


def apply_preprocessing(data, config_file): #TODO Revisar lo que hace la función en profundidad
    """
    Aplica el preprocesado a los datos basándose en un archivo JSON.
    """
    #Leer el archivo JSON
    file = open(config_file, 'r')
    config = json.load(file)

    opciones = config.get("preprocessing", {}) #Nos quedamos con el segundo JSON de parámetros de preprocesado dentro del principal
    print(f"Aplicando preprocesado desde {config_file}...")

    #Eliminar atributos innecesarios (aquellos que no queramos usar para el entrenamiento)
    if "drop_features" in opciones and len(opciones["drop_features"]) > 0: #Si existe una llave "drop..." y NO está vacía
        columnas_a_borrar = []
        for col in opciones["drop_features"]: #Para toda columna que se quiera eliminar
            if col in data.columns: #Si la columna se encuentra en el DataFrame
                columnas_a_borrar.append(col) #Añade la columna a la lista que habrá que borrar luego

    data = data.drop(columns=columnas_a_borrar) #Borra del DataFrame todas las columnas que se hayan indicado en el JSON y no interesan
    print(f" -> Columnas eliminadas: {columnas_a_borrar}")

    #Separamos temporalmente las columnas de atributos de la clase objetivo para no alterarla
    columnas_x = data.columns[:-1] #Desde la primera a la penúltima
    columna_y = data.columns[-1] #La última (previamente hemos ordenado para que la objetivo siempre esté al final

    #Tratar valores faltantes
    if opciones.get("missing_values") == "impute": #Si la clave "missing..." dice que hay que imputar valores
        estrategia = opciones.get("impute_strategy", "mean")  #Cogemos el valor que se indique en la estrategía.
                                                              #Si está vacío, se coge la media por defecto (de ahí el segundo param.)
        print(f" -> Imputando valores faltantes usando la estrategia: {estrategia}")

        # Seleccionamos solo las columnas numéricas para imputar (evita errores con texto)
        num_cols = data[columnas_x].select_dtypes(include=[np.number]).columns #Del DataFrame nos quedamos con las filas de las columnas que no son la columna a predecir.
                                                                               #Nos quedamos con las filas de aquellas columnas que sean numéricas (include=[np.number]).
                                                                               #.columns no da de ese DataFrame final sin columnas categóricas da los nombres de las columnas.
                                                                               #El objetivo es sacar los nombres de las columnas numéricas.

        if len(num_cols) > 0: #Si hay al menos una columna numérica
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy=estrategia) #Prepara la herramienta de imputación de valores
            data[num_cols] = imputer.fit_transform(data[num_cols]) #Imputamos los valores faltantes con la estrategía extraída previamente.

    #Preprocesamiento de texto (TF-IDF)
    if opciones.get("text_process") == "tf-idf": #TODO Habría que hacer alternativa para BOW tanto One-Hot como de frecuencia
        #Buscamos columnas de tipo 'object' (texto)
        text_cols = data[columnas_x].select_dtypes(include=['object']).columns #De las columnas que no son la que hay que predecir,
                                                                               #nos quedamos con los nombres de aquellas que son texto (al revés que arriba).
        if len(text_cols) > 0: #Si existe alguna columna que sea texto
            print(f" -> Aplicando TF-IDF a las columnas de texto: {list(text_cols)}")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer() #Creamos la herramienta de TF-IDF
            for col in text_cols:
                #Transformamos la columna de texto en una matriz de características numéricas
                #Cada fila de la matriz representa una fila (una instancia) de la columna que estamos convirtiendo a TF-IDF en forma de vector numérico.
                #De tal forma que cada elemento en ese vector es un número con el valor TF-IDF asignado.
                tfidf_matrix = vectorizer.fit_transform(data[col].astype(str)) #El astype convierte cualquier cosa (ya sea filas de solo números o filas vacías en string)
                                                                               #Si no la función fallaría

                # Convertimos la matriz en un DataFrame con nombres de columnas
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                                        columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
                # Eliminamos la columna de texto original y unimos las nuevas numéricas
                data = data.drop(columns=[col]).join(tfidf_df)

    # Volvemos a asegurar que la columna objetivo (y) esté al final tras las posibles modificaciones
    columnas_finales = data.columns.tolist()
    columnas_finales.remove(columna_y)
    columnas_finales.append(columna_y)
    data = data[columnas_finales]

    return data




def calculate_metrics(y_test, y_pred):
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



def calculate_confusion_matrix(y_test, y_pred):
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

def kNN(data, k, weights, p):
    """
    Función para implementar el algoritmo kNN
    
    :param data: Datos a clasificar
    :type data: pandas.DataFrame
    :param k: Número de vecinos más cercanos
    :type k: int
    :param weights: Pesos utilizados en la predicción ('uniform' o 'distance')
    :type weights: str
    :param p: Parámetro para la distancia métrica (1 para Manhattan, 2 para Euclídea)
    :type p: int
    :return: Clasificación de los datos
    :rtype: tuple
    """
    # Seleccionamos las características y la clase.
    # El .values se usa para convertirlo de DataFrame a matriz normal, que es lo que usa Skicit.
    X = data.iloc[:, :-1].values # Todas las columnas menos la última (atributos que se van a usar para entrenar)
    y = data.iloc[:, -1].values # Última columna (atributo a predecir). Sí o sí está en la última columna

    # Dividimos los datos en entrenamiento y test
    from sklearn.model_selection import train_test_split
    np.random.seed(42)  # Set a random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) #El último parámetro es la proporción
    
    # Escalamos los datos
    from sklearn.preprocessing import StandardScaler #Importamos la libreria para escalar valores
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train) #Escalamos el dataset de training con Z-Score
    X_test = sc.transform(X_test) #Escalamos el dataset de test con Z-Score
    
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
    # Comprobamos que se han introducido los parámetros correctos
    if len(sys.argv) < 4:
        print("Error en los parámetros de entrada")
        print("Uso: kNN.py <fichero*> <columna_objetivo*> <k*> [<weights>] [<p>] [-c <config.json>]")
        sys.exit(1)

    # Asignamos las variables desde la consola para que sea más fácil de leer
    fichero = sys.argv[1]
    columna_objetivo = sys.argv[2]
    k = int(sys.argv[3])

    #Variables opcionales con valores por defecto
    weights = 'uniform'
    p = 2
    config_file = None

    # Parseo manual para buscar el -c y su archivo JSON
    #TODO Revisar en profundidad qué coño es todo esto del "-c" en el parámetro del JSON
    if "-c" in sys.argv:
        indice_c = sys.argv.index("-c")
        if indice_c + 1 < len(sys.argv):
            config_file = sys.argv[indice_c + 1]
            # Eliminamos '-c' y el nombre del archivo de sys.argv para que no interfieran con weights y p
            sys.argv.pop(indice_c)
            sys.argv.pop(indice_c)

            # Asignamos weights y p si se proporcionaron (ahora que sys.argv está limpio del -c)
    if len(sys.argv) > 4:
        weights = sys.argv[4]
    if len(sys.argv) > 5:
        p = int(sys.argv[5])

    # Cargamos los datos
    data = load_data(fichero, columna_objetivo)

    #Si se ha especificado un archivo JSON, aplicamos el preprocesado
    if config_file:
        data = apply_preprocessing(data, config_file)

    # Implementamos el algoritmo kNN
    y_test, y_pred = kNN(data, k, weights, p)
    
    # Mostramos la matriz de confusión
    print("\nMatriz de confusión:")
    print(calculate_confusion_matrix(y_test, y_pred)) #Creamos la matriz de confusión con las clases reales de test
                                                      #y las predicciones sobre la clase real hechas

    # Mostramos el F-score, Precision y Recall, tanto Micro como Macro
    calculate_metrics(y_test, y_pred)

    #Guardamos los resultados en el CSV
    #guardar_resultados_csv(k, p, weights, y_test, y_pred) #TODO decidir cómo hacerlo, si con macro, micro o weighted
