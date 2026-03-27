# 1. Introducción

En este repositorio encontrarás un pequeño proyecto hecho por 2 estudiantes de 3º de carrera para la asignatura de *Sistemas de apoyo a la decisión* del grado de *Ingeniería Informática de Gestión y Sistemas de Información*.

El objetivo principal de este proyecto ha sido la implementación de 4 modelos de predicción supervisada (KNN, Árboles de Decisión, Random Forest y Naïve Bayes). Para ello, se ha tenido que desarrollar un sistema que permite preprocesar los datos de origen para que el modelo escogido por el usuario sepa interpretarlos sin problemas.

# 2. Cómo utilizar el programa

Aquí se explica de forma concisa cómo utilizar el programa. Hemos intentado explicar los aspecto más importantes de la forma más sencilla para que una persona sin muchos conocimientos técnicos en programación lo entienda.

## Requisitos
- Tener un entorno virtual de Python 3.12
- Tener instalados en el entorno virtual los paquetes indicados en el fichero *requirements.txt*

## Ejecución del programa
1. Activa el entorno virtual utilizado para el programa.
2. Sitúate dentro del directorio del repositorio que has clonado.
3. Modifica todos los parámetros que desees dentro del fichero *config_file.json*. Ten en cuenta que esta configuración variará en función del dataset a predecir y del modelo que se quiera utilizar.
4. Crea un modelo predictivo basándote en tu configuración previa con el siguiente comando:
```bash
python entrenadorModelos.py fichero_train.csv "columna_objetivo" -c config_file.json
```
o si no te funciona el anterior:
```bash
py entrenadorModelos.py fichero_train.csv "columna_objetivo" -c config_file.json
```
*El parámetro "fichero_train.csv" será el fichero CSV que tenga tus datos de train. El parámetro "columna_objetivo" será el nombre de la columna a predecir en el fichero CSV.*
5. Una vez se ha creado el modelo (fichero terminado en la extensión .pkl) para la evaluación de los datos de test (otro fichero CSV estructurado igual que el de train) se tiene que ejecutar el siguiente comando.
```bash
python evaluadorModelos.py fichero_test.csv "columna_objetivo" mejor_modelo_X.pkl -c config_file.json
```
o si no te funciona el anterior:
```bash
py evaluadorModelos.py fichero_test.csv "columna_objetivo" mejor_modelo_X.pkl -c config_file.json
```

*El parámetro "fichero_test.csv" será el fichero CSV que tenga tus datos de test. El parámetro "columna_objetivo" será el nombre de la columna a predecir en el fichero CSV.*


## Configuración del Pipeline (`config_file.json`)

El archivo `config_file.json` actúa como el panel de mandos central del proyecto. Permite alterar por completo el comportamiento del preprocesado y probar diferentes algoritmos sin tener que modificar ni una sola línea de código Python.

### 1. Configuración General
* **`dataset_language`**: Define el idioma principal del dataset (ej. `"english"`, `"spanish"`). Es vital para aplicar el diccionario de *stopwords* correcto cuando se procesa texto libre.
* **`algorithm`**: El algoritmo de Machine Learning que se va a entrenar. Opciones soportadas: `"KNN"`, `"DecisionTree"`, `"RandomForest"`, `"NaiveBayes"`.
* **`metric_to_evaluate`**: Métrica utilizada para decidir qué combinación de hiperparámetros es la ganadora (`"micro"`, `"macro"`, `"binary"`).
*"binary" en caso de que sea una clasificación binaria.*

### 2. Módulo de Preprocesamiento (`preprocessing`)
El orden de ejecución de estas transformaciones está estrictamente diseñado para evitar la fuga de datos (*Data Leakage*).
* **Limpieza Básica**: 
  * `separator`: El carácter que separa las columnas en el CSV (ej. `","`).
  * `drop_features`: Lista de columnas irrelevantes (ej. IDs, nombres) que se eliminarán antes de empezar.
* **Gestión de Nulos**: 
  * `missing_values`: Activa la imputación con `"impute"` o la desactiva con `"none"`.
  * `impute_strategy`: Estrategia matemática a usar (`"mean"`, `"median"`, `"most_frequent"`).
* **Discretización**: 
  * `continuous_features_discretize`: Lista de columnas numéricas continuas que se transformarán en categorías puras.
  * `discretize_bins`: Número de agrupaciones o "cajas" a crear.
* **Texto Libre y Categóricas**: 
  * `categorical_features_convert`: Lista de columnas de texto a procesar.
  * `text_preprocess`: Herramienta de vectorización/codificación. Opciones: `"tf-idf"` (importancia en contexto NLP), `"frequency"` (conteo de palabras) o `"one-hot"` (columnas binarias)).
* **Escalado**: 
  * `scaling`: Ajusta los valores numéricos para algoritmos geométricos. Opciones: `"standard"`, `"max-min"`, `"max"`, o `"none"`. *Nota: Debe ser "none" para algoritmos basados en probabilidad (Naive Bayes) o árboles.*
  * `features_scale`: Lista específica de columnas a escalar.
* **Balanceo de Clases**: 
  * `sampling`: Trata problemas de clases desbalanceadas (como detecciones médicas raras). Usa `"oversampling"` (SMOTE) para generar casos sintéticos, `"undersampling"` o `"none"`.
  * `balance`: Ratio deseado entre las clases (ej. `0.5` para equilibrar al 50/50).


### 3. Configuración de Hiperparámetros
El código incluye un enrutador inteligente que busca automáticamente el mejor modelo iterando sobre los rangos definidos en esta sección. Debes modificar el diccionario correspondiente al `algorithm` seleccionado.

* **KNN (`hyperparametersKNN`)**: 
  * Rango de vecinos (`k_min`, `k_max`) y distancias (`p_min`, `p_max`).
* **Árboles (`hyperparametersDecisionTree` / `hyperparametersRandomForest`)**: 
  * Profundidad (`min_depth`, `max_depth`), tamaño mínimo para dividir ramas (`min_samples_split`, `min_samples_leaf`) y número de árboles en caso del bosque (`n_estimators`).
* **Naive Bayes (`hyperparametersNaiveBayes`)**: 
  * **`tipo`**: El subtipo de algoritmo. Usa `"gaussian"` (datos numéricos puros), `"categorical"` (etiquetas de texto o discretizadas) o `"multinomial"` (conteos / NLP).
  * **`alpha`**: Rangos del hiperparámetro de Suavizado de Laplace (`min_alpha`, `max_alpha`, `step_alpha`) para variantes categóricas y multinomiales.
