# 1. Introducción

En este repositorio encontrarás un pequeño proyecto hecho por 2 estudiantes de 3º de carrera para la asignatura de *Sistemas de apoyo a la decisión* del grado de *Ingeniería Informática de Gestión y Sistemas de Información*.

El objetivo principal de este proyecto ha evolucionado para abarcar tres grandes áreas de la Inteligencia Artificial:
1. **Clasificación supervisada:** Implementación de 4 modelos de predicción (KNN, Árboles de Decisión, Random Forest y Naïve Bayes) con un sistema de preprocesamiento dinámico.
2. **Aprendizaje no supervisado (Clustering):** Agrupación de instancias mediante el algoritmo K-Means para descubrir patrones ocultos y extraer los tópicos más relevantes utilizando procesamiento de lenguaje natural (NLP).
3. **IA generativa:** Integración con modelos de lenguaje locales (LLMs) mediante Ollama para clasificar texto o generar datos sintéticos basados en el contexto original del dataset.

# 2. Arquitectura del proyecto
El proyecto está dividido en varios scripts, cada uno con una responsabilidad única:
- **`clasificacion/entrenadorModelos.py`**: El núcleo de aprendizaje supervisado. Aplica técnicas de preprocesamiento (limpieza, imputación, vectorización de texto, escalado, balanceo SMOTE), entrena el algoritmo seleccionado, busca la mejor combinación de hiperparámetros y exporta el modelo ganador en formato `.pkl`.
- **`clasificacion/evaluadorModelos.py`**: Toma un modelo entrenado (`.pkl`) y un dataset de prueba, aplica ciegamente las reglas de preprocesamiento aprendidas durante el entrenamiento (evitando el Data Leakage) y evalúa el rendimiento generando una matriz de confusión y métricas (F1, Precision, Recall).
- **`clustering/dividirDatos.py`**: Herramienta de utilidad que fragmenta un dataset principal en varios archivos CSV más pequeños, separados en base a las etiquetas únicas de la columna objetivo (ej. POSITIVO, NEGATIVO, NEUTRO).
- **`clustering/plantillaClustering.py`**: Script de aprendizaje no supervisado. Ejecuta K-Means, muestra una gráfica del método del codo para que el usuario decida el número óptimo de clústeres (*K*), clasifica las instancias en cada cluster y extrae las palabras clave de cada tópico utilizando el vectorizador TF-IDF/Frequency.
- **`generacion/plantillaGenerativa.py`**: Utiliza modelos locales de la familia Ollama (ej. gemma2:2b) para evaluar sentimientos Zero-Shot o generar instancias sintéticas (Few-Shot) para balancear clases minoritarias o para clasificar instancias.


# 3. Requisitos
- Tener un entorno virtual de Python 3.12
- Tener instalados en el entorno virtual los paquetes indicados en el fichero *requirements.txt*
- *(Opcional, solo para IA Generativa):* Tener instalado Ollama en tu sistema y haber descargado el modelo necesario ejecutando en tu terminal: `ollama pull gemma2:2b`.

# 4. Cómo utilizar el programa

Aquí se explica de forma concisa cómo utilizar el programa. Hemos intentado explicar los aspecto más importantes de la forma más sencilla para que una persona sin muchos conocimientos técnicos en programación lo entienda.


## 4.1 Modelos supervisados (entrenamiento y evaluación)
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


## 4.2 Aprendizaje no supervisado (Clustering)
1. Activa el entorno virtual utilizado para el programa.
2. Sitúate dentro del directorio del repositorio que has clonado.
3. Modifica todos los parámetros que desees dentro del fichero *config_file.json*. Ten en cuenta que esta configuración variará en función del dataset a dividir en clusters.
4. Separa tu dataset en múltiples ficheros CSV basados en las distintas clases de la columna objetivo:
```bash
python dividirDatos.py fichero_datos.csv "columna_objetivo" -c config_file.json
```
5. Aplicar algoritmo de clustering (en función del fichero de configuración JSON) para obtener instancias clasificadas en clusters y palabras más significativas de cada cluster:
```bash
python plantillaClustering.py fichero_datos.csv "columna_objetivo" -c config_file.json
```
o si no te funciona el anterior:
```bash
py plantillaClustering.py fichero_datos.csv "columna_objetivo" -c config_file.json
```
*El script te mostrará una gráfica interactiva en el caso de que uses K-Means (Método del Codo). Deberás observar dónde se aplana la curva, cerrar la ventana gráfica y escribir por consola el número óptimo de **K** que deseas aplicar.*

6. Los archivos resultantes se guardarán en la carpeta ./clustering/ficheros_divididos/.


# 5. Configuración del Pipeline (`config_file.json`)

El archivo `config_file.json` actúa como el panel de mandos central del proyecto. Permite alterar por completo el comportamiento del preprocesado y probar diferentes algoritmos sin tener que modificar el código Python.

### 5.1. Configuración General
* **`proyecto`**: Booleano (`true` / `false`). Si se activa, aplica la lógica específica y particular de este proyecto (como divisiones predefinidas de train/dev/test y un filtrado especial de *stopwords*).
* **`dataset_language`**: Define el idioma principal del dataset (ej. `"english"`, `"spanish"`). Es vital para cargar el diccionario adecuado de *stopwords* y aplicar la lematización o *stemming* correctos en los textos.
* **`algorithm`**: El algoritmo principal que se va a ejecutar (opciones: `"KNN"`, `"DecisionTree"`, `"RandomForest"`, `"NaiveBayes"`, `"K-Means"`).
* **`metric_to_evaluate`**: Métrica utilizada para evaluar y decidir el modelo ganador durante la búsqueda de hiperparámetros (`"micro"`, `"macro"`, `"binary"`).

### 5.2. Módulo de Preprocesamiento (`preprocessing`)
El orden de las transformaciones está diseñado para evitar fugas de datos (*Data Leakage*).
* **Limpieza y Preparación Básica**:
  * `agrupar_sentimiento_proyecto`: Booleano que, si es `true`, convierte valoraciones numéricas (1-5) automáticamente en etiquetas categóricas puras (`NEGATIVO`, `NEUTRO`, `POSITIVO`).
  * `separator`: El carácter que separa las columnas en el CSV (ej. `","`).
  * `drop_features`: Lista de columnas irrelevantes que se eliminarán antes de empezar (ej. `["reviewId", "gender", "location", "date"]`).
* **Gestión de Nulos**:
  * `missing_values`: Activa la imputación con `"impute"` o la desactiva con `"none"`.
  * `impute_strategy`: Estrategia matemática a usar para rellenar huecos (ej. `"mean"`, `"median"`, `"most_frequent"`).
* **Discretización**:
  * `continuous_features_discretize`: Lista de columnas numéricas continuas que se transformarán en rangos o categorías.
  * `discretize_bins`: Número de agrupaciones o "cajas" a crear (ej. `10`).
* **Texto Libre (NLP)**:
  * `categorical_features_convert`: Lista de columnas de texto libre a procesar (ej. `["content"]`).
  * `text_preprocess`: Técnica de codificación a aplicar (`"tf-idf"`, `"frequency"`, o `"one-hot"`).
  * `ngramas_tfidf`: Número máximo de n-gramas a considerar si se usa la estrategia TF-IDF (ej. `3` para incluir desde unigramas hasta trigramas).
* **Escalado**:
  * `scaling`: Ajusta las métricas para algoritmos basados en distancias geométricas (`"standard"`, `"max-min"`, `"max"`, `"none"`).
  * `features_scale`: Columnas concretas a las que aplicar el escalado (ej. `["bill_length_mm", "body_mass_g"]`).
* **Balanceo de Clases**:
  * `sampling`: Trata problemas de clases desbalanceadas. Usa `"oversampling"` (SMOTE), `"undersampling"` o `"none"`.
  * `balance`: Ratio de balanceo deseado (ej. `"auto"` para equilibrar automáticamente).

### 5.3. Configuración de Hiperparámetros
El script evaluará combinaciones de forma automatizada iterando sobre los rangos y arrays que definas en estos diccionarios:
* **KNN (`hyperparametersKNN`)**:
  * Rangos de vecinos (`k_min`, `k_max`), parámetros de cálculo de distancia de Minkowski (`p_min`, `p_max`), función de peso (`w`: `["uniform", "distance"]`) y el salto de iteración (`step`).
* **Árboles de Decisión (`hyperparametersDecisionTree`)**:
  * Profundidad del árbol (`min_depth`, `max_depth`), tamaño mínimo para dividir ramas (`min_samples_split`), mínimo de muestras en hojas (`min_samples_leaf`) y métricas de calidad de la partición (`criterion`: `["gini", "entropy"]`).
* **Random Forest (`hyperparametersRandomForest`)**:
  * Se alimenta de la configuración del árbol de decisión para construir el bosque y añade la cantidad total de árboles a generar (`n_estimators`).
* **Naive Bayes (`hyperparametersNaiveBayes`)**:
  * `type`: Variante matemática a usar (`"gaussian"`, `"categorical"`, `"multinomial"`, `"complement"`).
  * Suavizado de Laplace para las variantes categóricas/multinomiales (`min_alpha`, `max_alpha`, `step_alpha`).
* **K-Means (`hyperparametersKMeans`)**:
  * Rango de clústeres a evaluar para trazar la gráfica del codo (`k_min`, `k_max`), salto (`step`) y número de inicializaciones independientes (`n_inicios`).