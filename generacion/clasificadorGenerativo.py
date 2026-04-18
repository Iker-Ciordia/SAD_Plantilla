from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import argparse
import pandas as pd
from pandas.core.dtypes.missing import construct_1d_array_from_inferred_fill_value
from sklearn.metrics import confusion_matrix, precision_score, \
    f1_score, recall_score  # Importamos el modulo para hacer la matriz de confusión
import json

#run "ollama pull gemma2:2b" in your terminal before running this script

parser=argparse.ArgumentParser(description='ollama LLM evaluation')
parser.add_argument('--model', type=str, default='gemma2:2b', help='ollama model name')
parser.add_argument('--file', type=str, required=True, help='Ruta al archivo CSV a clasificar')
parser.add_argument('--text_col', type=str, required=True, help='Nombre de la columna que contiene el texto de la opinión')
parser.add_argument('--target_col', type=str, required=True, help='Nombre de la columna objetivo (POSITIVO/NEGATIVO/NEUTRO)')
parser.add_argument('--sample', type=int, default=200, help='Límite de filas a evaluar (-1 para evaluar todo el CSV)')
parser.add_argument('--config', type=str, required=True, help='Ruta del fichero JSON de configuración')
args=parser.parse_args()

template = """Eres un analizador de sentimientos experto. Clasifica el comentario en 'Positivo', 'Negativo' o 'Neutro'. Responde ÚNICAMENTE con la etiqueta.
Comentario: {texto}
Etiqueta:"""

prompt = PromptTemplate.from_template(template)
model = OllamaLLM(model=args.model,temperature=0,top_k=10,top_p=0.5) #deterministic
chain = prompt | model

ok = 0
wrongOut = 0

print(f"[*] Cargando datos desde {args.file}...")
df = pd.read_csv(args.file, sep=",")
etiquetas_validas = ['POSITIVO', 'NEGATIVO', 'NEUTRO']

print("Predicción", "Real")
array_real = []
array_prediccion = []
for n, row in df.iterrows():
    if n == args.sample:
        break

    texto = str(row[args.text_col]) #Obtenemos el texto a clasificar

    real = str(row[args.target_col]).strip().upper() #Obtenemos su clase real para compararla luego con lo predicho.

    respuesta = chain.invoke({'texto': texto}).strip().upper() #Llamamos al modelo con la instancia a clasificar y nos guardamos su respuesta
                                                                #El modelo ya tiene el prompt metido ---> chain = prompt | model
    if respuesta in etiquetas_validas:
        array_real.append(real)
        array_prediccion.append(respuesta)
        print(respuesta, row[args.target_col].strip().upper(), n)


etiquetas_fijas = ['NEGATIVO', 'NEUTRO', 'POSITIVO']
#Calcular matriz de confusión
cm = confusion_matrix(array_real, array_prediccion)
#print(real)


#Creamos los nombres para las filas (Realidad) y columnas (Predicción)
nombres_filas = [f"Realidad: {e}" for e in etiquetas_fijas]
nombres_columnas = [f"Predicción: {e}" for e in etiquetas_fijas]

#Juntamos ambos en un DataFrame de pandas para que se imprima bonito
matriz_bonita = pd.DataFrame(cm, index=nombres_filas, columns=nombres_columnas)

print(matriz_bonita)


#Calcular métricas
config_file = args.config
file = open(config_file, 'r')
config = json.load(file)
tipo_metrica = config["metric_to_evaluate"]

print("\nPrecision:")
precision = precision_score(array_real, array_prediccion, average=tipo_metrica)
print(str(tipo_metrica) + ": " + str(precision))

print("\nRecall:")
recall = recall_score(array_real, array_prediccion, average=tipo_metrica)
print(str(tipo_metrica) + ": " + str(recall))

print("\nF-score:")
fscore = f1_score(array_real, array_prediccion, average=tipo_metrica)
print(str(tipo_metrica) + ": " + str(fscore))
