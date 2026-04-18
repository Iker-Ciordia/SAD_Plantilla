from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import argparse
import pandas as pd
from pandas.core.dtypes.missing import construct_1d_array_from_inferred_fill_value
from sklearn.metrics import confusion_matrix, precision_score, \
    f1_score, recall_score  # Importamos el modulo para hacer la matriz de confusión
import json
import re

#run "ollama pull gemma2:2b" in your terminal before running this script

parser=argparse.ArgumentParser(description='ollama LLM evaluation')
parser.add_argument('--model', type=str, default='gemma2:2b', help='ollama model name')
parser.add_argument('--file', type=str, required=True, help='Ruta al archivo CSV a clasificar')
parser.add_argument('--text_col', type=str, required=True, help='Nombre de la columna que contiene el texto de la opinión')
parser.add_argument('--target_col', type=str, required=True, help='Nombre de la columna objetivo (POSITIVO/NEGATIVO/NEUTRO)')
parser.add_argument('--sample', type=int, default=200, help='Límite de filas a evaluar (-1 para evaluar todo el CSV)')
parser.add_argument('--config', type=str, required=True, help='Ruta del fichero JSON de configuración')
parser.add_argument('--mode', type=str, choices=['classify', 'generate'], default='classify', help='Modo de ejecución: classify o generate')
parser.add_argument('--gen_class', type=str, default='POSITIVO', help='Clase a generar (ej. POSITIVO, NEGATIVO, NEUTRO)')
parser.add_argument('--gen_count', type=int, default=10, help='Número de instancias a generar')
parser.add_argument('--out_file', type=str, default='instancias_generadas.csv', help='Ruta para guardar los datos generados')
args=parser.parse_args()

def clasificar_instancias(args):
    # Prompt actualizado para disuadir el uso de <think>
    template = """You are an expert sentiment analyzer. Classify the comment as 'POSITIVO', 'NEGATIVO', or 'NEUTRO'. 
Respond ONLY with the label. Do not use <think> reasoning tags or provide explanations.
Comment: {texto}
Label:"""

    prompt = PromptTemplate.from_template(template)
    model = OllamaLLM(model=args.model,temperature=0,top_k=10,top_p=0.5) #deterministic
    chain = prompt | model

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

        # --- EXTRACCIÓN Y LIMPIEZA REGEX ---
        respuesta_cruda = chain.invoke({'texto': texto})
        respuesta_limpia = re.sub(r'<think>.*?</think>', '', respuesta_cruda, flags=re.IGNORECASE | re.DOTALL)
        respuesta = respuesta_limpia.strip().upper() 
        # -----------------------------------

        if respuesta in etiquetas_validas:
            array_real.append(real)
            array_prediccion.append(respuesta)
            print(respuesta, row[args.target_col].strip().upper(), n)

    etiquetas_fijas = ['NEGATIVO', 'NEUTRO', 'POSITIVO']
    #Calcular matriz de confusión
    cm = confusion_matrix(array_real, array_prediccion, labels=etiquetas_fijas)

    #Creamos los nombres para las filas (Realidad) y columnas (Predicción)
    nombres_filas = [f"Realidad: {e}" for e in etiquetas_fijas]
    nombres_columnas = [f"Predicción: {e}" for e in etiquetas_fijas]

    #Juntamos ambos en un DataFrame de pandas para que se imprima bonito
    matriz_bonita = pd.DataFrame(cm, index=nombres_filas, columns=nombres_columnas)
    print(matriz_bonita)

    #Calcular métricas
    file = open(args.config, 'r')
    config = json.load(file)
    tipo_metrica = config["metric_to_evaluate"]

    print("\nPrecision:")
    precision = precision_score(array_real, array_prediccion, average=tipo_metrica, zero_division=0)
    print(str(tipo_metrica) + ": " + str(precision))

    print("\nRecall:")
    recall = recall_score(array_real, array_prediccion, average=tipo_metrica, zero_division=0)
    print(str(tipo_metrica) + ": " + str(recall))

    print("\nF-score:")
    fscore = f1_score(array_real, array_prediccion, average=tipo_metrica, zero_division=0)
    print(str(tipo_metrica) + ": " + str(fscore))


def generar_instancias(args):
    # 1. Intentar cargar datos reales para usarlos como contexto (Few-Shot)
    tiene_ejemplos = False
    df_filtrado = None
    try:
        df_real = pd.read_csv(args.file, sep=",")
        # Filtramos quedándonos solo con los textos de la clase que queremos generar
        df_filtrado = df_real[df_real[args.target_col].str.strip().str.upper() == args.gen_class.upper()]
        tiene_ejemplos = not df_filtrado.empty
    except Exception as e:
        print(f"[!] Aviso: No se pudieron cargar ejemplos de contexto de {args.file} ({e})")

    # Prompt actualizado para disuadir el uso de <think>
    template = """You are an expert data generator for Machine Learning training. 
Generate a NEW, realistic user comment or review that clearly expresses a {sentimiento} sentiment.
The comment must be natural, varied, of the same style and length as the examples, and strictly in English.

Here are some real examples with {sentimiento} sentiment to give you context:
{ejemplos}

Respond ONLY with the text of the generated comment. Do not include quotes, introductions, additional notes, or <think> reasoning tags."""
    
    prompt = PromptTemplate.from_template(template)
    # Aumentamos la temperatura para que los comentarios generados sean creativos y variados
    model = OllamaLLM(model=args.model, temperature=0.8) 
    chain = prompt | model

    print(f"[*] Generando {args.gen_count} instancias sintéticas para la clase: '{args.gen_class}'...")
    if tiene_ejemplos:
        print(f"[*] Usando ejemplos de contexto extraídos de {args.file}")
    
    nuevas_instancias = []
    for i in range(args.gen_count):
        ejemplos_texto = "No hay ejemplos disponibles."
        if tiene_ejemplos:
            # Extraer hasta 3 ejemplos aleatorios reales diferentes en cada iteración para mayor variedad
            num_ejemplos = min(3, len(df_filtrado))
            ejemplos = df_filtrado.sample(n=num_ejemplos)[args.text_col].tolist()
            ejemplos_texto = "\n".join([f"- {ej}" for ej in ejemplos])

        # --- EXTRACCIÓN Y LIMPIEZA REGEX ---
        respuesta_cruda = chain.invoke({
            'sentimiento': args.gen_class,
            'ejemplos': ejemplos_texto
        })
        
        # Eliminamos el bloque de pensamiento de la salida en bruto
        texto_generado = re.sub(r'<think>.*?</think>', '', respuesta_cruda, flags=re.IGNORECASE | re.DOTALL).strip()
        
        # Limpieza básica por si el modelo incluye comillas
        texto_generado = texto_generado.strip('"').strip("'")
        # -----------------------------------
        
        nuevas_instancias.append({
            args.text_col: texto_generado,
            args.target_col: args.gen_class
        })
        print(f"  [{i+1}/{args.gen_count}] Generado: {texto_generado}")

    df_generado = pd.DataFrame(nuevas_instancias)
    df_generado.to_csv(args.out_file, index=False, encoding='utf-8')
    print(f"\n[+] Proceso finalizado. Las instancias se han guardado en: {args.out_file}")


if __name__ == "__main__":
    if args.mode == 'classify':
        clasificar_instancias(args)
    elif args.mode == 'generate':
        generar_instancias(args)