import os
import json
import re
import pandas as pd
from types import SimpleNamespace
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score

ruta_json = 'generacion/generacion.json'  # Asegúrate de que el nombre coincide con tu archivo

try:
    with open(ruta_json, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
        # Convertimos el diccionario en un objeto para seguir usando args.propiedad
        args = SimpleNamespace(**config_dict)
except FileNotFoundError:
    print(f"[!] ERROR: No se ha encontrado el archivo de configuración '{ruta_json}'.")
    exit(1)

# Asignación de valores por defecto si no vienen en el JSON
if not hasattr(args, 'out_file') or args.out_file is None:
    modelo_seguro = args.model.replace(':', '_')
    args.out_file = f'nuevo_{modelo_seguro}.csv'

if not hasattr(args, 'sample'):
    args.sample = -1

def normalizar(val):
    val = val.strip()
    if val in ['4', '5']: return "POSITIVO"
    if val == '3': return "NEUTRO"
    if val in ['1', '2']: return "NEGATIVO"
    return val.upper()


def clasificar_instancias(args):
    print(f"[*] Cargando datos de evaluación desde {args.file}...")
    df = pd.read_csv(args.file, sep=",")

    if getattr(args, 'train_file', None) is None:
        print("[!] ERROR: En el JSON necesitas definir 'train_file' para los ejemplos")
        return

    df_train = pd.read_csv(args.train_file, sep=",")
    df_ejemplos = df_train.copy()
    df_ejemplos['label_temp'] = df_ejemplos[args.target_col].astype(str).str.strip()
    df_ejemplos['label_temp'] = df_ejemplos['label_temp'].apply(normalizar)

    tipos_de_prompt = ['0-shot', '1-shot', 'few-shot']
    registro_prompts_total = []

    for tipo_prompt in tipos_de_prompt:
        str_ejemplos = ""
        ejemplos_list = []

        if tipo_prompt == 'few-shot':
            for clase in ['POSITIVO', 'NEGATIVO', 'NEUTRO']:
                pool = df_ejemplos[df_ejemplos['label_temp'] == clase]
                if not pool.empty:
                    ex = pool.sample(n=1).iloc[0]
                    ejemplos_list.append(f"- Comment: \"{ex[args.text_col]}\" -> Label: {clase}")
            str_ejemplos = "\n".join(ejemplos_list)

        elif tipo_prompt == '1-shot':
            pool = df_ejemplos[df_ejemplos['label_temp'].isin(['POSITIVO', 'NEGATIVO', 'NEUTRO'])]
            if not pool.empty:
                ex = pool.sample(n=1).iloc[0]
                ejemplos_list.append(f"- Comment: \"{ex[args.text_col]}\" -> Label: {ex['label_temp']}")
            str_ejemplos = "\n".join(ejemplos_list)

        # Construimos el template correspondiente
        if tipo_prompt == "few-shot":
            template = f"""You are an expert sentiment analyzer. Classify the comment as 'POSITIVO', 'NEGATIVO', or 'NEUTRO'. 
Respond ONLY with the label. Do not use <think> reasoning tags or provide explanations.

Examples:
{str_ejemplos}

Comment: {{texto}}
Label:"""
        elif tipo_prompt == "1-shot":
            template = f"""You are an expert sentiment analyzer. Classify the comment as 'POSITIVO', 'NEGATIVO', or 'NEUTRO'. 
Respond ONLY with the label. Do not use <think> reasoning tags or provide explanations.

Example:
{str_ejemplos}

Comment: {{texto}}
Label:"""
        else:
            template = f"""You are an expert sentiment analyzer. Classify the comment as 'POSITIVO', 'NEGATIVO', or 'NEUTRO'. 
Respond ONLY with the label. Do not use <think> reasoning tags or provide explanations.

Comment: {{texto}}
Label:"""

        prompt = PromptTemplate.from_template(template)
        model = OllamaLLM(model=args.model, temperature=0.5, top_k=10, top_p=0.5)
        chain = prompt | model

        array_real = []
        array_prediccion = []
        etiquetas_validas = ['POSITIVO', 'NEGATIVO', 'NEUTRO']

        for n, row in df.iterrows():
            if n == args.sample:
                break

            texto = str(row[args.text_col])
            real = str(row[args.target_col]).strip()
            real = normalizar(real)

            # Invocación al modelo
            respuesta_cruda = chain.invoke({'texto': texto})
            respuesta_limpia = re.sub(r'<think>.*?</think>', '', respuesta_cruda, flags=re.IGNORECASE | re.DOTALL)
            respuesta = respuesta_limpia.strip().upper()

            registro_prompts_total.append({
                "modelo/tamaño": args.model,
                "tipo de prompt": tipo_prompt,
                "prompt": template.replace("{texto}", texto),
                "entrada": texto,
                "salida": respuesta_limpia.strip()
            })

            if respuesta in etiquetas_validas:
                array_real.append(real)
                array_prediccion.append(respuesta)

        # --- Calcular e imprimir métricas de ESTE EXPERIMENTO ---
        etiquetas_fijas = ['NEGATIVO', 'NEUTRO', 'POSITIVO']
        cm = confusion_matrix(array_real, array_prediccion, labels=etiquetas_fijas)
        nombres_filas = [f"Realidad: {e}" for e in etiquetas_fijas]
        nombres_columnas = [f"Predicción: {e}" for e in etiquetas_fijas]
        matriz_bonita = pd.DataFrame(cm, index=nombres_filas, columns=nombres_columnas)

        print(f"\n--- Resultados para {tipo_prompt.upper()} ---")
        print(matriz_bonita)

        # Usamos la métrica directamente desde el JSON general
        tipo_metrica = args.metric_to_evaluate

        print(
            f"\nPrecision ({tipo_metrica}): {precision_score(array_real, array_prediccion, average=tipo_metrica, zero_division=0):.4f}")
        print(
            f"Recall ({tipo_metrica}): {recall_score(array_real, array_prediccion, average=tipo_metrica, zero_division=0):.4f}")
        print(
            f"F-score ({tipo_metrica}): {f1_score(array_real, array_prediccion, average=tipo_metrica, zero_division=0):.4f}")

    df_prompts = pd.DataFrame(registro_prompts_total)
    nombre_archivo = "prompts_test.csv"
    hdr = not os.path.exists(nombre_archivo)
    df_prompts.to_csv(nombre_archivo, mode='a', index=False, header=hdr, encoding='utf-8')


def generar_instancias(args):
    tiene_ejemplos = False
    df_filtrado = None
    try:
        df_real = pd.read_csv(args.file, sep=",")
        df_filtrado = df_real[df_real[args.target_col].str.strip().str.upper() == args.gen_class.upper()]
        tiene_ejemplos = not df_filtrado.empty
    except Exception as e:
        print(f"[!] Aviso: No se pudieron cargar ejemplos de contexto de {args.file} ({e})")

    template = """You are an expert data generator for Machine Learning training. 
Generate a NEW, realistic user comment or review that clearly expresses a {sentimiento} sentiment.
The comment must be natural, varied, of the same style and length as the examples, and strictly in English.

Here are some real examples with {sentimiento} sentiment to give you context:
{ejemplos}

Respond ONLY with the text of the generated comment. Do not include quotes, commas (,), introductions, additional notes, or <think> reasoning tags.
CRITICAL: NEVER use commas in your response as it will break the CSV formatting."""

    prompt = PromptTemplate.from_template(template)
    model = OllamaLLM(model=args.model, temperature=0.8)
    chain = prompt | model

    print(f"[*] Generando {args.gen_count} instancias sintéticas para la clase: '{args.gen_class}'...")
    if tiene_ejemplos:
        print(f"[*] Usando ejemplos de contexto extraídos de {args.file}")

    nuevas_instancias = []
    for i in range(args.gen_count):
        ejemplos_texto = "No hay ejemplos disponibles."
        if tiene_ejemplos:
            num_ejemplos = min(3, len(df_filtrado))
            ejemplos = df_filtrado.sample(n=num_ejemplos)[args.text_col].tolist()
            ejemplos_texto = "\n".join([f"- {ej}" for ej in ejemplos])

        respuesta_cruda = chain.invoke({
            'sentimiento': args.gen_class,
            'ejemplos': ejemplos_texto
        })

        texto_generado = re.sub(r'<think>.*?</think>', '', respuesta_cruda, flags=re.IGNORECASE | re.DOTALL).strip()
        texto_generado = texto_generado.strip('"').strip("'")

        nuevas_instancias.append({
            args.text_col: texto_generado,
            args.target_col: args.gen_class
        })
        print(f"  [{i + 1}/{args.gen_count}] Generado: {texto_generado}")

    df_generado = pd.DataFrame(nuevas_instancias)
    df_generado.to_csv(args.out_file, index=False, encoding='utf-8')
    print(f"\n[+] Proceso finalizado. Las instancias se han guardado en: {args.out_file}")


if __name__ == "__main__":
    if args.mode == 'classify':
        clasificar_instancias(args)
    elif args.mode == 'generate':
        generar_instancias(args)