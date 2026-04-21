import os
import sys
import json
import pandas as pd

# Forzamos a Python a mirar una carpeta más arriba (la raíz SAD_Plantilla)
ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_raiz not in sys.path:
    sys.path.append(ruta_raiz)
from clasificacion import entrenadorModelos

if __name__ == "__main__":
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

    data = entrenadorModelos.load_data(fichero, columna_objetivo, config)

    agrupar = config.get("preprocessing").get("agrupar_sentimiento_proyecto", False)

    if agrupar:
        diccionario_sustitucion = {
            1: 'NEGATIVO', 2: 'NEGATIVO', 3: 'NEUTRO', 4: 'POSITIVO', 5: 'POSITIVO'
        }
        if columna_objetivo in data.columns:
            data[columna_objetivo] = data[columna_objetivo].replace(diccionario_sustitucion)
            print(f"[*] JSON indica agrupar: Columna '{columna_objetivo}' convertida a NEGATIVO/NEUTRO/POSITIVO")

    # --- INICIO DEL TROCEADO DEL DATASET ---
    print("\n[->] Iniciando la división del dataset por clases...")

    # 1. Creamos una carpeta para guardar los resultados limpios (exist_ok=True evita el error si ya existe)
    carpeta_salida = "./clustering/ficheros_divididos"
    os.makedirs(carpeta_salida, exist_ok=True)

    # 2. Extraemos el nombre original del archivo para reciclarlo (ej: de "datos/Instagram.csv" saca "Instagram")
    nombre_base = os.path.splitext(os.path.basename(fichero))[0]

    # 3. Sacamos la lista de las etiquetas únicas que existen en la columna objetivo
    # (Serán 'POSITIVO', 'NEGATIVO', 'NEUTRO', o las que hubiera si no se agrupó)
    etiquetas_unicas = data[columna_objetivo].unique()

    # 4. Filtramos y guardamos
    for etiqueta in etiquetas_unicas:
        # Asegurarnos de que no es un valor nulo/NaN
        if pd.isna(etiqueta):
            continue

        # Filtramos: Nos quedamos solo con las filas donde la columna objetivo sea igual a la etiqueta actual
        data_filtrada = data[data[columna_objetivo] == etiqueta]

        # Limpiamos el nombre de la etiqueta por si tiene espacios (para que el nombre del archivo sea válido)
        nombre_etiqueta_limpia = str(etiqueta).replace(" ", "_").upper()

        # Construimos la ruta del nuevo fichero (ej: ficheros_divididos/Instagram_POSITIVO.csv)
        ruta_nuevo_fichero = os.path.join(carpeta_salida, f"{nombre_base}_{nombre_etiqueta_limpia}.csv")

        # Guardamos el DataFrame filtrado (sin procesar, tal cual estaba)
        data_filtrada.to_csv(ruta_nuevo_fichero, index=False)

        print(f" [+] Generado: {ruta_nuevo_fichero} -> Contiene {len(data_filtrada)} comentarios.")

    print("\n[V] ¡División completada con éxito!")