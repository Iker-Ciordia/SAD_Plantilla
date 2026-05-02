import pandas as pd


def preparar_y_unir_datasets(ruta_original, rutas_generados, ruta_salida="train_final_combinado.csv"):
    """
    Lee un CSV original, convierte sus puntuaciones numéricas a categóricas,
    le añade DataFrames adicionales (generados por IA) y guarda el resultado barajado.
    """
    print(f"Cargando dataset original: {ruta_original}...")
    df_original = pd.read_csv(ruta_original)

    # 1. Función para convertir la columna 'score'
    def convertir_score(score):
        score_str = str(score).strip()
        if score_str in ['1', '2']:
            return "NEGATIVO"
        elif score_str == '3':
            return "NEUTRO"
        elif score_str in ['4', '5']:
            return "POSITIVO"
        return score_str  # Por si algún valor ya viene como texto (ej. "NEGATIVO")

    # Aplicar la conversión
    df_original['score'] = df_original['score'].apply(convertir_score)
    print("Columna 'score' convertida a valores categóricos.")

    # 2. Cargar los datasets generados por la IA
    lista_dfs = [df_original]
    for ruta in rutas_generados:
        try:
            df_gen = pd.read_csv(ruta)
            lista_dfs.append(df_gen)
            print(f"  Añadido dataset IA: {ruta} ({len(df_gen)} filas)")
        except Exception as e:
            print(f"  [!] Error cargando {ruta}: {e}")

    # 3. Unir todos los DataFrames
    # pd.concat rellenará con NaN (nulos) las columnas como 'gender' o 'location'
    # que no existen en los CSVs de la IA, lo cual es correcto.
    df_final = pd.concat(lista_dfs, ignore_index=True)

    # 4. Desordenar (Shuffle) para evitar sesgos en el entrenamiento
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Guardar el CSV resultante
    df_final.to_csv(ruta_salida, index=False, encoding='utf-8')

    print("\n--- RESUMEN ---")
    print(f"Dataset combinado guardado en: '{ruta_salida}'")
    print(f"Total de filas: {len(df_final)}")
    print("\nDistribución final de clases:")
    print(df_final['score'].value_counts())

    return df_final


# ==========================================
# EJEMPLO DE USO CON TUS ARCHIVOS
# ==========================================
if __name__ == "__main__":
    archivo_original = "../../../ficheros_csv/Instagram_train.csv"
    archivos_ia = [
        "NEGATIVAS-gemma4-FINAL.csv",
        "NEGATIVAS-llama-FINAL.csv",
        "NEUTRAS-gemma4-FINAL.csv",
        "NEUTRAS-llama-FINAL.csv"
    ]

    df = preparar_y_unir_datasets(
        ruta_original=archivo_original,
        rutas_generados=archivos_ia,
        ruta_salida="Instagram_train_combinado_IA.csv"
    )