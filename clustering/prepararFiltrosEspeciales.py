import pandas as pd
import os


def generar_subconjuntos():
    """
    Filtra los datasets originales por aplicación, sentimiento y rango de años
    para preparar ficheros listos para el algoritmo de clustering.
    """
    carpeta_salida = "./clustering/ficheros_divididos"
    os.makedirs(carpeta_salida, exist_ok=True)

    # --- 1. TIKTOK: NEGATIVAS DE 2010 A 2019 ---
    ruta_tiktok = "clustering/ficheros_divididos/TikTok_NEGATIVO.csv"

    if os.path.exists(ruta_tiktok):
        df_tiktok = pd.read_csv(ruta_tiktok)

        # Calculamos los años al vuelo (aislado del dataframe) con dayfirst=True
        años_tiktok = pd.to_datetime(df_tiktok['date'], errors='coerce', dayfirst=True).dt.year

        # Filtramos directamente cruzando con esos años calculados
        tiktok_filtrado = df_tiktok[
            (df_tiktok['score'] == 'NEGATIVO') &
            (años_tiktok >= 2016) &
            (años_tiktok <= 2019)
            ]

        salida_tiktok = os.path.join(carpeta_salida, "TikTok_NEGATIVO_2016_2019.csv")
        tiktok_filtrado.to_csv(salida_tiktok, index=False)
        print(f"[+] Generado: {salida_tiktok} ({len(tiktok_filtrado)} filas)")
    else:
        print(f"[-] ERROR: No se encontró el archivo {ruta_tiktok}")

    # --- 2. INSTAGRAM: NEGATIVAS DE 2020 A 2025 ---
    ruta_ig = "clustering/ficheros_divididos/Instagram_NEGATIVO.csv"

    if os.path.exists(ruta_ig):
        df_ig = pd.read_csv(ruta_ig)

        # Lo mismo para Instagram
        años_ig = pd.to_datetime(df_ig['date'], errors='coerce', dayfirst=True).dt.year

        ig_filtrado = df_ig[
            (df_ig['score'] == 'NEGATIVO') &
            (años_ig >= 2020) &
            (años_ig <= 2025)
            ]

        salida_ig = os.path.join(carpeta_salida, "Instagram_NEGATIVO_2020_2025.csv")
        ig_filtrado.to_csv(salida_ig, index=False)
        print(f"[+] Generado: {salida_ig} ({len(ig_filtrado)} filas)")
    else:
        print(f"[-] ERROR: No se encontró el archivo {ruta_ig}")


if __name__ == "__main__":
    generar_subconjuntos()