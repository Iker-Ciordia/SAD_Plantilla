import pandas as pd
import os

def generar_subconjuntos():
    """
    Filtra los datasets originales por aplicación, sentimiento y rango de años
    para preparar ficheros listos para el algoritmo de clustering.
    """
    # Creamos la carpeta de salida si no existe
    carpeta_salida = "./clustering/ficheros_divididos"
    os.makedirs(carpeta_salida, exist_ok=True)

    # --- 1. TIKTOK: NEGATIVAS DE 2010 A 2019 ---
    # Ajusta esta ruta a donde tengas realmente tu CSV original de TikTok
    ruta_tiktok = "clasificacion/ficheros_csv/TikTok.csv" 
    
    if os.path.exists(ruta_tiktok):
        df_tiktok = pd.read_csv(ruta_tiktok)
        # Convertimos la columna de fecha a tipo datetime para extraer el año
        df_tiktok['date'] = pd.to_datetime(df_tiktok['date'], errors='coerce')
        
        tiktok_filtrado = df_tiktok[
            (df_tiktok['score'] == 'NEGATIVO') & 
            (df_tiktok['date'].dt.year >= 2010) & 
            (df_tiktok['date'].dt.year <= 2019)
        ]
        
        salida_tiktok = os.path.join(carpeta_salida, "TikTok_NEGATIVO_2010_2019.csv")
        tiktok_filtrado.to_csv(salida_tiktok, index=False)
        print(f"[+] Generado: {salida_tiktok} ({len(tiktok_filtrado)} filas)")
    else:
        print(f"[-] ERROR: No se encontró el archivo {ruta_tiktok}")

    # --- 2. INSTAGRAM: NEGATIVAS DE 2020 A 2025 ---
    # Ajusta esta ruta a donde tengas realmente tu CSV original de Instagram
    ruta_ig = "clasificacion/ficheros_csv/Instagram.csv"
    
    if os.path.exists(ruta_ig):
        df_ig = pd.read_csv(ruta_ig)
        df_ig['date'] = pd.to_datetime(df_ig['date'], errors='coerce')
        
        ig_filtrado = df_ig[
            (df_ig['score'] == 'NEGATIVO') & 
            (df_ig['date'].dt.year >= 2020) & 
            (df_ig['date'].dt.year <= 2025)
        ]
        
        salida_ig = os.path.join(carpeta_salida, "Instagram_NEGATIVO_2020_2025.csv")
        ig_filtrado.to_csv(salida_ig, index=False)
        print(f"[+] Generado: {salida_ig} ({len(ig_filtrado)} filas)")
    else:
        print(f"[-] ERROR: No se encontró el archivo {ruta_ig}")

if __name__ == "__main__":
    generar_subconjuntos()