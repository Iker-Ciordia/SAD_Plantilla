import sys
import subprocess
import os

if __name__ == "__main__":
    # Comprobamos los parámetros del lanzador
    if len(sys.argv) < 7:
        print("Uso: python lanzador.py <fichero> <columna_objetivo> <k_min> <k_max> <p_min> <p_max>")
        sys.exit(1)

    fichero = sys.argv[1]
    columna_objetivo = sys.argv[2]
    k_min = int(sys.argv[3])
    k_max = int(sys.argv[4])
    p_min = int(sys.argv[5])
    p_max = int(sys.argv[6])

    pesos_lista = ['uniform', 'distance']

    # Borramos el CSV antiguo si existe para empezar limpios
    if os.path.exists('resultados.csv'):
        os.remove('resultados.csv')

    print("Iniciando barrido de hiperparámetros...")

    # Bucle para generar las combinaciones
    # range(k_min, k_max + 1, 2) avanza de 2 en 2 (ej: 1, 3, 5) para que k sea impar
    for k in range(k_min, k_max + 1, 2):
        for p in range(p_min, p_max + 1):
            for pesos in pesos_lista:
                print("\n")
                print(f"--> Ejecutando modelo con: k={k}, p={p}, weights={pesos}")

                # Preparamos el comando de consola como si lo escribiera a mano
                comando = ["python", "kNN.py", fichero, columna_objetivo, str(k), pesos, str(p)]

                # Ejecutamos kNN.py y esperamos a que termine
                subprocess.run(comando)

    #print("\nBarrido completado Abre el archivo 'resultados.csv' para ver qué modelo ganó.")