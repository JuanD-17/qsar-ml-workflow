# =============================================================================
# DESCRIPCIÓN GENERAL DEL SCRIPT:
#
# Este script realiza las siguientes funciones:
# 1. Lee un archivo CSV de entrada (`molecule_definitive.csv`) que contiene datos de ChEMBL.
# 2. Filtra las filas donde la columna `assay_description` contiene las palabras clave
#    "Antioxidant", "antioxidant" o "lipid peroxidation" (sensible a mayúsculas).
# 3. Guarda las filas filtradas en un nuevo archivo CSV (`molecule_filter.csv`).
#
# El script incluye manejo de errores para verificar la existencia del archivo de entrada
# y la presencia de la columna `assay_description`, además de proporcionar retroalimentación
# sobre el número de filas encontradas y el estado del proceso.
# =============================================================================

import pandas as pd
import os

def filter_chembl_data(input_file: str, output_file: str) -> None:
    """
    Filtra datos de un archivo CSV de ChEMBL para mantener solo las filas donde
    assay_description contiene 'Antioxidant', 'antioxidant' o 'lipid peroxidation'
    y guarda los resultados en un nuevo archivo CSV.
    """
    try:
        # Verificar si el archivo de entrada existe
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"El archivo de entrada {input_file} no existe")

        # Leer el archivo CSV de entrada
        try:
            df = pd.read_csv(input_file, dtype=str, low_memory=False)
        except Exception as e:
            print(f"Error al leer el archivo CSV: {str(e)}")
            raise

        # Verificar si la columna requerida 'assay_description' existe
        if 'assay_description' not in df.columns:
            raise ValueError("La columna 'assay_description' no se encontró en el archivo de entrada")

        # Definir las palabras clave para el filtrado
        keywords = ['Antioxidant', 'antioxidant', 'lipid peroxidation']

        # Filtrar filas donde assay_description contiene alguna de las palabras clave
        mask = df['assay_description'].str.contains('|'.join(keywords), case=True, na=False)
        filtered_df = df[mask]

        # Verificar si se encontraron filas
        if filtered_df.empty:
            print("⚠️ No se encontraron filas con las palabras clave especificadas en assay_description")
        else:
            print(f"✅ Se encontraron {len(filtered_df)} filas con las palabras clave especificadas")

        # Guardar los datos filtrados en el archivo CSV de salida
        filtered_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ Datos filtrados guardados en: {output_file}")

    except Exception as e:
        print(f"❌ Error crítico durante el procesamiento: {str(e)}")
        raise

# Punto de entrada principal
if __name__ == "__main__":
    # Archivos de entrada y salida
    input_file = "molecule_definitive.csv"
    output_file = "molecule_filter.csv"

    # Ejecutar el proceso de filtrado
    filter_chembl_data(input_file, output_file)