# =============================================================================
# DESCRIPCIÓN GENERAL DEL SCRIPT:
# 
# Este script realiza las siguientes funciones:
# 1. Lee un archivo CSV de entrada que contiene IDs de ensayos (assay_chembl_id)
# 2. Para cada ID de ensayo, consulta la API de ChEMBL para obtener datos de actividad
# 3. Para cada molécula encontrada en los datos de actividad, consulta información adicional
# 4. Combina los datos de actividad y molécula en un conjunto completo de información
# 5. Guarda todos los resultados en un archivo CSV de salida
# 
# El script maneja tanto respuestas en formato JSON como XML de la API, implementa
# un sistema de caché para evitar consultas repetidas sobre la misma molécula,
# y cuenta con mecanismos de reintento en caso de fallos de conexión.
# =============================================================================

import pandas as pd
import requests
import csv
import time
import json
import os
import xml.etree.ElementTree as ET

def fetch_chembl_data(input_file: str, output_file: str) -> None:
    """
    Obtiene datos combinados de actividad y moléculas desde la API de ChEMBL
    para IDs de ensayos (assay_chembl_id) dados y los guarda en CSV.
    Soporta respuestas en formato XML o JSON.
    """
    try:
        # Verificar si el archivo existe
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"El archivo de entrada {input_file} no existe")

        # Lee el archivo CSV de entrada
        try:
            df = pd.read_csv(input_file, sep=";", dtype=str, low_memory=False)
        except Exception as e:
            print(f"Error al leer el archivo CSV: {str(e)}")
            raise

        # Verifica si existe la columna requerida (puede ser 'aidextid' o 'assay_chembl_id')
        id_column = None
        for col in ['aidextid', 'assay_chembl_id']:
            if col in df.columns:
                id_column = col
                break

        if id_column is None:
            raise ValueError("Columna 'aidextid' o 'assay_chembl_id' no encontrada en el archivo de entrada")

        # Limpia espacios en blanco y elimina valores vacíos/nulos
        df[id_column] = df[id_column].str.strip()
        df = df[df[id_column].notna() & (df[id_column] != "")]

        # Define los campos a extraer
        field_names = [
            "molecule_chembl_id", "molecule_pref_name", "max_phase", "full_mwt",
            "num_ro5_violations", "alogp", "compound_key", "canonical_smiles", "standard_type",
            "standard_relation", "standard_value", "standard_units", "pchembl_value",
            "data_validity_comment", "activity_comment", "uo_units", "ligand_efficiency",
            "ligand_efficiency_bei", "ligand_efficiency_le", "ligand_efficiency_lle",
            "ligand_efficiency_sei", "potential_duplicate", "assay_chembl_id", "assay_description",
            "assay_type", "bao_format", "bao_label", "assay_organism", "assay_tissue_chembl_id",
            "assay_tissue_name", "assay_cell_type", "assay_subcellular_fraction",
            "assay_parameters", "assay_variant_accession", "assay_variant_mutation",
            "target_chembl_id", "target_pref_name", "target_organism", "target_type",
            "document_chembl_id", "src_id", "src_description", "document_journal",
            "document_year", "cell_chembl_id", "properties", "action_type",
            "standard_text_value", "value"
        ]

        # Crea el archivo CSV de salida con encabezados
        with open(output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(field_names)

        # Obtiene IDs de ensayo únicos
        unique_ids = df[id_column].unique()
        print(f"Procesando {len(unique_ids)} IDs de ensayo únicos...")

        # Diccionario para almacenar información de moléculas (caché)
        molecule_cache = {}

        # Contador para seguimiento
        processed_count = 0
        total_activities = 0

        # Recorre cada ID de ensayo único
        for assay_id in unique_ids:
            processed_count += 1
            print(f"[{processed_count}/{len(unique_ids)}] Procesando ID de ensayo: {assay_id}")

            # Construye la URL de la API
            url = f"https://www.ebi.ac.uk/chembl/api/data/activity?assay_chembl_id={assay_id}"
            
            # Intentar con formato JSON explícito
            headers_json = {'Accept': 'application/json'}
            
            try:
                # Realizar la petición HTTP con reintentos
                max_retries = 3
                retry_count = 0
                response = None

                while retry_count < max_retries:
                    try:
                        # Primero intentamos con JSON explícito
                        response = requests.get(url, headers=headers_json, timeout=15)
                        break  # Si la petición es exitosa, salimos del bucle
                    except requests.exceptions.RequestException as e:
                        retry_count += 1
                        print(f" Reintento {retry_count}/{max_retries} para {assay_id}: {str(e)}")
                        time.sleep(5)  # Esperar más tiempo entre reintentos

                if response is None:
                    print(f" ❌ No se pudo conectar a la API después de {max_retries} intentos para {assay_id}")
                    continue

                # Procesar la respuesta
                if response.status_code == 200:
                    # Verificar contenido de la respuesta
                    print(f" Response status: {response.status_code}")
                    print(f" Response content preview: {response.text[:200]}")  # Print first 200 chars
                    
                    # Determinar si la respuesta es XML o JSON
                    is_xml = response.text.strip().startswith('<?xml')
                    
                    activities = []
                    
                    if is_xml:
                        print(" Procesando respuesta en formato XML")
                        try:
                            # Parsear XML
                            root = ET.fromstring(response.text)
                            # En XML, las actividades estarán en <activities><activity>
                            xml_activities = root.findall('.//activity')
                            
                            # Convertir los elementos XML a diccionarios
                            for xml_activity in xml_activities:
                                activity_dict = {}
                                for elem in xml_activity:
                                    # Guardar todos los elementos como pares clave-valor
                                    activity_dict[elem.tag] = elem.text if elem.text else ""
                                activities.append(activity_dict)
                                
                            print(f" ✅ Encontradas {len(activities)} actividades para {assay_id} (XML)")
                                
                        except ET.ParseError as e:
                            print(f" ❌ Error al analizar XML para {assay_id}: {str(e)}")
                            continue
                    else:
                        # Intentar procesar como JSON
                        try:
                            response_data = response.json()
                            # Verificar si hay actividades en la respuesta
                            if "activities" in response_data:
                                activities = response_data["activities"]
                                print(f" ✅ Encontradas {len(activities)} actividades para {assay_id} (JSON)")
                            else:
                                print(f" ⚠️ No se encontraron actividades para {assay_id} en formato JSON")
                        except json.JSONDecodeError as e:
                            print(f" ❌ Error al decodificar JSON para {assay_id}: {str(e)}")
                            print(f" Contenido de respuesta (primeros 100 caracteres): {response.text[:100]}")
                            continue  # Pasar al siguiente ID
                    
                    # Procesar cada actividad (común para XML o JSON)
                    total_activities += len(activities)
                    activity_count = 0
                    
                    for activity in activities:
                        activity_count += 1
                        if activity_count % 10 == 0:
                            print(f" Procesando actividad {activity_count}/{len(activities)}")

                        # Obtener el ID de la molécula
                        molecule_id = activity.get("molecule_chembl_id")
                        if not molecule_id:
                            print(" ⚠️ Actividad sin molecule_chembl_id, ignorando")
                            continue

                        # Si la molécula no está en caché, obtener información adicional
                        if molecule_id not in molecule_cache:
                            molecule_data = {}
                            try:
                                # Consultar la API de moléculas
                                molecule_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{molecule_id}"

                                # Reintentos para la solicitud de molécula
                                mol_retry_count = 0
                                molecule_response = None

                                while mol_retry_count < max_retries:
                                    try:
                                        # Intentar con JSON explícito
                                        molecule_response = requests.get(molecule_url, headers=headers_json, timeout=15)
                                        break
                                    except requests.exceptions.RequestException as e:
                                        mol_retry_count += 1
                                        print(f" Reintento {mol_retry_count}/{max_retries} para molécula {molecule_id}: {str(e)}")
                                        time.sleep(3)

                                if molecule_response is None:
                                    print(f" ⚠️ No se pudo obtener datos para molécula {molecule_id} después de {max_retries} intentos")
                                    molecule_cache[molecule_id] = {}
                                    continue

                                if molecule_response.status_code == 200:
                                    # Determinar si la respuesta es XML o JSON
                                    mol_is_xml = molecule_response.text.strip().startswith('<?xml')
                                    
                                    if mol_is_xml:
                                        print(f" Procesando molécula {molecule_id} en formato XML")
                                        try:
                                            # Parsear XML
                                            mol_root = ET.fromstring(molecule_response.text)
                                            # Construir un diccionario equivalente para mantener compatibilidad
                                            molecule_data = {"molecule_properties": {}, "molecule_structures": {}}
                                            
                                            # Extraer propiedades básicas
                                            for elem in mol_root.findall('.//molecule'):
                                                for child in elem:
                                                    if child.tag not in ["molecule_properties", "molecule_structures"]:
                                                        molecule_data[child.tag] = child.text if child.text else ""
                                            
                                            # Extraer propiedades específicas
                                            properties = mol_root.find('.//molecule_properties')
                                            if properties is not None:
                                                for prop in properties:
                                                    molecule_data["molecule_properties"][prop.tag] = prop.text if prop.text else ""
                                                    
                                            # Extraer estructuras
                                            structures = mol_root.find('.//molecule_structures')
                                            if structures is not None:
                                                for struct in structures:
                                                    molecule_data["molecule_structures"][struct.tag] = struct.text if struct.text else ""
                                                
                                        except ET.ParseError as e:
                                            print(f" ❌ Error al analizar XML para molécula {molecule_id}: {str(e)}")
                                            molecule_data = {}
                                    else:
                                        # Intentar procesar como JSON
                                        try:
                                            molecule_data = molecule_response.json()
                                        except json.JSONDecodeError as e:
                                            print(f" ❌ Error al decodificar JSON para molécula {molecule_id}: {str(e)}")
                                            print(f" Contenido de respuesta (primeros 100 caracteres): {molecule_response.text[:100]}")
                                            molecule_data = {}
                                    
                                    molecule_cache[molecule_id] = molecule_data
                                else:
                                    print(f" ⚠️ Error HTTP {molecule_response.status_code} para molécula {molecule_id}")
                                    molecule_cache[molecule_id] = {}
                            except Exception as e:
                                print(f" ⚠️ Error al procesar molécula {molecule_id}: {str(e)}")
                                molecule_cache[molecule_id] = {}

                            # Pausa breve para evitar sobrecargar la API
                            time.sleep(1)

                        # Obtener datos de molécula del caché
                        molecule_data = molecule_cache.get(molecule_id, {})

                        # Extraer propiedades específicas
                        molecule_properties = molecule_data.get("molecule_properties", {})
                        molecule_structures = molecule_data.get("molecule_structures", {})

                        # Función para obtener valor de un diccionario (funciona igual para XML o JSON)
                        def get_value(dict_obj, key, default=""):
                            if isinstance(dict_obj, dict):
                                return dict_obj.get(key, default)
                            return default

                        # Preparar datos combinados
                        combined_data = {
                            "molecule_chembl_id": molecule_id,
                            "molecule_pref_name": get_value(activity, "molecule_pref_name") or get_value(molecule_data, "pref_name", ""),
                            "max_phase": get_value(molecule_data, "max_phase", ""),
                            "full_mwt": get_value(molecule_properties, "full_mwt", ""),
                            "num_ro5_violations": get_value(molecule_properties, "num_ro5_violations", ""),
                            "alogp": get_value(molecule_properties, "alogp", ""),
                            "compound_key": "",  # No disponible directamente
                            "canonical_smiles": get_value(activity, "canonical_smiles") or get_value(molecule_structures, "canonical_smiles", ""),
                            "standard_type": get_value(activity, "standard_type", ""),
                            "standard_relation": get_value(activity, "standard_relation", ""),
                            "standard_value": get_value(activity, "standard_value", ""),
                            "standard_units": get_value(activity, "standard_units", ""),
                            "pchembl_value": get_value(activity, "pchembl_value", ""),
                            "data_validity_comment": get_value(activity, "data_validity_comment", ""),
                            "activity_comment": get_value(activity, "activity_comment", ""),
                            "uo_units": get_value(activity, "uo_units", ""),
                            "ligand_efficiency": get_value(activity, "ligand_efficiency", ""),
                            "ligand_efficiency_bei": "",  # No disponible directamente
                            "ligand_efficiency_le": "",  # No disponible directamente
                            "ligand_efficiency_lle": "",  # No disponible directamente
                            "ligand_efficiency_sei": "",  # No disponible directamente
                            "potential_duplicate": get_value(activity, "potential_duplicate", ""),
                            "assay_chembl_id": get_value(activity, "assay_chembl_id", ""),
                            "assay_description": get_value(activity, "assay_description", ""),
                            "assay_type": get_value(activity, "assay_type", ""),
                            "bao_format": get_value(activity, "bao_format", ""),
                            "bao_label": get_value(activity, "bao_label", ""),
                            "assay_organism": "",  # No disponible directamente
                            "assay_tissue_chembl_id": "",  # No disponible directamente
                            "assay_tissue_name": "",  # No disponible directamente
                            "assay_cell_type": "",  # No disponible directamente
                            "assay_subcellular_fraction": "",  # No disponible directamente
                            "assay_parameters": "",  # No disponible directamente
                            "assay_variant_accession": get_value(activity, "assay_variant_accession", ""),
                            "assay_variant_mutation": get_value(activity, "assay_variant_mutation", ""),
                            "target_chembl_id": get_value(activity, "target_chembl_id", ""),
                            "target_pref_name": get_value(activity, "target_pref_name", ""),
                            "target_organism": get_value(activity, "target_organism", ""),
                            "target_type": "",  # No disponible directamente
                            "document_chembl_id": get_value(activity, "document_chembl_id", ""),
                            "src_id": get_value(activity, "src_id", ""),
                            "src_description": "",  # No disponible directamente
                            "document_journal": get_value(activity, "document_journal", ""),
                            "document_year": get_value(activity, "document_year", ""),
                            "cell_chembl_id": "",  # No disponible directamente
                            "properties": "",  # No disponible directamente
                            "action_type": get_value(activity, "action_type", ""),
                            "standard_text_value": get_value(activity, "standard_text_value", ""),
                            "value": get_value(activity, "value", "")
                        }

                        # Escribir al archivo CSV
                        with open(output_file, mode="a", newline="", encoding="utf-8") as file:
                            writer = csv.writer(file)
                            # Extraer valores en el orden especificado
                            row_values = [combined_data.get(field, "") for field in field_names]
                            writer.writerow(row_values)
                else:
                    print(f" ❌ Error HTTP {response.status_code} para {assay_id}")
                    print(f" Respuesta: {response.text[:200]}")  # Mostrar respuesta para depuración
            except Exception as e:
                print(f" ❌ Error general para {assay_id}: {str(e)}")

            # Pausa entre solicitudes para evitar sobrecargar la API
            time.sleep(5)  # Aumentado a 5 segundos para evitar limitaciones de tasa

        print(f"\n✅ Procesamiento completado: {processed_count} IDs de ensayo, {total_activities} actividades totales")
        print(f"✅ Archivo de salida guardado en: {output_file}")
    except Exception as e:
        print(f"❌ Error crítico en el procesamiento: {str(e)}")
        raise

# Punto de entrada principal
if __name__ == "__main__":
    # Archivos de entrada y salida
    input_file = "molecules.csv"  # Contiene los IDs de ensayo
    output_file = "molecule_definitive.csv"  # Archivo de salida

    # Ejecutar el proceso
    fetch_chembl_data(input_file, output_file)