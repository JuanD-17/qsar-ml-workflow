# ESTE SCRIPT REALIZA UN META-ANÁLISIS PARA DETERMINAR LA CATEGORÍA DE POTENCIA DE MOLÉCULAS:
# 1. CARGA CONSOLIDADA: Carga Cannabaceae_molecules.csv y las predicciones de TODOS los modelos de los análisis anteriores (Bajo y Medio), asegurando una fila única por molécula.
# 2. CÁLCULO DE DESCRIPTORES: Calcula descriptores químicos clave para cada molécula.
# 3. LÓGICA DE CLASIFICACIÓN HÍBRIDA:
#    A. INTENTO DE CONSENSO: Se realiza una votación ponderada entre todas las categorías de modelos ("Bajo", "Medio" y "Alto"). La ponderación se basa en la precisión de cada modelo (inverso de la varianza (1/sigma^2, con sigma estimada desde CI95)). Se calcula una confianza de votación.
#    B. DESEMPATE POR EXPERTO (FALLBACK): Si la confianza de la votación de consenso es baja (inferior a un umbral del 66.6%), el sistema descarta el consenso. En su lugar, identifica al único modelo "experto" (el que tiene el CI más estrecho de todos) y asigna la categoría basándose únicamente en la decisión de ese modelo.
# 4. CÁLCULO DE MÉTRICAS GLOBALES:
#    - Weighted_Prediction_Log10_nM: La predicción numérica final, calculada como el promedio ponderado de TODOS los modelos.
#    - Consensus_Level: Mide la dispersión entre las predicciones de todos los modelos.
#    - Category_Confidence: La confianza calculada en la votación de consenso.
#    - Classification_Method: Indica si la categoría final se decidió por "Consensus" o por "Best_Model_Fallback".
# 5. SALIDA: Guarda un único archivo CSV con la categoría final y todas las métricas de diagnóstico para cada molécula.

import os
import pandas as pd
import numpy as np
from scipy import stats
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
import logging


def _xlsx_escape(value):
    return (str(value)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))


def _xlsx_col_name(index):
    name = ''
    while index > 0:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def write_simple_xlsx(df, output_path, sheet_name='Predictions'):
    from datetime import datetime, timezone
    from zipfile import ZipFile, ZIP_DEFLATED

    rows = [list(df.columns)] + df.astype(object).where(df.notna(), '').values.tolist()
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            cell_ref = f"{_xlsx_col_name(col_idx)}{row_idx}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
            else:
                text = _xlsx_escape(value)
                cells.append(f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        '</worksheet>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{_xlsx_escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '</Relationships>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        '</Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        '</Types>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>'
        '</cp:coreProperties>'
    )
    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>Microsoft Excel</Application>'
        '</Properties>'
    )
    with ZipFile(output_path, 'w', ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', content_types)
        zf.writestr('_rels/.rels', root_rels)
        zf.writestr('docProps/core.xml', core_xml)
        zf.writestr('docProps/app.xml', app_xml)
        zf.writestr('xl/workbook.xml', workbook_xml)
        zf.writestr('xl/_rels/workbook.xml.rels', workbook_rels)
        zf.writestr('xl/styles.xml', styles_xml)
        zf.writestr('xl/worksheets/sheet1.xml', sheet_xml)

# --- CONFIGURACIÓN ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)

output_dir = "ensayo334"
os.makedirs(output_dir, exist_ok=True)

# --- DEFINICIONES DE ENTRADA (TODOS LOS MODELOS) ---
# Se mapea cada modelo a su categoría principal ("Bajo", "Medio" o "Alto")
MODEL_MAPPING = {
    # Modelos de Categoría BAJO
    "Bajo_1": ("ensayo177", "cannabaceae_predictions1.csv", "Bajo"),
    "Bajo_2": ("ensayo178", "cannabaceae_predictions2.csv", "Bajo"),
    "Bajo_3": ("ensayo179", "cannabaceae_predictions3.csv", "Bajo"),
    # Modelos de Categoría MEDIO
    "Medio_1": ("ensayo180", "cannabaceae_predictions4.csv", "Medio"),
    "Medio_2": ("ensayo181", "cannabaceae_predictions5.csv", "Medio"),
    "Medio_3": ("ensayo182", "cannabaceae_predictions6.csv", "Medio"),
    "Medio_4": ("ensayo183", "cannabaceae_predictions7.csv", "Medio"),
    # Modelos de Categoría ALTO (integrados ensayo184)
    "Alto_1": ("ensayo184", "cannabaceae_predictions8.csv", "Alto"),
}

# --- FUNCIONES ---
def calculate_descriptors_batch(smiles_batch):
    results = []
    descriptor_functions = [
        ('MolLogP', Crippen.MolLogP),
        ('TPSA', rdMolDescriptors.CalcTPSA),
        ('MolWt', Descriptors.MolWt)
    ]
    for smiles in smiles_batch:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: raise ValueError("Invalid SMILES")
            results.append({name: func(mol) for name, func in descriptor_functions})
        except Exception as e:
            logging.warning(f"Descriptor calculation failed for SMILES {smiles}: {e}")
            results.append(None)
    return results

# --- CARGA DE DATOS ---
logging.info("Cargando y deduplicando moléculas base...")
try:
    df_base = pd.read_csv('Cannabaceae_molecules.csv', sep=';')
    df_base = df_base.drop_duplicates(subset=['Smiles']).reset_index(drop=True)
    df_base.columns = df_base.columns.str.strip()
except Exception as e:
    logging.error(f"Error al cargar Cannabaceae_molecules.csv: {e}")
    exit()

logging.info("Cargando y combinando todas las predicciones de modelos...")
merged_df = df_base[['Smiles', 'Molecule_name']].copy()
merged_df.columns = ['SMILES', 'Molecule_name']

model_names = []
for model_name, (in_dir, pred_file, category) in MODEL_MAPPING.items():
    file_path = os.path.join(in_dir, pred_file)
    try:
        df = pd.read_csv(file_path, sep=';')
        df.columns = df.columns.str.strip()
        df = df.drop_duplicates(subset=['SMILES'], keep='first')
        
        # Renombrar columnas para evitar colisiones
        rename_cols = {
            'Predicted_Log10_Value_nM': f'Pred_{model_name}',
            'Lower_CI_95': f'Lower_{model_name}',
            'Upper_CI_95': f'Upper_{model_name}',
            'CI_95_Width': f'Width_{model_name}'
        }
        df = df[['SMILES'] + list(rename_cols.keys())].rename(columns=rename_cols)
        
        merged_df = pd.merge(merged_df, df, on='SMILES', how='inner')
        model_names.append(model_name)
        logging.info(f"Cargado y unido: {model_name} desde {file_path}")
    except Exception as e:
        logging.error(f"Fallo al procesar {file_path}: {e}")
        # No salir, solo registrar y continuar, para maximizar las moléculas procesadas
        continue

logging.info(f"DataFrame combinado con {len(merged_df)} moléculas presentes en todos los modelos.")

# --- CÁLCULO DE DESCRIPTORES ---
logging.info("Calculando descriptores químicos...")
descriptors = calculate_descriptors_batch(merged_df["SMILES"].tolist())
descriptor_df = pd.DataFrame([d for d in descriptors if d is not None])
valid_indices = [i for i, d in enumerate(descriptors) if d is not None]
merged_df = merged_df.iloc[valid_indices].reset_index(drop=True)
merged_df = pd.concat([merged_df, descriptor_df], axis=1)

# --- LÓGICA DE CLASIFICACIÓN HÍBRIDA (VECTORIZADA) ---
logging.info("Iniciando clasificación híbrida...")

# 1. Extraer datos en arrays de NumPy para eficiencia
pred_cols = [f'Pred_{m}' for m in model_names]
width_cols = [f'Width_{m}' for m in model_names]
predictions = merged_df[pred_cols].values
ci_widths = merged_df[width_cols].values

# 2. Calcular pesos y normalizarlos
sigma = (ci_widths / 3.92) + 1e-12
weights = 1 / (sigma ** 2)
normalized_weights = weights / weights.sum(axis=1, keepdims=True)

# 3. Votación de Consenso
logging.info("Realizando votación de consenso entre Bajo, Medio y Alto...")
category_votes = {}
for category in ["Bajo", "Medio", "Alto"]:
    # Sumar los pesos de los modelos que pertenecen a esta categoría
    cat_model_indices = [i for i, name in enumerate(model_names) if MODEL_MAPPING[name][2] == category]
    if not cat_model_indices:
        category_votes[category] = np.zeros(len(merged_df))
        continue
    category_votes[category] = weights[:, cat_model_indices].sum(axis=1)

vote_df = pd.DataFrame(category_votes)
total_votes = vote_df.sum(axis=1)
# Evitar división por cero si todos los pesos son cero para una fila
with np.errstate(divide='ignore', invalid='ignore'):
    consensus_winner = vote_df.idxmax(axis=1)
    category_confidence = vote_df.max(axis=1) / total_votes
    category_confidence.fillna(0, inplace=True)

# 4. Fallback: Identificar el mejor modelo individual para cada molécula
logging.info("Identificando el mejor modelo individual (fallback)...")
best_model_indices = np.argmin(ci_widths, axis=1)
best_model_names = [model_names[i] for i in best_model_indices]
fallback_winner = [MODEL_MAPPING[name][2] for name in best_model_names]

# 5. Aplicar la lógica híbrida
logging.info("Aplicando lógica híbrida para decisión final...")
confidence_threshold = 2/3  # Umbral de confianza (66.6%)
use_consensus = category_confidence >= confidence_threshold

merged_df['Final_Category'] = np.where(use_consensus, consensus_winner, fallback_winner)
merged_df['Classification_Method'] = np.where(use_consensus, 'Consensus', 'Best_Model_Fallback')
merged_df['Category_Confidence'] = category_confidence

# --- CÁLCULO DE MÉTRICAS FINALES Y CI ---
logging.info("Calculando métricas finales y el intervalo de confianza correspondiente...")

# a. Predicción ponderada global y Nivel de Consenso (se calculan para todas las filas)
merged_df['Weighted_Prediction_Log10_nM'] = np.sum(predictions * normalized_weights, axis=1)
weighted_avg = merged_df['Weighted_Prediction_Log10_nM'].values[:, np.newaxis]
weighted_variance = np.sum(normalized_weights * (predictions - weighted_avg)**2, axis=1)
merged_df['Consensus_Level'] = np.sqrt(weighted_variance)

# b. Calcular siempre el CI final a partir de la varianza combinada del
# meta-estimador ponderado. Esto mantiene consistencia entre el estimador
# central, sus pesos de precisión y su incertidumbre final.
combined_variance = 1 / np.sum(weights, axis=1)
combined_sigma = np.sqrt(combined_variance)
weighted_pred = merged_df['Weighted_Prediction_Log10_nM'].values
weighted_lower_ci = weighted_pred - 1.96 * combined_sigma
weighted_upper_ci = weighted_pred + 1.96 * combined_sigma
weighted_ci_width = weighted_upper_ci - weighted_lower_ci

merged_df['Final_Lower_CI_95'] = weighted_lower_ci
merged_df['Final_Upper_CI_95'] = weighted_upper_ci
merged_df['Final_CI_Width'] = weighted_ci_width

# --- PREPARAR SALIDA ---
logging.info("Preparando archivo de salida final...")
output_cols = [
    "SMILES", "Molecule_name", "Final_Category", "Classification_Method",
    "Category_Confidence", "Consensus_Level", "Weighted_Prediction_Log10_nM",
    "Final_Lower_CI_95", "Final_Upper_CI_95", "Final_CI_Width"
]
output_df = merged_df[output_cols]

output_path = os.path.join(output_dir, "Cannabaceae_consolidated_predictions.csv")
output_df.to_csv(output_path, sep=';', index=False, float_format='%.6f')

output_xlsx_path = os.path.join(output_dir, "Cannabaceae_consolidated_predictions.xlsx")
write_simple_xlsx(output_df, output_xlsx_path)

logging.info(f"Resultados consolidados guardados en {output_path}")

# --- RESUMEN FINAL ---
logging.info("\nResumen del Meta-Análisis:")
logging.info(f"Moléculas procesadas: {len(output_df)}")
logging.info(f"Distribución de Categorías Finales:\n{output_df['Final_Category'].value_counts().to_string()}")
logging.info(f"Métodos de Clasificación Utilizados:\n{output_df['Classification_Method'].value_counts().to_string()}")
logging.info(f"Confianza de Categoría Promedio: {output_df['Category_Confidence'].mean():.3f}")
logging.info(f"Nivel de Consenso Promedio: {output_df['Consensus_Level'].mean():.3f}")
logging.info("Proceso completado.")
