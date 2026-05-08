# Este script realiza un análisis de predicción de propiedades químicas (Log10 Value en nM, probablemente una medida de actividad antioxidante como IC50 o EC50) para moléculas de la familia Cannabaceae, utilizando un modelo de Random Forest previamente entrenado en el ensayo112 con selección de características Boruta, Mutual Information, y descriptores con andamios de Murcko. El flujo de trabajo incluye:
# 1. Cargar un archivo CSV con SMILES y nombres de moléculas (Molecule_name) de Cannabaceae, usando punto y coma (;) como separador.
# 2. Verificar que las columnas 'Smiles' y 'Molecule_name' existan, manejando espacios en blanco, y validar las SMILES.
# 3. Calcular descriptores químicos y fingerprints Morgan (radius=3, fpSize=512) consistentes con el ensayo112, incluyendo descriptores de andamios de Murcko.
# 4. Preprocesar los descriptores (imputación y escalado) utilizando los objetos guardados en ensayo112_fixed.
# 5. Realizar predicciones con intervalos de confianza del 95% usando el modelo Random Forest entrenado.
# 6. Guardar los resultados en un archivo CSV con las columnas SMILES, Molecule_name, predicciones, límites del intervalo de confianza, y amplitud del intervalo.
# 7. Registrar estadísticas resumidas.
# Modificación: Se asegura la compatibilidad con el modelo y descriptores del ensayo112. Se elimina la clasificación de potencia (Potency_Category). Se fuerza el uso de puntos como separador decimal al guardar el CSV para evitar problemas de formato regional.
# Categoría: Medio - Valores altos - potencia
# El script está diseñado para ejecutarse en un entorno virtual de Python con las librerías necesarias instaladas (pandas, numpy, rdkit, sklearn, scipy, etc.).

import pandas as pd
import numpy as np
import os
import logging
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from scipy import stats

# Configura el sistema de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)

# Define directorios de entrada y salida
input_dir = "ensayo112_fixed"
output_dir = "ensayo202"
os.makedirs(output_dir, exist_ok=True)

# Función para contar enlaces dobles, excluyendo los aromáticos
def count_double_bonds(mol):
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE and not bond.GetIsAromatic())

# Función para validar SMILES
def validate_smiles(smiles_list):
    valid_smiles = []
    valid_idx = []
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and mol.GetNumAtoms() > 0:
                valid_smiles.append(smiles)
                valid_idx.append(i)
            else:
                logging.warning(f"Invalid SMILES (None or 0 atoms): {smiles}")
        except Exception as e:
            logging.warning(f"Error validating SMILES {smiles}: {e}")
    return valid_smiles, valid_idx

# Función para calcular descriptores químicos y fingerprints en lotes
def calculate_descriptors_batch(smiles_batch):
    results = []
    morgan_gen = AllChem.GetMorganGenerator(radius=3, fpSize=512)
    
    descriptor_functions = [
        ('MolLogP', lambda mol: Crippen.MolLogP(mol)),
        ('TPSA', lambda mol: rdMolDescriptors.CalcTPSA(mol)),
        ('MolWt', lambda mol: Descriptors.MolWt(mol)),
        ('HeavyAtomCount', lambda mol: Descriptors.HeavyAtomCount(mol)),
        ('RingCount', lambda mol: rdMolDescriptors.CalcNumRings(mol)),
        ('AromaticRingCount', lambda mol: rdMolDescriptors.CalcNumAromaticRings(mol)),
        ('HBondDonors', lambda mol: Lipinski.NumHDonors(mol)),
        ('HBondAcceptors', lambda mol: Lipinski.NumHAcceptors(mol)),
        ('NumRotatableBonds', lambda mol: rdMolDescriptors.CalcNumRotatableBonds(mol)),
        ('FractionCSP3', lambda mol: Descriptors.FractionCSP3(mol))
    ]
    
    for smiles in smiles_batch:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                logging.warning(f"Invalid SMILES in descriptor calculation: {smiles}")
                results.append(None)
                continue
            
            fp = morgan_gen.GetFingerprint(mol)
            fp_arr = np.array(list(fp))
            
            descriptors_dict = {}
            for name, func in descriptor_functions:
                try:
                    descriptors_dict[name] = func(mol)
                except Exception as e:
                    logging.warning(f"Error in descriptor {name} for {smiles}: {e}")
                    descriptors_dict[name] = 0.0
            
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None or scaffold.GetNumAtoms() == 0:
                scaffold_fp = np.zeros(512)
                scaffold_descriptors = {
                    'Scaffold_MolWt': 0.0,
                    'Scaffold_HeavyAtomCount': 0,
                    'Scaffold_RingCount': 0,
                    'Scaffold_AromaticRingCount': 0,
                    'Scaffold_TPSA': 0.0
                }
            else:
                scaffold_fp = morgan_gen.GetFingerprint(scaffold)
                scaffold_fp = np.array(list(scaffold_fp))
                scaffold_descriptors = {
                    'Scaffold_MolWt': Descriptors.MolWt(scaffold),
                    'Scaffold_HeavyAtomCount': Descriptors.HeavyAtomCount(scaffold),
                    'Scaffold_RingCount': rdMolDescriptors.CalcNumRings(scaffold),
                    'Scaffold_AromaticRingCount': rdMolDescriptors.CalcNumAromaticRings(scaffold),
                    'Scaffold_TPSA': rdMolDescriptors.CalcTPSA(scaffold)
                }
                
            descriptors_dict.update(scaffold_descriptors)
            final_descriptors = np.concatenate([fp_arr, np.array([value for value in descriptors_dict.values()]), scaffold_fp])
            results.append(np.nan_to_num(final_descriptors, nan=0.0, posinf=0.0, neginf=0.0))
        except Exception as e:
            logging.warning(f"Error in descriptor calculation for SMILES {smiles}: {e}")
            results.append(None)
    return results

# Función para calcular descriptores en paralelo
def get_descriptors(smiles_list):
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    batch_size = max(1, len(smiles_list) // (n_cores * 4))
    batches = [smiles_list[i:i+batch_size] for i in range(0, len(smiles_list), batch_size)]
    
    X_list = []
    valid_idx = []
    failed_smiles = []
    
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        batch_results = list(executor.map(calculate_descriptors_batch, batches))
    
    for batch_result, batch_smiles in zip(batch_results, batches):
        for smiles, desc in zip(batch_smiles, batch_result):
            if desc is not None:
                valid_idx.append(smiles_list.index(smiles))
                X_list.append(desc)
            else:
                failed_smiles.append(smiles)
    
    if failed_smiles:
        logging.warning(f"Failed to calculate descriptors for {len(failed_smiles)} SMILES: {failed_smiles[:5]}...")
    
    logging.info(f"Total valid descriptors calculated: {len(X_list)}")
    return np.array(X_list), valid_idx

# Carga el modelo y objetos de preprocesamiento
try:
    rf_model = joblib.load(os.path.join(input_dir, 'rf_model_With_Murcko.joblib'))
    scaler = joblib.load(os.path.join(input_dir, 'scaler_With_Murcko.joblib'))
    imputer = joblib.load(os.path.join(input_dir, 'imputer_With_Murcko.joblib'))
    selected_features = joblib.load(os.path.join(input_dir, 'selected_features_With_Murcko.joblib'))
    logging.info("Model and preprocessing objects loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Error loading model or preprocessing objects: {e}")
    exit()

# Carga las moléculas de Cannabaceae
logging.info("Loading Cannabaceae molecules...")
try:
    df_cannabaceae = pd.read_csv('Cannabaceae_molecules2.csv', sep=';')
    logging.info(f"Initial rows in Cannabaceae_molecules2.csv: {len(df_cannabaceae)}")
except Exception as e:
    logging.error(f"Error loading Cannabaceae_molecules2.csv: {e}")
    exit()

# Verifica columnas 'Smiles' y 'Molecule_name'
smiles_column = 'Smiles'
molecule_name_column = 'Molecule_name'
df_cannabaceae.columns = df_cannabaceae.columns.str.strip()
if smiles_column not in df_cannabaceae.columns:
    logging.error(f"La columna '{smiles_column}' no se encuentra en el archivo CSV.")
    logging.info(f"Columnas disponibles: {df_cannabaceae.columns.tolist()}")
    exit()
if molecule_name_column not in df_cannabaceae.columns:
    logging.error(f"La columna '{molecule_name_column}' no se encuentra en el archivo CSV.")
    logging.info(f"Columnas disponibles: {df_cannabaceae.columns.tolist()}")
    exit()

# Valida las SMILES
logging.info("Validating SMILES for Cannabaceae molecules...")
smiles_list, valid_idx = validate_smiles(df_cannabaceae[smiles_column].tolist())
logging.info(f"Valid SMILES: {len(smiles_list)}")
if not smiles_list:
    logging.error("No valid SMILES found in Cannabaceae_molecules2.csv.")
    exit()

# Obtiene los nombres de moléculas correspondientes
molecule_names = [df_cannabaceae[molecule_name_column].iloc[i] for i in valid_idx]

# Calcula descriptores
logging.info(f"Calculating descriptors for {len(smiles_list)} valid SMILES...")
X_cannabaceae, valid_idx = get_descriptors(smiles_list)
logging.info(f"Descriptors shape: {X_cannabaceae.shape}")

# Preprocesa los descriptores
X_cannabaceae = imputer.transform(X_cannabaceae)
X_cannabaceae_scaled = scaler.transform(X_cannabaceae)
X_cannabaceae_selected = X_cannabaceae_scaled[:, selected_features]
logging.info(f"Selected features shape: {X_cannabaceae_selected.shape}")

# Realiza predicciones con intervalos de confianza
logging.info("Making predictions with confidence intervals...")
predictions = []
lower_ci = []
upper_ci = []
confidence_level = 0.95
z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

tree_predictions = np.array([tree.predict(X_cannabaceae_selected) for tree in rf_model.estimators_])
mean_predictions = np.mean(tree_predictions, axis=0)
std_predictions = np.std(tree_predictions, axis=0)
ci_margin = z_score * std_predictions / np.sqrt(len(rf_model.estimators_))

for i, (mean_pred, ci) in enumerate(zip(mean_predictions, ci_margin)):
    predictions.append(mean_pred)
    lower_ci.append(mean_pred - ci)
    upper_ci.append(mean_pred + ci)

# Crea un DataFrame con los resultados
results_df = pd.DataFrame({
    'SMILES': [smiles_list[i] for i in range(len(smiles_list))],
    'Molecule_name': molecule_names,
    'Predicted_Log10_Value_nM': predictions,
    'Lower_CI_95': lower_ci,
    'Upper_CI_95': upper_ci,
    'CI_95_Width': [upper - lower for upper, lower in zip(upper_ci, lower_ci)]
})

# Guarda los resultados en un archivo CSV, asegurando puntos como separador decimal
results_df.to_csv(os.path.join(output_dir, 'cannabaceae_predictions4.csv'), sep=';', index=False, float_format='%.6f')
logging.info(f"Predictions saved to {os.path.join(output_dir, 'cannabaceae_predictions4.csv')}")

# Registra estadísticas resumidas
logging.info("\nPrediction Summary:")
logging.info(f"Number of molecules predicted: {len(predictions)}")
logging.info(f"Average predicted Log10 Value (nM): {np.mean(predictions):.3f}")
logging.info(f"Standard deviation of predictions: {np.std(predictions):.3f}")
logging.info(f"Average CI width: {np.mean([upper - lower for upper, lower in zip(upper_ci, lower_ci)]):.3f}")
logging.info("Process completed successfully.")