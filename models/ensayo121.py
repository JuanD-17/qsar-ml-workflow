# Ensayo 121 - Optimización de un modelo QSAR utilizando Random Forest y selección de características
# ANDAMIOS DE MURCKO Y ECFP
# Random Forest con selección de características Boruta y Mutual Information
# Categoría log(Bajo) - Potencia Media

# Importa pandas para manejo de datos
import pandas as pd
# Importa numpy para cálculos numéricos
import numpy as np
# Importa random para generación de números aleatorios
import random
# Importa os para manejo de directorios y archivos
import os
# Importa matplotlib para visualización de gráficos
import matplotlib.pyplot as plt
# Importa joblib para guardar y cargar modelos
import joblib
# Importa RDKit para manipulación de moléculas químicas
from rdkit import Chem
# Importa módulos de RDKit para descriptores moleculares
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Crippen, Lipinski
# Importa MurckoScaffold para calcular andamios de Murcko
from rdkit.Chem.Scaffolds import MurckoScaffold
# Importa RandomForestRegressor para modelado
from sklearn.ensemble import RandomForestRegressor
# Importa herramientas de scikit-learn para división de datos y validación
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
# Importa métricas para evaluar el modelo
from sklearn.metrics import r2_score, mean_squared_error
# Importa StandardScaler para escalado de características
from sklearn.preprocessing import StandardScaler
# Importa SimpleImputer para imputación de valores faltantes
from sklearn.impute import SimpleImputer
# Importa SelectFromModel para selección de características
from sklearn.feature_selection import SelectFromModel
# Importa LassoCV para selección de características con regularización
from sklearn.linear_model import LassoCV
# Importa BorutaPy para selección avanzada de características
from boruta import BorutaPy
# Importa mutual_info_regression para selección basada en información mutua
from sklearn.feature_selection import mutual_info_regression
# Importa ThreadPoolExecutor para procesamiento paralelo
from concurrent.futures import ThreadPoolExecutor
# Importa multiprocessing para usar múltiples núcleos de CPU
import multiprocessing
# Importa logging para registro de eventos
import logging

# Configura el sistema de registro con nivel INFO y formato con marca de tiempo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fija la semilla de numpy para reproducibilidad
np.random.seed(42)
# Fija la semilla de random para reproducibilidad
random.seed(42)

# Define el nombre del directorio para guardar resultados
output_dir = "ensayo121"
# Crea el directorio "ensayo121", no genera error si ya existe
os.makedirs(output_dir, exist_ok=True)

# Define la ruta del archivo de caché para descriptores
cache_file = os.path.join(output_dir, "descriptors_cache_with_murcko.joblib")
# Verifica si el archivo de caché existe
if os.path.exists(cache_file):
    # Elimina el archivo de caché si existe
    os.remove(cache_file)
    # Registra la eliminación del archivo de caché
    logging.info(f"Cache eliminado: {cache_file}")

# Define función para contar enlaces dobles no aromáticos
def count_double_bonds(mol):
    """Count the number of double bonds in a molecule, excluding aromatic bonds."""
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE and not bond.GetIsAromatic())

# Define función para validar cadenas SMILES
def validate_smiles(smiles_list):
    # Inicializa lista para almacenar SMILES válidos
    valid_smiles = []
    # Inicializa lista para almacenar índices de SMILES válidos
    valid_idx = []
    # Itera sobre la lista de SMILES con índices
    for i, smiles in enumerate(smiles_list):
        try:
            # Convierte SMILES a objeto molécula
            mol = Chem.MolFromSmiles(smiles)
            # Verifica si la molécula es válida y tiene átomos
            if mol is not None and mol.GetNumAtoms() > 0:
                # Agrega SMILES válido a la lista
                valid_smiles.append(smiles)
                # Agrega índice del SMILES válido
                valid_idx.append(i)
            else:
                # Registra advertencia para SMILES inválido
                logging.warning(f"SMILES inválido (mol None o 0 átomos): {smiles}")
        except Exception as e:
            # Registra advertencia con el error
            logging.warning(f"Error validando SMILES {smiles}: {e}")
            # Continúa con el siguiente SMILES
            continue
    # Devuelve SMILES válidos e índices
    return valid_smiles, valid_idx

# Define función optimizada para calcular descriptores en lote
def calculate_descriptors_batch(smiles_batch):
    # Inicializa lista para resultados
    results = []
    # Configura generador de fingerprints Morgan con radio 3 y tamaño 512
    morgan_gen = AllChem.GetMorganGenerator(radius=3, fpSize=512)
    
    # Define lista de funciones para calcular descriptores moleculares
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
        ('FractionCSP3', lambda mol: Descriptors.FractionCSP3(mol)),
        ('AromaticityIndex', lambda mol: rdMolDescriptors.CalcNumAromaticRings(mol) / max(1, rdMolDescriptors.CalcNumRings(mol)) if rdMolDescriptors.CalcNumRings(mol) > 0 else 0),
        ('DipoleMoment', lambda mol: Crippen.MolLogP(mol) / max(1, Descriptors.MolWt(mol))),
        ('ChargeDistribution', lambda mol: (Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol)) / max(1, Descriptors.HeavyAtomCount(mol))),
        ('PartialAtomicCharge', lambda mol: Descriptors.NumValenceElectrons(mol) / max(1, Descriptors.HeavyAtomCount(mol))),
        ('ConjugatedPiSystems', lambda mol: (count_double_bonds(mol) + rdMolDescriptors.CalcNumAromaticRings(mol)) if (count_double_bonds(mol) + rdMolDescriptors.CalcNumAromaticRings(mol)) > 0 else 0),
        ('NumRadicalElectrons', lambda mol: Descriptors.NumRadicalElectrons(mol)),
        ('MaxPartialCharge', lambda mol: Descriptors.MaxPartialCharge(mol) if Descriptors.MaxPartialCharge(mol) is not None else 0),
        ('MinPartialCharge', lambda mol: Descriptors.MinPartialCharge(mol) if Descriptors.MinPartialCharge(mol) is not None else 0),
        ('MaxAbsPartialCharge', lambda mol: Descriptors.MaxAbsPartialCharge(mol) if Descriptors.MaxAbsPartialCharge(mol) is not None else 0),
        ('MinAbsPartialCharge', lambda mol: Descriptors.MinAbsPartialCharge(mol) if Descriptors.MinAbsPartialCharge(mol) is not None else 0),
        ('BertzCT', lambda mol: Descriptors.BertzCT(mol)),
        ('Chi0v', lambda mol: Descriptors.Chi0v(mol)),
        ('Chi1v', lambda mol: Descriptors.Chi1v(mol)),
        ('Chi2v', lambda mol: Descriptors.Chi2v(mol)),
        ('HallKierAlpha', lambda mol: Descriptors.HallKierAlpha(mol)),
        ('ExactMolWt', lambda mol: Descriptors.ExactMolWt(mol)),
        ('NumAliphaticRings', lambda mol: rdMolDescriptors.CalcNumAliphaticRings(mol)),
        ('NumOxygenAtoms', lambda mol: len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8])),
        ('OxygenBonds', lambda mol: len([bond for bond in mol.GetBonds() if bond.GetBeginAtom().GetAtomicNum() == 8 or bond.GetEndAtom().GetAtomicNum() == 8])),
        ('MolMR', lambda mol: Descriptors.MolMR(mol)),
        ('FormalChargeCount', lambda mol: len([atom for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0])),
        ('RadicalElectronCount', lambda mol: len([atom for atom in mol.GetAtoms() if atom.GetNumRadicalElectrons() > 0]))
    ]
    
    # Itera sobre el lote de SMILES
    for smiles in smiles_batch:
        try:
            # Convierte SMILES a objeto molécula
            mol = Chem.MolFromSmiles(smiles)
            # Verifica si la molécula es válida
            if mol is None or mol.GetNumAtoms() == 0:
                # Registra advertencia para SMILES inválido
                logging.warning(f"SMILES inválido en descriptor: {smiles}")
                results.append(None)
                continue
            
            # Calcula fingerprints Morgan
            fp = morgan_gen.GetFingerprint(mol)
            # Convierte fingerprints a arreglo numpy
            fp_arr = np.array(list(fp))
            
            # Inicializa diccionario para descriptores
            descriptors_dict = {}
            # Calcula cada descriptor
            for name, func in descriptor_functions:
                try:
                    # Aplica función del descriptor
                    descriptors_dict[name] = func(mol)
                except Exception as e:
                    # Registra advertencia si falla el cálculo del descriptor
                    logging.warning(f"Error en descriptor {name} para {smiles}: {e}")
                    descriptors_dict[name] = 0.0
            
            # Calcula andamio de Murcko
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            # Verifica si el andamio es válido
            if scaffold is None or scaffold.GetNumAtoms() == 0:
                # Crea fingerprints vacíos para andamio inválido
                scaffold_fp = np.zeros(512)
                # Define descriptores vacíos para andamio
                scaffold_descriptors = {
                    'Scaffold_MolWt': 0.0,
                    'Scaffold_HeavyAtomCount': 0,
                    'Scaffold_RingCount': 0,
                    'Scaffold_AromaticRingCount': 0,
                    'Scaffold_TPSA': 0.0
                }
            else:
                # Calcula fingerprints Morgan para el andamio
                scaffold_fp = morgan_gen.GetFingerprint(scaffold)
                scaffold_fp = np.array(list(scaffold_fp))
                # Calcula descriptores del andamio
                scaffold_descriptors = {
                    'Scaffold_MolWt': Descriptors.MolWt(scaffold),
                    'Scaffold_HeavyAtomCount': Descriptors.HeavyAtomCount(scaffold),
                    'Scaffold_RingCount': rdMolDescriptors.CalcNumRings(scaffold),
                    'Scaffold_AromaticRingCount': rdMolDescriptors.CalcNumAromaticRings(scaffold),
                    'Scaffold_TPSA': rdMolDescriptors.CalcTPSA(scaffold)
                }
                
            # Actualiza diccionario con descriptores del andamio
            descriptors_dict.update(scaffold_descriptors)
            # Combina fingerprints y descriptores en un solo arreglo
            final_descriptors = np.concatenate([fp_arr, np.array([value for value in descriptors_dict.values()]), scaffold_fp])
            # Reemplaza valores NaN o infinitos por 0
            results.append(np.nan_to_num(final_descriptors, nan=0.0, posinf=0.0, neginf=0.0))
        except Exception as e:
            # Registra advertencia si falla el cálculo de descriptores
            logging.warning(f"Error en descriptor para SMILES {smiles}: {e}")
            results.append(None)
    # Devuelve resultados del lote
    return results

# Define función para obtener descriptores con caché
def get_descriptors_with_cache(smiles_list):
    # Verifica si existe el archivo de caché
    if os.path.exists(cache_file):
        # Carga datos del caché
        cached_data = joblib.load(cache_file)
        # Extrae SMILES del caché
        cached_smiles = cached_data['smiles']
        # Extrae descriptores del caché
        cached_descriptors = cached_data['descriptors']
        # Identifica SMILES nuevos para calcular
        smiles_to_calculate = [s for s in smiles_list if s not in cached_smiles]
        # Registra uso del caché
        logging.info(f"Usando caché para {len(cached_smiles)} SMILES, calculando {len(smiles_to_calculate)} nuevos.")
    else:
        # Inicializa lista vacía de SMILES en caché
        cached_smiles = []
        # Inicializa lista vacía de descriptores en caché
        cached_descriptors = []
        # Todos los SMILES deben calcularse
        smiles_to_calculate = smiles_list
        # Registra que no hay caché
        logging.info(f"No se encontró caché, calculando {len(smiles_to_calculate)} SMILES.")
    
    # Inicializa lista para descriptores válidos
    X_list = []
    # Inicializa lista para índices válidos
    valid_idx = []
    # Inicializa lista para SMILES fallidos
    failed_smiles = []
    
    # Si hay SMILES para calcular
    if smiles_to_calculate:
        # Determina número de núcleos disponibles
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        # Calcula tamaño de lote para procesamiento paralelo
        batch_size = max(1, len(smiles_to_calculate) // (n_cores * 4))
        # Divide SMILES en lotes
        batches = [smiles_to_calculate[i:i+batch_size] for i in range(0, len(smiles_to_calculate), batch_size)]
        
        # Inicializa lista para nuevos descriptores
        all_new_descriptors = []
        # Inicia executor para procesamiento paralelo
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            # Procesa lotes en paralelo
            batch_results = list(executor.map(calculate_descriptors_batch, batches))
        
        # Itera sobre resultados y lotes
        for batch_result, batch_smiles in zip(batch_results, batches):
            # Itera sobre SMILES y descriptores
            for smiles, desc in zip(batch_smiles, batch_result):
                # Si el descriptor es válido
                if desc is not None:
                    # Agrega índice del SMILES válido
                    valid_idx.append(smiles_list.index(smiles))
                    # Agrega descriptor a la lista
                    X_list.append(desc)
                    # Agrega SMILES al caché
                    cached_smiles.append(smiles)
                    # Agrega descriptor al caché
                    cached_descriptors.append(desc)
                else:
                    # Agrega SMILES a la lista de fallidos
                    failed_smiles.append(smiles)
        
        # Si hay SMILES fallidos
        if failed_smiles:
            # Registra advertencia con SMILES fallidos
            logging.warning(f"Fallaron {len(failed_smiles)} SMILES al calcular descriptores: {failed_smiles[:5]}...")
    
    # Registra número de descriptores válidos
    logging.info(f"Total descriptores válidos calculados: {len(X_list)}")
    
    # Guarda SMILES y descriptores en caché
    joblib.dump({'smiles': cached_smiles, 'descriptors': cached_descriptors}, cache_file)
    
    # Devuelve matriz de descriptores e índices válidos
    return np.array(X_list), valid_idx

# Define función para selección de características con Boruta
def boruta_feature_selection(X, y):
    # Crea modelo Random Forest para Boruta
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    # Configura BorutaPy
    boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=100)
    # Ajusta Boruta a los datos
    boruta.fit(X, y)
    # Devuelve máscara de características seleccionadas
    return boruta.support_

# Define función para selección con Mutual Information
def mutual_info_selection(X, y, n_features=50):
    # Calcula puntajes de información mutua
    mi_scores = mutual_info_regression(X, y, random_state=42)
    # Selecciona los índices de las n_features con mayor puntaje
    top_features = np.argsort(mi_scores)[-n_features:]
    # Crea máscara inicial de ceros
    mask = np.zeros(X.shape[1], dtype=bool)
    # Marca las características seleccionadas como True
    mask[top_features] = True
    # Devuelve máscara de características seleccionadas
    return mask

# Define función para procesar y entrenar modelo Random Forest
def process_and_train_rf_model(X, y, output_dir, model_name):
    # Divide datos en entrenamiento (75%) y prueba (25%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Registra tamaño de los conjuntos
    logging.info(f"Train set: {len(y_train)} muestras, Test set: {len(y_test)} muestras.")
    
    # Crea imputador para valores faltantes con estrategia de media
    imputer = SimpleImputer(strategy='mean')
    # Ajusta y transforma datos de entrenamiento
    X_train = imputer.fit_transform(X_train)
    # Transforma datos de prueba
    X_test = imputer.transform(X_test)
    
    # Crea escalador para estandarizar características
    scaler = StandardScaler()
    # Ajusta y transforma datos de entrenamiento
    X_train_scaled = scaler.fit_transform(X_train)
    # Transforma datos de prueba
    X_test_scaled = scaler.transform(X_test)
    
    # Inicializa máscara para selección de características
    combined_mask = np.ones(X_train_scaled.shape[1], dtype=bool)
    
    # Aplica selección de características con Boruta
    boruta_mask = boruta_feature_selection(X_train_scaled, y_train)
    # Combina máscara con Boruta (AND lógico)
    combined_mask = np.logical_and(combined_mask, boruta_mask)
    # Registra número de características seleccionadas por Boruta
    logging.info(f"Características seleccionadas por Boruta: {np.sum(boruta_mask)}")
    
    # Crea figura para gráfico de Boruta
    plt.figure(figsize=(10, 6))
    # Grafica máscara de Boruta
    plt.bar(range(len(boruta_mask)), boruta_mask)
    # Establece título del gráfico
    plt.title(f'Boruta Feature Selection - {model_name}')
    # Etiqueta eje X
    plt.xlabel('Feature Index')
    # Etiqueta eje Y
    plt.ylabel('Selected (1) / Not Selected (0)')
    # Guarda gráfico de Boruta
    plt.savefig(os.path.join(output_dir, f'boruta_features_{model_name}.png'))
    # Cierra la figura
    plt.close()
    
    # Aplica selección con Mutual Information
    mi_mask = mutual_info_selection(X_train_scaled, y_train)
    # Combina máscara con Mutual Information (AND lógico)
    combined_mask = np.logical_and(combined_mask, mi_mask)
    # Registra número de características seleccionadas por MI
    logging.info(f"Características seleccionadas por Mutual Information: {np.sum(mi_mask)}")
    
    # Calcula puntajes de Mutual Information
    mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)
    # Crea figura para gráfico de MI
    plt.figure(figsize=(10, 6))
    # Grafica puntajes de MI
    plt.bar(range(len(mi_scores)), mi_scores)
    # Establece título del gráfico
    plt.title(f'Mutual Information Scores - {model_name}')
    # Etiqueta eje X
    plt.xlabel('Feature Index')
    # Etiqueta eje Y
    plt.ylabel('Mutual Information Score')
    # Guarda gráfico de MI
    plt.savefig(os.path.join(output_dir, f'mi_features_{model_name}.png'))
    # Cierra la figura
    plt.close()
    
    # Crea modelo Random Forest inicial
    initial_rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    # Ajusta modelo inicial a datos de entrenamiento
    initial_rf.fit(X_train_scaled, y_train)
    # Crea selector de características basado en importancia media
    sfm = SelectFromModel(initial_rf, threshold="mean", prefit=True)
    # Obtiene máscara de características seleccionadas por Random Forest
    sfm_mask = sfm.get_support()
    
    # Crea modelo LassoCV para selección de características
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=20000, n_jobs=-1)
    # Ajusta LassoCV a datos de entrenamiento
    lasso_cv.fit(X_train_scaled, y_train)
    # Crea máscara de características no nulas
    lasso_mask = lasso_cv.coef_ != 0
    
    # Combina máscaras con OR lógico
    combined_mask = np.logical_or(combined_mask, sfm_mask, lasso_mask)
    # Selecciona características en datos de entrenamiento
    X_train_selected = X_train_scaled[:, combined_mask]
    # Selecciona características en datos de prueba
    X_test_selected = X_test_scaled[:, combined_mask]
    
    # Registra número de características seleccionadas
    logging.info(f"Total características seleccionadas para {model_name}: {X_train_selected.shape[1]} de {X_train_scaled.shape[1]}")
    
    # Guarda máscara combinada
    joblib.dump(combined_mask, os.path.join(output_dir, f'selected_features_{model_name}.joblib'))
    # Guarda selector de Random Forest
    joblib.dump(sfm, os.path.join(output_dir, f'sfm_selector_{model_name}.joblib'))
    # Guarda selector de Lasso
    joblib.dump(lasso_cv, os.path.join(output_dir, f'lasso_selector_{model_name}.joblib'))
    # Guarda escalador
    joblib.dump(scaler, os.path.join(output_dir, f'scaler_{model_name}.joblib'))
    # Guarda imputador
    joblib.dump(imputer, os.path.join(output_dir, f'imputer_{model_name}.joblib'))
    
    # Define cuadrícula de hiperparámetros para GridSearchCV
    param_grid_rf = {
        # Número de árboles
        'n_estimators': [150, 250, 300],
        # Profundidad máxima
        'max_depth': [8, 12, None],
        # Mínimo de muestras para dividir un nodo
        'min_samples_split': [2, 3, 5],
        # Mínimo de muestras por hoja
        'min_samples_leaf': [1, 2],
        # Número de características por división
        'max_features': ['sqrt', 0.7]
    }
    # Crea objeto GridSearchCV para optimización
    grid_search_rf = GridSearchCV(
        # Modelo base
        RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=True),
        # Cuadrícula de hiperparámetros
        param_grid_rf,
        # Validación cruzada de 5 pliegues
        cv=5,
        # Métrica de evaluación
        scoring='neg_mean_squared_error',
        # Usa todos los núcleos disponibles
        n_jobs=-1,
        # Muestra progreso
        verbose=1
    )
    # Ajusta GridSearchCV a datos seleccionados
    grid_search_rf.fit(X_train_selected, y_train)
    
    # Crea modelo final con mejores hiperparámetros
    final_rf = RandomForestRegressor(**grid_search_rf.best_params_, random_state=42, n_jobs=-1, bootstrap=True)
    # Ajusta modelo final a datos de entrenamiento
    final_rf.fit(X_train_selected, y_train)
    
    # Genera predicciones para entrenamiento
    y_train_pred_rf = final_rf.predict(X_train_selected)
    # Genera predicciones para prueba
    y_test_pred_rf = final_rf.predict(X_test_selected)
    # Calcula R² para entrenamiento
    train_r2_rf = r2_score(y_train, y_train_pred_rf)
    # Calcula R² para prueba
    test_r2_rf = r2_score(y_test, y_test_pred_rf)
    # Calcula RMSE para entrenamiento
    train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
    # Calcula RMSE para prueba
    test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
    
    # Configura validación cruzada de 5 pliegues
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # Calcula R² en validación cruzada
    cv_scores_r2_rf = cross_val_score(final_rf, X_train_selected, y_train, cv=cv, scoring='r2', n_jobs=-1)
    # Calcula MSE en validación cruzada
    cv_scores_mse_rf = -cross_val_score(final_rf, X_train_selected, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    # Convierte MSE a RMSE
    cv_scores_rmse_rf = np.sqrt(cv_scores_mse_rf)
    
    # Registra inicio de resultados
    logging.info(f"\nResultados del modelo {model_name} (Random Forest):")
    # Registra R² de entrenamiento
    logging.info(f"R² Training: {train_r2_rf:.3f}")
    # Registra RMSE de entrenamiento
    logging.info(f"RMSE Training: {train_rmse_rf:.3f}")
    # Registra R² de prueba
    logging.info(f"R² Validation: {test_r2_rf:.3f}")
    # Registra RMSE de prueba
    logging.info(f"RMSE Validation: {test_rmse_rf:.3f}")
    # Registra R² promedio de validación cruzada
    logging.info(f"R² CV 5-fold: {cv_scores_r2_rf.mean():.3f} ± {cv_scores_r2_rf.std():.3f}")
    # Registra RMSE promedio de validación cruzada
    logging.info(f"RMSE CV 5-fold: {cv_scores_rmse_rf.mean():.3f} ± {cv_scores_rmse_rf.std():.3f}")
    
    # Guarda modelo Random Forest
    joblib.dump(final_rf, os.path.join(output_dir, f'rf_model_{model_name}.joblib'))
    
    # Crea figura para gráfico de dispersión
    plt.figure(figsize=(10, 8))
    # Grafica datos de entrenamiento
    plt.scatter(y_train, y_train_pred_rf, c='blue', label=f'Training: R²={train_r2_rf:.3f}, RMSE={train_rmse_rf:.3f}', alpha=0.6)
    # Grafica datos de prueba
    plt.scatter(y_test, y_test_pred_rf, c='red', label=f'Validation: R²={test_r2_rf:.3f}, RMSE={test_rmse_rf:.3f}', alpha=0.6)
    # Grafica línea ideal
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Ideal line (Y=X)')
    # Etiqueta eje X
    plt.xlabel('Observed Antioxidant Activity (Log10 Value (nM))')
    # Etiqueta eje Y
    plt.ylabel('Predicted')
    # Establece título del gráfico
    plt.title(f'QSAR Random Forest Model - {model_name}')
    # Muestra leyenda
    plt.legend()
    # Agrega cuadrícula
    plt.grid(True)
    # Guarda gráfico
    plt.savefig(os.path.join(output_dir, f'rf_model_plot_{model_name}.png'), dpi=150)
    # Cierra la figura
    plt.close()
    
    # Crea diccionario con resultados
    results = {
        # Nombre del modelo
        'model_name': model_name,
        # R² de entrenamiento
        'train_r2_rf': train_r2_rf,
        # R² de prueba
        'test_r2_rf': test_r2_rf,
        # RMSE de entrenamiento
        'train_rmse_rf': train_rmse_rf,
        # RMSE de prueba
        'test_rmse_rf': test_rmse_rf,
        # Promedio de R² en validación cruzada
        'cv_r2_mean_rf': cv_scores_r2_rf.mean(),
        # Desviación estándar de R²
        'cv_r2_std_rf': cv_scores_r2_rf.std(),
        # Promedio de RMSE en validación cruzada
        'cv_rmse_mean_rf': cv_scores_rmse_rf.mean(),
        # Desviación estándar de RMSE
        'cv_rmse_std_rf': cv_scores_rmse_rf.std()
    }
    
    # Crea DataFrame para resultados
    comparison = pd.DataFrame({
        # Nombre del modelo
        'Modelo': ['Random Forest'],
        # R² de entrenamiento
        'R² Training': [results['train_r2_rf']],
        # RMSE de entrenamiento
        'RMSE Training': [results['train_rmse_rf']],
        # R² de prueba
        'R² Validation': [results['test_r2_rf']],
        # RMSE de prueba
        'RMSE Validation': [results['test_rmse_rf']],
        # Promedio de R² en validación cruzada
        'R² CV Mean': [results['cv_r2_mean_rf']],
        # Desviación estándar de R²
        'R² CV Std': [results['cv_r2_std_rf']],
        # Promedio de RMSE en validación cruzada
        'RMSE CV Mean': [results['cv_rmse_mean_rf']],
        # Desviación estándar de RMSE
        'RMSE CV Std': [results['cv_rmse_std_rf']]
    })
    # Guarda resultados en CSV
    comparison.to_csv(os.path.join(output_dir, f'model_comparison_{model_name}.csv'), index=False)
    
    # Devuelve diccionario de resultados
    return results

# Registra inicio de carga de datos
logging.info("Cargando datos...")
# Carga archivo CSV con datos moleculares
df = pd.read_csv('Antioxidant13.csv')

# Define categoría deseada
categoria_deseada = 'Bajo'
# Define subcategoría deseada
subcategoria_deseada = 'Bajo - Potencia Media'
# Filtra datos por categoría
df_categoria = df[df['Category'] == categoria_deseada].copy()
# Filtra datos por subcategoría
df_subcategoria = df_categoria[df_categoria['subcategoria_nombre'] == subcategoria_deseada].copy()

# Verifica si hay datos en la subcategoría
if df_subcategoria.empty:
    # Registra error si no hay datos
    logging.error(f"No se encontraron datos para la categoría '{categoria_deseada}' y subcategoría '{subcategoria_deseada}'.")
    # Termina ejecución
    exit()

# Registra número inicial de filas
logging.info(f"Filas iniciales en df_subcategoria: {len(df_subcategoria)}")
# Convierte columna de actividad a numérica
df_subcategoria['Log10 Value (nM)'] = pd.to_numeric(df_subcategoria['Log10 Value (nM)'], errors='coerce')
# Registra filas tras conversión
logging.info(f"Filas después de conversión numérica: {len(df_subcategoria)}")
# Elimina filas con valores faltantes
df_subcategoria = df_subcategoria.dropna(subset=['Log10 Value (nM)', 'Smiles'])
# Registra filas tras eliminar NaN
logging.info(f"Filas después de eliminar NaN: {len(df_subcategoria)}")

# Registra inicio de validación de SMILES
logging.info(f"Validando SMILES para '{categoria_deseada} - {subcategoria_deseada}'...")
# Valida SMILES de la subcategoría
smiles_list, valid_idx = validate_smiles(df_subcategoria['Smiles'].tolist())
# Registra número de SMILES válidos
logging.info(f"SMILES válidos: {len(smiles_list)}")
# Verifica si hay SMILES válidos
if not smiles_list:
    # Registra error si no hay SMILES válidos
    logging.error("No se encontraron SMILES válidos.")
    # Termina ejecución
    exit()

# Registra inicio de cálculo de descriptores
logging.info(f"Calculando descriptores con andamios de Murcko para {len(smiles_list)} SMILES válidos...")
# Calcula descriptores usando caché
X_with_murcko, valid_idx = get_descriptors_with_cache(smiles_list)

# Convierte descriptores a arreglo numpy
X_with_murcko = np.array(X_with_murcko)
# Extrae valores objetivo correspondientes a SMILES válidos
y = df_subcategoria['Log10 Value (nM)'].iloc[valid_idx].values

# Registra dimensiones de matrices
logging.info(f"X_with_murcko shape: {X_with_murcko.shape}, y shape: {y.shape}")
# Verifica si hay suficientes datos
if len(X_with_murcko) < 2 or len(y) < 2:
    # Registra error si no hay suficientes datos
    logging.error("No hay suficientes datos válidos para la regresión.")
    # Termina ejecución
    exit()

# Registra inicio de entrenamiento
logging.info("\nEntrenando modelo Random Forest con andamios de Murcko...")
# Entrena y evalúa modelo
results_rf = process_and_train_rf_model(X_with_murcko, y, output_dir, "With_Murcko")

# Registra ubicación de resultados guardados
logging.info(f"Resultados guardados en '{output_dir}'")
# Registra finalización exitosa
logging.info("Proceso completado exitosamente.")