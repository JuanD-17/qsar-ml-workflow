# Ensayo 171 - Optimización de un modelo QSAR utilizando Random Forest, MLP y Stacking
# ANDAMIOS DE MURCKO Y ECFP
# Random Forest con selección de características Boruta y Mutual Information
# MLP Change: Implementación de red neuronal MLPRegressor
# Stacking: Combina Random Forest y MLP con un meta-modelo de regresión lineal
# Categoria Alto (sin subcategorías)
# Radius 3, ECFP 512 bits

# Importa las bibliotecas necesarias
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Crippen, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression
from boruta import BorutaPy
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import logging
from sklearn.neural_network import MLPRegressor

# Configuración de logging y semillas
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)
random.seed(42)

output_dir = "ensayo171"
os.makedirs(output_dir, exist_ok=True)

cache_file = os.path.join(output_dir, "descriptors_cache_with_murcko.joblib")
if os.path.exists(cache_file):
    os.remove(cache_file)
    logging.info(f"Cache eliminado: {cache_file}")

# Funciones existentes
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
                logging.warning(f"SMILES inválido (mol None o 0 átomos): {smiles}")
        except Exception as e:
            logging.warning(f"Error validando SMILES {smiles}: {e}")
            continue
    return valid_smiles, valid_idx

def calculate_descriptors_batch(smiles_batch):
    results = []
    morgan_gen = AllChem.GetMorganGenerator(radius=3, fpSize=512)
    descriptor_functions = [
        ('MolLogP', Crippen.MolLogP),
        ('TPSA', rdMolDescriptors.CalcTPSA),
        ('MolWt', Descriptors.MolWt),
        ('HeavyAtomCount', Descriptors.HeavyAtomCount),
        ('RingCount', rdMolDescriptors.CalcNumRings),
        ('AromaticRingCount', rdMolDescriptors.CalcNumAromaticRings),
        ('HBondDonors', Lipinski.NumHDonors),
        ('HBondAcceptors', Lipinski.NumHAcceptors),
        ('NumRotatableBonds', rdMolDescriptors.CalcNumRotatableBonds),
        ('FractionCSP3', Descriptors.FractionCSP3)
    ]
    
    for smiles in smiles_batch:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                logging.warning(f"SMILES inválido en descriptor: {smiles}")
                results.append(None)
                continue
            
            fp = morgan_gen.GetFingerprint(mol)
            fp_arr = np.array(list(fp))
            
            descriptors_dict = {}
            for name, func in descriptor_functions:
                try:
                    descriptors_dict[name] = func(mol)
                except Exception as e:
                    logging.warning(f"Error en descriptor {name} para {smiles}: {e}")
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
            logging.warning(f"Error en descriptor para SMILES {smiles}: {e}")
            results.append(None)
    return results

def get_descriptors_with_cache(smiles_list):
    if os.path.exists(cache_file):
        cached_data = joblib.load(cache_file)
        cached_smiles = cached_data['smiles']
        cached_descriptors = cached_data['descriptors']
        smiles_to_calculate = [s for s in smiles_list if s not in cached_smiles]
        logging.info(f"Usando caché para {len(cached_smiles)} SMILES, calculando {len(smiles_to_calculate)} nuevos.")
    else:
        cached_smiles = []
        cached_descriptors = []
        smiles_to_calculate = smiles_list
        logging.info(f"No se encontró caché, calculando {len(smiles_to_calculate)} SMILES.")
    
    X_list = []
    valid_idx = []
    failed_smiles = []
    
    if smiles_to_calculate:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        batch_size = max(1, len(smiles_to_calculate) // (n_cores * 4))
        batches = [smiles_to_calculate[i:i+batch_size] for i in range(0, len(smiles_to_calculate), batch_size)]
        
        all_new_descriptors = []
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            batch_results = list(executor.map(calculate_descriptors_batch, batches))
        
        for batch_result, batch_smiles in zip(batch_results, batches):
            for smiles, desc in zip(batch_smiles, batch_result):
                if desc is not None:
                    valid_idx.append(smiles_list.index(smiles))
                    X_list.append(desc)
                    cached_smiles.append(smiles)
                    cached_descriptors.append(desc)
                else:
                    failed_smiles.append(smiles)
        
        if failed_smiles:
            logging.warning(f"Fallaron {len(failed_smiles)} SMILES al calcular descriptores: {failed_smiles[:5]}...")
    
    logging.info(f"Total descriptores válidos calculados: {len(X_list)}")
    
    joblib.dump({'smiles': cached_smiles, 'descriptors': cached_descriptors}, cache_file)
    
    return np.array(X_list), valid_idx

def boruta_feature_selection(X, y):
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=100)
    boruta.fit(X, y)
    return boruta.support_

def mutual_info_selection(X, y, n_features=50):
    mi_scores = mutual_info_regression(X, y, random_state=42)
    top_features = np.argsort(mi_scores)[-n_features:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_features] = True
    return mask

def process_and_train_models(X, y, output_dir, model_name):
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    logging.info(f"Train set: {len(y_train)} muestras, Test set: {len(y_test)} muestras.")
    
    # Preprocesamiento
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Selección de características
    combined_mask = np.ones(X_train_scaled.shape[1], dtype=bool)
    
    boruta_mask = boruta_feature_selection(X_train_scaled, y_train)
    combined_mask = np.logical_and(combined_mask, boruta_mask)
    logging.info(f"Características seleccionadas por Boruta: {np.sum(boruta_mask)}")
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(boruta_mask)), boruta_mask)
    plt.title(f'Boruta Feature Selection - {model_name}')
    plt.xlabel('Feature Index')
    plt.ylabel('Selected (1) / Not Selected (0)')
    plt.savefig(os.path.join(output_dir, f'boruta_features_{model_name}.png'))
    plt.close()
    
    mi_mask = mutual_info_selection(X_train_scaled, y_train)
    combined_mask = np.logical_and(combined_mask, mi_mask)
    logging.info(f"Características seleccionadas por Mutual Information: {np.sum(mi_mask)}")
    
    mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(mi_scores)), mi_scores)
    plt.title(f'Mutual Information Scores - {model_name}')
    plt.xlabel('Feature Index')
    plt.ylabel('Mutual Information Score')
    plt.savefig(os.path.join(output_dir, f'mi_features_{model_name}.png'))
    plt.close()
    
    initial_rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    initial_rf.fit(X_train_scaled, y_train)
    sfm = SelectFromModel(initial_rf, threshold="mean", prefit=True)
    sfm_mask = sfm.get_support()
    
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=20000, n_jobs=-1)
    lasso_cv.fit(X_train_scaled, y_train)
    lasso_mask = lasso_cv.coef_ != 0
    
    combined_mask = np.logical_or(combined_mask, sfm_mask, lasso_mask)
    X_train_selected = X_train_scaled[:, combined_mask]
    X_test_selected = X_test_scaled[:, combined_mask]
    
    logging.info(f"Total características seleccionadas para {model_name}: {X_train_selected.shape[1]} de {X_train_scaled.shape[1]}")
    
    # Guardar preprocesadores
    joblib.dump(combined_mask, os.path.join(output_dir, f'selected_features_{model_name}.joblib'))
    joblib.dump(sfm, os.path.join(output_dir, f'sfm_selector_{model_name}.joblib'))
    joblib.dump(lasso_cv, os.path.join(output_dir, f'lasso_selector_{model_name}.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, f'scaler_{model_name}.joblib'))
    joblib.dump(imputer, os.path.join(output_dir, f'imputer_{model_name}.joblib'))
    
    # Random Forest Model
    param_grid_rf = {
        'n_estimators': [150, 250, 300],
        'max_depth': [8, 12, None],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 0.7]
    }
    grid_search_rf = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=True),
        param_grid_rf,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search_rf.fit(X_train_selected, y_train)
    
    final_rf = RandomForestRegressor(**grid_search_rf.best_params_, random_state=42, n_jobs=-1, bootstrap=True)
    final_rf.fit(X_train_selected, y_train)
    
    y_train_pred_rf = final_rf.predict(X_train_selected)
    y_test_pred_rf = final_rf.predict(X_test_selected)
    train_r2_rf = r2_score(y_train, y_train_pred_rf)
    test_r2_rf = r2_score(y_test, y_test_pred_rf)
    train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
    test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_r2_rf = cross_val_score(final_rf, X_train_selected, y_train, cv=cv, scoring='r2', n_jobs=-1)
    cv_scores_mse_rf = -cross_val_score(final_rf, X_train_selected, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    cv_scores_rmse_rf = np.sqrt(cv_scores_mse_rf)
    
    logging.info(f"\nResultados del modelo {model_name} (Random Forest):")
    logging.info(f"R² Training: {train_r2_rf:.3f}")
    logging.info(f"RMSE Training: {train_rmse_rf:.3f}")
    logging.info(f"R² Validation: {test_r2_rf:.3f}")
    logging.info(f"RMSE Validation: {test_rmse_rf:.3f}")
    logging.info(f"R² CV 5-fold: {cv_scores_r2_rf.mean():.3f} ± {cv_scores_r2_rf.std():.3f}")
    logging.info(f"RMSE CV 5-fold: {cv_scores_rmse_rf.mean():.3f} ± {cv_scores_rmse_rf.std():.3f}")
    
    joblib.dump(final_rf, os.path.join(output_dir, f'rf_model_{model_name}.joblib'))
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_train_pred_rf, c='blue', label=f'Training: R²={train_r2_rf:.3f}, RMSE={train_rmse_rf:.3f}', alpha=0.6)
    plt.scatter(y_test, y_test_pred_rf, c='red', label=f'Validation: R²={test_r2_rf:.3f}, RMSE={test_rmse_rf:.3f}', alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Ideal line (Y=X)')
    plt.xlabel('Observed Antioxidant Activity (Log10 Value (nM))')
    plt.ylabel('Predicted')
    plt.title(f'QSAR Random Forest Model - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'rf_model_plot_{model_name}.png'), dpi=150)
    plt.close()
    
    # MLP Neural Network Model
    param_grid_mlp_convergence = {
        'mlp__hidden_layer_sizes': [(50,), (100,), (50, 25), (75, 50)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__alpha': [0.01, 0.1, 0.5],
        'mlp__solver': ['lbfgs', 'adam'],
        'mlp__learning_rate_init': [0.01, 0.1],
        'mlp__max_iter': [5000, 10000],
        'mlp__tol': [1e-3, 1e-4],
        'mlp__early_stopping': [True],
        'mlp__validation_fraction': [0.2],
        'mlp__n_iter_no_change': [50, 100],
        'mlp__beta_1': [0.9],
        'mlp__beta_2': [0.999]
    }
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('mlp', MLPRegressor(random_state=42))
    ])
    grid_search_mlp = GridSearchCV(
        pipeline,
        param_grid_mlp_convergence,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search_mlp.fit(X_train_selected, y_train)
    
    final_mlp = Pipeline([
        ('scaler', RobustScaler()),
        ('mlp', MLPRegressor(**{key.replace('mlp__', ''): value for key, value in grid_search_mlp.best_params_.items()}, random_state=42))
    ])
    final_mlp.fit(X_train_selected, y_train)
    
    y_train_pred_mlp = final_mlp.predict(X_train_selected)
    y_test_pred_mlp = final_mlp.predict(X_test_selected)
    train_r2_mlp = r2_score(y_train, y_train_pred_mlp)
    test_r2_mlp = r2_score(y_test, y_test_pred_mlp)
    train_rmse_mlp = np.sqrt(mean_squared_error(y_train, y_train_pred_mlp))
    test_rmse_mlp = np.sqrt(mean_squared_error(y_test, y_test_pred_mlp))
    
    cv_scores_r2_mlp = cross_val_score(final_mlp, X_train_selected, y_train, cv=cv, scoring='r2', n_jobs=-1)
    cv_scores_mse_mlp = -cross_val_score(final_mlp, X_train_selected, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    cv_scores_rmse_mlp = np.sqrt(cv_scores_mse_mlp)
    
    logging.info(f"\nResultados del modelo {model_name} (MLP Neural Network):")
    logging.info(f"R² Training: {train_r2_mlp:.3f}")
    logging.info(f"RMSE Training: {train_rmse_mlp:.3f}")
    logging.info(f"R² Validation: {test_r2_mlp:.3f}")
    logging.info(f"RMSE Validation: {test_rmse_mlp:.3f}")
    logging.info(f"R² CV 5-fold: {cv_scores_r2_mlp.mean():.3f} ± {cv_scores_r2_mlp.std():.3f}")
    logging.info(f"RMSE CV 5-fold: {cv_scores_rmse_mlp.mean():.3f} ± {cv_scores_rmse_mlp.std():.3f}")
    
    joblib.dump(final_mlp, os.path.join(output_dir, f'mlp_model_{model_name}.joblib'))
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_train_pred_mlp, c='blue', label=f'Training: R²={train_r2_mlp:.3f}, RMSE={train_rmse_mlp:.3f}', alpha=0.6)
    plt.scatter(y_test, y_test_pred_mlp, c='red', label=f'Validation: R²={test_r2_mlp:.3f}, RMSE={test_rmse_mlp:.3f}', alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Ideal line (Y=X)')
    plt.xlabel('Observed Antioxidant Activity (Log10 Value (nM))')
    plt.ylabel('Predicted')
    plt.title(f'QSAR MLP Neural Network Model - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'mlp_model_plot_{model_name}.png'), dpi=150)
    plt.close()
    
    # Stacking Ensemble Model
    logging.info(f"\nEntrenando modelo Stacking para {model_name}...")
    estimators = [
        ('rf', RandomForestRegressor(**grid_search_rf.best_params_, random_state=42, n_jobs=-1, bootstrap=True)),
        ('mlp', Pipeline([
            ('scaler', RobustScaler()),
            ('mlp', MLPRegressor(**{key.replace('mlp__', ''): value for key, value in grid_search_mlp.best_params_.items()}, random_state=42))
        ]))
    ]
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    stacking_regressor.fit(X_train_selected, y_train)
    
    y_train_pred_stack = stacking_regressor.predict(X_train_selected)
    y_test_pred_stack = stacking_regressor.predict(X_test_selected)
    train_r2_stack = r2_score(y_train, y_train_pred_stack)
    test_r2_stack = r2_score(y_test, y_test_pred_stack)
    train_rmse_stack = np.sqrt(mean_squared_error(y_train, y_train_pred_stack))
    test_rmse_stack = np.sqrt(mean_squared_error(y_test, y_test_pred_stack))
    
    cv_scores_r2_stack = cross_val_score(stacking_regressor, X_train_selected, y_train, cv=cv, scoring='r2', n_jobs=-1)
    cv_scores_mse_stack = -cross_val_score(stacking_regressor, X_train_selected, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    cv_scores_rmse_stack = np.sqrt(cv_scores_mse_stack)
    
    logging.info(f"\nResultados del modelo {model_name} (Stacking Ensemble):")
    logging.info(f"R² Training: {train_r2_stack:.3f}")
    logging.info(f"RMSE Training: {train_rmse_stack:.3f}")
    logging.info(f"R² Validation: {test_r2_stack:.3f}")
    logging.info(f"RMSE Validation: {test_rmse_stack:.3f}")
    logging.info(f"R² CV 5-fold: {cv_scores_r2_stack.mean():.3f} ± {cv_scores_r2_stack.std():.3f}")
    logging.info(f"RMSE CV 5-fold: {cv_scores_rmse_stack.mean():.3f} ± {cv_scores_rmse_stack.std():.3f}")
    
    joblib.dump(stacking_regressor, os.path.join(output_dir, f'stacking_model_{model_name}.joblib'))
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_train_pred_stack, c='blue', label=f'Training: R²={train_r2_stack:.3f}, RMSE={train_rmse_stack:.3f}', alpha=0.6)
    plt.scatter(y_test, y_test_pred_stack, c='red', label=f'Validation: R²={test_r2_stack:.3f}, RMSE={test_rmse_stack:.3f}', alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Ideal line (Y=X)')
    plt.xlabel('Observed Antioxidant Activity (Log10 Value (nM))')
    plt.ylabel('Predicted')
    plt.title(f'QSAR Stacking Ensemble Model - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'stacking_model_plot_{model_name}.png'), dpi=150)
    plt.close()
    
    # Actualizar diccionario de resultados con métricas del modelo de stacking
    results = {
        'model_name': model_name,
        'train_r2_rf': train_r2_rf,
        'test_r2_rf': test_r2_rf,
        'train_rmse_rf': train_rmse_rf,
        'test_rmse_rf': test_rmse_rf,
        'cv_r2_mean_rf': cv_scores_r2_rf.mean(),
        'cv_r2_std_rf': cv_scores_r2_rf.std(),
        'cv_rmse_rf': cv_scores_rmse_rf.mean(),
        'cv_rmse_std_rf': cv_scores_rmse_rf.std(),
        'train_r2_mlp': train_r2_mlp,
        'test_r2_mlp': test_r2_mlp,
        'train_rmse_mlp': train_rmse_mlp,
        'test_rmse_mlp': test_rmse_mlp,
        'cv_r2_mean_mlp': cv_scores_r2_mlp.mean(),
        'cv_r2_std_mlp': cv_scores_r2_mlp.std(),
        'cv_rmse_mlp': cv_scores_rmse_mlp.mean(),
        'cv_rmse_std_mlp': cv_scores_rmse_mlp.std(),
        'train_r2_stack': train_r2_stack,
        'test_r2_stack': test_r2_stack,
        'train_rmse_stack': train_rmse_stack,
        'test_rmse_stack': test_rmse_stack,
        'cv_r2_mean_stack': cv_scores_r2_stack.mean(),
        'cv_r2_std_stack': cv_scores_r2_stack.std(),
        'cv_rmse_stack': cv_scores_rmse_stack.mean(),
        'cv_rmse_std_stack': cv_scores_rmse_stack.std()
    }
    
    # Actualizar DataFrame de comparación para incluir stacking
    comparison = pd.DataFrame({
        'Modelo': ['Random Forest', 'MLP Neural Network', 'Stacking Ensemble'],
        'R² Training': [results['train_r2_rf'], results['train_r2_mlp'], results['train_r2_stack']],
        'RMSE Training': [results['train_rmse_rf'], results['train_rmse_mlp'], results['train_rmse_stack']],
        'R² Validation': [results['test_r2_rf'], results['test_r2_mlp'], results['test_r2_stack']],
        'RMSE Validation': [results['test_rmse_rf'], results['test_rmse_mlp'], results['test_rmse_stack']],
        'R² CV Mean': [results['cv_r2_mean_rf'], results['cv_r2_mean_mlp'], results['cv_r2_mean_stack']],
        'R² CV Std': [results['cv_r2_std_rf'], results['cv_r2_std_mlp'], results['cv_r2_std_stack']],
        'RMSE CV Mean': [results['cv_rmse_rf'], results['cv_rmse_mlp'], results['cv_rmse_stack']],
        'RMSE CV Std': [results['cv_rmse_std_rf'], results['cv_rmse_std_mlp'], results['cv_rmse_std_stack']]
    })
    comparison.to_csv(os.path.join(output_dir, f'model_comparison_{model_name}.csv'), index=False)
    
    return results

# Código principal
logging.info("Cargando datos...")
df = pd.read_csv('antioxidant18.csv')

categoria_deseada = 'Alto'
df_categoria = df[df['Category'] == categoria_deseada].copy()

if df_categoria.empty:
    logging.error(f"No se encontraron datos para la categoría '{categoria_deseada}'.")
    exit()

logging.info(f"Filas iniciales en df_categoria: {len(df_categoria)}")
df_categoria['Log10 Value (nM)'] = pd.to_numeric(df_categoria['Log10 Value (nM)'], errors='coerce')
logging.info(f"Filas después de conversión numérica: {len(df_categoria)}")
df_categoria = df_categoria.dropna(subset=['Log10 Value (nM)', 'Smiles'])
logging.info(f"Filas después de eliminar NaN: {len(df_categoria)}")

logging.info(f"Validando SMILES para '{categoria_deseada}'...")
smiles_list, valid_idx = validate_smiles(df_categoria['Smiles'].tolist())
logging.info(f"SMILES válidos: {len(smiles_list)}")
if not smiles_list:
    logging.error("No se encontraron SMILES válidos.")
    exit()

logging.info(f"Calculando descriptores con andamios de Murcko para {len(smiles_list)} SMILES válidos...")
X_with_murcko, valid_idx = get_descriptors_with_cache(smiles_list)

X_with_murcko = np.array(X_with_murcko)
y = df_categoria['Log10 Value (nM)'].iloc[valid_idx].values

logging.info(f"X_with_murcko shape: {X_with_murcko.shape}, y shape: {y.shape}")
if len(X_with_murcko) < 2 or len(y) < 2:
    logging.error("No hay suficientes datos válidos para la regresión.")
    exit()

logging.info("\nEntrenando modelos Random Forest, MLP y Stacking con andamios de Murcko...")
results = process_and_train_models(X_with_murcko, y, output_dir, "Alto")

logging.info(f"Resultados guardados en '{output_dir}'")
logging.info("Proceso completado exitosamente.")