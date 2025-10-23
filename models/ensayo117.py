# Ensayo 117 - Optimización de un modelo QSAR utilizando Random Forest y selección de características
# ANDAMIOS DE MURCKO Y ECFP
# Random Forest con selección de características Boruta y Mutual Information
# Categoria Medio - rango superior medio - potencia

import pandas as pd  # Importa pandas para manejo de datos
import numpy as np  # Importa numpy para cálculos numéricos
import random  # Importa random para generación de números aleatorios
import os  # Importa os para manejo de directorios y archivos
import matplotlib.pyplot as plt  # Importa matplotlib para visualización de gráficos
import joblib  # Importa joblib para guardar y cargar modelos
from rdkit import Chem  # Importa RDKit para manipulación de moléculas químicas
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Crippen, Lipinski  # Importa módulos de RDKit para descriptores moleculares
from rdkit.Chem.Scaffolds import MurckoScaffold  # Importa MurckoScaffold para calcular andamios de Murcko
from sklearn.ensemble import RandomForestRegressor  # Importa RandomForestRegressor para modelado
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score  # Importa herramientas de scikit-learn para división de datos y validación
from sklearn.metrics import r2_score, mean_squared_error  # Importa métricas para evaluar el modelo
from sklearn.preprocessing import StandardScaler  # Importa StandardScaler para escalado de características
from sklearn.impute import SimpleImputer  # Importa SimpleImputer para imputación de valores faltantes
from sklearn.feature_selection import SelectFromModel  # Importa SelectFromModel para selección de características
from sklearn.linear_model import LassoCV  # Importa LassoCV para selección de características con regularización
from boruta import BorutaPy  # Importa BorutaPy para selección avanzada de características
from sklearn.feature_selection import mutual_info_regression  # Importa mutual_info_regression para selección basada en información mutua
from concurrent.futures import ThreadPoolExecutor  # Importa ThreadPoolExecutor para procesamiento paralelo
import multiprocessing  # Importa multiprocessing para usar múltiples núcleos de CPU
import logging  # Importa logging para registro de eventos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Configura el sistema de registro con nivel INFO y formato con marca de tiempo

np.random.seed(42)  # Fija la semilla de numpy para reproducibilidad
random.seed(42)  # Fija la semilla de random para reproducibilidad

output_dir = "ensayo117"  # Define el nombre del directorio para guardar resultados
os.makedirs(output_dir, exist_ok=True)  # Crea el directorio "ensayo113", no genera error si ya existe

cache_file = os.path.join(output_dir, "descriptors_cache_with_murcko.joblib")  # Define la ruta del archivo de caché para descriptores
if os.path.exists(cache_file):  # Verifica si el archivo de caché existe
    os.remove(cache_file)  # Elimina el archivo de caché si existe
    logging.info(f"Cache eliminado: {cache_file}")  # Registra la eliminación del archivo de caché

def validate_smiles(smiles_list):  # Define función para validar cadenas SMILES
    valid_smiles = []  # Inicializa lista para almacenar SMILES válidos
    valid_idx = []  # Inicializa lista para almacenar índices de SMILES válidos
    for i, smiles in enumerate(smiles_list):  # Itera sobre la lista de SMILES con índices
        try:  # Inicia bloque try para manejar excepciones
            mol = Chem.MolFromSmiles(smiles)  # Convierte SMILES a objeto molécula
            if mol is not None and mol.GetNumAtoms() > 0:  # Verifica si la molécula es válida y tiene átomos
                valid_smiles.append(smiles)  # Agrega SMILES válido a la lista
                valid_idx.append(i)  # Agrega índice del SMILES válido
            else:  # Si la molécula no es válida
                logging.warning(f"SMILES inválido (mol None o 0 átomos): {smiles}")  # Registra advertencia para SMILES inválido
        except Exception as e:  # Captura cualquier excepción
            logging.warning(f"Error validando SMILES {smiles}: {e}")  # Registra advertencia con el error
            continue  # Continúa con el siguiente SMILES
    return valid_smiles, valid_idx  # Devuelve SMILES válidos e índices

def calculate_descriptors_batch(smiles_batch):  # Define función para calcular descriptores en lote
    results = []  # Inicializa lista para almacenar resultados de descriptores
    morgan_gen = AllChem.GetMorganGenerator(radius=3, fpSize=512)  # Crea generador de huellas Morgan (radio 3, 512 bits)
    descriptor_functions = [  # Define lista de tuplas con nombres y funciones de descriptores
        ('MolLogP', Crippen.MolLogP),  # Descriptor de hidrofobicidad
        ('TPSA', rdMolDescriptors.CalcTPSA),  # Área superficial polar topológica
        ('MolWt', Descriptors.MolWt),  # Peso molecular
        ('HeavyAtomCount', Descriptors.HeavyAtomCount),  # Conteo de átomos pesados
        ('RingCount', rdMolDescriptors.CalcNumRings),  # Número de anillos
        ('AromaticRingCount', rdMolDescriptors.CalcNumAromaticRings),  # Número de anillos aromáticos
        ('HBondDonors', Lipinski.NumHDonors),  # Donadores de enlaces de hidrógeno
        ('HBondAcceptors', Lipinski.NumHAcceptors),  # Aceptores de enlaces de hidrógeno
        ('NumRotatableBonds', rdMolDescriptors.CalcNumRotatableBonds),  # Enlaces rotables
        ('FractionCSP3', Descriptors.FractionCSP3)  # Fracción de carbonos sp3
    ]
    
    for smiles in smiles_batch:  # Itera sobre el lote de SMILES
        try:  # Inicia bloque try para manejar excepciones
            mol = Chem.MolFromSmiles(smiles)  # Convierte SMILES a objeto molécula
            if mol is None or mol.GetNumAtoms() == 0:  # Verifica si la molécula es válida
                logging.warning(f"SMILES inválido en descriptor: {smiles}")  # Registra advertencia para SMILES inválido
                results.append(None)  # Agrega None a los resultados
                continue  # Continúa con el siguiente SMILES
            
            fp = morgan_gen.GetFingerprint(mol)  # Calcula huellas Morgan para la molécula
            fp_arr = np.array(list(fp))  # Convierte huellas a arreglo numpy
            
            descriptors_dict = {}  # Inicializa diccionario para descriptores
            for name, func in descriptor_functions:  # Itera sobre funciones de descriptores
                try:  # Inicia bloque try para manejar excepciones
                    descriptors_dict[name] = func(mol)  # Calcula descriptor y lo almacena
                except Exception as e:  # Captura cualquier excepción
                    logging.warning(f"Error en descriptor {name} para {smiles}: {e}")  # Registra advertencia con el error
                    descriptors_dict[name] = 0.0  # Asigna 0.0 en caso de error
            
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)  # Obtiene el andamio de Murcko
            if scaffold is None or scaffold.GetNumAtoms() == 0:  # Verifica si el andamio es válido
                scaffold_fp = np.zeros(512)  # Crea huellas vacías (512 bits) si es inválido
                scaffold_descriptors = {  # Define descriptores de andamio vacíos
                    'Scaffold_MolWt': 0.0,  # Peso molecular del andamio
                    'Scaffold_HeavyAtomCount': 0,  # Conteo de átomos pesados del andamio
                    'Scaffold_RingCount': 0,  # Número de anillos del andamio
                    'Scaffold_AromaticRingCount': 0,  # Anillos aromáticos del andamio
                    'Scaffold_TPSA': 0.0  # Área superficial polar del andamio
                }
            else:  # Si el andamio es válido
                scaffold_fp = morgan_gen.GetFingerprint(scaffold)  # Calcula huellas Morgan del andamio
                scaffold_fp = np.array(list(scaffold_fp))  # Convierte huellas a arreglo numpy
                scaffold_descriptors = {  # Calcula descriptores del andamio
                    'Scaffold_MolWt': Descriptors.MolWt(scaffold),  # Peso molecular
                    'Scaffold_HeavyAtomCount': Descriptors.HeavyAtomCount(scaffold),  # Átomos pesados
                    'Scaffold_RingCount': rdMolDescriptors.CalcNumRings(scaffold),  # Anillos
                    'Scaffold_AromaticRingCount': rdMolDescriptors.CalcNumAromaticRings(scaffold),  # Anillos aromáticos
                    'Scaffold_TPSA': rdMolDescriptors.CalcTPSA(scaffold)  # Área superficial polar
                }
                
            descriptors_dict.update(scaffold_descriptors)  # Combina descriptores de molécula y andamio
            final_descriptors = np.concatenate([fp_arr, np.array([value for value in descriptors_dict.values()]), scaffold_fp])  # Concatena huellas y descriptores
            results.append(np.nan_to_num(final_descriptors, nan=0.0, posinf=0.0, neginf=0.0))  # Reemplaza valores inválidos por 0.0
        except Exception as e:  # Captura cualquier excepción
            logging.warning(f"Error en descriptor para SMILES {smiles}: {e}")  # Registra advertencia con el error
            results.append(None)  # Agrega None a los resultados
    return results  # Devuelve lista de descriptores calculados

def get_descriptors_with_cache(smiles_list):  # Define función para obtener descriptores con caché
    if os.path.exists(cache_file):  # Verifica si existe el archivo de caché
        cached_data = joblib.load(cache_file)  # Carga datos del caché
        cached_smiles = cached_data['smiles']  # Extrae SMILES del caché
        cached_descriptors = cached_data['descriptors']  # Extrae descriptores del caché
        smiles_to_calculate = [s for s in smiles_list if s not in cached_smiles]  # Identifica SMILES nuevos para calcular
        logging.info(f"Usando caché para {len(cached_smiles)} SMILES, calculando {len(smiles_to_calculate)} nuevos.")  # Registra uso del caché
    else:  # Si no existe caché
        cached_smiles = []  # Inicializa lista vacía de SMILES en caché
        cached_descriptors = []  # Inicializa lista vacía de descriptores en caché
        smiles_to_calculate = smiles_list  # Todos los SMILES deben calcularse
        logging.info(f"No se encontró caché, calculando {len(smiles_to_calculate)} SMILES.")  # Registra que no hay caché
    
    X_list = []  # Inicializa lista para descriptores válidos
    valid_idx = []  # Inicializa lista para índices válidos
    failed_smiles = []  # Inicializa lista para SMILES fallidos
    
    if smiles_to_calculate:  # Si hay SMILES para calcular
        n_cores = max(1, multiprocessing.cpu_count() - 1)  # Determina número de núcleos disponibles
        batch_size = max(1, len(smiles_to_calculate) // (n_cores * 4))  # Calcula tamaño de lote para procesamiento paralelo
        batches = [smiles_to_calculate[i:i+batch_size] for i in range(0, len(smiles_to_calculate), batch_size)]  # Divide SMILES en lotes
        
        all_new_descriptors = []  # Inicializa lista para nuevos descriptores
        with ThreadPoolExecutor(max_workers=n_cores) as executor:  # Inicia executor para procesamiento paralelo
            batch_results = list(executor.map(calculate_descriptors_batch, batches))  # Procesa lotes en paralelo
        
        for batch_result, batch_smiles in zip(batch_results, batches):  # Itera sobre resultados y lotes
            for smiles, desc in zip(batch_smiles, batch_result):  # Itera sobre SMILES y descriptores
                if desc is not None:  # Si el descriptor es válido
                    valid_idx.append(smiles_list.index(smiles))  # Agrega índice del SMILES válido
                    X_list.append(desc)  # Agrega descriptor a la lista
                    cached_smiles.append(smiles)  # Agrega SMILES al caché
                    cached_descriptors.append(desc)  # Agrega descriptor al caché
                else:  # Si el descriptor no es válido
                    failed_smiles.append(smiles)  # Agrega SMILES a la lista de fallidos
        
        if failed_smiles:  # Si hay SMILES fallidos
            logging.warning(f"Fallaron {len(failed_smiles)} SMILES al calcular descriptores: {failed_smiles[:5]}...")  # Registra advertencia con SMILES fallidos

    logging.info(f"Total descriptores válidos calculados: {len(X_list)}")  # Registra número de descriptores válidos
    
    joblib.dump({'smiles': cached_smiles, 'descriptors': cached_descriptors}, cache_file)  # Guarda SMILES y descriptores en caché
    
    return np.array(X_list), valid_idx  # Devuelve matriz de descriptores e índices válidos

def boruta_feature_selection(X, y):  # Define función para selección de características con Boruta
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)  # Crea modelo Random Forest para Boruta
    boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=100)  # Configura BorutaPy
    boruta.fit(X, y)  # Ajusta Boruta a los datos
    return boruta.support_  # Devuelve máscara de características seleccionadas

def mutual_info_selection(X, y, n_features=50):  # Define función para selección con Mutual Information
    mi_scores = mutual_info_regression(X, y, random_state=42)  # Calcula puntajes de información mutua
    top_features = np.argsort(mi_scores)[-n_features:]  # Selecciona los índices de las n_features con mayor puntaje
    mask = np.zeros(X.shape[1], dtype=bool)  # Crea máscara inicial de ceros
    mask[top_features] = True  # Marca las características seleccionadas como True
    return mask  # Devuelve máscara de características seleccionadas

def process_and_train_rf_model(X, y, output_dir, model_name):  # Define función para procesar y entrenar modelo Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  # Divide datos en entrenamiento (75%) y prueba (25%)
    logging.info(f"Train set: {len(y_train)} muestras, Test set: {len(y_test)} muestras.")  # Registra tamaño de los conjuntos
    
    imputer = SimpleImputer(strategy='mean')  # Crea imputador para valores faltantes con estrategia de media
    X_train = imputer.fit_transform(X_train)  # Ajusta y transforma datos de entrenamiento
    X_test = imputer.transform(X_test)  # Transforma datos de prueba
    
    scaler = StandardScaler()  # Crea escalador para estandarizar características
    X_train_scaled = scaler.fit_transform(X_train)  # Ajusta y transforma datos de entrenamiento
    X_test_scaled = scaler.transform(X_test)  # Transforma datos de prueba
    
    combined_mask = np.ones(X_train_scaled.shape[1], dtype=bool)  # Inicializa máscara para selección de características
    
    boruta_mask = boruta_feature_selection(X_train_scaled, y_train)  # Aplica selección de características con Boruta
    combined_mask = np.logical_and(combined_mask, boruta_mask)  # Combina máscara con Boruta (AND lógico)
    logging.info(f"Características seleccionadas por Boruta: {np.sum(boruta_mask)}")  # Registra número de características seleccionadas por Boruta
    
    plt.figure(figsize=(10, 6))  # Crea figura para gráfico de Boruta
    plt.bar(range(len(boruta_mask)), boruta_mask)  # Grafica máscara de Boruta
    plt.title(f'Boruta Feature Selection - {model_name}')  # Establece título del gráfico
    plt.xlabel('Feature Index')  # Etiqueta eje X
    plt.ylabel('Selected (1) / Not Selected (0)')  # Etiqueta eje Y
    plt.savefig(os.path.join(output_dir, f'boruta_features_{model_name}.png'))  # Guarda gráfico de Boruta
    plt.close()  # Cierra la figura
    
    mi_mask = mutual_info_selection(X_train_scaled, y_train)  # Aplica selección con Mutual Information
    combined_mask = np.logical_and(combined_mask, mi_mask)  # Combina máscara con Mutual Information (AND lógico)
    logging.info(f"Características seleccionadas por Mutual Information: {np.sum(mi_mask)}")  # Registra número de características seleccionadas por MI
    
    mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)  # Calcula puntajes de Mutual Information
    plt.figure(figsize=(10, 6))  # Crea figura para gráfico de MI
    plt.bar(range(len(mi_scores)), mi_scores)  # Grafica puntajes de MI
    plt.title(f'Mutual Information Scores - {model_name}')  # Establece título del gráfico
    plt.xlabel('Feature Index')  # Etiqueta eje X
    plt.ylabel('Mutual Information Score')  # Etiqueta eje Y
    plt.savefig(os.path.join(output_dir, f'mi_features_{model_name}.png'))  # Guarda gráfico de MI
    plt.close()  # Cierra la figura
    
    initial_rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)  # Crea modelo Random Forest inicial
    initial_rf.fit(X_train_scaled, y_train)  # Ajusta modelo inicial a datos de entrenamiento
    sfm = SelectFromModel(initial_rf, threshold="mean", prefit=True)  # Crea selector de características basado en importancia media
    sfm_mask = sfm.get_support()  # Obtiene máscara de características seleccionadas por Random Forest
    
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=20000, n_jobs=-1)  # Crea modelo LassoCV para selección de características
    lasso_cv.fit(X_train_scaled, y_train)  # Ajusta LassoCV a datos de entrenamiento
    lasso_mask = lasso_cv.coef_ != 0  # Crea máscara de características no nulas
    
    combined_mask = np.logical_or(combined_mask, sfm_mask, lasso_mask)  # Combina máscaras con OR lógico
    X_train_selected = X_train_scaled[:, combined_mask]  # Selecciona características en datos de entrenamiento
    X_test_selected = X_test_scaled[:, combined_mask]  # Selecciona características en datos de prueba
    
    logging.info(f"Total características seleccionadas para {model_name}: {X_train_selected.shape[1]} de {X_train_scaled.shape[1]}")  # Registra número de características seleccionadas
    
    joblib.dump(combined_mask, os.path.join(output_dir, f'selected_features_{model_name}.joblib'))  # Guarda máscara combinada
    joblib.dump(sfm, os.path.join(output_dir, f'sfm_selector_{model_name}.joblib'))  # Guarda selector de Random Forest
    joblib.dump(lasso_cv, os.path.join(output_dir, f'lasso_selector_{model_name}.joblib'))  # Guarda selector de Lasso
    joblib.dump(scaler, os.path.join(output_dir, f'scaler_{model_name}.joblib'))  # Guarda escalador
    joblib.dump(imputer, os.path.join(output_dir, f'imputer_{model_name}.joblib'))  # Guarda imputador
    
    param_grid_rf = {  # Define cuadrícula de hiperparámetros para GridSearchCV
        'n_estimators': [150, 250, 300],  # Número de árboles
        'max_depth': [8, 12, None],  # Profundidad máxima
        'min_samples_split': [2, 3, 5],  # Mínimo de muestras para dividir un nodo
        'min_samples_leaf': [1, 2],  # Mínimo de muestras por hoja
        'max_features': ['sqrt', 0.7]  # Número de características por división
    }
    grid_search_rf = GridSearchCV(  # Crea objeto GridSearchCV para optimización
        RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=True),  # Modelo base
        param_grid_rf,  # Cuadrícula de hiperparámetros
        cv=5,  # Validación cruzada de 5 pliegues
        scoring='neg_mean_squared_error',  # Métrica de evaluación
        n_jobs=-1,  # Usa todos los núcleos disponibles
        verbose=1  # Muestra progreso
    )
    grid_search_rf.fit(X_train_selected, y_train)  # Ajusta GridSearchCV a datos seleccionados
    
    final_rf = RandomForestRegressor(**grid_search_rf.best_params_, random_state=42, n_jobs=-1, bootstrap=True)  # Crea modelo final con mejores hiperparámetros
    final_rf.fit(X_train_selected, y_train)  # Ajusta modelo final a datos de entrenamiento
    
    y_train_pred_rf = final_rf.predict(X_train_selected)  # Genera predicciones para entrenamiento
    y_test_pred_rf = final_rf.predict(X_test_selected)  # Genera predicciones para prueba
    train_r2_rf = r2_score(y_train, y_train_pred_rf)  # Calcula R² para entrenamiento
    test_r2_rf = r2_score(y_test, y_test_pred_rf)  # Calcula R² para prueba
    train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))  # Calcula RMSE para entrenamiento
    test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))  # Calcula RMSE para prueba
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)  # Configura validación cruzada de 5 pliegues
    cv_scores_r2_rf = cross_val_score(final_rf, X_train_selected, y_train, cv=cv, scoring='r2', n_jobs=-1)  # Calcula R² en validación cruzada
    cv_scores_mse_rf = -cross_val_score(final_rf, X_train_selected, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)  # Calcula MSE en validación cruzada
    cv_scores_rmse_rf = np.sqrt(cv_scores_mse_rf)  # Convierte MSE a RMSE
    
    logging.info(f"\nResultados del modelo {model_name} (Random Forest):")  # Registra inicio de resultados
    logging.info(f"R² Training: {train_r2_rf:.3f}")  # Registra R² de entrenamiento
    logging.info(f"RMSE Training: {train_rmse_rf:.3f}")  # Registra RMSE de entrenamiento
    logging.info(f"R² Validation: {test_r2_rf:.3f}")  # Registra R² de prueba
    logging.info(f"RMSE Validation: {test_rmse_rf:.3f}")  # Registra RMSE de prueba
    logging.info(f"R² CV 5-fold: {cv_scores_r2_rf.mean():.3f} ± {cv_scores_r2_rf.std():.3f}")  # Registra R² promedio de validación cruzada
    logging.info(f"RMSE CV 5-fold: {cv_scores_rmse_rf.mean():.3f} ± {cv_scores_rmse_rf.std():.3f}")  # Registra RMSE promedio de validación cruzada
    
    joblib.dump(final_rf, os.path.join(output_dir, f'rf_model_{model_name}.joblib'))  # Guarda modelo Random Forest
    
    plt.figure(figsize=(10, 8))  # Crea figura para gráfico de dispersión
    plt.scatter(y_train, y_train_pred_rf, c='blue', label=f'Training: R²={train_r2_rf:.3f}, RMSE={train_rmse_rf:.3f}', alpha=0.6)  # Grafica datos de entrenamiento
    plt.scatter(y_test, y_test_pred_rf, c='red', label=f'Validation: R²={test_r2_rf:.3f}, RMSE={test_rmse_rf:.3f}', alpha=0.6)  # Grafica datos de prueba
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Ideal line (Y=X)')  # Grafica línea ideal
    plt.xlabel('Observed Antioxidant Activity (Log10 Value (nM))')  # Etiqueta eje X
    plt.ylabel('Predicted')  # Etiqueta eje Y
    plt.title(f'QSAR Random Forest Model - {model_name}')  # Establece título del gráfico
    plt.legend()  # Muestra leyenda
    plt.grid(True)  # Agrega cuadrícula
    plt.savefig(os.path.join(output_dir, f'rf_model_plot_{model_name}.png'), dpi=150)  # Guarda gráfico
    plt.close()  # Cierra la figura
    
    results = {  # Crea diccionario con resultados
        'model_name': model_name,  # Nombre del modelo
        'train_r2_rf': train_r2_rf,  # R² de entrenamiento
        'test_r2_rf': test_r2_rf,  # R² de prueba
        'train_rmse_rf': train_rmse_rf,  # RMSE de entrenamiento
        'test_rmse_rf': test_rmse_rf,  # RMSE de prueba
        'cv_r2_mean_rf': cv_scores_r2_rf.mean(),  # Promedio de R² en validación cruzada
        'cv_r2_std_rf': cv_scores_r2_rf.std(),  # Desviación estándar de R²
        'cv_rmse_mean_rf': cv_scores_rmse_rf.mean(),  # Promedio de RMSE en validación cruzada
        'cv_rmse_std_rf': cv_scores_rmse_rf.std()  # Desviación estándar de RMSE
    }
    
    comparison = pd.DataFrame({  # Crea DataFrame para resultados
        'Modelo': ['Random Forest'],  # Nombre del modelo
        'R² Training': [results['train_r2_rf']],  # R² de entrenamiento
        'RMSE Training': [results['train_rmse_rf']],  # RMSE de entrenamiento
        'R² Validation': [results['test_r2_rf']],  # R² de prueba
        'RMSE Validation': [results['test_rmse_rf']],  # RMSE de prueba
        'R² CV Mean': [results['cv_r2_mean_rf']],  # Promedio de R² en validación cruzada
        'R² CV Std': [results['cv_r2_std_rf']],  # Desviación estándar de R²
        'RMSE CV Mean': [results['cv_rmse_mean_rf']],  # Promedio de RMSE en validación cruzada
        'RMSE CV Std': [results['cv_rmse_std_rf']]  # Desviación estándar de RMSE
    })
    comparison.to_csv(os.path.join(output_dir, f'model_comparison_{model_name}.csv'), index=False)  # Guarda resultados en CSV
    
    return results  # Devuelve diccionario de resultados

logging.info("Cargando datos...")  # Registra inicio de carga de datos
df = pd.read_csv('antioxidant15.csv')  # Carga archivo CSV con datos moleculares

categoria_deseada = 'Medio'  # Define categoría deseada
subcategoria_deseada = 'rango superior medio - potencia'  # Define subcategoría deseada
df_categoria = df[df['Category'] == categoria_deseada].copy()  # Filtra datos por categoría
df_subcategoria = df_categoria[df_categoria['Subcategory'] == subcategoria_deseada].copy()  # Filtra datos por subcategoría

if df_subcategoria.empty:  # Verifica si hay datos en la subcategoría
    logging.error(f"No se encontraron datos para la categoría '{categoria_deseada}' y subcategoría '{subcategoria_deseada}'.")  # Registra error si no hay datos
    exit()  # Termina ejecución

logging.info(f"Filas iniciales en df_subcategoria: {len(df_subcategoria)}")  # Registra número inicial de filas
df_subcategoria['Log10 Value (nM)'] = pd.to_numeric(df_subcategoria['Log10 Value (nM)'], errors='coerce')  # Convierte columna de actividad a numérica
logging.info(f"Filas después de conversión numérica: {len(df_subcategoria)}")  # Registra filas tras conversión
df_subcategoria = df_subcategoria.dropna(subset=['Log10 Value (nM)', 'Smiles'])  # Elimina filas con valores faltantes
logging.info(f"Filas después de eliminar NaN: {len(df_subcategoria)}")  # Registra filas tras eliminar NaN

logging.info(f"Validando SMILES para '{categoria_deseada} - {subcategoria_deseada}'...")  # Registra inicio de validación de SMILES
smiles_list, valid_idx = validate_smiles(df_subcategoria['Smiles'].tolist())  # Valida SMILES de la subcategoría
logging.info(f"SMILES válidos: {len(smiles_list)}")  # Registra número de SMILES válidos
if not smiles_list:  # Verifica si hay SMILES válidos
    logging.error("No se encontraron SMILES válidos.")  # Registra error si no hay SMILES válidos
    exit()  # Termina ejecución

logging.info(f"Calculando descriptores con andamios de Murcko para {len(smiles_list)} SMILES válidos...")  # Registra inicio de cálculo de descriptores
X_with_murcko, valid_idx = get_descriptors_with_cache(smiles_list)  # Calcula descriptores usando caché

X_with_murcko = np.array(X_with_murcko)  # Convierte descriptores a arreglo numpy
y = df_subcategoria['Log10 Value (nM)'].iloc[valid_idx].values  # Extrae valores objetivo correspondientes a SMILES válidos

logging.info(f"X_with_murcko shape: {X_with_murcko.shape}, y shape: {y.shape}")  # Registra dimensiones de matrices
if len(X_with_murcko) < 2 or len(y) < 2:  # Verifica si hay suficientes datos
    logging.error("No hay suficientes datos válidos para la regresión.")  # Registra error si no hay suficientes datos
    exit()  # Termina ejecución

logging.info("\nEntrenando modelo Random Forest con andamios de Murcko...")  # Registra inicio de entrenamiento
results_rf = process_and_train_rf_model(X_with_murcko, y, output_dir, "With_Murcko")  # Entrena y evalúa modelo

logging.info(f"Resultados guardados en '{output_dir}'")  # Registra ubicación de resultados guardados
logging.info("Proceso completado exitosamente.")  # Registra finalización exitosa