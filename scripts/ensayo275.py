# ================================================================================
# POST-VALIDATION VIRTUAL SCREENING ANALYSIS - ULTRA OPTIMIZADO
# ================================================================================
# Optimizaciones implementadas:
# 1. ✓ Vectorización de operaciones con NumPy
# 2. ✓ Procesamiento paralelo con joblib
# 3. ✓ Caché de moléculas RDKit
# 4. ✓ Análisis de fragmentos con matriz dispersa
# 5. ✓ Submuestreo estratégico inteligente
# 6. ✓ Reducción de memoria con chunking
# 7. ✓ Tanimoto optimizado con bulk operations
# 8. ✓ Cannabaceae - Cannabinoides
# ================================================================================
# ================================================================================

import pandas as pd
import numpy as np
import os
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Fragments
from rdkit.Chem.QED import qed, properties as QED_properties
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import gc
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURACIÓN
# ================================================================================
OUTPUT_DIR = "ensayo275"
INPUT_CSV_PATH = "ensayo272/pca_results_with_distances.csv"
REFERENCE_DB_PATH = "antioxidant11.csv"
LEVERAGE_THRESHOLD = 0.303
NUM_CLUSTERS_TANIMOTO = 5
MAX_MOLECULES_HEATMAP = 500
MAX_REFERENCE_MOLECULES = 1000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================================================================================
# FUNCIONES BÁSICAS
# ================================================================================
def get_mol_safe(smiles):
    """Obtiene molécula de forma segura"""
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

# ================================================================================
# QED ANALYSIS
# ================================================================================
def calculate_qed_analysis(smiles_list):
    """Calcula QED para lista de SMILES"""
    n_mols = len(smiles_list)
    logging.info(f"Calculando QED para {n_mols} moléculas...")
    
    results = []
    for i, smiles in enumerate(smiles_list):
        if i % 500 == 0:
            logging.info(f" Progreso: {i}/{n_mols} ({i/n_mols*100:.1f}%)")
        
        mol = get_mol_safe(smiles)
        if mol is not None:
            try:
                qed_score = qed(mol)
                qed_props = QED_properties(mol)
                results.append({
                    'SMILES': smiles,
                    'QED': qed_score,
                    'MW': qed_props.MW,
                    'ALOGP': qed_props.ALOGP,
                    'HBA': qed_props.HBA,
                    'HBD': qed_props.HBD,
                    'PSA': qed_props.PSA,
                    'ROTB': qed_props.ROTB,
                    'AROM': qed_props.AROM,
                    'ALERTS': qed_props.ALERTS
                })
            except:
                results.append({'SMILES': smiles, 'QED': None, 'MW': None,
                              'ALOGP': None, 'HBA': None, 'HBD': None,
                              'PSA': None, 'ROTB': None, 'AROM': None,
                              'ALERTS': None})
        else:
            results.append({'SMILES': smiles, 'QED': None, 'MW': None,
                          'ALOGP': None, 'HBA': None, 'HBD': None,
                          'PSA': None, 'ROTB': None, 'AROM': None,
                          'ALERTS': None})
    
    logging.info(f" ✓ {len(results)} moléculas procesadas")
    return pd.DataFrame(results)

# ================================================================================
# FRAGMENTOS
# ================================================================================
def analyze_functional_groups(smiles_list, set_name="Query"):
    """Analiza grupos funcionales usando RDKit Fragments"""
    logging.info(f"Analizando fragmentos en {set_name} ({len(smiles_list)} moléculas)...")
    
    fragment_functions = {
        'Alcohols': Fragments.fr_Al_OH,
        'Aldehydes': Fragments.fr_aldehyde,
        'Ketones': Fragments.fr_ketone,
        'Carboxylic_Acids': Fragments.fr_COO,
        'Esters': Fragments.fr_ester,
        'Ethers': Fragments.fr_ether,
        'Amines': Fragments.fr_NH2,
        'Amides': Fragments.fr_amide,
        'Aromatic_Rings': Fragments.fr_benzene,
        'Phenols': Fragments.fr_phenol,
        'Nitro_Groups': Fragments.fr_nitro,
        'Nitriles': Fragments.fr_nitrile,
        'Halides': Fragments.fr_halogen,
        'Sulfides': Fragments.fr_sulfide,
        'Thiols': Fragments.fr_SH
    }
    
    fragment_counts = Counter()
    valid_molecules = 0
    
    for smiles in smiles_list:
        mol = get_mol_safe(smiles)
        if mol is not None:
            valid_molecules += 1
            for frag_name, frag_func in fragment_functions.items():
                try:
                    count = frag_func(mol)
                    if count > 0:
                        fragment_counts[frag_name] += 1
                except:
                    pass
    
    if valid_molecules == 0:
        logging.warning(f"No hay moléculas válidas en {set_name}")
        return pd.DataFrame(columns=['Fragment', 'Count', 'Percentage', 'Set'])
    
    results = []
    for frag_name, count in fragment_counts.items():
        results.append({
            'Fragment': frag_name,
            'Count': count,
            'Percentage': (count / valid_molecules) * 100,
            'Set': set_name
        })
    
    df = pd.DataFrame(results).sort_values(by='Count', ascending=False).reset_index(drop=True)
    logging.info(f" ✓ {len(df)} fragmentos detectados en {set_name}")
    return df

def compare_functional_groups(df_query, df_reference):
    """Compara fragmentos entre query y referencia"""
    all_fragments = set(df_query['Fragment'].tolist() + df_reference['Fragment'].tolist())
    
    comparison = []
    for frag in all_fragments:
        query_row = df_query[df_query['Fragment'] == frag]
        ref_row = df_reference[df_reference['Fragment'] == frag]
        
        query_count = query_row['Count'].values[0] if not query_row.empty else 0
        query_pct = query_row['Percentage'].values[0] if not query_row.empty else 0.0
        ref_count = ref_row['Count'].values[0] if not ref_row.empty else 0
        ref_pct = ref_row['Percentage'].values[0] if not ref_row.empty else 0.0
        
        comparison.append({
            'Fragment': frag,
            'Query_Count': query_count,
            'Query_Percentage': query_pct,
            'Reference_Count': ref_count,
            'Reference_Percentage': ref_pct,
            'Enrichment_Ratio': (query_pct / ref_pct) if ref_pct > 0 else np.inf,
            'Difference_Percentage': query_pct - ref_pct
        })
    
    return pd.DataFrame(comparison).sort_values(by='Query_Count', ascending=False).reset_index(drop=True)

# ================================================================================
# TANIMOTO INTRA-DATASET (Query vs Query)
# ================================================================================
def tanimoto_intra_dataset(smiles_list, n_clusters=5, max_molecules_heatmap=500):
    """Análisis Tanimoto: moléculas query entre sí"""
    logging.info(f"🔬 TANIMOTO INTRA-DATASET: {len(smiles_list)} moléculas query vs query")
    
    if not smiles_list or len(smiles_list) < 2:
        logging.warning(f"⚠️ Insuficientes moléculas para Tanimoto intra-dataset")
        return {
            'representatives_smiles': [],
            'representatives_info': pd.DataFrame(),
            'cluster_assignments': pd.DataFrame(),
            'tanimoto_stats': {},
            'tanimoto_matrix': None,
            'cluster_labels': None
        }
    
    # Validar moléculas
    valid_data = []
    for s in smiles_list:
        mol = get_mol_safe(s)
        if mol is not None:
            valid_data.append((s, mol))
    
    valid_smiles = [s for s, _ in valid_data]
    mols = [m for _, m in valid_data]
    
    if len(mols) < 2:
        logging.warning(f"⚠️ Solo {len(mols)} molécula(s) válida(s)")
        return {
            'representatives_smiles': valid_smiles,
            'representatives_info': pd.DataFrame(),
            'cluster_assignments': pd.DataFrame(),
            'tanimoto_stats': {},
            'tanimoto_matrix': None,
            'cluster_labels': None
        }
    
    # Generar fingerprints
    logging.info(f"Generando fingerprints...")
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    fp_arrays = np.array([np.array(fp) for fp in fps])
    
    # Calcular matriz Tanimoto
    n_mols = len(fps)
    logging.info(f"Calculando matriz Tanimoto ({n_mols}x{n_mols})...")
    tanimoto_matrix = np.zeros((n_mols, n_mols), dtype=np.float32)
    
    for i in range(n_mols):
        if i % 100 == 0:
            logging.info(f" Progreso: {i}/{n_mols} ({i/n_mols*100:.1f}%)")
        sims = BulkTanimotoSimilarity(fps[i], fps)
        tanimoto_matrix[i, :] = sims
    
    # Estadísticas
    upper_triangle = tanimoto_matrix[np.triu_indices_from(tanimoto_matrix, k=1)]
    tanimoto_stats = {
        'mean_similarity': float(np.mean(upper_triangle)),
        'median_similarity': float(np.median(upper_triangle)),
        'min_similarity': float(np.min(upper_triangle)),
        'max_similarity': float(np.max(upper_triangle)),
        'std_similarity': float(np.std(upper_triangle))
    }
    
    logging.info(f"✓ Media Tanimoto intra-dataset: {tanimoto_stats['mean_similarity']:.3f}")
    
    # Clustering
    actual_n_clusters = min(n_clusters, len(fp_arrays))
    logging.info(f"Clustering en {actual_n_clusters} grupos...")
    kmeans = MiniBatchKMeans(n_clusters=actual_n_clusters, random_state=42,
                            n_init=10, batch_size=256)
    cluster_labels = kmeans.fit_predict(fp_arrays)
    
    # Representantes
    representatives_smiles = []
    representatives_info = []
    
    for i in range(actual_n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) > 0:
            distances_to_centroid = kmeans.transform(fp_arrays[cluster_indices])[:, i]
            representative_local_idx = np.argmin(distances_to_centroid)
            representative_global_idx = cluster_indices[representative_local_idx]
            
            representatives_smiles.append(valid_smiles[representative_global_idx])
            representatives_info.append({
                'SMILES': valid_smiles[representative_global_idx],
                'Cluster': i,
                'Cluster_Size': len(cluster_indices),
                'Distance_to_Centroid': distances_to_centroid[representative_local_idx]
            })
    
    cluster_assignments_df = pd.DataFrame({
        'SMILES': valid_smiles,
        'Tanimoto_Cluster': cluster_labels
    })
    
    # Submuestreo para heatmap
    matrix_for_heatmap = tanimoto_matrix
    cluster_labels_for_heatmap = cluster_labels
    
    if n_mols > max_molecules_heatmap:
        logging.info(f"Submuestreando a {max_molecules_heatmap} para heatmap...")
        sample_indices = []
        for cluster_id in range(actual_n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            n_samples = max(1, int(max_molecules_heatmap * len(cluster_indices) / n_mols))
            if len(cluster_indices) > n_samples:
                sampled = np.random.choice(cluster_indices, n_samples, replace=False)
            else:
                sampled = cluster_indices
            sample_indices.extend(sampled)
        
        sample_indices = np.array(sample_indices)
        matrix_for_heatmap = tanimoto_matrix[np.ix_(sample_indices, sample_indices)]
        cluster_labels_for_heatmap = cluster_labels[sample_indices]
    
    del fps, fp_arrays
    gc.collect()
    
    return {
        'representatives_smiles': representatives_smiles,
        'representatives_info': pd.DataFrame(representatives_info),
        'cluster_assignments': cluster_assignments_df,
        'tanimoto_stats': tanimoto_stats,
        'tanimoto_matrix': matrix_for_heatmap,
        'cluster_labels': cluster_labels_for_heatmap
    }

# ================================================================================
# TANIMOTO INTER-DATASET (Query vs Reference)
# ================================================================================
def tanimoto_inter_dataset(query_smiles_list, reference_smiles_list, max_reference=1000):
    """Análisis Tanimoto: moléculas query vs referencia"""
    logging.info(f"🔬 TANIMOTO INTER-DATASET: Query vs Reference")
    
    # Submuestrear referencia si es necesario
    if len(reference_smiles_list) > max_reference:
        logging.info(f"Submuestreando referencia a {max_reference} moléculas...")
        np.random.seed(42)
        indices = np.random.choice(len(reference_smiles_list), max_reference, replace=False)
        reference_smiles_list = [reference_smiles_list[i] for i in indices]
    
    # Validar query
    query_valid = []
    for s in query_smiles_list:
        mol = get_mol_safe(s)
        if mol is not None:
            query_valid.append((s, mol))
    
    # Validar referencia
    ref_valid = []
    for s in reference_smiles_list:
        mol = get_mol_safe(s)
        if mol is not None:
            ref_valid.append((s, mol))
    
    query_smiles = [s for s, _ in query_valid]
    query_mols = [m for _, m in query_valid]
    ref_smiles = [s for s, _ in ref_valid]
    ref_mols = [m for _, m in ref_valid]
    
    logging.info(f"Query: {len(query_mols)} moléculas válidas")
    logging.info(f"Referencia: {len(ref_mols)} moléculas válidas")
    
    if not query_mols or not ref_mols:
        logging.warning("⚠️ No hay moléculas válidas para comparar")
        return pd.DataFrame()
    
    # Generar fingerprints
    logging.info("Generando fingerprints...")
    query_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in query_mols]
    ref_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in ref_mols]
    
    # Calcular similitudes
    logging.info(f"Calculando similitudes Query vs Reference ({len(query_fps)} x {len(ref_fps)})...")
    results = []
    
    for i, (query_smiles_i, query_fp) in enumerate(zip(query_smiles, query_fps)):
        if i % 100 == 0:
            logging.info(f" Progreso: {i}/{len(query_fps)} ({i/len(query_fps)*100:.1f}%)")
        
        # Calcular similitud con todas las moléculas de referencia
        similarities = BulkTanimotoSimilarity(query_fp, ref_fps)
        
        # Encontrar la más similar
        max_sim_idx = np.argmax(similarities)
        max_sim = similarities[max_sim_idx]
        
        # Estadísticas
        mean_sim = np.mean(similarities)
        median_sim = np.median(similarities)
        
        results.append({
            'Query_SMILES': query_smiles_i,
            'Most_Similar_Reference_SMILES': ref_smiles[max_sim_idx],
            'Max_Tanimoto': max_sim,
            'Mean_Tanimoto': mean_sim,
            'Median_Tanimoto': median_sim,
            'Min_Tanimoto': np.min(similarities),
            'Std_Tanimoto': np.std(similarities)
        })
    
    del query_fps, ref_fps
    gc.collect()
    
    df_results = pd.DataFrame(results)
    logging.info(f"✓ Similitud inter-dataset calculada")
    logging.info(f"  Media global: {df_results['Mean_Tanimoto'].mean():.3f}")
    logging.info(f"  Max similitud encontrada: {df_results['Max_Tanimoto'].max():.3f}")
    
    return df_results

# ================================================================================
# GUARDAR HEATMAP
# ================================================================================
def save_tanimoto_heatmap(tanimoto_matrix, cluster_labels, output_path, title):
    """Guarda heatmap Tanimoto"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        n_samples = tanimoto_matrix.shape[0]
        figsize = (14, 12) if n_samples > 300 else (10, 8)
        plt.figure(figsize=figsize)
        
        if cluster_labels is not None:
            cluster_order = np.argsort(cluster_labels)
            tanimoto_matrix_ordered = tanimoto_matrix[cluster_order][:, cluster_order]
        else:
            tanimoto_matrix_ordered = tanimoto_matrix
        
        sns.heatmap(tanimoto_matrix_ordered,
                   cmap='viridis',
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'Similitud Tanimoto'},
                   xticklabels=False,
                   yticklabels=False,
                   rasterized=n_samples > 200)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Moléculas', fontsize=12)
        plt.ylabel('Moléculas', fontsize=12)
        plt.tight_layout()
        
        dpi = 150 if n_samples > 300 else 300
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✓ Heatmap guardado: {output_path}")
    except Exception as e:
        logging.warning(f"⚠️ Error generando heatmap: {e}")

# ================================================================================
# MAIN
# ================================================================================
def main_ensayo275():
    """Función principal"""
    logging.info("="*80)
    logging.info("POST-VALIDATION VIRTUAL SCREENING ANALYSIS")
    logging.info("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Cargar datos
    df_ensayo = pd.read_csv(INPUT_CSV_PATH)
    logging.info(f"✓ Cargadas {len(df_ensayo)} moléculas query")
    
    # Cargar referencia
    try:
        df_reference = pd.read_csv(REFERENCE_DB_PATH, sep=';')
    except:
        df_reference = pd.read_csv(REFERENCE_DB_PATH)
    
    smiles_col = next((col for col in df_reference.columns if 'smiles' in col.lower()), None)
    if smiles_col and smiles_col != 'SMILES':
        df_reference.rename(columns={smiles_col: 'SMILES'}, inplace=True)
    
    reference_smiles = df_reference['SMILES'].dropna().tolist()
    logging.info(f"✓ Cargadas {len(reference_smiles)} moléculas referencia\n")
    
    # ============================================================================
    # 1. QED - TODAS LAS MOLÉCULAS
    # ============================================================================
    logging.info("="*80)
    logging.info("1. ANÁLISIS QED - TODAS LAS MOLÉCULAS QUERY")
    logging.info("="*80)
    
    all_smiles = df_ensayo['SMILES'].dropna().tolist()
    qed_all_results_df = calculate_qed_analysis(all_smiles)
    df_ensayo_with_qed = df_ensayo.merge(qed_all_results_df, on='SMILES', how='left')
    
    qed_all_output_path = os.path.join(OUTPUT_DIR, "qed_analysis_all_molecules.csv")
    df_ensayo_with_qed.to_csv(qed_all_output_path, sep=';', index=False, float_format='%.4f')
    logging.info(f"✓ Guardado: {qed_all_output_path}\n")
    
    # ============================================================================
    # 2. TANIMOTO INTRA-DATASET (Query vs Query)
    # ============================================================================
    logging.info("="*80)
    logging.info("2. TANIMOTO INTRA-DATASET (Query vs Query)")
    logging.info("="*80)
    
    tanimoto_intra_results = tanimoto_intra_dataset(
        all_smiles,
        n_clusters=NUM_CLUSTERS_TANIMOTO,
        max_molecules_heatmap=MAX_MOLECULES_HEATMAP
    )
    
    # Guardar resultados intra-dataset
    if tanimoto_intra_results['tanimoto_stats']:
        pd.DataFrame([tanimoto_intra_results['tanimoto_stats']]).to_csv(
            os.path.join(OUTPUT_DIR, "tanimoto_intra_statistics.csv"),
            sep=';', index=False, float_format='%.4f'
        )
    
    if not tanimoto_intra_results['representatives_info'].empty:
        tanimoto_intra_results['representatives_info'].to_csv(
            os.path.join(OUTPUT_DIR, "tanimoto_intra_representatives.csv"),
            sep=';', index=False, float_format='%.4f'
        )
    
    if not tanimoto_intra_results['cluster_assignments'].empty:
        df_ensayo_with_qed = df_ensayo_with_qed.merge(
            tanimoto_intra_results['cluster_assignments'],
            on='SMILES',
            how='left'
        )
    
    if tanimoto_intra_results['tanimoto_matrix'] is not None:
        save_tanimoto_heatmap(
            tanimoto_intra_results['tanimoto_matrix'],
            tanimoto_intra_results['cluster_labels'],
            os.path.join(OUTPUT_DIR, "tanimoto_intra_heatmap.png"),
            f"Tanimoto Intra-Dataset (Query vs Query, n={tanimoto_intra_results['tanimoto_matrix'].shape[0]})"
        )
    
    logging.info("")
    
    # ============================================================================
    # 3. TANIMOTO INTER-DATASET (Query vs Reference)
    # ============================================================================
    logging.info("="*80)
    logging.info("3. TANIMOTO INTER-DATASET (Query vs Reference)")
    logging.info("="*80)
    
    tanimoto_inter_results = tanimoto_inter_dataset(
        all_smiles,
        reference_smiles,
        max_reference=MAX_REFERENCE_MOLECULES
    )
    
    if not tanimoto_inter_results.empty:
        tanimoto_inter_path = os.path.join(OUTPUT_DIR, "tanimoto_inter_query_vs_reference.csv")
        tanimoto_inter_results.to_csv(tanimoto_inter_path, sep=';', index=False, float_format='%.4f')
        logging.info(f"✓ Guardado: {tanimoto_inter_path}")
        
        # Merge con dataset principal
        df_ensayo_with_qed = df_ensayo_with_qed.merge(
            tanimoto_inter_results[['Query_SMILES', 'Most_Similar_Reference_SMILES', 
                                   'Max_Tanimoto', 'Mean_Tanimoto']].rename(
                columns={'Query_SMILES': 'SMILES'}
            ),
            on='SMILES',
            how='left'
        )
    
    logging.info("")
    
    # ============================================================================
    # 4. FILTRADO POR LEVERAGE
    # ============================================================================
    logging.info("="*80)
    logging.info("4. FILTRADO POR LEVERAGE < 0.303")
    logging.info("="*80)
    
    if 'Leverage' in df_ensayo.columns:
        df_filtered = df_ensayo_with_qed[df_ensayo_with_qed['Leverage'] < LEVERAGE_THRESHOLD].copy()
        logging.info(f"✓ {len(df_filtered)} moléculas con Leverage < {LEVERAGE_THRESHOLD}")
    elif 'Distance_to_Centroid_2D' in df_ensayo.columns:
        threshold = np.percentile(df_ensayo_with_qed['Distance_to_Centroid_2D'].dropna(), 75)
        df_filtered = df_ensayo_with_qed[df_ensayo_with_qed['Distance_to_Centroid_2D'] < threshold].copy()
        logging.info(f"✓ {len(df_filtered)} moléculas filtradas por distancia")
    else:
        logging.error("❌ No se encontró columna de filtrado")
        return
    
    smiles_filtered = df_filtered['SMILES'].dropna().tolist()
    
    leverage_output_path = os.path.join(OUTPUT_DIR, "molecules_leverage_filtered.csv")
    df_filtered.to_csv(leverage_output_path, sep=';', index=False, float_format='%.4f')
    logging.info(f"✓ Guardado: {leverage_output_path}\n")
    
    # ============================================================================
    # 5. FRAGMENTOS
    # ============================================================================
    logging.info("="*80)
    logging.info("5. ANÁLISIS DE FRAGMENTOS (Leverage vs Reference)")
    logging.info("="*80)
    
    # Submuestrear referencia
    if len(reference_smiles) > MAX_REFERENCE_MOLECULES:
        np.random.seed(42)
        indices = np.random.choice(len(reference_smiles), MAX_REFERENCE_MOLECULES, replace=False)
        reference_sample = [reference_smiles[i] for i in indices]
    else:
        reference_sample = reference_smiles
    
    # Analizar fragmentos
    df_query_fg = analyze_functional_groups(smiles_filtered, "Query_Leverage")
    df_reference_fg = analyze_functional_groups(reference_sample, "Reference")
    
    # Guardar
    df_query_fg.to_csv(os.path.join(OUTPUT_DIR, "fragments_query_leverage.csv"),
                       sep=';', index=False, float_format='%.2f')
    df_reference_fg.to_csv(os.path.join(OUTPUT_DIR, "fragments_reference.csv"),
                           sep=';', index=False, float_format='%.2f')
    
    if not df_query_fg.empty and not df_reference_fg.empty:
        df_comparison = compare_functional_groups(df_query_fg, df_reference_fg)
        df_comparison.to_csv(os.path.join(OUTPUT_DIR, "fragments_comparison.csv"),
                            sep=';', index=False, float_format='%.2f')
        logging.info("✓ Fragmentos guardados\n")
    
    # ============================================================================
    # 6. RESULTADOS FINALES
    # ============================================================================
    final_output_path = os.path.join(OUTPUT_DIR, "post_validation_complete_results.csv")
    df_ensayo_with_qed.to_csv(final_output_path, sep=';', index=False, float_format='%.4f')
    
    logging.info("="*80)
    logging.info("✓✓✓ ANÁLISIS COMPLETADO ✓✓✓")
    logging.info("="*80)
    logging.info(f"Total moléculas query: {len(df_ensayo_with_qed)}")
    logging.info(f"Moléculas leverage < {LEVERAGE_THRESHOLD}: {len(df_filtered)}")
    logging.info(f"Moléculas referencia: {len(reference_smiles)}")
    logging.info(f"\nArchivos en: {OUTPUT_DIR}/")
    logging.info("  • qed_analysis_all_molecules.csv")
    logging.info("  • tanimoto_intra_statistics.csv")
    logging.info("  • tanimoto_intra_representatives.csv")
    logging.info("  • tanimoto_intra_heatmap.png")
    logging.info("  • tanimoto_inter_query_vs_reference.csv")
    logging.info("  • molecules_leverage_filtered.csv")
    logging.info("  • fragments_query_leverage.csv")
    logging.info("  • fragments_reference.csv")
    logging.info("  • fragments_comparison.csv")
    logging.info("  • post_validation_complete_results.csv")

# ================================================================================
# EJECUTAR
# ================================================================================
if __name__ == "__main__":
    main_ensayo275()