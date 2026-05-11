"""
GENERACIÓN DE FIGURA 1 - CARACTERÍSTICAS DEL DATASET Y DISTRIBUCIONES DE ACTIVIDAD
Este script genera una figura científica de publicación Q1 con tres paneles que muestran:
PANEL A - COMPOSICIÓN DEL DATASET:
* Gráfico de barras mostrando el dataset inicial, datos curados de IC50/EC50, y datos removidos
* Etiquetas con números absolutos y porcentajes
* Línea de referencia horizontal para el total inicial
  PANEL B - COMPARACIÓN DE DISTRIBUCIONES (VIOLIN PLOT):
* Violin plots superpuestos con box plots para IC50 y EC50
* Visualización de densidad de probabilidad de cada distribución
* Marcadores para media (rombo) y mediana (línea)
* Prueba estadística de Mann-Whitney U con anotación de significancia
* Cajas informativas con estadísticas (n, mediana, IQR)
* Leyenda explicando componentes del violin plot
  PANEL C - ESTRATIFICACIÓN POR TERTILES:
* Histogramas de densidad superpuestos para IC50 y EC50
* Líneas verticales marcando tertiles (33.33% y 66.67%)
* Regiones sombreadas clasificando actividad en: ALTA, MEDIA, BAJA
* Permite identificar visualmente la distribución de actividad molecular
  CARACTERÍSTICAS TÉCNICAS:
* Estilo de publicación Q1 con fuentes Arial/Helvetica
* Paleta de colores profesional y accesible
* Sin spines superiores ni derechos (estilo minimalista)
* Exportación en múltiples formatos: PNG, PDF, TIFF, SVG (300 DPI)
* Manejo robusto de casos con solo IC50 o IC50+EC50
  ESTADÍSTICAS CALCULADAS:
* Media, mediana, desviación estándar
* Cuartiles (Q1, Q3) y rango intercuartílico (IQR)
* Tertiles para estratificación de actividad
* Prueba no paramétrica de Mann-Whitney U para comparar distribuciones
  ENTRADA: antioxidant10.csv (en el directorio actual)
  SALIDA: Carpeta 'ensayo285/' con:
* Figure1_Dataset_Characteristics.png/pdf/tiff/svg
* Figure1_Statistics_Report.txt
  """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')
# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================
# Directorio de salida
OUTPUT_DIR = "ensayo315"
# Entradas directas desde CSV curado(s). Si tienes separados, cambia las rutas.
IC50_INPUT_FILE = "/Users/juandavidhoyostrejos/Documents/Icesi/qsar/ensayo284/datos_curados.csv"
EC50_INPUT_FILE = "/Users/juandavidhoyostrejos/Documents/Icesi/qsar/ensayo284/datos_curados.csv"
IC50_ASSAY_FILTER = "IC50"
EC50_ASSAY_FILTER = "EC50"
# Configuración de estilo para publicación Q1
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 15,
    'xtick.labelsize': 11.5,
    'ytick.labelsize': 11.5,
    'legend.fontsize': 11.5,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi': 1200,
    'savefig.dpi': 1200,
    'savefig.bbox': 'tight',
    'axes.grid': False,
    'legend.frameon': False,
})
sns.set_style('whitegrid')
# Paleta de colores profesional
COLORS = {
    'ic50': '#2E86AB',  # Azul para IC50
    'ec50': '#E94F37',  # Rojo-naranja para EC50
    'removed': '#8D99AE',  # Gris para datos removidos
    'high': '#1A936F',  # Verde para alta actividad
    'medium': '#F6AE2D',  # Amarillo para actividad media
    'low': '#C44536',  # Rojo para baja actividad
    'text': '#1F2430',  # Gris oscuro para texto
    'grid': '#E0E0E0',  # Gris claro para grillas
    'axis': '#202020'
}

PANEL_X_LABEL_FONTSIZE = 12.2
PANEL_Y_LABEL_FONTSIZE = 12.2
# ============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# ============================================================================
def explore_and_load_data(filepath):
    """
        Carga y explora el dataset inicial.
        Parámetros:
        -----------
        filepath : str
            Ruta al archivo CSV
        Retorna:
        --------
        DataFrame de pandas con los datos cargados
        """
    print("=" * 70)
    print("EXPLORACIÓN DEL DATASET")
    print("=" * 70)
    df = pd.read_csv(filepath, low_memory=False)
    print(f"\n📊 Archivo: {filepath}")
    print(f"📊 Total de registros: {len(df):,}")
    print(f"📊 Columnas encontradas: {len(df.columns)}")
    # Mostrar tipos de medición disponibles
    if 'Standard Type' in df.columns:
        print(f"\n🔍 Tipos de medición en 'Standard Type':")
        type_counts = df['Standard Type'].value_counts()
        for t, c in type_counts.head(15).items():
            print(f" '{t}': {c:,}")
        if len(type_counts) > 15:
            print(f" ... y {len(type_counts) - 15} tipos más")
    else:
        print("\n⚠️ Columna 'Standard Type' no encontrada")
    return df


def _resolve_activity_column(df):
    """Selecciona columna de actividad logarítmica o deriva desde normalizada."""
    preferred = ["Log10 Value (nM)", "log10_value", "log_activity", "Activity_Log"]
    for col in preferred:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").values

    normalized_candidates = ["Normalized Value (nM)", "Standard Value", "Activity", "Value"]
    for col in normalized_candidates:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce").values
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log10(numeric)

    raise ValueError(
        "❌ No se encontró una columna de actividad compatible. "
        "Se esperaba Log10 Value (nM) o una columna normalizada convertible a log10."
    )


def _load_activity_from_csv(csv_path, assay_filter=None):
    """Carga valores de actividad desde CSV, opcionalmente filtrando por tipo de ensayo."""
    df = pd.read_csv(csv_path, low_memory=False)

    if assay_filter:
        if "Standard Type" not in df.columns:
            raise ValueError(
                f"❌ El archivo {csv_path} no tiene 'Standard Type' para filtrar {assay_filter}."
            )
        assay_upper = assay_filter.upper()
        mask = df["Standard Type"].astype(str).str.upper().str.contains(assay_upper, na=False)
        df = df[mask].copy()

    values = _resolve_activity_column(df)
    values = values[np.isfinite(values)]
    return values


def _extract_precomputed_tertiles(df):
    """Intenta extraer T1/T2 preguardados desde columnas del CSV."""
    if df is None or df.empty:
        return None

    def norm(txt):
        return "".join(ch for ch in str(txt).lower() if ch.isalnum())

    col_map = {norm(c): c for c in df.columns}
    candidate_pairs = [
        ("tertile33", "tertile66"),
        ("tercil33", "tercil66"),
        ("q33", "q66"),
        ("p33", "p66"),
        ("t1", "t2"),
        ("tertile1", "tertile2"),
        ("tercil1", "tercil2"),
        ("lowertertile", "uppertertile"),
    ]

    for left_key, right_key in candidate_pairs:
        if left_key in col_map and right_key in col_map:
            left = pd.to_numeric(df[col_map[left_key]], errors="coerce").dropna()
            right = pd.to_numeric(df[col_map[right_key]], errors="coerce").dropna()
            if len(left) == 0 or len(right) == 0:
                continue
            t1 = float(left.iloc[0])
            t2 = float(right.iloc[0])
            if np.isfinite(t1) and np.isfinite(t2) and t1 < t2:
                return (t1, t2)
    return None


def extract_ic50_ec50_from_csvs(ic50_csv_path, ec50_csv_path, ic50_assay_filter=None, ec50_assay_filter=None):
    """
        Extrae y cura datos de IC50 y EC50 del dataset.
        Proceso:
        --------
        1. Identifica automáticamente las columnas de IC50/EC50
        2. Convierte valores a numérico
        3. Elimina NaN e infinitos
        4. Calcula conteos y estadísticas de curación
        Parámetros:
        -----------
        df : DataFrame
            Dataset completo
        Retorna:
        --------
        dict con:
            - ic50_values: array con valores de IC50
            - ec50_values: array con valores de EC50
            - initial_count: total de moléculas inicial
            - ic50_count: moléculas IC50 curadas
            - ec50_count: moléculas EC50 curadas
            - removed_count: moléculas removidas
            - has_ic50: bool indicando si hay datos IC50
            - has_ec50: bool indicando si hay datos EC50
        """
    print("\n" + "=" * 70)
    print("EXTRACCIÓN Y CURACIÓN DE DATOS IC50/EC50")
    print("=" * 70)
    if not os.path.exists(ic50_csv_path):
        raise FileNotFoundError(f"❌ No se encontró el CSV de IC50: {ic50_csv_path}")
    if not os.path.exists(ec50_csv_path):
        raise FileNotFoundError(f"❌ No se encontró el CSV de EC50: {ec50_csv_path}")

    df_ic50 = pd.read_csv(ic50_csv_path, low_memory=False)
    df_ec50 = pd.read_csv(ec50_csv_path, low_memory=False)

    if ic50_assay_filter:
        if "Standard Type" not in df_ic50.columns:
            raise ValueError(
                f"❌ El archivo {ic50_csv_path} no tiene 'Standard Type' para filtrar {ic50_assay_filter}."
            )
        mask = df_ic50["Standard Type"].astype(str).str.upper().str.contains(ic50_assay_filter.upper(), na=False)
        df_ic50 = df_ic50[mask].copy()

    if ec50_assay_filter:
        if "Standard Type" not in df_ec50.columns:
            raise ValueError(
                f"❌ El archivo {ec50_csv_path} no tiene 'Standard Type' para filtrar {ec50_assay_filter}."
            )
        mask = df_ec50["Standard Type"].astype(str).str.upper().str.contains(ec50_assay_filter.upper(), na=False)
        df_ec50 = df_ec50[mask].copy()

    ic50_values = _resolve_activity_column(df_ic50)
    ec50_values = _resolve_activity_column(df_ec50)
    ic50_values = ic50_values[np.isfinite(ic50_values)]
    ec50_values = ec50_values[np.isfinite(ec50_values)]

    ic50_pre_t = _extract_precomputed_tertiles(df_ic50)
    ec50_pre_t = _extract_precomputed_tertiles(df_ec50)

    print(f"\n✓ IC50 CSV: {ic50_csv_path}")
    print(f"✓ EC50 CSV: {ec50_csv_path}")
    if ic50_assay_filter:
        print(f"✓ Filtro IC50: {ic50_assay_filter}")
    if ec50_assay_filter:
        print(f"✓ Filtro EC50: {ec50_assay_filter}")

    print(f"\n📊 IC50 curado: {len(ic50_values):,}")
    print(f"📊 EC50 curado: {len(ec50_values):,}")
    if ic50_pre_t is not None:
        print(f"✓ IC50 terciles preguardados detectados: [{ic50_pre_t[0]:.4f}, {ic50_pre_t[1]:.4f}]")
    if ec50_pre_t is not None:
        print(f"✓ EC50 terciles preguardados detectados: [{ec50_pre_t[0]:.4f}, {ec50_pre_t[1]:.4f}]")
    # Calcular totales
    total_curated = len(ic50_values) + len(ec50_values)
    # Al usar CSV ya curados por propiedad, no hay removidos adicionales en este paso.
    initial_count = total_curated
    removed = 0
    print(f"\n📊 Resumen de curación:")
    print(f" Dataset inicial: {initial_count:,}")
    print(f" IC50 curated: {len(ic50_values):,}")
    print(f" EC50 curated: {len(ec50_values):,}")
    print(f" Total curated: {total_curated:,}")
    print(f" Removidos: {removed:,} ({removed/initial_count*100:.1f}%)")
    return {
        'ic50_values': ic50_values,
        'ec50_values': ec50_values,
        'initial_count': initial_count,
        'ic50_count': len(ic50_values),
        'ec50_count': len(ec50_values),
        'removed_count': removed,
        'has_ic50': len(ic50_values) > 0,
        'has_ec50': len(ec50_values) > 0,
        'ic50_precomputed_tertiles': ic50_pre_t,
        'ec50_precomputed_tertiles': ec50_pre_t,
    }


def calculate_statistics(data):
    """
        Calcula estadísticas descriptivas completas para IC50 y EC50.
        Estadísticas calculadas:
        ------------------------
        - Media, mediana, desviación estándar
        - Mínimo, máximo, rango
        - Cuartiles Q1 y Q3 (25%, 75%)
        - Tertiles (33.33%, 66.67%) para estratificación
        - Conteos por categoría de actividad (alta, media, baja)
        - Prueba de Mann-Whitney U (si hay IC50 y EC50)
        Parámetros:
        -----------
        data : dict
            Diccionario con valores de IC50 y EC50
        Retorna:
        --------
        dict con todas las estadísticas calculadas
        """
    print("\n" + "=" * 70)
    print("ANÁLISIS ESTADÍSTICO DETALLADO")
    print("=" * 70)
    ic50 = data['ic50_values']
    ec50 = data['ec50_values']
    stats_results = {}
    # ===== ESTADÍSTICAS IC50 =====
    if len(ic50) > 0:
        print(f"\n📈 IC50 (n={len(ic50):,}):")
        stats_results['ic50_mean'] = np.mean(ic50)
        stats_results['ic50_median'] = np.median(ic50)
        stats_results['ic50_std'] = np.std(ic50, ddof=1)  # Desviación estándar muestral
        stats_results['ic50_min'] = np.min(ic50)
        stats_results['ic50_max'] = np.max(ic50)
        stats_results['ic50_q1'] = np.percentile(ic50, 25)
        stats_results['ic50_q3'] = np.percentile(ic50, 75)
        if data.get('ic50_precomputed_tertiles') is not None:
            stats_results['ic50_tertiles'] = np.array(data['ic50_precomputed_tertiles'], dtype=float)
            ic50_tertile_source = "preguardados CSV"
        else:
            stats_results['ic50_tertiles'] = np.percentile(ic50, [33.33, 66.67])
            ic50_tertile_source = "calculados (percentil 33.33/66.67)"
        print(f" Media: {stats_results['ic50_mean']:.4f}")
        print(f" Mediana: {stats_results['ic50_median']:.4f}")
        print(f" Desv. Est.: {stats_results['ic50_std']:.4f}")
        print(f" Rango: [{stats_results['ic50_min']:.4f}, {stats_results['ic50_max']:.4f}]")
        print(f" IQR: [{stats_results['ic50_q1']:.4f}, {stats_results['ic50_q3']:.4f}]")
        print(f" Tertiles ({ic50_tertile_source}): [{stats_results['ic50_tertiles'][0]:.4f}, "
              f"{stats_results['ic50_tertiles'][1]:.4f}]")
        # Conteo por tertiles (estratificación de actividad)
        t1, t2 = stats_results['ic50_tertiles']
        stats_results['ic50_tertile_counts'] = (
            np.sum(ic50 < t1),  # Alta actividad
            np.sum((ic50 >= t1) & (ic50 < t2)),  # Actividad media
            np.sum(ic50 >= t2)  # Baja actividad
        )
        print(f" Actividad Alta: {stats_results['ic50_tertile_counts'][0]:,}, "
              f"Media: {stats_results['ic50_tertile_counts'][1]:,}, "
              f"Baja: {stats_results['ic50_tertile_counts'][2]:,}")
    # ===== ESTADÍSTICAS EC50 =====
    if len(ec50) > 0:
        print(f"\n📈 EC50 (n={len(ec50):,}):")
        stats_results['ec50_mean'] = np.mean(ec50)
        stats_results['ec50_median'] = np.median(ec50)
        stats_results['ec50_std'] = np.std(ec50, ddof=1)
        stats_results['ec50_min'] = np.min(ec50)
        stats_results['ec50_max'] = np.max(ec50)
        stats_results['ec50_q1'] = np.percentile(ec50, 25)
        stats_results['ec50_q3'] = np.percentile(ec50, 75)
        if data.get('ec50_precomputed_tertiles') is not None:
            stats_results['ec50_tertiles'] = np.array(data['ec50_precomputed_tertiles'], dtype=float)
            ec50_tertile_source = "preguardados CSV"
        else:
            stats_results['ec50_tertiles'] = np.percentile(ec50, [33.33, 66.67])
            ec50_tertile_source = "calculados (percentil 33.33/66.67)"
        print(f" Media: {stats_results['ec50_mean']:.4f}")
        print(f" Mediana: {stats_results['ec50_median']:.4f}")
        print(f" Desv. Est.: {stats_results['ec50_std']:.4f}")
        print(f" Rango: [{stats_results['ec50_min']:.4f}, {stats_results['ec50_max']:.4f}]")
        print(f" IQR: [{stats_results['ec50_q1']:.4f}, {stats_results['ec50_q3']:.4f}]")
        print(f" Tertiles ({ec50_tertile_source}): [{stats_results['ec50_tertiles'][0]:.4f}, "
              f"{stats_results['ec50_tertiles'][1]:.4f}]")
        # Conteo por tertiles
        t1, t2 = stats_results['ec50_tertiles']
        stats_results['ec50_tertile_counts'] = (
            np.sum(ec50 < t1),
            np.sum((ec50 >= t1) & (ec50 < t2)),
            np.sum(ec50 >= t2)
        )
        print(f" Actividad Alta: {stats_results['ec50_tertile_counts'][0]:,}, "
              f"Media: {stats_results['ec50_tertile_counts'][1]:,}, "
              f"Baja: {stats_results['ec50_tertile_counts'][2]:,}")
    # ===== PRUEBA DE MANN-WHITNEY U =====
    if len(ic50) > 0 and len(ec50) > 0:
        print(f"\n📊 Prueba de Mann-Whitney U (IC50 vs EC50):")
        u_stat, p_value = stats.mannwhitneyu(ic50, ec50, alternative='two-sided')
        stats_results['mann_whitney_u'] = u_stat
        stats_results['mann_whitney_p'] = p_value
        print(f" Estadístico U: {u_stat:,.1f}")
        print(f" Valor p: {p_value:.2e}")
        if p_value < 0.05:
            print(f" ✓ Diferencia SIGNIFICATIVA (p < 0.05)")
        else:
            print(f" ✗ Diferencia NO significativa (p ≥ 0.05)")
    return stats_results


# ============================================================================
# FUNCIONES DE GENERACIÓN DE PANELES
# ============================================================================
def create_panel_a(ax, data):
    """
        Panel A: Gráfico de barras de composición del dataset.
        Muestra:
        - Dataset inicial
        - IC50 curated
        - EC50 curated (si existe)
        - Datos removidos
        Con etiquetas de valores absolutos y porcentajes.
        """
    initial = data['initial_count']
    ic50 = data['ic50_count']
    ec50 = data['ec50_count']
    removed = data['removed_count']
    # Configurar categorías según disponibilidad de EC50
    if data['has_ec50']:
        categories = ['Initial\nDataset', 'IC50\nCurated', 'EC50\nCurated', 'Removed']
        values = [initial, ic50, ec50, removed]
        colors = [COLORS['text'], COLORS['ic50'], COLORS['ec50'], COLORS['removed']]
    else:
        categories = ['Initial\nDataset', 'IC50\nCurated', 'Removed']
        values = [initial, ic50, removed]
        colors = [COLORS['text'], COLORS['ic50'], COLORS['removed']]
    # Crear barras
    bars = ax.bar(categories, values, color=colors, edgecolor='white',
                  linewidth=0.5, width=0.65)
    # Etiquetas de valores sobre las barras
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold',
                    color=COLORS['text'])
    # Porcentajes para IC50
    ax.annotate(f'({ic50/initial*100:.1f}%)',
                xy=(1, ic50), xytext=(0, 20),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=7, color=COLORS['ic50'], fontweight='bold')
    # Porcentajes para EC50 (si existe)
    if data['has_ec50']:
        ax.annotate(f'({ec50/initial*100:.1f}%)',
                    xy=(2, ec50), xytext=(0, 20),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7, color=COLORS['ec50'], fontweight='bold')
    # Configuración de ejes
    ax.set_ylabel('Number of Molecules', fontweight='bold')
    ax.set_ylim(0, initial * 1.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_title('Dataset Composition', fontweight='bold', pad=10)
    # Línea de referencia horizontal
    ax.axhline(y=initial, color=COLORS['grid'], linestyle='--',
               linewidth=0.8, alpha=0.7)


def create_panel_b(ax, data, stats_results):
    """
        Panel B: Violin plot con box plot superpuesto.
        Componentes:
        - Violin plot: muestra densidad de probabilidad
        - Box plot: muestra IQR, mediana y whiskers
        - Media: marcada con rombo blanco
        - Estadísticas en cajas informativas
        - Prueba de Mann-Whitney U (si hay IC50 y EC50)
        """
    ic50 = data['ic50_values']
    ec50 = data['ec50_values']
    if len(ic50) == 0 or len(ec50) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "Insufficient data for violin comparison", ha="center", va="center")
        return

    violin_data = [ic50, ec50]
    positions = [0.92, 2.08]
    colors_v = [COLORS['ic50'], COLORS['ec50']]

    # ===== VIOLIN PLOT =====
    parts = ax.violinplot(
        violin_data,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.82,
    )
    for pc, color in zip(parts['bodies'], colors_v):
        pc.set_facecolor(color)
        pc.set_edgecolor('white')
        pc.set_alpha(0.62)
        pc.set_linewidth(1.2)

    # ===== BOX PLOT SUPERPUESTO =====
    bp = ax.boxplot(violin_data, positions=positions, widths=0.15, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors_v):
        patch.set_facecolor('white')
        patch.set_edgecolor(color)
        patch.set_linewidth(1.9)
    for element in ['whiskers', 'caps']:
        for j, line in enumerate(bp[element]):
            line.set_color(colors_v[j // 2])
            line.set_linewidth(1.7)
    for median, color in zip(bp['medians'], colors_v):
        median.set_color(color)
        median.set_linewidth(2.3)

    # ===== MARCADORES DE MEDIA =====
    ax.scatter(
        [1], [stats_results['ic50_mean']],
        color='white', s=88, zorder=5, edgecolor=COLORS['ic50'], linewidth=2.2, marker='D'
    )
    ax.scatter(
        [2], [stats_results['ec50_mean']],
        color='white', s=88, zorder=5, edgecolor=COLORS['ec50'], linewidth=2.2, marker='D'
    )

    # ===== ESTADÍSTICA MANN-WHITNEY (solo para subtítulo superior) =====
    u_stat = stats_results['mann_whitney_u']
    p_val = stats_results['mann_whitney_p']
    u_str = f"{u_stat / 1e6:.2f}M" if u_stat >= 1e6 else f"{u_stat:,.0f}"
    p_str = "p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"
    stars = "***" if p_val < 0.001 else "*"
    y_max = max(float(np.max(ic50)), float(np.max(ec50))) + 0.45

    # ===== CAJAS INFORMATIVAS INFERIORES =====
    stats_ic50 = (
        f"n = {len(ic50):,}\n"
        f"Median: {stats_results['ic50_median']:.2f}\n"
        f"IQR: [{stats_results['ic50_q1']:.2f}, {stats_results['ic50_q3']:.2f}]"
    )
    stats_ec50 = (
        f"n = {len(ec50):,}\n"
        f"Median: {stats_results['ec50_median']:.2f}\n"
        f"IQR: [{stats_results['ec50_q1']:.2f}, {stats_results['ec50_q3']:.2f}]"
    )
    ax.text(
        0.02, 0.02, stats_ic50,
        transform=ax.transAxes,
        fontsize=9.4,
        verticalalignment='bottom',
        color=COLORS['ic50'],
        bbox=dict(boxstyle='round,pad=0.30', facecolor='white', edgecolor=COLORS['ic50'], alpha=0.95, linewidth=1.0),
    )
    ax.text(
        0.98, 0.02, stats_ec50,
        transform=ax.transAxes,
        fontsize=9.4,
        verticalalignment='bottom',
        ha='right',
        color=COLORS['ec50'],
        bbox=dict(boxstyle='round,pad=0.30', facecolor='white', edgecolor=COLORS['ec50'], alpha=0.95, linewidth=1.0),
    )

    # ===== LEYENDA =====
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['ic50'], alpha=0.62, edgecolor='white', label='IC$_{50}$ density'),
        mpatches.Patch(facecolor=COLORS['ec50'], alpha=0.62, edgecolor='white', label='EC$_{50}$ density'),
        Line2D([0], [0], color='gray', linewidth=2.2, label='Median'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(0.005, 0.965),
        fontsize=9.6,
        framealpha=0.95,
        edgecolor=COLORS['grid'],
        fancybox=False,
        borderaxespad=0.32,
        labelspacing=0.28,
        handletextpad=0.48,
    )

    # ===== CONFIGURACIÓN DE EJES =====
    ax.set_xticks(positions)
    ax.set_xticks(positions)
    ax.set_xticklabels([r'log$_{\mathbf{10}}$(IC$_{\mathbf{50}}$) [nM]', r'log$_{\mathbf{10}}$(EC$_{\mathbf{50}}$) [nM]'], fontsize=13.2, fontweight='bold')
    ax.set_xlim(0.35, 2.65)
    ax.set_ylabel(r'log$_{\mathbf{10}}$(Activity) [nM]', fontsize=PANEL_Y_LABEL_FONTSIZE, fontweight='bold')
    ax.set_title('', pad=0)
    ax.text(0.5, 1.030, 'Mann–Whitney U = 11.91M, ***p < 0.0001',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=12.4,
            fontweight='normal', color=COLORS['text'], clip_on=False)
    ax.set_ylim(-3.0, max(10.5, y_max + 0.75))
    ax.tick_params(axis='both', labelsize=11.5, width=1.0, color=COLORS['axis'])
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.15)
        ax.spines[spine].set_color(COLORS['axis'])
    ax.grid(axis='y', alpha=0.20, linestyle='--', linewidth=0.7)


def create_panel_c(ax, data, stats_results):
    """
        Panel C: Histograma con estratificación por tertiles.
        Características:
        - Histogramas de densidad superpuestos
        - Líneas verticales marcando tertiles (33.33%, 66.67%)
        - Regiones sombreadas: ALTA, MEDIA, BAJA actividad
        - Diferente estilo de línea para IC50 (--) y EC50 (:)
        """
    ic50 = data['ic50_values']
    ec50 = data['ec50_values']
    # Determinar bins óptimos
    if data['has_ec50']:
        all_data = np.concatenate([ic50, ec50])
    else:
        all_data = ic50
    bins = np.linspace(all_data.min() - 0.2, all_data.max() + 0.2, 55)
    # ===== HISTOGRAMAS =====
    ax.hist(ic50, bins=bins, alpha=0.52, color=COLORS['ic50'],
            edgecolor='white', linewidth=0.3, label='IC50', density=True)
    if data['has_ec50']:
        ax.hist(ec50, bins=bins, alpha=0.52, color=COLORS['ec50'],
                edgecolor='white', linewidth=0.3, label='EC50', density=True)
    ymax = ax.get_ylim()[1] * 1.05  # Aumentar un poco el ymax para espacio
    ax.set_ylim(0, ymax)
    # ===== LÍNEAS DE TERTILES IC50 =====
    t1_ic50, t2_ic50 = stats_results['ic50_tertiles']
    ax.axvline(x=t1_ic50, color=COLORS['ic50'], linestyle='--',
               linewidth=2.2, alpha=0.95)
    ax.axvline(x=t2_ic50, color=COLORS['ic50'], linestyle='--',
               linewidth=2.2, alpha=0.95)
    # ===== LÍNEAS DE TERTILES EC50 =====
    if data['has_ec50']:
        t1_ec50, t2_ec50 = stats_results['ec50_tertiles']
        ax.axvline(x=t1_ec50, color=COLORS['ec50'], linestyle=':',
                   linewidth=2.2, alpha=0.95)
        ax.axvline(x=t2_ec50, color=COLORS['ec50'], linestyle=':',
                   linewidth=2.2, alpha=0.95)
    # ===== REGIONES SOMBREADAS (basadas en tertiles IC50) =====
    xmin, xmax = ax.get_xlim()
    ax.axvspan(xmin, t1_ic50, alpha=0.22, color=COLORS['high'])
    ax.axvspan(t1_ic50, t2_ic50, alpha=0.24, color=COLORS['medium'])
    ax.axvspan(t2_ic50, xmax, alpha=0.22, color=COLORS['low'])
    # ===== ETIQUETAS DE REGIONES (con bbox mejorado para visibilidad, posición ajustada para MEDIUM) =====
    bbox_base = dict(boxstyle='round,pad=0.42', facecolor='white', alpha=0.98, linewidth=1.0)
    # HIGH
    ax.text((xmin + t1_ic50) / 2, ymax * 0.93, 'HIGH', fontsize=11.5,
            color='black', fontweight='bold', ha='center', va='top',
            bbox=dict(bbox_base, edgecolor=COLORS['high']))
    # MEDIUM - Posición más baja para evitar solapamiento
    ax.text((t1_ic50 + t2_ic50) / 2, ymax * 0.71, 'MEDIUM',
            fontsize=11.5, color='black', fontweight='bold',
            ha='center', va='top',
            bbox=dict(bbox_base, edgecolor=COLORS['medium']))
    # LOW
    ax.text((t2_ic50 + xmax) / 2, ymax * 0.93, 'LOW', fontsize=11.5,
            color='black', fontweight='bold', ha='center', va='top',
            bbox=dict(bbox_base, edgecolor=COLORS['low']))
    # ===== LEYENDA =====
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['ic50'], alpha=0.6, edgecolor='white',
                       label='IC$_{50}$' + f' (n={len(ic50):,})'),
        Line2D([0], [0], color=COLORS['ic50'], linestyle='--', linewidth=1.8,
               label='IC$_{50}$ tertiles'),
    ]
    if data['has_ec50']:
        legend_elements.extend([
            mpatches.Patch(facecolor=COLORS['ec50'], alpha=0.6, edgecolor='white',
                           label='EC$_{50}$' + f' (n={len(ec50):,})'),
            Line2D([0], [0], color=COLORS['ec50'], linestyle=':', linewidth=1.8,
                   label='EC$_{50}$ tertiles'),
        ])
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10.2,
              framealpha=0.95, edgecolor=COLORS['grid'], borderpad=0.50, labelspacing=0.40, handletextpad=0.50)
    # ===== CONFIGURACIÓN DE EJES =====
    ax.set_xlabel(r'log$_{\mathbf{10}}$(Activity) [nM]', fontsize=PANEL_X_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Density', fontsize=PANEL_Y_LABEL_FONTSIZE, fontweight='bold')
    ax.set_title('', pad=0)
    ax.tick_params(axis='both', labelsize=11.5, width=1.0, color=COLORS['axis'])
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.15)
        ax.spines[spine].set_color(COLORS['axis'])


# ============================================================================
# FUNCIÓN PRINCIPAL DE GENERACIÓN DE FIGURA
# ============================================================================
def create_figure(data, stats_results, output_folder):
    """
        Crea la figura completa con dos paneles.
        Layout:
        - Figura de 12x5 pulgadas
        - 2 subplots en disposición horizontal (1 fila, 2 columnas)
        - Espaciado optimizado para publicación
        - Exportación en múltiples formatos
        Parámetros:
        -----------
        data : dict
            Datos extraídos de IC50/EC50
        stats_results : dict
            Resultados estadísticos
        output_folder : str
            Carpeta de salida
        """
    print("\n" + "=" * 70)
    print("GENERANDO FIGURA CIENTÍFICA")
    print("=" * 70)
    # Crear figura y grid
    fig = plt.figure(figsize=(13.0, 6.0), facecolor='white')
    gs = fig.add_gridspec(1, 2, wspace=0.39, left=0.07, right=0.985,
                          top=0.875, bottom=0.14)
    # Crear subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    # Generar paneles
    print("Generando Panel A (antes C): Tertile Stratification...")
    create_panel_c(ax1, data, stats_results)
    print("Generando Panel B: Violin Plot Comparison...")
    create_panel_b(ax2, data, stats_results)
    # Etiquetas de paneles (A, B)
    for ax, label in zip([ax1, ax2], ['A', 'B']):
        ax.text(-0.08, 1.035, label, transform=ax.transAxes, fontsize=19,
                fontweight='bold', va='top', ha='left', color=COLORS['text'], clip_on=False)
    # Crear carpeta de salida
    os.makedirs(output_folder, exist_ok=True)
    # Guardar en múltiples formatos
    base_name = os.path.join(output_folder, 'Figure1_Dataset_Characteristics')
    formats = {
        '.png': 1200,
        '.pdf': 1200,
        '.tiff': 1200,
        '.svg': None  # SVG es vectorial, no necesita DPI
    }
    for ext, dpi in formats.items():
        filepath = base_name + ext
        if ext == '.tiff':
            fig.savefig(
                filepath,
                facecolor='white',
                edgecolor='none',
                bbox_inches='tight',
                dpi=dpi,
                pil_kwargs={'compression': 'tiff_lzw'},
            )
        elif dpi is not None:
            fig.savefig(filepath, facecolor='white', edgecolor='none',
                        bbox_inches='tight', dpi=dpi)
        else:
            fig.savefig(filepath, facecolor='white', edgecolor='none',
                        bbox_inches='tight', format='svg')
        print(f"✓ Guardado: {os.path.basename(filepath)}")
    plt.close()


# ============================================================================
# FUNCIÓN DE REPORTE ESTADÍSTICO
# ============================================================================
def save_statistics_report(data, stats_results, output_folder):
    """
        Guarda un reporte de texto con todas las estadísticas.
        Contenido del reporte:
        - Composición del dataset
        - Estadísticas descriptivas de IC50
        - Estadísticas descriptivas de EC50 (si existe)
        - Resultados de prueba de Mann-Whitney U
        """
    report_path = os.path.join(output_folder, 'Figure1_Statistics_Report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("FIGURE 1 - STATISTICAL ANALYSIS REPORT\n")
        f.write("Dataset Characteristics and Activity Distributions\n")
        f.write("=" * 70 + "\n\n")
        # ===== COMPOSICIÓN DEL DATASET =====
        f.write("DATASET COMPOSITION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Initial molecules: {data['initial_count']:,}\n")
        f.write(f"IC50 curated: {data['ic50_count']:,} "
                f"({data['ic50_count']/data['initial_count']*100:.2f}%)\n")
        if data['has_ec50']:
            f.write(f"EC50 curated: {data['ec50_count']:,} "
                    f"({data['ec50_count']/data['initial_count']*100:.2f}%)\n")
        f.write(f"Removed: {data['removed_count']:,} "
                f"({data['removed_count']/data['initial_count']*100:.2f}%)\n\n")
        # ===== ESTADÍSTICAS IC50 =====
        if data['has_ic50']:
            f.write("IC50 STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Sample size: n = {data['ic50_count']:,}\n\n")
            f.write("Measures of central tendency:\n")
            f.write(f" Mean: {stats_results['ic50_mean']:.4f}\n")
            f.write(f" Median: {stats_results['ic50_median']:.4f}\n\n")
            f.write("Measures of dispersion:\n")
            f.write(f" Std Dev: {stats_results['ic50_std']:.4f}\n")
            f.write(f" Min: {stats_results['ic50_min']:.4f}\n")
            f.write(f" Max: {stats_results['ic50_max']:.4f}\n")
            f.write(f" Range: {stats_results['ic50_max'] - stats_results['ic50_min']:.4f}\n\n")
            f.write("Quartiles:\n")
            f.write(f" Q1 (25%): {stats_results['ic50_q1']:.4f}\n")
            f.write(f" Q2 (50%, Median): {stats_results['ic50_median']:.4f}\n")
            f.write(f" Q3 (75%): {stats_results['ic50_q3']:.4f}\n")
            f.write(f" IQR (Q3-Q1): {stats_results['ic50_q3'] - stats_results['ic50_q1']:.4f}\n\n")
            f.write("Tertiles (for activity stratification):\n")
            f.write(f" T1 (33.33%): {stats_results['ic50_tertiles'][0]:.4f}\n")
            f.write(f" T2 (66.67%): {stats_results['ic50_tertiles'][1]:.4f}\n\n")
            f.write("Activity classification counts:\n")
            f.write(f" High activity (< T1): {stats_results['ic50_tertile_counts'][0]:,}\n")
            f.write(f" Medium activity (T1-T2): {stats_results['ic50_tertile_counts'][1]:,}\n")
            f.write(f" Low activity (> T2): {stats_results['ic50_tertile_counts'][2]:,}\n\n")
        # ===== ESTADÍSTICAS EC50 =====
        if data['has_ec50']:
            f.write("EC50 STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Sample size: n = {data['ec50_count']:,}\n\n")
            f.write("Measures of central tendency:\n")
            f.write(f" Mean: {stats_results['ec50_mean']:.4f}\n")
            f.write(f" Median: {stats_results['ec50_median']:.4f}\n\n")
            f.write("Measures of dispersion:\n")
            f.write(f" Std Dev: {stats_results['ec50_std']:.4f}\n")
            f.write(f" Min: {stats_results['ec50_min']:.4f}\n")
            f.write(f" Max: {stats_results['ec50_max']:.4f}\n")
            f.write(f" Range: {stats_results['ec50_max'] - stats_results['ec50_min']:.4f}\n\n")
            f.write("Quartiles:\n")
            f.write(f" Q1 (25%): {stats_results['ec50_q1']:.4f}\n")
            f.write(f" Q2 (50%, Median): {stats_results['ec50_median']:.4f}\n")
            f.write(f" Q3 (75%): {stats_results['ec50_q3']:.4f}\n")
            f.write(f" IQR (Q3-Q1): {stats_results['ec50_q3'] - stats_results['ec50_q1']:.4f}\n\n")
            f.write("Tertiles (for activity stratification):\n")
            f.write(f" T1 (33.33%): {stats_results['ec50_tertiles'][0]:.4f}\n")
            f.write(f" T2 (66.67%): {stats_results['ec50_tertiles'][1]:.4f}\n\n")
            f.write("Activity classification counts:\n")
            f.write(f" High activity (< T1): {stats_results['ec50_tertile_counts'][0]:,}\n")
            f.write(f" Medium activity (T1-T2): {stats_results['ec50_tertile_counts'][1]:,}\n")
            f.write(f" Low activity (> T2): {stats_results['ec50_tertile_counts'][2]:,}\n\n")
        # ===== PRUEBA DE MANN-WHITNEY U =====
        if 'mann_whitney_u' in stats_results:
            f.write("STATISTICAL COMPARISON (IC50 vs EC50)\n")
            f.write("-" * 40 + "\n")
            f.write("Mann-Whitney U Test (non-parametric):\n")
            f.write(f" U statistic: {stats_results['mann_whitney_u']:,.1f}\n")
            f.write(f" p-value: {stats_results['mann_whitney_p']:.4e}\n")
            f.write(f" Significance level: α = 0.05\n")
            if stats_results['mann_whitney_p'] < 0.05:
                f.write(f" Result: SIGNIFICANT (p < 0.05)\n")
                f.write(f" Interpretation: The distributions of IC50 and EC50 are\n")
                f.write(f" significantly different.\n")
            else:
                f.write(f" Result: NOT SIGNIFICANT (p ≥ 0.05)\n")
                f.write(f" Interpretation: No significant difference between IC50\n")
                f.write(f" and EC50 distributions.\n")
    print(f"✓ Guardado: {os.path.basename(report_path)}")


def create_qq_plots(data, output_folder):
    """
    Generate supplementary Q-Q plots for IC50 and EC50 normality assessment.
    """
    ic50 = data['ic50_values']
    ec50 = data['ec50_values']

    if len(ic50) == 0 or len(ec50) == 0:
        print("⚠️ Q-Q plot omitido: faltan datos IC50 o EC50.")
        return

    fig = plt.figure(figsize=(11.2, 5.6), facecolor='white')
    gs = fig.add_gridspec(1, 2, wspace=0.24, left=0.08, right=0.98, top=0.91, bottom=0.14)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for ax, values, label, color in [
        (ax1, ic50, r"$\mathbf{IC}_{\mathbf{50}}$", COLORS['ic50']),
        (ax2, ec50, r"$\mathbf{EC}_{\mathbf{50}}$", COLORS['ec50']),
    ]:
        (osm, osr), (slope, intercept, r) = stats.probplot(values, dist="norm")
        point_size = 14 if label == r"IC$_{\mathbf{50}}$" else 12
        point_alpha = 0.98 if label == r"IC$_{\mathbf{50}}$" else 0.96

        ax.scatter(
            osm,
            osr,
            s=point_size,
            color=color,
            alpha=point_alpha,
            edgecolors='none',
            linewidths=0.0,
        )

        ax.plot(
            osm,
            slope * osm + intercept,
            color='black',
            linewidth=1.9,
            linestyle='--',
            alpha=0.9
        )

        ax.set_title(f"Q-Q Plot ({label})", fontweight='bold', pad=8)
        ax.set_xlabel("Theoretical Quantiles", fontweight='bold', fontsize=12)
        ax.set_ylabel("Sample Quantiles", fontweight='bold', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(True, linestyle='--', linewidth=0.85, alpha=0.35, color='#D1D7E0')
        ax.set_axisbelow(True)

        ax.text(
            0.03,
            0.97,
            f"R² = {r**2:.4f}",
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=11,
            fontweight='bold',
            color=COLORS['text'],
            bbox=dict(
                boxstyle='round,pad=0.28',
                facecolor='white',
                edgecolor='#D3D3D3',
                linewidth=0.9,
                alpha=1.0
            ),
        )

    ax1.text(
        -0.18, 1.06, "A",
        transform=ax1.transAxes,
        fontsize=16,
        fontweight='bold',
        color=COLORS['text'],
        ha='left',
        va='top'
    )
    ax2.text(
        -0.18, 1.06, "B",
        transform=ax2.transAxes,
        fontsize=16,
        fontweight='bold',
        color=COLORS['text'],
        ha='left',
        va='top'
    )

    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.join(output_folder, "Supplementary_QQPlot_IC50_EC50")
    outputs = {
        ".tiff": 1200,
        ".png": 1200,
        ".pdf": 1200,
        ".svg": None,
    }

    for ext, dpi in outputs.items():
        out = base_name + ext
        if ext == ".tiff":
            fig.savefig(
                out,
                facecolor='white',
                edgecolor='none',
                bbox_inches='tight',
                dpi=dpi,
                pil_kwargs={'compression': 'tiff_lzw'}
            )
        elif dpi is None:
            fig.savefig(
                out,
                facecolor='white',
                edgecolor='none',
                bbox_inches='tight',
                format='svg'
            )
        else:
            fig.savefig(
                out,
                facecolor='white',
                edgecolor='none',
                bbox_inches='tight',
                dpi=dpi
            )
        print(f"✓ Guardado: {os.path.basename(out)}")

    plt.close(fig)


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FIGURE 1 - DATASET CHARACTERISTICS")
    print("QSAR Cannabaceae - Publication Q1")
    print("=" * 70)
    try:
        # 1. Cargar IC50/EC50 desde CSV directos (separados o un CSV con filtros por tipo)
        data = extract_ic50_ec50_from_csvs(
            IC50_INPUT_FILE,
            EC50_INPUT_FILE,
            ic50_assay_filter=IC50_ASSAY_FILTER,
            ec50_assay_filter=EC50_ASSAY_FILTER,
        )
        # 3. Calcular estadísticas
        stats_results = calculate_statistics(data)
        # 4. Crear figura
        create_figure(data, stats_results, OUTPUT_DIR)
        # 5. Guardar reporte estadístico
        save_statistics_report(data, stats_results, OUTPUT_DIR)
        # 6. Q-Q plots suplementarios
        create_qq_plots(data, OUTPUT_DIR)
        # ===== RESUMEN FINAL =====
        print("\n" + "=" * 70)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print(f"\nArchivos guardados en: {OUTPUT_DIR}/\n")
        print("Archivos generados:")
        print(" 📊 Figure1_Dataset_Characteristics.png")
        print(" 📊 Figure1_Dataset_Characteristics.pdf")
        print(" 📊 Figure1_Dataset_Characteristics.tiff")
        print(" 📊 Figure1_Dataset_Characteristics.svg")
        print(" 📄 Figure1_Statistics_Report.txt")
        print(" 📊 Supplementary_QQPlot_IC50_EC50.tiff/.png/.pdf/.svg")
        print("\n" + "=" * 70 + "\n")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print(" Verifica las rutas IC50_INPUT_FILE y EC50_INPUT_FILE.")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
