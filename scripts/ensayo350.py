# Este script construye PCA editoriales tipo JCIM usando bibliotecas de referencia
# de IC50 y EC50, proyecta cannabinoides y terpenos externos y exporta figuras 2D
# suplementarias refinadas listas para Supporting Information.

from __future__ import annotations

import importlib
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors, rdFingerprintGenerator
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path('/Users/juandavidhoyostrejos/Documents/Icesi/qsar')
OUTPUT_DIR = BASE_DIR / 'ensayo350'
TRAINING_PATH = BASE_DIR / 'antioxidant12.csv'
CANNABINOIDS_PATH = BASE_DIR / 'ensayo197' / 'Cannabaceae_consolidated_predictions.csv'
TERPENES_PATH = BASE_DIR / 'ensayo329' / 'terpenos_predictions8.csv'
EC50_TRAINING_PATH = BASE_DIR / 'EC50_dataset.csv'
EC50_CANNABINOIDS_PATH = BASE_DIR / 'Cannabinoides_consolidated_predictions(EC50).csv'
EC50_TERPENES_PATH = BASE_DIR / 'Terpenes(EC50).csv'

FIG_DPI = 1200
FIGSIZE_MAIN = (14.6, 6.4)
FONT_FAMILY = ['Arial', 'Helvetica', 'DejaVu Sans']
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
RANDOM_STATE = 42
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS)

TRAINING_COLOR = '#CCCCCC'
CANN_COLOR = '#2F6FB0'
TERP_COLOR = '#C9961A'
HULL_COLOR = '#7A7F87'
EC50_HULL_COLOR = '#8E939A'
ELLIPSE95 = '#7AA6D8'
GRID_COLOR = '#E6E8EB'
AXIS_COLOR = '#54575C'

X_LIMITS = (-12, 32)
Y_LIMITS = (-28, 14)

CANN_LABELS = {'THC', 'CBD', 'CBG', 'CBGA', 'THCA'}
TERP_LABELS = {
    'beta-Mirceno',
    'p-cymene',
    'terpinolene',
    'alfa-Pineno',
    'd-Limoneno',
    'Linalool',
    'beta-Cariopilene',
    '(-)-Oxido de Cariofileno',
}

PHYS_DESCRIPTORS = [
    'MolWt',
    'MolLogP',
    'TPSA',
    'NumRotatableBonds',
    'NumHDonors',
    'NumHAcceptors',
    'FractionCSP3',
    'NumAromaticRings',
]


def configure_style() -> None:
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': FONT_FAMILY,
        'font.size': 9.2,
        'axes.titlesize': 12.7,
        'axes.titleweight': 'semibold',
        'axes.labelsize': 11.5,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 9.0,
        'ytick.labelsize': 9.0,
        'legend.fontsize': 9.2,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.edgecolor': AXIS_COLOR,
        'axes.linewidth': 0.8,
        'grid.color': GRID_COLOR,
        'grid.alpha': 0.55,
        'grid.linewidth': 0.4,
    })


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_required_modules() -> object | None:
    missing = []
    for module_name in ['pandas', 'numpy', 'matplotlib', 'sklearn', 'rdkit', 'scipy']:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        raise SystemExit('Missing required modules: ' + ', '.join(missing))
    try:
        return importlib.import_module('adjustText').adjust_text
    except Exception:
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for column in df.columns:
        clean = column.strip()
        if clean == 'Smiles':
            rename_map[column] = 'SMILES'
        elif clean in {'Molecule Name', 'Molecule_name', 'Molecule Name '}:
            rename_map[column] = 'Molecule_name'
        elif clean == 'Log10 Value (nM)':
            rename_map[column] = 'Log10_Value_nM'
    return df.rename(columns=rename_map)


def read_csv_auto(path: Path) -> pd.DataFrame:
    with path.open('r', encoding='utf-8-sig', errors='replace') as handle:
        header = handle.readline()
    sep = ';' if header.count(';') >= header.count(',') else ','
    return normalize_columns(pd.read_csv(path, sep=sep, encoding='utf-8-sig'))


def validate_smiles_table(df: pd.DataFrame, smiles_col: str = 'SMILES') -> pd.DataFrame:
    mols = []
    valid_rows = []
    for idx, smiles in enumerate(df[smiles_col].astype(str)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mols.append(mol)
        valid_rows.append(idx)
    out = df.iloc[valid_rows].copy().reset_index(drop=True)
    out['Mol'] = mols
    return out


def load_training_set() -> pd.DataFrame:
    df = read_csv_auto(TRAINING_PATH)
    if 'SMILES' not in df.columns:
        raise ValueError(f'SMILES column not found in {TRAINING_PATH}')
    if 'Molecule_name' not in df.columns:
        df['Molecule_name'] = 'Training molecule'
    df = df.dropna(subset=['SMILES']).copy()
    df['SMILES'] = df['SMILES'].astype(str).str.strip()
    df = df[df['SMILES'] != ''].copy()
    df = validate_smiles_table(df)
    df = df.drop_duplicates(subset=['SMILES']).reset_index(drop=True)
    df['Group'] = 'Training set'
    return df


def load_external_set(path: Path, group_name: str) -> pd.DataFrame:
    df = read_csv_auto(path)
    if 'SMILES' not in df.columns or 'Molecule_name' not in df.columns:
        raise ValueError(f'SMILES/Molecule_name columns not found in {path}')
    df = df[['SMILES', 'Molecule_name']].dropna(subset=['SMILES', 'Molecule_name']).copy()
    df['SMILES'] = df['SMILES'].astype(str).str.strip()
    df['Molecule_name'] = df['Molecule_name'].astype(str).str.strip()
    df = df[(df['SMILES'] != '') & (df['Molecule_name'] != '')].copy()
    df = validate_smiles_table(df)
    df = df.drop_duplicates(subset=['SMILES']).reset_index(drop=True)
    df['Group'] = group_name
    return df


def descriptor_cache_path(label: str) -> Path:
    return OUTPUT_DIR / f'{label}_descriptor_cache.pkl'


def compute_descriptor_matrix(df: pd.DataFrame, cache_label: str) -> tuple[np.ndarray, list[str]]:
    cache_path = descriptor_cache_path(cache_label)
    current_smiles = df['SMILES'].tolist()
    if cache_path.exists():
        with cache_path.open('rb') as handle:
            cache = pickle.load(handle)
        if cache.get('smiles') == current_smiles:
            return cache['matrix'], cache['feature_names']

    matrix = np.zeros((len(df), len(PHYS_DESCRIPTORS) + FINGERPRINT_BITS), dtype=np.float32)
    feature_names = PHYS_DESCRIPTORS + [f'Morgan_{i}' for i in range(FINGERPRINT_BITS)]

    for row_idx, mol in enumerate(df['Mol']):
        phys_values = [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Descriptors.FractionCSP3(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
        ]
        matrix[row_idx, : len(PHYS_DESCRIPTORS)] = np.asarray(phys_values, dtype=np.float32)
        fp = MORGAN_GENERATOR.GetFingerprint(mol)
        fp_array = np.zeros((FINGERPRINT_BITS,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        matrix[row_idx, len(PHYS_DESCRIPTORS):] = fp_array

    with cache_path.open('wb') as handle:
        pickle.dump({'smiles': current_smiles, 'matrix': matrix, 'feature_names': feature_names}, handle)
    return matrix, feature_names


def fit_pca(training_df: pd.DataFrame, cannabinoids_df: pd.DataFrame, terpenes_df: pd.DataFrame, cache_prefix: str) -> tuple[pd.DataFrame, np.ndarray]:
    print('Computing molecular descriptors...', flush=True)
    x_train, _ = compute_descriptor_matrix(training_df, f'{cache_prefix}_training')
    x_cann, _ = compute_descriptor_matrix(cannabinoids_df, f'{cache_prefix}_cannabinoids')
    x_terp, _ = compute_descriptor_matrix(terpenes_df, f'{cache_prefix}_terpenes')

    print('Scaling and fitting PCA on training set only...', flush=True)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_cann_scaled = scaler.transform(x_cann)
    x_terp_scaled = scaler.transform(x_terp)

    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    train_coords = pca.fit_transform(x_train_scaled)
    cann_coords = pca.transform(x_cann_scaled)
    terp_coords = pca.transform(x_terp_scaled)

    coord_columns = ['PC1', 'PC2', 'PC3']
    train_out = training_df[['SMILES', 'Molecule_name', 'Group']].copy()
    cann_out = cannabinoids_df[['SMILES', 'Molecule_name', 'Group']].copy()
    terp_out = terpenes_df[['SMILES', 'Molecule_name', 'Group']].copy()

    for idx, col in enumerate(coord_columns):
        train_out[col] = train_coords[:, idx]
        cann_out[col] = cann_coords[:, idx]
        terp_out[col] = terp_coords[:, idx]

    combined = pd.concat([train_out, cann_out, terp_out], ignore_index=True)
    return combined, pca.explained_variance_ratio_ * 100


def confidence_ellipse(x: np.ndarray, y: np.ndarray, ax: plt.Axes, n_std: float, edgecolor: str, linewidth: float, alpha: float = 1.0) -> None:
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(max(cov[0, 0] * cov[1, 1], 1e-12))
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', edgecolor=edgecolor, linewidth=linewidth, linestyle='-', alpha=alpha)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    angle = np.rad2deg(np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1]) / 2)
    transform = plt.matplotlib.transforms.Affine2D().rotate_deg(angle).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transform + ax.transData)
    ax.add_patch(ellipse)


def plot_convex_hull(points: np.ndarray, ax: plt.Axes, color: str, linewidth: float, alpha: float) -> None:
    if len(points) < 3:
        return
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_points = np.vstack([hull_points, hull_points[0]])
    ax.plot(hull_points[:, 0], hull_points[:, 1], color=color, linewidth=linewidth, alpha=alpha, zorder=2)


def abbreviate_label(name: str, max_len: int = 24) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1].rstrip() + '…'


def filter_labels(external_df: pd.DataFrame, keep_names: set[str]) -> pd.DataFrame:
    def canonical(value: str) -> str:
        return str(value).strip()
    df = external_df.copy()
    df['canonical_name'] = df['Molecule_name'].map(canonical)
    filtered = df[df['canonical_name'].isin(keep_names)].copy()
    return filtered.reset_index(drop=True)


def add_labels(ax: plt.Axes, labeled_df: pd.DataFrame, box_edge: str, adjust_text) -> None:
    if labeled_df.empty:
        return
    texts = []
    for _, row in labeled_df.iterrows():
        text = ax.text(
            row['PC1'],
            row['PC2'],
            abbreviate_label(str(row['Molecule_name'])),
            fontsize=6.9,
            fontweight='regular',
            color='black',
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=box_edge, lw=0.4, alpha=0.12),
            zorder=8,
        )
        texts.append(text)

    if adjust_text is not None and texts:
        adjust_text(
            texts,
            ax=ax,
            only_move={'points': 'xy', 'texts': 'xy'},
            expand_points=(1.12, 1.16),
            expand_text=(1.08, 1.12),
            force_points=(0.12, 0.18),
            force_text=(0.16, 0.24),
            arrowprops=dict(arrowstyle='-', color=box_edge, lw=0.35, alpha=0.5),
            min_arrow_len=6,
            prevent_crossings=True,
            ensure_inside_axes=True,
            lim=800,
        )


def style_axes(ax: plt.Axes, panel_letter: str, variance_pct: np.ndarray) -> None:
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.set_ylabel(f'PC2 ({variance_pct[1]:.1f}% variance explained)', fontsize=15.4, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=13.2, width=0.8, length=3.3)
    ax.tick_params(axis='both', which='minor', width=0.55, length=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which='major')
    ax.grid(False, which='minor')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color(AXIS_COLOR)
    ax.spines['bottom'].set_color(AXIS_COLOR)
    ax.text(0.012, 0.985, panel_letter, transform=ax.transAxes, fontsize=20.5, fontweight='bold', ha='left', va='top', color='#222222')


def draw_training_background(ax: plt.Axes, training: pd.DataFrame, include_ellipse: bool, hull_color: str, hull_linewidth: float, hull_alpha: float) -> None:
    points = training[['PC1', 'PC2']].to_numpy()
    ax.scatter(training['PC1'], training['PC2'], s=3.2, c=TRAINING_COLOR, alpha=0.09, edgecolors='none', zorder=1)
    plot_convex_hull(points, ax, hull_color, hull_linewidth, hull_alpha)
    if include_ellipse:
        confidence_ellipse(training['PC1'].to_numpy(), training['PC2'].to_numpy(), ax, n_std=2.45, edgecolor=ELLIPSE95, linewidth=0.8, alpha=0.72)


def build_panel(ax: plt.Axes, training: pd.DataFrame, external_df: pd.DataFrame, point_color: str, marker: str, panel_letter: str, title: str, variance_pct: np.ndarray, adjust_text, keep_labels: set[str], include_ellipse: bool = True, hull_color: str = HULL_COLOR, hull_linewidth: float = 1.25, hull_alpha: float = 0.8) -> None:
    draw_training_background(ax, training, include_ellipse, hull_color, hull_linewidth, hull_alpha)
    ax.scatter(
        external_df['PC1'],
        external_df['PC2'],
        s=62,
        c=point_color,
        marker=marker,
        edgecolors='white',
        linewidths=0.7,
        alpha=0.98,
        zorder=6,
    )
    # Subplot title intentionally omitted for cleaner supplementary layout
    style_axes(ax, panel_letter, variance_pct)


def build_main_figure(combined: pd.DataFrame, variance_pct: np.ndarray, figure_title: str, stem: str, include_ellipse: bool = True, hull_color: str = HULL_COLOR, hull_linewidth: float = 1.25, hull_alpha: float = 0.8) -> None:
    training = combined[combined['Group'] == 'Training set'].copy()
    cannabinoids = combined[combined['Group'] == 'Cannabinoids'].copy()
    terpenes = combined[combined['Group'] == 'Terpenes'].copy()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MAIN, dpi=FIG_DPI)
    build_panel(axes[0], training, cannabinoids, CANN_COLOR, 'o', 'A', 'Cannabinoids', variance_pct, None, CANN_LABELS, include_ellipse=include_ellipse, hull_color=hull_color, hull_linewidth=hull_linewidth, hull_alpha=hull_alpha)
    build_panel(axes[1], training, terpenes, TERP_COLOR, 'D', 'B', 'Terpenes', variance_pct, None, TERP_LABELS, include_ellipse=include_ellipse, hull_color=hull_color, hull_linewidth=hull_linewidth, hull_alpha=hull_alpha)
    axes[1].set_ylabel('')

    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=TRAINING_COLOR, markeredgecolor='none', markersize=4.4, alpha=0.7, label='Training set'),
        Line2D([0], [0], color=hull_color, lw=hull_linewidth, alpha=hull_alpha, label='Training convex hull'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=CANN_COLOR, markeredgecolor='white', markeredgewidth=0.7, markersize=6.8, label='Cannabinoids'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor=TERP_COLOR, markeredgecolor='white', markeredgewidth=0.7, markersize=6.8, label='Terpenes'),
    ]

    # Figure title intentionally omitted for cleaner supplementary layout
    fig.supxlabel(f'PC1 ({variance_pct[0]:.1f}% variance explained)', fontsize=15.4, fontweight='bold', y=0.085)
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False, fontsize=12.8, handlelength=1.6, columnspacing=1.3, handletextpad=0.55)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.19, wspace=0.24)
    save_figure(fig, stem)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f'{stem}.tiff', dpi=FIG_DPI, format='tiff', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / f'{stem}.png', dpi=FIG_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def print_summary(training_df: pd.DataFrame, cannabinoids_df: pd.DataFrame, terpenes_df: pd.DataFrame, variance_pct: np.ndarray, adjust_text) -> None:
    print('\n=== PCA Summary ===')
    print(f'Training molecules   : {len(training_df)}')
    print(f'Cannabinoids         : {len(cannabinoids_df)}')
    print(f'Terpenes             : {len(terpenes_df)}')
    print(f'PC1 variance         : {variance_pct[0]:.2f}%')
    print(f'PC2 variance         : {variance_pct[1]:.2f}%')
    print(f'PC3 variance         : {variance_pct[2]:.2f}%')
    print(f'adjustText available : {adjust_text is not None}')
    print(f'Outputs              : {OUTPUT_DIR}')


def run_endpoint_pipeline(
    training_path: Path,
    cannabinoids_path: Path,
    terpenes_path: Path,
    coordinates_name: str,
    figure_title: str,
    stem: str,
    summary_label: str,
    adjust_text,
    include_ellipse: bool = True,
    hull_color: str = HULL_COLOR,
    hull_linewidth: float = 1.25,
    hull_alpha: float = 0.8,
) -> None:
    print('Reading input tables...', flush=True)
    training_df = load_training_set() if training_path == TRAINING_PATH else load_external_set(training_path, 'Training set')
    cannabinoids_df = load_external_set(cannabinoids_path, 'Cannabinoids')
    terpenes_df = load_external_set(terpenes_path, 'Terpenes')

    combined, variance_pct = fit_pca(training_df, cannabinoids_df, terpenes_df, summary_label.lower())
    combined.to_csv(OUTPUT_DIR / coordinates_name, index=False)
    print('Building editorial PCA figure...', flush=True)
    build_main_figure(combined, variance_pct, figure_title, stem, include_ellipse=include_ellipse, hull_color=hull_color, hull_linewidth=hull_linewidth, hull_alpha=hull_alpha)
    print(f'\n=== PCA Summary ({summary_label}) ===')
    print(f'Training molecules   : {len(training_df)}')
    print(f'Cannabinoids         : {len(cannabinoids_df)}')
    print(f'Terpenes             : {len(terpenes_df)}')
    print(f'PC1 variance         : {variance_pct[0]:.2f}%')
    print(f'PC2 variance         : {variance_pct[1]:.2f}%')
    print(f'PC3 variance         : {variance_pct[2]:.2f}%')
    print(f'adjustText available : {adjust_text is not None}')
    print(f'Coordinates CSV      : {OUTPUT_DIR / coordinates_name}')
    print(f'Figure stem          : {OUTPUT_DIR / stem}')




TANIMOTO_NAME_MAP = {
    'beta-pineno': 'β-Pinene',
    'beta-pinene': 'β-Pinene',
    'beta-mirceno': 'β-Myrcene',
    'beta-mircene': 'β-Myrcene',
    'beta-myrcene': 'β-Myrcene',
    'beta-cariopilene': 'β-Caryophyllene',
    'beta-caryophyllene': 'β-Caryophyllene',
    'alfa-pineno': 'α-Pinene',
    'alpha-pinene': 'α-Pinene',
    '(-)-alfa-bisabolol': '(−)-α-Bisabolol',
    '(-)-alpha-bisabolol': '(−)-α-Bisabolol',
    '(-)-oxido de cariofileno': '(−)-Caryophyllene oxide',
    '(-)-oxide of caryophyllene': '(−)-Caryophyllene oxide',
    'd-limoneno': 'D-Limonene',
    'd-limonene': 'D-Limonene',
    'p-cymene': 'p-Cymene',
    'linalool': 'Linalool',
    'geraniol': 'Geraniol',
    'nerolidol 1': 'Nerolidol 1',
    'nerolidol 2': 'Nerolidol 2',
    'ocimeno 1': 'Ocimene 1',
    'ocimene 1': 'Ocimene 1',
    'camphene': 'Camphene',
}


def standardize_tanimoto_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        return text
    key = text.casefold()
    return TANIMOTO_NAME_MAP.get(key, text)


def build_tanimoto_similarity_table(training_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    training_fps = [MORGAN_GENERATOR.GetFingerprint(mol) for mol in training_df['Mol']]
    rows = []
    for _, row in external_df.iterrows():
        fp = MORGAN_GENERATOR.GetFingerprint(row['Mol'])
        sims = DataStructs.BulkTanimotoSimilarity(fp, training_fps)
        max_sim = max(sims) if sims else 0.0
        rows.append({'Molecule_name': standardize_tanimoto_name(row['Molecule_name']), 'MaxSimilarity': float(max_sim)})
    out = pd.DataFrame(rows)
    out = out.sort_values('MaxSimilarity', ascending=False).reset_index(drop=True)
    return out


def _style_tanimoto_texts(ax: plt.Axes) -> None:
    for text in ax.texts:
        content = text.get_text().strip()
        if not content:
            continue
        try:
            float(content)
        except ValueError:
            if 'Similarity threshold' in content:
                text.set_fontsize(12.4)
                text.set_fontweight('bold')
                text.set_color('0.30')
                text.set_position((0.615, 0.05))
            continue
        text.set_fontweight('bold')
        text.set_fontsize(13)
        text.set_color('black')


def _draw_tanimoto_panel(ax: plt.Axes, df: pd.DataFrame, panel_letter: str, title: str, color: str, threshold: float = 0.6) -> None:
    y = np.arange(len(df))
    ax.barh(y, df['MaxSimilarity'], color=color, edgecolor='#62666d', linewidth=0.8, height=0.70, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df['Molecule_name'], fontsize=15.2, color=AXIS_COLOR)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.grid(False)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.55, alpha=0.24, color=GRID_COLOR, zorder=0)
    ax.set_title(title, fontsize=14.8, fontweight='bold', pad=10)
    ax.text(-0.14, 1.03, panel_letter, transform=ax.transAxes, fontsize=23.0, fontweight='bold', ha='left', va='top', color='#222222')
    ax.axvline(threshold, color='#5f6368', linestyle=(0, (4, 3)), linewidth=1.25, alpha=0.95, zorder=2)
    ax.text(threshold + 0.015, 0.062, 'Similarity threshold = 0.60', transform=ax.get_xaxis_transform(), fontsize=12.4, fontweight='bold', color='0.30', ha='left', va='bottom')
    ax.tick_params(axis='x', labelsize=13.4, width=0.85, length=3.6, colors=AXIS_COLOR)
    ax.tick_params(axis='y', labelsize=15.2, width=0.0, length=0, colors=AXIS_COLOR, pad=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)
    ax.spines['left'].set_color(AXIS_COLOR)
    ax.spines['bottom'].set_color(AXIS_COLOR)
    ax.set_xlabel('')
    for yi, value in zip(y, df['MaxSimilarity']):
        ax.text(min(value + 0.012, 1.03), yi, f'{value:.3f}', va='center', ha='left', fontsize=13, fontweight='bold', color='black')
    _style_tanimoto_texts(ax)


def build_tanimoto_figure(training_path: Path, cannabinoids_path: Path, terpenes_path: Path, stem: str, endpoint_label: str) -> None:
    training_df = load_training_set() if training_path == TRAINING_PATH else load_external_set(training_path, 'Training set')
    cannabinoids_df = load_external_set(cannabinoids_path, 'Cannabinoids')
    terpenes_df = load_external_set(terpenes_path, 'Terpenes')

    cannabinoids_sim = build_tanimoto_similarity_table(training_df, cannabinoids_df)
    terpenes_sim = build_tanimoto_similarity_table(training_df, terpenes_df).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 8.7), dpi=FIG_DPI)
    _draw_tanimoto_panel(axes[0], cannabinoids_sim, 'A', 'Cannabinoids', '#4C72B0')
    _draw_tanimoto_panel(axes[1], terpenes_sim, 'B', 'Terpenes', '#DD8452')
    for ax in axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xlim(0.0, 1.05)
        _style_tanimoto_texts(ax)
    fig.supxlabel('Maximum Tanimoto similarity to training set', x=0.5, ha='center', fontsize=15.4, fontweight='bold', y=0.05)
    fig.subplots_adjust(left=0.225, right=0.985, top=0.945, bottom=0.12, wspace=0.40)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUTPUT_DIR / f'{stem}.png', dpi=FIG_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / f'{stem}.tiff', dpi=FIG_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / f'{stem}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def build_tanimoto_figure_ic50() -> None:
    build_tanimoto_figure(TRAINING_PATH, CANNABINOIDS_PATH, TERPENES_PATH, 'Figure_SX_Tanimoto', 'IC50 model')


def build_tanimoto_figure_ec50() -> None:
    build_tanimoto_figure(EC50_TRAINING_PATH, EC50_CANNABINOIDS_PATH, EC50_TERPENES_PATH, 'Figure_SX_Tanimoto_EC50', 'EC50 model')

def main() -> None:
    adjust_text = check_required_modules()
    configure_style()
    ensure_output_dir()
    run_endpoint_pipeline(
        TRAINING_PATH,
        CANNABINOIDS_PATH,
        TERPENES_PATH,
        'pca_coordinates.csv',
        r'Chemical Space PCA of Cannabinoids and Terpenes ($\mathbf{IC_{50}}$)',
        'PCA_ChemicalSpace_SI_JCIM_Final',
        'IC50',
        adjust_text,
    )
    build_tanimoto_figure_ic50()
    run_endpoint_pipeline(
        EC50_TRAINING_PATH,
        EC50_CANNABINOIDS_PATH,
        EC50_TERPENES_PATH,
        'pca_coordinates_ec50.csv',
        r'Chemical Space PCA of Cannabinoids and Terpenes ($\mathbf{EC_{50}}$)',
        'PCA_ChemicalSpace_EC50_SI_JCIM_Final',
        'EC50',
        adjust_text,
        include_ellipse=False,
        hull_color=EC50_HULL_COLOR,
        hull_linewidth=1.05,
        hull_alpha=0.62,
    )
    build_tanimoto_figure_ec50()


if __name__ == '__main__':
    main()
