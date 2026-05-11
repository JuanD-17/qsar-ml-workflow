# Este script compara valores experimentales vs. predichos para IC50 y EC50,
# clasifica la concordancia por error absoluto y resume qué rasgos estructurales
# parecen estar mejor o peor capturados por el modelo QSAR.

from __future__ import annotations

import csv
import shutil
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as _pd
    import numpy as _np
    from numpy.typing import NDArray

    DataFrame = _pd.DataFrame
    Series = _pd.Series
    FloatArray = NDArray[_np.float64]
else:
    DataFrame = Any
    Series = Any
    FloatArray = Any


REQUIRED_MODULES = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scipy': 'scipy',
    'rdkit': 'rdkit',
}


def check_dependencies():
    missing = []
    imported = {}
    for package, import_name in REQUIRED_MODULES.items():
        try:
            imported[import_name] = __import__(import_name)
        except Exception:
            missing.append(package)
    if missing:
        print('Missing required dependencies:')
        for package in missing:
            print(f'  - {package}')
        print(f"\nInstall them with: pip install {' '.join(missing)}")
        raise SystemExit(1)
    return imported


mods = check_dependencies()
pd = mods['pandas']
np = mods['numpy']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, VPacker, TextArea
from scipy.stats import gaussian_kde
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog('rdApp.*')


BASE_DIR = Path('/Users/juandavidhoyostrejos/Documents/Icesi/qsar')
OUTPUT_DIR = BASE_DIR / 'ensayo349'
PREDICTION_FILES = {
    'IC50': BASE_DIR / 'ensayo336' / 'molecules3_consolidated_predictions.csv',
    'EC50': BASE_DIR / 'molecules3(EC50).csv',
}
EXPERIMENTAL_CSV = BASE_DIR / 'ensayo348' / 'external_validation_results2.csv'
CHEBI_ANNOTATION_CSV = BASE_DIR / 'ensayo331' / 'chebi_annotation_combined.csv'
PREDICTED_COL = 'Weighted_Prediction_Log10_nM'
EXPERIMENTAL_COL = 'Log10 Value (nM)'

CLEAN_RESULTS = {
    'IC50': OUTPUT_DIR / 'IC50_clean_results.csv',
    'EC50': OUTPUT_DIR / 'EC50_clean_results.csv',
}
STRUCTURAL_SUMMARY = {
    'IC50': OUTPUT_DIR / 'IC50_structural_summary.csv',
    'EC50': OUTPUT_DIR / 'EC50_structural_summary.csv',
}
COMPARISON_PANEL_PNG = OUTPUT_DIR / 'IC50_EC50_comparison_panel.png'
COMPARISON_PANEL_TIFF = OUTPUT_DIR / 'IC50_EC50_comparison_panel.tiff'
COMPARISON_PANEL_PDF = OUTPUT_DIR / 'IC50_EC50_comparison_panel.pdf'
COMPARISON_PANEL_SVG = OUTPUT_DIR / 'IC50_EC50_comparison_panel.svg'

AGREEMENT_ORDER = [
    'good_agreement_strict',
    'good_agreement_relaxed',
    'poor_agreement',
]

AGREEMENT_LABELS = {
    'good_agreement_strict': 'Good agreement (<= 0.3)',
    'good_agreement_relaxed': 'Relaxed agreement (0.3-0.5)',
    'poor_agreement': 'Poor agreement (> 0.5)',
}

def sniff_delimiter(path: Path) -> str:
    sample = path.read_text(errors='replace').splitlines()[:20]
    sample_text = '\n'.join(sample)
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=',;\t|').delimiter
    except Exception:
        return ','


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_column_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(name).strip().lower())


def normalize_column_names(df: DataFrame) -> Dict[str, str]:
    return {normalize_column_name(col): col for col in df.columns}


def column_as_series(df: DataFrame, column_name: str) -> Series:
    data = df.loc[:, column_name]
    if getattr(data, 'ndim', 1) == 2:
        return data.iloc[:, 0]
    return data


def clean_name_column(series: Series) -> Series:
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
    )


def equivalent_columns(df: DataFrame, kind: str) -> List[str]:
    normalized = normalize_column_names(df)
    target_map = {
        'CID': {'cid', 'pubchemcid', 'compoundcid'},
        'InChIKey': {'inchikey', 'inchikey14', 'inchi_key'},
        'SMILES': {'smiles', 'canonicalsmiles', 'inputsmiles', 'matchedcanonicalsmiles'},
        'Compound Name': {'compoundname', 'moleculename', 'inputmoleculename', 'matchedmoleculename', 'name'},
    }
    targets = {normalize_column_name(value) for value in target_map.get(kind, set())}
    return [original for norm, original in normalized.items() if norm in targets]


def detect_merge_key(left_df: DataFrame, right_df: DataFrame) -> Tuple[str, str, str]:
    for key_kind in ['CID', 'InChIKey', 'SMILES', 'Compound Name']:
        left_candidates = equivalent_columns(left_df, key_kind)
        right_candidates = equivalent_columns(right_df, key_kind)
        if not left_candidates or not right_candidates:
            continue
        for left_col in left_candidates:
            for right_col in right_candidates:
                left = left_df[[left_col]].dropna().copy()
                right = right_df[[right_col]].dropna().copy()
                if key_kind == 'Compound Name':
                    left['_merge_key'] = clean_name_column(column_as_series(left, left_col))
                    right['_merge_key'] = clean_name_column(column_as_series(right, right_col))
                else:
                    left['_merge_key'] = column_as_series(left, left_col).astype(str).str.strip()
                    right['_merge_key'] = column_as_series(right, right_col).astype(str).str.strip()
                matches = left.merge(right, on='_merge_key', how='inner')
                if len(matches) > 0:
                    print(f'[{key_kind}] merge key selected: {left_col} <-> {right_col} ({len(matches):,} matches)')
                    return key_kind, left_col, right_col
    raise SystemExit('No compatible merge key found between prediction and experimental datasets.')


def load_endpoint_pair(endpoint: str) -> Tuple[DataFrame, DataFrame]:
    prediction_csv = PREDICTION_FILES[endpoint]
    pred = pd.read_csv(prediction_csv, sep=sniff_delimiter(prediction_csv))
    exp_all = pd.read_csv(EXPERIMENTAL_CSV, sep=sniff_delimiter(EXPERIMENTAL_CSV))
    exp = exp_all.loc[
        exp_all['standard_type'].astype(str).str.upper() == endpoint.upper()
    ].copy()

    print(f'\n=== {endpoint} input inspection ===')
    print(f'Prediction shape: {pred.shape}')
    print(f'Experimental shape: {exp.shape}')
    print(f'Prediction columns: {list(pred.columns)}')
    print(f'Experimental columns: {list(exp.columns)}')
    return pred, exp


def choose_preferred_column(df: DataFrame, candidates: List[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def smiles_to_inchikey(smiles: str) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        return ''
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''
        return Chem.MolToInchiKey(mol)
    except Exception:
        return ''


def load_chebi_annotations() -> DataFrame:
    if not CHEBI_ANNOTATION_CSV.exists():
        print(f'ChEBI annotation file not found: {CHEBI_ANNOTATION_CSV}')
        return pd.DataFrame(columns=['SMILES', 'InChIKey', 'CHEBI_Primary_Class'])

    chebi = pd.read_csv(CHEBI_ANNOTATION_CSV, low_memory=False)
    smiles_col = choose_preferred_column(chebi, ['Smiles', 'SMILES', 'smiles'])
    if not smiles_col:
        print('No SMILES column found in ChEBI annotation file.')
        return pd.DataFrame(columns=['SMILES', 'InChIKey', 'CHEBI_Primary_Class'])

    chebi_work = chebi.copy()
    chebi_work['SMILES'] = chebi_work[smiles_col].astype(str).str.strip()
    chebi_work['InChIKey'] = chebi_work.get('InChIKey', '').fillna('').astype(str).str.strip()
    chebi_work['CHEBI_Primary_Class'] = (
        chebi_work.get('CHEBI_Primary_Class', 'Unclassified')
        .fillna('Unclassified')
        .astype(str)
        .str.strip()
        .replace({'': 'Unclassified'})
    )
    chebi_work = chebi_work.loc[chebi_work['SMILES'] != ''].copy()
    chebi_work = chebi_work[['SMILES', 'InChIKey', 'CHEBI_Primary_Class']].drop_duplicates()
    print(f'ChEBI annotations loaded: {len(chebi_work):,} unique SMILES')
    return chebi_work


def merge_and_aggregate(endpoint: str, chebi_annotations: DataFrame) -> DataFrame:
    pred, exp = load_endpoint_pair(endpoint)
    key_kind, pred_key, exp_key = detect_merge_key(pred, exp)

    pred_work = pred.copy()
    exp_work = exp.copy()
    if key_kind == 'Compound Name':
        pred_work['_merge_key'] = clean_name_column(column_as_series(pred_work, pred_key))
        exp_work['_merge_key'] = clean_name_column(column_as_series(exp_work, exp_key))
    else:
        pred_work['_merge_key'] = column_as_series(pred_work, pred_key).astype(str).str.strip()
        exp_work['_merge_key'] = column_as_series(exp_work, exp_key).astype(str).str.strip()

    merged = pred_work.merge(exp_work, on='_merge_key', how='inner', suffixes=('_pred', '_exp'))
    merged['experimental_log10_nM'] = pd.to_numeric(merged[EXPERIMENTAL_COL], errors='coerce')
    merged['predicted_log10_nM'] = pd.to_numeric(merged[PREDICTED_COL], errors='coerce')
    merged = merged.dropna(subset=['experimental_log10_nM', 'predicted_log10_nM']).copy()
    merged = merged.drop_duplicates().copy()

    smiles_col = choose_preferred_column(
        merged,
        ['SMILES', 'SMILES_pred', 'input_smiles', 'matched_canonical_smiles', 'canonical_smiles', 'smiles'],
    )
    name_col = choose_preferred_column(
        merged,
        ['Molecule_name', 'input_molecule_name', 'matched_molecule_name', 'Compound Name', 'compound_name', 'name'],
    )
    cid_col = choose_preferred_column(merged, ['CID', 'cid', 'pubchem_cid'])
    inchikey_col = choose_preferred_column(merged, ['InChIKey', 'inchikey'])

    grouped = merged.groupby('_merge_key', dropna=False)
    aggregated = grouped.agg(
        experimental_log10_nM=('experimental_log10_nM', 'median'),
        predicted_log10_nM=('predicted_log10_nM', 'mean'),
        n_measurements=('experimental_log10_nM', 'size'),
    ).reset_index()

    if name_col:
        names = grouped[name_col].agg(lambda s: s.dropna().astype(str).iloc[0] if not s.dropna().empty else '')
        aggregated['compound_name'] = aggregated['_merge_key'].map(names)
    else:
        aggregated['compound_name'] = aggregated['_merge_key']

    if smiles_col:
        smiles = grouped[smiles_col].agg(lambda s: s.dropna().astype(str).iloc[0] if not s.dropna().empty else '')
        aggregated['SMILES'] = aggregated['_merge_key'].map(smiles)
    else:
        aggregated['SMILES'] = ''

    if cid_col:
        cid_values = grouped[cid_col].agg(lambda s: s.dropna().astype(str).iloc[0] if not s.dropna().empty else '')
        aggregated['CID'] = aggregated['_merge_key'].map(cid_values)
    if inchikey_col:
        inchikey_values = grouped[inchikey_col].agg(lambda s: s.dropna().astype(str).iloc[0] if not s.dropna().empty else '')
        aggregated['InChIKey'] = aggregated['_merge_key'].map(inchikey_values)

    aggregated['endpoint'] = endpoint
    aggregated['residual'] = aggregated['predicted_log10_nM'] - aggregated['experimental_log10_nM']
    aggregated['absolute_error'] = (aggregated['predicted_log10_nM'] - aggregated['experimental_log10_nM']).abs()
    aggregated['agreement_category'] = aggregated['absolute_error'].apply(classify_agreement)
    aggregated['InChIKey'] = aggregated['SMILES'].apply(smiles_to_inchikey)
    chebi_by_smiles = chebi_annotations.drop_duplicates(subset=['SMILES'])
    chebi_by_inchikey = chebi_annotations.loc[
        chebi_annotations['InChIKey'].fillna('').astype(str).str.strip() != ''
    ].drop_duplicates(subset=['InChIKey'])

    aggregated = aggregated.merge(
        chebi_by_smiles.rename(columns={'InChIKey': 'chebi_inchikey_from_smiles', 'CHEBI_Primary_Class': 'chebi_class_from_smiles'}),
        on='SMILES',
        how='left',
    )
    aggregated = aggregated.merge(
        chebi_by_inchikey.rename(columns={'SMILES': 'chebi_smiles_from_inchikey', 'CHEBI_Primary_Class': 'chebi_class_from_inchikey'}),
        on='InChIKey',
        how='left',
    )
    aggregated['CHEBI_Primary_Class'] = aggregated['chebi_class_from_smiles'].combine_first(
        aggregated['chebi_class_from_inchikey']
    )
    aggregated['CHEBI_Primary_Class'] = (
        aggregated['CHEBI_Primary_Class'].fillna('Unclassified').astype(str).replace({'': 'Unclassified'})
    )
    aggregated = aggregated.drop(
        columns=[
            'chebi_inchikey_from_smiles',
            'chebi_class_from_smiles',
            'chebi_smiles_from_inchikey',
            'chebi_class_from_inchikey',
        ],
        errors='ignore',
    )

    export_cols = [
        'SMILES',
        'InChIKey',
        'compound_name',
        'endpoint',
        'experimental_log10_nM',
        'predicted_log10_nM',
        'residual',
        'absolute_error',
        'agreement_category',
        'CHEBI_Primary_Class',
    ]
    aggregated[export_cols].to_csv(CLEAN_RESULTS[endpoint], index=False)
    print(f'{endpoint}: merged measurements = {len(merged):,}, molecules exported = {len(aggregated):,}')
    return aggregated


def classify_agreement(absolute_error: float) -> str:
    if absolute_error <= 0.3:
        return 'good_agreement_strict'
    if absolute_error <= 0.5:
        return 'good_agreement_relaxed'
    return 'poor_agreement'


def build_structural_summary(df: DataFrame, endpoint: str) -> DataFrame:
    summary = (
        df.groupby(['CHEBI_Primary_Class', 'agreement_category'], dropna=False)
        .size()
        .reset_index(name='n_compounds')
        .pivot(index='CHEBI_Primary_Class', columns='agreement_category', values='n_compounds')
        .fillna(0)
        .reset_index()
    )
    for category in AGREEMENT_ORDER:
        if category not in summary.columns:
            summary[category] = 0
    summary['total_compounds'] = summary[AGREEMENT_ORDER].sum(axis=1)
    summary.insert(0, 'endpoint', endpoint)
    summary = summary[['endpoint', 'CHEBI_Primary_Class', 'total_compounds', *AGREEMENT_ORDER]]
    summary = summary.sort_values(['total_compounds', 'CHEBI_Primary_Class'], ascending=[False, True]).reset_index(drop=True)
    summary.to_csv(STRUCTURAL_SUMMARY[endpoint], index=False)
    return summary


def configure_plot_style() -> None:
    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 12,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9.5,
        'axes.linewidth': 1.0,
        'grid.color': '#e8e8e8',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.4,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def equal_axis_limits(x: FloatArray, y: FloatArray, margin: float = 0.05) -> Tuple[float, float]:
    minimum = float(min(np.min(x), np.min(y)))
    maximum = float(max(np.max(x), np.max(y)))
    span = maximum - minimum if maximum > minimum else 1.0
    pad = span * margin
    return minimum - pad, maximum + pad


def scatter_density(x: FloatArray, y: FloatArray) -> Tuple[FloatArray, FloatArray, FloatArray]:
    xy = np.vstack([x, y])
    densities = gaussian_kde(xy)(xy)
    order = np.argsort(densities)
    return x[order], y[order], densities[order]


def draw_concordance_panel(ax, df: DataFrame, endpoint_display: str, panel_label: str) -> None:
    x = df['experimental_log10_nM'].to_numpy(dtype=float)
    y = df['predicted_log10_nM'].to_numpy(dtype=float)
    x_sorted, y_sorted, density = scatter_density(x, y)
    limits = equal_axis_limits(x, y, margin=0.07)
    endpoint_bold = r'$\mathbf{IC}_{\mathbf{50}}$' if 'IC' in endpoint_display else r'$\mathbf{EC}_{\mathbf{50}}$'
    log_bold = r'$\mathbf{log}_{\mathbf{10}}$'
    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    metric_x, metric_y = (0.035, 0.965)

    scatter = ax.scatter(
        x_sorted,
        y_sorted,
        c=density,
        cmap='RdBu_r',
        s=66,
        alpha=0.85,
        edgecolors='black',
        linewidths=0.4,
        zorder=3,
    )
    ax.plot(limits, limits, '--', color='black', linewidth=0.95, label='Identity line', zorder=2)

    if len(df) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(limits[0], limits[1], 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='#c94f3d', linewidth=2.0, label='OLS fit', zorder=2)

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'Experimental {log_bold} {endpoint_bold} (nM)', fontweight='bold', labelpad=8)
    ax.set_ylabel(f'Predicted {log_bold} {endpoint_bold} (nM)', fontweight='bold', labelpad=8)
    ax.text(
        -0.03,
        1.035,
        f'{panel_label}. {endpoint_bold}',
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=16,
        fontweight='bold',
        color='#2b2f38',
        clip_on=False,
        zorder=5,
    )
    metric_style = dict(size=11.8, color='#2b2f38')
    metric_labels = VPacker(
        children=[
            TextArea(r'$\mathbf{R^2}$', textprops=metric_style),
            TextArea(r'$\mathbf{RMSE}$', textprops=metric_style),
        ],
        align='left',
        pad=0,
        sep=2,
    )
    metric_values = VPacker(
        children=[
            TextArea(f'= {r_squared:.2f}', textprops=metric_style),
            TextArea(f'= {rmse:.2f}', textprops=metric_style),
        ],
        align='left',
        pad=0,
        sep=2,
    )
    metric_block = HPacker(children=[metric_labels, metric_values], align='center', pad=0, sep=7)
    metric_box = AnchoredOffsetbox(
        loc='upper left',
        child=metric_block,
        pad=0.25,
        borderpad=0.0,
        frameon=True,
        bbox_to_anchor=(metric_x, metric_y),
        bbox_transform=ax.transAxes,
    )
    metric_box.patch.set_facecolor('white')
    metric_box.patch.set_edgecolor('none')
    metric_box.patch.set_alpha(0.56)
    metric_box.patch.set_linewidth(0.0)
    metric_box.set_zorder(6)
    ax.add_artist(metric_box)
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(1.0)
    ax.grid(True, color='#e8e8e8', linewidth=0.6, alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=11, width=0.9, length=4)
    ax.legend(
        loc='lower right',
        frameon=True,
        facecolor='white',
        edgecolor='#d9d9d9',
        framealpha=0.82,
        fancybox=False,
        borderpad=0.45,
        handlelength=2.0,
        labelspacing=0.35,
    )
    return scatter


def build_comparison_figure(ic50_df: DataFrame, ec50_df: DataFrame):
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2), constrained_layout=False)
    scatter_a = draw_concordance_panel(axes[0], ic50_df, r'IC$_{50}$', 'A')
    scatter_b = draw_concordance_panel(axes[1], ec50_df, r'EC$_{50}$', 'B')
    fig.subplots_adjust(left=0.07, right=0.875, top=0.965, bottom=0.13, wspace=0.20)
    cbar_ax = fig.add_axes([0.895, 0.13, 0.022, 0.835])
    colorbar = fig.colorbar(scatter_b, cax=cbar_ax)
    colorbar.set_label(r'$\mathbf{Point\ density}$', fontsize=13, fontweight='bold', labelpad=10)
    colorbar.ax.tick_params(labelsize=10.5, width=0.8, length=3.5)
    return fig


def save_figure(fig, png_path: Path, tiff_path: Path, pdf_path: Path, svg_path: Path) -> None:
    fig.savefig(png_path, dpi=1200, bbox_inches='tight', facecolor='white')
    fig.savefig(tiff_path, dpi=1200, bbox_inches='tight', facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def print_category_summary(df: DataFrame, endpoint: str) -> None:
    counts = df['agreement_category'].value_counts().reindex(AGREEMENT_ORDER, fill_value=0)
    print(f'\n=== {endpoint} concordance summary ===')
    for category in AGREEMENT_ORDER:
        print(f'{category}: {int(counts[category])}')


def main() -> None:
    reset_output_dir(OUTPUT_DIR)
    chebi_annotations = load_chebi_annotations()

    ic50_df = merge_and_aggregate('IC50', chebi_annotations)
    ec50_df = merge_and_aggregate('EC50', chebi_annotations)

    build_structural_summary(ic50_df, 'IC50')
    build_structural_summary(ec50_df, 'EC50')

    print_category_summary(ic50_df, 'IC50')
    print_category_summary(ec50_df, 'EC50')

    fig = build_comparison_figure(ic50_df, ec50_df)
    save_figure(fig, COMPARISON_PANEL_PNG, COMPARISON_PANEL_TIFF, COMPARISON_PANEL_PDF, COMPARISON_PANEL_SVG)

    print('\n=== Exported files ===')
    for endpoint in ['IC50', 'EC50']:
        print(CLEAN_RESULTS[endpoint])
        print(STRUCTURAL_SUMMARY[endpoint])
    print(COMPARISON_PANEL_PNG)
    print(COMPARISON_PANEL_TIFF)
    print(COMPARISON_PANEL_PDF)
    print(COMPARISON_PANEL_SVG)


if __name__ == '__main__':
    main()
