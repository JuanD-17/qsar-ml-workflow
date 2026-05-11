#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Figura tipo Cleveland para IC50 y EC50 con estilo homogéneo de manuscrito Q1.
# Mantiene los datos originales y mejora únicamente la presentación visual.

from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path('/Users/juandavidhoyostrejos/Documents/Icesi/qsar')
OUT_DIR = BASE_DIR / 'ensayo339'
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_A_IC50 = BASE_DIR / 'ensayo335' / 'Cannabaceae_consolidated_predictions.csv'
INPUT_B_IC50 = BASE_DIR / 'ensayo336' / 'molecules3_consolidated_predictions.csv'
INPUT_A_EC50 = BASE_DIR / 'Cannabaceae_(EC50).csv'
INPUT_B_EC50 = BASE_DIR / 'molecules3(EC50).csv'

COLORS = {
    'dataset_a': '#2f6f9f',
    'dataset_b': '#b6543c',
    'connector': '#bdc8d6',
    'grid': '#d8e1ec',
    'text': '#273142',
}

PNG_DPI = 300
TIFF_DPI = 1200
TOP_N = 10
EXCLUDED_EC50 = {
    "1,3-Benzodioxole, 5,5'-(tetrahydro-1H,3H-furo(3,4-c)furan-1,4-diyl)bis-, (1S-(1alpha,3aalpha,4alpha,6aalpha))-"
}
EC50_XLIM = (5.20, 5.40)
Y_OFFSET = 0.10
LABEL_A = 'Cannabaceae phytochemicals'
LABEL_B = 'Natural Antioxidant Validation Set'
TITLE_IC50 = r'$\mathbf{IC_{50}}$'
TITLE_EC50 = r'$\mathbf{EC_{50}}$'
XLABEL_IC50 = r'$\mathbf{Predicted\ log_{10}(IC_{50})\ (nM)}$'
XLABEL_EC50 = r'$\mathbf{Predicted\ log_{10}(EC_{50})\ (nM)}$'


def read_predictions(path: Path) -> pd.DataFrame:
    # Lee cada CSV detectando separador automáticamente y valida columnas mínimas.
    if not path.exists():
        raise FileNotFoundError(f'No existe el archivo: {path}')
    first_line = path.read_text(encoding='utf-8', errors='ignore').splitlines()[0]
    sep = ';' if first_line.count(';') >= first_line.count(',') else ','
    df = pd.read_csv(path, sep=sep)
    required = {'Molecule_name', 'Final_Category', 'Weighted_Prediction_Log10_nM'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Faltan columnas en {path.name}: {sorted(missing)}')
    return df


def normalize_molecule_name(name: str) -> str:
    # Normaliza nombres para detectar moléculas repetidas aunque cambie capitalización o espacios.
    return re.sub(r'\s+', ' ', str(name).strip()).casefold()


def choose_display_name(names: pd.Series) -> str:
    # Conserva una versión legible del nombre compartido priorizando mayúsculas químicas habituales.
    candidates = [str(n).strip() for n in names if str(n).strip()]
    if not candidates:
        return ''
    def score(value: str) -> tuple[int, int, int]:
        return (
            sum(ch.isupper() for ch in value),
            -value.count('_'),
            -len(value),
        )
    return sorted(candidates, key=score, reverse=True)[0]


def prepare_dataset(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    # Conserva moléculas categoría Bajo y deja solo la mejor predicción por molécula, unificando nombres repetidos.
    work = df.copy()
    work['Molecule_name'] = work['Molecule_name'].astype(str).str.strip()
    work = work[work['Final_Category'].astype(str).str.strip().str.lower() == 'bajo'].copy()
    work['Weighted_Prediction_Log10_nM'] = pd.to_numeric(work['Weighted_Prediction_Log10_nM'], errors='coerce')
    work = work.dropna(subset=['Molecule_name', 'Weighted_Prediction_Log10_nM'])
    work['NameKey'] = work['Molecule_name'].map(normalize_molecule_name)
    work = work.sort_values('Weighted_Prediction_Log10_nM', ascending=True)
    work = work.drop_duplicates(subset=['NameKey'], keep='first')
    work['Source'] = source_label
    return work[['Molecule_name', 'NameKey', 'Weighted_Prediction_Log10_nM', 'Source']]


def build_long_table(path_a: Path, path_b: Path) -> pd.DataFrame:
    # Une ambas fuentes en formato largo y colapsa moléculas repetidas entre datasets.
    a = prepare_dataset(read_predictions(path_a), LABEL_A)
    b = prepare_dataset(read_predictions(path_b), LABEL_B)
    long_df = pd.concat([a, b], ignore_index=True)
    display_map = (
        long_df.groupby('NameKey', as_index=False)['Molecule_name']
        .agg(choose_display_name)
        .rename(columns={'Molecule_name': 'Display_name'})
    )
    long_df = long_df.merge(display_map, on='NameKey', how='left')
    long_df['Molecule_name'] = long_df['Display_name']
    long_df = long_df.drop(columns=['Display_name'])
    return long_df.rename(columns={'Weighted_Prediction_Log10_nM': 'Predicted_log10_nM'})


def select_order(long_df: pd.DataFrame, top_n: Optional[int] = None, excluded: Optional[set[str]] = None) -> list[str]:
    # Ordena moléculas por menor valor predicho disponible y aplica exclusiones opcionales.
    order_df = (
        long_df.groupby('Molecule_name', as_index=False)['Predicted_log10_nM']
        .min()
        .rename(columns={'Predicted_log10_nM': 'sort_value'})
        .sort_values('sort_value', ascending=True)
    )
    if excluded:
        order_df = order_df[~order_df['Molecule_name'].isin(excluded)]
    if top_n is not None:
        order_df = order_df.head(top_n)
    return order_df['Molecule_name'].tolist()


def make_wide_table(long_df: pd.DataFrame, ordered_molecules: list[str]) -> pd.DataFrame:
    # Pivota a formato ancho manteniendo el orden visual final.
    return (
        long_df.pivot(index='Molecule_name', columns='Source', values='Predicted_log10_nM')
        .reindex(ordered_molecules)
        .reset_index()
    )


def save_csvs(prefix: str, long_df: pd.DataFrame, wide_df: pd.DataFrame) -> None:
    # Guarda tablas auxiliares del panel para trazabilidad.
    long_df.to_csv(OUT_DIR / f'{prefix}_long.csv', index=False)
    wide_df.to_csv(OUT_DIR / f'{prefix}_wide.csv', index=False)


def style_axis(ax: plt.Axes) -> None:
    # Aplica un estilo uniforme y limpio, con grilla sutil solo en el eje X.
    ax.set_facecolor('white')
    ax.grid(True, axis='x', linestyle='--', linewidth=0.8, alpha=0.45, color=COLORS['grid'])
    ax.grid(False, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.tick_params(axis='x', labelsize=17.8, width=1.0, colors=COLORS['text'])
    ax.tick_params(axis='y', labelsize=16.2, width=1.0, colors=COLORS['text'])


def _compute_xlim(data_values: np.ndarray) -> tuple[float, float]:
    # Calcula márgenes homogéneos sin forzar el mismo rango entre endpoints.
    x_min = np.nanmin(data_values)
    x_max = np.nanmax(data_values)
    pad = max((x_max - x_min) * 0.10, 0.04)
    return x_min - pad, x_max + pad


def plot_cleveland(
    ax: plt.Axes,
    wide_df: pd.DataFrame,
    panel_letter: str,
    endpoint_title: str,
    xlabel: str,
    show_ylabel: bool,
    xlim: Optional[tuple[float, float]] = None,
) -> tuple[list, list]:
    # Dibuja un Cleveland plot con offsets verticales sutiles para evitar solapamiento.
    y_positions = np.arange(len(wide_df))
    style_axis(ax)

    x_a = wide_df[LABEL_A].to_numpy(dtype=float) if LABEL_A in wide_df else np.full(len(wide_df), np.nan)
    x_b = wide_df[LABEL_B].to_numpy(dtype=float) if LABEL_B in wide_df else np.full(len(wide_df), np.nan)

    # Línea conectora tenue para moléculas presentes en ambos conjuntos.
    both_mask = ~np.isnan(x_a) & ~np.isnan(x_b)
    for i in np.where(both_mask)[0]:
        ax.plot([x_a[i], x_b[i]], [y_positions[i], y_positions[i]], color=COLORS['connector'], linewidth=1.25, alpha=0.9, zorder=1)

    y_a = y_positions + Y_OFFSET
    y_b = y_positions - Y_OFFSET
    scat_a = ax.scatter(
        x_a,
        y_a,
        s=121,
        color=COLORS['dataset_a'],
        edgecolors='white',
        linewidths=1.0,
        label=LABEL_A,
        zorder=3,
    )
    scat_b = ax.scatter(
        x_b,
        y_b,
        s=125,
        marker='^',
        color=COLORS['dataset_b'],
        edgecolors='white',
        linewidths=1.0,
        label=LABEL_B,
        zorder=4,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(wide_df['Molecule_name'], fontsize=16.2, color=COLORS['text'])
    ax.invert_yaxis()

    if xlim is None:
        ax.set_xlim(*_compute_xlim(wide_df.iloc[:, 1:].to_numpy(dtype=float)))
    else:
        ax.set_xlim(*xlim)

    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax.set_xlabel(xlabel, fontsize=20.1, fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Molecule' if show_ylabel else '', fontsize=20.1, fontweight='bold', color=COLORS['text'])
    ax.set_title(endpoint_title, fontsize=22.4, fontweight='bold', pad=14, color=COLORS['text'])
    ax.text(-0.12, 1.02, panel_letter, transform=ax.transAxes, fontsize=32.2, fontweight='bold', color=COLORS['text'], va='bottom', ha='left')
    return [scat_a, scat_b], [LABEL_A, LABEL_B]


def build_data(path_a: Path, path_b: Path, top_n: Optional[int] = None, excluded: Optional[set[str]] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Construye tablas larga y ancha ya filtradas y ordenadas para un endpoint.
    long_df = build_long_table(path_a, path_b)
    ordered = select_order(long_df, top_n=top_n, excluded=excluded)
    long_df = long_df[long_df['Molecule_name'].isin(ordered)].copy()
    long_df['Molecule_name'] = pd.Categorical(long_df['Molecule_name'], categories=ordered, ordered=True)
    long_df = long_df.sort_values(['Molecule_name', 'Source'])
    wide_df = make_wide_table(long_df, ordered)
    return long_df, wide_df


def save_individual_panel(
    wide_df: pd.DataFrame,
    endpoint_title: str,
    xlabel: str,
    panel_letter: str,
    stem: str,
    xlim: Optional[tuple[float, float]] = None,
) -> None:
    # Guarda un panel individual manteniendo la estética del panel combinado.
    fig, ax = plt.subplots(figsize=(13.6, 9.6), constrained_layout=False)
    handles, labels = plot_cleveland(
        ax,
        wide_df,
        panel_letter=panel_letter,
        endpoint_title=endpoint_title,
        xlabel=xlabel,
        show_ylabel=True,
        xlim=xlim,
    )
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=14.3, handletextpad=0.6, columnspacing=1.6)
    plt.subplots_adjust(left=0.29, right=0.98, top=0.88, bottom=0.12)
    fig.savefig(OUT_DIR / f'{stem}.png', dpi=PNG_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / f'{stem}.tiff', dpi=TIFF_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / f'{stem}.svg', bbox_inches='tight', facecolor='white')
    try:
        fig.savefig(OUT_DIR / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    except Exception as exc:
        print(f'Advertencia: no se pudo exportar PDF para {stem}: {exc}')
    plt.close(fig)


def build_combined(ic50_wide: pd.DataFrame, ec50_wide: pd.DataFrame) -> None:
    # Construye el panel AB final sin título global y con una sola leyenda compartida.
    fig, axes = plt.subplots(1, 2, figsize=(20.0, 10.2), constrained_layout=False)
    handles, labels = plot_cleveland(
        axes[0],
        ic50_wide,
        panel_letter='A',
        endpoint_title=TITLE_IC50,
        xlabel=XLABEL_IC50,
        show_ylabel=True,
    )
    plot_cleveland(
        axes[1],
        ec50_wide,
        panel_letter='B',
        endpoint_title=TITLE_EC50,
        xlabel=XLABEL_EC50,
        show_ylabel=False,
        xlim=EC50_XLIM,
    )

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.992), ncol=2, frameon=False, fontsize=15.4, handletextpad=0.6, columnspacing=1.6)
    plt.subplots_adjust(left=0.11, right=0.985, top=0.88, bottom=0.11, wspace=0.34)

    fig.savefig(OUT_DIR / 'Cleveland_High_Potency_Panels_AB.png', dpi=PNG_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'Cleveland_High_Potency_Panels_AB.tiff', dpi=TIFF_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'Cleveland_High_Potency_Panels_AB.svg', bbox_inches='tight', facecolor='white')
    try:
        fig.savefig(OUT_DIR / 'Cleveland_High_Potency_Panels_AB.pdf', bbox_inches='tight', facecolor='white')
    except Exception as exc:
        print(f'Advertencia: no se pudo exportar PDF para el combinado: {exc}')
    plt.close(fig)


def main() -> None:
    # Genera paneles individuales y el panel combinado conservando el contenido original.
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titleweight': 'bold',
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
    })

    ic50_long, ic50_wide = build_data(INPUT_A_IC50, INPUT_B_IC50, top_n=TOP_N, excluded=None)
    ec50_long, ec50_wide = build_data(INPUT_A_EC50, INPUT_B_EC50, top_n=TOP_N, excluded=EXCLUDED_EC50)

    save_csvs('cleveland_high_potency_panel_a', ic50_long, ic50_wide)
    save_csvs('cleveland_high_potency_panel_b', ec50_long, ec50_wide)

    save_individual_panel(ic50_wide, TITLE_IC50, XLABEL_IC50, 'A', 'Cleveland_High_Potency_Panel_A')
    save_individual_panel(ec50_wide, TITLE_EC50, XLABEL_EC50, 'B', 'Cleveland_High_Potency_Panel_B', xlim=EC50_XLIM)
    build_combined(ic50_wide, ec50_wide)


if __name__ == '__main__':
    main()
