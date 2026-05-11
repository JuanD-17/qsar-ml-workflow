# Este script filtra los resultados de validacion externa con evidencia antioxidante para actividades IC50 y EC50,
# normaliza los valores convertibles a nM, calcula su transformacion Log10, y exporta un CSV final
# solo con los registros que se pudieron convertir correctamente, junto con un resumen del proceso.

from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd

BASE_DIR = Path('/Users/juandavidhoyostrejos/Documents/Icesi/qsar')
INPUT_CSV = BASE_DIR / 'external_validation_results.csv'
OUTPUT_DIR = BASE_DIR / 'ensayo348'
OUTPUT_CSV = OUTPUT_DIR / 'external_validation_results2.csv'
SUMMARY_TXT = OUTPUT_DIR / 'ensayo348_summary.txt'

VALID_STANDARD_TYPES = {'IC50', 'EC50'}


def clean_text(value) -> str:
    if pd.isna(value):
        return ''
    return str(value).strip()


def to_float(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(',', '.')
    try:
        return float(text)
    except ValueError:
        return np.nan


def normalize_units(units: str) -> str:
    text = clean_text(units)
    if not text:
        return ''
    text = text.replace('μ', 'u').replace('µ', 'u')
    text = re.sub(r'\s+', '', text.lower())
    return text


def antioxidant_row_mask(df: pd.DataFrame) -> pd.Series:
    evidence = df['antioxidant_evidence'].fillna('').astype(str).str.strip().ne('')
    related = df['is_antioxidant_related'].fillna(False).astype(bool)
    endpoint = df['standard_type'].fillna('').astype(str).str.upper().isin(VALID_STANDARD_TYPES)
    assay_present = df['assay_chembl_id'].fillna('').astype(str).str.strip().ne('')
    return evidence & related & endpoint & assay_present


def standard_to_nm(standard_value, standard_units):
    value = to_float(standard_value)
    units = normalize_units(standard_units)

    if pd.isna(value) or value <= 0 or not units:
        return np.nan, 'not_convertible'

    if units == 'nm':
        return value, 'standard_value'
    if units in {'um', 'microm/min', 'microm'}:
        return value * 1_000.0, 'standard_value'
    if units in {'mm', 'mmol/l'}:
        return value * 1_000_000.0, 'standard_value'
    if units in {'m', 'mol/l'}:
        return value * 1_000_000_000.0, 'standard_value'
    if units == '10^-5um':
        return value * (10 ** -5) * 1_000.0, 'standard_value'

    # Units like ug.mL-1, ug, ppm, %, 10^-5mg/ml need molecular weight or a more specific rule.
    return np.nan, 'requires_fallback'


def pchembl_to_nm(pchembl_value):
    pchembl = to_float(pchembl_value)
    if pd.isna(pchembl):
        return np.nan
    return 10 ** (9.0 - pchembl)


def compute_normalized_and_log10(row: pd.Series) -> pd.Series:
    normalized_nm, source = standard_to_nm(row['standard_value'], row['standard_units'])

    if pd.isna(normalized_nm):
        fallback_nm = pchembl_to_nm(row['pchembl_value'])
        if pd.notna(fallback_nm) and fallback_nm > 0:
            normalized_nm = fallback_nm
            source = 'pchembl_value'

    log10_nm = round(math.log10(normalized_nm), 3) if pd.notna(normalized_nm) and normalized_nm > 0 else np.nan
    return pd.Series({
        'Normalized Value (nM)': normalized_nm,
        'Log10 Value (nM)': log10_nm,
        'Normalization Source': source,
    })


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df['standard_value'] = df['standard_value'].apply(to_float)
    df['pchembl_value'] = df['pchembl_value'].apply(to_float)

    filtered = df.loc[antioxidant_row_mask(df)].copy()
    filtered['antioxidant_activity_confirmed'] = True
    filtered['endpoint_type_valid'] = filtered['standard_type'].fillna('').astype(str).str.upper().isin(VALID_STANDARD_TYPES)

    conversions = filtered.apply(compute_normalized_and_log10, axis=1)
    filtered = pd.concat([filtered, conversions], axis=1)
    filtered = filtered[filtered['Normalized Value (nM)'].notna() & filtered['Log10 Value (nM)'].notna()].copy()

    ordered_columns = list(filtered.columns)
    for extra in ['Normalized Value (nM)', 'Log10 Value (nM)', 'Normalization Source']:
        if extra in ordered_columns:
            ordered_columns.remove(extra)
    ordered_columns.extend(['Normalized Value (nM)', 'Log10 Value (nM)', 'Normalization Source'])
    filtered = filtered[ordered_columns]

    filtered.to_csv(OUTPUT_CSV, index=False)

    summary_lines = [
        'ensayo348 summary',
        f'Input rows: {len(df):,}',
        f'Filtered rows (antioxidant evidence + IC50/EC50 + assay): {len(filtered):,}',
        '',
        'Standard type counts:',
        filtered['standard_type'].fillna('NA').value_counts().to_string(),
        '',
        'Standard units counts:',
        filtered['standard_units'].fillna('NA').value_counts().to_string(),
        '',
        'Normalization source counts:',
        filtered['Normalization Source'].fillna('NA').value_counts().to_string(),
        '',
        f'Rows with Normalized Value (nM): {filtered["Normalized Value (nM)"].notna().sum():,}',
        f'Rows with Log10 Value (nM): {filtered["Log10 Value (nM)"].notna().sum():,}',
    ]
    SUMMARY_TXT.write_text('\n'.join(summary_lines))

    print(f'Output CSV: {OUTPUT_CSV}')
    print(f'Summary: {SUMMARY_TXT}')
    print(f'Rows exported: {len(filtered):,}')
    print('Normalization source counts:')
    print(filtered['Normalization Source'].fillna('NA').value_counts().to_string())


if __name__ == '__main__':
    main()
