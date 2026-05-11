"""
Ensayo 332 - Conteo de clases ChEBI por actividad biologica (IC50 vs EC50)

Entrada:
- ensayo331/chebi_annotation_antioxidant11.csv
- ensayo331/chebi_annotation_antioxidant19.csv
  (si no existen, usa ensayo331/chebi_annotation_combined.csv)

Salida (ensayo332/):
- chebi_counts_by_activity_primary.csv
- chebi_counts_percent_by_activity_primary.csv
- chebi_counts_by_activity_multilabel.csv
- chebi_counts_percent_by_activity_multilabel.csv
- chebi_counts_long_primary.csv
- ensayo332_log.txt

Objetivo:
- Contar cuantas moleculas hay por clase estructural (ChEBI) en cada actividad
  biologica (IC50, EC50) usando los resultados anotados en ensayo331.
"""

from __future__ import annotations

import os
import sys
from typing import List

import pandas as pd


INPUT_DIR = "ensayo331"
OUTPUT_DIR = "ensayo332"

FILES_PRIORITY = [
    "chebi_annotation_antioxidant11.csv",
    "chebi_annotation_antioxidant19.csv",
]
FALLBACK_FILE = "chebi_annotation_combined.csv"

ACTIVITY_COL = "Standard Type"
PRIMARY_CLASS_COL = "CHEBI_Primary_Class"
MULTI_CLASS_COL = "CHEBI_Class_List"

ALLOWED_ACTIVITIES = {"IC50", "EC50"}


def log(msg: str, fh=None) -> None:
    print(msg)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


def load_input_tables(input_dir: str, log_fh=None) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []
    for f in FILES_PRIORITY:
        p = os.path.join(input_dir, f)
        if os.path.exists(p):
            df = pd.read_csv(p, low_memory=False)
            df["__source_file"] = f
            tables.append(df)
            log(f"✓ Cargado: {p} ({len(df):,} filas)", log_fh)

    if not tables:
        p = os.path.join(input_dir, FALLBACK_FILE)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"No se encontraron archivos en {input_dir}. "
                f"Esperado: {FILES_PRIORITY} o {FALLBACK_FILE}."
            )
        df = pd.read_csv(p, low_memory=False)
        df["__source_file"] = FALLBACK_FILE
        tables = [df]
        log(f"✓ Cargado fallback: {p} ({len(df):,} filas)", log_fh)

    out = pd.concat(tables, ignore_index=True)
    return out


def normalize_activity(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def ensure_columns(df: pd.DataFrame) -> None:
    missing = [
        c
        for c in [ACTIVITY_COL, PRIMARY_CLASS_COL, MULTI_CLASS_COL]
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para ensayo332: {missing}")


def build_primary_counts(df: pd.DataFrame) -> pd.DataFrame:
    # Conteo por clase primaria y actividad
    pivot = (
        df.pivot_table(
            index=PRIMARY_CLASS_COL,
            columns=ACTIVITY_COL,
            values="SMILES" if "SMILES" in df.columns else df.columns[0],
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    for col in ["IC50", "EC50"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["Total"] = pivot["IC50"] + pivot["EC50"]
    pivot = pivot.sort_values("Total", ascending=False).reset_index(drop=True)
    return pivot[[PRIMARY_CLASS_COL, "IC50", "EC50", "Total"]]


def build_primary_percent(primary_counts: pd.DataFrame) -> pd.DataFrame:
    out = primary_counts.copy()
    ic50_total = max(int(out["IC50"].sum()), 1)
    ec50_total = max(int(out["EC50"].sum()), 1)
    total = max(int(out["Total"].sum()), 1)
    out["IC50_%"] = (out["IC50"] / ic50_total * 100.0).round(2)
    out["EC50_%"] = (out["EC50"] / ec50_total * 100.0).round(2)
    out["Total_%"] = (out["Total"] / total * 100.0).round(2)
    return out


def build_multilabel_counts(df: pd.DataFrame) -> pd.DataFrame:
    # Cada molecula puede aportar a varias clases (CHEBI_Class_List)
    expanded = df.copy()
    expanded[MULTI_CLASS_COL] = expanded[MULTI_CLASS_COL].fillna("Unclassified").astype(str)
    expanded[MULTI_CLASS_COL] = expanded[MULTI_CLASS_COL].str.split(r"\s*\|\s*")
    expanded = expanded.explode(MULTI_CLASS_COL)
    expanded[MULTI_CLASS_COL] = expanded[MULTI_CLASS_COL].fillna("Unclassified")

    pivot = (
        expanded.pivot_table(
            index=MULTI_CLASS_COL,
            columns=ACTIVITY_COL,
            values="SMILES" if "SMILES" in expanded.columns else expanded.columns[0],
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={MULTI_CLASS_COL: "CHEBI_Class"})
    )

    for col in ["IC50", "EC50"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["Total"] = pivot["IC50"] + pivot["EC50"]
    pivot = pivot.sort_values("Total", ascending=False).reset_index(drop=True)
    return pivot[["CHEBI_Class", "IC50", "EC50", "Total"]]


def build_multilabel_percent(multilabel_counts: pd.DataFrame) -> pd.DataFrame:
    out = multilabel_counts.copy()
    ic50_total = max(int(out["IC50"].sum()), 1)
    ec50_total = max(int(out["EC50"].sum()), 1)
    total = max(int(out["Total"].sum()), 1)
    out["IC50_%"] = (out["IC50"] / ic50_total * 100.0).round(2)
    out["EC50_%"] = (out["EC50"] / ec50_total * 100.0).round(2)
    out["Total_%"] = (out["Total"] / total * 100.0).round(2)
    return out


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "ensayo332_log.txt")

    with open(log_path, "w", encoding="utf-8") as log_fh:
        log("=" * 72, log_fh)
        log("ENSAYO 332 - CONTEO ChEBI POR ACTIVIDAD BIOLOGICA", log_fh)
        log("=" * 72, log_fh)

        df = load_input_tables(INPUT_DIR, log_fh=log_fh)
        ensure_columns(df)

        df[ACTIVITY_COL] = df[ACTIVITY_COL].apply(normalize_activity)
        df = df[df[ACTIVITY_COL].isin(ALLOWED_ACTIVITIES)].copy()
        if df.empty:
            raise ValueError("No hay filas IC50/EC50 despues de filtrar 'Standard Type'.")

        # Para estabilidad de conteo primario
        df[PRIMARY_CLASS_COL] = df[PRIMARY_CLASS_COL].fillna("Unclassified")

        log(
            f"Filas validas IC50/EC50: {len(df):,} | "
            f"IC50={int((df[ACTIVITY_COL] == 'IC50').sum()):,} | "
            f"EC50={int((df[ACTIVITY_COL] == 'EC50').sum()):,}",
            log_fh,
        )

        # Conteo primario
        primary_counts = build_primary_counts(df)
        primary_pct = build_primary_percent(primary_counts)

        # Conteo multilabel
        multilabel_counts = build_multilabel_counts(df)
        multilabel_pct = build_multilabel_percent(multilabel_counts)

        # Long format para graficas/reportes
        long_primary = (
            primary_counts.melt(
                id_vars=[PRIMARY_CLASS_COL],
                value_vars=["IC50", "EC50", "Total"],
                var_name="Activity",
                value_name="Count",
            )
            .sort_values(["Activity", "Count"], ascending=[True, False])
            .reset_index(drop=True)
        )

        # Guardar
        primary_counts.to_csv(
            os.path.join(OUTPUT_DIR, "chebi_counts_by_activity_primary.csv"), index=False
        )
        primary_pct.to_csv(
            os.path.join(OUTPUT_DIR, "chebi_counts_percent_by_activity_primary.csv"), index=False
        )
        multilabel_counts.to_csv(
            os.path.join(OUTPUT_DIR, "chebi_counts_by_activity_multilabel.csv"), index=False
        )
        multilabel_pct.to_csv(
            os.path.join(OUTPUT_DIR, "chebi_counts_percent_by_activity_multilabel.csv"), index=False
        )
        long_primary.to_csv(
            os.path.join(OUTPUT_DIR, "chebi_counts_long_primary.csv"), index=False
        )

        log("✓ Guardados archivos de conteo en ensayo332/", log_fh)
        log("=" * 72, log_fh)
        log("Proceso completado.", log_fh)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR en ensayo332: {e}")
        sys.exit(1)
