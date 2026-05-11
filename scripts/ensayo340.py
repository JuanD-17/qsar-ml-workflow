"""
Ensayo 340 - Heatmap de comparaciones por pares (solo Dilution 50).
"""

# -----------------------------
# IMPORTS Y CONFIGURACION BASE
# -----------------------------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator

# -----------------------------
# RUTAS DE ENTRADA / SALIDA
# -----------------------------
# Carpeta donde se guardan todas las figuras finales de este ensayo.
OUTPUT_DIR = "/Users/juandavidhoyostrejos/Documents/Icesi/qsar/ensayo340"
# Se crea la carpeta si no existe (evita errores al guardar).
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV de entrada con estimates y p-values de ABTS/DPPH por comparacion y dilucion.
INPUT_CSV = "/Users/juandavidhoyostrejos/Documents/Icesi/qsar/pairwise_antioxidant_comparisons.csv"

# -----------------------------
# ESTILO GLOBAL DE LA FIGURA
# -----------------------------
# Se define la tipografia y tamanos para mantener consistencia visual con el resto del articulo.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.9,
        # Se mantiene salida raster en 1200 DPI para publicacion.
        "figure.dpi": 1200,
        "savefig.dpi": 1200,
        "savefig.bbox": "tight",
    }
)

# Colores de texto base usados en titulos y anotaciones generales.
COLORS = {"text": "#2D3142"}


# -----------------------------
# FUNCIONES AUXILIARES DE DATOS
# -----------------------------
def _clean_numeric(series: pd.Series) -> pd.Series:
    # Convierte texto numerico a float, incluyendo casos con coma decimal (ej. "1,23").
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _translate_context_to_english(text: str) -> str:
    # Traduce etiquetas de contexto de espanol a ingles para ejes/publicacion.
    mapping = {
        "hoja": "leaf",
        "flor": "flower",
        "fruto": "fruit",
        "oscura": "dark",
        "citrica": "citrus",
    }
    t = str(text).strip()
    return mapping.get(t.lower(), t)


def _format_taxon_block(block: str) -> str:
    # Formatea un bloque tipo "Cannabis_sativa_(oscura)" a "C. sativa (dark)".
    b = str(block).strip()
    core = b
    suffix = ""

    # Separa parte taxonomica y parte entre parentesis, si existe.
    if "_(" in b:
        core, suffix = b.split("_(", 1)
        suffix = suffix.rstrip(")")

    # Divide genero/especie por guiones bajos.
    parts = [p for p in core.split("_") if p]

    # Si hay genero y especie, abrevia genero (ej. Cannabis -> C.).
    if len(parts) >= 2:
        short = f"{parts[0][0]}. {parts[1]}"
    else:
        # Fallback defensivo si viene en otro formato.
        short = core.replace("_", " ")

    # Traduce el sufijo contextual a ingles cuando aplique.
    suffix_en = _translate_context_to_english(suffix) if suffix else ""

    # Reconstruye texto final con sufijo (tejido/tipo) si existe.
    return f"{short} ({suffix_en})" if suffix_en else short


def _format_comparison_label(comp: str) -> str:
    # Convierte una comparacion completa "A-B" a "A vs B" para legibilidad.
    c = str(comp).strip()

    # Si no hay separador, limpia guiones bajos y retorna directo.
    if "-" not in c:
        return c.replace("_", " ").strip(" -")

    # Si hay separador, formatea ambos lados por separado.
    left, right = c.split("-", 1)
    return f"{_format_taxon_block(left)} vs {_format_taxon_block(right)}".strip(" -")


def _italicize_species_only(label: str) -> str:
    # Aplica cursiva solo al nombre cientifico (no al resto del texto contextual).
    out = str(label)
    out = out.replace("C. sativa", r"$\it{C.\ sativa}$")
    out = out.replace("H. lupulus", r"$\it{H.\ lupulus}$")
    out = out.replace("T. micrantha", r"$\it{T.\ micrantha}$")
    return out


def _p_to_stars(pval: float, has_data: bool) -> str:
    # Mapea p-value a etiqueta de significancia solicitada.
    # Regla de negocio:
    # - ns    si p >= 0.05
    # - *     si p < 0.05
    # - **    si p < 0.01
    # - ***   si p < 0.001
    if not has_data or pd.isna(pval):
        return ""
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def _text_color_for_cell(cmap, norm, value: float) -> str:
    # Ajusta color del texto segun contraste del fondo de la celda (luminancia).
    # Fondo oscuro -> texto blanco; fondo claro -> gris oscuro.
    r, g, b, _ = cmap(norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.52 else "#333333"


def load_data(path: str) -> pd.DataFrame:
    # Carga CSV como texto para controlar conversion numerica manualmente.
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")

    # Limpia nombres de columnas por si vienen con espacios/BOM.
    df.columns = [c.strip() for c in df.columns]

    # Verifica que esten las columnas minimas necesarias para construir el heatmap.
    req = ["dilution", "comparison", "ABTS_estimate", "ABTS_pvalue", "DPPH_estimate", "DPPH_pvalue"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    # Convierte columnas numericas a float/int de forma robusta.
    df["dilution"] = _clean_numeric(df["dilution"]).astype("Int64")
    df["ABTS_estimate"] = _clean_numeric(df["ABTS_estimate"])
    df["ABTS_pvalue"] = _clean_numeric(df["ABTS_pvalue"])
    df["DPPH_estimate"] = _clean_numeric(df["DPPH_estimate"])
    df["DPPH_pvalue"] = _clean_numeric(df["DPPH_pvalue"])

    # Normaliza texto de comparacion.
    df["comparison"] = df["comparison"].astype(str).str.strip()
    return df


def build_panel_matrices(df: pd.DataFrame, dilution_value: int):
    # Filtra SOLO la dilucion objetivo (en este ensayo: 50).
    panel_df = df[df["dilution"] == dilution_value].copy().reset_index(drop=True)

    # Crea etiquetas legibles para eje Y.
    panel_df["comparison_pretty"] = panel_df["comparison"].map(_format_comparison_label)
    panel_df["comparison_pretty_italic"] = panel_df["comparison_pretty"].map(_italicize_species_only)

    # Matriz 2 columnas [ABTS, DPPH] con los estimates.
    est = panel_df[["ABTS_estimate", "DPPH_estimate"]].to_numpy(dtype=float)

    # Matriz paralela con simbolos de significancia por celda.
    stars = np.empty(est.shape, dtype=object)
    abts_p = panel_df["ABTS_pvalue"].to_numpy(dtype=float)
    dpph_p = panel_df["DPPH_pvalue"].to_numpy(dtype=float)

    for i in range(len(panel_df)):
        stars[i, 0] = _p_to_stars(abts_p[i], np.isfinite(est[i, 0]))
        stars[i, 1] = _p_to_stars(dpph_p[i], np.isfinite(est[i, 1]))

    return panel_df, est, stars


# -----------------------------
# CONSTRUCCION DE LA FIGURA
# -----------------------------
def draw_figure(df: pd.DataFrame):
    # Genera matrices para la dilucion 50 unicamente.
    panel_df, est, stars = build_panel_matrices(df, 50)
    if len(panel_df) == 0:
        raise ValueError("Dilution 50 is not present in the CSV.")

    # Colormap divergente: azul (negativo) -> blanco (0) -> rojo (positivo).
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-30.0, vcenter=0.0, vmax=30.0)

    # Figura principal de un solo panel.
    fig, ax = plt.subplots(1, 1, figsize=(11, 10), facecolor="white")

    # Dibuja heatmap base sin anotaciones automaticas (las agregamos manualmente).
    sns.heatmap(
        est,
        ax=ax,
        cmap=cmap,
        norm=norm,
        cbar=False,
        linewidths=0.4,
        linecolor="#D9D9D9",
        xticklabels=["ABTS", "DPPH"],
        yticklabels=panel_df["comparison_pretty_italic"].tolist(),
        annot=False,
    )

    # Formato de ejes y titulo del panel.
    ax.set_title("10 mg/mL extract", pad=12, fontsize=15, fontweight="bold", color=COLORS["text"])
    ax.set_xlabel("Assay", fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0, length=0, labelsize=11)

    # Marco externo fino para apariencia limpia tipo publicacion.
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.75)
        ax.spines[side].set_color("black")

    # Colorbar unico del panel, con etiqueta Estimate.
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar_ax = fig.add_axes([0.935, 0.17, 0.018, 0.66])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Estimate", fontsize=12, fontstyle="normal", fontweight="bold")
    cbar.ax.yaxis.label.set_fontweight("bold")
    cbar.ax.tick_params(labelsize=11, width=0.8, length=3.5)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    cbar.ax.tick_params(which="minor", length=2.2, width=0.7)

    # Anotaciones por celda: solo simbolo de significancia (sin valores numericos).
    for i in range(est.shape[0]):
        for j in range(est.shape[1]):
            val = est[i, j]
            if np.isfinite(val):
                txt_color = _text_color_for_cell(cmap, norm, val)
                ax.text(
                    j + 0.50,
                    i + 0.50,
                    stars[i, j],
                    ha="center",
                    va="center",
                    fontsize=10.6,
                    color=txt_color,
                    fontweight="bold",
                )

    # Ajuste de margenes para que no se recorten etiquetas largas del eje Y.
    fig.subplots_adjust(left=0.32, right=0.88, top=0.92, bottom=0.08)
    return fig


# -----------------------------
# EXPORTACION DE RESULTADOS
# -----------------------------
def save_outputs(fig):
    # Se exporta a raster y vectorial para uso en manuscrito y edicion posterior.
    outputs = {
        "Panel_Heatmap_Dilution50.png": 1200,
        "Panel_Heatmap_Dilution50.tiff": 1200,
        "Panel_Heatmap_Dilution50.pdf": None,
        "Panel_Heatmap_Dilution50.svg": None,
    }

    for name, dpi in outputs.items():
        out_path = os.path.join(OUTPUT_DIR, name)
        if dpi is None:
            # PDF/SVG: salida vectorial.
            fig.savefig(out_path, facecolor="white")
        else:
            # PNG/TIFF: salida raster de alta resolucion.
            fig.savefig(out_path, facecolor="white", dpi=dpi)
        print(f"Saved: {out_path}")


# -----------------------------
# FLUJO PRINCIPAL
# -----------------------------
def main():
    # 1) Carga y limpieza de datos.
    df = load_data(INPUT_CSV)

    # 2) Construccion de la figura de Dilution 50.
    fig = draw_figure(df)

    # 3) Exportacion en formatos finales.
    save_outputs(fig)

    # 4) Cierre explicito para liberar memoria.
    plt.close(fig)
    print(f"\nDone. Output directory: {OUTPUT_DIR}")


# Punto de entrada del script.
if __name__ == "__main__":
    main()
