# -*- coding: utf-8 -*-
# Este script reconstruye el panel de los Top 5 Bemis-Murcko scaffolds usando
# exclusivamente la imagen fuente scaffold.tif. La estrategia no depende de
# deteccion automatica fragmentada: usa recortes manuales estables sobre la
# imagen fuente, ajusta cada estructura para que no se salga de su recuadro,
# conserva la informacion estadistica de los scaffold_summary.csv y exporta la
# figura final en formatos listos para manuscrito.

from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ============================
# Configuracion general
# ============================
OUTPUT_DIR = Path("/Users/juandavidhoyostrejos/Documents/Icesi/qsar/ensayo317")
TIFF_SOURCE = Path("/Users/juandavidhoyostrejos/Documents/Icesi/qsar/scaffold.tif")
IC50_SUMMARY_PATH = Path("/Users/juandavidhoyostrejos/Documents/Icesi/qsar/ensayo291/scaffold_summary.csv")
EC50_SUMMARY_PATH = Path("/Users/juandavidhoyostrejos/Documents/Icesi/qsar/ensayo301/scaffold_summary.csv")
RECORTES_DIR = OUTPUT_DIR / "recortes"
RASTER_DPI = 1200

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 14,
    "figure.dpi": RASTER_DPI,
    "savefig.dpi": RASTER_DPI,
    "savefig.bbox": "tight",
})

# ============================
# Coordenadas manuales de recorte en scaffold.tif
# ============================
# Los recortes se definieron sobre la imagen fuente completa para evitar que
# las estructuras queden fragmentadas. Luego cada recorte se autoajusta al
# bounding box real y se le agrega padding blanco.
MANUAL_CROPS = {
    "benzene": (0, 10, 220, 240),
    "flavone": (250, 0, 700, 255),
    "chromane": (700, 10, 960, 240),
    "butenolide": (950, 10, 1161, 240),
    "purine_nucleoside_like_scaffold": (0, 240, 370, 529),
    "benzimidazole_linked_phenyl": (360, 250, 860, 529),
}

DISPLAY_NAMES = {
    "benzene": "Benzene",
    "flavone": "2-phenyl-4H-chromen-4-one",
    "chromane": "Chromane",
    "butenolide": "furan-2(5H)-one",
    "purine_nucleoside_like_scaffold": "(S)-9-(tetrahydrofuran-2-yl)-9H-purine",
    "benzimidazole_linked_phenyl": "2-phenyl-1H-benzo[d]imidazole",
}

# Orden final solicitado por el usuario para el panel.
IC50_ORDER = [
    "benzene",
    "flavone",
    "butenolide",
    "chromane",
    "purine_nucleoside_like_scaffold",
]
EC50_ORDER = [
    "benzene",
    "benzimidazole_linked_phenyl",
    "flavone",
    "butenolide",
    "chromane",
]

# ============================
# Utilidades de directorios
# ============================
def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RECORTES_DIR.mkdir(parents=True, exist_ok=True)


# ============================
# Carga de estadisticas de scaffold
# ============================
def load_top_scaffolds(summary_path: Path, expected_order: list, endpoint_total: int) -> dict:
    df = pd.read_csv(summary_path).copy()
    total_count = endpoint_total
    df = df.sort_values("molecule_count", ascending=False).head(len(expected_order)).reset_index(drop=True)
    df["frequency_pct"] = df["molecule_count"] / total_count * 100
    df["mean_activity"] = df["mean_activity"].apply(lambda v: float(str(v).replace("'", "")))
    rows = {}
    for idx, key in enumerate(expected_order):
        row = df.iloc[idx].copy()
        row["count"] = int(row["molecule_count"])
        rows[key] = row
    return rows


# ============================
# Recorte limpio desde scaffold.tif
# ============================
def trim_and_pad(crop: Image.Image, threshold: int = 245, padding: int = 24) -> Image.Image:
    gray = crop.convert("L")
    arr = np.array(gray)
    mask = arr < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return crop.convert("RGBA")
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    trimmed = crop.crop((x0, y0, x1, y1)).convert("RGBA")
    return ImageOps.expand(trimmed, border=padding, fill="white")


def extract_manual_crops(source_path: Path) -> dict:
    source = Image.open(source_path).convert("RGBA")
    crops = {}
    for key, (x0, y0, x1, y1) in MANUAL_CROPS.items():
        raw_crop = source.crop((x0, y0, x1, y1))
        clean_crop = trim_and_pad(raw_crop, threshold=245, padding=28)
        crops[key] = clean_crop
        clean_crop.save(RECORTES_DIR / f"{key}.png")
    return crops


# ============================
# Construccion del panel
# ============================
def format_scaffold_name(title: str) -> str:
    if "\n" in title:
        return title
    if title == "2-phenyl-1H-benzo[d]imidazole":
        return title
    if title == "(S)-9-(tetrahydrofuran-2-yl)-9H-purine":
        return "(S)-9-(tetrahydrofuran-2-yl)-\n9H-purine"
    if len(title) <= 18:
        return title
    return fill(title, width=22, break_long_words=False, break_on_hyphens=True)


def place_scaffold_card(
    ax,
    structure_img: Image.Image,
    slot_label: str,
    title: str,
    stats_row,
    border_color: str = "#E1E5EC",
    structure_scale: float = 1.0,
    title_font_boost: float = 0.0,
    stats_font_boost: float = 0.0,
    slot_font_boost: float = 0.0,
    box_x: float = 0.03,
    box_y: float = 0.04,
    box_w: float = 0.94,
    box_h: float = 0.90,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box = FancyBboxPatch(
        (box_x, box_y), box_w, box_h,
        boxstyle="round,pad=0.006,rounding_size=0.016",
        linewidth=1.05,
        edgecolor=border_color,
        facecolor="#FFFFFF",
    )
    ax.add_patch(box)

    bubble_color = "#444444"
    ax.text(
        0.11, 0.86, slot_label,
        ha="center", va="center",
        fontsize=9.2 + slot_font_boost, fontweight="bold", color="white",
        bbox=dict(boxstyle="circle,pad=0.22", fc=bubble_color, ec=bubble_color, lw=0.7),
        zorder=5,
    )

    arr = np.array(structure_img)
    h, w = arr.shape[:2]
    target_w, target_h = 0.74 * structure_scale, 0.47 * structure_scale
    scale = min(target_w / w, target_h / h)
    draw_w = w * scale
    draw_h = h * scale
    center_x = 0.50
    center_y = 0.68
    extent = [center_x - draw_w / 2, center_x + draw_w / 2, center_y - draw_h / 2, center_y + draw_h / 2]
    ax.imshow(arr, extent=extent, interpolation="lanczos", zorder=3)

    title_wrapped = format_scaffold_name(title)
    stats_text = (
        f"n = {int(stats_row['count']):,} ({stats_row['frequency_pct']:.1f}%)\n"
        f"μ = {stats_row['mean_activity']:.2f}"
    )

    title_fontsize = 11.5 if len(title_wrapped.replace("\n", " ")) <= 28 else 10.2
    title_y = 0.180
    stats_y = 0.088
    if title == "(S)-9-(tetrahydrofuran-2-yl)-9H-purine":
        title_fontsize = 8.9
        title_y = 0.198
        stats_y = 0.074
    title_fontsize += title_font_boost
    if title == "2-phenyl-1H-benzo[d]imidazole":
        ax.annotate(
            "2-phenyl-1H-benzo[d]imidazole",
            xy=(0.50, 0.182),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=9.6 + title_font_boost,
            fontweight="semibold",
            color="#374151",
            annotation_clip=False,
            clip_on=True,
            zorder=6,
        )
    else:
        ax.text(0.50, title_y, title_wrapped, ha="center", va="center", fontsize=title_fontsize, fontweight="semibold", color="#374151", wrap=False, clip_on=False)
    ax.text(0.50, stats_y, stats_text, ha="center", va="center", fontsize=10.8 + stats_font_boost, color="#4B5563", linespacing=1.05)


def build_panel(ic50_stats: dict, ec50_stats: dict, crops: dict) -> None:
    fig = plt.figure(figsize=(19.0, 10.9), facecolor="white")
    gs = fig.add_gridspec(
        2, 5,
        height_ratios=[1.0, 1.0],
        hspace=0.16,
        wspace=0.16,
        left=0.03,
        right=0.985,
        top=0.93,
        bottom=0.065,
    )

    fig.text(0.5, 0.958, "IC50", ha="center", va="center", fontsize=17.8, fontweight="bold", color="#000000")
    fig.text(0.5, 0.497, "EC50", ha="center", va="center", fontsize=17.8, fontweight="bold", color="#000000")

    fig.text(0.012, 0.848, "A", fontsize=23.0, fontweight="bold", color="#374151")
    fig.text(0.012, 0.386, "B", fontsize=23.0, fontweight="bold", color="#374151")

    for idx, key in enumerate(IC50_ORDER):
        ax_card = fig.add_subplot(gs[0, idx])
        place_scaffold_card(
            ax_card,
            crops[key],
            f"S{idx+1}",
            DISPLAY_NAMES[key],
            ic50_stats[key],
            title_font_boost=5.2,
            stats_font_boost=4.0,
            slot_font_boost=2.2,
            box_x=0.012,
            box_w=0.976,
            box_y=0.032,
            box_h=0.91,
        )

    for idx, key in enumerate(EC50_ORDER):
        ax_card = fig.add_subplot(gs[1, idx])
        place_scaffold_card(
            ax_card,
            crops[key],
            f"S{idx+1}",
            DISPLAY_NAMES[key],
            ec50_stats[key],
            title_font_boost=5.2,
            stats_font_boost=4.0,
            slot_font_boost=2.2,
            box_x=0.012,
            box_w=0.976,
            box_y=0.032,
            box_h=0.91,
        )

    base = OUTPUT_DIR / "Panel_A_Top5_Scaffolds_IC50_EC50_Combined"
    outputs = [
        (base.with_suffix(".png"), {"facecolor": "white", "dpi": RASTER_DPI}),
        (base.with_suffix(".tiff"), {"facecolor": "white", "dpi": RASTER_DPI}),
        (base.with_suffix(".pdf"), {"facecolor": "white"}),
        (base.with_suffix(".svg"), {"facecolor": "white"}),
    ]
    for out_path, kwargs in outputs:
        try:
            fig.savefig(out_path, **kwargs)
        except Exception as exc:
            print(f"Advertencia: no se pudo guardar {out_path.name}: {exc}")
    plt.close(fig)

def build_panel_vertical(ic50_stats: dict, ec50_stats: dict, crops: dict) -> None:
    fig = plt.figure(figsize=(11.4, 16.2), facecolor="white")
    gs = fig.add_gridspec(
        5, 2,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0],
        hspace=0.11,
        wspace=0.08,
        left=0.045,
        right=0.99,
        top=0.93,
        bottom=0.04,
    )

    fig.suptitle("Top 5 Bemis-Murcko Scaffolds: IC50 vs. EC50", fontsize=16, fontweight="bold", y=0.985)

    ic50_axes = []
    for idx, key in enumerate(IC50_ORDER):
        ax_card = fig.add_subplot(gs[idx, 0])
        ic50_axes.append(ax_card)
        place_scaffold_card(
            ax_card,
            crops[key],
            f"S{idx+1}",
            DISPLAY_NAMES[key],
            ic50_stats[key],
            structure_scale=1.10,
            title_font_boost=0.1,
            stats_font_boost=0.6,
            box_x=0.017,
            box_y=0.03,
            box_w=0.968,
            box_h=0.925,
        )

    ec50_axes = []
    for idx, key in enumerate(EC50_ORDER):
        ax_card = fig.add_subplot(gs[idx, 1])
        ec50_axes.append(ax_card)
        place_scaffold_card(
            ax_card,
            crops[key],
            f"S{idx+1}",
            DISPLAY_NAMES[key],
            ec50_stats[key],
            structure_scale=1.10,
            title_font_boost=0.1,
            stats_font_boost=0.6,
            box_x=0.017,
            box_y=0.03,
            box_w=0.968,
            box_h=0.925,
        )

    ic50_pos = ic50_axes[0].get_position()
    ec50_pos = ec50_axes[0].get_position()
    fig.text((ic50_pos.x0 + ic50_pos.x1) / 2, 0.938, "A. IC50", ha="center", va="center", fontsize=14, fontweight="bold", color="#000000")
    fig.text((ec50_pos.x0 + ec50_pos.x1) / 2, 0.938, "B. EC50", ha="center", va="center", fontsize=14, fontweight="bold", color="#000000")

    base = OUTPUT_DIR / "Panel_A_Top5_Scaffolds_IC50_EC50_Vertical"
    outputs = [
        (base.with_suffix(".png"), {"facecolor": "white", "dpi": RASTER_DPI}),
        (base.with_suffix(".pdf"), {"facecolor": "white"}),
        (base.with_suffix(".svg"), {"facecolor": "white"}),
    ]
    for out_path, kwargs in outputs:
        try:
            fig.savefig(out_path, **kwargs)
        except Exception as exc:
            print(f"Advertencia: no se pudo guardar {out_path.name}: {exc}")
    plt.close(fig)


# ============================
# Ejecucion principal
# ============================
def main() -> None:
    ensure_dirs()
    ic50_stats = load_top_scaffolds(IC50_SUMMARY_PATH, IC50_ORDER, endpoint_total=2726)
    ec50_stats = load_top_scaffolds(EC50_SUMMARY_PATH, EC50_ORDER, endpoint_total=447)
    crops = extract_manual_crops(TIFF_SOURCE)
    build_panel(ic50_stats, ec50_stats, crops)
    build_panel_vertical(ic50_stats, ec50_stats, crops)
    print("Panel de scaffolds regenerado desde scaffold.tif")
    print(f"Recortes guardados en: {RECORTES_DIR}")


if __name__ == "__main__":
    main()
