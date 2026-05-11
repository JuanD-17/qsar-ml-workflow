"""
Ensayo 320 - Pipeline maestro Panel A + Panel B (Q1)

Flujo:
1) Genera el panel A desde el script fuente de activity cliffs globales.
2) Genera el panel B desde el script fuente de representative activity cliff pairs.
3) Combina A (arriba) + B (abajo) en una sola figura final.
4) Guarda todo en ensayo320 a 1200 DPI.
"""

import base64
import importlib.util
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

# Evita limitaciones de PIL para imágenes grandes al combinar paneles a alta resolución
Image.MAX_IMAGE_PIXELS = None

# ============================================================================
# CONFIGURACION
# ============================================================================
OUTPUT_DIR = "ensayo320"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PANEL_A_SCRIPT_CANDIDATES = [
    "ensayo317.py",
    "ensayo304.py",
    "ensayo303.py",
]
PANEL_B_SCRIPT_CANDIDATES = [
    "ensayo300.py",
    "ensayo299.py",
]
PANEL_C_SCRIPT_CANDIDATES = [
    "ensayo319.py",
]

PANEL_A_BASE = "Panel_B_Activity_Cliffs_IC50_EC50_Combined"
PANEL_A_IC50_BASE = "Panel_A_Activity_Cliffs_IC50"
PANEL_A_EC50_BASE = "Panel_B_Activity_Cliffs_EC50"
PANEL_B_BASE = "Figure_Representative_Cliff_Pairs"
OUTPUT_BASE = os.path.join(OUTPUT_DIR, "Panel_AB_Q1_Combined")
WORD_OUTPUT = os.path.join(OUTPUT_DIR, "Panels_A_C.docx")
STRUCTURE_ACTIVITY_TIF = "Structure_activity_cliff.tif"
CLIFF_RECORTES_DIR = os.path.join(OUTPUT_DIR, "recortes_structure_activity_cliff")
os.makedirs(CLIFF_RECORTES_DIR, exist_ok=True)

INTER_PANEL_GAP = 18
OUTER_MARGIN = 10
WHITE_THRESHOLD = 248

# Control de tamaño final para mantener legibilidad y evitar errores por archivos gigantes
TARGET_WIDTH_PX = 16000
RASTER_DPI = 1200
FINAL_SCALE_FACTOR = 1.16

# Recortes manuales dentro de Structure_activity_cliff.tif para las cuatro
# estructuras del panel representativo (dos IC50 arriba y dos EC50 abajo).
CLIFF_MANUAL_CROPS = {
    "ic50_left": (0, 0, 990, 300),
    "ic50_right": (20, 470, 910, 790),
    "ec50_left": (0, 800, 590, 1223),
    "ec50_right": (880, 800, 1476, 1223),
}

CLIFF_DISPLAY_NAMES = {
    ("IC50", "left"): "2-(1,2-dihydroxyethyl)-4-(dodecyloxy)-5-hydroxyfuran-3(2H)-one",
    ("IC50", "right"): "2-(1,2-dihydroxyethyl)-5-hydroxy-4-(nonyloxy)furan-3(2H)-one",
    ("EC50", "left"): "2-(1-((1,3-dioxoisoindolin-2-yl)methyl)-1,2,3,4-tetrahydroisoquinoline-2-carbonyl)cyclohexane-1-carboxylic acid",
    ("EC50", "right"): "(1S,2R)-2-((R)-1-((1,3-dioxoisoindolin-2-yl)methyl)-1,2,3,4-tetrahydroisoquinoline-2-carbonyl)cyclohexane-1-carboxylic acid",
}

CLIFF_BOX_EDGE = "#8fa3b8"
CLIFF_BOX_FACE = "white"
CLIFF_BOX_LINEWIDTH = 1.05
CLIFF_TITLE_BLUE = "#2f5f8a"
CLIFF_TITLE_GREEN = "#3e7a66"
CLIFF_FONT_FAMILY = "DejaVu Sans"


# ============================================================================
# UTILIDADES
# ============================================================================
def resolve_existing_path(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No se encontro ninguno de estos scripts: {candidates}")


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el modulo: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trim_white_borders(img: Image.Image, threshold: int = WHITE_THRESHOLD) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    bg = Image.new("RGB", img.size, (threshold, threshold, threshold))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    return img.crop(bbox) if bbox else img


def resize_to_width(img: Image.Image, target_width: int) -> Image.Image:
    if img.width == target_width:
        return img
    new_h = int(img.height * (target_width / img.width))
    return img.resize((target_width, new_h), Image.Resampling.LANCZOS)


def save_multi_format(canvas: Image.Image, output_base: str):
    out_png = f"{output_base}.png"
    canvas.save(out_png, format="PNG", dpi=(RASTER_DPI, RASTER_DPI), optimize=True)
    print(f"✓ Guardado: {os.path.basename(out_png)}")


def wrap_iupac_name(name: str, max_chars: int = 34) -> str:
    """Envuelve nombres IUPAC respetando el texto original y priorizando guiones existentes."""
    if len(name) <= max_chars:
        return name

    parts = name.split("-")
    if len(parts) == 1:
        return textwrap.fill(name, width=max_chars)

    lines = []
    current = parts[0]
    for part in parts[1:]:
        candidate = f"{current}-{part}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = part
    if current:
        lines.append(current)
    return "\n".join(lines)

    out_tiff = f"{output_base}.tiff"
    canvas.save(out_tiff, format="TIFF", dpi=(RASTER_DPI, RASTER_DPI), compression="tiff_lzw")
    print(f"✓ Guardado: {os.path.basename(out_tiff)}")

    out_pdf = f"{output_base}.pdf"
    try:
        canvas.save(out_pdf, format="PDF", resolution=RASTER_DPI)
        print(f"✓ Guardado: {os.path.basename(out_pdf)}")
    except Exception as exc:
        print(f"⚠️ No se pudo guardar {os.path.basename(out_pdf)}: {exc}")

    out_svg = f"{output_base}.svg"
    with open(out_png, "rb") as f:
        b64_png = base64.b64encode(f.read()).decode("ascii")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas.width}" height="{canvas.height}" '
        f'viewBox="0 0 {canvas.width} {canvas.height}">'
        f'<image href="data:image/png;base64,{b64_png}" width="{canvas.width}" height="{canvas.height}"/></svg>'
    )
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"✓ Guardado: {os.path.basename(out_svg)}")


def add_centered_caption(doc: Document, text: str, size: int = 11):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = True
    run.font.name = "Arial"
    run.font.size = Pt(size)


def add_centered_image(doc: Document, image_path: str, width_inches: float = 6.9):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(image_path, width=Inches(width_inches))


def export_panels_to_word():
    scaffold_candidates = [
        os.path.join(OUTPUT_DIR, "Panel_A_Top5_Scaffolds_IC50_EC50_Combined.png"),
        os.path.join("ensayo317", "Panel_A_Top5_Scaffolds_IC50_EC50_Combined.png"),
    ]
    panel_scaffolds = next((path for path in scaffold_candidates if os.path.exists(path)), None)
    panel_cliff_pairs = os.path.join(OUTPUT_DIR, f"{PANEL_B_BASE}.png")

    if panel_scaffolds is None:
        raise FileNotFoundError(
            "No existe el panel de scaffolds para Word en ensayo320 ni en ensayo317."
        )
    if not os.path.exists(panel_cliff_pairs):
        raise FileNotFoundError(f"No existe el panel representativo para Word: {panel_cliff_pairs}")

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Arial"
    normal_style.font.size = Pt(10)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run("Panels A and C for Manuscript Preparation")
    title_run.bold = True
    title_run.font.name = "Arial"
    title_run.font.size = Pt(14)

    doc.add_paragraph()
    add_centered_caption(doc, "Panel A. Top 5 Bemis-Murcko Scaffolds (IC50 vs EC50)")
    add_centered_image(doc, panel_scaffolds, width_inches=6.9)

    doc.add_paragraph()
    add_centered_caption(doc, "Panel C. Representative Activity Cliff Pairs (Highest Delta Activity)")
    add_centered_image(doc, panel_cliff_pairs, width_inches=6.9)

    doc.save(WORD_OUTPUT)
    print(f"✓ Guardado: {os.path.basename(WORD_OUTPUT)}")


# ============================================================================
# GENERACION DE PANELES DESDE CODIGO FUENTE
# ============================================================================
def generate_panel_a_into_ensayo320():
    panel_b_script = resolve_existing_path(PANEL_B_SCRIPT_CANDIDATES)
    mod = load_module(panel_b_script, "panel_b_src")
    mod.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(mod.OUTPUT_DIR, exist_ok=True)

    ic50_cliffs = mod.load_activity_cliffs_data(mod.IC50_INPUT_DIR, "IC50")
    ic50_pairs = mod.load_all_pairs_data(mod.IC50_INPUT_DIR, seed=42)
    ec50_cliffs = mod.load_activity_cliffs_data(mod.EC50_INPUT_DIR, "EC50")
    ec50_pairs = mod.load_all_pairs_data(mod.EC50_INPUT_DIR, seed=43)
    y_max = max(
        float(ic50_pairs["Activity_Difference"].max()) if len(ic50_pairs) else 0.0,
        float(ec50_pairs["Activity_Difference"].max()) if len(ec50_pairs) else 0.0,
        float(ic50_cliffs["Activity_Difference"].max()) if len(ic50_cliffs) else 0.0,
        float(ec50_cliffs["Activity_Difference"].max()) if len(ec50_cliffs) else 0.0,
    )
    y_max = max(mod.ACTIVITY_THRESHOLD + 0.5, y_max * 1.08)

    def _draw_uniform(ax, cliffs_df, all_pairs_df, assay_label, show_ylabel):
        if len(all_pairs_df) > 120000:
            sample_indices = np.random.choice(len(all_pairs_df), 120000, replace=False)
            plot_df = all_pairs_df.iloc[sample_indices]
        else:
            plot_df = all_pairs_df

        ax.hexbin(
            plot_df["Tanimoto_Similarity"],
            plot_df["Activity_Difference"],
            gridsize=85,
            extent=(0.0, 1.02, 0.0, y_max),
            mincnt=1,
            cmap="Greys",
            linewidths=0,
            alpha=0.28,
            bins="log",
            zorder=0,
        )

        ax.fill_between(
            [mod.SIMILARITY_THRESHOLD, 1.0],
            mod.ACTIVITY_THRESHOLD,
            y_max,
            alpha=0.08,
            color="#D99A9A",
            zorder=1,
        )
        ax.axhline(
            y=mod.ACTIVITY_THRESHOLD,
            color="#A65D4D",
            linestyle=(0, (4, 2)),
            linewidth=1.35,
            alpha=0.82,
            zorder=2,
        )
        ax.axvline(
            x=mod.SIMILARITY_THRESHOLD,
            color="#3F7DA3",
            linestyle=(0, (4, 2)),
            linewidth=1.35,
            alpha=0.82,
            zorder=2,
        )

        if len(cliffs_df) > 0:
            ax.scatter(
                cliffs_df["Tanimoto_Similarity"],
                cliffs_df["Activity_Difference"],
                c="#9E2F25",
                marker="X",
                s=155,
                linewidths=1.0,
                edgecolors="#6F1C15",
                zorder=5,
                alpha=0.96,
            )

        stats_text = (
            f"n = {len(cliffs_df)} cliffs\nmax Δ = {cliffs_df['Activity_Difference'].max():.2f}"
        ) if len(cliffs_df) else "n = 0 cliffs"
        ax.text(
            0.035,
            0.955,
            stats_text,
            transform=ax.transAxes,
            fontsize=15.4,
            verticalalignment="top",
            color=mod.COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.34",
                facecolor="white",
                edgecolor="#D3D9E1",
                alpha=0.96,
                linewidth=0.8,
            ),
            family="DejaVu Sans",
            fontweight="normal",
            linespacing=1.15,
        )

        assay_display = "IC₅₀" if assay_label == "IC50" else "EC₅₀"
        ax.set_title(assay_display, fontweight="bold", fontsize=19.6, pad=10, fontfamily="DejaVu Sans")
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(-0.1, y_max)
        ax.grid(True, axis="x", alpha=0.16, linestyle=(0, (1.5, 2.8)), color="#D7DDE5", linewidth=0.7)
        ax.grid(False, axis="y")
        ax.set_xlabel("")
        ax.set_ylabel("|Δlog₁₀(nM)|" if show_ylabel else "", fontweight="bold", fontsize=17.6, fontfamily="DejaVu Sans")
        ax.tick_params(axis="both", labelsize=14.7, width=1.0, colors=mod.COLORS["text"])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("DejaVu Sans")
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.spines["left"].set_color(mod.COLORS["text"])
        ax.spines["bottom"].set_color(mod.COLORS["text"])

    fig, axes = plt.subplots(1, 2, figsize=(18.2, 8.0), facecolor="white")
    _draw_uniform(axes[0], ic50_cliffs, ic50_pairs, "IC50", True)
    _draw_uniform(axes[1], ec50_cliffs, ec50_pairs, "EC50", False)

    common_legend = [
        mod.mpatches.Patch(color="#C7CDD6", alpha=0.45, label="All molecular pairs"),
        mod.mpatches.Patch(color="#D99A9A", alpha=0.12, label=f"Cliff zone (Δ ≥ {mod.ACTIVITY_THRESHOLD})"),
        mod.Line2D([0], [0], color="#A65D4D", linestyle=(0, (4, 2)), linewidth=1.35, label=f"Activity threshold (Δ ≥ {mod.ACTIVITY_THRESHOLD})"),
        mod.Line2D([0], [0], color="#3F7DA3", linestyle=(0, (4, 2)), linewidth=1.35, label=f"Similarity threshold (≥ {mod.SIMILARITY_THRESHOLD})"),
    ]
    fig.legend(
        handles=common_legend,
        ncol=4,
        bbox_to_anchor=(0.5, 0.982),
        loc="upper center",
        frameon=False,
        fontsize=15.2,
        fancybox=False,
        handlelength=1.5,
        columnspacing=1.6,
        handletextpad=0.6,
        borderpad=0.2,
    )
    axes[0].text(-0.085, 1.015, "A", transform=axes[0].transAxes, fontsize=21.6, fontweight="bold",
                 fontfamily="DejaVu Sans", color=mod.COLORS["text"], va="top", ha="left")
    axes[1].text(-0.072, 1.015, "B", transform=axes[1].transAxes, fontsize=21.6, fontweight="bold",
                 fontfamily="DejaVu Sans", color=mod.COLORS["text"], va="top", ha="left")
    fig.supxlabel("Tanimoto Similarity", fontsize=17.6, fontweight="bold", y=0.035, fontfamily="DejaVu Sans")
    fig.subplots_adjust(left=0.085, right=0.94, top=0.875, bottom=0.11, wspace=0.15)
    base_name = os.path.join(OUTPUT_DIR, PANEL_A_BASE)
    mod._save_multi_format(fig, base_name)
    plt.close(fig)

    # Exporta tambien cada endpoint por separado, para usarlos como panel A y panel B
    # individuales sin afectar el combinado principal.
    single_specs = [
        ("A", "IC50", ic50_cliffs, ic50_pairs, True, PANEL_A_IC50_BASE),
        ("B", "EC50", ec50_cliffs, ec50_pairs, True, PANEL_A_EC50_BASE),
    ]
    for panel_letter, assay_label, cliffs_df, pairs_df, show_ylabel, base_stub in single_specs:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(10.8, 9.0), facecolor="white")
        _draw_uniform(ax_single, cliffs_df, pairs_df, assay_label, show_ylabel)
        fig_single.text(
            0.02, 0.972, panel_letter,
            fontsize=28, fontweight="bold", fontfamily="DejaVu Sans",
            color=mod.COLORS["text"], va="top", ha="left"
        )
        fig_single.supxlabel("Tanimoto Similarity", fontsize=12.5, fontweight="bold", y=0.045, fontfamily="DejaVu Sans")
        fig_single.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.10)
        mod._save_multi_format(fig_single, os.path.join(OUTPUT_DIR, base_stub))
        plt.close(fig_single)


def trim_and_pad_crop(img: Image.Image, threshold: int = 245, padding: int = 10) -> Image.Image:
    gray = img.convert("L")
    arr = np.array(gray)
    mask = arr < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img.convert("RGBA")
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    trimmed = img.crop((x0, y0, x1, y1)).convert("RGBA")
    return ImageOps.expand(trimmed, border=padding, fill="white")


def extract_structure_activity_cliff_crops():
    source = Image.open(STRUCTURE_ACTIVITY_TIF).convert("RGBA")
    crops = {}
    for key, (x0, y0, x1, y1) in CLIFF_MANUAL_CROPS.items():
        crop = source.crop((x0, y0, x1, y1))
        clean = trim_and_pad_crop(crop, threshold=245, padding=18)
        out_path = os.path.join(CLIFF_RECORTES_DIR, f"{key}.png")
        clean.save(out_path)
        crops[key] = clean
    return crops


def add_pair_row_from_tif(fig, gs, row_idx, pair_row, assay_label, assay_color, left_img, right_img, mod):
    ax_left = fig.add_subplot(gs[row_idx, 0])
    ax_mid = fig.add_subplot(gs[row_idx, 1])
    ax_right = fig.add_subplot(gs[row_idx, 2])

    if pair_row is None:
        for ax in (ax_left, ax_right):
            ax.axis("off")
            ax.text(0.5, 0.5, f"{assay_label}\nNo activity cliffs", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color=mod.COLORS["text"],
                    bbox=dict(boxstyle="round,pad=0.42", facecolor="white", edgecolor=assay_color, linewidth=1.6))
        ax_mid.axis("off")
        ax_mid.text(0.5, 0.5, f"{assay_label}\n\nNo representative cliff pair\nunder current thresholds.",
                    ha="center", va="center", fontsize=13, color=mod.COLORS["text"],
                    bbox=dict(boxstyle="round,pad=0.60", facecolor="white", edgecolor=assay_color, linewidth=1.6))
        return

    left_title = wrap_iupac_name(CLIFF_DISPLAY_NAMES[(assay_label, "left")], max_chars=32)
    right_title = wrap_iupac_name(CLIFF_DISPLAY_NAMES[(assay_label, "right")], max_chars=32)

    ax_left.imshow(np.array(left_img))
    ax_left.axis("off")
    if row_idx == 0:
        ax_left.set_anchor("N")
    ax_left.set_title(
        left_title,
        color=assay_color,
        fontsize=19.4,
        fontweight="bold",
        fontfamily=CLIFF_FONT_FAMILY,
        pad=4,
        y=0.985,
    )

    ax_right.imshow(np.array(right_img))
    ax_right.axis("off")
    if row_idx == 0:
        ax_right.set_anchor("N")
    ax_right.set_title(
        right_title,
        color=assay_color,
        fontsize=17.6,
        fontweight="bold",
        fontfamily=CLIFF_FONT_FAMILY,
        pad=4,
        y=0.985,
    )

    activity_y = -0.118 if row_idx == 0 else -0.148
    for ax, activity in [(ax_left, pair_row['Activity_1']), (ax_right, pair_row['Activity_2'])]:
        ax.text(
            0.5,
            activity_y,
            f"Activity\n{activity:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom" if row_idx == 0 else "top",
            fontsize=17.6,
            color=mod.COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.34",
                facecolor=CLIFF_BOX_FACE,
                edgecolor=CLIFF_BOX_EDGE,
                linewidth=CLIFF_BOX_LINEWIDTH,
            ),
            clip_on=False,
        )

    info = (
        f"Tanimoto similarity: {pair_row['Tanimoto_Similarity']:.3f}\n"
        f"Delta log activity: {pair_row['Activity_Difference']:.2f}\n"
        f"Fold change: {pair_row['Fold_Change']:.0f}x"
    )
    ax_mid.axis("off")
    info_y = 0.60 if row_idx == 0 else 0.48
    ax_mid.text(
        0.5,
        info_y,
        info,
        ha="center",
        va="center",
        fontsize=19.8,
        color=mod.COLORS["text"],
        bbox=dict(
            boxstyle="round,pad=0.58",
            facecolor=CLIFF_BOX_FACE,
            edgecolor=CLIFF_BOX_EDGE,
            linewidth=CLIFF_BOX_LINEWIDTH,
        ),
    )


def generate_panel_b_into_ensayo320():
    panel_c_script = resolve_existing_path(PANEL_C_SCRIPT_CANDIDATES)
    mod = load_module(panel_c_script, "panel_c_src")
    mod.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(mod.OUTPUT_DIR, exist_ok=True)

    crops = extract_structure_activity_cliff_crops()
    df_ic50 = mod.load_cliffs(mod.IC50_CLIFFS_FILE)
    df_ec50 = mod.load_cliffs(mod.EC50_CLIFFS_FILE)
    rep_ic50 = mod.select_representative_pair(df_ic50)
    rep_ec50 = mod.select_representative_pair(df_ec50)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })

    fig = plt.figure(figsize=(22.4, 14.6), facecolor="white")
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.72, 0.82, 1.72],
        hspace=0.34,
        wspace=0.12,
        left=0.028,
        right=0.972,
        top=0.90,
        bottom=0.095,
    )

    add_pair_row_from_tif(fig, gs, 0, rep_ic50, "IC50", CLIFF_TITLE_BLUE, crops["ic50_left"], crops["ic50_right"], mod)
    add_pair_row_from_tif(fig, gs, 1, rep_ec50, "EC50", CLIFF_TITLE_GREEN, crops["ec50_left"], crops["ec50_right"], mod)

    fig.text(0.022, 0.925, "A", ha="left", va="center", fontsize=26.9, fontweight="bold", color=mod.COLORS["text"], fontfamily=CLIFF_FONT_FAMILY)
    fig.text(0.5, 0.94, "IC50", ha="center", va="center", fontsize=26.9, fontweight="bold", color=mod.COLORS["text"], fontfamily=CLIFF_FONT_FAMILY)
    fig.text(0.022, 0.492, "B", ha="left", va="center", fontsize=26.9, fontweight="bold", color=mod.COLORS["text"], fontfamily=CLIFF_FONT_FAMILY)
    fig.text(0.5, 0.505, "EC50", ha="center", va="center", fontsize=26.9, fontweight="bold", color=mod.COLORS["text"], fontfamily=CLIFF_FONT_FAMILY)

    base = os.path.join(OUTPUT_DIR, PANEL_B_BASE)
    for ext, dpi in mod.EXPORT_FORMATS.items():
        path = base + ext
        try:
            if dpi is None:
                fig.savefig(path, facecolor="white", format="svg")
            else:
                fig.savefig(path, facecolor="white", dpi=dpi)
            print(f"✓ Guardado: {os.path.basename(path)}")
        except Exception as exc:
            print(f"⚠️ No se pudo guardar {os.path.basename(path)}: {exc}")
    plt.close(fig)


def combine_a_b_from_ensayo320():
    a_path = os.path.join(OUTPUT_DIR, f"{PANEL_A_BASE}.png")
    b_path = os.path.join(OUTPUT_DIR, f"{PANEL_B_BASE}.png")

    if not os.path.exists(a_path):
        raise FileNotFoundError(f"No existe Panel A generado: {a_path}")
    if not os.path.exists(b_path):
        raise FileNotFoundError(f"No existe Panel B generado: {b_path}")

    img_a = trim_white_borders(Image.open(a_path).convert("RGB"))
    img_b = trim_white_borders(Image.open(b_path).convert("RGB"))

    max_width = max(img_a.width, img_b.width)
    target_width = min(max_width, TARGET_WIDTH_PX)

    img_a = resize_to_width(img_a, target_width)
    img_b = resize_to_width(img_b, target_width)

    if FINAL_SCALE_FACTOR != 1.0:
        scaled_width = int(target_width * FINAL_SCALE_FACTOR)
        img_a = resize_to_width(img_a, scaled_width)
        img_b = resize_to_width(img_b, scaled_width)
        target_width = scaled_width

    canvas_w = target_width + (OUTER_MARGIN * 2)
    canvas_h = img_a.height + img_b.height + INTER_PANEL_GAP + (OUTER_MARGIN * 2)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color="white")

    y = OUTER_MARGIN
    panel_a_y = y
    canvas.paste(img_a, (OUTER_MARGIN, y))
    y += img_a.height + INTER_PANEL_GAP
    panel_b_y = y
    canvas.paste(img_b, (OUTER_MARGIN, y))

    # Mantiene los paneles tal como salen de sus figuras fuente;\n    # no agrega letras extra en la etapa de combinacion para evitar duplicados.\n\n    save_multi_format(canvas, OUTPUT_BASE)


def main():
    print("=" * 70)
    print("ENSAYO 320 - PANEL A + B DESDE CODIGO FUENTE")
    print("=" * 70)

    print("\n[1/3] Generando Panel A en ensayo320...")
    generate_panel_a_into_ensayo320()

    print("\n[2/3] Generando Panel B en ensayo320...")
    generate_panel_b_into_ensayo320()

    print("\n[3/3] Combinando A + B...")
    combine_a_b_from_ensayo320()

    print("\n[4/4] Exportando Panel A y Panel C a Word...")
    try:
        export_panels_to_word()
    except Exception as exc:
        print(f"⚠️ No se pudo exportar Word: {exc}")

    print("\n✅ Completado")
    print(f"Resultados en: {OUTPUT_DIR}/")
    print(f"- {PANEL_A_BASE}.png/.pdf/.tiff/.svg")
    print(f"- {PANEL_B_BASE}.png/.pdf/.tiff/.svg")
    print("- Panel_AB_Q1_Combined.png/.pdf/.tiff/.svg")
    print("- Panels_A_C.docx")


if __name__ == "__main__":
    main()
