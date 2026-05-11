# Ensayo 314
# Este script genera figuras comparativas de Williams plots para evaluar
# el dominio de aplicabilidad de predicciones QSAR para IC50 y EC50.
# Flujo general:
# 1. Carga los datasets de entrenamiento IC50 y EC50 y calcula descriptores.
# 2. Carga las predicciones consolidadas para cannabinoides y terpenos.
# 3. Calcula leverage (h*) y residuals estandarizados para cada endpoint.
# 4. Dibuja una figura 2x2 con Williams plots (IC50 + EC50).
# 5. Dibuja una figura separada solo para EC50 (paneles C y D).
# 6. Exporta ambas figuras en PNG, TIFF, PDF y SVG.

import logging
import os
from typing import Dict, Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.lines import Line2D

try:
    from adjustText import adjust_text  # pyright: ignore[reportMissingImports]
except Exception:
    adjust_text = None
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
RDLogger.DisableLog("rdApp.*")
np.random.seed(42)

OUTPUT_DIR = "/Users/juandavidhoyostrejos/Documents/Icesi/qsar/ensayo314"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAINING_DATA_PATHS = {
    "IC50": "/Users/juandavidhoyostrejos/Documents/Icesi/qsar/antioxidant11.csv",
    "EC50": "/Users/juandavidhoyostrejos/Documents/Icesi/qsar/EC50_dataset.csv",
}

PREDICTION_PATHS = {
    "A": {
        "endpoint": "IC50",
        "title": r"Cannabinoids – Applicability Domain ($\mathbf{IC_{50}}$)",
        "tag": "A",
        "path": "/Users/juandavidhoyostrejos/Documents/Icesi/qsar/ensayo334/Cannabaceae_consolidated_predictions.csv",
    },
    "B": {
        "endpoint": "IC50",
        "title": r"Terpenes – Applicability Domain ($\mathbf{IC_{50}}$)",
        "tag": "B",
        "path": "/Users/juandavidhoyostrejos/Documents/Icesi/qsar/ensayo330/Cannabaceae_consolidated_predictions.csv",
    },
    "C": {
        "endpoint": "EC50",
        "title": r"Cannabinoids – Applicability Domain ($\mathbf{EC_{50}}$)",
        "tag": "C",
        "path": "/Users/juandavidhoyostrejos/Documents/Icesi/qsar/Cannabinoides_consolidated_predictions(EC50).csv",
    },
    "D": {
        "endpoint": "EC50",
        "title": r"Terpenes – Applicability Domain ($\mathbf{EC_{50}}$)",
        "tag": "D",
        "path": "/Users/juandavidhoyostrejos/Documents/Icesi/qsar/Terpenes(EC50).csv",
    },
}

COLORS = {
    # Softer contextual background for the training set so predicted molecules remain primary.
    "train": "#BCC6D6",
    "train_edge": "#9BA8BA",
    # Journal-style red for predicted compounds with a darker edge for crisp rendering.
    "pred": "#C23B32",
    "pred_edge": "#8F231D",
    "text": "#28303F",
    "grid": "#D7DDE5",
    "threshold": "#2F7EA1",
    "residual": "#7A7A7A",
    "label_text": "#7F1D1D",
    "label_edge": "#B84A4A",
    "stats_edge": "#CCD3DD",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.4,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

MAX_LABELS_HIGH = 12
MAX_LABELS_LOW = 18
LABEL_FONT_SIZE = 11.3
PANEL_TAG_FONTSIZE = 16.5
PANEL_TITLE_FONTSIZE = 13.5
AXIS_LABEL_FONTSIZE = 12.0
TICK_FONTSIZE = 10.5
STATS_FONTSIZE = 8.5
LEGEND_FONTSIZE = 10.5
TRAINING_ALPHA = 0.08
TRAINING_SIZE = 8
PREDICTED_SIZE = 46
PREDICTED_EDGEWIDTH = 0.75
ANNOTATION_LINEWIDTH = 0.78
ANNOTATION_COLOR = "#8B0000"
OFFSET_X_DENSE = 0.022
OFFSET_X_SPARSE = 0.026
OFFSET_Y = 0.34
HSTAR_FONTSIZE = 12.0
AXIS_SPINE_WIDTH = 1.3
TICK_WIDTH = 1.2
TICK_LENGTH = 4.0
SIGMA_LINEWIDTH = 1.0
HSTAR_LINEWIDTH = 1.2
LEGEND_TRAIN_MARKERSIZE = 5.6
LEGEND_PRED_MARKERSIZE = 9.2

# Visual x-axis limits are truncated only for display so extreme leverage values do not
# compress the main cloud of points. All points remain in the calculations.
DISPLAY_X_LIMITS = {
    "IC50": None,
    "EC50": None,
}

# Prioritize a few recognizable compounds when labels are shown so the plot remains readable.
KEY_LABELS = {
    "Cannabinoids": ["CBD", "CBN", "CBC", "CBGA", "CBG", "CBDV", "THC", "THCA", "THCV"],
    "Terpenes": [
        "Geraniol",
        "Linalool",
        "beta-Mirceno",
        "d-Limoneno",
        "Ocimeno 1",
        "alfa-Pineno",
        "beta-pineno",
        "beta-Cariopilene",
        "Nerolidol 1",
        "(-)-Oxido de Cariofileno",
        "terpinene",
    ],
}

PANEL_LABEL_SELECTIONS = {
    "A": ["CBC", "CBGA", "CBG", "CBN", "CBDV", "CBD", "THCV", "THC", "THCA"],
    "B": ["p-cymene", "Geraniol", "Linalool", "beta-Mirceno", "Nerolidol 1", "Ocimeno 1", "d-Limoneno", "alfa-Pineno", "beta-pineno", "beta-Cariopilene", "(-)-Oxido de Cariofileno"],
    "C": ["CBG", "CBGA", "CBD", "CBDV", "CBC", "CBN", "THCV", "THC", "THCA"],
    "D": ["terpinolene", "d-Limoneno", "Geraniol", "Linalool", "alfa-Pineno", "Nerolidol 1", "beta-Mirceno", "(-)-Oxido de Cariofileno", "Ocimeno 1", "terpinene"],
}

MANUAL_LABEL_OFFSETS = {
    "A": {"THCV": (22, 12)},
    "B": {
        "p-cymene": (-12, 24),
        "Geraniol": (-10, 18),
        "Linalool": (-8, 12),
        "beta-Mirceno": (-2, 6),
        "Nerolidol 1": (8, -8),
        "Ocimeno 1": (12, -14),
        "d-Limoneno": (18, 22),
        "alfa-Pineno": (24, 14),
        "beta-pineno": (30, 10),
        "beta-Cariopilene": (34, 16),
        "(-)-Oxido de Cariofileno": (38, 10),
    },
    "D": {
        "terpinolene": (-18, 20),
        "d-Limoneno": (-10, 16),
        "Geraniol": (4, 8),
        "Linalool": (10, 12),
        "alfa-Pineno": (16, 14),
        "Nerolidol 1": (24, 12),
        "beta-Mirceno": (34, 10),
        "(-)-Oxido de Cariofileno": (10, -18),
        "Ocimeno 1": (16, -24),
        "terpinene": (20, -30),
    },
}

FULL_LABEL_MAP = {
    "alfa-Pineno": "alpha-Pinene",
    "beta-pineno": "beta-Pinene",
    "beta-Mirceno": "beta-Myrcene",
    "beta-Cariopilene": "beta-Caryophyllene",
    "(-)-Oxido de Cariofileno": "(-)-Oxide of Caryophyllene",
    "p-cymene": "p-Cymene",
    "d-Limoneno": "d-Limonene",
    "Ocimeno 1": "Ocimene 1",
    "terpinolene": "Terpinolene",
    "terpinene": "Terpinene",
    "Linalool": "Linalool",
    "Geraniol": "Geraniol",
    "Nerolidol 1": "Nerolidol 1",
    "Nerolidol 2": "Nerolidol 2",
    "(-)-Guaiol": "(-)-Guaiol",
    "(-)-alfa-Bisabolol": "(-)-alpha-Bisabolol",
    "alfa-Humeleno": "alpha-Humulene",
    "camphene": "Camphene",
    "Δ3-carene": "Δ3-Carene",
    "Eucaliptol": "Eucalyptol",
    "isopulegol": "Isopulegol",
}

CAPTION_NOTE = "For EC50, leverage was computed in PCA score space to account for the different descriptor dimensionality."


def validate_smiles(smiles_list):
    valid_smiles, valid_idx = [], []
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol and mol.GetNumAtoms() > 0:
                valid_smiles.append(smiles)
                valid_idx.append(i)
        except Exception:
            continue
    return valid_smiles, valid_idx


def display_label(name: str, panel_family: str) -> str:
    text = str(name).strip()
    if panel_family == "Terpenes":
        return FULL_LABEL_MAP.get(text, text)
    return text


def get_panel_selected_indices(df_predictions_valid, visible_indices, panel_tag: str) -> np.ndarray:
    selected_names = PANEL_LABEL_SELECTIONS.get(panel_tag)
    if not selected_names:
        return np.array([], dtype=int)
    selected_set = set(selected_names)
    ordered = []
    for idx in visible_indices:
        mol_name = str(df_predictions_valid.iloc[idx].get("Molecule_name", ""))
        if mol_name in selected_set:
            ordered.append(int(idx))
    return np.array(ordered, dtype=int)


def calculate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        descriptors = {
            "MolLogP": Crippen.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "MolWt": Descriptors.MolWt(mol),
            "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "NumHDonors": Lipinski.NumHDonors(mol),
            "NumHAcceptors": Lipinski.NumHAcceptors(mol),
            "FractionCSP3": Descriptors.FractionCSP3(mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        }

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=512)
        for i in range(512):
            descriptors[f"Morgan_{i}"] = float(fp[i])

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_fp = (
            AllChem.GetMorganFingerprintAsBitVect(scaffold, 3, nBits=512)
            if scaffold.GetNumAtoms() > 0
            else [0] * 512
        )
        descriptors["Scaffold_MolWt"] = Descriptors.MolWt(scaffold) if scaffold.GetNumAtoms() > 0 else 0
        for i in range(512):
            descriptors[f"Scaffold_Morgan_{i}"] = float(scaffold_fp[i])

        return descriptors
    except Exception:
        return None


def get_descriptors_for_df(df, smiles_col="SMILES"):
    descriptor_list = [calculate_descriptors(s) for s in df[smiles_col]]
    valid_indices = [i for i, d in enumerate(descriptor_list) if d is not None]
    if not valid_indices:
        return pd.DataFrame(), []
    desc_df = pd.DataFrame([descriptor_list[i] for i in valid_indices])
    return desc_df, valid_indices


def load_training_dataset(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    logging.info("Loading training dataset: %s", path)
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip().split(","))

    header = lines[0]
    header[0] = header[0].lstrip("\ufeff")
    data = [line for line in lines[1:] if len(line) == len(header)]
    df_train_raw = pd.DataFrame(data, columns=header)
    df_train_raw.rename(columns={"Log10 Value (nM)": "Activity", "Smiles": "SMILES"}, inplace=True)
    df_train_raw["Activity"] = pd.to_numeric(df_train_raw["Activity"], errors="coerce")
    df_train_raw.dropna(subset=["SMILES", "Activity"], inplace=True)

    _, valid_idx_train = validate_smiles(df_train_raw["SMILES"])
    df_train = df_train_raw.iloc[valid_idx_train].reset_index(drop=True)
    X_train_df, train_indices = get_descriptors_for_df(df_train)
    y_train = df_train.iloc[train_indices]["Activity"].reset_index(drop=True).astype(float)
    X_train_df = X_train_df.reset_index(drop=True)
    logging.info("Training set loaded: %s molecules", len(X_train_df))
    return X_train_df, y_train


def load_predictions(path: str, label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction file not found for {label}: {path}")

    df_pred = pd.read_csv(path, sep=";")
    df_pred = df_pred.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    X_pred_df, valid_idx = get_descriptors_for_df(df_pred, smiles_col="SMILES")
    df_pred_valid = df_pred.iloc[valid_idx].reset_index(drop=True)
    X_pred_df = X_pred_df.reset_index(drop=True)
    logging.info("%s loaded: %s molecules", label, len(df_pred_valid))
    return X_pred_df, df_pred_valid


def compute_williams_metrics(X_train_scaled, y_train, X_pred_scaled, endpoint):
    residuals_threshold = 3
    if endpoint == "EC50":
        n_components_pca = min(197, X_train_scaled.shape[1], X_train_scaled.shape[0] - 1)
        pca_ad = PCA(n_components=n_components_pca, random_state=42)
        T_train = pca_ad.fit_transform(X_train_scaled)
        T_pred = pca_ad.transform(X_pred_scaled)
        n_tr = T_train.shape[0]
        TtT_inv = np.linalg.pinv(T_train.T @ T_train)
        leverage_train = (1.0 / n_tr) + np.einsum("ij,jk,ik->i", T_train, TtT_inv, T_train)
        leverage_pred = (1.0 / n_tr) + np.einsum("ij,jk,ik->i", T_pred, TtT_inv, T_pred)
        leverage_threshold = 3 * n_components_pca / n_tr
        temp_rf = RandomForestRegressor(random_state=42).fit(T_train, y_train)
        residuals = y_train - temp_rf.predict(T_train)
        residuals_std = (residuals - residuals.mean()) / residuals.std()
        logging.info("EC50 PCA-AD: %s components, h*=%.4f", n_components_pca, leverage_threshold)
        return leverage_train, leverage_pred, residuals_std, leverage_threshold, residuals_threshold

    X_train_intercept = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
    try:
        hat_matrix_inv = np.linalg.inv(X_train_intercept.T @ X_train_intercept)
    except np.linalg.LinAlgError:
        hat_matrix_inv = np.linalg.pinv(X_train_intercept.T @ X_train_intercept)

    leverage_train = np.einsum("ij,ji->i", X_train_intercept, hat_matrix_inv @ X_train_intercept.T)
    X_pred_intercept = np.c_[np.ones(X_pred_scaled.shape[0]), X_pred_scaled]
    leverage_pred = np.einsum("ij,ji->i", X_pred_intercept, hat_matrix_inv @ X_pred_intercept.T)
    temp_rf = RandomForestRegressor(random_state=42).fit(X_train_scaled, y_train)
    residuals = y_train - temp_rf.predict(X_train_scaled)
    residuals_std = (residuals - residuals.mean()) / residuals.std()
    p, n = X_train_scaled.shape[1] + 1, len(y_train)
    leverage_threshold = 3 * p / n
    return leverage_train, leverage_pred, residuals_std, leverage_threshold, residuals_threshold


def create_williams_plot(ax, X_train_scaled, y_train, X_pred_scaled, df_predictions_valid, panel_tag, title_suffix, endpoint, show_legend=True, show_molecule_labels=True):
    """Draw a single Williams plot panel with a publication-oriented visual hierarchy."""
    leverage_train, leverage_pred, residuals_std, leverage_threshold, residuals_threshold = compute_williams_metrics(
        X_train_scaled, y_train, X_pred_scaled, endpoint
    )

    x_display_max = DISPLAY_X_LIMITS.get(endpoint)
    if x_display_max is None:
        finite_values = np.concatenate([
            leverage_train[np.isfinite(leverage_train)],
            leverage_pred[np.isfinite(leverage_pred)],
        ])
        x_display_max = max(0.82, np.percentile(finite_values, 99.5)) if len(finite_values) else 0.82
    x_display_max = max(x_display_max, leverage_threshold * 1.08 + 0.02)

    # Contextual training cloud: soft, small, and transparent so it supports rather than dominates.
    ax.scatter(
        leverage_train,
        residuals_std,
        c=COLORS["train"],
        alpha=TRAINING_ALPHA,
        s=TRAINING_SIZE,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )

    # Predicted molecules remain the primary signal in the figure.
    ax.scatter(
        leverage_pred,
        np.zeros_like(leverage_pred),
        c=COLORS["pred"],
        alpha=0.98,
        s=PREDICTED_SIZE,
        marker="D",
        edgecolors=COLORS["pred_edge"],
        linewidths=PREDICTED_EDGEWIDTH,
        zorder=4,
    )

    max_opt_y = 0.0
    label_annotations = []
    if show_molecule_labels:
        label_pool = np.arange(len(leverage_pred))
        visible_mask = leverage_pred <= x_display_max
        visible_indices = label_pool[visible_mask]

        panel_family = "Cannabinoids" if "Cannabinoids" in title_suffix else "Terpenes"
        selected_indices = get_panel_selected_indices(df_predictions_valid, visible_indices, panel_tag)
        selected_name_set = set(PANEL_LABEL_SELECTIONS.get(panel_tag, []))
        preferred_indices: List[int] = [int(idx) for idx in selected_indices]

        max_labels_for_panel = len(PANEL_LABEL_SELECTIONS.get(panel_tag, [])) if PANEL_LABEL_SELECTIONS.get(panel_tag) else (11 if panel_family == "Terpenes" else 9)
        if panel_family == "Terpenes":
            if len(visible_indices) and not len(selected_indices):
                lowest_visible_idx = int(visible_indices[np.argmin(leverage_pred[visible_indices])])
                if lowest_visible_idx not in preferred_indices:
                    preferred_indices.append(lowest_visible_idx)
            preferred_sorted = np.array(preferred_indices, dtype=int)[np.argsort(leverage_pred[np.array(preferred_indices, dtype=int)])] if preferred_indices else np.array([], dtype=int)
            sorted_visible = visible_indices[np.argsort(leverage_pred[visible_indices])] if len(visible_indices) else np.array([], dtype=int)
        else:
            preferred_sorted = np.array(preferred_indices, dtype=int)[np.argsort(leverage_pred[np.array(preferred_indices, dtype=int)])[::-1]] if preferred_indices else np.array([], dtype=int)
            sorted_visible = visible_indices[np.argsort(leverage_pred[visible_indices])[::-1]] if len(visible_indices) else np.array([], dtype=int)

        priority_indices: List[int] = [int(idx) for idx in preferred_sorted]
        if not selected_name_set:
            for idx in sorted_visible:
                if int(idx) not in priority_indices:
                    priority_indices.append(int(idx))
                if len(priority_indices) >= max_labels_for_panel:
                    break

        all_indices = np.array(priority_indices[:max_labels_for_panel], dtype=int)

        if panel_family == "Terpenes":
            terp_sorted = all_indices[np.argsort(leverage_pred[all_indices])] if len(all_indices) else np.array([], dtype=int)
            left_count = int(np.ceil(len(terp_sorted) * 0.58))
            left_indices = terp_sorted[:left_count]
            right_indices = terp_sorted[left_count:]

            left_upper = np.array([2.85, 2.35, 1.85, 1.35])
            left_lower = np.array([-1.35, -1.95, -2.55])
            right_upper = np.array([2.70, 2.20, 1.70, 1.20])
            right_lower = np.array([-1.25, -1.85, -2.45])

            terp_placements = []
            for order, idx in enumerate(left_indices):
                lane_pool = left_upper if order < len(left_upper) else left_lower
                lane_y = float(lane_pool[order] if order < len(left_upper) else lane_pool[min(order - len(left_upper), len(left_lower) - 1)])
                x_anchor = float(leverage_pred[idx])
                x_text = max(0.05, x_anchor - (0.085 + 0.018 * min(order, 4)))
                terp_placements.append((idx, x_text, lane_y, display_label(df_predictions_valid.iloc[idx].get("Molecule_name", f"Pred_{idx}"), panel_family), "right"))

            for order, idx in enumerate(right_indices):
                lane_pool = right_upper if order < len(right_upper) else right_lower
                lane_y = float(lane_pool[order] if order < len(right_upper) else lane_pool[min(order - len(right_upper), len(right_lower) - 1)])
                x_anchor = float(leverage_pred[idx])
                x_text = min(x_display_max - 0.03, x_anchor + (0.070 + 0.024 * min(order, 4)))
                terp_placements.append((idx, x_text, lane_y, display_label(df_predictions_valid.iloc[idx].get("Molecule_name", f"Pred_{idx}"), panel_family), "left"))

            for i, x_text, lane_y, full_label, ha in terp_placements:
                raw_name = str(df_predictions_valid.iloc[i].get("Molecule_name", f"Pred_{i}"))
                dx, dy = MANUAL_LABEL_OFFSETS.get(panel_tag, {}).get(raw_name, (0, 0))
                x_text = x_text + dx / 72.0
                lane_y = lane_y + dy / 72.0
                rad = -0.07 if ha == "right" else 0.07
                ann = ax.annotate(
                    full_label,
                    xy=(leverage_pred[i], 0),
                    xytext=(x_text, lane_y),
                    fontsize=LABEL_FONT_SIZE,
                    fontweight="semibold",
                    color=ANNOTATION_COLOR,
                    ha=ha,
                    va="center",
                    arrowprops=dict(
                        arrowstyle="-",
                        color=ANNOTATION_COLOR,
                        lw=0.72,
                        alpha=0.68,
                        shrinkA=2,
                        shrinkB=4,
                        connectionstyle=f"arc3,rad={rad:.2f}",
                    ),
                    bbox=dict(
                        boxstyle="round,pad=0.22",
                        facecolor="white",
                        edgecolor=ANNOTATION_COLOR,
                        alpha=0.94,
                        linewidth=0.82,
                    ),
                    zorder=5,
                )
                label_annotations.append(ann)
                max_opt_y = max(max_opt_y, abs(lane_y))
        else:
            compact_y_offsets = [2.20, 1.75, 1.30, 0.88, -1.08, -1.52, -1.96, 2.55, -2.30]
            compact_x_offsets = [0.030, 0.024, 0.020, 0.018, 0.024, 0.020, 0.026, 0.034, 0.034]
            compact_align = ["left", "left", "left", "left", "left", "left", "left", "left", "left"]

            cannabinoid_sorted = all_indices[np.argsort(leverage_pred[all_indices])[::-1]] if len(all_indices) else np.array([], dtype=int)
            compact_placements = []
            for order, idx in enumerate(cannabinoid_sorted):
                y_off = compact_y_offsets[min(order, len(compact_y_offsets) - 1)]
                x_off = compact_x_offsets[min(order, len(compact_x_offsets) - 1)]
                x_anchor = float(leverage_pred[idx])
                x_text = min(x_display_max - 0.03, x_anchor + x_off)
                label = display_label(str(df_predictions_valid.iloc[idx].get("Molecule_name", f"Pred_{idx}")), panel_family)
                compact_placements.append((idx, x_text, y_off, label, compact_align[min(order, len(compact_align) - 1)]))

            for i, x_text, lane_y, short_label, ha in compact_placements:
                raw_name = str(df_predictions_valid.iloc[i].get("Molecule_name", f"Pred_{i}"))
                if raw_name == "THCV" and panel_tag == "A":
                    ann = ax.annotate(
                        short_label,
                        xy=(leverage_pred[i], 0),
                        xytext=(22, 12),
                        textcoords="offset points",
                        fontsize=LABEL_FONT_SIZE,
                        fontweight="semibold",
                        color=ANNOTATION_COLOR,
                        ha="left",
                        va="center",
                        arrowprops=dict(
                            arrowstyle="-",
                            color=ANNOTATION_COLOR,
                            lw=0.66,
                            alpha=0.66,
                            shrinkA=2,
                            shrinkB=4,
                            connectionstyle="arc3,rad=0.02",
                        ),
                        bbox=dict(
                            boxstyle="round,pad=0.22",
                            facecolor="white",
                            edgecolor=ANNOTATION_COLOR,
                            alpha=0.94,
                            linewidth=0.82,
                        ),
                        zorder=5,
                    )
                    max_opt_y = max(max_opt_y, 0.25)
                    continue
                ann = ax.annotate(
                    short_label,
                    xy=(leverage_pred[i], 0),
                    xytext=(x_text, lane_y),
                    fontsize=LABEL_FONT_SIZE,
                    fontweight="semibold",
                    color=ANNOTATION_COLOR,
                    ha=ha,
                    va="center",
                    arrowprops=dict(
                        arrowstyle="-",
                        color=ANNOTATION_COLOR,
                        lw=0.66,
                        alpha=0.66,
                        shrinkA=2,
                        shrinkB=4,
                        connectionstyle="arc3,rad=0.02",
                    ),
                    bbox=dict(
                        boxstyle="round,pad=0.22",
                        facecolor="white",
                        edgecolor=ANNOTATION_COLOR,
                        alpha=0.94,
                        linewidth=0.82,
                    ),
                    zorder=5,
                )
                label_annotations.append(ann)
                max_opt_y = max(max_opt_y, abs(lane_y))

    if show_molecule_labels and adjust_text is not None and label_annotations:
        try:
            adjust_text(
                label_annotations,
                ax=ax,
                only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"} if panel_family == "Terpenes" else {"text": "y", "static": "xy", "explode": "y", "pull": "y"},
                force_text=(0.18, 0.26) if panel_family == "Terpenes" else (0.09, 0.16),
                force_static=(0.08, 0.08),
                expand=(1.18, 1.36) if panel_family == "Terpenes" else (1.04, 1.12),
                ensure_inside_axes=True,
                time_lim=0.80 if panel_family == "Terpenes" else 0.45,
                iter_lim=280 if panel_family == "Terpenes" else 160,
            )
        except Exception:
            pass

    # Reference lines are visible but restrained, with consistent styling across all panels.
    ax.axhline(y=residuals_threshold, color=COLORS["residual"], linestyle=(0, (4, 2)), linewidth=SIGMA_LINEWIDTH, alpha=0.58, zorder=2)
    ax.axhline(y=-residuals_threshold, color=COLORS["residual"], linestyle=(0, (4, 2)), linewidth=SIGMA_LINEWIDTH, alpha=0.58, zorder=2)
    ax.axvline(x=leverage_threshold, color=COLORS["threshold"], linestyle=(0, (5, 2)), linewidth=HSTAR_LINEWIDTH, alpha=0.80, zorder=2)

    hstar_x = min(leverage_threshold + 0.015, x_display_max - 0.12)
    ax.text(
        hstar_x,
        0.92,
        f"h* = {leverage_threshold:.3f}",
        transform=ax.get_xaxis_transform(),
        fontsize=HSTAR_FONTSIZE,
        fontweight="semibold",
        color=COLORS["text"],
        bbox=dict(boxstyle="round,pad=0.36", facecolor="white", edgecolor=COLORS["threshold"], linewidth=HSTAR_LINEWIDTH, alpha=0.98),
        zorder=6,
    )

    pred_in_ad = int(np.sum(leverage_pred <= leverage_threshold))
    pred_total = int(len(leverage_pred))
    pred_in_ad_pct = (pred_in_ad / pred_total * 100.0) if pred_total else 0.0
    train_in_residual = int(np.sum(np.abs(residuals_std) <= residuals_threshold))
    train_total = int(len(residuals_std))
    train_in_residual_pct = (train_in_residual / train_total * 100.0) if train_total else 0.0
    panel_x = 0.02
    ax.text(
        panel_x,
        0.975,
        panel_tag,
        transform=ax.transAxes,
        fontsize=PANEL_TAG_FONTSIZE,
        fontweight="bold",
        color=COLORS["text"],
        ha="left",
        va="top",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title_suffix, fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold", pad=16)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_SPINE_WIDTH)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, width=TICK_WIDTH, length=TICK_LENGTH)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=TICK_WIDTH, length=TICK_LENGTH)

    if show_legend:
        legend_handles = [
            Line2D([], [], marker="o", linestyle="None", markersize=LEGEND_TRAIN_MARKERSIZE, markerfacecolor=COLORS["train"], markeredgecolor="none", alpha=TRAINING_ALPHA, label="Training set"),
            Line2D([], [], marker="D", linestyle="None", markersize=LEGEND_PRED_MARKERSIZE, markerfacecolor=COLORS["pred"], markeredgecolor=COLORS["pred_edge"], markeredgewidth=PREDICTED_EDGEWIDTH, label="Predicted molecules"),
            Line2D([], [], color=COLORS["threshold"], linestyle=(0, (5, 2)), linewidth=HSTAR_LINEWIDTH, label="h* threshold"),
            Line2D([], [], color=COLORS["residual"], linestyle=(0, (4, 2)), linewidth=SIGMA_LINEWIDTH, label="±3σ limits"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            bbox_to_anchor=(0.988, 0.08),
            framealpha=0.98,
            edgecolor=COLORS["stats_edge"],
            facecolor="white",
            fancybox=False,
            fontsize=LEGEND_FONTSIZE,
            borderpad=0.48,
            handlelength=1.95,
            handletextpad=0.62,
            labelspacing=0.36,
        )

    ax.grid(True, axis="x", alpha=0.22, linestyle=(0, (1.2, 2.4)), linewidth=0.75, color=COLORS["grid"])
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.set_xlim(left=-0.01, right=x_display_max)

    residuals_min = residuals_std.min()
    residuals_max = residuals_std.max()
    ylim_bottom = min(-residuals_threshold - 0.55, residuals_min - 0.5)
    ylim_top = max(max(residuals_threshold, residuals_max) + 0.55, max_opt_y + 0.95)
    ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

def prepare_scaled_matrices(endpoint: str, panel_key: str, training_cache, prediction_cache):
    X_train_df, y_train = training_cache[endpoint]
    X_pred_df, df_pred_valid = prediction_cache[panel_key]
    common_cols = sorted(set(X_train_df.columns) & set(X_pred_df.columns))
    X_train = X_train_df[common_cols]
    X_pred = X_pred_df[common_cols]
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), y_train, scaler.transform(X_pred), df_pred_valid


def save_figure(fig, base_name: str, dpi: int = 1200):
    png_out = os.path.join(OUTPUT_DIR, f"{base_name}.png")
    try:
        fig.savefig(png_out, dpi=dpi, facecolor="white")
        logging.info("✓ Saved: %s", png_out)
    except Exception as exc:
        logging.warning("Could not save %s: %s", png_out, exc)
        png_out = None

    tiff_out = os.path.join(OUTPUT_DIR, f"{base_name}.tiff")
    if png_out and os.path.exists(png_out):
        try:
            with Image.open(png_out) as im:
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                im.save(tiff_out, compression="tiff_lzw", dpi=(dpi, dpi))
            logging.info("✓ Saved: %s", tiff_out)
        except Exception as exc:
            logging.warning("Could not save %s: %s", tiff_out, exc)

    for ext in ("pdf", "svg"):
        out = os.path.join(OUTPUT_DIR, f"{base_name}.{ext}")
        try:
            fig.savefig(out, facecolor="white")
            logging.info("✓ Saved: %s", out)
        except Exception as exc:
            logging.warning("Could not save %s: %s", out, exc)


def save_final_jcim_outputs(fig, base_name: str, dpi: int = 1200):
    """Save the final JCIM-oriented figure in PDF and TIFF, rebuilding TIFF from a temporary PNG."""
    png_out = os.path.join(OUTPUT_DIR, f"{base_name}__temp.png")
    pdf_out = os.path.join(OUTPUT_DIR, f"{base_name}.pdf")
    tiff_out = os.path.join(OUTPUT_DIR, f"{base_name}.tiff")
    try:
        fig.savefig(png_out, dpi=dpi, facecolor="white", bbox_inches="tight")
        logging.info("✓ Saved temporary PNG: %s", png_out)
    except Exception as exc:
        logging.warning("Could not save temporary PNG %s: %s", png_out, exc)
        png_out = None

    if png_out and os.path.exists(png_out):
        try:
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(png_out) as im:
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                im.save(tiff_out, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
            logging.info("✓ Saved: %s", tiff_out)
        except Exception as exc:
            logging.warning("Could not save %s: %s", tiff_out, exc)
        finally:
            try:
                os.remove(png_out)
            except OSError:
                pass

    try:
        fig.savefig(pdf_out, facecolor="white", bbox_inches="tight")
        logging.info("✓ Saved: %s", pdf_out)
    except Exception as exc:
        logging.warning("Could not save %s: %s", pdf_out, exc)


def main():
    global PANEL_TAG_FONTSIZE, PANEL_TITLE_FONTSIZE, AXIS_LABEL_FONTSIZE, TICK_FONTSIZE
    global LEGEND_FONTSIZE, HSTAR_FONTSIZE, PREDICTED_SIZE, TRAINING_ALPHA, TRAINING_SIZE
    training_cache: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    for endpoint, path in TRAINING_DATA_PATHS.items():
        training_cache[endpoint] = load_training_dataset(path)

    prediction_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for panel_key, panel_meta in PREDICTION_PATHS.items():
        prediction_cache[panel_key] = load_predictions(panel_meta["path"], panel_meta["title"])

    logging.info("Generating combined Williams Plot figure (IC50 + EC50)...")
    fig_all, axes_all = plt.subplots(nrows=2, ncols=2, figsize=(18.6, 16.4), dpi=300, facecolor="white")
    axes_all = axes_all.flatten()
    for ax, panel_key in zip(axes_all, ["A", "B", "C", "D"]):
        endpoint = PREDICTION_PATHS[panel_key]["endpoint"]
        X_train_scaled, y_train, X_pred_scaled, df_pred_valid = prepare_scaled_matrices(endpoint, panel_key, training_cache, prediction_cache)
        create_williams_plot(ax, X_train_scaled, y_train, X_pred_scaled, df_pred_valid, PREDICTION_PATHS[panel_key]["tag"], PREDICTION_PATHS[panel_key]["title"], endpoint)
    row1_bottom = min(axes_all[0].get_ylim()[0], axes_all[1].get_ylim()[0])
    row1_top = max(axes_all[0].get_ylim()[1], axes_all[1].get_ylim()[1])
    row2_bottom = min(axes_all[2].get_ylim()[0], axes_all[3].get_ylim()[0])
    row2_top = max(axes_all[2].get_ylim()[1], axes_all[3].get_ylim()[1])
    for ax in axes_all[:2]:
        ax.set_ylim(row1_bottom, row1_top)
    for ax in axes_all[2:]:
        ax.set_ylim(row2_bottom, row2_top)
    fig_all.supxlabel("Leverage (hᵢ)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", y=0.055)
    fig_all.supylabel("Standardized Residuals", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", x=0.035)
    fig_all.subplots_adjust(left=0.09, right=0.985, top=0.955, bottom=0.10, hspace=0.34, wspace=0.33)
    save_figure(fig_all, "Figure_Williams_Combined_IC50_EC50")
    save_figure(fig_all, "Figure_Williams_AD_optimized")
    plt.close(fig_all)

    logging.info("Generating combined Williams Plot figure without panel legends...")
    fig_all_nolegend, axes_all_nolegend = plt.subplots(nrows=2, ncols=2, figsize=(18.6, 16.4), dpi=300, facecolor="white")
    axes_all_nolegend = axes_all_nolegend.flatten()
    for ax, panel_key in zip(axes_all_nolegend, ["A", "B", "C", "D"]):
        endpoint = PREDICTION_PATHS[panel_key]["endpoint"]
        X_train_scaled, y_train, X_pred_scaled, df_pred_valid = prepare_scaled_matrices(endpoint, panel_key, training_cache, prediction_cache)
        create_williams_plot(ax, X_train_scaled, y_train, X_pred_scaled, df_pred_valid, PREDICTION_PATHS[panel_key]["tag"], PREDICTION_PATHS[panel_key]["title"], endpoint, show_legend=False)
    row1_bottom = min(axes_all_nolegend[0].get_ylim()[0], axes_all_nolegend[1].get_ylim()[0])
    row1_top = max(axes_all_nolegend[0].get_ylim()[1], axes_all_nolegend[1].get_ylim()[1])
    row2_bottom = min(axes_all_nolegend[2].get_ylim()[0], axes_all_nolegend[3].get_ylim()[0])
    row2_top = max(axes_all_nolegend[2].get_ylim()[1], axes_all_nolegend[3].get_ylim()[1])
    for ax in axes_all_nolegend[:2]:
        ax.set_ylim(row1_bottom, row1_top)
    for ax in axes_all_nolegend[2:]:
        ax.set_ylim(row2_bottom, row2_top)
    fig_all_nolegend.supxlabel("Leverage (hᵢ)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", y=0.055)
    fig_all_nolegend.supylabel("Standardized Residuals", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", x=0.035)
    fig_all_nolegend.subplots_adjust(left=0.09, right=0.985, top=0.955, bottom=0.10, hspace=0.34, wspace=0.33)
    save_figure(fig_all_nolegend, "Figure_Williams_Combined_IC50_EC50_NoLegend")
    plt.close(fig_all_nolegend)

    logging.info("Generating combined Williams Plot figure without molecule labels...")
    fig_all_nolabels, axes_all_nolabels = plt.subplots(nrows=2, ncols=2, figsize=(18.6, 16.4), dpi=300, facecolor="white")
    axes_all_nolabels = axes_all_nolabels.flatten()
    for ax, panel_key in zip(axes_all_nolabels, ["A", "B", "C", "D"]):
        endpoint = PREDICTION_PATHS[panel_key]["endpoint"]
        X_train_scaled, y_train, X_pred_scaled, df_pred_valid = prepare_scaled_matrices(endpoint, panel_key, training_cache, prediction_cache)
        create_williams_plot(
            ax,
            X_train_scaled,
            y_train,
            X_pred_scaled,
            df_pred_valid,
            PREDICTION_PATHS[panel_key]["tag"],
            PREDICTION_PATHS[panel_key]["title"],
            endpoint,
            show_legend=True,
            show_molecule_labels=False,
        )
    row1_bottom = min(axes_all_nolabels[0].get_ylim()[0], axes_all_nolabels[1].get_ylim()[0])
    row1_top = max(axes_all_nolabels[0].get_ylim()[1], axes_all_nolabels[1].get_ylim()[1])
    row2_bottom = min(axes_all_nolabels[2].get_ylim()[0], axes_all_nolabels[3].get_ylim()[0])
    row2_top = max(axes_all_nolabels[2].get_ylim()[1], axes_all_nolabels[3].get_ylim()[1])
    for ax in axes_all_nolabels[:2]:
        ax.set_ylim(row1_bottom, row1_top)
    for ax in axes_all_nolabels[2:]:
        ax.set_ylim(row2_bottom, row2_top)
    fig_all_nolabels.supxlabel("Leverage (hᵢ)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", y=0.055)
    fig_all_nolabels.supylabel("Standardized Residuals", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", x=0.035)
    fig_all_nolabels.subplots_adjust(left=0.09, right=0.985, top=0.955, bottom=0.10, hspace=0.34, wspace=0.33)
    save_figure(fig_all_nolabels, "Figure_Williams_Combined_IC50_EC50_NoLabels")
    plt.close(fig_all_nolabels)

    logging.info("Generating JCIM final Williams Plot figure (scaled no-label version)...")
    style_backup = {
        "panel_tag": PANEL_TAG_FONTSIZE,
        "panel_title": PANEL_TITLE_FONTSIZE,
        "axis_label": AXIS_LABEL_FONTSIZE,
        "tick": TICK_FONTSIZE,
        "legend": LEGEND_FONTSIZE,
        "hstar": HSTAR_FONTSIZE,
        "predicted_size": PREDICTED_SIZE,
        "training_alpha": TRAINING_ALPHA,
        "training_size": TRAINING_SIZE,
        "axis_spine": AXIS_SPINE_WIDTH,
        "tick_width": TICK_WIDTH,
        "tick_length": TICK_LENGTH,
        "sigma_lw": SIGMA_LINEWIDTH,
        "hstar_lw": HSTAR_LINEWIDTH,
        "legend_train_ms": LEGEND_TRAIN_MARKERSIZE,
        "legend_pred_ms": LEGEND_PRED_MARKERSIZE,
    }
    PANEL_TAG_FONTSIZE = 27
    PANEL_TITLE_FONTSIZE = 24
    AXIS_LABEL_FONTSIZE = 28
    TICK_FONTSIZE = 24
    LEGEND_FONTSIZE = 17
    HSTAR_FONTSIZE = 16.8
    PREDICTED_SIZE = 60
    TRAINING_ALPHA = 0.16
    TRAINING_SIZE = 15
    AXIS_SPINE_WIDTH = 1.3
    TICK_WIDTH = 1.3
    TICK_LENGTH = 5.5
    SIGMA_LINEWIDTH = 1.2
    HSTAR_LINEWIDTH = 1.5
    LEGEND_TRAIN_MARKERSIZE = 8.2
    LEGEND_PRED_MARKERSIZE = 11.8

    fig_jcim, axes_jcim = plt.subplots(nrows=2, ncols=2, figsize=(24, 20.5), dpi=300, facecolor="white")
    axes_jcim = axes_jcim.flatten()
    for ax, panel_key in zip(axes_jcim, ["A", "B", "C", "D"]):
        endpoint = PREDICTION_PATHS[panel_key]["endpoint"]
        X_train_scaled, y_train, X_pred_scaled, df_pred_valid = prepare_scaled_matrices(endpoint, panel_key, training_cache, prediction_cache)
        create_williams_plot(
            ax,
            X_train_scaled,
            y_train,
            X_pred_scaled,
            df_pred_valid,
            PREDICTION_PATHS[panel_key]["tag"],
            PREDICTION_PATHS[panel_key]["title"],
            endpoint,
            show_legend=False,
            show_molecule_labels=False,
        )
    row1_bottom = min(axes_jcim[0].get_ylim()[0], axes_jcim[1].get_ylim()[0])
    row1_top = max(axes_jcim[0].get_ylim()[1], axes_jcim[1].get_ylim()[1])
    row2_bottom = min(axes_jcim[2].get_ylim()[0], axes_jcim[3].get_ylim()[0])
    row2_top = max(axes_jcim[2].get_ylim()[1], axes_jcim[3].get_ylim()[1])
    for ax in axes_jcim[:2]:
        ax.set_ylim(row1_bottom, row1_top)
    for ax in axes_jcim[2:]:
        ax.set_ylim(row2_bottom, row2_top)
    fig_jcim.supxlabel("Leverage (hᵢ)", fontsize=29.4, fontweight="bold", y=0.055)
    fig_jcim.supylabel("Standardized Residuals", fontsize=28, fontweight="bold", x=0.035)
    fig_jcim.subplots_adjust(left=0.09, right=0.985, top=0.955, bottom=0.10, hspace=0.30, wspace=0.32)
    save_final_jcim_outputs(fig_jcim, "Williams_plot_JCIM_final", dpi=1200)
    plt.close(fig_jcim)

    PANEL_TAG_FONTSIZE = style_backup["panel_tag"]
    PANEL_TITLE_FONTSIZE = style_backup["panel_title"]
    AXIS_LABEL_FONTSIZE = style_backup["axis_label"]
    TICK_FONTSIZE = style_backup["tick"]
    LEGEND_FONTSIZE = style_backup["legend"]
    HSTAR_FONTSIZE = style_backup["hstar"]
    PREDICTED_SIZE = style_backup["predicted_size"]
    TRAINING_ALPHA = style_backup["training_alpha"]
    TRAINING_SIZE = style_backup["training_size"]
    AXIS_SPINE_WIDTH = style_backup["axis_spine"]
    TICK_WIDTH = style_backup["tick_width"]
    TICK_LENGTH = style_backup["tick_length"]
    SIGMA_LINEWIDTH = style_backup["sigma_lw"]
    HSTAR_LINEWIDTH = style_backup["hstar_lw"]
    LEGEND_TRAIN_MARKERSIZE = style_backup["legend_train_ms"]
    LEGEND_PRED_MARKERSIZE = style_backup["legend_pred_ms"]

    logging.info("Generating EC50-only Williams Plot figure (panels C and D)...")
    fig_ec50, axes_ec50 = plt.subplots(nrows=1, ncols=2, figsize=(7.7, 3.35), dpi=300, facecolor="white")
    for ax, panel_key in zip(axes_ec50, ["C", "D"]):
        endpoint = PREDICTION_PATHS[panel_key]["endpoint"]
        X_train_scaled, y_train, X_pred_scaled, df_pred_valid = prepare_scaled_matrices(endpoint, panel_key, training_cache, prediction_cache)
        create_williams_plot(ax, X_train_scaled, y_train, X_pred_scaled, df_pred_valid, PREDICTION_PATHS[panel_key]["tag"], PREDICTION_PATHS[panel_key]["title"], endpoint)
    fig_ec50.supxlabel("Leverage (hᵢ)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", y=0.09)
    fig_ec50.supylabel("Standardized Residuals", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", x=0.04)
    ec_bottom = min(ax.get_ylim()[0] for ax in axes_ec50)
    ec_top = max(ax.get_ylim()[1] for ax in axes_ec50)
    for ax in axes_ec50:
        ax.set_ylim(ec_bottom, ec_top)
    fig_ec50.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.18, wspace=0.33)
    save_figure(fig_ec50, "Figure_Williams_EC50_Panels_CD")
    plt.close(fig_ec50)

    caption_note_path = os.path.join(OUTPUT_DIR, "Figure_Williams_EC50_caption_note.txt")
    with open(caption_note_path, "w", encoding="utf-8") as f:
        f.write(CAPTION_NOTE + "\n")
    logging.info("✓ Saved: %s", caption_note_path)

    logging.info("Done. Output directory: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
