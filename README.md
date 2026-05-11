# qsar-ml-workflow

**Associated manuscript:**
Machine Learning-Based Chemoinformatics Screening and Experimental Validation of Antioxidant Activity in Cannabaceae

---

## Overview

This repository hosts machine learning–based QSAR models designed to predict the antioxidant activity of natural compounds, with a particular focus on metabolites derived from the **Cannabaceae** family (*Humulus lupulus*, *Trema micrantha*, *Cannabis* spp.).

The models were constructed using curated experimental IC₅₀ data retrieved from ChEMBL, molecular descriptors, and feature selection strategies to elucidate structure–activity relationships relevant to antioxidant mechanisms. Predictions are generated through a multi-model consensus framework in which specialized submodels independently evaluate molecules across different activity regions before generating a final weighted prediction.

---

## Repository Structure

```
qsar-ml-workflow/
├── data_acquisition/        # Scripts for bioactivity retrieval and phytochemical mining
├── data/
│   ├── raw/                 # Original datasets from external databases
│   ├── curated/             # Cleaned and normalized datasets
│   ├── processed/           # Modeling-ready datasets
│   └── prediction_datasets/ # External datasets for QSAR prediction and validation
├── models/                  # Trained QSAR models and evaluation results
├── predictions/             # Consolidated QSAR prediction workflows
│   ├── terpenes/
│   ├── Cannabaceae_molecules/
│   ├── Cannabaceae_molecules2/
│   ├── molecules3/
│   └── consolidated_predictions/
└── scripts/                 # Analysis scripts and derived figures
    ├── ensayo314/           # Applicability domain (Williams plot)
    ├── ensayo315/           # Mann–Whitney + stratification
    ├── ensayo317/           # Top 5 scaffolds
    ├── ensayo320/           # Activity cliffs
    ├── ensayo331/           # ChEBI annotation
    ├── ensayo332/           # ChEBI ontology summary
    ├── ensayo339/           # Cleveland plot
    ├── ensayo340/           # ABTS/DPPH heatmap
    ├── ensayo348/           # External validation (step 1)
    ├── ensayo349/           # External validation (step 2)
    ├── ensayo350/           # PCA + Tanimoto chemical space
    ├── ensayo275/           # QED — main dataset
    ├── ensayo276/           # QED — terpene subset
    └── ensayo279/           # QED — extended dataset
```

---

## Dataset Description

- **Data sources:** ChEMBL (v35) and PubChem BioAssay
- **Initial dataset:** 24,394 compounds
- **Curated dataset:** 15,026 molecules with IC₅₀ values; units standardized to nM

The data processing pipeline follows a sequential curation workflow:

```
data_acquisition → data/raw → data/curated → data/processed → models
```

A parallel phytochemical mining workflow targets Cannabaceae-specific compounds for downstream QSAR prediction and external validation.

---

## Data Acquisition

The `data_acquisition/` directory contains scripts for two parallel workflows.

**ChEMBL bioactivity retrieval** (`parte2.py`, `parte3.py`)
Queries the ChEMBL API using curated assay IDs (`molecules.csv`) to retrieve activity-level and molecule-level information. Results are filtered to retain only antioxidant-related assays based on keyword matching in the `assay_description` field (`Antioxidant`, `antioxidant`, `lipid peroxidation`). The filtered output (`molecule_filter.csv`) enters the data curation workflow in `data/raw/`.

**Cannabaceae phytochemical mining** (`mine_cannabaceae_phytochemicals.py`)
Mines and enriches phytochemical compound data for *Humulus lupulus*, *Trema micrantha*, and *Cannabis* spp. using PubChem and NCBI Entrez/PubMed. Outputs a structured compound dataset (`cannabaceae_phytochemicals_complete.csv`) used for prediction dataset construction and external validation.

> Requires local configuration of `Entrez.email`, `Entrez.api_key`, `SAVE_DIR`, `OUTPUT_FILE`, `CACHE_DIR`, and `FORCE_UPDATE` before execution.

---

## Modeling Strategy

All QSAR models follow a standardized computational workflow:

1. Molecular representation
2. Descriptor preprocessing
3. Feature selection
4. Model training and hyperparameter optimization
5. Model evaluation and performance reporting

### Molecular Representation

Three complementary molecular representations were used:

| Representation | Method | Parameters |
|---|---|---|
| ECFP (Morgan Fingerprints) | Extended Connectivity Fingerprints | Radius 3, 512 bits |
| Physicochemical descriptors | RDKit | MolWt, MolLogP, TPSA, HBD, HBA, rotatable bonds, Fsp³, Bertz complexity, Kier–Hall indices, and others |
| Murcko scaffold | Bemis–Murcko decomposition | Scaffold-level Morgan FP + physicochemical descriptors |

### Data Preprocessing

- SMILES validation using RDKit
- Removal of invalid or empty molecular structures
- Parallelized descriptor calculation with multi-core processing
- Descriptor caching to reduce redundant computations
- Missing value imputation (mean)
- Feature standardization (z-score normalization, `StandardScaler`)

### Feature Selection

Feature selection was performed using multiple complementary methods:

- **Boruta** — Random Forest-based all-relevant feature selection
- **Mutual Information Regression** — detection of non-linear dependencies
- **Random Forest Importance Thresholding** — mean importance filtering via `SelectFromModel`
- **LASSO Regression** — L1-regularized sparse feature selection

Final descriptor subsets combine selection masks to retain statistically relevant, non-redundant, and predictively informative features.

### Model Training

| Parameter | Value |
|---|---|
| Training / validation split | 75% / 25% |
| Hyperparameter optimization | GridSearchCV, 5-fold cross-validation |
| Base algorithm | Random Forest Regressor |
| Extended algorithms (low-activity branch) | MLP Neural Network, Stacking Ensemble (RF + MLP → Linear Regression meta-model) |

---

## Model Performance (Internal Validation)

The predictive performance of the developed QSAR models was evaluated using internal validation metrics including the coefficient of determination (R²) and the root mean squared error (RMSE). Metrics are reported for the training set, internal validation set, and 5-fold cross-validation (mean ± standard deviation).

The trained model artifacts are available in the [`models/`](models/) directory.

---

## Available Models

Models are organized by activity branch:

| Branch | Scripts | Algorithm |
|---|---|---|
| High activity | `ensayo120.py`, `ensayo121.py`, `ensayo122.py` | Random Forest Regressor |
| Medium activity | `ensayo112.py`, `ensayo117.py`, `ensayo118.py`, `ensayo119.py` | Random Forest Regressor |
| Low activity | `ensayo171.py` | Random Forest, MLP, Stacking Ensemble |

Each model script produces trained model artifacts (`.joblib`), feature selection masks, scaler and imputer objects, hyperparameter optimization results, performance metrics (CSV), observed vs. predicted plots, and feature selection diagnostic plots.

---

## QSAR Prediction Workflows

Predictions are generated through a **multi-model consensus framework**. Each workflow applies 8 specialized submodels organized into Low, Medium, and High activity branches, followed by a weighted consensus consolidation step.

| Workflow | Base dataset | Consolidation script | Output file |
|---|---|---|---|
| Terpenes | `Terpenos.csv` | `ensayo330` | `Cannabaceae_consolidated_predictions.csv` |
| Cannabaceae main | `Cannabaceae_molecules.csv` | `ensayo334` | `Cannabaceae_consolidated_predictions.csv` |
| Extended Cannabaceae | `Cannabaceae_molecules2.csv` | `ensayo335` | `Cannabaceae_consolidated_predictions.csv` |
| molecules3 | `molecules3.csv` | `ensayo336` | `molecules3_consolidated_predictions.csv` |

### Final Prediction Output

Each consolidated prediction file contains:

| Column | Description |
|---|---|
| `SMILES` | Molecular structure |
| `Molecule_name` | Molecule identifier |
| `Final_Category` | Predicted activity category (Low / Medium / High) |
| `Classification_Method` | Final decision strategy |
| `Category_Confidence` | Confidence of the assigned category |
| `Consensus_Level` | Agreement between submodels |
| `Weighted_Prediction_Log10_nM` | **Primary quantitative output** |
| `Final_Lower_CI_95` | Lower 95% confidence interval |
| `Final_Upper_CI_95` | Upper 95% confidence interval |
| `Final_CI_Width` | Width of the confidence interval |

> The main quantitative variable used in downstream prioritization and comparative analyses is `Weighted_Prediction_Log10_nM`.

---

## Analytical Scripts and Derived Figures

The `scripts/` directory consolidates the main analytical blocks performed after modeling and prediction. Each block is documented with its primary assay, key inputs, outputs, and role within the project.

### 1. Applicability Domain — Williams Plot (`ensayo314`)

Characterizes the applicability domain of the IC50 and EC50 models using leverage values and standardized residuals. Allows identification of structural outliers and evaluation of prediction reliability for new compounds.

Key outputs: `Williams_plot_JCIM_final.tiff`, `Williams_plot_leverage_values_IC50_EC50.csv`

---

### 2. ChEBI Ontological Annotation (`ensayo331`, `ensayo332`)

Resolves ChEBI ontological annotations for antioxidant compounds and summarizes chemical class distributions, supporting structural and functional interpretation of the modeled chemical space.

| Assay | Role |
|---|---|
| `ensayo331` | ChEBI annotation of `antioxidant11.csv` and `antioxidant19.csv`; combined annotation table |
| `ensayo332` | ChEBI class counts and percentages by activity level (primary and multilabel) |

---

### 3. Drug-likeness Assessment — QED (`ensayo275`, `ensayo276`, `ensayo279`)

Applies the Quantitative Estimate of Drug-likeness (QED) to prioritize molecules according to overall chemical quality across different molecular subsets.

| Assay | Subset |
|---|---|
| `ensayo275` | Main dataset (all molecules and leverage-filtered subset) |
| `ensayo276` | Terpene subset |
| `ensayo279` | Extended dataset |

---

### 4. Chemical Space Exploration — PCA (`ensayo350`)

Visualizes the chemical space coverage of cannabinoids and terpenes relative to the training set, using principal component analysis of molecular descriptors.

Key outputs: `PCA_ChemicalSpace_JCIM_Q1.tiff`, `pca_coordinates.csv`, `pca_coordinates_ec50.csv`

> `ensayo350` also contains the final Tanimoto similarity figures. PCA and Tanimoto analyses are consolidated in the same assay folder.

---

### 5. Activity Cliffs (`ensayo320`)

Identifies structurally similar molecules displaying large differences in antioxidant activity, both at the global dataset level and in representative structural pairs.

Key outputs: `Panel_B_Activity_Cliffs_IC50_EC50_Combined.tiff`, `Figure_Representative_Cliff_Pairs.tiff`

---

### 6. Dominant Scaffolds — Top 5 (`ensayo317`)

Summarizes the most frequent Bemis–Murcko scaffolds in the IC50 and EC50 datasets, including chemical structure, frequency, and aggregated activity metrics.

Key outputs: `Panel_A_Top5_Scaffolds_IC50_EC50_Combined.tiff/.svg`, `Panel_A_Top5_Scaffolds_IC50_EC50_Vertical.tiff/.svg`

> The complete `ensayo317/` folder should be uploaded to GitHub as it contains both final figures and the image crops (`recortes/`) used for figure assembly.

---

### 7. Potency Comparison — Cleveland Plot (`ensayo339`)

Compares project molecules against reference natural antioxidants, highlighting compounds with high predicted potency.

Key outputs: `Cleveland_High_Potency_Panels_AB.tiff`, `Natural_Antioxidant_Validation_Set_ranked.csv`

---

### 8. Experimental Antioxidant Activity — ABTS/DPPH Heatmap (`ensayo340`)

Summarizes pairwise comparisons of experimental antioxidant activity (ABTS and DPPH assays) at 10 mg/mL extract concentration using statistical heatmaps.

Input: `pairwise_antioxidant_comparisons.csv` (columns: `dilution`, `comparison`, `ABTS_estimate`, `ABTS_pvalue`, `DPPH_estimate`, `DPPH_pvalue`)

Key output: `Panel_Heatmap_Dilution50.tiff/.svg`

---

### 9. Dataset Characterization — Mann–Whitney + Stratification (`ensayo315`)

Integrates activity tertile stratification, IC50 vs. EC50 distribution comparisons, and Mann–Whitney U tests, together with supplementary Q-Q plots for normality assessment.

Key outputs: `Figure1_Dataset_Characteristics.tiff/.svg`, `Supplementary_QQPlot_IC50_EC50.tiff`

---

### 10. Structural Similarity — Tanimoto (`ensayo350`)

Evaluates the maximum Tanimoto similarity of cannabinoids and terpenes against the training set for both IC50 and EC50 analyses, assessing structural novelty and applicability domain coverage.

Key outputs: `Figure_SX_Tanimoto.tiff`, `Figure_SX_Tanimoto_EC50.tiff`

> Shares the `ensayo350/` folder with the PCA chemical space analysis (see section 4).

---

### 11. External Validation: Prediction vs. Experimental (`ensayo348`, `ensayo349`)

Documents the concordance between experimental and predicted IC50 and EC50 values in the external validation set. The workflow runs sequentially:

```
external_validation_results.csv
        ↓
    ensayo348.py → ensayo348/external_validation_results2.csv
        ↓
    ensayo349.py → IC50_EC50_comparison_panel.tiff/.svg
```

Input (`antioxidant19.csv`-derived): `external_validation_results.csv`

Key outputs: `ensayo348/external_validation_results2.csv`, `ensayo349/IC50_EC50_comparison_panel.tiff`

---

### Analysis Scripts — Quick Reference

| Block | Assay(s) | Role |
|---|---|---|
| Applicability domain | `ensayo314` | Williams plot, leverage and residuals |
| Dataset characterization | `ensayo315` | Tertile stratification + Mann–Whitney |
| Top 5 scaffolds | `ensayo317` | Dominant Bemis–Murcko scaffolds |
| Activity cliffs | `ensayo320` | Structural similarity vs. activity divergence |
| ChEBI annotation | `ensayo331`, `ensayo332` | Ontological class resolution and summary |
| Cleveland plot | `ensayo339` | Potency ranking vs. reference antioxidants |
| ABTS/DPPH heatmap | `ensayo340` | Experimental activity comparisons |
| External validation | `ensayo348`, `ensayo349` | Predicted vs. experimental concordance |
| PCA + Tanimoto | `ensayo350` | Chemical space and structural novelty |
| QED | `ensayo275`, `ensayo276`, `ensayo279` | Drug-likeness prioritization |

---

## Requirements

Python ≥ 3.8

| Library | Use |
|---|---|
| `pandas` | Data manipulation |
| `requests` | ChEMBL API queries |
| `rdkit` | Molecular descriptor calculation and SMILES validation |
| `scikit-learn` | Model training, feature selection, preprocessing |
| `joblib` | Model serialization |
| `tqdm` | Progress tracking |
| `pubchempy` | PubChem compound queries |
| `biopython` | NCBI Entrez access (`Bio.Entrez`) |
| `boruta` | Boruta feature selection |
| `matplotlib` / `seaborn` | Figure generation |

---

