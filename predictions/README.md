# QSAR Consolidated Prediction Workflows

This directory documents the main consolidated QSAR prediction workflows used throughout the project for antioxidant activity prioritization in Cannabaceae-related molecular datasets.

The prediction strategy follows a **multi-model consensus framework** in which specialized submodels independently evaluate molecules across different activity regions before generating a final weighted prediction.

The workflows summarized here include:
- Terpene-focused prediction datasets
- Cannabaceae phytochemical prediction datasets
- Extended phytochemical screening datasets
- External molecular collections used for virtual screening

---

## General Prediction Strategy

Each prediction workflow consists of:

- **Eight independent submodel predictions** generated from specialized QSAR models, organized into **Low**, **Medium**, and **High** activity branches.
- **One final consolidated prediction** generated through weighted consensus integration, accompanied by confidence metrics and final uncertainty estimates.

Thus, every prediction block documented here contains:
- 8 previous prediction outputs
- 1 final consolidated prediction file

---

## Common Prediction Output Structure

Each consolidated prediction workflow contains individual prediction `.csv` files, a final consolidation script, and a final consensus prediction dataset with the following columns:

| Column | Description |
|--------|-------------|
| `SMILES` | Molecular structure representation |
| `Molecule_name` | Molecule identifier |
| `Final_Category` | Final assigned activity category |
| `Classification_Method` | Final decision strategy |
| `Category_Confidence` | Confidence associated with the assigned category |
| `Consensus_Level` | Agreement between submodels |
| `Weighted_Prediction_Log10_nM` | Final weighted quantitative prediction |
| `Final_Lower_CI_95` | Lower 95% confidence interval |
| `Final_Upper_CI_95` | Upper 95% confidence interval |
| `Final_CI_Width` | Width of the final confidence interval |

> The main quantitative output used in downstream analyses is **`Weighted_Prediction_Log10_nM`**.

---

## Directory Structure

```
predictions/
├── terpenes/
├── Cannabaceae_molecules/
├── Cannabaceae_molecules2/
├── molecules3/
└── consolidated_predictions/
```

---

## Prediction Workflows

### 1. Terpenes

| Item | Value |
|------|-------|
| Final consolidated prediction | `ensayo330` |
| Output file | `Cannabaceae_consolidated_predictions.csv` |
| Base dataset | `Terpenos.csv` |

**Submodel predictions:**

| Assay | Prediction Branch |
|-------|------------------|
| ensayo322 | Low_1 |
| ensayo323 | Low_2 |
| ensayo324 | Low_3 |
| ensayo325 | Medium_1 |
| ensayo326 | Medium_2 |
| ensayo327 | Medium_3 |
| ensayo328 | Medium_4 |
| ensayo329 | High_1 |

- `ensayo322–324` → Low-activity specialized submodels
- `ensayo325–328` → Medium-activity specialized submodels
- `ensayo329` → High-activity specialized submodel
- `ensayo330` → Final weighted consensus prediction

---

### 2. Cannabaceae Main

| Item | Value |
|------|-------|
| Final consolidated prediction | `ensayo334` |
| Output file | `Cannabaceae_consolidated_predictions.csv` |
| Base dataset | `Cannabaceae_molecules.csv` |

**Submodel predictions:**

| Assay | Prediction Branch |
|-------|------------------|
| ensayo177 | Low_1 |
| ensayo178 | Low_2 |
| ensayo179 | Low_3 |
| ensayo180 | Medium_1 |
| ensayo181 | Medium_2 |
| ensayo182 | Medium_3 |
| ensayo183 | Medium_4 |
| ensayo184 | High_1 |

- `ensayo177–179` → Low-activity specialized submodels
- `ensayo180–183` → Medium-activity specialized submodels
- `ensayo184` → High-activity specialized submodel
- `ensayo334` → Final weighted consensus prediction

---

### 3. Extended Cannabaceae

| Item | Value |
|------|-------|
| Final consolidated prediction | `ensayo335` |
| Output file | `Cannabaceae_consolidated_predictions.csv` |
| Base dataset | `Cannabaceae_molecules2.csv` |

**Submodel predictions:**

| Assay | Prediction Branch |
|-------|------------------|
| ensayo199 | Low_1 |
| ensayo200 | Low_2 |
| ensayo201 | Low_3 |
| ensayo202 | Medium_1 |
| ensayo203 | Medium_2 |
| ensayo204 | Medium_3 |
| ensayo205 | Medium_4 |
| ensayo206 | High_1 |

- `ensayo199–201` → Low-activity specialized submodels
- `ensayo202–205` → Medium-activity specialized submodels
- `ensayo206` → High-activity specialized submodel
- `ensayo335` → Final weighted consensus prediction

---

### 4. molecules3

| Item | Value |
|------|-------|
| Final consolidated prediction | `ensayo336` |
| Output file | `molecules3_consolidated_predictions.csv` |
| Base dataset | `molecules3.csv` |

**Submodel predictions:**

| Assay | Prediction Branch |
|-------|------------------|
| ensayo221 | Low_1 |
| ensayo222 | Low_2 |
| ensayo223 | Low_3 |
| ensayo224 | Medium_1 |
| ensayo225 | Medium_2 |
| ensayo226 | Medium_3 |
| ensayo227 | Medium_4 |
| ensayo228 | High_1 |

- `ensayo221–223` → Low-activity specialized submodels
- `ensayo224–227` → Medium-activity specialized submodels
- `ensayo228` → High-activity specialized submodel
- `ensayo336` → Final weighted consensus prediction

---

## Interpretation of Final Predictions

The final consolidated prediction files contain both categorical and quantitative outputs.

| Column | Interpretation |
|--------|---------------|
| `Final_Category` | Final predicted activity category (Low, Medium, High) |
| `Classification_Method` | Final decision strategy (Consensus or fallback approach) |
| `Category_Confidence` | Confidence associated with the assigned category |
| `Consensus_Level` | Degree of agreement between independent submodels |
| `Weighted_Prediction_Log10_nM` | Final weighted quantitative prediction |
| `Final_CI_Width` | Final uncertainty interval width |

> The central quantitative variable used for downstream prioritization and comparative analyses is **`Weighted_Prediction_Log10_nM`**.

---

## General Workflow Summary

```
Input dataset
    │
    ▼
8 specialized submodels (Low / Medium / High branches)
    │
    ▼
Final consolidation workflow
    │
    ▼
Single robust prediction per molecule
```

This strategy leverages local model specialization while maintaining a unified final prediction framework suitable for downstream prioritization, applicability domain analysis, and experimental interpretation.

---

## Final Summary

This README provides a unified overview of the four major consolidated QSAR prediction workflows used throughout the project. Each workflow explicitly contains **8 previous independent submodel predictions** and **1 final weighted consolidated prediction**, improving reproducibility, prediction traceability.
