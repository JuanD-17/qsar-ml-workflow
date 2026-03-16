# Curated datasets

This folder contains curated and progressively processed antioxidant activity datasets derived from the consolidated raw dataset located in `data/raw`.

The datasets in this directory represent intermediate and advanced preprocessing stages used to normalize activity values, restrict endpoints, and prepare the data for QSAR modeling workflows.

---

# Dataset lineage: `antioxidant9.csv` → `antioxidant12.csv`

This document summarizes the curation and transformation stages across the intermediate antioxidant datasets used before QSAR modeling.

---

## 1) `antioxidant9.csv` — Harmonized raw export (IC50 + EC50)

### Role in pipeline
- First standardized tabular export after initial ChEMBL filtering.
- Retains both biological endpoints (`IC50`, `EC50`) within a single dataset.
- Still includes heterogeneous experimental units and mixed assay conditions.

### Observed structure
- **Rows:** 17,103  
- **Columns:** 49  
- **Delimiter:** semicolon (`;`)

### Activity composition
- `IC50`: 15,090  
- `EC50`: 2,013  

### Common units
- `nM` (majority)
- `ug.mL-1`
- `%`
- other heterogeneous activity units

### Interpretation
This dataset functions as a **harmonized curated export** preserving full assay metadata while consolidating antioxidant activity records.  
However, heterogeneous units and mixed endpoints still prevent direct QSAR modeling.

---

## 2) `antioxidant10.csv` — Unit normalization + log transform stage

### Role in pipeline
Introduces activity normalization and the base logarithmic transformation required for modeling workflows.

### Observed structure
- **Rows:** 17,006  
- **Columns:** 51  

### New columns introduced
- `Normalized Value (nM)`
- `Log10 Value (nM)`

### Endpoint composition
- `IC50`: 15,026  
- `EC50`: 1,980  

### Important note
`Log10 Value (nM)` may contain extreme or non-finite values (e.g., `-inf`) caused by problematic normalized values or zero activity entries.

### Interpretation
This dataset represents a **partially normalized stage**, where activity units have been harmonized to nM and log-transformed, but additional cleaning is still required before model training.

---

## 3) `antioxidant11.csv` — Modeling-ready IC50 subset

### Role in pipeline
Restricts the dataset to a single biological endpoint (**IC50**) to ensure consistency for QSAR modeling.

### Observed structure
- **Rows:** 14,908  
- **Columns:** 60  

### Endpoint composition
- `IC50`: 14,908 (100%)

### Added statistical diagnostics
Global distribution descriptors were incorporated to characterize the activity landscape:

- `Deviation_from_mean`
- `Deviation_from_median`
- `Mean_log10`
- `Median_log10`
- `Std_log10`
- `Anderson_stat`
- `Anderson_critical_value_5pct`
- `Skewness`
- `Kurtosis`

### Interpretation
This dataset represents the **first model-oriented processed dataset**, where endpoint heterogeneity has been removed and statistical diagnostics are available to support distribution analysis.

---

## 4) `antioxidant12.csv` — Stratified IC50 dataset

### Role in pipeline
Builds upon `antioxidant11.csv` by introducing potency-based stratification used for category-specific QSAR modeling.

### Observed structure
- **Rows:** 14,908  
- **Columns:** 61  

### New column
- `Category`

### Category distribution

| Category | Records |
|--------|--------|
| Bajo | 4,965 |
| Medio | 4,981 |
| Alto | 4,962 |

### Interpretation
This dataset is a **stratified IC50 dataset** enabling potency-specific modeling workflows (e.g., high-, medium-, and low-activity models).

---

# Practical summary

| Dataset | Stage | Endpoint status | Main transformation |
|------|------|------|------|
| `antioxidant9.csv` | Curated base export | Mixed (`IC50`, `EC50`) | Harmonized export with heterogeneous units |
| `antioxidant10.csv` | Normalization stage | Mixed (`IC50`, `EC50`) | Activity normalization and log transformation |
| `antioxidant11.csv` | Modeling preprocessing | IC50 only | Endpoint restriction and distribution diagnostics |
| `antioxidant12.csv` | Stratified modeling dataset | IC50 only | Potency class labeling (`Bajo`, `Medio`, `Alto`) |

---


