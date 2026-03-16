# Data directory

This directory contains all datasets used throughout the QSAR modeling
workflow for antioxidant activity prediction.

The datasets are organized according to their stage in the data processing
pipeline, from raw data acquisition to modeling-ready datasets.

---

# Directory structure

data/

raw/  
Original datasets obtained from external databases (e.g., ChEMBL, PubChem) before strict curation.

curated/  
Cleaned and normalized datasets derived from the raw data. These datasets contain harmonized units, endpoint filtering, and statistical diagnostics.

processed/  
Final modeling datasets derived from curated data and used directly by QSAR models.

---

# Data pipeline overview

The datasets follow the pipeline:

data_acquisition → data/raw → data/curated → data/processed → models

Where:

data_acquisition contains scripts used to retrieve bioactivity data from external databases.

data/raw contains heterogeneous activity datasets prior to strict normalization and endpoint filtering.

data/curated contains progressively cleaned datasets where activity values are normalized and endpoints are harmonized.

data/processed contains modeling-ready datasets used for QSAR model training and evaluation.

---

# Dataset lineage summary

The main dataset transformation workflow is:

raw datasets  
↓  
curated datasets  
↓  
antioxidant12.csv (IC50 processed dataset with potency category)  
↓  
processed datasets for model training

Processed datasets are then used by the modeling scripts located in the `models/` directory.

---
