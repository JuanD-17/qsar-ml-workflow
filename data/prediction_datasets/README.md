# QSAR Prediction Datasets

## Overview
These `.csv` files contain external molecular datasets used as input for QSAR-based antioxidant activity prediction, phytochemical screening, and external validation analyses. Each dataset groups compounds from different experimental, phytochemical, or bibliographic sources and serves as the basis for downstream prediction workflows and comparative analyses.

---

## Included Files

### 1. `Cannabaceae_molecules.csv`
Primary dataset containing phytocannabinoids reported mainly in *Cannabis sativa*.

**Expected content:**
- `Molecule_name`: compound name
- `plant_name`: associated plant species
- `retention_time`: experimental retention time
- `concentration`: reported concentration
- `units`: concentration units
- `plant_part`: analyzed plant tissue
- `Smiles`: chemical structure in SMILES format
- `Chembl`: associated ChEMBL identifier or reference

**Usage:**
Used as an input dataset for QSAR prediction of cannabinoid compounds with potential antioxidant activity.

---

### 2. `Cannabaceae_molecules2.csv`
Complementary dataset containing Cannabaceae-derived compounds, especially metabolites reported in *Humulus lupulus* and additional literature-derived sources.

**Expected content:**
- `Molecule_name`: compound name
- `Smiles`: chemical structure in SMILES format
- `plant_name`: associated plant species
- `reference`: bibliographic reference or source

**Usage:**
Used as an extended dataset to expand the chemical space of Cannabaceae compounds evaluated through QSAR workflows.

---

### 3. `Terpenos.csv`
Dataset containing individual terpenes used for QSAR prediction analyses.

**Expected content:**
- `Molecule_name`: terpene name
- `Smiles`: chemical structure in SMILES format
- `Chembl`: chemical reference or identifier

**Notes:**
- The dataset may contain equivalent names or stereochemical variants, for example:
  - `beta-Cariopilene`
  - `Nerolidol 1`
  - `Nerolidol 2`

**Usage:**
Used for QSAR prediction of terpene compounds and subsequent integration into mixture analyses and additive baseline calculations.

---

### 4. `molecules3.csv`
External reference dataset containing natural antioxidants and additional compounds collected from multiple public databases and literature sources.

**Expected content:**
- `Molecule_name`: compound name
- `Smiles`: chemical structure in SMILES format
- `Data_Source`: source database or literature origin

**Usage:**
Serves as an external reference dataset for validation, chemical comparison, and evaluation of QSAR predictions outside the core Cannabaceae chemical space.

---

## General Format
- Main separator: `;`
- Recommended encoding: `UTF-8`
- The key structural column is `Smiles`
- Molecular names may require normalization for downstream integration and comparison analyses

---

## Important Notes
- Metabolite names may contain orthographic variants or alternative nomenclatures between datasets.
- For integrated analyses, the creation of a normalized molecular identifier based on `Molecule_name` is recommended.
- These files represent molecular input datasets and not final QSAR prediction outputs.
- QSAR prediction results derived from these datasets are stored separately in downstream prediction and consolidated analysis files.
