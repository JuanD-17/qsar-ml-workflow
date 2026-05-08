# Cannabaceae Extended QSAR Prediction Workflow

This directory contains the complete computational workflow used to generate antioxidant IC50 predictions for the extended Cannabaceae molecular dataset.

The workflow integrates multiple activity-stratified QSAR submodels followed by an inverse-variance weighted consensus prediction framework to obtain final consolidated predictions and confidence intervals for each molecule.

The final output of this workflow is the consolidated prediction dataset generated in `ensayo335`, which summarizes both the final potency classification and the weighted quantitative prediction for each compound included in the extended Cannabaceae molecular dataset.

---

# Overview

The workflow is divided into two major levels:

1. Individual activity-stratified QSAR predictions (`ensayo199–206`)
2. Final weighted consensus integration (`ensayo335`)

The overall strategy was designed to improve predictive robustness across heterogeneous IC50 activity regions by integrating specialized local QSAR models trained on different potency distributions.

---

# Workflow Structure

```text id="9t6j2n"
data/prediction_dataset/Cannabaceae_molecules2.csv
        ↓
ensayo199 – ensayo206
(activity-stratified QSAR submodels)
        ↓
Individual predictions + confidence intervals
        ↓
ensayo335
(weighted consensus integration)
        ↓
Final consolidated predictions
```

---

# Directory Structure

```text id="7z2m4p"
predictions/
└── Cannabaceae_molecules2/
    ├── ensayo199/
    ├── ensayo200/
    ├── ensayo201/
    ├── ensayo202/
    ├── ensayo203/
    ├── ensayo204/
    ├── ensayo205/
    ├── ensayo206/
    └── ensayo335/
```

Each directory corresponds to an independent QSAR prediction workflow.

The directories `ensayo199` to `ensayo206` contain individual activity-stratified QSAR prediction pipelines, while `ensayo335` contains the weighted consensus integration workflow used to generate the final consolidated predictions.

---

# Input Dataset

Input file:

```text id="4b4f1x"
data/prediction_dataset/Cannabaceae_molecules2.csv
```

The dataset contains additional Cannabaceae-derived compounds, particularly metabolites reported in *Humulus lupulus*, *Cannabis sativa*, *Trema micrantha*.

Typical columns include:

| Column        | Description                     |
| ------------- | ------------------------------- |
| Molecule_name | Compound name                   |
| SMILES        | Canonical SMILES representation |
| plant_name    | Plant source                    |
| reference     | Literature reference or source  |

This dataset expands the chemical space relative to the primary `Cannabaceae_molecules.csv` dataset.

---

# Activity-Stratified QSAR Models

The workflow uses multiple local QSAR models trained on different IC50 activity regions.

The activity regions correspond to:

| Region      | Description                          |
| ----------- | ------------------------------------ |
| Low IC50    | Higher predicted antioxidant potency |
| Medium IC50 | Intermediate predicted potency       |
| High IC50   | Lower predicted antioxidant potency  |

Lower IC50 values indicate stronger predicted antioxidant activity.

Each submodel independently generates:

* Predicted log10(IC50 nM)
* 95% confidence intervals
* Prediction uncertainty estimates

This activity-stratified strategy was implemented to reduce prediction discontinuities and improve local model specialization across heterogeneous potency ranges.

---

# Individual Prediction Workflows (`ensayo199–206`)

All individual prediction workflows follow the same general computational procedure:

1. SMILES validation
2. Molecular descriptor calculation
3. Morgan fingerprint (ECFP6) generation
4. Feature preprocessing
5. QSAR prediction
6. Confidence interval estimation
7. Export of prediction tables

Each workflow produces an independent prediction table later integrated during the consensus stage.

---

# Individual QSAR Prediction Blocks

## `ensayo199`

* Internal category: `Bajo_1`
* Activity region: Low IC50
* Reference training workflow: `ensayo120`

## `ensayo200`

* Internal category: `Bajo_2`
* Activity region: Low IC50
* Reference training workflow: `ensayo121`

## `ensayo201`

* Internal category: `Bajo_3`
* Activity region: Low IC50
* Reference training workflow: `ensayo122`

## `ensayo202`

* Internal category: `Medio_1`
* Activity region: Medium IC50
* Reference training workflow: `ensayo112_fixed`

## `ensayo203`

* Internal category: `Medio_2`
* Activity region: Medium IC50
* Reference training workflow: `ensayo117`

## `ensayo204`

* Internal category: `Medio_3`
* Activity region: Medium IC50
* Reference training workflow: `ensayo118`

## `ensayo205`

* Internal category: `Medio_4`
* Activity region: Medium IC50
* Reference training workflow: `ensayo119`

## `ensayo206`

* Internal category: `Alto_1`
* Activity region: High IC50
* Reference training workflow: `ensayo171`
* Model type: stacking model

---

# Molecular Representation

The QSAR workflows use molecular fingerprints and physicochemical descriptors generated from RDKit.

Primary molecular representations include:

* Extended Connectivity Fingerprints (ECFP6)
* Physicochemical descriptors
* Preprocessed feature matrices compatible with trained QSAR models

Feature preprocessing procedures were matched to the original model training pipelines.

---

# Consensus Prediction Framework (`ensayo335`)

The workflow `ensayo335` acts as a weighted meta-analytic integration layer.

This stage does not retrain models. Instead, it consolidates the eight previously generated predictions into a single uncertainty-aware consensus estimate.

The workflow performs the following operations:

1. Loads the base extended Cannabaceae molecular dataset
2. Imports the eight individual prediction tables generated in `ensayo199–206`
3. Estimates prediction precision from uncertainty intervals
4. Calculates weighted quantitative predictions
5. Assigns the final activity category
6. Propagates final uncertainty estimates

---

# Weighted Prediction Strategy

The final consolidated prediction is stored in:

* `Weighted_Prediction_Log10_nM`

Each individual QSAR model contributes:

* a predicted `log10(nM)` value
* a 95% confidence interval

The workflow converts model uncertainty into a precision estimate and assigns weights approximately proportional to:

```text id="5v8h2m"
weight ~ 1 / σ²
```

Where:

* σ² represents the variance associated with each prediction
* Lower uncertainty predictions receive greater weight during final integration

The final prediction corresponds to a weighted average of all individual activity-stratified predictions.

---

# Final Category Assignment

The final categorical classification is determined using:

1. Weighted consensus voting among:

   * `Bajo`
   * `Medio`
   * `Alto`

2. Fallback classification using the best-performing model when consensus confidence is insufficient.

The workflow generates:

* `Final_Category`
* `Classification_Method`
* `Category_Confidence`
* `Consensus_Level`

---

# Final Output Variables

The consolidated prediction file generated in `ensayo335` includes the following key variables:

| Column                       | Description                                 |
| ---------------------------- | ------------------------------------------- |
| SMILES                       | Canonical molecular representation          |
| Molecule_name                | Compound name                               |
| Final_Category               | Final predicted potency category            |
| Classification_Method        | Consensus or fallback classification method |
| Category_Confidence          | Confidence score for final classification   |
| Consensus_Level              | Agreement level among submodels             |
| Weighted_Prediction_Log10_nM | Final weighted quantitative prediction      |
| Final_Lower_CI_95            | Lower 95% confidence interval               |
| Final_Upper_CI_95            | Upper 95% confidence interval               |
| Final_CI_Width               | Width of final confidence interval          |

---

# Output Directory

Final consolidated predictions are exported to:

```text id="9r7d3k"
predictions/Cannabaceae_molecules2/ensayo335/
```

The exported files contain:

* final weighted predictions
* confidence interval estimates
* consensus metrics
* prediction provenance metadata

---

# Applicability Domain Considerations

Predictions should be interpreted within the applicability domain of the trained QSAR models.

Prediction reliability may decrease for compounds located outside the structural and physicochemical space represented in the original training datasets.

Confidence intervals provide an uncertainty-aware framework for interpreting prediction robustness.

---

# Computational Environment

The workflow was developed using:

* Python 3.9.6
* RDKit
* scikit-learn
* pandas
* numpy

Additional utilities were used for:

* molecular preprocessing
* descriptor generation
* uncertainty estimation
* weighted prediction consolidation

QSAR models were previously trained and serialized prior to external prediction workflows.

---

# Conceptual Summary of the Workflow

* `ensayo199–201` = Low IC50 prediction block
* `ensayo202–205` = Medium IC50 prediction block
* `ensayo206` = High IC50 prediction block
* `ensayo335` = Final weighted integration workflow

In practical terms:

* `ensayo199–206` generate the individual QSAR predictions
* `ensayo335` generates the final consolidated prediction and final activity classification for each molecule in the extended Cannabaceae dataset

---

# Notes

* Predictions correspond to computationally estimated antioxidant IC50 values.
* Lower predicted IC50 values indicate higher predicted antioxidant potency.
* Confidence intervals reflect model uncertainty and should not be interpreted as experimental variability.
* The workflow is intended for hypothesis generation and compound prioritization rather than direct biological confirmation.

---

