# Cannabaceae QSAR Prediction Workflow

This directory contains the complete computational workflow used to generate antioxidant IC50 predictions for the primary Cannabaceae molecular dataset.

The workflow integrates multiple activity-stratified QSAR submodels followed by an inverse-variance weighted consensus prediction framework to obtain final consolidated predictions and confidence intervals for each molecule.

The final output of this workflow is the consolidated prediction dataset generated in `ensayo334`, which summarizes both the final potency classification and the weighted quantitative prediction for each compound included in the Cannabaceae molecular dataset.

---

# Overview

The workflow is divided into two major levels:

1. Individual activity-stratified QSAR predictions (`ensayo177–184`)
2. Final weighted consensus integration (`ensayo334`)

The overall strategy was designed to improve predictive robustness across heterogeneous IC50 activity regions by integrating specialized local QSAR models trained on different potency distributions.

---

# Workflow Structure

```text id="qf8k4d"
data/prediction_dataset/Cannabaceae_molecules.csv
        ↓
ensayo177 – ensayo184
(activity-stratified QSAR submodels)
        ↓
Individual predictions + confidence intervals
        ↓
ensayo334
(weighted consensus integration)
        ↓
Final consolidated predictions
```

---

# Directory Structure

```text id="d2rv0k"
predictions/
└── Cannabaceae_molecules/
    ├── ensayo177/
    ├── ensayo178/
    ├── ensayo179/
    ├── ensayo180/
    ├── ensayo181/
    ├── ensayo182/
    ├── ensayo183/
    ├── ensayo184/
    └── ensayo334/
```

Each directory corresponds to an independent QSAR prediction workflow.

The directories `ensayo177` to `ensayo184` contain individual activity-stratified QSAR prediction pipelines, while `ensayo334` contains the weighted consensus integration workflow used to generate the final consolidated predictions.

---

# Input Dataset

Input file:

```text id="s3qj7w"
data/prediction_dataset/Cannabaceae_molecules.csv
```

The dataset contains phytocannabinoids and structurally related Cannabaceae-derived compounds represented by canonical SMILES strings and associated metadata.

Typical columns include:

| Column         | Description                               |
| -------------- | ----------------------------------------- |
| Molecule_name  | Compound name                             |
| plant_name     | Plant source                              |
| retention_time | Retention time from metabolomic profiling |
| concentration  | Measured abundance                        |
| units          | Concentration units                       |
| plant_part     | Plant tissue source                       |
| SMILES         | Canonical SMILES representation           |
| Chembl         | Associated ChEMBL identifier              |

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

# Individual Prediction Workflows (`ensayo177–184`)

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

## `ensayo177`

* Internal category: `Bajo_1`
* Activity region: Low IC50
* Reference training workflow: `ensayo120`

## `ensayo178`

* Internal category: `Bajo_2`
* Activity region: Low IC50
* Reference training workflow: `ensayo121`

## `ensayo179`

* Internal category: `Bajo_3`
* Activity region: Low IC50
* Reference training workflow: `ensayo122`

## `ensayo180`

* Internal category: `Medio_1`
* Activity region: Medium IC50
* Reference training workflow: `ensayo112_fixed`

## `ensayo181`

* Internal category: `Medio_2`
* Activity region: Medium IC50
* Reference training workflow: `ensayo117`

## `ensayo182`

* Internal category: `Medio_3`
* Activity region: Medium IC50
* Reference training workflow: `ensayo118`

## `ensayo183`

* Internal category: `Medio_4`
* Activity region: Medium IC50
* Reference training workflow: `ensayo119`

## `ensayo184`

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

# Consensus Prediction Framework (`ensayo334`)

The workflow `ensayo334` acts as a weighted meta-analytic integration layer.

This stage does not retrain models. Instead, it consolidates the eight previously generated predictions into a single uncertainty-aware consensus estimate.

The workflow performs the following operations:

1. Loads the base Cannabaceae molecular dataset
2. Imports the eight individual prediction tables generated in `ensayo177–184`
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

```text id="4wq2zz"
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

The consolidated prediction file generated in `ensayo334` includes the following key variables:

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

```text id="4ehrrk"
predictions/Cannabaceae_molecules/ensayo334/
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

* `ensayo177–179` = Low IC50 prediction block
* `ensayo180–183` = Medium IC50 prediction block
* `ensayo184` = High IC50 prediction block
* `ensayo334` = Final weighted integration workflow

In practical terms:

* `ensayo177–184` generate the individual QSAR predictions
* `ensayo334` generates the final consolidated prediction and final activity classification for each Cannabaceae molecule

---

# Notes

* Predictions correspond to computationally estimated antioxidant IC50 values.
* Lower predicted IC50 values indicate higher predicted antioxidant potency.
* Confidence intervals reflect model uncertainty and should not be interpreted as experimental variability.
* The workflow is intended for hypothesis generation and compound prioritization rather than direct biological confirmation.

---


