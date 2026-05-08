# Terpene QSAR Prediction Workflow

This directory contains the complete computational workflow used to generate antioxidant IC50 predictions for terpene-derived compounds identified from Cannabaceae phytochemical datasets.

The workflow integrates multiple activity-stratified QSAR submodels followed by an inverse-variance weighted consensus prediction framework to obtain final consolidated predictions and confidence intervals.

The prediction infrastructure was designed to improve predictive robustness across heterogeneous IC50 activity regions by combining specialized local QSAR models trained on distinct activity distributions.

---

# Overview

The workflow follows four major stages:

1. Molecular preprocessing and descriptor generation
2. Activity-stratified QSAR prediction
3. Confidence interval estimation
4. Weighted consensus prediction consolidation

This strategy enables the integration of multiple local QSAR models to improve prediction stability and uncertainty estimation for structurally diverse terpene-derived compounds.

---

# Workflow Structure

```text id="8nrxj2"
data/prediction_dataset/Terpenos.csv
        ↓
ensayo322 – ensayo329
(activity-stratified QSAR submodels)
        ↓
Individual predictions + confidence intervals
        ↓
ensayo330
(weighted consensus integration)
        ↓
Final consolidated predictions
```

---

# Directory Structure

```text id="f0v4j8"
predictions/
└── terpenes/
    ├── ensayo322/
    ├── ensayo323/
    ├── ensayo324/
    ├── ensayo325/
    ├── ensayo326/
    ├── ensayo327/
    ├── ensayo328/
    ├── ensayo329/
    └── ensayo330/
```

Each directory corresponds to an independent QSAR prediction workflow.

The directories `ensayo322` to `ensayo329` contain individual activity-stratified QSAR prediction pipelines, while `ensayo330` contains the weighted consensus integration workflow used to generate the final consolidated predictions.

---

# Input Dataset

Input file:

```text id="k7x4eu"
data/prediction_dataset/Terpenos.csv
```

The dataset contains terpene-derived compounds represented by canonical SMILES strings and associated metadata.

Typical columns include:

| Column                | Description                         |
| --------------------- | ----------------------------------- |
| Molecule_name         | Compound name                       |
| SMILES                | Canonical SMILES representation     |
| Final_Category        | Final chemical classification       |
| Classification_Method | Chemical ontology assignment method |
| Category_Confidence   | Confidence score for classification |

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

# ensayo322 – ensayo329 Workflows

Each workflow follows the same general computational procedure:

1. SMILES validation
2. Molecular descriptor calculation
3. Morgan fingerprint (ECFP6) generation
4. Feature preprocessing
5. QSAR prediction
6. Confidence interval estimation
7. Export of prediction tables

Each workflow produces an independent prediction table used later during the consensus integration stage.

---

# Molecular Representation

The QSAR workflows use molecular fingerprints and physicochemical descriptors generated from RDKit.

Primary molecular representations include:

* Extended Connectivity Fingerprints (ECFP6)
* Physicochemical descriptors
* Preprocessed feature matrices compatible with trained QSAR models

Feature preprocessing procedures were matched to the original model training pipelines.

---

# Consensus Prediction Framework (ensayo330)

Final consolidated predictions are generated using an inverse-variance weighting strategy inspired by meta-analysis methodologies.

Each individual prediction contributes proportionally according to its associated uncertainty:

```text id="2nq2x5"
weight ~ 1 / σ²
```

Where:

* σ² represents the variance associated with each prediction
* Lower uncertainty predictions receive greater weight during final prediction consolidation

The weighted integration framework combines all activity-stratified predictions into a single consensus estimate.

---

# Final Output Variables

The consensus workflow produces the following variables:

| Column                       | Description                             |
| ---------------------------- | --------------------------------------- |
| Weighted_Prediction_Log10_nM | Final weighted consensus prediction     |
| Final_Lower_CI_95            | Lower 95% confidence interval           |
| Final_Upper_CI_95            | Upper 95% confidence interval           |
| Final_CI_Width               | Width of confidence interval            |
| Consensus_Level              | Agreement level among submodels         |
| Prediction_Source            | Source model contributing to prediction |

---

# Output Directory

Final consolidated predictions are exported to:

```text id="w0z55r"
predictions/terpenes/ensayo330/
```

The exported files contain:

* Final weighted predictions
* Confidence interval estimates
* Consensus metrics
* Prediction provenance metadata

---

# Applicability Domain Considerations

Predictions should be interpreted within the applicability domain of the trained QSAR models.

The reliability of predictions may decrease for compounds located outside the structural and physicochemical space represented in the original training datasets.

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

# Notes

* Predictions correspond to computationally estimated antioxidant IC50 values.
* The workflow was designed for external phytochemical screening and compound prioritization.
* Lower predicted IC50 values indicate higher predicted antioxidant potency.
* Confidence intervals reflect model uncertainty and should not be interpreted as experimental variability.
* The workflow is intended for hypothesis generation and prioritization rather than direct biological confirmation.

---

