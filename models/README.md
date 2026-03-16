# QSAR Models

This directory contains trained **QSAR models for the prediction of antioxidant activity** using curated datasets derived from ChEMBL bioassays.

Models are organized according to **assay identifier and activity range**, and were developed using molecular fingerprints, physicochemical descriptors, and scaffold-based representations.

---

# Modeling Workflow

All QSAR models follow a standardized computational workflow:

1. Molecular representation  
2. Descriptor preprocessing  
3. Feature selection  
4. Model training and hyperparameter optimization  
5. Model evaluation and performance reporting  

---

# Molecular Representation

## 1. ECFP (Morgan Fingerprints)

Extended Connectivity Fingerprints (ECFP) were used to encode molecular topology.

- Radius: **3**
- Bit size: **512**

These fingerprints capture local atomic environments and are widely used in QSAR modeling.

---

## 2. Physicochemical Descriptors

Molecular descriptors were calculated using RDKit and include:

- Molecular weight (MolWt)
- Lipophilicity (MolLogP)
- Topological polar surface area (TPSA)
- Heavy atom count
- Hydrogen bond donors
- Hydrogen bond acceptors
- Total ring count
- Aromatic ring count
- Aliphatic ring count
- Number of rotatable bonds
- Fraction Csp³
- Molar refractivity
- Bertz molecular complexity index
- Kier–Hall connectivity indices
- Charge-related descriptors
- Conjugation and π-system proxies

These descriptors capture physicochemical properties relevant to antioxidant activity.

---

## 3. Murcko Scaffold Representation

Bemis–Murcko scaffolds were extracted from each molecule to capture core structural frameworks.

Scaffold representations include:

- Morgan fingerprints (512 bits)
- Scaffold-level physicochemical descriptors:
  - Scaffold molecular weight
  - Scaffold heavy atom count
  - Scaffold ring count
  - Scaffold aromatic ring count
  - Scaffold TPSA

This representation helps capture **scaffold-level structure–activity relationships**.

---

# Data Processing

Prior to model training, molecular data undergo several preprocessing steps:

- Validation of **SMILES strings using RDKit**
- Removal of invalid or empty molecular structures
- Parallelized descriptor calculation using multi-core processing
- Descriptor caching to reduce redundant computations
- Handling of missing values using **mean imputation**
- Replacement of NaN and infinite values
- Feature standardization using **z-score normalization (StandardScaler)**

These steps ensure numerical stability and reproducibility of the modeling pipeline.

---

# Feature Selection

Feature selection is performed using multiple complementary methods to identify informative and non-redundant descriptors.

Methods include:

- **Boruta**  
  Random Forest-based all-relevant feature selection

- **Mutual Information Regression**  
  Detection of non-linear dependencies between descriptors and activity

- **Random Forest Importance Thresholding**  
  Mean importance filtering using `SelectFromModel`

- **LASSO Regression**  
  L1-regularized sparse feature selection

Final descriptor subsets are obtained by combining selection masks to retain features that are:

- Statistically relevant  
- Non-redundant  
- Predictively informative  

---

# Model Training and Optimization

## Data Splitting

Datasets are divided into:

- **75% training set**
- **25% internal validation set**

---

## Hyperparameter Optimization

Hyperparameters are optimized using:

**GridSearchCV with 5-fold cross-validation**

Optimized parameters for Random Forest models include:

- Number of trees (`n_estimators`)
- Maximum tree depth
- Minimum samples per split
- Minimum samples per leaf
- Feature subsampling strategy (`max_features`)

For neural network models, additional parameters include:

- Hidden layer architecture
- Activation functions
- Regularization strength
- Solver type
- Learning rate
- Early stopping configuration

---

# Model Evaluation

Model performance is evaluated using:

- **R² (Coefficient of Determination)**
- **RMSE (Root Mean Squared Error)**

Metrics are reported for:

- Training set
- Internal validation set
- 5-fold cross-validation (mean ± standard deviation)

Observed vs. predicted activity plots are also generated for visual assessment of model performance.

---

# Available Models

The following QSAR models are implemented in this directory.

## High-Activity Models

**`ensayo120.py`**

- Algorithm: Random Forest Regressor  
- Activity subset: High activity (upper range)

**`ensayo121.py`**

- Algorithm: Random Forest Regressor  
- Activity subset: High activity (mid range)

**`ensayo122.py`**

- Algorithm: Random Forest Regressor  
- Activity subset: High activity (lower range)

---

## Medium-Activity Models

**`ensayo112.py`**

- Algorithm: Random Forest Regressor  
- Activity subset: Medium activity (upper-high range)

**`ensayo117.py`**

- Algorithm: Random Forest Regressor  
- Activity subset: Medium activity (upper-mid range)

**`ensayo118.py`**

- Algorithm: Random Forest Regressor  
- Activity subset: Medium activity (lower-mid range)

**`ensayo119.py`**

- Algorithm: Random Forest Regressor  
- Activity subset: Medium activity (lower-low range)

---

## Low-Activity Models

**`ensayo171.py`**

Algorithms implemented:

- Random Forest Regressor
- MLP Neural Network
- Stacking Ensemble

Stacking configuration:

- Base learners: Random Forest and MLP
- Meta-model: Linear Regression
- Cross-validation: 5-fold stacking validation

---

# Model Outputs

Each model script produces the following outputs:

- Trained model (`.joblib`)
- Feature selection masks
- Random Forest and LASSO selector objects
- Scaler and imputer objects
- Hyperparameter optimization results
- Model performance metrics (`CSV`)
- Observed vs. predicted activity plots
- Feature selection diagnostic plots (Boruta and Mutual Information)

---

# Role in the Project Pipeline

These QSAR models correspond to the **modeling stage of the project pipeline**.

---
