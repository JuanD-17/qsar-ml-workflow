# QSAR Models

## Available Models

- **Ensayo 120**
  - Algorithm: Random Forest Regressor
  - Task: Regression (antioxidant activity prediction)
  - Target variable: Log₁₀-transformed antioxidant activity (nM)
  - Category used: High-activity (upper range)
 
## Molecular Representation

Each molecule is encoded using a hybrid descriptor space composed of:

1. ECFP (Morgan fingerprints)
  - Radius: 3
  - Bit size: 512
    
2. Physicochemical Descriptors

  - Molecular weight and exact mass
  - Lipophilicity (MolLogP)
  - Topological polar surface area (TPSA)
  - Hydrogen bond donors and acceptors
  - Ring systems (aromatic and aliphatic)
  - Rotatable bonds and fraction Csp³
  - Electronic and charge-related descriptors
  - Conjugation and π-system proxies
    
3. Murcko Scaffolds
  - Extraction of Bemis–Murcko scaffolds
  - Independent calculation of:
  - Morgan fingerprints (512 bits)
  - Scaffold-level physicochemical descriptors

## Data Processing
  - SMILES strings are validated prior to descriptor calculation
  - Invalid or empty molecular structures are excluded
  - Missing values are handled using mean imputation
  - Features are standardized using z-score normalization
  - Descriptor calculation is parallelized and cached to improve efficiency

## Feature Selection
  - Boruta (Random Forest-based relevance testing)
  - Mutual Information Regression (non-linear dependency detection)
  - Random Forest Importance Thresholding (mean importance)
  - LASSO Regression (L1 regularization)
Final feature sets are obtained by combining selectors to retain:
  - Statistically relevant
  - Non-redundant
  - Predictively informative descriptors

## Model Training and Optimization
Data split:
  - 75% training
  - 25% internal validation
Hyperparameter optimization:
  - GridSearchCV (5-fold cross-validation)
Parameters tuned include:
  - Number of trees
  - Tree depth
  - Minimum samples per split and leaf
  - Feature subsampling strategy

## Model Evaluation

  - R² (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)
Evaluations are reported for:
  - Training set
  - Internal validation set
  - 5-fold cross-validation (mean ± SD)

## Outputs
Running this script generates the following artifacts in the [ensayo120/](./ensayo120/) directory:
- Trained Random Forest model (.joblib)
- Feature selection masks and selectors
- Scaler and imputer objects
- Hyperparameter optimization results
- Performance metrics (CSV)
- Feature selection plots (Boruta, Mutual Information)
- Observed vs. predicted activity plots









