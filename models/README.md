# QSAR Models

## Available Models

**Ensayo 120**
  - Algorithm: Random Forest Regressor
  - Task: Regression (antioxidant activity prediction)
  - Target variable: Log₁₀-transformed antioxidant activity (nM)
  - Category used: High-activity (upper range)
 
## Molecular Representation
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
- Trained Random Forest model (.joblib)
- Feature selection masks and selectors
- Scaler and imputer objects
- Hyperparameter optimization results
- Performance metrics (CSV)
- Feature selection plots (Boruta, Mutual Information)
- Observed vs. predicted activity plots

**Ensayo 121**
- Algorithm: Random Forest Regressor
- Task: Regression (antioxidant activity prediction)
- Target variable: Log₁₀-transformed antioxidant activity (nM)
- Category used: High-activity (mid range)

## Molecular Representation
1. ECFP (Morgan Fingerprints)
  - Radius: 3
  - Bit size: 512

2. Physicochemical Descriptors

  - Molecular weight and exact molecular weight
  - Lipophilicity (MolLogP)
  - Topological polar surface area (TPSA)
  - Heavy atom count
  - Hydrogen bond donors and acceptors
  - Total, aromatic, and aliphatic ring counts
  - Rotatable bonds and fraction Csp³
  - Aromaticity index
  - Dipole moment proxy
  - Charge distribution and partial charge descriptors
  - Conjugated π-system proxies
  - Radical electron counts
  - Bertz complexity index
  - Kier–Hall connectivity indices
  - Molar refractivity
  - Oxygen atom counts and oxygen-involving bonds
  - Formal charge-related descriptors

3. Murcko Scaffolds
  - Morgan fingerprints (512 bits) 
Scaffold-level descriptors are calculated, including:
  - Molecular weight
  - Heavy atom count
  - Ring count
  - Aromatic ring count
  - TPSA

## Data Processing
  - SMILES strings are validated and filtered to ensure chemical correctness
  - Invalid or empty molecular structures are excluded
  - Descriptor calculation is parallelized using multi-core processing
  - Descriptor results are cached to disk to avoid redundant computations
  - Missing values are handled using mean imputation
  - Features are standardized using z-score normalization
  - NaN and infinite values are replaced with zero

## Feature Selection
  - Boruta: Random Forest-based relevance testing
  - Mutual Information Regression: Non-linear dependency detection
  - Random Forest Importance Thresholding: Mean importance filtering
  - LASSO Regression (L1 regularization): Sparse feature selection

## Model Training and Optimization
Data Split
  - 75% training set
  - 25% internal validation set
Hyperparameter Optimization
  - GridSearchCV with 5-fold cross-validation
Optimized parameters include:
  - Number of trees (n_estimators)
  - Maximum tree depth
  - Minimum samples per split
  - Minimum samples per leaf
  - Feature subsampling strategy

## Model Evaluation
  - R² (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)
Metrics are reported for:
  - Training set
  - Internal validation set
  - 5-fold cross-validation (mean ± standard deviation)

## Outputs
  - Trained Random Forest model (.joblib)
  - Selected feature masks
  - Boruta and Mutual Information selection plots
  - Random Forest and LASSO selector objects
  - Scaler and imputer objects
  - Hyperparameter optimization results
  - Model performance metrics (CSV)
  - Observed vs. predicted antioxidant activity plots

**Ensayo 122**
  - Algorithm: Random Forest Regressor
  - Task: Regression (antioxidant activity prediction)
  - Target variable: Log₁₀-transformed antioxidant activity (nM)
  - Category used: High-activity (low range)

## Molecular Representation
1. ECFP (Morgan Fingerprints)
  - Radius: 3
  - Bit size: 512

2. Physicochemical Descriptors
  - Molecular weight and exact molecular weight
  - Lipophilicity (MolLogP)
  - Topological polar surface area (TPSA)
  - Heavy atom count
  - Total, aromatic, and aliphatic ring counts
  - Hydrogen bond donors and acceptors
  - Number of rotatable bonds and fraction Csp³
  - Aromaticity index
  - Dipole moment proxy
  - Charge distribution descriptors
  - Partial atomic charge proxies
  - Conjugated π-system descriptors
  - Radical electron counts
  - Bertz molecular complexity index
  - Kier–Hall connectivity indices
  - Molar refractivity
  - Oxygen atom counts and oxygen-involving bonds
  - Formal charge-related descriptors

3. Murcko Scaffolds
  - Morgan fingerprints (512 bits) 
Scaffold-level physicochemical descriptors are calculated, including:
  - Molecular weight
  - Heavy atom count
  - Ring count
  - Aromatic ring count
  - TPSA

## Data Processing
  - SMILES strings are validated prior to descriptor calculation
  - Invalid or empty molecular structures are excluded
  - Descriptor calculation is parallelized using multi-core processing
  - Descriptor values are cached to avoid redundant computations
  - Missing values are handled using mean imputation
  - Features are standardized using z-score normalization
  - NaN and infinite values are replaced with zero

## Feature Selection
  - Boruta: Random Forest-based relevance testing
  - Mutual Information Regression: Detection of non-linear dependencies
  - Random Forest Importance Thresholding: Mean importance filtering
  - LASSO Regression: L1-regularized feature selection
Final feature sets are obtained by combining selection masks to retain descriptors that are:
  - Statistically relevant
  - Non-redundant
  - Predictively informative

## Model Training and Optimization
Data Split
  - 75% training set
  - 25% internal validation set
Hyperparameter Optimization
  - GridSearchCV with 5-fold cross-validation
Optimized parameters include:
  - Number of trees (n_estimators)
  - Maximum tree depth
  - Minimum samples per split
  - Minimum samples per leaf
  - Feature subsampling strategy

## Model Evaluation
  - R² (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)
Evaluations are reported for:
  - Training set
  - Internal validation set
  - 5-fold cross-validation (mean ± standard deviation)

## Outputs
  - Trained Random Forest model (.joblib)
  - Feature selection masks
  - Boruta feature selection plots
  - Mutual Information score plots
  - Random Forest and LASSO selector objects
  - Scaler and imputer objects
  - Hyperparameter optimization results
  - Model performance metrics (CSV)
  - Observed vs. predicted antioxidant activity plots

**Ensayo 112**

Algorithm: Random Forest Regressor
Task: Regression (antioxidant activity prediction)
Target variable: Log₁₀-transformed antioxidant activity (nM)
Category used: Medium-activity (upper-high range)

## Molecular Representation

1. ECFP (Morgan Fingerprints)
Radius: 3
Bit size: 512

2. Physicochemical Descriptors
  - Molecular weight
  - Lipophilicity (MolLogP)
  - Topological polar surface area (TPSA)
  - Heavy atom count
  - Total ring count
  - Aromatic ring count
  - Hydrogen bond donors
  - Hydrogen bond acceptors
  - Number of rotatable bonds
  - Fraction Csp³

3. Murcko Scaffolds
  - Extraction of Bemis–Murcko scaffolds
  - Morgan fingerprints (512 bits) calculated at the scaffold level
Additionally, scaffold-level physicochemical descriptors are calculated, including:
  - Scaffold molecular weight
  - Scaffold heavy atom count
  - Scaffold ring count
  - Scaffold aromatic ring count
  - Scaffold TPSA

## Data Processing
  - SMILES strings are validated prior to descriptor calculation
  - Invalid or empty molecular structures are excluded
  - Descriptor calculation is parallelized using multi-core processing
  - Descriptor values are cached to improve computational efficiency
  - Missing values are handled using mean imputation
  - Features are standardized using z-score normalization
  - NaN and infinite values are replaced with zero to ensure numerical stability

## Feature Selection
  - Boruta: Random Forest-based relevance testing
  - Mutual Information Regression: Detection of non-linear dependencies between descriptors and activity
  - Random Forest Importance Thresholding: Mean feature importance filtering
  - LASSO Regression: L1-regularized feature selection
Final feature sets are obtained by combining selection masks to retain descriptors that are:
  - Statistically relevant
  - Non-redundant
  - Predictively informative

## Model Training and Optimization
Data Split
  - 75% training set
  - 25% internal validation set
Hyperparameter Optimization
  - GridSearchCV with 5-fold cross-validation
Optimized parameters include:
  - Number of trees (n_estimators)
  - Maximum tree depth
  - Minimum samples per split
  - Minimum samples per leaf
  - Feature subsampling strategy (max_features)

## Model Evaluation
  - R² (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)
Evaluations are reported for:
  - Training set
  - Internal validation set
  - 5-fold cross-validation (mean ± standard deviation)

## Outputs
  - Trained Random Forest model (.joblib)
  - Feature selection masks
  - Boruta feature selection plots
  - Mutual Information score plots
  - Random Forest and LASSO selector objects
  - Scaler and imputer objects
  - Hyperparameter optimization results
  - Model performance metrics (CSV)
  - Observed vs. predicted antioxidant activity plots

**Ensayo 117**
Algorithm: Random Forest Regressor
Task: Regression (antioxidant activity prediction)
Target variable: Log₁₀-transformed antioxidant activity (nM)
Category used: Medium-activity (upper-mid range)

## Molecular Representation
1. ECFP (Morgan Fingerprints)
  - Radius: 3
  - Bit size: 512

2. Physicochemical Descriptors
  - Molecular weight
  - Lipophilicity (MolLogP)
  - Topological polar surface area (TPSA)
  - Heavy atom count
  - Total ring count
  - Aromatic ring count
  - Hydrogen bond donors
  - Hydrogen bond acceptors
  - Number of rotatable bonds
  - Fraction Csp³

3. Murcko Scaffolds
  - Bemis–Murcko scaffolds are extracted from each molecule
  - Morgan fingerprints (512 bits) are computed at the scaffold level
Additionally, scaffold-specific descriptors are calculated:
  - Scaffold molecular weight
  - Scaffold heavy atom count
  - Scaffold ring count
  - Scaffold aromatic ring count
  - Scaffold TPSA

## Data Processing
  - SMILES strings are validated prior to descriptor calculation
  - Invalid or empty molecular structures are excluded
  - Descriptor calculation is parallelized using multi-core processing
  - Descriptor values are cached to reduce redundant computation
  - Missing values are imputed using the mean strategy
  - All descriptors are standardized using z-score normalization
  - NaN and infinite values are replaced with zero to ensure numerical robustness

## Feature Selection
  - Boruta: Random Forest-based all-relevant feature selection
  - Mutual Information Regression: Identification of non-linear dependencies
  - Random Forest Importance Filtering: Mean importance thresholding
  - LASSO Regression: L1-regularized feature selection
Final descriptor subsets are generated by combining selection masks, ensuring:
  - High relevance
  - Reduced redundancy
  - Improved generalization capability

## Model Training and Optimization
Data Splitting
  - 75% training set
  - 25% internal validation set
Hyperparameter Tuning
  - GridSearchCV with 5-fold cross-validation
Optimized parameters include:
  - Number of trees (n_estimators)
  - Maximum tree depth
  - Minimum samples per split
  - Minimum samples per leaf
  - Feature subsampling strategy (max_features)

## Model Evaluation
  - R² (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)
Metrics are reported for:
  - Training set
  - Internal validation set
  - 5-fold cross-validation (mean ± standard deviation)

## Outputs
  - Trained Random Forest model (.joblib)
  - Selected feature masks
  - Boruta feature selection plots
  - Mutual Information score plots
  - Random Forest and LASSO selector objects
  - Scaler and imputer objects
  - Hyperparameter optimization results
  - Model performance summary (CSV)
  - Observed vs. predicted QSAR plots

**Ensayo 118**
  - Algorithm: Random Forest Regressor
  - Task: Regression (antioxidant activity prediction)
  - Target variable: Log₁₀-transformed antioxidant activity (nM)
  - Category used: Medium-activity (lower-mid range)

## Molecular Representation
1. ECFP (Morgan Fingerprints)
  - Radius: 3
  - Bit size: 512

2. Physicochemical Descriptors
  - Molecular weight
  - Lipophilicity (MolLogP)
  - Topological polar surface area (TPSA)
  - Heavy atom count
  - Total ring count
  - Aromatic ring count
  - Hydrogen bond donors
  - Hydrogen bond acceptors
  - Number of rotatable bonds
  - Fraction Csp³

3. Murcko Scaffolds
  - Extraction of Bemis–Murcko scaffolds
  - Morgan fingerprints (512 bits) computed at the scaffold level
Additionally, scaffold-level physicochemical descriptors are calculated, including:
  - Scaffold molecular weight
  - Scaffold heavy atom count
  - Scaffold ring count
  - Scaffold aromatic ring count
  - Scaffold TPSA

## Data Processing
  - SMILES strings are validated prior to descriptor calculation
  - Invalid or empty molecular structures are excluded
  - Descriptor calculation is parallelized using multi-core processing
  - Descriptor values are cached to reduce redundant computations
  - Missing values are handled using mean imputation
  - Features are standardized using z-score normalization
  - NaN and infinite values are replaced with zero to ensure numerical stability

## Feature Selection
  - Boruta: Random Forest-based all-relevant feature selection
  - Mutual Information Regression: Identification of non-linear dependencies
  - Random Forest Importance Thresholding: Mean importance filtering via SelectFromModel
  - LASSO Regression: L1-regularized feature selection
Final feature sets are obtained by combining selection masks to retain descriptors that are:
  - Statistically relevant
  - Non-redundant
  - Predictively informative

## Model Training and Optimization
Data Split
  - 75% training set
  - 25% internal validation set
Hyperparameter Optimization
  - GridSearchCV with 5-fold cross-validation
Optimized hyperparameters include:
  - Number of trees (n_estimators)
  - Maximum tree depth
  - Minimum samples per split
  - Minimum samples per leaf
  - Feature subsampling strategy (max_features)

## Model Evaluation
  - R² (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)
Evaluations are reported for:
  - Training set
  - Internal validation set
  - 5-fold cross-validation (mean ± standard deviation)

## Outputs
  - Trained Random Forest model (.joblib)
  - Combined feature selection masks
  - Boruta feature selection plots
  - Mutual Information score plots
  - Random Forest (SelectFromModel) selector objects
  - LASSO selector objects
  - Scaler and imputer objects
  - Hyperparameter optimization results
  - Model performance metrics (CSV)
  - Observed vs. predicted antioxidant activity plots

**Ensayo 119**
Algorithm: Random Forest Regressor
Task: Regression (antioxidant activity prediction)
Target variable: Log₁₀-transformed antioxidant activity (nM)
Category used: Medium-activity (lower-low range)

## Molecular Representation
1. ECFP (Morgan Fingerprints)
  - Radius: 3
  - Bit size: 512

2. Physicochemical Descriptors
  - Molecular weight (MolWt)
  - Lipophilicity (MolLogP)
  - Topological polar surface area (TPSA)
  - Heavy atom count
  - Total ring count
  - Aromatic ring count
  - Hydrogen bond donors
  - Hydrogen bond acceptors
  - Number of rotatable bonds
  - Fraction Csp³

3. Murcko Scaffolds
  - Extraction of Bemis–Murcko scaffolds 
  - Morgan fingerprints (512 bits) computed at the scaffold level
Additionally, scaffold-level physicochemical descriptors are calculated, including:
  - Scaffold molecular weight
  - Scaffold heavy atom count
  - Scaffold ring count
  - Scaffold aromatic ring count
  - Scaffold TPSA

## Data Processing
  - SMILES strings are validated prior to descriptor calculation using RDKit
  - Invalid or empty molecular structures are excluded from the analysis
  - Descriptor calculation is parallelized using multi-core processing (ThreadPoolExecutor)
  - Descriptor values are cached using joblib to reduce redundant computations
  - Missing values are handled using mean imputation
  - Features are standardized using z-score normalization (StandardScaler)
  - NaN and infinite values are replaced with zero to ensure numerical stability

## Feature Selection
  - Boruta: Random Forest-based all-relevant feature selection
  - Mutual Information Regression: Identification of non-linear dependencies between descriptors and activity
  - Random Forest Importance Thresholding: Mean-importance filtering via SelectFromModel
  - LASSO Regression: L1-regularized feature selection using LassoCV
Final feature sets are obtained by combining selection masks to retain descriptors that are:
  - Statistically relevant
  - Non-redundant
  - Predictively informative

## Model Training and Optimization
Data Split
  - 75% training set
  - 25% internal validation set
Hyperparameter Optimization
  - GridSearchCV with 5-fold cross-validation
Optimized hyperparameters include:
  - Number of trees (n_estimators)
  - Maximum tree depth
  - Minimum samples per split
  - Minimum samples per leaf
  - Feature subsampling strategy (max_features)

## Model Evaluation
  - R² (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)
Evaluations are reported for:
  - Training set
  - Internal validation set
  - 5-fold cross-validation (mean ± standard deviation)

## Outputs
  - Trained Random Forest model (.joblib)
  - Combined feature selection masks
  - Boruta feature selection plots
  - Mutual Information score plots
  - Random Forest (SelectFromModel) selector objects
  - LASSO selector objects
  - Scaler and imputer objects
  - Hyperparameter optimization results
  - Model performance metrics (CSV)
  - Observed vs. predicted antioxidant activity plots

**Ensayo 171**
  - Algorithms: Random Forest Regressor, MLP Regressor, Stacking Ensemble
  - Task: Regression (antioxidant activity prediction)
  - Target variable: Log₁₀-transformed antioxidant activity (nM)
  - Category used: Low-activity (single subset)

## Molecular Representation
1. ECFP (Morgan Fingerprints)
  - Radius: 3
  - Bit size: 512

2. Physicochemical Descriptors
  - Molecular weight (MolWt)
  - Lipophilicity (MolLogP)
  - Topological polar surface area (TPSA)
  - Heavy atom count
  - Total ring count
  - Aromatic ring count
  - Hydrogen bond donors
  - Hydrogen bond acceptors
  - Number of rotatable bonds
  - Fraction Csp³

3. Murcko Scaffolds
  - Extraction of Bemis–Murcko scaffolds from each molecule
  - Morgan fingerprints (512 bits) computed at the scaffold level
Additionally, scaffold-level physicochemical descriptors are calculated, including:
  - Scaffold molecular weight
  - Scaffold heavy atom count
  - Scaffold ring count
  - Scaffold aromatic ring count
  - Scaffold TPSA

## Data Processing
  - SMILES strings are validated prior to descriptor calculation using RDKit
  - Invalid or empty molecular structures are excluded from the analysis
  - Descriptor calculation is parallelized using multi-core processing (ThreadPoolExecutor)
  - Descriptor values are cached using joblib to reduce redundant computations
  - Missing values are handled using mean imputation
  - Features are standardized using z-score normalization (StandardScaler)
  - NaN and infinite values are replaced with zero to ensure numerical stability

## Feature Selection
  - Boruta: Random Forest-based all-relevant feature selection
  - Mutual Information Regression: Identification of non-linear dependencies between descriptors and activity
  - Random Forest Importance Thresholding: Mean-importance filtering via SelectFromModel
  - LASSO Regression: L1-regularized feature selection using LassoCV
Final feature sets are obtained by combining selection masks to retain descriptors that are:
  - Statistically relevant
  - Non-redundant
  - Predictively informative

## Model Training and Optimization
Data Split
  - 75% training set
  - 25% internal validation set
Hyperparameter Optimization
  - GridSearchCV with 5-fold cross-validation
Optimized hyperparameters include:

    ## Random Forest
  - Number of trees (n_estimators)
  - Maximum tree depth
  - Minimum samples per split
  - Minimum samples per leaf
  - Feature subsampling strategy (max_features)

 ## MLP Neural Network
  - Hidden layer architectures
  - Activation functions
  - Regularization strength (alpha)
  - Solver type
  - Learning rate
  - Maximum iterations
  - Early stopping parameters
    
 ## Stacking Ensemble
  - Base learners: Random Forest and MLP
  - Meta-model: Linear Regression
  - 5-fold cross-validation for stacking

## Model Evaluation
  - R² (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)
Evaluations are reported for:
  - Training set
  - Internal validation set
  - 5-fold cross-validation (mean ± standard deviation)
Metrics are computed independently for:
  - Random Forest model
  - MLP Neural Network model
  - Stacking ensemble model

## Outputs
  - Trained Random Forest model (.joblib)
  - Trained MLP Neural Network model (.joblib)
  - Trained Stacking Ensemble model (.joblib)
  - Combined feature selection masks
  - Boruta feature selection plots
  - Mutual Information score plots
  - Random Forest (SelectFromModel) selector objects
  - LASSO selector objects
  - Scaler and imputer objects
  - Hyperparameter optimization results for Random Forest and MLP
  - Model performance comparison table (CSV)
Observed vs. predicted antioxidant activity plots for:
  - Random Forest
  - MLP Neural Network
  - Stacking Ensemble







