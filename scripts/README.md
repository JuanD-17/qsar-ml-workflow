# README — QSAR Analysis Scripts and Derived Figures

## Overview
This folder compiles the main scripts and outputs associated with the QSAR project, including:
- applicability domain analysis,
- chemical/ontological annotation,
- molecular quality assessment,
- chemical space exploration,
- structural similarity analysis,
- activity cliffs,
- dominant scaffolds,
- external validation,
- and complementary statistical/experimental analyses.

The purpose of this README is to provide minimal traceability for each analytical block by documenting:
- main assay,
- relevant input files,
- final outputs,
- and the role of each analysis within the project.

---

## 1. Williams Plot / Applicability Domain

### Main assay
- `ensayo314`

### Main files
- `ensayo314/Williams_plot_JCIM_final.tiff`
- `ensayo314/Williams_plot_JCIM_final.pdf`
- `ensayo314/Figure_Williams_Combined_IC50_EC50_FinalReadable.tiff`
- `ensayo314/Williams_plot_leverage_values_IC50_EC50.csv`

### Function
This block documents the applicability domain of the IC50 and EC50 models using leverage and standardized residuals, allowing identification of structural outliers and evaluation of prediction reliability.

---

## 2. ChEBI

### Main assays
- `ensayo331`
- `ensayo332`

### Main files

#### `ensayo331`
- `ensayo331/chebi_annotation_antioxidant11.csv`
- `ensayo331/chebi_annotation_antioxidant19.csv`
- `ensayo331/chebi_annotation_combined.csv`
- `ensayo331/chebi_class_summary.csv`
- `ensayo331/chebi_version_metadata.txt`

#### `ensayo332`
- `ensayo332/chebi_counts_by_activity_primary.csv`
- `ensayo332/chebi_counts_percent_by_activity_primary.csv`
- `ensayo332/chebi_counts_by_activity_multilabel.csv`
- `ensayo332/chebi_counts_percent_by_activity_multilabel.csv`
- `ensayo332/chebi_counts_long_primary.csv`

### Function
This block resolves ChEBI ontological annotations and summarizes chemical classes associated with antioxidant compounds, facilitating structural and functional interpretation.

---

## 3. QED

### Main assays
- `ensayo275`
- `ensayo276`
- `ensayo279`

### Main files

#### `ensayo275`
- `ensayo275/qed_analysis_all_molecules.csv`
- `ensayo275/qed_analysis_leverage_filtered.csv`

#### `ensayo276`
- `ensayo276/qed_analysis_all_molecules.csv`

#### `ensayo279`
- `ensayo279/qed_analysis_all_molecules.csv`

### Function
The QED (Quantitative Estimate of Drug-likeness) analysis is used to prioritize molecules according to overall chemical quality. Within this block, `ensayo276` corresponds to the terpene subset included in this evaluation.

---

## 4. PCA of Chemical Space

### Main assay
- `ensayo350`

### Main files
- `ensayo350/PCA_ChemicalSpace_JCIM_Q1.tiff`
- `ensayo350/PCA_ChemicalSpace_JCIM_Q1.png`
- `ensayo350/PCA_ChemicalSpace_JCIM_Q1.pdf`
- `ensayo350/PCA_ChemicalSpace_EC50_SI_JCIM_Final.tiff`
- `ensayo350/PCA_ChemicalSpace_EC50_SI_JCIM_Final.png`
- `ensayo350/PCA_ChemicalSpace_EC50_SI_JCIM_Final.pdf`
- `ensayo350/pca_coordinates.csv`
- `ensayo350/pca_coordinates_ec50.csv`

### Function
This block visualizes the chemical space coverage of cannabinoids and terpenes, as well as their relationship with the training set.

### Important note
The same `ensayo350` folder also contains the final Tanimoto figures. Therefore, PCA and Tanimoto analyses were consolidated within the same assay.

---

## 5. Activity Cliffs

### Main assay
- `ensayo320`

### Main files
- `ensayo320/Panel_B_Activity_Cliffs_IC50_EC50_Combined.tiff`
- `ensayo320/Panel_B_Activity_Cliffs_IC50_EC50_Combined.png`
- `ensayo320/Panel_B_Activity_Cliffs_IC50_EC50_Combined.pdf`
- `ensayo320/Figure_Representative_Cliff_Pairs.tiff`
- `ensayo320/Figure_Representative_Cliff_Pairs.png`
- `ensayo320/Figure_Representative_Cliff_Pairs.pdf`

### Function
This block identifies structurally similar molecules displaying large activity differences, both in global analyses and representative structural pairs.

---

## 6. Top 5 Scaffolds

### Main assay
- `ensayo317`

### Main files
- `ensayo317/Panel_A_Top5_Scaffolds_IC50_EC50_Combined.tiff`
- `ensayo317/Panel_A_Top5_Scaffolds_IC50_EC50_Combined.png`
- `ensayo317/Panel_A_Top5_Scaffolds_IC50_EC50_Combined.pdf`
- `ensayo317/Panel_A_Top5_Scaffolds_IC50_EC50_Combined.svg`
- `ensayo317/Panel_A_Top5_Scaffolds_IC50_EC50_Vertical.tiff`
- `ensayo317/Panel_A_Top5_Scaffolds_IC50_EC50_Vertical.png`
- `ensayo317/Panel_A_Top5_Scaffolds_IC50_EC50_Vertical.pdf`
- `ensayo317/Panel_A_Top5_Scaffolds_IC50_EC50_Vertical.svg`

### Important auxiliary files
- `ensayo317/recortes`

### Function
This block summarizes the most frequent Bemis–Murcko scaffolds in IC50 and EC50 datasets, including chemical structure, frequency, and aggregated metrics.
### GitHub recommendation
For this block, it is recommended to upload the complete `ensayo317` folder, since it contains both the final figures and the image crops used for figure assembly.

Suggested structure:

```text
ensayo317/
├── images/
│   └── .gitkeep
└── recortes/
    └── .gitkeep
```

If created directly from GitHub, folders can be generated using:
- `ensayo317/images/.gitkeep`
- `ensayo317/recortes/.gitkeep`

---

## 7. Cleveland Plot

### Main assay
- `ensayo339`

### Main files
- `ensayo339/Cleveland_High_Potency_Panels_AB.tiff`
- `ensayo339/Cleveland_High_Potency_Panels_AB.png`
- `ensayo339/Cleveland_High_Potency_Panels_AB.pdf`
- `ensayo339/Natural_Antioxidant_Validation_Set_ranked.csv`

### Auxiliary tables
- `ensayo339/cleveland_high_potency_panel_a_long.csv`
- `ensayo339/cleveland_high_potency_panel_a_wide.csv`
- `ensayo339/cleveland_high_potency_panel_b_long.csv`
- `ensayo339/cleveland_high_potency_panel_b_wide.csv`

### Function
This block compares project molecules against reference natural antioxidants, highlighting compounds with high predicted potency.

---

## 8. ABTS/DPPH Heatmap

### Main assay
- `ensayo340`

### Main input file
- `pairwise_antioxidant_comparisons.csv`

### Final files
- `ensayo340/Panel_Heatmap_Dilution50.tiff`
- `ensayo340/Panel_Heatmap_Dilution50.png`
- `ensayo340/Panel_Heatmap_Dilution50.pdf`
- `ensayo340/Panel_Heatmap_Dilution50.svg`

### Key columns in the input CSV
- `dilution`
- `comparison`
- `ABTS_estimate`
- `ABTS_pvalue`
- `DPPH_estimate`
- `DPPH_pvalue`

### Function
This block summarizes pairwise comparisons of experimental antioxidant activity for ABTS and DPPH at 10 mg/mL extract concentration.

---

## 9. Mann–Whitney + Stratification

### Main assay
- `ensayo315`

### Main files
- `ensayo315/Figure1_Dataset_Characteristics.tiff`
- `ensayo315/Figure1_Dataset_Characteristics.png`
- `ensayo315/Figure1_Dataset_Characteristics.pdf`
- `ensayo315/Figure1_Dataset_Characteristics.svg`

### Supplementary files
- `ensayo315/Supplementary_QQPlot_IC50_EC50.tiff`
- `ensayo315/Supplementary_QQPlot_IC50_EC50.png`

### Function
This block integrates activity tertile stratification, IC50 vs EC50 distribution comparisons, and the Mann–Whitney U test, together with supplementary Q-Q plots.

---

## 10. Tanimoto

### Main assay
- `ensayo350`

### Main files
- `ensayo350/Figure_SX_Tanimoto.tiff`
- `ensayo350/Figure_SX_Tanimoto.png`
- `ensayo350/Figure_SX_Tanimoto.pdf`
- `ensayo350/Figure_SX_Tanimoto_EC50.tiff`
- `ensayo350/Figure_SX_Tanimoto_EC50.png`
- `ensayo350/Figure_SX_Tanimoto_EC50.pdf`

### Function
This block evaluates maximum Tanimoto similarity against the training set for cannabinoids and terpenes in both IC50 and EC50 analyses.

### Important note
The same `ensayo350` folder also contains the final PCA figures. Therefore, Tanimoto and PCA belong to the same final chemical space analysis block.

---

## 11. External Validation: Prediction vs Experimental Data

### Main assays
- `ensayo348`
- `ensayo349`

### Base input file
- `external_validation_results.csv`

This file is the direct input for `ensayo348.py`.

### Logical workflow
1. `external_validation_results.csv`
2. `ensayo348.py`
3. output from `ensayo348`: `ensayo348/external_validation_results2.csv`
4. `ensayo349.py`
5. final prediction vs experimental concordance figure

### Main files from `ensayo348`
- `ensayo348/external_validation_results2.csv`
- `ensayo348/ensayo348_summary.txt`

### Main files from `ensayo349`
- `ensayo349/IC50_EC50_comparison_panel.tiff`
- `ensayo349/IC50_EC50_comparison_panel.png`
- `ensayo349/IC50_EC50_comparison_panel.pdf`
- `ensayo349/IC50_EC50_comparison_panel.svg`

### Function
This block documents the concordance between experimental and predicted IC50 and EC50 values in the external validation set.

### GitHub recommendation
It is recommended to upload at least:
- `external_validation_results.csv`
- `ensayo348.py`
- `ensayo349.py`
- `ensayo348/external_validation_results2.csv`
- final figure files from `ensayo349`

---

## Final Note

This README consolidates the main QSAR analytical blocks and their derived figures. It is intended to accompany a `Scripts` folder in GitHub and facilitate traceability between scripts, intermediate files, and final outputs.
