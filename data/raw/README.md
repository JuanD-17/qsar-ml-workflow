# Raw datasets

This folder contains the original consolidated datasets obtained from ChEMBL,
PubChem and literature sources before strict QSAR curation.

These datasets may contain mixed assay types, units, and relation operators.

---

## Dataset: `antioxidant8.csv`

### Overview
`antioxidant8.csv` is a bioactivity dataset focused on antioxidant-related
measurements (including IC50 and EC50) for small molecules.

- File format: CSV (semicolon-separated `;`)
- Rows: 24,394
- Columns: 49
- File size: ~8.9 MB
- Source path in project: `data/raw/antioxidant8.csv`

### Main content
Each row corresponds to one activity record and includes:

**Molecule identifiers and structure**
- Molecule ChEMBL ID
- Molecule Name
- SMILES
- Molecular Weight
- AlogP
- #RO5 Violations

**Bioactivity fields**
- Standard Type
- Standard Relation
- Standard Value
- Standard Units
- pChEMBL Value

**Assay metadata**
- Assay ChEMBL ID
- Assay Description
- Assay Type

**Target metadata**
- Target ChEMBL ID
- Target Name
- Target Organism
- Target Type

**Document metadata**
- Document ChEMBL ID
- Document Journal
- Document Year
- Source Description
