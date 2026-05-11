"""
Ensayo 331 - Clasificacion ChEBI por estructura (SMILES) via API

Entrada:
- antioxidant11.csv
- antioxidant19.csv

Salida (ensayo331/):
- chebi_annotation_antioxidant11.csv
- chebi_annotation_antioxidant19.csv
- chebi_annotation_combined.csv
- chebi_class_summary.csv
- unresolved_structures.csv
- cache_unique_antioxidant11.csv
- cache_unique_antioxidant19.csv
- ensayo331_log.txt

Notas:
- Requiere internet para APIs de ChEMBL/UniChem/OLS.
- Reanudable: si se interrumpe, relanzar y continua desde cache.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import List, Optional, Tuple
from urllib.parse import quote

import pandas as pd
import requests

try:
    from rdkit import Chem  # type: ignore
except Exception:
    Chem = None


INPUT_FILES = ["antioxidant11.csv", "antioxidant19.csv"]
OUTPUT_DIR = "ensayo331"
REQUEST_TIMEOUT = 25
REQUEST_SLEEP = 0.05
MAX_RETRIES = 5
RETRY_BACKOFF = 1.6
SAVE_EVERY = 100

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
UNICHEM_INCHIKEY_API = "https://www.ebi.ac.uk/unichem/rest/inchikey/{inchikey}"
OLS_TERM_API = "https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms/{encoded_iri}"
OLS_ANCESTORS_API = "https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms/{encoded_iri}/ancestors?size=1000"
OLS_CHEBI_ONTOLOGY_API = "https://www.ebi.ac.uk/ols4/api/ontologies/chebi"

UNICHEM_CHEBI_SRC_IDS = {7, "7"}

CLASS_RULES = {
    "Phenolic compounds": ["phenol", "polyphenol", "phenolic"],
    "Flavonoids": ["flavonoid", "flavone", "flavonol", "flavan", "isoflav"],
    "Terpenoids": ["terpene", "terpenoid", "sesquiterpene", "monoterpene", "diterpene", "triterpene"],
    "Alkaloids": ["alkaloid"],
    "Glycosides": ["glycoside", "glycosidic"],
    "Steroids": ["steroid", "sterol"],
    "Lipids/Fatty derivatives": ["fatty acid", "lipid", "acyl", "glyceride"],
    "Quinones": ["quinone"],
    "Aromatic compounds": ["aromatic compound", "benzene", "aryl"],
    "Heterocyclic compounds": ["heterocyclic", "heterocycle"],
}

PRIMARY_CLASS_ORDER = [
    "Flavonoids",
    "Phenolic compounds",
    "Terpenoids",
    "Alkaloids",
    "Glycosides",
    "Steroids",
    "Quinones",
    "Lipids/Fatty derivatives",
    "Aromatic compounds",
    "Heterocyclic compounds",
]


def log(msg: str, fh=None) -> None:
    print(msg)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


def detect_smiles_col(df: pd.DataFrame) -> str:
    for c in ["Smiles", "SMILES", "smiles", "canonical_smiles"]:
        if c in df.columns:
            return c
    raise ValueError("No se encontro columna de SMILES.")


def normalize_chebi_id(raw_id: Optional[str]) -> Optional[str]:
    if raw_id is None:
        return None
    txt = str(raw_id).strip()
    if not txt:
        return None
    m = re.search(r"(\d+)", txt)
    if not m:
        return None
    return f"CHEBI:{m.group(1)}"


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    if Chem is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def encode_ols_iri(chebi_id: str) -> str:
    iri = f"http://purl.obolibrary.org/obo/{chebi_id.replace(':', '_')}"
    return quote(quote(iri, safe=""), safe="")


def safe_get_json(session: requests.Session, url: str) -> Tuple[Optional[dict], Optional[str]]:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json(), None
            if resp.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}"
                time.sleep((RETRY_BACKOFF ** attempt) + 0.2)
                continue
            return None, f"HTTP {resp.status_code}"
        except requests.exceptions.Timeout as e:
            last_err = f"Timeout: {e}"
        except requests.exceptions.ConnectionError as e:
            last_err = f"ConnectionError: {e}"
        except Exception as e:
            return None, str(e)

        time.sleep((RETRY_BACKOFF ** attempt) + 0.2)
    return None, (last_err or "request failed")


def get_chebi_ontology_version(session: requests.Session) -> Tuple[str, str]:
    """
    Devuelve (version, source_url) de la ontologia ChEBI en OLS.
    Si no se puede resolver, retorna 'unknown'.
    """
    data, err = safe_get_json(session, OLS_CHEBI_ONTOLOGY_API)
    if err or data is None:
        return "unknown", OLS_CHEBI_ONTOLOGY_API

    candidates = [
        data.get("loaded"),
        data.get("updated"),
        data.get("config", {}).get("version"),
        data.get("version"),
        data.get("ontologyId"),
    ]
    for c in candidates:
        if c is not None and str(c).strip():
            return str(c).strip(), OLS_CHEBI_ONTOLOGY_API
    return "unknown", OLS_CHEBI_ONTOLOGY_API


def resolve_chebi_from_chembl(session: requests.Session, chembl_id: Optional[str]) -> Optional[str]:
    if not chembl_id:
        return None
    chembl_id = str(chembl_id).strip()
    if not chembl_id or chembl_id.lower() == "nan":
        return None
    data, err = safe_get_json(session, CHEMBL_API.format(chembl_id=chembl_id))
    if err or data is None:
        return None
    return normalize_chebi_id(data.get("chebi_par_id"))


def resolve_chebi_from_inchikey(session: requests.Session, inchikey: Optional[str]) -> Optional[str]:
    if not inchikey:
        return None
    data, err = safe_get_json(session, UNICHEM_INCHIKEY_API.format(inchikey=inchikey))
    if err or data is None or not isinstance(data, list):
        return None
    for row in data:
        if row.get("src_id") in UNICHEM_CHEBI_SRC_IDS:
            chebi_id = normalize_chebi_id(row.get("src_compound_id") or row.get("assignment"))
            if chebi_id:
                return chebi_id
    return None


def get_chebi_label_ancestors(session: requests.Session, chebi_id: str) -> Tuple[Optional[str], List[str], Optional[str]]:
    encoded_iri = encode_ols_iri(chebi_id)
    term_data, term_err = safe_get_json(session, OLS_TERM_API.format(encoded_iri=encoded_iri))
    if term_err or term_data is None:
        return None, [], term_err

    label = term_data.get("label")
    anc_data, anc_err = safe_get_json(session, OLS_ANCESTORS_API.format(encoded_iri=encoded_iri))
    if anc_err or anc_data is None:
        return label, [], anc_err

    ancestors = []
    for t in anc_data.get("_embedded", {}).get("terms", []):
        lbl = t.get("label")
        if lbl:
            ancestors.append(str(lbl))
    return label, sorted(set(ancestors)), None


def classify_from_chebi(ancestors: List[str], label: Optional[str]) -> List[str]:
    text = " | ".join([*(ancestors or []), label or ""]).lower()
    matched = [cls for cls, kws in CLASS_RULES.items() if any(k in text for k in kws)]
    return matched if matched else ["Unclassified"]


def pick_primary_class(classes: List[str]) -> str:
    for cls in PRIMARY_CLASS_ORDER:
        if cls in classes:
            return cls
    return classes[0] if classes else "Unclassified"


def _cache_key(smiles: str, chembl_id: str) -> str:
    return f"{smiles}||{chembl_id}"


def annotate_unique_structures(df: pd.DataFrame, smiles_col: str, cache_path: str, log_fh=None) -> pd.DataFrame:
    session = requests.Session()
    uniques = (
        df[[smiles_col, "Molecule ChEMBL ID"]]
        .dropna(subset=[smiles_col])
        .drop_duplicates(subset=[smiles_col])
        .reset_index(drop=True)
    )

    if os.path.exists(cache_path):
        cached_df = pd.read_csv(cache_path, low_memory=False)
        log(f"  Cache detectado: {cache_path} ({len(cached_df):,} filas)", log_fh)
    else:
        cached_df = pd.DataFrame()

    rows = cached_df.to_dict(orient="records") if len(cached_df) else []
    done = set()
    if len(cached_df):
        for _, rr in cached_df.iterrows():
            done.add(_cache_key(str(rr.get(smiles_col, "")).strip(), str(rr.get("Molecule ChEMBL ID", "") or "").strip()))

    total = len(uniques)
    processed_new = 0
    for i, r in uniques.iterrows():
        smiles = str(r[smiles_col]).strip()
        chembl_id = str(r.get("Molecule ChEMBL ID", "")).strip()
        if chembl_id.lower() == "nan":
            chembl_id = ""

        key = _cache_key(smiles, chembl_id)
        if key in done:
            continue

        if i % 300 == 0:
            log(f"  Procesando estructuras unicas: {i:,}/{total:,}", log_fh)

        inchikey = smiles_to_inchikey(smiles)
        chebi_id = resolve_chebi_from_chembl(session, chembl_id)
        source = "chembl_id"
        if chebi_id is None:
            chebi_id = resolve_chebi_from_inchikey(session, inchikey)
            source = "inchikey_unichem" if chebi_id else "unresolved"

        chebi_label, ancestors, err = None, [], None
        if chebi_id is not None:
            chebi_label, ancestors, err = get_chebi_label_ancestors(session, chebi_id)
            if err:
                source = f"{source}_with_ols_error"
        else:
            err = "CHEBI ID not resolved"

        classes = classify_from_chebi(ancestors, chebi_label)
        primary_class = pick_primary_class(classes)

        rows.append(
            {
                smiles_col: smiles,
                "Molecule ChEMBL ID": chembl_id if chembl_id else None,
                "InChIKey": inchikey,
                "CHEBI_ID": chebi_id,
                "CHEBI_Label": chebi_label,
                "CHEBI_Ancestors_Count": len(ancestors),
                "CHEBI_Ancestors": " | ".join(ancestors) if ancestors else None,
                "CHEBI_Class_List": " | ".join(classes),
                "CHEBI_Primary_Class": primary_class,
                "Resolution_Source": source,
                "Resolution_Error": err,
            }
        )

        done.add(key)
        processed_new += 1
        if processed_new % SAVE_EVERY == 0:
            pd.DataFrame(rows).to_csv(cache_path, index=False)
            log(f"    checkpoint guardado ({processed_new:,} nuevas)", log_fh)

        time.sleep(REQUEST_SLEEP)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(cache_path, index=False)
    return out_df


def save_summary(combined_df: pd.DataFrame) -> None:
    summary = (
        combined_df.groupby("CHEBI_Primary_Class", dropna=False)
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    summary["Percent"] = (summary["Count"] / max(len(combined_df), 1) * 100.0).round(2)
    summary.to_csv(os.path.join(OUTPUT_DIR, "chebi_class_summary.csv"), index=False)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "ensayo331_log.txt")
    metadata_path = os.path.join(OUTPUT_DIR, "chebi_version_metadata.txt")

    with open(log_path, "w", encoding="utf-8") as log_fh:
        metadata_session = requests.Session()
        chebi_version, chebi_version_source = get_chebi_ontology_version(metadata_session)

        log("=" * 78, log_fh)
        log("ENSAYO 331 - CLASIFICACION ChEBI POR ESTRUCTURA (SMILES)", log_fh)
        log("=" * 78, log_fh)
        log(f"ChEBI ontology version (OLS): {chebi_version}", log_fh)
        log(f"ChEBI metadata source: {chebi_version_source}", log_fh)

        per_file_annotated = []
        for filename in INPUT_FILES:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"No se encontro: {filename}")

            log(f"\n[1/3] Cargando {filename}", log_fh)
            df = pd.read_csv(filename, low_memory=False)
            smiles_col = detect_smiles_col(df)
            log(f"  Filas: {len(df):,} | Columna SMILES: '{smiles_col}'", log_fh)

            base = os.path.splitext(os.path.basename(filename))[0]
            cache_path = os.path.join(OUTPUT_DIR, f"cache_unique_{base}.csv")
            log(f"[2/3] Resolviendo ChEBI para estructuras unicas ({base})", log_fh)
            annotation_df = annotate_unique_structures(df, smiles_col, cache_path, log_fh)
            log(f"  Estructuras unicas anotadas: {len(annotation_df):,}", log_fh)

            unresolved = annotation_df[annotation_df["CHEBI_ID"].isna()]
            log(f"  Sin resolver ChEBI: {len(unresolved):,}", log_fh)

            log(f"[3/3] Uniendo anotaciones en {filename}", log_fh)
            annotated_df = df.merge(annotation_df, on=[smiles_col, "Molecule ChEMBL ID"], how="left")
            annotated_df["Source_File"] = filename
            annotated_df["CHEBI_Ontology_Version"] = chebi_version
            out_file = os.path.join(OUTPUT_DIR, f"chebi_annotation_{base}.csv")
            annotated_df.to_csv(out_file, index=False)
            log(f"  ✓ Guardado: {out_file}", log_fh)
            per_file_annotated.append(annotated_df)

        combined_df = pd.concat(per_file_annotated, ignore_index=True)
        combined_out = os.path.join(OUTPUT_DIR, "chebi_annotation_combined.csv")
        combined_df.to_csv(combined_out, index=False)
        log(f"\n✓ Guardado combinado: {combined_out}", log_fh)

        unresolved_out = os.path.join(OUTPUT_DIR, "unresolved_structures.csv")
        combined_df[combined_df["CHEBI_ID"].isna()].to_csv(unresolved_out, index=False)
        log(f"✓ Guardado no resueltos: {unresolved_out}", log_fh)

        save_summary(combined_df)
        log(f"✓ Guardado resumen: {os.path.join(OUTPUT_DIR, 'chebi_class_summary.csv')}", log_fh)

        with open(metadata_path, "w", encoding="utf-8") as mf:
            mf.write("ENSAYO 331 - ChEBI Metadata\n")
            mf.write(f"CHEBI_Ontology_Version: {chebi_version}\n")
            mf.write(f"Metadata_Source: {chebi_version_source}\n")
        log(f"✓ Guardado metadata ChEBI: {metadata_path}", log_fh)

        log("\nProceso completado.", log_fh)
        log("=" * 78, log_fh)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario (KeyboardInterrupt). Puedes reanudar ejecutando de nuevo.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR en ensayo331: {e}")
        sys.exit(1)
