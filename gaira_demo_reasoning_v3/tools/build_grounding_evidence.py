"""Build the real per-axis grounding EVIDENCE table for demo v2.

Replaces the hardcoded per-axis family counts inherited from v1 with a table
derived from the actual grounding registries. Deliberately conservative:

  * counts UNIQUE reference analytes, not duplicated files or augmented spectra
  * keeps measured spectra separate from literature papers
  * never infers a molecular assignment from a nearby peak
  * where the legacy 8-axis grounding cannot be resolved to a single v11 child
    axis, the per-axis analyte count is NA (not 0), with the shared pool size
    recorded in unmapped_records + mapping_notes

Inputs:
  * <v2>/data/legacy/grounding_molecule_bsv.csv   (bundled; 202 RamanBioLib refs, 8-axis dominant)
  * <GAIRA_DATA>/…/grounding_backbone_v1/tables/grounding_peak_support_summary.csv  (measured counts)
  * …/warehouse_source_registry.csv               (source families)  — for the corpus summary only

Outputs (under <v2>/data/generated/):
  * per_axis_grounding_evidence.csv
  * grounding_corpus_summary.json
Re-run: `python tools/build_grounding_evidence.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_ROOT))
from gaira_core import config as cfg          # noqa: E402  (path constants only)
from gaira_core import paths as gpaths        # noqa: E402

OUT_DIR = DEMO_ROOT / "data" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Legacy 8-axis → v11 mapping (mirror of cfg.LEGACY8_TO_V11), with resolvability.
LEGACY8_TO_V11 = cfg.LEGACY8_TO_V11
# Reverse: for each v11 axis, which legacy axis feeds it and is it a 1:1 map?
V11_FROM_LEGACY: dict[str, tuple[str, bool]] = {}
for legacy, children in LEGACY8_TO_V11.items():
    resolvable = len(children) == 1          # 1:1 → resolvable; split → ambiguous
    for ch in children:
        V11_FROM_LEGACY[ch] = (legacy, resolvable)


def build_per_axis() -> pd.DataFrame:
    mol = pd.read_csv(gpaths.BUNDLED_LEGACY_DIR / "grounding_molecule_bsv.csv")
    dom = mol["dominant_axis"].value_counts().to_dict()   # legacy-axis → n unique analytes

    rows = []
    for axis in cfg.BSV_AXES:
        legacy, resolvable = V11_FROM_LEGACY.get(axis, (None, False))
        pool = int(dom.get(legacy, 0)) if legacy else 0
        siblings = LEGACY8_TO_V11.get(legacy, ()) if legacy else ()
        if legacy is None:
            n_analytes, status, note, unmapped = ("NA", "not_axis_mapped",
                "No legacy 8-axis reference maps to this v11 axis.", 0)
        elif resolvable:
            n_analytes, status, note, unmapped = (pool, "resolved",
                f"1:1 legacy '{legacy}' → {axis}; {pool} unique reference analytes "
                f"(dominant-axis count from the 202-molecule RamanBioLib table).", 0)
        else:
            sib_labels = " + ".join(cfg.axis_short(s) for s in siblings)
            n_analytes, status, note, unmapped = ("NA", "ambiguous_8axis_split",
                f"Legacy '{legacy}' splits into {sib_labels}; the 8-axis grounding "
                f"cannot resolve which of these v11 children each analyte belongs to. "
                f"Shared pool = {pool} unique analytes (not double-counted across children).",
                pool)
        rows.append({
            "axis": axis,
            "axis_short": cfg.axis_short(axis),
            "unique_reference_analytes": n_analytes,
            "measured_reference_spectra": "NA",         # measured counts are per-source, not per-axis
            "direct_spectral_sources": "NA",            # registry has no per-axis attribution
            "supporting_literature_sources": "NA",      # literature is corpus-level, never axis-mapped here
            "unmapped_records": unmapped,
            "mapping_status": status,
            "mapping_notes": note,
        })
    return pd.DataFrame(rows)


def build_corpus_summary() -> dict:
    mol = pd.read_csv(gpaths.BUNDLED_LEGACY_DIR / "grounding_molecule_bsv.csv")
    summary = {
        "unique_reference_analytes_total": int(mol["id"].nunique()),
        "reference_analyte_source": "RamanBioLib 202-molecule reference table (dominant 8-axis)",
        "resolved_axes": ["G03_pyrimidine_nucleotide", "G04_nucleic_acid_phosphate",
                          "G05_glycan_carbohydrate", "G06_protein_peptide_backbone",
                          "G07_aromatic_residue"],
        "ambiguous_axis_pairs": {
            "purine_nucleotide→(G01,G02)": int(mol["dominant_axis"].value_counts().get("purine_nucleotide", 0)),
            "membrane_lipid→(G08,G09)": int(mol["dominant_axis"].value_counts().get("membrane_lipid", 0)),
            "redox_metabolite→(G10,G11)": int(mol["dominant_axis"].value_counts().get("redox_metabolite", 0)),
        },
        "measured_reference_spectra": None,
        "measured_reference_spectra_by_source": {},
        "direct_spectral_source_rows": None,
        "direct_spectral_source_unique": None,
        "supporting_literature_sources": None,
        "registry_notes": [],
        "data_source_mode": gpaths.get_data_status().mode,
    }

    ar = gpaths.autoresearch_root()
    if ar is not None:
        tbl = ar / "gaira_evidence_warehouse_grounding_backbone_v1" / "tables"
        try:
            s = pd.read_csv(tbl / "grounding_peak_support_summary.csv")
            by = {r["source_id"]: int(r["summary_spectra_count"]) for _, r in s.iterrows()}
            summary["measured_reference_spectra_by_source"] = by
            summary["measured_reference_spectra"] = int(sum(by.values()))
        except Exception as e:
            summary["registry_notes"].append(f"peak_support_summary unreadable: {e}")
        try:
            reg = pd.read_csv(tbl / "warehouse_source_registry.csv")
            ref = reg[reg["source_family"].isin(["reference_molecule", "serum_grounding"])]
            lit = reg[reg["source_family"] == "disease_or_stress_paper"]
            summary["direct_spectral_source_rows"] = int(len(ref))
            summary["direct_spectral_source_unique"] = int(ref["source_id"].nunique())
            summary["supporting_literature_sources"] = int(lit["source_id"].nunique())
            dup = ref["source_id"].value_counts()
            dup = dup[dup > 1]
            if len(dup):
                summary["registry_notes"].append(
                    "duplicate registry rows: " + ", ".join(f"{k}×{v}" for k, v in dup.items()))
        except Exception as e:
            summary["registry_notes"].append(f"warehouse_source_registry unreadable: {e}")
    else:
        summary["registry_notes"].append(
            "GAIRA_DATA volume not resolved — corpus-level measured/source counts "
            "unavailable; per-axis analyte counts (from bundled molecule table) are still valid.")
    return summary


if __name__ == "__main__":
    df = build_per_axis()
    df.to_csv(OUT_DIR / "per_axis_grounding_evidence.csv", index=False)
    summ = build_corpus_summary()
    (OUT_DIR / "grounding_corpus_summary.json").write_text(json.dumps(summ, indent=2))
    print("wrote", OUT_DIR / "per_axis_grounding_evidence.csv")
    print(df[["axis_short", "unique_reference_analytes", "mapping_status"]].to_string(index=False))
    print("\ncorpus summary:")
    print(json.dumps(summ, indent=2))
