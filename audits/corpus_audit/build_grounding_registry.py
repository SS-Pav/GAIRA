"""Build the molecular-grounding registries from the 202-analyte reference
table + the 43-source warehouse + the grounding peak-support summary.

Deterministic, read-only. Outputs:
  data_audit/grounding_analyte_registry.csv
  data_audit/grounding_spectrum_registry.csv   (source-level measured-spectra rows)
  data_audit/grounding_totals.json
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
LEGACY = REPO / "streamlit_apps" / "gaira_demo" / "data"
WARE = (REPO_VOL := Path("/Volumes/SSD_Rad/GAIRA_DATA")) / "processed" / "gaira_autoresearch" / \
    "gaira_autoresearch_v1" / "gaira_evidence_warehouse_grounding_backbone_v1" / "tables"
OUT = REPO / "data_audit"; OUT.mkdir(exist_ok=True)

# legacy 8 -> v11 (mirror of config.LEGACY8_TO_V11)
L8 = {"membrane_lipid": ["G08_lipid_acyl_membrane", "G09_sterol_neutral_lipid"],
      "protein_backbone": ["G06_protein_peptide_backbone"],
      "aromatic_amino_acid": ["G07_aromatic_residue"],
      "purine_nucleotide": ["G01_purine_nucleotide", "G02_purine_metabolite"],
      "pyrimidine_nucleotide": ["G03_pyrimidine_nucleotide"],
      "glycan_carbohydrate": ["G05_glycan_carbohydrate"],
      "redox_metabolite": ["G10_sulfur_thiol_redox", "G11_metabolic_small_molecule"],
      "nucleic_acid_backbone": ["G04_nucleic_acid_phosphate"]}


def main():
    bsv = pd.read_csv(LEGACY / "grounding_molecule_bsv.csv")
    idx = pd.read_csv(LEGACY / "grounding_molecule_index.csv")
    m = bsv.merge(idx[["id", "type", "sample_substrate", "laser_wavelength"]], on="id", how="left")

    def modality(sub):
        s = str(sub).lower()
        if "caf2" in s or "glass" in s or "coverslip" in s or "compartment" in s:
            return "Raman"
        if "metal" in s or "gold" in s or "ag" in s or "au" in s:
            return "SERS(likely)"
        return "unknown"

    rows = []
    for _, r in m.iterrows():
        split = L8.get(r["dominant_axis"], [])
        rows.append({
            "canonical_analyte_name": r["component"],
            "chemical_class": r["type"],
            "biochemical_family": r["family"],
            "reference_dataset": "RamanBioLib(grounding_molecule_bsv)",
            "reference_modality": modality(r["sample_substrate"]),
            "substrate": r["sample_substrate"],
            "excitation_nm": r["laser_wavelength"],
            "number_of_measured_spectra": "NA(1 collapsed BSV row; raw not in this table)",
            "is_pure_analyte": True, "is_literature_only": False,
            "is_digitized_spectrum": "unknown",
            "dominant_legacy_axis": r["dominant_axis"],
            "mapped_gaira_axes": ";".join(split),
            "mapping_confidence": "resolved" if len(split) == 1 else "derived_split(ambiguous)",
            "evidence_tier": 1,
            "source_path": str(LEGACY / "grounding_molecule_bsv.csv"),
            "notes": "one row per analyte; BSV is precomputed 8-axis; raw spectrum not co-located",
        })
    adf = pd.DataFrame(rows)
    adf.to_csv(OUT / "grounding_analyte_registry.csv", index=False)

    # source-level measured-spectra registry (warehouse peak-support summary)
    spec_rows = []
    totals_measured = {"raman": 0, "sers": 0}
    if (WARE / "grounding_peak_support_summary.csv").exists():
        s = pd.read_csv(WARE / "grounding_peak_support_summary.csv")
        for _, r in s.iterrows():
            mod = str(r["modality"]).lower()
            n = int(r["summary_spectra_count"])
            spec_rows.append({
                "source_id": r["source_id"], "source_family": r["source_family"],
                "modality": r["modality"], "measured_spectra": n,
                "detected_peaks": r["detected_peak_count"], "class_count": r["class_count"],
                "raw_or_processed": "measured/summary",
                "used_in_demo": r["source_id"] in ("adenine_sers_control",),
                "notes": "measured reference spectra counted in the grounding warehouse"})
            if "raman" in mod:
                totals_measured["raman"] += n
            elif "sers" in mod:
                totals_measured["sers"] += n
    pd.DataFrame(spec_rows).to_csv(OUT / "grounding_spectrum_registry.csv", index=False)

    # warehouse composition
    ware = {}
    if (WARE / "warehouse_source_registry.csv").exists():
        w = pd.read_csv(WARE / "warehouse_source_registry.csv")
        ware = {"n_rows": len(w), "unique_source_ids": int(w["source_id"].nunique()),
                "source_family": w["source_family"].value_counts().to_dict(),
                "modality": w["modality"].value_counts().to_dict(),
                "biosample_type": w["biosample_type"].value_counts().to_dict()}

    totals = {
        "table_defining_202": str(LEGACY / "grounding_molecule_bsv.csv"),
        "n_rows_202_table": len(bsv),
        "unique_analyte_names": int(m["component"].nunique()),
        "duplicate_analyte_names": int(len(m) - m["component"].nunique()),
        "entity_type_distribution": m["type"].value_counts().to_dict(),
        "modality_distribution_202": adf["reference_modality"].value_counts().to_dict(),
        "substrate_distribution_202": m["sample_substrate"].value_counts().to_dict(),
        "dominant_axis_distribution": m["dominant_axis"].value_counts().to_dict(),
        "measured_reference_spectra_by_source": {r["source_id"]: r["measured_spectra"] for r in spec_rows},
        "total_measured_reference_spectra": int(sum(r["measured_spectra"] for r in spec_rows)),
        "measured_by_modality": totals_measured,
        "warehouse": ware,
        "note_202": "202 = rows in grounding_molecule_bsv.csv; each is one analyte with a precomputed "
                    "8-axis BSV. Raw spectra for these 202 are NOT co-located in this table; only the "
                    "warehouse reference sources (adenine 12, amino_acid 20, metabolite63 64, serum_ag 64) "
                    "carry counted MEASURED spectra.",
    }
    (OUT / "grounding_totals.json").write_text(json.dumps(totals, indent=2, default=str))
    print("202-table unique analytes:", totals["unique_analyte_names"],
          "| duplicates:", totals["duplicate_analyte_names"])
    print("entity types:", totals["entity_type_distribution"])
    print("modality (202):", totals["modality_distribution_202"])
    print("measured reference spectra by source:", totals["measured_reference_spectra_by_source"],
          "=> total", totals["total_measured_reference_spectra"], "by modality", totals_measured)
    print("warehouse:", ware.get("source_family"))


if __name__ == "__main__":
    main()
