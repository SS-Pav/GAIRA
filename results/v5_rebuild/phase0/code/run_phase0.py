"""GAIRA V5 Phase 0 (V5.0) — emit canonical registries + run the admission gate.

Read-only. Emits registries to results/v5_rebuild/phase0/tables/ and a log.
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
import pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src"))
from gaira.data import loader, admits_to_joint_analysis  # noqa

OUT = REPO / "results/v5_rebuild/phase0/tables"; OUT.mkdir(parents=True, exist_ok=True)
LOG = REPO / "results/v5_rebuild/phase0/logs"; LOG.mkdir(parents=True, exist_ok=True)


def main():
    specs = loader.load_all()
    # grounding_spectrum_registry (one row per loaded observation)
    rows = []
    for s in specs:
        r = s.record.to_row()
        ok, reason = admits_to_joint_analysis(s.record)
        r["admitted_to_joint_analysis"] = ok
        r["admission_reason"] = reason
        r["has_full_spectrum"] = s.has_spectrum
        rows.append(r)
    spec_df = pd.DataFrame(rows)
    spec_df.to_csv(OUT / "grounding_spectrum_registry.csv", index=False)

    # grounding_analyte_registry (unique analyte x modality x source)
    an = (spec_df.groupby(["canonical_analyte_name", "modality", "source_dataset",
                           "substrate_geometry"])
          .agg(n_observations=("spectrum_id", "count"),
               excitations=("excitation_nm", lambda s: sorted(set(x for x in s if pd.notna(x)))))
          .reset_index())
    an.to_csv(OUT / "grounding_analyte_registry.csv", index=False)

    # acquisition_domain_registry
    dom = (spec_df[spec_df.has_full_spectrum]
           .groupby(["modality", "substrate_geometry", "excitation_nm"])
           .agg(n_spectra=("spectrum_id", "count"),
                n_analytes=("canonical_analyte_name", "nunique"),
                sources=("source_dataset", lambda s: sorted(set(s)))).reset_index())
    dom.to_csv(OUT / "acquisition_domain_registry.csv", index=False)

    # substrate_registry
    sub = (spec_df.groupby(["substrate_material", "substrate_geometry", "modality"])
           .agg(n_spectra=("spectrum_id", "count"),
                sources=("source_dataset", lambda s: sorted(set(s)))).reset_index())
    sub.to_csv(OUT / "substrate_registry.csv", index=False)

    # pointers to held-out registries (built in V4; roles enforced here)
    pointers = pd.DataFrame([
        {"registry": "controlled_perturbation_registry",
         "source": "data_audit/v4_controlled_perturbation_evaluation_registry.csv",
         "rule": "held out from axis/observation/coordinate/motif fitting and training"},
        {"registry": "biological_dataset_registry", "source": "data_audit/biological_dataset_registry.csv",
         "rule": "challenge only; never defines molecular coordinate system"},
        {"registry": "physics_evidence_registry",
         "source": "data_audit/v4_ag_flakes_metabolite24_peak_registry.csv + physics_atlas_registry.csv",
         "rule": "peak/ambiguity/collision evidence; not full-spectrum grounding"},
    ])
    pointers.to_csv(OUT / "heldout_registry_pointers.csv", index=False)

    # admission summary
    n_ok = int(spec_df.admitted_to_joint_analysis.sum())
    excl = collections.Counter(spec_df[~spec_df.admitted_to_joint_analysis]["admission_reason"])
    summary = {
        "total_loaded": len(spec_df),
        "by_source": spec_df.source_dataset.value_counts().to_dict(),
        "admitted_to_joint_analysis": n_ok,
        "excluded": dict(excl),
        "acquisition_domains_full_spectrum": len(dom),
        "raman_excitations": sorted(set(dom[dom.modality == "raman"]["excitation_nm"].dropna())),
        "sers_domains": dom[dom.modality == "sers"][["substrate_geometry", "excitation_nm", "n_spectra"]].to_dict("records"),
    }
    (LOG / "phase0_summary.txt").write_text(str(summary))
    print("== Phase 0 registries emitted ==")
    print("total:", len(spec_df), "| admitted:", n_ok, "| excluded:", dict(excl))
    print("acquisition domains (full-spectrum):", len(dom))
    print(dom[["modality", "substrate_geometry", "excitation_nm", "n_spectra", "n_analytes"]].to_string(index=False))
    return spec_df, dom


if __name__ == "__main__":
    main()
