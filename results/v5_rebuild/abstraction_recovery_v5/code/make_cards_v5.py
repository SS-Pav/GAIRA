"""V5 per-analyte evidence cards from the committed abstraction table. Physics-aware, graded
(present vs specific), explicit 'not tested'/'unassigned'; no molecular-ID claims from motif/theme."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

BASE = Path("/Users/surajpg/projects/GAIRA/results/v5_rebuild/abstraction_recovery_v5")
(BASE / "analytes").mkdir(exist_ok=True)
df = pd.read_csv(BASE / "tables/per_analyte_abstraction_recovery.csv")
cards = {}
for _, r in df.iterrows():
    def tier(present, specific):
        return "specific" if specific else ("present (non-specific)" if present else "not recovered")
    c = {
        "analyte": r.analyte, "broad_family": r.broad_family, "subclass": r.subclass,
        "subclass_exploratory": bool(r.subclass_exploratory),
        "exact_identity": {"latent": bool(r.latent_identity_recovered), "mss": bool(r.mss_identity_recovered),
                           "theme": bool(r.theme_identity_recovered),
                           "verdict": "specifically recovered" if r.latent_identity_recovered else "not specifically recovered"},
        "nmf_component": {"top3_overlap": r.comp_top3_overlap, "mass_retained": r.comp_mass_retained,
                          "recovered": bool(r.component_recovered)},
        "mss_motif": {"expected": r.expected_mss, "rank_agsers": r.mss_rank_S,
                      "enrich_over_null": r.mss_enrich_null, "present_top3": bool(r.mss_present_top3),
                      "specific": bool(r.mss_motif_recovered), "status": r.mss_status,
                      "tier": tier(r.mss_present_top3, r.mss_motif_recovered)},
        "molecular_subclass": {"assigned": r.subclass,
                               "loao_centroid_recovered": bool(r.subclass_loao_recovered),
                               "nn_same_subclass": bool(r.nn_same_subclass),
                               "note": "cross-modal subclass classification is at chance in aggregate"},
        "broad_family": {"nn_same_family": bool(r.nn_same_family), "loao_recovered": bool(r.family_loao_recovered)},
        "biochemical_theme": {"expected": r.expected_theme, "rank_agsers": r.expected_theme_rank_S,
                              "enrich_over_null": r.theme_enrich_null, "present_top3": bool(r.theme_present_top3),
                              "specific": bool(r.theme_recovered), "tier": tier(r.theme_present_top3, r.theme_recovered)},
        "perturbation": {"status": r.perturbation_status, "validated": r.perturbation_status != "not tested"},
        "matrix": {"tier": r.serum_tier, "recovered": bool(r.matrix_recovered)},
        "delta_purine": r.delta_purine, "ood_sers": r.ood_sers, "confidence_sers": r.confidence_sers,
        "highest_recovered_level": r.highest_recovered_level,
        "evidence_profile": r.evidence_profile,
    }
    # human conclusion
    if r.latent_identity_recovered:
        concl = "Exact analyte specifically recovered from Ag-SERS."
    elif r.mss_motif_recovered:
        concl = f"Exact identity lost, but the expected {r.expected_mss} motif is specifically recovered."
    elif r.perturbation_status != "not tested":
        concl = "Static identity weak, but functional perturbation response validates the expected chemistry."
    elif r.mss_present_top3 or r.theme_present_top3:
        concl = ("Ag-SERS shows the expected motif/theme in its top-3, but this PRESENCE is not analyte-"
                 "specific above the shared background — no molecular identification is claimed.")
    else:
        concl = "No specific biochemical evidence survives Ag-SERS abstraction for this analyte."
    c["conclusion"] = concl
    cards[r.analyte] = c
    safe = r.analyte.replace("/", "_").replace(" ", "_")
    (BASE / "analytes" / f"{safe}.json").write_text(json.dumps(c, indent=2, default=float))
(BASE / "artifacts/all_cards_v5.json").write_text(json.dumps(cards, indent=1, default=float))
print(f"V5 cards: {len(cards)} written")
