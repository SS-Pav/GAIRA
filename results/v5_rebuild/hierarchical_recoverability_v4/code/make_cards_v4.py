"""V4 per-analyte evidence cards (JSON + all_cards) from the committed evidence profile.
Physics-aware language; explicit 'not tested'; independent recovery flags, no single score.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

BASE = Path("/Users/surajpg/projects/GAIRA/results/v5_rebuild/hierarchical_recoverability_v4")
(BASE / "analytes").mkdir(exist_ok=True)
df = pd.read_csv(BASE / "tables/per_analyte_evidence_profile.csv")

cards = {}
for _, r in df.iterrows():
    c = {
        "analyte": r.analyte, "family": r.family, "expected_theme": r.expected_theme,
        "evidence_profile": r.evidence_profile,
        "level1_latent": {"cosine": r.C_latent, "null95": r.latent_null95, "rank": int(r.latent_rank),
                          "fdr_q": r.latent_q, "recovered": bool(r.latent_recovered),
                          "supporting": bool(r.latent_supporting)},
        "level2_mss": {"cosine": r.C_MSS, "null95": r.MSS_null95, "rank": int(r.MSS_rank),
                       "fdr_q": r.MSS_q, "recovered": bool(r.MSS_recovered),
                       "supporting": bool(r.MSS_supporting)},
        "level3_theme": {"raw_cosine": r.C_theme_raw, "raw_null_med": r.theme_raw_null_med,
                         "identity_residual": r.theme_identity, "identity_null95": r.theme_id_null95,
                         "identity_rank": int(r.theme_id_rank), "expected_theme_top3": bool(r.expected_theme_top3),
                         "recovered": bool(r.theme_recovered),
                         "rank_rho": r.rank_rho, "top2": r.top2, "top3": r.top3, "top3_null": r.top3_null,
                         "argmax_agree": bool(r.argmax_agree), "argmax_robust": bool(r.argmax_robust),
                         "note": "raw theme cosine is broad interpretation, NOT analyte identity"},
        "level4_perturbation": ({"status": r.perturbation_status, "validated": True}
                                if r.perturbation_validated else
                                {"status": "not tested", "validated": False}),
        "level5_matrix": ({"tier": r.serum_tier, "recovered": bool(r.matrix_recovered)}
                          if r.serum_tested else {"tier": "not tested", "recovered": False}),
        "purine": {"delta_purine": r.delta_purine, "purine_R": r.purine_R, "purine_S": r.purine_S},
        "ood_sers": r.ood_sers, "confidence_sers": r.confidence_sers,
    }
    cards[r.analyte] = c
    safe = r.analyte.replace("/", "_").replace(" ", "_")
    (BASE / "analytes" / f"{safe}.json").write_text(json.dumps(c, indent=2, default=float))
(BASE / "artifacts/all_cards_v4.json").write_text(json.dumps(cards, indent=1, default=float))
print(f"V4 cards: {len(cards)} written")
