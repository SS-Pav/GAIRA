"""V3 per-analyte assessment cards — 9 layers each. Uses hierarchical, physics-aware language
(latent redistribution / adsorption-driven observation bias / identity-specific preservation /
functional perturbation validation), NEVER binary "theme preserved / failed". Additive.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
BASE = REPO / "results/v5_rebuild/representation_hierarchy_v3"
(BASE / "analytes").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "tables/per_analyte_hierarchy.csv").set_index("analyte")
mat = pd.read_csv(BASE / "tables/matrix_robustness.csv").set_index("analyte")
val = json.loads((REPO / "results/v5_rebuild/foundation_audit/tables/validation_results.json").read_text())

PERTURB = {
    "adenine": "functional perturbation validation — a 14-point concentration series drives the "
               "nucleic_purine theme monotonically (ρ=0.996) along a saturating Langmuir law "
               "(K=0.89 µM); the biochemical abstraction is dynamically confirmed, not merely static.",
    "ergothioneine": "functional perturbation validation — the sulfur_antioxidant theme rises "
                     "monotonically and saturates with dose (ρ=0.927, K=1.52 µM).",
    "urate": "directional perturbation validation — enzymatic (uricase) removal drops the "
             "oxopurine-carbonyl motif sharply (Δ=−0.060); validates response DIRECTION at the "
             "motif layer, not a dose magnitude.",
}


def interpret(a, r):
    """Layer 8 — physics-aware interpretation, no binary preserved/failed."""
    L1, idn, sep, argmax, dpur = (r.L1_latent_fingerprint, r.L3b_theme_identity,
                                  r.L4_rank_separation, r.L6_argmax_agreement, r.delta_purine)
    parts = []
    if L1 >= 0.55 and idn >= 0.5:
        parts.append("Identity-specific preservation across the hierarchy: both the latent "
                     "fingerprint and the distinctive biochemical abstraction transfer to silver.")
    elif L1 < 0.55 and (idn >= 0.4 or (argmax and sep > 0)):
        parts.append("Latent redistribution with retained biochemical abstraction: the 24-coordinate "
                     "fingerprint is reshaped by adsorption, yet the higher-level theme identity survives.")
    elif idn < 0 and dpur > 0.03:
        parts.append("Adsorption-driven observation bias: on silver this analyte is pulled toward the "
                     "nucleic_purine attractor, so its distinctive Raman abstraction is not recovered "
                     "even where surface-level structure partially transfers.")
    else:
        parts.append("Partial mid-level transfer: motif structure carries over but the identity-specific "
                     "theme abstraction is weak — a surface-physics limit, not a representation error.")
    if a in PERTURB:
        parts.append(PERTURB[a])
    return " ".join(parts)


def limitations(a, r):
    """Layer 9 — explicit limitations."""
    lims = ["Raw theme cosine and raw rank ρ are baseline-inflated and are NOT stand-alone evidence; "
            f"this analyte's identity-specific signal is cosine {r.L3b_theme_identity:+.2f} / rank "
            f"separation {r.L4_rank_separation:+.3f}."]
    if a not in PERTURB:
        lims.append("No dynamic perturbation validation exists for this analyte (Level 4 not measured).")
    if a in mat.index:
        disp = float(mat.loc[a, "serum_spike_displacement"])
        if disp < 0.02:
            lims.append(f"Weak serum-matrix recoverability (displacement {disp:.3f}): competition on "
                        "colloid suppresses recovery.")
    else:
        lims.append("Not in the serum spike-in panel (Level 5 not measured).")
    return lims


def card(a):
    r = df.loc[a]
    c = {
        "analyte": a,
        "layer1_latent_fingerprint": {"component_cosine": r.L1_latent_fingerprint,
            "meaning": "cosine over 24 NMF coordinates — surface-physics-dominated"},
        "layer2_mss_motif": {"mss_cosine": r.L2_mss_motif,
            "meaning": "cosine over 12 biochemical MSS motifs — mid-level structure"},
        "layer3_theme_cosine": {"raw": r.L3a_theme_raw, "identity_specific": r.L3b_theme_identity,
            "identity_null": r.L3b_identity_null, "identity_separation": r.L3b_identity_separation,
            "meaning": "raw is baseline-inflated; identity-specific is baseline-subtracted vs a null"},
        "layer4_theme_rank_correlation": {"spearman_rho": r.L4_theme_rank_rho,
            "rank_null": r.L4_rank_null, "rank_separation": r.L4_rank_separation,
            "meaning": "Spearman ρ of the 11-theme ordering; separation = identity-specific part"},
        "layer5_top3_overlap": {"top2": r.L5_top2_overlap, "top3": r.L5_top3_overlap,
            "meaning": "fraction of top themes shared — avoids argmax instability"},
        "layer6_argmax_agreement": {"agree": bool(r.L6_argmax_agreement),
            "raman_dominant": r.raman_dominant, "sers_dominant": r.sers_dominant,
            "meaning": "strict single-dominant-theme agreement — intentionally strict and unstable"},
        "layer7_family": {"family": r.family, "expected_theme": r.expected_theme,
            "delta_purine": r.delta_purine, "purine_share_raman": r.purine_share_raman,
            "purine_share_sers": r.purine_share_sers},
        "layer8_interpretation": interpret(a, r),
        "layer9_limitations": limitations(a, r),
    }
    if a in mat.index:
        m = mat.loc[a]
        c["layer7_family"]["serum_spike_displacement"] = float(m.serum_spike_displacement)
    return c


def card_md(c):
    a = c["analyte"]; r = df.loc[a]
    L = c
    lines = [f"# {a} · Representation-hierarchy assessment (V3)",
             f"*Family: {L['layer7_family']['family']} · expected theme: {L['layer7_family']['expected_theme']}*", ""]
    lines += ["| Layer | Metric | Value |", "|---|---|---|",
              f"| 1 · Latent fingerprint | component cosine | {L['layer1_latent_fingerprint']['component_cosine']} |",
              f"| 2 · MSS motif | mss cosine | {L['layer2_mss_motif']['mss_cosine']} |",
              f"| 3 · Theme (raw) | theme cosine raw | {L['layer3_theme_cosine']['raw']} |",
              f"| 3 · Theme (identity) | baseline-subtracted | {L['layer3_theme_cosine']['identity_specific']} "
              f"(null {L['layer3_theme_cosine']['identity_null']}, sep {L['layer3_theme_cosine']['identity_separation']}) |",
              f"| 4 · Theme rank | Spearman ρ | {L['layer4_theme_rank_correlation']['spearman_rho']} "
              f"(sep {L['layer4_theme_rank_correlation']['rank_separation']}) |",
              f"| 5 · Top-k overlap | top-2 / top-3 | {L['layer5_top3_overlap']['top2']} / {L['layer5_top3_overlap']['top3']} |",
              f"| 6 · Argmax agreement | dominant theme | {L['layer6_argmax_agreement']['raman_dominant']} → "
              f"{L['layer6_argmax_agreement']['sers_dominant']} ({'agree' if L['layer6_argmax_agreement']['agree'] else 'differ'}) |",
              f"| 7 · Family | ΔPurine share | {L['layer7_family']['delta_purine']} "
              f"({L['layer7_family']['purine_share_raman']} → {L['layer7_family']['purine_share_sers']}) |", ""]
    lines += ["## Layer 8 — Interpretation", L["layer8_interpretation"], "",
              "## Layer 9 — Limitations"]
    lines += [f"- {x}" for x in L["layer9_limitations"]]
    lines += ["", "*Frozen atlas 09ed804a…; raw theme/rank are baseline-inflated and never read alone.*"]
    return "\n".join(lines)


cards = {}
for a in df.index:
    c = card(a); cards[a] = c
    safe = a.replace("/", "_").replace(" ", "_")
    (BASE / "analytes" / f"{safe}.json").write_text(json.dumps(c, indent=2, default=float))
    (BASE / "analytes" / f"{safe}.md").write_text(card_md(c))
(BASE / "artifacts").mkdir(exist_ok=True)
(BASE / "artifacts/all_cards_v3.json").write_text(json.dumps(cards, indent=1, default=float))
print(f"V3 cards written: {len(cards)} (9 layers each) to analytes/")
print("example interpretations:")
for a in ["adenine", "glucose", "guanine", "uracil"]:
    print(f"  {a}: {cards[a]['layer8_interpretation'][:120]}...")
