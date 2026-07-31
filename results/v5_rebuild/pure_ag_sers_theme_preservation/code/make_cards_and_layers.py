"""Parts 5-7: per-analyte transfer cards + the perturbation-sensitivity layer + the
matrix-recoverability linkage. Levels 3 (perturbation) and 4 (matrix) are attached ONLY
where real data exist; every other analyte is explicitly 'Not tested' — never imputed.

Reads the committed metrics table + validation_results.json + phase7_serum_vs_pure.csv.
Additive; frozen atlas untouched.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
BASE = REPO / "results/v5_rebuild/pure_ag_sers_theme_preservation"
(BASE / "analytes").mkdir(parents=True, exist_ok=True)
(BASE / "tables").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "tables/per_analyte_transfer_metrics.csv")
val = json.loads((REPO / "results/v5_rebuild/foundation_audit/tables/validation_results.json").read_text())
phase7 = pd.read_csv(REPO / "results/v5_rebuild/spike_validation/tables/phase7_serum_vs_pure.csv")

# ── Level 3: perturbation sensitivity — ONLY adenine, ergothioneine (dose), uricase/urate (directional) ──
PERT = {
    "adenine": {"kind": "concentration dose-response", "theme": val["3_adenine_dose"]["theme"],
                "monotonicity_rho": val["3_adenine_dose"]["monotonicity_rho"],
                "dose_model": val["3_adenine_dose"]["best_dose_model"],
                "saturating_K_uM": val["3_adenine_dose"]["saturating_K_uM"],
                "saturating_r2": val["3_adenine_dose"]["saturating_r2"],
                "n_levels": len(val["3_adenine_dose"]["levels_uM"]),
                "statement": "purine theme rises monotonically and saturates (Langmuir) — a reproducible dose-response"},
    "ergothioneine": {"kind": "concentration dose-response", "theme": val["4_ergothioneine_dose"]["theme"],
                      "monotonicity_rho": val["4_ergothioneine_dose"]["monotonicity_rho"],
                      "dose_model": val["4_ergothioneine_dose"]["best_dose_model"],
                      "saturating_K_uM": val["4_ergothioneine_dose"]["saturating_K_uM"],
                      "saturating_r2": val["4_ergothioneine_dose"]["saturating_r2"],
                      "n_levels": len(val["4_ergothioneine_dose"]["levels_uM"]),
                      "statement": "sulfur theme rises monotonically and saturates — a reproducible dose-response"},
    # uricase depletes URATE. This is DIRECTIONAL depletion sensitivity, NOT a dose series.
    "urate": {"kind": "directional depletion (uricase)", "theme": "nucleic_purine (oxopurine motif)",
              "delta_oxopurine_motif": val["6_uricase_depletion"]["delta_oxopurine_motif"],
              "delta_purine_ring_motif": val["6_uricase_depletion"]["delta_purine_ring_motif"],
              "purine_theme_delta": val["6_uricase_depletion"]["purine_delta"],
              "localises_to_purine_theme": val["6_uricase_depletion"]["localises_to_purine"],
              "statement": "enzymatic urate removal drops the oxopurine-carbonyl MOTIF sharply "
                           "(theme layer is diffuse); validates perturbation DIRECTION, not a dose score"},
}

pert_rows = []
for a, d in PERT.items():
    row = {"analyte": a, "perturbation_kind": d["kind"], "target_theme": d.get("theme")}
    row.update({k: v for k, v in d.items() if k not in ("kind", "theme", "statement")})
    row["statement"] = d["statement"]
    pert_rows.append(row)
pd.DataFrame(pert_rows).to_csv(BASE / "tables/perturbation_sensitivity.csv", index=False)

# ── Level 4: matrix recoverability — join serum spike recovery from phase7 ──
p7 = phase7.set_index("analyte")

def serum_tier(disp, dircos):
    if disp is None or np.isnan(disp):
        return None
    if disp >= 0.05 and dircos >= 0.35:
        return "strong"
    if disp >= 0.02:
        return "moderate"
    return "weak"

matrix_rows = []
for a in df.analyte:
    if a in p7.index:
        r = p7.loc[a]
        matrix_rows.append({
            "analyte": a,
            "serum_spike_displacement": round(float(r.spike_displacement_norm), 4),
            "serum_replicate_direction_cos": round(float(r.replicate_direction_cos), 4),
            "serum_vs_pureSERS_cos": round(float(r.cos_spike_vs_pureSERS), 4),
            "serum_vs_pureRaman_cos": round(float(r.cos_spike_vs_pureRaman), 4),
            "serum_recovery_tier": serum_tier(float(r.spike_displacement_norm),
                                              float(r.replicate_direction_cos)),
            "serum_tested": True,
        })
    else:
        matrix_rows.append({"analyte": a, "serum_tested": False, "serum_recovery_tier": None})
matrix = pd.DataFrame(matrix_rows)
# link to pure transfer for the recoverability-boundary comparison
link = matrix.merge(df[["analyte", "family", "component_cosine", "theme_cosine_distinct",
                        "mss_cosine", "quadrant"]], on="analyte", how="left")
link.to_csv(BASE / "tables/matrix_recoverability_linkage.csv", index=False)

# ── Part 5: per-analyte transfer cards (JSON + MD), all four levels ──
pert_by = {r["analyte"]: r for r in pert_rows}
matrix_by = {r["analyte"]: r for r in matrix_rows}

def card(a):
    r = df[df.analyte == a].iloc[0].to_dict()
    c = {
        "analyte": a, "family": r["family"], "expected_theme": r["expected_theme"],
        "n_raman": int(r["n_raman"]), "n_sers": int(r["n_sers"]),
        "level1_latent_fingerprint": {
            "component_cosine": r["component_cosine"], "tier": r["component_tier"],
            "meaning": "similarity of the 24 NMF coordinates (adsorption-limited)"},
        "level2_biochemical_theme": {
            "dominant_theme_match": bool(r["dominant_theme_match"]),
            "raman_dominant": r["raman_dominant"], "sers_dominant": r["sers_dominant"],
            "theme_cosine_raw": r["theme_cosine"],
            "theme_cosine_distinct": r["theme_cosine_distinct"],
            "theme_null_mean": r["theme_null_mean"], "theme_separation": r["theme_separation"],
            "self_is_nearest_theme": bool(r["self_is_nearest_theme"]),
            "expected_rank_raman": r["expected_rank_raman"], "expected_rank_sers": r["expected_rank_sers"],
            "expected_retained_top3": bool(r["expected_retained_top3"]),
            "mss_cosine": r["mss_cosine"], "dominant_mss_match": bool(r["dominant_mss_match"]),
            "redistribution": {"theme_jsd": r["theme_jsd"], "l1_theme_shift": r["l1_theme_shift"],
                               "gained_theme": r["gained_theme"], "lost_theme": r["lost_theme"],
                               "gained_component": r["gained_component"], "lost_component": r["lost_component"],
                               "gained_mss": r["gained_mss"], "lost_mss": r["lost_mss"]}},
        "level3_perturbation_sensitivity": pert_by.get(a, {"tested": False, "status": "Not tested",
            "why": "no controlled perturbation series exists for this analyte"}),
        "level4_matrix_recoverability": (
            {k: v for k, v in matrix_by[a].items()} if matrix_by[a].get("serum_tested")
            else {"tested": False, "status": "Not tested", "why": "not in the serum spike-in panel"}),
        "quadrant": r["quadrant"],
        "ood_sers": r["ood_sers"], "confidence_sers": r["confidence_sers"],
    }
    return c

def card_md(c):
    a = c["analyte"]; L1 = c["level1_latent_fingerprint"]; L2 = c["level2_biochemical_theme"]
    L3 = c["level3_perturbation_sensitivity"]; L4 = c["level4_matrix_recoverability"]
    red = L2["redistribution"]
    lines = [f"# {a}  ·  cross-modal transfer card",
             f"*Family: {c['family']} · expected theme: {c['expected_theme']} · "
             f"{c['n_raman']} Raman / {c['n_sers']} Ag-SERS spectra · quadrant: {c['quadrant']}*", ""]
    lines += ["## Level 1 — Latent fingerprint preservation",
              f"- component cosine **{L1['component_cosine']}** ({L1['tier']}) — {L1['meaning']}", ""]
    lines += ["## Level 2 — Biochemical theme preservation",
              f"- dominant theme: Raman **{L2['raman_dominant']}** → Ag-SERS **{L2['sers_dominant']}**  "
              f"({'preserved' if L2['dominant_theme_match'] else 'not preserved'})",
              f"- theme cosine: raw {L2['theme_cosine_raw']} · **distinctive {L2['theme_cosine_distinct']}** "
              f"(null {L2['theme_null_mean']}, separation {L2['theme_separation']}, "
              f"self-nearest {L2['self_is_nearest_theme']})",
              f"- expected theme rank: Raman #{L2['expected_rank_raman']} → Ag-SERS #{L2['expected_rank_sers']} "
              f"(top-3 retained: {L2['expected_retained_top3']})",
              f"- MSS motif cosine {L2['mss_cosine']} (dominant motif preserved: {L2['dominant_mss_match']})",
              f"- redistribution: JSD {red['theme_jsd']}, L1 {red['l1_theme_shift']}; "
              f"gained **{red['gained_theme']}**, lost **{red['lost_theme']}** "
              f"(motif +{red['gained_mss']} / −{red['lost_mss']})", ""]
    lines += ["## Level 3 — Perturbation sensitivity"]
    if L3.get("tested") is False or L3.get("status") == "Not tested":
        lines.append(f"- **Not tested** — {L3.get('why','no perturbation data')}")
    else:
        lines.append(f"- **{L3['perturbation_kind']}** on {L3.get('target_theme')}: {L3['statement']}")
        for k in ("monotonicity_rho", "saturating_K_uM", "saturating_r2",
                  "delta_oxopurine_motif", "delta_purine_ring_motif", "purine_theme_delta"):
            if k in L3 and L3[k] is not None:
                lines.append(f"  - {k}: {L3[k]}")
    lines += ["", "## Level 4 — Matrix recoverability (serum)"]
    if not L4.get("serum_tested"):
        lines.append(f"- **Not tested** — {L4.get('why','not in serum panel')}")
    else:
        lines.append(f"- serum recovery tier **{L4['serum_recovery_tier']}** "
                     f"(displacement {L4['serum_spike_displacement']}, "
                     f"direction cos {L4['serum_replicate_direction_cos']}, "
                     f"vs pure-SERS {L4['serum_vs_pureSERS_cos']})")
    lines += ["", f"*OOD(SERS) {c['ood_sers']} · confidence(SERS) {c['confidence_sers']}. "
              f"Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*"]
    return "\n".join(lines)

cards = {}
for a in df.analyte:
    c = card(a); cards[a] = c
    safe = a.replace("/", "_").replace(" ", "_")
    (BASE / "analytes" / f"{safe}.json").write_text(json.dumps(c, indent=2))
    (BASE / "analytes" / f"{safe}.md").write_text(card_md(c))
(BASE / "artifacts/all_cards.json").write_text(json.dumps(cards, indent=1))

print(f"cards written: {len(cards)} to analytes/")
print(f"perturbation rows: {len(pert_rows)} (adenine, ergothioneine dose; urate/uricase directional)")
print(f"matrix linkage: {matrix.serum_tested.sum()}/{len(matrix)} analytes have serum data")
print("\nperturbation_sensitivity.csv:")
print(pd.DataFrame(pert_rows)[["analyte", "perturbation_kind", "target_theme", "statement"]].to_string(index=False))
print("\nserum recovery tiers among matched pure analytes:")
print(link.serum_recovery_tier.value_counts(dropna=False))
