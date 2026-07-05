"""gaira_base_4 cross-pilot synthesis v1.

Synthesize Pilot 1.1 + Pilot 2.1 normalized passive readouts.
Identify robust cross-pilot signals vs substrate/cohort-sensitive findings.

NO classifier. NO threshold tuning. NO target fitting. NO engine change.
"""
from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_gaira_base_4_hybrid_bsv_build_v1 import BSV_GROUPS
from run_gaira_base_4_hybrid_bsv_controlled_calibration_v2 import FAMILY_LABELS


ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_cross_pilot_synthesis_v1")
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"

P1_DIR = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_passive_target_pilot_1_1_normalization_sensitivity/tables"
)
P2_DIR = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_passive_target_pilot_2_1_normalization_sensitivity/tables"
)

BSV_GROUPS_ORDER = [g["group_id"] for g in BSV_GROUPS]


def _cohens_d(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2: return 0.0
    pooled = np.sqrt(((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1))
                       / max(len(x)+len(y)-2, 1))
    return (np.mean(x) - np.mean(y)) / (pooled if pooled > 0 else 1.0)


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — load + harmonize
# ─────────────────────────────────────────────────────────────────────

def stage1_load():
    print("\n[STAGE 1] load + harmonize")
    p1_eff = pd.read_csv(P1_DIR / "pilot1_1_effect_size_survival.csv")
    p2_eff = pd.read_csv(P2_DIR / "pilot2_1_effect_size_survival.csv")
    p1_mass = pd.read_csv(P1_DIR / "pilot1_1_bsv_mass.csv")
    p2_mass = pd.read_csv(P2_DIR / "pilot2_1_bsv_mass_metrics.csv")

    # Pilot 2.1 also has normalized vectors — we need pairwise CCA-vs-HCC etc.
    p2_vec = pd.read_csv(P2_DIR / "pilot2_1_normalized_bsv_vectors.csv")

    # Pilot 1: comparison label is "H0T_vs_CTR"
    p1_eff = p1_eff.copy()
    p1_eff["pilot"] = "P1_Gurian_HCC"
    p1_eff["substrate"] = "Ag_colloid_untyped (Gurian Ag)"
    p1_eff["comparison"] = "P1_HCC_vs_CTR"
    p1_eff["n_target"] = 72; p1_eff["n_control"] = 72

    # Pilot 2 vs-NC effects
    p2_eff = p2_eff.copy()
    p2_eff["pilot"] = "P2_label_free_SERS_nanosensor"
    p2_eff["substrate"] = "unknown_SERS"
    p2_eff["comparison"] = p2_eff["comparison"].apply(lambda c: f"P2_{c}")
    n_per = {"NC": 44, "HCC": 49, "CCA": 58, "LM": 44}
    p2_eff["n_control"] = 44
    p2_eff["n_target"] = p2_eff["comparison"].apply(
        lambda c: n_per.get(c.replace("P2_", "").split("_vs_")[0], 0)
    )

    # Compute Pilot 2 pairwise (CCA_vs_HCC, LM_vs_HCC, CCA_vs_LM) for sumnorm + clr
    pair_rows = []
    for rep in ["abs", "sumnorm", "clr", "delta_nc", "delta_sumnorm"]:
        for a, b in [("CCA", "HCC"), ("LM", "HCC"), ("CCA", "LM")]:
            for g in BSV_GROUPS_ORDER:
                col = f"{rep}_{g}"
                if col not in p2_vec.columns: continue
                x = p2_vec[p2_vec.class_label == a][col].values
                y = p2_vec[p2_vec.class_label == b][col].values
                d = _cohens_d(x, y)
                pair_rows.append({
                    "pilot": "P2_label_free_SERS_nanosensor",
                    "substrate": "unknown_SERS",
                    "comparison": f"P2_{a}_vs_{b}",
                    "representation": rep,
                    "family": g, "family_label": FAMILY_LABELS.get(g, g),
                    "cohens_d": round(float(d), 3),
                    "abs_d": round(abs(float(d)), 3),
                    "ci95_low": np.nan, "ci95_high": np.nan,
                    "ci_excludes_zero": False,   # not bootstrapped here
                    "direction": "+" if d > 0 else ("-" if d < 0 else "0"),
                    "n_target": n_per[a], "n_control": n_per[b],
                })
    p2_pairs = pd.DataFrame(pair_rows)

    # Concatenate everything with a common schema
    common_cols = ["pilot", "substrate", "comparison", "representation", "family",
                    "family_label", "cohens_d", "abs_d", "ci95_low", "ci95_high",
                    "ci_excludes_zero", "n_target", "n_control"]
    for d in (p1_eff, p2_eff, p2_pairs):
        for c in common_cols:
            if c not in d.columns: d[c] = np.nan
    harm = pd.concat([p1_eff[common_cols], p2_eff[common_cols], p2_pairs[common_cols]],
                       ignore_index=True)
    harm.to_csv(TABLES / "cross_pilot_harmonized_effect_sizes_v1.csv", index=False)

    # Inventory
    inv_rows = [
        {"pilot": "Pilot_1_1_HCC_holdout",
         "source": "Gurian_Bonifacio_2020 hcc_serum",
         "substrate": "Ag colloid (variant undocumented) → Ag_colloid_untyped",
         "n_total": 144, "comparisons": "P1_HCC_vs_CTR (72 vs 72)",
         "amplitude_offset_pct": 0.24,
         "key_finding": "normalization AMPLIFIES signal (raw d=0.26 → sumnorm d=0.57)"},
        {"pilot": "Pilot_2_1_CCA_HCC_LM_NC",
         "source": "label-free SERS nanosensor cca_hcc_lm_serum_sers",
         "substrate": "label-free SERS nanosensor (chemistry undocumented) → unknown_SERS",
         "n_total": 195,
         "comparisons": "P2_HCC_vs_NC (49v44), P2_CCA_vs_NC (58v44), P2_LM_vs_NC (44v44), P2_CCA_vs_HCC, P2_LM_vs_HCC, P2_CCA_vs_LM",
         "amplitude_offset_pct": "CCA +4.0%, LM +3.3%, HCC +0.1%",
         "key_finding": "normalization REVEALS biology hidden by amplitude offset (raw 11/11→sumnorm 9/11)"},
    ]
    pd.DataFrame(inv_rows).to_csv(TABLES / "cross_pilot_input_inventory_v1.csv", index=False)

    print(f"  Pilot 1.1 effects: {len(p1_eff)} rows")
    print(f"  Pilot 2.1 vs-NC effects: {len(p2_eff)} rows")
    print(f"  Pilot 2 pairwise effects (computed): {len(p2_pairs)} rows")
    print(f"  Total harmonized: {len(harm)} rows")

    lines = [
        "# Cross-Pilot Input Harmonization v1",
        "",
        "## Inputs",
        "",
        "| pilot | source | substrate | n_total | amplitude offset | finding |",
        "|---|---|---|---:|---|---|",
    ]
    for r in inv_rows:
        lines.append(f"| {r['pilot']} | {r['source']} | {r['substrate']} | "
                     f"{r['n_total']} | {r['amplitude_offset_pct']} | {r['key_finding']} |")
    lines += [
        "",
        "## Comparisons covered",
        "",
        "- P1_HCC_vs_CTR (Gurian Ag colloid)",
        "- P2_HCC_vs_NC, P2_CCA_vs_NC, P2_LM_vs_NC (label-free SERS nanosensor)",
        "- P2_CCA_vs_HCC, P2_LM_vs_HCC, P2_CCA_vs_LM (within-pilot pairwise; computed from normalized vectors)",
        "",
        "## Representations",
        "- abs (raw)",
        "- sumnorm (compositional)",
        "- clr (centered log-ratio)",
        "- delta_nc / delta_ctr (raw Δ vs control)",
        "- delta_sumnorm (sum-normalized Δ vs control)",
    ]
    (REPORTS / "REPORT_cross_pilot_input_harmonization_v1.md").write_text("\n".join(lines))
    return harm, p1_mass, p2_mass


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — interpretation rules
# ─────────────────────────────────────────────────────────────────────

def stage2_rules():
    lines = [
        "# Cross-Pilot Interpretation Rules v1",
        "",
        "## Hierarchy",
        "",
        "1. **Sum-normalized BSV** + **CLR** agreement → primary compositional biology layer",
        "2. **Raw BSV** → amplitude / context layer only (susceptible to per-cohort intensity offset)",
        "3. **ΔBSV (sum-normalized)** → reference-relative supporting signal",
        "4. **CI excluding zero** in at least one pilot → stronger evidence",
        "5. **Same-direction effect across pilots / cohorts** → highest confidence",
        "6. **Opposite-direction across pilots** → substrate or cohort-sensitive (do NOT generalize)",
        "",
        "## Effect-size thresholds",
        "",
        "- |d| < 0.15: noise; do not interpret",
        "- 0.15 ≤ |d| < 0.30: weak signal; report with caveat",
        "- |d| ≥ 0.30 with CI ✓ in normalized representation: meaningful evidence",
        "- |d| ≥ 0.50 with CI ✓ in two pilots: high-confidence cross-pilot signal",
        "",
        "## Forbidden",
        "",
        "- No diagnostic-accuracy claims",
        "- No exact molecule identity",
        "- No clinical decision claims",
        "- No claim of subtype discrimination from passive BSV alone",
    ]
    (REPORTS / "REPORT_cross_pilot_interpretation_rules_v1.md").write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — family consensus
# ─────────────────────────────────────────────────────────────────────

def stage3_consensus(harm):
    print("\n[STAGE 3] Cross-pilot family consensus")
    # Focus on sum-normalized layer for consensus
    sn = harm[harm.representation == "sumnorm"].copy()
    rows = []
    for fam in BSV_GROUPS_ORDER:
        sub = sn[sn.family == fam]
        # Per-comparison
        d = {r["comparison"]: r["cohens_d"] for _, r in sub.iterrows()}
        ci = {r["comparison"]: r["ci_excludes_zero"] for _, r in sub.iterrows()}
        # Disease-vs-control comparisons
        vs_ctl = ["P1_HCC_vs_CTR", "P2_HCC_vs_NC", "P2_CCA_vs_NC", "P2_LM_vs_NC"]
        signs = [np.sign(d.get(c, 0)) for c in vs_ctl if c in d and not np.isnan(d.get(c, np.nan))]
        same_sign = (all(s > 0 for s in signs) or all(s < 0 for s in signs)) and len(signs) >= 2
        ci_count = sum(int(ci.get(c, False)) for c in vs_ctl if c in ci)
        max_d = max(abs(d.get(c, 0)) for c in vs_ctl if c in d) if signs else 0.0

        # Categorize
        cca_d = d.get("P2_CCA_vs_NC", 0); lm_d = d.get("P2_LM_vs_NC", 0)
        hcc_p1 = d.get("P1_HCC_vs_CTR", 0); hcc_p2 = d.get("P2_HCC_vs_NC", 0)

        # Advanced-cancer signal: large in CCA + LM but small in either HCC
        is_advanced = (abs(cca_d) >= 0.5 and abs(lm_d) >= 0.5
                         and (np.sign(cca_d) == np.sign(lm_d))
                         and abs(hcc_p2) < 0.2)
        # Cross-pilot consistent: same direction in P1 HCC + P2 HCC AND P2 CCA/LM
        hcc_consistent = (np.sign(hcc_p1) == np.sign(hcc_p2)) and abs(hcc_p1) >= 0.3
        cross_consistent = (np.sign(hcc_p1) == np.sign(cca_d) == np.sign(lm_d)) and abs(hcc_p1) >= 0.3 and abs(cca_d) >= 0.5
        # Substrate-sensitive: opposite direction P1 vs P2 cancer cohorts
        substrate_sens = (abs(hcc_p1) >= 0.3 and abs(cca_d) >= 0.5
                            and np.sign(hcc_p1) != np.sign(cca_d))

        # Priority order matters: substrate-sensitivity (sign flip P1↔P2) must be
        # checked BEFORE advanced-cancer to avoid mislabeling sign-flipping families.
        if substrate_sens:
            cat = "SUBSTRATE_OR_COHORT_SENSITIVE"
        elif cross_consistent:
            cat = "CONSISTENT_CROSS_PILOT_SIGNAL"
        elif is_advanced:
            cat = "ADVANCED_CANCER_SIGNAL"
        elif hcc_consistent:
            cat = "DISEASE_SUBTYPE_SPECIFIC_SIGNAL"
        elif max_d < 0.15:
            cat = "NO_SIGNAL"
        else:
            cat = "WEAK_OR_INCONSISTENT"

        rows.append({
            "family": fam, "family_label": FAMILY_LABELS.get(fam, fam),
            "P1_HCC_vs_CTR_d_sn": round(hcc_p1, 3),
            "P2_HCC_vs_NC_d_sn": round(hcc_p2, 3),
            "P2_CCA_vs_NC_d_sn": round(cca_d, 3),
            "P2_LM_vs_NC_d_sn": round(lm_d, 3),
            "max_abs_d_disease_vs_control": round(max_d, 3),
            "n_disease_vs_control_with_CI": ci_count,
            "category": cat,
        })
    cons = pd.DataFrame(rows)
    cons.to_csv(TABLES / "cross_pilot_family_consensus_v1.csv", index=False)

    print("  Family consensus categories:")
    for _, r in cons.iterrows():
        print(f"    {r['family']:5s} {r['family_label']:14s}  "
              f"P1HCC={r['P1_HCC_vs_CTR_d_sn']:+.2f}  P2HCC={r['P2_HCC_vs_NC_d_sn']:+.2f}  "
              f"P2CCA={r['P2_CCA_vs_NC_d_sn']:+.2f}  P2LM={r['P2_LM_vs_NC_d_sn']:+.2f}  → {r['category']}")

    cat_counts = cons["category"].value_counts().to_dict()
    lines = [
        "# Cross-Pilot Family Consensus v1",
        "",
        "## Per-family classification (sum-normalized layer)",
        "",
        "| family | P1_HCC d | P2_HCC d | P2_CCA d | P2_LM d | max |d| | CI count | category |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in cons.iterrows():
        lines.append(
            f"| **{r['family']}** {r['family_label']} | "
            f"{r['P1_HCC_vs_CTR_d_sn']:+.2f} | {r['P2_HCC_vs_NC_d_sn']:+.2f} | "
            f"{r['P2_CCA_vs_NC_d_sn']:+.2f} | {r['P2_LM_vs_NC_d_sn']:+.2f} | "
            f"{r['max_abs_d_disease_vs_control']:.2f} | {r['n_disease_vs_control_with_CI']} | "
            f"**{r['category']}** |"
        )
    lines += [
        "",
        "## Category counts",
        "",
    ]
    for k, v in cat_counts.items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Category definitions",
        "",
        "- **CONSISTENT_CROSS_PILOT_SIGNAL**: same direction across P1 HCC + P2 HCC + P2 CCA + P2 LM with meaningful magnitudes (P1 |d|≥0.3, P2 |d|≥0.5).",
        "- **ADVANCED_CANCER_SIGNAL**: large + same-direction in CCA + LM but small in HCC — signal scales with advanced disease state, not present in earlier-stage HCC.",
        "- **SUBSTRATE_OR_COHORT_SENSITIVE**: opposite direction between Pilot 1 (Gurian Ag) and Pilot 2 (label-free SERS nanosensor) — substrate or cohort-driven sign flip.",
        "- **DISEASE_SUBTYPE_SPECIFIC_SIGNAL**: same direction in P1 HCC + P2 HCC but distinct from CCA/LM behavior.",
        "- **WEAK_OR_INCONSISTENT**: signal exists but not coherent across pilots.",
        "- **NO_SIGNAL**: max |d| < 0.15 across all pilots.",
    ]
    (REPORTS / "REPORT_cross_pilot_family_consensus_v1.md").write_text("\n".join(lines))
    return cons


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — Cohort biochemical state map
# ─────────────────────────────────────────────────────────────────────

def stage4_state_map(harm):
    print("\n[STAGE 4] Cohort biochemical state map")
    sn = harm[harm.representation == "sumnorm"]
    state_rows = []
    cohorts = [
        ("HCC_Pilot1", "P1_HCC_vs_CTR", "Gurian Ag colloid (untyped)", "n=72 vs 72"),
        ("HCC_Pilot2", "P2_HCC_vs_NC",  "label-free SERS nanosensor", "n=49 vs 44"),
        ("CCA_Pilot2", "P2_CCA_vs_NC",  "label-free SERS nanosensor", "n=58 vs 44"),
        ("LM_Pilot2",  "P2_LM_vs_NC",   "label-free SERS nanosensor", "n=44 vs 44"),
    ]
    for cohort_name, comp, sub_str, ns in cohorts:
        sub = sn[sn.comparison == comp].sort_values("cohens_d", ascending=False)
        elevated = sub[sub.cohens_d > 0.30]
        depleted = sub[sub.cohens_d < -0.30]
        ci_pos = sub[sub["ci_excludes_zero"] == True]
        state_rows.append({
            "cohort": cohort_name,
            "substrate": sub_str,
            "n": ns,
            "elevated_families_d_ge_03": ";".join(f"{r['family']}({r['cohens_d']:+.2f})"
                                                       for _, r in elevated.iterrows()),
            "depleted_families_d_le_neg03": ";".join(f"{r['family']}({r['cohens_d']:+.2f})"
                                                          for _, r in depleted.iterrows()),
            "max_abs_d": round(float(sub["abs_d"].max()), 2),
            "n_ci_significant": int(len(ci_pos)),
            "confidence_level": ("HIGH" if len(ci_pos) >= 4 else
                                   "MEDIUM" if len(ci_pos) >= 1 else "LOW"),
        })
    smap = pd.DataFrame(state_rows)
    smap.to_csv(TABLES / "cross_pilot_cohort_state_map_v1.csv", index=False)

    lines = [
        "# Cross-Pilot Cohort Biochemical State Map v1",
        "",
        "## Per-cohort sum-normalized BSV signature",
        "",
        "| cohort | substrate | n | elevated (|d|≥0.30) | depleted (|d|≥0.30) | max |d| | CI-sig | confidence |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for _, r in smap.iterrows():
        lines.append(
            f"| **{r['cohort']}** | {r['substrate']} | {r['n']} | "
            f"{r['elevated_families_d_ge_03'] or '—'} | "
            f"{r['depleted_families_d_le_neg03'] or '—'} | "
            f"{r['max_abs_d']} | {r['n_ci_significant']}/11 | **{r['confidence_level']}** |"
        )
    lines += [
        "",
        "## Substrate / cohort caveats",
        "",
        "- HCC_Pilot1: Gurian Ag colloid; substrate variant unconfirmed (Ag_colloid_untyped block, inference OFF).",
        "- HCC_Pilot2 / CCA_Pilot2 / LM_Pilot2: label-free SERS nanosensor; substrate type unknown (unknown_SERS block, inference OFF).",
        "- All cohorts: SENSITIVE-tier output policy applies; no diagnostic claims.",
    ]
    (REPORTS / "REPORT_cross_pilot_cohort_state_map_v1.md").write_text("\n".join(lines))
    return smap


# ─────────────────────────────────────────────────────────────────────
# Stage 5 — Robust vs caution signals
# ─────────────────────────────────────────────────────────────────────

def stage5_robust_vs_caution(cons, harm):
    print("\n[STAGE 5] Robust vs caution signals")
    rows = []
    for _, r in cons.iterrows():
        cat = r["category"]
        fam = r["family"]
        # Get CI-significance counts in disease-vs-control sumnorm
        sn = harm[(harm.representation == "sumnorm") & (harm.family == fam) &
                    (harm.comparison.isin(["P1_HCC_vs_CTR", "P2_HCC_vs_NC", "P2_CCA_vs_NC", "P2_LM_vs_NC"]))]
        ci_count = int(sn["ci_excludes_zero"].sum())
        n_meaningful = int((sn["abs_d"] >= 0.30).sum())
        n_consistent = int((sn["cohens_d"] > 0).sum() if (sn["cohens_d"] > 0).sum() > (sn["cohens_d"] < 0).sum() else (sn["cohens_d"] < 0).sum())

        # Robust = consistent direction + meaningful in ≥2 cohorts + CI in ≥1
        is_robust = (cat in ("CONSISTENT_CROSS_PILOT_SIGNAL", "ADVANCED_CANCER_SIGNAL")
                       and n_meaningful >= 2 and ci_count >= 1)
        # Caution = substrate-sensitive OR weak/inconsistent OR subtype-specific
        is_caution = cat in ("SUBSTRATE_OR_COHORT_SENSITIVE", "DISEASE_SUBTYPE_SPECIFIC_SIGNAL")

        rows.append({
            "family": fam, "family_label": r["family_label"],
            "category": cat,
            "n_meaningful_cohorts": n_meaningful,
            "n_ci_significant_cohorts": ci_count,
            "is_robust": is_robust,
            "is_caution": is_caution,
            "tier": ("ROBUST" if is_robust else
                       "CAUTION" if is_caution else
                       ("WEAK" if cat == "WEAK_OR_INCONSISTENT" else "NO_SIGNAL")),
            "summary": _signal_summary(cat, r, fam),
        })
    rs_df = pd.DataFrame(rows)
    rs_df.to_csv(TABLES / "cross_pilot_robust_vs_caution_signals_v1.csv", index=False)

    print("  signal tiers:")
    print(rs_df["tier"].value_counts().to_string())

    lines = [
        "# Cross-Pilot Robust vs Caution Signals v1",
        "",
        "## ROBUST signals",
        "",
        "Survives normalization, CI-supported in ≥1 pilot, same direction across multiple cohorts.",
        "",
        "| family | category | n_meaningful | n_CI | summary |",
        "|---|---|---:|---:|---|",
    ]
    for _, r in rs_df[rs_df["is_robust"]].iterrows():
        lines.append(f"| **{r['family']}** {r['family_label']} | {r['category']} | "
                     f"{r['n_meaningful_cohorts']}/4 | {r['n_ci_significant_cohorts']}/4 | "
                     f"{r['summary']} |")
    lines += [
        "",
        "## CAUTION signals",
        "",
        "Direction flips between pilots OR signal exists in only one cohort/pilot.",
        "",
        "| family | category | n_meaningful | n_CI | summary |",
        "|---|---|---:|---:|---|",
    ]
    for _, r in rs_df[rs_df["is_caution"]].iterrows():
        lines.append(f"| {r['family']} {r['family_label']} | {r['category']} | "
                     f"{r['n_meaningful_cohorts']}/4 | {r['n_ci_significant_cohorts']}/4 | "
                     f"{r['summary']} |")
    lines += [
        "",
        "## WEAK / NO_SIGNAL families",
        "",
        "| family | category | tier | summary |",
        "|---|---|---|---|",
    ]
    for _, r in rs_df[rs_df["tier"].isin(["WEAK", "NO_SIGNAL"])].iterrows():
        lines.append(f"| {r['family']} {r['family_label']} | {r['category']} | {r['tier']} | "
                     f"{r['summary']} |")
    (REPORTS / "REPORT_cross_pilot_robust_vs_caution_signals_v1.md").write_text("\n".join(lines))
    return rs_df


def _signal_summary(cat, row, fam):
    p1 = row["P1_HCC_vs_CTR_d_sn"]
    cca = row["P2_CCA_vs_NC_d_sn"]
    lm = row["P2_LM_vs_NC_d_sn"]
    if cat == "ADVANCED_CANCER_SIGNAL":
        direction = "elevated" if cca > 0 else "depleted"
        return f"{direction} in CCA ({cca:+.2f}) and LM ({lm:+.2f}); near-zero in HCC P2"
    if cat == "CONSISTENT_CROSS_PILOT_SIGNAL":
        return f"consistent across pilots: P1 HCC ({p1:+.2f}), P2 CCA ({cca:+.2f}), P2 LM ({lm:+.2f})"
    if cat == "SUBSTRATE_OR_COHORT_SENSITIVE":
        return f"OPPOSITE: P1 HCC {p1:+.2f}, P2 CCA {cca:+.2f} (sign flips between substrates)"
    if cat == "DISEASE_SUBTYPE_SPECIFIC_SIGNAL":
        return f"P1 HCC {p1:+.2f}, P2 HCC {row['P2_HCC_vs_NC_d_sn']:+.2f} same direction"
    return f"weak / not interpretable; max |d|={row['max_abs_d_disease_vs_control']}"


# ─────────────────────────────────────────────────────────────────────
# Stage 6 — Figures
# ─────────────────────────────────────────────────────────────────────

def stage6_figures(harm, cons, smap, rs_df):
    print("\n[STAGE 6] Figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal_cohort = {"P1_HCC_vs_CTR": "#9467bd",
                   "P2_HCC_vs_NC":  "#d62728",
                   "P2_CCA_vs_NC":  "#ff7f0e",
                   "P2_LM_vs_NC":   "#2ca02c"}

    # 1. Cross-pilot sumnorm effect-size heatmap
    sn = harm[harm.representation == "sumnorm"]
    comps = ["P1_HCC_vs_CTR", "P2_HCC_vs_NC", "P2_CCA_vs_NC", "P2_LM_vs_NC"]
    pivot = sn[sn.comparison.isin(comps)].pivot(
        index="family", columns="comparison", values="cohens_d"
    ).reindex(BSV_GROUPS_ORDER)[comps]
    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = float(np.abs(pivot.values).max()) or 0.5
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(BSV_GROUPS_ORDER)))
    ax.set_yticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER])
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, rotation=20, ha="right", fontsize=9)
    ax.set_title("Cross-pilot sum-normalized Cohen's d (disease vs control)")
    # CI markers
    ci_pivot = sn[sn.comparison.isin(comps)].pivot(
        index="family", columns="comparison", values="ci_excludes_zero"
    ).reindex(BSV_GROUPS_ORDER)[comps]
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            ci_mark = " *" if bool(ci_pivot.iloc[i, j]) else ""
            ax.text(j, i, f"{v:+.2f}{ci_mark}", ha="center", va="center", fontsize=8,
                     color="white" if abs(v) > vmax*0.55 else "black")
    fig.colorbar(im, ax=ax, label="Cohen's d (sumnorm)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_cross_pilot_sumnorm_effect_heatmap_v1.png", dpi=150)
    plt.close(fig)

    # 2. CLR heatmap
    cl = harm[harm.representation == "clr"]
    pivot = cl[cl.comparison.isin(comps)].pivot(
        index="family", columns="comparison", values="cohens_d"
    ).reindex(BSV_GROUPS_ORDER)[comps]
    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = float(np.abs(pivot.values).max()) or 0.5
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(BSV_GROUPS_ORDER)))
    ax.set_yticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER])
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, rotation=20, ha="right", fontsize=9)
    ax.set_title("Cross-pilot CLR Cohen's d (disease vs control)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                     color="white" if abs(v) > vmax*0.55 else "black")
    fig.colorbar(im, ax=ax, label="Cohen's d (CLR)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_cross_pilot_clr_effect_heatmap_v1.png", dpi=150)
    plt.close(fig)

    # 3. Robust signal summary
    robust = rs_df[rs_df["is_robust"]]
    fig, ax = plt.subplots(figsize=(11, 4))
    if len(robust):
        for fam in robust["family"]:
            sub = sn[(sn.family == fam) & (sn.comparison.isin(comps))]
            xs = np.arange(len(comps))
            ds = [float(sub[sub.comparison == c]["cohens_d"].iloc[0]) if len(sub[sub.comparison == c]) else 0
                   for c in comps]
            ax.plot(xs, ds, marker="o", linewidth=2,
                     label=f"{fam} {FAMILY_LABELS.get(fam, fam)}")
        ax.set_xticks(np.arange(len(comps))); ax.set_xticklabels(comps, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Cohen's d (sumnorm)"); ax.axhline(0, color="k", lw=0.5)
        ax.set_title("ROBUST cross-pilot signals — same direction in ≥2 cohorts")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No robust signals identified", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_cross_pilot_robust_signal_summary_v1.png", dpi=150)
    plt.close(fig)

    # 4. Cohort biochemical radar (sumnorm means per cohort)
    angles = np.linspace(0, 2*np.pi, len(BSV_GROUPS_ORDER), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for comp in comps:
        sub = sn[sn.comparison == comp]
        # Use cohens_d per family directly as the "shift profile"
        vals = []
        for g in BSV_GROUPS_ORDER:
            row = sub[sub.family == g]
            vals.append(float(row["cohens_d"].iloc[0]) if len(row) else 0)
        vals += vals[:1]
        ax.plot(angles, vals, label=comp, color=pal_cohort[comp], linewidth=1.6)
        ax.fill(angles, vals, alpha=0.07, color=pal_cohort[comp])
    ax.plot(angles, [0]*len(angles), color="k", linewidth=0.6, linestyle="--")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], fontsize=9)
    ax.set_title("Cross-pilot disease-vs-control sum-normalized Cohen's d radar", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_cross_pilot_normalized_bsv_radar_v1.png", dpi=180)
    plt.close(fig)

    # 5. Direction-consistency plot (per family direction across 4 comps)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(BSV_GROUPS_ORDER))
    width = 0.2
    for i, comp in enumerate(comps):
        sub = sn[sn.comparison == comp]
        ds = []
        for g in BSV_GROUPS_ORDER:
            row = sub[sub.family == g]
            ds.append(float(row["cohens_d"].iloc[0]) if len(row) else 0)
        ax.bar(x + (i - 1.5) * width, ds, width, label=comp, color=pal_cohort[comp])
    ax.set_xticks(x); ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Cohen's d (sumnorm)")
    ax.set_title("Direction consistency per family across cohort comparisons")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_cross_pilot_direction_consistency_v1.png", dpi=150)
    plt.close(fig)

    # 6. Raw vs normalized comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    raw = harm[(harm.representation == "abs") & (harm.comparison.isin(comps))]
    raw_pivot = raw.pivot(index="family", columns="comparison", values="cohens_d").reindex(BSV_GROUPS_ORDER)[comps]
    sn_pivot = sn[sn.comparison.isin(comps)].pivot(index="family", columns="comparison", values="cohens_d").reindex(BSV_GROUPS_ORDER)[comps]
    vmax = max(float(np.abs(raw_pivot.values).max()), float(np.abs(sn_pivot.values).max()))
    for ax_, pv, title in zip(axes, [raw_pivot, sn_pivot], ["RAW (abs)", "SUM-NORMALIZED"]):
        im = ax_.imshow(pv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax_.set_yticks(range(len(BSV_GROUPS_ORDER)))
        ax_.set_yticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER])
        ax_.set_xticks(range(len(comps)))
        ax_.set_xticklabels(comps, rotation=30, ha="right", fontsize=8)
        ax_.set_title(title)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.iloc[i, j]
                ax_.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                          color="white" if abs(v) > vmax*0.5 else "black")
        fig.colorbar(im, ax=ax_, fraction=0.05)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_cross_pilot_raw_vs_normalized_v1.png", dpi=150)
    plt.close(fig)

    # 7. Amplitude offset summary
    fig, ax = plt.subplots(figsize=(8, 4))
    cohort_lbl = ["P1\nHCC", "P2\nHCC", "P2\nCCA", "P2\nLM"]
    pcts = [0.24, 0.10, 4.00, 3.30]
    colors = ["#9467bd", "#d62728", "#ff7f0e", "#2ca02c"]
    ax.bar(cohort_lbl, pcts, color=colors)
    ax.set_ylabel("amplitude offset vs control cohort (%)")
    ax.set_title("Per-cohort BSV mass offset — amplitude artifact magnitude")
    for i, v in enumerate(pcts):
        ax.text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_cross_pilot_amplitude_offsets_v1.png", dpi=150)
    plt.close(fig)

    # 8. Interpretation schematic
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    ax.text(0.05, 0.85, "GAIRA passive target interpretation pipeline",
             fontsize=14, fontweight="bold")
    boxes = [
        (0.05, 0.55, "raw spectrum"),
        (0.30, 0.55, "v4.5 hybrid pipeline\n(motif → MSS → BSV)"),
        (0.58, 0.55, "11-axis BSV\nper spectrum"),
        (0.05, 0.25, "raw BSV =\namplitude/context"),
        (0.30, 0.25, "sum-normalized\n+ CLR"),
        (0.58, 0.25, "robust signals\n(G09 ↓, G04 ↑)"),
        (0.78, 0.25, "caution signals\n(G05, G03)"),
    ]
    for x, y, text in boxes:
        ax.text(x, y, text, fontsize=10, ha="left", va="center",
                  bbox=dict(boxstyle="round", facecolor="#cce", edgecolor="black"))
    # Arrows
    ax.annotate("", xy=(0.30, 0.55), xytext=(0.18, 0.55),
                  arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(0.58, 0.55), xytext=(0.50, 0.55),
                  arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(0.18, 0.32), xytext=(0.18, 0.50),
                  arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(0.42, 0.32), xytext=(0.42, 0.50),
                  arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(0.58, 0.25), xytext=(0.46, 0.25),
                  arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(0.78, 0.25), xytext=(0.46, 0.25),
                  arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.05, 0.05, "Interpretation rule: normalized BSV / CLR is the chemistry layer; "
                          "raw BSV is amplitude/context only. Substrate caveat applies throughout.",
              fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_gaira_passive_target_interpretation_schematic_v1.png", dpi=150)
    plt.close(fig)

    print("  8 figures emitted")


# ─────────────────────────────────────────────────────────────────────
# Stage 7 — Cross-pilot interpretation
# ─────────────────────────────────────────────────────────────────────

def stage7_interpretation(rs_df, smap):
    print("\n[STAGE 7] Cross-pilot interpretation")
    robust = rs_df[rs_df["is_robust"]]
    caution = rs_df[rs_df["is_caution"]]

    lines = [
        "# Cross-Pilot Biochemical Interpretation v1 (CAUTIOUS)",
        "",
        "## What signals replicate across pilots?",
        "",
    ]
    if len(robust):
        for _, r in robust.iterrows():
            lines.append(f"- **{r['family']} {r['family_label']}** ({r['category']}) — {r['summary']}")
    else:
        lines.append("- No signals fully meet the robust threshold (≥2 cohorts meaningful + CI in ≥1 pilot).")
    lines += [
        "",
        "## What signals are HCC-specific vs advanced-cancer-like?",
        "",
        "- **HCC-specific signature (Pilot 1 HCC vs CTR, sum-normalized)**:",
        "  - elevated G05 Glycan + G04 Nucl-bbone",
        "  - depleted G09 Sterol-lipid + G03 Pyrimidine",
        "  - HCC Pilot 2 weak — does NOT reproduce the Pilot 1 specific axes (different substrate, possibly different disease state)",
        "- **Advanced-cancer signature (CCA + LM in Pilot 2)**:",
        "  - elevated G03 Pyrimidine + G06 Protein + G07 Aromatic + G01 Purine-nuc + G04 Nucl-bbone",
        "  - depleted G09 Sterol-lipid + G05 Glycan + G08 Lipid-acyl + G11 Metab-SM",
        "  - shared chemistry of advanced liver malignancy regardless of HCC/CCA/LM subtype",
        "",
        "## What signals are substrate / cohort-sensitive?",
        "",
    ]
    for _, r in caution.iterrows():
        lines.append(f"- **{r['family']} {r['family_label']}** — {r['summary']}")
    lines += [
        "",
        "## What did normalization reveal?",
        "",
        "- **Pilot 2 raw 11/11 elevated pattern was MOSTLY real biology**, not artifact. After sum-normalization 9/11 of the CCA effects + 9/11 of the LM effects survive as CI-significant. The 2 collapsing axes (G10, G02) were the amplitude-driven illusions.",
        "- **Pilot 1 raw modest signal was UNDER-amplified** by within-cohort amplitude noise. Sum-normalization doubled the effect sizes (raw d=0.26 → sumnorm d=0.57 on G05) and revealed 4 CI-significant axes that raw analysis missed.",
        "- **Same normalization tool, opposite directions** of correction across pilots. Both pilots' chemistry signal becomes interpretable only after compositional normalization.",
        "",
        "## What should GAIRA report as ROBUST?",
        "",
    ]
    for _, r in robust.iterrows():
        lines.append(f"- **{r['family']} {r['family_label']}** — {r['summary']}")
    if not len(robust):
        lines.append("- (no signals meeting robust threshold)")
    lines += [
        "",
        "## What should GAIRA report only with CAUTION?",
        "",
    ]
    for _, r in caution.iterrows():
        lines.append(f"- **{r['family']} {r['family_label']}** — {r['summary']}")
    lines += [
        "",
        "## What should NOT be claimed",
        "",
        "- Diagnostic separation of HCC, CCA, LM from healthy via passive BSV alone.",
        "- Subtype discrimination (HCC vs CCA vs LM) at clinical accuracy.",
        "- Exact molecule identity from any single elevated/depleted axis.",
        "- Generalization beyond the two substrate families tested (Gurian Ag colloid, label-free SERS nanosensor).",
        "- That early-stage HCC produces a strong BSV-level signature (it does not in either pilot).",
        "",
        "## Substrate / cohort caveats",
        "",
        "All Pilot 1 + Pilot 2 datasets used substrate blocks `Ag_colloid_untyped` or `unknown_SERS` — the engine ran with substrate-specific physics OFF for inference. This is the conservative correct behaviour given the substrate documentation gap. Passive readout output tier = SENSITIVE for all cohorts.",
        "",
        "## Convergent finding (cleanest cross-pilot signal)",
        "",
        "**G09 Sterol-lipid depletion in serum** is observed in:",
        "- Pilot 1 HCC vs CTR (Gurian Ag colloid) — d_sumnorm = −0.46 (CI ✓)",
        "- Pilot 2 CCA vs NC (label-free SERS) — d_sumnorm = −1.39 (CI ✓)",
        "- Pilot 2 LM vs NC (label-free SERS) — d_sumnorm = −1.19 (CI ✓)",
        "",
        "Three cohorts, two substrates, same direction — **highest-confidence cross-pilot biochemical pattern in GAIRA passive target readouts to date**.",
    ]
    (REPORTS / "REPORT_cross_pilot_biochemical_interpretation_v1.md").write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# Stage 8 — Readiness decision
# ─────────────────────────────────────────────────────────────────────

def stage8_readiness(rs_df, smap):
    print("\n[STAGE 8] Synthesis readiness")
    n_robust = int(rs_df["is_robust"].sum())
    n_caution = int(rs_df["is_caution"].sum())
    n_high_conf = int((smap["confidence_level"] == "HIGH").sum())

    if n_robust >= 2 and n_high_conf >= 2:
        decision = "READY_FOR_PASSIVE_GAIRA_DEMO_REPORT"
    elif n_robust >= 1 or n_high_conf >= 1:
        decision = "READY_WITH_SUBSTRATE_COHORT_CAVEATS"
    elif n_caution >= 4:
        decision = "NEEDS_MORE_TARGET_QC"
    else:
        decision = "NEEDS_MORE_COHORTS_BEFORE_SYNTHESIS"

    lines = [
        "# Cross-Pilot Synthesis Readiness v1",
        "",
        f"**Decision: {decision}**",
        "",
        "## Headline counts",
        "",
        f"- ROBUST signals: **{n_robust}/11**",
        f"- CAUTION signals: **{n_caution}/11**",
        f"- HIGH-confidence cohort state maps: **{n_high_conf}/4**",
        "",
        "## Suitable use levels",
        "",
    ]
    if decision == "READY_FOR_PASSIVE_GAIRA_DEMO_REPORT":
        lines.append(
            "- **GAIRA demo**: yes — the cross-pilot signature is interpretable.\n"
            "- **Manuscript-style results**: yes, with mandatory substrate caveats and sum-normalisation reporting.\n"
            "- **Internal validation**: yes."
        )
    elif decision == "READY_WITH_SUBSTRATE_COHORT_CAVEATS":
        lines.append(
            "- **GAIRA demo**: yes, with substrate / cohort caveats prominently displayed.\n"
            "- **Manuscript-style results**: yes, scoped to the specific substrate families tested.\n"
            "- **Internal validation**: yes."
        )
    elif decision == "NEEDS_MORE_TARGET_QC":
        lines.append(
            "- **GAIRA demo**: not yet — caution signals dominate.\n"
            "- **Manuscript**: not yet — substrate / cohort sensitivity needs more replication.\n"
            "- **Internal validation**: yes for engineering only."
        )
    else:
        lines.append(
            "- More cohorts required before any external use.\n"
            "- Internal-only at this stage."
        )
    lines += [
        "",
        "## Invariants preserved",
        "",
        "- Engine v4.5 / taxonomy / motif / MSS v4.3 / substrate physics v1.2: read-only",
        "- No classifier training, no threshold tuning, no label-driven feature selection",
        "- No target-label fitting",
        "- No DART-Met",
    ]
    (REPORTS / "REPORT_cross_pilot_synthesis_readiness_v1.md").write_text("\n".join(lines))
    return decision, n_robust, n_caution, n_high_conf


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_4_cross_pilot_synthesis_v1")
    print("=" * 78)
    for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    harm, p1_mass, p2_mass = stage1_load()
    stage2_rules()
    cons = stage3_consensus(harm)
    smap = stage4_state_map(harm)
    rs_df = stage5_robust_vs_caution(cons, harm)
    stage6_figures(harm, cons, smap, rs_df)
    stage7_interpretation(rs_df, smap)
    decision, n_robust, n_caution, n_high_conf = stage8_readiness(rs_df, smap)

    # Audit log
    lines = [
        "# gaira_base_4_cross_pilot_synthesis_v1 — Audit Log",
        "",
        "## Inputs",
        f"- Pilot 1.1: {P1_DIR}",
        f"- Pilot 2.1: {P2_DIR}",
        "",
        "## Harmonized comparisons",
        "- P1_HCC_vs_CTR (Gurian Ag colloid)",
        "- P2_HCC_vs_NC, P2_CCA_vs_NC, P2_LM_vs_NC (label-free SERS nanosensor)",
        "- P2_CCA_vs_HCC, P2_LM_vs_HCC, P2_CCA_vs_LM (within-pilot pairwise; computed)",
        "",
        "## Outputs",
        "- 5 tables (harmonized effect sizes, input inventory, consensus, cohort state map, robust vs caution)",
        "- 8 figures (sumnorm heatmap, CLR heatmap, robust signal summary, normalized radar, direction consistency, raw vs normalized, amplitude offsets, interpretation schematic)",
        "- 7 reports (input harmonization, interpretation rules, family consensus, cohort state map, robust vs caution, biochemical interpretation, synthesis readiness)",
        "",
        f"## Counts: ROBUST={n_robust}/11, CAUTION={n_caution}/11, HIGH_conf cohorts={n_high_conf}/4",
        "",
        f"## Final decision: **{decision}**",
        "",
        "## Invariants",
        "- engine v4.5 / taxonomy / motif / MSS / substrate physics: unchanged",
        "- no classifier training, no threshold tuning, no label-driven feature select",
        "- no target-label fitting",
        "- no DART-Met",
    ]
    (AUDIT / "gaira_base_4_cross_pilot_synthesis_v1_audit_log.md").write_text("\n".join(lines))

    p = Path(__file__)
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)

    print(f"\n[complete] decision: {decision}")
    print(f"  ROBUST={n_robust}/11  CAUTION={n_caution}/11  HIGH-conf cohorts={n_high_conf}/4")


if __name__ == "__main__":
    main()
