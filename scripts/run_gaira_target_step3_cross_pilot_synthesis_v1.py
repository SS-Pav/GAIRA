"""GAIRA Target Step 3 — canonical cross-pilot synthesis v1.

Strictly comparable synthesis using ONLY:
  - Pilot 1 (HCC holdout, canonical, STRICTLY_COMPARABLE)
  - Pilot 2b (CCA raw, canonical, STRICTLY_COMPARABLE)
  - Step 2 axis reliability matrix (locked v1)

No pooling. No cross-dataset normalization. No fallback pilots as primary
evidence. Tier 3 axes are surfaced but never used as primary claims.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_target_step3_cross_pilot_synthesis_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral.window_panel import BSV_COMPONENTS


# ──────────────────────────────────────────────────────────────────────
# Paths (locked inputs)
# ──────────────────────────────────────────────────────────────────────

PILOT1_TABLES  = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_target_pilot1_hcc_holdout_bsv/tables")
PILOT2B_TABLES = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/pilot2b_cca_raw/tables")
STEP2_DIR      = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/step2_axis_reliability_v1")
POLICY_YAML    = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/config/gaira_target_pipeline_policy_v1.yaml")

OUT_ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/step3_cross_pilot_synthesis_v1")


BSV_SHORT = {
    "membrane_lipid": "Lipid", "protein_backbone": "Protein",
    "aromatic_amino_acid": "Aromatic AA", "purine_nucleotide": "Purine",
    "pyrimidine_nucleotide": "Pyrimidine", "glycan_carbohydrate": "Glycan",
    "redox_metabolite": "Redox", "nucleic_acid_backbone": "Nuc.Backbone",
}
TIER_COLOR = {
    "TIER_1_ROBUST":    "#4ADE80",
    "TIER_2_CONTEXTUAL": "#FBBF24",
    "TIER_3_UNSTABLE":  "#F87171",
}
USE_COLOR = {
    "PRIMARY_CROSS_PILOT":    "#059669",
    "SECONDARY_CONTEXT_ONLY": "#D97706",
    "EXCLUDE_PRIMARY_CLAIMS": "#B91C1C",
}


# ──────────────────────────────────────────────────────────────────────
# Load canonical inputs + policy gate
# ──────────────────────────────────────────────────────────────────────

def _direction(d: float) -> str:
    if d > 0.002:
        return "up"
    if d < -0.002:
        return "down"
    return "flat"


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a) * len(b)
    if n == 0:
        return 0.0
    greater = int(sum((ai > b).sum() for ai in a))
    less = int(sum((ai < b).sum() for ai in a))
    return float((greater - less) / n)


def load_inputs():
    # Policy gate
    policy = yaml.safe_load(POLICY_YAML.read_text())
    required_version = policy["policy"]["version"]
    assert required_version == "v1", f"unexpected policy version {required_version}"

    # Pilot 1 effect sizes (wide format). Primary contrast: HCC vs CTR.
    p1_eff = pd.read_csv(PILOT1_TABLES / "pilot1_hcc_axis_effect_sizes.csv").set_index("axis")

    # Pilot 2b effect sizes (long format; filter to compare_class='cca').
    p2b_eff_long = pd.read_csv(PILOT2B_TABLES / "pilot2b_cca_raw_axis_effect_sizes.csv")
    p2b_eff = p2b_eff_long[p2b_eff_long["compare_class"] == "cca"].set_index("axis")

    # Step 2 reliability matrix.
    rel = pd.read_csv(STEP2_DIR / "axis_reliability_matrix.csv")

    # Distances-to-centroid per-spectrum (for Theme 5).
    p1_dist_df = pd.read_csv(PILOT1_TABLES / "pilot1_hcc_per_spectrum_delta_bsv.csv")
    p2b_dist_df = pd.read_csv(PILOT2B_TABLES / "pilot2b_cca_raw_per_spectrum_delta_bsv.csv")
    p1_dist_hc  = p1_dist_df[p1_dist_df["class"] == "healthy_control"]["distance_to_healthy_centroid"].to_numpy()
    p1_dist_dis = p1_dist_df[p1_dist_df["class"] == "hcc"]["distance_to_healthy_centroid"].to_numpy()
    p2b_dist_hc = p2b_dist_df[p2b_dist_df["class"] == "healthy_control"]["distance_to_healthy_centroid"].to_numpy()
    p2b_dist_dis = p2b_dist_df[p2b_dist_df["class"] == "cca"]["distance_to_healthy_centroid"].to_numpy()

    distances = {
        "p1_hc_median":  float(np.median(p1_dist_hc)),
        "p1_dis_median": float(np.median(p1_dist_dis)),
        "p1_cliffs_delta_vs_hc": _cliffs_delta(p1_dist_dis, p1_dist_hc),
        "p2b_hc_median":  float(np.median(p2b_dist_hc)),
        "p2b_dis_median": float(np.median(p2b_dist_dis)),
        "p2b_cliffs_delta_vs_hc": _cliffs_delta(p2b_dist_dis, p2b_dist_hc),
    }

    return policy, p1_eff, p2b_eff, rel, distances


# ──────────────────────────────────────────────────────────────────────
# Axis comparison table (Level 1 + Level 2)
# ──────────────────────────────────────────────────────────────────────

def _tier_lookup(rel: pd.DataFrame, pilot: str, axis: str) -> str:
    row = rel[(rel["pilot_name"] == pilot) & (rel["axis"] == axis)]
    if row.empty:
        return "UNKNOWN"
    return row.iloc[0]["tier"]


def _classify_use(t1: str, t2b: str) -> str:
    if t1 == "TIER_1_ROBUST" and t2b == "TIER_1_ROBUST":
        return "PRIMARY_CROSS_PILOT"
    if "TIER_3_UNSTABLE" in (t1, t2b):
        return "EXCLUDE_PRIMARY_CLAIMS"
    # Remaining: (T1, T2), (T2, T1), (T2, T2)
    if {t1, t2b} == {"TIER_1_ROBUST", "TIER_2_CONTEXTUAL"}:
        return "SECONDARY_CONTEXT_ONLY"
    # Both Tier 2 → exclude from primary (not Tier 1 anywhere)
    return "EXCLUDE_PRIMARY_CLAIMS"


def _interpretation_note(axis: str, p1_d: float, p2b_d: float, use_class: str,
                          sign_agree: bool) -> str:
    name = BSV_SHORT[axis]
    if use_class == "EXCLUDE_PRIMARY_CLAIMS":
        return "Excluded from primary cross-pilot claims per Step 2 reliability."
    if use_class == "SECONDARY_CONTEXT_ONLY":
        direction_phrase = "same direction" if sign_agree else "opposite direction"
        return f"{name} carries {direction_phrase} in both pilots but reliability is asymmetric; use as secondary context only."
    # PRIMARY
    if sign_agree:
        dir_txt = "elevated" if p1_d > 0 else "depressed"
        return (f"{name} is reliably detected and {dir_txt} in BOTH canonical pilots "
                f"(direction agrees).")
    return (f"{name} is reliably detected in BOTH canonical pilots but with OPPOSITE "
            f"direction (HCC {p1_d:+.2f} vs CCA {p2b_d:+.2f}); axis-level engagement is shared "
            f"but the underlying biology diverges.")


def build_axis_comparison(p1_eff: pd.DataFrame, p2b_eff: pd.DataFrame,
                            rel: pd.DataFrame) -> pd.DataFrame:
    # Rank by |cohens_d| within each pilot
    def _rank(df):
        s = df["cohens_d"].abs().sort_values(ascending=False)
        return {axis: r + 1 for r, axis in enumerate(s.index)}

    p1_rank  = _rank(p1_eff)
    p2b_rank = _rank(p2b_eff)

    rows = []
    for axis in BSV_COMPONENTS:
        p1_d = float(p1_eff.loc[axis, "cohens_d"])
        p2_d = float(p2b_eff.loc[axis, "cohens_d"])
        p1_dir = _direction(p1_d)
        p2_dir = _direction(p2_d)
        sign_agree = (
            p1_dir == p2_dir and p1_dir in ("up", "down")
        )
        t1  = _tier_lookup(rel, "pilot1_hcc_holdout", axis)
        t2b = _tier_lookup(rel, "pilot2b_cca_raw", axis)
        use_class = _classify_use(t1, t2b)
        rows.append({
            "axis": axis,
            "pilot1_direction": p1_dir,
            "pilot2b_direction": p2_dir,
            "sign_agreement": bool(sign_agree),
            "pilot1_effect_size": p1_d,
            "pilot2b_effect_size": p2_d,
            "pilot1_effect_size_abs": abs(p1_d),
            "pilot2b_effect_size_abs": abs(p2_d),
            "pilot1_rank": int(p1_rank[axis]),
            "pilot2b_rank": int(p2b_rank[axis]),
            "rank_delta": int(p2b_rank[axis] - p1_rank[axis]),
            "pilot1_tier": t1,
            "pilot2b_tier": t2b,
            "primary_use_flag": use_class,
            "interpretation_note": _interpretation_note(axis, p1_d, p2_d, use_class, sign_agree),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Carry-forward axis classification (Level 1 output)
# ──────────────────────────────────────────────────────────────────────

def build_carry_forward(comp_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in comp_df.iterrows():
        axis = r["axis"]
        use = r["primary_use_flag"]
        rationale = (
            f"Pilot 1 tier = {r['pilot1_tier']}; Pilot 2b tier = {r['pilot2b_tier']}; "
            f"direction {r['pilot1_direction']}/{r['pilot2b_direction']} "
            f"({'agree' if r['sign_agreement'] else 'differ'}); "
            f"|d| {r['pilot1_effect_size_abs']:.2f}/{r['pilot2b_effect_size_abs']:.2f}."
        )
        rows.append({
            "axis": axis,
            "axis_label": BSV_SHORT[axis],
            "pilot1_tier": r["pilot1_tier"],
            "pilot2b_tier": r["pilot2b_tier"],
            "use_class": use,
            "rationale": rationale,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Theme synthesis (Level 3)
# ──────────────────────────────────────────────────────────────────────

def _support_label(abs_d: float, tier: str) -> str:
    if tier == "TIER_3_UNSTABLE":
        return "unreliable"
    if abs_d >= 0.8:
        return "very_strong"
    if abs_d >= 0.5:
        return "strong"
    if abs_d >= 0.3:
        return "moderate"
    if abs_d >= 0.2:
        return "weak"
    return "null"


def build_theme_synthesis(comp_df: pd.DataFrame, distances: dict) -> pd.DataFrame:
    comp = comp_df.set_index("axis")

    def row(axis): return comp.loc[axis]

    themes = []

    # Theme 1 — shared top-prominence cluster
    top3_p1 = set(comp.nsmallest(3, "pilot1_rank").index)
    top3_p2b = set(comp.nsmallest(3, "pilot2b_rank").index)
    overlap = sorted(top3_p1 & top3_p2b, key=lambda a: BSV_COMPONENTS.index(a))
    overlap_label = ", ".join(BSV_SHORT[a] for a in overlap)
    themes.append({
        "theme_name": "shared_top_3_axis_prominence",
        "supporting_axes": overlap_label,
        "pilot1_support": f"ranks {sorted(int(comp.loc[a, 'pilot1_rank']) for a in overlap)}",
        "pilot2b_support": f"ranks {sorted(int(comp.loc[a, 'pilot2b_rank']) for a in overlap)}",
        "agreement_level": "shared_axes" if len(overlap) == 3 else f"{len(overlap)}/3 shared",
        "confidence_level": "high" if len(overlap) == 3 else "moderate",
        "notes": (f"The same axes occupy the top 3 by |d| in both canonical pilots: "
                  f"{overlap_label}. Rank order differs slightly but the axis set is stable."),
    })

    # Theme 2 — Glycan-associated elevation (shared direction, both T1)
    r = row("glycan_carbohydrate")
    themes.append({
        "theme_name": "glycan_associated_elevation",
        "supporting_axes": "Glycan",
        "pilot1_support": f"{_support_label(r['pilot1_effect_size_abs'], r['pilot1_tier'])} "
                            f"(d={r['pilot1_effect_size']:+.2f}, {r['pilot1_tier']})",
        "pilot2b_support": f"{_support_label(r['pilot2b_effect_size_abs'], r['pilot2b_tier'])} "
                             f"(d={r['pilot2b_effect_size']:+.2f}, {r['pilot2b_tier']})",
        "agreement_level": "agree" if r["sign_agreement"] else "disagree",
        "confidence_level": "high" if r["primary_use_flag"] == "PRIMARY_CROSS_PILOT" and r["sign_agreement"] else "moderate",
        "notes": ("Glycan axis is elevated versus the within-dataset healthy centroid in BOTH "
                  "canonical pilots. Only Tier 1 carry-forward axis where direction agrees. "
                  "Cross-pilot shared theme."),
    })

    # Theme 3 — Purine axis inversion (HCC down, CCA up; both T1)
    r = row("purine_nucleotide")
    themes.append({
        "theme_name": "divergent_purine_direction",
        "supporting_axes": "Purine",
        "pilot1_support": f"{_support_label(r['pilot1_effect_size_abs'], r['pilot1_tier'])} "
                            f"(d={r['pilot1_effect_size']:+.2f}, {r['pilot1_tier']})",
        "pilot2b_support": f"{_support_label(r['pilot2b_effect_size_abs'], r['pilot2b_tier'])} "
                             f"(d={r['pilot2b_effect_size']:+.2f}, {r['pilot2b_tier']})",
        "agreement_level": "divergent_direction_shared_axis",
        "confidence_level": "high" if r["primary_use_flag"] == "PRIMARY_CROSS_PILOT" else "moderate",
        "notes": ("Purine axis is the #1 axis by |d| in BOTH pilots, but direction differs "
                  "(HCC depressed, CCA elevated). Axis-level engagement is shared; underlying "
                  "biology diverges."),
    })

    # Theme 4 — Nuc.Backbone axis inversion (HCC up, CCA down; both T1)
    r = row("nucleic_acid_backbone")
    themes.append({
        "theme_name": "divergent_nucleic_acid_backbone_direction",
        "supporting_axes": "Nuc.Backbone",
        "pilot1_support": f"{_support_label(r['pilot1_effect_size_abs'], r['pilot1_tier'])} "
                            f"(d={r['pilot1_effect_size']:+.2f}, {r['pilot1_tier']})",
        "pilot2b_support": f"{_support_label(r['pilot2b_effect_size_abs'], r['pilot2b_tier'])} "
                             f"(d={r['pilot2b_effect_size']:+.2f}, {r['pilot2b_tier']})",
        "agreement_level": "divergent_direction_shared_axis",
        "confidence_level": "high" if r["primary_use_flag"] == "PRIMARY_CROSS_PILOT" else "moderate",
        "notes": ("Nucleic-acid-backbone axis is reliably engaged in both (T1/T1) but with "
                  "opposite direction (HCC elevated, CCA depressed). Note: this axis has only 1 "
                  "mapped atlas window — high engagement but atlas-expansion work should precede "
                  "any mechanistic interpretation."),
    })

    # Theme 5 — broader displacement from healthy centroid (dataset-wide magnitude)
    themes.append({
        "theme_name": "shared_displacement_from_healthy_centroid",
        "supporting_axes": "distance_metric (all 8 axes)",
        "pilot1_support": f"Cliff's δ(disease vs HC) = {distances['p1_cliffs_delta_vs_hc']:+.2f}"
                            f" · medians HC={distances['p1_hc_median']:.4f} / disease={distances['p1_dis_median']:.4f}",
        "pilot2b_support": f"Cliff's δ(disease vs HC) = {distances['p2b_cliffs_delta_vs_hc']:+.2f}"
                             f" · medians HC={distances['p2b_hc_median']:.4f} / disease={distances['p2b_dis_median']:.4f}",
        "agreement_level": ("agree" if distances['p1_cliffs_delta_vs_hc'] > 0 and
                             distances['p2b_cliffs_delta_vs_hc'] > 0 else "disagree"),
        "confidence_level": "high" if (distances['p1_cliffs_delta_vs_hc'] > 0.25 and
                                       distances['p2b_cliffs_delta_vs_hc'] > 0.25) else "moderate",
        "notes": ("Both disease cohorts sit further from their own healthy centroids than "
                  "the healthy cohort does. CCA displacement is substantially larger than HCC, "
                  "but both are positive. Whole-representation shift, not a single-axis claim."),
    })

    # Theme 6 — secondary protein attenuation
    r = row("protein_backbone")
    themes.append({
        "theme_name": "secondary_protein_attenuation",
        "supporting_axes": "Protein",
        "pilot1_support": f"{_support_label(r['pilot1_effect_size_abs'], r['pilot1_tier'])} "
                            f"(d={r['pilot1_effect_size']:+.2f}, {r['pilot1_tier']})",
        "pilot2b_support": f"{_support_label(r['pilot2b_effect_size_abs'], r['pilot2b_tier'])} "
                             f"(d={r['pilot2b_effect_size']:+.2f}, {r['pilot2b_tier']})",
        "agreement_level": "agree_but_asymmetric_reliability",
        "confidence_level": "low_moderate",
        "notes": ("Protein axis is depressed in both cohorts, but reliability is asymmetric "
                  "(T2 in Pilot 1; T1 in Pilot 2b). Use as secondary context; do NOT elevate to "
                  "a primary cross-pilot claim."),
    })

    return pd.DataFrame(themes)


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 180, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


_DIR_GLYPH = {"up": "▲", "down": "▼", "flat": "·"}


def fig_direction_agreement(comp_df: pd.DataFrame, path: Path):
    axes = BSV_COMPONENTS
    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    row_ys = {"Pilot 1 (HCC)": 2, "Pilot 2b (CCA)": 1, "Agreement": 0}

    for j, axis in enumerate(axes):
        r = comp_df[comp_df["axis"] == axis].iloc[0]
        # Pilot 1 cell
        c1 = TIER_COLOR[r["pilot1_tier"]]
        ax.add_patch(plt.Rectangle((j - 0.42, row_ys["Pilot 1 (HCC)"] - 0.42), 0.84, 0.84,
                                     facecolor=c1, edgecolor="white", linewidth=1.2))
        ax.text(j, row_ys["Pilot 1 (HCC)"], f"{_DIR_GLYPH[r['pilot1_direction']]}\nd={r['pilot1_effect_size']:+.2f}",
                 ha="center", va="center", fontsize=10, color="#111", fontweight="bold")

        # Pilot 2b cell
        c2 = TIER_COLOR[r["pilot2b_tier"]]
        ax.add_patch(plt.Rectangle((j - 0.42, row_ys["Pilot 2b (CCA)"] - 0.42), 0.84, 0.84,
                                     facecolor=c2, edgecolor="white", linewidth=1.2))
        ax.text(j, row_ys["Pilot 2b (CCA)"], f"{_DIR_GLYPH[r['pilot2b_direction']]}\nd={r['pilot2b_effect_size']:+.2f}",
                 ha="center", va="center", fontsize=10, color="#111", fontweight="bold")

        # Agreement cell
        use = r["primary_use_flag"]
        if use == "EXCLUDE_PRIMARY_CLAIMS":
            label, color = "excluded", "#F87171"
        elif use == "SECONDARY_CONTEXT_ONLY":
            label, color = ("same dir" if r["sign_agreement"] else "opp dir"), "#FBBF24"
        else:
            label, color = ("agree" if r["sign_agreement"] else "divergent"), "#4ADE80"
        ax.add_patch(plt.Rectangle((j - 0.42, row_ys["Agreement"] - 0.42), 0.84, 0.84,
                                     facecolor=color, edgecolor="white", linewidth=1.2))
        ax.text(j, row_ys["Agreement"], label, ha="center", va="center",
                 fontsize=10, color="#111", fontweight="bold")

    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels([BSV_SHORT[a] for a in axes], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(list(row_ys.values()))
    ax.set_yticklabels(list(row_ys.keys()))
    ax.set_xlim(-0.7, len(axes) - 0.3)
    ax.set_ylim(-0.7, 2.7)
    ax.tick_params(axis="both", length=0)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Cross-pilot direction agreement (color = Step 2 reliability tier)",
                  fontsize=13, pad=12)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=TIER_COLOR["TIER_1_ROBUST"],    label="Tier 1 Robust"),
        Patch(facecolor=TIER_COLOR["TIER_2_CONTEXTUAL"], label="Tier 2 Contextual"),
        Patch(facecolor=TIER_COLOR["TIER_3_UNSTABLE"],  label="Tier 3 Unstable"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.0, 1.05),
              frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def fig_rank_comparison(comp_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    df = comp_df.sort_values("pilot1_rank").reset_index(drop=True)
    for _, r in df.iterrows():
        col = USE_COLOR[r["primary_use_flag"]]
        # dashed line if direction disagrees on primary axes
        ls = "-" if r["sign_agreement"] or r["primary_use_flag"] != "PRIMARY_CROSS_PILOT" else "--"
        ax.plot([0, 1], [r["pilot1_rank"], r["pilot2b_rank"]], color=col, lw=2.2, alpha=0.85,
                 linestyle=ls)
        ax.scatter(0, r["pilot1_rank"], color=col, s=56, zorder=3, edgecolor="white")
        ax.scatter(1, r["pilot2b_rank"], color=col, s=56, zorder=3, edgecolor="white")
        # Labels on both sides
        ax.text(-0.06, r["pilot1_rank"], BSV_SHORT[r["axis"]],
                 ha="right", va="center", fontsize=10, color="#1F2937")
        ax.text(1.06, r["pilot2b_rank"], BSV_SHORT[r["axis"]],
                 ha="left", va="center", fontsize=10, color="#1F2937")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pilot 1\nHCC holdout", "Pilot 2b\nCCA raw"], fontsize=11)
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylim(8.6, 0.4)
    ax.set_ylabel("|effect size| rank (1 = largest)")
    ax.set_title("Cross-pilot axis ranking by |Cohen's d|",
                  fontsize=13, pad=12)
    ax.grid(alpha=0.15, axis="y", linestyle=":")

    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([], [], color=USE_COLOR["PRIMARY_CROSS_PILOT"], lw=2.2,
                label="Primary cross-pilot (T1/T1)"),
        Line2D([], [], color=USE_COLOR["SECONDARY_CONTEXT_ONLY"], lw=2.2,
                label="Secondary context (T1/T2)"),
        Line2D([], [], color=USE_COLOR["EXCLUDE_PRIMARY_CLAIMS"], lw=2.2,
                label="Excluded (T3 in either)"),
        Line2D([], [], color="#555", lw=2.2, linestyle="--",
                label="Dashed = primary axis with direction disagreement"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    _save(fig, path)


def fig_tier_overlay(comp_df: pd.DataFrame, path: Path):
    axes = BSV_COMPONENTS
    pilots = [("pilot1_hcc_holdout",  "Pilot 1 (HCC)",  "pilot1"),
              ("pilot2b_cca_raw",      "Pilot 2b (CCA)", "pilot2b")]

    fig, ax = plt.subplots(figsize=(13, 3.6))
    for i, (pname, plabel, prefix) in enumerate(pilots):
        for j, axis in enumerate(axes):
            r = comp_df[comp_df["axis"] == axis].iloc[0]
            tier = r[f"{prefix}_tier"]
            direction = r[f"{prefix}_direction"]
            d = r[f"{prefix}_effect_size"]
            color = TIER_COLOR[tier]
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.90, 0.90,
                                         facecolor=color, edgecolor="white", linewidth=1.4))
            ax.text(j, i, f"{_DIR_GLYPH[direction]}  d={d:+.2f}",
                     ha="center", va="center", fontsize=10, color="#111", fontweight="bold")

    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels([BSV_SHORT[a] for a in axes], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(pilots)))
    ax.set_yticklabels([p[1] for p in pilots], fontsize=10)
    ax.set_xlim(-0.7, len(axes) - 0.3)
    ax.set_ylim(-0.7, len(pilots) - 0.3)
    ax.invert_yaxis()
    ax.tick_params(axis="both", length=0)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Cross-pilot tier × direction overlay",
                  fontsize=13, pad=10)

    # Add a small use-class bar underneath
    ax2 = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.18,
                         ax.get_position().width, 0.10])
    for j, axis in enumerate(axes):
        r = comp_df[comp_df["axis"] == axis].iloc[0]
        c = USE_COLOR[r["primary_use_flag"]]
        ax2.add_patch(plt.Rectangle((j - 0.45, 0), 0.90, 1.0, facecolor=c, edgecolor="white"))
        short = {"PRIMARY_CROSS_PILOT": "PRIMARY",
                  "SECONDARY_CONTEXT_ONLY": "SECONDARY",
                  "EXCLUDE_PRIMARY_CLAIMS": "EXCLUDE"}[r["primary_use_flag"]]
        ax2.text(j, 0.5, short, ha="center", va="center", fontsize=8,
                  color="white", fontweight="bold")
    ax2.set_xlim(-0.7, len(axes) - 0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xticks([]); ax2.set_yticks([])
    for s in ["top", "right", "left", "bottom"]:
        ax2.spines[s].set_visible(False)
    ax2.set_ylabel("use", rotation=0, labelpad=28, fontsize=9, va="center")

    _save(fig, path)


def fig_theme_summary(theme_df: pd.DataFrame, path: Path):
    n = len(theme_df)
    fig, ax = plt.subplots(figsize=(14, 0.65 * n + 1.6))

    conf_color = {"high": "#059669", "moderate": "#D97706",
                  "low": "#B91C1C", "low_moderate": "#D97706"}
    agree_color = {
        "agree": "#4ADE80",
        "shared_axes": "#4ADE80",
        "divergent_direction_shared_axis": "#FBBF24",
        "agree_but_asymmetric_reliability": "#FBBF24",
        "disagree": "#F87171",
    }

    for i, (_, r) in enumerate(theme_df.iterrows()):
        y = i
        # Theme name
        ax.text(0.02, y, r["theme_name"].replace("_", " "),
                 ha="left", va="center", fontsize=11, color="#111", fontweight="bold")
        # Supporting axes
        ax.text(0.42, y, r["supporting_axes"], ha="left", va="center",
                 fontsize=10, color="#333")

        # Agreement chip
        ac = agree_color.get(r["agreement_level"], "#CBD5E1")
        ax.add_patch(plt.Rectangle((0.68, y - 0.28), 0.16, 0.56, facecolor=ac,
                                     edgecolor="white"))
        ax.text(0.76, y, r["agreement_level"].replace("_", " "),
                 ha="center", va="center", fontsize=8.5, color="#111")

        # Confidence chip
        cc = conf_color.get(r["confidence_level"], "#CBD5E1")
        ax.add_patch(plt.Rectangle((0.86, y - 0.28), 0.12, 0.56, facecolor=cc,
                                     edgecolor="white"))
        ax.text(0.92, y, r["confidence_level"].replace("_", " "),
                 ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    # Column headers
    ax.text(0.02, -0.42, "Theme", fontsize=10, fontweight="bold", color="#555")
    ax.text(0.42, -0.42, "Supporting axes", fontsize=10, fontweight="bold", color="#555")
    ax.text(0.76, -0.42, "Agreement", fontsize=10, fontweight="bold", color="#555",
             ha="center")
    ax.text(0.92, -0.42, "Confidence", fontsize=10, fontweight="bold", color="#555",
             ha="center")

    fig.suptitle("Cross-pilot theme synthesis (canonical pilots only)",
                  fontsize=13, y=0.99)
    fig.tight_layout()
    _save(fig, path)


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def write_report(comp_df: pd.DataFrame, theme_df: pd.DataFrame,
                  carry_df: pd.DataFrame, distances: dict, policy: dict,
                  path: Path):
    primary_axes = carry_df[carry_df["use_class"] == "PRIMARY_CROSS_PILOT"]["axis_label"].tolist()
    secondary_axes = carry_df[carry_df["use_class"] == "SECONDARY_CONTEXT_ONLY"]["axis_label"].tolist()
    excluded_axes = carry_df[carry_df["use_class"] == "EXCLUDE_PRIMARY_CLAIMS"]["axis_label"].tolist()

    md = []
    md.append("# REPORT — Step 3 · Canonical cross-pilot synthesis v1")
    md.append("")
    md.append("First strictly comparable cross-pilot synthesis for GAIRA target pilots. "
              "Strictly structure-aware. No pooling, no cross-dataset normalization.")
    md.append("")
    md.append("## A. Scope and guardrails")
    md.append("")
    md.append(f"- **Policy gate:** `gaira_target_pipeline_policy_{policy['policy']['version']}.yaml`.")
    md.append(f"- **Only canonical STRICTLY_COMPARABLE pilots** were used: Pilot 1 (HCC holdout) and Pilot 2b (CCA raw). Both carry preprocessing tag `raw_asls_sg_l2`.")
    md.append(f"- **Pilot 2a (fallback, `npz_l2`) was not used** as primary evidence. Its cross-preprocessing role was already consumed inside Step 2's sensitivity column and does not enter Step 3 directly.")
    md.append(f"- **No cross-dataset normalization or alignment.** Magnitudes are interpreted within-pilot first; cross-pilot comparison is limited to direction and relative prominence.")
    md.append(f"- **Step 2 reliability matrix** gates axis-level use: Tier 1 axes are primary; Tier 2 secondary; Tier 3 excluded from primary claims.")
    md.append("")
    md.append("## B. Included pilots and why")
    md.append("")
    md.append("| Pilot | Dataset | Preprocess | Canonical / Fallback | Comparability |")
    md.append("|---|---|---|---|---|")
    md.append("| Pilot 1 | HCC holdout (Gurian 2020) | `raw_asls_sg_l2` | canonical | STRICTLY_COMPARABLE |")
    md.append("| Pilot 2b | CCA raw (`cca_hcc_lm_serum_sers`) | `raw_asls_sg_l2` | canonical | STRICTLY_COMPARABLE |")
    md.append("")
    md.append("## C. Axis-level synthesis")
    md.append("")
    md.append("### Primary carry-forward axes (Tier 1 in both)")
    md.append("")
    md.append("| Axis | P1 d | P2b d | Direction | Rank (P1 → P2b) | Interpretation |")
    md.append("|---|---:|---:|---|---:|---|")
    for _, r in comp_df[comp_df["primary_use_flag"] == "PRIMARY_CROSS_PILOT"].iterrows():
        direction = "agree ({})".format("up" if r["pilot1_direction"] == "up" else "down") \
            if r["sign_agreement"] else "DIVERGENT (P1 {} · P2b {})".format(r["pilot1_direction"], r["pilot2b_direction"])
        md.append(f"| {BSV_SHORT[r['axis']]} | `{r['pilot1_effect_size']:+.2f}` | `{r['pilot2b_effect_size']:+.2f}` | {direction} | {r['pilot1_rank']} → {r['pilot2b_rank']} | {r['interpretation_note']} |")
    md.append("")
    md.append("### Secondary context (Tier 1 in one, Tier 2 in the other)")
    md.append("")
    sec = comp_df[comp_df["primary_use_flag"] == "SECONDARY_CONTEXT_ONLY"]
    if sec.empty:
        md.append("*(none)*")
    else:
        md.append("| Axis | P1 d / tier | P2b d / tier | Direction | Interpretation |")
        md.append("|---|---:|---:|---|---|")
        for _, r in sec.iterrows():
            direction = "agree" if r["sign_agreement"] else "disagree"
            md.append(f"| {BSV_SHORT[r['axis']]} | `{r['pilot1_effect_size']:+.2f}` ({r['pilot1_tier']}) | `{r['pilot2b_effect_size']:+.2f}` ({r['pilot2b_tier']}) | {direction} | {r['interpretation_note']} |")
    md.append("")
    md.append("### Excluded from primary claims (Tier 3 in at least one canonical pilot)")
    md.append("")
    exc = comp_df[comp_df["primary_use_flag"] == "EXCLUDE_PRIMARY_CLAIMS"]
    md.append("| Axis | P1 tier | P2b tier | Reason |")
    md.append("|---|---|---|---|")
    for _, r in exc.iterrows():
        md.append(f"| {BSV_SHORT[r['axis']]} | {r['pilot1_tier']} | {r['pilot2b_tier']} | excluded from primary synthesis per Step 2 reliability |")
    md.append("")
    md.append("## D. Theme-level synthesis")
    md.append("")
    for _, t in theme_df.iterrows():
        md.append(f"### {t['theme_name'].replace('_', ' ').capitalize()}")
        md.append("")
        md.append(f"- **Supporting axes:** {t['supporting_axes']}")
        md.append(f"- **Pilot 1:** {t['pilot1_support']}")
        md.append(f"- **Pilot 2b:** {t['pilot2b_support']}")
        md.append(f"- **Agreement:** `{t['agreement_level']}`")
        md.append(f"- **Confidence:** `{t['confidence_level']}`")
        md.append(f"- {t['notes']}")
        md.append("")
    md.append("## E. Carry-forward conclusions")
    md.append("")
    md.append("**Primary (both pilots):** " + (", ".join(primary_axes) if primary_axes else "— none —"))
    md.append("")
    md.append("**Secondary context only:** " + (", ".join(secondary_axes) if secondary_axes else "— none —"))
    md.append("")
    md.append("**Excluded from primary claims:** " + (", ".join(excluded_axes) if excluded_axes else "— none —"))
    md.append("")
    md.append("What the canonical data supports (and only this):")
    md.append("")
    md.append("- **Glycan axis is elevated vs within-dataset healthy centroid in both HCC and CCA cohorts** (Tier 1 both, same direction, both rank in top 3 by |d|). This is currently the only same-direction Tier 1 cross-pilot claim.")
    md.append("- **Purine and Nucleic-acid-backbone axes are both reliably engaged in both pilots, but with opposite direction.** Axis-level engagement is shared; the underlying biology diverges between HCC and CCA. Do **not** collapse these into a single directional claim.")
    md.append("- **Both disease cohorts are displaced from their own healthy centroid** in 8-D BSV space — CCA more strongly than HCC. This is a whole-representation shift, not a single-axis claim.")
    md.append("- **Secondary context:** Protein axis is depressed in both cohorts, but reliability is asymmetric (P1 Tier 2 · P2b Tier 1). Surface as supporting context only; do not elevate to a primary cross-pilot claim.")
    md.append("")
    md.append("## F. Not-yet-claimable items")
    md.append("")
    md.append("The following are explicitly **not** supported by canonical evidence at this stage and must not be overstated:")
    md.append("")
    md.append("- **A single shared \"liver-disease signature\"** across HCC and CCA — the two Tier 1 axes with opposite direction (Purine, Nuc.Backbone) rule that out at this resolution.")
    md.append("- **Mechanistic/molecular claims on Glycan, Purine, or Nuc.Backbone axes** — axes aggregate many atlas windows; per-axis biology is not a molecule call.")
    md.append("- **Anything from Tier 3 axes** (Aromatic AA, Redox, Lipid, Pyrimidine at the cross-pilot level) — these are unreliable under canonical preprocessing in at least one pilot.")
    md.append("- **That the Purine / Nuc.Backbone direction inversion between HCC and CCA reflects a single clean biological contrast** — the direction flip is a data fact; its mechanistic interpretation is out of scope until a third canonical pilot constrains it.")
    md.append("- **Cross-pilot numerical magnitude claims** (e.g. \"CCA has 2× stronger Glycan effect\"). Magnitudes are pipeline-level; only direction and rank are used for cross-pilot comparison under this policy.")
    md.append("")
    md.append("## G. Recommendation for next step")
    md.append("")
    md.append("1. **Run a third canonical pilot** (e.g. LM from `cca_hcc_lm_serum_sers`, or another disease cohort with raw spectra available) through the locked `raw_asls_sg_l2` path. Focus the next synthesis on whether the Purine / Nuc.Backbone direction pattern is disease-specific or matrix-specific.")
    md.append("2. **Atlas-expansion work on Nucleic-acid-backbone** — the axis has only 1 mapped window in the current atlas; engagement is strong in both canonical pilots, so expanding its window coverage would improve interpretability.")
    md.append("3. **Targeted grounding for the Glycan axis** — the only same-direction Tier 1 cross-pilot carry-forward. A grounded biochemical-theme interpretation here would be the highest-value next claim.")
    md.append("4. **Per-patient validation for the CCA dataset** — Pilot 2b's sample-level replication lets us run a per-sample rollup as a future robustness layer. Do this only after the third canonical pilot is in.")
    md.append("")
    md.append("## H. Outputs")
    md.append("")
    md.append("- `cross_pilot_axis_comparison.csv`")
    md.append("- `cross_pilot_theme_synthesis.csv`")
    md.append("- `cross_pilot_carry_forward_axes.csv`")
    md.append("- `fig_cross_pilot_direction_agreement.png`")
    md.append("- `fig_cross_pilot_rank_comparison.png`")
    md.append("- `fig_cross_pilot_tier_overlay.png`")
    md.append("- `fig_cross_pilot_theme_summary.png`")
    md.append("")

    path.write_text("\n".join(md))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[step3] out: {OUT_ROOT}")

    policy, p1_eff, p2b_eff, rel, distances = load_inputs()
    print(f"[step3] inputs loaded (policy={policy['policy']['version']}; "
          f"rel rows={len(rel)}; dist P1 δ={distances['p1_cliffs_delta_vs_hc']:+.2f} · "
          f"P2b δ={distances['p2b_cliffs_delta_vs_hc']:+.2f})")

    comp_df = build_axis_comparison(p1_eff, p2b_eff, rel)
    comp_df.to_csv(OUT_ROOT / "cross_pilot_axis_comparison.csv", index=False)
    print("[step3] wrote cross_pilot_axis_comparison.csv")

    theme_df = build_theme_synthesis(comp_df, distances)
    theme_df.to_csv(OUT_ROOT / "cross_pilot_theme_synthesis.csv", index=False)
    print("[step3] wrote cross_pilot_theme_synthesis.csv")

    carry_df = build_carry_forward(comp_df)
    carry_df.to_csv(OUT_ROOT / "cross_pilot_carry_forward_axes.csv", index=False)
    print("[step3] wrote cross_pilot_carry_forward_axes.csv")

    fig_direction_agreement(comp_df, OUT_ROOT / "fig_cross_pilot_direction_agreement.png")
    fig_rank_comparison(comp_df, OUT_ROOT / "fig_cross_pilot_rank_comparison.png")
    fig_tier_overlay(comp_df, OUT_ROOT / "fig_cross_pilot_tier_overlay.png")
    fig_theme_summary(theme_df, OUT_ROOT / "fig_cross_pilot_theme_summary.png")
    print("[step3] figures written")

    write_report(comp_df, theme_df, carry_df, distances, policy,
                  OUT_ROOT / "REPORT_step3_cross_pilot_synthesis_v1.md")
    print("[step3] report written")
    print("[step3] done")


if __name__ == "__main__":
    main()
