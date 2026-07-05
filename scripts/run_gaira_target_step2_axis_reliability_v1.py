"""GAIRA Target Step 2 — Axis reliability classification v1.

Post-analysis rule-based classifier. Consumes existing canonical pilot outputs
(Pilot 1 HCC holdout; Pilot 2b CCA raw) and emits a deterministic TIER per
(pilot × axis) using the rules defined in `axis_reliability_rules_v1.md`.

No scorer / atlas / axis changes. No pilot reruns beyond computing two
diagnostic artifacts for Pilot 1 (axis correlation + contribution
diagnostics) that Pilot 1 never produced.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_target_step2_axis_reliability_v1.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral.dataset_loader import load_dataset
from gaira.spectral.preprocessing import _preprocess_raw
from gaira.spectral.window_panel import BSV_COMPONENTS, WINDOW_DEFS, extract_window_features
from gaira.spectral.bsv_projection import project_to_bsv


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

P1_TABLES = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_target_pilot1_hcc_holdout_bsv/tables")
P2B_TABLES = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/pilot2b_cca_raw/tables")
OUT_ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/step2_axis_reliability_v1")


# ──────────────────────────────────────────────────────────────────────
# Thresholds (deterministic, locked for v1)
# ──────────────────────────────────────────────────────────────────────

D_TIER1_MIN           = 0.50  # large-ish effect
D_BORDERLINE_MIN      = 0.30  # borderline effect, only accepts Tier 1 with cross-pipeline support
D_WEAK_MAX            = 0.20  # below this + CI crosses zero → Tier 3 disqualifier
CORR_ENTANGLED_MIN    = 0.50  # max |r| to other axes ≥ 0.5 → entangled → not Tier 1
CORR_SEVERE_MIN       = 0.80  # max |r| to other axes ≥ 0.8 → Tier 3 disqualifier
SAMPLE_CONSIST_MIN    = 0.60  # ≥ 60% of samples agree in sign
SINGLE_WINDOW_SHARE   = 0.75  # ≥ 75% of |Δ| from top window → single-window-dominant
SINGLE_WINDOW_MIN_N   = 3     # only flag if axis has ≥ 3 windows (else by-design)


BSV_SHORT = {
    "membrane_lipid": "Lipid", "protein_backbone": "Protein",
    "aromatic_amino_acid": "Aromatic AA", "purine_nucleotide": "Purine",
    "pyrimidine_nucleotide": "Pyrimidine", "glycan_carbohydrate": "Glycan",
    "redox_metabolite": "Redox", "nucleic_acid_backbone": "Nuc.Backbone",
}

TIER_COLORS = {
    "TIER_1_ROBUST":    "#4ADE80",   # emerald
    "TIER_2_CONTEXTUAL": "#FBBF24",   # amber
    "TIER_3_UNSTABLE":  "#F87171",   # red
}


# ──────────────────────────────────────────────────────────────────────
# Pilot 1 diagnostic recomputation (missing from Pilot 1 outputs)
# ──────────────────────────────────────────────────────────────────────

def compute_pilot1_diagnostics(diag_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute axis correlation + contribution diagnostics for Pilot 1.

    This re-executes only the diagnostic steps (preprocess → windows → BSV);
    it does NOT modify Pilot 1 outputs. Results are saved under step2/diagnostics/.
    """
    ds = load_dataset("hcc_holdout_vornoli2020")
    Xn, _ = _preprocess_raw(ds)
    wf = extract_window_features(Xn, ds.wavenumbers)
    bsv = project_to_bsv(wf)

    is_hc = ds.cohorts == "healthy_control"
    delta = bsv - bsv[is_hc].mean(axis=0)

    # Axis correlation (same definition as Pilot 2b)
    corr = np.corrcoef(delta.T)
    corr_df = (
        pd.DataFrame(corr, index=BSV_COMPONENTS, columns=BSV_COMPONENTS)
        .reset_index().rename(columns={"index": "axis"})
    )
    corr_df.to_csv(diag_dir / "pilot1_axis_correlation.csv", index=False)

    # Contribution diagnostics: HCC-vs-HC per window, grouped by axis
    is_dis = ds.cohorts == "hcc"
    rows = []
    for ci, comp in enumerate(BSV_COMPONENTS):
        win_idx = [j for j, (_, _, _, c) in enumerate(WINDOW_DEFS) if c == comp]
        for j in win_idx:
            w_id, w_lo, w_hi, _ = WINDOW_DEFS[j]
            hc_m = float(wf[is_hc, j].mean())
            dis_m = float(wf[is_dis, j].mean())
            rows.append({
                "axis": comp, "window_id": w_id,
                "window_range_cm1": f"{int(w_lo)}-{int(w_hi)}",
                "hc_mean": hc_m, "disease_mean": dis_m,
                "delta_mean": dis_m - hc_m,
            })
    contrib_df = pd.DataFrame(rows)
    contrib_df.to_csv(diag_dir / "pilot1_contribution_diagnostics.csv", index=False)
    return corr_df, contrib_df


# ──────────────────────────────────────────────────────────────────────
# Per-axis metric builders
# ──────────────────────────────────────────────────────────────────────

def _n_windows_per_axis() -> dict:
    counts = {c: 0 for c in BSV_COMPONENTS}
    for _, _, _, c in WINDOW_DEFS:
        if c in counts:
            counts[c] += 1
    return counts


def _top_window_share(contrib_df: pd.DataFrame, axis: str) -> float:
    sub = contrib_df[contrib_df["axis"] == axis].copy()
    sub["abs"] = sub["delta_mean"].abs()
    total = float(sub["abs"].sum())
    if total <= 0 or sub.empty:
        return float("nan")
    return float(sub["abs"].max() / total)


def _max_abs_corr_row(corr_df: pd.DataFrame, axis: str) -> float:
    row = corr_df.set_index("axis").loc[axis]
    row = row.drop(labels=[axis])
    return float(row.abs().max())


def _ci_excludes_zero(lo: float, hi: float) -> bool:
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


def _sensitivity_supported(d_a: float, d_b: float) -> str:
    """'supported' | 'contradicted' | 'partial'."""
    direction = np.sign(d_a) == np.sign(d_b) and abs(d_a) > 1e-9 and abs(d_b) > 1e-9
    magnitude_ok = abs(d_b) >= 0.5 * abs(d_a)
    if direction and magnitude_ok:
        return "supported"
    if not direction:
        return "contradicted"
    # Direction matches but magnitude collapsed < half
    return "contradicted"


# Pilot 1 ─────────────────────────────────────────────────────────────

def build_pilot1_metrics(corr_df: pd.DataFrame, contrib_df: pd.DataFrame) -> pd.DataFrame:
    eff = pd.read_csv(P1_TABLES / "pilot1_hcc_axis_effect_sizes.csv").set_index("axis")
    batch = pd.read_csv(P1_TABLES / "pilot1_hcc_batch_summary.csv")

    n_win = _n_windows_per_axis()
    rows = []
    for axis in BSV_COMPONENTS:
        e = eff.loc[axis]
        d = float(e["cohens_d"])
        lo, hi = float(e["cohens_d_ci_low"]), float(e["cohens_d_ci_high"])
        # Batch-level sign consistency for HCC (Pilot 1 has 3 substrate batches)
        sub = batch[(batch["class"] == "hcc") & (batch["axis"] == axis)]
        signs = [np.sign(v) for v in sub["mean_delta_bsv"].dropna().tolist()
                 if not np.isnan(v) and v != 0]
        batch_consistent = len(signs) >= 2 and len(set(signs)) == 1
        stability = "high" if batch_consistent else "low"

        tws = _top_window_share(contrib_df, axis)
        nw = n_win[axis]

        rows.append({
            "pilot_name": "pilot1_hcc_holdout",
            "dataset_name": "hcc_serum",
            "compare_class": "hcc_vs_healthy_control",
            "axis": axis,
            "cohens_d": d,
            "cohens_d_abs": abs(d),
            "cohens_d_ci_low": lo,
            "cohens_d_ci_high": hi,
            "ci_excludes_zero": _ci_excludes_zero(lo, hi),
            "max_abs_axis_correlation": _max_abs_corr_row(corr_df, axis),
            "top_window_share": tws,
            "n_windows_mapped": nw,
            "stability_kind": "batch_sign_consistency",
            "stability_value": "consistent" if batch_consistent else "inconsistent",
            "stability": stability,
            "sensitivity_branch": "not_available",
            "sample_level_consistency": float("nan"),
            "batch_level_consistent": batch_consistent,
            "pilot2a_cohens_d": float("nan"),
            "pilot2b_cohens_d": float("nan"),
        })
    return pd.DataFrame(rows)


# Pilot 2b ────────────────────────────────────────────────────────────

def build_pilot2b_metrics() -> pd.DataFrame:
    eff = pd.read_csv(P2B_TABLES / "pilot2b_cca_raw_axis_effect_sizes.csv")
    eff = eff[eff["compare_class"] == "cca"].set_index("axis")
    corr_df = pd.read_csv(P2B_TABLES / "pilot2b_cca_raw_axis_correlation.csv")
    contrib_df = pd.read_csv(P2B_TABLES / "pilot2b_cca_raw_contribution_diagnostics.csv")
    sens = pd.read_csv(P2B_TABLES / "pilot2ab_axis_effect_sizes_comparison.csv")
    sens = sens[sens["compare_class"] == "cca"].set_index("axis")
    per_spec = pd.read_csv(P2B_TABLES / "pilot2b_cca_raw_per_spectrum_delta_bsv.csv")

    # Sample-level consistency for CCA
    cca = per_spec[per_spec["class"] == "cca"]
    consistency = {}
    for axis in BSV_COMPONENTS:
        col = f"delta_bsv_{axis}"
        cohort_sign = np.sign(cca[col].mean())
        per_sample = cca.groupby("sample_id")[col].mean()
        sample_signs = np.sign(per_sample.values)
        consistency[axis] = float((sample_signs == cohort_sign).mean())

    n_win = _n_windows_per_axis()
    rows = []
    for axis in BSV_COMPONENTS:
        e = eff.loc[axis]
        d = float(e["cohens_d"])
        lo, hi = float(e["cohens_d_ci_low"]), float(e["cohens_d_ci_high"])

        sr = sens.loc[axis]
        d_a, d_b = float(sr["cohens_d_2a"]), float(sr["cohens_d_2b"])
        sens_tag = _sensitivity_supported(d_a, d_b)

        sl = consistency[axis]
        stability = "high" if sl >= SAMPLE_CONSIST_MIN else "low"

        rows.append({
            "pilot_name": "pilot2b_cca_raw",
            "dataset_name": "cca_hcc_lm_serum_sers",
            "compare_class": "cca_vs_healthy_control",
            "axis": axis,
            "cohens_d": d,
            "cohens_d_abs": abs(d),
            "cohens_d_ci_low": lo,
            "cohens_d_ci_high": hi,
            "ci_excludes_zero": _ci_excludes_zero(lo, hi),
            "max_abs_axis_correlation": _max_abs_corr_row(corr_df, axis),
            "top_window_share": _top_window_share(contrib_df, axis),
            "n_windows_mapped": n_win[axis],
            "stability_kind": "sample_sign_consistency",
            "stability_value": f"{sl*100:.0f}%_samples_agree",
            "stability": stability,
            "sensitivity_branch": sens_tag,
            "sample_level_consistency": sl,
            "batch_level_consistent": None,
            "pilot2a_cohens_d": d_a,
            "pilot2b_cohens_d": d_b,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Deterministic tier classifier
# ──────────────────────────────────────────────────────────────────────

def _fragile_single_window(m: pd.Series) -> bool:
    return (
        not np.isnan(m["top_window_share"])
        and m["top_window_share"] >= SINGLE_WINDOW_SHARE
        and m["n_windows_mapped"] >= SINGLE_WINDOW_MIN_N
        and m["sensitivity_branch"] != "supported"
    )


def _robust_criteria(m: pd.Series) -> dict:
    fragile = _fragile_single_window(m)
    return {
        "effect_meaningful":        bool(m["cohens_d_abs"] >= D_TIER1_MIN),
        "ci_excludes_zero":         bool(m["ci_excludes_zero"]),
        "sensitivity_not_contradicted": m["sensitivity_branch"] in ("supported", "not_available"),
        "stability_ok":             m["stability"] in ("high", "not_applicable"),
        "not_entangled":            bool(m["max_abs_axis_correlation"] < CORR_ENTANGLED_MIN),
        "not_fragile_single_window": not fragile,
    }


def classify_tier(m: pd.Series) -> tuple[str, str, str]:
    """Return (tier, reason_short, reason_long)."""
    # Tier 3 disqualifiers (any trigger = Tier 3)
    t3 = []
    if m["sensitivity_branch"] == "contradicted":
        t3.append(
            f"sensitivity-contradicted (2a→2b direction flip or magnitude collapse: "
            f"d_2a={m['pilot2a_cohens_d']:+.2f} → d_2b={m['pilot2b_cohens_d']:+.2f})"
        )
    if m["cohens_d_abs"] < D_WEAK_MAX and not m["ci_excludes_zero"]:
        t3.append(f"weak effect (|d|={m['cohens_d_abs']:.2f}) with CI crossing zero")
    if m["max_abs_axis_correlation"] >= CORR_SEVERE_MIN:
        t3.append(f"severe entanglement (max |r|={m['max_abs_axis_correlation']:.2f} ≥ {CORR_SEVERE_MIN})")
    if _fragile_single_window(m):
        t3.append(
            f"single-window-dominant ({m['top_window_share']*100:.0f}% of |Δ|) across "
            f"{int(m['n_windows_mapped'])} mapped windows with no cross-pipeline support"
        )
    if t3:
        return "TIER_3_UNSTABLE", "; ".join(t3[:2]), " | ".join(t3)

    # Tier 1 vs 2
    rc = _robust_criteria(m)
    n_pass = sum(1 for v in rc.values() if v)

    # Borderline gate: d < 0.5 requires cross-pipeline support to stay Tier 1
    if (n_pass >= 5
        and (m["cohens_d_abs"] >= D_TIER1_MIN or m["sensitivity_branch"] == "supported")):
        passed = [k for k, v in rc.items() if v]
        return ("TIER_1_ROBUST",
                f"{n_pass}/6 robustness criteria pass",
                f"passed: {', '.join(passed)}")

    # Tier 2
    failed = [k for k, v in rc.items() if not v]
    if not failed:
        return ("TIER_2_CONTEXTUAL",
                "borderline effect requires cross-pipeline support for Tier 1",
                f"d_abs={m['cohens_d_abs']:.2f} · sensitivity={m['sensitivity_branch']}")
    return ("TIER_2_CONTEXTUAL",
            f"{len(failed)}/6 robustness criteria failed: {', '.join(failed)}",
            " | ".join([f"{k}=fail" for k in failed]))


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


def fig_heatmap(matrix_df: pd.DataFrame, path: Path):
    pilots = matrix_df["pilot_name"].unique().tolist()
    axes_in_order = BSV_COMPONENTS
    tier_to_int = {"TIER_1_ROBUST": 0, "TIER_2_CONTEXTUAL": 1, "TIER_3_UNSTABLE": 2}

    Z = np.zeros((len(pilots), len(axes_in_order)), dtype=int)
    annot = np.empty((len(pilots), len(axes_in_order)), dtype=object)
    for i, p in enumerate(pilots):
        for j, a in enumerate(axes_in_order):
            row = matrix_df[(matrix_df["pilot_name"] == p) & (matrix_df["axis"] == a)]
            if row.empty:
                Z[i, j] = 1; annot[i, j] = ""
                continue
            Z[i, j] = tier_to_int[row.iloc[0]["tier"]]
            annot[i, j] = f"{row.iloc[0]['tier'].split('_')[1][0]}\nd={row.iloc[0]['cohens_d']:+.2f}"

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([TIER_COLORS["TIER_1_ROBUST"],
                            TIER_COLORS["TIER_2_CONTEXTUAL"],
                            TIER_COLORS["TIER_3_UNSTABLE"]])

    fig, ax = plt.subplots(figsize=(12, max(3.6, 1.5 * len(pilots) + 1.2)))
    im = ax.imshow(Z, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(axes_in_order)))
    ax.set_xticklabels([BSV_SHORT[a] for a in axes_in_order], rotation=30, ha="right")
    ax.set_yticks(range(len(pilots)))
    ax.set_yticklabels(pilots)
    for i in range(len(pilots)):
        for j in range(len(axes_in_order)):
            ax.text(j, i, annot[i, j], ha="center", va="center",
                     fontsize=9, color="#111", fontweight="bold")
    ax.set_title("Axis reliability classification — canonical pilots (v1)", fontsize=13, pad=12)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=TIER_COLORS["TIER_1_ROBUST"], label="Tier 1 — Robust"),
        Patch(facecolor=TIER_COLORS["TIER_2_CONTEXTUAL"], label="Tier 2 — Contextual"),
        Patch(facecolor=TIER_COLORS["TIER_3_UNSTABLE"], label="Tier 3 — Unstable"),
    ]
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def fig_reason_panel(matrix_df: pd.DataFrame, path: Path):
    criteria_order = [
        ("effect_meaningful",              "|d| ≥ 0.5"),
        ("ci_excludes_zero",               "CI ≠ 0"),
        ("sensitivity_not_contradicted",   "sens OK"),
        ("stability_ok",                   "stability"),
        ("not_entangled",                  "|r| < 0.5"),
        ("not_fragile_single_window",      "multi-window"),
    ]

    n = len(matrix_df)
    fig, ax = plt.subplots(figsize=(13, 0.48 * n + 2.6))

    # Y-axis: pilot_name + axis
    row_labels = []
    for _, r in matrix_df.iterrows():
        label = f"{r['pilot_name'][:7]} | {BSV_SHORT[r['axis']]}"
        row_labels.append(label)
    y_positions = list(range(n))

    for i, (_, r) in enumerate(matrix_df.iterrows()):
        # Criterion cells
        for j, (key, _) in enumerate(criteria_order):
            val = r["crit_" + key]
            if val is True:
                color = "#4ADE80"; marker = "✓"
            elif val is False:
                color = "#F87171"; marker = "✗"
            else:
                color = "#CBD5E1"; marker = "·"
            ax.add_patch(plt.Rectangle((j - 0.42, i - 0.42), 0.84, 0.84,
                                         facecolor=color, edgecolor="white", linewidth=1.2))
            ax.text(j, i, marker, ha="center", va="center", fontsize=11,
                     color="#0F172A", fontweight="bold")

        # Tier tag
        tier_col_x = len(criteria_order) + 0.3
        tier_color = TIER_COLORS[r["tier"]]
        ax.add_patch(plt.Rectangle((tier_col_x - 0.48, i - 0.42), 2.2, 0.84,
                                     facecolor=tier_color, edgecolor="white", linewidth=1.2))
        tier_short = {"TIER_1_ROBUST": "TIER 1", "TIER_2_CONTEXTUAL": "TIER 2",
                       "TIER_3_UNSTABLE": "TIER 3"}[r["tier"]]
        ax.text(tier_col_x + 0.6, i, f"{tier_short}  |d|={r['cohens_d_abs']:.2f}",
                 ha="center", va="center", fontsize=10, color="#0F172A", fontweight="bold")

    ax.set_xlim(-0.75, len(criteria_order) + 2.8)
    ax.set_ylim(-0.75, n - 0.25)
    ax.invert_yaxis()
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xticks(list(range(len(criteria_order))) + [len(criteria_order) + 0.9])
    ax.set_xticklabels([lab for _, lab in criteria_order] + ["Tier"], fontsize=9)
    ax.tick_params(axis="both", length=0)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Axis reliability — criterion pass/fail panel (v1)", fontsize=12, pad=10)

    # Pilot separator
    pilot_changes = np.where(
        matrix_df["pilot_name"].shift() != matrix_df["pilot_name"]
    )[0]
    for pc in pilot_changes:
        if pc > 0:
            ax.axhline(pc - 0.5, color="#94A3B8", lw=1.0)
    fig.tight_layout()
    _save(fig, path)


# ──────────────────────────────────────────────────────────────────────
# Rules doc
# ──────────────────────────────────────────────────────────────────────

def write_rules_doc(path: Path):
    txt = dedent(f"""\
    # Axis reliability rules — v1

    **Purpose:** classify each BSV axis, within each canonical target pilot,
    as `TIER_1_ROBUST`, `TIER_2_CONTEXTUAL`, or `TIER_3_UNSTABLE` from a
    deterministic rule set so downstream synthesis never has to guess which
    axes to trust.

    Applies **only** to canonical pilots (strictly comparable per
    `gaira_target_pipeline_policy_v1.yaml` — pipeline tag `raw_asls_sg_l2`).

    ## Thresholds (locked for v1)

    - `D_TIER1_MIN`         = **{D_TIER1_MIN}** — |d| needed for Tier 1 via the "large effect" path.
    - `D_BORDERLINE_MIN`    = **{D_BORDERLINE_MIN}** — floor under which Tier 1 requires cross-pipeline support.
    - `D_WEAK_MAX`          = **{D_WEAK_MAX}** — below this AND CI crosses zero → Tier 3.
    - `CORR_ENTANGLED_MIN`  = **{CORR_ENTANGLED_MIN}** — max |r| to other axes ≥ 0.5 → not Tier 1.
    - `CORR_SEVERE_MIN`     = **{CORR_SEVERE_MIN}** — max |r| to other axes ≥ 0.8 → Tier 3.
    - `SAMPLE_CONSIST_MIN`  = **{SAMPLE_CONSIST_MIN}** — fraction of samples whose sign matches cohort sign (when replicates exist).
    - `SINGLE_WINDOW_SHARE` = **{SINGLE_WINDOW_SHARE}** — top window's share of |Δ| across the axis's mapped windows.
    - `SINGLE_WINDOW_MIN_N` = **{SINGLE_WINDOW_MIN_N}** — an axis with fewer mapped windows is not flagged as fragile (by-design limitation).

    ## Classification procedure

    For each `(pilot, axis)` row:

    ### Step 1 — Tier 3 disqualifiers (any triggers → Tier 3)

    1. `sensitivity_branch == "contradicted"` (a parallel preprocessing branch flipped the sign or collapsed magnitude below ½).
    2. `|d| < {D_WEAK_MAX}` AND CI crosses zero (truly null effect).
    3. `max_abs_axis_correlation ≥ {CORR_SEVERE_MIN}` (severe entanglement; cannot read as independent evidence).
    4. `top_window_share ≥ {SINGLE_WINDOW_SHARE}` AND `n_windows_mapped ≥ {SINGLE_WINDOW_MIN_N}` AND `sensitivity_branch != "supported"` (fragile single-window axis with no cross-pipeline support).

    ### Step 2 — Tier 1 (if Tier 3 not triggered, and ≥ 5 of 6 robustness criteria pass)

    Robustness criteria (each is a boolean):

    1. **effect_meaningful**: `|d| ≥ {D_TIER1_MIN}`.
    2. **ci_excludes_zero**: bootstrap CI does not span zero.
    3. **sensitivity_not_contradicted**: `sensitivity_branch ∈ {{"supported", "not_available"}}`.
    4. **stability_ok**: `stability ∈ {{"high", "not_applicable"}}`, where "high" means sample-level sign agreement ≥ {SAMPLE_CONSIST_MIN} (when replicates exist) or all batches share direction (when batch metadata exists).
    5. **not_entangled**: `max_abs_axis_correlation < {CORR_ENTANGLED_MIN}`.
    6. **not_fragile_single_window**: not disqualified by the Step 1.4 condition.

    **Borderline gate:** if the axis passes ≥ 5/6 but `|d| < {D_TIER1_MIN}`, Tier 1 is granted **only if** `sensitivity_branch == "supported"`. This prevents Tier 1 on borderline effects that cannot be cross-validated.

    ### Step 3 — Tier 2 (everything else)

    Any row that is not Tier 3 and does not reach Tier 1 by Step 2 lands in Tier 2.

    ## How metrics are obtained per pilot

    | Metric | Source |
    |---|---|
    | `cohens_d` + CI | pilot's `axis_effect_sizes.csv` |
    | `max_abs_axis_correlation` | pilot's `axis_correlation.csv` (recomputed from per-spectrum ΔBSV if missing) |
    | `top_window_share`, `n_windows_mapped` | pilot's `contribution_diagnostics.csv` (recomputed if missing) |
    | `stability` | sample-level sign consistency if replicates exist; otherwise batch-level sign consistency across all substrate batches |
    | `sensitivity_branch` | `pilot2ab_axis_effect_sizes_comparison.csv` for Pilot 2b; `"not_available"` when no cross-preprocessing branch exists for that pilot |

    ## Tie-break logic

    - Disqualifiers in Step 1 are evaluated before robustness counts; a single disqualifier forces Tier 3 regardless of other strengths.
    - If two robust criteria fail equivalently, Tier 2 is the ceiling.
    - Sensitivity-branch "partial" outcomes do not exist in v1 — the comparison collapses to `supported | contradicted | not_available`.

    ## Caveats

    - Reliability is a **pipeline-stability** claim, not a biology claim. A Tier 1 axis just means GAIRA can measure that axis reliably under the canonical preprocessing path; it does not mean the biology at that axis is the same disease-to-disease.
    - The rules presume a canonical front end. Fallback pilots (e.g. Pilot 2a) are not classified here and must not be compared axis-by-axis to canonical pilots in reliability terms.
    - Axes with <3 mapped windows (Lipid, Pyrimidine, Glycan, Redox, Nuc.Backbone) cannot be flagged as "single-window-dominant fragile" even if their one or two windows account for all of the signal; this is a stated by-design limitation and should drive atlas-expansion work, not reliability downgrades.
    - Thresholds are frozen for v1. Any change to thresholds or disqualifier rules requires v2 of this doc.
    """)
    path.write_text(txt)


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def write_report(matrix_df: pd.DataFrame, summary_df: pd.DataFrame,
                  path: Path):
    pilots = matrix_df["pilot_name"].unique().tolist()

    # Carry-forward: axes Tier 1 in both canonical pilots
    t1_by_pilot = {
        p: set(matrix_df[(matrix_df["pilot_name"] == p)
                          & (matrix_df["tier"] == "TIER_1_ROBUST")]["axis"])
        for p in pilots
    }
    t3_by_pilot = {
        p: set(matrix_df[(matrix_df["pilot_name"] == p)
                          & (matrix_df["tier"] == "TIER_3_UNSTABLE")]["axis"])
        for p in pilots
    }
    t1_in_all = set.intersection(*t1_by_pilot.values()) if t1_by_pilot else set()
    t1_in_any = set.union(*t1_by_pilot.values()) if t1_by_pilot else set()
    t3_in_any = set.union(*t3_by_pilot.values()) if t3_by_pilot else set()

    def _axis_list(axes):
        return ", ".join(BSV_SHORT[a] for a in BSV_COMPONENTS if a in axes) or "— none —"

    def _pilot_block(p):
        sub = matrix_df[matrix_df["pilot_name"] == p]
        lines = [f"### {p}"]
        lines.append("")
        lines.append("| Axis | Tier | \\|d\\| | CI excl. 0 | max\\|r\\| | top-window share (n win) | stability | sensitivity | reason |")
        lines.append("|---|---|---:|---:|---:|---:|---|---|---|")
        for _, r in sub.iterrows():
            stab = r["stability"]
            if r["stability_kind"] == "sample_sign_consistency":
                stab = f"{r['sample_level_consistency']*100:.0f}% samples agree"
            elif r["stability_kind"] == "batch_sign_consistency":
                stab = "batch-consistent" if r["batch_level_consistent"] else "batch-inconsistent"
            tws = f"{r['top_window_share']*100:.0f}% ({int(r['n_windows_mapped'])})" \
                if not np.isnan(r["top_window_share"]) else f"— ({int(r['n_windows_mapped'])})"
            lines.append(
                f"| {BSV_SHORT[r['axis']]} | "
                f"**{r['tier'].replace('TIER_', 'T').split('_')[0]}** | "
                f"`{r['cohens_d_abs']:.2f}` | "
                f"{'✓' if r['ci_excludes_zero'] else '✗'} | "
                f"`{r['max_abs_axis_correlation']:.2f}` | "
                f"{tws} | {stab} | {r['sensitivity_branch']} | "
                f"{r['reason_short']} |"
            )
        return "\n".join(lines)

    carry_set = t1_in_all  # intersection across canonical pilots
    carry_axes_pretty = _axis_list(carry_set)
    exclude_pretty = _axis_list(t3_in_any)

    md = []
    md.append("# REPORT — Step 2 · Axis reliability v1")
    md.append("")
    md.append("Post-analysis, rule-based classifier applied to GAIRA **canonical** target "
              "pilots. Not a scorer change; not a pilot rerun.")
    md.append("")
    md.append("## A. Why this layer exists")
    md.append("")
    md.append("The 8 GAIRA BSV axes are not assumed equally trustworthy on a given dataset. "
              "Some axes may carry a large, CI-excluding, cross-pipeline-stable signal; others may be "
              "weak, preprocess-sensitive, entangled with neighbouring axes, or dominated by a single "
              "fragile spectral window. This layer sorts each axis into three tiers deterministically so "
              "downstream cross-pilot synthesis can carry forward only the axes that deserve primary "
              "interpretation weight.")
    md.append("")
    md.append("Reliability is a pipeline-stability claim, not a biology claim.")
    md.append("")
    md.append("## B. Rule set (v1)")
    md.append("")
    md.append(f"See [`axis_reliability_rules_v1.md`](axis_reliability_rules_v1.md) for the full text. "
              f"Tier assignment is deterministic from: `|d|`, `CI_excludes_zero`, "
              f"`max_abs_axis_correlation`, `top_window_share × n_windows_mapped`, `stability` "
              f"(sample- or batch-level), and `sensitivity_branch_support` (cross-preprocessing).")
    md.append("")
    md.append(f"Thresholds locked for v1: D_TIER1_MIN=`{D_TIER1_MIN}`, "
              f"D_BORDERLINE_MIN=`{D_BORDERLINE_MIN}`, D_WEAK_MAX=`{D_WEAK_MAX}`, "
              f"CORR_ENTANGLED_MIN=`{CORR_ENTANGLED_MIN}`, CORR_SEVERE_MIN=`{CORR_SEVERE_MIN}`, "
              f"SAMPLE_CONSIST_MIN=`{SAMPLE_CONSIST_MIN}`, "
              f"SINGLE_WINDOW_SHARE=`{SINGLE_WINDOW_SHARE}` "
              f"(only flagged if n_windows ≥ `{SINGLE_WINDOW_MIN_N}`).")
    md.append("")
    md.append("## C. Pilot 1 classification (HCC holdout)")
    md.append("")
    md.append(_pilot_block("pilot1_hcc_holdout"))
    md.append("")
    p1 = summary_df[summary_df["pilot_name"] == "pilot1_hcc_holdout"].iloc[0]
    md.append(f"- Tier 1 (robust): **{p1['n_tier1']}** — {p1['tier1_axes']}")
    md.append(f"- Tier 2 (contextual): **{p1['n_tier2']}** — {p1['tier2_axes']}")
    md.append(f"- Tier 3 (unstable): **{p1['n_tier3']}** — {p1['tier3_axes']}")
    md.append("")
    md.append("*Pilot 1 has no cross-preprocessing sensitivity branch, so `sensitivity_branch = "
              "\"not_available\"` for every axis. Stability is evaluated via substrate-batch sign "
              "consistency across batches A/B/C.*")
    md.append("")
    md.append("## D. Pilot 2b classification (CCA raw, canonical)")
    md.append("")
    md.append(_pilot_block("pilot2b_cca_raw"))
    md.append("")
    p2b = summary_df[summary_df["pilot_name"] == "pilot2b_cca_raw"].iloc[0]
    md.append(f"- Tier 1 (robust): **{p2b['n_tier1']}** — {p2b['tier1_axes']}")
    md.append(f"- Tier 2 (contextual): **{p2b['n_tier2']}** — {p2b['tier2_axes']}")
    md.append(f"- Tier 3 (unstable): **{p2b['n_tier3']}** — {p2b['tier3_axes']}")
    md.append("")
    md.append("*Pilot 2b evaluates `sensitivity_branch_support` against Pilot 2a's `npz_l2` "
              "preprocessing. Stability = sample-level sign agreement across the dataset's "
              "multi-spectrum-per-sample replicate structure.*")
    md.append("")
    md.append("## E. Explicit carry-forward recommendation")
    md.append("")
    md.append("### Safe for primary interpretation (Tier 1 in BOTH canonical pilots)")
    md.append(f"**{carry_axes_pretty}**")
    md.append("")
    md.append("### Usable as secondary context (Tier 1 in at least one pilot, or Tier 2 in both)")
    only_one = sorted(t1_in_any - carry_set, key=BSV_COMPONENTS.index)
    md.append(f"{_axis_list(only_one)}")
    md.append("")
    md.append("### Exclude from primary claims (Tier 3 in at least one canonical pilot)")
    md.append(f"{exclude_pretty}")
    md.append("")
    md.append("Exclusion here means an axis must not enter primary cross-pilot synthesis without "
              "a pipeline-specific caveat. An axis listed here may still be **re-promoted** in a "
              "future pilot if it passes all v1 rules in that pilot's canonical run.")
    md.append("")
    md.append("## F. Final recommendation")
    md.append("")
    md.append(f"**Carry-forward set for cross-pilot synthesis:** `{sorted(carry_set, key=BSV_COMPONENTS.index)}`.")
    md.append("")
    md.append("Use these axes as the primary interpretation channels. Any axis outside this set may be "
              "included as supporting context only, with an explicit reliability-class label on each "
              "claim. Do **not** combine Tier 1 and Tier 3 axes into a single disease-state claim.")
    md.append("")
    md.append("## G. Outputs")
    md.append("")
    md.append("- `axis_reliability_rules_v1.md`")
    md.append("- `axis_reliability_matrix.csv`")
    md.append("- `axis_reliability_summary.csv`")
    md.append("- `fig_axis_reliability_heatmap.png`")
    md.append("- `fig_axis_reliability_reason_panel.png`")
    md.append("- `diagnostics/pilot1_axis_correlation.csv`")
    md.append("- `diagnostics/pilot1_contribution_diagnostics.csv`")
    md.append("")

    path.write_text("\n".join(md))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    diag_dir = OUT_ROOT / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    print(f"[step2] out: {OUT_ROOT}")

    print("[step2] recomputing Pilot 1 axis correlation + contribution diagnostics ...")
    p1_corr, p1_contrib = compute_pilot1_diagnostics(diag_dir)

    print("[step2] building Pilot 1 metrics ...")
    p1 = build_pilot1_metrics(p1_corr, p1_contrib)
    print("[step2] building Pilot 2b metrics ...")
    p2b = build_pilot2b_metrics()

    matrix = pd.concat([p1, p2b], ignore_index=True)

    tiers, reasons_short, reasons_long = [], [], []
    rc_cols = []
    for _, m in matrix.iterrows():
        t, rs, rl = classify_tier(m)
        tiers.append(t); reasons_short.append(rs); reasons_long.append(rl)
        rc_cols.append(_robust_criteria(m))
    matrix["tier"] = tiers
    matrix["reason_short"] = reasons_short
    matrix["reason_long"] = reasons_long
    for k in rc_cols[0].keys():
        matrix[f"crit_{k}"] = [rc[k] for rc in rc_cols]

    # Final matrix schema for disk
    export_cols = [
        "pilot_name", "dataset_name", "compare_class", "axis",
        "tier",
        "cohens_d", "cohens_d_abs", "cohens_d_ci_low", "cohens_d_ci_high",
        "ci_excludes_zero",
        "max_abs_axis_correlation",
        "top_window_share", "n_windows_mapped",
        "stability_kind", "stability_value", "stability",
        "sample_level_consistency", "batch_level_consistent",
        "sensitivity_branch", "pilot2a_cohens_d", "pilot2b_cohens_d",
        "reason_short", "reason_long",
    ]
    # Add the boolean criteria for the figure consumer
    export_cols += [c for c in matrix.columns if c.startswith("crit_")]
    matrix_out = matrix[export_cols].copy()
    matrix_out.to_csv(OUT_ROOT / "axis_reliability_matrix.csv", index=False)
    print(f"[step2] wrote axis_reliability_matrix.csv  ({len(matrix_out)} rows)")

    # Summary (one row per pilot)
    rows = []
    for p, sub in matrix.groupby("pilot_name"):
        def _axes(t):
            return ", ".join(BSV_SHORT[a] for a in BSV_COMPONENTS
                             if a in sub[sub["tier"] == t]["axis"].values)
        rows.append({
            "pilot_name": p,
            "dataset_name": sub["dataset_name"].iloc[0],
            "compare_class": sub["compare_class"].iloc[0],
            "n_axes_total": len(sub),
            "n_tier1": int((sub["tier"] == "TIER_1_ROBUST").sum()),
            "n_tier2": int((sub["tier"] == "TIER_2_CONTEXTUAL").sum()),
            "n_tier3": int((sub["tier"] == "TIER_3_UNSTABLE").sum()),
            "tier1_axes": _axes("TIER_1_ROBUST") or "— none —",
            "tier2_axes": _axes("TIER_2_CONTEXTUAL") or "— none —",
            "tier3_axes": _axes("TIER_3_UNSTABLE") or "— none —",
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_ROOT / "axis_reliability_summary.csv", index=False)
    print(f"[step2] wrote axis_reliability_summary.csv  ({len(summary_df)} rows)")

    # Figures
    fig_heatmap(matrix_out, OUT_ROOT / "fig_axis_reliability_heatmap.png")
    fig_reason_panel(matrix, OUT_ROOT / "fig_axis_reliability_reason_panel.png")
    print("[step2] figures written")

    # Rules + report
    write_rules_doc(OUT_ROOT / "axis_reliability_rules_v1.md")
    write_report(matrix_out, summary_df, OUT_ROOT / "REPORT_step2_axis_reliability_v1.md")
    print("[step2] rules + report written")
    print("[step2] done")


if __name__ == "__main__":
    main()
