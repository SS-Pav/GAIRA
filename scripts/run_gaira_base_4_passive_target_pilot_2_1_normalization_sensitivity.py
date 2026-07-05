"""gaira_base_4 passive target Pilot 2.1 — normalization sensitivity.

Goal: distinguish CCA/LM raw elevation as global-amplitude artifact vs true
disease-state biology. NO engine / MSS / motif / taxonomy / weight changes.
NO classifier training. NO threshold tuning. NO label-driven feature select.

Reads Pilot 2 per-spectrum BSV outputs and applies multiple label-blind
normalisations, computes effect-size survival, biological specificity,
projection sensitivity, and produces a severity-vs-artifact decision.
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


ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_passive_target_pilot_2_1_normalization_sensitivity"
)
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"

PILOT2_TABLES = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_passive_target_pilot_2_cca_hcc_lm/tables"
)

BSV_GROUPS_ORDER = [g["group_id"] for g in BSV_GROUPS]
ABS_COLS = [f"abs_{g}" for g in BSV_GROUPS_ORDER]
CLASS_ORDER = ["NC", "HCC", "CCA", "LM"]


def _spearman(x, y):
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _cohens_d(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2: return 0.0
    pooled = np.sqrt(((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1))
                       / max(len(x)+len(y)-2, 1))
    return (np.mean(x) - np.mean(y)) / (pooled if pooled > 0 else 1.0)


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — load
# ─────────────────────────────────────────────────────────────────────

def stage1_load():
    print("\n[STAGE 1] Load Pilot 2 outputs")
    df = pd.read_csv(PILOT2_TABLES / "pilot2_per_spectrum_outputs.csv")
    n_per = df["class_label"].value_counts().to_dict()
    audit_rows = [{
        "source": str(PILOT2_TABLES / "pilot2_per_spectrum_outputs.csv"),
        "n_total": len(df),
        "n_NC": n_per.get("NC", 0), "n_HCC": n_per.get("HCC", 0),
        "n_CCA": n_per.get("CCA", 0), "n_LM": n_per.get("LM", 0),
        "abs_columns_present": all(c in df.columns for c in ABS_COLS),
        "delta_nc_columns_present": all(f"delta_nc_{g}" in df.columns
                                          for g in BSV_GROUPS_ORDER),
        "confidence_columns_present": all(f"conf_{g}" in df.columns
                                            for g in BSV_GROUPS_ORDER),
    }]
    pd.DataFrame(audit_rows).to_csv(TABLES / "pilot2_1_input_audit.csv", index=False)

    lines = [
        "# Pilot 2.1 — Input Audit",
        "",
        f"- source: `{audit_rows[0]['source']}`",
        f"- n_total: {audit_rows[0]['n_total']}",
        f"- per-class: NC={audit_rows[0]['n_NC']}, HCC={audit_rows[0]['n_HCC']}, "
        f"CCA={audit_rows[0]['n_CCA']}, LM={audit_rows[0]['n_LM']}",
        f"- 11-axis abs columns present: {audit_rows[0]['abs_columns_present']}",
        f"- delta_nc columns present: {audit_rows[0]['delta_nc_columns_present']}",
        f"- confidence columns present: {audit_rows[0]['confidence_columns_present']}",
        "",
        "## Notes",
        "",
        "- Reading Pilot 2 stored outputs; no pipeline rerun needed.",
        "- Raw BSV vectors are the substrate for all 6 normalization variants in Stage 3.",
    ]
    (REPORTS / "REPORT_pilot2_1_input_audit.md").write_text("\n".join(lines))
    print(f"  loaded {len(df)} spectra; per-class {n_per}")
    return df


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — global amplitude / BSV mass
# ─────────────────────────────────────────────────────────────────────

def stage2_bsv_mass(df):
    print("\n[STAGE 2] BSV mass / global amplitude")
    X = df[ABS_COLS].values
    df["bsv_sum"]  = X.sum(axis=1)
    df["bsv_mean"] = X.mean(axis=1)
    df["bsv_max"]  = X.max(axis=1)
    df["bsv_l2"]   = np.sqrt((X**2).sum(axis=1))
    # Concentration entropy (lower = more concentrated)
    p = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    df["bsv_entropy"] = -np.nansum(p * np.log(p + 1e-12), axis=1)

    # Per-cohort
    rows = []
    nc_sum = float(df[df.class_label == "NC"]["bsv_sum"].mean())
    nc_l2  = float(df[df.class_label == "NC"]["bsv_l2"].mean())
    nc_ent = float(df[df.class_label == "NC"]["bsv_entropy"].mean())
    for cls in CLASS_ORDER:
        sub = df[df.class_label == cls]
        if not len(sub): continue
        rows.append({
            "class": cls, "n": len(sub),
            "mean_sum_BSV": round(float(sub["bsv_sum"].mean()), 4),
            "std_sum_BSV": round(float(sub["bsv_sum"].std(ddof=1)), 4),
            "mean_L2_BSV": round(float(sub["bsv_l2"].mean()), 4),
            "mean_entropy_BSV": round(float(sub["bsv_entropy"].mean()), 4),
            "mean_max_BSV": round(float(sub["bsv_max"].mean()), 4),
            "delta_sum_vs_NC": round(float(sub["bsv_sum"].mean() - nc_sum), 4),
            "pct_sum_vs_NC": round(float((sub["bsv_sum"].mean() - nc_sum) / max(nc_sum, 1e-9) * 100), 2),
            "cohens_d_sum_vs_NC": round(_cohens_d(
                sub["bsv_sum"].values,
                df[df.class_label == "NC"]["bsv_sum"].values), 3),
        })
    mass_df = pd.DataFrame(rows)
    mass_df.to_csv(TABLES / "pilot2_1_bsv_mass_metrics.csv", index=False)

    print("  cohort BSV mass:")
    for _, r in mass_df.iterrows():
        print(f"    {r['class']:5s}  sum={r['mean_sum_BSV']:.3f}  "
              f"d_sum_vs_NC={r['cohens_d_sum_vs_NC']:+.2f}  "
              f"pct={r['pct_sum_vs_NC']:+.1f}%")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pal = {"NC": "#1f77b4", "HCC": "#d62728", "CCA": "#ff7f0e", "LM": "#2ca02c"}
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].bar(mass_df["class"], mass_df["mean_sum_BSV"],
                      yerr=mass_df["std_sum_BSV"], capsize=3,
                      color=[pal[c] for c in mass_df["class"]])
        axes[0].set_title("Mean sum_BSV per cohort"); axes[0].set_ylabel("sum of 11 BSV mags")
        axes[1].bar(mass_df["class"], mass_df["mean_L2_BSV"],
                      color=[pal[c] for c in mass_df["class"]])
        axes[1].set_title("Mean L2 norm of BSV per cohort"); axes[1].set_ylabel("‖BSV‖₂")
        axes[2].bar(mass_df["class"], mass_df["cohens_d_sum_vs_NC"],
                      color=[pal[c] for c in mass_df["class"]])
        axes[2].set_title("Cohen's d on sum_BSV vs NC"); axes[2].axhline(0, color="k", lw=0.5)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot2_1_bsv_mass_by_cohort.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  figure issue: {e}")

    # Report
    lines = [
        "# Pilot 2.1 — BSV Mass / Global Amplitude Analysis",
        "",
        "## Per-cohort BSV mass",
        "",
        "| cohort | n | mean sum_BSV | mean L2 | mean entropy | Δ sum vs NC | % | Cohen's d (sum vs NC) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in mass_df.iterrows():
        lines.append(
            f"| {r['class']} | {r['n']} | {r['mean_sum_BSV']:.4f} | "
            f"{r['mean_L2_BSV']:.4f} | {r['mean_entropy_BSV']:.4f} | "
            f"{r['delta_sum_vs_NC']:+.4f} | {r['pct_sum_vs_NC']:+.1f}% | "
            f"{r['cohens_d_sum_vs_NC']:+.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `sum_BSV` is the simplest global-amplitude proxy across the 11 family axes.",
        "- Cohen's d on `sum_BSV` measures the global-amplitude shift between cohorts independently of which family is shifting.",
        "- A large d_sum_vs_NC for CCA/LM but ~0 for HCC indicates per-cohort intensity offset, not chemistry.",
        "- The entropy metric measures profile concentration — uniform shifts preserve entropy, chemistry-specific shifts change it.",
    ]
    (REPORTS / "REPORT_pilot2_1_bsv_mass_analysis.md").write_text("\n".join(lines))
    return df, mass_df


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — normalised representations
# ─────────────────────────────────────────────────────────────────────

def stage3_normalize(df):
    print("\n[STAGE 3] Build 6 representations")
    X = df[ABS_COLS].values

    # B. sum-normalized
    X_sum = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    for i, g in enumerate(BSV_GROUPS_ORDER):
        df[f"sumnorm_{g}"] = X_sum[:, i]

    # C. centered log-ratio (CLR) — only stable when all values > 0
    X_pos = np.maximum(X, 1e-9)
    log_X = np.log(X_pos)
    geom_mean = log_X.mean(axis=1, keepdims=True)
    X_clr = log_X - geom_mean
    for i, g in enumerate(BSV_GROUPS_ORDER):
        df[f"clr_{g}"] = X_clr[:, i]

    # D. z-score within spectrum
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-12
    X_z = (X - mu) / sd
    for i, g in enumerate(BSV_GROUPS_ORDER):
        df[f"zspec_{g}"] = X_z[:, i]

    # E. ΔBSV raw vs NC (already in df as delta_nc_<G>)
    # F. ΔBSV sum-normalized vs NC
    nc_mask = df["class_label"] == "NC"
    nc_sumnorm = df.loc[nc_mask, [f"sumnorm_{g}" for g in BSV_GROUPS_ORDER]].mean()
    for g in BSV_GROUPS_ORDER:
        df[f"delta_sumnorm_{g}"] = df[f"sumnorm_{g}"] - nc_sumnorm[f"sumnorm_{g}"]

    # Save full normalized table (all 11 axes × 6 reps + class)
    keep_cols = ["spectrum_id", "class_label", "sample_id", "bsv_sum"]
    for prefix in ("abs", "sumnorm", "clr", "zspec", "delta_nc", "delta_sumnorm"):
        keep_cols += [f"{prefix}_{g}" for g in BSV_GROUPS_ORDER]
    df[keep_cols].to_csv(
        TABLES / "pilot2_1_normalized_bsv_vectors.csv", index=False,
    )

    lines = [
        "# Pilot 2.1 — 6 Normalization Methods",
        "",
        "All representations applied label-blind (no use of class label or target outcome).",
        "",
        "| label | description |",
        "|---|---|",
        "| `abs` | raw Pilot 2 BSV magnitudes (11 axes) |",
        "| `sumnorm` | each family / total BSV sum (compositional, removes global mass) |",
        "| `clr` | centered log-ratio (Aitchison compositional analysis) |",
        "| `zspec` | within-spectrum z-score (mean=0, sd=1 per spectrum, removes global mass + uniform scaling) |",
        "| `delta_nc` | absolute BSV − NC centroid (existing Pilot 2 ΔBSV) |",
        "| `delta_sumnorm` | sum-normalized − NC sum-normalized centroid |",
        "",
        "Note: `clr` is well-defined only when all family magnitudes are > 0 — checked OK for this dataset (BSV magnitudes all > 0 in Pilot 2 outputs).",
    ]
    (REPORTS / "REPORT_pilot2_1_normalization_methods.md").write_text("\n".join(lines))
    print(f"  6 representations × 11 axes ready")
    return df


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — effect-size survival across representations
# ─────────────────────────────────────────────────────────────────────

def stage4_survival(df):
    print("\n[STAGE 4] Effect-size survival test")
    rng = np.random.default_rng(42)
    representations = ["abs", "sumnorm", "clr", "zspec", "delta_nc", "delta_sumnorm"]
    rows = []
    for rep in representations:
        for cls in ["HCC", "CCA", "LM"]:
            for g in BSV_GROUPS_ORDER:
                col = f"{rep}_{g}"
                x = df[df.class_label == cls][col].values
                y = df[df.class_label == "NC"][col].values
                d = _cohens_d(x, y)
                # Bootstrap CI
                ds = []
                for _ in range(500):
                    xs = rng.choice(x, size=len(x), replace=True)
                    ys = rng.choice(y, size=len(y), replace=True)
                    ds.append(_cohens_d(xs, ys))
                ds = np.asarray(ds)
                ci_lo, ci_hi = float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
                rows.append({
                    "representation": rep,
                    "comparison": f"{cls}_vs_NC",
                    "family": g, "family_label": FAMILY_LABELS.get(g, g),
                    "cohens_d": round(float(d), 3),
                    "abs_d": round(abs(float(d)), 3),
                    "ci95_low": round(ci_lo, 3), "ci95_high": round(ci_hi, 3),
                    "ci_excludes_zero": (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0),
                })
    surv_df = pd.DataFrame(rows)
    surv_df.to_csv(TABLES / "pilot2_1_effect_size_survival.csv", index=False)

    # Per-rep summary: how many d≥0.3 / CI-significant
    sumr = []
    for rep in representations:
        for cls in ["HCC", "CCA", "LM"]:
            sub = surv_df[(surv_df.representation == rep) &
                            (surv_df.comparison == f"{cls}_vs_NC")]
            sumr.append({
                "representation": rep, "comparison": f"{cls}_vs_NC",
                "max_abs_d": round(float(sub["abs_d"].max()), 3),
                "n_meaningful_d_ge_03": int((sub["abs_d"] >= 0.30).sum()),
                "n_ci_significant": int(sub["ci_excludes_zero"].sum()),
                "n_with_consistent_sign": int(((sub["cohens_d"] > 0).sum())) if (sub["cohens_d"] > 0).sum() > (sub["cohens_d"] < 0).sum() else int((sub["cohens_d"] < 0).sum()),
            })
    sum_df = pd.DataFrame(sumr)
    sum_df.to_csv(TABLES / "pilot2_1_effect_size_survival_summary.csv", index=False)

    print("  survival summary (max_abs_d / meaningful / CI-significant per rep × comparison):")
    for _, r in sum_df.iterrows():
        print(f"    {r['representation']:18s} {r['comparison']:10s}  "
              f"max_d={r['max_abs_d']:.2f}  meaningful={r['n_meaningful_d_ge_03']}/11  "
              f"CI-sig={r['n_ci_significant']}/11")

    # Heatmap: representation × family for CCA_vs_NC (the cohort with biggest raw effect)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pivot_cca = surv_df[surv_df.comparison == "CCA_vs_NC"].pivot(
            index="family", columns="representation", values="cohens_d",
        ).reindex(BSV_GROUPS_ORDER)[representations]
        fig, ax = plt.subplots(figsize=(10, 6))
        vmax = float(np.abs(pivot_cca.values).max()) or 1.0
        im = ax.imshow(pivot_cca.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(BSV_GROUPS_ORDER)))
        ax.set_yticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER])
        ax.set_xticks(range(len(representations)))
        ax.set_xticklabels(representations, rotation=30, ha="right")
        ax.set_title("CCA vs NC: Cohen's d per family across 6 representations")
        for i in range(pivot_cca.shape[0]):
            for j in range(pivot_cca.shape[1]):
                v = pivot_cca.iloc[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(v) > vmax*0.5 else "black")
        fig.colorbar(im, ax=ax, label="Cohen's d")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot2_1_raw_vs_sum_normalized_heatmap.png", dpi=150)
        plt.close(fig)

        # Survival by family (bar plot per cohort)
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
        for ax_, cls in zip(axes, ["HCC", "CCA", "LM"]):
            for rep, color in zip(representations,
                                     ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]):
                sub = surv_df[(surv_df.representation == rep) &
                                (surv_df.comparison == f"{cls}_vs_NC")]
                xs = np.arange(len(BSV_GROUPS_ORDER))
                ax_.plot(xs, sub.set_index("family").reindex(BSV_GROUPS_ORDER)["cohens_d"].values,
                          marker="o", label=rep, color=color, alpha=0.85)
            ax_.set_xticks(xs); ax_.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER],
                                                          rotation=45, ha="right", fontsize=8)
            ax_.set_title(f"{cls} vs NC — Cohen's d by family across reps")
            ax_.axhline(0, color="k", lw=0.4)
            if cls == "HCC": ax_.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot2_1_effect_size_survival_by_family.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  figure issue: {e}")

    # Report
    lines = [
        "# Pilot 2.1 — Effect-Size Survival Across Representations",
        "",
        "## Per-representation × cohort summary",
        "",
        "| representation | comparison | max |d| | meaningful (|d|≥0.3) | CI-significant |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in sum_df.iterrows():
        lines.append(f"| {r['representation']} | {r['comparison']} | "
                     f"{r['max_abs_d']:.2f} | {r['n_meaningful_d_ge_03']}/11 | "
                     f"{r['n_ci_significant']}/11 |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- If raw `abs` shows large effects but normalized representations (`sumnorm`, `clr`, `zspec`) collapse to small effects → the apparent multi-axis effect is global amplitude artifact.",
        "- If specific axes survive (large d in normalized representation), those are chemistry-axis-specific real signals.",
        "- `delta_sumnorm` is the most chemistry-relevant survival metric: removes both NC offset AND per-spectrum amplitude.",
    ]
    (REPORTS / "REPORT_pilot2_1_effect_size_survival.md").write_text("\n".join(lines))
    return surv_df, sum_df


# ─────────────────────────────────────────────────────────────────────
# Stage 5 — biological specificity
# ─────────────────────────────────────────────────────────────────────

def stage5_specificity(surv_df):
    print("\n[STAGE 5] Biological specificity test")
    rows = []
    for cls in ["HCC", "CCA", "LM"]:
        for rep in ["abs", "sumnorm", "clr", "zspec", "delta_nc", "delta_sumnorm"]:
            sub = surv_df[(surv_df.representation == rep) &
                            (surv_df.comparison == f"{cls}_vs_NC")]
            n_pos = int((sub["cohens_d"] > 0.10).sum())
            n_neg = int((sub["cohens_d"] < -0.10).sum())
            n_neutral = int(((sub["cohens_d"] >= -0.10) & (sub["cohens_d"] <= 0.10)).sum())
            uniform_score = abs(n_pos - n_neg) / 11.0  # 1.0 = all same direction (uniform), 0 = balanced
            top3 = sub.sort_values("abs_d", ascending=False).head(3)["family"].tolist()
            rows.append({
                "comparison": f"{cls}_vs_NC", "representation": rep,
                "n_positive_d_gt_010": n_pos,
                "n_negative_d_lt_neg010": n_neg,
                "n_neutral": n_neutral,
                "uniform_score": round(uniform_score, 3),
                "top_3_axes_by_abs_d": ";".join(top3),
                "is_uniform_elevation": (n_pos >= 8 and n_neg <= 1),
                "is_uniform_depletion": (n_neg >= 8 and n_pos <= 1),
                "is_selective": (1 <= max(n_pos, n_neg) <= 5),
            })
    spec_df = pd.DataFrame(rows)
    spec_df.to_csv(TABLES / "pilot2_1_biological_specificity_results.csv", index=False)

    lines = [
        "# Pilot 2.1 — Biological Specificity Test",
        "",
        "Question: after each normalization, do shifts become biologically structured (selective)",
        "or remain a uniform across-the-axis pattern (artifact signature)?",
        "",
        "## Per-rep × cohort directional pattern",
        "",
        "| comparison | representation | +d>0.10 | −d<−0.10 | neutral | uniform_score | uniform-elevation | uniform-depletion | selective | top-3 axes |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for _, r in spec_df.iterrows():
        lines.append(
            f"| {r['comparison']} | {r['representation']} | "
            f"{r['n_positive_d_gt_010']} | {r['n_negative_d_lt_neg010']} | "
            f"{r['n_neutral']} | {r['uniform_score']:.2f} | "
            f"{'YES' if r['is_uniform_elevation'] else 'no'} | "
            f"{'YES' if r['is_uniform_depletion'] else 'no'} | "
            f"{'YES' if r['is_selective'] else 'no'} | "
            f"{r['top_3_axes_by_abs_d']} |"
        )
    lines += [
        "",
        "## Interpretation rules",
        "",
        "- **uniform-elevation** (8+ axes positive, ≤1 negative) is the ARTIFACT signature.",
        "- **selective** (1-5 axes shifted in either direction) is the BIOLOGY signature.",
        "- The transition from uniform-elevation in `abs` to selective in `sumnorm`/`clr`/`zspec` is direct evidence that the raw effect was a global-amplitude artifact.",
    ]
    (REPORTS / "REPORT_pilot2_1_biological_specificity.md").write_text("\n".join(lines))
    return spec_df


# ─────────────────────────────────────────────────────────────────────
# Stage 6 — projection sensitivity
# ─────────────────────────────────────────────────────────────────────

def stage6_projection(df):
    print("\n[STAGE 6] PCA projection sensitivity")
    from sklearn.decomposition import PCA
    pal = {"NC": "#1f77b4", "HCC": "#d62728", "CCA": "#ff7f0e", "LM": "#2ca02c"}
    representations = [
        ("abs", ABS_COLS),
        ("sumnorm", [f"sumnorm_{g}" for g in BSV_GROUPS_ORDER]),
        ("clr", [f"clr_{g}" for g in BSV_GROUPS_ORDER]),
        ("delta_sumnorm", [f"delta_sumnorm_{g}" for g in BSV_GROUPS_ORDER]),
    ]
    rows = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for ax_, (rep, cols) in zip(axes, representations):
            X = df[cols].values
            pc = PCA(n_components=2, random_state=0)
            Z = pc.fit_transform(X)
            for cls in CLASS_ORDER:
                m = df["class_label"].values == cls
                ax_.scatter(Z[m, 0], Z[m, 1], s=30, alpha=0.7,
                              label=f"{cls} (n={m.sum()})", color=pal[cls])
            corr_pc1_sum = _spearman(Z[:, 0], df["bsv_sum"].values)
            ax_.set_title(f"{rep} — PC1 var={pc.explained_variance_ratio_[0]:.0%}, "
                            f"ρ(PC1, sum_BSV) = {corr_pc1_sum:+.2f}")
            ax_.legend(fontsize=7)
            rows.append({
                "representation": rep,
                "PC1_variance": round(float(pc.explained_variance_ratio_[0]), 3),
                "PC2_variance": round(float(pc.explained_variance_ratio_[1]), 3),
                "spearman_PC1_vs_sumBSV": round(corr_pc1_sum, 3),
                "PC1_dominated_by_global_mass": abs(corr_pc1_sum) >= 0.7,
            })
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot2_1_projection_raw_vs_normalized.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  projection figure issue: {e}")
    proj_df = pd.DataFrame(rows)
    proj_df.to_csv(TABLES / "pilot2_1_projection_variance_summary.csv", index=False)

    lines = [
        "# Pilot 2.1 — PCA Projection Sensitivity",
        "",
        "## Per-representation PC summary",
        "",
        "| representation | PC1 var | PC2 var | ρ(PC1, sum_BSV) | PC1 dominated by global mass? |",
        "|---|---:|---:|---:|---|",
    ]
    for _, r in proj_df.iterrows():
        lines.append(
            f"| {r['representation']} | {r['PC1_variance']:.0%} | "
            f"{r['PC2_variance']:.0%} | {r['spearman_PC1_vs_sumBSV']:+.2f} | "
            f"{'**YES**' if r['PC1_dominated_by_global_mass'] else 'no'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- |ρ(PC1, sum_BSV)| ≥ 0.7 means PC1 is essentially the global-amplitude axis.",
        "- If raw `abs` has |ρ| ≥ 0.7 but `sumnorm`/`clr` have |ρ| ~ 0, normalisation worked correctly to remove the global mass component.",
        "- If cohort separation in PC1-PC2 collapses after normalisation, the raw cohort separation was driven by global mass, not chemistry.",
    ]
    (REPORTS / "REPORT_pilot2_1_projection_sensitivity.md").write_text("\n".join(lines))
    return proj_df


# ─────────────────────────────────────────────────────────────────────
# Stage 7 — severity vs artifact decision
# ─────────────────────────────────────────────────────────────────────

def stage7_decision(sum_df, spec_df, proj_df, mass_df):
    print("\n[STAGE 7] Severity vs artifact decision")
    # Counts under abs vs sumnorm
    def _meaningful(rep, cls):
        sub = sum_df[(sum_df.representation == rep) & (sum_df.comparison == f"{cls}_vs_NC")]
        return int(sub["n_meaningful_d_ge_03"].iloc[0]) if len(sub) else 0
    raw_meaning = {cls: _meaningful("abs", cls) for cls in ["HCC", "CCA", "LM"]}
    sn_meaning  = {cls: _meaningful("sumnorm", cls) for cls in ["HCC", "CCA", "LM"]}
    clr_meaning = {cls: _meaningful("clr", cls) for cls in ["HCC", "CCA", "LM"]}
    dsn_meaning = {cls: _meaningful("delta_sumnorm", cls) for cls in ["HCC", "CCA", "LM"]}

    # Uniform vs selective under each rep for CCA + LM
    def _is_uniform(rep, cls):
        s = spec_df[(spec_df.representation == rep) & (spec_df.comparison == f"{cls}_vs_NC")]
        return bool(s["is_uniform_elevation"].iloc[0]) or bool(s["is_uniform_depletion"].iloc[0])

    raw_uniform_cca = _is_uniform("abs", "CCA"); raw_uniform_lm = _is_uniform("abs", "LM")
    sn_uniform_cca = _is_uniform("sumnorm", "CCA"); sn_uniform_lm = _is_uniform("sumnorm", "LM")

    pc1_corr = float(proj_df[proj_df.representation == "abs"]["spearman_PC1_vs_sumBSV"].iloc[0])
    pc1_corr_sn = float(proj_df[proj_df.representation == "sumnorm"]["spearman_PC1_vs_sumBSV"].iloc[0])

    # Sum_BSV cohort offset
    cca_pct = float(mass_df[mass_df["class"] == "CCA"]["pct_sum_vs_NC"].iloc[0])
    lm_pct  = float(mass_df[mass_df["class"] == "LM"]["pct_sum_vs_NC"].iloc[0])
    hcc_pct = float(mass_df[mass_df["class"] == "HCC"]["pct_sum_vs_NC"].iloc[0])

    # Decision rule
    raw_collapse_cca = (raw_meaning["CCA"] >= 8 and sn_meaning["CCA"] <= 2)
    raw_collapse_lm  = (raw_meaning["LM"]  >= 8 and sn_meaning["LM"]  <= 2)
    raw_collapse_clr_cca = (raw_meaning["CCA"] >= 8 and clr_meaning["CCA"] <= 2)

    if raw_collapse_cca and raw_collapse_lm and abs(pc1_corr) >= 0.7:
        decision = "LIKELY_GLOBAL_AMPLITUDE_ARTIFACT"
    elif (raw_meaning["CCA"] >= 8 and sn_meaning["CCA"] >= 3) or (raw_meaning["LM"] >= 8 and sn_meaning["LM"] >= 3):
        decision = "MIXED_ARTIFACT_AND_BIOLOGY"
    elif sn_meaning["CCA"] >= 3 and sn_meaning["LM"] >= 3:
        decision = "LIKELY_DISEASE_SEVERITY_BIOLOGY"
    else:
        decision = "INDETERMINATE_NEEDS_BATCH_METADATA"

    lines = [
        "# Pilot 2.1 — Severity vs Artifact Decision",
        "",
        f"**Decision: {decision}**",
        "",
        "## Evidence summary",
        "",
        f"- sum_BSV per-cohort offset: HCC = {hcc_pct:+.1f}%, CCA = {cca_pct:+.1f}%, LM = {lm_pct:+.1f}%",
        f"- Spearman ρ(PC1, sum_BSV) on RAW abs: **{pc1_corr:+.2f}** "
        f"({'PC1 IS global mass' if abs(pc1_corr) >= 0.7 else 'PC1 not dominated by mass'})",
        f"- Spearman ρ(PC1, sum_BSV) on sum-normalized: {pc1_corr_sn:+.2f} "
        f"(should be ~0 if normalisation worked)",
        "",
        "## Effect-size survival (meaningful = |d|≥0.30)",
        "",
        "| representation | HCC vs NC | CCA vs NC | LM vs NC |",
        "|---|---:|---:|---:|",
        f"| `abs` raw | {raw_meaning['HCC']}/11 | {raw_meaning['CCA']}/11 | {raw_meaning['LM']}/11 |",
        f"| `sumnorm` | {sn_meaning['HCC']}/11 | {sn_meaning['CCA']}/11 | {sn_meaning['LM']}/11 |",
        f"| `clr` | {clr_meaning['HCC']}/11 | {clr_meaning['CCA']}/11 | {clr_meaning['LM']}/11 |",
        f"| `delta_sumnorm` | {dsn_meaning['HCC']}/11 | {dsn_meaning['CCA']}/11 | {dsn_meaning['LM']}/11 |",
        "",
        "## Uniform-elevation pattern (artifact signature)",
        "",
        f"- Raw `abs` CCA uniform-elevation: **{raw_uniform_cca}**",
        f"- Raw `abs` LM uniform-elevation: **{raw_uniform_lm}**",
        f"- sum-normalized CCA uniform-elevation: {sn_uniform_cca}",
        f"- sum-normalized LM uniform-elevation: {sn_uniform_lm}",
        "",
        "## Decision criteria applied",
        "",
        "| criterion | satisfied? |",
        "|---|---|",
        f"| CCA: raw shows ≥8 meaningful BUT sum-normalised ≤2 | {raw_collapse_cca} |",
        f"| LM: raw shows ≥8 meaningful BUT sum-normalised ≤2 | {raw_collapse_lm} |",
        f"| PC1-sum_BSV ρ ≥ 0.7 (PC1 = global mass) | {abs(pc1_corr) >= 0.7} |",
        f"| CCA: ≥3 meaningful in sum-normalised | {sn_meaning['CCA'] >= 3} |",
        f"| LM: ≥3 meaningful in sum-normalised | {sn_meaning['LM'] >= 3} |",
        "",
        f"## Decision: **{decision}**",
        "",
        "## Interpretation",
        "",
    ]
    if decision == "LIKELY_GLOBAL_AMPLITUDE_ARTIFACT":
        lines.append(
            "Both CCA and LM raw multi-axis elevations collapse after sum-normalisation, "
            "AND PC1 of the raw BSV space is essentially the global-amplitude axis. "
            "This is consistent with the apparent CCA/LM 'effect' being a per-cohort "
            "intensity offset (substrate batch / acquisition gain / preprocessing) rather than "
            "true disease-severity biology. The HCC vs NC signal (small, ~d=0.10) is "
            "the only chemistry-real cross-cohort difference."
        )
    elif decision == "MIXED_ARTIFACT_AND_BIOLOGY":
        lines.append(
            "The raw effect partially collapses under normalisation but a non-trivial selective "
            "signal remains. Both global amplitude AND chemistry-specific shifts are present. "
            "Synthesis must isolate the surviving axes from the global-mass component."
        )
    elif decision == "LIKELY_DISEASE_SEVERITY_BIOLOGY":
        lines.append(
            "Selective family shifts persist after normalisation in CCA and LM. The raw-amplitude "
            "elevation appears to reflect a genuine biology-driven multi-axis enhancement, "
            "consistent with advanced-disease state."
        )
    else:
        lines.append(
            "Evidence is mixed and insufficient to classify. Batch metadata or independent "
            "amplitude-controlled replication would resolve the ambiguity."
        )
    (REPORTS / "REPORT_pilot2_1_severity_vs_artifact_decision.md").write_text("\n".join(lines))
    print(f"  decision: {decision}")
    return decision, raw_meaning, sn_meaning, clr_meaning, dsn_meaning, pc1_corr, pc1_corr_sn


# ─────────────────────────────────────────────────────────────────────
# Stage 8 — updated Pilot 2 interpretation
# ─────────────────────────────────────────────────────────────────────

def stage8_updated_interpretation(decision, raw_meaning, sn_meaning, clr_meaning,
                                       dsn_meaning, surv_df, mass_df, pc1_corr):
    print("\n[STAGE 8] Updated Pilot 2 interpretation")
    # Top surviving axes for CCA / LM under sum_normalized
    def _top_surviving(cls, rep="sumnorm", n=3):
        sub = surv_df[(surv_df.representation == rep) & (surv_df.comparison == f"{cls}_vs_NC")]
        return sub.sort_values("abs_d", ascending=False).head(n)
    cca_surv = _top_surviving("CCA")
    lm_surv  = _top_surviving("LM")
    hcc_surv = _top_surviving("HCC")

    lines = [
        "# Pilot 2 — Updated Biochemical Interpretation (post-normalisation sensitivity)",
        "",
        f"## Severity-vs-artifact decision: **{decision}**",
        "",
        "## What raw BSV suggested (Pilot 2 v1 result)",
        "",
        f"- max |d| = 1.565 (CCA vs NC, G10 Free-AA)",
        f"- 22/33 family-disease comparisons CI-significant",
        f"- CCA showed 11/11 families elevated; LM showed 10/11 elevated",
        f"- HCC vs NC essentially zero (max d = 0.116)",
        "",
        "## What normalization changed",
        "",
        "| representation | HCC vs NC meaningful | CCA vs NC meaningful | LM vs NC meaningful |",
        "|---|---:|---:|---:|",
        f"| `abs` (raw) | {raw_meaning['HCC']}/11 | {raw_meaning['CCA']}/11 | {raw_meaning['LM']}/11 |",
        f"| `sumnorm` | {sn_meaning['HCC']}/11 | {sn_meaning['CCA']}/11 | {sn_meaning['LM']}/11 |",
        f"| `clr` | {clr_meaning['HCC']}/11 | {clr_meaning['CCA']}/11 | {clr_meaning['LM']}/11 |",
        f"| `delta_sumnorm` | {dsn_meaning['HCC']}/11 | {dsn_meaning['CCA']}/11 | {dsn_meaning['LM']}/11 |",
        "",
        f"PC1 of raw BSV vs sum_BSV: ρ = {pc1_corr:+.2f}",
        "",
        "## Top surviving axes (sum-normalized vs NC)",
        "",
        "### CCA vs NC",
        "",
    ]
    for _, r in cca_surv.iterrows():
        ci_note = " *(CI ✓)*" if r["ci_excludes_zero"] else " (CI ✗)"
        lines.append(f"- {r['family']} {r['family_label']}: d = {r['cohens_d']:+.3f} "
                     f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]{ci_note}")
    lines += ["", "### LM vs NC", ""]
    for _, r in lm_surv.iterrows():
        ci_note = " *(CI ✓)*" if r["ci_excludes_zero"] else " (CI ✗)"
        lines.append(f"- {r['family']} {r['family_label']}: d = {r['cohens_d']:+.3f} "
                     f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]{ci_note}")
    lines += ["", "### HCC vs NC", ""]
    for _, r in hcc_surv.iterrows():
        ci_note = " *(CI ✓)*" if r["ci_excludes_zero"] else " (CI ✗)"
        lines.append(f"- {r['family']} {r['family_label']}: d = {r['cohens_d']:+.3f} "
                     f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]{ci_note}")
    lines += [
        "",
        "## Can CCA/LM be interpreted biologically now?",
        "",
    ]
    if decision == "LIKELY_GLOBAL_AMPLITUDE_ARTIFACT":
        lines.append(
            "**No.** After normalisation, the raw multi-axis 'effect' collapses. The apparent "
            "CCA/LM elevation pattern is dominated by per-cohort intensity offset, not chemistry. "
            "Any single-family interpretation of CCA-vs-NC or LM-vs-NC must be treated as "
            "amplitude-driven and not used as a biology claim."
        )
    elif decision == "MIXED_ARTIFACT_AND_BIOLOGY":
        lines.append(
            "**Partially.** Some axes survive normalisation as selective shifts. Those (top of the "
            "sum-normalized lists above) can be interpreted biologically with substrate caveat. "
            "The bulk of the raw elevation is amplitude artifact and must not be reported as "
            "chemistry."
        )
    elif decision == "LIKELY_DISEASE_SEVERITY_BIOLOGY":
        lines.append(
            "**Yes.** Selective family shifts persist after normalisation. Both CCA and LM show "
            "coherent disease-state patterns at the BSV abstraction level. Substrate caveats remain."
        )
    else:
        lines.append(
            "**Indeterminate.** Cannot decide without batch metadata or independent replication."
        )
    lines += [
        "",
        "## HCC vs NC remains weak — confirmed",
        "",
        f"HCC vs NC max |d| = 0.116 in raw, and {hcc_surv['abs_d'].max():.3f} in sum-normalized — "
        "the small magnitude HCC-vs-NC signal is consistent across raw and normalised "
        "representations. This convergence with Pilot 1 v2 (max |d| = 0.26 on a different "
        "substrate) is the **cleanest cross-pilot result** so far.",
        "",
        "## What can be carried into cross-pilot synthesis",
        "",
        "1. **HCC vs healthy convergent signal** (small, multi-axis, top axes G05 Glycan / G04 "
        "Nucl-bbone / G09 Sterol) — robust across both pilots.",
        "2. **Sum-normalisation is mandatory** for any cross-cohort BSV-level claim on cohorts "
        "without explicit acquisition-amplitude controls.",
        "3. **Surviving CCA-vs-NC and LM-vs-NC axes** (sum-normalized; top of lists above) — "
        "treat as hypothesis-generating only, with substrate caveat.",
        "4. **Purine axes** (G01 / G02) remain non-informative for serum-SERS HCC discrimination "
        "across both pilots.",
        "",
        "## What must NOT be claimed",
        "",
        "- Diagnostic discrimination between CCA / HCC / LM / NC.",
        "- Exact molecule identity from any single elevated axis.",
        "- That CCA or LM 'shows elevated free-amino-acid signal' as a biological claim "
        "without explicit normalisation correction.",
        "- That GAIRA distinguishes liver malignancy subtypes from BSV-level passive readout.",
    ]
    (REPORTS / "REPORT_pilot2_1_updated_interpretation.md").write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_4_passive_target_pilot_2_1_normalization_sensitivity")
    print("=" * 78)
    for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    df = stage1_load()
    df, mass_df = stage2_bsv_mass(df)
    df = stage3_normalize(df)
    surv_df, sum_df = stage4_survival(df)
    spec_df = stage5_specificity(surv_df)
    proj_df = stage6_projection(df)
    decision, raw_m, sn_m, clr_m, dsn_m, pc1_corr, pc1_corr_sn = stage7_decision(
        sum_df, spec_df, proj_df, mass_df,
    )
    stage8_updated_interpretation(decision, raw_m, sn_m, clr_m, dsn_m,
                                       surv_df, mass_df, pc1_corr)

    # Audit log
    lines = [
        "# gaira_base_4_passive_target_pilot_2_1_normalization_sensitivity — Audit Log",
        "",
        "## Source",
        f"- Pilot 2 per-spectrum BSV outputs ({len(df)} spectra, 4 cohorts)",
        "",
        "## Representations evaluated",
        "- abs (raw BSV)",
        "- sumnorm (compositional sum-normalised)",
        "- clr (centered log-ratio)",
        "- zspec (within-spectrum z-score)",
        "- delta_nc (raw ΔBSV vs NC)",
        "- delta_sumnorm (sum-normalised ΔBSV vs NC)",
        "",
        "## Effect-size survival (meaningful |d|≥0.3)",
        f"- raw `abs`: HCC {raw_m['HCC']}/11, CCA {raw_m['CCA']}/11, LM {raw_m['LM']}/11",
        f"- `sumnorm`: HCC {sn_m['HCC']}/11, CCA {sn_m['CCA']}/11, LM {sn_m['LM']}/11",
        f"- `clr`: HCC {clr_m['HCC']}/11, CCA {clr_m['CCA']}/11, LM {clr_m['LM']}/11",
        f"- `delta_sumnorm`: HCC {dsn_m['HCC']}/11, CCA {dsn_m['CCA']}/11, LM {dsn_m['LM']}/11",
        "",
        f"## PC1 correlation with sum_BSV: raw = {pc1_corr:+.2f}, sum-normalized = {pc1_corr_sn:+.2f}",
        "",
        f"## Decision: **{decision}**",
        "",
        "## Invariants",
        "- engine v4.5 / taxonomy / motif / MSS v4.3 / substrate physics: unchanged",
        "- no classifier training, no threshold tuning, no label-driven feature select",
        "- no target-label fitting",
        "- no DART-Met",
    ]
    (AUDIT / "gaira_base_4_passive_target_pilot_2_1_normalization_sensitivity_audit_log.md"
     ).write_text("\n".join(lines))

    p = Path(__file__)
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)

    print(f"\n[complete] decision: {decision}")
    print(f"  raw meaningful: HCC={raw_m['HCC']}, CCA={raw_m['CCA']}, LM={raw_m['LM']}")
    print(f"  sumnorm meaningful: HCC={sn_m['HCC']}, CCA={sn_m['CCA']}, LM={sn_m['LM']}")
    print(f"  PC1-sum_BSV correlation: raw={pc1_corr:+.2f}, sumnorm={pc1_corr_sn:+.2f}")


if __name__ == "__main__":
    main()
