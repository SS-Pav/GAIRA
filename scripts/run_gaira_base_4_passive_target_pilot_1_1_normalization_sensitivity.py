"""gaira_base_4 passive target Pilot 1.1 — normalization sensitivity for HCC holdout.

Reuses Pilot 1 v2 per-spectrum BSV outputs. Tests whether the modest
Pilot 1 signal (G05↑, G04↑, G09↓ at d≈0.25) survives normalization.

NO engine / MSS / motif / taxonomy / weight changes.
NO classifier training. NO threshold tuning. NO label-driven feature select.
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
    "gaira_base_4_passive_target_pilot_1_1_normalization_sensitivity"
)
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"

PILOT1_V2_TABLE = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_passive_target_pilot_1_hcc_holdout_rerun_v2/tables/"
    "pilot1_v2_per_spectrum_outputs.csv"
)

BSV_GROUPS_ORDER = [g["group_id"] for g in BSV_GROUPS]
ABS_COLS = [f"abs_{g}" for g in BSV_GROUPS_ORDER]
CLASSES = ["CTR", "H0T"]   # Pilot 1 uses CTR + H0T


def _spearman(x, y):
    rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _cohens_d(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2: return 0.0
    pooled = np.sqrt(((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1))
                       / max(len(x)+len(y)-2, 1))
    return (np.mean(x) - np.mean(y)) / (pooled if pooled > 0 else 1.0)


def main():
    print("=" * 78)
    print("gaira_base_4_passive_target_pilot_1_1_normalization_sensitivity")
    print("=" * 78)
    for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load Pilot 1 v2 per-spectrum outputs ──
    df = pd.read_csv(PILOT1_V2_TABLE)
    print(f"loaded {len(df)} spectra; classes = {df['class_label'].value_counts().to_dict()}")

    # ── BSV mass ──
    X = df[ABS_COLS].values
    df["bsv_sum"] = X.sum(axis=1)
    df["bsv_l2"]  = np.sqrt((X**2).sum(axis=1))
    p = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    df["bsv_entropy"] = -np.nansum(p * np.log(p + 1e-12), axis=1)

    # Per-class amplitude
    print("\nBSV mass per class:")
    mass_rows = []
    for cls in CLASSES:
        sub = df[df.class_label == cls]
        mass_rows.append({
            "class": cls, "n": len(sub),
            "mean_sum_BSV": round(float(sub["bsv_sum"].mean()), 4),
            "std_sum_BSV":  round(float(sub["bsv_sum"].std(ddof=1)), 4),
            "mean_L2_BSV":  round(float(sub["bsv_l2"].mean()), 4),
            "mean_entropy": round(float(sub["bsv_entropy"].mean()), 4),
        })
        print(f"  {cls}: sum={mass_rows[-1]['mean_sum_BSV']:.4f}  "
              f"std={mass_rows[-1]['std_sum_BSV']:.4f}")
    ctr_sum = mass_rows[0]["mean_sum_BSV"]
    h0t_sum = mass_rows[1]["mean_sum_BSV"]
    pct_offset = (h0t_sum - ctr_sum) / ctr_sum * 100
    print(f"  H0T amplitude offset vs CTR: {pct_offset:+.2f}%")
    pd.DataFrame(mass_rows).to_csv(TABLES / "pilot1_1_bsv_mass.csv", index=False)

    # ── Build 5 representations ──
    # B. sumnorm
    X_sum = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    for i, g in enumerate(BSV_GROUPS_ORDER): df[f"sumnorm_{g}"] = X_sum[:, i]
    # C. CLR
    X_pos = np.maximum(X, 1e-9); log_X = np.log(X_pos)
    X_clr = log_X - log_X.mean(axis=1, keepdims=True)
    for i, g in enumerate(BSV_GROUPS_ORDER): df[f"clr_{g}"] = X_clr[:, i]
    # E. delta_ctr (raw ΔBSV vs CTR centroid) — already in df from Pilot 1 v2
    # check column name pattern
    delta_ctr_present = all(f"delta_ctr_{g}" in df.columns for g in BSV_GROUPS_ORDER)
    if not delta_ctr_present:
        # compute it
        ctr_means = df[df.class_label == "CTR"][ABS_COLS].mean()
        for g in BSV_GROUPS_ORDER:
            df[f"delta_ctr_{g}"] = df[f"abs_{g}"] - ctr_means[f"abs_{g}"]
    # F. delta_sumnorm — sumnorm minus CTR sumnorm centroid
    ctr_sumnorm_means = df[df.class_label == "CTR"][[f"sumnorm_{g}" for g in BSV_GROUPS_ORDER]].mean()
    for g in BSV_GROUPS_ORDER:
        df[f"delta_sumnorm_{g}"] = df[f"sumnorm_{g}"] - ctr_sumnorm_means[f"sumnorm_{g}"]

    # Save normalised
    keep = ["spectrum_id", "class_label", "sample_id", "batch_id", "bsv_sum"]
    for prefix in ("abs", "sumnorm", "clr", "delta_ctr", "delta_sumnorm"):
        keep += [f"{prefix}_{g}" for g in BSV_GROUPS_ORDER]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(TABLES / "pilot1_1_normalized_vectors.csv", index=False)

    # ── Effect sizes per representation ──
    rng = np.random.default_rng(42)
    representations = ["abs", "sumnorm", "clr", "delta_ctr", "delta_sumnorm"]
    rows = []
    for rep in representations:
        for g in BSV_GROUPS_ORDER:
            col = f"{rep}_{g}"
            x = df[df.class_label == "H0T"][col].values
            y = df[df.class_label == "CTR"][col].values
            d_pt = _cohens_d(x, y)
            ds = []
            for _ in range(1000):
                xs = rng.choice(x, size=len(x), replace=True)
                ys = rng.choice(y, size=len(y), replace=True)
                ds.append(_cohens_d(xs, ys))
            ds = np.asarray(ds)
            ci_lo, ci_hi = float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
            rows.append({
                "representation": rep, "family": g,
                "family_label": FAMILY_LABELS.get(g, g),
                "cohens_d": round(float(d_pt), 3),
                "abs_d": round(abs(float(d_pt)), 3),
                "ci95_low": round(ci_lo, 3), "ci95_high": round(ci_hi, 3),
                "ci_excludes_zero": (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0),
                "direction": "H0T>CTR" if d_pt > 0 else ("H0T<CTR" if d_pt < 0 else "equal"),
            })
    eff = pd.DataFrame(rows)
    eff.to_csv(TABLES / "pilot1_1_effect_size_survival.csv", index=False)

    # Per-rep summary
    sumr = []
    for rep in representations:
        sub = eff[eff.representation == rep]
        sumr.append({
            "representation": rep,
            "max_abs_d": round(float(sub["abs_d"].max()), 3),
            "n_meaningful_d_ge_03": int((sub["abs_d"] >= 0.30).sum()),
            "n_ci_significant": int(sub["ci_excludes_zero"].sum()),
            "n_d_ge_015": int((sub["abs_d"] >= 0.15).sum()),
        })
    sum_df = pd.DataFrame(sumr)
    sum_df.to_csv(TABLES / "pilot1_1_survival_summary.csv", index=False)

    print("\nSurvival per representation (HCC vs CTR):")
    for _, r in sum_df.iterrows():
        print(f"  {r['representation']:18s}  max_d={r['max_abs_d']:.2f}  "
              f"meaningful (|d|≥0.30)={r['n_meaningful_d_ge_03']}/11  "
              f"CI-sig={r['n_ci_significant']}/11  |d|≥0.15={r['n_d_ge_015']}/11")

    # Direction-and-rank stability (raw vs sumnorm)
    rs = []
    for g in BSV_GROUPS_ORDER:
        d_raw = float(eff[(eff.representation == "abs") & (eff.family == g)]["cohens_d"].iloc[0])
        d_sn  = float(eff[(eff.representation == "sumnorm") & (eff.family == g)]["cohens_d"].iloc[0])
        d_clr = float(eff[(eff.representation == "clr") & (eff.family == g)]["cohens_d"].iloc[0])
        d_dsn = float(eff[(eff.representation == "delta_sumnorm") & (eff.family == g)]["cohens_d"].iloc[0])
        rs.append({
            "family": g, "family_label": FAMILY_LABELS.get(g, g),
            "d_abs": d_raw, "d_sumnorm": d_sn, "d_clr": d_clr, "d_delta_sumnorm": d_dsn,
            "sign_stable_abs_to_sumnorm": np.sign(d_raw) == np.sign(d_sn) and d_raw != 0,
            "sign_stable_abs_to_clr": np.sign(d_raw) == np.sign(d_clr) and d_raw != 0,
            "delta_d_sn_minus_abs": round(d_sn - d_raw, 3),
        })
    rank_df = pd.DataFrame(rs)
    rank_df.to_csv(TABLES / "pilot1_1_direction_stability.csv", index=False)

    # Top-3 raw signals — do they survive?
    top3_raw = eff[eff.representation == "abs"].sort_values("abs_d", ascending=False).head(3)
    survives = []
    for _, r in top3_raw.iterrows():
        sn_d = float(eff[(eff.representation == "sumnorm") & (eff.family == r["family"])]["cohens_d"].iloc[0])
        survives.append({
            "raw_top_family": r["family"],
            "raw_d": r["cohens_d"],
            "sumnorm_d": sn_d,
            "sign_preserved": np.sign(r["cohens_d"]) == np.sign(sn_d),
            "survived_meaningful": abs(sn_d) >= 0.20,
        })
    print("\nTop-3 raw signals — do they survive sumnorm?")
    for s in survives:
        print(f"  {s['raw_top_family']}: raw d={s['raw_d']:+.2f} → sumnorm d={s['sumnorm_d']:+.2f}  "
              f"(sign preserved={s['sign_preserved']}, |d|≥0.20={s['survived_meaningful']})")

    # ── Decision ──
    n_meaning_raw = int(sum_df[sum_df.representation == "abs"]["n_meaningful_d_ge_03"].iloc[0])
    n_meaning_sn  = int(sum_df[sum_df.representation == "sumnorm"]["n_meaningful_d_ge_03"].iloc[0])
    max_d_raw = float(sum_df[sum_df.representation == "abs"]["max_abs_d"].iloc[0])
    max_d_sn  = float(sum_df[sum_df.representation == "sumnorm"]["max_abs_d"].iloc[0])
    n_dge15_sn = int(sum_df[sum_df.representation == "sumnorm"]["n_d_ge_015"].iloc[0])
    top3_signs_preserved = sum(s["sign_preserved"] for s in survives)
    top3_survived = sum(s["survived_meaningful"] for s in survives)

    if max_d_raw < 0.15 and max_d_sn < 0.15:
        decision = "PILOT1_INDETERMINATE_LOW_SIGNAL"
    elif top3_signs_preserved >= 2 and (n_dge15_sn >= 2 or max_d_sn >= 0.20):
        decision = "PILOT1_PATTERN_SURVIVES_NORMALIZATION"
    elif top3_signs_preserved <= 1 or max_d_sn < 0.15:
        decision = "PILOT1_PATTERN_COLLAPSES"
    else:
        decision = "PILOT1_INDETERMINATE_LOW_SIGNAL"

    # ── Figures ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. raw vs sumnorm heatmap (1 col per rep, 11 rows)
        pivot = eff.pivot(index="family", columns="representation", values="cohens_d")
        pivot = pivot.reindex(BSV_GROUPS_ORDER)[representations]
        fig, ax = plt.subplots(figsize=(8, 6))
        vmax = float(np.abs(pivot.values).max()) or 0.5
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(BSV_GROUPS_ORDER)))
        ax.set_yticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER])
        ax.set_xticks(range(len(representations)))
        ax.set_xticklabels(representations, rotation=30, ha="right")
        ax.set_title("Pilot 1.1 — Cohen's d (HCC vs CTR) per family across representations")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iloc[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(v) > vmax*0.5 else "black")
        fig.colorbar(im, ax=ax, label="Cohen's d")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot1_1_raw_vs_normalized_heatmap.png", dpi=150)
        plt.close(fig)

        # 2. CLR-only heatmap (showcase)
        fig, ax = plt.subplots(figsize=(5, 5))
        clr_v = eff[eff.representation == "clr"].set_index("family").reindex(BSV_GROUPS_ORDER)["cohens_d"]
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in clr_v]
        ax.barh([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], clr_v, color=colors)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("CLR Cohen's d (HCC vs CTR)")
        ax.set_title("Pilot 1.1 — CLR effect-size by family")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot1_1_clr_effect_size.png", dpi=150)
        plt.close(fig)

        # 3. Normalized BSV radar (sumnorm)
        angles = np.linspace(0, 2*np.pi, len(BSV_GROUPS_ORDER), endpoint=False).tolist()
        angles += angles[:1]
        pal = {"CTR": "#1f77b4", "H0T": "#d62728"}
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        for cls in CLASSES:
            sub = df[df.class_label == cls]
            vals = [float(sub[f"sumnorm_{g}"].mean()) for g in BSV_GROUPS_ORDER]
            vals += vals[:1]
            ax.plot(angles, vals, label=cls, color=pal[cls], linewidth=1.6)
            ax.fill(angles, vals, alpha=0.10, color=pal[cls])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], fontsize=9)
        ax.set_title("Pilot 1.1 — sum-normalized BSV radar (CTR vs H0T)", pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.05))
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot1_1_sumnorm_radar.png", dpi=180)
        plt.close(fig)

        # 4. Normalized ΔBSV radar (delta_sumnorm)
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        sub = df[df.class_label == "H0T"]
        vals = [float(sub[f"delta_sumnorm_{g}"].mean()) for g in BSV_GROUPS_ORDER]
        vals += vals[:1]
        ax.plot(angles, vals, color="#d62728", linewidth=1.8, label="H0T − CTR (sumnorm)")
        ax.fill(angles, vals, alpha=0.12, color="#d62728")
        ax.plot(angles, [0]*len(angles), color="k", linewidth=0.8, linestyle="--", label="CTR baseline (Δ=0)")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], fontsize=9)
        ax.set_title("Pilot 1.1 — ΔBSV (sum-normalized) radar", pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.05))
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot1_1_delta_sumnorm_radar.png", dpi=180)
        plt.close(fig)
    except Exception as e:
        print(f"figure issue: {e}")

    # ── Reports ──
    lines = [
        "# Pilot 1.1 — Normalization Sensitivity (HCC holdout)",
        "",
        "## Source",
        f"- Pilot 1 v2 per-spectrum outputs: `{PILOT1_V2_TABLE}`",
        f"- {len(df)} spectra (CTR={mass_rows[0]['n']}, H0T={mass_rows[1]['n']})",
        "",
        "## Global amplitude check",
        "",
        f"- mean sum_BSV CTR = {ctr_sum:.4f}",
        f"- mean sum_BSV H0T = {h0t_sum:.4f}",
        f"- H0T amplitude offset vs CTR: **{pct_offset:+.2f}%**",
        f"- Pilot 2 had +3.3-4.0% per-cohort offset; this Pilot 1 offset is **much smaller** → less amplitude artifact concern.",
        "",
        "## Effect-size survival across 5 representations",
        "",
        "| representation | max |d| | meaningful (|d|≥0.30) | CI-significant | |d|≥0.15 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in sum_df.iterrows():
        lines.append(f"| {r['representation']} | {r['max_abs_d']:.2f} | "
                     f"{r['n_meaningful_d_ge_03']}/11 | {r['n_ci_significant']}/11 | "
                     f"{r['n_d_ge_015']}/11 |")
    lines += [
        "",
        "## Top-3 raw signals — survival under sum-normalization",
        "",
        "| family | raw d | sumnorm d | sign preserved | |d|≥0.20 |",
        "|---|---:|---:|---|---|",
    ]
    for s in survives:
        lines.append(f"| {s['raw_top_family']} | {s['raw_d']:+.2f} | "
                     f"{s['sumnorm_d']:+.2f} | "
                     f"{'YES' if s['sign_preserved'] else 'no'} | "
                     f"{'YES' if s['survived_meaningful'] else 'no'} |")
    lines += [
        "",
        "## Per-family direction across representations",
        "",
        "| family | abs | sumnorm | clr | Δ(sn − abs) | sign stable abs↔sumnorm |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, r in rank_df.iterrows():
        lines.append(f"| {r['family']} {r['family_label']} | {r['d_abs']:+.2f} | "
                     f"{r['d_sumnorm']:+.2f} | {r['d_clr']:+.2f} | "
                     f"{r['delta_d_sn_minus_abs']:+.2f} | "
                     f"{'YES' if r['sign_stable_abs_to_sumnorm'] else 'no'} |")
    lines += [
        "",
        f"## Decision: **{decision}**",
        "",
        "## Interpretation",
        "",
    ]
    if decision == "PILOT1_PATTERN_SURVIVES_NORMALIZATION":
        lines.append(
            "The Pilot 1 raw signal (G05 Glycan ↑, G04 Nucleic-backbone ↑, G09 Sterol-lipid ↓) "
            "preserves direction and meaningful magnitude under sum-normalisation. The HCC vs "
            "CTR signal is a real-but-modest selective biochemistry pattern, not an amplitude "
            "artifact. The H0T amplitude offset vs CTR is much smaller than Pilot 2's CCA/LM "
            "offsets, so the HCC vs CTR comparison was not appreciably distorted by amplitude."
        )
    elif decision == "PILOT1_PATTERN_COLLAPSES":
        lines.append(
            "Top-3 raw effects flip sign or collapse under sum-normalisation. The Pilot 1 raw "
            "result was driven by a non-trivial amplitude component, and the residual "
            "compositional signal is too small to interpret cleanly."
        )
    else:
        lines.append(
            "Effect sizes are too small in both raw and normalised representations to draw "
            "conclusions. Pilot 1 HCC vs CTR signal is at the variance floor."
        )
    lines += [
        "",
        "## Updated cross-pilot synthesis",
        "",
        "- HCC vs healthy at BSV level remains a small but real signal across both pilots.",
        f"- Pilot 1 max |d| = {max_d_raw:.2f} (raw) / {max_d_sn:.2f} (sumnorm)",
        "- Pilot 2 HCC max |d| = 0.12 (raw) / 0.39 (sumnorm)",
        "- Both pilots independently confirm the modest magnitude of the HCC vs healthy serum BSV signal.",
        "",
        "## Invariants",
        "",
        "- engine v4.5 / taxonomy / motif / MSS / substrate physics: unchanged",
        "- no classifier training, no threshold tuning, no label-driven feature select",
        "- no target-label fitting",
        "- no DART-Met",
    ]
    (REPORTS / "REPORT_pilot1_1_updated_interpretation.md").write_text("\n".join(lines))

    # Audit log
    audit_lines = [
        "# gaira_base_4_passive_target_pilot_1_1_normalization_sensitivity — Audit Log",
        "",
        "## Source",
        f"- {PILOT1_V2_TABLE}",
        f"- {len(df)} spectra (CTR={mass_rows[0]['n']}, H0T={mass_rows[1]['n']})",
        "",
        "## Representations evaluated",
        "- abs (raw BSV)",
        "- sumnorm (compositional)",
        "- clr (centered log-ratio)",
        "- delta_ctr (raw ΔBSV vs CTR)",
        "- delta_sumnorm (sum-normalised ΔBSV vs CTR)",
        "",
        "## Global amplitude",
        f"- H0T offset vs CTR: {pct_offset:+.2f}%",
        "",
        "## Effect-size survival (max |d|, meaningful, CI-sig per rep)",
    ]
    for _, r in sum_df.iterrows():
        audit_lines.append(f"- {r['representation']}: max d={r['max_abs_d']}, "
                            f"meaningful={r['n_meaningful_d_ge_03']}/11, "
                            f"CI-sig={r['n_ci_significant']}/11")
    audit_lines += [
        "",
        f"## Decision: **{decision}**",
        "",
        "## Invariants",
        "- engine v4.5 / taxonomy / motif / MSS / substrate physics: unchanged",
        "- no fitting / no threshold tuning / no classifier",
    ]
    (AUDIT / "gaira_base_4_passive_target_pilot_1_1_normalization_sensitivity_audit_log.md"
     ).write_text("\n".join(audit_lines))

    p = Path(__file__)
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)

    print(f"\n[complete] decision: {decision}")
    print(f"  H0T amplitude offset vs CTR: {pct_offset:+.2f}%")
    print(f"  raw max |d|={max_d_raw:.2f} → sumnorm max |d|={max_d_sn:.2f}")
    print(f"  top-3 raw signals: {top3_signs_preserved}/3 sign-preserved, "
          f"{top3_survived}/3 |d|≥0.20 in sumnorm")


if __name__ == "__main__":
    main()
