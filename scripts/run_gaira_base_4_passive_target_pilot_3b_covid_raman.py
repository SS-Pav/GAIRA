"""gaira_base_4 passive target Pilot 3B — COVID serum Raman full readout.

NO classifier. NO threshold tuning. NO target-label fitting. NO engine change.

Dataset: COVID serum Raman (Pilot 3A audited).
Regime: Raman → substrate-aware physics OFF.
Cohorts: Healthy (ref) / COVID / Suspected / Tube (negative control).
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

from gaira.spectral import canonical_master_axis
from run_gaira_base_4_hybrid_bsv_build_v1 import (
    BSV_GROUPS, compute_motif_firings, compute_mss_scores_v43,
)
from run_gaira_base_4_hybrid_bsv_refinement_v4_5_triglyceride_veto import (
    compute_hybrid_bsv_v45,
)
from run_gaira_base_4_hybrid_bsv_controlled_calibration_v2 import FAMILY_LABELS


ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_passive_target_pilot_3b_covid_raman"
)
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"

DATA = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/covid_serum_raman")
MSS_V43 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_decision_enrichment_v1/"
    "registry/grounding_molecular_signatures_v4_3.csv"
)
LEARNED_MOTIFS = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_3_grounding_trained_ontology_v1/"
    "registry/learned_motif_registry_v1.csv"
)

BSV_GROUPS_ORDER = [g["group_id"] for g in BSV_GROUPS]
ABS_COLS = [f"abs_{g}" for g in BSV_GROUPS_ORDER]
COHORT_FILES = {
    "Healthy":   "raw_Helthy.txt",
    "Suspected": "raw_Suspected.txt",
    "COVID":     "raw_COVID.txt",
    "Tube":      "raw_Tube.txt",
}
COHORT_ORDER = ["Healthy", "Suspected", "COVID", "Tube"]


def _cohens_d(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2: return 0.0
    pooled = np.sqrt(((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1))
                       / max(len(x)+len(y)-2, 1))
    return (np.mean(x) - np.mean(y)) / (pooled if pooled > 0 else 1.0)


# ─────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────

def load_covid_raman(master_x):
    print("\n[load] COVID serum Raman dataset")
    wn_full = np.loadtxt(DATA / "wave_number.txt")
    refs = []
    for cls, fname in COHORT_FILES.items():
        arr = np.loadtxt(DATA / fname)  # 900 rows × N cols
        # Drop boundary zero rows (per Pilot 3A)
        keep = np.arange(1, arr.shape[0] - 1)  # rows 1..898
        arr = arr[keep]
        wn = wn_full[keep]
        order = np.argsort(wn)
        for i in range(arr.shape[1]):
            y = arr[:, i]
            y_rs = np.interp(master_x, wn[order], y[order],
                               left=np.nan, right=np.nan)
            refs.append({
                "spectrum_id": f"covid::{cls}_{i+1:03d}",
                "sample_id": f"{cls}_{(i // 3) + 1}",  # 3-experimenter design
                "experimenter_id": (i % 3) + 1,
                "class_label": cls,
                "regime": "Raman",
                "substrate_family": "none (Raman)",
                "spectrum": y_rs,
                "preprocessing_tag": "source_baselined + gaira_canonical_resample (no re-baseline)",
            })
        print(f"  {cls}: {arr.shape[1]} spectra")
    return refs


# ─────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────

def run_pipeline(refs, master_x, motif_df, mss_df, motif_id_to_group,
                   motif_ids, analyte_to_group):
    print("\n[pipeline] Raman, substrate physics OFF")
    rows = []
    for r in refs:
        mf = compute_motif_firings(r["spectrum"], master_x, motif_df)
        order = np.argsort(-mf)
        top_motif_families = []
        for idx in order[:5]:
            g = motif_id_to_group.get(motif_ids[idx], None)
            if g and g not in top_motif_families:
                top_motif_families.append(g)
            if len(top_motif_families) >= 3: break

        ms = compute_mss_scores_v43(r["spectrum"], master_x, mss_df)
        top_mss = sorted(ms.items(), key=lambda kv: -kv[1])[:5]

        bsv = compute_hybrid_bsv_v45(
            r["spectrum"], master_x, mf, ms, motif_id_to_group, motif_ids,
            analyte_to_group, regime="Raman",
            apply_sers_physics=False, apply_tg_veto=True,
        )
        per_group = bsv["per_group"]
        bsv_vec = {g: round(per_group.get(g, {}).get("magnitude", 0.0), 4)
                    for g in BSV_GROUPS_ORDER}
        conf_vec = {g: round(per_group.get(g, {}).get("confidence", 0.0), 4)
                     for g in BSV_GROUPS_ORDER}
        sorted_g = sorted(per_group.items(), key=lambda kv: -kv[1]["magnitude"])
        top3 = [g for g, _ in sorted_g[:3]]

        row = {
            "spectrum_id": r["spectrum_id"], "sample_id": r["sample_id"],
            "experimenter_id": r["experimenter_id"],
            "class_label": r["class_label"], "regime": r["regime"],
            "substrate_family": r["substrate_family"],
            "preprocessing_tag": r["preprocessing_tag"],
            "substrate_block": "n/a (Raman)",
            "apply_sers_physics": False,
            "top_motif_family": top_motif_families[0] if top_motif_families else None,
            "top_3_motif_families": ";".join(top_motif_families[:3]),
            "top_mss_hits": ";".join(n for n, _ in top_mss),
            "top_mss_scores": ";".join(str(round(s, 3)) for _, s in top_mss),
            "top_bsv_family": bsv["top_group"],
            "top_3_bsv_families": ";".join(top3),
            "bsv_vector_11axis": ";".join(f"{g}:{v}" for g, v in bsv_vec.items()),
            "confidence_vector_11axis": ";".join(f"{g}:{v}" for g, v in conf_vec.items()),
            "ambiguity_flag": bsv["ambiguity_flag"],
            "spillover_ratio": round(bsv["spillover_ratio"], 4),
            "top_confidence": round(per_group.get(bsv["top_group"], {}).get("confidence", 0.0), 4),
            "nearest_competing_family": sorted_g[1][0] if len(sorted_g) > 1 else None,
            "interpretation_tier": "RAMAN_NO_SUBSTRATE_CAVEAT",
        }
        row.update({f"abs_{g}": bsv_vec[g] for g in BSV_GROUPS_ORDER})
        row.update({f"conf_{g}": conf_vec[g] for g in BSV_GROUPS_ORDER})
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Representations
# ─────────────────────────────────────────────────────────────────────

def add_representations(df):
    X = df[ABS_COLS].values
    df["bsv_sum"] = X.sum(axis=1)
    df["bsv_l2"]  = np.sqrt((X**2).sum(axis=1))
    p = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    df["bsv_entropy"] = -np.nansum(p * np.log(p + 1e-12), axis=1)
    # sumnorm
    X_sum = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    for i, g in enumerate(BSV_GROUPS_ORDER): df[f"sumnorm_{g}"] = X_sum[:, i]
    # CLR
    X_pos = np.maximum(X, 1e-9); log_X = np.log(X_pos)
    X_clr = log_X - log_X.mean(axis=1, keepdims=True)
    for i, g in enumerate(BSV_GROUPS_ORDER): df[f"clr_{g}"] = X_clr[:, i]
    # Δ vs Healthy
    h = df[df.class_label == "Healthy"]
    h_means_abs = h[ABS_COLS].mean()
    h_means_sn  = h[[f"sumnorm_{g}" for g in BSV_GROUPS_ORDER]].mean()
    for g in BSV_GROUPS_ORDER:
        df[f"delta_raw_{g}"]     = df[f"abs_{g}"]     - h_means_abs[f"abs_{g}"]
        df[f"delta_sumnorm_{g}"] = df[f"sumnorm_{g}"] - h_means_sn[f"sumnorm_{g}"]
    return df


# ─────────────────────────────────────────────────────────────────────
# Effect sizes + trajectory
# ─────────────────────────────────────────────────────────────────────

def effect_sizes(df, comparisons, representations, n_boot=1000):
    print("\n[effects] effect sizes + bootstrap CIs")
    rng = np.random.default_rng(42)
    rows = []
    for rep in representations:
        for a, b in comparisons:
            for g in BSV_GROUPS_ORDER:
                col = f"{rep}_{g}"
                if col not in df.columns: continue
                x = df[df.class_label == a][col].values
                y = df[df.class_label == b][col].values
                d_pt = _cohens_d(x, y)
                ds = []
                for _ in range(n_boot):
                    xs = rng.choice(x, size=len(x), replace=True)
                    ys = rng.choice(y, size=len(y), replace=True)
                    ds.append(_cohens_d(xs, ys))
                ds = np.asarray(ds)
                ci_lo, ci_hi = float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
                rows.append({
                    "representation": rep, "comparison": f"{a}_vs_{b}",
                    "family": g, "family_label": FAMILY_LABELS.get(g, g),
                    "cohens_d": round(float(d_pt), 3),
                    "abs_d": round(abs(float(d_pt)), 3),
                    "ci95_low": round(ci_lo, 3), "ci95_high": round(ci_hi, 3),
                    "ci_excludes_zero": (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0),
                    "direction": "↑" if d_pt > 0 else ("↓" if d_pt < 0 else "0"),
                })
    return pd.DataFrame(rows)


def trajectory_scores(df, rep="sumnorm"):
    """Per-family Healthy → Suspected → COVID monotonicity."""
    rows = []
    for g in BSV_GROUPS_ORDER:
        col = f"{rep}_{g}"
        if col not in df.columns: continue
        h = df[df.class_label == "Healthy"][col].mean()
        s = df[df.class_label == "Suspected"][col].mean()
        c = df[df.class_label == "COVID"][col].mean()
        # monotonic increase
        if (h <= s <= c) and not (h == s == c): score = +1
        elif (h >= s >= c) and not (h == s == c): score = -1
        else: score = 0
        rows.append({
            "family": g, "family_label": FAMILY_LABELS.get(g, g),
            "Healthy_mean": round(float(h), 4),
            "Suspected_mean": round(float(s), 4),
            "COVID_mean": round(float(c), 4),
            "delta_S_minus_H": round(float(s - h), 4),
            "delta_C_minus_H": round(float(c - h), 4),
            "delta_C_minus_S": round(float(c - s), 4),
            "monotonic_score": score,
            "monotonic_label": "monotonic_increase" if score == +1
                else ("monotonic_decrease" if score == -1 else "non-monotonic"),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_4_passive_target_pilot_3b_covid_raman")
    print("=" * 78)
    for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    master_x = canonical_master_axis()
    mss_df = pd.read_csv(MSS_V43)
    motif_df = pd.read_csv(LEARNED_MOTIFS)
    motif_ids = motif_df["learned_motif_id"].tolist()
    motif_id_to_group = {}
    for g in BSV_GROUPS:
        for m_id in g["dominant_motifs"]:
            motif_id_to_group[m_id] = g["group_id"]
    bc_to_group = {bc: g["group_id"] for g in BSV_GROUPS
                    for bc in g["member_broad_classes"]}
    analyte_to_group = {}
    for _, r in mss_df.iterrows():
        analyte_to_group[r["analyte_name"]] = bc_to_group.get(r["broad_class"], "G11")

    # 1. Load
    refs = load_covid_raman(master_x)
    inv = pd.DataFrame([{
        "cohort": cls,
        "n_spectra": sum(1 for r in refs if r["class_label"] == cls),
    } for cls in COHORT_ORDER])
    inv.to_csv(TABLES / "pilot3b_cohort_inventory.csv", index=False)

    # 2. Pipeline
    df = run_pipeline(refs, master_x, motif_df, mss_df, motif_id_to_group,
                        motif_ids, analyte_to_group)

    # 3. Representations + Δ refs
    df = add_representations(df)
    df.to_csv(TABLES / "pilot3b_per_spectrum_outputs.csv", index=False)

    # 4. Amplitude audit
    print("\n[amplitude audit]")
    h_sum = float(df[df.class_label == "Healthy"]["bsv_sum"].mean())
    amp_rows = []
    for cls in COHORT_ORDER:
        sub = df[df.class_label == cls]
        ms = float(sub["bsv_sum"].mean())
        amp_rows.append({
            "cohort": cls, "n": len(sub),
            "mean_sum_BSV": round(ms, 4),
            "std_sum_BSV": round(float(sub["bsv_sum"].std(ddof=1)), 4),
            "mean_L2": round(float(sub["bsv_l2"].mean()), 4),
            "mean_entropy": round(float(sub["bsv_entropy"].mean()), 4),
            "delta_sum_vs_Healthy": round(ms - h_sum, 4),
            "pct_offset_vs_Healthy": round((ms - h_sum) / h_sum * 100, 2),
            "cohens_d_sum_vs_Healthy": round(_cohens_d(
                sub["bsv_sum"].values,
                df[df.class_label == "Healthy"]["bsv_sum"].values), 3),
        })
        print(f"  {cls:10s}  sum={ms:.4f}  pct_offset={amp_rows[-1]['pct_offset_vs_Healthy']:+.1f}%  "
              f"d_sum={amp_rows[-1]['cohens_d_sum_vs_Healthy']:+.2f}")
    pd.DataFrame(amp_rows).to_csv(TABLES / "pilot3b_amplitude_audit.csv", index=False)

    # 5+6. Comparisons + effect sizes
    primary = [("COVID", "Healthy"), ("Suspected", "Healthy")]
    trajectory = [("COVID", "Suspected")]
    qc = [("Tube", "Healthy"), ("Tube", "COVID")]
    all_pairs = primary + trajectory + qc
    eff = effect_sizes(df, all_pairs,
                          ["abs", "sumnorm", "clr", "delta_raw", "delta_sumnorm"])
    eff.to_csv(TABLES / "pilot3b_effect_sizes_all.csv", index=False)

    # 7. Trajectory analysis
    traj = trajectory_scores(df, rep="sumnorm")
    traj.to_csv(TABLES / "pilot3b_trajectory_scores.csv", index=False)
    n_mono_inc = int((traj["monotonic_score"] == +1).sum())
    n_mono_dec = int((traj["monotonic_score"] == -1).sum())
    print(f"\n[trajectory] Healthy→Suspected→COVID monotonic: "
          f"{n_mono_inc} ↑, {n_mono_dec} ↓, {11 - n_mono_inc - n_mono_dec} non-monotonic")

    # 8. QC validation (Tube)
    tube_eff = eff[(eff.representation == "sumnorm") & (eff.comparison == "Tube_vs_Healthy")]
    tube_n_ci = int(tube_eff["ci_excludes_zero"].sum())
    tube_max_d = float(tube_eff["abs_d"].max())
    tube_mean_abs_delta = float(np.mean([
        abs(df[df.class_label == "Tube"][f"delta_sumnorm_{g}"].mean()) for g in BSV_GROUPS_ORDER
    ]))
    qc_rows = [{
        "metric": "tube_mean_abs_delta_sumnorm_vs_Healthy", "value": round(tube_mean_abs_delta, 4)},
        {"metric": "tube_n_families_CI_significant_vs_Healthy", "value": tube_n_ci},
        {"metric": "tube_max_abs_d_vs_Healthy", "value": round(tube_max_d, 3)},
        {"metric": "tube_n_spectra", "value": int((df.class_label == "Tube").sum())},
    ]
    pd.DataFrame(qc_rows).to_csv(TABLES / "pilot3b_qc_validation.csv", index=False)

    # 9. Figures
    print("\n[figures]")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pal = {"Healthy": "#1f77b4", "Suspected": "#ff7f0e",
                "COVID": "#d62728", "Tube": "#7f7f7f"}

        # 1. effect-size heatmap (sumnorm) — 5 comparisons
        comps = ["COVID_vs_Healthy", "Suspected_vs_Healthy", "COVID_vs_Suspected",
                  "Tube_vs_Healthy", "Tube_vs_COVID"]
        sn = eff[(eff.representation == "sumnorm") & (eff.comparison.isin(comps))]
        pivot = sn.pivot(index="family", columns="comparison", values="cohens_d").reindex(BSV_GROUPS_ORDER)[comps]
        ci_pivot = sn.pivot(index="family", columns="comparison",
                              values="ci_excludes_zero").reindex(BSV_GROUPS_ORDER)[comps]
        fig, ax = plt.subplots(figsize=(11, 6))
        vmax = float(np.abs(pivot.values).max()) or 0.5
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(BSV_GROUPS_ORDER)))
        ax.set_yticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER])
        ax.set_xticks(range(len(comps)))
        ax.set_xticklabels(comps, rotation=20, ha="right", fontsize=9)
        ax.set_title("Pilot 3B — sum-normalized Cohen's d (* = CI excludes 0)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iloc[i, j]
                star = "*" if bool(ci_pivot.iloc[i, j]) else ""
                ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=8,
                         color="white" if abs(v) > vmax*0.55 else "black")
        fig.colorbar(im, ax=ax, label="Cohen's d (sumnorm)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot3b_sumnorm_effect_heatmap.png", dpi=150)
        plt.close(fig)

        # 2. CLR effect-size heatmap
        cl = eff[(eff.representation == "clr") & (eff.comparison.isin(comps))]
        pivot_cl = cl.pivot(index="family", columns="comparison", values="cohens_d").reindex(BSV_GROUPS_ORDER)[comps]
        fig, ax = plt.subplots(figsize=(11, 6))
        vmax = float(np.abs(pivot_cl.values).max()) or 0.5
        im = ax.imshow(pivot_cl.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(BSV_GROUPS_ORDER)))
        ax.set_yticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER])
        ax.set_xticks(range(len(comps)))
        ax.set_xticklabels(comps, rotation=20, ha="right", fontsize=9)
        ax.set_title("Pilot 3B — CLR Cohen's d")
        for i in range(pivot_cl.shape[0]):
            for j in range(pivot_cl.shape[1]):
                v = pivot_cl.iloc[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(v) > vmax*0.55 else "black")
        fig.colorbar(im, ax=ax, label="Cohen's d (CLR)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot3b_clr_effect_heatmap.png", dpi=150)
        plt.close(fig)

        # 3. Trajectory plot — Healthy → Suspected → COVID per family
        fig, ax = plt.subplots(figsize=(11, 5))
        x_labels = ["Healthy", "Suspected", "COVID"]
        x = np.arange(3)
        cmap = plt.cm.tab20(np.linspace(0, 1, 11))
        for i, g in enumerate(BSV_GROUPS_ORDER):
            row = traj[traj.family == g].iloc[0]
            ys = [row["Healthy_mean"], row["Suspected_mean"], row["COVID_mean"]]
            mark = "o" if row["monotonic_score"] != 0 else "x"
            ls  = "-" if row["monotonic_score"] != 0 else "--"
            ax.plot(x, ys, marker=mark, linestyle=ls, color=cmap[i], linewidth=1.4,
                     label=f"{g} {FAMILY_LABELS.get(g, g)}")
        ax.set_xticks(x); ax.set_xticklabels(x_labels)
        ax.set_ylabel("sumnorm mean")
        ax.set_title("Pilot 3B — family trajectories Healthy → Suspected → COVID "
                       "(o = monotonic, x = non-monotonic)")
        ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot3b_trajectory_plot.png", dpi=150)
        plt.close(fig)

        # 4a. Sumnorm BSV radar per cohort
        angles = np.linspace(0, 2*np.pi, len(BSV_GROUPS_ORDER), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        for cls in COHORT_ORDER:
            sub = df[df.class_label == cls]
            vals = [float(sub[f"sumnorm_{g}"].mean()) for g in BSV_GROUPS_ORDER]
            vals += vals[:1]
            ax.plot(angles, vals, label=f"{cls} (n={len(sub)})", color=pal[cls], linewidth=1.6)
            ax.fill(angles, vals, alpha=0.07, color=pal[cls])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], fontsize=9)
        ax.set_title("Pilot 3B — sum-normalized BSV radar per cohort", pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot3b_sumnorm_radar.png", dpi=180)
        plt.close(fig)

        # 4b. ΔBSV vs Healthy radar
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        for cls in ["Suspected", "COVID", "Tube"]:
            sub = df[df.class_label == cls]
            vals = [float(sub[f"delta_sumnorm_{g}"].mean()) for g in BSV_GROUPS_ORDER]
            vals += vals[:1]
            ax.plot(angles, vals, label=f"{cls} − Healthy", color=pal[cls], linewidth=1.6)
            ax.fill(angles, vals, alpha=0.10, color=pal[cls])
        ax.plot(angles, [0]*len(angles), color="k", linewidth=0.8, linestyle="--",
                 label="Healthy baseline (Δ=0)")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], fontsize=9)
        ax.set_title("Pilot 3B — ΔBSV (sumnorm) radar vs Healthy", pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot3b_delta_sumnorm_radar.png", dpi=180)
        plt.close(fig)

        # 5. Amplitude distributions per cohort
        fig, ax = plt.subplots(figsize=(10, 4))
        for cls in COHORT_ORDER:
            sub = df[df.class_label == cls]
            ax.hist(sub["bsv_sum"], bins=20, alpha=0.6, label=f"{cls} (n={len(sub)})",
                     color=pal[cls])
        ax.set_xlabel("sum_BSV (per spectrum)"); ax.set_ylabel("count")
        ax.set_title("Pilot 3B — sum_BSV distribution per cohort")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot3b_amplitude_distributions.png", dpi=150)
        plt.close(fig)

        # 6. PCA on sumnorm BSV
        from sklearn.decomposition import PCA
        X = df[[f"sumnorm_{g}" for g in BSV_GROUPS_ORDER]].values
        pc = PCA(n_components=2, random_state=0).fit_transform(X)
        fig, ax = plt.subplots(figsize=(8, 6))
        for cls in COHORT_ORDER:
            m = df["class_label"].values == cls
            ax.scatter(pc[m, 0], pc[m, 1], s=35, alpha=0.7,
                         label=f"{cls} (n={m.sum()})", color=pal[cls])
        ax.set_xlabel("PC1 of sumnorm BSV"); ax.set_ylabel("PC2 of sumnorm BSV")
        ax.set_title("Pilot 3B — unsupervised PCA of sumnorm BSV (colored post-hoc)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot3b_pca_sumnorm.png", dpi=150)
        plt.close(fig)

        # 7. Tube QC plot — ΔBSV per family
        fig, ax = plt.subplots(figsize=(11, 4))
        sub = df[df.class_label == "Tube"]
        ds = [sub[f"delta_sumnorm_{g}"].mean() for g in BSV_GROUPS_ORDER]
        sems = [sub[f"delta_sumnorm_{g}"].sem() for g in BSV_GROUPS_ORDER]
        x = np.arange(len(BSV_GROUPS_ORDER))
        colors = ["#2ca02c" if abs(d) < 0.05 else "#d62728" for d in ds]
        ax.bar(x, ds, yerr=sems, capsize=2, color=colors)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER],
                                                  rotation=45, ha="right")
        ax.set_ylabel("Δ sumnorm (Tube − Healthy)")
        ax.set_title("Pilot 3B — Tube QC: ΔBSV vs Healthy "
                       "(green = |Δ|<0.05 expected near-zero)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot3b_tube_qc.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  figure issue: {e}")

    # 10. Per-comparison summary
    print("\n[summary]")
    rep_summary = []
    for rep in ["abs", "sumnorm", "clr", "delta_raw", "delta_sumnorm"]:
        for comp_a, comp_b in primary + trajectory:
            sub = eff[(eff.representation == rep) & (eff.comparison == f"{comp_a}_vs_{comp_b}")]
            rep_summary.append({
                "representation": rep,
                "comparison": f"{comp_a}_vs_{comp_b}",
                "max_abs_d": round(float(sub["abs_d"].max()), 3),
                "n_meaningful_d_ge_03": int((sub["abs_d"] >= 0.30).sum()),
                "n_ci_significant": int(sub["ci_excludes_zero"].sum()),
            })
    sum_df = pd.DataFrame(rep_summary)
    sum_df.to_csv(TABLES / "pilot3b_per_comparison_summary.csv", index=False)
    print(sum_df.to_string(index=False))

    # 11. Decision logic
    sn_cv = sum_df[(sum_df.representation == "sumnorm") &
                     (sum_df.comparison == "COVID_vs_Healthy")].iloc[0]
    sn_sus = sum_df[(sum_df.representation == "sumnorm") &
                      (sum_df.comparison == "Suspected_vs_Healthy")].iloc[0]
    n_mono = (traj["monotonic_score"].abs() == 1).sum()
    tube_qc_pass = (tube_n_ci <= 2 and tube_max_d < 0.5)

    if (int(sn_cv["n_ci_significant"]) >= 4 and float(sn_cv["max_abs_d"]) >= 0.5
            and tube_qc_pass and n_mono >= 4):
        decision = "READY_STRONG_SIGNAL"
    elif (int(sn_cv["n_ci_significant"]) >= 2 and float(sn_cv["max_abs_d"]) >= 0.30
            and tube_qc_pass):
        decision = "READY_MODERATE_SIGNAL"
    elif tube_qc_pass:
        decision = "MIXED_SIGNAL_WITH_CAVEATS"
    else:
        decision = "QC_FAILURE"

    print(f"\n[decision] {decision}")
    print(f"  COVID_vs_Healthy sumnorm: max |d|={sn_cv['max_abs_d']}, "
          f"n_meaningful={sn_cv['n_meaningful_d_ge_03']}, CI-sig={sn_cv['n_ci_significant']}")
    print(f"  Trajectory monotonic families: {n_mono}/11")
    print(f"  Tube QC: max |d|={tube_max_d:.2f}, n_CI_sig={tube_n_ci}, pass={tube_qc_pass}")

    # ── Reports ──
    # Cohort comparison report
    lines = [
        "# Pilot 3B — Cohort Comparison Report",
        "",
        f"## Final decision: **{decision}**",
        "",
        "## Per-comparison summary (5 representations)",
        "",
        "| representation | comparison | max |d| | meaningful (|d|≥0.30) | CI-significant |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in sum_df.iterrows():
        lines.append(f"| {r['representation']} | {r['comparison']} | "
                     f"{r['max_abs_d']:.2f} | {r['n_meaningful_d_ge_03']}/11 | "
                     f"{r['n_ci_significant']}/11 |")
    lines += [
        "",
        "## COVID vs Healthy — top-3 sumnorm shifts (CI-significant only)",
        "",
    ]
    cv_eff = eff[(eff.representation == "sumnorm") &
                  (eff.comparison == "COVID_vs_Healthy") &
                  (eff.ci_excludes_zero == True)].sort_values("abs_d", ascending=False)
    if len(cv_eff):
        for _, r in cv_eff.head(5).iterrows():
            lines.append(f"- {r['family']} {r['family_label']}: d = {r['cohens_d']:+.3f} "
                         f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}] {r['direction']}")
    else:
        lines.append("- (no CI-significant axes)")
    lines += [
        "",
        "## Suspected vs Healthy — top-3 sumnorm shifts (CI-significant only)",
        "",
    ]
    sus_eff = eff[(eff.representation == "sumnorm") &
                    (eff.comparison == "Suspected_vs_Healthy") &
                    (eff.ci_excludes_zero == True)].sort_values("abs_d", ascending=False)
    if len(sus_eff):
        for _, r in sus_eff.head(5).iterrows():
            lines.append(f"- {r['family']} {r['family_label']}: d = {r['cohens_d']:+.3f} "
                         f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}] {r['direction']}")
    else:
        lines.append("- (no CI-significant axes)")
    lines += [
        "",
        "## Cohort amplitude audit",
        "",
        "| cohort | n | mean sum_BSV | std | % offset vs Healthy | d_sum vs Healthy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in amp_rows:
        lines.append(f"| {r['cohort']} | {r['n']} | {r['mean_sum_BSV']:.4f} | "
                     f"{r['std_sum_BSV']:.4f} | {r['pct_offset_vs_Healthy']:+.1f}% | "
                     f"{r['cohens_d_sum_vs_Healthy']:+.2f} |")
    (REPORTS / "REPORT_pilot3b_cohort_comparison.md").write_text("\n".join(lines))

    # Trajectory interpretation report
    lines = [
        "# Pilot 3B — Trajectory Interpretation",
        "",
        "## Per-family Healthy → Suspected → COVID monotonicity (sumnorm means)",
        "",
        "| family | Healthy | Suspected | COVID | Δ(C−H) | monotonic |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, r in traj.iterrows():
        mono_lbl = "↑" if r["monotonic_score"] == 1 else ("↓" if r["monotonic_score"] == -1 else "—")
        lines.append(f"| {r['family']} {r['family_label']} | "
                     f"{r['Healthy_mean']:.4f} | {r['Suspected_mean']:.4f} | "
                     f"{r['COVID_mean']:.4f} | {r['delta_C_minus_H']:+.4f} | {mono_lbl} |")
    lines += [
        "",
        f"## Summary",
        f"- monotonic increase: {n_mono_inc}/11 families",
        f"- monotonic decrease: {n_mono_dec}/11 families",
        f"- non-monotonic: {11 - n_mono_inc - n_mono_dec}/11",
        "",
        "Suspected lying between Healthy and COVID is the expected severity-progression pattern. "
        "Family-level monotonicity (either direction) supports a coherent disease-trajectory readout.",
    ]
    (REPORTS / "REPORT_pilot3b_trajectory_interpretation.md").write_text("\n".join(lines))

    # Normalization sensitivity summary
    lines = [
        "# Pilot 3B — Normalization Sensitivity Summary",
        "",
        "## Effect-size survival across representations (COVID vs Healthy)",
        "",
        "| representation | max |d| | meaningful (|d|≥0.30) | CI-significant |",
        "|---|---:|---:|---:|",
    ]
    for _, r in sum_df[sum_df.comparison == "COVID_vs_Healthy"].iterrows():
        lines.append(f"| {r['representation']} | {r['max_abs_d']:.2f} | "
                     f"{r['n_meaningful_d_ge_03']}/11 | {r['n_ci_significant']}/11 |")
    raw_max = float(sum_df[(sum_df.comparison == "COVID_vs_Healthy") &
                              (sum_df.representation == "abs")]["max_abs_d"].iloc[0])
    sn_max = float(sn_cv["max_abs_d"])
    if abs(raw_max - sn_max) < 0.05:
        norm_verdict = "Normalization is OPTIONAL — raw vs sumnorm produce nearly identical signals (no per-cohort amplitude offset)."
    elif sn_max > raw_max + 0.10:
        norm_verdict = "Normalization is HELPFUL — sumnorm AMPLIFIES signal vs raw (within-cohort amplitude noise was masking the chemistry)."
    elif sn_max < raw_max - 0.10:
        norm_verdict = "Normalization is CORRECTIVE — raw exaggerates the signal due to per-cohort amplitude offset; sumnorm is the chemistry layer."
    else:
        norm_verdict = "Normalization adds modest interpretability."
    lines += ["", f"## Verdict", "", norm_verdict]
    (REPORTS / "REPORT_pilot3b_normalization_sensitivity.md").write_text("\n".join(lines))

    # QC validation report
    lines = [
        "# Pilot 3B — Tube QC Validation Report",
        "",
        f"- Tube vs Healthy max |d| (sumnorm): **{tube_max_d:.3f}**",
        f"- Tube vs Healthy CI-significant axes: **{tube_n_ci}/11**",
        f"- Tube mean |Δ_sumnorm|: **{tube_mean_abs_delta:.4f}**",
        f"- Tube n: {int((df.class_label == 'Tube').sum())}",
        "",
        "## Verdict",
        "",
    ]
    if tube_qc_pass:
        lines.append("**PASS** — Tube cohort behaves as expected (near-zero structured signal). "
                       "GAIRA is not generating spurious biology from instrument/buffer background.")
    else:
        lines.append("**FAIL** — Tube cohort shows structured multi-axis signal. "
                       "GAIRA may be reading instrument/buffer features as biology. Investigate.")
    (REPORTS / "REPORT_pilot3b_qc_validation.md").write_text("\n".join(lines))

    # Audit log
    lines = [
        "# gaira_base_4_passive_target_pilot_3b_covid_raman — Audit Log",
        "",
        f"## Dataset",
        f"- {DATA}",
        f"- Cohorts: " + ", ".join(f"{c}={int(inv[inv.cohort==c].n_spectra.iloc[0])}" for c in COHORT_ORDER),
        "",
        "## Pipeline",
        "- engine v4.5 + v3 fixes; UNCHANGED",
        "- regime: Raman; substrate physics OFF for inference and interpretation",
        "- 11-axis BSV + sumnorm + CLR + delta_raw + delta_sumnorm per spectrum",
        "- bootstrap 95% CIs (1000 resamples) per family per comparison",
        "",
        "## Comparisons",
        "- PRIMARY: COVID vs Healthy, Suspected vs Healthy",
        "- TRAJECTORY: COVID vs Suspected",
        "- QC: Tube vs Healthy, Tube vs COVID",
        "",
        "## Results",
        f"- COVID vs Healthy sumnorm: max |d| = {sn_cv['max_abs_d']}, "
        f"meaningful = {sn_cv['n_meaningful_d_ge_03']}/11, CI-sig = {sn_cv['n_ci_significant']}/11",
        f"- Suspected vs Healthy sumnorm: max |d| = {sn_sus['max_abs_d']}, "
        f"meaningful = {sn_sus['n_meaningful_d_ge_03']}/11, CI-sig = {sn_sus['n_ci_significant']}/11",
        f"- Trajectory monotonic families: {n_mono}/11",
        f"- Tube QC: max |d| = {tube_max_d:.2f}, CI-sig = {tube_n_ci}/11, pass = {tube_qc_pass}",
        "",
        f"## Final decision: **{decision}**",
        "",
        "## Invariants",
        "- engine v4.5 / taxonomy / motif / MSS v4.3 / substrate physics v1.2: unchanged",
        "- no classifier training, no threshold tuning, no label-driven feature select",
        "- no target-label fitting; no DART-Met",
    ]
    (AUDIT / "gaira_base_4_passive_target_pilot_3b_covid_raman_audit_log.md"
     ).write_text("\n".join(lines))

    p = Path(__file__)
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)


if __name__ == "__main__":
    main()
