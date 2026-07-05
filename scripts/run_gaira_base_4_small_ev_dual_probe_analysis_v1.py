"""gaira_base_4_small_ev_dual_probe_analysis_v1

Phase: FULL GAIRA analysis on small2023_ev dual-probe dataset — cross-probe
reproducibility test at RAW + 11-axis BSV + MSS layers.

Strict invariants:
- Engine v4.5 / MSS kernel / motif / BSV / preprocessing UNCHANGED
- Probe 1 and Probe 2 kept SEPARATE throughout
- Canonical preprocessing: interp-to-master + AsLS + Savitzky-Golay + L2
- Overlap region: 670-1800 cm⁻¹ (Probe 1 native; Probe 2 sliced to this range)
- No substrate-physics wrapper (probe physics still unknown)
- No classifier
- No disease labels (dataset has cell-line mixture cohorts, no disease labels)

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_4_small_ev_dual_probe_analysis_v1.py
"""
from __future__ import annotations

import shutil
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis  # noqa: E402

from run_gaira_base_4_mss_resolution_reporting_layer_v1 import (  # noqa: E402
    baseline_correct, mss_anchor_score, load_templates,
)


# ──────────────────────────────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_small_ev_dual_probe_analysis_v1")
TABLES  = ROOT / "tables"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT   = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/small2023_ev")
P1_MAT = DATA_DIR / "NormedProbe1.mat"
P2_MAT = DATA_DIR / "NormedProbe2.mat"
FIG_S7 = DATA_DIR / "Fig_S7 (1).xlsx"

COHORTS = ["c00", "c01", "c10", "c25", "c50", "c100"]
# Mixture ratio on x-axis: HT-1080 fraction (%) inferred from Fig5 Labels.csv
#   c00 = thp100 (0% HT)    c01 = thp50ht1 (1%)    c10 = thp50ht10 (10%)
#   c25 = thp50ht25 (25%)   c50 = thp50ht50 (50%)  c100 = ht100 (100%)
COHORT_HT_FRAC = {"c00": 0.0, "c01": 1.0, "c10": 10.0, "c25": 25.0, "c50": 50.0, "c100": 100.0}

# Cap spectra per cohort to keep runtime tractable (sampling is random_state-fixed)
N_MAX_PER_COHORT = 2000

# 11 BSV families (BIOLOGY_AXES_V11)
BSV_FAMILIES = (
    ("G01", "purine_nucleotide"),
    ("G02", "purine_metabolite"),
    ("G03", "pyrimidine_nucleotide"),
    ("G04", "phosphate_nucleic_adjacent"),
    ("G05", "glycan_carbohydrate"),
    ("G06", "protein_peptide_backbone"),
    ("G07", "aromatic_residue"),
    ("G08", "lipid_acyl_membrane"),
    ("G09", "sterol_neutral_lipid"),
    ("G10", "sulfur_thiol_redox"),
    ("G11", "metabolic_small_molecule"),
)


# ──────────────────────────────────────────────────────────────────────
# STEP 0 — data prep
# ──────────────────────────────────────────────────────────────────────
def load_probe_spectra():
    print("[STEP 0] loading probe spectra + Calx from Fig_S7")
    # Calx from Fig_S7 — both probes land on integer cm⁻¹ grid, step 1
    wn1 = np.arange(670, 1801)   # Probe 1: 670..1800 (1131 pts)
    wn2 = np.arange(401, 1801)   # Probe 2: 401..1800 (1400 pts)
    assert len(wn1) == 1131 and len(wn2) == 1400, "Calx-length mismatch"

    p1 = sio.loadmat(P1_MAT, squeeze_me=False)["normed1"][0, 0]
    p2 = sio.loadmat(P2_MAT, squeeze_me=False)["Normed"][0, 0]

    # Overlap region = 670-1800 cm⁻¹ (Probe 1 full range; Probe 2 indices 269..1400)
    overlap_wn = wn1.copy()
    p2_overlap_start_idx = int(np.where(wn2 == 670)[0][0])

    probe_data = {"Probe1": {}, "Probe2": {}}
    for cohort in COHORTS:
        arr1 = p1[cohort]
        arr2 = p2[cohort]
        assert arr1.shape[1] == 1131 and arr2.shape[1] == 1400, "shape vs Calx mismatch"
        # Slice Probe 2 to 670-1800
        arr2_overlap = arr2[:, p2_overlap_start_idx:]
        assert arr2_overlap.shape[1] == 1131, "Probe2 overlap shape mismatch"
        probe_data["Probe1"][cohort] = arr1
        probe_data["Probe2"][cohort] = arr2_overlap
    return probe_data, overlap_wn


def canonical_preprocess_spectra(arr, wn_in, master_x):
    """Interpolate each row to master_x, then apply baseline_correct.
    Returns (N, len(master_x)) canonical-pp array."""
    out = np.full((arr.shape[0], len(master_x)), np.nan)
    for i in range(arr.shape[0]):
        y_rs = np.interp(master_x, wn_in, arr[i], left=np.nan, right=np.nan)
        y_pp = baseline_correct(y_rs)
        if np.isfinite(y_pp).any() and float(np.linalg.norm(y_pp)) >= 1e-12:
            out[i] = y_pp
    return out


def subsample_cohort(arr, n_max, seed_offset=0):
    n = arr.shape[0]
    if n <= n_max: return arr, np.arange(n)
    rng = np.random.default_rng(42 + seed_offset)
    idx = rng.choice(n, n_max, replace=False)
    return arr[idx], idx


def prepare_data(probe_data, overlap_wn):
    """Preprocess + subsample per cohort per probe. Returns a dict of arrays
    on the canonical master axis for BSV/MSS scoring, and metadata dataframe."""
    master_x = canonical_master_axis()
    out = {"Probe1": {}, "Probe2": {}}
    meta_rows = []
    for tag in ["Probe1", "Probe2"]:
        for i_coh, cohort in enumerate(COHORTS):
            raw = probe_data[tag][cohort]
            sub, idx_sel = subsample_cohort(raw, N_MAX_PER_COHORT, seed_offset=i_coh)
            print(f"  {tag}/{cohort}: {raw.shape[0]} spectra → subsample {sub.shape[0]}")
            # Preprocess: interp from overlap_wn to master_x (400-1800 step 1), then AsLS+SG+L2
            pp = canonical_preprocess_spectra(sub, overlap_wn, master_x)
            out[tag][cohort] = pp
            for j in range(pp.shape[0]):
                meta_rows.append({
                    "probe": tag, "cohort": cohort, "orig_idx": int(idx_sel[j]),
                    "ht_frac": COHORT_HT_FRAC[cohort],
                })
    meta_df = pd.DataFrame(meta_rows)
    return out, master_x, meta_df


# ──────────────────────────────────────────────────────────────────────
# Per-spectrum BSV scoring via family-aggregated MSS anchor scores
# (same kernel as MSS resolution layer v1 — no new scoring logic)
# ──────────────────────────────────────────────────────────────────────
def compute_bsv_per_spectrum(Y_pp, master_x, templates_by_mol):
    """Returns (N, 11) raw BSV scores — max-over-molecules per family."""
    family_mols = defaultdict(list)
    for mol, tps in templates_by_mol.items():
        # Use first-available template to pick family
        t = next(iter(tps.values()))
        family_mols[t["bsv_family_id"]].append((mol, t))
    n = Y_pp.shape[0]
    out = np.zeros((n, len(BSV_FAMILIES)))
    for i in range(n):
        if i % 500 == 0: print(f"    bsv {i}/{n}")
        y = Y_pp[i]
        if not np.isfinite(y).any(): continue
        for k, (fid, _) in enumerate(BSV_FAMILIES):
            best = 0.0
            for mol, t in family_mols.get(fid, []):
                # Prefer SERS template if available
                tps = templates_by_mol[mol]
                tuse = tps.get("SERS") or tps.get("Raman") or t
                sc, _, _ = mss_anchor_score(y, master_x, tuse["anchors"], tuse["supports"])
                if sc > best: best = sc
            out[i, k] = best
    return out


def bsv_transforms(bsv_raw):
    """Return dict of {'raw','sumnorm','clr'} (N, 11) arrays."""
    eps = 1e-6
    s = bsv_raw.sum(axis=1, keepdims=True)
    sumnorm = np.where(s > 0, bsv_raw / np.maximum(s, 1e-9), 0.0)
    gv = bsv_raw + eps
    log_g = np.log(gv)
    gm = log_g.mean(axis=1, keepdims=True)
    clr = log_g - gm
    return {"raw": bsv_raw, "sumnorm": sumnorm, "clr": clr}


# ──────────────────────────────────────────────────────────────────────
# BLOCK 1 — PCA (RAW vs BSV) per probe + overlay
# ──────────────────────────────────────────────────────────────────────
def block1_pca(pp_probes, bsv_probes_clr, meta_df):
    print("[BLOCK 1] PCA raw vs BSV")
    # Concatenate per probe, keep a probe label
    # Raw PCA: each spectrum is a 1401-dim vector (canonical master axis)
    X_raw = []; X_bsv = []; y_probe = []; y_cohort = []; y_htfrac = []
    for tag in ["Probe1", "Probe2"]:
        for cohort in COHORTS:
            pp = pp_probes[tag][cohort]
            bsv = bsv_probes_clr[tag][cohort]
            mask = np.isfinite(pp).all(axis=1)
            X_raw.append(pp[mask])
            X_bsv.append(bsv[mask])
            y_probe += [tag] * int(mask.sum())
            y_cohort += [cohort] * int(mask.sum())
            y_htfrac += [COHORT_HT_FRAC[cohort]] * int(mask.sum())
    X_raw = np.vstack(X_raw); X_bsv = np.vstack(X_bsv)
    y_probe = np.array(y_probe); y_cohort = np.array(y_cohort)
    y_htfrac = np.array(y_htfrac)
    print(f"  joint matrix shapes: raw {X_raw.shape}, bsv {X_bsv.shape}")

    # PCA
    pca_raw = PCA(n_components=2).fit(X_raw); Zraw = pca_raw.transform(X_raw)
    pca_bsv = PCA(n_components=2).fit(X_bsv); Zbsv = pca_bsv.transform(X_bsv)

    # Figure: 2×2 grid — (raw/bsv) × (probe coloring / cohort coloring)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        probe_colors = {"Probe1": "#4C72B0", "Probe2": "#DD8452"}
        for ax, Z, title in [(axes[0, 0], Zraw, "RAW  — by probe"),
                                (axes[1, 0], Zbsv, "BSV (CLR)  — by probe")]:
            for tag, color in probe_colors.items():
                m = y_probe == tag
                ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.35, color=color, label=tag)
            ax.set_title(title); ax.legend(fontsize=8)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

        for ax, Z, title in [(axes[0, 1], Zraw, "RAW  — by HT fraction"),
                                (axes[1, 1], Zbsv, "BSV (CLR)  — by HT fraction")]:
            sc = ax.scatter(Z[:, 0], Z[:, 1], s=5, alpha=0.4, c=y_htfrac, cmap="viridis")
            ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            plt.colorbar(sc, ax=ax, label="HT-1080 fraction (%)")
        fig.suptitle("Joint PCA — raw (top) vs BSV-CLR (bottom); probe overlay (left) vs HT-frac (right)",
                        y=1.01)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_raw_vs_bsv_pca_probe_overlay_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig1 issue: {e}")

    # Compute simple metrics: silhouette-like probe-clustering in each space
    def _probe_separation(Z, y):
        # Distance between probe-mean centroids divided by within-probe mean spread
        d = np.linalg.norm(Z[y == "Probe1"].mean(axis=0) - Z[y == "Probe2"].mean(axis=0))
        s = 0.5 * (Z[y == "Probe1"].std(axis=0).mean() + Z[y == "Probe2"].std(axis=0).mean())
        return float(d / max(s, 1e-9))
    sep_raw = _probe_separation(Zraw, y_probe)
    sep_bsv = _probe_separation(Zbsv, y_probe)
    pd.DataFrame([{
        "space": "raw_spectra (canonical pp)", "probe_centroid_dist_over_spread": sep_raw,
    }, {
        "space": "bsv_clr_11", "probe_centroid_dist_over_spread": sep_bsv,
    }]).to_csv(TABLES / "pca_probe_separation_metric_v1.csv", index=False)
    return sep_raw, sep_bsv


# ──────────────────────────────────────────────────────────────────────
# BLOCK 2 — 11-axis BSV trajectory per cohort per probe + cross-probe correlation
# ──────────────────────────────────────────────────────────────────────
def _spearman(x, y):
    x = pd.Series(x); y = pd.Series(y)
    valid = x.notna() & y.notna()
    if valid.sum() < 3: return np.nan
    rx = x[valid].rank(); ry = y[valid].rank()
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3: return np.nan
    if np.std(x) == 0 or np.std(y) == 0: return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def block2_trajectories(bsv_probes):
    """bsv_probes: dict per probe per cohort → (N, 11) sumnorm or clr vectors.
    Returns per-cohort mean and cross-probe correlation per axis."""
    print("[BLOCK 2] 11-axis trajectories")
    # Build per-probe cohort-mean table: (probe, cohort, axis) → mean
    rows = []
    trajectories = {"Probe1": {}, "Probe2": {}}
    for tag in ["Probe1", "Probe2"]:
        for cohort in COHORTS:
            mat = bsv_probes[tag][cohort]  # (N, 11) sumnorm
            if mat is None or mat.shape[0] == 0: continue
            for k, (fid, fname) in enumerate(BSV_FAMILIES):
                rows.append({
                    "probe": tag, "cohort": cohort, "axis": fid, "axis_name": fname,
                    "ht_frac": COHORT_HT_FRAC[cohort],
                    "mean": float(np.nanmean(mat[:, k])),
                    "sd":   float(np.nanstd(mat[:, k])),
                    "n":    int(mat.shape[0]),
                })
        # Per-axis trajectory vector over cohorts
        for k, (fid, _) in enumerate(BSV_FAMILIES):
            vec = [float(np.nanmean(bsv_probes[tag][c][:, k])) for c in COHORTS]
            trajectories[tag][fid] = vec
    trajectory_df = pd.DataFrame(rows)
    trajectory_df.to_csv(TABLES / "bsv_trajectory_per_cohort_per_probe_v1.csv", index=False)

    # Cross-probe trajectory correlation per axis
    corr_rows = []
    for k, (fid, fname) in enumerate(BSV_FAMILIES):
        v1 = np.array(trajectories["Probe1"][fid], dtype=float)
        v2 = np.array(trajectories["Probe2"][fid], dtype=float)
        rho = _spearman(v1, v2)
        r = _pearson(v1, v2)
        mono_p1 = _spearman(np.array([COHORT_HT_FRAC[c] for c in COHORTS]), v1)
        mono_p2 = _spearman(np.array([COHORT_HT_FRAC[c] for c in COHORTS]), v2)
        # Effect size c00 vs c100 per probe
        eff_p1 = v1[-1] - v1[0]
        eff_p2 = v2[-1] - v2[0]
        direction_match = np.sign(eff_p1) == np.sign(eff_p2) and abs(eff_p1) > 1e-6
        corr_rows.append({
            "axis": fid, "axis_name": fname,
            "probe1_trajectory": ";".join(f"{v:.4f}" for v in v1),
            "probe2_trajectory": ";".join(f"{v:.4f}" for v in v2),
            "pearson_cross_probe": r,
            "spearman_cross_probe": rho,
            "monotonicity_probe1": mono_p1,
            "monotonicity_probe2": mono_p2,
            "effect_c100_minus_c00_probe1": eff_p1,
            "effect_c100_minus_c00_probe2": eff_p2,
            "direction_agreement": bool(direction_match),
        })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(TABLES / "trajectory_correlation_table_v1.csv", index=False)

    # Figures — 11 axes trajectory per probe
    try:
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
        for ax, (fid, fname) in zip(axes.flat, BSV_FAMILIES):
            v1 = trajectories["Probe1"][fid]; v2 = trajectories["Probe2"][fid]
            x = [COHORT_HT_FRAC[c] for c in COHORTS]
            ax.plot(x, v1, "-o", color="#4C72B0", label="Probe1", lw=1.5)
            ax.plot(x, v2, "-s", color="#DD8452", label="Probe2", lw=1.5)
            r = corr_df[corr_df.axis == fid]["pearson_cross_probe"].iloc[0]
            ax.set_title(f"{fid} {fname}  (r={r:+.2f})", fontsize=10)
            ax.set_xscale("symlog", linthresh=1); ax.grid(alpha=0.3)
        axes.flat[-1].axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("HT-1080 fraction (%)")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("11-axis BSV (sumnorm) trajectories across HT-1080 mixture ratio", y=1.01)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_trajectory_plots_all_axes_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig trajectory issue: {e}")

    # Heatmap: cross-probe correlation per axis
    try:
        fig, ax = plt.subplots(figsize=(10, 3))
        arr = corr_df[["pearson_cross_probe", "spearman_cross_probe",
                          "monotonicity_probe1", "monotonicity_probe2"]].values.T
        im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_yticks(range(4))
        ax.set_yticklabels(["Pearson cross-probe", "Spearman cross-probe",
                              "Probe1 monotonicity vs HTfrac", "Probe2 monotonicity vs HTfrac"],
                             fontsize=9)
        ax.set_xticks(range(len(BSV_FAMILIES)))
        ax.set_xticklabels([f"{fid}" for fid, _ in BSV_FAMILIES], fontsize=9)
        plt.colorbar(im, ax=ax, label="correlation")
        ax.set_title("Trajectory correlations per BSV axis (cross-probe + monotonicity vs HT fraction)")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                              fontsize=7, color="white" if abs(v) > 0.6 else "black")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_trajectory_correlation_heatmap_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig corr heatmap issue: {e}")

    return trajectory_df, corr_df, trajectories


# ──────────────────────────────────────────────────────────────────────
# BLOCK 3 — top BSV axes per probe (data-driven; no forcing)
# ──────────────────────────────────────────────────────────────────────
def block3_top_axes(trajectories, bsv_probes):
    print("[BLOCK 3] top BSV axes per probe")
    rows = {"Probe1": [], "Probe2": []}
    for tag in ["Probe1", "Probe2"]:
        for k, (fid, fname) in enumerate(BSV_FAMILIES):
            traj = trajectories[tag][fid]
            # Variance across cohort means
            var_c = float(np.std(traj))
            # Effect size c00 vs c100 (endpoints)
            eff = float(traj[-1] - traj[0])
            # Monotonicity vs HT fraction
            mono = _spearman(np.array([COHORT_HT_FRAC[c] for c in COHORTS]),
                                np.array(traj, float))
            rows[tag].append({
                "axis": fid, "axis_name": fname,
                "cohort_mean_variance": var_c,
                "effect_c100_minus_c00": eff,
                "monotonicity_vs_htfrac": mono,
                "combined_rank_score": abs(eff) + var_c,  # simple combined metric
            })
    top_rows = {}
    for tag in ["Probe1", "Probe2"]:
        df = pd.DataFrame(rows[tag]).sort_values("combined_rank_score", ascending=False)
        df["rank"] = np.arange(1, len(df) + 1)
        df.to_csv(TABLES / f"top_axes_{tag.lower()}_v1.csv", index=False)
        top_rows[tag] = df.head(5)["axis"].tolist()
        print(f"  {tag} top 5: {top_rows[tag]}")
    return top_rows


# ──────────────────────────────────────────────────────────────────────
# BLOCK 4 — MSS resolution constrained to molecules in top BSV axes
# ──────────────────────────────────────────────────────────────────────
def block4_mss_resolution(pp_probes, meta_df, top_axes, master_x, templates_by_mol):
    """For each probe, for each selected top BSV axis, compute per-spectrum
    MSS scores for molecules in that axis. Return per-cohort effect sizes +
    trajectories + ranking."""
    print("[BLOCK 4] MSS resolution constrained to top BSV axes")
    family_mols = defaultdict(list)
    for mol, tps in templates_by_mol.items():
        t = next(iter(tps.values()))
        family_mols[t["bsv_family_id"]].append(mol)

    rows = []
    per_spec_scores = {"Probe1": {}, "Probe2": {}}
    for tag in ["Probe1", "Probe2"]:
        sel_axes = top_axes[tag]
        sel_mols = sorted({m for fid in sel_axes for m in family_mols.get(fid, [])})
        print(f"  {tag} selected molecules from {sel_axes}: {sel_mols}")
        for cohort in COHORTS:
            Y = pp_probes[tag][cohort]
            # Score each spectrum against each selected molecule
            mat = np.zeros((Y.shape[0], len(sel_mols)))
            for j, mol in enumerate(sel_mols):
                tps = templates_by_mol[mol]
                t = tps.get("SERS") or tps.get("Raman") or next(iter(tps.values()))
                for i in range(Y.shape[0]):
                    if not np.isfinite(Y[i]).any(): continue
                    sc, _, _ = mss_anchor_score(Y[i], master_x, t["anchors"], t["supports"])
                    mat[i, j] = sc
            per_spec_scores[tag][cohort] = {"mols": sel_mols, "mat": mat}

        # Compute cohort means + cross-cohort effect sizes
        first = per_spec_scores[tag][COHORTS[0]]["mat"]
        last  = per_spec_scores[tag][COHORTS[-1]]["mat"]
        for j, mol in enumerate(sel_mols):
            trajectory = [float(per_spec_scores[tag][c]["mat"][:, j].mean())
                             for c in COHORTS]
            eff = trajectory[-1] - trajectory[0]
            mono = _spearman(np.array([COHORT_HT_FRAC[c] for c in COHORTS]),
                                np.array(trajectory, float))
            # Find the axis this molecule belongs to (first axis match)
            ax_id = next((fid for fid in sel_axes
                              if mol in family_mols.get(fid, [])), "?")
            rows.append({
                "probe":         tag,
                "selected_axis": ax_id,
                "molecule":      mol,
                "n_per_cohort":  ";".join(str(per_spec_scores[tag][c]["mat"].shape[0]) for c in COHORTS),
                "trajectory":    ";".join(f"{v:.4f}" for v in trajectory),
                "effect_c100_minus_c00": eff,
                "monotonicity_vs_htfrac": mono,
            })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "mss_effect_sizes_v1.csv", index=False)

    # MSS trajectory figure for top 8 molecules per probe
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for ax, tag in zip(axes, ["Probe1", "Probe2"]):
            sub = df[df.probe == tag].sort_values(
                "effect_c100_minus_c00", key=lambda s: s.abs(), ascending=False).head(8)
            for _, r in sub.iterrows():
                traj = [float(x) for x in r["trajectory"].split(";")]
                ax.plot([COHORT_HT_FRAC[c] for c in COHORTS], traj, "-o",
                          label=f"{r['molecule']} ({r['selected_axis']})", lw=1.3)
            ax.set_title(f"{tag} — top 8 MSS molecules by |Δ c100-c00|")
            ax.set_xlabel("HT-1080 fraction (%)"); ax.set_ylabel("MSS score")
            ax.set_xscale("symlog", linthresh=1); ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_mss_trajectory_top_molecules_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig mss trajectory issue: {e}")

    return df, per_spec_scores


# ──────────────────────────────────────────────────────────────────────
# BLOCK 5 — cross-probe MSS consistency
# ──────────────────────────────────────────────────────────────────────
def block5_cross_probe_mss(mss_df):
    print("[BLOCK 5] cross-probe MSS consistency")
    # For molecules appearing in both probes' MSS tables
    p1 = mss_df[mss_df.probe == "Probe1"].set_index("molecule")
    p2 = mss_df[mss_df.probe == "Probe2"].set_index("molecule")
    common = sorted(set(p1.index) & set(p2.index))
    rows = []
    for mol in common:
        traj1 = np.array([float(x) for x in p1.loc[mol, "trajectory"].split(";")])
        traj2 = np.array([float(x) for x in p2.loc[mol, "trajectory"].split(";")])
        r = _pearson(traj1, traj2)
        rho = _spearman(traj1, traj2)
        eff1 = float(p1.loc[mol, "effect_c100_minus_c00"])
        eff2 = float(p2.loc[mol, "effect_c100_minus_c00"])
        direction_agree = (np.sign(eff1) == np.sign(eff2)) and min(abs(eff1), abs(eff2)) > 1e-3
        magnitude_ratio = min(abs(eff1), abs(eff2)) / max(abs(eff1), abs(eff2), 1e-6)
        if direction_agree and magnitude_ratio > 0.5 and (r or 0) > 0.6:
            cls = "CONSISTENT"
        elif direction_agree:
            cls = "PARTIAL"
        else:
            cls = "PROBE_SPECIFIC"
        rows.append({
            "molecule":    mol,
            "probe1_axis": p1.loc[mol, "selected_axis"],
            "probe2_axis": p2.loc[mol, "selected_axis"],
            "effect_probe1": eff1, "effect_probe2": eff2,
            "direction_agreement": bool(direction_agree),
            "pearson_trajectory": r,
            "spearman_trajectory": rho,
            "magnitude_ratio": magnitude_ratio,
            "classification": cls,
        })
    df = pd.DataFrame(rows).sort_values("classification")
    df.to_csv(TABLES / "mss_cross_probe_consistency_v1.csv", index=False)

    # Figure: scatter of Probe1 effect vs Probe2 effect, colored by classification
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {"CONSISTENT": "#2ca02c", "PARTIAL": "#f39c12", "PROBE_SPECIFIC": "#c0392b"}
        for cls, color in colors.items():
            sub = df[df.classification == cls]
            ax.scatter(sub.effect_probe1, sub.effect_probe2, s=60,
                          color=color, label=f"{cls} (n={len(sub)})", alpha=0.75)
            for _, r in sub.iterrows():
                ax.annotate(r["molecule"], (r.effect_probe1, r.effect_probe2),
                              fontsize=7, alpha=0.7)
        lim = float(max(df.effect_probe1.abs().max(), df.effect_probe2.abs().max()) * 1.1)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, alpha=0.5)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("Probe 1 Δ c100-c00 (MSS)"); ax.set_ylabel("Probe 2 Δ c100-c00 (MSS)")
        ax.set_title("MSS cross-probe consistency — endpoint effect comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_mss_cross_probe_comparison_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig mss cross-probe issue: {e}")
    return df


# ──────────────────────────────────────────────────────────────────────
# BLOCK 6 — MSS top-K constrained to selected-axis molecules
# ──────────────────────────────────────────────────────────────────────
def block6_topk(per_spec_scores):
    print("[BLOCK 6] MSS top-K constrained to selected-axis molecules")
    rows = []
    for tag in ["Probe1", "Probe2"]:
        for cohort in COHORTS:
            d = per_spec_scores[tag][cohort]
            mols = d["mols"]; mat = d["mat"]
            if mat.shape[0] == 0: continue
            # For each spectrum, rank selected mols
            order = np.argsort(-mat, axis=1)  # (N, M)
            for k in (3, 5):
                top_k_idx = order[:, :k]
                for j, mol in enumerate(mols):
                    freq = float(np.mean(np.any(top_k_idx == j, axis=1)))
                    rows.append({
                        "probe": tag, "cohort": cohort, "k": k,
                        "molecule": mol, "freq": freq,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "mss_topk_frequency_v1.csv", index=False)
    return df


# ──────────────────────────────────────────────────────────────────────
# Final report
# ──────────────────────────────────────────────────────────────────────
def write_report(sep_raw, sep_bsv, corr_df, top_axes, mss_df, mss_xp_df, topk_df):
    lines = [
        "# REPORT — small2023_ev dual-probe GAIRA analysis v1\n",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Setup",
        "- Dataset: small2023_ev, Parlatan 2023 (EV mixture HT-1080:THP-1 series, 6 cohorts c00-c100).",
        "- Probes kept SEPARATE. Canonical preprocessing (interp→AsLS→SG→L2) applied per spectrum.",
        "- Overlap region 670-1800 cm⁻¹ (Probe 1 native; Probe 2 sliced).",
        "- Subsampled to ≤2000 spectra per cohort per probe.",
        "- NO substrate physics wrapper, NO classifier, engine/MSS/BSV UNCHANGED.",
        "",
        "## Required answers\n",
    ]

    # Q1: BSV vs raw probe separation
    lines.append("### 1. Does BSV collapse probe differences while preserving biology?")
    lines.append(f"- Probe-centroid distance normalized by within-probe spread:")
    lines.append(f"  - RAW PCA: {sep_raw:.2f}")
    lines.append(f"  - BSV-CLR PCA: {sep_bsv:.2f}")
    if sep_bsv < sep_raw * 0.7:
        lines.append(f"  → **YES**, BSV reduces probe separation by {100*(1 - sep_bsv/max(sep_raw,1e-9)):.0f}% vs raw.")
    elif sep_bsv < sep_raw:
        lines.append(f"  → **Partially**, BSV reduces but does not eliminate probe separation.")
    else:
        lines.append(f"  → **No**, BSV does not reduce probe separation relative to raw.")
    lines.append("")

    # Q2: axes that transfer
    lines.append("### 2. Which axes transfer across probes?")
    lines.append("| axis | Pearson cross-probe | direction match | Probe1 mono | Probe2 mono |")
    lines.append("|---|---:|---|---:|---:|")
    for _, r in corr_df.iterrows():
        lines.append(f"| {r['axis']} {r['axis_name']} | {r['pearson_cross_probe']:+.2f} | "
                        f"{'✓' if r['direction_agreement'] else '✗'} | "
                        f"{r['monotonicity_probe1']:+.2f} | {r['monotonicity_probe2']:+.2f} |")
    transfers = corr_df[(corr_df.pearson_cross_probe > 0.6) & (corr_df.direction_agreement)]
    lines.append("")
    lines.append(f"- **{len(transfers)}/11 axes** have cross-probe Pearson > 0.6 AND direction match.")
    if not transfers.empty:
        lines.append(f"  - transferring axes: {transfers['axis'].tolist()}")
    lines.append("")

    # Q3: mixture trajectory matching
    lines.append("### 3. Does mixture trajectory match across probes?")
    n_mono_p1 = int((corr_df["monotonicity_probe1"].abs() >= 0.6).sum())
    n_mono_p2 = int((corr_df["monotonicity_probe2"].abs() >= 0.6).sum())
    lines.append(f"- Probe1: {n_mono_p1}/11 axes show |monotonicity vs HT-fraction| ≥ 0.6")
    lines.append(f"- Probe2: {n_mono_p2}/11 axes show |monotonicity vs HT-fraction| ≥ 0.6")
    both = corr_df[(corr_df["monotonicity_probe1"].abs() >= 0.6) &
                       (corr_df["monotonicity_probe2"].abs() >= 0.6) &
                       (np.sign(corr_df["monotonicity_probe1"]) == np.sign(corr_df["monotonicity_probe2"]))]
    lines.append(f"- **{len(both)}/11 axes** show monotonicity≥0.6 same-direction on BOTH probes "
                    f"→ {both['axis'].tolist() if not both.empty else '(none)'}")
    lines.append("")

    # Q4: consistent MSS molecules
    lines.append("### 4. Which MSS molecules are consistent across probes?")
    n_consistent = int((mss_xp_df.classification == "CONSISTENT").sum())
    n_partial = int((mss_xp_df.classification == "PARTIAL").sum())
    n_probe_specific = int((mss_xp_df.classification == "PROBE_SPECIFIC").sum())
    lines.append(f"- Of {len(mss_xp_df)} molecules scored on both probes:")
    lines.append(f"  - CONSISTENT: {n_consistent} ({sorted(mss_xp_df[mss_xp_df.classification=='CONSISTENT']['molecule'].tolist())})")
    lines.append(f"  - PARTIAL: {n_partial} ({sorted(mss_xp_df[mss_xp_df.classification=='PARTIAL']['molecule'].tolist())})")
    lines.append(f"  - PROBE_SPECIFIC: {n_probe_specific} ({sorted(mss_xp_df[mss_xp_df.classification=='PROBE_SPECIFIC']['molecule'].tolist())})")
    lines.append("")

    # Q5: does MSS add resolution beyond BSV?
    lines.append("### 5. Does MSS add real resolution beyond BSV or just noise?")
    mss_p1 = mss_df[mss_df.probe == "Probe1"]["effect_c100_minus_c00"].abs()
    mss_p2 = mss_df[mss_df.probe == "Probe2"]["effect_c100_minus_c00"].abs()
    bsv_mass = corr_df[corr_df.direction_agreement]["effect_c100_minus_c00_probe1"].abs().mean() \
                   if not corr_df[corr_df.direction_agreement].empty else np.nan
    lines.append(f"- MSS median |Δ c100-c00| on Probe1: {float(mss_p1.median()):.3f}  (n={len(mss_p1)} molecules)")
    lines.append(f"- MSS median |Δ c100-c00| on Probe2: {float(mss_p2.median()):.3f}  (n={len(mss_p2)} molecules)")
    lines.append(f"- BSV-axis mean |Δ c100-c00| (direction-matched axes): {bsv_mass:.3f}")
    if n_consistent >= 2:
        lines.append(f"  → **YES** — {n_consistent} molecules transfer consistently across probes, "
                        f"providing per-molecule resolution beyond axis-level BSV.")
    else:
        lines.append(f"  → **Limited** — only {n_consistent} cross-probe CONSISTENT molecules; "
                        f"MSS resolution is mostly probe-specific on this dataset.")
    lines.append("")

    # Top BSV axes per probe
    lines.append("## Top BSV axes (data-driven, no forced overlap)\n")
    lines.append(f"- Probe 1: {top_axes['Probe1']}")
    lines.append(f"- Probe 2: {top_axes['Probe2']}")
    common_top = [a for a in top_axes['Probe1'] if a in top_axes['Probe2']]
    lines.append(f"- Overlap (top-5): {common_top}")
    lines.append("")

    # Top-k summary
    lines.append("## Top-K MSS frequency (constrained to selected-axis molecules)")
    lines.append("See `tables/mss_topk_frequency_v1.csv` — per probe × cohort × molecule frequency at k=3,5.")
    lines.append("")
    (REPORTS / "REPORT_small_ev_dual_probe_analysis_v1.md").write_text("\n".join(lines))


def write_audit():
    txt = [
        "# gaira_base_4_small_ev_dual_probe_analysis_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Strict invariants",
        "- Engine v4.5 / MSS scoring kernel / motif registry / MSS templates / 11-axis BSV / preprocessing ALL UNCHANGED",
        "- Probe 1 and Probe 2 kept SEPARATE through the entire pipeline",
        "- Canonical preprocessing: interp→master_x → AsLS → Savitzky-Golay → L2",
        "- Overlap region 670-1800 cm⁻¹ (Probe 1 native; Probe 2 sliced from 401-1800 to 670-1800)",
        "- NO substrate-physics wrapper (probe physics still unspecified in local files)",
        "- NO classifier",
        "- NO disease labels (dataset has no disease cohorts)",
        "- Per-cohort subsampling cap N_MAX_PER_COHORT=2000; deterministic RNG (seed=42 + cohort offset)",
        "",
        "## Inputs",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/small2023_ev/NormedProbe1.mat",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/small2023_ev/NormedProbe2.mat",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/small2023_ev/Fig_S7 (1).xlsx (Calx for both probes)",
        "",
        "## Outputs",
        "- tables/pca_probe_separation_metric_v1.csv",
        "- tables/bsv_trajectory_per_cohort_per_probe_v1.csv",
        "- tables/trajectory_correlation_table_v1.csv",
        "- tables/top_axes_probe1_v1.csv + tables/top_axes_probe2_v1.csv",
        "- tables/mss_effect_sizes_v1.csv",
        "- tables/mss_cross_probe_consistency_v1.csv",
        "- tables/mss_topk_frequency_v1.csv",
        "- figures: raw_vs_bsv_pca, trajectory_plots, trajectory_heatmap, mss_trajectory, mss_cross_probe",
        "- reports/REPORT_small_ev_dual_probe_analysis_v1.md",
    ]
    (AUDIT / "gaira_base_4_small_ev_dual_probe_analysis_v1_audit_log.md").write_text("\n".join(txt))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("gaira_base_4_small_ev_dual_probe_analysis_v1")
    print("=" * 78)

    print("[STEP 0] load + Calx slice + subsample + canonical preprocess")
    probe_data, overlap_wn = load_probe_spectra()
    pp_probes, master_x, meta_df = prepare_data(probe_data, overlap_wn)

    print("[BSV] computing 11-axis BSV per spectrum per probe/cohort")
    templates, _, _ = load_templates()
    by_mol = defaultdict(dict)
    for t in templates:
        by_mol[t["molecule"]][t["regime"]] = t

    bsv_raw = {"Probe1": {}, "Probe2": {}}
    bsv_sumnorm = {"Probe1": {}, "Probe2": {}}
    bsv_clr = {"Probe1": {}, "Probe2": {}}
    for tag in ["Probe1", "Probe2"]:
        for cohort in COHORTS:
            Y = pp_probes[tag][cohort]
            bsv_r = compute_bsv_per_spectrum(Y, master_x, by_mol)
            bt = bsv_transforms(bsv_r)
            bsv_raw[tag][cohort] = bt["raw"]
            bsv_sumnorm[tag][cohort] = bt["sumnorm"]
            bsv_clr[tag][cohort] = bt["clr"]

    sep_raw, sep_bsv = block1_pca(pp_probes, bsv_clr, meta_df)
    traj_df, corr_df, trajectories = block2_trajectories(bsv_sumnorm)
    top_axes = block3_top_axes(trajectories, bsv_sumnorm)
    mss_df, per_spec_scores = block4_mss_resolution(
        pp_probes, meta_df, top_axes, master_x, by_mol)
    mss_xp_df = block5_cross_probe_mss(mss_df)
    topk_df = block6_topk(per_spec_scores)
    write_report(sep_raw, sep_bsv, corr_df, top_axes, mss_df, mss_xp_df, topk_df)
    write_audit()
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print("[done]")


if __name__ == "__main__":
    main()
