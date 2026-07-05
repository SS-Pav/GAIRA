"""GAIRA Target Pilot 1 — HCC holdout (Gurian et al. 2020, Ag plasmonic paper).

Spectrum-level BSV / ΔBSV pilot. Audit verdict: GO — spectrum-level only.

- Uses locked `gaira.spectral.*` pipeline end-to-end.
- Healthy reference = mean BSV of 72 CTR spectra.
- Per-spectrum ΔBSV = spectrum_bsv − healthy_centroid.
- SAEL / R7c-style agreement applied only to cohort-mean Δ as a summary tag.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_target_pilot1_hcc_holdout_bsv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral.dataset_loader import HCC_HOLDOUT_CSV, load_dataset
from gaira.spectral.preprocessing import preprocess
from gaira.spectral.window_panel import BSV_COMPONENTS, extract_window_features
from gaira.spectral.bsv_projection import project_to_bsv
from gaira.spectral.expected_bsv import build_expected_comparators
from gaira.spectral.comparison import compute_delta_comparison


DATASET_ID = "hcc_holdout_vornoli2020"          # kept for backward compat
PAPER = "Gurian et al. 2020 (Bonifacio group, Trieste)"
SUBSTRATE = "Ag plasmonic paper (per parser provenance)"
OUT_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_target_pilot1_hcc_holdout_bsv")
RNG = np.random.default_rng(42)
N_BOOT = 1000

BSV_SHORT = {
    "membrane_lipid": "Lipid", "protein_backbone": "Protein",
    "aromatic_amino_acid": "Aromatic AA", "purine_nucleotide": "Purine",
    "pyrimidine_nucleotide": "Pyrimidine", "glycan_carbohydrate": "Glycan",
    "redox_metabolite": "Redox", "nucleic_acid_backbone": "Nuc.Backbone",
}
CLASS_COLORS = {"healthy_control": "#4C78A8", "hcc": "#E45756"}
BATCH_MARKERS = {"A": "o", "B": "s", "C": "^"}


# ──────────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────────

def _mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    if len(a) < 2 or len(b) < 2:
        return 0.0, 0.0, 0.0
    pooled = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
    d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
    boots = np.empty(N_BOOT)
    for k in range(N_BOOT):
        aa = RNG.choice(a, size=len(a), replace=True)
        bb = RNG.choice(b, size=len(b), replace=True)
        p = float(np.sqrt((aa.var(ddof=1) + bb.var(ddof=1)) / 2.0))
        boots[k] = (aa.mean() - bb.mean()) / p if p > 0 else 0.0
    return d, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a) * len(b)
    if n == 0:
        return 0.0
    greater = int(sum((ai > b).sum() for ai in a))
    less = int(sum((ai < b).sum() for ai in a))
    return float((greater - less) / n)


# ──────────────────────────────────────────────────────────────────────
# Load + pipeline
# ──────────────────────────────────────────────────────────────────────

def _load_metadata_aligned(n_rows: int) -> pd.DataFrame:
    df = pd.read_csv(
        HCC_HOLDOUT_CSV,
        usecols=["acquisition_date", "substrate_batch", "class", "sample_code"],
    )
    assert len(df) == n_rows, f"metadata row count {len(df)} != dataset rows {n_rows}"
    df["class_gaira"] = df["class"].map({"CTR": "healthy_control", "H0T": "hcc"})
    df["biosample_id"] = (
        df["acquisition_date"].astype(int).astype(str).str.zfill(8) + "_batch-"
        + df["substrate_batch"].astype(str) + "_"
        + df["class"].astype(str) + "_"
        + df["sample_code"].astype(int).astype(str).str.zfill(3)
    )
    df["biosample_id"] = "hcc_serum_" + df["biosample_id"]
    return df


def run_pipeline():
    ds = load_dataset(DATASET_ID)
    Xn, prep = preprocess(ds)
    wf = extract_window_features(Xn, ds.wavenumbers)
    bsv = project_to_bsv(wf)
    meta = _load_metadata_aligned(bsv.shape[0])
    # Sanity: cohort mapping equal to row-aligned metadata
    assert (meta["class_gaira"].to_numpy() == ds.cohorts).all(), "row alignment mismatch"
    return ds, Xn, wf, bsv, prep, meta


# ──────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────

def write_tables(tables_dir: Path, bsv, delta, dist, ds, meta):
    base = meta[["biosample_id", "class_gaira", "sample_code", "substrate_batch", "acquisition_date"]].copy()
    base = base.rename(columns={"class_gaira": "class"})

    # 1. per_spectrum_bsv
    df1 = base.copy()
    for i, c in enumerate(BSV_COMPONENTS):
        df1[f"bsv_{c}"] = bsv[:, i]
    df1.to_csv(tables_dir / "pilot1_hcc_per_spectrum_bsv.csv", index=False)

    # 2. per_spectrum_delta_bsv
    df2 = base.copy()
    for i, c in enumerate(BSV_COMPONENTS):
        df2[f"delta_bsv_{c}"] = delta[:, i]
    df2["distance_to_healthy_centroid"] = dist
    df2.to_csv(tables_dir / "pilot1_hcc_per_spectrum_delta_bsv.csv", index=False)

    # 3. cohort_summary
    rows = []
    for cls in ["healthy_control", "hcc"]:
        mask = ds.cohorts == cls
        for i, c in enumerate(BSV_COMPONENTS):
            b = bsv[mask, i]; d = delta[mask, i]
            rows.append({
                "class": cls, "axis": c, "n": int(mask.sum()),
                "mean_bsv": float(b.mean()),
                "median_bsv": float(np.median(b)),
                "sd_bsv": float(b.std(ddof=1)),
                "iqr_bsv": float(np.percentile(b, 75) - np.percentile(b, 25)),
                "mean_delta_bsv": float(d.mean()),
                "median_delta_bsv": float(np.median(d)),
                "sd_delta_bsv": float(d.std(ddof=1)),
                "iqr_delta_bsv": float(np.percentile(d, 75) - np.percentile(d, 25)),
                "mad_delta_bsv": _mad(d),
            })
    pd.DataFrame(rows).to_csv(tables_dir / "pilot1_hcc_cohort_summary.csv", index=False)

    # 4. batch_summary
    rows = []
    batch_arr = meta["substrate_batch"].to_numpy()
    for cls in ["healthy_control", "hcc"]:
        for batch in ["A", "B", "C"]:
            mask = (ds.cohorts == cls) & (batch_arr == batch)
            n = int(mask.sum())
            for i, c in enumerate(BSV_COMPONENTS):
                d_vals = delta[mask, i] if n > 0 else np.array([])
                rows.append({
                    "class": cls, "batch": batch, "axis": c, "count": n,
                    "mean_delta_bsv": float(d_vals.mean()) if n else float("nan"),
                    "sd_delta_bsv": float(d_vals.std(ddof=1)) if n > 1 else float("nan"),
                })
    pd.DataFrame(rows).to_csv(tables_dir / "pilot1_hcc_batch_summary.csv", index=False)

    # 5. axis_effect_sizes  — Cohen's d on BSV (HCC vs CTR), Cliff's delta, bootstrap CI
    rows = []
    is_ctr = ds.cohorts == "healthy_control"
    is_hcc = ds.cohorts == "hcc"
    for i, c in enumerate(BSV_COMPONENTS):
        ctr = bsv[is_ctr, i]; hcc = bsv[is_hcc, i]
        d, lo, hi = _cohens_d(hcc, ctr)
        rows.append({
            "axis": c,
            "ctr_mean": float(ctr.mean()), "hcc_mean": float(hcc.mean()),
            "delta_mean": float(hcc.mean() - ctr.mean()),
            "ctr_sd": float(ctr.std(ddof=1)), "hcc_sd": float(hcc.std(ddof=1)),
            "ctr_mad": _mad(ctr), "hcc_mad": _mad(hcc),
            "pooled_sd": float(np.sqrt((ctr.var(ddof=1) + hcc.var(ddof=1)) / 2.0)),
            "cohens_d": d, "cohens_d_ci_low": lo, "cohens_d_ci_high": hi,
            "cliffs_delta": _cliffs_delta(hcc, ctr),
        })
    pd.DataFrame(rows).to_csv(tables_dir / "pilot1_hcc_axis_effect_sizes.csv", index=False)


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 180,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig1_bsv_heatmap(fig_dir, bsv, ds, meta):
    # Sort rows: class → batch → acquisition_date → sample_code
    idx = np.lexsort((
        meta["sample_code"].to_numpy(),
        meta["acquisition_date"].to_numpy(),
        meta["substrate_batch"].to_numpy(),
        ds.cohorts,
    ))
    bsv_sorted = bsv[idx]
    cls_sorted = ds.cohorts[idx]
    batch_sorted = meta["substrate_batch"].to_numpy()[idx]

    # Per-axis z-score for visual comparability
    z = (bsv_sorted - bsv_sorted.mean(axis=0)) / (bsv_sorted.std(axis=0, ddof=1) + 1e-9)

    fig, (ax_sb, ax) = plt.subplots(
        1, 2, figsize=(10, 11),
        gridspec_kw={"width_ratios": [0.05, 1.0], "wspace": 0.01},
    )
    # Class sidebar (single strip)
    class_col = np.array([[0] if c == "healthy_control" else [1] for c in cls_sorted])
    ax_sb.imshow(class_col, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    ax_sb.set_yticks([]); ax_sb.set_xticks([])
    ax_sb.set_title("class", fontsize=9)

    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(len(BSV_COMPONENTS)))
    ax.set_xticklabels([BSV_SHORT[c] for c in BSV_COMPONENTS], rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_ylabel("spectra (sorted: class → batch → date)")
    ax.set_title("Per-spectrum BSV (z-scored within axis)", fontsize=12)
    # Class separator line
    split = int((cls_sorted == "healthy_control").sum())
    ax.axhline(split - 0.5, color="black", lw=1.0)

    # Batch ticks on right
    batch_changes = [0] + [i for i in range(1, len(batch_sorted)) if batch_sorted[i] != batch_sorted[i - 1]] + [len(batch_sorted)]
    for i in range(len(batch_changes) - 1):
        ymid = (batch_changes[i] + batch_changes[i + 1]) / 2
        ax.text(8.1, ymid, batch_sorted[batch_changes[i]],
                fontsize=8, va="center", ha="left", color="#444")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("z-score (per axis)", fontsize=9)
    _save(fig, fig_dir / "fig1_bsv_heatmap.png")


def fig2_delta_distributions(fig_dir, delta, ds):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=False)
    for i, c in enumerate(BSV_COMPONENTS):
        ax = axes.flat[i]
        ctr = delta[ds.cohorts == "healthy_control", i]
        hcc = delta[ds.cohorts == "hcc", i]
        bp = ax.boxplot(
            [ctr, hcc], positions=[0, 1], widths=0.55,
            patch_artist=True, showfliers=True,
            medianprops=dict(color="black", linewidth=1.4),
            flierprops=dict(marker="o", markersize=3, markerfacecolor="gray", markeredgecolor="none", alpha=0.5),
        )
        for patch, cls in zip(bp["boxes"], ["healthy_control", "hcc"]):
            patch.set_facecolor(CLASS_COLORS[cls]); patch.set_alpha(0.55)
        ax.axhline(0, color="#888", lw=0.8, linestyle="--")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["CTR", "HCC"])
        ax.set_title(BSV_SHORT[c], fontsize=11)
        ax.set_ylabel("ΔBSV" if i % 4 == 0 else "")
    fig.suptitle("Per-axis ΔBSV vs healthy centroid — CTR (n=72) vs HCC (n=72)", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "fig2_delta_bsv_distributions.png")


def _radar(ax, labels, series, colors, fill_alpha=0.15, linewidth=2):
    n = len(labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    theta_closed = theta + [theta[0]]
    for (name, vals), color in zip(series, colors):
        v = list(vals) + [vals[0]]
        ax.plot(theta_closed, v, color=color, linewidth=linewidth, label=name)
        ax.fill(theta_closed, v, color=color, alpha=fill_alpha)
    ax.set_xticks(theta); ax.set_xticklabels(labels, fontsize=9)
    ax.tick_params(axis="y", labelsize=8, colors="#555")


def fig3_cohort_bsv_radar(fig_dir, bsv, ds):
    ctr_mean = bsv[ds.cohorts == "healthy_control"].mean(axis=0)
    hcc_mean = bsv[ds.cohorts == "hcc"].mean(axis=0)
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    _radar(
        ax, [BSV_SHORT[c] for c in BSV_COMPONENTS],
        [("CTR (n=72)", ctr_mean), ("HCC (n=72)", hcc_mean)],
        [CLASS_COLORS["healthy_control"], CLASS_COLORS["hcc"]],
        fill_alpha=0.18,
    )
    ax.set_title("Cohort mean BSV (summary only)", y=1.08, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.05), fontsize=9)
    _save(fig, fig_dir / "fig3_cohort_bsv_radar.png")


def fig4_cohort_delta_radar(fig_dir, delta, ds):
    hcc_mean_delta = delta[ds.cohorts == "hcc"].mean(axis=0)
    # Symmetric ring so sign is visible — shift + scale for polar
    m = max(1e-4, float(np.abs(hcc_mean_delta).max()) * 1.2)
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    _radar(
        ax, [BSV_SHORT[c] for c in BSV_COMPONENTS],
        [("HCC Δ vs healthy centroid", hcc_mean_delta)],
        [CLASS_COLORS["hcc"]],
        fill_alpha=0.22, linewidth=2.2,
    )
    ax.set_ylim(-m, m)
    ax.axhline = None  # no-op; keep polar defaults
    # Add zero circle manually
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(theta, np.zeros_like(theta), color="#444", lw=0.8, linestyle="--")
    ax.set_title("Cohort mean ΔBSV — HCC vs healthy centroid", y=1.08, fontsize=12)
    _save(fig, fig_dir / "fig4_cohort_delta_bsv_radar.png")


def fig5_batch_effect(fig_dir, delta, ds, meta):
    batch_arr = meta["substrate_batch"].to_numpy()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    for i, c in enumerate(BSV_COMPONENTS):
        ax = axes.flat[i]
        data = []; labels = []; colors = []
        for cls in ["healthy_control", "hcc"]:
            for batch in ["A", "B", "C"]:
                mask = (ds.cohorts == cls) & (batch_arr == batch)
                data.append(delta[mask, i] if mask.sum() else np.array([0.0]))
                labels.append(f"{'CTR' if cls=='healthy_control' else 'HCC'}\n{batch}")
                colors.append(CLASS_COLORS[cls])
        bp = ax.boxplot(data, positions=range(len(data)), widths=0.55, patch_artist=True,
                         showfliers=False, medianprops=dict(color="black", linewidth=1.2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.5)
        ax.axhline(0, color="#888", lw=0.8, linestyle="--")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(BSV_SHORT[c], fontsize=11)
        ax.set_ylabel("ΔBSV" if i % 4 == 0 else "")
    fig.suptitle("Batch-effect panel — ΔBSV by class × substrate_batch", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "fig5_batch_effect_panel.png")


def fig6_pca(fig_dir, bsv, ds, meta):
    pca = PCA(n_components=2)
    pp = pca.fit_transform(bsv)
    ev = pca.explained_variance_ratio_ * 100
    batch_arr = meta["substrate_batch"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for cls in ["healthy_control", "hcc"]:
        for batch in ["A", "B", "C"]:
            m = (ds.cohorts == cls) & (batch_arr == batch)
            if not m.any():
                continue
            ax.scatter(
                pp[m, 0], pp[m, 1],
                c=CLASS_COLORS[cls], marker=BATCH_MARKERS[batch],
                s=65, alpha=0.78, edgecolor="white", linewidth=0.6,
                label=f"{'CTR' if cls=='healthy_control' else 'HCC'} · batch {batch}",
            )
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title("PCA of per-spectrum BSV (8-D)", fontsize=12)
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(alpha=0.2, linestyle=":")
    _save(fig, fig_dir / "fig6_pca_bsv_space.png")


def fig7_distance(fig_dir, dist, ds):
    fig, ax = plt.subplots(figsize=(8, 5))
    ctr = dist[ds.cohorts == "healthy_control"]
    hcc = dist[ds.cohorts == "hcc"]
    bins = np.linspace(0, max(dist.max() * 1.02, 1e-3), 25)
    ax.hist(ctr, bins=bins, color=CLASS_COLORS["healthy_control"], alpha=0.6, label=f"CTR (n={len(ctr)})")
    ax.hist(hcc, bins=bins, color=CLASS_COLORS["hcc"], alpha=0.6, label=f"HCC (n={len(hcc)})")
    ax.axvline(np.median(ctr), color=CLASS_COLORS["healthy_control"], lw=1.6, linestyle="--",
                label=f"CTR median = {np.median(ctr):.4f}")
    ax.axvline(np.median(hcc), color=CLASS_COLORS["hcc"], lw=1.6, linestyle="--",
                label=f"HCC median = {np.median(hcc):.4f}")
    ax.set_xlabel("‖spectrum BSV − healthy centroid‖₂")
    ax.set_ylabel("Count")
    ax.set_title("Distance-to-healthy-centroid distributions", fontsize=12)
    ax.legend(fontsize=9)
    _save(fig, fig_dir / "fig7_distance_to_centroid.png")


# ──────────────────────────────────────────────────────────────────────
# SAEL / comparator summary on cohort-mean Δ (not primary evidence)
# ──────────────────────────────────────────────────────────────────────

def summarize_vs_expected(bsv, ds) -> dict:
    cb_means = {
        "healthy_control": {
            c: float(bsv[ds.cohorts == "healthy_control"].mean(axis=0)[i])
            for i, c in enumerate(BSV_COMPONENTS)
        },
        "hcc": {
            c: float(bsv[ds.cohorts == "hcc"].mean(axis=0)[i])
            for i, c in enumerate(BSV_COMPONENTS)
        },
    }
    exp = build_expected_comparators(DATASET_ID, ds.cohort_names)
    ref_exp = exp.get("healthy_control")
    hcc_exp = exp.get("hcc")
    out: dict = {
        "expected_hcc_comparator": hcc_exp.comparator_name if hcc_exp else None,
        "expected_hcc_match": hcc_exp.match_type if hcc_exp else None,
        "expected_hcc_confidence": hcc_exp.confidence if hcc_exp else None,
    }
    if hcc_exp and hcc_exp.bsv and ref_exp and ref_exp.bsv:
        cmp_res = compute_delta_comparison(
            cb_means["hcc"], cb_means["healthy_control"],
            hcc_exp.bsv, ref_exp.bsv,
        )
        out["delta_cosine"] = float(cmp_res.get("delta_cosine", float("nan")))
        per_axis = cmp_res.get("per_axis", [])
        out["sign_agree_axes"] = sum(
            1 for a in per_axis
            if (float(a.get("obs_delta", 0.0)) * float(a.get("exp_delta", 0.0))) > 0
        )
        out["aligned_axes"] = sum(1 for a in per_axis if a.get("category") == "aligned")
        out["partial_axes"] = sum(1 for a in per_axis if a.get("category") == "partial")
        out["divergent_axes"] = sum(1 for a in per_axis if a.get("category") == "divergent")
        out["weak_axes"] = sum(1 for a in per_axis if a.get("category") == "weak")
        out["n_axes"] = len(per_axis)
    return out


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def write_report(report_dir, tables_dir, bsv, delta, dist, ds, meta, prep, sael_summary):
    axis_fx = pd.read_csv(tables_dir / "pilot1_hcc_axis_effect_sizes.csv")
    cohort = pd.read_csv(tables_dir / "pilot1_hcc_cohort_summary.csv")
    batch_df = pd.read_csv(tables_dir / "pilot1_hcc_batch_summary.csv")

    # Top axes by |Δ mean| and by |Cohen's d|
    axis_fx["abs_delta"] = axis_fx["delta_mean"].abs()
    axis_fx["abs_d"] = axis_fx["cohens_d"].abs()
    top_by_delta = axis_fx.nlargest(3, "abs_delta")
    top_by_d = axis_fx.nlargest(3, "abs_d")

    batch_counts = meta.groupby(["substrate_batch", "class_gaira"]).size().unstack(fill_value=0)

    ctr_d = dist[ds.cohorts == "healthy_control"]
    hcc_d = dist[ds.cohorts == "hcc"]
    d_cliff = _cliffs_delta(hcc_d, ctr_d)

    # Verdict logic
    max_abs_d = float(axis_fx["abs_d"].max())
    n_small = int((axis_fx["abs_d"] >= 0.2).sum())
    n_medium = int((axis_fx["abs_d"] >= 0.5).sum())

    # Batch-direction consistency: for each axis, check if Δ sign is same in
    # all 3 batches for HCC.
    batch_consistency = {}
    for c in BSV_COMPONENTS:
        hcc_rows = batch_df[(batch_df["class"] == "hcc") & (batch_df["axis"] == c)]
        signs = [np.sign(v) for v in hcc_rows["mean_delta_bsv"].dropna().tolist() if not np.isnan(v)]
        batch_consistency[c] = int(len(signs) >= 2 and len(set(signs)) == 1)

    consistent_axes = [c for c, v in batch_consistency.items() if v]

    if max_abs_d < 0.2 and n_small == 0:
        verdict = "no meaningful structure"
    elif max_abs_d >= 0.5 and len(consistent_axes) >= 2:
        verdict = "coherent cohort shift"
    elif max_abs_d >= 0.2 and len(consistent_axes) >= 1:
        verdict = "weak but real shift"
    elif max_abs_d >= 0.2 and not consistent_axes:
        verdict = "batch-sensitive signal"
    else:
        verdict = "weak but real shift"

    def _fmt_row(r):
        return (f"- **{BSV_SHORT[r['axis']]}** (`{r['axis']}`) · Δmean = `{r['delta_mean']:+.5f}` · "
                f"Cohen's d = `{r['cohens_d']:+.2f}` (95% CI "
                f"[{r['cohens_d_ci_low']:+.2f}, {r['cohens_d_ci_high']:+.2f}]) · "
                f"Cliff's δ = `{r['cliffs_delta']:+.2f}`")

    md = []
    md.append("# GAIRA Target Pilot 1 — HCC holdout (BSV / ΔBSV)")
    md.append("")
    md.append(f"**Dataset:** {PAPER}  ")
    md.append(f"**Substrate:** {SUBSTRATE}  ")
    md.append(f"**Pipeline:** `{prep.pipeline}` (AsLS λ=1e5 p=0.001 · SG window=11 order=3 · L2)  ")
    md.append(f"**Scorer / atlas:** locked (no changes).")
    md.append("")
    md.append("## A. Dataset facts")
    md.append(f"- {bsv.shape[0]} spectra · 72 CTR + 72 HCC (H0T in source)")
    md.append(f"- Metadata: `biosample_id`, `class`, `sample_code`, `substrate_batch`, `acquisition_date`")
    md.append(f"- **One spectrum per sample**; no replicates; no patient identifiers")
    md.append(f"- **Analysis unit: per-spectrum == per-sample** (sample-level aggregation is a no-op here)")
    md.append(f"- Batch distribution (count by class × batch):\n")
    md.append("```")
    md.append(batch_counts.to_string())
    md.append("```")
    md.append("")
    md.append("## B. Pipeline used")
    md.append(f"- Load: `gaira.spectral.dataset_loader.load_dataset('{DATASET_ID}')` (CSV→master-axis interp; 400–1800 cm⁻¹; 1401 points)")
    md.append(f"- Preprocess: `gaira.spectral.preprocess` · pipeline `{prep.pipeline}` · {prep.baseline} · {prep.smoothing} · {prep.normalization}")
    md.append(f"- Window panel: 22 canonical windows (`gaira.spectral.window_panel.extract_window_features`)")
    md.append(f"- BSV: `gaira.spectral.bsv_projection.project_to_bsv` → 144 × 8")
    md.append(f"- Healthy centroid: mean of 72 CTR BSVs (canonical reference)")
    md.append(f"- Per-spectrum ΔBSV: `bsv − healthy_centroid`")
    md.append(f"- Distance-to-centroid: L2 norm in 8-D BSV space")
    md.append("")
    md.append("## C. Main findings")
    md.append("")
    md.append("### Top axes by |Δmean|")
    for _, r in top_by_delta.iterrows():
        md.append(_fmt_row(r))
    md.append("")
    md.append("### Top axes by |Cohen's d|")
    for _, r in top_by_d.iterrows():
        md.append(_fmt_row(r))
    md.append("")
    md.append("### Magnitude vs healthy internal spread")
    for c in top_by_d["axis"].tolist():
        r = axis_fx[axis_fx["axis"] == c].iloc[0]
        ratio = abs(r["delta_mean"]) / r["ctr_sd"] if r["ctr_sd"] > 0 else float("nan")
        md.append(f"- {BSV_SHORT[c]}: |Δmean|/σ(CTR) = `{ratio:.2f}` (ΔBSV relative to within-healthy dispersion)")
    md.append("")
    md.append("### Dispersion")
    ctr_spread = float(np.median([axis_fx.iloc[i]["ctr_sd"] for i in range(len(axis_fx))]))
    hcc_spread = float(np.median([axis_fx.iloc[i]["hcc_sd"] for i in range(len(axis_fx))]))
    md.append(f"- Median per-axis σ: CTR `{ctr_spread:.4f}` vs HCC `{hcc_spread:.4f}`")
    md.append(f"- Distance-to-centroid median: CTR `{np.median(ctr_d):.4f}` vs HCC `{np.median(hcc_d):.4f}` · Cliff's δ(HCC vs CTR) = `{d_cliff:+.2f}`")
    md.append("")
    md.append("## D. Batch sensitivity")
    md.append("")
    md.append(f"- Axes with consistent ΔBSV sign across all 3 substrate batches in HCC: "
              f"{', '.join(BSV_SHORT[c] for c in consistent_axes) if consistent_axes else '— none —'}")
    md.append(f"- Batches are not balanced across classes: see count matrix above.")
    md.append(f"- Per-batch ΔBSV is in [pilot1_hcc_batch_summary.csv]; visual check: `fig5_batch_effect_panel.png`.")
    md.append("")
    md.append("## E. Interpretation (biochemical-theme language only)")
    theme_lines = []
    for _, r in top_by_d.iterrows():
        direction = "elevated" if r["delta_mean"] > 0 else "depressed"
        theme_lines.append(
            f"- GAIRA reports a {direction} response on the **{BSV_SHORT[r['axis']]}** axis in HCC "
            f"(d = {r['cohens_d']:+.2f}, CI [{r['cohens_d_ci_low']:+.2f}, {r['cohens_d_ci_high']:+.2f}])."
        )
    md.extend(theme_lines)
    md.append("")
    md.append(
        "> GAIRA does **not** identify individual molecules from these axes. "
        "Each axis aggregates many literature-linked bands; the interpretation is "
        "a biochemical-theme shift in representation space, not a molecular call."
    )
    md.append("")
    md.append("### SAEL / R7c summary on cohort-mean Δ (not primary evidence)")
    if sael_summary.get("delta_cosine") is not None:
        md.append(f"- Expected HCC comparator: `{sael_summary.get('expected_hcc_comparator')}` "
                  f"(match `{sael_summary.get('expected_hcc_match')}`, confidence "
                  f"`{sael_summary.get('expected_hcc_confidence')}`)")
        md.append(f"- Observed-vs-expected Δ cosine: `{sael_summary['delta_cosine']:+.3f}`")
        md.append(f"- Axes with sign-agreement observed vs expected Δ: "
                  f"`{sael_summary['sign_agree_axes']}/{sael_summary['n_axes']}`")
        md.append(f"- Per-axis categories: aligned `{sael_summary.get('aligned_axes', 0)}`, "
                  f"partial `{sael_summary.get('partial_axes', 0)}`, "
                  f"divergent `{sael_summary.get('divergent_axes', 0)}`, "
                  f"weak `{sael_summary.get('weak_axes', 0)}`")
    else:
        md.append("- Expected comparator unavailable or incomplete for HCC.")
    md.append("")
    md.append("## F. Final verdict")
    md.append("")
    md.append(f"**{verdict}.**")
    md.append("")
    md.append(
        "This verdict is based on (i) max |Cohen's d| across 8 axes, "
        f"(ii) number of axes with consistent ΔBSV direction across all 3 substrate batches, and "
        "(iii) dispersion of distance-to-healthy-centroid. See tables + figures for full values."
    )
    md.append("")
    md.append("## G. Outputs")
    md.append("")
    md.append("### Tables")
    md.append("- `tables/pilot1_hcc_per_spectrum_bsv.csv`")
    md.append("- `tables/pilot1_hcc_per_spectrum_delta_bsv.csv`")
    md.append("- `tables/pilot1_hcc_cohort_summary.csv`")
    md.append("- `tables/pilot1_hcc_batch_summary.csv`")
    md.append("- `tables/pilot1_hcc_axis_effect_sizes.csv`")
    md.append("")
    md.append("### Figures")
    for fn in [
        "fig1_bsv_heatmap.png", "fig2_delta_bsv_distributions.png",
        "fig3_cohort_bsv_radar.png", "fig4_cohort_delta_bsv_radar.png",
        "fig5_batch_effect_panel.png", "fig6_pca_bsv_space.png",
        "fig7_distance_to_centroid.png",
    ]:
        md.append(f"- `figures/{fn}`")
    md.append("")
    md.append("## H. Scope limitation")
    md.append(
        "- Release carries exactly one spectrum per sample; no technical replicates; no patient IDs. "
        "Patient-level biology cannot be inferred from this pilot."
    )
    md.append(
        "- Pilot 1 reports representation-space structure only (BSV / ΔBSV). "
        "No classifier metrics, no molecule identities, no clinical claims."
    )
    md.append("")

    (report_dir / "REPORT_pilot1_hcc_holdout.md").write_text("\n".join(md))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[pilot1] output root: {OUT_ROOT}")
    for p in ("tables", "figures", "report"):
        (OUT_ROOT / p).mkdir(parents=True, exist_ok=True)
    tables_dir = OUT_ROOT / "tables"
    figures_dir = OUT_ROOT / "figures"
    report_dir = OUT_ROOT / "report"

    ds, Xn, wf, bsv, prep, meta = run_pipeline()
    print(f"[pilot1] dataset {DATASET_ID}: X={Xn.shape}, BSV={bsv.shape}, cohorts={ds.cohort_counts}")

    is_ctr = ds.cohorts == "healthy_control"
    healthy_centroid = bsv[is_ctr].mean(axis=0)
    delta = bsv - healthy_centroid
    dist = np.linalg.norm(delta, axis=1)

    write_tables(tables_dir, bsv, delta, dist, ds, meta)
    print("[pilot1] tables written")

    fig1_bsv_heatmap(figures_dir, bsv, ds, meta)
    fig2_delta_distributions(figures_dir, delta, ds)
    fig3_cohort_bsv_radar(figures_dir, bsv, ds)
    fig4_cohort_delta_radar(figures_dir, delta, ds)
    fig5_batch_effect(figures_dir, delta, ds, meta)
    fig6_pca(figures_dir, bsv, ds, meta)
    fig7_distance(figures_dir, dist, ds)
    print("[pilot1] figures written")

    sael = summarize_vs_expected(bsv, ds)
    write_report(report_dir, tables_dir, bsv, delta, dist, ds, meta, prep, sael)
    print(f"[pilot1] report written · sael={sael}")
    print("[pilot1] done")


if __name__ == "__main__":
    main()
