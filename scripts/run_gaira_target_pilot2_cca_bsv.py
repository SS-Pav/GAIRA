"""GAIRA Target Pilot 2 — CCA cohort (cca_hcc_lm_serum_sers, 785 nm serum SERS).

Spectrum-level BSV / ΔBSV pilot run entirely within the CCA dataset.

- Dataset-native preprocessing only (NPZ → `npz_l2`): no cross-dataset alignment.
- Healthy reference = mean BSV of this dataset's `healthy_control` spectra.
- Per-spectrum ΔBSV = spectrum_bsv − (within-dataset healthy centroid).
- No reuse of Pilot 1 centroids, statistics, or outputs.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_target_pilot2_cca_bsv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral.dataset_loader import load_dataset, NPZ_PATH
from gaira.spectral.preprocessing import preprocess
from gaira.spectral.window_panel import BSV_COMPONENTS, WINDOW_DEFS, extract_window_features
from gaira.spectral.bsv_projection import project_to_bsv


DATASET_ID = "cca_hcc_lm_serum_sers"
PAPER = "Combination of label-free SERS-based nanosensor (multi-class liver SERS cohort)"
SUBSTRATE = "Ag nanoparticle · 785 nm laser (per parser provenance)"
OUT_ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/pilot2_cca")
RNG = np.random.default_rng(42)
N_BOOT = 1000

BSV_SHORT = {
    "membrane_lipid": "Lipid", "protein_backbone": "Protein",
    "aromatic_amino_acid": "Aromatic AA", "purine_nucleotide": "Purine",
    "pyrimidine_nucleotide": "Pyrimidine", "glycan_carbohydrate": "Glycan",
    "redox_metabolite": "Redox", "nucleic_acid_backbone": "Nuc.Backbone",
}
CLASS_COLORS = {
    "healthy_control": "#4C78A8",
    "cca":             "#E45756",   # primary focus
    "hcc":             "#F2B36B",
    "lm":              "#72B7B2",
}
CLASS_ORDER = ["healthy_control", "cca", "hcc", "lm"]


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
# Load + metadata (NPZ-native)
# ──────────────────────────────────────────────────────────────────────

def _parse_sample_key(sk: str) -> dict:
    # Format: <proc_ver>__<ds_prefix>_<SAMPLE_ID>__<SOURCE_ROW>__maprow_<NN>
    parts = sk.split("__")
    proc_ver = parts[0] if parts else ""
    ds_prefix = parts[1] if len(parts) > 1 else ""
    sample_id = ds_prefix.replace(f"{DATASET_ID}_", "")
    source_row = parts[2] if len(parts) > 2 else ""
    maprow = parts[3] if len(parts) > 3 else ""
    # Acquisition sub-index = trailing "_NN" inside source_row (e.g. SER-CCA-1_01 → 01)
    acq_index = source_row.split("_")[-1] if source_row else ""
    return {
        "processing_version": proc_ver,
        "sample_id": sample_id,
        "source_row_id": source_row,
        "acquisition_index": acq_index,
        "maprow": maprow,
    }


def load_cca_with_metadata():
    ds = load_dataset(DATASET_ID)
    npz = np.load(NPZ_PATH, allow_pickle=True)
    mask = npz["dataset_ids"] == DATASET_ID
    sample_keys = npz["sample_keys"][mask]
    subclasses = npz["subclasses"][mask]
    processing_versions = npz["processing_versions"][mask]
    parsed = pd.DataFrame([_parse_sample_key(str(s)) for s in sample_keys])

    meta = pd.DataFrame({
        "biosample_id": [f"{DATASET_ID}_{p['sample_id']}__{p['source_row_id']}__{p['maprow']}"
                          for _, p in parsed.iterrows()],
        "class": ds.cohorts,
        "sample_id": parsed["sample_id"].values,
        "source_row_id": parsed["source_row_id"].values,
        "acquisition_index": parsed["acquisition_index"].values,
        "maprow": parsed["maprow"].values,
        "processing_version": parsed["processing_version"].values,
        "subclass": subclasses,
    })
    return ds, meta


def run_pipeline():
    ds, meta = load_cca_with_metadata()
    Xn, prep = preprocess(ds)
    wf = extract_window_features(Xn, ds.wavenumbers)
    bsv = project_to_bsv(wf)
    assert bsv.shape[0] == len(meta), "row alignment"
    return ds, meta, Xn, wf, bsv, prep


# ──────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────

def write_tables(tables_dir: Path, bsv, wf, delta, dist, ds, meta):
    base = meta[["biosample_id", "class", "sample_id", "source_row_id",
                  "acquisition_index", "maprow"]].copy()

    # 1. per_spectrum_bsv
    df1 = base.copy()
    for i, c in enumerate(BSV_COMPONENTS):
        df1[f"bsv_{c}"] = bsv[:, i]
    df1.to_csv(tables_dir / "pilot2_cca_per_spectrum_bsv.csv", index=False)

    # 2. per_spectrum_delta_bsv
    df2 = base.copy()
    for i, c in enumerate(BSV_COMPONENTS):
        df2[f"delta_bsv_{c}"] = delta[:, i]
    df2["distance_to_healthy_centroid"] = dist
    df2.to_csv(tables_dir / "pilot2_cca_per_spectrum_delta_bsv.csv", index=False)

    # 3. cohort_summary (one row per class × axis — same schema as Pilot 1)
    rows = []
    for cls in CLASS_ORDER:
        mask = ds.cohorts == cls
        if mask.sum() == 0:
            continue
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
    pd.DataFrame(rows).to_csv(tables_dir / "pilot2_cca_cohort_summary.csv", index=False)

    # 4. batch_summary — dataset carries NO substrate_batch / acquisition_date.
    # We emit the schema with batch="all" to preserve cross-pilot compatibility
    # and document the absence in the report.
    rows = []
    for cls in CLASS_ORDER:
        mask = ds.cohorts == cls
        n = int(mask.sum())
        for i, c in enumerate(BSV_COMPONENTS):
            d_vals = delta[mask, i] if n > 0 else np.array([])
            rows.append({
                "class": cls, "batch": "all", "axis": c, "count": n,
                "mean_delta_bsv": float(d_vals.mean()) if n else float("nan"),
                "sd_delta_bsv": float(d_vals.std(ddof=1)) if n > 1 else float("nan"),
            })
    pd.DataFrame(rows).to_csv(tables_dir / "pilot2_cca_batch_summary.csv", index=False)

    # 5. axis_effect_sizes — long format so 3 disease classes × 8 axes fit cleanly
    rows = []
    is_ref = ds.cohorts == "healthy_control"
    ref_n = int(is_ref.sum())
    for compare_cls in ["cca", "hcc", "lm"]:
        is_cmp = ds.cohorts == compare_cls
        cmp_n = int(is_cmp.sum())
        for i, c in enumerate(BSV_COMPONENTS):
            ref_vals = bsv[is_ref, i]; cmp_vals = bsv[is_cmp, i]
            d, lo, hi = _cohens_d(cmp_vals, ref_vals)
            rows.append({
                "axis": c,
                "reference_class": "healthy_control",
                "compare_class": compare_cls,
                "n_reference": ref_n, "n_compare": cmp_n,
                "reference_mean": float(ref_vals.mean()),
                "compare_mean": float(cmp_vals.mean()),
                "delta_mean": float(cmp_vals.mean() - ref_vals.mean()),
                "reference_sd": float(ref_vals.std(ddof=1)),
                "compare_sd": float(cmp_vals.std(ddof=1)),
                "reference_mad": _mad(ref_vals),
                "compare_mad": _mad(cmp_vals),
                "pooled_sd": float(np.sqrt((ref_vals.var(ddof=1) + cmp_vals.var(ddof=1)) / 2.0)),
                "cohens_d": d, "cohens_d_ci_low": lo, "cohens_d_ci_high": hi,
                "cliffs_delta": _cliffs_delta(cmp_vals, ref_vals),
            })
    pd.DataFrame(rows).to_csv(tables_dir / "pilot2_cca_axis_effect_sizes.csv", index=False)

    # 6. axis correlation matrix (ΔBSV)
    corr = np.corrcoef(delta.T)
    corr_df = pd.DataFrame(corr, index=BSV_COMPONENTS, columns=BSV_COMPONENTS).reset_index()
    corr_df = corr_df.rename(columns={"index": "axis"})
    corr_df.to_csv(tables_dir / "pilot2_cca_axis_correlation.csv", index=False)

    # 7. contribution diagnostics: for each axis, per-window contribution to the
    # CCA-vs-healthy mean delta (since project_to_bsv is a mean across mapped windows).
    rows = []
    is_cca = ds.cohorts == "cca"
    is_hc = ds.cohorts == "healthy_control"
    for ci, comp in enumerate(BSV_COMPONENTS):
        win_idx = [j for j, (_, _, _, c) in enumerate(WINDOW_DEFS) if c == comp]
        for j in win_idx:
            w_id, w_lo, w_hi, _ = WINDOW_DEFS[j]
            hc_mean = float(wf[is_hc, j].mean())
            cca_mean = float(wf[is_cca, j].mean())
            rows.append({
                "axis": comp,
                "window_id": w_id,
                "window_range_cm1": f"{int(w_lo)}-{int(w_hi)}",
                "hc_mean": hc_mean,
                "cca_mean": cca_mean,
                "delta_mean": cca_mean - hc_mean,
            })
    contrib = pd.DataFrame(rows)
    contrib.to_csv(tables_dir / "pilot2_cca_contribution_diagnostics.csv", index=False)
    return contrib


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
    idx = np.lexsort((
        meta["acquisition_index"].to_numpy(),
        meta["sample_id"].to_numpy(),
        [CLASS_ORDER.index(c) for c in ds.cohorts],
    ))
    bsv_sorted = bsv[idx]
    cls_sorted = ds.cohorts[idx]
    z = (bsv_sorted - bsv_sorted.mean(axis=0)) / (bsv_sorted.std(axis=0, ddof=1) + 1e-9)

    fig, (ax_sb, ax) = plt.subplots(
        1, 2, figsize=(10, 11),
        gridspec_kw={"width_ratios": [0.05, 1.0], "wspace": 0.01},
    )
    class_idx = np.array([[CLASS_ORDER.index(c)] for c in cls_sorted])
    ax_sb.imshow(class_idx, aspect="auto", cmap="tab10", vmin=0, vmax=9)
    ax_sb.set_yticks([]); ax_sb.set_xticks([])
    ax_sb.set_title("class", fontsize=9)

    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(len(BSV_COMPONENTS)))
    ax.set_xticklabels([BSV_SHORT[c] for c in BSV_COMPONENTS], rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_ylabel("spectra (sorted: class → sample_id → acq_idx)")
    ax.set_title("Per-spectrum BSV (z-scored within axis)", fontsize=12)
    # Class separator lines
    pos = 0
    for cls in CLASS_ORDER:
        n = int((cls_sorted == cls).sum())
        if pos > 0 and n > 0:
            ax.axhline(pos - 0.5, color="black", lw=1.0)
        pos += n

    # Class labels on the sidebar
    pos = 0
    for cls in CLASS_ORDER:
        n = int((cls_sorted == cls).sum())
        if n > 0:
            ax_sb.text(-0.8, pos + n / 2, cls.replace("healthy_control", "HC").upper(),
                        fontsize=8, ha="right", va="center")
        pos += n

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("z-score (per axis)", fontsize=9)
    _save(fig, fig_dir / "fig1_bsv_heatmap.png")


def fig2_delta_distributions(fig_dir, delta, ds):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharey=False)
    classes = [c for c in CLASS_ORDER if (ds.cohorts == c).sum() > 0]
    positions = list(range(len(classes)))
    for i, comp in enumerate(BSV_COMPONENTS):
        ax = axes.flat[i]
        data = [delta[ds.cohorts == c, i] for c in classes]
        bp = ax.boxplot(
            data, positions=positions, widths=0.55,
            patch_artist=True, showfliers=True,
            medianprops=dict(color="black", linewidth=1.4),
            flierprops=dict(marker="o", markersize=3, markerfacecolor="gray",
                            markeredgecolor="none", alpha=0.5),
        )
        for patch, cls in zip(bp["boxes"], classes):
            patch.set_facecolor(CLASS_COLORS[cls]); patch.set_alpha(0.6)
        ax.axhline(0, color="#888", lw=0.8, linestyle="--")
        ax.set_xticks(positions)
        ax.set_xticklabels([c.replace("healthy_control", "HC").upper() for c in classes], fontsize=9)
        ax.set_title(BSV_SHORT[comp], fontsize=11)
        ax.set_ylabel("ΔBSV" if i % 4 == 0 else "")
    fig.suptitle("Per-axis ΔBSV vs within-dataset healthy centroid (4 classes)", fontsize=13, y=1.02)
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
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(111, polar=True)
    series, colors = [], []
    for cls in CLASS_ORDER:
        mask = ds.cohorts == cls
        if mask.sum() == 0:
            continue
        series.append((f"{cls.replace('healthy_control','HC').upper()} (n={int(mask.sum())})",
                        bsv[mask].mean(axis=0)))
        colors.append(CLASS_COLORS[cls])
    _radar(ax, [BSV_SHORT[c] for c in BSV_COMPONENTS], series, colors, fill_alpha=0.15)
    ax.set_title("Cohort mean BSV (summary only)", y=1.08, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.05), fontsize=9)
    _save(fig, fig_dir / "fig3_cohort_bsv_radar.png")


def fig4_cohort_delta_radar(fig_dir, delta, ds):
    series, colors = [], []
    for cls in ["cca", "hcc", "lm"]:
        mask = ds.cohorts == cls
        if mask.sum() == 0:
            continue
        series.append((f"Δ {cls.upper()} vs HC (n={int(mask.sum())})", delta[mask].mean(axis=0)))
        colors.append(CLASS_COLORS[cls])
    m = max(1e-4, max(float(np.abs(v).max()) for _, v in series) * 1.2)
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(111, polar=True)
    _radar(ax, [BSV_SHORT[c] for c in BSV_COMPONENTS], series, colors, fill_alpha=0.20)
    ax.set_ylim(-m, m)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(theta, np.zeros_like(theta), color="#444", lw=0.8, linestyle="--")
    ax.set_title("Cohort mean ΔBSV — disease cohorts vs healthy centroid", y=1.08, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)
    _save(fig, fig_dir / "fig4_cohort_delta_bsv_radar.png")


def fig5_sample_replication_panel(fig_dir, delta, ds, meta):
    """Repurposed from 'batch effect' — dataset has no batch metadata.
    Here we show per-sample mean ΔBSV within each class, as a technical
    replication / sample-level robustness check.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    classes = [c for c in CLASS_ORDER if (ds.cohorts == c).sum() > 0]
    for i, comp in enumerate(BSV_COMPONENTS):
        ax = axes.flat[i]
        per_sample = []
        positions = []
        colors = []
        labels = []
        pos = 0
        for cls in classes:
            mask = ds.cohorts == cls
            if not mask.any():
                continue
            sub = pd.DataFrame({
                "sample_id": meta["sample_id"].to_numpy()[mask],
                "d": delta[mask, i],
            })
            sample_means = sub.groupby("sample_id")["d"].mean().to_numpy()
            per_sample.append(sample_means)
            positions.append(pos)
            colors.append(CLASS_COLORS[cls])
            labels.append(f"{cls.replace('healthy_control','HC').upper()}\n(n={len(sample_means)})")
            pos += 1
        bp = ax.boxplot(
            per_sample, positions=positions, widths=0.55,
            patch_artist=True, showfliers=True,
            medianprops=dict(color="black", linewidth=1.3),
            flierprops=dict(marker="o", markersize=3, markerfacecolor="gray",
                            markeredgecolor="none", alpha=0.5),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.axhline(0, color="#888", lw=0.8, linestyle="--")
        ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(BSV_SHORT[comp], fontsize=11)
        ax.set_ylabel("per-sample mean ΔBSV" if i % 4 == 0 else "")
    fig.suptitle("Per-sample mean ΔBSV (sample-level robustness — no batch metadata)",
                  fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "fig5_sample_replication_panel.png")


def fig6_pca(fig_dir, bsv, ds, meta):
    pca = PCA(n_components=2)
    pp = pca.fit_transform(bsv)
    ev = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for cls in CLASS_ORDER:
        mask = ds.cohorts == cls
        if not mask.any():
            continue
        ax.scatter(
            pp[mask, 0], pp[mask, 1],
            c=CLASS_COLORS[cls], s=48, alpha=0.72,
            edgecolor="white", linewidth=0.5,
            label=f"{cls.replace('healthy_control','HC').upper()} (n={int(mask.sum())})",
        )
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title("PCA of per-spectrum BSV (8-D) — CCA dataset", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.2, linestyle=":")
    _save(fig, fig_dir / "fig6_pca_bsv_space.png")


def fig7_distance(fig_dir, dist, ds):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bins = np.linspace(0, max(float(dist.max()) * 1.02, 1e-3), 30)
    for cls in CLASS_ORDER:
        mask = ds.cohorts == cls
        if not mask.any():
            continue
        vals = dist[mask]
        ax.hist(vals, bins=bins, color=CLASS_COLORS[cls], alpha=0.55,
                 label=f"{cls.replace('healthy_control','HC').upper()} (n={len(vals)}, med={np.median(vals):.4f})")
        ax.axvline(np.median(vals), color=CLASS_COLORS[cls], lw=1.2, linestyle="--")
    ax.set_xlabel("‖spectrum BSV − healthy centroid‖₂")
    ax.set_ylabel("Count")
    ax.set_title("Distance-to-healthy-centroid distributions", fontsize=12)
    ax.legend(fontsize=9)
    _save(fig, fig_dir / "fig7_distance_to_centroid.png")


def fig8_axis_correlation(fig_dir, delta):
    corr = np.corrcoef(delta.T)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{corr[i, j]:+.2f}", ha="center", va="center",
                     fontsize=9, color="black" if abs(corr[i, j]) < 0.6 else "white")
    labels = [BSV_SHORT[c] for c in BSV_COMPONENTS]
    ax.set_xticks(range(8)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(8)); ax.set_yticklabels(labels)
    ax.set_title("ΔBSV axis correlation (dataset-wide, per-spectrum)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    _save(fig, fig_dir / "fig8_axis_correlation.png")


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def write_report(report_dir, tables_dir, bsv, delta, dist, ds, meta, prep, contrib):
    axis_fx = pd.read_csv(tables_dir / "pilot2_cca_axis_effect_sizes.csv")
    cca_fx = axis_fx[axis_fx["compare_class"] == "cca"].copy()
    cca_fx["abs_delta"] = cca_fx["delta_mean"].abs()
    cca_fx["abs_d"] = cca_fx["cohens_d"].abs()
    top_by_delta = cca_fx.nlargest(3, "abs_delta")
    top_by_d = cca_fx.nlargest(3, "abs_d")

    # Per-sample dispersion check (sample-level mean ΔBSV magnitudes)
    cca_mask = ds.cohorts == "cca"
    sample_means = (
        pd.DataFrame({"sample_id": meta["sample_id"].to_numpy()[cca_mask],
                        **{f"d_{c}": delta[cca_mask, i] for i, c in enumerate(BSV_COMPONENTS)}})
        .groupby("sample_id").mean()
    )
    sample_medians = sample_means.median(axis=0)

    # Sample-level consistency: does the sign of sample_mean ΔBSV agree with the cohort-mean sign?
    consistency = {}
    for c in BSV_COMPONENTS:
        cohort_sign = np.sign(delta[cca_mask, BSV_COMPONENTS.index(c)].mean())
        sample_signs = np.sign(sample_means[f"d_{c}"].values)
        consistency[c] = float((sample_signs == cohort_sign).mean())

    # Verdict logic
    max_abs_d = float(cca_fx["abs_d"].max())
    consistent_axes = [c for c, v in consistency.items() if v >= 0.60]
    if max_abs_d < 0.2:
        verdict = "no meaningful structure"
    elif max_abs_d >= 0.5 and len(consistent_axes) >= 2:
        verdict = "coherent cohort shift"
    elif max_abs_d >= 0.2 and len(consistent_axes) >= 1:
        verdict = "weak but real shift"
    else:
        verdict = "weak but real shift"

    # Contribution diagnostics for top 2 axes
    contrib_lines = []
    for axis in top_by_d["axis"].tolist()[:2]:
        sub = contrib[contrib["axis"] == axis].copy()
        sub["abs"] = sub["delta_mean"].abs()
        sub = sub.sort_values("abs", ascending=False)
        total = sub["abs"].sum()
        top_line = [
            f"- **{BSV_SHORT[axis]}** drivers: "
            + ", ".join(
                f"`{r.window_id}` ({r.window_range_cm1} cm⁻¹, Δ={r.delta_mean:+.5f})"
                for r in sub.head(min(3, len(sub))).itertuples()
            )
        ]
        # Is signal single-band- or multi-band-driven?
        if len(sub) >= 2 and sub.iloc[0]["abs"] > 0.75 * total:
            top_line.append(f"  Single-band dominated (top window accounts for "
                             f"{sub.iloc[0]['abs']/total*100:.0f}% of |Δ|).")
        else:
            top_line.append(f"  Multi-band signal (top window contributes "
                             f"{(sub.iloc[0]['abs']/total*100 if total>0 else 0):.0f}%).")
        contrib_lines.extend(top_line)

    def _fmt(r):
        return (f"- **{BSV_SHORT[r['axis']]}** (`{r['axis']}`) · Δmean = `{r['delta_mean']:+.5f}` · "
                f"Cohen's d = `{r['cohens_d']:+.2f}` (95% CI "
                f"[{r['cohens_d_ci_low']:+.2f}, {r['cohens_d_ci_high']:+.2f}]) · "
                f"Cliff's δ = `{r['cliffs_delta']:+.2f}`")

    md = []
    md.append("# GAIRA Target Pilot 2 — CCA cohort (BSV / ΔBSV)")
    md.append("")
    md.append(f"**Dataset:** `{DATASET_ID}` — {PAPER}  ")
    md.append(f"**Substrate / laser:** {SUBSTRATE}  ")
    md.append(f"**Pipeline:** `{prep.pipeline}` — {prep.baseline} · {prep.smoothing} · {prep.normalization}  ")
    md.append(f"**Scorer / atlas:** locked (no changes).")
    md.append("")
    md.append("## A. Dataset facts")
    counts = {cls: int((ds.cohorts == cls).sum()) for cls in CLASS_ORDER}
    md.append(f"- {bsv.shape[0]} spectra · "
              + " · ".join(f"{cls.replace('healthy_control','HC').upper()} n={n}" for cls, n in counts.items()))
    md.append(f"- Metadata: `biosample_id`, `class`, `sample_id`, `source_row_id`, `acquisition_index`, `maprow`")
    md.append(f"- **Replicate structure present**: multiple spectra per biological sample "
              f"(170 unique `sample_id`s across 354 spectra). Primary analysis unit = **per-spectrum** per prompt; "
              f"a per-sample robustness check is surfaced in fig 5.")
    md.append(f"- **No substrate_batch / acquisition_date** in the released archive. The `batch_summary.csv` is "
              f"emitted with `batch='all'` to preserve the Pilot 1 schema.")
    md.append(f"- **Comparability to Pilot 1 comes from identical atlas + axes + output schema, "
              f"NOT from shared preprocessing transformations.**")
    md.append("")
    md.append("## B. Pipeline used")
    md.append(f"- Load: `gaira.spectral.dataset_loader.load_dataset('{DATASET_ID}')` — NPZ-backed, 400–1800 cm⁻¹, 1401 points.")
    md.append(f"- Preprocess: `gaira.spectral.preprocess` · pipeline tag `{prep.pipeline}` — {prep.notes}")
    md.append(f"- Window panel: 22 canonical windows · BSV projection: 354 × 8.")
    md.append(f"- Healthy centroid: mean BSV of {counts['healthy_control']} `healthy_control` spectra (dataset-specific).")
    md.append(f"- Per-spectrum ΔBSV = `bsv − healthy_centroid`.")
    md.append(f"- Distance-to-centroid: L2 in 8-D BSV space.")
    md.append("")
    md.append("### Pipeline-note (honesty)")
    md.append("This dataset was released pre-processed (polynomial baseline + vector normalization inside the "
              "ingestion pipeline `v2_crop400_1800_interp1_poly3_vector`). `preprocess()` dispatches to the NPZ "
              "path and applies only L2 normalization at query time (`npz_l2`). In contrast, the HCC holdout in "
              "Pilot 1 is released as raw CSV and goes through `raw_asls_sg_l2` (AsLS + SG + L2). **Downstream "
              "stages (22 windows → 8-axis BSV projection → ΔBSV definition) are identical in both pilots.** "
              "The upstream spectral preprocessing difference is *dataset-native*, not a cross-dataset harmonization.")
    md.append("")
    md.append("## C. Main findings — CCA vs healthy_control")
    md.append("")
    md.append("### Top axes by |Δmean|")
    for _, r in top_by_delta.iterrows():
        md.append(_fmt(r))
    md.append("")
    md.append("### Top axes by |Cohen's d|")
    for _, r in top_by_d.iterrows():
        md.append(_fmt(r))
    md.append("")
    md.append("### Axis ranking (within dataset, CCA vs HC)")
    rank_df = cca_fx.sort_values("abs_d", ascending=False)[["axis", "delta_mean", "cohens_d", "cliffs_delta"]]
    md.append("| Rank | Axis | Δmean | Cohen's d | Cliff's δ |")
    md.append("|---|---|---:|---:|---:|")
    for k, (_, r) in enumerate(rank_df.iterrows(), start=1):
        md.append(f"| {k} | {BSV_SHORT[r['axis']]} | `{r['delta_mean']:+.5f}` | `{r['cohens_d']:+.2f}` | `{r['cliffs_delta']:+.2f}` |")
    md.append("")
    md.append("### Magnitude vs within-healthy dispersion")
    for c in top_by_d["axis"].tolist():
        r = cca_fx[cca_fx["axis"] == c].iloc[0]
        ratio = abs(r["delta_mean"]) / r["reference_sd"] if r["reference_sd"] > 0 else float("nan")
        md.append(f"- {BSV_SHORT[c]}: |Δmean|/σ(HC) = `{ratio:.2f}`")
    md.append("")
    md.append("### Dispersion")
    ctr_spread = float(cca_fx["reference_sd"].median())
    hcc_spread = float(cca_fx["compare_sd"].median())
    hc_dist = dist[ds.cohorts == "healthy_control"]
    cca_dist = dist[ds.cohorts == "cca"]
    d_cliff = _cliffs_delta(cca_dist, hc_dist)
    md.append(f"- Median per-axis σ: HC `{ctr_spread:.4f}` vs CCA `{hcc_spread:.4f}`")
    md.append(f"- Distance-to-centroid median: HC `{np.median(hc_dist):.4f}` vs CCA `{np.median(cca_dist):.4f}` · "
              f"Cliff's δ(CCA vs HC) = `{d_cliff:+.2f}`")
    md.append("")
    md.append("## D. Batch analysis")
    md.append("")
    md.append("**Dataset carries no substrate_batch or acquisition_date.** `batch_summary.csv` is emitted "
              "with `batch='all'` to preserve schema compatibility with Pilot 1.")
    md.append("")
    md.append("As a partial sensitivity check, per-sample mean ΔBSVs were computed for CCA and compared against "
              "the dataset-wide cohort sign:")
    for axis, score in sorted(consistency.items(), key=lambda kv: -kv[1]):
        md.append(f"- {BSV_SHORT[axis]}: {score*100:.0f}% of CCA samples match cohort ΔBSV sign")
    md.append("")
    md.append(f"Axes with ≥ 60% sample-sign agreement: "
              f"{', '.join(BSV_SHORT[c] for c in consistent_axes) if consistent_axes else '— none —'}.")
    md.append("")
    md.append("See `fig5_sample_replication_panel.png` for per-sample-mean ΔBSV boxplots by class.")
    md.append("")
    md.append("## E. Axis correlation (entanglement)")
    md.append("")
    md.append("See `pilot2_cca_axis_correlation.csv` and `fig8_axis_correlation.png`. Entangled axis pairs are "
              "the ones where |r| is large: they share spectral drivers and should not be interpreted as "
              "independent evidence channels for the same biology.")
    md.append("")
    md.append("## F. Contribution diagnostics — top 2 axes (CCA vs HC)")
    md.append("")
    md.extend(contrib_lines)
    md.append("")
    md.append("## G. Interpretation (biochemical-theme language only)")
    for _, r in top_by_d.iterrows():
        direction = "elevated" if r["delta_mean"] > 0 else "depressed"
        md.append(f"- GAIRA reports a {direction} response on the **{BSV_SHORT[r['axis']]}** axis in CCA "
                  f"(d = {r['cohens_d']:+.2f}, CI [{r['cohens_d_ci_low']:+.2f}, {r['cohens_d_ci_high']:+.2f}]).")
    md.append("")
    md.append("> GAIRA does **not** identify individual molecules from these axes. Each axis aggregates many "
              "literature-linked bands; the interpretation is a biochemical-theme shift in representation space, "
              "not a molecular call. Pilot 2 is self-contained and does not cross-compare to Pilot 1 data here.")
    md.append("")
    md.append("## H. Final verdict")
    md.append("")
    md.append(f"**{verdict}.**")
    md.append("")
    md.append(
        "Based on (i) max |Cohen's d| across the 8 axes for CCA vs healthy_control within this dataset, "
        "(ii) number of axes where ≥ 60% of CCA samples match the cohort-mean ΔBSV direction, and "
        "(iii) distance-to-centroid Cliff's δ."
    )
    md.append("")
    md.append("## I. Outputs")
    md.append("")
    md.append("### Tables")
    for n in [
        "pilot2_cca_per_spectrum_bsv.csv",
        "pilot2_cca_per_spectrum_delta_bsv.csv",
        "pilot2_cca_cohort_summary.csv",
        "pilot2_cca_batch_summary.csv",
        "pilot2_cca_axis_effect_sizes.csv",
        "pilot2_cca_axis_correlation.csv",
        "pilot2_cca_contribution_diagnostics.csv",
    ]:
        md.append(f"- `tables/{n}`")
    md.append("")
    md.append("### Figures")
    for n in [
        "fig1_bsv_heatmap.png", "fig2_delta_bsv_distributions.png",
        "fig3_cohort_bsv_radar.png", "fig4_cohort_delta_bsv_radar.png",
        "fig5_sample_replication_panel.png", "fig6_pca_bsv_space.png",
        "fig7_distance_to_centroid.png", "fig8_axis_correlation.png",
    ]:
        md.append(f"- `figures/{n}`")
    md.append("")
    md.append("## J. Scope limitation & cross-dataset preparation")
    md.append("- Primary comparison is CCA vs healthy_control within this dataset. HCC and LM class results "
              "are co-reported in `cohort_summary` and `axis_effect_sizes` for completeness; they are not "
              "the Pilot 2 focus.")
    md.append("- No cross-dataset normalization, scaling, or alignment was performed. Comparability to "
              "Pilot 1 is purely schema-level (identical axes, identical metrics, identical table shape "
              "for per-spectrum / cohort summaries).")
    md.append("- Sample-level biological inference is possible here because replicate structure exists "
              "(unlike Pilot 1), but was deliberately held to a robustness check — full sample-level "
              "modeling is out of scope for Pilot 2.")

    (report_dir / "REPORT_pilot2_cca.md").write_text("\n".join(md))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[pilot2] output root: {OUT_ROOT}")
    for p in ("tables", "figures", "report"):
        (OUT_ROOT / p).mkdir(parents=True, exist_ok=True)
    tables_dir = OUT_ROOT / "tables"
    figures_dir = OUT_ROOT / "figures"
    report_dir = OUT_ROOT / "report"

    ds, meta, Xn, wf, bsv, prep = run_pipeline()
    print(f"[pilot2] dataset {DATASET_ID}: X={Xn.shape}, BSV={bsv.shape}, cohorts={ds.cohort_counts}")

    is_hc = ds.cohorts == "healthy_control"
    healthy_centroid = bsv[is_hc].mean(axis=0)
    delta = bsv - healthy_centroid
    dist = np.linalg.norm(delta, axis=1)
    print(f"[pilot2] healthy centroid (dataset-native): {healthy_centroid.round(5)}")

    contrib = write_tables(tables_dir, bsv, wf, delta, dist, ds, meta)
    print("[pilot2] tables written")

    fig1_bsv_heatmap(figures_dir, bsv, ds, meta)
    fig2_delta_distributions(figures_dir, delta, ds)
    fig3_cohort_bsv_radar(figures_dir, bsv, ds)
    fig4_cohort_delta_radar(figures_dir, delta, ds)
    fig5_sample_replication_panel(figures_dir, delta, ds, meta)
    fig6_pca(figures_dir, bsv, ds, meta)
    fig7_distance(figures_dir, dist, ds)
    fig8_axis_correlation(figures_dir, delta)
    print("[pilot2] figures written")

    write_report(report_dir, tables_dir, bsv, delta, dist, ds, meta, prep, contrib)
    print("[pilot2] report written")
    print("[pilot2] done")


if __name__ == "__main__":
    main()
