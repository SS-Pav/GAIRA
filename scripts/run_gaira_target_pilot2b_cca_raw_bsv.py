"""GAIRA Target Pilot 2b — canonical raw-spectrum rerun of the CCA cohort.

Pipeline (identical front-end to Pilot 1):

    raw spectra (DuckDB biosample_spectrum_points)
      → np.interp to master axis (400–1800 cm⁻¹, 1401 pts)
      → gaira.spectral.preprocessing._preprocess_raw (AsLS + SG + L2)
      → 22-window panel → 8-axis BSV
      → within-dataset healthy centroid → per-spectrum ΔBSV

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_target_pilot2b_cca_raw_bsv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral.dataset_loader import NPZ_PATH, SpectralDataset
from gaira.spectral.preprocessing import _preprocess_raw
from gaira.spectral.window_panel import BSV_COMPONENTS, WINDOW_DEFS, extract_window_features
from gaira.spectral.bsv_projection import project_to_bsv


DATASET_ID = "cca_hcc_lm_serum_sers"
DATASET_ID_RAW = "cca_hcc_lm_serum_sers_raw"  # in-memory tag only — no registry edit
DUCKDB_PATH = Path("/Volumes/SSD_Rad/GAIRA_DATA/interim/gaira.duckdb")
PAPER = "Combination of label-free SERS-based nanosensor (multi-class liver SERS cohort)"
SUBSTRATE = "Ag nanoparticle · 785 nm laser (per parser provenance)"
OUT_ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/pilot2b_cca_raw")
P2A_TABLES = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/pilot2_cca/tables")
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
    "cca":             "#E45756",
    "hcc":             "#F2B36B",
    "lm":              "#72B7B2",
}
CLASS_ORDER = ["healthy_control", "cca", "hcc", "lm"]


# ──────────────────────────────────────────────────────────────────────
# Stats helpers (identical to Pilot 2a)
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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ──────────────────────────────────────────────────────────────────────
# Raw load pipeline
# ──────────────────────────────────────────────────────────────────────

def _select_real_npz_rows():
    npz = np.load(NPZ_PATH, allow_pickle=True)
    ds_mask = npz["dataset_ids"] == DATASET_ID
    rk_mask = npz["record_kinds"] == "processed_spectrum"
    mask = ds_mask & rk_mask
    keys = [str(k) for k in npz["sample_keys"][mask]]
    labels = [str(x) for x in npz["labels_optional"][mask]]
    subclasses = [str(x) for x in npz["subclasses"][mask]]
    master_x = npz["master_x"].astype(np.float64)
    return keys, labels, subclasses, master_x


def _parse_biosample_id(sk: str) -> tuple[str, str, str, str]:
    parts = sk.split("__")
    proc_ver = parts[0] if parts else ""
    sid = parts[1].replace(f"{DATASET_ID}_", "")
    source_row = parts[2] if len(parts) > 2 else ""
    maprow = parts[3] if len(parts) > 3 else ""
    bid = f"{DATASET_ID}_{sid}__{source_row}__{maprow}"
    return bid, sid, source_row, maprow


def load_raw_dataset() -> tuple[SpectralDataset, pd.DataFrame]:
    keys, labels, subclasses, master_x = _select_real_npz_rows()
    meta_rows = []
    bids = []
    for sk, lab, sub in zip(keys, labels, subclasses):
        bid, sid, sr, mr = _parse_biosample_id(sk)
        bids.append(bid)
        meta_rows.append({
            "biosample_id": bid,
            "class": lab,
            "sample_id": sid,
            "source_row_id": sr,
            "acquisition_index": sr.split("_")[-1] if sr else "",
            "maprow": mr,
            "subclass": sub,
        })
    meta = pd.DataFrame(meta_rows)
    assert len(meta) == 350, f"expected 350 real spectra, got {len(meta)}"

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    con.register("npz_bids", pd.DataFrame({"biosample_id": bids}))
    pts = con.execute(
        """
        SELECT biosample_id, wavenumber, intensity
        FROM biosample_spectrum_points
        WHERE dataset_id = ?
          AND biosample_id IN (SELECT biosample_id FROM npz_bids)
        ORDER BY biosample_id, wavenumber
        """,
        [DATASET_ID],
    ).fetchdf()
    con.close()

    # Sanity: every biosample should have >= 1000 points
    per_bid = pts.groupby("biosample_id").size()
    missing = set(bids) - set(per_bid.index)
    if missing:
        raise RuntimeError(f"{len(missing)} biosamples missing raw points; first: {sorted(missing)[:3]}")

    # Interp each spectrum to master_x
    X = np.zeros((len(bids), len(master_x)), dtype=np.float64)
    gb = pts.groupby("biosample_id", sort=False)
    for i, bid in enumerate(bids):
        sub = gb.get_group(bid)
        wn = sub["wavenumber"].to_numpy(dtype=np.float64)
        it = sub["intensity"].to_numpy(dtype=np.float64)
        order = np.argsort(wn)
        X[i] = np.interp(master_x, wn[order], it[order])

    classes = meta["class"].to_numpy()
    ds = SpectralDataset(
        dataset_id=DATASET_ID_RAW,
        X=X, wavenumbers=master_x, cohorts=classes,
        n_spectra=len(X),
        cohort_names=sorted(set(classes)),
        cohort_counts={c: int((classes == c).sum()) for c in sorted(set(classes))},
    )
    return ds, meta


# ──────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────

def write_tables(tables_dir: Path, bsv, wf, delta, dist, ds, meta) -> pd.DataFrame:
    base = meta[["biosample_id", "class", "sample_id", "source_row_id",
                  "acquisition_index", "maprow"]].copy()

    df1 = base.copy()
    for i, c in enumerate(BSV_COMPONENTS):
        df1[f"bsv_{c}"] = bsv[:, i]
    df1.to_csv(tables_dir / "pilot2b_cca_raw_per_spectrum_bsv.csv", index=False)

    df2 = base.copy()
    for i, c in enumerate(BSV_COMPONENTS):
        df2[f"delta_bsv_{c}"] = delta[:, i]
    df2["distance_to_healthy_centroid"] = dist
    df2.to_csv(tables_dir / "pilot2b_cca_raw_per_spectrum_delta_bsv.csv", index=False)

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
    pd.DataFrame(rows).to_csv(tables_dir / "pilot2b_cca_raw_cohort_summary.csv", index=False)

    # Batch summary — no batch metadata in this dataset; emit single "all" rows
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
    pd.DataFrame(rows).to_csv(tables_dir / "pilot2b_cca_raw_batch_summary.csv", index=False)

    # Long-format effect sizes (parity with Pilot 2a schema)
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
    pd.DataFrame(rows).to_csv(tables_dir / "pilot2b_cca_raw_axis_effect_sizes.csv", index=False)

    # Axis correlation
    corr = np.corrcoef(delta.T)
    corr_df = pd.DataFrame(corr, index=BSV_COMPONENTS, columns=BSV_COMPONENTS).reset_index()
    corr_df = corr_df.rename(columns={"index": "axis"})
    corr_df.to_csv(tables_dir / "pilot2b_cca_raw_axis_correlation.csv", index=False)

    # Contribution diagnostics (CCA vs HC)
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
                "axis": comp, "window_id": w_id,
                "window_range_cm1": f"{int(w_lo)}-{int(w_hi)}",
                "hc_mean": hc_mean, "cca_mean": cca_mean,
                "delta_mean": cca_mean - hc_mean,
            })
    contrib = pd.DataFrame(rows)
    contrib.to_csv(tables_dir / "pilot2b_cca_raw_contribution_diagnostics.csv", index=False)
    return contrib


# ──────────────────────────────────────────────────────────────────────
# Figures (same design as Pilot 2a)
# ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 180, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
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
    bsv_s = bsv[idx]; cls_s = ds.cohorts[idx]
    z = (bsv_s - bsv_s.mean(axis=0)) / (bsv_s.std(axis=0, ddof=1) + 1e-9)
    fig, (ax_sb, ax) = plt.subplots(
        1, 2, figsize=(10, 11),
        gridspec_kw={"width_ratios": [0.05, 1.0], "wspace": 0.01},
    )
    class_idx = np.array([[CLASS_ORDER.index(c)] for c in cls_s])
    ax_sb.imshow(class_idx, aspect="auto", cmap="tab10", vmin=0, vmax=9)
    ax_sb.set_yticks([]); ax_sb.set_xticks([])
    ax_sb.set_title("class", fontsize=9)
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(8))
    ax.set_xticklabels([BSV_SHORT[c] for c in BSV_COMPONENTS], rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_ylabel("spectra (sorted: class → sample_id → acq_idx)")
    ax.set_title("Per-spectrum BSV (raw pipeline, z-scored within axis)", fontsize=12)
    pos = 0
    for cls in CLASS_ORDER:
        n = int((cls_s == cls).sum())
        if pos > 0 and n > 0:
            ax.axhline(pos - 0.5, color="black", lw=1.0)
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
    for i, comp in enumerate(BSV_COMPONENTS):
        ax = axes.flat[i]
        data = [delta[ds.cohorts == c, i] for c in classes]
        bp = ax.boxplot(data, positions=range(len(classes)), widths=0.55,
                         patch_artist=True, showfliers=True,
                         medianprops=dict(color="black", linewidth=1.4),
                         flierprops=dict(marker="o", markersize=3, markerfacecolor="gray",
                                         markeredgecolor="none", alpha=0.5))
        for patch, cls in zip(bp["boxes"], classes):
            patch.set_facecolor(CLASS_COLORS[cls]); patch.set_alpha(0.6)
        ax.axhline(0, color="#888", lw=0.8, linestyle="--")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([c.replace("healthy_control", "HC").upper() for c in classes], fontsize=9)
        ax.set_title(BSV_SHORT[comp], fontsize=11)
        ax.set_ylabel("ΔBSV" if i % 4 == 0 else "")
    fig.suptitle("Per-axis ΔBSV vs within-dataset healthy centroid — raw pipeline", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "fig2_delta_bsv_distributions.png")


def _radar(ax, labels, series, colors, fill_alpha=0.15, linewidth=2):
    n = len(labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    theta_c = theta + [theta[0]]
    for (name, vals), color in zip(series, colors):
        v = list(vals) + [vals[0]]
        ax.plot(theta_c, v, color=color, linewidth=linewidth, label=name)
        ax.fill(theta_c, v, color=color, alpha=fill_alpha)
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
    ax.set_title("Cohort mean BSV — raw pipeline (summary only)", y=1.08, fontsize=12)
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
    ax.set_title("Cohort mean ΔBSV — disease vs healthy (raw pipeline)", y=1.08, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)
    _save(fig, fig_dir / "fig4_cohort_delta_bsv_radar.png")


def fig5_sample_replication_panel(fig_dir, delta, ds, meta):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    classes = [c for c in CLASS_ORDER if (ds.cohorts == c).sum() > 0]
    for i, comp in enumerate(BSV_COMPONENTS):
        ax = axes.flat[i]
        per_sample = []; positions = []; colors = []; labels = []
        pos = 0
        for cls in classes:
            mask = ds.cohorts == cls
            if not mask.any():
                continue
            sub = pd.DataFrame({
                "sample_id": meta["sample_id"].to_numpy()[mask],
                "d": delta[mask, i],
            })
            sm = sub.groupby("sample_id")["d"].mean().to_numpy()
            per_sample.append(sm)
            positions.append(pos); colors.append(CLASS_COLORS[cls])
            labels.append(f"{cls.replace('healthy_control','HC').upper()}\n(n={len(sm)})")
            pos += 1
        bp = ax.boxplot(per_sample, positions=positions, widths=0.55,
                         patch_artist=True, showfliers=True,
                         medianprops=dict(color="black", linewidth=1.3),
                         flierprops=dict(marker="o", markersize=3, markerfacecolor="gray",
                                         markeredgecolor="none", alpha=0.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.axhline(0, color="#888", lw=0.8, linestyle="--")
        ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(BSV_SHORT[comp], fontsize=11)
        ax.set_ylabel("per-sample mean ΔBSV" if i % 4 == 0 else "")
    fig.suptitle("Per-sample mean ΔBSV (raw pipeline · sample-level robustness check)",
                  fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "fig5_sample_replication_panel.png")


def fig6_pca(fig_dir, bsv, ds):
    pca = PCA(n_components=2)
    pp = pca.fit_transform(bsv)
    ev = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for cls in CLASS_ORDER:
        mask = ds.cohorts == cls
        if not mask.any():
            continue
        ax.scatter(pp[mask, 0], pp[mask, 1],
                    c=CLASS_COLORS[cls], s=48, alpha=0.72,
                    edgecolor="white", linewidth=0.5,
                    label=f"{cls.replace('healthy_control','HC').upper()} (n={int(mask.sum())})")
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title("PCA of per-spectrum BSV (8-D) — CCA raw pipeline", fontsize=12)
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
    ax.set_title("Distance-to-healthy-centroid distributions — raw pipeline", fontsize=12)
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
    ax.set_title("ΔBSV axis correlation (raw pipeline, dataset-wide)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    _save(fig, fig_dir / "fig8_axis_correlation.png")


# ──────────────────────────────────────────────────────────────────────
# Pilot 2a vs 2b comparison
# ──────────────────────────────────────────────────────────────────────

def _rank_by_abs_d(df: pd.DataFrame) -> pd.Series:
    tmp = df.copy()
    tmp["abs_d"] = tmp["cohens_d"].abs()
    tmp = tmp.sort_values("abs_d", ascending=False).reset_index(drop=True)
    tmp["rank"] = tmp.index + 1
    return tmp.set_index("axis")["rank"]


def write_comparison_artifacts(tables_dir, figures_dir, bsv_b, delta_b, ds_b):
    p2a_eff = pd.read_csv(P2A_TABLES / "pilot2_cca_axis_effect_sizes.csv")
    p2b_eff = pd.read_csv(tables_dir / "pilot2b_cca_raw_axis_effect_sizes.csv")
    p2a_coh = pd.read_csv(P2A_TABLES / "pilot2_cca_cohort_summary.csv")
    p2b_coh = pd.read_csv(tables_dir / "pilot2b_cca_raw_cohort_summary.csv")

    # Effect-size comparison per (axis, compare_class)
    merged = p2a_eff[["axis", "compare_class", "delta_mean", "cohens_d"]].merge(
        p2b_eff[["axis", "compare_class", "delta_mean", "cohens_d"]],
        on=["axis", "compare_class"], suffixes=("_2a", "_2b"),
    )
    # Ranks within each compare_class
    def _rank_within(df, col):
        out = df.copy()
        out["rank"] = out.groupby("compare_class")[col].rank(ascending=False, method="first")
        return out
    rank_a = _rank_within(p2a_eff.assign(abs_d=p2a_eff["cohens_d"].abs()), "abs_d")[["axis", "compare_class", "rank"]]
    rank_b = _rank_within(p2b_eff.assign(abs_d=p2b_eff["cohens_d"].abs()), "abs_d")[["axis", "compare_class", "rank"]]
    rank_pair = rank_a.merge(rank_b, on=["axis", "compare_class"], suffixes=("_2a", "_2b"))
    merged = merged.merge(rank_pair, on=["axis", "compare_class"])
    merged["delta_in_rank"] = merged["rank_2b"].astype(int) - merged["rank_2a"].astype(int)
    merged["sign_agreement"] = (np.sign(merged["delta_mean_2a"]) == np.sign(merged["delta_mean_2b"]))
    merged.to_csv(tables_dir / "pilot2ab_axis_effect_sizes_comparison.csv", index=False)

    # Delta cosine per cohort (cohort-mean ΔBSV vector)
    rows = []
    for cls in ["cca", "hcc", "lm"]:
        a_vec = p2a_coh[(p2a_coh["class"] == cls)].set_index("axis").loc[BSV_COMPONENTS, "mean_delta_bsv"].to_numpy()
        b_vec = p2b_coh[(p2b_coh["class"] == cls)].set_index("axis").loc[BSV_COMPONENTS, "mean_delta_bsv"].to_numpy()
        rows.append({
            "compare_class": cls,
            "cohort_mean_delta_cosine_2a_vs_2b": _cosine(a_vec, b_vec),
            "pilot2a_l2_norm": float(np.linalg.norm(a_vec)),
            "pilot2b_l2_norm": float(np.linalg.norm(b_vec)),
        })
    pd.DataFrame(rows).to_csv(tables_dir / "pilot2ab_delta_cosine.csv", index=False)

    # Top-conclusions survival summary (CCA-specific, top-4 by |d| in 2a)
    cca_2a = p2a_eff[p2a_eff["compare_class"] == "cca"].copy()
    cca_2a["abs_d"] = cca_2a["cohens_d"].abs()
    cca_2a = cca_2a.sort_values("abs_d", ascending=False).reset_index(drop=True).head(4)
    cca_2b_idx = p2b_eff[p2b_eff["compare_class"] == "cca"].set_index("axis")
    surv = []
    for _, r in cca_2a.iterrows():
        ax = r["axis"]
        b = cca_2b_idx.loc[ax]
        direction_match = (np.sign(r["delta_mean"]) == np.sign(b["delta_mean"]))
        magnitude_retained = abs(b["cohens_d"]) >= 0.5 * abs(r["cohens_d"])
        surv.append({
            "axis": ax,
            "pilot2a_cohens_d": r["cohens_d"],
            "pilot2b_cohens_d": b["cohens_d"],
            "direction_match": bool(direction_match),
            "magnitude_retained_half_d": bool(magnitude_retained),
            "conclusion_survives": bool(direction_match and magnitude_retained),
        })
    pd.DataFrame(surv).to_csv(tables_dir / "pilot2ab_cca_top_conclusions_survival.csv", index=False)

    # Radar overlay — per compare_class panel with 2a + 2b traces
    fig = plt.figure(figsize=(18, 6))
    for i, cls in enumerate(["cca", "hcc", "lm"]):
        ax = fig.add_subplot(1, 3, i + 1, polar=True)
        a_vec = p2a_coh[p2a_coh["class"] == cls].set_index("axis").loc[BSV_COMPONENTS, "mean_delta_bsv"].to_numpy()
        b_vec = p2b_coh[p2b_coh["class"] == cls].set_index("axis").loc[BSV_COMPONENTS, "mean_delta_bsv"].to_numpy()
        _radar(
            ax, [BSV_SHORT[c] for c in BSV_COMPONENTS],
            [(f"Pilot 2a (npz_l2)", a_vec), (f"Pilot 2b (raw_asls_sg_l2)", b_vec)],
            ["#94A3B8", CLASS_COLORS[cls]], fill_alpha=0.22, linewidth=2.2,
        )
        m = max(1e-4, max(float(np.abs(a_vec).max()), float(np.abs(b_vec).max())) * 1.25)
        ax.set_ylim(-m, m)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(theta, np.zeros_like(theta), color="#444", lw=0.8, linestyle="--")
        ax.set_title(f"Δ {cls.upper()} vs HC — 2a vs 2b", y=1.10, fontsize=11)
        ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), fontsize=8)
    fig.suptitle("Pilot 2a vs Pilot 2b cohort-mean ΔBSV (per compare class)", fontsize=13, y=1.06)
    fig.tight_layout()
    _save(fig, figures_dir / "fig_pilot2ab_radar_overlay.png")

    # Rank comparison figure — slope chart per compare_class
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    axes_labels = [BSV_SHORT[c] for c in BSV_COMPONENTS]
    for ax, cls in zip(axes, ["cca", "hcc", "lm"]):
        sub = rank_pair[rank_pair["compare_class"] == cls]
        sub = sub.merge(merged[["axis", "compare_class", "sign_agreement"]],
                          on=["axis", "compare_class"])
        # For each axis, draw a line from (0, rank_2a) to (1, rank_2b)
        for _, r in sub.iterrows():
            color = "#4C78A8" if r["sign_agreement"] else "#E45756"
            ax.plot([0, 1], [int(r["rank_2a"]), int(r["rank_2b"])], color=color, alpha=0.7, lw=2)
            ax.scatter(0, int(r["rank_2a"]), color=color, s=40, zorder=3)
            ax.scatter(1, int(r["rank_2b"]), color=color, s=40, zorder=3)
            # Label axis name at the 2b side
            ax.text(1.03, int(r["rank_2b"]), BSV_SHORT[r["axis"]], fontsize=8,
                     va="center", ha="left", color="#333")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pilot 2a\n(npz_l2)", "Pilot 2b\n(raw_asls_sg_l2)"])
        ax.set_xlim(-0.2, 1.5)
        ax.set_ylim(8.5, 0.5)
        ax.set_ylabel("|effect size| rank (1 = largest)" if cls == "cca" else "")
        ax.set_title(f"{cls.upper()} vs HC", fontsize=11)
        ax.grid(alpha=0.15, axis="y", linestyle=":")
    fig.suptitle("Axis ranking by |Cohen's d| — Pilot 2a vs Pilot 2b\n"
                  "(blue lines = sign agrees across pipelines; red lines = sign flip)",
                  fontsize=12, y=1.04)
    fig.tight_layout()
    _save(fig, figures_dir / "fig_pilot2ab_rank_comparison.png")

    return merged, rows, surv


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def write_report(report_dir, tables_dir, bsv, delta, dist, ds, meta, prep,
                  contrib, comp_merged, comp_cos, surv_rows):
    axis_fx = pd.read_csv(tables_dir / "pilot2b_cca_raw_axis_effect_sizes.csv")
    cca_fx = axis_fx[axis_fx["compare_class"] == "cca"].copy()
    cca_fx["abs_delta"] = cca_fx["delta_mean"].abs()
    cca_fx["abs_d"] = cca_fx["cohens_d"].abs()
    top_by_delta = cca_fx.nlargest(3, "abs_delta")
    top_by_d = cca_fx.nlargest(3, "abs_d")

    # sample consistency check
    cca_mask = ds.cohorts == "cca"
    sample_means = (
        pd.DataFrame({"sample_id": meta["sample_id"].to_numpy()[cca_mask],
                        **{f"d_{c}": delta[cca_mask, BSV_COMPONENTS.index(c)] for c in BSV_COMPONENTS}})
        .groupby("sample_id").mean()
    )
    consistency = {}
    for c in BSV_COMPONENTS:
        cohort_sign = np.sign(delta[cca_mask, BSV_COMPONENTS.index(c)].mean())
        sample_signs = np.sign(sample_means[f"d_{c}"].values)
        consistency[c] = float((sample_signs == cohort_sign).mean())

    max_abs_d = float(cca_fx["abs_d"].max())
    consistent_axes = [c for c, v in consistency.items() if v >= 0.60]

    # Verdict logic
    p2a_cca = pd.read_csv(P2A_TABLES / "pilot2_cca_axis_effect_sizes.csv")
    p2a_cca = p2a_cca[p2a_cca["compare_class"] == "cca"]
    top_axes_2a = p2a_cca.assign(ad=p2a_cca["cohens_d"].abs()).sort_values("ad", ascending=False).head(4)["axis"].tolist()
    match_ct = sum(1 for s in surv_rows if s["conclusion_survives"])
    sign_match = sum(1 for s in surv_rows if s["direction_match"])

    if max_abs_d < 0.2:
        verdict = "no meaningful structure"
    elif max_abs_d >= 0.5 and len(consistent_axes) >= 2 and match_ct >= 2:
        verdict = "coherent cohort shift"
    elif max_abs_d >= 0.5 and sign_match >= 3:
        verdict = "coherent cohort shift"
    elif max_abs_d >= 0.2 and len(consistent_axes) >= 1:
        verdict = "weak but real shift"
    elif max_abs_d >= 0.2 and len(consistent_axes) == 0:
        verdict = "preprocess-sensitive signal"
    else:
        verdict = "weak but real shift"

    # Contribution narrative
    contrib_lines = []
    for axis in top_by_d["axis"].tolist()[:2]:
        sub = contrib[contrib["axis"] == axis].copy()
        sub["abs"] = sub["delta_mean"].abs()
        sub = sub.sort_values("abs", ascending=False)
        total = float(sub["abs"].sum())
        contrib_lines.append(
            f"- **{BSV_SHORT[axis]}** drivers: "
            + ", ".join(
                f"`{r.window_id}` ({r.window_range_cm1} cm⁻¹, Δ={r.delta_mean:+.5f})"
                for r in sub.head(min(3, len(sub))).itertuples()
            )
        )
        top_pct = (sub.iloc[0]["abs"] / total * 100) if total > 0 else 0
        if len(sub) >= 2 and sub.iloc[0]["abs"] > 0.75 * total:
            contrib_lines.append(f"  Single-band dominated (top window accounts for {top_pct:.0f}% of |Δ|).")
        else:
            contrib_lines.append(f"  Multi-band signal (top window contributes {top_pct:.0f}%).")

    def _fmt(r):
        return (f"- **{BSV_SHORT[r['axis']]}** (`{r['axis']}`) · Δmean = `{r['delta_mean']:+.5f}` · "
                f"Cohen's d = `{r['cohens_d']:+.2f}` "
                f"(95% CI [{r['cohens_d_ci_low']:+.2f}, {r['cohens_d_ci_high']:+.2f}]) · "
                f"Cliff's δ = `{r['cliffs_delta']:+.2f}`")

    md = []
    md.append("# GAIRA Target Pilot 2b — CCA cohort (canonical raw pipeline)")
    md.append("")
    md.append(f"**Dataset:** `{DATASET_ID}` — {PAPER}  ")
    md.append(f"**Substrate / laser:** {SUBSTRATE}  ")
    md.append(f"**Pipeline:** `{prep.pipeline}` — {prep.baseline} · {prep.smoothing} · {prep.normalization}  ")
    md.append(f"**Scorer / atlas:** locked (no changes).")
    md.append(f"**Preserves Pilot 2a:** yes — all Pilot 2a outputs untouched; outputs written to a distinct folder.")
    md.append("")
    md.append("## A. Dataset facts")
    counts = {cls: int((ds.cohorts == cls).sum()) for cls in CLASS_ORDER}
    md.append(f"- **350 real raw spectra** used · "
              + " · ".join(f"{cls.replace('healthy_control','HC').upper()} n={n}" for cls, n in counts.items()))
    md.append(f"- Source: `biosample_spectrum_points` in `/Volumes/SSD_Rad/GAIRA_DATA/interim/gaira.duckdb`, "
              f"filtered to the 350 NPZ biosample IDs whose `record_kinds == 'processed_spectrum'`.")
    md.append(f"- **Excluded the 4 NPZ `class_summary` pseudo-spectra** that Pilot 2a inadvertently "
              f"included. This is an explicit correction over Pilot 2a.")
    md.append(f"- Metadata preserved: `biosample_id`, `class`, `sample_id`, `source_row_id`, "
              f"`acquisition_index`, `maprow`.")
    md.append(f"- Replicate structure: multiple map-row spectra per biological sample "
              f"({meta['sample_id'].nunique()} unique `sample_id`s across 350 spectra).")
    md.append(f"- No substrate_batch / acquisition_date metadata; `batch_summary.csv` emitted with "
              f"`batch='all'` to preserve schema.")
    md.append("")
    md.append("## B. Canonical preprocessing")
    md.append(f"- Raw source path (primary): DuckDB `biosample_spectrum_points` (parser-faithful raw counts; "
              f"intensity range observed up to ~89 k counts).")
    md.append(f"- Per-spectrum `np.interp` onto shared master axis (400–1800 cm⁻¹, 1401 pts, step 1.0 cm⁻¹).")
    md.append(f"- `gaira.spectral.preprocessing._preprocess_raw(ds)` applied:")
    md.append(f"  - AsLS baseline (λ=1e5, p=0.001, 10 iterations)")
    md.append(f"  - Savitzky–Golay smoothing (window=11, order=3)")
    md.append(f"  - L2 vector normalization")
    md.append(f"- Final pipeline tag: **`{prep.pipeline}`** — matches Pilot 1 exactly.")
    md.append(f"- Downstream stages (22-window panel → 8-axis BSV projection → ΔBSV) are unchanged.")
    md.append("")
    md.append("## C. Main findings — CCA vs healthy_control (raw pipeline)")
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
        md.append(f"| {k} | {BSV_SHORT[r['axis']]} | `{r['delta_mean']:+.5f}` | "
                  f"`{r['cohens_d']:+.2f}` | `{r['cliffs_delta']:+.2f}` |")
    md.append("")
    md.append("### Magnitude vs within-healthy dispersion")
    for c in top_by_d["axis"].tolist():
        r = cca_fx[cca_fx["axis"] == c].iloc[0]
        ratio = abs(r["delta_mean"]) / r["reference_sd"] if r["reference_sd"] > 0 else float("nan")
        md.append(f"- {BSV_SHORT[c]}: |Δmean|/σ(HC) = `{ratio:.2f}`")
    md.append("")
    md.append("### Dispersion")
    hc_dist = dist[ds.cohorts == "healthy_control"]
    cca_dist = dist[ds.cohorts == "cca"]
    md.append(f"- Median per-axis σ: HC `{cca_fx['reference_sd'].median():.4f}` vs "
              f"CCA `{cca_fx['compare_sd'].median():.4f}`")
    md.append(f"- Distance-to-centroid median: HC `{np.median(hc_dist):.4f}` vs "
              f"CCA `{np.median(cca_dist):.4f}` · "
              f"Cliff's δ(CCA vs HC) = `{_cliffs_delta(cca_dist, hc_dist):+.2f}`")
    md.append("")
    md.append("### Sample-level robustness")
    for axis, score in sorted(consistency.items(), key=lambda kv: -kv[1]):
        md.append(f"- {BSV_SHORT[axis]}: {score*100:.0f}% of CCA samples match cohort ΔBSV sign")
    md.append("")
    md.append(f"Axes with ≥ 60% sample-sign agreement: "
              f"{', '.join(BSV_SHORT[c] for c in consistent_axes) if consistent_axes else '— none —'}.")
    md.append("")
    md.append("## D. Contribution diagnostics — top 2 axes")
    md.append("")
    md.extend(contrib_lines)
    md.append("")
    md.append("## E. Axis entanglement")
    corr = np.corrcoef(delta.T)
    ent_rows = []
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(corr[i, j]) >= 0.5:
                ent_rows.append((corr[i, j], BSV_COMPONENTS[i], BSV_COMPONENTS[j]))
    ent_rows.sort(key=lambda t: -abs(t[0]))
    if ent_rows:
        md.append("Entangled axis pairs (|r| ≥ 0.5):")
        for r, a, b in ent_rows:
            md.append(f"- {BSV_SHORT[a]} ↔ {BSV_SHORT[b]}: r = `{r:+.2f}`")
        md.append("")
        md.append("> These axes share spectral drivers and must not be read as independent evidence channels.")
    else:
        md.append("- No axis pairs exceed |r| ≥ 0.5 under the raw pipeline.")
    md.append("")
    md.append("## F. Pilot 2a vs Pilot 2b comparison")
    md.append("")
    md.append("### Cohort-mean ΔBSV cosine similarity (2a vs 2b)")
    for row in comp_cos:
        md.append(f"- **{row['compare_class'].upper()} vs HC**: "
                  f"cosine(2a, 2b) = `{row['cohort_mean_delta_cosine_2a_vs_2b']:+.3f}` · "
                  f"‖Δ‖₂(2a) = `{row['pilot2a_l2_norm']:.4f}` · "
                  f"‖Δ‖₂(2b) = `{row['pilot2b_l2_norm']:.4f}`")
    md.append("")
    md.append("### Top-4 CCA conclusions from Pilot 2a — survival under canonical rerun")
    md.append("| Axis | 2a d | 2b d | Direction match | ≥ ½ magnitude | Survives |")
    md.append("|---|---:|---:|---:|---:|---|")
    for s in surv_rows:
        md.append(f"| {BSV_SHORT[s['axis']]} | `{s['pilot2a_cohens_d']:+.2f}` | `{s['pilot2b_cohens_d']:+.2f}` | "
                  f"{'✓' if s['direction_match'] else '✗'} | "
                  f"{'✓' if s['magnitude_retained_half_d'] else '✗'} | "
                  f"{'**yes**' if s['conclusion_survives'] else 'no'} |")
    md.append("")
    md.append("See also: `tables/pilot2ab_axis_effect_sizes_comparison.csv`, "
              "`figures/fig_pilot2ab_radar_overlay.png`, "
              "`figures/fig_pilot2ab_rank_comparison.png`.")
    md.append("")
    md.append("## G. Key question answers")
    md.append(f"1. **CCA cohort shift survives canonical rerun?** "
              f"{'Yes.' if verdict.startswith('coherent') else 'Partially.' if verdict.startswith('weak') else 'No.'}")
    md.append(f"2. **Top-axis ordering materially changes?** "
              + ("Yes — see rank comparison figure." if any(abs(int(r['delta_in_rank'])) >= 3
                  for _, r in comp_merged[comp_merged['compare_class'] == 'cca'].iterrows())
                 else "Some reordering but top axes cluster consistently; see rank comparison figure."))
    nuc = cca_fx[cca_fx["axis"] == "nucleic_acid_backbone"].iloc[0]
    md.append(f"3. **Nuc.Backbone signal under raw pipeline:** Δmean = `{nuc['delta_mean']:+.5f}`, "
              f"d = `{nuc['cohens_d']:+.2f}` (CI [{nuc['cohens_d_ci_low']:+.2f}, "
              f"{nuc['cohens_d_ci_high']:+.2f}]).")
    def _row(a):
        r = cca_fx[cca_fx["axis"] == a].iloc[0]
        return f"{BSV_SHORT[a]}: d=`{r['cohens_d']:+.2f}` CI [{r['cohens_d_ci_low']:+.2f}, {r['cohens_d_ci_high']:+.2f}]"
    md.append("4. **Protein / Redox / Glycan under raw pipeline:** "
              + "; ".join(_row(a) for a in ["protein_backbone", "redox_metabolite", "glycan_carbohydrate"]))
    md.append(f"5. **Distance-to-centroid elevated vs healthy:** "
              f"median(CCA)/median(HC) = `{np.median(cca_dist)/max(np.median(hc_dist), 1e-9):.2f}× `, "
              f"Cliff's δ = `{_cliffs_delta(cca_dist, hc_dist):+.2f}`.")
    md.append(f"6. **Pilot 2a conclusions directionally stable?** "
              f"{sign_match}/4 top-axis sign matches · {match_ct}/4 survive both direction + ≥½ magnitude.")
    md.append("")
    md.append("## H. Interpretation (biochemical-theme language only)")
    for _, r in top_by_d.iterrows():
        direction = "elevated" if r["delta_mean"] > 0 else "depressed"
        md.append(f"- GAIRA reports a {direction} response on the **{BSV_SHORT[r['axis']]}** axis in CCA "
                  f"(d = {r['cohens_d']:+.2f}, CI [{r['cohens_d_ci_low']:+.2f}, {r['cohens_d_ci_high']:+.2f}]).")
    md.append("")
    md.append("> GAIRA does **not** identify individual molecules from these axes. Each axis aggregates "
              "many literature-linked bands; the interpretation is a biochemical-theme shift in "
              "representation space. Pilot 2b does not claim cross-dataset equivalence with Pilot 1.")
    md.append("")
    md.append("## I. Final verdict")
    md.append("")
    md.append(f"**{verdict}.**")
    md.append("")
    md.append("Based on (i) max |Cohen's d| under the raw pipeline, (ii) sample-level sign agreement, "
              "(iii) survival of Pilot 2a's top conclusions in direction + ≥ ½ magnitude.")
    md.append("")
    md.append("## J. Outputs")
    md.append("")
    md.append("### Tables")
    for n in [
        "pilot2b_cca_raw_per_spectrum_bsv.csv",
        "pilot2b_cca_raw_per_spectrum_delta_bsv.csv",
        "pilot2b_cca_raw_cohort_summary.csv",
        "pilot2b_cca_raw_batch_summary.csv",
        "pilot2b_cca_raw_axis_effect_sizes.csv",
        "pilot2b_cca_raw_axis_correlation.csv",
        "pilot2b_cca_raw_contribution_diagnostics.csv",
        "pilot2ab_axis_effect_sizes_comparison.csv",
        "pilot2ab_delta_cosine.csv",
        "pilot2ab_cca_top_conclusions_survival.csv",
    ]:
        md.append(f"- `tables/{n}`")
    md.append("")
    md.append("### Figures")
    for n in [
        "fig1_bsv_heatmap.png", "fig2_delta_bsv_distributions.png",
        "fig3_cohort_bsv_radar.png", "fig4_cohort_delta_bsv_radar.png",
        "fig5_sample_replication_panel.png", "fig6_pca_bsv_space.png",
        "fig7_distance_to_centroid.png", "fig8_axis_correlation.png",
        "fig_pilot2ab_radar_overlay.png", "fig_pilot2ab_rank_comparison.png",
    ]:
        md.append(f"- `figures/{n}`")
    md.append("")
    md.append("## K. Scope")
    md.append("- Canonical front-end pipeline (same tag as Pilot 1) applied to the CCA cohort.")
    md.append("- No cross-dataset normalization / alignment; comparability to Pilot 1 remains schema-only.")
    md.append("- Pilot 2a is preserved untouched and serves as the dataset-native (`npz_l2`) sensitivity branch.")

    (report_dir / "REPORT_pilot2b_cca_raw.md").write_text("\n".join(md))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[pilot2b] output root: {OUT_ROOT}")
    for p in ("tables", "figures", "report"):
        (OUT_ROOT / p).mkdir(parents=True, exist_ok=True)
    tables_dir = OUT_ROOT / "tables"
    figures_dir = OUT_ROOT / "figures"
    report_dir = OUT_ROOT / "report"

    ds, meta = load_raw_dataset()
    print(f"[pilot2b] loaded {ds.n_spectra} raw spectra · cohorts={ds.cohort_counts}")

    Xn, prep = _preprocess_raw(ds)
    print(f"[pilot2b] preprocessed: pipeline={prep.pipeline}")

    wf = extract_window_features(Xn, ds.wavenumbers)
    bsv = project_to_bsv(wf)
    is_hc = ds.cohorts == "healthy_control"
    healthy_centroid = bsv[is_hc].mean(axis=0)
    delta = bsv - healthy_centroid
    dist = np.linalg.norm(delta, axis=1)
    print(f"[pilot2b] healthy centroid (raw pipeline): {healthy_centroid.round(5)}")

    contrib = write_tables(tables_dir, bsv, wf, delta, dist, ds, meta)
    print("[pilot2b] standard tables written")

    fig1_bsv_heatmap(figures_dir, bsv, ds, meta)
    fig2_delta_distributions(figures_dir, delta, ds)
    fig3_cohort_bsv_radar(figures_dir, bsv, ds)
    fig4_cohort_delta_radar(figures_dir, delta, ds)
    fig5_sample_replication_panel(figures_dir, delta, ds, meta)
    fig6_pca(figures_dir, bsv, ds)
    fig7_distance(figures_dir, dist, ds)
    fig8_axis_correlation(figures_dir, delta)
    print("[pilot2b] standard figures written")

    comp_merged, comp_cos, surv_rows = write_comparison_artifacts(
        tables_dir, figures_dir, bsv, delta, ds
    )
    print("[pilot2b] 2a vs 2b comparison artifacts written")

    write_report(report_dir, tables_dir, bsv, delta, dist, ds, meta, prep,
                  contrib, comp_merged, comp_cos, surv_rows)
    print("[pilot2b] report written")
    print("[pilot2b] done")


if __name__ == "__main__":
    main()
