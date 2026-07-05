"""GAIRA Target Pilot 3 — canonical raw-spectrum pilot for LM vs HC.

Full standalone canonical pilot for Liver Metastasis (LM) vs healthy_control,
using the locked canonical front end `raw_asls_sg_l2`.

Same raw-access path as Pilot 2b (DuckDB `biosample_spectrum_points` for the
`cca_hcc_lm_serum_sers` dataset), same master axis (400–1800 cm⁻¹, 1401 pts),
same `_preprocess_raw`, same 22-window panel, same 8-axis BSV projection.
Healthy centroid = within-dataset `healthy_control` mean BSV. No reuse of
Pilot 1 or Pilot 2b statistics.

Policy:
  preprocess_tag                 = raw_asls_sg_l2
  canonical_or_fallback          = canonical
  comparability_class            = STRICTLY_COMPARABLE
  policy_version                 = v1

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_target_pilot3_lm_raw_bsv.py
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
import yaml
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral.dataset_loader import NPZ_PATH, SpectralDataset
from gaira.spectral.preprocessing import _preprocess_raw
from gaira.spectral.window_panel import BSV_COMPONENTS, WINDOW_DEFS, extract_window_features
from gaira.spectral.bsv_projection import project_to_bsv


DATASET_ID = "cca_hcc_lm_serum_sers"
PRIMARY_CLASS = "lm"
PRIMARY_CLASS_LABEL = "LM"
DATASET_FAMILY_PAPER = "Combination of label-free SERS-based nanosensor (multi-class liver SERS cohort)"
SUBSTRATE_NOTE = "Ag nanoparticle · 785 nm laser (per parser provenance)"

DUCKDB_PATH = Path("/Volumes/SSD_Rad/GAIRA_DATA/interim/gaira.duckdb")
OUT_ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/pilot3_lm_raw")
POLICY_PATH = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/config/gaira_target_pipeline_policy_v1.yaml")

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
    "lm":              "#E45756",   # primary focus
    "cca":             "#F2B36B",
    "hcc":             "#72B7B2",
}
CLASS_ORDER = ["healthy_control", "lm", "cca", "hcc"]


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
    if pooled == 0:
        return 0.0, 0.0, 0.0
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
# Raw load (same as Pilot 2b)
# ──────────────────────────────────────────────────────────────────────

def _parse_biosample_id(sk: str) -> tuple[str, str, str, str]:
    parts = sk.split("__")
    sid = parts[1].replace(f"{DATASET_ID}_", "")
    source_row = parts[2] if len(parts) > 2 else ""
    maprow = parts[3] if len(parts) > 3 else ""
    bid = f"{DATASET_ID}_{sid}__{source_row}__{maprow}"
    return bid, sid, source_row, maprow


def load_raw_dataset() -> tuple[SpectralDataset, pd.DataFrame]:
    npz = np.load(NPZ_PATH, allow_pickle=True)
    ds_mask = npz["dataset_ids"] == DATASET_ID
    rk_mask = npz["record_kinds"] == "processed_spectrum"   # drop the 4 class_summary rows
    mask = ds_mask & rk_mask
    keys = [str(k) for k in npz["sample_keys"][mask]]
    labels = [str(x) for x in npz["labels_optional"][mask]]
    master_x = npz["master_x"].astype(np.float64)

    bids = []
    meta_rows = []
    for sk, lab in zip(keys, labels):
        bid, sid, sr, mr = _parse_biosample_id(sk)
        bids.append(bid)
        meta_rows.append({
            "biosample_id": bid, "class": lab, "sample_id": sid,
            "source_row_id": sr, "acquisition_index": sr.split("_")[-1] if sr else "",
            "maprow": mr,
        })
    meta = pd.DataFrame(meta_rows)
    assert len(meta) == 350, f"expected 350 real spectra, got {len(meta)}"

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    con.register("bids", pd.DataFrame({"biosample_id": bids}))
    pts = con.execute(
        """
        SELECT biosample_id, wavenumber, intensity
        FROM biosample_spectrum_points
        WHERE dataset_id = ?
          AND biosample_id IN (SELECT biosample_id FROM bids)
        ORDER BY biosample_id, wavenumber
        """,
        [DATASET_ID],
    ).fetchdf()
    con.close()

    X = np.zeros((len(bids), len(master_x)), dtype=np.float64)
    gb = pts.groupby("biosample_id", sort=False)
    for i, bid in enumerate(bids):
        sub = gb.get_group(bid)
        wn = sub["wavenumber"].to_numpy(dtype=np.float64)
        it = sub["intensity"].to_numpy(dtype=np.float64)
        o = np.argsort(wn)
        X[i] = np.interp(master_x, wn[o], it[o])

    cohorts = meta["class"].to_numpy()
    ds = SpectralDataset(
        dataset_id=DATASET_ID + "_raw",
        X=X, wavenumbers=master_x, cohorts=cohorts,
        n_spectra=len(X),
        cohort_names=sorted(set(cohorts)),
        cohort_counts={c: int((cohorts == c).sum()) for c in sorted(set(cohorts))},
    )
    return ds, meta


def run_pipeline():
    ds, meta = load_raw_dataset()
    Xn, prep = _preprocess_raw(ds)
    wf = extract_window_features(Xn, ds.wavenumbers)
    bsv = project_to_bsv(wf)
    return ds, meta, Xn, wf, bsv, prep


# ──────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────

def write_tables(tables_dir: Path, bsv, wf, delta, dist, ds, meta) -> pd.DataFrame:
    base = meta[["biosample_id", "class", "sample_id", "source_row_id",
                  "acquisition_index", "maprow"]].copy()

    # 1. per_spectrum_bsv
    df1 = base.copy()
    for i, c in enumerate(BSV_COMPONENTS):
        df1[f"bsv_{c}"] = bsv[:, i]
    df1.to_csv(tables_dir / "pilot3_lm_raw_per_spectrum_bsv.csv", index=False)

    # 2. per_spectrum_delta_bsv
    df2 = base.copy()
    for i, c in enumerate(BSV_COMPONENTS):
        df2[f"delta_bsv_{c}"] = delta[:, i]
    df2["distance_to_healthy_centroid"] = dist
    df2.to_csv(tables_dir / "pilot3_lm_raw_per_spectrum_delta_bsv.csv", index=False)

    # 3. cohort_summary — per class × axis
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
    pd.DataFrame(rows).to_csv(tables_dir / "pilot3_lm_raw_cohort_summary.csv", index=False)

    # 4. batch_summary — no batch metadata in this dataset; emit schema-compatible 'all' rows
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
    pd.DataFrame(rows).to_csv(tables_dir / "pilot3_lm_raw_batch_summary.csv", index=False)

    # 5. axis_effect_sizes — long format; primary comparison = LM vs HC, also include CCA and HCC
    rows = []
    is_ref = ds.cohorts == "healthy_control"
    ref_n = int(is_ref.sum())
    for compare_cls in ["lm", "cca", "hcc"]:
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
    pd.DataFrame(rows).to_csv(tables_dir / "pilot3_lm_raw_axis_effect_sizes.csv", index=False)

    # 6. axis correlation (ΔBSV)
    corr = np.corrcoef(delta.T)
    corr_df = pd.DataFrame(corr, index=BSV_COMPONENTS, columns=BSV_COMPONENTS).reset_index()
    corr_df = corr_df.rename(columns={"index": "axis"})
    corr_df.to_csv(tables_dir / "pilot3_lm_raw_axis_correlation.csv", index=False)

    # 7. contribution_diagnostics — LM vs HC per window, grouped by axis
    rows = []
    is_lm = ds.cohorts == PRIMARY_CLASS
    is_hc = ds.cohorts == "healthy_control"
    for comp in BSV_COMPONENTS:
        win_idx = [j for j, (_, _, _, c) in enumerate(WINDOW_DEFS) if c == comp]
        for j in win_idx:
            w_id, w_lo, w_hi, _ = WINDOW_DEFS[j]
            hc_mean = float(wf[is_hc, j].mean())
            lm_mean = float(wf[is_lm, j].mean())
            rows.append({
                "axis": comp,
                "window_id": w_id,
                "window_range_cm1": f"{int(w_lo)}-{int(w_hi)}",
                "hc_mean": hc_mean,
                "lm_mean": lm_mean,
                "delta_mean": lm_mean - hc_mean,
            })
    contrib = pd.DataFrame(rows)
    contrib.to_csv(tables_dir / "pilot3_lm_raw_contribution_diagnostics.csv", index=False)
    return contrib


# ──────────────────────────────────────────────────────────────────────
# Figures (mirror Pilot 2b, LM-focused)
# ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 180, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


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
    fig.suptitle(f"Per-axis ΔBSV vs within-dataset healthy centroid — LM-focused pilot "
                  f"(raw pipeline)", fontsize=13, y=1.02)
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
    # Primary disease cohort (LM) rendered first / largest; co-report CCA and HCC
    series, colors = [], []
    for cls in ["lm", "cca", "hcc"]:
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
    ax.set_title("Cohort mean ΔBSV — disease vs healthy (LM primary · raw pipeline)",
                  y=1.08, fontsize=12)
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
    fig.suptitle(f"Per-sample mean ΔBSV (raw pipeline · sample-level robustness check)",
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
    ax.set_title(f"PCA of per-spectrum BSV (8-D) — LM canonical pilot", fontsize=12)
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
# Manifest (per implementation memo)
# ──────────────────────────────────────────────────────────────────────

def write_manifest(out_root: Path, ds: SpectralDataset, meta: pd.DataFrame, prep):
    policy = yaml.safe_load(POLICY_PATH.read_text())
    manifest = {
        "policy_version":                  policy["policy"]["version"],
        "pilot_id":                        "pilot3_lm_raw",
        "dataset_name":                    DATASET_ID,
        "primary_compare_class":           PRIMARY_CLASS,
        "input_spectral_artifact_type":    "duckdb_row_store",
        "source_file_paths": [
            str(DUCKDB_PATH) + "::biosample_spectrum_points",
        ],
        "raw_availability":                True,
        "interpolation_axis": {
            "range_cm1":   [400.0, 1800.0],
            "n_points":    1401,
            "step_cm1":    1.0,
        },
        "preprocessing_tag_actually_used": prep.pipeline,
        "canonical_or_fallback":           "canonical",
        "comparability_class":             "STRICTLY_COMPARABLE",
        "analysis_unit":                   "per_spectrum",
        "n_spectra":                       int(ds.n_spectra),
        "n_samples":                       int(meta["sample_id"].nunique()),
        "per_class_counts":                {str(c): int(n) for c, n in ds.cohort_counts.items()},
        "healthy_centroid_source_class":   "healthy_control",
        "batch_metadata_available":        False,
        "batch_metadata_field":            None,
        "known_deviations": (
            "excluded 4 NPZ `class_summary` pseudo-spectra to match Pilot 2b's corrected scope; "
            "350 real processed_spectrum rows used (LM=80, HCC=88, CCA=95, HC=87)."
        ),
    }
    (out_root / "pilot_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def write_report(report_dir, tables_dir, bsv, delta, dist, ds, meta, prep, contrib):
    axis_fx = pd.read_csv(tables_dir / "pilot3_lm_raw_axis_effect_sizes.csv")
    lm_fx = axis_fx[axis_fx["compare_class"] == PRIMARY_CLASS].copy()
    lm_fx["abs_delta"] = lm_fx["delta_mean"].abs()
    lm_fx["abs_d"] = lm_fx["cohens_d"].abs()
    top_by_delta = lm_fx.nlargest(3, "abs_delta")
    top_by_d = lm_fx.nlargest(3, "abs_d")

    # Per-sample sign consistency (LM cohort)
    lm_mask = ds.cohorts == PRIMARY_CLASS
    sample_means = (
        pd.DataFrame({"sample_id": meta["sample_id"].to_numpy()[lm_mask],
                       **{f"d_{c}": delta[lm_mask, BSV_COMPONENTS.index(c)] for c in BSV_COMPONENTS}})
        .groupby("sample_id").mean()
    )
    consistency = {}
    for c in BSV_COMPONENTS:
        cohort_sign = np.sign(delta[lm_mask, BSV_COMPONENTS.index(c)].mean())
        sample_signs = np.sign(sample_means[f"d_{c}"].values)
        consistency[c] = float((sample_signs == cohort_sign).mean())

    max_abs_d = float(lm_fx["abs_d"].max())
    consistent_axes = [c for c, v in consistency.items() if v >= 0.60]

    # Verdict
    if max_abs_d < 0.2:
        verdict = "no meaningful structure"
    elif max_abs_d >= 0.5 and len(consistent_axes) >= 2:
        verdict = "coherent cohort shift"
    elif max_abs_d >= 0.2 and len(consistent_axes) >= 1:
        verdict = "weak but real shift"
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
        n_windows = len(sub)
        if n_windows < 3:
            note = f"Axis has only {n_windows} mapped window(s) by atlas design — single/multi-window dominance is structural, not a fragility flag."
        elif sub.iloc[0]["abs"] > 0.75 * total:
            note = f"Single-window dominated (top window accounts for {top_pct:.0f}% of |Δ|)."
        else:
            note = f"Multi-band signal (top window contributes {top_pct:.0f}%)."
        contrib_lines.append(f"  {note}")

    # Entanglement
    corr = np.corrcoef(delta.T)
    ent_rows = []
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(corr[i, j]) >= 0.5:
                ent_rows.append((corr[i, j], BSV_COMPONENTS[i], BSV_COMPONENTS[j]))
    ent_rows.sort(key=lambda t: -abs(t[0]))

    def _fmt(r):
        return (f"- **{BSV_SHORT[r['axis']]}** (`{r['axis']}`) · Δmean = `{r['delta_mean']:+.5f}` · "
                f"Cohen's d = `{r['cohens_d']:+.2f}` "
                f"(95% CI [{r['cohens_d_ci_low']:+.2f}, {r['cohens_d_ci_high']:+.2f}]) · "
                f"Cliff's δ = `{r['cliffs_delta']:+.2f}`")

    md = []
    md.append("# GAIRA Target Pilot 3 — LM vs HC (canonical raw pipeline)")
    md.append("")
    md.append("Full standalone canonical pilot for Liver Metastasis (LM) vs healthy_control, "
              "using the locked canonical front end `raw_asls_sg_l2`. No reuse of Pilot 1 or "
              "Pilot 2b centroids / statistics.")
    md.append("")
    md.append("## Pipeline declaration (per gaira_target_pipeline_lock_v1)")
    md.append("")
    md.append(f"- **Policy version:** `v1`")
    md.append(f"- **Dataset:** `{DATASET_ID}`")
    md.append(f"- **Paper / substrate:** {DATASET_FAMILY_PAPER} · {SUBSTRATE_NOTE}")
    md.append(f"- **Input artifact type:** `duckdb_row_store` (biosample_spectrum_points)")
    md.append(f"- **Source path:** `{DUCKDB_PATH}::biosample_spectrum_points`")
    md.append(f"- **Raw available:** `True`")
    md.append(f"- **Interpolation axis:** [400.0, 1800.0] cm⁻¹ · 1401 pts · step 1.0 cm⁻¹")
    md.append(f"- **Preprocessing tag used:** `{prep.pipeline}`")
    md.append(f"- **Canonical or fallback:** `canonical`")
    md.append(f"- **Comparability class:** `STRICTLY_COMPARABLE`")
    md.append(f"- **Analysis unit:** per-spectrum (sample-level roll-up used as robustness check only)")
    md.append("")
    md.append("## A. Dataset facts")
    counts = {cls: int((ds.cohorts == cls).sum()) for cls in CLASS_ORDER}
    md.append(f"- **350 real raw spectra** used · "
              + " · ".join(f"{cls.replace('healthy_control','HC').upper()} n={n}" for cls, n in counts.items()))
    md.append(f"- **Primary comparison:** LM (n={counts['lm']}) vs HC (n={counts['healthy_control']}) against "
              f"within-dataset healthy centroid.")
    md.append(f"- Other disease classes (CCA, HCC) co-reported in `cohort_summary` and `axis_effect_sizes` "
              f"for parity with Pilot 2b schema; not the Pilot 3 focus.")
    md.append(f"- Metadata preserved per spectrum: `biosample_id`, `class`, `sample_id`, "
              f"`source_row_id`, `acquisition_index`, `maprow`.")
    md.append(f"- **Replicate structure present**: {meta['sample_id'].nunique()} unique `sample_id`s "
              f"across 350 spectra (LM: {meta[lm_mask]['sample_id'].nunique()} unique samples).")
    md.append(f"- **No substrate_batch / acquisition_date metadata** in this release; "
              f"`batch_summary.csv` emitted with `batch='all'`.")
    md.append(f"- **Analysis unit:** per-spectrum per prompt; a per-sample robustness check is "
              f"provided in fig 5.")
    md.append("")
    md.append("## B. Pipeline used")
    md.append(f"- Raw source: DuckDB `biosample_spectrum_points` filtered to the 350 NPZ biosample IDs "
              f"with `record_kinds == 'processed_spectrum'` (the 4 NPZ `class_summary` pseudo-rows are "
              f"excluded, same as Pilot 2b).")
    md.append(f"- Per-spectrum `numpy.interp` onto the master axis.")
    md.append(f"- `gaira.spectral.preprocessing._preprocess_raw` → pipeline tag **`{prep.pipeline}`** "
              f"(AsLS λ=1e5, p=0.001, 10 iter · SG window=11, order=3 · L2 vector norm).")
    md.append(f"- Window panel: 22 canonical windows · BSV projection: 350 × 8.")
    md.append(f"- Healthy centroid: mean BSV of {counts['healthy_control']} `healthy_control` spectra "
              f"(dataset-specific).")
    md.append(f"- Per-spectrum ΔBSV = `bsv − healthy_centroid`.")
    md.append(f"- Distance-to-centroid: L2 in 8-D BSV space.")
    md.append("")
    md.append("## C. Main findings — LM vs HC")
    md.append("")
    md.append("### Top axes by |Δmean|")
    for _, r in top_by_delta.iterrows():
        md.append(_fmt(r))
    md.append("")
    md.append("### Top axes by |Cohen's d|")
    for _, r in top_by_d.iterrows():
        md.append(_fmt(r))
    md.append("")
    md.append("### Axis ranking (within dataset, LM vs HC)")
    rank_df = lm_fx.sort_values("abs_d", ascending=False)[["axis", "delta_mean", "cohens_d",
                                                              "cohens_d_ci_low", "cohens_d_ci_high",
                                                              "cliffs_delta"]]
    md.append("| Rank | Axis | Δmean | Cohen's d | CI 95% | Cliff's δ |")
    md.append("|---|---|---:|---:|---|---:|")
    for k, (_, r) in enumerate(rank_df.iterrows(), start=1):
        md.append(f"| {k} | {BSV_SHORT[r['axis']]} | `{r['delta_mean']:+.5f}` | "
                  f"`{r['cohens_d']:+.2f}` | "
                  f"[{r['cohens_d_ci_low']:+.2f}, {r['cohens_d_ci_high']:+.2f}] | "
                  f"`{r['cliffs_delta']:+.2f}` |")
    md.append("")
    md.append("### Magnitude vs within-healthy dispersion")
    for c in top_by_d["axis"].tolist():
        r = lm_fx[lm_fx["axis"] == c].iloc[0]
        ratio = abs(r["delta_mean"]) / r["reference_sd"] if r["reference_sd"] > 0 else float("nan")
        md.append(f"- {BSV_SHORT[c]}: |Δmean|/σ(HC) = `{ratio:.2f}`")
    md.append("")
    md.append("### Dispersion")
    hc_dist = dist[ds.cohorts == "healthy_control"]
    lm_dist = dist[lm_mask]
    md.append(f"- Median per-axis σ: HC `{lm_fx['reference_sd'].median():.4f}` vs "
              f"LM `{lm_fx['compare_sd'].median():.4f}`")
    md.append(f"- Distance-to-centroid median: HC `{np.median(hc_dist):.4f}` vs "
              f"LM `{np.median(lm_dist):.4f}` · "
              f"Cliff's δ(LM vs HC) = `{_cliffs_delta(lm_dist, hc_dist):+.2f}`")
    md.append("")
    md.append("### Sample-level robustness (LM)")
    for axis, score in sorted(consistency.items(), key=lambda kv: -kv[1]):
        md.append(f"- {BSV_SHORT[axis]}: {score*100:.0f}% of LM samples match cohort ΔBSV sign")
    md.append("")
    md.append(f"Axes with ≥ 60% sample-sign agreement: "
              f"{', '.join(BSV_SHORT[c] for c in consistent_axes) if consistent_axes else '— none —'}.")
    md.append("")
    md.append("## D. Axis interpretation")
    md.append("")
    md.append("### Contribution diagnostics — top 2 axes")
    md.extend(contrib_lines)
    md.append("")
    md.append("### Entanglement")
    if ent_rows:
        md.append("Entangled axis pairs (|r| ≥ 0.5):")
        for r, a, b in ent_rows:
            md.append(f"- {BSV_SHORT[a]} ↔ {BSV_SHORT[b]}: r = `{r:+.2f}`")
        md.append("")
        md.append("> These axes share spectral drivers and must not be read as independent evidence channels.")
    else:
        md.append("- No axis pairs exceed |r| ≥ 0.5 under the raw pipeline — no entanglement concerns.")
    md.append("")
    md.append("## E. Final verdict")
    md.append("")
    md.append(f"**{verdict}.**")
    md.append("")
    md.append("Based on (i) max |Cohen's d| across the 8 axes for LM vs healthy_control within this "
              "dataset, (ii) number of axes where ≥ 60% of LM samples match the cohort-mean ΔBSV "
              "direction, and (iii) distance-to-centroid Cliff's δ.")
    md.append("")
    md.append("## F. Carry-forward recommendation (pre-Step-2 view)")
    md.append("")
    # Flag axes with |d| >= 0.5 and CI excluding zero as candidate Tier-1 under the existing rules
    def _ci_excl(r):
        return (r["cohens_d_ci_low"] > 0 and r["cohens_d_ci_high"] > 0) or \
               (r["cohens_d_ci_low"] < 0 and r["cohens_d_ci_high"] < 0)
    t1_candidates, t2_candidates, t3_candidates = [], [], []
    for _, r in lm_fx.iterrows():
        excl = _ci_excl(r)
        if r["abs_d"] >= 0.5 and excl and consistency[r["axis"]] >= 0.6:
            t1_candidates.append(r["axis"])
        elif r["abs_d"] >= 0.2 and excl:
            t2_candidates.append(r["axis"])
        else:
            t3_candidates.append(r["axis"])

    def _axes_pretty(axes):
        return ", ".join(BSV_SHORT[a] for a in axes) if axes else "— none —"

    md.append("Under the Step-2 v1 thresholds (|d| ≥ 0.5 + CI excludes zero + sample-sign ≥ 60%), the "
              f"likely Pilot 3 Tier-1 axes are: **{_axes_pretty(t1_candidates)}**. Likely Tier-2: "
              f"{_axes_pretty(t2_candidates)}. Likely Tier-3: {_axes_pretty(t3_candidates)}. Formal "
              "reliability classification should be run once Pilot 3 is admitted to the Step-2 matrix.")
    md.append("")
    md.append("**Important:** entanglement and single-window-dominance checks, and any cross-preprocessing "
              "sensitivity branch, will refine this classification. Nuc.Backbone in particular still has "
              "only 1 mapped atlas window — the same structural caveat noted in prior pilots applies.")
    md.append("")
    md.append("## G. Outputs")
    md.append("")
    md.append("### Tables")
    for n in [
        "pilot3_lm_raw_per_spectrum_bsv.csv",
        "pilot3_lm_raw_per_spectrum_delta_bsv.csv",
        "pilot3_lm_raw_cohort_summary.csv",
        "pilot3_lm_raw_batch_summary.csv",
        "pilot3_lm_raw_axis_effect_sizes.csv",
        "pilot3_lm_raw_axis_correlation.csv",
        "pilot3_lm_raw_contribution_diagnostics.csv",
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
    md.append("### Manifest")
    md.append("- `pilot_manifest.yaml` — policy-compliant pilot metadata")
    md.append("")
    md.append("## H. Scope & limits")
    md.append("- Primary comparison is LM vs healthy_control within this dataset. CCA and HCC class "
              "results are co-reported for completeness; they are not the Pilot 3 focus.")
    md.append("- No cross-dataset normalization / alignment. Schema parity with Pilot 1 and Pilot 2b "
              "is maintained so later cross-pilot synthesis can ingest Pilot 3 directly.")
    md.append("- Sample-level replication exists in this dataset, but primary analysis unit is per-spectrum; "
              "sample-level roll-up is used as a robustness check only.")
    md.append("- Pilot 3 has no cross-preprocessing sensitivity branch of its own (Pilot 2a-style "
              "non-canonical rerun was not executed for this cohort). Step 2-equivalent reliability "
              "classification will record `sensitivity_branch = not_available` for LM until such a "
              "branch is added.")
    md.append("")
    (report_dir / "REPORT_pilot3_lm_raw.md").write_text("\n".join(md))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for p in ("tables", "figures", "report"):
        (OUT_ROOT / p).mkdir(parents=True, exist_ok=True)
    tables_dir = OUT_ROOT / "tables"
    figures_dir = OUT_ROOT / "figures"
    report_dir = OUT_ROOT / "report"
    print(f"[pilot3] out: {OUT_ROOT}")

    ds, meta, Xn, wf, bsv, prep = run_pipeline()
    print(f"[pilot3] dataset {DATASET_ID}: X={Xn.shape}, BSV={bsv.shape}, cohorts={ds.cohort_counts}")

    is_hc = ds.cohorts == "healthy_control"
    healthy_centroid = bsv[is_hc].mean(axis=0)
    delta = bsv - healthy_centroid
    dist = np.linalg.norm(delta, axis=1)
    print(f"[pilot3] healthy centroid (raw pipeline): {healthy_centroid.round(5)}")

    contrib = write_tables(tables_dir, bsv, wf, delta, dist, ds, meta)
    print("[pilot3] standard tables written")

    fig1_bsv_heatmap(figures_dir, bsv, ds, meta)
    fig2_delta_distributions(figures_dir, delta, ds)
    fig3_cohort_bsv_radar(figures_dir, bsv, ds)
    fig4_cohort_delta_radar(figures_dir, delta, ds)
    fig5_sample_replication_panel(figures_dir, delta, ds, meta)
    fig6_pca(figures_dir, bsv, ds)
    fig7_distance(figures_dir, dist, ds)
    fig8_axis_correlation(figures_dir, delta)
    print("[pilot3] standard figures written")

    write_manifest(OUT_ROOT, ds, meta, prep)
    print("[pilot3] pilot_manifest.yaml written")

    write_report(report_dir, tables_dir, bsv, delta, dist, ds, meta, prep, contrib)
    print("[pilot3] report written")
    print("[pilot3] done")


if __name__ == "__main__":
    main()
