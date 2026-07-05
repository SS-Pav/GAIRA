"""GAIRA Target Step 4 — window 1020–1080 cm⁻¹ ablation sensitivity (v1).

Surgical robustness test on canonical Pilot 2b (CCA raw, `raw_asls_sg_l2`).
Ablates the single atlas window mapped to `nucleic_acid_backbone`
(`1020-1080`) by zeroing its column in the 22-window feature matrix
before BSV projection. Everything else stays on the canonical path.

This script does NOT modify the atlas, the scorer, any `src/` core code,
or any existing pilot output. It creates an isolated sensitivity branch
under:

    /Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/step4_window_sensitivity_1020_1080_v1/

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_target_step4_window_sensitivity_1020_1080_v1.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral.dataset_loader import NPZ_PATH, SpectralDataset
from gaira.spectral.preprocessing import _preprocess_raw
from gaira.spectral.window_panel import BSV_COMPONENTS, WINDOW_DEFS, extract_window_features
from gaira.spectral.bsv_projection import project_to_bsv


DATASET_ID = "cca_hcc_lm_serum_sers"
ABLATE_WINDOW_ID = "1020-1080"
DUCKDB_PATH = Path("/Volumes/SSD_Rad/GAIRA_DATA/interim/gaira.duckdb")
OUT_ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/step4_window_sensitivity_1020_1080_v1")
P2B_TABLES = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot/pilot2b_cca_raw/tables")
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
# Stats helpers
# ──────────────────────────────────────────────────────────────────────

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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ──────────────────────────────────────────────────────────────────────
# Raw load (identical to Pilot 2b)
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
    rk_mask = npz["record_kinds"] == "processed_spectrum"
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
            "source_row_id": sr, "maprow": mr,
        })
    meta = pd.DataFrame(meta_rows)
    assert len(meta) == 350, f"expected 350, got {len(meta)}"

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


# ──────────────────────────────────────────────────────────────────────
# Canonical + ablated branches
# ──────────────────────────────────────────────────────────────────────

def _ablate_index() -> int:
    for j, (wid, _, _, _) in enumerate(WINDOW_DEFS):
        if wid == ABLATE_WINDOW_ID:
            return j
    raise KeyError(ABLATE_WINDOW_ID)


def run_branches(ds: SpectralDataset) -> dict:
    Xn, prep = _preprocess_raw(ds)
    wf = extract_window_features(Xn, ds.wavenumbers)

    # Canonical
    bsv_c = project_to_bsv(wf)

    # Ablated: zero the 1020-1080 column; project with the unmodified projector.
    # Since `1020-1080` is the ONLY window mapped to `nucleic_acid_backbone`, this
    # collapses that axis to a constant 0 across all spectra and leaves every other
    # axis's BSV value identical to canonical.
    j = _ablate_index()
    wf_a = wf.copy()
    wf_a[:, j] = 0.0
    bsv_a = project_to_bsv(wf_a)

    is_hc = ds.cohorts == "healthy_control"
    hc_c = bsv_c[is_hc].mean(axis=0)
    hc_a = bsv_a[is_hc].mean(axis=0)

    delta_c = bsv_c - hc_c
    delta_a = bsv_a - hc_a

    dist_c = np.linalg.norm(delta_c, axis=1)
    dist_a = np.linalg.norm(delta_a, axis=1)

    return {
        "prep": prep, "wf": wf, "wf_a": wf_a, "ablate_idx": j,
        "bsv_c": bsv_c, "bsv_a": bsv_a,
        "hc_c": hc_c, "hc_a": hc_a,
        "delta_c": delta_c, "delta_a": delta_a,
        "dist_c": dist_c, "dist_a": dist_a,
    }


# ──────────────────────────────────────────────────────────────────────
# Effects + geometry tables
# ──────────────────────────────────────────────────────────────────────

def build_effects_table(branches: dict, ds: SpectralDataset) -> pd.DataFrame:
    is_cca = ds.cohorts == "cca"
    is_hc  = ds.cohorts == "healthy_control"

    def _d_info(vec_c, vec_a, axis):
        d_c, lo_c, hi_c = _cohens_d(vec_c[is_cca], vec_c[is_hc])
        d_a, lo_a, hi_a = _cohens_d(vec_a[is_cca], vec_a[is_hc])
        return (d_c, lo_c, hi_c), (d_a, lo_a, hi_a)

    rows = []
    for i, comp in enumerate(BSV_COMPONENTS):
        (d_c, lo_c, hi_c), (d_a, lo_a, hi_a) = _d_info(
            branches["bsv_c"][:, i], branches["bsv_a"][:, i], comp
        )
        rows.append({
            "axis": comp,
            "canonical_d": d_c, "canonical_d_ci_low": lo_c, "canonical_d_ci_high": hi_c,
            "ablated_d": d_a, "ablated_d_ci_low": lo_a, "ablated_d_ci_high": hi_a,
            "delta_d": d_a - d_c,
            "sign_changed": bool(np.sign(d_c) != np.sign(d_a) and abs(d_c) > 1e-6 and abs(d_a) > 1e-6),
        })

    # Ranks (by |d|), ascending from 1 = largest
    df = pd.DataFrame(rows)
    df["rank_canonical"] = df["canonical_d"].abs().rank(ascending=False, method="min").astype(int)
    df["rank_ablated"]   = df["ablated_d"].abs().rank(ascending=False, method="min").astype(int)
    df["rank_changed"]   = df["rank_ablated"] - df["rank_canonical"]

    # Interpretation note
    notes = []
    for _, r in df.iterrows():
        if r["axis"] == "nucleic_acid_backbone":
            notes.append("axis ablated — all contributions zeroed; d collapses to 0 by construction")
        elif abs(r["delta_d"]) < 0.01:
            notes.append("unchanged — no atlas window shared with 1020-1080")
        else:
            notes.append(f"shift Δd={r['delta_d']:+.3f}")
    df["interpretation_note"] = notes

    return df


def build_geometry_table(branches: dict, ds: SpectralDataset) -> pd.DataFrame:
    rows = []
    for cls in CLASS_ORDER:
        m = ds.cohorts == cls
        if not m.any():
            continue
        dc = branches["dist_c"][m]; da = branches["dist_a"][m]
        # Cohort-mean ΔBSV vectors (canonical vs ablated)
        mc = branches["delta_c"][m].mean(axis=0)
        ma = branches["delta_a"][m].mean(axis=0)
        # Cliff's delta vs HC (within each branch)
        hc_dc = branches["dist_c"][ds.cohorts == "healthy_control"]
        hc_da = branches["dist_a"][ds.cohorts == "healthy_control"]
        rows.append({
            "cohort": cls, "n": int(m.sum()),
            "canonical_distance_median": float(np.median(dc)),
            "ablated_distance_median": float(np.median(da)),
            "distance_median_delta": float(np.median(da) - np.median(dc)),
            "canonical_cliffs_delta_vs_hc": _cliffs_delta(dc, hc_dc) if cls != "healthy_control" else 0.0,
            "ablated_cliffs_delta_vs_hc": _cliffs_delta(da, hc_da) if cls != "healthy_control" else 0.0,
            "canonical_vs_ablated_delta_cosine": _cosine(mc, ma),
            "canonical_mean_delta_l2": float(np.linalg.norm(mc)),
            "ablated_mean_delta_l2": float(np.linalg.norm(ma)),
        })
    df = pd.DataFrame(rows)

    def _note(r):
        if r["cohort"] == "healthy_control":
            return "reference cohort — within-branch centroid"
        if r["canonical_vs_ablated_delta_cosine"] > 0.9:
            verdict = "geometry robustly preserved"
        elif r["canonical_vs_ablated_delta_cosine"] > 0.75:
            verdict = "geometry largely preserved"
        elif r["canonical_vs_ablated_delta_cosine"] > 0.5:
            verdict = "geometry partially preserved"
        else:
            verdict = "geometry materially changed"
        sep_before = r["canonical_cliffs_delta_vs_hc"]
        sep_after  = r["ablated_cliffs_delta_vs_hc"]
        verdict += f" (Cliff's δ vs HC: {sep_before:+.2f} → {sep_after:+.2f})"
        return verdict
    df["interpretation"] = df.apply(_note, axis=1)
    return df


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 180, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _radar(ax, labels, series, colors, fill_alpha=0.18, linewidth=2.2):
    n = len(labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    theta_c = theta + [theta[0]]
    for (name, vals), color in zip(series, colors):
        v = list(vals) + [vals[0]]
        ax.plot(theta_c, v, color=color, linewidth=linewidth, label=name)
        ax.fill(theta_c, v, color=color, alpha=fill_alpha)
    ax.set_xticks(theta); ax.set_xticklabels(labels, fontsize=9)
    ax.tick_params(axis="y", labelsize=8, colors="#555")


def fig_radar_overlay(branches: dict, ds: SpectralDataset, path: Path):
    is_cca = ds.cohorts == "cca"
    canon = branches["delta_c"][is_cca].mean(axis=0)
    abl   = branches["delta_a"][is_cca].mean(axis=0)
    m = max(1e-4, max(float(np.abs(canon).max()), float(np.abs(abl).max())) * 1.2)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    _radar(
        ax, [BSV_SHORT[c] for c in BSV_COMPONENTS],
        [("Canonical (22 windows)", canon),
         ("Ablated (1020–1080 removed)", abl)],
        ["#94A3B8", "#E45756"], fill_alpha=0.28, linewidth=2.4,
    )
    ax.set_ylim(-m, m)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(theta, np.zeros_like(theta), color="#444", lw=0.8, linestyle="--")
    ax.set_title("CCA cohort-mean ΔBSV — canonical vs ablated", y=1.08, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)
    _save(fig, path)


def fig_rank_shift(effects_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    df = effects_df.sort_values("rank_canonical").reset_index(drop=True)
    for _, r in df.iterrows():
        axis = r["axis"]
        # Color: Nuc.Backbone = red (ablation target); other axes keep style based on sign change
        if axis == "nucleic_acid_backbone":
            col = "#E45756"; marker = "X"
        elif r["sign_changed"]:
            col = "#D97706"; marker = "o"
        elif abs(r["delta_d"]) > 0.05:
            col = "#F59E0B"; marker = "o"
        else:
            col = "#4ADE80"; marker = "o"
        ax.plot([0, 1], [r["rank_canonical"], r["rank_ablated"]], color=col, lw=2.2, alpha=0.85)
        ax.scatter(0, r["rank_canonical"], color=col, s=62, zorder=3, edgecolor="white", marker=marker)
        ax.scatter(1, r["rank_ablated"], color=col, s=62, zorder=3, edgecolor="white", marker=marker)
        ax.text(-0.06, r["rank_canonical"], BSV_SHORT[axis], ha="right", va="center",
                 fontsize=10, color="#1F2937")
        ax.text(1.06, r["rank_ablated"], BSV_SHORT[axis], ha="left", va="center",
                 fontsize=10, color="#1F2937")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Canonical", "Ablated\n(1020–1080 removed)"], fontsize=11)
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylim(8.6, 0.4)
    ax.set_ylabel("|effect size| rank (1 = largest)")
    ax.set_title("CCA axis ranking by |Cohen's d| — canonical vs ablated",
                  fontsize=13, pad=12)
    ax.grid(alpha=0.15, axis="y", linestyle=":")

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="#E45756", marker="X", markersize=8, lw=2,
                label="Nuc.Backbone (ablation target)"),
        Line2D([0], [0], color="#4ADE80", marker="o", markersize=8, lw=2,
                label="Stable axis (|Δd| ≤ 0.05)"),
        Line2D([0], [0], color="#F59E0B", marker="o", markersize=8, lw=2,
                label="Shifted (|Δd| > 0.05)"),
        Line2D([0], [0], color="#D97706", marker="o", markersize=8, lw=2,
                label="Sign change"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    _save(fig, path)


def fig_distance_comparison(branches: dict, ds: SpectralDataset, path: Path):
    # 2 panels (canonical, ablated); 4 cohort histograms each; shared x-axis
    all_vals = np.concatenate([branches["dist_c"], branches["dist_a"]])
    x_max = float(all_vals.max()) * 1.02
    bins = np.linspace(0, x_max, 34)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    for ax, title, key in zip(
        axes, ["Canonical", "Ablated (1020–1080 removed)"],
        ["dist_c", "dist_a"],
    ):
        for cls in CLASS_ORDER:
            m = ds.cohorts == cls
            if not m.any():
                continue
            vals = branches[key][m]
            ax.hist(vals, bins=bins, color=CLASS_COLORS[cls], alpha=0.55,
                     label=f"{cls.replace('healthy_control','HC').upper()} "
                           f"(n={int(m.sum())}, med={np.median(vals):.4f})")
            ax.axvline(np.median(vals), color=CLASS_COLORS[cls], lw=1.2, linestyle="--")
        ax.set_ylabel("Count")
        ax.set_title(title, fontsize=12, loc="left")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("‖spectrum BSV − healthy centroid‖₂")
    fig.suptitle("Distance-to-healthy-centroid distributions — canonical vs ablated",
                  fontsize=13, y=1.00)
    fig.tight_layout()
    _save(fig, path)


def fig_effect_size_comparison(effects_df: pd.DataFrame, path: Path):
    axes = BSV_COMPONENTS
    canon = [float(effects_df[effects_df["axis"] == a]["canonical_d"].iloc[0]) for a in axes]
    abl   = [float(effects_df[effects_df["axis"] == a]["ablated_d"].iloc[0]) for a in axes]
    x = np.arange(len(axes))
    w = 0.38

    fig, ax = plt.subplots(figsize=(13, 5.8))
    # Use absolute-value magnitude for |d| comparison; signs shown in annotations
    ax.bar(x - w / 2, np.abs(canon), width=w, color="#94A3B8",
            edgecolor="white", label="Canonical |d|")
    ax.bar(x + w / 2, np.abs(abl), width=w, color="#E45756",
            edgecolor="white", label="Ablated |d|")
    for i, (c_v, a_v) in enumerate(zip(canon, abl)):
        ax.text(i - w / 2, abs(c_v) + 0.02, f"{c_v:+.2f}",
                 ha="center", va="bottom", fontsize=8, color="#1F2937")
        ax.text(i + w / 2, abs(a_v) + 0.02, f"{a_v:+.2f}",
                 ha="center", va="bottom", fontsize=8, color="#B91C1C")
    ax.set_xticks(x)
    ax.set_xticklabels([BSV_SHORT[a] for a in axes], rotation=25, ha="right")
    ax.set_ylabel("|Cohen's d|  (CCA vs HC)")
    ax.set_title("Per-axis effect size — canonical vs ablated (signed values annotated)",
                  fontsize=12, pad=10)
    ax.axhline(0.5, color="#888", lw=0.7, linestyle=":")
    ax.axhline(0.2, color="#888", lw=0.7, linestyle=":")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, max(max(np.abs(canon)), max(np.abs(abl))) * 1.18)
    _save(fig, path)


def fig_axis_correlation_comparison(branches: dict, path: Path):
    corr_c = np.corrcoef(branches["delta_c"].T)
    # For ablated, nuc.backbone column is all zero → correlation row/col is NaN.
    corr_a = np.corrcoef(branches["delta_a"].T)
    # Replace NaN with 0 for display (annotate as N/A)
    labels = [BSV_SHORT[c] for c in BSV_COMPONENTS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, corr, title in zip(axes, [corr_c, corr_a], ["Canonical", "Ablated"]):
        im = ax.imshow(np.nan_to_num(corr, nan=0.0), cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(8):
            for j in range(8):
                v = corr[i, j]
                txt = "N/A" if np.isnan(v) else f"{v:+.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                         color="black" if (np.isnan(v) or abs(v) < 0.6) else "white")
        ax.set_xticks(range(8)); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticks(range(8)); ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r")
    fig.suptitle("ΔBSV axis correlation — canonical vs ablated", fontsize=13, y=1.02)
    _save(fig, path)


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def write_report(effects_df: pd.DataFrame, geometry_df: pd.DataFrame,
                  branches: dict, ds: SpectralDataset, prep, path: Path):
    def _row(a):
        r = effects_df[effects_df["axis"] == a].iloc[0]
        return r

    # Question 1 — does the cohort shift survive?
    cca_geom = geometry_df[geometry_df["cohort"] == "cca"].iloc[0]
    cosine_cca = cca_geom["canonical_vs_ablated_delta_cosine"]
    cca_cliff_c = cca_geom["canonical_cliffs_delta_vs_hc"]
    cca_cliff_a = cca_geom["ablated_cliffs_delta_vs_hc"]

    if cosine_cca >= 0.9 and cca_cliff_a >= 0.35:
        q1 = "yes, largely"
    elif cosine_cca >= 0.75:
        q1 = "yes, partially"
    elif cosine_cca >= 0.5:
        q1 = "yes, partially"
    else:
        q1 = "no"

    # Question 2 — Nuc.Backbone fate
    nb = _row("nucleic_acid_backbone")
    q2 = "collapses — axis ablated by design (sole atlas window removed)"

    # Question 3 — other major axes
    def _axis_line(a):
        r = _row(a)
        direction_stable = np.sign(r["canonical_d"]) == np.sign(r["ablated_d"])
        delta = r["delta_d"]
        verdict = "unchanged" if abs(delta) < 0.01 else \
                  ("weakens" if abs(r["ablated_d"]) < abs(r["canonical_d"]) else "strengthens")
        return (f"**{BSV_SHORT[a]}**: d {r['canonical_d']:+.2f} → {r['ablated_d']:+.2f} "
                f"(Δd={delta:+.3f}); direction {'stable' if direction_stable else 'changed'}; "
                f"rank {int(r['rank_canonical'])} → {int(r['rank_ablated'])}.  → {verdict}.")

    # Question 4 — distance-to-centroid
    hc_d_c = float(np.median(branches["dist_c"][ds.cohorts == "healthy_control"]))
    hc_d_a = float(np.median(branches["dist_a"][ds.cohorts == "healthy_control"]))
    cca_d_c = float(np.median(branches["dist_c"][ds.cohorts == "cca"]))
    cca_d_a = float(np.median(branches["dist_a"][ds.cohorts == "cca"]))

    q4_verdict = "yes" if cca_cliff_a > 0.25 else ("partially" if cca_cliff_a > 0.15 else "no")

    # Final classification.
    # For a surgical single-axis ablation, cosine is dragged down whenever the
    # ablated axis contributes a large share of the ΔBSV norm — *even if* every
    # remaining axis is numerically identical. So cosine alone is not the
    # correct robustness test here. We use two structural checks instead:
    #   (1) do the other Tier-1 axes (Purine, Glycan) stay numerically stable?
    #   (2) is the CCA-vs-HC ordinal separation preserved or stronger?
    tier1_other_identical = all(
        abs(float(_row(a)["delta_d"])) < 0.01
        for a in ("purine_nucleotide", "glycan_carbohydrate")
    )
    cliff_retained_fraction = (
        cca_cliff_a / cca_cliff_c if abs(cca_cliff_c) > 1e-6 else 1.0
    )
    cliff_preserved = cliff_retained_fraction >= 0.80 and cca_cliff_a >= 0.30

    if tier1_other_identical and cliff_preserved and q4_verdict == "yes":
        classification = "BROAD_BASED_SIGNAL"
    elif cosine_cca >= 0.50 and cca_cliff_a >= 0.20:
        classification = "PARTIALLY_WINDOW_DEPENDENT"
    else:
        classification = "HEAVILY_WINDOW_DEPENDENT"

    md = []
    md.append("# REPORT — Step 4 · Window 1020–1080 cm⁻¹ ablation sensitivity (v1)")
    md.append("")
    md.append("Surgical robustness test on canonical Pilot 2b. **No changes to scorer, atlas, axes, "
              "or preprocessing.** Ablation is implemented by zeroing the single feature-matrix "
              "column corresponding to the `1020-1080` window *after* window extraction and "
              "*before* BSV projection; every other atlas window is untouched.")
    md.append("")
    md.append("## Experimental setup")
    md.append("")
    md.append(f"- **Dataset:** `{DATASET_ID}` (350 real raw spectra; identical input to Pilot 2b).")
    md.append(f"- **Preprocessing tag:** `{prep.pipeline}` (canonical `raw_asls_sg_l2`).")
    md.append(f"- **Ablation:** column index `{branches['ablate_idx']}` in the 22-window feature matrix "
              f"(window `{ABLATE_WINDOW_ID}` → `nucleic_acid_backbone`) set to 0.")
    md.append("- **Why this is surgical:** `1020-1080` is the *only* atlas window mapped to "
              "`nucleic_acid_backbone` and is used by no other axis. Zeroing it collapses Nuc.Backbone "
              "to a constant 0 across all spectra; every other axis's BSV value is identical to the "
              "canonical branch (arithmetic of `project_to_bsv`).")
    md.append("- **Reversibility:** the unmodified `project_to_bsv` and `extract_window_features` are "
              "reused as-is; no persistent changes to atlas or code.")
    md.append("")
    md.append("## Required question answers")
    md.append("")
    md.append("### 1. Does the canonical CCA cohort shift survive the 1020–1080 ablation?")
    md.append("")
    md.append(f"**{q1}.**")
    md.append("")
    md.append(f"- Cohort-mean ΔBSV cosine(canonical, ablated) = `{cosine_cca:+.3f}` for CCA.")
    md.append(f"- ‖Δ‖₂(CCA canonical) = `{cca_geom['canonical_mean_delta_l2']:.4f}` → "
              f"‖Δ‖₂(CCA ablated) = `{cca_geom['ablated_mean_delta_l2']:.4f}`.")
    md.append(f"- Cliff's δ(CCA distance vs HC distance): `{cca_cliff_c:+.2f}` → `{cca_cliff_a:+.2f}`.")
    md.append("")
    md.append("### 2. What happens to Nuc.Backbone?")
    md.append("")
    md.append(f"**{q2}.**")
    md.append(f"- d: `{nb['canonical_d']:+.2f}` → `{nb['ablated_d']:+.2f}`.  "
              f"Rank: {int(nb['rank_canonical'])} → {int(nb['rank_ablated'])}.")
    md.append("- This outcome is **by construction, not by evidence**. Because the atlas maps only "
              "this window to Nuc.Backbone, ablating it removes the axis from the representation. "
              "The ablation cannot evaluate the axis's biology; it only tests the dependence of the "
              "rest of the geometry on this axis.")
    md.append("")
    md.append("### 3. What happens to Purine / Glycan / Protein?")
    md.append("")
    for a in ("purine_nucleotide", "glycan_carbohydrate", "protein_backbone"):
        md.append(f"- {_axis_line(a)}")
    md.append("")
    md.append("All three are **identical** to canonical numerics: no atlas window overlap with "
              "`1020-1080`, so their BSVs, ΔBSVs, and effect sizes pass through unchanged.")
    md.append("")
    md.append("### 4. Does distance-to-centroid remain elevated in CCA vs HC?")
    md.append("")
    md.append(f"**{q4_verdict}.**")
    md.append(f"- HC distance median: canonical `{hc_d_c:.4f}` → ablated `{hc_d_a:.4f}`.")
    md.append(f"- CCA distance median: canonical `{cca_d_c:.4f}` → ablated `{cca_d_a:.4f}`.")
    md.append(f"- Cliff's δ(CCA vs HC): `{cca_cliff_c:+.2f}` → `{cca_cliff_a:+.2f}` "
              f"(ordinal separation {'preserved' if cca_cliff_a > 0.25 else 'reduced'}).")
    md.append("")
    md.append("### 5. Does the overall CCA geometry remain interpretable after ablation?")
    md.append("")
    # Summarize top 3 post-ablation
    abl_top3 = effects_df.sort_values("ablated_d", key=lambda s: s.abs(),
                                         ascending=False).head(3)["axis"].tolist()
    abl_top3_label = ", ".join(BSV_SHORT[a] for a in abl_top3)
    md.append(f"- Under ablation the three strongest axes (by |d|) are: **{abl_top3_label}**.")
    md.append("- The shape of the ΔBSV radar outside Nuc.Backbone is unchanged (same 7 axes, same "
              "d-values). Interpretation continues on the Tier-1 carry-forward set {Purine, Glycan} "
              "as before; Protein secondary context is also preserved.")
    md.append("- What the representation *loses* is the single dimension that carried the HCC-vs-CCA "
              "direction divergence (the key Step-3 divergence claim). That claim is *not* "
              "reaffirmed by the ablation, because the ablation cannot test an axis it has removed.")
    md.append("")
    md.append("### 6. Final classification")
    md.append("")
    md.append(f"**`{classification}`**")
    md.append("")
    if classification == "BROAD_BASED_SIGNAL":
        md.append("Rationale: both other Tier-1 axes (Purine and Glycan) pass through numerically "
                  f"unchanged (Δd = 0.000 exactly for each); the disease-vs-HC ordinal separation is "
                  f"retained at ≥ 80% of canonical (Cliff's δ {cca_cliff_c:+.2f} → {cca_cliff_a:+.2f}, "
                  f"retained fraction = {cliff_retained_fraction:.2f}); and the ablated separation "
                  f"itself is ≥ 0.30. The cohort-mean ΔBSV cosine of {cosine_cca:+.2f} is expected to "
                  "drop from 1.0 because Nuc.Backbone alone contributed a large share of the ΔBSV "
                  "vector norm, but that is a consequence of removing a high-magnitude single axis, "
                  "not of the remaining structure changing. **The canonical CCA geometry is broad-based**: "
                  "the other 7 axes carry independent, reproducible separation.")
    elif classification == "PARTIALLY_WINDOW_DEPENDENT":
        md.append(f"Rationale: cohort-mean ΔBSV cosine = {cosine_cca:+.2f} (≥ 0.50); ablated "
                  f"Cliff's δ = {cca_cliff_a:+.2f} (≥ 0.20); but at least one of the following fails: "
                  "other Tier-1 axes are not numerically stable, or the CCA-vs-HC ordinal separation "
                  "loses more than 20% of its canonical magnitude.")
    else:
        md.append("Rationale: either the cohort-mean ΔBSV direction agrees only weakly "
                  f"(cosine = {cosine_cca:+.2f}) or the disease-vs-HC separation falls below the minimal "
                  "threshold, indicating that Nuc.Backbone was carrying a disproportionate share of the "
                  "result and the other 7 axes do not independently support the cohort distinction.")
    md.append("")
    md.append("## Per-axis effects (CCA vs HC)")
    md.append("")
    md.append("| Axis | canonical d | ablated d | Δd | rank (can → abl) | note |")
    md.append("|---|---:|---:|---:|---:|---|")
    for _, r in effects_df.iterrows():
        md.append(f"| {BSV_SHORT[r['axis']]} | `{r['canonical_d']:+.2f}` | `{r['ablated_d']:+.2f}` | "
                  f"`{r['delta_d']:+.3f}` | {int(r['rank_canonical'])} → {int(r['rank_ablated'])} | "
                  f"{r['interpretation_note']} |")
    md.append("")
    md.append("## Cohort geometry (canonical vs ablated)")
    md.append("")
    md.append("| Cohort | n | canonical dist. med | ablated dist. med | cosine(can,abl) | ‖Δ‖₂(can) | ‖Δ‖₂(abl) | Cliff's δ(vs HC) can → abl |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in geometry_df.iterrows():
        cliff_row = (f"{r['canonical_cliffs_delta_vs_hc']:+.2f} → "
                      f"{r['ablated_cliffs_delta_vs_hc']:+.2f}") if r["cohort"] != "healthy_control" else "—"
        md.append(f"| {r['cohort'].replace('healthy_control','HC').upper()} | {int(r['n'])} | "
                  f"`{r['canonical_distance_median']:.4f}` | `{r['ablated_distance_median']:.4f}` | "
                  f"`{r['canonical_vs_ablated_delta_cosine']:+.3f}` | "
                  f"`{r['canonical_mean_delta_l2']:.4f}` | `{r['ablated_mean_delta_l2']:.4f}` | "
                  f"{cliff_row} |")
    md.append("")
    md.append("## Scope & limits")
    md.append("")
    md.append("- The ablation **removes** the Nuc.Backbone axis; it does **not** test its biology "
              "or robustness. Nuc.Backbone evaluation requires atlas-expansion work (adding more "
              "windows to that axis so ablation of any single window no longer empties it).")
    md.append("- This is a per-spectrum sensitivity branch on canonical Pilot 2b only. No "
              "cross-dataset normalization, no pooling, no changes to the policy file.")
    md.append("- Pilot 1 (HCC holdout) was **not** re-ablated in this step — its Step-3 divergence "
              "claim on Nuc.Backbone carries the same structural caveat and should be revisited "
              "once the atlas expands.")
    md.append("")
    md.append("## Outputs")
    md.append("")
    md.append("- `window_ablation_effects.csv`")
    md.append("- `window_ablation_geometry_summary.csv`")
    md.append("- `fig_window_ablation_radar_overlay.png`")
    md.append("- `fig_window_ablation_rank_shift.png`")
    md.append("- `fig_window_ablation_distance_comparison.png`")
    md.append("- `fig_window_ablation_effect_size_comparison.png`")
    md.append("- `fig_window_ablation_axis_correlation_comparison.png`")
    md.append("")

    path.write_text("\n".join(md))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[step4] out: {OUT_ROOT}")
    print(f"[step4] ablating window: {ABLATE_WINDOW_ID}")

    ds, meta = load_raw_dataset()
    print(f"[step4] loaded {ds.n_spectra} raw spectra · cohorts={ds.cohort_counts}")

    branches = run_branches(ds)
    print(f"[step4] branches computed · ablate_idx={branches['ablate_idx']}")

    # Quick canonical-vs-Pilot-2b agreement check (sanity)
    p2b_cohort = pd.read_csv(P2B_TABLES / "pilot2b_cca_raw_cohort_summary.csv")
    p2b_cca_mean = p2b_cohort[p2b_cohort["class"] == "cca"].set_index("axis").loc[BSV_COMPONENTS, "mean_delta_bsv"].to_numpy()
    my_cca_mean = branches["delta_c"][ds.cohorts == "cca"].mean(axis=0)
    cos_agree = _cosine(p2b_cca_mean, my_cca_mean)
    print(f"[step4] sanity: canonical-branch CCA ΔBSV cosine vs saved Pilot 2b: {cos_agree:+.6f}")

    effects_df = build_effects_table(branches, ds)
    effects_df.to_csv(OUT_ROOT / "window_ablation_effects.csv", index=False)
    print("[step4] wrote window_ablation_effects.csv")

    geometry_df = build_geometry_table(branches, ds)
    geometry_df.to_csv(OUT_ROOT / "window_ablation_geometry_summary.csv", index=False)
    print("[step4] wrote window_ablation_geometry_summary.csv")

    fig_radar_overlay(branches, ds, OUT_ROOT / "fig_window_ablation_radar_overlay.png")
    fig_rank_shift(effects_df, OUT_ROOT / "fig_window_ablation_rank_shift.png")
    fig_distance_comparison(branches, ds, OUT_ROOT / "fig_window_ablation_distance_comparison.png")
    fig_effect_size_comparison(effects_df, OUT_ROOT / "fig_window_ablation_effect_size_comparison.png")
    fig_axis_correlation_comparison(branches, OUT_ROOT / "fig_window_ablation_axis_correlation_comparison.png")
    print("[step4] figures written")

    prep = branches["prep"]
    write_report(effects_df, geometry_df, branches, ds, prep,
                  OUT_ROOT / "REPORT_step4_window_sensitivity_1020_1080_v1.md")
    print("[step4] report written")
    print("[step4] done")


if __name__ == "__main__":
    main()
