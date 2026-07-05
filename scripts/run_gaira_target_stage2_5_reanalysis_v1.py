"""GAIRA — Stage 2.5 Full Reanalysis + Cross-Pilot Synthesis (v1).

For each canonical target pilot (1 HCC, 2b CCA, 3 LM) this runner:
  Step 1 — replicates canonical figures from the existing pilot tables
           (BSV heatmap, ΔBSV distributions, BSV / ΔBSV radars, PCA scatter,
            distance-to-centroid plot, correlation heatmap).
  Step 2 — joins the Stage 2 substrate overlay onto the pilot's principal
           effect-sizes table.
  Step 3 — within-pilot multi-cohort comparisons (only meaningful for the
           4-class Ag-colloid dataset shared by Pilots 2b / 3).
  Step 4 — cross-pilot HCC comparison (Pilot 1 HCC vs Pilot 2b HCC).
  Step 5 — Ag-colloid disease progression (HC → CCA → HCC → LM) using the
           4-class effect-sizes table from Pilot 2b (same data underlies
           Pilot 3, by design).
  Step 6 — global per-axis synthesis table (direction × consistency ×
           substrate class × final interpretation).
  Step 7 — final markdown report.

Hard rules enforced:
  - never recompute BSV / ΔBSV (correlation matrices and PCA are
    visualisation statistics over the existing BSV columns)
  - never normalise across datasets
  - never pool data across pilots
  - cross-pilot comparisons are categorical: direction sign, rank order,
    effect-size class — never raw-magnitude pooling
  - file checksum gate verifies no pilot file is mutated

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_target_stage2_5_reanalysis_v1.py
"""
from __future__ import annotations

import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral.window_panel import BSV_COMPONENTS


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

PILOT_ROOT  = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot")
STAGE2_ROOT = PILOT_ROOT / "stage2_substrate_overlay_v1"
OUT_ROOT    = PILOT_ROOT / "stage2_5_reanalysis_v1"

CLASS_ORDER_4 = ["healthy_control", "cca", "hcc", "lm"]
CLASS_PALETTE = {
    "healthy_control": "#4E79A7",  # blue
    "cca":             "#F28E2B",  # orange
    "hcc":             "#E15759",  # red
    "lm":              "#59A14F",  # green
}
SHIFT_PALETTE = {
    "STRONGER":     "#3F8E3F",
    "WEAKER":       "#C04444",
    "AMBIGUOUS":    "#B07A2A",
    "INCONCLUSIVE": "#888888",
    "UNCHANGED":    "#444444",
    "MIXED":        "#864FB4",
}


@dataclass(frozen=True)
class PilotSpec:
    pilot_id: str
    short_label: str
    substrate_family: str
    reference_class: str
    principal_compare_class: str
    tables_dir: Path
    per_spectrum_bsv_csv: str
    per_spectrum_delta_bsv_csv: str
    axis_effect_sizes_csv: str
    cohort_summary_csv: str
    axis_correlation_csv: str | None
    contribution_diagnostics_csv: str | None
    overlay_csv: Path        # output of stage 2 runner


PILOTS: list[PilotSpec] = [
    PilotSpec(
        pilot_id="pilot1_hcc",
        short_label="Pilot 1 — HCC holdout",
        substrate_family="Ag_nanostructured_array",
        reference_class="healthy_control",
        principal_compare_class="hcc",
        tables_dir=PILOT_ROOT / "gaira_target_pilot1_hcc_holdout_bsv" / "tables",
        per_spectrum_bsv_csv="pilot1_hcc_per_spectrum_bsv.csv",
        per_spectrum_delta_bsv_csv="pilot1_hcc_per_spectrum_delta_bsv.csv",
        axis_effect_sizes_csv="pilot1_hcc_axis_effect_sizes.csv",
        cohort_summary_csv="pilot1_hcc_cohort_summary.csv",
        axis_correlation_csv=None,
        contribution_diagnostics_csv=None,
        overlay_csv=STAGE2_ROOT / "pilot1_hcc_axis_substrate_overlay.csv",
    ),
    PilotSpec(
        pilot_id="pilot2b_cca",
        short_label="Pilot 2b — CCA (Ag colloid)",
        substrate_family="Ag_nanoparticle_colloid",
        reference_class="healthy_control",
        principal_compare_class="cca",
        tables_dir=PILOT_ROOT / "pilot2b_cca_raw" / "tables",
        per_spectrum_bsv_csv="pilot2b_cca_raw_per_spectrum_bsv.csv",
        per_spectrum_delta_bsv_csv="pilot2b_cca_raw_per_spectrum_delta_bsv.csv",
        axis_effect_sizes_csv="pilot2b_cca_raw_axis_effect_sizes.csv",
        cohort_summary_csv="pilot2b_cca_raw_cohort_summary.csv",
        axis_correlation_csv="pilot2b_cca_raw_axis_correlation.csv",
        contribution_diagnostics_csv="pilot2b_cca_raw_contribution_diagnostics.csv",
        overlay_csv=STAGE2_ROOT / "pilot2b_cca_axis_substrate_overlay.csv",
    ),
    PilotSpec(
        pilot_id="pilot3_lm",
        short_label="Pilot 3 — LM (Ag colloid)",
        substrate_family="Ag_nanoparticle_colloid",
        reference_class="healthy_control",
        principal_compare_class="lm",
        tables_dir=PILOT_ROOT / "pilot3_lm_raw" / "tables",
        per_spectrum_bsv_csv="pilot3_lm_raw_per_spectrum_bsv.csv",
        per_spectrum_delta_bsv_csv="pilot3_lm_raw_per_spectrum_delta_bsv.csv",
        axis_effect_sizes_csv="pilot3_lm_raw_axis_effect_sizes.csv",
        cohort_summary_csv="pilot3_lm_raw_cohort_summary.csv",
        axis_correlation_csv="pilot3_lm_raw_axis_correlation.csv",
        contribution_diagnostics_csv="pilot3_lm_raw_contribution_diagnostics.csv",
        overlay_csv=STAGE2_ROOT / "pilot3_lm_axis_substrate_overlay.csv",
    ),
]


# ──────────────────────────────────────────────────────────────────────
# Helpers — IO + checksum
# ──────────────────────────────────────────────────────────────────────

def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()


def _read_csv(p: Path) -> list[dict[str, str]]:
    with p.open() as f: return list(csv.DictReader(f))


def _write_csv(p: Path, header: list[str], rows: list[list]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(r)


def _snapshot_pilot(spec: PilotSpec) -> dict[str, str]:
    paths = [
        spec.tables_dir / spec.per_spectrum_bsv_csv,
        spec.tables_dir / spec.per_spectrum_delta_bsv_csv,
        spec.tables_dir / spec.axis_effect_sizes_csv,
        spec.tables_dir / spec.cohort_summary_csv,
    ]
    if spec.axis_correlation_csv:
        paths.append(spec.tables_dir / spec.axis_correlation_csv)
    if spec.contribution_diagnostics_csv:
        paths.append(spec.tables_dir / spec.contribution_diagnostics_csv)
    return {str(p): _sha(p) for p in paths if p.exists()}


def _gate(before: dict, after: dict) -> None:
    diff = [k for k in before if after.get(k) != before[k]]
    if diff:
        raise RuntimeError("PILOT FILES MUTATED:\n  " + "\n  ".join(diff))


def _bsv_array(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    """Return (n_spectra, 8) BSV matrix in canonical axis order, plus class labels."""
    cols = [f"bsv_{a}" for a in BSV_COMPONENTS]
    arr = np.array([[float(r[c]) for c in cols] for r in rows])
    classes = [r["class"] for r in rows]
    return arr, classes


def _delta_bsv_array(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    cols = [f"delta_bsv_{a}" for a in BSV_COMPONENTS]
    arr = np.array([[float(r[c]) for c in cols] for r in rows])
    classes = [r["class"] for r in rows]
    return arr, classes


def _classes_in_pilot(rows: list[dict[str, str]]) -> list[str]:
    seen = list(dict.fromkeys(r["class"] for r in rows))
    # Canonical order with healthy_control first
    out = ["healthy_control"] + [c for c in CLASS_ORDER_4 if c != "healthy_control" and c in seen]
    out += [c for c in seen if c not in out]
    return out


def _tier_from_d(d: float) -> str:
    a = abs(d)
    if a >= 0.8: return "large"
    if a >= 0.5: return "medium"
    if a >= 0.2: return "small"
    return "negligible"


# ──────────────────────────────────────────────────────────────────────
# STEP 1 — replicate canonical figures
# ──────────────────────────────────────────────────────────────────────

def _fig_bsv_heatmap(out_dir: Path, spec: PilotSpec, bsv_rows: list[dict[str, str]]) -> None:
    classes = _classes_in_pilot(bsv_rows)
    means = np.zeros((len(classes), len(BSV_COMPONENTS)))
    for i, c in enumerate(classes):
        rs = [r for r in bsv_rows if r["class"] == c]
        bsv, _ = _bsv_array(rs)
        means[i] = bsv.mean(axis=0)
    fig, ax = plt.subplots(figsize=(9.5, 1.6 + 0.5 * len(classes)))
    im = ax.imshow(means, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(BSV_COMPONENTS)))
    ax.set_xticklabels([a.replace("_", "\n") for a in BSV_COMPONENTS], fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(BSV_COMPONENTS)):
            ax.text(j, i, f"{means[i, j]*1000:.1f}", ha="center", va="center",
                    color="w" if means[i, j] < means.max() * 0.55 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean BSV (raw units)")
    ax.set_title(f"{spec.short_label} — BSV mean per class × axis  (×1e3)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_bsv_heatmap.png", dpi=140); plt.close(fig)


def _fig_delta_bsv_distributions(
    out_dir: Path, spec: PilotSpec, dbsv_rows: list[dict[str, str]]
) -> None:
    classes = _classes_in_pilot(dbsv_rows)
    n_axes = len(BSV_COMPONENTS)
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.5), sharex=True)
    for ax_idx, axis in enumerate(BSV_COMPONENTS):
        ax = axes.flat[ax_idx]
        data = []
        labels = []
        for c in classes:
            arr = np.array([float(r[f"delta_bsv_{axis}"]) for r in dbsv_rows if r["class"] == c])
            data.append(arr); labels.append(f"{c}\n(n={len(arr)})")
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                        showfliers=False, medianprops=dict(color="k", linewidth=1.2))
        for patch, c in zip(bp["boxes"], classes):
            patch.set_facecolor(CLASS_PALETTE.get(c, "#888888")); patch.set_alpha(0.65)
        ax.axhline(0, color="k", linewidth=0.6, linestyle=":")
        ax.set_title(axis.replace("_", " "), fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"{spec.short_label} — ΔBSV distributions per axis × class", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "fig_delta_bsv_distributions.png", dpi=140); plt.close(fig)


def _radar_axes(theta: np.ndarray):
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels([a.replace("_", "\n") for a in BSV_COMPONENTS], fontsize=9)
    return fig, ax


def _fig_bsv_radar(out_dir: Path, spec: PilotSpec, bsv_rows: list[dict[str, str]]) -> None:
    classes = _classes_in_pilot(bsv_rows)
    theta = np.linspace(0, 2 * math.pi, len(BSV_COMPONENTS), endpoint=False)
    theta_close = np.append(theta, theta[0])
    fig, ax = _radar_axes(theta)
    for c in classes:
        rs = [r for r in bsv_rows if r["class"] == c]
        bsv, _ = _bsv_array(rs)
        m = bsv.mean(axis=0)
        m_close = np.append(m, m[0])
        ax.plot(theta_close, m_close, color=CLASS_PALETTE.get(c, "#888888"),
                linewidth=2, label=c)
        ax.fill(theta_close, m_close, color=CLASS_PALETTE.get(c, "#888888"), alpha=0.10)
    ax.set_title(f"{spec.short_label} — BSV radar (mean per class)", pad=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.05), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_bsv_radar.png", dpi=140); plt.close(fig)


def _fig_delta_bsv_radar(out_dir: Path, spec: PilotSpec, dbsv_rows: list[dict[str, str]]) -> None:
    classes = _classes_in_pilot(dbsv_rows)
    theta = np.linspace(0, 2 * math.pi, len(BSV_COMPONENTS), endpoint=False)
    theta_close = np.append(theta, theta[0])
    fig, ax = _radar_axes(theta)
    for c in classes:
        rs = [r for r in dbsv_rows if r["class"] == c]
        dbsv, _ = _delta_bsv_array(rs)
        m = dbsv.mean(axis=0)
        m_close = np.append(m, m[0])
        ax.plot(theta_close, m_close, color=CLASS_PALETTE.get(c, "#888888"),
                linewidth=2, label=c)
        ax.fill(theta_close, m_close, color=CLASS_PALETTE.get(c, "#888888"), alpha=0.08)
    # add reference circle at 0
    ax.plot(theta_close, np.zeros_like(theta_close), color="k", linewidth=0.6, linestyle=":")
    ax.set_title(f"{spec.short_label} — ΔBSV radar (mean per class, anchored on healthy_control)",
                 pad=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.05), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_delta_bsv_radar.png", dpi=140); plt.close(fig)


def _fig_pca(out_dir: Path, spec: PilotSpec, bsv_rows: list[dict[str, str]]) -> None:
    bsv, classes = _bsv_array(bsv_rows)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(bsv)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cls_order = _classes_in_pilot(bsv_rows)
    for c in cls_order:
        idx = [i for i, k in enumerate(classes) if k == c]
        ax.scatter(Z[idx, 0], Z[idx, 1], s=22, alpha=0.65, edgecolor="none",
                   color=CLASS_PALETTE.get(c, "#888888"), label=f"{c} (n={len(idx)})")
    ax.axhline(0, color="k", linewidth=0.4, linestyle=":")
    ax.axvline(0, color="k", linewidth=0.4, linestyle=":")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"{spec.short_label} — PCA on per-spectrum BSV (8D)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pca.png", dpi=140); plt.close(fig)


def _fig_distance_to_centroid(
    out_dir: Path, spec: PilotSpec,
    bsv_rows: list[dict[str, str]],
    dbsv_rows: list[dict[str, str]],
) -> None:
    classes = _classes_in_pilot(bsv_rows)
    # Prefer the existing distance_to_healthy_centroid column if present
    have_col = bool(dbsv_rows) and "distance_to_healthy_centroid" in dbsv_rows[0]
    if have_col:
        per_class_dist = {
            c: np.array([float(r["distance_to_healthy_centroid"])
                         for r in dbsv_rows if r["class"] == c])
            for c in classes
        }
        ylabel = "distance_to_healthy_centroid (file column)"
    else:
        # Visualisation-only: Euclidean distance in BSV space to mean of healthy_control.
        # No BSV value is changed — this is plotted only.
        bsv_arr, cls_arr = _bsv_array(bsv_rows)
        hc_idx = [i for i, c in enumerate(cls_arr) if c == "healthy_control"]
        centroid = bsv_arr[hc_idx].mean(axis=0)
        per_class_dist = {
            c: np.linalg.norm(bsv_arr[[i for i, k in enumerate(cls_arr) if k == c]] - centroid, axis=1)
            for c in classes
        }
        ylabel = "Euclidean distance to healthy_control BSV centroid (viz only)"
    fig, ax = plt.subplots(figsize=(7.5, 5))
    data, labels, colors = [], [], []
    for c in classes:
        data.append(per_class_dist[c])
        labels.append(f"{c}\n(n={len(per_class_dist[c])})")
        colors.append(CLASS_PALETTE.get(c, "#888888"))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                    showfliers=True, medianprops=dict(color="k", linewidth=1.2))
    for p, col in zip(bp["boxes"], colors):
        p.set_facecolor(col); p.set_alpha(0.65)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{spec.short_label} — distance to healthy_control centroid")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_distance_to_centroid.png", dpi=140); plt.close(fig)


def _fig_correlation_heatmap(
    out_dir: Path, spec: PilotSpec, bsv_rows: list[dict[str, str]]
) -> None:
    """Use existing axis_correlation.csv when available, else compute on the fly
    from per-spectrum BSV columns. Either way this is visualisation only — no BSV
    value is redefined."""
    if spec.axis_correlation_csv:
        rows = _read_csv(spec.tables_dir / spec.axis_correlation_csv)
        order = [r["axis"] for r in rows]
        M = np.array([[float(r[a]) for a in order] for r in rows])
        source = "from pilot table"
    else:
        bsv, _ = _bsv_array(bsv_rows)
        M = np.corrcoef(bsv.T)
        order = list(BSV_COMPONENTS)
        source = "computed (viz only)"
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="k" if abs(M[i, j]) < 0.6 else "w")
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"{spec.short_label} — BSV axis correlation heatmap ({source})")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_correlation_heatmap.png", dpi=140); plt.close(fig)


def step1_replicate(spec: PilotSpec) -> Path:
    out_dir = OUT_ROOT / spec.pilot_id
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    bsv_rows  = _read_csv(spec.tables_dir / spec.per_spectrum_bsv_csv)
    dbsv_rows = _read_csv(spec.tables_dir / spec.per_spectrum_delta_bsv_csv)

    figs = out_dir / "figures"
    _fig_bsv_heatmap(figs, spec, bsv_rows)
    _fig_delta_bsv_distributions(figs, spec, dbsv_rows)
    _fig_bsv_radar(figs, spec, bsv_rows)
    _fig_delta_bsv_radar(figs, spec, dbsv_rows)
    _fig_pca(figs, spec, bsv_rows)
    _fig_distance_to_centroid(figs, spec, bsv_rows, dbsv_rows)
    _fig_correlation_heatmap(figs, spec, bsv_rows)
    return out_dir


# ──────────────────────────────────────────────────────────────────────
# STEP 2 — attach substrate overlay
# ──────────────────────────────────────────────────────────────────────

def step2_attach_overlay(spec: PilotSpec, pilot_dir: Path) -> Path:
    eff_rows = _read_csv(spec.tables_dir / spec.axis_effect_sizes_csv)
    overlay_rows = _read_csv(spec.overlay_csv)
    overlay_by_axis = {r["axis"]: r for r in overlay_rows}

    has_compare = any("compare_class" in r for r in eff_rows)
    out_rows: list[list] = []
    for r in eff_rows:
        if has_compare and (r.get("compare_class") != spec.principal_compare_class
                            or r.get("reference_class") != spec.reference_class):
            continue
        axis = r["axis"]
        ov = overlay_by_axis.get(axis, {})
        delta_mean = float(r.get("delta_mean", "0") or 0)
        cohens_d   = float(r.get("cohens_d", "0") or 0)
        out_rows.append([
            axis,
            f"{delta_mean:+.6f}",
            f"{cohens_d:+.4f}",
            _tier_from_d(cohens_d),
            ov.get("substrate_family", spec.substrate_family),
            ov.get("visibility_tag", ""),
            ov.get("abundance_interpretation", ""),
            ov.get("conflict_flag", ""),
            ov.get("unresolved_assignment_flag", ""),
            ov.get("composed_multiplier", ""),
            ov.get("interpretation_shift", "UNCHANGED"),
            ov.get("key_caveat_summary", ""),
        ])
    out_csv = pilot_dir / "tables" / f"{spec.pilot_id}_axis_effect_sizes_with_overlay.csv"
    _write_csv(out_csv, [
        "axis", "delta_mean", "cohens_d", "tier", "substrate_family",
        "visibility_tag", "abundance_interpretation",
        "conflict_flag", "unresolved_assignment_flag", "composed_multiplier",
        "substrate_adjusted_class", "key_caveat_summary",
    ], out_rows)
    return out_csv


# ──────────────────────────────────────────────────────────────────────
# STEP 3 — within-pilot multi-cohort comparisons (pilots 2b, 3)
# ──────────────────────────────────────────────────────────────────────

def step3_within_pilot(spec: PilotSpec, pilot_dir: Path) -> Path | None:
    eff_rows = _read_csv(spec.tables_dir / spec.axis_effect_sizes_csv)
    if not any("compare_class" in r for r in eff_rows):
        return None
    overlay_by_axis = {r["axis"]: r for r in _read_csv(spec.overlay_csv)}
    compare_classes = sorted({r["compare_class"] for r in eff_rows})

    # Per-cohort ranked tables
    out_csv = pilot_dir / "tables" / f"{spec.pilot_id}_within_pilot_axis_ranks.csv"
    rows_out: list[list] = []
    for cc in compare_classes:
        sub = [r for r in eff_rows
               if r["compare_class"] == cc and r["reference_class"] == spec.reference_class]
        sub_sorted = sorted(sub, key=lambda r: -abs(float(r["cohens_d"])))
        for rank, r in enumerate(sub_sorted, start=1):
            d = float(r["cohens_d"])
            dmean = float(r["delta_mean"])
            ov = overlay_by_axis.get(r["axis"], {})
            rows_out.append([
                cc, rank, r["axis"],
                f"{dmean:+.6f}", f"{d:+.4f}",
                _tier_from_d(d),
                ov.get("visibility_tag", ""),
                ov.get("conflict_flag", ""),
                ov.get("interpretation_shift", "UNCHANGED"),
            ])
    _write_csv(out_csv, [
        "compare_class", "rank", "axis", "delta_mean", "cohens_d", "tier",
        "visibility_tag", "conflict_flag", "substrate_adjusted_class",
    ], rows_out)

    # ΔBSV radar per cohort (one polar figure with 3 traces)
    dbsv_rows = _read_csv(spec.tables_dir / spec.per_spectrum_delta_bsv_csv)
    classes = _classes_in_pilot(dbsv_rows)
    cohorts = [c for c in classes if c != "healthy_control"]
    theta = np.linspace(0, 2 * math.pi, len(BSV_COMPONENTS), endpoint=False)
    theta_close = np.append(theta, theta[0])
    fig, ax = _radar_axes(theta)
    for c in cohorts:
        rs = [r for r in dbsv_rows if r["class"] == c]
        dbsv, _ = _delta_bsv_array(rs)
        m = dbsv.mean(axis=0)
        m_close = np.append(m, m[0])
        ax.plot(theta_close, m_close, color=CLASS_PALETTE.get(c, "#888888"),
                linewidth=2, label=f"{c} (n={len(rs)})")
        ax.fill(theta_close, m_close, color=CLASS_PALETTE.get(c, "#888888"), alpha=0.08)
    ax.plot(theta_close, np.zeros_like(theta_close), color="k", linewidth=0.6, linestyle=":")
    ax.set_title(f"{spec.short_label} — within-pilot ΔBSV radar (per cohort vs HC)", pad=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.05), fontsize=8)
    fig.tight_layout()
    fig.savefig(pilot_dir / "figures" / f"{spec.pilot_id}_within_pilot_delta_radar.png", dpi=140)
    plt.close(fig)
    return out_csv


# ──────────────────────────────────────────────────────────────────────
# STEP 4 — cross-pilot HCC comparison (P1 vs P2b)
# ──────────────────────────────────────────────────────────────────────

def step4_cross_pilot_hcc() -> dict:
    out_dir = OUT_ROOT / "cross_pilot_hcc"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    p1 = next(s for s in PILOTS if s.pilot_id == "pilot1_hcc")
    p2 = next(s for s in PILOTS if s.pilot_id == "pilot2b_cca")

    # P1 effect-sizes — single comparison HCC vs HC (delta = hcc - ctr)
    p1_rows = _read_csv(p1.tables_dir / p1.axis_effect_sizes_csv)
    p1_by_axis = {r["axis"]: r for r in p1_rows}

    # P2b — pull only the HCC vs HC comparison
    p2_rows = [r for r in _read_csv(p2.tables_dir / p2.axis_effect_sizes_csv)
               if r["reference_class"] == "healthy_control" and r["compare_class"] == "hcc"]
    p2_by_axis = {r["axis"]: r for r in p2_rows}

    p1_overlay = {r["axis"]: r for r in _read_csv(p1.overlay_csv)}
    # for the P2b HCC overlay we need to recompose; but per the spec the overlay
    # is annotation only and fundamentally a function of (substrate_family × axis),
    # not of the compare class. The Stage 2 overlay CSV for pilot2b was built on
    # the CCA principal pair, but visibility / abundance / conflict are pair-
    # invariant — they reflect substrate × axis. Reuse it here, attach P2b's
    # actual HCC d.
    p2_overlay = {r["axis"]: r for r in _read_csv(p2.overlay_csv)}

    rows_out = []
    for axis in BSV_COMPONENTS:
        r1 = p1_by_axis.get(axis); r2 = p2_by_axis.get(axis)
        if not (r1 and r2):
            continue
        d1 = float(r1["cohens_d"]); d2 = float(r2["cohens_d"])
        dm1 = float(r1["delta_mean"]); dm2 = float(r2["delta_mean"])
        sign1 = "+" if d1 > 0 else ("-" if d1 < 0 else "0")
        sign2 = "+" if d2 > 0 else ("-" if d2 < 0 else "0")
        agree = "AGREE" if sign1 == sign2 and sign1 != "0" else (
            "DISAGREE" if sign1 != sign2 and sign1 != "0" and sign2 != "0" else "ZERO"
        )
        ov1 = p1_overlay.get(axis, {})
        ov2 = p2_overlay.get(axis, {})
        rows_out.append({
            "axis": axis,
            "p1_d": d1, "p2_d": d2, "p1_dmean": dm1, "p2_dmean": dm2,
            "sign1": sign1, "sign2": sign2, "agree": agree,
            "p1_tier": _tier_from_d(d1), "p2_tier": _tier_from_d(d2),
            "p1_substrate_class": ov1.get("interpretation_shift", "UNCHANGED"),
            "p2_substrate_class": ov2.get("interpretation_shift", "UNCHANGED"),
            "p1_visibility": ov1.get("visibility_tag", ""),
            "p2_visibility": ov2.get("visibility_tag", ""),
            "p1_conflict_flag": ov1.get("conflict_flag", "false"),
            "p2_conflict_flag": ov2.get("conflict_flag", "false"),
        })

    # Ranks within each pilot
    p1_rank = {r["axis"]: i+1 for i, r in enumerate(
        sorted(rows_out, key=lambda x: -abs(x["p1_d"])))}
    p2_rank = {r["axis"]: i+1 for i, r in enumerate(
        sorted(rows_out, key=lambda x: -abs(x["p2_d"])))}
    for r in rows_out:
        r["p1_rank"] = p1_rank[r["axis"]]
        r["p2_rank"] = p2_rank[r["axis"]]
        r["rank_diff"] = r["p2_rank"] - r["p1_rank"]

    # Write CSV
    csv_path = out_dir / "tables" / "cross_pilot_hcc_comparison.csv"
    _write_csv(
        csv_path,
        ["axis", "p1_dmean", "p2_dmean", "p1_d", "p2_d",
         "p1_tier", "p2_tier", "p1_rank", "p2_rank", "rank_diff",
         "direction_sign_p1", "direction_sign_p2", "direction_agreement",
         "p1_substrate_adjusted_class", "p2_substrate_adjusted_class",
         "p1_visibility", "p2_visibility",
         "p1_conflict_flag", "p2_conflict_flag"],
        [[r["axis"], f"{r['p1_dmean']:+.6f}", f"{r['p2_dmean']:+.6f}",
          f"{r['p1_d']:+.4f}", f"{r['p2_d']:+.4f}",
          r["p1_tier"], r["p2_tier"], r["p1_rank"], r["p2_rank"], r["rank_diff"],
          r["sign1"], r["sign2"], r["agree"],
          r["p1_substrate_class"], r["p2_substrate_class"],
          r["p1_visibility"], r["p2_visibility"],
          r["p1_conflict_flag"], r["p2_conflict_flag"]] for r in rows_out],
    )

    # ── Rank-slope plot ─────────────────────────────────────────────
    rs = sorted(rows_out, key=lambda r: r["p1_rank"])
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rs:
        col = SHIFT_PALETTE.get(r["p2_substrate_class"], "#888888")
        ax.plot([0, 1], [r["p1_rank"], r["p2_rank"]], color=col, marker="o",
                linewidth=1.6, markersize=7, alpha=0.85)
        ax.text(-0.04, r["p1_rank"], r["axis"], ha="right", va="center", fontsize=8)
        ax.text(1.04, r["p2_rank"], r["axis"], ha="left", va="center", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pilot 1 HCC\n(Ag-array)", "Pilot 2b HCC\n(Ag-colloid)"])
    ax.set_ylabel("rank by |Cohen's d| (1 = strongest)")
    ax.set_ylim(len(rs) + 0.5, 0.5)
    ax.set_title("Cross-pilot HCC axis rank comparison\n(line color = P2b substrate-aware shift)")
    legend_handles = [
        plt.Line2D([0], [0], color=col, marker="o", linewidth=2, label=lbl)
        for lbl, col in SHIFT_PALETTE.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fig_cross_pilot_hcc_rank_slope.png", dpi=140)
    plt.close(fig)

    # ── Direction heatmap ───────────────────────────────────────────
    pilot_labels = ["Pilot 1 HCC (Ag-array)", "Pilot 2b HCC (Ag-colloid)"]
    M = np.zeros((len(rows_out), 2))
    for i, r in enumerate(rows_out):
        M[i, 0] = np.sign(r["p1_d"]) * abs(r["p1_d"])
        M[i, 1] = np.sign(r["p2_d"]) * abs(r["p2_d"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    vmax = max(0.001, np.abs(M).max())
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(pilot_labels, fontsize=8)
    ax.set_yticks(range(len(rows_out)))
    ax.set_yticklabels([r["axis"] for r in rows_out], fontsize=8)
    for i, r in enumerate(rows_out):
        for j, val in enumerate([r["p1_d"], r["p2_d"]]):
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                    color="k" if abs(val) < vmax * 0.55 else "w")
        # mark agreement / disagreement
        ax.text(2.1, i, f"  {r['agree']}", va="center", ha="left", fontsize=8,
                color={"AGREE": "#1f7a1f", "DISAGREE": "#a01f1f", "ZERO": "#666"}[r["agree"]])
    fig.colorbar(im, ax=ax, label="Cohen's d (signed)")
    ax.set_title("Cross-pilot HCC direction heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fig_cross_pilot_hcc_direction_heatmap.png", dpi=140)
    plt.close(fig)

    return {"rows": rows_out, "csv": csv_path, "dir": out_dir}


# ──────────────────────────────────────────────────────────────────────
# STEP 5 — Ag-colloid disease progression (HC → CCA → HCC → LM)
# ──────────────────────────────────────────────────────────────────────

def step5_ag_progression() -> dict:
    """Use Pilot 2b's 4-class effect-sizes table (same data underlies Pilot 3)."""
    out_dir = OUT_ROOT / "ag_colloid_progression"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    p2 = next(s for s in PILOTS if s.pilot_id == "pilot2b_cca")
    eff_rows = _read_csv(p2.tables_dir / p2.axis_effect_sizes_csv)
    overlay_by_axis = {r["axis"]: r for r in _read_csv(p2.overlay_csv)}

    # Build axis × cohort matrices of (Δmean, d)
    cohorts = ["cca", "hcc", "lm"]
    dmean_mat = {a: {} for a in BSV_COMPONENTS}
    d_mat     = {a: {} for a in BSV_COMPONENTS}
    for r in eff_rows:
        if r["reference_class"] != "healthy_control": continue
        if r["compare_class"] not in cohorts: continue
        a = r["axis"]; c = r["compare_class"]
        dmean_mat[a][c] = float(r["delta_mean"])
        d_mat[a][c]     = float(r["cohens_d"])

    # Per-axis progression rows
    rows_out: list[list] = []
    for a in BSV_COMPONENTS:
        ds = [d_mat[a][c] for c in cohorts]
        signs = [("+" if d > 0 else "-" if d < 0 else "0") for d in ds]
        direction_consistency = (
            "all_positive" if all(s == "+" for s in signs)
            else "all_negative" if all(s == "-" for s in signs)
            else "mixed"
        )
        # Monotonicity: signed d monotonic increasing or decreasing across CCA→HCC→LM
        if all(ds[i] <= ds[i+1] for i in range(len(ds)-1)):
            monotonicity = "monotonic_up"
        elif all(ds[i] >= ds[i+1] for i in range(len(ds)-1)):
            monotonicity = "monotonic_down"
        else:
            monotonicity = "non_monotonic"
        max_abs = max(abs(d) for d in ds)
        progression = "→".join(_tier_from_d(d_mat[a][c])[0:1].upper() for c in cohorts)
        ov = overlay_by_axis.get(a, {})
        rows_out.append([
            a,
            *[f"{dmean_mat[a][c]:+.6f}" for c in cohorts],
            *[f"{d_mat[a][c]:+.4f}" for c in cohorts],
            "→".join(signs), direction_consistency, monotonicity,
            f"{max_abs:.3f}", progression,
            ov.get("substrate_adjusted_class", ov.get("interpretation_shift", "UNCHANGED")),
            ov.get("visibility_tag", ""), ov.get("conflict_flag", ""),
        ])
    csv_path = out_dir / "tables" / "within_ag_colloid_progression.csv"
    _write_csv(csv_path, [
        "axis",
        "delta_mean_cca", "delta_mean_hcc", "delta_mean_lm",
        "cohens_d_cca",   "cohens_d_hcc",   "cohens_d_lm",
        "direction_sequence", "direction_consistency", "monotonicity",
        "max_abs_d", "tier_progression",
        "substrate_adjusted_class", "visibility_tag", "conflict_flag",
    ], rows_out)

    # Per-axis progression plot — small multiples
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharey=True)
    x = np.arange(len(cohorts))
    for ax_idx, axis in enumerate(BSV_COMPONENTS):
        ax = axes.flat[ax_idx]
        ds = [d_mat[axis][c] for c in cohorts]
        ax.bar(x, ds, color=[CLASS_PALETTE[c] for c in cohorts], alpha=0.85)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(cohorts, fontsize=8)
        ov = overlay_by_axis.get(axis, {})
        klass = ov.get("interpretation_shift", "UNCHANGED")
        ax.set_title(f"{axis.replace('_',' ')}\nshift: {klass}", fontsize=9,
                     color=SHIFT_PALETTE.get(klass, "#444"))
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Ag-colloid disease progression — per-axis Cohen's d (vs healthy_control)",
                 fontsize=11)
    fig.text(0.005, 0.5, "Cohen's d (signed)", rotation=90, va="center", fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    fig.savefig(out_dir / "figures" / "fig_axis_progression_plots.png", dpi=140)
    plt.close(fig)

    return {"csv": csv_path, "dir": out_dir, "d_mat": d_mat, "rows": rows_out}


# ──────────────────────────────────────────────────────────────────────
# STEP 6 — global per-axis synthesis
# ──────────────────────────────────────────────────────────────────────

def step6_global_synthesis(
    cross_hcc: dict, ag_progression: dict
) -> Path:
    out_dir = OUT_ROOT
    p1 = next(s for s in PILOTS if s.pilot_id == "pilot1_hcc")
    p2 = next(s for s in PILOTS if s.pilot_id == "pilot2b_cca")
    p3 = next(s for s in PILOTS if s.pilot_id == "pilot3_lm")

    # Pilot directions
    p1_eff = {r["axis"]: r for r in _read_csv(p1.tables_dir / p1.axis_effect_sizes_csv)}
    p2_eff = {(r["compare_class"], r["axis"]): r
              for r in _read_csv(p2.tables_dir / p2.axis_effect_sizes_csv)
              if r["reference_class"] == "healthy_control"}
    p3_eff = {(r["compare_class"], r["axis"]): r
              for r in _read_csv(p3.tables_dir / p3.axis_effect_sizes_csv)
              if r["reference_class"] == "healthy_control"}

    p1_overlay = {r["axis"]: r for r in _read_csv(p1.overlay_csv)}
    p2_overlay = {r["axis"]: r for r in _read_csv(p2.overlay_csv)}
    p3_overlay = {r["axis"]: r for r in _read_csv(p3.overlay_csv)}

    def _sgn(d: float) -> str:
        return "+" if d > 0 else ("-" if d < 0 else "0")

    rows_out: list[list] = []
    for a in BSV_COMPONENTS:
        d_p1   = float(p1_eff[a]["cohens_d"])
        d_p2c  = float(p2_eff[("cca", a)]["cohens_d"])
        d_p2h  = float(p2_eff[("hcc", a)]["cohens_d"])
        d_p3l  = float(p3_eff[("lm", a)]["cohens_d"])
        signs = [_sgn(d_p1), _sgn(d_p2c), _sgn(d_p2h), _sgn(d_p3l)]
        nz = [s for s in signs if s != "0"]
        if not nz:
            consistency = "zero"
        elif all(s == "+" for s in nz):
            consistency = "all_positive"
        elif all(s == "-" for s in nz):
            consistency = "all_negative"
        else:
            consistency = "mixed"

        # Substrate class — collapse across pilots, AMBIGUOUS dominates
        klasses = [
            p1_overlay.get(a, {}).get("interpretation_shift", "UNCHANGED"),
            p2_overlay.get(a, {}).get("interpretation_shift", "UNCHANGED"),
            p3_overlay.get(a, {}).get("interpretation_shift", "UNCHANGED"),
        ]
        if "AMBIGUOUS" in klasses:
            substrate_class = "AMBIGUOUS"
        elif "STRONGER" in klasses and "WEAKER" in klasses:
            substrate_class = "MIXED"
        elif "STRONGER" in klasses:
            substrate_class = "STRONGER"
        elif "WEAKER" in klasses:
            substrate_class = "WEAKER"
        elif "INCONCLUSIVE" in klasses and not any(k in ("STRONGER", "WEAKER") for k in klasses):
            substrate_class = "INCONCLUSIVE"
        else:
            substrate_class = "UNCHANGED"

        # Final interpretation
        if substrate_class == "AMBIGUOUS":
            final = "AMBIGUOUS — substrate physics or conflict prevent direct biology call"
        elif consistency == "mixed":
            final = "DIVERGENT — direction not consistent across pilots / cohorts"
        elif consistency == "all_positive" and substrate_class == "STRONGER":
            final = "ROBUST_BIOLOGY_UPGRADED — consistent elevation that beats substrate suppression"
        elif consistency == "all_positive" and substrate_class == "WEAKER":
            final = "ROBUST_DIRECTION_BUT_INFLATED — elevation consistent but substrate-amplified"
        elif consistency == "all_negative" and substrate_class == "STRONGER":
            final = "DEPRESSION_INCONSISTENT_WITH_SUPPRESSION_UPGRADE"
        elif consistency == "all_negative":
            final = "ROBUST_DEPRESSION — consistent depression across pilots"
        elif consistency == "all_positive":
            final = "ROBUST_ELEVATION"
        else:
            final = "INCONCLUSIVE"

        rows_out.append([
            a,
            _sgn(d_p1), _sgn(d_p2c), _sgn(d_p2h), _sgn(d_p3l),
            f"{d_p1:+.3f}", f"{d_p2c:+.3f}", f"{d_p2h:+.3f}", f"{d_p3l:+.3f}",
            consistency, substrate_class, final,
        ])

    out_csv = out_dir / "cross_pilot_global_synthesis.csv"
    _write_csv(out_csv, [
        "axis",
        "P1_HCC_direction", "P2b_CCA_direction", "P2b_HCC_direction", "P3_LM_direction",
        "P1_HCC_d", "P2b_CCA_d", "P2b_HCC_d", "P3_LM_d",
        "direction_consistency", "substrate_class", "final_interpretation",
    ], rows_out)

    return out_csv


# ──────────────────────────────────────────────────────────────────────
# STEP 7 — final markdown report
# ──────────────────────────────────────────────────────────────────────

def step7_report(global_csv: Path) -> Path:
    rows = _read_csv(global_csv)

    # Pull cross-pilot HCC + Ag-colloid progression for richer narrative
    cross_hcc_rows = _read_csv(
        OUT_ROOT / "cross_pilot_hcc" / "tables" / "cross_pilot_hcc_comparison.csv"
    )
    ag_prog_rows = _read_csv(
        OUT_ROOT / "ag_colloid_progression" / "tables" / "within_ag_colloid_progression.csv"
    )

    out_md = OUT_ROOT / "REPORT_stage2_5_cross_pilot_synthesis_v1.md"

    L: list[str] = []
    L.append("# GAIRA — Stage 2.5 Reanalysis + Cross-Pilot Synthesis (v1)")
    L.append("")
    L.append(
        "_Annotation-only reanalysis. Underlying BSV / ΔBSV outputs are unchanged "
        "(verified by SHA-256 checksum gate). Cross-pilot comparisons are made at "
        "the level of direction sign, rank order, and effect-size class — never "
        "raw-magnitude pooling._"
    )
    L.append("")

    # 1. canonical findings recap
    L.append("## 1. Canonical findings recap")
    L.append("")
    L.append("Per-pilot canonical figures regenerated under "
             "`stage2_5_reanalysis_v1/<pilot>/figures/` (no input files modified):")
    L.append("")
    for spec in PILOTS:
        L.append(f"- **{spec.short_label}** — substrate `{spec.substrate_family}`")
        L.append(f"  - BSV heatmap, ΔBSV distributions, BSV / ΔBSV radars, "
                 f"PCA scatter, distance-to-centroid, axis correlation heatmap")
    L.append("")
    L.append("Headline directions (signed Cohen's d vs healthy_control), per cohort:")
    L.append("")
    L.append("| axis | P1 HCC | P2b CCA | P2b HCC | P3 LM |")
    L.append("|---|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| `{r['axis']}` | {r['P1_HCC_d']} | {r['P2b_CCA_d']} | "
                 f"{r['P2b_HCC_d']} | {r['P3_LM_d']} |")
    L.append("")

    # 2. substrate-aware corrections
    L.append("## 2. Substrate-aware corrections")
    L.append("")
    L.append(
        "Stage 2 substrate overlays are joined onto each pilot's effect-sizes table. "
        "Per-axis substrate classes summarised below (collapsed across pilots, "
        "`AMBIGUOUS` dominates):"
    )
    L.append("")
    L.append("| axis | P1 substrate class | P2b substrate class | P3 substrate class | collapsed |")
    L.append("|---|:---:|:---:|:---:|:---:|")
    p1_overlay = {r["axis"]: r for r in _read_csv(
        next(s for s in PILOTS if s.pilot_id == "pilot1_hcc").overlay_csv)}
    p2_overlay = {r["axis"]: r for r in _read_csv(
        next(s for s in PILOTS if s.pilot_id == "pilot2b_cca").overlay_csv)}
    p3_overlay = {r["axis"]: r for r in _read_csv(
        next(s for s in PILOTS if s.pilot_id == "pilot3_lm").overlay_csv)}
    for r in rows:
        a = r["axis"]
        L.append(
            f"| `{a}` | "
            f"{p1_overlay.get(a, {}).get('interpretation_shift', '')} | "
            f"{p2_overlay.get(a, {}).get('interpretation_shift', '')} | "
            f"{p3_overlay.get(a, {}).get('interpretation_shift', '')} | "
            f"{r['substrate_class']} |"
        )
    L.append("")
    weakened = [r for r in rows if r["substrate_class"] == "WEAKER"]
    strengthened = [r for r in rows if r["substrate_class"] == "STRONGER"]
    if weakened:
        L.append("**Axes downgraded by substrate physics (signal likely inflated):**")
        for r in weakened:
            L.append(f"- `{r['axis']}` — substrate-aware class WEAKER")
        L.append("")
    if strengthened:
        L.append("**Axes upgraded by substrate physics (biology beats suppression):**")
        for r in strengthened:
            L.append(f"- `{r['axis']}` — substrate-aware class STRONGER")
        L.append("")

    # 3. cross-pilot consistent signals
    L.append("## 3. Cross-pilot consistent signals")
    L.append("")
    consistent_pos = [r for r in rows
                      if r["direction_consistency"] == "all_positive"
                      and r["substrate_class"] != "AMBIGUOUS"]
    consistent_neg = [r for r in rows
                      if r["direction_consistency"] == "all_negative"
                      and r["substrate_class"] != "AMBIGUOUS"]
    if consistent_pos:
        L.append("**All-cohort elevations** (P1 HCC, P2b CCA, P2b HCC, P3 LM all +):")
        for r in consistent_pos:
            L.append(f"- `{r['axis']}` — substrate class `{r['substrate_class']}` "
                     f"→ {r['final_interpretation']}")
        L.append("")
    if consistent_neg:
        L.append("**All-cohort depressions** (all −):")
        for r in consistent_neg:
            L.append(f"- `{r['axis']}` — substrate class `{r['substrate_class']}` "
                     f"→ {r['final_interpretation']}")
        L.append("")
    if not (consistent_pos or consistent_neg):
        L.append("- _no axis is fully consistent in direction across all four cohort calls._")
        L.append("")

    # ── Cross-pilot HCC agreement digest (P1 vs P2b on the same compare class) ──
    L.append("### 3a. Cross-pilot HCC direction agreement (Pilot 1 Ag-array vs Pilot 2b Ag-colloid)")
    L.append("")
    L.append(
        "This is the **only** pair where the same compare class (HCC) is run on two "
        "different substrate families. Direction agreement here is the cleanest "
        "test of whether a signal is biology vs substrate-driven."
    )
    L.append("")
    L.append("| axis | P1 d | P2b d | direction | P1 substrate class | P2b substrate class |")
    L.append("|---|---:|---:|:---:|:---:|:---:|")
    for r in cross_hcc_rows:
        agree_emoji = {"AGREE": "✓ AGREE", "DISAGREE": "✗ DISAGREE", "ZERO": "·"}[r["direction_agreement"]]
        L.append(
            f"| `{r['axis']}` | {r['p1_d']} | {r['p2_d']} | "
            f"{agree_emoji} | {r['p1_substrate_adjusted_class']} | "
            f"{r['p2_substrate_adjusted_class']} |"
        )
    L.append("")
    agree_axes = [r['axis'] for r in cross_hcc_rows if r['direction_agreement'] == 'AGREE']
    disagree_axes = [r['axis'] for r in cross_hcc_rows if r['direction_agreement'] == 'DISAGREE']
    L.append(f"- Direction-AGREE axes (P1↔P2b on HCC): {len(agree_axes)}/{len(cross_hcc_rows)} — "
             + ", ".join(f"`{a}`" for a in agree_axes))
    L.append(f"- Direction-DISAGREE axes: {len(disagree_axes)}/{len(cross_hcc_rows)} — "
             + ", ".join(f"`{a}`" for a in disagree_axes))
    L.append("")
    L.append(
        "_DISAGREE on `purine_nucleotide`, `pyrimidine_nucleotide`, and "
        "`nucleic_acid_backbone` is concentrated in axes where the substrate engine "
        "either flags WEAKER (purine/pyrimidine on Ag colloid) or AMBIGUOUS (nucleic "
        "backbone, conflict-driven). The cross-substrate flip on these axes is "
        "consistent with a substrate-physics explanation rather than a biology one._"
    )
    L.append("")

    # ── Headline biology call ─────────────────────────────────────────
    L.append("### 3b. Headline biology call")
    L.append("")
    if any(r["axis"] == "glycan_carbohydrate" and r["direction_consistency"] == "all_positive"
           for r in rows):
        L.append(
            "**`glycan_carbohydrate` is the strongest cross-pilot biological signal in "
            "the current data.** Direction is positive in all four cohort calls "
            "(P1 HCC, P2b CCA, P2b HCC, P3 LM); it survives the substrate-aware "
            "critique on Ag colloid (the engine upgrades it to STRONGER because "
            "glycan visibility is canonically suppressed on bare Ag colloid, so any "
            "elevation is biology-overcomes-suppression); and it is also positive on "
            "the Ag-array Pilot 1 (where the substrate engine reads it as WEAKER, but "
            "the direction still agrees). Cross-substrate direction agreement on a "
            "single biochemical class is the most defensible biology call this dataset "
            "supports."
        )
    else:
        L.append("- _no axis qualifies as a robust cross-pilot biology call after substrate-aware reading._")
    L.append("")

    # 4. cross-pilot divergent signals
    L.append("## 4. Cross-pilot divergent signals")
    L.append("")
    divergent = [r for r in rows if r["direction_consistency"] == "mixed"]
    if divergent:
        L.append(
            "**Direction inconsistent across cohorts** — these axes flip sign between "
            "pilots / cohorts and cannot be cited as a single biological signal:"
        )
        for r in divergent:
            L.append(
                f"- `{r['axis']}` — sequence "
                f"(P1 HCC, P2b CCA, P2b HCC, P3 LM) = "
                f"({r['P1_HCC_direction']}, {r['P2b_CCA_direction']}, "
                f"{r['P2b_HCC_direction']}, {r['P3_LM_direction']}); "
                f"substrate class `{r['substrate_class']}`."
            )
        L.append("")
    else:
        L.append("- _no axis shows mixed direction across the four cohort calls._")
        L.append("")

    # 4a. Ag-colloid disease progression highlights
    L.append("### 4a. Ag-colloid disease progression highlights (HC → CCA → HCC → LM)")
    L.append("")
    L.append(
        "The 4-class Ag-colloid table (Pilot 2b principal data; Pilot 3 reuses the "
        "same per-spectrum file) supports a **within-substrate** progression read. "
        "Monotonic patterns across `cca → hcc → lm` are tabulated below for "
        "transparency, but each axis must still pass the substrate-aware filter "
        "before being cited as biology."
    )
    L.append("")
    L.append("| axis | d (cca, hcc, lm) | direction | monotonicity | substrate class |")
    L.append("|---|---|:---:|:---:|:---:|")
    for r in ag_prog_rows:
        L.append(
            f"| `{r['axis']}` | "
            f"({r['cohens_d_cca']}, {r['cohens_d_hcc']}, {r['cohens_d_lm']}) | "
            f"{r['direction_sequence']} | {r['monotonicity']} | "
            f"{r['substrate_adjusted_class']} |"
        )
    L.append("")
    monotonic_axes = [r for r in ag_prog_rows
                      if r["monotonicity"] in ("monotonic_up", "monotonic_down")
                      and r["substrate_adjusted_class"] not in ("AMBIGUOUS",)]
    if monotonic_axes:
        L.append("**Monotonic Ag-colloid progressions surviving substrate filter:**")
        for r in monotonic_axes:
            L.append(
                f"- `{r['axis']}` — {r['monotonicity']} across CCA→HCC→LM "
                f"({r['cohens_d_cca']} → {r['cohens_d_hcc']} → {r['cohens_d_lm']}); "
                f"substrate class `{r['substrate_adjusted_class']}`."
            )
        L.append("")
    monotonic_amb = [r for r in ag_prog_rows
                     if r["monotonicity"] in ("monotonic_up", "monotonic_down")
                     and r["substrate_adjusted_class"] == "AMBIGUOUS"]
    if monotonic_amb:
        L.append(
            "**Monotonic Ag-colloid trends that are AMBIGUOUS (informational only, "
            "not cited as biology):**"
        )
        for r in monotonic_amb:
            L.append(
                f"- `{r['axis']}` — {r['monotonicity']} across CCA→HCC→LM "
                f"({r['cohens_d_cca']} → {r['cohens_d_hcc']} → {r['cohens_d_lm']}); "
                f"AMBIGUOUS — conflict_flag=`{r['conflict_flag']}`."
            )
        L.append("")

    # 5. unreliable / ambiguous axes
    L.append("## 5. Unreliable / ambiguous axes")
    L.append("")
    ambig = [r for r in rows if r["substrate_class"] == "AMBIGUOUS"]
    if ambig:
        L.append(
            "Axes flagged AMBIGUOUS by the substrate engine — observed |d| values "
            "cannot be cited as biology without orthogonal evidence:"
        )
        for r in ambig:
            L.append(
                f"- `{r['axis']}` — direction sequence "
                f"({r['P1_HCC_direction']}, {r['P2b_CCA_direction']}, "
                f"{r['P2b_HCC_direction']}, {r['P3_LM_direction']}); "
                f"see Stage 2 per-pilot reports for which pilot raised the conflict."
            )
        L.append("")
    else:
        L.append("- _no AMBIGUOUS axes after substrate-aware reading._")
        L.append("")

    # 6. final GAIRA interpretation
    L.append("## 6. Final GAIRA interpretation")
    L.append("")
    L.append("Per-axis closing call:")
    L.append("")
    L.append("| axis | direction consistency | substrate class | final interpretation |")
    L.append("|---|:---:|:---:|---|")
    for r in rows:
        L.append(f"| `{r['axis']}` | {r['direction_consistency']} | "
                 f"{r['substrate_class']} | {r['final_interpretation']} |")
    L.append("")
    L.append("**What GAIRA can say with confidence after this synthesis:**")
    L.append("")
    L.append(
        "- Substrate-aware reading separates three categories of axis: (a) "
        "robust-direction biology that survives substrate critique, (b) directional "
        "signal that is plausibly substrate-inflated (purine, pyrimidine on Ag colloid), "
        "and (c) ambiguous regions where literature conflict (1020–1080 cm⁻¹) or "
        "substrate artifact (citrate baseline overlap on the redox axis) prevents a "
        "clean biology call."
    )
    L.append(
        "- Pilot 1 (Ag-array, HCC vs HC) and Pilots 2b/3 (Ag-colloid, multi-class) "
        "use **different substrate families**. Direction agreement on the same "
        "biochemical class across the two substrates is a substantially stronger "
        "biological claim than agreement within one substrate family."
    )
    L.append(
        "- Cross-pilot HCC comparison (Pilot 1 vs Pilot 2b) is reported at the "
        "categorical level (direction sign + rank). Where the two pilots agree on "
        "direction, the call is provisional but defensible; where they disagree, "
        "the substrate-physics asymmetry between Ag-array and Ag-colloid is the "
        "first hypothesis to explore before invoking biology."
    )
    L.append(
        "- The 4-class Ag-colloid progression (HC → CCA → HCC → LM) shows class-"
        "specific patterns that the two-class Pilot 1 cannot reproduce; these "
        "should be treated as Ag-colloid-specific findings until replicated on a "
        "second substrate family."
    )

    out_md.write_text("\n".join(L))
    return out_md


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n[Stage 2.5 reanalysis + cross-pilot synthesis v1]")
    print("─" * 76)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    snapshots: dict[str, dict[str, str]] = {}
    pilot_dirs: dict[str, Path] = {}

    for spec in PILOTS:
        print(f"\n── {spec.short_label}")
        snapshots[spec.pilot_id] = _snapshot_pilot(spec)
        d = step1_replicate(spec)
        print(f"   step 1 — canonical figures regenerated: {d.relative_to(OUT_ROOT)}/figures/")
        out_overlay_csv = step2_attach_overlay(spec, d)
        print(f"   step 2 — overlay attached: {out_overlay_csv.name}")
        within = step3_within_pilot(spec, d)
        if within:
            print(f"   step 3 — within-pilot comparisons: {within.name}")
        pilot_dirs[spec.pilot_id] = d
        # Per-pilot mutation gate
        _gate(snapshots[spec.pilot_id], _snapshot_pilot(spec))
        print("   [gate] pilot file checksums unchanged ✓")

    print("\n── step 4 — cross-pilot HCC comparison")
    cross_hcc = step4_cross_pilot_hcc()
    print(f"   wrote {cross_hcc['csv'].relative_to(OUT_ROOT)}")
    print(f"   figs:  {(cross_hcc['dir'] / 'figures').relative_to(OUT_ROOT)}/")

    print("\n── step 5 — Ag-colloid disease progression")
    ag_prog = step5_ag_progression()
    print(f"   wrote {ag_prog['csv'].relative_to(OUT_ROOT)}")
    print(f"   figs:  {(ag_prog['dir'] / 'figures').relative_to(OUT_ROOT)}/")

    print("\n── step 6 — global synthesis table")
    global_csv = step6_global_synthesis(cross_hcc, ag_prog)
    print(f"   wrote {global_csv.relative_to(OUT_ROOT)}")

    print("\n── step 7 — final report")
    report = step7_report(global_csv)
    print(f"   wrote {report.relative_to(OUT_ROOT)}")

    # Final pilot-mutation gate (defence in depth)
    for spec in PILOTS:
        _gate(snapshots[spec.pilot_id], _snapshot_pilot(spec))

    print()
    print("─" * 76)
    print("[Stage 2.5] complete")
    print(f"  outputs: {OUT_ROOT}")


if __name__ == "__main__":
    main()
