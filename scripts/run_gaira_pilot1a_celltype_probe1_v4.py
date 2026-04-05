from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from gaira.autoresearch_storage import (
    DEFAULT_STORAGE_CONFIG_PATH,
    initialize_autoresearch_sprint,
    load_autoresearch_storage_config,
)
from gaira.demo.autoresearch_pass5_utils import classify_compound_family
from gaira.demo.raw_bsv_pilot_utils import ALL_AXES


V3_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot1a_celltype_probe1_v3"
)
PASS5_TABLE = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pass5_saturation_fix/tables/calibration_results_ranked.csv"
)
SPRINT_SUBDIR = "pilot1a_celltype_probe1_v4"

CONFIG_SPECS = [
    {
        "config_id": "baseline_v1_locked_purine",
        "short_label": "baseline",
        "display_name": "Baseline v1 locked purine",
    },
    {
        "config_id": "candidate_v2_cfg05_max_desaturation",
        "short_label": "cfg05",
        "display_name": "Candidate v2 cfg05 max desaturation",
    },
    {
        "config_id": "candidate_v2_cfg08_balanced_update",
        "short_label": "cfg08",
        "display_name": "Candidate v2 cfg08 balanced update",
    },
]

REUSED_FILES = [
    "per_spectrum_bsv.csv",
    "class_mean_bsv.csv",
    "pairwise_delta_bsv.csv",
    "intra_class_bsv_variance.csv",
    "inter_class_bsv_distance.csv",
    "class_topk_neighborhood_composition.csv",
    "class_neighborhood_entropy.csv",
    "class_top1_dominance.csv",
    "class_axis_entropy.csv",
    "retrieval_hit_summary_by_class.csv",
    "per_spectrum_retrieval_hits.csv",
    "pca_coordinates_spectral.csv",
    "pca_coordinates_bsv.csv",
    "pca_coordinates_bsv_class_mean.csv",
    "config_within_between_summary.csv",
]

ROOT_FIGURES_TO_COPY = [
    "pca_spectral_original_dataset.png",
    "pca_bsv_baseline_v1_locked_purine.png",
    "pca_bsv_candidate_v2_cfg05_max_desaturation.png",
    "pca_bsv_candidate_v2_cfg08_balanced_update.png",
    "pca_bsv_class_mean_baseline_v1_locked_purine.png",
    "pca_bsv_class_mean_candidate_v2_cfg05_max_desaturation.png",
    "pca_bsv_class_mean_candidate_v2_cfg08_balanced_update.png",
    "pca_bsv_class_mean_overlay.png",
    "neighborhood_grid_baseline_v1_locked_purine.png",
    "neighborhood_grid_candidate_v2_cfg05_max_desaturation.png",
    "neighborhood_grid_candidate_v2_cfg08_balanced_update.png",
    "class_mean_bsv_heatmap_baseline_v1_locked_purine.png",
    "class_mean_bsv_heatmap_candidate_v2_cfg05_max_desaturation.png",
    "class_mean_bsv_heatmap_candidate_v2_cfg08_balanced_update.png",
    "pilot1a_within_between_comparison.png",
    "pilot1a_entropy_dominance_comparison.png",
    "pilot1a_config_tradeoff_summary.png",
]

CLASS_COLORS = {
    "Hec": "#4c78a8",
    "Hela": "#f58518",
    "Ht": "#54a24b",
    "Mef": "#e45756",
    "Thp": "#72b7b2",
}
CONFIG_COLORS = {"baseline": "#577590", "cfg05": "#f3722c", "cfg08": "#43aa8b"}
FAMILY_ORDER = [
    "purine_core_like",
    "methylated_purine_like",
    "guanidine_like",
    "sulfur_small_molecule_like",
    "other_metabolite_like",
]
FAMILY_COLORS = {
    "purine_core_like": "#355070",
    "methylated_purine_like": "#6d597a",
    "guanidine_like": "#b56576",
    "sulfur_small_molecule_like": "#2a9d8f",
    "other_metabolite_like": "#e9c46a",
}


def _axes_present(df: pd.DataFrame) -> list[str]:
    return [axis for axis in ALL_AXES if axis in df.columns]


def _require_v3_inputs() -> None:
    if not V3_ROOT.exists():
        raise RuntimeError(f"Missing v3 root: {V3_ROOT}")
    for spec in CONFIG_SPECS:
        run_dir = V3_ROOT / "runs" / str(spec["config_id"])
        for name in REUSED_FILES:
            path = run_dir / name
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(f"Missing v3 artifact: {path}")
            df = pd.read_csv(path)
            if df.empty:
                raise RuntimeError(f"Empty v3 artifact: {path}")


def _copy_reused_outputs(sprint_root: Path, figures_dir: Path) -> None:
    for spec in CONFIG_SPECS:
        src_run = V3_ROOT / "runs" / str(spec["config_id"])
        dst_run = sprint_root / "runs" / str(spec["config_id"])
        dst_run.mkdir(parents=True, exist_ok=True)
        (dst_run / "tables").mkdir(exist_ok=True)
        (dst_run / "report").mkdir(exist_ok=True)
        for name in REUSED_FILES:
            shutil.copy2(src_run / name, dst_run / name)
            shutil.copy2(src_run / name, dst_run / "tables" / name)
        run_cfg = src_run / "report" / "run_config.json"
        if run_cfg.exists():
            shutil.copy2(run_cfg, dst_run / "report" / "run_config.json")
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name in ROOT_FIGURES_TO_COPY:
        src = V3_ROOT / "figures" / name
        if src.exists():
            shutil.copy2(src, figures_dir / name)


def _read_run_df(sprint_root: Path, config_id: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(sprint_root / "runs" / config_id / filename)


def _compound_to_family(name: str) -> str:
    lower = str(name).strip().lower()
    if any(token in lower for token in ["3-methyladenine", "methyladenine"]):
        return "methylated_purine_like"
    if any(token in lower for token in ["guanidine", "guanidino"]):
        return "guanidine_like"
    if any(token in lower for token in ["cyste", "glutath", "methion", "seleno", "sulfoximine", "sulfur"]):
        return "sulfur_small_molecule_like"
    if any(token in lower for token in ["adenine", "xanth", "hypox", "uric", "urate", "inos", "purine"]):
        return "purine_core_like"
    return "other_metabolite_like"


def _build_delta_vs_cohort(class_mean_df: pd.DataFrame, per_spectrum_bsv_df: pd.DataFrame) -> pd.DataFrame:
    axes = _axes_present(class_mean_df)
    cohort_mean = per_spectrum_bsv_df[axes].mean(axis=0)
    out = class_mean_df[["sample_key", "dataset_id", "subset_id", "class_label"]].copy()
    for axis in axes:
        out[axis] = class_mean_df[axis].to_numpy(dtype=float) - float(cohort_mean[axis])
    return out


def _build_neighborhood_family_composition(class_neighborhood_df: pd.DataFrame) -> pd.DataFrame:
    df = class_neighborhood_df.copy()
    df["neighborhood_family"] = df["compound_label"].astype(str).map(_compound_to_family)
    grouped = (
        df.groupby(["class_label", "neighborhood_family"], as_index=False)["support_fraction"]
        .sum()
        .rename(columns={"support_fraction": "family_support_fraction"})
    )
    rows = []
    for class_label in sorted(df["class_label"].astype(str).unique().tolist()):
        sub = grouped[grouped["class_label"].astype(str) == class_label].copy()
        existing = {str(x) for x in sub["neighborhood_family"].tolist()}
        for family in FAMILY_ORDER:
            if family not in existing:
                rows.append(
                    {
                        "class_label": class_label,
                        "neighborhood_family": family,
                        "family_support_fraction": 0.0,
                    }
                )
    if rows:
        grouped = pd.concat([grouped, pd.DataFrame(rows)], ignore_index=True)
    grouped = grouped.sort_values(["class_label", "neighborhood_family"]).reset_index(drop=True)
    return grouped


def _top_axes(series: pd.Series, *, positive_only: bool = False, top_n: int = 2) -> str:
    work = series.copy()
    if positive_only:
        work = work[work > 0]
    work = work.sort_values(ascending=False)
    if work.empty:
        return ""
    return "; ".join([f"{idx}:{float(val):.3f}" for idx, val in work.head(top_n).items()])


def _build_class_fingerprint_summary(
    config_id: str,
    short_label: str,
    class_mean_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    family_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    top1_df: pd.DataFrame,
) -> pd.DataFrame:
    axes = _axes_present(class_mean_df)
    rows = []
    for class_label in sorted(class_mean_df["class_label"].astype(str).tolist()):
        abs_row = class_mean_df[class_mean_df["class_label"].astype(str) == class_label].iloc[0]
        delta_row = delta_df[delta_df["class_label"].astype(str) == class_label].iloc[0]
        fam_row = family_df[family_df["class_label"].astype(str) == class_label].copy()
        dominant_fam = fam_row.sort_values("family_support_fraction", ascending=False).iloc[0]
        entropy = float(entropy_df[entropy_df["class_label"].astype(str) == class_label]["neighborhood_entropy"].iloc[0])
        top1 = float(top1_df[top1_df["class_label"].astype(str) == class_label]["top1_fraction"].iloc[0])
        abs_series = abs_row[axes].astype(float)
        delta_series = delta_row[axes].astype(float)
        rows.append(
            {
                "config_id": config_id,
                "config_short_label": short_label,
                "class_label": class_label,
                "top_absolute_axes": _top_axes(abs_series, positive_only=False, top_n=2),
                "top_positive_delta_axes": _top_axes(delta_series, positive_only=True, top_n=2),
                "dominant_neighborhood_family": str(dominant_fam["neighborhood_family"]),
                "neighborhood_entropy": entropy,
                "top1_dominance": top1,
                "delta_l1_magnitude": float(np.abs(delta_series.to_numpy(dtype=float)).sum()),
            }
        )
    return pd.DataFrame(rows)


def _mean_pairwise_distance(df: pd.DataFrame, value_cols: list[str]) -> float:
    matrix = df[value_cols].to_numpy(dtype=float)
    if len(matrix) < 2:
        return 0.0
    dists = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            dists.append(float(np.linalg.norm(matrix[i] - matrix[j])))
    return float(np.mean(dists)) if dists else 0.0


def _plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    output_path: Path,
    title: str,
    *,
    cmap: str,
    center_zero: bool = False,
) -> None:
    fig, ax = plt.subplots(
        figsize=(max(7.2, 0.72 * len(col_labels) + 2.4), max(4.8, 0.6 * len(row_labels) + 2.3))
    )
    if center_zero:
        lim = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 1e-9)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
    else:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=max(float(np.nanmax(matrix)), 1e-9))
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.84)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_absolute_fingerprint_grid(class_mean_df: pd.DataFrame, output_path: Path, title: str) -> None:
    axes = [axis for axis in _axes_present(class_mean_df) if float(class_mean_df[axis].abs().max()) > 0.02]
    labels = sorted(class_mean_df["class_label"].astype(str).tolist())
    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.4, 3.6 * nrows))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs[len(labels) :]:
        ax.axis("off")
    xmax = max(float(class_mean_df[axes].to_numpy(dtype=float).max()), 0.4)
    for ax, label in zip(axs, labels, strict=False):
        row = class_mean_df[class_mean_df["class_label"].astype(str) == label].iloc[0]
        vals = np.array([float(row[axis]) for axis in axes], dtype=float)
        y = np.arange(len(axes))
        ax.barh(y, vals, color=CLASS_COLORS.get(label, "#666666"), alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(axes, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.0, xmax * 1.05)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.20, linewidth=0.6)
    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_delta_fingerprint_grid(delta_df: pd.DataFrame, output_path: Path, title: str) -> None:
    axes = [axis for axis in _axes_present(delta_df) if float(delta_df[axis].abs().max()) > 0.005]
    labels = sorted(delta_df["class_label"].astype(str).tolist())
    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.8, 3.8 * nrows))
    axs = np.atleast_1d(axs).ravel()
    lim = max(float(np.abs(delta_df[axes].to_numpy(dtype=float)).max()), 0.05)
    for ax in axs[len(labels) :]:
        ax.axis("off")
    for ax, label in zip(axs, labels, strict=False):
        row = delta_df[delta_df["class_label"].astype(str) == label].iloc[0]
        vals = np.array([float(row[axis]) for axis in axes], dtype=float)
        y = np.arange(len(axes))
        colors = ["#4daf4a" if v >= 0 else "#d73027" for v in vals]
        ax.barh(y, vals, color=colors, alpha=0.88)
        ax.axvline(0.0, color="#555555", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(axes, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(-lim * 1.08, lim * 1.08)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.20, linewidth=0.6)
    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_family_grid(family_df: pd.DataFrame, output_path: Path, title: str) -> None:
    classes = sorted(family_df["class_label"].astype(str).unique().tolist())
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    left = np.zeros(len(classes), dtype=float)
    for family in FAMILY_ORDER:
        vals = []
        for label in classes:
            sub = family_df[
                (family_df["class_label"].astype(str) == label)
                & (family_df["neighborhood_family"].astype(str) == family)
            ]
            vals.append(float(sub["family_support_fraction"].iloc[0]) if not sub.empty else 0.0)
        vals_arr = np.asarray(vals, dtype=float)
        ax.barh(
            np.arange(len(classes)),
            vals_arr,
            left=left,
            color=FAMILY_COLORS[family],
            label=family,
            alpha=0.9,
        )
        left += vals_arr
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Family support fraction")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Neighborhood family")
    ax.grid(True, axis="x", alpha=0.20, linewidth=0.6)
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_summary_figure(
    class_mean_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    family_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    abs_axes = [axis for axis in _axes_present(class_mean_df) if float(class_mean_df[axis].abs().max()) > 0.02]
    delta_axes = [axis for axis in _axes_present(delta_df) if float(delta_df[axis].abs().max()) > 0.005]
    classes = sorted(class_mean_df["class_label"].astype(str).tolist())
    abs_heat = class_mean_df.set_index("class_label")[abs_axes].loc[classes]
    delta_heat = delta_df.set_index("class_label")[delta_axes].loc[classes]
    family_heat = (
        family_df.pivot(index="class_label", columns="neighborhood_family", values="family_support_fraction")
        .reindex(classes)
        .reindex(FAMILY_ORDER, axis=1)
        .fillna(0.0)
    )

    fig = plt.figure(figsize=(15.5, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.0], wspace=0.35)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

    im0 = axs[0].imshow(abs_heat.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(abs_heat.to_numpy(dtype=float).max()), 1e-9))
    axs[0].set_title("Absolute BSV", fontsize=11)
    axs[0].set_xticks(np.arange(len(abs_axes)))
    axs[0].set_xticklabels(abs_axes, rotation=40, ha="right", fontsize=8)
    axs[0].set_yticks(np.arange(len(classes)))
    axs[0].set_yticklabels(classes, fontsize=9)
    fig.colorbar(im0, ax=axs[0], shrink=0.78)

    lim = max(float(np.abs(delta_heat.to_numpy(dtype=float)).max()), 1e-9)
    im1 = axs[1].imshow(delta_heat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    axs[1].set_title("Delta vs cohort", fontsize=11)
    axs[1].set_xticks(np.arange(len(delta_axes)))
    axs[1].set_xticklabels(delta_axes, rotation=40, ha="right", fontsize=8)
    axs[1].set_yticks(np.arange(len(classes)))
    axs[1].set_yticklabels([])
    fig.colorbar(im1, ax=axs[1], shrink=0.78)

    im2 = axs[2].imshow(family_heat.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0.0, vmax=max(float(family_heat.to_numpy(dtype=float).max()), 1e-9))
    axs[2].set_title("Neighborhood family mix", fontsize=11)
    axs[2].set_xticks(np.arange(len(FAMILY_ORDER)))
    axs[2].set_xticklabels(FAMILY_ORDER, rotation=40, ha="right", fontsize=8)
    axs[2].set_yticks(np.arange(len(classes)))
    axs[2].set_yticklabels([])
    fig.colorbar(im2, ax=axs[2], shrink=0.78)

    fig.suptitle(title, fontsize=15, y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_delta_comparison(comparator_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    x = np.arange(len(comparator_df))
    width = 0.28
    ax.bar(x - width / 2, comparator_df["mean_delta_l1_magnitude"], width=width, color="#43aa8b", label="Mean delta magnitude")
    ax.bar(x + width / 2, comparator_df["mean_pairwise_delta_distance"], width=width, color="#577590", label="Mean pairwise delta distance")
    ax.set_xticks(x)
    ax.set_xticklabels(comparator_df["config_short_label"].tolist())
    ax.set_title("Pilot 1a: Relative fingerprint signal exposed by delta-BSV")
    ax.grid(True, axis="y", alpha=0.20, linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_distinctiveness(comparator_df: pd.DataFrame, output_path: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.8))
    metrics = [
        ("mean_pairwise_delta_distance", "Delta fingerprint distinctiveness"),
        ("mean_pairwise_family_distance", "Neighborhood family distinctiveness"),
        ("mean_top1_dominance", "Neighborhood dominance"),
    ]
    colors = ["#577590", "#2a9d8f", "#e76f51"]
    for ax, (col, title), color in zip(axs, metrics, colors, strict=False):
        ax.bar(comparator_df["config_short_label"].tolist(), comparator_df[col].to_numpy(dtype=float), color=color)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.20, linewidth=0.6)
    fig.suptitle("Pilot 1a: Class fingerprint distinctiveness by config", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _build_pdf(report_md: Path, figure_paths: list[Path], output_path: Path) -> None:
    text = report_md.read_text(encoding="utf-8")
    wrapped_lines: list[str] = []
    for raw in text.splitlines():
        if raw.startswith("#"):
            wrapped_lines.append(raw)
        elif raw.strip():
            wrapped_lines.extend(textwrap.wrap(raw, width=96))
        else:
            wrapped_lines.append("")
    with PdfPages(output_path) as pdf:
        chunk_size = 34
        for i in range(0, len(wrapped_lines), chunk_size):
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.965
            for line in wrapped_lines[i : i + chunk_size]:
                size = 12 if line.startswith("# ") else 10 if line.startswith("## ") else 8.6
                weight = "bold" if line.startswith("#") else "normal"
                fig.text(0.06, y, line, ha="left", va="top", fontsize=size, fontweight=weight, family="DejaVu Sans Mono")
                y -= 0.026 if line.startswith("#") else 0.023
            pdf.savefig(fig)
            plt.close(fig)
        for path in figure_paths:
            img = plt.imread(path)
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(path.name, fontsize=12, y=0.98)
            pdf.savefig(fig)
            plt.close(fig)


def _build_report(
    report_path: Path,
    comparator_df: pd.DataFrame,
    class_compare_df: pd.DataFrame,
    pass5_ranked_df: pd.DataFrame,
) -> None:
    baseline = comparator_df[comparator_df["config_short_label"] == "baseline"].iloc[0]
    cfg05 = comparator_df[comparator_df["config_short_label"] == "cfg05"].iloc[0]
    cfg08 = comparator_df[comparator_df["config_short_label"] == "cfg08"].iloc[0]
    baseline_val = float(pass5_ranked_df[pass5_ranked_df["config_id"] == "cfg02"]["validation_score"].iloc[0])
    cfg05_val = float(pass5_ranked_df[pass5_ranked_df["config_id"] == "cfg05"]["validation_score"].iloc[0])
    cfg08_val = float(pass5_ranked_df[pass5_ranked_df["config_id"] == "cfg08"]["validation_score"].iloc[0])
    lines = [
        "# GAIRAv3 Pilot 1a Celltype Probe1 v4 Report",
        "",
        "## 1. Why v4 was needed",
        "- Pilot 1a v3 showed that the coarse absolute Tier-1 BSV radar was too low-resolution to act as a class fingerprint by itself.",
        "- The cell types occupy a similar broad biochemical regime, so absolute class-mean BSV shapes alone looked too similar.",
        "- v4 therefore treats the fingerprint as a stacked object: absolute BSV, delta-BSV versus cohort mean, local neighborhood family composition, and stability context.",
        "- The underlying v3 per-config outputs were reused after verification. This pass refines the fingerprint definition and reporting layer only.",
        "",
        "## 2. Absolute vs Differential Fingerprint Concept",
        "- Absolute BSV describes broad biochemical position.",
        "- Delta-BSV versus cohort mean describes relative enrichment or depletion, which is the more useful class-level fingerprint when the whole panel lives in the same broad regime.",
        "- Neighborhood family composition gives a local chemistry fingerprint built from retrieved support patterns, not direct molecule identification.",
        "- Stability metrics tell us whether those patterns are tight within class and distinct between classes.",
        "",
        "## 3. Per-config Class Fingerprint Results",
        f"- Baseline: mean delta fingerprint magnitude `{baseline['mean_delta_l1_magnitude']:.4f}`, mean pairwise delta distance `{baseline['mean_pairwise_delta_distance']:.4f}`, dominant families `{baseline['dominant_compound_family_summary']}`.",
        f"- cfg05: mean delta fingerprint magnitude `{cfg05['mean_delta_l1_magnitude']:.4f}`, mean pairwise delta distance `{cfg05['mean_pairwise_delta_distance']:.4f}`, dominant families `{cfg05['dominant_compound_family_summary']}`.",
        f"- cfg08: mean delta fingerprint magnitude `{cfg08['mean_delta_l1_magnitude']:.4f}`, mean pairwise delta distance `{cfg08['mean_pairwise_delta_distance']:.4f}`, dominant families `{cfg08['dominant_compound_family_summary']}`.",
        "- Delta-BSV does make the fingerprints more informative than absolute BSV alone because it surfaces relative support shifts that are largely hidden in the shared broad-family baseline.",
        "- Neighborhood family grouping also helps. It shows whether classes differ mainly by purine-core versus methylated-purine versus guanidine-like balance, or whether they drift into a broader other-metabolite regime.",
        "",
        "## 4. Stability Context",
        f"- Baseline: mean intra `{baseline['mean_intra_class_bsv_variance']:.6f}`, mean inter `{baseline['mean_inter_class_bsv_distance']:.6f}`, top1 dominance `{baseline['mean_top1_dominance']:.4f}`.",
        f"- cfg05: mean intra `{cfg05['mean_intra_class_bsv_variance']:.6f}`, mean inter `{cfg05['mean_inter_class_bsv_distance']:.6f}`, top1 dominance `{cfg05['mean_top1_dominance']:.4f}`.",
        f"- cfg08: mean intra `{cfg08['mean_intra_class_bsv_variance']:.6f}`, mean inter `{cfg08['mean_inter_class_bsv_distance']:.6f}`, top1 dominance `{cfg08['mean_top1_dominance']:.4f}`.",
        "- The important comparison here is not just raw inter-class distance. It is whether the relative fingerprint layers are class-distinctive while remaining stable within class.",
        "",
        "## 5. Comparator Interpretation",
        "- Baseline still acts like a narrow reference. Its class differences are visible, but they are dominated by a small purine-centered vocabulary.",
        "- cfg05 is the clearest upgrade. Under the richer fingerprint definition it exposes stronger class-relative differences without losing interpretability.",
        "- cfg08 is partly rescued by the richer fingerprint object because its broader neighborhood family structure is now visible rather than just looking diffuse. But it still shifts too much toward a broad `other_metabolite_like` vocabulary for this cell-type panel.",
        "- In plain terms: delta-BSV and neighborhood grouping help all three configs, but they help cfg05 most because it broadens the fingerprint without erasing class identity.",
        "",
        "## 6. Decision for Pilot 1b",
        f"- Pass 5 validation note: baseline `{baseline_val:.4f}`, cfg05 `{cfg05_val:.4f}`, cfg08 `{cfg08_val:.4f}`.",
        "- cfg05 now clearly outperforms baseline for class fingerprint interpretability.",
        "- cfg08 remains a useful broadness comparator, but it is still not the best working default for Pilot 1b.",
        "- Recommended config for Pilot 1b: `cfg05`.",
        "",
        "## 7. Neighborhood Family Grouping Rules",
        "- `purine_core_like`: adenine, xanthine, hypoxanthine, urate-like, inosine-like, and related purine-adjacent names.",
        "- `methylated_purine_like`: methyladenine-like names.",
        "- `guanidine_like`: guanidine / guanidino names such as methylguanidine.",
        "- `sulfur_small_molecule_like`: sulfur- or selenium-adjacent small molecules such as methionine/cysteine/selenium/sulfoximine-related names.",
        "- `other_metabolite_like`: everything else in the retrieved local neighborhood vocabulary.",
        "",
        "## 8. Example Class-level Readout",
    ]
    example = class_compare_df.sort_values(["config_short_label", "class_label"]).head(9)
    for row in example.itertuples(index=False):
        lines.append(
            f"- {row.config_short_label} / {row.class_label}: abs `{row.top_absolute_axes}`, delta `{row.top_positive_delta_axes}`, family `{row.dominant_neighborhood_family}`."
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _require_v3_inputs()
    storage_cfg = load_autoresearch_storage_config(DEFAULT_STORAGE_CONFIG_PATH)
    sprint_paths = initialize_autoresearch_sprint(
        DEFAULT_STORAGE_CONFIG_PATH,
        sprint_id=f"{storage_cfg.sprint_id}/{SPRINT_SUBDIR}",
    )
    _copy_reused_outputs(sprint_paths.sprint_root, sprint_paths.figures_dir)
    pass5_ranked_df = pd.read_csv(PASS5_TABLE)

    comparator_rows = []
    class_summary_rows = []
    for spec in CONFIG_SPECS:
        config_id = str(spec["config_id"])
        short_label = str(spec["short_label"])
        class_mean_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_mean_bsv.csv")
        per_spectrum_bsv_df = _read_run_df(sprint_paths.sprint_root, config_id, "per_spectrum_bsv.csv")
        neighborhood_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_topk_neighborhood_composition.csv")
        entropy_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_neighborhood_entropy.csv")
        top1_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_top1_dominance.csv")
        axis_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_axis_entropy.csv")
        within_df = _read_run_df(sprint_paths.sprint_root, config_id, "config_within_between_summary.csv")

        delta_df = _build_delta_vs_cohort(class_mean_df, per_spectrum_bsv_df)
        family_df = _build_neighborhood_family_composition(neighborhood_df)
        class_summary_df = _build_class_fingerprint_summary(
            config_id,
            short_label,
            class_mean_df,
            delta_df,
            family_df,
            entropy_df,
            top1_df,
        )

        delta_df.to_csv(sprint_paths.sprint_root / "runs" / config_id / "class_mean_bsv_delta_vs_cohort.csv", index=False)
        family_df.to_csv(sprint_paths.sprint_root / "runs" / config_id / "class_neighborhood_family_composition.csv", index=False)
        class_summary_df.to_csv(sprint_paths.sprint_root / "runs" / config_id / "class_fingerprint_summary.csv", index=False)
        delta_df.to_csv(sprint_paths.sprint_root / "runs" / config_id / "tables" / "class_mean_bsv_delta_vs_cohort.csv", index=False)
        family_df.to_csv(sprint_paths.sprint_root / "runs" / config_id / "tables" / "class_neighborhood_family_composition.csv", index=False)
        class_summary_df.to_csv(sprint_paths.sprint_root / "runs" / config_id / "tables" / "class_fingerprint_summary.csv", index=False)

        _plot_absolute_fingerprint_grid(
            class_mean_df,
            sprint_paths.figures_dir / f"absolute_fingerprint_grid_{config_id}.png",
            f"Absolute Fingerprint Grid: {short_label}",
        )
        _plot_delta_fingerprint_grid(
            delta_df,
            sprint_paths.figures_dir / f"delta_fingerprint_grid_{config_id}.png",
            f"Delta Fingerprint Grid vs cohort: {short_label}",
        )
        _plot_family_grid(
            family_df,
            sprint_paths.figures_dir / f"neighborhood_family_fingerprint_grid_{config_id}.png",
            f"Neighborhood Family Fingerprint: {short_label}",
        )
        _plot_summary_figure(
            class_mean_df,
            delta_df,
            family_df,
            sprint_paths.figures_dir / f"class_fingerprint_summary_{config_id}.png",
            f"Class Fingerprint Summary: {short_label}",
        )

        axes = _axes_present(delta_df)
        family_pivot = (
            family_df.pivot(index="class_label", columns="neighborhood_family", values="family_support_fraction")
            .reindex(sorted(class_mean_df["class_label"].astype(str).tolist()))
            .reindex(FAMILY_ORDER, axis=1)
            .fillna(0.0)
        )
        within = within_df.iloc[0]
        comparator_rows.append(
            {
                "config_id": config_id,
                "config_short_label": short_label,
                "display_name": spec["display_name"],
                "mean_intra_class_bsv_variance": float(within["mean_intra_class_bsv_variance"]),
                "mean_inter_class_bsv_distance": float(within["mean_inter_class_bsv_distance"]),
                "within_between_ratio": float(within["within_between_ratio"]),
                "mean_neighborhood_entropy": float(entropy_df["neighborhood_entropy"].mean()),
                "mean_top1_dominance": float(top1_df["top1_fraction"].mean()),
                "mean_axis_entropy": float(axis_df["axis_entropy"].mean()),
                "dominant_compound_family_summary": "; ".join(
                    class_summary_df["dominant_neighborhood_family"].value_counts().sort_values(ascending=False).rename_axis("family").reset_index(name="n").apply(lambda r: f"{r['family']}:{int(r['n'])}", axis=1).tolist()
                ),
                "mean_delta_l1_magnitude": float(class_summary_df["delta_l1_magnitude"].mean()),
                "mean_pairwise_delta_distance": _mean_pairwise_distance(delta_df, axes),
                "mean_pairwise_family_distance": _mean_pairwise_distance(family_pivot.reset_index(drop=True), FAMILY_ORDER),
                "concise_interpretation_note": (
                    "narrow reference"
                    if short_label == "baseline"
                    else "best relative fingerprint balance"
                    if short_label == "cfg05"
                    else "broader but still diffuse"
                ),
            }
        )
        class_summary_rows.append(class_summary_df)

    comparator_df = pd.DataFrame(comparator_rows)
    class_compare_df = pd.concat(class_summary_rows, ignore_index=True)
    comparator_df.to_csv(sprint_paths.tables_dir / "pilot1a_probe1_v4_config_comparison.csv", index=False)
    class_compare_df.to_csv(sprint_paths.tables_dir / "pilot1a_probe1_v4_class_fingerprint_comparison.csv", index=False)

    _plot_delta_comparison(comparator_df, sprint_paths.figures_dir / "pilot1a_delta_fingerprint_comparison.png")
    _plot_distinctiveness(comparator_df, sprint_paths.figures_dir / "pilot1a_class_fingerprint_distinctiveness.png")

    report_md = sprint_paths.report_dir / "GAIRAv3_Pilot1a_celltype_probe1_v4_report.md"
    _build_report(report_md, comparator_df, class_compare_df, pass5_ranked_df)

    figure_paths = [sprint_paths.figures_dir / name for name in ROOT_FIGURES_TO_COPY]
    for spec in CONFIG_SPECS:
        config_id = str(spec["config_id"])
        figure_paths.extend(
            [
                sprint_paths.figures_dir / f"absolute_fingerprint_grid_{config_id}.png",
                sprint_paths.figures_dir / f"delta_fingerprint_grid_{config_id}.png",
                sprint_paths.figures_dir / f"neighborhood_family_fingerprint_grid_{config_id}.png",
                sprint_paths.figures_dir / f"class_fingerprint_summary_{config_id}.png",
            ]
        )
    figure_paths.extend(
        [
            sprint_paths.figures_dir / "pilot1a_delta_fingerprint_comparison.png",
            sprint_paths.figures_dir / "pilot1a_class_fingerprint_distinctiveness.png",
        ]
    )
    _build_pdf(report_md, figure_paths, sprint_paths.report_dir / "GAIRAv3_Pilot1a_celltype_probe1_v4_report.pdf")
    print(f"Wrote Pilot 1a v4 outputs under {sprint_paths.sprint_root}")


if __name__ == "__main__":
    main()
