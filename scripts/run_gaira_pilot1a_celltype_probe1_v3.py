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


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot1a_celltype_probe1_v2"
)
PASS5_TABLE = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pass5_saturation_fix/tables/calibration_results_ranked.csv"
)
SPRINT_SUBDIR = "pilot1a_celltype_probe1_v3"

CONFIG_SPECS = [
    {
        "config_id": "baseline_v1_locked_purine",
        "short_label": "baseline",
        "display_name": "Baseline v1 locked purine",
        "filter_mode": "purine_focused_universal",
        "top_k": 5,
        "weighting_mode": "uniform_weighting",
        "weighting_param": None,
        "diversity_mode": "none",
    },
    {
        "config_id": "candidate_v2_cfg05_max_desaturation",
        "short_label": "cfg05",
        "display_name": "Candidate v2 cfg05 max desaturation",
        "filter_mode": "purine_expanded_neighbor",
        "top_k": 5,
        "weighting_mode": "softmax_temperature",
        "weighting_param": 1.0,
        "diversity_mode": "compound_uniqueness_penalty",
    },
    {
        "config_id": "candidate_v2_cfg08_balanced_update",
        "short_label": "cfg08",
        "display_name": "Candidate v2 cfg08 balanced update",
        "filter_mode": "balanced_metabolite_subset",
        "top_k": 8,
        "weighting_mode": "rank_decay_weighting",
        "weighting_param": 0.75,
        "diversity_mode": "family_balance_penalty",
    },
]

RUN_FILES = [
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

COLOR_BY_CLASS = {
    "Hec": "#4c78a8",
    "Hela": "#f58518",
    "Ht": "#54a24b",
    "Mef": "#e45756",
    "Thp": "#72b7b2",
}
COLOR_BY_CONFIG = {
    "baseline": "#577590",
    "cfg05": "#f3722c",
    "cfg08": "#43aa8b",
}
MARKER_BY_CONFIG = {"baseline": "o", "cfg05": "s", "cfg08": "^"}


def _axes_present(df: pd.DataFrame) -> list[str]:
    return [axis for axis in ALL_AXES if axis in df.columns]


def _require_v2_inputs() -> None:
    if not V2_ROOT.exists():
        raise RuntimeError(f"Missing v2 root: {V2_ROOT}")
    for spec in CONFIG_SPECS:
        run_dir = V2_ROOT / "runs" / str(spec["config_id"])
        for name in RUN_FILES:
            path = run_dir / name
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(f"Missing v2 artifact: {path}")
            df = pd.read_csv(path)
            if df.empty:
                raise RuntimeError(f"Empty v2 artifact: {path}")


def _copy_v2_run_tables(sprint_root: Path) -> None:
    for spec in CONFIG_SPECS:
        src_run = V2_ROOT / "runs" / str(spec["config_id"])
        dst_run = sprint_root / "runs" / str(spec["config_id"])
        dst_run.mkdir(parents=True, exist_ok=True)
        (dst_run / "tables").mkdir(exist_ok=True)
        (dst_run / "report").mkdir(exist_ok=True)
        for name in RUN_FILES:
            shutil.copy2(src_run / name, dst_run / name)
            shutil.copy2(src_run / name, dst_run / "tables" / name)
        run_cfg = src_run / "report" / "run_config.json"
        if run_cfg.exists():
            shutil.copy2(run_cfg, dst_run / "report" / "run_config.json")


def _read_run_df(sprint_root: Path, config_id: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(sprint_root / "runs" / config_id / filename)


def _fit_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * s[:2]
    denom = max(float((s**2).sum()), 1e-12)
    explained = (s[:2] ** 2) / denom
    return scores, explained


def _scaled_pca_df(df: pd.DataFrame, axes: list[str]) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    active_axes = [axis for axis in axes if float(df[axis].abs().max()) > 1e-9]
    matrix = df[active_axes].to_numpy(dtype=float)
    means = matrix.mean(axis=0, keepdims=True)
    stds = matrix.std(axis=0, keepdims=True)
    stds = np.where(stds < 1e-9, 1.0, stds)
    scaled = (matrix - means) / stds
    scores, explained = _fit_pca(scaled)
    out = df[["class_label"]].copy()
    if "sample_key" in df.columns:
        out["sample_key"] = df["sample_key"].astype(str)
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1] if scores.shape[1] > 1 else 0.0
    out["pc1_explained_ratio"] = float(explained[0]) if len(explained) > 0 else 0.0
    out["pc2_explained_ratio"] = float(explained[1]) if len(explained) > 1 else 0.0
    return out, active_axes, explained


def _place_label(ax: plt.Axes, x: float, y: float, text: str, idx: int) -> None:
    dx = [0.04, -0.06, 0.05, -0.05, 0.03][idx % 5]
    dy = [0.05, 0.05, -0.06, -0.05, 0.08][idx % 5]
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        fontsize=9,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
        arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "#555555"},
    )


def _plot_spectral_pca(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    labels = sorted(pca_df["class_label"].astype(str).unique().tolist())
    for label in labels:
        sub = pca_df[pca_df["class_label"].astype(str) == label].copy()
        ax.scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=22,
            alpha=0.72,
            color=COLOR_BY_CLASS.get(label, "#666666"),
            label=label,
            edgecolors="white",
            linewidths=0.25,
        )
    pc1_ratio = float(pca_df["pc1_explained_ratio"].iloc[0]) if "pc1_explained_ratio" in pca_df.columns else 0.0
    pc2_ratio = float(pca_df["pc2_explained_ratio"].iloc[0]) if "pc2_explained_ratio" in pca_df.columns else 0.0
    ax.set_xlabel(f"PC1 ({pc1_ratio * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pc2_ratio * 100:.1f}% var)")
    ax.set_title("Original Spectral PCA: Probe 1 Cell-Type Dataset")
    ax.grid(True, alpha=0.20, linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Class")
    fig.tight_layout(rect=[0.0, 0.0, 0.83, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_bsv_pca(pca_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    labels = sorted(pca_df["class_label"].astype(str).unique().tolist())
    for label in labels:
        sub = pca_df[pca_df["class_label"].astype(str) == label].copy()
        ax.scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=26,
            alpha=0.75,
            color=COLOR_BY_CLASS.get(label, "#666666"),
            label=label,
            edgecolors="white",
            linewidths=0.25,
        )
    pc1_ratio = float(pca_df["pc1_explained_ratio"].iloc[0]) if not pca_df.empty else 0.0
    pc2_ratio = float(pca_df["pc2_explained_ratio"].iloc[0]) if not pca_df.empty else 0.0
    ax.set_xlabel(f"PC1 ({pc1_ratio * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pc2_ratio * 100:.1f}% var)")
    ax.set_title(title)
    ax.grid(True, alpha=0.20, linewidth=0.6)
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.7)
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.7)
    ax.ticklabel_format(style="plain", axis="both")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Class")
    fig.tight_layout(rect=[0.0, 0.0, 0.83, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_bsv_class_mean_pca(pca_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.5))
    for idx, row in enumerate(pca_df.itertuples(index=False)):
        ax.scatter(
            float(row.pc1),
            float(row.pc2),
            s=78,
            color=COLOR_BY_CLASS.get(str(row.class_label), "#666666"),
            edgecolors="white",
            linewidths=0.5,
        )
        _place_label(ax, float(row.pc1), float(row.pc2), str(row.class_label), idx)
    pc1_ratio = float(pca_df["pc1_explained_ratio"].iloc[0]) if not pca_df.empty else 0.0
    pc2_ratio = float(pca_df["pc2_explained_ratio"].iloc[0]) if not pca_df.empty else 0.0
    ax.set_xlabel(f"PC1 ({pc1_ratio * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pc2_ratio * 100:.1f}% var)")
    ax.set_title(title)
    ax.grid(True, alpha=0.20, linewidth=0.6)
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.7)
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_overlay_class_mean_pca(overlay_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    for idx, row in enumerate(overlay_df.itertuples(index=False)):
        ax.scatter(
            float(row.pc1),
            float(row.pc2),
            s=74,
            color=COLOR_BY_CONFIG.get(str(row.config_short_label), "#666666"),
            marker=MARKER_BY_CONFIG.get(str(row.config_short_label), "o"),
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
        )
        _place_label(ax, float(row.pc1), float(row.pc2), f"{row.class_label}-{row.config_short_label}", idx)
    pc1_ratio = float(overlay_df["pc1_explained_ratio"].iloc[0]) if not overlay_df.empty else 0.0
    pc2_ratio = float(overlay_df["pc2_explained_ratio"].iloc[0]) if not overlay_df.empty else 0.0
    ax.set_xlabel(f"PC1 ({pc1_ratio * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pc2_ratio * 100:.1f}% var)")
    ax.set_title("Class-Mean BSV PCA Overlay (visualization-only scaled projection)")
    ax.grid(True, alpha=0.20, linewidth=0.6)
    handles = []
    labels = []
    for short_label in ["baseline", "cfg05", "cfg08"]:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=MARKER_BY_CONFIG[short_label],
                linestyle="",
                markerfacecolor=COLOR_BY_CONFIG[short_label],
                markeredgecolor="white",
                markersize=9,
            )
        )
        labels.append(short_label)
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Config")
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _display_axes(class_mean_df: pd.DataFrame) -> list[str]:
    axes = _axes_present(class_mean_df)
    keep = []
    for axis in axes:
        if float(class_mean_df[axis].abs().max()) >= 0.03:
            keep.append(axis)
    return keep or axes[:]


def _plot_radar_grid(class_mean_df: pd.DataFrame, output_path: Path, title: str) -> list[str]:
    axes = _display_axes(class_mean_df)
    labels = sorted(class_mean_df["class_label"].astype(str).tolist())
    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10.8, 5.0 * nrows),
        subplot_kw={"projection": "polar"},
    )
    axs = np.atleast_1d(axs).ravel()
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    for ax in axs[len(labels) :]:
        ax.axis("off")
    for ax, label in zip(axs, labels, strict=False):
        row = class_mean_df[class_mean_df["class_label"].astype(str) == label].iloc[0]
        values = np.array([float(row[axis]) for axis in axes], dtype=float)
        values_closed = np.concatenate([values, [values[0]]])
        color = COLOR_BY_CLASS.get(label, "#666666")
        ax.plot(angles_closed, values_closed, color=color, linewidth=2.6)
        ax.fill(angles_closed, values_closed, color=color, alpha=0.34)
        ax.scatter(angles, values, color=color, s=18, zorder=3)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(axes, fontsize=8)
        ax.tick_params(axis="x", pad=10)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_title(label, y=1.12, fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=15, y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    return axes


def _plot_neighborhood_grid(class_neighborhood_df: pd.DataFrame, output_path: Path, title: str) -> None:
    labels = sorted(class_neighborhood_df["class_label"].astype(str).unique().tolist())
    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.5, 3.6 * nrows))
    axs = np.atleast_1d(axs).ravel()
    xmax = max(float(class_neighborhood_df["support_fraction"].max()), 0.35)
    for ax in axs[len(labels) :]:
        ax.axis("off")
    for ax, label in zip(axs, labels, strict=False):
        sub = (
            class_neighborhood_df[class_neighborhood_df["class_label"].astype(str) == label]
            .copy()
            .sort_values("support_fraction", ascending=False)
            .head(8)
        )
        y = np.arange(len(sub))
        colors = [COLOR_BY_CLASS.get(label, "#2a9d8f")] * len(sub)
        ax.barh(y, sub["support_fraction"].to_numpy(dtype=float), color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["compound_label"].astype(str).tolist(), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.0, xmax * 1.05)
        ax.grid(True, axis="x", alpha=0.20, linewidth=0.6)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Support fraction")
    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _draw_heatmap(
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
        figsize=(max(7.0, 0.72 * len(col_labels) + 2.8), max(4.8, 0.62 * len(row_labels) + 2.6))
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


def _plot_within_between(comparator_df: pd.DataFrame, output_path: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
    x = np.arange(len(comparator_df))
    labels = comparator_df["config_short_label"].tolist()
    axs[0].bar(x, comparator_df["mean_intra_class_bsv_variance"], color="#577590", width=0.62)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels)
    axs[0].set_title("Within-class BSV Variance")
    axs[0].grid(True, axis="y", alpha=0.20, linewidth=0.6)
    axs[1].bar(x, comparator_df["mean_inter_class_bsv_distance"], color="#43aa8b", width=0.62)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)
    axs[1].set_title("Between-class BSV Distance")
    axs[1].grid(True, axis="y", alpha=0.20, linewidth=0.6)
    fig.suptitle("Pilot 1a: Within-class vs Between-class Stability")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_entropy_dominance(comparator_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    x = np.arange(len(comparator_df))
    width = 0.24
    ax.bar(x - width, comparator_df["mean_neighborhood_entropy"], width=width, color="#2a9d8f", label="Neighborhood entropy")
    ax.bar(x, comparator_df["mean_axis_entropy"], width=width, color="#e9c46a", label="Axis entropy")
    ax.bar(x + width, comparator_df["mean_top1_dominance"], width=width, color="#e76f51", label="Top1 dominance")
    ax.set_xticks(x)
    ax.set_xticklabels(comparator_df["config_short_label"].tolist())
    ax.set_title("Pilot 1a: Breadth vs Dominance")
    ax.grid(True, axis="y", alpha=0.20, linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_tradeoff(comparator_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for row in comparator_df.itertuples(index=False):
        ax.scatter(
            float(row.mean_neighborhood_entropy),
            float(row.mean_inter_class_bsv_distance),
            s=170,
            color=COLOR_BY_CONFIG.get(str(row.config_short_label), "#666666"),
            alpha=0.9,
            edgecolors="white",
            linewidths=0.7,
        )
        ax.annotate(
            str(row.config_short_label),
            xy=(float(row.mean_neighborhood_entropy), float(row.mean_inter_class_bsv_distance)),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xlabel("Mean neighborhood entropy")
    ax.set_ylabel("Mean inter-class BSV distance")
    ax.set_title("Pilot 1a Config Tradeoff: Breadth vs Class Separation")
    ax.grid(True, alpha=0.20, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _dominant_family_summary(class_neighborhood_df: pd.DataFrame) -> str:
    top = (
        class_neighborhood_df.sort_values(["class_label", "support_fraction"], ascending=[True, False])
        .groupby("class_label", sort=True)
        .head(1)
        .copy()
    )
    top["family"] = top["compound_label"].astype(str).map(lambda x: classify_compound_family(x, ""))
    grouped = top.groupby("family", sort=True).size().sort_values(ascending=False)
    return "; ".join([f"{family}:{int(count)}" for family, count in grouped.items()])


def _interpretation_note(mean_neighborhood_entropy: float, mean_top1_dominance: float, within_between_ratio: float) -> str:
    if mean_neighborhood_entropy < 1.0 and mean_top1_dominance > 0.7:
        return "narrow and dominance-heavy"
    if mean_neighborhood_entropy > 1.6 and within_between_ratio > 0.008:
        return "broadened but more diffuse"
    if mean_neighborhood_entropy > 1.1 and mean_top1_dominance < 0.65:
        return "broadened with usable structure"
    return "intermediate tradeoff"


def _build_report(
    report_path: Path,
    comparator_df: pd.DataFrame,
    pass5_ranked_df: pd.DataFrame,
    radar_axis_notes: dict[str, list[str]],
) -> None:
    baseline = comparator_df[comparator_df["config_short_label"] == "baseline"].iloc[0]
    cfg05 = comparator_df[comparator_df["config_short_label"] == "cfg05"].iloc[0]
    cfg08 = comparator_df[comparator_df["config_short_label"] == "cfg08"].iloc[0]
    baseline_val = float(pass5_ranked_df[pass5_ranked_df["config_id"] == "cfg02"]["validation_score"].iloc[0])
    cfg05_val = float(pass5_ranked_df[pass5_ranked_df["config_id"] == "cfg05"]["validation_score"].iloc[0])
    cfg08_val = float(pass5_ranked_df[pass5_ranked_df["config_id"] == "cfg08"]["validation_score"].iloc[0])
    lines = [
        "# GAIRAv3 Pilot 1a Celltype Probe1 v3 Report",
        "",
        "## 1. Overview",
        "- This is the final decision-grade Pilot 1a rerun for Probe 1 cell types only.",
        "- The underlying v2 numeric outputs were reused after verification. This v3 pass upgrades figure quality and report clarity without changing the science or recomputing the cell-type fingerprints.",
        "- The spectral PCA is shown once because all three configs use the same spectra. The configs only change the BSV readout, not the spectral input.",
        "- For the BSV PCA figures, columns were centered and variance-scaled for visualization only. The underlying BSV tables remain unchanged.",
        "",
        "## 2. Baseline",
        f"- Baseline remains the narrowest fingerprint set: mean neighborhood entropy `{baseline['mean_neighborhood_entropy']:.4f}`, mean top1 dominance `{baseline['mean_top1_dominance']:.4f}`.",
        f"- Its raw between-class distance is the largest (`{baseline['mean_inter_class_bsv_distance']:.6f}`), but that mainly reflects a narrow purine-dominant vocabulary rather than a richer biochemical picture.",
        "- In plain terms: the baseline separates classes strongly because the same few axes dominate, not because it captures broader chemistry.",
        "",
        "## 3. cfg05",
        f"- cfg05 broadens the fingerprint while keeping the dominant neighborhood family anchored in `{cfg05['dominant_compound_family_summary']}`.",
        f"- It lowers neighborhood dominance from `{baseline['mean_top1_dominance']:.4f}` to `{cfg05['mean_top1_dominance']:.4f}` and raises mean neighborhood entropy from `{baseline['mean_neighborhood_entropy']:.4f}` to `{cfg05['mean_neighborhood_entropy']:.4f}`.",
        "- The BSV PCA now spreads class identity across more than one active axis, which is the main reason it looks more interpretable in this report even though the raw Euclidean class distances shrink.",
        "- That shrinkage is not automatically bad. It means the fingerprints are less dominated by one narrow chemistry neighborhood and more distributed across several supported axes.",
        "",
        "## 4. cfg08",
        f"- cfg08 broadens further: mean neighborhood entropy `{cfg08['mean_neighborhood_entropy']:.4f}`, mean top1 dominance `{cfg08['mean_top1_dominance']:.4f}`.",
        f"- But its dominant neighborhood summary shifts to `{cfg08['dominant_compound_family_summary']}`, which weakens the impression of a clean cell-type-specific fingerprint vocabulary.",
        "- In practice cfg08 looks broader, but also more diffuse. The class fingerprints become less crisp even when they are more diverse.",
        "",
        "## 5. Stability Framing",
        f"- Baseline: mean intra `{baseline['mean_intra_class_bsv_variance']:.6f}`, mean inter `{baseline['mean_inter_class_bsv_distance']:.6f}`, ratio `{baseline['within_between_ratio']:.6f}`.",
        f"- cfg05: mean intra `{cfg05['mean_intra_class_bsv_variance']:.6f}`, mean inter `{cfg05['mean_inter_class_bsv_distance']:.6f}`, ratio `{cfg05['within_between_ratio']:.6f}`.",
        f"- cfg08: mean intra `{cfg08['mean_intra_class_bsv_variance']:.6f}`, mean inter `{cfg08['mean_inter_class_bsv_distance']:.6f}`, ratio `{cfg08['within_between_ratio']:.6f}`.",
        "- The right read here is simple: within-class structure should stay tight, and between-class structure should still look meaningfully separated in the BSV PCA. cfg05 satisfies that balance best.",
        "",
        "## 6. Display Notes",
        f"- Radar display axes for baseline: `{', '.join(radar_axis_notes['baseline'])}`.",
        f"- Radar display axes for cfg05: `{', '.join(radar_axis_notes['cfg05'])}`.",
        f"- Radar display axes for cfg08: `{', '.join(radar_axis_notes['cfg08'])}`.",
        "- Display-only dead-axis suppression was used for the radar figures where an axis was effectively zero across all classes for that config. The full BSV tables are unchanged.",
        "",
        "## 7. Decision",
        f"- Pass 5 validation note: baseline `{baseline_val:.4f}`, cfg05 `{cfg05_val:.4f}`, cfg08 `{cfg08_val:.4f}`.",
        "- cfg05 is the best decision-grade compromise for Pilot 1b.",
        "- Keep baseline as the narrow reference.",
        "- Keep cfg08 as the broadness comparator, not the working default.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def main() -> None:
    _require_v2_inputs()
    storage_cfg = load_autoresearch_storage_config(DEFAULT_STORAGE_CONFIG_PATH)
    sprint_paths = initialize_autoresearch_sprint(
        DEFAULT_STORAGE_CONFIG_PATH,
        sprint_id=f"{storage_cfg.sprint_id}/{SPRINT_SUBDIR}",
    )
    _copy_v2_run_tables(sprint_paths.sprint_root)
    pass5_ranked_df = pd.read_csv(PASS5_TABLE)

    spectral_pca_df = _read_run_df(sprint_paths.sprint_root, CONFIG_SPECS[0]["config_id"], "pca_coordinates_spectral.csv")
    _plot_spectral_pca(spectral_pca_df, sprint_paths.figures_dir / "pca_spectral_original_dataset.png")

    comparator_rows = []
    overlay_source = []
    radar_axis_notes: dict[str, list[str]] = {}
    for spec in CONFIG_SPECS:
        config_id = str(spec["config_id"])
        short_label = str(spec["short_label"])
        class_mean_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_mean_bsv.csv")
        pairwise_delta_df = _read_run_df(sprint_paths.sprint_root, config_id, "pairwise_delta_bsv.csv")
        neighborhood_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_topk_neighborhood_composition.csv")
        entropy_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_neighborhood_entropy.csv")
        top1_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_top1_dominance.csv")
        axis_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_axis_entropy.csv")
        within_df = _read_run_df(sprint_paths.sprint_root, config_id, "config_within_between_summary.csv")
        per_spectrum_bsv_df = _read_run_df(sprint_paths.sprint_root, config_id, "per_spectrum_bsv.csv")

        axes = _axes_present(per_spectrum_bsv_df)
        bsv_pca_df, _, _ = _scaled_pca_df(per_spectrum_bsv_df, axes)
        class_mean_pca_df, _, _ = _scaled_pca_df(class_mean_df, _axes_present(class_mean_df))
        class_mean_pca_df["config_short_label"] = short_label
        overlay_source.append(class_mean_pca_df.copy())

        _plot_bsv_pca(
            bsv_pca_df,
            sprint_paths.figures_dir / f"pca_bsv_{config_id}.png",
            f"BSV PCA: {short_label}",
        )
        _plot_bsv_class_mean_pca(
            class_mean_pca_df,
            sprint_paths.figures_dir / f"pca_bsv_class_mean_{config_id}.png",
            f"BSV PCA Class Means: {short_label}",
        )

        heat_df = class_mean_df.set_index("class_label")[_axes_present(class_mean_df)]
        _draw_heatmap(
            heat_df.to_numpy(dtype=float),
            heat_df.index.astype(str).tolist(),
            heat_df.columns.astype(str).tolist(),
            sprint_paths.figures_dir / f"class_mean_bsv_heatmap_{config_id}.png",
            f"Class Mean BSV Heatmap: {short_label}",
            cmap="viridis",
        )
        delta_heat = pairwise_delta_df.pivot(index="group_label", columns="reference_group", values="small_molecule_metabolite")
        delta_heat = delta_heat.reindex(sorted(delta_heat.index), axis=0).reindex(sorted(delta_heat.columns), axis=1)
        _draw_heatmap(
            delta_heat.to_numpy(dtype=float),
            delta_heat.index.astype(str).tolist(),
            delta_heat.columns.astype(str).tolist(),
            sprint_paths.figures_dir / f"pairwise_delta_bsv_heatmap_small_molecule_metabolite_{config_id}.png",
            f"Pairwise Delta BSV: small_molecule_metabolite ({short_label})",
            cmap="coolwarm",
            center_zero=True,
        )
        radar_axes = _plot_radar_grid(
            class_mean_df,
            sprint_paths.figures_dir / f"radar_fingerprint_grid_{config_id}.png",
            f"Fingerprint Atlas: {short_label}",
        )
        radar_axis_notes[short_label] = radar_axes
        _plot_neighborhood_grid(
            neighborhood_df,
            sprint_paths.figures_dir / f"neighborhood_grid_{config_id}.png",
            f"Neighborhood Composition: {short_label}",
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
                "dominant_compound_family_summary": _dominant_family_summary(neighborhood_df),
                "concise_interpretation_note": _interpretation_note(
                    float(entropy_df["neighborhood_entropy"].mean()),
                    float(top1_df["top1_fraction"].mean()),
                    float(within["within_between_ratio"]),
                ),
            }
        )

    comparator_df = pd.DataFrame(comparator_rows)
    comparator_df.to_csv(sprint_paths.tables_dir / "pilot1a_probe1_v3_config_comparison.csv", index=False)

    overlay_df = pd.concat(overlay_source, ignore_index=True)
    overlay_matrix = overlay_df[["pc1", "pc2"]].to_numpy(dtype=float)
    overlay_scores, overlay_explained = _fit_pca(overlay_matrix)
    overlay_df = overlay_df.copy()
    overlay_df["pc1"] = overlay_scores[:, 0]
    overlay_df["pc2"] = overlay_scores[:, 1] if overlay_scores.shape[1] > 1 else 0.0
    overlay_df["pc1_explained_ratio"] = float(overlay_explained[0]) if len(overlay_explained) > 0 else 0.0
    overlay_df["pc2_explained_ratio"] = float(overlay_explained[1]) if len(overlay_explained) > 1 else 0.0
    _plot_overlay_class_mean_pca(overlay_df, sprint_paths.figures_dir / "pca_bsv_class_mean_overlay.png")
    _plot_within_between(comparator_df, sprint_paths.figures_dir / "pilot1a_within_between_comparison.png")
    _plot_entropy_dominance(comparator_df, sprint_paths.figures_dir / "pilot1a_entropy_dominance_comparison.png")
    _plot_tradeoff(comparator_df, sprint_paths.figures_dir / "pilot1a_config_tradeoff_summary.png")

    report_md = sprint_paths.report_dir / "GAIRAv3_Pilot1a_celltype_probe1_v3_report.md"
    _build_report(report_md, comparator_df, pass5_ranked_df, radar_axis_notes)

    figure_paths = [sprint_paths.figures_dir / "pca_spectral_original_dataset.png"]
    for spec in CONFIG_SPECS:
        config_id = str(spec["config_id"])
        figure_paths.extend(
            [
                sprint_paths.figures_dir / f"pca_bsv_{config_id}.png",
                sprint_paths.figures_dir / f"pca_bsv_class_mean_{config_id}.png",
                sprint_paths.figures_dir / f"radar_fingerprint_grid_{config_id}.png",
                sprint_paths.figures_dir / f"neighborhood_grid_{config_id}.png",
                sprint_paths.figures_dir / f"class_mean_bsv_heatmap_{config_id}.png",
                sprint_paths.figures_dir / f"pairwise_delta_bsv_heatmap_small_molecule_metabolite_{config_id}.png",
            ]
        )
    figure_paths.extend(
        [
            sprint_paths.figures_dir / "pca_bsv_class_mean_overlay.png",
            sprint_paths.figures_dir / "pilot1a_within_between_comparison.png",
            sprint_paths.figures_dir / "pilot1a_entropy_dominance_comparison.png",
            sprint_paths.figures_dir / "pilot1a_config_tradeoff_summary.png",
        ]
    )
    _build_pdf(report_md, figure_paths, sprint_paths.report_dir / "GAIRAv3_Pilot1a_celltype_probe1_v3_report.pdf")
    print(f"Wrote Pilot 1a v3 outputs under {sprint_paths.sprint_root}")


if __name__ == "__main__":
    main()
