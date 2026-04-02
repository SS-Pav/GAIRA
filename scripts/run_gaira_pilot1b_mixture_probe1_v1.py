from __future__ import annotations

import json
import math
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
from gaira.demo.gaira_pilot_utils import (
    ALL_AXES,
    build_class_axis_entropy,
    build_class_neighborhood_entropy,
    build_class_top1_dominance,
    build_class_topk_neighborhood_composition,
    build_pdf_report,
    infer_mixture_order,
    pairwise_delta_bsv,
    plot_bsv_heatmap,
)


ROOT = Path(__file__).resolve().parents[1]
COMPARATOR_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot1_comparator_cfg05_cfg08"
)
PILOT1A_V5_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot1a_celltype_probe1_v5"
)
SPRINT_SUBDIR = "pilot1b_mixture_probe1_v1"
SUBSET_ALIAS = "small2023_mixture_probe1"

CONFIG_SPECS = [
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

REUSED_SOURCE_FILES = [
    "per_spectrum_bsv.csv",
    "class_mean_bsv.csv",
    "pairwise_delta_bsv.csv",
    "class_topk_neighborhood_composition.csv",
    "class_neighborhood_entropy.csv",
    "class_top1_dominance.csv",
    "class_axis_entropy.csv",
    "retrieval_hit_summary_by_class.csv",
    "per_spectrum_retrieval_hits.csv",
    "pca_coordinates.csv",
    "mixture_progression_summary.csv",
    "inter_class_bsv_distance.csv",
    "intra_class_bsv_variance.csv",
]

FAMILY_ORDER = [
    "purine_core_like",
    "methylated_purine_like",
    "guanidine_like",
    "sulfur_small_molecule_like",
    "aromatic_small_molecule_like",
    "generic_other_metabolite",
]

CLASS_COLORS = {
    "c00": "#355070",
    "c01": "#6d597a",
    "c10": "#b56576",
    "c25": "#e56b6f",
    "c50": "#eaac8b",
    "c100": "#f4a261",
}

CONFIG_COLORS = {"cfg05": "#f3722c", "cfg08": "#43aa8b"}


def _require_inputs() -> None:
    if not COMPARATOR_ROOT.exists():
        raise RuntimeError(f"Missing comparator root: {COMPARATOR_ROOT}")
    if not PILOT1A_V5_ROOT.exists():
        raise RuntimeError(f"Missing Pilot 1a v5 root: {PILOT1A_V5_ROOT}")
    for spec in CONFIG_SPECS:
        src_root = COMPARATOR_ROOT / "runs" / str(spec["config_id"]) / SUBSET_ALIAS / "tables"
        for name in REUSED_SOURCE_FILES:
            path = src_root / name
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(f"Missing source artifact: {path}")
        endpoint_root = PILOT1A_V5_ROOT / "runs" / str(spec["config_id"])
        for name in [
            "class_mean_bsv.csv",
            "delta_class_mean_bsv.csv",
            "class_family_fingerprint.csv",
            "class_top1_dominance.csv",
            "class_neighborhood_entropy.csv",
        ]:
            path = endpoint_root / name
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(f"Missing endpoint artifact: {path}")


def _read_source_df(config_id: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(COMPARATOR_ROOT / "runs" / config_id / SUBSET_ALIAS / "tables" / filename)


def _copy_reused_tables(run_dir: Path, config_id: str) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name in REUSED_SOURCE_FILES:
        src = COMPARATOR_ROOT / "runs" / config_id / SUBSET_ALIAS / "tables" / name
        dst_name = "pca_coordinates_spectral.csv" if name == "pca_coordinates.csv" else name
        shutil.copy2(src, run_dir / dst_name)
        shutil.copy2(src, tables_dir / dst_name)


def _axes_present(df: pd.DataFrame) -> list[str]:
    return [axis for axis in ALL_AXES if axis in df.columns]


def _compound_to_fine_family(name: str) -> str:
    lower = str(name).strip().lower()
    if any(token in lower for token in ["3-methyladenine", "methyladenine"]):
        return "methylated_purine_like"
    if any(token in lower for token in ["guanidine", "guanidino"]):
        return "guanidine_like"
    if any(token in lower for token in ["cyste", "glutath", "methion", "seleno", "sulfoximine", "sulfur"]):
        return "sulfur_small_molecule_like"
    if any(token in lower for token in ["tyr", "trypt", "phenyl", "indole", "dopamine", "3-methoxytyramine"]):
        return "aromatic_small_molecule_like"
    if any(token in lower for token in ["adenine", "xanth", "hypox", "uric", "urate", "inos", "purine"]):
        return "purine_core_like"
    return "generic_other_metabolite"


def _build_family_fingerprint(class_neighborhood_df: pd.DataFrame) -> pd.DataFrame:
    df = class_neighborhood_df.copy()
    df["neighborhood_family"] = df["compound_label"].astype(str).map(_compound_to_fine_family)
    grouped = (
        df.groupby(["class_label", "neighborhood_family"], as_index=False)["support_fraction"]
        .sum()
        .rename(columns={"support_fraction": "family_support_fraction"})
    )
    rows = []
    for class_label in sorted(df["class_label"].astype(str).unique().tolist(), key=_mixture_sort_key):
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
    for class_label in sorted(class_mean_df["class_label"].astype(str).tolist(), key=_mixture_sort_key):
        abs_row = class_mean_df[class_mean_df["class_label"].astype(str) == class_label].iloc[0]
        delta_row = delta_df[delta_df["class_label"].astype(str) == class_label].iloc[0]
        fam_row = family_df[family_df["class_label"].astype(str) == class_label].copy()
        dominant_fam = fam_row.sort_values("family_support_fraction", ascending=False).iloc[0]
        entropy = float(
            entropy_df[entropy_df["class_label"].astype(str) == class_label]["neighborhood_entropy"].iloc[0]
        )
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


def _mixture_sort_key(label: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    return (int(digits) if digits else 10**9, str(label))


def _fit_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, keepdims=True)
    std = np.where(std < 1e-9, 1.0, std)
    centered = centered / std
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * s[:2]
    denom = max(float((s**2).sum()), 1e-12)
    explained = (s[:2] ** 2) / denom if len(s) >= 2 else np.array([1.0, 0.0])
    return scores, explained


def _pca_dataframe(df: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    matrix = df[axes].to_numpy(dtype=float)
    scores, explained = _fit_pca(matrix)
    out = df[["class_label"]].copy()
    if "sample_key" in df.columns:
        out["sample_key"] = df["sample_key"].astype(str)
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1] if scores.shape[1] > 1 else 0.0
    out["pc1_explained_ratio"] = float(explained[0])
    out["pc2_explained_ratio"] = float(explained[1]) if len(explained) > 1 else 0.0
    return out


def _plot_scatter_pca(
    pca_df: pd.DataFrame,
    output_path: Path,
    title: str,
    *,
    annotate: bool = False,
) -> None:
    labels = sorted(pca_df["class_label"].astype(str).unique().tolist(), key=_mixture_sort_key)
    fig, ax = plt.subplots(figsize=(9.4, 6.8))
    for label in labels:
        sub = pca_df[pca_df["class_label"].astype(str) == label]
        ax.scatter(
            sub["pc1"],
            sub["pc2"],
            s=40 if not annotate else 85,
            alpha=0.84,
            label=label,
            color=CLASS_COLORS.get(label, "#4c78a8"),
            edgecolors="white",
            linewidths=0.5,
        )
        if annotate and not sub.empty:
            row = sub.iloc[0]
            ax.annotate(
                label,
                (float(row["pc1"]), float(row["pc2"])),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.72},
            )
    ax.set_xlabel(f"PC1 ({float(pca_df['pc1_explained_ratio'].iloc[0]) * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({float(pca_df['pc2_explained_ratio'].iloc[0]) * 100:.1f}% var)")
    ax.set_title(title)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    if not annotate:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout(rect=[0.0, 0.0, 0.83, 1.0])
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_delta_heatmap(delta_df: pd.DataFrame, output_path: Path, title: str) -> None:
    axes = _axes_present(delta_df)
    work = delta_df.copy()
    work["class_label"] = work["class_label"].astype(str)
    work = work.sort_values("class_label", key=lambda s: s.map(_mixture_sort_key))
    matrix = work.set_index("class_label")[axes].to_numpy(dtype=float)
    lim = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 1e-8)
    fig, ax = plt.subplots(figsize=(max(8.2, 0.8 * len(axes) + 3.5), 5.2))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_xticks(np.arange(len(axes)))
    ax.set_xticklabels(axes, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(work)))
    ax.set_yticklabels(work["class_label"].tolist())
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_family_heatmap(family_df: pd.DataFrame, output_path: Path, title: str) -> None:
    labels = sorted(family_df["class_label"].astype(str).unique().tolist(), key=_mixture_sort_key)
    heat = (
        family_df.pivot(index="class_label", columns="neighborhood_family", values="family_support_fraction")
        .reindex(labels)
        .reindex(FAMILY_ORDER, axis=1)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0.0, vmax=max(float(heat.to_numpy(dtype=float).max()), 1e-9))
    ax.set_xticks(np.arange(len(FAMILY_ORDER)))
    ax.set_xticklabels(FAMILY_ORDER, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_progression_alignment(alignment_df: pd.DataFrame, output_path: Path, title: str) -> None:
    work = alignment_df.sort_values("mixture_code_numeric").copy()
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for col, label, marker in [
        ("absolute_toward_high_score", "absolute", "o"),
        ("delta_toward_high_score", "delta", "s"),
        ("family_toward_high_score", "family", "^"),
        ("combined_toward_high_score", "combined", "D"),
    ]:
        ax.plot(work["class_label"], work[col], marker=marker, linewidth=2.0, label=label)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Toward high-endpoint score")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_adjacent_distance_progression(step_df: pd.DataFrame, output_path: Path, title: str) -> None:
    work = step_df.sort_values("left_code_numeric").copy()
    labels = [f"{a}->{b}" for a, b in zip(work["left_class"], work["right_class"], strict=False)]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    for i, (col, label) in enumerate(
        [
            ("absolute_adjacent_distance", "absolute"),
            ("delta_adjacent_distance", "delta"),
            ("family_adjacent_distance", "family"),
            ("combined_adjacent_distance", "combined"),
        ]
    ):
        ax.bar(x + (i - 1.5) * width, work[col], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Adjacent distance")
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_top1_progression(top1_df: pd.DataFrame, output_path: Path, title: str) -> None:
    work = top1_df.sort_values("class_label", key=lambda s: s.map(_mixture_sort_key)).copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.plot(work["class_label"], work["top1_fraction"], marker="o", linewidth=2.0, color="#6d597a")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Top1 dominance")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_endpoint_distance(alignment_df: pd.DataFrame, output_path: Path, title: str) -> None:
    work = alignment_df.sort_values("mixture_code_numeric").copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    for ax, prefix in zip(axes, ["delta", "family"], strict=False):
        ax.plot(work["class_label"], work[f"{prefix}_distance_to_low_endpoint"], marker="o", linewidth=2.0, label="to low")
        ax.plot(work["class_label"], work[f"{prefix}_distance_to_high_endpoint"], marker="s", linewidth=2.0, label="to high")
        ax.set_title(prefix.replace("_", " ").title())
        ax.set_ylabel("Distance")
        ax.grid(True, axis="y", alpha=0.24, linewidth=0.5)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 0.84, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_endpoint_reference(endpoint_rows: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(len(endpoint_rows), 2, figsize=(12.4, 4.4 * len(endpoint_rows)))
    axes = np.atleast_2d(axes)
    for i, row in enumerate(endpoint_rows):
        abs_df = row["absolute_df"]
        fam_df = row["family_df"]
        heat = abs_df.set_index("class_label")[_axes_present(abs_df)]
        im = axes[i, 0].imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(heat.to_numpy(dtype=float).max()), 1e-9))
        axes[i, 0].set_xticks(np.arange(len(heat.columns)))
        axes[i, 0].set_xticklabels(heat.columns.tolist(), rotation=35, ha="right", fontsize=8)
        axes[i, 0].set_yticks(np.arange(len(heat.index)))
        axes[i, 0].set_yticklabels(heat.index.tolist(), fontsize=9)
        axes[i, 0].set_title(f"{row['short_label']} endpoint absolute BSV")
        fam = (
            fam_df.pivot(index="class_label", columns="family", values="family_fraction")
            .reindex(heat.index.tolist())
            .reindex(FAMILY_ORDER, axis=1)
            .fillna(0.0)
        )
        axes[i, 1].imshow(fam.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0.0, vmax=max(float(fam.to_numpy(dtype=float).max()), 1e-9))
        axes[i, 1].set_xticks(np.arange(len(fam.columns)))
        axes[i, 1].set_xticklabels(fam.columns.tolist(), rotation=35, ha="right", fontsize=8)
        axes[i, 1].set_yticks(np.arange(len(fam.index)))
        axes[i, 1].set_yticklabels(fam.index.tolist(), fontsize=9)
        axes[i, 1].set_title(f"{row['short_label']} endpoint family mix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_tradeoff(config_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    panels = [
        ("progression_combined_spearman", "Combined progression"),
        ("noncollapse_ratio", "Noncollapse ratio"),
        ("mean_top1_dominance", "Mean top1 dominance"),
        ("mean_neighborhood_entropy", "Mean neighborhood entropy"),
    ]
    for ax, (col, title) in zip(axes.ravel(), panels, strict=False):
        x = np.arange(len(config_df))
        ax.bar(
            x,
            config_df[col].to_numpy(dtype=float),
            color=[CONFIG_COLORS.get(x_, "#4c78a8") for x_ in config_df["config_short_label"]],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(config_df["config_short_label"].tolist())
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.5)
    fig.suptitle("Pilot 1b cfg05 vs cfg08 tradeoff summary")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _load_endpoint_tables(config_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_root = PILOT1A_V5_ROOT / "runs" / config_id
    return (
        pd.read_csv(run_root / "class_mean_bsv.csv"),
        pd.read_csv(run_root / "delta_class_mean_bsv.csv"),
        pd.read_csv(run_root / "class_family_fingerprint.csv"),
    )


def _choose_endpoint_pair(
    mixture_class_mean_df: pd.DataFrame,
    mixture_delta_df: pd.DataFrame,
    endpoint_class_mean_df: pd.DataFrame,
    endpoint_delta_df: pd.DataFrame,
) -> tuple[str, str]:
    mix_order = infer_mixture_order(mixture_class_mean_df["class_label"].astype(str).tolist())
    mix_low = mix_order[0]
    mix_high = mix_order[-1]
    axes = _axes_present(mixture_delta_df)
    endpoint_work = endpoint_delta_df.set_index("class_label")
    mix_work = mixture_delta_df.set_index("class_label")
    low_vec = mix_work.loc[mix_low, axes].to_numpy(dtype=float)
    high_vec = mix_work.loc[mix_high, axes].to_numpy(dtype=float)
    endpoint_labels = endpoint_work.index.astype(str).tolist()
    low_dists = sorted(
        [(label, float(np.linalg.norm(low_vec - endpoint_work.loc[label, axes].to_numpy(dtype=float)))) for label in endpoint_labels],
        key=lambda x: x[1],
    )
    high_dists = sorted(
        [(label, float(np.linalg.norm(high_vec - endpoint_work.loc[label, axes].to_numpy(dtype=float)))) for label in endpoint_labels],
        key=lambda x: x[1],
    )
    low_label = low_dists[0][0]
    high_label = high_dists[0][0]
    if high_label == low_label:
        for label, _ in high_dists[1:]:
            if label != low_label:
                high_label = label
                break
    return str(low_label), str(high_label)


def _build_endpoint_alignment_summary(
    *,
    config_id: str,
    short_label: str,
    mixture_class_mean_df: pd.DataFrame,
    mixture_delta_df: pd.DataFrame,
    mixture_family_df: pd.DataFrame,
    endpoint_class_mean_df: pd.DataFrame,
    endpoint_delta_df: pd.DataFrame,
    endpoint_family_df: pd.DataFrame,
    low_endpoint_class: str,
    high_endpoint_class: str,
) -> pd.DataFrame:
    ordered = infer_mixture_order(mixture_class_mean_df["class_label"].astype(str).tolist())
    axes = _axes_present(mixture_class_mean_df)
    abs_mix = mixture_class_mean_df.set_index("class_label")
    delta_mix = mixture_delta_df.set_index("class_label")
    fam_mix = (
        mixture_family_df.pivot(index="class_label", columns="neighborhood_family", values="family_support_fraction")
        .reindex(columns=FAMILY_ORDER)
        .fillna(0.0)
    )
    abs_end = endpoint_class_mean_df.set_index("class_label")
    delta_end = endpoint_delta_df.set_index("class_label")
    fam_end = (
        endpoint_family_df.pivot(index="class_label", columns="family", values="family_fraction")
        .reindex(columns=FAMILY_ORDER)
        .fillna(0.0)
    )

    rows = []
    for label in ordered:
        abs_vec = abs_mix.loc[label, axes].to_numpy(dtype=float)
        delta_vec = delta_mix.loc[label, axes].to_numpy(dtype=float)
        fam_vec = fam_mix.loc[label, FAMILY_ORDER].to_numpy(dtype=float)

        abs_low = abs_end.loc[low_endpoint_class, axes].to_numpy(dtype=float)
        abs_high = abs_end.loc[high_endpoint_class, axes].to_numpy(dtype=float)
        delta_low = delta_end.loc[low_endpoint_class, axes].to_numpy(dtype=float)
        delta_high = delta_end.loc[high_endpoint_class, axes].to_numpy(dtype=float)
        fam_low = fam_end.loc[low_endpoint_class, FAMILY_ORDER].to_numpy(dtype=float)
        fam_high = fam_end.loc[high_endpoint_class, FAMILY_ORDER].to_numpy(dtype=float)

        dist_abs_low = float(np.linalg.norm(abs_vec - abs_low))
        dist_abs_high = float(np.linalg.norm(abs_vec - abs_high))
        dist_delta_low = float(np.linalg.norm(delta_vec - delta_low))
        dist_delta_high = float(np.linalg.norm(delta_vec - delta_high))
        dist_fam_low = float(np.linalg.norm(fam_vec - fam_low))
        dist_fam_high = float(np.linalg.norm(fam_vec - fam_high))

        def toward_high(d_low: float, d_high: float) -> float:
            return d_low / max(d_low + d_high, 1e-12)

        rows.append(
            {
                "config_id": config_id,
                "config_short_label": short_label,
                "class_label": label,
                "mixture_code_numeric": int("".join(ch for ch in label if ch.isdigit()) or 0),
                "low_endpoint_class": low_endpoint_class,
                "high_endpoint_class": high_endpoint_class,
                "absolute_distance_to_low_endpoint": dist_abs_low,
                "absolute_distance_to_high_endpoint": dist_abs_high,
                "absolute_toward_high_score": toward_high(dist_abs_low, dist_abs_high),
                "delta_distance_to_low_endpoint": dist_delta_low,
                "delta_distance_to_high_endpoint": dist_delta_high,
                "delta_toward_high_score": toward_high(dist_delta_low, dist_delta_high),
                "family_distance_to_low_endpoint": dist_fam_low,
                "family_distance_to_high_endpoint": dist_fam_high,
                "family_toward_high_score": toward_high(dist_fam_low, dist_fam_high),
            }
        )
    out = pd.DataFrame(rows)
    out["combined_toward_high_score"] = out[
        ["absolute_toward_high_score", "delta_toward_high_score", "family_toward_high_score"]
    ].mean(axis=1)
    return out.sort_values("mixture_code_numeric").reset_index(drop=True)


def _compute_progression_metrics(alignment_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = alignment_df.sort_values("mixture_code_numeric").copy()
    numeric = pd.Series(work["mixture_code_numeric"].to_numpy(dtype=float))
    steps = []
    for left, right in zip(work.itertuples(index=False), work.iloc[1:].itertuples(index=False), strict=False):
        steps.append(
            {
                "config_id": left.config_id,
                "config_short_label": left.config_short_label,
                "left_class": left.class_label,
                "right_class": right.class_label,
                "left_code_numeric": left.mixture_code_numeric,
                "absolute_adjacent_distance": abs(float(right.absolute_toward_high_score) - float(left.absolute_toward_high_score)),
                "delta_adjacent_distance": abs(float(right.delta_toward_high_score) - float(left.delta_toward_high_score)),
                "family_adjacent_distance": abs(float(right.family_toward_high_score) - float(left.family_toward_high_score)),
                "combined_adjacent_distance": abs(float(right.combined_toward_high_score) - float(left.combined_toward_high_score)),
            }
        )
    step_df = pd.DataFrame(steps)
    summary = {
        "config_id": str(work["config_id"].iloc[0]),
        "config_short_label": str(work["config_short_label"].iloc[0]),
        "progression_absolute_spearman": float(numeric.corr(work["absolute_toward_high_score"], method="spearman")),
        "progression_delta_spearman": float(numeric.corr(work["delta_toward_high_score"], method="spearman")),
        "progression_family_spearman": float(numeric.corr(work["family_toward_high_score"], method="spearman")),
        "progression_combined_spearman": float(numeric.corr(work["combined_toward_high_score"], method="spearman")),
        "max_combined_jump": float(step_df["combined_adjacent_distance"].max()) if not step_df.empty else 0.0,
        "mean_combined_adjacent_distance": float(step_df["combined_adjacent_distance"].mean()) if not step_df.empty else 0.0,
        "endpoint_combined_separation": float(work["combined_toward_high_score"].iloc[-1] - work["combined_toward_high_score"].iloc[0]),
        "collapse_region_count": int((step_df["combined_adjacent_distance"] < 0.05).sum()) if not step_df.empty else 0,
    }
    return pd.DataFrame([summary]), step_df


def _compute_noncollapse_metrics(
    *,
    config_id: str,
    short_label: str,
    class_mean_bsv_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    family_df: pd.DataFrame,
    top1_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
) -> pd.DataFrame:
    ordered = infer_mixture_order(class_mean_bsv_df["class_label"].astype(str).tolist())
    abs_work = class_mean_bsv_df.set_index("class_label").loc[ordered, _axes_present(class_mean_bsv_df)]
    delta_work = delta_df.set_index("class_label").loc[ordered, _axes_present(delta_df)]
    fam_work = (
        family_df.pivot(index="class_label", columns="neighborhood_family", values="family_support_fraction")
        .reindex(ordered)
        .reindex(FAMILY_ORDER, axis=1)
        .fillna(0.0)
    )
    unique_abs = np.unique(np.round(abs_work.to_numpy(dtype=float), 8), axis=0).shape[0]
    unique_delta = np.unique(np.round(delta_work.to_numpy(dtype=float), 8), axis=0).shape[0]
    adj_delta = []
    for left, right in zip(ordered[:-1], ordered[1:], strict=False):
        diff = delta_work.loc[left].to_numpy(dtype=float) - delta_work.loc[right].to_numpy(dtype=float)
        adj_delta.append(float(np.linalg.norm(diff)))
    intermediate = [label for label in ordered if label not in {ordered[0], ordered[-1]}]
    intermediate_distinct = 0
    for label in intermediate:
        vec = delta_work.loc[label].to_numpy(dtype=float)
        others = [o for o in ordered if o != label]
        min_dist = min(float(np.linalg.norm(vec - delta_work.loc[o].to_numpy(dtype=float))) for o in others)
        if min_dist > 1e-3:
            intermediate_distinct += 1
    endpoint_sep = float(np.linalg.norm(delta_work.loc[ordered[0]].to_numpy(dtype=float) - delta_work.loc[ordered[-1]].to_numpy(dtype=float)))
    return pd.DataFrame(
        [
            {
                "config_id": config_id,
                "config_short_label": short_label,
                "noncollapse_ratio": unique_delta / max(len(ordered), 1),
                "unique_absolute_profile_ratio": unique_abs / max(len(ordered), 1),
                "adjacent_nonzero_ratio": float(np.mean([x > 1e-6 for x in adj_delta])) if adj_delta else 0.0,
                "min_adjacent_delta_distance": float(min(adj_delta)) if adj_delta else 0.0,
                "mean_adjacent_delta_distance": float(np.mean(adj_delta)) if adj_delta else 0.0,
                "intermediate_distinct_count": int(intermediate_distinct),
                "endpoint_delta_separation": endpoint_sep,
                "mean_top1_dominance": float(top1_df["top1_fraction"].mean()),
                "mean_neighborhood_entropy": float(entropy_df["neighborhood_entropy"].mean()),
                "mean_family_distance": float(
                    np.mean(
                        [
                            np.linalg.norm(
                                fam_work.loc[a].to_numpy(dtype=float) - fam_work.loc[b].to_numpy(dtype=float)
                            )
                            for i, a in enumerate(ordered)
                            for b in ordered[i + 1 :]
                        ]
                    )
                )
                if len(ordered) > 1
                else 0.0,
            }
        ]
    )


def _build_report_markdown(
    output_path: Path,
    *,
    config_metrics_df: pd.DataFrame,
    endpoint_alignment_df: pd.DataFrame,
) -> None:
    cfg05 = config_metrics_df[config_metrics_df["config_short_label"] == "cfg05"].iloc[0]
    cfg08 = config_metrics_df[config_metrics_df["config_short_label"] == "cfg08"].iloc[0]
    cfg05_endpoints = endpoint_alignment_df[endpoint_alignment_df["config_short_label"] == "cfg05"].iloc[0]
    cfg08_endpoints = endpoint_alignment_df[endpoint_alignment_df["config_short_label"] == "cfg08"].iloc[0]
    lines = [
        "# GAIRAv3 Pilot 1b: small2023 mixture Probe 1",
        "",
        "## 1. Overview",
        "- This is a fixed-config Probe 1 mixture evaluation only.",
        "- The endpoint references come from the locked Pilot 1a v5 cell-type fingerprints.",
        "- No new autoresearch or parameter tuning was run here.",
        "",
        "## 2. Endpoint Anchoring",
        f"- cfg05 low/high endpoint references: `{cfg05_endpoints['low_endpoint_class']}` -> `{cfg05_endpoints['high_endpoint_class']}`",
        f"- cfg08 low/high endpoint references: `{cfg08_endpoints['low_endpoint_class']}` -> `{cfg08_endpoints['high_endpoint_class']}`",
        "- These anchors were selected by closest-match to the observed mixture endpoints in delta-BSV space, using the existing Pilot 1a v5 endpoint fingerprints.",
        "",
        "## 3. cfg05 vs cfg08",
        f"- cfg05 progression spearman `{cfg05['progression_combined_spearman']:.4f}`, noncollapse `{cfg05['noncollapse_ratio']:.4f}`, mean top1 dominance `{cfg05['mean_top1_dominance']:.4f}`",
        f"- cfg08 progression spearman `{cfg08['progression_combined_spearman']:.4f}`, noncollapse `{cfg08['noncollapse_ratio']:.4f}`, mean top1 dominance `{cfg08['mean_top1_dominance']:.4f}`",
        f"- cfg05 intermediate distinct classes `{int(cfg05['intermediate_distinct_count'])}` of 4",
        f"- cfg08 intermediate distinct classes `{int(cfg08['intermediate_distinct_count'])}` of 4",
        "",
        "## 4. Interpretation",
        "- The key question here is not perfect linearity. It is whether the ordered series shows a coherent movement rather than the old collapse-plus-endpoint-jump pattern.",
        "- Absolute BSV, delta-BSV, and neighborhood-family alignment were evaluated jointly.",
        "- Molecular language remains conservative: these are support-pattern shifts within the current vocabulary, not direct molecule identification.",
        "",
        "## 5. Direct Answers",
        f"1. cfg05 produces a meaningful ordered mixture progression: `{bool(cfg05['progression_combined_spearman'] > 0.3 and cfg05['noncollapse_ratio'] > 0.6)}`",
        f"2. cfg08 produces a more convincing progression than cfg05: `{bool(cfg08['progression_combined_spearman'] > cfg05['progression_combined_spearman'] and cfg08['noncollapse_ratio'] >= cfg05['noncollapse_ratio'])}`",
        f"3. Intermediate classes are distinguishable under cfg05: `{bool(cfg05['intermediate_distinct_count'] >= 3)}`; under cfg08: `{bool(cfg08['intermediate_distinct_count'] >= 3)}`",
        f"4. Mixtures move toward endpoint fingerprints coherently under cfg05: `{bool(cfg05['endpoint_combined_separation'] > 0.5)}`; under cfg08: `{bool(cfg08['endpoint_combined_separation'] > 0.5)}`",
        f"5. Progression is smoother under cfg05: `{bool(cfg05['max_combined_jump'] < 0.45)}`; under cfg08: `{bool(cfg08['max_combined_jump'] < 0.45)}`",
    ]
    if float(cfg05["noncollapse_ratio"]) > float(cfg08["noncollapse_ratio"]) and int(
        cfg05["intermediate_distinct_count"]
    ) > int(cfg08["intermediate_distinct_count"]):
        lines.append("6. Recommendation: move `cfg05` forward to Pilot 1c for probe-consistency testing.")
    else:
        lines.append("6. Recommendation: move `cfg08` forward to Pilot 1c for probe-consistency testing.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _require_inputs()
    storage_cfg = load_autoresearch_storage_config(DEFAULT_STORAGE_CONFIG_PATH)
    sprint_paths = initialize_autoresearch_sprint(
        DEFAULT_STORAGE_CONFIG_PATH,
        sprint_id=f"{storage_cfg.sprint_id}/{SPRINT_SUBDIR}",
    )
    sprint_paths.tables_dir.mkdir(parents=True, exist_ok=True)
    sprint_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    sprint_paths.report_dir.mkdir(parents=True, exist_ok=True)

    source_spectral_pca = _read_source_df(CONFIG_SPECS[0]["config_id"], "pca_coordinates.csv")
    shutil.copy2(
        COMPARATOR_ROOT / "runs" / CONFIG_SPECS[0]["config_id"] / SUBSET_ALIAS / "tables" / "pca_coordinates.csv",
        sprint_paths.tables_dir / "pca_spectral_probe1_mixture.csv",
    )
    _plot_scatter_pca(
        source_spectral_pca.rename(columns={"pc1": "pc1", "pc2": "pc2"}),
        sprint_paths.figures_dir / "pca_spectral_probe1_mixture.png",
        "Spectral PCA: small2023 mixture Probe 1",
        annotate=False,
    )

    comparator_rows = []
    alignment_rows = []
    endpoint_reference_rows = []
    report_figure_paths = [sprint_paths.figures_dir / "pca_spectral_probe1_mixture.png"]

    for spec in CONFIG_SPECS:
        config_id = str(spec["config_id"])
        short_label = str(spec["short_label"])
        run_dir = sprint_paths.runs_dir / config_id
        run_dir.mkdir(parents=True, exist_ok=True)
        tables_dir = run_dir / "tables"
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        _copy_reused_tables(run_dir, config_id)

        per_spectrum_bsv_df = pd.read_csv(run_dir / "per_spectrum_bsv.csv")
        class_mean_bsv_df = pd.read_csv(run_dir / "class_mean_bsv.csv")
        class_neighborhood_df = pd.read_csv(run_dir / "class_topk_neighborhood_composition.csv")
        class_neighborhood_entropy_df = pd.read_csv(run_dir / "class_neighborhood_entropy.csv")
        class_top1_dominance_df = pd.read_csv(run_dir / "class_top1_dominance.csv")
        class_axis_entropy_df = pd.read_csv(run_dir / "class_axis_entropy.csv")

        axes = _axes_present(class_mean_bsv_df)
        cohort_mean = per_spectrum_bsv_df[axes].mean(axis=0)
        delta_class_df = class_mean_bsv_df[["sample_key", "dataset_id", "subset_id", "class_label"]].copy()
        delta_per_spectrum_df = per_spectrum_bsv_df[["sample_key", "dataset_id", "subset_id", "class_label"]].copy()
        for axis in axes:
            delta_class_df[axis] = class_mean_bsv_df[axis].to_numpy(dtype=float) - float(cohort_mean[axis])
            delta_per_spectrum_df[axis] = per_spectrum_bsv_df[axis].to_numpy(dtype=float) - float(cohort_mean[axis])
        delta_class_df.to_csv(run_dir / "class_mean_bsv_delta_vs_cohort.csv", index=False)
        delta_class_df.to_csv(tables_dir / "class_mean_bsv_delta_vs_cohort.csv", index=False)

        family_df = _build_family_fingerprint(class_neighborhood_df)
        family_df.to_csv(run_dir / "class_neighborhood_family_composition.csv", index=False)
        family_df.to_csv(tables_dir / "class_neighborhood_family_composition.csv", index=False)

        summary_df = _build_class_fingerprint_summary(
            config_id,
            short_label,
            class_mean_bsv_df,
            delta_class_df,
            family_df,
            class_neighborhood_entropy_df,
            class_top1_dominance_df,
        )
        summary_df.to_csv(run_dir / "class_fingerprint_summary.csv", index=False)
        summary_df.to_csv(tables_dir / "class_fingerprint_summary.csv", index=False)

        pca_bsv_df = _pca_dataframe(per_spectrum_bsv_df, axes)
        pca_bsv_class_mean_df = _pca_dataframe(class_mean_bsv_df, axes)
        pca_delta_df = _pca_dataframe(delta_per_spectrum_df, axes)
        pca_bsv_df.to_csv(run_dir / "pca_coordinates_bsv.csv", index=False)
        pca_bsv_df.to_csv(tables_dir / "pca_coordinates_bsv.csv", index=False)
        pca_bsv_class_mean_df.to_csv(run_dir / "pca_coordinates_bsv_class_mean.csv", index=False)
        pca_bsv_class_mean_df.to_csv(tables_dir / "pca_coordinates_bsv_class_mean.csv", index=False)

        endpoint_abs_df, endpoint_delta_df, endpoint_family_df = _load_endpoint_tables(config_id)
        low_endpoint, high_endpoint = _choose_endpoint_pair(
            class_mean_bsv_df,
            delta_class_df,
            endpoint_abs_df,
            endpoint_delta_df,
        )

        endpoint_alignment_df = _build_endpoint_alignment_summary(
            config_id=config_id,
            short_label=short_label,
            mixture_class_mean_df=class_mean_bsv_df,
            mixture_delta_df=delta_class_df,
            mixture_family_df=family_df,
            endpoint_class_mean_df=endpoint_abs_df,
            endpoint_delta_df=endpoint_delta_df,
            endpoint_family_df=endpoint_family_df,
            low_endpoint_class=low_endpoint,
            high_endpoint_class=high_endpoint,
        )
        progression_df, step_df = _compute_progression_metrics(endpoint_alignment_df)
        noncollapse_df = _compute_noncollapse_metrics(
            config_id=config_id,
            short_label=short_label,
            class_mean_bsv_df=class_mean_bsv_df,
            delta_df=delta_class_df,
            family_df=family_df,
            top1_df=class_top1_dominance_df,
            entropy_df=class_neighborhood_entropy_df,
        )
        progression_summary_df = endpoint_alignment_df.copy()

        endpoint_alignment_df.to_csv(run_dir / "endpoint_alignment_summary.csv", index=False)
        endpoint_alignment_df.to_csv(tables_dir / "endpoint_alignment_summary.csv", index=False)
        progression_summary_df.to_csv(run_dir / "mixture_progression_summary.csv", index=False)
        progression_summary_df.to_csv(tables_dir / "mixture_progression_summary.csv", index=False)
        progression_df.to_csv(run_dir / "progression_metrics.csv", index=False)
        progression_df.to_csv(tables_dir / "progression_metrics.csv", index=False)
        noncollapse_df.to_csv(run_dir / "noncollapse_metrics.csv", index=False)
        noncollapse_df.to_csv(tables_dir / "noncollapse_metrics.csv", index=False)
        step_df.to_csv(run_dir / "adjacent_progression_steps.csv", index=False)
        step_df.to_csv(tables_dir / "adjacent_progression_steps.csv", index=False)

        _plot_scatter_pca(
            pca_bsv_df,
            sprint_paths.figures_dir / f"pca_bsv_{short_label}.png",
            f"BSV PCA: {short_label}",
        )
        _plot_scatter_pca(
            pca_delta_df,
            sprint_paths.figures_dir / f"pca_delta_bsv_{short_label}.png",
            f"Delta-BSV PCA: {short_label}",
        )
        plot_bsv_heatmap(
            class_mean_bsv_df,
            sprint_paths.figures_dir / f"class_mean_bsv_heatmap_{short_label}.png",
            f"Class Mean BSV: {short_label}",
        )
        _plot_delta_heatmap(
            delta_class_df,
            sprint_paths.figures_dir / f"delta_fingerprint_grid_{short_label}.png",
            f"Delta fingerprint grid: {short_label}",
        )
        _plot_family_heatmap(
            family_df,
            sprint_paths.figures_dir / f"neighborhood_family_fingerprint_grid_{short_label}.png",
            f"Neighborhood family fingerprint grid: {short_label}",
        )
        _plot_progression_alignment(
            endpoint_alignment_df,
            sprint_paths.figures_dir / f"mixture_progression_alignment_{short_label}.png",
            f"Mixture progression alignment: {short_label}",
        )
        _plot_adjacent_distance_progression(
            step_df,
            sprint_paths.figures_dir / f"adjacent_distance_progression_{short_label}.png",
            f"Adjacent distance progression: {short_label}",
        )
        _plot_top1_progression(
            class_top1_dominance_df,
            sprint_paths.figures_dir / f"top1_dominance_progression_{short_label}.png",
            f"Top1 dominance progression: {short_label}",
        )
        _plot_endpoint_distance(
            endpoint_alignment_df,
            sprint_paths.figures_dir / f"mixture_to_endpoint_distance_{short_label}.png",
            f"Mixture-to-endpoint distances: {short_label}",
        )

        report_figure_paths.extend(
            [
                sprint_paths.figures_dir / f"pca_bsv_{short_label}.png",
                sprint_paths.figures_dir / f"pca_delta_bsv_{short_label}.png",
                sprint_paths.figures_dir / f"class_mean_bsv_heatmap_{short_label}.png",
                sprint_paths.figures_dir / f"delta_fingerprint_grid_{short_label}.png",
                sprint_paths.figures_dir / f"neighborhood_family_fingerprint_grid_{short_label}.png",
                sprint_paths.figures_dir / f"mixture_progression_alignment_{short_label}.png",
                sprint_paths.figures_dir / f"adjacent_distance_progression_{short_label}.png",
                sprint_paths.figures_dir / f"top1_dominance_progression_{short_label}.png",
                sprint_paths.figures_dir / f"mixture_to_endpoint_distance_{short_label}.png",
            ]
        )

        metric_row = progression_df.merge(noncollapse_df, on=["config_id", "config_short_label"], how="left")
        comparator_rows.append(metric_row)
        alignment_rows.append(endpoint_alignment_df)
        endpoint_reference_rows.append(
            {
                "short_label": short_label,
                "absolute_df": endpoint_abs_df[endpoint_abs_df["class_label"].astype(str).isin([low_endpoint, high_endpoint])].copy(),
                "family_df": endpoint_family_df[endpoint_family_df["class_label"].astype(str).isin([low_endpoint, high_endpoint])].copy(),
            }
        )

        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "subset_alias": SUBSET_ALIAS,
                    "representation_mode": "raw_direct_bsv_input",
                    "grounding_mode": "universal_only",
                    "universal_grounding_filter_mode": spec["filter_mode"],
                    "aggregation_mode": "class_mean_spectrum_then_bsv",
                    "ontology_mode": "tier1_plus_subclass",
                    "similarity_metric": "cosine",
                    "plausibility_scoring_mode": "baseline_plausibility",
                    "pca_grouping_mode": "class_label_groups",
                    "top_k": spec["top_k"],
                    "weighting_mode": spec["weighting_mode"],
                    "weighting_param": spec["weighting_param"],
                    "diversity_mode": spec["diversity_mode"],
                    "source_run_root": str(COMPARATOR_ROOT / "runs" / config_id / SUBSET_ALIAS),
                    "endpoint_reference_root": str(PILOT1A_V5_ROOT / "runs" / config_id),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    comparator_df = pd.concat(comparator_rows, ignore_index=True)
    endpoint_alignment_all_df = pd.concat(alignment_rows, ignore_index=True)

    comparator_df.to_csv(sprint_paths.tables_dir / "pilot1b_cfg05_vs_cfg08_comparison.csv", index=False)
    endpoint_alignment_all_df.to_csv(sprint_paths.tables_dir / "pilot1b_endpoint_alignment_comparison.csv", index=False)
    comparator_df.to_csv(sprint_paths.tables_dir / "progression_metrics.csv", index=False)
    comparator_df[
        [
            "config_id",
            "config_short_label",
            "noncollapse_ratio",
            "unique_absolute_profile_ratio",
            "adjacent_nonzero_ratio",
            "min_adjacent_delta_distance",
            "mean_adjacent_delta_distance",
            "intermediate_distinct_count",
            "endpoint_delta_separation",
            "mean_top1_dominance",
            "mean_neighborhood_entropy",
            "mean_family_distance",
        ]
    ].to_csv(sprint_paths.tables_dir / "noncollapse_metrics.csv", index=False)

    _plot_endpoint_reference(
        endpoint_reference_rows,
        sprint_paths.figures_dir / "endpoint_fingerprint_reference.png",
    )
    _plot_tradeoff(
        comparator_df,
        sprint_paths.figures_dir / "pilot1b_cfg05_vs_cfg08_tradeoff.png",
    )
    report_figure_paths.extend(
        [
            sprint_paths.figures_dir / "endpoint_fingerprint_reference.png",
            sprint_paths.figures_dir / "pilot1b_cfg05_vs_cfg08_tradeoff.png",
        ]
    )

    report_md = sprint_paths.report_dir / "GAIRAv3_Pilot1b_mixture_probe1_v1_report.md"
    _build_report_markdown(
        report_md,
        config_metrics_df=comparator_df,
        endpoint_alignment_df=endpoint_alignment_all_df,
    )
    report_pdf = sprint_paths.report_dir / "GAIRAv3_Pilot1b_mixture_probe1_v1_report.pdf"
    build_pdf_report(report_md, report_figure_paths, report_pdf)


if __name__ == "__main__":
    main()
