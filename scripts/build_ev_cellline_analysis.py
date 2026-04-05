#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from gaira.demo.ev_analysis_utils import (
    DEFAULT_ANCHOR_AUDIT_DIR,
    DEFAULT_CLUSTER_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_GROUNDING_DIR,
    DEFAULT_RUN_DIR,
    THEME_COLORS,
    THEME_ORDER,
    balanced_sample,
    cluster_composition_summary,
    compute_theme_profiles,
    decode_direct_matrix,
    entropy_normalized,
    knn_label_metrics,
    load_common_artifacts,
    load_direct_processed_metadata,
    load_direct_processed_spectra_by_ids,
    normalize_rows,
    reduce_for_plot,
    save_heatmap,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/ev_cellline_analysis_v1")
MIXTURE_CLASSES = ["c00", "c01", "c10", "c25", "c50", "c100"]
MIXTURE_PROBES = ["normedprobe1", "normedprobe2"]
CELLLINE_CLASSES = ["Hec", "Hela", "Ht", "Mef", "Thp"]
SMALL2023_PROCESSING_VERSION = "v2_crop670_1800_interp1_poly3_vector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EV cell-line analysis for GAIRAM.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--grounding-dir", type=Path, default=DEFAULT_GROUNDING_DIR)
    parser.add_argument("--anchor-audit-dir", type=Path, default=DEFAULT_ANCHOR_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k-grounding", type=int, default=12)
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--balanced-per-group", type=int, default=250)
    parser.add_argument("--plot-per-group", type=int, default=120)
    return parser.parse_args()


def sampled_silhouette(values: np.ndarray, labels: np.ndarray, *, seed: int, max_points: int = 2500) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    if len(values) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(values), size=max_points, replace=False)
        values = values[idx]
        labels = labels[idx]
    return float(silhouette_score(values, labels))


def direct_embedding_metric_rows(
    values: np.ndarray,
    class_labels: np.ndarray,
    probe_labels: np.ndarray,
    *,
    knn_k: int,
    seed: int,
) -> pd.DataFrame:
    class_knn = knn_label_metrics(values, class_labels, k=knn_k)
    probe_knn = knn_label_metrics(values, probe_labels, k=knn_k)
    metrics = [
        {"metric": "silhouette_class", "value": sampled_silhouette(values, class_labels, seed=seed)},
        {"metric": "silhouette_probe", "value": sampled_silhouette(values, probe_labels, seed=seed)},
        {"metric": "nn_purity_class", "value": class_knn["nn_purity"]},
        {"metric": "nn_purity_probe", "value": probe_knn["nn_purity"]},
        {"metric": "neighbor_entropy_class", "value": class_knn["neighbor_entropy"]},
        {"metric": "neighbor_entropy_probe", "value": probe_knn["neighbor_entropy"]},
        {"metric": "top1_match_class", "value": class_knn["top1_match"]},
        {"metric": "top1_match_probe", "value": probe_knn["top1_match"]},
        {
            "metric": "class_probe_ratio",
            "value": float(class_knn["nn_purity"] / probe_knn["nn_purity"]) if probe_knn["nn_purity"] not in (0.0, np.nan) else np.nan,
        },
    ]
    return pd.DataFrame(metrics)


def neighborhood_entropy(values: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(values)), metric="cosine", algorithm="brute")
    nn.fit(values)
    _, indices = nn.kneighbors(values)
    entropies = []
    for row in labels[indices[:, 1:]]:
        entropies.append(entropy_normalized(row))
    return float(np.mean(entropies)) if entropies else float("nan")


def plot_projection(coords: np.ndarray, labels: np.ndarray, *, title: str, output_path: Path) -> None:
    unique = list(dict.fromkeys(pd.Series(labels).astype(str).tolist()))
    cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    for idx, label in enumerate(unique):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=16, alpha=0.72, color=cmap(idx % cmap.N), label=str(label))
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(fontsize=8, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metric_bars(comparison_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    metrics = ["silhouette_class", "silhouette_probe", "nn_purity_class", "nn_purity_probe", "class_probe_ratio"]
    direct_vals = [comparison_df.loc[comparison_df["metric"] == metric, "direct_value"].iloc[0] for metric in metrics]
    embedding_vals = [comparison_df.loc[comparison_df["metric"] == metric, "embedding_value"].iloc[0] for metric in metrics]
    x = np.arange(len(metrics))
    width = 0.36
    ax.bar(x - width / 2, direct_vals, width=width, color="#7c8793", label="direct processed spectra")
    ax.bar(x + width / 2, embedding_vals, width=width, color="#3e6ea1", label="GAIRAM embeddings")
    ax.set_xticks(x)
    ax.set_xticklabels([metric.replace("_", "\n") for metric in metrics])
    ax.set_title("Direct vs embedding class/probe organization")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def class_composition_summary(cellline_df: pd.DataFrame, theme_cols: list[str]) -> pd.DataFrame:
    rows = []
    for class_label, group in cellline_df.groupby("label_optional", sort=True):
        mean_profile = group[theme_cols].mean()
        top = mean_profile.sort_values(ascending=False)
        rows.append(
            {
                "class_label": class_label,
                **{theme: float(mean_profile[theme]) for theme in theme_cols},
                "dominant_theme": top.index[0],
                "secondary_theme": top.index[1] if len(top) > 1 and top.iloc[1] > 0 else "none",
                "composition_variance": float(group[theme_cols].var().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("class_label").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_common_artifacts(
        run_dir=args.run_dir,
        eval_dir=args.eval_dir,
        cluster_dir=args.cluster_dir,
        grounding_dir=args.grounding_dir,
        anchor_audit_dir=args.anchor_audit_dir,
    )
    embeddings = artifacts["embeddings"]  # type: ignore[assignment]
    metadata = artifacts["metadata"].copy()  # type: ignore[assignment]
    cluster_assignments = artifacts["cluster_assignments"].copy()  # type: ignore[assignment]
    grounding_theme_table = artifacts["grounding_theme_table"].copy()  # type: ignore[assignment]

    small_meta = metadata[(metadata["dataset_id"] == "small2023_ev") & (metadata["record_kind"] == "processed_spectrum")].copy()
    small_meta = small_meta.merge(cluster_assignments[["sample_key", "within_type_cluster_id"]], on="sample_key", how="left")

    mixture_meta = load_direct_processed_metadata(
        dataset_id="small2023_ev",
        processing_version=SMALL2023_PROCESSING_VERSION,
        class_filter=MIXTURE_CLASSES,
        subclass_filter=MIXTURE_PROBES,
    )
    mixture_meta["sample_key"] = mixture_meta["sample_key"].astype(str)
    sampled_meta = balanced_sample(
        mixture_meta,
        ["subclass_label", "class_label"],
        per_group=args.balanced_per_group,
        seed=args.seed,
    )
    mixture_direct = load_direct_processed_spectra_by_ids(
        dataset_id="small2023_ev",
        processing_version=SMALL2023_PROCESSING_VERSION,
        sample_keys=sampled_meta["sample_key"].astype(str).tolist(),
    )
    mixture_direct["sample_key"] = mixture_direct["sample_key"].astype(str)
    _, direct_matrix = decode_direct_matrix(mixture_direct)
    direct_scaled = StandardScaler().fit_transform(direct_matrix)

    mixture_embed_meta = small_meta[small_meta["sample_key"].isin(mixture_direct["sample_key"])].copy()
    mixture_embed = normalize_rows(embeddings[mixture_embed_meta.index.to_numpy()])
    class_labels = mixture_direct["class_label"].to_numpy()
    probe_labels = mixture_direct["subclass_label"].to_numpy()

    direct_metrics = direct_embedding_metric_rows(direct_scaled, class_labels, probe_labels, knn_k=args.knn_k, seed=args.seed)
    direct_metrics.to_csv(args.output_dir / "direct_spectra_metrics.csv", index=False)

    embedding_metrics = direct_embedding_metric_rows(mixture_embed, class_labels, probe_labels, knn_k=args.knn_k, seed=args.seed)
    embedding_metrics.to_csv(args.output_dir / "embedding_space_metrics.csv", index=False)

    comparison = direct_metrics.merge(embedding_metrics, on="metric", suffixes=("_direct", "_embedding"))
    comparison = comparison.rename(columns={"value_direct": "direct_value", "value_embedding": "embedding_value"})
    comparison["delta_embedding_minus_direct"] = comparison["embedding_value"] - comparison["direct_value"]
    comparison.to_csv(args.output_dir / "direct_vs_embedding_comparison.csv", index=False)

    direct_report = textwrap.dedent(
        """\
        # small2023 EV Direct Processed-Spectra Report

        Direct-space metrics were computed on the balanced mixture subset only:
        `c00/c01/c10/c25/c50/c100` across `normedprobe1` and `normedprobe2`.

        This is the cleanest way to compare biological-class organization against probe nuisance,
        because the explicit cell-line labels live only in the separate `fig3_norm_archive`.
        """
    )
    (args.output_dir / "direct_spectra_report.md").write_text(direct_report, encoding="utf-8")

    embedding_report = textwrap.dedent(
        """\
        # small2023 EV Embedding-Space Report

        The same balanced mixture subset used in the direct baseline was projected through the
        GAIRAM v7 embedding. Metrics were then recomputed without retraining or reclustering.
        """
    )
    (args.output_dir / "embedding_space_report.md").write_text(embedding_report, encoding="utf-8")

    compare_report = textwrap.dedent(
        """\
        # small2023 EV Direct vs Embedding Comparison

        The comparison table focuses on whether embedding space improves class organization
        while reducing probe-family nuisance. Read this as a structural benchmark, not as a
        claim that every small2023 label is a stable biological class.
        """
    )
    (args.output_dir / "direct_vs_embedding_report.md").write_text(compare_report, encoding="utf-8")

    plot_direct = balanced_sample(mixture_direct, ["subclass_label", "class_label"], per_group=args.plot_per_group, seed=args.seed)
    _, plot_direct_matrix = decode_direct_matrix(plot_direct)
    plot_direct_coords = reduce_for_plot(plot_direct_matrix, seed=args.seed)
    plot_projection(plot_direct_coords, plot_direct["class_label"].to_numpy(), title="Direct processed spectra by mixture class", output_path=args.output_dir / "direct_map_by_class.png")
    plot_projection(plot_direct_coords, plot_direct["subclass_label"].to_numpy(), title="Direct processed spectra by probe family", output_path=args.output_dir / "direct_map_by_probe.png")

    plot_embed_meta = mixture_embed_meta[mixture_embed_meta["sample_key"].isin(plot_direct["sample_key"])].copy()
    plot_embed = normalize_rows(embeddings[plot_embed_meta.index.to_numpy()])
    plot_embed_coords = reduce_for_plot(plot_embed, seed=args.seed)
    plot_projection(plot_embed_coords, plot_direct["class_label"].to_numpy(), title="GAIRAM embeddings by mixture class", output_path=args.output_dir / "embedding_map_by_class.png")
    plot_projection(plot_embed_coords, plot_direct["subclass_label"].to_numpy(), title="GAIRAM embeddings by probe family", output_path=args.output_dir / "embedding_map_by_probe.png")
    plot_metric_bars(comparison, args.output_dir / "direct_vs_embedding_metric_bars.png")

    grounding_metadata = grounding_theme_table.copy().reset_index(drop=True)
    grounding_keys = set(grounding_metadata["sample_key"].astype(str))
    grounding_idx = metadata.index[metadata["sample_key"].astype(str).isin(grounding_keys)].to_numpy()
    grounding_embeddings = embeddings[grounding_idx]
    grounding_themes = grounding_metadata["grounding_theme"].astype(str).to_numpy()

    cellline_meta = small_meta[small_meta["label_optional"].isin(CELLLINE_CLASSES)].copy().reset_index(drop=False)
    cellline_embed = normalize_rows(embeddings[cellline_meta["index"].to_numpy()])
    cellline_profiles = compute_theme_profiles(cellline_embed, normalize_rows(grounding_embeddings), grounding_themes, top_k=args.top_k_grounding)
    cellline_profiles = cellline_meta[["sample_key", "label_optional", "subclass_label", "within_type_cluster_id"]].reset_index(drop=True).join(cellline_profiles)

    cellline_cluster_profiles, cluster_metrics = cluster_composition_summary(
        cellline_profiles,
        cluster_col="within_type_cluster_id",
        theme_cols=THEME_ORDER,
    )
    cellline_cluster_profiles = cellline_cluster_profiles.rename(columns={"within_type_cluster_id": "cluster_id"})
    cellline_cluster_profiles.to_csv(args.output_dir / "cellline_per_cluster_composition_profiles.csv", index=False)

    class_summary = class_composition_summary(cellline_profiles, THEME_ORDER)
    class_summary.to_csv(args.output_dir / "cellline_class_composition_summary.csv", index=False)

    composition_report = textwrap.dedent(
        """\
        # small2023 EV Cell-Line Composition Report

        Cell-line composition profiling uses only the explicit `Hec/Hela/Ht/Mef/Thp` spectra from the
        `fig3_norm_archive`. These labels do not support a probe-nuisance comparison, so they are treated
        as a separate interpretation layer rather than mixed into the probe benchmark.
        """
    )
    (args.output_dir / "cellline_composition_report.md").write_text(composition_report, encoding="utf-8")

    heatmap = class_summary.set_index("class_label")[THEME_ORDER]
    save_heatmap(
        heatmap,
        output_path=args.output_dir / "cellline_composition_heatmap.png",
        title="Cell-line class biochemical composition summary",
        figsize=(8.2, 5.2),
    )

    cellline_coords = reduce_for_plot(cellline_embed, seed=args.seed)
    plot_projection(cellline_coords, cellline_meta["label_optional"].to_numpy(), title="Cell-line embedding map by class", output_path=args.output_dir / "cellline_class_composition_scatter.png")

    summary_lines = [
        "# EV Cell-Line Summary",
        "",
        "Assessment split used in this analysis:",
        "- class-vs-probe structure: balanced mixture subset across `normedprobe1` and `normedprobe2`.",
        "- cell-line interpretation: explicit `Hec/Hela/Ht/Mef/Thp` spectra from `fig3_norm_archive`.",
        "",
        f"- direct class silhouette: {float(direct_metrics.loc[direct_metrics.metric == 'silhouette_class', 'value'].iloc[0]):.4f}",
        f"- embedding class silhouette: {float(embedding_metrics.loc[embedding_metrics.metric == 'silhouette_class', 'value'].iloc[0]):.4f}",
        f"- direct probe silhouette: {float(direct_metrics.loc[direct_metrics.metric == 'silhouette_probe', 'value'].iloc[0]):.4f}",
        f"- embedding probe silhouette: {float(embedding_metrics.loc[embedding_metrics.metric == 'silhouette_probe', 'value'].iloc[0]):.4f}",
        "",
        "Assessment:",
        "- Embedding space can be judged honestly against direct spectra for mixture-class vs probe nuisance.",
        "- Explicit cell-line interpretation is possible, but it is supported by the separate fig3 archive rather than the probe benchmark subset.",
        "- This page is supportable for a demo if it is framed as 'small2023 structural benchmark + cell-line composition interpretation', not as a single unified biological assay.",
    ]
    (args.output_dir / "ev_cellline_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
