#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


DEFAULT_RUN_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v7_anchor_gpu_run1")
DEFAULT_EVAL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_eval_v2/embedding_v7_anchor_gpu_run1_eval_v2")
DEFAULT_COMPARE_RUN_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v5_full_true_gpu_run1")
DEFAULT_COMPARE_EVAL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_eval_v2/embedding_v5_full_true_gpu_run1_eval_v2")
DEFAULT_ANCHOR_TABLE = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_anchor_audit/embedding_anchor_table_v1.csv")
DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_cluster_analysis_v7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build latent cluster analysis for GAIRAM v7 embeddings.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--anchor-table", type=Path, default=DEFAULT_ANCHOR_TABLE)
    parser.add_argument("--compare-run-dir", type=Path, default=DEFAULT_COMPARE_RUN_DIR)
    parser.add_argument("--compare-eval-dir", type=Path, default=DEFAULT_COMPARE_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--global-threshold", type=float, default=0.075)
    parser.add_argument("--within-threshold", type=float, default=0.065)
    return parser.parse_args()


def load_run(run_dir: Path, eval_dir: Path, anchor_table_path: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    embeddings = np.load(run_dir / "embeddings.npy")
    metadata = pd.read_csv(run_dir / "metadata.csv")
    anchor_table = pd.read_csv(anchor_table_path)
    projection = pd.read_csv(eval_dir / "embedding_projection_v2.csv") if (eval_dir / "embedding_projection_v2.csv").exists() else pd.DataFrame()

    anchor_cols = [
        "sample_key",
        "proposed_harmonized_anchor",
        "anchor_type",
        "anchor_confidence",
        "cross_dataset_usable",
        "notes",
    ]
    metadata = metadata.merge(anchor_table[anchor_cols], on="sample_key", how="left")
    metadata["proposed_harmonized_anchor"] = metadata["proposed_harmonized_anchor"].fillna("")
    metadata["anchor_type"] = metadata["anchor_type"].fillna("")
    metadata["anchor_confidence"] = metadata["anchor_confidence"].fillna("")
    metadata["cross_dataset_usable"] = metadata["cross_dataset_usable"].fillna(False)
    metadata["notes"] = metadata["notes"].fillna("")
    return embeddings, metadata, projection, anchor_table


def infer_cluster_count(
    n_samples: int,
    *,
    min_clusters: int,
    max_clusters: int,
    scale_divisor: float = 20.0,
) -> int:
    estimate = int(round(math.sqrt(max(n_samples, 1) / scale_divisor)))
    return max(min_clusters, min(max_clusters, estimate))


def minibatch_cluster_labels(
    embeddings: np.ndarray,
    *,
    min_clusters: int,
    max_clusters: int,
) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    if len(embeddings) <= min_clusters:
        return np.arange(len(embeddings), dtype=int)

    n_clusters = infer_cluster_count(
        len(embeddings),
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )
    n_components = min(32, embeddings.shape[1], max(8, embeddings.shape[1] // 2))
    reduced = PCA(n_components=n_components, random_state=7).fit_transform(embeddings)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=7,
        batch_size=4096,
        n_init=5,
    )
    return model.fit_predict(reduced)


def run_clustering(embeddings: np.ndarray, metadata: pd.DataFrame, *, global_threshold: float, within_threshold: float) -> pd.DataFrame:
    working = metadata.copy().reset_index(drop=True)
    _ = global_threshold
    _ = within_threshold
    global_labels = minibatch_cluster_labels(
        embeddings,
        min_clusters=24,
        max_clusters=160,
    )
    working["global_cluster_id"] = [f"g_{int(label):04d}" for label in global_labels]

    within_labels = np.empty(len(working), dtype=object)
    for sample_type, subset in working.groupby("sample_type", sort=False):
        subset_indices = subset.index.to_numpy()
        subset_labels = minibatch_cluster_labels(
            embeddings[subset_indices],
            min_clusters=4,
            max_clusters=96,
        )
        for idx, label in zip(subset_indices, subset_labels, strict=False):
            within_labels[idx] = f"{sample_type}_{int(label):04d}"
    working["within_type_cluster_id"] = within_labels
    return working


def dominant_share(series: pd.Series) -> tuple[str, float, int]:
    valid = series.fillna("").astype(str)
    valid = valid[valid != ""]
    if valid.empty:
        return "", 0.0, 0
    counts = valid.value_counts()
    label = str(counts.index[0])
    count = int(counts.iloc[0])
    return label, count / len(valid), int(valid.nunique())


def entropy_normalized(series: pd.Series) -> float:
    valid = series.fillna("").astype(str)
    valid = valid[valid != ""]
    if valid.empty:
        return 0.0
    counts = valid.value_counts()
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def grounding_anchor_centroids(embeddings: np.ndarray, metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    subset = metadata[
        (metadata["sample_type"].astype(str) == "grounding")
        & (metadata["anchor_type"].astype(str) == "grounding_linked_biochemical_theme")
        & (metadata["proposed_harmonized_anchor"].astype(str) != "")
    ].copy()
    centroids = {}
    for anchor, group in subset.groupby("proposed_harmonized_anchor", sort=True):
        centroids[str(anchor)] = embeddings[group.index.to_numpy()].mean(axis=0)
    return centroids


def top_nearest_grounding_anchors(centroid: np.ndarray, anchor_centroids: dict[str, np.ndarray], top_k: int = 3) -> str:
    if not anchor_centroids:
        return ""
    centroid_norm = centroid / max(np.linalg.norm(centroid), 1e-8)
    scored = []
    for anchor, anchor_vec in anchor_centroids.items():
        anchor_norm = anchor_vec / max(np.linalg.norm(anchor_vec), 1e-8)
        score = float(np.dot(centroid_norm, anchor_norm))
        scored.append((anchor, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return "; ".join(f"{anchor}:{score:.3f}" for anchor, score in scored[:top_k])


def interpret_cluster(row: pd.Series) -> str:
    dominant_type = row["dominant_sample_type"]
    dataset_pure = bool(row["dataset_pure"])
    cross_dataset_mixed = bool(row["cross_dataset_mixed"])
    anchor_coherent = bool(row["anchor_coherent"])
    dominant_anchor_type = str(row["dominant_anchor_type"])
    if dominant_type == "":
        return "mixed_unresolved_cluster"
    if dominant_anchor_type == "process_or_protocol_anchor":
        return "likely_protocol_or_batch_cluster"
    if cross_dataset_mixed and anchor_coherent:
        return f"anchor_coherent_cross_dataset_{dominant_type}"
    if dataset_pure:
        return f"dataset_pure_{dominant_type}_cluster"
    if anchor_coherent:
        return f"anchor_coherent_{dominant_type}_cluster"
    if row["dominant_sample_type_share"] < 0.75:
        return "mixed_sample_type_cluster"
    return "mixed_biochemical_theme_cluster"


def summarize_clusters(
    embeddings: np.ndarray,
    assignments: pd.DataFrame,
    *,
    cluster_col: str,
    cluster_scope: str,
    anchor_centroids: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    interpretation_rows = []
    for cluster_id, group in assignments.groupby(cluster_col, sort=True):
        centroid = embeddings[group.index.to_numpy()].mean(axis=0)
        dominant_sample_type, dominant_sample_type_share, sample_type_count = dominant_share(group["sample_type"])
        dominant_dataset, dominant_dataset_share, dataset_count = dominant_share(group["dataset_id"])
        dominant_anchor, dominant_anchor_share, anchor_count = dominant_share(group["proposed_harmonized_anchor"])
        dominant_family, dominant_family_share, family_count = dominant_share(group["family_label"])
        dominant_class, dominant_class_share, class_count = dominant_share(group["label_optional"])
        dominant_anchor_type, dominant_anchor_type_share, _ = dominant_share(group["anchor_type"])
        row = {
            "cluster_scope": cluster_scope,
            "cluster_id": cluster_id,
            "cluster_size": int(len(group)),
            "dominant_sample_type": dominant_sample_type,
            "dominant_sample_type_share": dominant_sample_type_share,
            "sample_type_count": sample_type_count,
            "dominant_dataset": dominant_dataset,
            "dominant_dataset_share": dominant_dataset_share,
            "dataset_count": dataset_count,
            "dominant_anchor": dominant_anchor,
            "dominant_anchor_share": dominant_anchor_share,
            "anchor_count": anchor_count,
            "dominant_anchor_type": dominant_anchor_type,
            "dominant_anchor_type_share": dominant_anchor_type_share,
            "dominant_family": dominant_family,
            "dominant_family_share": dominant_family_share,
            "family_count": family_count,
            "dominant_class": dominant_class,
            "dominant_class_share": dominant_class_share,
            "class_count": class_count,
            "dataset_entropy": entropy_normalized(group["dataset_id"]),
            "anchor_entropy": entropy_normalized(group["proposed_harmonized_anchor"]),
            "class_entropy": entropy_normalized(group["label_optional"]),
            "nearest_grounding_anchors": top_nearest_grounding_anchors(centroid, anchor_centroids),
        }
        row["dataset_pure"] = bool(dataset_count == 1 or dominant_dataset_share >= 0.90)
        row["cross_dataset_mixed"] = bool(
            dominant_sample_type_share >= 0.80 and dataset_count >= 2 and dominant_dataset_share < 0.80
        )
        row["anchor_coherent"] = bool(dominant_anchor != "" and dominant_anchor_share >= 0.70)
        row["class_coherent"] = bool(dominant_class != "" and dominant_class_share >= 0.70)
        row["interpretation_label"] = interpret_cluster(pd.Series(row))
        row["datasets"] = "; ".join(group["dataset_id"].astype(str).value_counts().index[:6].tolist())
        row["anchors"] = "; ".join(group["proposed_harmonized_anchor"].astype(str).replace("", "none").value_counts().index[:6].tolist())
        row["classes"] = "; ".join(group["label_optional"].astype(str).replace("", "none").value_counts().index[:6].tolist())
        summary_rows.append(row)
        interpretation_rows.append(
            {
                "cluster_scope": cluster_scope,
                "cluster_id": cluster_id,
                "interpretation_label": row["interpretation_label"],
                "dominant_sample_type": dominant_sample_type,
                "dominant_dataset": dominant_dataset,
                "dominant_anchor": dominant_anchor,
                "dominant_class": dominant_class,
                "dataset_pure": row["dataset_pure"],
                "cross_dataset_mixed": row["cross_dataset_mixed"],
                "anchor_coherent": row["anchor_coherent"],
                "class_coherent": row["class_coherent"],
                "nearest_grounding_anchors": row["nearest_grounding_anchors"],
                "notes": (
                    f"datasets={row['datasets']} | anchors={row['anchors']} | classes={row['classes']}"
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(interpretation_rows)


def cross_dataset_cluster_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cluster_scope, dominant_sample_type), subset in summary_df.groupby(["cluster_scope", "dominant_sample_type"], dropna=False):
        if dominant_sample_type == "":
            continue
        rows.append(
            {
                "cluster_scope": cluster_scope,
                "sample_type": dominant_sample_type,
                "cluster_count": int(len(subset)),
                "cross_dataset_cluster_count": int((subset["dataset_count"] >= 2).sum()),
                "cross_dataset_mixed_cluster_count": int(subset["cross_dataset_mixed"].sum()),
                "dataset_pure_cluster_count": int(subset["dataset_pure"].sum()),
                "anchor_coherent_cluster_count": int(subset["anchor_coherent"].sum()),
                "class_coherent_cluster_count": int(subset["class_coherent"].sum()),
                "mean_dataset_purity": float(subset["dominant_dataset_share"].mean()),
                "mean_anchor_purity": float(subset["dominant_anchor_share"].mean()),
                "mean_class_purity": float(subset["dominant_class_share"].mean()),
                "mean_dataset_entropy": float(subset["dataset_entropy"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster_scope", "sample_type"]).reset_index(drop=True)


def compare_cluster_metrics(v5_metrics: pd.DataFrame, v7_metrics: pd.DataFrame) -> pd.DataFrame:
    merged = v5_metrics.merge(
        v7_metrics,
        on=["cluster_scope", "sample_type"],
        how="outer",
        suffixes=("_v5", "_v7"),
    )
    for col in [
        "cluster_count",
        "cross_dataset_cluster_count",
        "cross_dataset_mixed_cluster_count",
        "dataset_pure_cluster_count",
        "anchor_coherent_cluster_count",
        "class_coherent_cluster_count",
        "mean_dataset_purity",
        "mean_anchor_purity",
        "mean_class_purity",
        "mean_dataset_entropy",
    ]:
        merged[f"{col}_delta"] = merged[f"{col}_v7"] - merged[f"{col}_v5"]
    return merged


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def merge_projection_with_clusters(projection_df: pd.DataFrame, assignments_df: pd.DataFrame) -> pd.DataFrame:
    if projection_df.empty:
        return pd.DataFrame()
    cols = [
        "sample_key",
        "global_cluster_id",
        "within_type_cluster_id",
        "proposed_harmonized_anchor",
        "anchor_type",
        "anchor_confidence",
        "cross_dataset_usable",
    ]
    return projection_df.merge(assignments_df[cols], on="sample_key", how="left")


def plot_categorical_scatter(df: pd.DataFrame, color_col: str, output_path: Path, title: str, top_n: int | None = None) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    plot_df[color_col] = plot_df[color_col].fillna("").astype(str)
    if top_n is not None:
        top_labels = plot_df[color_col].value_counts().head(top_n).index.tolist()
        plot_df[color_col] = plot_df[color_col].where(plot_df[color_col].isin(top_labels), other="other")
    labels = plot_df[color_col].astype("category")
    codes = labels.cat.codes.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(plot_df["dim1"], plot_df["dim2"], c=codes, cmap="tab20", s=5, alpha=0.75, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    legend_labels = labels.cat.categories.tolist()[:15]
    handles = []
    for idx, label in enumerate(legend_labels):
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color=scatter.cmap(scatter.norm(idx)), label=label, markersize=5))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_reports(
    output_dir: Path,
    cluster_summary_df: pd.DataFrame,
    interpretation_df: pd.DataFrame,
    cross_metrics_df: pd.DataFrame,
    compare_df: pd.DataFrame,
) -> None:
    cluster_report = textwrap.dedent(
        f"""
        Cluster interpretation report

        Global and within-sample-type clusters were summarized from full-embedding clustering.
        Interpretation labels are intentionally cautious: they describe structure, anchor coherence, and dataset mixing,
        not hard biochemical identity claims.

        Top interpreted clusters:
        {interpretation_df.head(40).to_string(index=False)}
        """
    ).strip() + "\n"
    (output_dir / "cluster_interpretation_report.md").write_text(cluster_report, encoding="utf-8")

    cross_report = textwrap.dedent(
        f"""
        Cross-dataset cluster report

        This report asks whether v7 produced cross-dataset neighborhoods within sample type, rather than merely perturbing
        existing dataset islands.

        {cross_metrics_df.to_string(index=False)}
        """
    ).strip() + "\n"
    (output_dir / "cross_dataset_cluster_report.md").write_text(cross_report, encoding="utf-8")

    compare_report = textwrap.dedent(
        f"""
        v5 vs v7 cluster comparison report

        Positive deltas for `cross_dataset_mixed_cluster_count` and anchor coherence indicate more biologically shared latent neighborhoods.
        Negative deltas for mean dataset purity are desirable when they occur without destroying class or anchor coherence.

        {compare_df.to_string(index=False)}
        """
    ).strip() + "\n"
    (output_dir / "v5_vs_v7_cluster_report.md").write_text(compare_report, encoding="utf-8")

    ev_v5 = compare_df[(compare_df["sample_type"] == "ev") & (compare_df["cluster_scope"] == "within_type_cluster_id")]
    serum_v5 = compare_df[(compare_df["sample_type"] == "serum") & (compare_df["cluster_scope"] == "within_type_cluster_id")]
    memo_lines = [
        "# v7 Decision Memo",
        "",
        "1. Is v7 improving the latent biology story?",
        "Yes, if cross-dataset mixed and anchor-coherent clusters increase without total collapse of sample-type structure.",
        "",
        "2. Is EV now mixing across datasets in a meaningful way?",
        f"EV comparison rows:\n\n{ev_v5.to_string(index=False) if not ev_v5.empty else 'No EV comparison row available.'}",
        "",
        "3. Is serum improving at all?",
        f"Serum comparison rows:\n\n{serum_v5.to_string(index=False) if not serum_v5.empty else 'No serum comparison row available.'}",
        "",
        "4. Are we progressing in the right direction for a Raman foundation model?",
        "Progress is real only if anchor-coherent, cross-dataset clusters are appearing inside sample type rather than replacing one dataset island with another.",
        "",
        "5. Is another training iteration justified, or should work move toward graph / inference layers?",
        "If EV shows materially more cross-dataset mixed anchor-coherent clusters than v5, another training iteration is justified. If not, the next bottleneck may already be better handled by anchor-grounded graph and inference layers.",
        "",
    ]
    (output_dir / "v7_decision_memo.md").write_text("\n".join(memo_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    v7_embeddings, v7_metadata, v7_projection, _ = load_run(args.run_dir, args.eval_dir, args.anchor_table)
    v5_embeddings, v5_metadata, v5_projection, _ = load_run(args.compare_run_dir, args.compare_eval_dir, args.anchor_table)

    v7_assignments = run_clustering(
        v7_embeddings,
        v7_metadata,
        global_threshold=args.global_threshold,
        within_threshold=args.within_threshold,
    )
    v5_assignments = run_clustering(
        v5_embeddings,
        v5_metadata,
        global_threshold=args.global_threshold,
        within_threshold=args.within_threshold,
    )

    v7_anchor_centroids = grounding_anchor_centroids(v7_embeddings, v7_assignments)
    v5_anchor_centroids = grounding_anchor_centroids(v5_embeddings, v5_assignments)

    v7_global_summary, v7_global_interp = summarize_clusters(
        v7_embeddings,
        v7_assignments,
        cluster_col="global_cluster_id",
        cluster_scope="global_cluster_id",
        anchor_centroids=v7_anchor_centroids,
    )
    v7_within_summary, v7_within_interp = summarize_clusters(
        v7_embeddings,
        v7_assignments,
        cluster_col="within_type_cluster_id",
        cluster_scope="within_type_cluster_id",
        anchor_centroids=v7_anchor_centroids,
    )
    v5_global_summary, _ = summarize_clusters(
        v5_embeddings,
        v5_assignments,
        cluster_col="global_cluster_id",
        cluster_scope="global_cluster_id",
        anchor_centroids=v5_anchor_centroids,
    )
    v5_within_summary, _ = summarize_clusters(
        v5_embeddings,
        v5_assignments,
        cluster_col="within_type_cluster_id",
        cluster_scope="within_type_cluster_id",
        anchor_centroids=v5_anchor_centroids,
    )

    v7_cluster_summary = pd.concat([v7_global_summary, v7_within_summary], ignore_index=True)
    v7_interpretation = pd.concat([v7_global_interp, v7_within_interp], ignore_index=True)
    v7_cluster_summary.to_csv(args.output_dir / "cluster_summary.csv", index=False)
    v7_interpretation.to_csv(args.output_dir / "cluster_interpretation_table.csv", index=False)

    v7_assignments.to_csv(args.output_dir / "cluster_assignments.csv", index=False)

    v7_cross_metrics = cross_dataset_cluster_metrics(v7_cluster_summary)
    v7_cross_metrics.to_csv(args.output_dir / "cross_dataset_cluster_metrics.csv", index=False)

    v5_metrics = cross_dataset_cluster_metrics(pd.concat([v5_global_summary, v5_within_summary], ignore_index=True))
    compare_df = compare_cluster_metrics(v5_metrics, v7_cross_metrics)
    compare_df.to_csv(args.output_dir / "v5_vs_v7_cluster_comparison.csv", index=False)

    build_reports(args.output_dir, v7_cluster_summary, v7_interpretation, v7_cross_metrics, compare_df)

    v7_projection_clustered = merge_projection_with_clusters(v7_projection, v7_assignments)
    plot_categorical_scatter(v7_projection_clustered, "global_cluster_id", args.output_dir / "umap_cluster_id.png", "v7 sampled UMAP by global cluster", top_n=20)
    plot_categorical_scatter(v7_projection_clustered, "proposed_harmonized_anchor", args.output_dir / "umap_harmonized_anchor.png", "v7 sampled UMAP by harmonized anchor", top_n=12)
    plot_categorical_scatter(v7_projection_clustered, "dataset_id", args.output_dir / "umap_dataset_id.png", "v7 sampled UMAP by dataset", top_n=12)
    plot_categorical_scatter(v7_projection_clustered, "sample_type", args.output_dir / "umap_sample_type.png", "v7 sampled UMAP by sample type")
    for sample_type in ["ev", "serum", "grounding"]:
        sub = v7_projection_clustered[v7_projection_clustered["sample_type"] == sample_type].copy()
        plot_categorical_scatter(
            sub,
            "within_type_cluster_id",
            args.output_dir / f"umap_{sample_type}_within_type_clusters.png",
            f"v7 sampled UMAP: {sample_type} only by within-type cluster",
            top_n=20,
        )

    print(f"Saved cluster analysis to {args.output_dir}")
    print(f"Global clusters: {v7_global_summary['cluster_id'].nunique()}")
    print(
        "Within-type clusters: "
        + ", ".join(
            f"{sample_type}={v7_within_summary[v7_within_summary['dominant_sample_type'] == sample_type]['cluster_id'].nunique()}"
            for sample_type in sorted(v7_within_summary["dominant_sample_type"].dropna().unique().tolist())
        )
    )


if __name__ == "__main__":
    main()
