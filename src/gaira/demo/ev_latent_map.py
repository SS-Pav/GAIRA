from __future__ import annotations

from pathlib import Path

import pandas as pd

from gaira.demo.ev_graph_builder import DEFAULT_OUTPUT_DIR


def prepare_latent_map_tables(
    graph_state: dict[str, pd.DataFrame],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    projection = graph_state["projection"].copy()
    assignments = graph_state["cluster_assignments"].copy()
    clusters = graph_state["ev_cluster_interpretation"].copy()
    dataset_comp = graph_state["dataset_composition"].copy()
    edges = graph_state["edges"].copy()

    projection = projection.merge(
        assignments[["sample_key", "within_type_cluster_id"]],
        on="sample_key",
        how="left",
    )
    sampled_ev = projection[projection["sample_type"] == "ev"].copy()
    sampled_ev = sampled_ev.merge(
        clusters,
        left_on="within_type_cluster_id",
        right_on="cluster_id",
        how="left",
        suffixes=("", "_cluster"),
    )

    cluster_centroids = (
        sampled_ev.groupby("cluster_id", as_index=False)
        .agg(
            dim1=("dim1", "mean"),
            dim2=("dim2", "mean"),
            sampled_point_count=("sample_key", "count"),
        )
        .merge(clusters, on="cluster_id", how="left")
        .sort_values(["theme_support_strength_rank", "cluster_size"], ascending=[True, False])
        .reset_index(drop=True)
    )

    neighbor_edges = edges[edges["edge_type"] == "cluster_adjacent_cluster"].copy()
    neighbor_edges["source_cluster_id"] = neighbor_edges["source"].str.replace("cluster::", "", regex=False)
    neighbor_edges["target_cluster_id"] = neighbor_edges["target"].str.replace("cluster::", "", regex=False)

    summary_lines = [
        "# EV Latent Map Summary",
        "",
        f"- sampled EV points: {int(len(sampled_ev))}",
        f"- EV clusters in centroid view: {int(cluster_centroids['cluster_id'].nunique())}",
        f"- cross-dataset mixed clusters: {int(cluster_centroids['cross_dataset_mixed'].fillna(False).sum())}",
        "",
        "This centroid map is built from sampled UMAP coordinates. Clusters are learned from embeddings first; biochemical themes are painted onto those clusters afterward using grounding.",
    ]

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        cluster_centroids.to_csv(output_dir / "ev_cluster_centroids.csv", index=False)
        (output_dir / "ev_latent_map_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "sampled_ev": sampled_ev,
        "cluster_centroids": cluster_centroids,
        "neighbor_edges": neighbor_edges,
        "dataset_composition": dataset_comp,
    }
