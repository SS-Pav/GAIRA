from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v7_anchor_gpu_run1")
DEFAULT_EVAL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_eval_v2/embedding_v7_anchor_gpu_run1_eval_v2")
DEFAULT_CLUSTER_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_cluster_analysis_v7")
DEFAULT_GROUNDING_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_grounding_analysis_v7")
DEFAULT_ANCHOR_TABLE = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_anchor_audit/embedding_anchor_table_v1.csv")
DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_ev_graph_v1")

GRAPH_THEME_COLORS = {
    "cluster": "#5b6c7d",
    "anchor": "#c06c50",
    "theme": "#2f7f6f",
    "grounding_ref": "#a8b3c2",
    "dataset": "#7b6ea8",
}


def normalize_vectors(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return matrix / norms


def safe_label(value: str) -> str:
    return str(value).replace("_", " ").replace("/", " / ").strip()


def support_badge(strength: str) -> str:
    return {"strong": "Strong", "moderate": "Moderate", "weak": "Weak"}.get(strength, "Mixed")


def load_ev_source_tables(
    run_dir: Path = DEFAULT_RUN_DIR,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    cluster_dir: Path = DEFAULT_CLUSTER_DIR,
    grounding_dir: Path = DEFAULT_GROUNDING_DIR,
    anchor_table_path: Path = DEFAULT_ANCHOR_TABLE,
) -> dict[str, pd.DataFrame | np.ndarray]:
    embeddings = np.load(run_dir / "embeddings.npy")
    metadata = pd.read_csv(run_dir / "metadata.csv")
    projection = pd.read_csv(eval_dir / "embedding_projection_v2.csv")
    cluster_assignments = pd.read_csv(cluster_dir / "cluster_assignments.csv")
    cluster_summary = pd.read_csv(cluster_dir / "cluster_summary.csv")
    cluster_interpretation = pd.read_csv(cluster_dir / "cluster_interpretation_table.csv")
    ev_cluster_interpretation = pd.read_csv(grounding_dir / "ev_cluster_interpretation_table.csv")
    ev_cluster_theme_scores = pd.read_csv(grounding_dir / "ev_cluster_theme_scores.csv")
    ev_cluster_grounding_hits = pd.read_csv(grounding_dir / "ev_cluster_grounding_hits.csv")
    ev_cluster_grounding_summary = pd.read_csv(grounding_dir / "ev_cluster_grounding_summary.csv")
    grounding_theme_table = pd.read_csv(grounding_dir / "grounding_theme_table.csv")
    anchor_table = pd.read_csv(anchor_table_path)

    metadata = metadata.merge(
        anchor_table[["sample_key", "proposed_harmonized_anchor", "anchor_type", "anchor_confidence", "cross_dataset_usable"]],
        on="sample_key",
        how="left",
    )
    metadata["proposed_harmonized_anchor"] = metadata["proposed_harmonized_anchor"].fillna("")
    metadata["anchor_type"] = metadata["anchor_type"].fillna("")
    metadata["anchor_confidence"] = metadata["anchor_confidence"].fillna("")
    metadata["cross_dataset_usable"] = metadata["cross_dataset_usable"].fillna(False)
    metadata = metadata.merge(
        cluster_assignments[["sample_key", "global_cluster_id", "within_type_cluster_id"]],
        on="sample_key",
        how="left",
    )
    return {
        "embeddings": embeddings,
        "metadata": metadata,
        "projection": projection,
        "cluster_assignments": cluster_assignments,
        "cluster_summary": cluster_summary,
        "cluster_interpretation": cluster_interpretation,
        "ev_cluster_interpretation": ev_cluster_interpretation,
        "ev_cluster_theme_scores": ev_cluster_theme_scores,
        "ev_cluster_grounding_hits": ev_cluster_grounding_hits,
        "ev_cluster_grounding_summary": ev_cluster_grounding_summary,
        "grounding_theme_table": grounding_theme_table,
    }


def compute_ev_cluster_dataset_composition(metadata: pd.DataFrame) -> pd.DataFrame:
    ev = metadata[metadata["sample_type"] == "ev"].copy()
    comp = (
        ev.groupby(["within_type_cluster_id", "dataset_id"])
        .size()
        .reset_index(name="count")
        .rename(columns={"within_type_cluster_id": "cluster_id"})
    )
    totals = comp.groupby("cluster_id")["count"].sum().rename("cluster_total").reset_index()
    comp = comp.merge(totals, on="cluster_id", how="left")
    comp["share"] = comp["count"] / comp["cluster_total"]
    return comp.sort_values(["cluster_id", "share"], ascending=[True, False]).reset_index(drop=True)


def compute_cluster_adjacency(ev_cluster_interpretation: pd.DataFrame, metadata: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    ev = metadata[metadata["sample_type"] == "ev"].copy()
    norm_embeddings = normalize_vectors(embeddings)
    centroids = []
    for cluster_id, group in ev.groupby("within_type_cluster_id", sort=True):
        centroid = norm_embeddings[group.index.to_numpy()].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-8)
        centroids.append((cluster_id, centroid))
    if not centroids:
        return pd.DataFrame(columns=["source_cluster_id", "target_cluster_id", "similarity", "edge_weight"])

    ids = [item[0] for item in centroids]
    matrix = np.vstack([item[1] for item in centroids])
    sim = matrix @ matrix.T
    rows = []
    for i, cluster_id in enumerate(ids):
        neighbors = np.argsort(-sim[i])
        kept = 0
        for j in neighbors:
            if i == j:
                continue
            score = float(sim[i, j])
            if score < 0.55:
                continue
            rows.append(
                {
                    "source_cluster_id": cluster_id,
                    "target_cluster_id": ids[j],
                    "similarity": score,
                    "edge_weight": score,
                }
            )
            kept += 1
            if kept >= 3:
                break
    adj = pd.DataFrame(rows)
    if adj.empty:
        return adj
    adj["edge_key"] = adj.apply(lambda row: "::".join(sorted([row["source_cluster_id"], row["target_cluster_id"]])), axis=1)
    adj = adj.sort_values(["edge_key", "similarity"], ascending=[True, False]).drop_duplicates("edge_key")
    adj = adj.drop(columns=["edge_key"]).reset_index(drop=True)
    return adj


def build_graph_tables(
    run_dir: Path = DEFAULT_RUN_DIR,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    cluster_dir: Path = DEFAULT_CLUSTER_DIR,
    grounding_dir: Path = DEFAULT_GROUNDING_DIR,
    anchor_table_path: Path = DEFAULT_ANCHOR_TABLE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    top_theme_edges: int = 3,
    top_grounding_per_cluster: int = 3,
    write_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    loaded = load_ev_source_tables(run_dir, eval_dir, cluster_dir, grounding_dir, anchor_table_path)
    metadata = loaded["metadata"]
    ev_cluster_interpretation = loaded["ev_cluster_interpretation"].copy()
    ev_cluster_theme_scores = loaded["ev_cluster_theme_scores"].copy()
    ev_cluster_grounding_hits = loaded["ev_cluster_grounding_hits"].copy()
    grounding_theme_table = loaded["grounding_theme_table"].copy()
    embeddings = loaded["embeddings"]

    dataset_comp = compute_ev_cluster_dataset_composition(metadata)
    cluster_adj = compute_cluster_adjacency(ev_cluster_interpretation, metadata, embeddings)

    nodes = []
    edges = []

    for row in ev_cluster_interpretation.to_dict(orient="records"):
        node_id = f"cluster::{row['cluster_id']}"
        label = f"{row['cluster_id']}\n{support_badge(str(row['theme_support_strength']))}"
        title = (
            f"<b>{row['cluster_id']}</b><br>"
            f"Size: {int(row['cluster_size'])}<br>"
            f"Anchor: {safe_label(str(row['dominant_harmonized_anchor']))}<br>"
            f"Theme: {safe_label(str(row['top_biochemical_theme']))}<br>"
            f"Cross-dataset mixed: {bool(row['cross_dataset_mixed'])}"
        )
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "node_type": "cluster",
                "title": title,
                "color": GRAPH_THEME_COLORS["cluster"],
                "size": 18 + math.sqrt(max(int(row["cluster_size"]), 1)) * 0.55,
                "cluster_id": row["cluster_id"],
                "support_strength": row["theme_support_strength"],
                "cross_dataset_mixed": bool(row["cross_dataset_mixed"]),
                "dominant_anchor": row["dominant_harmonized_anchor"],
                "dominant_theme": row["top_biochemical_theme"],
            }
        )

    anchor_nodes = set()
    for row in ev_cluster_interpretation.to_dict(orient="records"):
        anchor = str(row["dominant_harmonized_anchor"])
        if not anchor:
            continue
        anchor_id = f"anchor::{anchor}"
        if anchor_id not in anchor_nodes:
            anchor_nodes.add(anchor_id)
            nodes.append(
                {
                    "id": anchor_id,
                    "label": safe_label(anchor),
                    "node_type": "anchor",
                    "title": f"<b>{safe_label(anchor)}</b><br>Harmonized EV anchor",
                    "color": GRAPH_THEME_COLORS["anchor"],
                    "size": 28,
                    "anchor_id": anchor,
                }
            )
        edges.append(
            {
                "source": f"cluster::{row['cluster_id']}",
                "target": anchor_id,
                "edge_type": "cluster_has_anchor",
                "weight": 1.0,
                "label": "dominant anchor",
                "cluster_id": row["cluster_id"],
            }
        )

    theme_nodes = set()
    theme_scores_top = (
        ev_cluster_theme_scores.sort_values(["cluster_id", "theme_share"], ascending=[True, False])
        .groupby("cluster_id", as_index=False)
        .head(top_theme_edges)
        .copy()
    )
    for row in theme_scores_top.to_dict(orient="records"):
        theme = str(row["grounding_theme"])
        theme_id = f"theme::{theme}"
        if theme_id not in theme_nodes:
            theme_nodes.add(theme_id)
            nodes.append(
                {
                    "id": theme_id,
                    "label": safe_label(theme),
                    "node_type": "theme",
                    "title": f"<b>{safe_label(theme)}</b><br>Grounding theme",
                    "color": GRAPH_THEME_COLORS["theme"],
                    "size": 24,
                    "theme_id": theme,
                }
            )
        edges.append(
            {
                "source": f"cluster::{row['cluster_id']}",
                "target": theme_id,
                "edge_type": "cluster_supported_by_theme",
                "weight": float(row["theme_share"]),
                "label": f"{row['theme_share']:.2f}",
                "cluster_id": row["cluster_id"],
            }
        )

    grounding_nodes = set()
    grounding_hits_top = (
        ev_cluster_grounding_hits.sort_values(["cluster_id", "weighted_score"], ascending=[True, False])
        .drop_duplicates(subset=["cluster_id", "sample_key"])
        .groupby("cluster_id", as_index=False)
        .head(top_grounding_per_cluster)
        .copy()
    )
    for row in grounding_hits_top.to_dict(orient="records"):
        ref_id = f"grounding::{row['sample_key']}"
        if ref_id not in grounding_nodes:
            grounding_nodes.add(ref_id)
            nodes.append(
                {
                    "id": ref_id,
                    "label": safe_label(str(row["label_optional"]))[:28],
                    "node_type": "grounding_ref",
                    "title": (
                        f"<b>{safe_label(str(row['label_optional']))}</b><br>"
                        f"Dataset: {row['dataset_id']}<br>"
                        f"Theme: {safe_label(str(row['grounding_theme']))}<br>"
                        f"Similarity: {float(row['similarity']):.3f}"
                    ),
                    "color": GRAPH_THEME_COLORS["grounding_ref"],
                    "size": 18,
                    "grounding_theme": row["grounding_theme"],
                    "dataset_id": row["dataset_id"],
                }
            )
        edges.append(
            {
                "source": f"cluster::{row['cluster_id']}",
                "target": ref_id,
                "edge_type": "cluster_retrieves_grounding",
                "weight": float(row["weighted_score"]),
                "label": f"{float(row['similarity']):.2f}",
                "cluster_id": row["cluster_id"],
            }
        )

    dataset_nodes = set()
    for row in dataset_comp.to_dict(orient="records"):
        dataset_id = str(row["dataset_id"])
        node_id = f"dataset::{dataset_id}"
        if node_id not in dataset_nodes:
            dataset_nodes.add(node_id)
            nodes.append(
                {
                    "id": node_id,
                    "label": safe_label(dataset_id),
                    "node_type": "dataset",
                    "title": f"<b>{safe_label(dataset_id)}</b><br>EV source dataset",
                    "color": GRAPH_THEME_COLORS["dataset"],
                    "size": 18,
                    "dataset_id": dataset_id,
                }
            )
        edges.append(
            {
                "source": f"cluster::{row['cluster_id']}",
                "target": node_id,
                "edge_type": "cluster_contains_dataset",
                "weight": float(row["share"]),
                "label": f"{float(row['share']):.2f}",
                "cluster_id": row["cluster_id"],
            }
        )

    for row in cluster_adj.to_dict(orient="records"):
        edges.append(
            {
                "source": f"cluster::{row['source_cluster_id']}",
                "target": f"cluster::{row['target_cluster_id']}",
                "edge_type": "cluster_adjacent_cluster",
                "weight": float(row["edge_weight"]),
                "label": f"{float(row['similarity']):.2f}",
                "cluster_id": row["source_cluster_id"],
            }
        )

    nodes_df = pd.DataFrame(nodes).drop_duplicates(subset=["id"]).reset_index(drop=True)
    edges_df = pd.DataFrame(edges).reset_index(drop=True)

    summary_lines = [
        "# EV Graph Summary",
        "",
        f"- clusters: {int((nodes_df['node_type'] == 'cluster').sum())}",
        f"- anchors: {int((nodes_df['node_type'] == 'anchor').sum())}",
        f"- themes: {int((nodes_df['node_type'] == 'theme').sum())}",
        f"- grounding refs: {int((nodes_df['node_type'] == 'grounding_ref').sum())}",
        f"- datasets: {int((nodes_df['node_type'] == 'dataset').sum())}",
        "",
        "Edge counts:",
        "",
        edges_df["edge_type"].value_counts().to_string(),
        "",
        "This graph is pruned for demo readability: dominant anchor only, top theme edges, top grounding refs per cluster, and a small adjacency layer.",
    ]

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        nodes_df.to_csv(output_dir / "ev_graph_nodes.csv", index=False)
        edges_df.to_csv(output_dir / "ev_graph_edges.csv", index=False)
        (output_dir / "ev_graph_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "nodes": nodes_df,
        "edges": edges_df,
        "ev_cluster_interpretation": ev_cluster_interpretation,
        "ev_cluster_theme_scores": ev_cluster_theme_scores,
        "ev_cluster_grounding_hits": ev_cluster_grounding_hits,
        "dataset_composition": dataset_comp,
        "projection": loaded["projection"],
        "cluster_assignments": loaded["cluster_assignments"],
        "grounding_theme_table": grounding_theme_table,
    }
