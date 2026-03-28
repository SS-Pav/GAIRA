from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors


RNG = np.random.default_rng(7)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.runtime import REMOTE_OUTPUT_ROOT, add_common_io_args

    parser = argparse.ArgumentParser(description="Evaluate spectral embedding geometry.")
    add_common_io_args(parser, default_run_name="embedding_v2", default_root=REMOTE_OUTPUT_ROOT)
    return parser.parse_args()


def label_silhouette(embeddings: np.ndarray, labels: pd.Series, label_name: str) -> dict[str, object]:
    valid = labels.fillna("").astype(str)
    valid_mask = valid != ""
    valid = valid[valid_mask]
    if valid.nunique() < 2:
        return {"metric": f"silhouette_{label_name}", "value": np.nan}
    return {"metric": f"silhouette_{label_name}", "value": float(silhouette_score(embeddings[valid_mask.to_numpy()], valid.to_numpy()))}


def neighbor_consistency(embeddings: np.ndarray, labels: pd.Series, label_name: str, n_neighbors: int = 6) -> dict[str, object]:
    valid = labels.fillna("").astype(str)
    valid_mask = valid != ""
    if valid_mask.sum() <= n_neighbors:
        return {"metric": f"nn_consistency_{label_name}", "value": np.nan}
    X = embeddings[valid_mask.to_numpy()]
    y = valid[valid_mask].to_numpy()
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)[:, 1:]
    score = float(np.mean([np.mean(y[row] == y[index]) for index, row in enumerate(indices)]))
    return {"metric": f"nn_consistency_{label_name}", "value": score}


def intra_inter_distance(embeddings: np.ndarray, labels: pd.Series, label_name: str) -> list[dict[str, object]]:
    valid = labels.fillna("").astype(str)
    valid_mask = valid != ""
    valid = valid[valid_mask]
    if valid.nunique() < 2:
        return []
    X = embeddings[valid_mask.to_numpy()]
    if len(X) > 1500:
        indices = np.sort(RNG.choice(np.arange(len(X)), size=1500, replace=False))
        X = X[indices]
        valid = valid.iloc[indices]
    distance_matrix = pairwise_distances(X, metric="cosine")
    rows: list[dict[str, object]] = []
    for label in valid.unique():
        label_mask = valid == label
        other_mask = valid != label
        intra = distance_matrix[np.ix_(label_mask.to_numpy(), label_mask.to_numpy())]
        inter = distance_matrix[np.ix_(label_mask.to_numpy(), other_mask.to_numpy())]
        rows.append(
            {
                "metric": f"intra_distance_{label_name}",
                "label": label,
                "intra_distance": float(np.mean(intra[np.triu_indices_from(intra, k=1)])) if intra.shape[0] > 1 else np.nan,
                "inter_distance": float(np.mean(inter)) if inter.size else np.nan,
            }
        )
    return rows


def family_label(metadata_df: pd.DataFrame) -> pd.Series:
    mapping = {
        "adenine_sers_control": "grounding_analyte",
        "metabolite_sers63_support": "grounding_analyte",
        "amino_acid_raman_grounding": "grounding_analyte",
        "small2023_ev": "ev_general",
        "shine_ev_sers": "ev_disease_or_stress",
        "diabetes_plasma_ev_sers": "ev_disease_or_stress",
        "covid_serum_raman": "serum_general",
        "serum_protocol_comparison": "serum_general",
        "serum_ag_colloids": "serum_general",
        "cspp_serum": "serum_general",
        "ergothioneine_serum": "serum_general",
        "cca_hcc_lm_serum_sers": "serum_liver_hepatobiliary",
    }
    return metadata_df["dataset_id"].map(mapping).fillna(metadata_df["sample_type"].astype(str))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.runtime import resolve_output_dir

    args = parse_args()
    output_dir = resolve_output_dir(args)
    embeddings = np.load(output_dir / "embeddings.npy")
    metadata_df = pd.read_csv(output_dir / "metadata.csv")
    metadata_df["family_label"] = family_label(metadata_df)

    metrics: list[dict[str, object]] = []
    metrics.append(label_silhouette(embeddings, metadata_df["sample_type"], "sample_type"))
    metrics.append(label_silhouette(embeddings, metadata_df["dataset_id"], "dataset_id"))
    metrics.append(label_silhouette(embeddings, metadata_df["label_optional"], "class"))
    metrics.append(label_silhouette(embeddings, metadata_df["family_label"], "family"))
    metrics.append(neighbor_consistency(embeddings, metadata_df["sample_type"], "sample_type"))
    metrics.append(neighbor_consistency(embeddings, metadata_df["dataset_id"], "dataset_id"))
    metrics.append(neighbor_consistency(embeddings, metadata_df["label_optional"], "class"))
    metrics.append(neighbor_consistency(embeddings, metadata_df["family_label"], "family"))

    distance_rows = (
        intra_inter_distance(embeddings, metadata_df["sample_type"], "sample_type")
        + intra_inter_distance(embeddings, metadata_df["dataset_id"], "dataset_id")
        + intra_inter_distance(embeddings, metadata_df["label_optional"], "class")
        + intra_inter_distance(embeddings, metadata_df["family_label"], "family")
    )

    metric_df = pd.DataFrame(metrics)
    distance_df = pd.DataFrame(distance_rows)
    metric_df.to_csv(output_dir / "embedding_metrics.csv", index=False)
    if not distance_df.empty:
        distance_df.to_csv(output_dir / "embedding_distance_metrics.csv", index=False)

    report = textwrap.dedent(
        f"""
        Embedding evaluation report

        Scalar metrics:

        {metric_df.to_string(index=False)}

        Distance summary:

        {distance_df.to_string(index=False) if not distance_df.empty else '_empty_'}

        Interpretation:

        - Higher silhouette by sample type, family, and dataset indicates non-random spectral geometry.
        - Higher nearest-neighbor consistency indicates local manifold coherence.
        - Useful embeddings should show intra-cluster distances below inter-cluster distances for at least sample type and dataset groupings.
        """
    )
    (output_dir / "embedding_report.md").write_text(report, encoding="utf-8")
    print(f"Saved metrics: {output_dir / 'embedding_metrics.csv'}")


if __name__ == "__main__":
    main()
