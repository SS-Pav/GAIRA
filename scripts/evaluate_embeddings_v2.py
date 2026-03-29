from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


DEFAULT_SAMPLE_SIZE = 15000


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.runtime import REMOTE_OUTPUT_ROOT, add_common_io_args

    parser = argparse.ArgumentParser(description="Scalable v2 embedding evaluation.")
    add_common_io_args(parser, default_run_name="embedding_v2", default_root=REMOTE_OUTPUT_ROOT)
    parser.add_argument("--report-dir", default=None, help="Directory to write v2 evaluation outputs. Defaults to output-dir.")
    parser.add_argument("--sample-size-global-metrics", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--neighbor-backend", choices=["sklearn", "pynndescent"], default="sklearn")
    return parser.parse_args()


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


def resolve_report_dir(output_dir: Path, report_dir: str | None) -> Path:
    return Path(report_dir).expanduser().resolve() if report_dir else output_dir


def _allocate_group_counts(group_counts: pd.Series, target_size: int) -> dict[str, int]:
    total = int(group_counts.sum())
    if target_size >= total:
        return {str(key): int(value) for key, value in group_counts.items()}

    raw = group_counts / total * target_size
    alloc = np.floor(raw).astype(int)
    if target_size >= len(group_counts):
        alloc = np.maximum(alloc, 1)
    while int(alloc.sum()) > target_size:
        idx = int(np.argmax(alloc))
        alloc.iloc[idx] -= 1
    remainder = target_size - int(alloc.sum())
    if remainder > 0:
        fractional = (raw - np.floor(raw)).sort_values(ascending=False)
        for key in fractional.index[:remainder]:
            alloc.loc[key] += 1
    return {str(key): int(value) for key, value in alloc.items()}


def stratified_sample_indices(metadata_df: pd.DataFrame, target_size: int, seed: int) -> np.ndarray:
    if target_size <= 0 or len(metadata_df) <= target_size:
        return metadata_df.index.to_numpy()

    rng = np.random.default_rng(seed)
    working = metadata_df.copy()
    working["label_bucket"] = working["label_optional"].fillna("").astype(str).replace("", "__nolabel__")
    working["strata"] = working["dataset_id"].astype(str) + "::" + working["label_bucket"].astype(str)
    group_counts = working.groupby("strata").size().sort_index()
    allocations = _allocate_group_counts(group_counts, target_size)

    selected: list[np.ndarray] = []
    for strata, group in working.groupby("strata", sort=False):
        n_select = allocations.get(str(strata), 0)
        if n_select <= 0:
            continue
        if len(group) <= n_select:
            selected.append(group.index.to_numpy())
        else:
            chosen = rng.choice(group.index.to_numpy(), size=n_select, replace=False)
            selected.append(np.sort(chosen))
    return np.sort(np.concatenate(selected)) if selected else np.array([], dtype=int)


def build_knn_indices(embeddings: np.ndarray, k: int, seed: int, backend: str) -> np.ndarray:
    target_k = min(k + 1, len(embeddings))
    if backend == "pynndescent":
        from pynndescent import NNDescent

        index = NNDescent(embeddings, metric="cosine", n_neighbors=target_k, random_state=seed)
        indices, _ = index.query(embeddings, k=target_k)
    else:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=target_k, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(embeddings)
        indices = nn.kneighbors(embeddings, return_distance=False)

    rows = []
    for row_index, row in enumerate(indices):
        filtered = [idx for idx in row.tolist() if int(idx) != row_index]
        if len(filtered) < k:
            filtered.extend([row_index] * (k - len(filtered)))
        rows.append(filtered[:k])
    return np.asarray(rows, dtype=int)


def neighbor_consistency(neighbor_indices: np.ndarray, labels: pd.Series, metric_name: str) -> list[dict[str, object]]:
    valid = labels.fillna("").astype(str)
    valid_mask = valid != ""
    if int(valid_mask.sum()) <= 1:
        return [
            {"metric": f"nn_consistency_{metric_name}", "value": np.nan, "evaluation_tier": "full_corpus"},
            {"metric": f"top1_match_{metric_name}", "value": np.nan, "evaluation_tier": "full_corpus"},
        ]

    rows = np.where(valid_mask.to_numpy())[0]
    y = valid.to_numpy()
    purity = []
    top1 = []
    for row in rows:
        neighbors = [idx for idx in neighbor_indices[row] if valid_mask.iloc[idx]]
        if not neighbors:
            continue
        top1.append(float(y[neighbors[0]] == y[row]))
        purity.append(float(np.mean([y[idx] == y[row] for idx in neighbors])))
    return [
        {"metric": f"nn_consistency_{metric_name}", "value": float(np.mean(purity)) if purity else np.nan, "evaluation_tier": "full_corpus"},
        {"metric": f"top1_match_{metric_name}", "value": float(np.mean(top1)) if top1 else np.nan, "evaluation_tier": "full_corpus"},
    ]


def label_silhouette(embeddings: np.ndarray, labels: pd.Series, label_name: str) -> dict[str, object]:
    valid = labels.fillna("").astype(str)
    valid_mask = valid != ""
    valid = valid[valid_mask]
    if valid.nunique() < 2:
        return {"metric": f"silhouette_{label_name}", "value": np.nan, "evaluation_tier": "sampled_global"}
    return {
        "metric": f"silhouette_{label_name}",
        "value": float(silhouette_score(embeddings[valid_mask.to_numpy()], valid.to_numpy())),
        "evaluation_tier": "sampled_global",
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.runtime import resolve_output_dir

    args = parse_args()
    output_dir = resolve_output_dir(args)
    report_dir = resolve_report_dir(output_dir, args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(output_dir / "embeddings.npy")
    metadata_df = pd.read_csv(output_dir / "metadata.csv")
    if "family_label" not in metadata_df.columns:
        metadata_df["family_label"] = family_label(metadata_df)

    neighbor_indices = build_knn_indices(embeddings, k=args.knn_k, seed=args.seed, backend=args.neighbor_backend)
    metrics: list[dict[str, object]] = []
    for column, name in [
        ("sample_type", "sample_type"),
        ("dataset_id", "dataset_id"),
        ("label_optional", "class"),
        ("family_label", "family"),
    ]:
        metrics.extend(neighbor_consistency(neighbor_indices, metadata_df[column], name))

    sample_indices = stratified_sample_indices(metadata_df, args.sample_size_global_metrics, args.seed)
    sampled_df = metadata_df.loc[sample_indices].copy()
    sampled_df["original_index"] = sample_indices
    sampled_df.to_csv(report_dir / "sample_manifest.csv", index=False)
    sampled_embeddings = embeddings[sample_indices]

    for column, name in [
        ("sample_type", "sample_type"),
        ("dataset_id", "dataset_id"),
        ("label_optional", "class"),
        ("family_label", "family"),
    ]:
        metrics.append(label_silhouette(sampled_embeddings, sampled_df[column], name))

    metrics_df = pd.DataFrame(metrics)
    metrics_df["source_output_dir"] = str(output_dir)
    metrics_df["sample_size_global_metrics"] = int(len(sampled_df))
    metrics_df["knn_k"] = int(args.knn_k)
    metrics_df["neighbor_backend"] = args.neighbor_backend
    metrics_df.to_csv(report_dir / "embedding_metrics_v2.csv", index=False)

    report = textwrap.dedent(
        f"""
        Embedding evaluation v2 report

        Source embedding folder:
        - {output_dir}

        Design:
        - Full-corpus metrics are local-neighborhood metrics computed from a kNN graph.
        - Sampled-global metrics are computed on a deterministic stratified subset.

        Full-corpus metrics:
        {metrics_df[metrics_df['evaluation_tier'] == 'full_corpus'].to_string(index=False)}

        Sampled-global metrics:
        {metrics_df[metrics_df['evaluation_tier'] == 'sampled_global'].to_string(index=False)}

        Sample manifest:
        - size = {len(sampled_df)}
        - file = {report_dir / 'sample_manifest.csv'}
        """
    )
    (report_dir / "embedding_report_v2.md").write_text(report, encoding="utf-8")
    print(f"Saved metrics: {report_dir / 'embedding_metrics_v2.csv'}")


if __name__ == "__main__":
    main()
