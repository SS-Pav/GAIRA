import json
import os
import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, silhouette_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET_ID = "small2023_ev"
PROCESSING_VERSION = "v1_crop670_1800_interp1_minmax"
TARGET_CLASSES = ["c00", "c01", "c10", "c25", "c50", "c100"]
TARGET_PROBES = ["normedprobe1", "normedprobe2"]
SEED = 42
BENCHMARK_PER_GROUP = 2000
PLOT_PER_GROUP = 150
EMBEDDING_DIM = 64


def load_processed_dataset(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        df = connection.execute(
            """
            SELECT
                p.processed_id,
                p.biosample_id,
                m.class_label,
                m.subclass_label,
                p.intensity_json
            FROM biosample_processed_spectra AS p
            JOIN biosample_metadata AS m
              ON p.biosample_id = m.biosample_id
             AND p.dataset_id = m.dataset_id
            WHERE p.dataset_id = ?
              AND p.processing_version = ?
              AND m.subclass_label IN (?, ?)
              AND m.class_label IN (?, ?, ?, ?, ?, ?)
            ORDER BY m.subclass_label, m.class_label, p.biosample_id
            """,
            [DATASET_ID, PROCESSING_VERSION, *TARGET_PROBES, *TARGET_CLASSES],
        ).fetchdf()

    if df.empty:
        raise RuntimeError("No processed small2023_ev spectra were found for the benchmark selection.")

    return df


def balanced_subset(df: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    grouped = []
    counts = []
    for (probe, class_label), group_df in df.groupby(["subclass_label", "class_label"], sort=True):
        if len(group_df) < per_group:
            raise ValueError(f"Group {(probe, class_label)} has only {len(group_df)} rows; need {per_group}.")
        sampled = group_df.sample(n=per_group, random_state=seed).sort_values("biosample_id")
        grouped.append(sampled)
        counts.append({"subclass_label": probe, "class_label": class_label, "n_used": len(sampled)})
    result_df = pd.concat(grouped, ignore_index=True)
    counts_df = pd.DataFrame(counts).sort_values(["subclass_label", "class_label"]).reset_index(drop=True)
    return result_df, counts_df


def decode_intensities(df: pd.DataFrame) -> np.ndarray:
    vectors = [np.asarray(json.loads(value), dtype=np.float32) for value in df["intensity_json"]]
    return np.vstack(vectors).astype(np.float32)


def fit_raw_baseline(X: np.ndarray, probe_labels: np.ndarray, class_labels: np.ndarray) -> pd.DataFrame:
    metrics = []
    for train_probe, test_probe in [("normedprobe1", "normedprobe2"), ("normedprobe2", "normedprobe1")]:
        train_mask = probe_labels == train_probe
        test_mask = probe_labels == test_probe

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])

        clf = LogisticRegression(
            max_iter=400,
            solver="lbfgs",
            random_state=SEED,
        )
        clf.fit(X_train, class_labels[train_mask])
        predictions = clf.predict(X_test)

        metrics.append(
            {
                "direction": f"{train_probe}->{test_probe}",
                "accuracy": accuracy_score(class_labels[test_mask], predictions),
                "balanced_accuracy": balanced_accuracy_score(class_labels[test_mask], predictions),
                "macro_f1": f1_score(class_labels[test_mask], predictions, average="macro"),
            }
        )

    return pd.DataFrame(metrics)


def train_embedding_model(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LabelEncoder, MLPClassifier]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, EMBEDDING_DIM),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=40,
        early_stopping=True,
        n_iter_no_change=8,
        validation_fraction=0.1,
        random_state=SEED,
        verbose=False,
    )
    mlp.fit(X_scaled, y_encoded)
    return scaler, label_encoder, mlp


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def transform_embeddings(scaler: StandardScaler, mlp: MLPClassifier, X: np.ndarray) -> np.ndarray:
    hidden = scaler.transform(X).astype(np.float32)
    for coef, intercept in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        hidden = hidden @ coef + intercept
        hidden = relu(hidden)
    return hidden.astype(np.float32)


def fit_embedding_probe(embeddings: np.ndarray, probe_labels: np.ndarray, class_labels: np.ndarray) -> pd.DataFrame:
    metrics = []
    for train_probe, test_probe in [("normedprobe1", "normedprobe2"), ("normedprobe2", "normedprobe1")]:
        train_mask = probe_labels == train_probe
        test_mask = probe_labels == test_probe

        scaler = StandardScaler()
        X_train = scaler.fit_transform(embeddings[train_mask])
        X_test = scaler.transform(embeddings[test_mask])

        clf = LogisticRegression(
            max_iter=300,
            solver="lbfgs",
            class_weight="balanced",
            random_state=SEED,
        )
        clf.fit(X_train, class_labels[train_mask])
        predictions = clf.predict(X_test)

        metrics.append(
            {
                "direction": f"{train_probe}->{test_probe}",
                "accuracy": accuracy_score(class_labels[test_mask], predictions),
                "balanced_accuracy": balanced_accuracy_score(class_labels[test_mask], predictions),
                "macro_f1": f1_score(class_labels[test_mask], predictions, average="macro"),
            }
        )

    return pd.DataFrame(metrics)


def stratified_plot_subset(df: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    parts = []
    for (_probe, _class_label), group_df in df.groupby(["subclass_label", "class_label"], sort=True):
        take_n = min(per_group, len(group_df))
        parts.append(group_df.sample(n=take_n, random_state=seed).sort_values("biosample_id"))
    return pd.concat(parts, ignore_index=True)


def tsne_projection(values: np.ndarray) -> np.ndarray:
    perplexity = min(30, max(5, len(values) // 20))
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    )
    return model.fit_transform(values)


def plot_projection(coords: np.ndarray, labels: np.ndarray, title: str, output_path: Path) -> None:
    unique_labels = list(dict.fromkeys(labels.tolist()))
    cmap = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(8, 6))
    for index, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.7,
            label=str(label),
            color=cmap(index % cmap.N),
        )
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(markerscale=1.5, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def centroid_distance_summary(
    raw_features: np.ndarray,
    embeddings: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for class_label in TARGET_CLASSES:
        probe1_mask = (df["class_label"] == class_label) & (df["subclass_label"] == "normedprobe1")
        probe2_mask = (df["class_label"] == class_label) & (df["subclass_label"] == "normedprobe2")
        raw_distance = float(
            np.linalg.norm(raw_features[probe1_mask].mean(axis=0) - raw_features[probe2_mask].mean(axis=0))
        )
        embedding_distance = float(
            np.linalg.norm(embeddings[probe1_mask].mean(axis=0) - embeddings[probe2_mask].mean(axis=0))
        )
        rows.append(
            {
                "class_label": class_label,
                "raw_cross_probe_centroid_distance": raw_distance,
                "embedding_cross_probe_centroid_distance": embedding_distance,
            }
        )
    summary_df = pd.DataFrame(rows)
    overall_row = pd.DataFrame(
        [
            {
                "class_label": "overall_mean",
                "raw_cross_probe_centroid_distance": summary_df["raw_cross_probe_centroid_distance"].mean(),
                "embedding_cross_probe_centroid_distance": summary_df[
                    "embedding_cross_probe_centroid_distance"
                ].mean(),
            }
        ]
    )
    return pd.concat([summary_df, overall_row], ignore_index=True)


def mixture_order_correlation(values: np.ndarray, class_labels: np.ndarray) -> float:
    order_map = {"c00": 0, "c01": 1, "c10": 2, "c25": 3, "c50": 4, "c100": 5}
    centroids = []
    for class_label in TARGET_CLASSES:
        mask = class_labels == class_label
        centroids.append(values[mask].mean(axis=0))
    centroids = np.vstack(centroids)

    order_diffs = []
    distances = []
    for i in range(len(TARGET_CLASSES)):
        for j in range(i + 1, len(TARGET_CLASSES)):
            order_diffs.append(abs(order_map[TARGET_CLASSES[i]] - order_map[TARGET_CLASSES[j]]))
            distances.append(float(np.linalg.norm(centroids[i] - centroids[j])))

    return float(np.corrcoef(np.asarray(order_diffs), np.asarray(distances))[0, 1])


def sampled_silhouette(values: np.ndarray, labels: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    sample_size = min(3000, len(values))
    indices = rng.choice(len(values), size=sample_size, replace=False)
    return float(silhouette_score(values[indices], labels[indices]))


def build_summary_text(
    counts_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    distance_df: pd.DataFrame,
) -> str:
    lines = []
    lines.append("small2023_ev substrate-invariant embedding benchmark")
    lines.append(f"Balanced spectra per probe/class group: {BENCHMARK_PER_GROUP}")
    lines.append("")
    lines.append("Spectra used per group:")
    lines.append(counts_df.to_string(index=False))
    lines.append("")
    lines.append("Raw-feature cross-probe baseline:")
    lines.append(baseline_df.to_string(index=False))
    lines.append("")
    lines.append("Embedding cross-probe linear probe:")
    lines.append(embedding_df.to_string(index=False))
    lines.append("")
    lines.append("Geometry metrics:")
    lines.append(geometry_df.to_string(index=False))
    lines.append("")
    lines.append("Cross-probe centroid distance summary:")
    lines.append(distance_df.to_string(index=False))
    lines.append("")
    baseline_mean = baseline_df["accuracy"].mean()
    embedding_mean = embedding_df["accuracy"].mean()
    probe_sep_before = geometry_df.loc[0, "raw_probe_silhouette"]
    probe_sep_after = geometry_df.loc[0, "embedding_probe_silhouette"]
    class_sep_before = geometry_df.loc[0, "raw_class_silhouette"]
    class_sep_after = geometry_df.loc[0, "embedding_class_silhouette"]
    lines.append("Interpretation:")
    lines.append(
        f"- Mean cross-probe accuracy changed from {baseline_mean:.4f} to {embedding_mean:.4f}."
    )
    lines.append(
        f"- Class separability changed from {class_sep_before:.4f} to {class_sep_after:.4f}."
    )
    lines.append(
        f"- Probe separability changed from {probe_sep_before:.4f} to {probe_sep_after:.4f}."
    )
    lines.append(
        "- Treat this as an invariant-benchmark baseline rather than a production model. "
        "The representation is supervised and deterministic but not a full contrastive deep-learning system."
    )
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import ensure_storage_dirs, resolve_storage_path

    storage_config = ensure_storage_dirs()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))
    if processed_root is None:
        raise RuntimeError("The storage config is missing processed_data.")

    output_dir = processed_root / "small2023_ev_invariant_embedding"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    db_path = project_root / "data" / "gaira.duckdb"

    full_df = load_processed_dataset(db_path)
    benchmark_df, counts_df = balanced_subset(full_df, BENCHMARK_PER_GROUP, SEED)
    X = decode_intensities(benchmark_df)
    class_labels = benchmark_df["class_label"].to_numpy()
    probe_labels = benchmark_df["subclass_label"].to_numpy()

    dataset_metadata_path = output_dir / "small2023_ev_invariant_dataset_metadata.csv"
    dataset_npz_path = output_dir / "small2023_ev_invariant_dataset.npz"
    counts_path = output_dir / "benchmark_sample_counts.csv"
    baseline_metrics_path = output_dir / "baseline_cross_probe_metrics.csv"
    embedding_metrics_path = output_dir / "embedding_cross_probe_metrics.csv"
    distance_summary_path = output_dir / "class_probe_distance_summary.csv"
    geometry_metrics_path = output_dir / "geometry_metrics.csv"
    summary_path = output_dir / "embedding_summary.txt"

    benchmark_df.drop(columns=["intensity_json"]).to_csv(dataset_metadata_path, index=False)
    np.savez_compressed(dataset_npz_path, X=X, class_labels=class_labels, probe_labels=probe_labels)
    counts_df.to_csv(counts_path, index=False)

    baseline_metrics_df = fit_raw_baseline(X, probe_labels, class_labels)
    baseline_metrics_df.to_csv(baseline_metrics_path, index=False)

    embedding_scaler, _label_encoder, embedding_model = train_embedding_model(X, class_labels)
    embeddings = transform_embeddings(embedding_scaler, embedding_model, X)
    embedding_metrics_df = fit_embedding_probe(embeddings, probe_labels, class_labels)
    embedding_metrics_df.to_csv(embedding_metrics_path, index=False)

    raw_scaler = StandardScaler()
    raw_scaled = raw_scaler.fit_transform(X)
    geometry_df = pd.DataFrame(
        [
            {
                "raw_class_silhouette": sampled_silhouette(raw_scaled, class_labels, SEED),
                "embedding_class_silhouette": sampled_silhouette(embeddings, class_labels, SEED),
                "raw_probe_silhouette": sampled_silhouette(raw_scaled, probe_labels, SEED),
                "embedding_probe_silhouette": sampled_silhouette(embeddings, probe_labels, SEED),
                "raw_mixture_order_correlation": mixture_order_correlation(raw_scaled, class_labels),
                "embedding_mixture_order_correlation": mixture_order_correlation(embeddings, class_labels),
            }
        ]
    )
    geometry_df.to_csv(geometry_metrics_path, index=False)

    distance_df = centroid_distance_summary(raw_scaled, embeddings, benchmark_df)
    distance_df.to_csv(distance_summary_path, index=False)

    plot_df = stratified_plot_subset(benchmark_df, PLOT_PER_GROUP, SEED).reset_index(drop=True)
    plot_X = decode_intensities(plot_df)
    plot_raw = raw_scaler.transform(plot_X)
    plot_embeddings = transform_embeddings(embedding_scaler, embedding_model, plot_X)

    raw_coords = tsne_projection(plot_raw)
    embedding_coords = tsne_projection(plot_embeddings)
    plot_projection(raw_coords, plot_df["class_label"].to_numpy(), "Raw Processed Spectra t-SNE by Class", output_dir / "raw_tsne_by_class.png")
    plot_projection(raw_coords, plot_df["subclass_label"].to_numpy(), "Raw Processed Spectra t-SNE by Probe", output_dir / "raw_tsne_by_probe.png")
    plot_projection(embedding_coords, plot_df["class_label"].to_numpy(), "Embedding t-SNE by Class", output_dir / "embedding_tsne_by_class.png")
    plot_projection(embedding_coords, plot_df["subclass_label"].to_numpy(), "Embedding t-SNE by Probe", output_dir / "embedding_tsne_by_probe.png")

    summary_text = build_summary_text(counts_df, baseline_metrics_df, embedding_metrics_df, geometry_df, distance_df)
    summary_path.write_text(summary_text, encoding="utf-8")

    print(f"Wrote benchmark outputs to: {output_dir}")
    print(counts_df.to_string(index=False))
    print("")
    print("Baseline metrics:")
    print(baseline_metrics_df.to_string(index=False))
    print("")
    print("Embedding metrics:")
    print(embedding_metrics_df.to_string(index=False))
    print("")
    print("Geometry metrics:")
    print(geometry_df.to_string(index=False))


if __name__ == "__main__":
    main()
