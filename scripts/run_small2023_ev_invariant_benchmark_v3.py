from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET_ID = "small2023_ev"
PROCESSING_VERSION = "v1_crop670_1800_interp1_minmax"
TARGET_CLASSES = ["c00", "c01", "c10", "c25", "c50", "c100"]
TARGET_PROBES = ["normedprobe1", "normedprobe2"]
SEED = 42
BENCHMARK_PER_GROUP = 2000
PLOT_PER_GROUP = 150
SILHOUETTE_PER_GROUP = 250
ORDER_MAP = {"c00": 0, "c01": 1, "c10": 2, "c25": 3, "c50": 4, "c100": 5}
V1_DIR = Path("/Volumes/SSD_SPG/GAIRA_DATA/processed/small2023_ev_invariant_embedding")
V2_DIR = Path("/Volumes/SSD_SPG/GAIRA_DATA/processed/small2023_ev_invariant_embedding_v2")
V3_DIR = Path("/Volumes/SSD_SPG/GAIRA_DATA/processed/small2023_ev_invariant_embedding_v3")
MPLCONFIGDIR = V3_DIR / ".mplconfig"


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
        raise RuntimeError("No processed small2023_ev spectra were found for the v3 benchmark selection.")
    return df


def balanced_subset(df: pd.DataFrame, per_group: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return np.vstack([np.asarray(json.loads(value), dtype=np.float32) for value in df["intensity_json"]]).astype(np.float32)


def fit_raw_baseline(X: np.ndarray, probe_labels: np.ndarray, class_labels: np.ndarray) -> pd.DataFrame:
    metrics = []
    for train_probe, test_probe in [("normedprobe1", "normedprobe2"), ("normedprobe2", "normedprobe1")]:
        train_mask = probe_labels == train_probe
        test_mask = probe_labels == test_probe
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])
        clf = LogisticRegression(max_iter=400, solver="lbfgs", random_state=SEED)
        clf.fit(X_train, class_labels[train_mask])
        predictions = clf.predict(X_test)
        metrics.append(
            {
                "model": "raw_baseline",
                "protocol": "strict_train_only",
                "direction": f"{train_probe}->{test_probe}",
                "accuracy": accuracy_score(class_labels[test_mask], predictions),
                "balanced_accuracy": balanced_accuracy_score(class_labels[test_mask], predictions),
                "macro_f1": f1_score(class_labels[test_mask], predictions, average="macro"),
            }
        )
    return pd.DataFrame(metrics)


def fit_source_only_embedding(X_train: np.ndarray, y_train: np.ndarray) -> tuple[StandardScaler, LinearDiscriminantAnalysis, Ridge]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)

    lda = LinearDiscriminantAnalysis(n_components=5)
    lda.fit(X_train_scaled, y_train)

    order_targets = np.asarray([ORDER_MAP[label] for label in y_train], dtype=np.float32)
    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_train_scaled, order_targets)
    return scaler, lda, ridge


def transform_source_only_embedding(
    scaler: StandardScaler,
    lda: LinearDiscriminantAnalysis,
    ridge: Ridge,
    X: np.ndarray,
) -> np.ndarray:
    X_scaled = scaler.transform(X).astype(np.float32)
    Z_class = lda.transform(X_scaled).astype(np.float32)
    order_score = ridge.predict(X_scaled).reshape(-1, 1).astype(np.float32)
    return np.hstack([Z_class, order_score]).astype(np.float32)


def fit_v3_directional_metrics(X: np.ndarray, probe_labels: np.ndarray, class_labels: np.ndarray) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    metrics = []
    models: dict[str, dict[str, object]] = {}
    for train_probe, test_probe in [("normedprobe1", "normedprobe2"), ("normedprobe2", "normedprobe1")]:
        train_mask = probe_labels == train_probe
        test_mask = probe_labels == test_probe
        scaler, lda, ridge = fit_source_only_embedding(X[train_mask], class_labels[train_mask])
        train_embeddings = transform_source_only_embedding(scaler, lda, ridge, X[train_mask])
        test_embeddings = transform_source_only_embedding(scaler, lda, ridge, X[test_mask])

        clf_scaler = StandardScaler()
        X_train = clf_scaler.fit_transform(train_embeddings)
        X_test = clf_scaler.transform(test_embeddings)
        clf = LogisticRegression(max_iter=400, solver="lbfgs", random_state=SEED)
        clf.fit(X_train, class_labels[train_mask])
        predictions = clf.predict(X_test)
        direction = f"{train_probe}->{test_probe}"
        metrics.append(
            {
                "model": "v3_embedding",
                "protocol": "strict_train_only",
                "direction": direction,
                "accuracy": accuracy_score(class_labels[test_mask], predictions),
                "balanced_accuracy": balanced_accuracy_score(class_labels[test_mask], predictions),
                "macro_f1": f1_score(class_labels[test_mask], predictions, average="macro"),
            }
        )
        models[train_probe] = {
            "source_probe": train_probe,
            "target_probe": test_probe,
            "scaler": scaler,
            "lda": lda,
            "ridge": ridge,
        }
    return pd.DataFrame(metrics), models


def build_consensus_v3_embedding(X: np.ndarray, source_models: dict[str, dict[str, object]]) -> np.ndarray:
    probe1_model = source_models["normedprobe1"]
    probe2_model = source_models["normedprobe2"]
    embed_from_probe1 = transform_source_only_embedding(
        probe1_model["scaler"], probe1_model["lda"], probe1_model["ridge"], X
    )
    embed_from_probe2 = transform_source_only_embedding(
        probe2_model["scaler"], probe2_model["lda"], probe2_model["ridge"], X
    )
    combined = np.hstack([embed_from_probe1, embed_from_probe2]).astype(np.float32)
    scaler = StandardScaler()
    return scaler.fit_transform(combined).astype(np.float32)


def stratified_subset(df: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    parts = []
    for (_probe, _class_label), group_df in df.groupby(["subclass_label", "class_label"], sort=True):
        parts.append(group_df.sample(n=min(per_group, len(group_df)), random_state=seed).sort_values("biosample_id"))
    return pd.concat(parts, ignore_index=True)


def mixture_order_correlation(values: np.ndarray, class_labels: np.ndarray) -> float:
    centroids = []
    for class_label in TARGET_CLASSES:
        centroids.append(values[class_labels == class_label].mean(axis=0))
    centroids = np.vstack(centroids)
    order_diffs = []
    distances = []
    for i in range(len(TARGET_CLASSES)):
        for j in range(i + 1, len(TARGET_CLASSES)):
            order_diffs.append(abs(ORDER_MAP[TARGET_CLASSES[i]] - ORDER_MAP[TARGET_CLASSES[j]]))
            distances.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    return float(np.corrcoef(np.asarray(order_diffs), np.asarray(distances))[0, 1])


def centroid_distance_summary(raw_features: np.ndarray, v3_embeddings: np.ndarray, df: pd.DataFrame, v1_mean: float, v2_mean: float) -> pd.DataFrame:
    rows = []
    for class_label in TARGET_CLASSES:
        probe1_mask = (df["class_label"] == class_label) & (df["subclass_label"] == "normedprobe1")
        probe2_mask = (df["class_label"] == class_label) & (df["subclass_label"] == "normedprobe2")
        raw_distance = float(np.linalg.norm(raw_features[probe1_mask].mean(axis=0) - raw_features[probe2_mask].mean(axis=0)))
        v3_distance = float(np.linalg.norm(v3_embeddings[probe1_mask].mean(axis=0) - v3_embeddings[probe2_mask].mean(axis=0)))
        rows.append(
            {
                "class_label": class_label,
                "raw_cross_probe_centroid_distance": raw_distance,
                "v1_embedding_cross_probe_centroid_distance": np.nan,
                "v2_embedding_cross_probe_centroid_distance": np.nan,
                "v3_embedding_cross_probe_centroid_distance": v3_distance,
            }
        )
    summary_df = pd.DataFrame(rows)
    overall = pd.DataFrame(
        [
            {
                "class_label": "overall_mean",
                "raw_cross_probe_centroid_distance": summary_df["raw_cross_probe_centroid_distance"].mean(),
                "v1_embedding_cross_probe_centroid_distance": v1_mean,
                "v2_embedding_cross_probe_centroid_distance": v2_mean,
                "v3_embedding_cross_probe_centroid_distance": summary_df["v3_embedding_cross_probe_centroid_distance"].mean(),
            }
        ]
    )
    return pd.concat([summary_df, overall], ignore_index=True)


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
        ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.72, label=str(label), color=cmap(index % cmap.N))
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(markerscale=1.5, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_metric_plot(comparison_df: pd.DataFrame, geometry_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    transfer_df = comparison_df[comparison_df["direction"].notna()].copy()
    metric_df = transfer_df.pivot(index="direction", columns="model", values="accuracy").reset_index()
    x = np.arange(len(metric_df))
    width = 0.18
    for idx, model in enumerate(["raw_baseline", "v1_embedding", "v2_embedding", "v3_embedding"]):
        axes[0].bar(x + (idx - 1.5) * width, metric_df[model], width=width, label=model.replace("_", " "))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_df["direction"])
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Cross-probe transfer")
    axes[0].legend(frameon=False, fontsize=8)

    g = geometry_df.iloc[0]
    labels = ["Class sep.", "Probe sep.", "Order corr."]
    raw_vals = [g["raw_class_silhouette"], g["raw_probe_silhouette"], g["raw_mixture_order_correlation"]]
    v1_vals = [g["v1_class_silhouette"], g["v1_probe_silhouette"], g["v1_mixture_order_correlation"]]
    v2_vals = [g["v2_class_silhouette"], g["v2_probe_silhouette"], g["v2_mixture_order_correlation"]]
    v3_vals = [g["v3_class_silhouette"], g["v3_probe_silhouette"], g["v3_mixture_order_correlation"]]
    x2 = np.arange(len(labels))
    for idx, vals in enumerate([raw_vals, v1_vals, v2_vals, v3_vals]):
        axes[1].bar(x2 + (idx - 1.5) * width, vals, width=width, label=["raw", "v1", "v2", "v3"][idx])
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Geometry summary")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_summary_text(counts_df: pd.DataFrame, comparison_df: pd.DataFrame, geometry_df: pd.DataFrame, distance_df: pd.DataFrame) -> str:
    overall = distance_df[distance_df["class_label"] == "overall_mean"].iloc[0]
    lines = [
        "small2023_ev invariant embedding benchmark v3",
        "Model/objective used: strict train-only source-supervised embedding",
        "- Source-only StandardScaler fit on training probe",
        "- Source-only supervised LDA fit on training probe class labels",
        "- Source-only Ridge ordinal component fit on training probe class order",
        "- Frozen transform applied to target probe without target labels",
        "- No joint class-supervised fitting across both probes in the primary benchmark",
        "- Geometry summary uses a consensus embedding from two independently source-only transforms",
        "",
        "Spectra used per group:",
        counts_df.to_string(index=False),
        "",
        "Comparison metrics:",
        comparison_df.to_string(index=False),
        "",
        "Geometry metrics:",
        geometry_df.to_string(index=False),
        "",
        "Centroid distance summary:",
        distance_df.to_string(index=False),
        "",
        f"Overall centroid distance raw/v1/v2/v3: {overall['raw_cross_probe_centroid_distance']:.4f} / "
        f"{overall['v1_embedding_cross_probe_centroid_distance']:.4f} / "
        f"{overall['v2_embedding_cross_probe_centroid_distance']:.4f} / "
        f"{overall['v3_embedding_cross_probe_centroid_distance']:.4f}",
    ]
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import ensure_storage_dirs

    ensure_storage_dirs()
    V3_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

    db_path = project_root / "data" / "gaira.duckdb"
    full_df = load_processed_dataset(db_path)
    benchmark_df, counts_df = balanced_subset(full_df, BENCHMARK_PER_GROUP, SEED)
    X = decode_intensities(benchmark_df)
    class_labels = benchmark_df["class_label"].to_numpy()
    probe_labels = benchmark_df["subclass_label"].to_numpy()

    np.savez_compressed(
        V3_DIR / "small2023_ev_invariant_dataset_v3.npz",
        X=X,
        class_labels=class_labels,
        probe_labels=probe_labels,
    )
    benchmark_df.drop(columns=["intensity_json"]).to_csv(V3_DIR / "small2023_ev_invariant_dataset_v3_metadata.csv", index=False)
    counts_df.to_csv(V3_DIR / "benchmark_sample_counts_v3.csv", index=False)

    raw_scaler = StandardScaler()
    raw_scaled = raw_scaler.fit_transform(X).astype(np.float32)
    baseline_df = fit_raw_baseline(X, probe_labels, class_labels)

    v1_embedding_df = pd.read_csv(V1_DIR / "embedding_cross_probe_metrics.csv").assign(model="v1_embedding")
    v2_embedding_df = pd.read_csv(V2_DIR / "comparison_cross_probe_metrics_v2.csv")
    v2_embedding_df = v2_embedding_df[v2_embedding_df["model"] == "v2_embedding"].copy()

    v1_geometry = pd.read_csv(V1_DIR / "geometry_metrics.csv").iloc[0]
    v2_geometry = pd.read_csv(V2_DIR / "geometry_metrics_v2.csv").iloc[0]
    v1_distances = pd.read_csv(V1_DIR / "class_probe_distance_summary.csv")
    v2_distances = pd.read_csv(V2_DIR / "class_probe_distance_summary_v2.csv")
    v1_mean_distance = float(v1_distances[v1_distances["class_label"] == "overall_mean"]["embedding_cross_probe_centroid_distance"].iloc[0])
    v2_mean_distance = float(v2_distances[v2_distances["class_label"] == "overall_mean"]["v2_embedding_cross_probe_centroid_distance"].iloc[0])

    v3_embedding_df, source_models = fit_v3_directional_metrics(X, probe_labels, class_labels)
    comparison_df = pd.concat(
        [
            baseline_df,
            v1_embedding_df,
            v2_embedding_df,
            v3_embedding_df,
        ],
        ignore_index=True,
    )
    comparison_df.to_csv(V3_DIR / "comparison_cross_probe_metrics_v3.csv", index=False)

    silhouette_df = stratified_subset(benchmark_df, SILHOUETTE_PER_GROUP, SEED)
    silhouette_X = decode_intensities(silhouette_df)
    silhouette_raw_scaled = raw_scaler.transform(silhouette_X).astype(np.float32)
    silhouette_v3 = build_consensus_v3_embedding(silhouette_X, source_models)

    geometry_df = pd.DataFrame(
        [
            {
                "raw_class_silhouette": silhouette_score(silhouette_raw_scaled, silhouette_df["class_label"].to_numpy()),
                "v1_class_silhouette": v1_geometry["embedding_class_silhouette"],
                "v2_class_silhouette": v2_geometry["v2_class_silhouette"],
                "v3_class_silhouette": silhouette_score(silhouette_v3, silhouette_df["class_label"].to_numpy()),
                "raw_probe_silhouette": silhouette_score(silhouette_raw_scaled, silhouette_df["subclass_label"].to_numpy()),
                "v1_probe_silhouette": v1_geometry["embedding_probe_silhouette"],
                "v2_probe_silhouette": v2_geometry["v2_probe_silhouette"],
                "v3_probe_silhouette": silhouette_score(silhouette_v3, silhouette_df["subclass_label"].to_numpy()),
                "raw_mixture_order_correlation": mixture_order_correlation(raw_scaled, class_labels),
                "v1_mixture_order_correlation": v1_geometry["embedding_mixture_order_correlation"],
                "v2_mixture_order_correlation": v2_geometry["v2_mixture_order_correlation"],
                "v3_mixture_order_correlation": mixture_order_correlation(build_consensus_v3_embedding(X, source_models), class_labels),
            }
        ]
    )
    geometry_df.to_csv(V3_DIR / "geometry_metrics_v3.csv", index=False)

    v3_consensus = build_consensus_v3_embedding(X, source_models)
    distance_df = centroid_distance_summary(raw_scaled, v3_consensus, benchmark_df, v1_mean_distance, v2_mean_distance)
    distance_df.to_csv(V3_DIR / "class_probe_distance_summary_v3.csv", index=False)

    mixture_df = pd.DataFrame(
        [
            {
                "raw_mixture_order_correlation": geometry_df.iloc[0]["raw_mixture_order_correlation"],
                "v1_mixture_order_correlation": geometry_df.iloc[0]["v1_mixture_order_correlation"],
                "v2_mixture_order_correlation": geometry_df.iloc[0]["v2_mixture_order_correlation"],
                "v3_mixture_order_correlation": geometry_df.iloc[0]["v3_mixture_order_correlation"],
            }
        ]
    )
    mixture_df.to_csv(V3_DIR / "mixture_ordering_summary_v3.csv", index=False)

    plot_df = stratified_subset(benchmark_df, PLOT_PER_GROUP, SEED)
    plot_X = decode_intensities(plot_df)
    plot_v3 = build_consensus_v3_embedding(plot_X, source_models)
    plot_coords = tsne_projection(plot_v3)
    plot_projection(plot_coords, plot_df["class_label"].to_numpy(), "v3 embedding t-SNE by class", V3_DIR / "embedding_tsne_by_class_v3.png")
    plot_projection(plot_coords, plot_df["subclass_label"].to_numpy(), "v3 embedding t-SNE by probe", V3_DIR / "embedding_tsne_by_probe_v3.png")

    build_metric_plot(comparison_df, geometry_df, V3_DIR / "v1_vs_v2_vs_v3_metric_comparison.png")

    summary_text = build_summary_text(counts_df, comparison_df, geometry_df, distance_df)
    (V3_DIR / "embedding_summary_v3.txt").write_text(summary_text, encoding="utf-8")

    print(f"Wrote v3 benchmark outputs to: {V3_DIR}")
    print(counts_df.to_string(index=False))
    print("")
    print("Comparison metrics:")
    print(comparison_df.to_string(index=False))
    print("")
    print("Geometry metrics:")
    print(geometry_df.to_string(index=False))


if __name__ == "__main__":
    main()
