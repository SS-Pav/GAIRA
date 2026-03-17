from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEEDS = [17, 23, 29, 31, 37]
BASE_SEED = 17
TARGET_DATASET = "small2023_ev"
TARGET_CLASSES = ["c00", "c01", "c10", "c25", "c50", "c100"]
TARGET_PROBES = ["normedprobe1", "normedprobe2"]
ORDER_MAP = {"c00": 0, "c01": 1, "c10": 2, "c25": 3, "c50": 4, "c100": 5}
DOMAIN_MAP = {"normedprobe1": 0, "normedprobe2": 1}
BENCHMARK_PER_GROUP = 2000
SILHOUETTE_PER_GROUP = 250
V1_DIR = Path("/Volumes/SSD_SPG/GAIRA_DATA/processed/small2023_ev_invariant_embedding")
V2_DIR = Path("/Volumes/SSD_SPG/GAIRA_DATA/processed/small2023_ev_invariant_embedding_v2")
VALIDATION_DIR = Path("/Volumes/SSD_SPG/GAIRA_DATA/processed/small2023_ev_invariant_embedding_v2_validation")


def load_processed_dataset(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            """
            SELECT
              p.processed_id,
              p.biosample_id,
              p.dataset_id,
              m.class_label,
              m.subclass_label,
              p.intensity_json
            FROM biosample_processed_spectra p
            JOIN biosample_metadata m
              ON p.biosample_id = m.biosample_id
             AND p.dataset_id = m.dataset_id
            WHERE p.dataset_id = ?
              AND p.processing_version = 'v1_crop670_1800_interp1_minmax'
              AND m.class_label IN ('c00', 'c01', 'c10', 'c25', 'c50', 'c100')
              AND m.subclass_label IN ('normedprobe1', 'normedprobe2')
            ORDER BY m.subclass_label, m.class_label, p.biosample_id
            """,
            [TARGET_DATASET],
        ).fetchdf()
    finally:
        con.close()


def decode_intensities(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["intensity_json"].map(lambda value: np.asarray(json.loads(value), dtype=np.float32)))


def balanced_subset(df: pd.DataFrame, per_group: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = []
    parts = []
    for (probe_label, class_label), group_df in df.groupby(["subclass_label", "class_label"], sort=True):
        take_n = min(per_group, len(group_df))
        sampled = group_df.sample(n=take_n, random_state=seed).sort_values("biosample_id")
        parts.append(sampled)
        counts.append(
            {
                "seed": seed,
                "subclass_label": probe_label,
                "class_label": class_label,
                "n_used": take_n,
            }
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(counts)


def fit_probe_logreg(X: np.ndarray, domain_labels: np.ndarray) -> np.ndarray:
    clf = LogisticRegression(max_iter=400, solver="lbfgs", random_state=BASE_SEED)
    clf.fit(X, domain_labels)
    coef = clf.coef_.astype(np.float32).ravel()
    norm = np.linalg.norm(coef)
    if norm == 0:
        return coef
    return coef / norm


def stage_features(
    X: np.ndarray,
    class_labels: np.ndarray,
    probe_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    global_mean = X_scaled.mean(axis=0)
    probe_means = {probe: X_scaled[probe_labels == probe].mean(axis=0) for probe in TARGET_PROBES}
    alpha = 0.75
    X_mean_aligned = np.empty_like(X_scaled)
    for probe in TARGET_PROBES:
        mask = probe_labels == probe
        X_mean_aligned[mask] = X_scaled[mask] - alpha * (probe_means[probe] - global_mean)

    raw_probe_direction = fit_probe_logreg(X_scaled, np.array([DOMAIN_MAP[p] for p in probe_labels]))
    X_probe_removed_only = X_scaled - np.outer(X_scaled @ raw_probe_direction, raw_probe_direction)

    aligned_probe_direction = fit_probe_logreg(
        X_mean_aligned, np.array([DOMAIN_MAP[p] for p in probe_labels])
    )
    X_domain_reduced = X_mean_aligned - np.outer(X_mean_aligned @ aligned_probe_direction, aligned_probe_direction)

    lda = LinearDiscriminantAnalysis(n_components=5)
    X_lda_only = lda.fit_transform(X_domain_reduced, class_labels).astype(np.float32)

    order_targets = np.array([ORDER_MAP[label] for label in class_labels], dtype=np.float32)
    ridge = Ridge(alpha=1.0, random_state=BASE_SEED)
    ridge.fit(X_domain_reduced, order_targets)
    X_ridge_only = ridge.predict(X_domain_reduced).reshape(-1, 1).astype(np.float32)

    X_full_v2 = np.hstack([X_lda_only, X_ridge_only]).astype(np.float32)
    return {
        "raw_standardized": X_scaled,
        "mean_alignment_only": X_mean_aligned,
        "probe_direction_removal_only": X_probe_removed_only,
        "lda_only": X_lda_only,
        "ordinal_ridge_only": X_ridge_only,
        "full_v2": X_full_v2,
    }


def cross_probe_transfer_metrics(features: np.ndarray, class_labels: np.ndarray, probe_labels: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for train_probe, test_probe in [("normedprobe1", "normedprobe2"), ("normedprobe2", "normedprobe1")]:
        train_mask = probe_labels == train_probe
        test_mask = probe_labels == test_probe
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_mask])
        X_test = scaler.transform(features[test_mask])
        clf = LogisticRegression(max_iter=400, solver="lbfgs", random_state=BASE_SEED)
        clf.fit(X_train, class_labels[train_mask])
        predictions = clf.predict(X_test)
        rows.append(
            {
                "direction": f"{train_probe}->{test_probe}",
                "accuracy": accuracy_score(class_labels[test_mask], predictions),
                "balanced_accuracy": balanced_accuracy_score(class_labels[test_mask], predictions),
                "macro_f1": f1_score(class_labels[test_mask], predictions, average="macro"),
            }
        )
    return rows


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


def silhouette_subset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    parts = []
    for (_probe, _class_label), group_df in df.groupby(["subclass_label", "class_label"], sort=True):
        take_n = min(SILHOUETTE_PER_GROUP, len(group_df))
        parts.append(group_df.sample(n=take_n, random_state=seed).sort_values("biosample_id"))
    return pd.concat(parts, ignore_index=True)


def exact_duplicate_audit(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    rows = []
    for class_label in TARGET_CLASSES:
        class_mask = df["class_label"].to_numpy() == class_label
        for probe in TARGET_PROBES:
            pass
        probe1 = X[class_mask & (df["subclass_label"].to_numpy() == "normedprobe1")]
        probe2 = X[class_mask & (df["subclass_label"].to_numpy() == "normedprobe2")]
        probe1_hashes = [hashlib.sha1(arr.tobytes()).hexdigest() for arr in probe1]
        probe2_hashes = [hashlib.sha1(arr.tobytes()).hexdigest() for arr in probe2]
        counts1 = pd.Series(probe1_hashes).value_counts()
        counts2 = pd.Series(probe2_hashes).value_counts()
        shared = sorted(set(counts1.index).intersection(counts2.index))
        pair_count = int(sum(min(int(counts1[h]), int(counts2[h])) for h in shared))
        rows.append(
            {
                "class_label": class_label,
                "probe1_count": len(probe1),
                "probe2_count": len(probe2),
                "shared_unique_fingerprints": len(shared),
                "exact_cross_probe_duplicate_pairs": pair_count,
            }
        )
    summary = pd.DataFrame(rows)
    overall = pd.DataFrame(
        [
            {
                "class_label": "overall",
                "probe1_count": int((df["subclass_label"] == "normedprobe1").sum()),
                "probe2_count": int((df["subclass_label"] == "normedprobe2").sum()),
                "shared_unique_fingerprints": int(summary["shared_unique_fingerprints"].sum()),
                "exact_cross_probe_duplicate_pairs": int(summary["exact_cross_probe_duplicate_pairs"].sum()),
            }
        ]
    )
    return pd.concat([summary, overall], ignore_index=True)


def nearest_neighbor_audit(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    rows = []
    subclass_labels = df["subclass_label"].to_numpy()
    class_labels = df["class_label"].to_numpy()
    for class_label in TARGET_CLASSES:
        probe1 = X[(class_labels == class_label) & (subclass_labels == "normedprobe1")]
        probe2 = X[(class_labels == class_label) & (subclass_labels == "normedprobe2")]
        for metric in ["cosine", "euclidean"]:
            nn_12 = NearestNeighbors(n_neighbors=1, metric=metric).fit(probe2)
            d12, _ = nn_12.kneighbors(probe1, return_distance=True)
            nn_21 = NearestNeighbors(n_neighbors=1, metric=metric).fit(probe1)
            d21, _ = nn_21.kneighbors(probe2, return_distance=True)
            for direction, distances in [
                ("normedprobe1->normedprobe2", d12.ravel()),
                ("normedprobe2->normedprobe1", d21.ravel()),
            ]:
                rows.append(
                    {
                        "class_label": class_label,
                        "direction": direction,
                        "metric": metric,
                        "mean_distance": float(np.mean(distances)),
                        "median_distance": float(np.median(distances)),
                        "min_distance": float(np.min(distances)),
                        "max_distance": float(np.max(distances)),
                        "p95_distance": float(np.quantile(distances, 0.95)),
                        "count_distance_le_1e-6": int(np.sum(distances <= 1e-6)),
                        "count_distance_le_1e-4": int(np.sum(distances <= 1e-4)),
                        "count_distance_le_1e-3": int(np.sum(distances <= 1e-3)),
                    }
                )
    return pd.DataFrame(rows)


def multi_seed_transfer(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        benchmark_df, _counts_df = balanced_subset(df, BENCHMARK_PER_GROUP, seed)
        X = decode_intensities(benchmark_df)
        class_labels = benchmark_df["class_label"].to_numpy()
        probe_labels = benchmark_df["subclass_label"].to_numpy()
        features = stage_features(X, class_labels, probe_labels)
        for model_name, model_features in [
            ("raw_standardized", features["raw_standardized"]),
            ("full_v2", features["full_v2"]),
        ]:
            for metric_row in cross_probe_transfer_metrics(model_features, class_labels, probe_labels):
                rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        **metric_row,
                    }
                )
    seed_df = pd.DataFrame(rows)
    summary_rows = []
    for (model_name, direction), group_df in seed_df.groupby(["model", "direction"], sort=True):
        summary_rows.append(
            {
                "model": model_name,
                "direction": direction,
                "accuracy_mean": float(group_df["accuracy"].mean()),
                "accuracy_std": float(group_df["accuracy"].std(ddof=0)),
                "balanced_accuracy_mean": float(group_df["balanced_accuracy"].mean()),
                "balanced_accuracy_std": float(group_df["balanced_accuracy"].std(ddof=0)),
                "macro_f1_mean": float(group_df["macro_f1"].mean()),
                "macro_f1_std": float(group_df["macro_f1"].std(ddof=0)),
            }
        )
    return pd.DataFrame(summary_rows)


def ablation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    benchmark_df, _counts_df = balanced_subset(df, BENCHMARK_PER_GROUP, BASE_SEED)
    X = decode_intensities(benchmark_df)
    class_labels = benchmark_df["class_label"].to_numpy()
    probe_labels = benchmark_df["subclass_label"].to_numpy()
    sil_df = silhouette_subset(benchmark_df, BASE_SEED)
    sil_X = decode_intensities(sil_df)
    sil_class = sil_df["class_label"].to_numpy()
    sil_probe = sil_df["subclass_label"].to_numpy()

    features = stage_features(X, class_labels, probe_labels)
    sil_features = stage_features(sil_X, sil_class, sil_probe)

    rows = []
    for stage_name in [
        "raw_standardized",
        "mean_alignment_only",
        "probe_direction_removal_only",
        "lda_only",
        "ordinal_ridge_only",
        "full_v2",
    ]:
        transfer_rows = cross_probe_transfer_metrics(features[stage_name], class_labels, probe_labels)
        metrics = {row["direction"]: row for row in transfer_rows}
        rows.append(
            {
                "stage": stage_name,
                "probe1_to_probe2_accuracy": metrics["normedprobe1->normedprobe2"]["accuracy"],
                "probe2_to_probe1_accuracy": metrics["normedprobe2->normedprobe1"]["accuracy"],
                "class_separability": float(silhouette_score(sil_features[stage_name], sil_class)),
                "probe_separability": float(silhouette_score(sil_features[stage_name], sil_probe)),
                "mixture_order_correlation": mixture_order_correlation(features[stage_name], class_labels),
            }
        )
    return pd.DataFrame(rows)


def plot_ablation(ablation_df: pd.DataFrame, output_path: Path) -> None:
    labels = ablation_df["stage"].tolist()
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(x - width / 2, ablation_df["probe1_to_probe2_accuracy"], width=width, label="Probe1->Probe2")
    axes[0].bar(x + width / 2, ablation_df["probe2_to_probe1_accuracy"], width=width, label="Probe2->Probe1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Ablation transfer accuracy")
    axes[0].legend(frameon=False)

    axes[1].plot(x, ablation_df["class_separability"], marker="o", label="Class separability")
    axes[1].plot(x, ablation_df["probe_separability"], marker="o", label="Probe separability")
    axes[1].plot(x, ablation_df["mixture_order_correlation"], marker="o", label="Mixture ordering")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_title("Ablation geometry summary")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_multi_seed(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_df = summary_df.copy()
    plot_df["label"] = plot_df["model"] + " " + plot_df["direction"]
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["accuracy_mean"], yerr=plot_df["accuracy_std"], capsize=4, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Multi-seed transfer robustness")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_summary(
    duplicate_df: pd.DataFrame,
    nn_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
) -> str:
    dup_overall = duplicate_df[duplicate_df["class_label"] == "overall"].iloc[0]
    seed_lines = []
    for _idx, row in seed_df.iterrows():
        seed_lines.append(
            f"- {row['model']} {row['direction']}: "
            f"{row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}"
        )

    best_stage = ablation_df.sort_values("probe1_to_probe2_accuracy", ascending=False).iloc[0]
    nn_cosine = nn_df[nn_df["metric"] == "cosine"]
    suspicious_cosine = int((nn_cosine["min_distance"] <= 1e-6).sum())

    lines = [
        "small2023_ev invariant embedding v2 validation",
        "",
        "Duplicate audit:",
        f"- Overall exact cross-probe duplicate pairs: {int(dup_overall['exact_cross_probe_duplicate_pairs'])}",
        f"- Overall shared unique fingerprints: {int(dup_overall['shared_unique_fingerprints'])}",
        f"- Cosine nearest-neighbor rows with min distance <= 1e-6: {suspicious_cosine}",
        "",
        "Multi-seed transfer summary:",
        *seed_lines,
        "",
        "Ablation summary:",
        ablation_df.to_string(index=False),
        "",
        f"Best Probe1->Probe2 ablation stage: {best_stage['stage']} ({best_stage['probe1_to_probe2_accuracy']:.4f})",
        "",
        "Interpretation:",
        "- No exact duplicate leakage is supported if duplicate pair count is zero.",
        "- Strong robustness across seeds supports stability rather than a fragile sampling artifact.",
        "- The benchmark remains optimistic because the embedding transform is fit jointly on both probe domains with class supervision before the transfer classifier.",
        "- That is not the same as train-only leakage, but it is still a transductive benchmark rather than a strict deployment protocol.",
    ]
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import ensure_storage_dirs, get_storage_config, resolve_storage_path

    ensure_storage_dirs()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    storage_config = get_storage_config()
    db_path = resolve_storage_path(storage_config.get("database"))
    if db_path is None:
        raise ValueError("Database path is not configured in config/storage.yaml")
    full_df = load_processed_dataset(db_path)
    benchmark_df, counts_df = balanced_subset(full_df, BENCHMARK_PER_GROUP, BASE_SEED)
    X = decode_intensities(benchmark_df)

    duplicate_df = exact_duplicate_audit(benchmark_df, X)
    duplicate_df.to_csv(VALIDATION_DIR / "duplicate_audit_summary.csv", index=False)

    nn_df = nearest_neighbor_audit(benchmark_df, X)
    nn_df.to_csv(VALIDATION_DIR / "cross_probe_nearest_neighbor_summary.csv", index=False)

    seed_df = multi_seed_transfer(full_df)
    seed_df.to_csv(VALIDATION_DIR / "multi_seed_transfer_summary.csv", index=False)

    ablation_df = ablation_metrics(full_df)
    ablation_df.to_csv(VALIDATION_DIR / "v2_ablation_metrics.csv", index=False)

    counts_df.to_csv(VALIDATION_DIR / "benchmark_sample_counts_validation.csv", index=False)
    plot_ablation(ablation_df, VALIDATION_DIR / "v2_ablation_plot.png")
    plot_multi_seed(seed_df, VALIDATION_DIR / "multi_seed_transfer_plot.png")

    summary_text = build_summary(duplicate_df, nn_df, seed_df, ablation_df)
    (VALIDATION_DIR / "v2_validation_summary.txt").write_text(summary_text, encoding="utf-8")

    print(f"Wrote validation outputs to: {VALIDATION_DIR}")
    print(duplicate_df.to_string(index=False))
    print()
    print(seed_df.to_string(index=False))
    print()
    print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    main()
