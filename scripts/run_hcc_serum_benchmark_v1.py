import json
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATASET_ID = "hcc_serum"
PROCESSING_VERSION = "v1_crop430_1730_interp1_minmax"
SEED = 42


def ensure_output_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import ensure_storage_dirs, resolve_storage_path

    storage_config = ensure_storage_dirs()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))
    if processed_root is None:
        raise ValueError("The storage config is missing processed_data.")

    output_dir = processed_root / "hcc_serum_benchmark_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_processed_dataset(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        df = connection.execute(
            """
            SELECT
                p.processed_id,
                p.biosample_id,
                p.intensity_json,
                m.class_label,
                m.sample_id AS sample_code,
                regexp_extract(m.replicate_id, 'batch-([A-Z])_', 1) AS substrate_batch
            FROM biosample_processed_spectra AS p
            JOIN biosample_metadata AS m
              ON p.biosample_id = m.biosample_id
             AND p.dataset_id = m.dataset_id
            WHERE p.dataset_id = ?
              AND p.processing_version = ?
            ORDER BY p.biosample_id
            """,
            [DATASET_ID, PROCESSING_VERSION],
        ).fetchdf()
    if df.empty:
        raise ValueError("No processed hcc_serum rows were found. Run process_biosample_dataset.py first.")
    return df


def parse_intensity_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.asarray([json.loads(values) for values in df["intensity_json"].tolist()], dtype=float)


def build_models() -> dict[str, Pipeline | LinearDiscriminantAnalysis]:
    return {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=SEED)),
            ]
        ),
        "pca_logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=20, random_state=SEED)),
                ("model", LogisticRegression(max_iter=5000, random_state=SEED)),
            ]
        ),
        "lda": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearDiscriminantAnalysis()),
            ]
        ),
    }


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def evaluate_split(
    X: np.ndarray,
    y: np.ndarray,
    splitter,
    split_name: str,
    groups: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    for model_name, model in build_models().items():
        for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_score = None
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                y_score = scores if scores.ndim == 1 else scores[:, 1]

            metrics = compute_binary_metrics(y_test, y_pred, y_score)
            rows.append(
                {
                    "split_name": split_name,
                    "model_name": model_name,
                    "fold": fold_index,
                    **metrics,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                }
            )
    return pd.DataFrame(rows)


def evaluate_batch_prediction(X: np.ndarray, batches: np.ndarray) -> pd.DataFrame:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=20, random_state=SEED)),
            ("model", LogisticRegression(max_iter=5000, random_state=SEED)),
        ]
    )
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, batches), start=1):
        model.fit(X[train_idx], batches[train_idx])
        pred = model.predict(X[test_idx])
        rows.append(
            {
                "fold": fold_index,
                "accuracy": float(accuracy_score(batches[test_idx], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(batches[test_idx], pred)),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
    return pd.DataFrame(rows)


def build_distribution_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_counts = pd.concat(
        [
            df.groupby("class_label")["sample_code"]
            .nunique()
            .reset_index(name="unique_sample_codes")
            .rename(columns={"class_label": "group_value"})
            .assign(group_type="class"),
            df.groupby("substrate_batch")["sample_code"]
            .nunique()
            .reset_index(name="unique_sample_codes")
            .rename(columns={"substrate_batch": "group_value"})
            .assign(group_type="substrate_batch"),
        ],
        ignore_index=True,
    )
    spectra_counts = pd.concat(
        [
            df.groupby("class_label")
            .size()
            .reset_index(name="n_spectra")
            .rename(columns={"class_label": "group_value"})
            .assign(group_type="class"),
            df.groupby("substrate_batch")
            .size()
            .reset_index(name="n_spectra")
            .rename(columns={"substrate_batch": "group_value"})
            .assign(group_type="substrate_batch"),
        ],
        ignore_index=True,
    )
    return sample_counts, spectra_counts


def compute_geometry_metrics(X: np.ndarray, class_ids: np.ndarray, batch_ids: np.ndarray) -> pd.DataFrame:
    pca = PCA(n_components=min(10, X.shape[1]), random_state=SEED)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    rows = [
        {
            "metric": "class_silhouette_raw_processed",
            "value": float(silhouette_score(X_pca, class_ids)),
        },
        {
            "metric": "batch_silhouette_raw_processed",
            "value": float(silhouette_score(X_pca, batch_ids)),
        },
    ]
    return pd.DataFrame(rows)


def make_scatter_plot(coords: np.ndarray, labels: np.ndarray, title: str, output_path: Path, label_name: str) -> None:
    unique_labels = list(dict.fromkeys(labels.tolist()))
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 5), dpi=300)
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.8,
            label=str(label),
            color=cmap(idx % 10),
            edgecolors="none",
        )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.legend(title=label_name, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_visualizations(X: np.ndarray, class_labels: np.ndarray, batch_labels: np.ndarray, output_dir: Path) -> None:
    X_scaled = StandardScaler().fit_transform(X)
    pca_coords = PCA(n_components=2, random_state=SEED).fit_transform(X_scaled)
    tsne_input = PCA(n_components=min(25, X.shape[1]), random_state=SEED).fit_transform(X_scaled)
    tsne_coords = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    ).fit_transform(tsne_input)

    make_scatter_plot(pca_coords, class_labels, "hcc_serum PCA by class", output_dir / "hcc_serum_pca_by_class.png", "class")
    make_scatter_plot(pca_coords, batch_labels, "hcc_serum PCA by batch", output_dir / "hcc_serum_pca_by_batch.png", "batch")
    make_scatter_plot(tsne_coords, class_labels, "hcc_serum t-SNE by class", output_dir / "hcc_serum_tsne_by_class.png", "class")
    make_scatter_plot(tsne_coords, batch_labels, "hcc_serum t-SNE by batch", output_dir / "hcc_serum_tsne_by_batch.png", "batch")


def write_summary(
    output_path: Path,
    metrics_df: pd.DataFrame,
    batch_diag_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
) -> None:
    mean_metrics = (
        metrics_df.groupby(["split_name", "model_name"])[["accuracy", "balanced_accuracy", "roc_auc"]]
        .mean()
        .reset_index()
    )
    batch_mean = batch_diag_df[["accuracy", "balanced_accuracy"]].mean()
    class_sil = geometry_df.loc[geometry_df["metric"] == "class_silhouette_raw_processed", "value"].iloc[0]
    batch_sil = geometry_df.loc[geometry_df["metric"] == "batch_silhouette_raw_processed", "value"].iloc[0]

    lines = [
        "hcc_serum benchmark v1 summary",
        "",
        "This is a conservative serum benchmark using processed spectra only.",
        "Primary task: CTR vs H0T classification.",
        "Confounder task: substrate_batch prediction and geometry checks.",
        "",
        "Mean model performance by split:",
        mean_metrics.to_string(index=False),
        "",
        "Batch-prediction mean metrics:",
        batch_mean.to_string(),
        "",
        f"Class silhouette: {class_sil:.6f}",
        f"Batch silhouette: {batch_sil:.6f}",
        "",
        "Interpretation guidance:",
        "- If standard CV is materially stronger than leave-one-batch-out, substrate batch may be inflating class performance.",
        "- If batch prediction is strong and batch silhouette stays high, batch remains a serious confounder.",
        "- If grouped-by-sample_code stays strong, the result is less likely to be driven only by repeated sample-code leakage.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"
    output_dir = ensure_output_dir()

    df = load_processed_dataset(db_path)
    X = parse_intensity_matrix(df)
    class_encoder = LabelEncoder()
    y = class_encoder.fit_transform(df["class_label"].to_numpy())
    batch_encoder = LabelEncoder()
    batches = batch_encoder.fit_transform(df["substrate_batch"].to_numpy())
    class_labels = df["class_label"].to_numpy()
    batch_labels = df["substrate_batch"].to_numpy()
    sample_codes = df["sample_code"].astype(str).to_numpy()

    sample_counts_df, spectra_counts_df = build_distribution_tables(df)
    batch_distribution_df = (
        df.groupby(["substrate_batch", "class_label"])
        .size()
        .reset_index(name="n_spectra")
        .sort_values(["substrate_batch", "class_label"])
        .reset_index(drop=True)
    )

    stratified_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    batch_splitter = LeaveOneGroupOut()
    sample_splitter = GroupKFold(n_splits=5)

    metrics_df = pd.concat(
        [
            evaluate_split(X, y, stratified_splitter, "stratified_cv"),
            evaluate_split(X, y, batch_splitter, "leave_one_batch_out", groups=batches),
            evaluate_split(X, y, sample_splitter, "grouped_sample_code", groups=sample_codes),
        ],
        ignore_index=True,
    )

    batch_diag_df = evaluate_batch_prediction(X, batches)
    geometry_df = compute_geometry_metrics(X, y, batches)

    build_visualizations(X, class_labels, batch_labels, output_dir)

    sample_counts_df.to_csv(output_dir / "hcc_serum_sample_counts.csv", index=False)
    spectra_counts_df.to_csv(output_dir / "hcc_serum_spectra_counts.csv", index=False)
    batch_distribution_df.to_csv(output_dir / "hcc_serum_batch_distribution.csv", index=False)
    metrics_df.to_csv(output_dir / "hcc_serum_benchmark_metrics.csv", index=False)
    batch_diag_df.to_csv(output_dir / "hcc_serum_batch_diagnostic_metrics.csv", index=False)
    geometry_df.to_csv(output_dir / "hcc_serum_geometry_metrics.csv", index=False)
    write_summary(output_dir / "hcc_serum_benchmark_summary.txt", metrics_df, batch_diag_df, geometry_df)

    print(f"Benchmark outputs written to: {output_dir}")
    print("\nMean metrics by split/model:")
    print(
        metrics_df.groupby(["split_name", "model_name"])[["accuracy", "balanced_accuracy", "roc_auc"]]
        .mean()
        .reset_index()
        .to_string(index=False)
    )
    print("\nBatch prediction mean metrics:")
    print(batch_diag_df[["accuracy", "balanced_accuracy"]].mean().to_string())
    print("\nGeometry metrics:")
    print(geometry_df.to_string(index=False))


if __name__ == "__main__":
    main()
