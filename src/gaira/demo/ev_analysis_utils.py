from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from gaira.config import get_database_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RUN_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v7_anchor_gpu_run1")
DEFAULT_EVAL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_eval_v2/embedding_v7_anchor_gpu_run1_eval_v2")
DEFAULT_CLUSTER_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_cluster_analysis_v7")
DEFAULT_GROUNDING_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_grounding_analysis_v7")
DEFAULT_ANCHOR_AUDIT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_anchor_audit")

THEME_ORDER = [
    "protein_peptide_associated",
    "purine_metabolite_associated",
    "oxidative_redox_associated",
    "lipid_membrane_associated",
    "carbohydrate_associated",
    "nucleic_acid_associated",
    "serum_matrix_associated",
]
THEME_COLORS = {
    "protein_peptide_associated": "#3e6ea1",
    "purine_metabolite_associated": "#cc7b37",
    "oxidative_redox_associated": "#7a4b9d",
    "lipid_membrane_associated": "#2c8c69",
    "carbohydrate_associated": "#c24d67",
    "nucleic_acid_associated": "#8d5b2a",
    "serum_matrix_associated": "#8f9499",
    "unresolved": "#b6bcc4",
}
STATE_COLORS = {
    "control_like": "#3e6ea1",
    "stress_or_toxicity_like": "#c24d67",
    "intermediate_or_ambiguous": "#c8a04b",
    "unmapped": "#8f9499",
}


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def load_common_artifacts(
    *,
    run_dir: Path = DEFAULT_RUN_DIR,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    cluster_dir: Path = DEFAULT_CLUSTER_DIR,
    grounding_dir: Path = DEFAULT_GROUNDING_DIR,
    anchor_audit_dir: Path = DEFAULT_ANCHOR_AUDIT_DIR,
) -> dict[str, pd.DataFrame | np.ndarray]:
    required = [
        run_dir / "embeddings.npy",
        run_dir / "metadata.csv",
        eval_dir / "embedding_projection_v2.csv",
        cluster_dir / "cluster_assignments.csv",
        cluster_dir / "cluster_summary.csv",
        grounding_dir / "ev_cluster_interpretation_table.csv",
        grounding_dir / "ev_cluster_theme_scores.csv",
        grounding_dir / "ev_cluster_grounding_hits.csv",
        grounding_dir / "grounding_theme_table.csv",
        anchor_audit_dir / "embedding_anchor_table_v1.csv",
    ]
    for path in required:
        ensure_exists(path)

    artifacts: dict[str, pd.DataFrame | np.ndarray] = {
        "embeddings": np.load(run_dir / "embeddings.npy"),
        "metadata": pd.read_csv(run_dir / "metadata.csv"),
        "projection": pd.read_csv(eval_dir / "embedding_projection_v2.csv"),
        "cluster_assignments": pd.read_csv(cluster_dir / "cluster_assignments.csv"),
        "cluster_summary": pd.read_csv(cluster_dir / "cluster_summary.csv"),
        "ev_cluster_interpretation": pd.read_csv(grounding_dir / "ev_cluster_interpretation_table.csv"),
        "ev_cluster_theme_scores": pd.read_csv(grounding_dir / "ev_cluster_theme_scores.csv"),
        "ev_cluster_grounding_hits": pd.read_csv(grounding_dir / "ev_cluster_grounding_hits.csv"),
        "grounding_theme_table": pd.read_csv(grounding_dir / "grounding_theme_table.csv"),
        "anchor_table": pd.read_csv(anchor_audit_dir / "embedding_anchor_table_v1.csv"),
    }
    return artifacts


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return values / norms


def entropy_normalized(labels: pd.Series | np.ndarray) -> float:
    values = pd.Series(labels).fillna("").astype(str)
    values = values[values != ""]
    if values.empty:
        return 0.0
    probs = values.value_counts(normalize=True)
    entropy = float(-(probs * np.log2(probs)).sum())
    max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def knn_label_metrics(values: np.ndarray, labels: np.ndarray, k: int = 6) -> dict[str, float]:
    labels = np.asarray(labels, dtype=object)
    valid_mask = pd.Series(labels).fillna("").astype(str).to_numpy() != ""
    if valid_mask.sum() <= k:
        return {"nn_purity": float("nan"), "neighbor_entropy": float("nan"), "top1_match": float("nan")}

    X = values[valid_mask]
    y = labels[valid_mask]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="cosine", algorithm="brute")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_indices = indices[:, 1:]
    neighbor_labels = y[neighbor_indices]
    purity = float((neighbor_labels == y[:, None]).mean())
    top1 = float((neighbor_labels[:, 0] == y).mean()) if neighbor_labels.shape[1] > 0 else float("nan")
    entropies = []
    for row in neighbor_labels:
        entropies.append(entropy_normalized(row))
    return {
        "nn_purity": purity,
        "neighbor_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "top1_match": top1,
    }


def sampled_global_metrics(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 7,
    max_points: int = 5000,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=object)
    valid_mask = pd.Series(labels).fillna("").astype(str).to_numpy() != ""
    X = values[valid_mask]
    y = labels[valid_mask]
    if len(np.unique(y)) < 2 or len(X) < 20:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }
    if len(X) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X), size=max_points, replace=False)
        X = X[indices]
        y = y[indices]
    return {
        "silhouette": float(silhouette_score(X, y)),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
        "calinski_harabasz": float(calinski_harabasz_score(X, y)),
    }


def cluster_label_enrichment(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    label_col: str,
) -> pd.DataFrame:
    working = df[[cluster_col, label_col]].copy()
    working[label_col] = working[label_col].fillna("").astype(str)
    working = working[working[label_col] != ""]
    if working.empty:
        return pd.DataFrame(columns=[cluster_col, label_col, "count", "cluster_fraction", "global_fraction", "log2_odds"])

    global_fraction = working[label_col].value_counts(normalize=True).to_dict()
    rows = []
    for cluster_id, group in working.groupby(cluster_col, sort=True):
        counts = group[label_col].value_counts()
        total = counts.sum()
        for label, count in counts.items():
            cluster_fraction = count / total
            baseline = max(global_fraction.get(label, 1e-6), 1e-6)
            odds = cluster_fraction / baseline
            rows.append(
                {
                    cluster_col: cluster_id,
                    label_col: label,
                    "count": int(count),
                    "cluster_fraction": float(cluster_fraction),
                    "global_fraction": float(baseline),
                    "log2_odds": float(np.log2(max(odds, 1e-6))),
                }
            )
    return pd.DataFrame(rows).sort_values([cluster_col, "log2_odds"], ascending=[True, False]).reset_index(drop=True)


def align_to_master_grid(x_values: np.ndarray, y_values: np.ndarray, master_x: np.ndarray) -> np.ndarray:
    order = np.argsort(x_values)
    return np.interp(master_x, x_values[order], y_values[order], left=0.0, right=0.0).astype(np.float32)


def load_direct_processed_spectra(
    *,
    dataset_id: str,
    processing_version: str,
    class_filter: list[str] | None = None,
    subclass_filter: list[str] | None = None,
) -> pd.DataFrame:
    db_path = get_database_path()
    query = """
        SELECT
          p.processed_id AS sample_key,
          p.dataset_id,
          p.processing_version,
          p.wavenumbers_json,
          p.intensity_json,
          m.class_label,
          m.subclass_label,
          m.source_file
        FROM biosample_processed_spectra p
        JOIN biosample_metadata m
          ON p.biosample_id = m.biosample_id
         AND p.dataset_id = m.dataset_id
        WHERE p.dataset_id = ?
          AND p.processing_version = ?
    """
    params: list[object] = [dataset_id, processing_version]
    if class_filter:
        query += " AND m.class_label IN (" + ",".join(["?"] * len(class_filter)) + ")"
        params.extend(class_filter)
    if subclass_filter:
        query += " AND m.subclass_label IN (" + ",".join(["?"] * len(subclass_filter)) + ")"
        params.extend(subclass_filter)
    query += " ORDER BY m.subclass_label, m.class_label, p.biosample_id"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        df = connection.execute(query, params).fetchdf()
    if df.empty:
        raise RuntimeError(f"No processed spectra found for {dataset_id} {processing_version}.")
    return df


def load_direct_processed_metadata(
    *,
    dataset_id: str,
    processing_version: str,
    class_filter: list[str] | None = None,
    subclass_filter: list[str] | None = None,
) -> pd.DataFrame:
    db_path = get_database_path()
    query = """
        SELECT
          p.processed_id AS sample_key,
          p.dataset_id,
          p.processing_version,
          m.class_label,
          m.subclass_label,
          m.source_file
        FROM biosample_processed_spectra p
        JOIN biosample_metadata m
          ON p.biosample_id = m.biosample_id
         AND p.dataset_id = m.dataset_id
        WHERE p.dataset_id = ?
          AND p.processing_version = ?
    """
    params: list[object] = [dataset_id, processing_version]
    if class_filter:
        query += " AND m.class_label IN (" + ",".join(["?"] * len(class_filter)) + ")"
        params.extend(class_filter)
    if subclass_filter:
        query += " AND m.subclass_label IN (" + ",".join(["?"] * len(subclass_filter)) + ")"
        params.extend(subclass_filter)
    query += " ORDER BY m.subclass_label, m.class_label, p.biosample_id"
    with duckdb.connect(str(db_path), read_only=True) as connection:
        df = connection.execute(query, params).fetchdf()
    if df.empty:
        raise RuntimeError(f"No processed metadata found for {dataset_id} {processing_version}.")
    return df


def load_direct_processed_spectra_by_ids(
    *,
    dataset_id: str,
    processing_version: str,
    sample_keys: list[str],
) -> pd.DataFrame:
    if not sample_keys:
        raise ValueError("sample_keys must not be empty")
    db_path = get_database_path()
    placeholders = ",".join(["?"] * len(sample_keys))
    query = f"""
        SELECT
          p.processed_id AS sample_key,
          p.dataset_id,
          p.processing_version,
          p.wavenumbers_json,
          p.intensity_json,
          m.class_label,
          m.subclass_label,
          m.source_file
        FROM biosample_processed_spectra p
        JOIN biosample_metadata m
          ON p.biosample_id = m.biosample_id
         AND p.dataset_id = m.dataset_id
        WHERE p.dataset_id = ?
          AND p.processing_version = ?
          AND p.processed_id IN ({placeholders})
    """
    params: list[object] = [dataset_id, processing_version, *sample_keys]
    with duckdb.connect(str(db_path), read_only=True) as connection:
        df = connection.execute(query, params).fetchdf()
    if df.empty:
        raise RuntimeError(f"No processed spectra found for selected IDs in {dataset_id} {processing_version}.")
    return df


def decode_direct_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_arrays = [np.asarray(json.loads(value), dtype=float) for value in df["wavenumbers_json"]]
    y_arrays = [np.asarray(json.loads(value), dtype=float) for value in df["intensity_json"]]
    all_x = np.unique(np.concatenate(x_arrays))
    matrix = np.vstack([align_to_master_grid(x, y, all_x) for x, y in zip(x_arrays, y_arrays, strict=False)])
    return all_x.astype(np.float32), matrix.astype(np.float32)


def balanced_sample(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    per_group: int,
    seed: int = 7,
) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby(group_cols, sort=True):
        n = min(per_group, len(group))
        parts.append(group.sample(n=n, random_state=seed).sort_values(group_cols))
    return pd.concat(parts, ignore_index=True)


def reduce_for_plot(values: np.ndarray, *, seed: int = 7) -> np.ndarray:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    n_components = min(20, scaled.shape[1], max(4, scaled.shape[1] // 2))
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(scaled)
    coords = PCA(n_components=2, random_state=seed).fit_transform(reduced)
    return coords.astype(np.float32)


def compute_theme_profiles(
    query_embeddings: np.ndarray,
    grounding_embeddings: np.ndarray,
    grounding_themes: np.ndarray,
    *,
    top_k: int = 12,
    chunk_size: int = 4096,
) -> pd.DataFrame:
    query_norm = normalize_rows(query_embeddings.astype(np.float32))
    grounding_norm = normalize_rows(grounding_embeddings.astype(np.float32))
    theme_index = {theme: idx for idx, theme in enumerate(THEME_ORDER)}
    rows = []
    for start in range(0, len(query_norm), chunk_size):
        stop = min(start + chunk_size, len(query_norm))
        scores = query_norm[start:stop] @ grounding_norm.T
        top_idx = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(top_scores, axis=1)[:, ::-1]
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        for row_idx in range(stop - start):
            theme_scores = np.zeros(len(THEME_ORDER), dtype=float)
            for idx, score in zip(top_idx[row_idx], top_scores[row_idx], strict=False):
                theme = str(grounding_themes[idx])
                if theme not in theme_index:
                    continue
                weight = max(float(score), 0.0)
                theme_scores[theme_index[theme]] += weight
            total = theme_scores.sum()
            if total > 0:
                theme_scores = theme_scores / total
            record = {theme: float(theme_scores[i]) for theme, i in theme_index.items()}
            if total > 0:
                probs = theme_scores[theme_scores > 0]
                entropy = float(-(probs * np.log2(probs)).sum())
                max_entropy = math.log2(len(THEME_ORDER)) if len(THEME_ORDER) > 1 else 1.0
                record["profile_entropy"] = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                record["profile_entropy"] = 0.0
            rows.append(record)
    return pd.DataFrame(rows)


def cluster_composition_summary(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    theme_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if theme_cols is None:
        theme_cols = THEME_ORDER
    grouped = df.groupby(cluster_col, sort=True)
    profiles = grouped[theme_cols].mean().reset_index()
    metrics = []
    for cluster_id, group in grouped:
        mean_profile = group[theme_cols].mean().to_numpy(dtype=float)
        distances = np.linalg.norm(group[theme_cols].to_numpy(dtype=float) - mean_profile, axis=1)
        top_order = mean_profile.argsort()[::-1]
        dominant = theme_cols[top_order[0]] if len(top_order) else "unresolved"
        secondary = theme_cols[top_order[1]] if len(top_order) > 1 and mean_profile[top_order[1]] > 0 else "none"
        top_share = float(mean_profile[top_order[0]]) if len(top_order) else 0.0
        metrics.append(
            {
                cluster_col: cluster_id,
                "cluster_size": int(len(group)),
                "dominant_theme": dominant,
                "secondary_theme": secondary,
                "top_theme_share": top_share,
                "composition_coherence": float(1.0 / (1.0 + distances.mean())) if len(distances) else float("nan"),
                "within_cluster_composition_variance": float(np.var(distances)) if len(distances) else float("nan"),
            }
        )
    return profiles, pd.DataFrame(metrics)


def save_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color_col: str,
    output_path: Path,
    title: str,
    color_map: dict[str, str] | None = None,
    size_col: str | None = None,
    marker_col: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    color_values = df[color_col].fillna("unresolved").astype(str)
    unique_colors = list(dict.fromkeys(color_values.tolist()))
    default_palette = plt.get_cmap("tab10")
    for idx, label in enumerate(unique_colors):
        mask = color_values == label
        kwargs = {
            "s": 28,
            "alpha": 0.78,
            "color": (color_map or {}).get(label, default_palette(idx % default_palette.N)),
            "label": label.replace("_", " "),
        }
        if size_col:
            kwargs["s"] = np.clip(df.loc[mask, size_col].to_numpy(dtype=float) / 10.0, 24, 180)
        marker = "o"
        if marker_col:
            marker_value = str(df.loc[mask, marker_col].iloc[0])
            marker = "D" if marker_value == "mixed" else "o"
        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col], marker=marker, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(fontsize=8, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_heatmap(
    values: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8.8, 6.6),
    x_rotation: int = 35,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    matrix = values.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels([str(col).replace("_", "\n") for col in values.columns], rotation=x_rotation, ha="right")
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(values.index.astype(str).tolist())
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("normalized weight", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
