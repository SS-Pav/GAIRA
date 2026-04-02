from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from gaira.demo.ev_analysis_utils import (
    DEFAULT_ANCHOR_AUDIT_DIR,
    DEFAULT_CLUSTER_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_GROUNDING_DIR,
    DEFAULT_RUN_DIR,
    STATE_COLORS,
    THEME_COLORS,
    THEME_ORDER,
    cluster_composition_summary,
    cluster_label_enrichment,
    compute_theme_profiles,
    decode_direct_matrix,
    entropy_normalized,
    knn_label_metrics,
    load_common_artifacts,
    load_direct_processed_metadata,
    load_direct_processed_spectra_by_ids,
    normalize_rows,
    reduce_for_plot,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


V5_RUN_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v5_full_true_gpu_run1")
V5_EVAL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_eval_v2/embedding_v5_full_true_gpu_run1_eval_v2")
V6_RUN_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v6_within_type_gpu_run1")
V6_EVAL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_eval_v2/embedding_v6_within_type_gpu_run1_eval_v2")
V7_RUN_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v7_anchor_gpu_run1")
V7_EVAL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_eval_v2/embedding_v7_anchor_gpu_run1_eval_v2")
V7_CLUSTER_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_cluster_analysis_v7")
V7_GROUNDING_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_grounding_analysis_v7")
SMALL2023_V1_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/small2023_ev_invariant_embedding")
SMALL2023_V2_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/small2023_ev_invariant_embedding_v2")
SMALL2023_V3_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/small2023_ev_invariant_embedding_v3")
EV_STRESS_V1_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/ev_stress_disease_analysis_v1")


def maybe_read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def maybe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_eval_metrics(eval_dir: Path) -> pd.DataFrame | None:
    if not eval_dir.exists():
        return None
    for name in ["embedding_metrics_v2.csv", "embedding_metrics.csv"]:
        path = eval_dir / name
        if path.exists():
            return pd.read_csv(path)
    return None


def metric_value(
    metrics: pd.DataFrame | None,
    metric: str,
    *,
    evaluation_tier: str | None = None,
    sample_type_filter: str | None = None,
) -> float:
    if metrics is None or metrics.empty:
        return float("nan")
    subset = metrics[metrics["metric"] == metric].copy()
    if evaluation_tier and "evaluation_tier" in subset.columns:
        subset = subset[subset["evaluation_tier"] == evaluation_tier]
    if sample_type_filter is not None and "sample_type_filter" in subset.columns:
        subset = subset[subset["sample_type_filter"].fillna("") == sample_type_filter]
    if subset.empty:
        return float("nan")
    return float(subset.iloc[0]["value"])


def save_barplot(df: pd.DataFrame, *, x: str, y: str, hue: str | None, title: str, output_path: Path) -> None:
    plt.figure(figsize=(9.2, 5.8))
    ax = sns.barplot(data=df, x=x, y=y, hue=hue, palette="deep")
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    if hue:
        ax.legend(frameon=False, title=hue.replace("_", " "))
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_boxplot(df: pd.DataFrame, *, x: str, y: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8.8, 5.6))
    ax = sns.boxplot(data=df, x=x, y=y, palette="deep")
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_heatmap(df: pd.DataFrame, *, title: str, output_path: Path, cmap: str = "mako") -> None:
    plt.figure(figsize=(9.2, max(4.8, 0.22 * len(df.index))))
    ax = sns.heatmap(df, cmap=cmap)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_scatter(df: pd.DataFrame, *, x: str, y: str, hue: str, style: str | None, size: str | None, title: str, output_path: Path, palette: dict | str = "deep") -> None:
    plt.figure(figsize=(9.0, 6.2))
    kwargs = {"data": df, "x": x, "y": y, "hue": hue, "palette": palette}
    if style:
        kwargs["style"] = style
    if size:
        kwargs["size"] = size
        kwargs["sizes"] = (30, 220)
    ax = sns.scatterplot(**kwargs)
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def load_v7_common() -> dict[str, pd.DataFrame | np.ndarray]:
    return load_common_artifacts(
        run_dir=V7_RUN_DIR,
        eval_dir=V7_EVAL_DIR,
        cluster_dir=V7_CLUSTER_DIR,
        grounding_dir=V7_GROUNDING_DIR,
        anchor_audit_dir=DEFAULT_ANCHOR_AUDIT_DIR,
    )


def broad_top_two(profile: pd.Series) -> tuple[str, str]:
    ordered = profile[THEME_ORDER].sort_values(ascending=False)
    primary = str(ordered.index[0]) if len(ordered) else "unresolved"
    secondary = str(ordered.index[1]) if len(ordered) > 1 and ordered.iloc[1] > 0 else "none"
    return primary, secondary


def composition_coherence(df: pd.DataFrame, *, cluster_col: str) -> pd.DataFrame:
    rows = []
    for cluster_id, group in df.groupby(cluster_col, sort=True):
        profile = group[THEME_ORDER].mean()
        diffs = group[THEME_ORDER].to_numpy(dtype=float) - profile.to_numpy(dtype=float)
        distances = np.linalg.norm(diffs, axis=1)
        rows.append(
            {
                "cluster_id": cluster_id,
                "composition_coherence": float(1.0 / (1.0 + distances.mean())) if len(distances) else float("nan"),
                "within_cluster_composition_variance": float(np.var(distances)) if len(distances) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def balanced_small2023_direct(per_group: int, *, seed: int, processing_version: str) -> pd.DataFrame:
    meta = load_direct_processed_metadata(
        dataset_id="small2023_ev",
        processing_version=processing_version,
        class_filter=["c00", "c01", "c10", "c25", "c50", "c100"],
        subclass_filter=["normedprobe1", "normedprobe2"],
    )
    sampled_meta = (
        meta.groupby(["subclass_label", "class_label"], group_keys=False)
        .apply(lambda group: group.sample(n=min(per_group, len(group)), random_state=seed))
        .reset_index(drop=True)
    )
    direct = load_direct_processed_spectra_by_ids(
        dataset_id="small2023_ev",
        processing_version=processing_version,
        sample_keys=sampled_meta["sample_key"].astype(str).tolist(),
    )
    direct["sample_key"] = direct["sample_key"].astype(str)
    return direct


def sampled_silhouette(values: np.ndarray, labels: np.ndarray, *, seed: int, max_points: int = 2500) -> float:
    labels = pd.Series(labels).fillna("unmapped").astype(str).to_numpy()
    if len(np.unique(labels)) < 2:
        return float("nan")
    if len(values) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(values), size=max_points, replace=False)
        values = values[idx]
        labels = labels[idx]
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(values, labels))


def class_probe_metrics(values: np.ndarray, class_labels: np.ndarray, probe_labels: np.ndarray, *, seed: int, k: int) -> pd.DataFrame:
    class_knn = knn_label_metrics(values, class_labels, k=k)
    probe_knn = knn_label_metrics(values, probe_labels, k=k)
    return pd.DataFrame(
        [
            {"metric": "silhouette_class", "value": sampled_silhouette(values, class_labels, seed=seed)},
            {"metric": "silhouette_probe", "value": sampled_silhouette(values, probe_labels, seed=seed)},
            {"metric": "nn_purity_class", "value": class_knn["nn_purity"]},
            {"metric": "nn_purity_probe", "value": probe_knn["nn_purity"]},
            {"metric": "neighbor_entropy_class", "value": class_knn["neighbor_entropy"]},
            {"metric": "neighbor_entropy_probe", "value": probe_knn["neighbor_entropy"]},
            {"metric": "top1_match_class", "value": class_knn["top1_match"]},
            {"metric": "top1_match_probe", "value": probe_knn["top1_match"]},
        ]
    )
