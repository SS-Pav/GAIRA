from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gaira.demo.ev_analysis_utils import decode_direct_matrix, load_direct_processed_spectra_by_ids, load_direct_processed_metadata, normalize_rows
from gaira.demo.v8_analysis_utils import SMALL2023_V2_DIR, V5_RUN_DIR, V7_RUN_DIR, class_probe_metrics, reduce_for_plot


DIRECT_PROCESSING_VERSION = "v2_crop670_1800_interp1_poly3_vector"
MODE_LABELS = {
    "cellline": ["Hec", "Hela", "Ht", "Mef", "Thp"],
    "mixture": ["c00", "c01", "c10", "c25", "c50", "c100"],
}


def mode_labels(mode: str) -> list[str]:
    if mode not in MODE_LABELS:
        raise ValueError(f"Unsupported small2023 mode: {mode}")
    return MODE_LABELS[mode]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_run_embeddings(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = pd.read_csv(run_dir / "metadata.csv")
    metadata["sample_key"] = metadata["sample_key"].astype(str)
    return metadata, np.load(run_dir / "embeddings.npy")


def load_branch_run(run_dir: Path, *, mode: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    metadata, embeddings = load_run_embeddings(run_dir)
    labels = set(mode_labels(mode))
    metadata = metadata[(metadata["dataset_id"] == "small2023_ev") & (metadata["branch_primary_label"].fillna("").isin(labels))].copy()
    metadata = metadata.reset_index(drop=False).rename(columns={"index": "source_index"})
    subset_embeddings = normalize_rows(embeddings[metadata["source_index"].to_numpy()])
    return metadata.reset_index(drop=True), subset_embeddings, read_json(run_dir / "run_config.json")


def load_direct_mode(mode: str, *, sample_keys: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = load_direct_processed_metadata(
        dataset_id="small2023_ev",
        processing_version=DIRECT_PROCESSING_VERSION,
        class_filter=mode_labels(mode),
    )
    metadata["sample_key"] = metadata["sample_key"].astype(str)
    if sample_keys is not None:
        metadata = metadata[metadata["sample_key"].isin(sample_keys)].copy()
    spectra = load_direct_processed_spectra_by_ids(
        dataset_id="small2023_ev",
        processing_version=DIRECT_PROCESSING_VERSION,
        sample_keys=metadata["sample_key"].astype(str).tolist(),
    )
    spectra["sample_key"] = spectra["sample_key"].astype(str)
    _, X = decode_direct_matrix(spectra)
    return spectra.reset_index(drop=True), StandardScaler().fit_transform(X)


def align_representation(
    metadata: pd.DataFrame,
    values: np.ndarray,
    sample_keys: list[str],
    *,
    label_col: str,
    probe_col: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    working = metadata.copy()
    working["sample_key"] = working["sample_key"].astype(str)
    sample_set = set(sample_keys)
    working = working[working["sample_key"].isin(sample_set)].copy()
    working = working.drop_duplicates("sample_key")
    order = pd.DataFrame({"sample_key": sample_keys})
    working = order.merge(working, on="sample_key", how="inner")
    idx = metadata.set_index(metadata["sample_key"].astype(str)).loc[working["sample_key"].astype(str), metadata.columns[0]].to_numpy() if metadata.columns[0] == "source_index" else None
    if idx is None:
        idx = metadata.reset_index().set_index(metadata["sample_key"].astype(str)).loc[working["sample_key"].astype(str), "index"].to_numpy()
    aligned = values[idx]
    working["class_label"] = working[label_col].astype(str)
    working["probe_label"] = working[probe_col].astype(str)
    return working.reset_index(drop=True), aligned


def _balanced_limit_indices(labels: pd.Series, *, max_samples: int, seed: int) -> np.ndarray:
    if len(labels) <= max_samples:
        return np.arange(len(labels))
    groups = labels.groupby(labels)
    per_group = max(1, max_samples // max(groups.ngroups, 1))
    sampled = groups.apply(lambda group: group.sample(n=min(per_group, len(group)), random_state=seed)).index.get_level_values(1).to_numpy()
    sampled = np.unique(sampled)
    if len(sampled) >= max_samples:
        return sampled[:max_samples]
    remaining = np.setdiff1d(np.arange(len(labels)), sampled, assume_unique=False)
    if len(remaining) > 0:
        rng = np.random.default_rng(seed)
        extra = rng.choice(remaining, size=min(max_samples - len(sampled), len(remaining)), replace=False)
        sampled = np.concatenate([sampled, extra])
    return np.sort(sampled)


def classifier_metrics(values: np.ndarray, labels: np.ndarray, *, seed: int, max_samples: int = 24000) -> dict[str, float]:
    labels = pd.Series(labels).fillna("").astype(str)
    valid = labels != ""
    labels = labels[valid]
    X = values[valid.to_numpy()]
    if labels.nunique() < 2 or len(labels) < 20:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    keep = _balanced_limit_indices(labels.reset_index(drop=True), max_samples=max_samples, seed=seed)
    labels = labels.reset_index(drop=True).iloc[keep]
    X = X[keep]
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        labels.to_numpy(),
        test_size=0.25,
        random_state=seed,
        stratify=labels.to_numpy(),
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
    }


def cross_probe_transfer_metrics(values: np.ndarray, class_labels: np.ndarray, probe_labels: np.ndarray, *, max_samples_per_probe: int = 12000, seed: int = 7) -> pd.DataFrame:
    labels = pd.Series(class_labels).fillna("").astype(str).to_numpy()
    probes = pd.Series(probe_labels).fillna("").astype(str).to_numpy()
    unique_probes = sorted([probe for probe in pd.unique(probes) if probe])
    if len(unique_probes) < 2:
        return pd.DataFrame(
            [{"direction": "not_available", "train_probe": "", "test_probe": "", "accuracy": float("nan"), "macro_f1": float("nan")}]
        )
    rows = []
    for train_probe, test_probe in [(unique_probes[0], unique_probes[1]), (unique_probes[1], unique_probes[0])]:
        train_mask = probes == train_probe
        test_mask = probes == test_probe
        train_classes = set(labels[train_mask])
        test_classes = set(labels[test_mask])
        shared_classes = sorted([label for label in train_classes & test_classes if label])
        if len(shared_classes) < 2:
            rows.append({"direction": f"{train_probe}_to_{test_probe}", "train_probe": train_probe, "test_probe": test_probe, "accuracy": float("nan"), "macro_f1": float("nan")})
            continue
        train_keep = train_mask & np.isin(labels, shared_classes)
        test_keep = test_mask & np.isin(labels, shared_classes)
        train_idx = np.where(train_keep)[0]
        test_idx = np.where(test_keep)[0]
        if len(train_idx) > max_samples_per_probe:
            rng = np.random.default_rng(seed)
            train_idx = np.sort(rng.choice(train_idx, size=max_samples_per_probe, replace=False))
        if len(test_idx) > max_samples_per_probe:
            rng = np.random.default_rng(seed + 1)
            test_idx = np.sort(rng.choice(test_idx, size=max_samples_per_probe, replace=False))
        clf = LogisticRegression(max_iter=2000)
        clf.fit(values[train_idx], labels[train_idx])
        preds = clf.predict(values[test_idx])
        rows.append(
            {
                "direction": f"{train_probe}_to_{test_probe}",
                "train_probe": train_probe,
                "test_probe": test_probe,
                "accuracy": float(accuracy_score(labels[test_idx], preds)),
                "macro_f1": float(f1_score(labels[test_idx], preds, average="macro")),
            }
        )
    return pd.DataFrame(rows)


def representation_scorecard(values: np.ndarray, class_labels: np.ndarray, probe_labels: np.ndarray, *, seed: int, k: int) -> dict[str, float]:
    metrics = class_probe_metrics(values, class_labels, probe_labels, seed=seed, k=k)
    metric_map = {str(row.metric): float(row.value) for row in metrics.itertuples(index=False)}
    class_clf = classifier_metrics(values, class_labels, seed=seed)
    probe_clf = classifier_metrics(values, probe_labels, seed=seed)
    metric_map["class_predict_accuracy"] = class_clf["accuracy"]
    metric_map["class_predict_macro_f1"] = class_clf["macro_f1"]
    metric_map["probe_predict_accuracy"] = probe_clf["accuracy"]
    metric_map["probe_predict_macro_f1"] = probe_clf["macro_f1"]
    metric_map["class_probe_silhouette_ratio"] = (
        metric_map["silhouette_class"] / max(abs(metric_map["silhouette_probe"]), 1e-6)
        if not np.isnan(metric_map["silhouette_class"]) and not np.isnan(metric_map["silhouette_probe"])
        else float("nan")
    )
    metric_map["class_probe_nn_purity_ratio"] = (
        metric_map["nn_purity_class"] / max(metric_map["nn_purity_probe"], 1e-6)
        if not np.isnan(metric_map["nn_purity_class"]) and not np.isnan(metric_map["nn_purity_probe"])
        else float("nan")
    )
    metric_map["class_minus_probe_predictability_gap"] = (
        metric_map["class_predict_macro_f1"] - metric_map["probe_predict_macro_f1"]
        if not np.isnan(metric_map["class_predict_macro_f1"]) and not np.isnan(metric_map["probe_predict_macro_f1"])
        else float("nan")
    )
    transfer = cross_probe_transfer_metrics(values, class_labels, probe_labels, seed=seed)
    metric_map["cross_probe_transfer_accuracy_mean"] = float(transfer["accuracy"].dropna().mean()) if transfer["accuracy"].notna().any() else float("nan")
    metric_map["cross_probe_transfer_macro_f1_mean"] = float(transfer["macro_f1"].dropna().mean()) if transfer["macro_f1"].notna().any() else float("nan")
    return metric_map


def pca_plot_frame(values: np.ndarray, metadata: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    coords = reduce_for_plot(values, seed=seed)
    plot_df = metadata.copy()
    plot_df["dim1"] = coords[:, 0]
    plot_df["dim2"] = coords[:, 1]
    return plot_df


def try_load_shared_subset(run_dir: Path, sample_keys: list[str], *, mode: str) -> tuple[pd.DataFrame, np.ndarray] | None:
    if not run_dir.exists():
        return None
    metadata, embeddings = load_run_embeddings(run_dir)
    labels = set(mode_labels(mode))
    metadata = metadata[(metadata["dataset_id"] == "small2023_ev") & (metadata["label_optional"].isin(labels))].copy()
    metadata = metadata.reset_index(drop=False).rename(columns={"index": "source_index"})
    metadata = metadata[metadata["sample_key"].astype(str).isin(sample_keys)].copy()
    if metadata.empty:
        return None
    ordered = pd.DataFrame({"sample_key": sample_keys}).merge(metadata, on="sample_key", how="inner")
    return ordered.reset_index(drop=True), normalize_rows(embeddings[ordered["source_index"].to_numpy()])


def try_load_v2_summary(mode: str) -> pd.DataFrame | None:
    if mode != "mixture":
        return None
    geometry_path = SMALL2023_V2_DIR / "geometry_metrics_v2.csv"
    transfer_path = SMALL2023_V2_DIR / "comparison_cross_probe_metrics_v2.csv"
    if not geometry_path.exists():
        return None
    geometry = pd.read_csv(geometry_path)
    row = geometry.iloc[0]
    transfer = pd.read_csv(transfer_path) if transfer_path.exists() else pd.DataFrame()
    transfer_f1 = float(transfer["macro_f1"].mean()) if "macro_f1" in transfer.columns else float("nan")
    transfer_acc = float(transfer["accuracy"].mean()) if "accuracy" in transfer.columns else float("nan")
    return pd.DataFrame(
        [
            {"representation": "v2_specialized", "metric": "silhouette_class", "value": float(row["v2_class_silhouette"])},
            {"representation": "v2_specialized", "metric": "silhouette_probe", "value": float(row["v2_probe_silhouette"])},
            {"representation": "v2_specialized", "metric": "cross_probe_transfer_accuracy_mean", "value": transfer_acc},
            {"representation": "v2_specialized", "metric": "cross_probe_transfer_macro_f1_mean", "value": transfer_f1},
        ]
    )
