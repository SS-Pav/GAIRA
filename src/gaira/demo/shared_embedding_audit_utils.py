from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from gaira.demo.ev_analysis_utils import (
    decode_direct_matrix,
    entropy_normalized,
    knn_label_metrics,
    load_direct_processed_spectra_by_ids,
    normalize_rows,
)
from gaira.demo.v8_analysis_utils import (
    V5_EVAL_DIR,
    V5_RUN_DIR,
    V6_EVAL_DIR,
    V6_RUN_DIR,
    V7_EVAL_DIR,
    V7_RUN_DIR,
)


@dataclass(frozen=True)
class AuditUnit:
    audit_unit_id: str
    dataset_id: str
    sample_type: str
    target_kind: str
    target_name: str
    nuisance_name: str | None
    label_values: tuple[str, ...] | None = None
    notes: str = ""


SHARED_RUN_CANDIDATES = [
    {
        "run_id": "embedding_v5_full_true_gpu_run1",
        "run_dir": V5_RUN_DIR,
        "eval_dir": V5_EVAL_DIR,
        "truly_shared_global": True,
        "notes": "Full shared backbone baseline.",
    },
    {
        "run_id": "embedding_v6_within_type_gpu_run1",
        "run_dir": V6_RUN_DIR,
        "eval_dir": V6_EVAL_DIR,
        "truly_shared_global": True,
        "notes": "Within-type shared run; artifacts missing locally.",
    },
    {
        "run_id": "embedding_v7_anchor_gpu_run1",
        "run_dir": V7_RUN_DIR,
        "eval_dir": V7_EVAL_DIR,
        "truly_shared_global": True,
        "notes": "Anchor-invariance shared backbone.",
    },
    {
        "run_id": "embedding_v8_ev_stress_gpu_run1",
        "run_dir": Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_ev_gpu_runs/embedding_v8_ev_stress_gpu_run1"),
        "eval_dir": Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_ev_gpu_eval/embedding_v8_ev_stress_gpu_run1_eval_v2"),
        "truly_shared_global": False,
        "notes": "EV branch run, not shared/global.",
    },
    {
        "run_id": "embedding_v8_ev_stress_branch_gpu_run1",
        "run_dir": Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_ev_branch_gpu_runs/embedding_v8_ev_stress_branch_gpu_run1"),
        "eval_dir": Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_ev_branch_gpu_eval/embedding_v8_ev_stress_branch_gpu_run1_eval_v2"),
        "truly_shared_global": False,
        "notes": "EV state-aware branch run, not shared/global.",
    },
]


THEME_TYPE_MAP = {
    "adenine_sers_control": ("universal_pure_reference", "all_sample_types"),
    "amino_acid_raman_grounding": ("universal_pure_reference", "all_sample_types"),
    "metabolite_sers63_support": ("universal_pure_reference", "all_sample_types"),
    "serum_ag_colloids_grounding": ("domain_specific_serum", "serum_only"),
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_run_metadata(run_dir: Path) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    meta_path = run_dir / "metadata.csv"
    emb_path = run_dir / "embeddings.npy"
    if not meta_path.exists() or not emb_path.exists():
        return None, None
    meta = pd.read_csv(meta_path)
    meta["sample_key"] = meta["sample_key"].astype(str)
    emb = np.load(emb_path)
    return meta, emb


def run_inventory() -> pd.DataFrame:
    rows = []
    for item in SHARED_RUN_CANDIDATES:
        run_dir = Path(item["run_dir"])
        eval_dir = Path(item["eval_dir"])
        rows.append(
            {
                "run_id": item["run_id"],
                "local_path": str(run_dir),
                "embeddings_exists": (run_dir / "embeddings.npy").exists(),
                "metadata_exists": (run_dir / "metadata.csv").exists(),
                "eval_v2_exists": (eval_dir / "embedding_metrics_v2.csv").exists() or (eval_dir / "embedding_metrics.csv").exists(),
                "truly_shared_global": bool(item["truly_shared_global"]),
                "usable_for_dataset_audit": bool(item["truly_shared_global"] and (run_dir / "embeddings.npy").exists() and (run_dir / "metadata.csv").exists()),
                "notes": item["notes"],
            }
        )
    return pd.DataFrame(rows)


def parse_shine_order(value: str) -> float | None:
    try:
        day_text, conc_text = value.split("_")
        day = int(day_text.replace("D", ""))
        conc = int(conc_text.replace("C", ""))
        conc_rank = {0: 0, 10: 1, 20: 2, 40: 3}.get(conc)
        if conc_rank is None:
            return None
        return float(day * 10 + conc_rank)
    except Exception:
        return None


def parse_erg_order(value: str) -> float | None:
    try:
        num = value.replace("erg_", "").replace("_uM", "").replace("p", ".")
        return float(num)
    except Exception:
        return None


def build_dataset_inventory(v7_metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[AuditUnit]]:
    meta = v7_metadata.copy()
    meta["sample_key"] = meta["sample_key"].astype(str)
    inventory_rows: list[dict[str, object]] = []
    units: list[AuditUnit] = []

    dataset_rows = (
        meta[meta["sample_type"].isin(["ev", "serum"])]
        .groupby(["dataset_id", "sample_type"], sort=True)
        .size()
        .reset_index(name="sample_count")
        .sort_values(["sample_type", "sample_count"], ascending=[True, False])
    )

    for row in dataset_rows.itertuples(index=False):
        dataset_id = str(row.dataset_id)
        sample_type = str(row.sample_type)
        sub = meta[meta["dataset_id"] == dataset_id].copy()
        label_values = sorted([v for v in sub["label_optional"].fillna("").astype(str).unique() if v])
        subclass_values = sorted([v for v in sub["subclass_label"].fillna("").astype(str).unique() if v])
        record_kind_values = sorted(sub["record_kind"].fillna("").astype(str).unique())
        audit_kind = "mixed_or_ambiguous"
        evaluation_notes = ""
        preferred_target = ""
        meaningful = True

        if dataset_id == "small2023_ev":
            audit_kind = "mixed_or_ambiguous"
            preferred_target = "split into mixture and cell-line audit units"
            evaluation_notes = "Contains two distinct label families; audit as separate units."
            units.extend(
                [
                    AuditUnit(
                        audit_unit_id="small2023_ev__mixture",
                        dataset_id=dataset_id,
                        sample_type=sample_type,
                        target_kind="discrete_classification",
                        target_name="mixture_class",
                        nuisance_name="probe_family",
                        label_values=("c00", "c01", "c10", "c25", "c50", "c100"),
                        notes="Main small2023 mixture task with two probes.",
                    ),
                    AuditUnit(
                        audit_unit_id="small2023_ev__cellline",
                        dataset_id=dataset_id,
                        sample_type=sample_type,
                        target_kind="discrete_classification",
                        target_name="cellline_class",
                        nuisance_name="probe_family",
                        label_values=("Hec", "Hela", "Ht", "Mef", "Thp"),
                        notes="Cell-line subset; only one probe family in corpus.",
                    ),
                ]
            )
        elif dataset_id == "shine_ev_sers":
            audit_kind = "ordered_condition"
            preferred_target = "dose_day_grid"
            evaluation_notes = "Use discrete condition labels and ordered day/dose index; Set9/Set10 as nuisance."
            units.append(
                AuditUnit(
                    audit_unit_id=dataset_id,
                    dataset_id=dataset_id,
                    sample_type=sample_type,
                    target_kind="discrete_plus_ordered",
                    target_name="dose_day_condition",
                    nuisance_name="acquisition_set",
                    notes=evaluation_notes,
                )
            )
        elif dataset_id == "diabetes_plasma_ev_sers":
            audit_kind = "discrete_classification"
            preferred_target = "impact_vs_strong_d"
            evaluation_notes = "Binary EV stress-state classification."
            units.append(
                AuditUnit(
                    audit_unit_id=dataset_id,
                    dataset_id=dataset_id,
                    sample_type=sample_type,
                    target_kind="discrete_classification",
                    target_name="disease_state",
                    nuisance_name=None,
                    notes=evaluation_notes,
                )
            )
        elif dataset_id == "cca_hcc_lm_serum_sers":
            audit_kind = "discrete_classification"
            preferred_target = "healthy_vs_cca_hcc_lm"
            evaluation_notes = "Multiclass serum cohort structure."
            units.append(
                AuditUnit(
                    audit_unit_id=dataset_id,
                    dataset_id=dataset_id,
                    sample_type=sample_type,
                    target_kind="discrete_classification",
                    target_name="serum_cohort_class",
                    nuisance_name=None,
                    notes=evaluation_notes,
                )
            )
        elif dataset_id == "covid_serum_raman":
            audit_kind = "discrete_classification"
            preferred_target = "healthy_suspected_confirmed_tube"
            evaluation_notes = "Discrete cohort organization with tube control caveat."
            units.append(
                AuditUnit(
                    audit_unit_id=dataset_id,
                    dataset_id=dataset_id,
                    sample_type=sample_type,
                    target_kind="discrete_classification",
                    target_name="covid_state_class",
                    nuisance_name=None,
                    notes=evaluation_notes,
                )
            )
        elif dataset_id == "serum_protocol_comparison":
            audit_kind = "discrete_classification"
            preferred_target = "protocol_id"
            evaluation_notes = "Protocol-only nuisance dataset."
            units.append(
                AuditUnit(
                    audit_unit_id=dataset_id,
                    dataset_id=dataset_id,
                    sample_type=sample_type,
                    target_kind="discrete_classification",
                    target_name="protocol_class",
                    nuisance_name=None,
                    notes=evaluation_notes,
                )
            )
        elif dataset_id == "ergothioneine_serum":
            audit_kind = "ordered_condition"
            preferred_target = "ergothioneine_concentration"
            evaluation_notes = "Calibration-like ordered concentration series."
            units.append(
                AuditUnit(
                    audit_unit_id=dataset_id,
                    dataset_id=dataset_id,
                    sample_type=sample_type,
                    target_kind="ordered_condition",
                    target_name="concentration_order",
                    nuisance_name=None,
                    notes=evaluation_notes,
                )
            )
        elif dataset_id == "serum_ag_colloids":
            audit_kind = "mixed_or_ambiguous"
            preferred_target = "spiked_serum_label"
            evaluation_notes = "Protocol/matrix-heavy serum reference set with spiking and donor subsets."
            units.append(
                AuditUnit(
                    audit_unit_id=dataset_id,
                    dataset_id=dataset_id,
                    sample_type=sample_type,
                    target_kind="discrete_classification",
                    target_name="serum_spiking_label",
                    nuisance_name="sub_archive",
                    notes=evaluation_notes,
                )
            )
        elif dataset_id == "cspp_serum":
            audit_kind = "mixed_or_ambiguous"
            preferred_target = "none"
            evaluation_notes = "Too heterogeneous for a single within-dataset target; protocol panels should be audited separately later."
            meaningful = False
        else:
            audit_kind = "mixed_or_ambiguous"
            preferred_target = "review_needed"
            evaluation_notes = "No audit-unit rule defined."
            meaningful = False

        inventory_rows.append(
            {
                "dataset_id": dataset_id,
                "sample_type": sample_type,
                "sample_count": int(row.sample_count),
                "available_label_values": "|".join(label_values[:40]),
                "available_subclass_values": "|".join(subclass_values[:40]),
                "record_kinds": "|".join(record_kind_values),
                "dataset_characterization": audit_kind,
                "preferred_audit_target": preferred_target,
                "within_dataset_evaluation_meaningful": meaningful,
                "evaluation_notes": evaluation_notes,
            }
        )
    return pd.DataFrame(inventory_rows), units


def sample_unit_metadata(metadata: pd.DataFrame, unit: AuditUnit, *, max_samples: int, seed: int) -> pd.DataFrame:
    sub = metadata[(metadata["dataset_id"] == unit.dataset_id) & (metadata["record_kind"] == "processed_spectrum")].copy()
    if unit.label_values is not None:
        sub = sub[sub["label_optional"].isin(unit.label_values)].copy()
    sub["target_label"] = sub["label_optional"].fillna("").astype(str)
    sub["nuisance_label"] = sub["subclass_label"].fillna("").astype(str)
    sub["ordered_value"] = np.nan
    if unit.dataset_id == "shine_ev_sers":
        sub["ordered_value"] = sub["label_optional"].fillna("").astype(str).map(parse_shine_order)
    elif unit.dataset_id == "ergothioneine_serum":
        sub["ordered_value"] = sub["label_optional"].fillna("").astype(str).map(parse_erg_order)
        sub["target_label"] = ""
    elif unit.dataset_id == "serum_protocol_comparison":
        sub["nuisance_label"] = ""
    elif unit.dataset_id in {"diabetes_plasma_ev_sers", "cca_hcc_lm_serum_sers", "covid_serum_raman"}:
        sub["nuisance_label"] = ""
    elif unit.dataset_id == "small2023_ev" and unit.audit_unit_id.endswith("__cellline"):
        sub["nuisance_label"] = ""
    elif unit.dataset_id == "serum_ag_colloids":
        sub["nuisance_label"] = sub["subclass_label"].fillna("").astype(str)

    if len(sub) <= max_samples:
        return sub.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    group_cols: list[str] = []
    if sub["target_label"].replace("", np.nan).notna().any():
        group_cols.append("target_label")
    if sub["nuisance_label"].replace("", np.nan).notna().any() and sub["nuisance_label"].nunique() > 1:
        group_cols.append("nuisance_label")
    if group_cols:
        parts = []
        grouped = sub.groupby(group_cols, sort=True, dropna=False)
        per_group = max(1, max_samples // max(grouped.ngroups, 1))
        for _, group in grouped:
            parts.append(group.sample(n=min(per_group, len(group)), random_state=seed))
        sampled = pd.concat(parts, ignore_index=True).drop_duplicates("sample_key")
        if len(sampled) < max_samples:
            remaining = sub.loc[~sub["sample_key"].isin(sampled["sample_key"])]
            if not remaining.empty:
                extra_idx = rng.choice(remaining.index.to_numpy(), size=min(max_samples - len(sampled), len(remaining)), replace=False)
                sampled = pd.concat([sampled, remaining.loc[extra_idx]], ignore_index=True)
        return sampled.head(max_samples).reset_index(drop=True)

    indices = rng.choice(sub.index.to_numpy(), size=max_samples, replace=False)
    return sub.loc[np.sort(indices)].reset_index(drop=True)


def load_direct_baseline(sampled_meta: pd.DataFrame) -> np.ndarray | None:
    if sampled_meta.empty:
        return None
    dataset_id = str(sampled_meta["dataset_id"].iloc[0])
    processing_version = str(sampled_meta["processing_version"].mode().iloc[0])
    sample_keys = sampled_meta["sample_key"].astype(str).tolist()
    direct = load_direct_processed_spectra_by_ids(
        dataset_id=dataset_id,
        processing_version=processing_version,
        sample_keys=sample_keys,
    )
    direct["sample_key"] = direct["sample_key"].astype(str)
    _, X = decode_direct_matrix(direct)
    matrix = StandardScaler().fit_transform(X)
    order = pd.DataFrame({"sample_key": sample_keys}).merge(direct[["sample_key"]], on="sample_key", how="inner")
    indexer = pd.Series(np.arange(len(direct)), index=direct["sample_key"].astype(str))
    return matrix[indexer.loc[order["sample_key"]].to_numpy()]


def align_run_values(run_meta: pd.DataFrame, run_values: np.ndarray, sample_keys: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    subset = run_meta[run_meta["sample_key"].astype(str).isin(sample_keys)].copy()
    if subset.empty:
        raise RuntimeError("No shared run overlap for requested sample keys.")
    subset = subset.reset_index(drop=False).rename(columns={"index": "source_index"})
    ordered = pd.DataFrame({"sample_key": sample_keys}).merge(subset, on="sample_key", how="inner")
    return ordered, normalize_rows(run_values[ordered["source_index"].to_numpy()])


def classifier_metrics(values: np.ndarray, labels: pd.Series, *, seed: int) -> dict[str, float]:
    labels = labels.fillna("").astype(str)
    valid = labels != ""
    labels = labels[valid].reset_index(drop=True)
    X = values[valid.to_numpy()]
    if labels.nunique() < 2 or len(labels) < 20:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels.to_numpy(),
        test_size=0.25,
        random_state=seed,
        stratify=labels.to_numpy(),
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
    }


def sampled_silhouette(values: np.ndarray, labels: pd.Series, *, seed: int, max_points: int = 3000) -> float:
    labels = labels.fillna("").astype(str)
    valid = labels != ""
    labels = labels[valid].reset_index(drop=True)
    X = values[valid.to_numpy()]
    if labels.nunique() < 2 or len(X) < 20:
        return float("nan")
    if len(X) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_points, replace=False)
        X = X[idx]
        labels = labels.iloc[idx].reset_index(drop=True)
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(X, labels.to_numpy()))


def ordered_metrics(values: np.ndarray, ordered_values: pd.Series, *, seed: int, k: int = 6) -> dict[str, float]:
    valid = ordered_values.notna()
    order = ordered_values[valid].astype(float).reset_index(drop=True)
    X = values[valid.to_numpy()]
    if len(order) < 20 or order.nunique() < 3:
        return {
            "ordered_spearman_pc1": float("nan"),
            "ordered_neighbor_continuity": float("nan"),
        }
    n_comp = max(1, min(8, X.shape[1], len(X) - 1))
    coords = PCA(n_components=n_comp, random_state=seed).fit_transform(X)
    pc1 = pd.Series(coords[:, 0])
    ordered_spearman = float(pc1.corr(order, method="spearman"))
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="cosine", algorithm="brute").fit(X)
    _, idx = nn.kneighbors(X)
    neighbor_order = order.to_numpy()[idx[:, 1:]].mean(axis=1)
    continuity = float(pd.Series(neighbor_order).corr(order, method="spearman"))
    return {
        "ordered_spearman_pc1": ordered_spearman,
        "ordered_neighbor_continuity": continuity,
    }


def cluster_metrics(
    values: np.ndarray,
    target_labels: pd.Series,
    ordered_values: pd.Series,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    n = len(values)
    target = target_labels.fillna("").astype(str)
    target_unique = max(target[target != ""].nunique(), 0)
    ordered_valid = ordered_values.notna().sum()
    k = target_unique if target_unique >= 2 else max(2, min(8, round(math.sqrt(max(n, 4) / 200.0))))
    if ordered_valid >= 24:
        k = max(k, min(8, int(min(ordered_values.nunique(dropna=True), 8))))
    k = int(max(2, min(k, max(2, min(12, n // 20 if n >= 80 else 2)))))
    n_comp = max(1, min(16, values.shape[1], len(values) - 1))
    coords = PCA(n_components=n_comp, random_state=seed).fit_transform(values)
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    clusters = model.fit_predict(coords)
    cluster_df = pd.DataFrame(
        {
            "cluster_id": clusters.astype(int),
            "target_label": target.to_numpy(),
            "ordered_value": ordered_values.to_numpy(),
        }
    )
    entropy_vals = []
    purity_weighted = []
    for _, group in cluster_df.groupby("cluster_id", sort=True):
        labels = group["target_label"]
        labels = labels[labels != ""]
        if labels.empty:
            continue
        entropy_vals.append(entropy_normalized(labels))
        purity_weighted.append(labels.value_counts(normalize=True).iloc[0] * len(group))
    weighted_purity = float(sum(purity_weighted) / max(len(cluster_df), 1)) if purity_weighted else float("nan")
    weighted_entropy = float(np.mean(entropy_vals)) if entropy_vals else float("nan")
    order_metric = float("nan")
    if ordered_values.notna().sum() >= 24:
        centers = pd.DataFrame({"cluster_id": clusters, "pc1": coords[:, 0], "ordered": ordered_values.to_numpy()})
        cluster_means = centers.groupby("cluster_id", sort=True).agg({"pc1": "mean", "ordered": "mean"}).dropna()
        if len(cluster_means) >= 3:
            order_metric = float(cluster_means["pc1"].corr(cluster_means["ordered"], method="spearman"))
    return cluster_df, {
        "cluster_count": k,
        "cluster_purity_target": weighted_purity,
        "cluster_entropy_target": weighted_entropy,
        "cluster_order_spearman": order_metric,
    }


def primary_composite(metrics: dict[str, float], target_kind: str) -> float:
    parts: list[float] = []
    for key in ["silhouette_class", "nn_purity_class", "top1_match_class", "class_predict_macro_f1"]:
        value = metrics.get(key)
        if value is not None and not math.isnan(value):
            parts.append(float(value))
    if target_kind in {"ordered_condition", "discrete_plus_ordered"}:
        for key in ["ordered_spearman_pc1", "ordered_neighbor_continuity", "cluster_order_spearman"]:
            value = metrics.get(key)
            if value is not None and not math.isnan(value):
                parts.append(abs(float(value)))
    return float(np.mean(parts)) if parts else float("nan")


def nuisance_composite(metrics: dict[str, float]) -> float:
    parts: list[float] = []
    for key in ["silhouette_nuisance", "nn_purity_nuisance", "top1_match_nuisance", "nuisance_predict_macro_f1"]:
        value = metrics.get(key)
        if value is not None and not math.isnan(value):
            parts.append(float(value))
    return float(np.mean(parts)) if parts else float("nan")


def evaluate_representation(
    values: np.ndarray,
    sampled_meta: pd.DataFrame,
    *,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    metrics: dict[str, float] = {}
    target = sampled_meta["target_label"]
    nuisance = sampled_meta["nuisance_label"]
    ordered = sampled_meta["ordered_value"]

    if target.replace("", np.nan).notna().sum() >= 20 and target[target != ""].nunique() >= 2:
        metrics["silhouette_class"] = sampled_silhouette(values, target, seed=seed)
        metrics.update({f"{k}_class": v for k, v in knn_label_metrics(values, target.to_numpy(), k=6).items()})
        class_clf = classifier_metrics(values, target, seed=seed)
        metrics["class_predict_accuracy"] = class_clf["accuracy"]
        metrics["class_predict_macro_f1"] = class_clf["macro_f1"]
    else:
        metrics["silhouette_class"] = float("nan")
        metrics["nn_purity_class"] = float("nan")
        metrics["neighbor_entropy_class"] = float("nan")
        metrics["top1_match_class"] = float("nan")
        metrics["class_predict_accuracy"] = float("nan")
        metrics["class_predict_macro_f1"] = float("nan")

    if nuisance.replace("", np.nan).notna().sum() >= 20 and nuisance[nuisance != ""].nunique() >= 2:
        metrics["silhouette_nuisance"] = sampled_silhouette(values, nuisance, seed=seed)
        nuisance_knn = knn_label_metrics(values, nuisance.to_numpy(), k=6)
        metrics["nn_purity_nuisance"] = nuisance_knn["nn_purity"]
        metrics["neighbor_entropy_nuisance"] = nuisance_knn["neighbor_entropy"]
        metrics["top1_match_nuisance"] = nuisance_knn["top1_match"]
        nuisance_clf = classifier_metrics(values, nuisance, seed=seed)
        metrics["nuisance_predict_accuracy"] = nuisance_clf["accuracy"]
        metrics["nuisance_predict_macro_f1"] = nuisance_clf["macro_f1"]
    else:
        metrics["silhouette_nuisance"] = float("nan")
        metrics["nn_purity_nuisance"] = float("nan")
        metrics["neighbor_entropy_nuisance"] = float("nan")
        metrics["top1_match_nuisance"] = float("nan")
        metrics["nuisance_predict_accuracy"] = float("nan")
        metrics["nuisance_predict_macro_f1"] = float("nan")

    metrics.update(ordered_metrics(values, ordered, seed=seed))
    clusters, cluster_summary = cluster_metrics(values, target, ordered, seed=seed)
    metrics.update(cluster_summary)
    return metrics, clusters


def grounding_inventory(grounding_theme_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_id, group in grounding_theme_table.groupby("dataset_id", sort=True):
        category, allowed = THEME_TYPE_MAP.get(str(dataset_id), ("other_relevant_type", "review_needed"))
        rows.append(
            {
                "grounding_dataset_id": dataset_id,
                "n_records": int(len(group)),
                "grounding_type": category,
                "proposed_allowed_sample_type": allowed,
                "top_theme_count": int(group["grounding_theme"].fillna("").astype(str).value_counts().iloc[0]) if not group.empty else 0,
                "cross_dataset_usable_fraction": float(group["cross_dataset_usable"].fillna(False).astype(bool).mean()) if "cross_dataset_usable" in group.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("n_records", ascending=False).reset_index(drop=True)


def ontology_rows() -> pd.DataFrame:
    rows = [
        ("protein_peptide", "generic_proteinaceous", "Broad proteinaceous / peptide backbone support", "tier2", "supports amide-backbone dominated spectra"),
        ("protein_peptide", "amide_backbone_dominated", "Amide-rich protein / peptide support", "tier2", "stronger than generic proteinaceous"),
        ("protein_peptide", "aromatic_aa_enriched", "Aromatic amino-acid enriched protein support", "tier2", "use when aromatic residues dominate"),
        ("lipid_membrane", "phospholipid_membrane", "Membrane phospholipid-like support", "tier2", "EV membrane-relevant"),
        ("lipid_membrane", "fatty_acid_like", "Fatty-acid-like lipid support", "tier2", "small-molecule lipid signatures"),
        ("nucleic_acid", "purine", "Purine-like nucleobase support", "tier2", "draft placement; may also behave as small-molecule context"),
        ("nucleic_acid", "pyrimidine", "Pyrimidine-like nucleobase support", "tier2", "draft placeholder"),
        ("nucleic_acid", "phosphate_backbone_general", "General nucleic-acid phosphate backbone support", "tier2", "broader than purine/pyrimidine"),
        ("carbohydrate", "generic_carbohydrate", "Generic carbohydrate / saccharide support", "tier2", "keep broad"),
        ("small_molecule_metabolite", "general_metabolite", "General small-molecule metabolite support", "tier2", "separate from purine"),
        ("small_molecule_metabolite", "amino_acid_free", "Free amino-acid / zwitterionic metabolite support", "tier2", "not protein backbone"),
        ("small_molecule_metabolite", "organic_acid", "Organic-acid metabolite support", "tier2", "draft placeholder"),
        ("small_molecule_metabolite", "redox_active_metabolite", "Redox-active metabolite support", "tier2", "use instead of broad oxidative/redox junk drawer"),
        ("small_molecule_metabolite", "aromatic_metabolite", "Aromatic small-molecule metabolite support", "tier2", "draft placeholder"),
        ("matrix_caveat", "serum_matrix", "Serum matrix / preparation caveat channel", "tier2", "supportive caveat, not headline biology"),
    ]
    return pd.DataFrame(rows, columns=["tier1", "tier2", "description", "level", "notes"])
