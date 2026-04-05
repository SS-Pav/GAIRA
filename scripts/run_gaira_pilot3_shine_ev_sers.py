from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from gaira.autoresearch_storage import (
    DEFAULT_STORAGE_CONFIG_PATH,
    initialize_autoresearch_sprint,
    load_autoresearch_storage_config,
)
from gaira.demo.autoresearch_pass5_utils import (
    Pass5HarnessConfig,
    apply_pass5_filter_mode,
    build_bsv_profiles_pass5,
)
from gaira.demo.gaira_experiment_runner_utils import (
    ResolvedExperiment,
    build_source_role_sets,
    load_architecture_registries,
    load_grounding_family_dataframe,
    load_query_dataframe,
    retrieval_hit_summary,
)
from gaira.demo.gaira_pilot_utils import (
    ALL_AXES,
    build_class_axis_entropy,
    build_class_neighborhood_entropy,
    build_class_top1_dominance,
    build_class_topk_neighborhood_composition,
    build_pdf_report,
    pairwise_delta_bsv,
    plot_pairwise_delta_heatmap,
)
from gaira.demo.raw_bsv_pilot_utils import (
    apply_source_role_policy,
    decode_and_align,
    load_ontology_rules,
    map_references_to_axes,
)


ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "reports" / "gaira_architecture_scaffold_v2"
PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"
ONTOLOGY_PATH = ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv"
PILOT2_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot2_target_validation_v1"
)

SPRINT_SUBDIR = "pilot3_shine_ev_sers"
SUBSET_ALIAS = "shine_ev_stress"

CONFIG_SPEC = {
    "config_id": "candidate_v2_cfg05_max_desaturation",
    "short_label": "cfg05",
    "display_name": "Candidate v2 cfg05 max desaturation",
    "filter_mode": "purine_expanded_neighbor",
    "top_k": 5,
    "weighting_mode": "softmax_temperature",
    "weighting_param": 1.0,
    "diversity_mode": "compound_uniqueness_penalty",
}

FIXED_RADAR_AXES = [
    "nucleic_acid",
    "protein_peptide",
    "lipid_membrane",
    "carbohydrate_glycan",
    "small_molecule_metabolite",
    "substrate_adsorption_bias",
]

FAMILY_ORDER = [
    "purine_core_like",
    "methylated_purine_like",
    "guanidine_like",
    "sulfur_small_molecule_like",
    "aromatic_small_molecule_like",
    "generic_other_metabolite",
]

DAY_ORDER = ["D0", "D1", "D2"]
DAY_COLORS = {"D0": "#355070", "D1": "#b56576", "D2": "#2a9d8f"}
FAMILY_COLORS = {
    "purine_core_like": "#355070",
    "methylated_purine_like": "#6d597a",
    "guanidine_like": "#b56576",
    "sulfur_small_molecule_like": "#2a9d8f",
    "aromatic_small_molecule_like": "#577590",
    "generic_other_metabolite": "#e9c46a",
}
CLUSTER_COLORS = ["#355070", "#b56576", "#6d597a", "#2a9d8f", "#e76f51", "#577590"]


def _resolve_alias(registries, subset_alias: str) -> ResolvedExperiment:
    matches = registries.dataset_experiments[
        registries.dataset_experiments["subset_alias"].astype(str) == str(subset_alias)
    ].copy()
    if matches.empty or len(matches) > 1:
        raise RuntimeError(f"Could not resolve unique subset alias {subset_alias}")
    dataset_row = matches.iloc[0]
    experiment_row = pd.Series(
        {
            "experiment_id": f"pilot3_target_validation__{subset_alias}",
            "subset_alias": subset_alias,
            "grounding_families_used": "universal_biochemical_grounding",
        }
    )
    return ResolvedExperiment(
        experiment_row=experiment_row,
        dataset_row=dataset_row,
        subset_alias=subset_alias,
        grounding_family_names=["universal_biochemical_grounding"],
    )


def _config_to_harness(spec: dict[str, object]) -> Pass5HarnessConfig:
    return Pass5HarnessConfig(
        config_id=str(spec["config_id"]),
        universal_grounding_filter_mode=str(spec["filter_mode"]),
        top_k=int(spec["top_k"]),
        weighting_mode=str(spec["weighting_mode"]),
        weighting_param=None if spec["weighting_param"] is None else float(spec["weighting_param"]),
        diversity_mode=str(spec["diversity_mode"]),
        family_min_coverage=0,
    )


def _prepare_grounding_and_mapping(registries, resolved: ResolvedExperiment, config_spec: dict[str, object]):
    harness_config = _config_to_harness(config_spec)
    original_grounding = list(resolved.grounding_family_names)
    object.__setattr__(resolved, "grounding_family_names", ["universal_biochemical_grounding"])
    try:
        grounding_df, family_to_sources, unavailable_sources = load_grounding_family_dataframe(resolved, registries)
    finally:
        object.__setattr__(resolved, "grounding_family_names", original_grounding)
    grounding_df = apply_pass5_filter_mode(grounding_df, harness_config.universal_grounding_filter_mode)
    primary_sources, caveat_only_sources = build_source_role_sets(resolved, family_to_sources)
    available_source_keys = set(grounding_df["source_key"].astype(str))
    primary_sources = {key for key in primary_sources if key in available_source_keys}
    caveat_only_sources = {key for key in caveat_only_sources if key in available_source_keys}
    ontology_rules = load_ontology_rules(ONTOLOGY_PATH)
    raw_mapping_df = map_references_to_axes(grounding_df, ontology_rules)
    mapping_df = apply_source_role_policy(
        raw_mapping_df,
        grounding_df,
        primary_sources=primary_sources,
        caveat_only_sources=caveat_only_sources,
    )
    return grounding_df, mapping_df, harness_config, unavailable_sources


def _axes_present(df: pd.DataFrame) -> list[str]:
    return [axis for axis in ALL_AXES if axis in df.columns]


def _ensure_fixed_axes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for axis in FIXED_RADAR_AXES:
        if axis not in out.columns:
            out[axis] = 0.0
    return out


def _parse_day(class_label: str) -> str:
    match = re.search(r"(D\d+)", str(class_label))
    return match.group(1) if match else str(class_label)


def _parse_concentration(class_label: str) -> int:
    match = re.search(r"C(\d+)", str(class_label))
    return int(match.group(1)) if match else -1


def _trajectory_index(day_label: str, concentration: int) -> int:
    day_num = int(str(day_label).replace("D", "")) if str(day_label).startswith("D") else 0
    return day_num * 100 + int(concentration)


def _extract_sample_id(row: pd.Series) -> str:
    if "sample_id" in row.index and pd.notna(row["sample_id"]) and str(row["sample_id"]).strip():
        return str(row["sample_id"]).strip()
    source_file = str(row.get("source_file", ""))
    match = re.search(r"/([^/]+)/s_\d+$", source_file)
    if match:
        return str(match.group(1))
    class_label = str(row.get("class_label", "sample"))
    source_match = re.search(r"(s_\d+)$", source_file)
    suffix = source_match.group(1) if source_match else "unknown"
    return f"{class_label}_{suffix}"


def _build_sample_query_df(query_df: pd.DataFrame) -> pd.DataFrame:
    master_x, matrix = decode_and_align(query_df)
    work = query_df.reset_index(drop=True).copy()
    work["derived_sample_id"] = work.apply(_extract_sample_id, axis=1)
    rows = []
    for sample_id, group in work.groupby("derived_sample_id", sort=True):
        idx = group.index.to_numpy()
        mean_vector = matrix[idx].mean(axis=0)
        class_label = str(group["class_label"].iloc[0])
        day_label = _parse_day(class_label)
        concentration = _parse_concentration(class_label)
        rows.append(
            {
                "sample_key": f"sample_mean__{sample_id}",
                "sample_id": str(sample_id),
                "dataset_id": str(group["dataset_id"].iloc[0]),
                "processing_version": str(group["processing_version"].iloc[0]),
                "wavenumbers_json": json.dumps(master_x.astype(float).tolist()),
                "intensity_json": json.dumps(mean_vector.astype(float).tolist()),
                "class_label": class_label,
                "broad_class_label": day_label,
                "trajectory_concentration": concentration,
                "trajectory_index": _trajectory_index(day_label, concentration),
                "subclass_label": str(group["subclass_label"].iloc[0]),
                "source_file": str(group["source_file"].iloc[0]),
                "source_key": str(group["source_key"].iloc[0]),
                "n_scans": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _cohort_delta(df: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    cohort_mean = df[axes].mean(axis=0)
    out = df.copy()
    for axis in axes:
        out[axis] = df[axis].to_numpy(dtype=float) - float(cohort_mean[axis])
    return out


def _fit_pca(matrix: np.ndarray, *, scale: bool) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if scale:
        std = centered.std(axis=0, keepdims=True)
        centered = centered / np.where(std < 1e-9, 1.0, std)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * s[:2]
    denom = max(float((s**2).sum()), 1e-12)
    explained = (s[:2] ** 2) / denom if len(s) >= 2 else np.array([1.0, 0.0])
    return scores, explained


def _pca_dataframe(df: pd.DataFrame, axes: list[str], *, scale: bool) -> pd.DataFrame:
    scores, explained = _fit_pca(df[axes].to_numpy(dtype=float), scale=scale)
    out = df[
        [
            "sample_key",
            "sample_id",
            "class_label",
            "broad_class_label",
            "trajectory_concentration",
            "trajectory_index",
        ]
    ].copy()
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1]
    out["pc1_explained_ratio"] = float(explained[0]) if len(explained) > 0 else 1.0
    out["pc2_explained_ratio"] = float(explained[1]) if len(explained) > 1 else 0.0
    return out


def _maybe_umap_dataframe(df: pd.DataFrame, axes: list[str]) -> pd.DataFrame | None:
    try:
        import umap  # type: ignore
    except Exception:
        return None
    X = StandardScaler().fit_transform(df[axes].to_numpy(dtype=float))
    reducer = umap.UMAP(
        n_neighbors=min(12, max(len(df) - 1, 2)),
        min_dist=0.25,
        metric="euclidean",
        random_state=42,
    )
    coords = reducer.fit_transform(X)
    out = df[
        [
            "sample_key",
            "sample_id",
            "class_label",
            "broad_class_label",
            "trajectory_concentration",
            "trajectory_index",
        ]
    ].copy()
    out["u1"] = coords[:, 0]
    out["u2"] = coords[:, 1]
    return out


def _compound_to_family(name: str) -> str:
    lower = str(name).strip().lower()
    if any(token in lower for token in ["3-methyladenine", "methyladenine"]):
        return "methylated_purine_like"
    if any(token in lower for token in ["guanidine", "guanidino"]):
        return "guanidine_like"
    if any(token in lower for token in ["cyste", "glutath", "methion", "seleno", "sulfoximine", "sulfur"]):
        return "sulfur_small_molecule_like"
    if any(token in lower for token in ["tyr", "trypt", "phenyl", "indole", "dopamine", "3-methoxytyramine"]):
        return "aromatic_small_molecule_like"
    if any(token in lower for token in ["adenine", "xanth", "hypox", "uric", "urate", "inos", "purine"]):
        return "purine_core_like"
    return "generic_other_metabolite"


def _family_fingerprint_from_neighborhood(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    work = df.copy()
    work["family"] = work["compound_label"].astype(str).map(_compound_to_family)
    grouped = work.groupby([label_col, "family"], as_index=False)["support_fraction"].sum()
    rows = []
    for label in sorted(work[label_col].astype(str).unique().tolist()):
        sub = grouped[grouped[label_col].astype(str) == label].copy()
        total = float(sub["support_fraction"].sum())
        existing = {str(x) for x in sub["family"].tolist()}
        for family in FAMILY_ORDER:
            val = 0.0
            if family in existing:
                val = float(sub[sub["family"].astype(str) == family]["support_fraction"].iloc[0])
            rows.append(
                {
                    label_col: label,
                    "family": family,
                    "family_fraction": (val / total) if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _sample_family_fingerprint(sample_retrieval_df: pd.DataFrame, sample_meta_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_retrieval_df.copy()
    df["family"] = df["reference_compound_label"].astype(str).map(_compound_to_family)
    grouped = (
        df.groupby(["query_sample_key", "query_class_label", "family"], as_index=False)["support_weight"]
        .sum()
        .rename(
            columns={
                "query_sample_key": "sample_key",
                "query_class_label": "class_label",
                "support_weight": "family_support",
            }
        )
    )
    sample_map = sample_meta_df.set_index("sample_key")[["sample_id", "broad_class_label", "trajectory_concentration", "trajectory_index"]]
    rows = []
    for sample_key, group in grouped.groupby("sample_key", sort=True):
        total = float(group["family_support"].sum())
        existing = {str(x) for x in group["family"].tolist()}
        class_label = str(group["class_label"].iloc[0])
        meta = sample_map.loc[str(sample_key)]
        for family in FAMILY_ORDER:
            val = 0.0
            if family in existing:
                val = float(group[group["family"].astype(str) == family]["family_support"].iloc[0])
            rows.append(
                {
                    "sample_key": str(sample_key),
                    "sample_id": str(meta["sample_id"]),
                    "class_label": class_label,
                    "broad_class_label": str(meta["broad_class_label"]),
                    "trajectory_concentration": int(meta["trajectory_concentration"]),
                    "trajectory_index": int(meta["trajectory_index"]),
                    "family": family,
                    "family_fraction": (val / total) if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _mean_within_variance(df: pd.DataFrame, axes: list[str], label_col: str, label: str) -> float:
    sub = df[df[label_col].astype(str) == str(label)].copy()
    if len(sub) <= 1:
        return 0.0
    return float(sub[axes].var(ddof=1).mean())


def _nearest_neighbor_purity(df: pd.DataFrame, axes: list[str], *, label_col: str, n_neighbors: int = 5) -> float:
    X = StandardScaler().fit_transform(df[axes].to_numpy(dtype=float))
    labels = df[label_col].astype(str).to_numpy()
    n_use = min(n_neighbors + 1, len(df))
    nn = NearestNeighbors(n_neighbors=n_use)
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)
    purities = []
    for i in range(len(df)):
        neigh = [j for j in indices[i] if j != i][:n_neighbors]
        purities.append(float(np.mean(labels[neigh] == labels[i])) if neigh else 0.0)
    return float(np.mean(purities)) if purities else 0.0


def _entropy_from_values(values: np.ndarray) -> float:
    safe = values[values > 0]
    if safe.size == 0:
        return 0.0
    return float(-(safe * np.log(safe)).sum())


def _broad_class_metrics(
    sample_bsv_df: pd.DataFrame,
    sample_delta_df: pd.DataFrame,
    sample_family_df: pd.DataFrame,
    sample_retrieval_df: pd.DataFrame,
    axes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    classes = [label for label in DAY_ORDER if label in set(sample_bsv_df["broad_class_label"].astype(str))]
    class_mean_bsv_df = sample_bsv_df.groupby("broad_class_label", as_index=False)[axes].mean().rename(
        columns={"broad_class_label": "class_label"}
    )
    class_mean_delta_df = sample_delta_df.groupby("broad_class_label", as_index=False)[axes].mean().rename(
        columns={"broad_class_label": "class_label"}
    )
    # Replace class labels with broad day labels.
    sample_to_broad = sample_bsv_df.set_index("sample_key")["broad_class_label"].to_dict()
    broad_hits = sample_retrieval_df.copy()
    broad_hits["query_class_label"] = broad_hits["query_sample_key"].astype(str).map(sample_to_broad)
    broad_neighborhood_df = build_class_topk_neighborhood_composition(broad_hits)
    broad_family_df = _family_fingerprint_from_neighborhood(broad_neighborhood_df, "class_label")
    neighborhood_entropy_df = build_class_neighborhood_entropy(broad_neighborhood_df)
    top1_df = build_class_top1_dominance(broad_neighborhood_df)
    axis_entropy_df = build_class_axis_entropy(class_mean_bsv_df)
    silhouette_abs = float(
        silhouette_score(StandardScaler().fit_transform(sample_bsv_df[axes]), sample_bsv_df["broad_class_label"].astype(str))
    ) if len(classes) > 1 else 0.0
    silhouette_delta = float(
        silhouette_score(StandardScaler().fit_transform(sample_delta_df[axes]), sample_delta_df["broad_class_label"].astype(str))
    ) if len(classes) > 1 else 0.0
    purity_abs = _nearest_neighbor_purity(sample_bsv_df, axes, label_col="broad_class_label")
    purity_delta = _nearest_neighbor_purity(sample_delta_df, axes, label_col="broad_class_label")
    pairwise_abs = pairwise_delta_bsv(class_mean_bsv_df, axes)
    pairwise_delta_df = pairwise_delta_bsv(class_mean_delta_df, axes)
    abs_centroid_map = class_mean_bsv_df.set_index("class_label")[axes]
    delta_centroid_map = class_mean_delta_df.set_index("class_label")[axes]
    rows = []
    for class_label in classes:
        centroid = delta_centroid_map.loc[class_label].to_numpy(dtype=float)
        others = class_mean_delta_df[class_mean_delta_df["class_label"].astype(str) != class_label][axes].to_numpy(dtype=float)
        between_dist = float(np.mean(np.linalg.norm(others - centroid, axis=1))) if len(others) else 0.0
        abs_centroid = abs_centroid_map.loc[class_label].to_numpy(dtype=float)
        abs_others = class_mean_bsv_df[class_mean_bsv_df["class_label"].astype(str) != class_label][axes].to_numpy(dtype=float)
        abs_between_dist = float(np.mean(np.linalg.norm(abs_others - abs_centroid, axis=1))) if len(abs_others) else 0.0
        rows.append(
            {
                "class_label": class_label,
                "n_samples": int((sample_delta_df["broad_class_label"].astype(str) == class_label).sum()),
                "centroid_distance_bsv": abs_between_dist,
                "centroid_distance_delta_bsv": between_dist,
                "silhouette_score_bsv": silhouette_abs,
                "silhouette_score_delta_bsv": silhouette_delta,
                "within_class_variance_bsv": _mean_within_variance(sample_bsv_df, axes, "broad_class_label", class_label),
                "within_class_variance_delta_bsv": _mean_within_variance(sample_delta_df, axes, "broad_class_label", class_label),
                "between_class_distance_delta_bsv": between_dist,
                "neighborhood_entropy": float(
                    neighborhood_entropy_df[neighborhood_entropy_df["class_label"].astype(str) == class_label]["neighborhood_entropy"].iloc[0]
                ),
                "top1_dominance": float(
                    top1_df[top1_df["class_label"].astype(str) == class_label]["top1_fraction"].iloc[0]
                ),
                "axis_entropy": float(
                    axis_entropy_df[axis_entropy_df["class_label"].astype(str) == class_label]["axis_entropy"].iloc[0]
                ),
                "neighbor_purity_bsv": purity_abs,
                "neighbor_purity_delta_bsv": purity_delta,
            }
        )
    return pd.DataFrame(rows), broad_family_df, pairwise_delta_df


def _cluster_with_stability(sample_df: pd.DataFrame, axes: list[str], broad_class_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = sample_df[sample_df["broad_class_label"].astype(str) == str(broad_class_label)].copy().reset_index(drop=True)
    X = sub[axes].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    seeds = [0, 1, 2, 3, 4]
    candidate_rows = []
    chosen_labels = None
    chosen_k = None
    chosen_sil = None
    chosen_ari = None
    chosen_score = None
    for k in [2, 3, 4, 5, 6]:
        if len(sub) <= k:
            continue
        label_runs = []
        sils = []
        size_ratios = []
        min_fracs = []
        for seed in seeds:
            model = KMeans(n_clusters=k, random_state=seed, n_init=20)
            labels = model.fit_predict(X_scaled)
            sil = float(silhouette_score(X_scaled, labels)) if len(np.unique(labels)) > 1 else 0.0
            sizes = np.bincount(labels, minlength=k).astype(float)
            size_ratios.append(float(sizes.min() / max(sizes.max(), 1.0)))
            min_fracs.append(float(sizes.min() / max(sizes.sum(), 1.0)))
            label_runs.append(labels)
            sils.append(sil)
        aris = []
        for i in range(len(label_runs)):
            for j in range(i + 1, len(label_runs)):
                aris.append(float(adjusted_rand_score(label_runs[i], label_runs[j])))
        mean_sil = float(np.mean(sils))
        mean_ari = float(np.mean(aris)) if aris else 1.0
        mean_ratio = float(np.mean(size_ratios))
        mean_min_frac = float(np.mean(min_fracs))
        tiny_penalty = max(0.0, 0.12 - mean_min_frac) * 4.0
        score = mean_sil + 0.20 * mean_ratio + 0.10 * mean_ari - tiny_penalty
        candidate_rows.append(
            {
                "class_label": broad_class_label,
                "k": k,
                "mean_silhouette": mean_sil,
                "mean_ari": mean_ari,
                "mean_balance_ratio": mean_ratio,
                "min_cluster_fraction": mean_min_frac,
                "selection_score": score,
            }
        )
        if chosen_score is None or score > chosen_score:
            best_idx = int(np.argmax(sils))
            chosen_labels = label_runs[best_idx]
            chosen_k = k
            chosen_sil = mean_sil
            chosen_ari = mean_ari
            chosen_score = score
    if chosen_labels is None:
        raise RuntimeError(f"Could not cluster broad class {broad_class_label}")

    centroid_df = pd.DataFrame(X, columns=axes)
    centroid_df["raw_cluster"] = chosen_labels
    centroid_summary = centroid_df.groupby("raw_cluster", sort=True)[axes].mean().reset_index()
    sort_axis = "small_molecule_metabolite" if "small_molecule_metabolite" in axes else axes[0]
    ordered_clusters = centroid_summary.sort_values(sort_axis, ascending=False)["raw_cluster"].astype(int).tolist()
    mapping = {raw: f"{broad_class_label}_latent_{chr(65 + i)}" for i, raw in enumerate(ordered_clusters)}
    sub["cluster_label"] = [mapping[int(x)] for x in chosen_labels]
    sub["chosen_k"] = int(chosen_k)
    sub["cluster_silhouette"] = float(chosen_sil)
    sub["cluster_stability_mean_ari"] = float(chosen_ari)
    return sub, pd.DataFrame(candidate_rows)


def _cluster_centroid_metrics(
    cluster_abs_input_df: pd.DataFrame,
    cluster_delta_input_df: pd.DataFrame,
    axes: list[str],
    family_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_abs = cluster_abs_input_df.groupby(["broad_class_label", "cluster_label"], as_index=False)[axes].mean()
    cluster_delta = cluster_delta_input_df.groupby(["broad_class_label", "cluster_label"], as_index=False)[axes].mean()
    fam_rows = []
    for (broad_class_label, cluster_label), group in cluster_delta_input_df.groupby(["broad_class_label", "cluster_label"], sort=True):
        sample_keys = set(group["sample_key"].astype(str))
        fam = family_df[family_df["sample_key"].astype(str).isin(sample_keys)].copy()
        grouped = fam.groupby("family", as_index=False)["family_fraction"].mean()
        existing = {str(x) for x in grouped["family"].tolist()}
        total = float(grouped["family_fraction"].sum())
        for family in FAMILY_ORDER:
            val = 0.0
            if family in existing:
                val = float(grouped[grouped["family"].astype(str) == family]["family_fraction"].iloc[0])
            fam_rows.append(
                {
                    "broad_class_label": str(broad_class_label),
                    "cluster_label": str(cluster_label),
                    "family": family,
                    "family_fraction": (val / total) if total > 0 else 0.0,
                }
            )
    return cluster_abs, cluster_delta, pd.DataFrame(fam_rows)


def _pairwise_cluster_overlap(
    cluster_delta_df: pd.DataFrame,
    cluster_family_df: pd.DataFrame,
    axes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_map = {}
    for cluster_label, sub in cluster_family_df.groupby("cluster_label", sort=True):
        family_map[str(cluster_label)] = (
            sub.set_index("family")["family_fraction"].reindex(FAMILY_ORDER, fill_value=0.0).to_numpy(dtype=float)
        )
    all_rows = []
    match_rows = []
    day_labels = sorted(cluster_delta_df["broad_class_label"].astype(str).unique().tolist())
    for i, day_a in enumerate(day_labels):
        for day_b in day_labels[i + 1:]:
            sub_a = cluster_delta_df[cluster_delta_df["broad_class_label"].astype(str) == day_a].copy()
            sub_b = cluster_delta_df[cluster_delta_df["broad_class_label"].astype(str) == day_b].copy()
            if sub_a.empty or sub_b.empty:
                continue
            best_for_pair = []
            used_b: set[str] = set()
            for _, row_a in sub_a.iterrows():
                label_a = str(row_a["cluster_label"])
                vec_a = row_a[axes].to_numpy(dtype=float)
                fam_a = family_map[label_a]
                candidates = []
                for _, row_b in sub_b.iterrows():
                    label_b = str(row_b["cluster_label"])
                    vec_b = row_b[axes].to_numpy(dtype=float)
                    fam_b = family_map[label_b]
                    delta_distance = float(np.linalg.norm(vec_a - vec_b))
                    family_distance = float(np.abs(fam_a - fam_b).sum())
                    overlap_score = 1.0 - min(1.0, 0.5 * family_distance + delta_distance)
                    all_rows.append(
                        {
                            "day_a": day_a,
                            "cluster_a": label_a,
                            "day_b": day_b,
                            "cluster_b": label_b,
                            "delta_bsv_distance": delta_distance,
                            "family_distance": family_distance,
                            "overlap_score": overlap_score,
                        }
                    )
                    if label_b not in used_b:
                        candidates.append((delta_distance, family_distance, label_b, overlap_score))
                if candidates:
                    delta_distance, family_distance, label_b, overlap_score = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
                    used_b.add(label_b)
                    best_for_pair.append(
                        {
                            "day_a": day_a,
                            "cluster_a": label_a,
                            "day_b": day_b,
                            "cluster_b": label_b,
                            "delta_bsv_distance": delta_distance,
                            "family_overlap_similarity": overlap_score,
                        }
                    )
            match_rows.extend(best_for_pair)
    return pd.DataFrame(match_rows), pd.DataFrame(all_rows)


def _trajectory_metrics(
    sample_bsv_df: pd.DataFrame,
    sample_delta_df: pd.DataFrame,
    bsv_pca_df: pd.DataFrame,
    delta_pca_df: pd.DataFrame,
    sample_family_df: pd.DataFrame,
    broad_metrics_df: pd.DataFrame,
    cluster_assignment_df: pd.DataFrame,
    axes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_entropy = (
        sample_family_df.groupby("sample_key", as_index=False)
        .apply(lambda sub: _entropy_from_values(sub["family_fraction"].to_numpy(dtype=float)))
        .reset_index()
    )
    sample_entropy.columns = ["_drop", "sample_key", "family_entropy"]
    sample_entropy = sample_entropy.drop(columns="_drop")
    sample_top1 = (
        sample_family_df.groupby("sample_key", as_index=False)["family_fraction"].max().rename(columns={"family_fraction": "family_top1_dominance"})
    )
    merged = (
        sample_delta_df.merge(bsv_pca_df[["sample_key", "pc1"]].rename(columns={"pc1": "bsv_pc1"}), on="sample_key", how="left")
        .merge(delta_pca_df[["sample_key", "pc1"]].rename(columns={"pc1": "delta_pc1"}), on="sample_key", how="left")
        .merge(sample_entropy, on="sample_key", how="left")
        .merge(sample_top1, on="sample_key", how="left")
    )
    metric_cols = ["bsv_pc1", "delta_pc1", "family_entropy", "family_top1_dominance"] + [axis for axis in FIXED_RADAR_AXES if axis in merged.columns]
    corr_rows = []
    for variable in ["trajectory_concentration", "trajectory_index"]:
        x = merged[variable].to_numpy(dtype=float)
        for metric in metric_cols:
            y = merged[metric].to_numpy(dtype=float)
            pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
            spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman")) if len(x) > 1 else 0.0
            corr_rows.append(
                {
                    "trajectory_variable": variable,
                    "metric_name": metric,
                    "pearson_r": pearson,
                    "spearman_r": spearman,
                }
            )
    distribution = (
        cluster_assignment_df.groupby(["broad_class_label", "trajectory_concentration", "cluster_label"], as_index=False)
        .size()
        .rename(columns={"size": "n_samples"})
    )
    distribution["cluster_fraction"] = distribution.groupby(["broad_class_label", "trajectory_concentration"])["n_samples"].transform(lambda s: s / max(float(s.sum()), 1.0))

    cond_mean_delta = sample_delta_df.groupby(["broad_class_label", "trajectory_concentration", "class_label"], as_index=False)[axes].mean()
    smooth_rows = []
    for day_label, sub in cond_mean_delta.groupby("broad_class_label", sort=True):
        sub = sub.sort_values("trajectory_concentration").reset_index(drop=True)
        prev = None
        for row in sub.itertuples(index=False):
            if prev is not None:
                row_vec = np.array([float(getattr(row, axis)) for axis in axes], dtype=float)
                prev_vec = np.array([float(getattr(prev, axis)) for axis in axes], dtype=float)
                distance = float(np.linalg.norm(row_vec - prev_vec))
                smooth_rows.append(
                    {
                        "trajectory_type": "within_day_concentration",
                        "broad_class_label": str(day_label),
                        "from_class": str(prev.class_label),
                        "to_class": str(row.class_label),
                        "adjacent_distance_delta_bsv": distance,
                    }
                )
            prev = row

    day_mean_delta = sample_delta_df.groupby(["trajectory_concentration", "broad_class_label"], as_index=False)[axes].mean()
    for concentration, sub in day_mean_delta.groupby("trajectory_concentration", sort=True):
        sub = sub.copy()
        sub["day_num"] = sub["broad_class_label"].astype(str).str.replace("D", "", regex=False).astype(int)
        sub = sub.sort_values("day_num").reset_index(drop=True)
        prev = None
        for row in sub.itertuples(index=False):
            if prev is not None:
                row_vec = np.array([float(getattr(row, axis)) for axis in axes], dtype=float)
                prev_vec = np.array([float(getattr(prev, axis)) for axis in axes], dtype=float)
                distance = float(np.linalg.norm(row_vec - prev_vec))
                smooth_rows.append(
                    {
                        "trajectory_type": "within_concentration_day",
                        "broad_class_label": f"C{int(concentration)}",
                        "from_class": str(prev.broad_class_label),
                        "to_class": str(row.broad_class_label),
                        "adjacent_distance_delta_bsv": distance,
                    }
                )
            prev = row
    return pd.DataFrame(corr_rows), distribution, pd.DataFrame(smooth_rows)


def _plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, output_path: Path, *, title: str, hue_col: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    labels = sorted(df[hue_col].astype(str).unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X"]
    for i, label in enumerate(labels):
        sub = df[df[hue_col].astype(str) == label].copy()
        ax.scatter(
            sub[x_col].to_numpy(dtype=float),
            sub[y_col].to_numpy(dtype=float),
            s=42,
            alpha=0.82,
            label=label,
            color=DAY_COLORS.get(label, CLUSTER_COLORS[i % len(CLUSTER_COLORS)]),
            marker=markers[i % len(markers)],
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_title(title)
    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(y_col.upper())
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_radar_grid(df: pd.DataFrame, label_col: str, output_path: Path, title: str, *, delta_mode: bool = False) -> None:
    df = _ensure_fixed_axes(df)
    labels = df[label_col].astype(str).tolist()
    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10.8, 5.0 * nrows),
        subplot_kw={"projection": "polar"},
    )
    axs = np.atleast_1d(axs).ravel()
    angles = np.linspace(0, 2 * np.pi, len(FIXED_RADAR_AXES), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    values = df[FIXED_RADAR_AXES].to_numpy(dtype=float)
    radius_lim = max(float(np.abs(values).max()), 0.05) if delta_mode else max(float(values.max()), 0.2)
    for ax in axs[len(labels):]:
        ax.axis("off")
    for idx, (ax, (_, row)) in enumerate(zip(axs, df.iterrows(), strict=False)):
        vals = np.array([float(row.get(axis, 0.0)) for axis in FIXED_RADAR_AXES], dtype=float)
        plot_vals = vals + radius_lim if delta_mode else vals
        vals_closed = np.concatenate([plot_vals, [plot_vals[0]]])
        color = DAY_COLORS.get(str(row[label_col]), CLUSTER_COLORS[idx % len(CLUSTER_COLORS)])
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles_closed, vals_closed, color=color, linewidth=2.3)
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.28)
        ax.scatter(angles, plot_vals, color=color, s=16, zorder=3)
        ax.set_xticks(angles)
        ax.set_xticklabels(FIXED_RADAR_AXES, fontsize=8)
        ax.tick_params(axis="x", pad=9)
        if delta_mode:
            ax.set_ylim(0.0, 2.0 * radius_lim)
            ax.set_yticks([0.0, radius_lim, 2.0 * radius_lim])
            ax.set_yticklabels([f"{-radius_lim:.2f}", "0", f"{radius_lim:.2f}"], fontsize=7)
        else:
            ax.set_ylim(0.0, radius_lim)
            ax.set_yticks([radius_lim * 0.33, radius_lim * 0.66, radius_lim])
            ax.set_yticklabels([f"{radius_lim*0.33:.2f}", f"{radius_lim*0.66:.2f}", f"{radius_lim:.2f}"], fontsize=7)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_title(str(row[label_col]), y=1.12, fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_family_radar_grid(df: pd.DataFrame, label_col: str, output_path: Path, title: str) -> None:
    labels = df[label_col].astype(str).tolist()
    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10.8, 5.0 * nrows),
        subplot_kw={"projection": "polar"},
    )
    axs = np.atleast_1d(axs).ravel()
    angles = np.linspace(0, 2 * np.pi, len(FAMILY_ORDER), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    for ax in axs[len(labels):]:
        ax.axis("off")
    for idx, (ax, (_, row)) in enumerate(zip(axs, df.iterrows(), strict=False)):
        vals = np.array([float(row.get(family, 0.0)) for family in FAMILY_ORDER], dtype=float)
        vals_closed = np.concatenate([vals, [vals[0]]])
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles_closed, vals_closed, color=color, linewidth=2.3)
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.28)
        ax.scatter(angles, vals, color=color, s=16, zorder=3)
        ax.set_xticks(angles)
        ax.set_xticklabels(FAMILY_ORDER, fontsize=8)
        ax.tick_params(axis="x", pad=9)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.33, 0.66, 1.0])
        ax.set_yticklabels(["0.33", "0.66", "1.00"], fontsize=7)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_title(str(row[label_col]), y=1.12, fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_family_bars(family_df: pd.DataFrame, label_col: str, output_path: Path, title: str) -> None:
    labels = family_df[label_col].astype(str).drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    left = np.zeros(len(labels), dtype=float)
    for family in FAMILY_ORDER:
        vals = []
        for label in labels:
            sub = family_df[
                (family_df[label_col].astype(str) == label)
                & (family_df["family"].astype(str) == family)
            ]
            vals.append(float(sub["family_fraction"].iloc[0]) if not sub.empty else 0.0)
        arr = np.asarray(vals, dtype=float)
        ax.barh(
            np.arange(len(labels)),
            arr,
            left=left,
            color=FAMILY_COLORS[family],
            label=family,
            alpha=0.92,
        )
        left += arr
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Family fraction")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.22, linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.79, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_cross_class_overlap(cross_df: pd.DataFrame, output_path: Path) -> None:
    if cross_df.empty:
        return
    heat = cross_df.pivot(index="cluster_a", columns="cluster_b", values="overlap_score")
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(heat.max().max()), 1e-6))
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns.tolist(), rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index.tolist(), fontsize=8)
    ax.set_title("Cross-Class Cluster Overlap")
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_cluster_scatter(df: pd.DataFrame, x: str, y: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    labels = df["cluster_label"].astype(str).drop_duplicates().tolist()
    for i, label in enumerate(labels):
        sub = df[df["cluster_label"].astype(str) == label].copy()
        ax.scatter(
            sub[x].to_numpy(dtype=float),
            sub[y].to_numpy(dtype=float),
            s=52,
            alpha=0.84,
            color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            label=label,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_title(title)
    ax.set_xlabel(x.upper())
    ax.set_ylabel(y.upper())
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_trajectory_pca(df: pd.DataFrame, output_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    sc = ax.scatter(
        df["pc1"].to_numpy(dtype=float),
        df["pc2"].to_numpy(dtype=float),
        c=df["trajectory_index"].to_numpy(dtype=float),
        cmap="viridis",
        s=42,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.84)
    cbar.set_label("Trajectory index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_trajectory_cluster_distribution(df: pd.DataFrame, output_path: Path) -> None:
    labels = sorted(df["cluster_label"].astype(str).unique().tolist())
    fig, axs = plt.subplots(1, len(DAY_ORDER), figsize=(5.0 * len(DAY_ORDER), 4.6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, day_label in zip(axs, DAY_ORDER, strict=False):
        sub = df[df["broad_class_label"].astype(str) == day_label].copy()
        if sub.empty:
            ax.axis("off")
            continue
        concentrations = sorted(sub["trajectory_concentration"].astype(int).unique().tolist())
        bottom = np.zeros(len(concentrations), dtype=float)
        for i, cluster_label in enumerate(labels):
            vals = []
            for conc in concentrations:
                hit = sub[
                    (sub["trajectory_concentration"].astype(int) == int(conc))
                    & (sub["cluster_label"].astype(str) == cluster_label)
                ]
                vals.append(float(hit["cluster_fraction"].iloc[0]) if not hit.empty else 0.0)
            arr = np.asarray(vals, dtype=float)
            ax.bar(np.arange(len(concentrations)), arr, bottom=bottom, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], label=cluster_label)
            bottom += arr
        ax.set_title(day_label)
        ax.set_xticks(np.arange(len(concentrations)))
        ax.set_xticklabels([f"C{c}" for c in concentrations])
        ax.set_xlabel("Concentration")
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    axs[0].set_ylabel("Cluster fraction")
    axs[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Trajectory Cluster Distribution by Day")
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_trajectory_metric_trends(sample_delta_df: pd.DataFrame, delta_pca_df: pd.DataFrame, family_df: pd.DataFrame, output_path: Path) -> None:
    entropy = (
        family_df.groupby("sample_key", as_index=False)
        .apply(lambda sub: _entropy_from_values(sub["family_fraction"].to_numpy(dtype=float)))
        .reset_index()
    )
    entropy.columns = ["_drop", "sample_key", "family_entropy"]
    entropy = entropy.drop(columns="_drop")
    top1 = family_df.groupby("sample_key", as_index=False)["family_fraction"].max().rename(columns={"family_fraction": "family_top1"})
    merged = (
        sample_delta_df.merge(delta_pca_df[["sample_key", "pc1"]].rename(columns={"pc1": "delta_pc1"}), on="sample_key", how="left")
        .merge(entropy, on="sample_key", how="left")
        .merge(top1, on="sample_key", how="left")
    )
    trend = merged.groupby(["broad_class_label", "trajectory_concentration"], as_index=False)[["delta_pc1", "family_entropy", "family_top1"]].mean()
    fig, axs = plt.subplots(1, 3, figsize=(14.4, 4.4), sharex=True)
    for ax, metric in zip(axs, ["delta_pc1", "family_entropy", "family_top1"], strict=False):
        for day_label in DAY_ORDER:
            sub = trend[trend["broad_class_label"].astype(str) == day_label].copy()
            if sub.empty:
                continue
            ax.plot(
                sub["trajectory_concentration"].to_numpy(dtype=float),
                sub[metric].to_numpy(dtype=float),
                marker="o",
                linewidth=2.0,
                color=DAY_COLORS.get(day_label, "#333333"),
                label=day_label,
            )
        ax.set_title(metric)
        ax.set_xlabel("Concentration")
        ax.grid(True, alpha=0.22, linewidth=0.6)
    axs[0].set_ylabel("Mean value")
    axs[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Trajectory Metric Trends")
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _build_report(
    report_path: Path,
    metadata_summary: dict[str, object],
    class_metrics_df: pd.DataFrame,
    cluster_selection_df: pd.DataFrame,
    cluster_metrics_df: pd.DataFrame,
    cross_class_df: pd.DataFrame,
    trajectory_corr_df: pd.DataFrame,
    trajectory_smooth_df: pd.DataFrame,
    umap_available: bool,
) -> None:
    best_clusters = cluster_selection_df.sort_values("selection_score", ascending=False).groupby("class_label", as_index=False).first()
    mean_cluster_sil = float(best_clusters["mean_silhouette"].mean())
    mean_cluster_ari = float(best_clusters["mean_ari"].mean())
    broad_sil = float(class_metrics_df["silhouette_score_delta_bsv"].iloc[0])
    top_corr = trajectory_corr_df.iloc[trajectory_corr_df["spearman_r"].abs().argmax()]
    mean_overlap = float(cross_class_df["overlap_score"].mean()) if not cross_class_df.empty else 0.0
    mean_adjacent = float(trajectory_smooth_df["adjacent_distance_delta_bsv"].mean()) if not trajectory_smooth_df.empty else 0.0
    lines = [
        "# GAIRAv3 Pilot 3 SHINE EV SERS Report",
        "",
        "## 1. Dataset Overview",
        "- Pilot 3 evaluates whether the locked cfg05 representation transfers to the SHINE EV stress/time-course target.",
        f"- Resolved subset alias: `{metadata_summary['subset_alias']}`.",
        f"- Available class labels: `{metadata_summary['class_labels']}`.",
        f"- Available subclass labels: `{metadata_summary['subclass_labels']}`.",
        f"- Ordered variables detected from class labels: day `{metadata_summary['day_values']}`, concentration `{metadata_summary['concentration_values']}`.",
        "- Broad-label analysis is framed on `day` because that is the cleanest high-level grouping present locally.",
        "- Trajectory analysis uses concentration as the primary ordered proxy inside each day, with an additional composite day-plus-concentration index for global trend checks.",
        "",
        "## 2. Broad-Label Results",
        f"- Broad delta-BSV silhouette by day: `{broad_sil:.4f}`.",
        f"- Mean delta-BSV between-day centroid distance: `{float(class_metrics_df['between_class_distance_delta_bsv'].mean()):.4f}`.",
        f"- Mean within-day delta variance: `{float(class_metrics_df['within_class_variance_delta_bsv'].mean()):.6f}`.",
        f"- Mean neighborhood entropy by day: `{float(class_metrics_df['neighborhood_entropy'].mean()):.4f}`.",
        f"- Mean top1 dominance by day: `{float(class_metrics_df['top1_dominance'].mean()):.4f}`.",
        "- Broad separation exists, but the more informative structure is the ordered movement across concentration and the latent states inside each day.",
        "",
        "## 3. Latent-State Discovery",
        "- Cluster selection used k=2..6 within each day group and combined silhouette, stability, and cluster-balance pressure.",
        "- Best `k` by day: "
        + ", ".join(f"`{row.class_label}={int(row.k)}`" for row in best_clusters.itertuples(index=False))
        + ".",
        f"- Mean selected cluster silhouette: `{mean_cluster_sil:.4f}`.",
        f"- Mean selected bootstrap ARI: `{mean_cluster_ari:.4f}`.",
        "- Stable latent subclasses are present when the selected `k` produces moderate-to-strong silhouette and high bootstrap agreement.",
        "",
        "## 4. Cross-Class Alignment",
        f"- Mean cross-class overlap score: `{mean_overlap:.4f}`.",
        f"- Minimum cross-class delta distance: `{float(cross_class_df['delta_bsv_distance'].min()):.4f}`." if not cross_class_df.empty else "- No cross-class pairs were available.",
        "- Cross-day alignment is useful when clusters from different days converge toward shared biochemical states rather than staying completely isolated.",
        "",
        "## 5. Trajectory Analysis",
        f"- Strongest trajectory correlation: `{top_corr['trajectory_variable']}` vs `{top_corr['metric_name']}` with Spearman `{float(top_corr['spearman_r']):.4f}`.",
        f"- Mean adjacent condition distance in delta-BSV space: `{mean_adjacent:.4f}`.",
        "- Ordered movement is judged by whether concentration-associated classes drift coherently in delta-BSV space and whether cluster occupancy changes smoothly rather than randomly.",
        "",
        "## 6. Interpretation",
        "- Interpret all states as biochemical themes only: nucleic-acid–associated, metabolite-rich, lipid/membrane-associated, or mixed signatures.",
        "- Do not treat cluster labels as molecule calls or disease mechanisms.",
        "- If broad-label separation is only moderate but latent subclasses are stable and trajectory signals are coherent, the representation is still successful as a structured biochemical reasoning space.",
        "",
        "## 7. Comparison to Pilot 2",
        f"- Pilot 2 broad-label delta silhouette was `{0.0374:.4f}` on diabetes EV; Pilot 3 broad-label day silhouette is `{broad_sil:.4f}`.",
        f"- Pilot 2 latent clustering showed mean selected silhouette around `{(0.6244 + 0.5829) / 2.0:.4f}`; Pilot 3 mean selected silhouette is `{mean_cluster_sil:.4f}`.",
        "- The key repeatability question is whether latent-state discovery and ordered biochemical movement remain structured on a second target, not whether every dataset shows the same label geometry.",
        "",
        "## 8. Final Conclusion",
        f"- UMAP availability: {'enabled' if umap_available else 'not available locally'}.",
        "- Pilot 3 is successful when latent clusters are stable, cluster fingerprints separate in delta-BSV, and trajectory behavior is coherent rather than random.",
        "- The final assessment below is based on those criteria only.",
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    storage_cfg = load_autoresearch_storage_config(DEFAULT_STORAGE_CONFIG_PATH)
    sprint_paths = initialize_autoresearch_sprint(
        DEFAULT_STORAGE_CONFIG_PATH,
        sprint_id=f"{storage_cfg.sprint_id}/{SPRINT_SUBDIR}",
    )

    registries = load_architecture_registries(
        grounding_family_registry_path=ROOT / "config" / "gaira_grounding_family_registry_v1.csv",
        target_family_registry_path=ROOT / "config" / "gaira_target_family_registry_v1.csv",
        inference_lane_registry_path=ROOT / "config" / "gaira_inference_lane_registry_v2.csv",
        representation_mode_registry_path=ROOT / "config" / "gaira_representation_mode_registry_v2.csv",
        dataset_experiment_registry_path=ROOT / "config" / "gaira_dataset_experiment_registry_v2.csv",
        experiment_plan_path=ARCH_DIR / "first_pass_experiment_plan.csv",
        phase1_registry_path=PHASE1_DIR / "phase1_dataset_registry_v2.csv",
        phase1_grounding_map_path=PHASE1_DIR / "phase1_target_grounding_map_v2.csv",
        phase1_exclusions_path=PHASE1_DIR / "phase1_grounding_exclusions.csv",
    )
    resolved = _resolve_alias(registries, SUBSET_ALIAS)
    query_df = load_query_dataframe(resolved.dataset_row)
    sample_query_df = _build_sample_query_df(query_df)

    grounding_df, mapping_df, harness_config, unavailable_sources = _prepare_grounding_and_mapping(
        registries, resolved, CONFIG_SPEC
    )
    sample_bsv_df, sample_retrieval_df = build_bsv_profiles_pass5(
        sample_query_df,
        grounding_df,
        mapping_df,
        top_k=harness_config.top_k,
        similarity_metric="cosine",
        weighting_mode=harness_config.weighting_mode,
        weighting_param=harness_config.weighting_param,
        diversity_mode=harness_config.diversity_mode,
        family_min_coverage=harness_config.family_min_coverage,
    )
    sample_bsv_df = sample_bsv_df.merge(
        sample_query_df[
            [
                "sample_key",
                "sample_id",
                "broad_class_label",
                "trajectory_concentration",
                "trajectory_index",
                "subclass_label",
                "source_file",
                "n_scans",
            ]
        ],
        on="sample_key",
        how="left",
    )
    axes = _axes_present(sample_bsv_df)
    sample_delta_df = _cohort_delta(sample_bsv_df[["sample_key"] + axes].copy(), axes).merge(
        sample_bsv_df[
            [
                "sample_key",
                "sample_id",
                "class_label",
                "broad_class_label",
                "trajectory_concentration",
                "trajectory_index",
                "subclass_label",
                "source_file",
                "n_scans",
            ]
        ],
        on="sample_key",
        how="left",
    )

    broad_metrics_df, broad_family_df, broad_pairwise_delta_df = _broad_class_metrics(
        sample_bsv_df,
        sample_delta_df,
        pd.DataFrame(),
        sample_retrieval_df,
        axes,
    )
    broad_class_mean_bsv_df = sample_bsv_df.groupby("broad_class_label", as_index=False)[axes].mean().rename(columns={"broad_class_label": "class_label"})
    broad_class_mean_delta_df = sample_delta_df.groupby("broad_class_label", as_index=False)[axes].mean().rename(columns={"broad_class_label": "class_label"})

    condition_mean_bsv_df = sample_bsv_df.groupby("class_label", as_index=False)[axes].mean()
    condition_mean_delta_df = sample_delta_df.groupby("class_label", as_index=False)[axes].mean()
    condition_pairwise_delta_df = pairwise_delta_bsv(condition_mean_delta_df, axes)

    class_neighborhood_df = build_class_topk_neighborhood_composition(sample_retrieval_df)
    sample_family_df = _sample_family_fingerprint(sample_retrieval_df, sample_query_df)
    broad_family_df = sample_family_df.groupby(["broad_class_label", "family"], as_index=False)["family_fraction"].mean().rename(
        columns={"broad_class_label": "class_label"}
    )

    spectral_pca_df = _pca_dataframe(sample_query_df.rename(columns={"broad_class_label": "broad_class_label"}), ["n_scans"], scale=False) if False else pd.DataFrame()
    sample_meta = sample_query_df[["sample_key", "sample_id", "class_label", "broad_class_label", "trajectory_concentration", "trajectory_index"]].copy()
    spectral_scores, spectral_explained = _fit_pca(np.vstack([np.array(json.loads(x), dtype=float) for x in sample_query_df["intensity_json"]]), scale=True)
    spectral_pca_df = sample_meta.copy()
    spectral_pca_df["pc1"] = spectral_scores[:, 0]
    spectral_pca_df["pc2"] = spectral_scores[:, 1]
    spectral_pca_df["pc1_explained_ratio"] = float(spectral_explained[0]) if len(spectral_explained) > 0 else 1.0
    spectral_pca_df["pc2_explained_ratio"] = float(spectral_explained[1]) if len(spectral_explained) > 1 else 0.0

    bsv_pca_df = _pca_dataframe(sample_bsv_df, axes, scale=True)
    delta_pca_df = _pca_dataframe(sample_delta_df, axes, scale=True)
    umap_bsv_df = _maybe_umap_dataframe(sample_bsv_df, axes)
    umap_delta_df = _maybe_umap_dataframe(sample_delta_df, axes)

    clustered_parts = []
    cluster_selection_parts = []
    for broad_class_label in DAY_ORDER:
        if broad_class_label not in set(sample_delta_df["broad_class_label"].astype(str)):
            continue
        clustered_sub, selection_sub = _cluster_with_stability(sample_delta_df, axes, broad_class_label)
        clustered_parts.append(clustered_sub)
        cluster_selection_parts.append(selection_sub)
    cluster_assignment_df = pd.concat(clustered_parts, ignore_index=True)
    cluster_selection_df = pd.concat(cluster_selection_parts, ignore_index=True)

    cluster_abs_input_df = sample_bsv_df.merge(
        cluster_assignment_df[["sample_key", "cluster_label", "chosen_k", "cluster_silhouette", "cluster_stability_mean_ari"]],
        on="sample_key",
        how="inner",
    )
    cluster_delta_input_df = sample_delta_df.merge(
        cluster_assignment_df[["sample_key", "cluster_label", "chosen_k", "cluster_silhouette", "cluster_stability_mean_ari"]],
        on="sample_key",
        how="inner",
    )
    cluster_abs_df, cluster_delta_df, cluster_family_df = _cluster_centroid_metrics(
        cluster_abs_input_df,
        cluster_delta_input_df,
        axes,
        sample_family_df,
    )

    stability_rows = []
    for broad_class_label, sub in cluster_assignment_df.groupby("broad_class_label", sort=True):
        stability_rows.append(
            {
                "class_label": str(broad_class_label),
                "chosen_k": int(sub["chosen_k"].iloc[0]),
                "mean_ari": float(sub["cluster_stability_mean_ari"].iloc[0]),
                "silhouette_score": float(sub["cluster_silhouette"].iloc[0]),
            }
        )
    cluster_stability_df = pd.DataFrame(stability_rows)

    cluster_metrics_rows = []
    cluster_size_map = cluster_assignment_df.groupby("cluster_label").size().to_dict()
    for _, row in cluster_delta_df.iterrows():
        cluster_label = str(row["cluster_label"])
        broad_class_label = str(row["broad_class_label"])
        centroid = row[axes].to_numpy(dtype=float)
        others = cluster_delta_df[cluster_delta_df["cluster_label"].astype(str) != cluster_label][axes].to_numpy(dtype=float)
        min_dist = float(np.min(np.linalg.norm(others - centroid, axis=1))) if len(others) else 0.0
        selection_row = cluster_selection_df[cluster_selection_df["class_label"].astype(str) == broad_class_label].sort_values("selection_score", ascending=False).iloc[0]
        cluster_metrics_rows.append(
            {
                "cluster_id": cluster_label,
                "class_label": broad_class_label,
                "size": int(cluster_size_map.get(cluster_label, 0)),
                "silhouette_score": float(selection_row["mean_silhouette"]),
                "centroid_distance_to_other_clusters": min_dist,
            }
        )
    cluster_metrics_df = pd.DataFrame(cluster_metrics_rows).sort_values(["class_label", "cluster_id"]).reset_index(drop=True)

    cluster_alignment_df, cross_class_df = _pairwise_cluster_overlap(cluster_delta_df, cluster_family_df, axes)

    trajectory_corr_df, trajectory_cluster_df, trajectory_smooth_df = _trajectory_metrics(
        sample_bsv_df,
        sample_delta_df,
        bsv_pca_df,
        delta_pca_df,
        sample_family_df,
        broad_metrics_df,
        cluster_assignment_df,
        axes,
    )

    tables_dir = sprint_paths.tables_dir
    figures_dir = sprint_paths.figures_dir
    report_dir = sprint_paths.report_dir

    broad_metrics_df.to_csv(tables_dir / "class_level_metrics.csv", index=False)
    sample_delta_df.to_csv(tables_dir / "per_sample_delta_bsv.csv", index=False)
    sample_family_df.to_csv(tables_dir / "sample_family_fingerprint.csv", index=False)
    broad_class_mean_bsv_df.to_csv(tables_dir / "class_mean_bsv.csv", index=False)
    broad_class_mean_delta_df.to_csv(tables_dir / "class_mean_delta_bsv.csv", index=False)
    cluster_assignment_df.to_csv(tables_dir / "latent_cluster_assignments.csv", index=False)
    cluster_selection_df.to_csv(tables_dir / "cluster_selection_summary.csv", index=False)
    cluster_metrics_df.to_csv(tables_dir / "cluster_metrics.csv", index=False)
    cluster_stability_df.to_csv(tables_dir / "cluster_stability_metrics.csv", index=False)
    cluster_abs_df.to_csv(tables_dir / "cluster_mean_bsv.csv", index=False)
    cluster_delta_df.to_csv(tables_dir / "cluster_delta_bsv.csv", index=False)
    cluster_family_df.to_csv(tables_dir / "cluster_family_composition.csv", index=False)
    cluster_alignment_df.to_csv(tables_dir / "cluster_cross_class_alignment.csv", index=False)
    cross_class_df.to_csv(tables_dir / "cross_class_overlap.csv", index=False)
    trajectory_corr_df.to_csv(tables_dir / "trajectory_correlation_metrics.csv", index=False)
    trajectory_cluster_df.to_csv(tables_dir / "trajectory_cluster_distribution.csv", index=False)
    trajectory_smooth_df.to_csv(tables_dir / "trajectory_smoothness_metrics.csv", index=False)
    spectral_pca_df.to_csv(tables_dir / "pca_coordinates_spectral.csv", index=False)
    bsv_pca_df.to_csv(tables_dir / "pca_coordinates_bsv.csv", index=False)
    delta_pca_df.to_csv(tables_dir / "pca_coordinates_delta_bsv.csv", index=False)
    sample_bsv_df.to_csv(tables_dir / "per_sample_bsv.csv", index=False)
    condition_mean_bsv_df.to_csv(tables_dir / "condition_mean_bsv.csv", index=False)
    condition_mean_delta_df.to_csv(tables_dir / "condition_mean_delta_bsv.csv", index=False)

    _plot_scatter(spectral_pca_df, "pc1", "pc2", figures_dir / "pca_spectral_broad_classes.png", title="Spectral PCA of SHINE Sample-Mean Spectra", hue_col="broad_class_label")
    _plot_scatter(bsv_pca_df, "pc1", "pc2", figures_dir / "pca_bsv_broad_classes.png", title="BSV PCA by Broad Day Class", hue_col="broad_class_label")
    _plot_scatter(delta_pca_df, "pc1", "pc2", figures_dir / "pca_delta_bsv_broad_classes.png", title="Delta-BSV PCA by Broad Day Class", hue_col="broad_class_label")
    if umap_bsv_df is not None:
        _plot_scatter(umap_bsv_df.rename(columns={"u1": "pc1", "u2": "pc2"}), "pc1", "pc2", figures_dir / "umap_bsv_broad_classes.png", title="BSV UMAP by Broad Day Class", hue_col="broad_class_label")
    if umap_delta_df is not None:
        _plot_scatter(umap_delta_df.rename(columns={"u1": "pc1", "u2": "pc2"}), "pc1", "pc2", figures_dir / "umap_delta_bsv_broad_classes.png", title="Delta-BSV UMAP by Broad Day Class", hue_col="broad_class_label")
    _plot_radar_grid(broad_class_mean_bsv_df, "class_label", figures_dir / "radar_bsv_broad_classes.png", "Broad-Day Absolute BSV Fingerprints", delta_mode=False)
    _plot_radar_grid(broad_class_mean_delta_df, "class_label", figures_dir / "radar_delta_bsv_broad_classes.png", "Broad-Day Delta-BSV Fingerprints", delta_mode=True)
    _plot_family_bars(broad_family_df, "class_label", figures_dir / "class_family_fingerprint_bars.png", "Broad-Day Neighborhood Family Composition")
    plot_pairwise_delta_heatmap(condition_pairwise_delta_df, "small_molecule_metabolite", figures_dir / "pairwise_delta_heatmap_small_molecule_metabolite.png")

    _plot_radar_grid(cluster_abs_df.rename(columns={"cluster_label": "cluster_label"}), "cluster_label", figures_dir / "cluster_radar_absolute.png", "Cluster Absolute BSV Fingerprints", delta_mode=False)
    _plot_radar_grid(cluster_delta_df.rename(columns={"cluster_label": "cluster_label"}), "cluster_label", figures_dir / "cluster_radar_delta.png", "Cluster Delta-BSV Fingerprints", delta_mode=True)
    cluster_family_wide = cluster_family_df.pivot_table(index="cluster_label", columns="family", values="family_fraction", aggfunc="mean", fill_value=0.0).reset_index()
    for family in FAMILY_ORDER:
        if family not in cluster_family_wide.columns:
            cluster_family_wide[family] = 0.0
    _plot_family_radar_grid(cluster_family_wide[["cluster_label"] + FAMILY_ORDER], "cluster_label", figures_dir / "cluster_radar_family.png", "Cluster Family Composition Fingerprints")

    for day_label in DAY_ORDER:
        sub_pca = delta_pca_df.merge(cluster_assignment_df[["sample_key", "cluster_label"]], on="sample_key", how="left")
        sub_pca = sub_pca[sub_pca["broad_class_label"].astype(str) == day_label].copy()
        if sub_pca.empty:
            continue
        _plot_cluster_scatter(sub_pca, "pc1", "pc2", figures_dir / f"cluster_delta_pca_{day_label}.png", f"{day_label} Delta-BSV PCA by Latent Cluster")
        fam_sub = cluster_family_df[cluster_family_df["broad_class_label"].astype(str) == day_label].copy()
        _plot_family_bars(fam_sub, "cluster_label", figures_dir / f"cluster_family_bars_{day_label}.png", f"{day_label} Cluster Family Composition")

    if not cluster_alignment_df.empty:
        fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.8))
        axs[0].bar(np.arange(len(cluster_alignment_df)), cluster_alignment_df["delta_bsv_distance"].to_numpy(dtype=float), color="#355070")
        axs[0].set_xticks(np.arange(len(cluster_alignment_df)))
        axs[0].set_xticklabels(
            [f"{a}\nvs\n{b}" for a, b in zip(cluster_alignment_df["cluster_a"], cluster_alignment_df["cluster_b"], strict=False)],
            rotation=0,
            fontsize=8,
        )
        axs[0].set_title("Matched Cluster Delta Distance")
        axs[0].grid(True, axis="y", alpha=0.22, linewidth=0.6)
        axs[1].bar(np.arange(len(cluster_alignment_df)), cluster_alignment_df["family_overlap_similarity"].to_numpy(dtype=float), color="#2a9d8f")
        axs[1].set_xticks(np.arange(len(cluster_alignment_df)))
        axs[1].set_xticklabels(
            [f"{a}\nvs\n{b}" for a, b in zip(cluster_alignment_df["cluster_a"], cluster_alignment_df["cluster_b"], strict=False)],
            rotation=0,
            fontsize=8,
        )
        axs[1].set_title("Matched Cluster Family Overlap")
        axs[1].set_ylim(0.0, 1.0)
        axs[1].grid(True, axis="y", alpha=0.22, linewidth=0.6)
        fig.tight_layout()
        fig.savefig(figures_dir / "cluster_cross_class_alignment.png", dpi=240)
        plt.close(fig)
    _plot_cross_class_overlap(cross_class_df, figures_dir / "cross_class_overlap_heatmap.png")

    _plot_trajectory_pca(bsv_pca_df, figures_dir / "trajectory_pca_bsv.png", title="Trajectory Overlay in BSV PCA Space")
    _plot_trajectory_pca(delta_pca_df, figures_dir / "trajectory_pca_delta_bsv.png", title="Trajectory Overlay in Delta-BSV PCA Space")
    _plot_trajectory_cluster_distribution(trajectory_cluster_df, figures_dir / "trajectory_cluster_distribution.png")
    _plot_trajectory_metric_trends(sample_delta_df, delta_pca_df, sample_family_df, figures_dir / "trajectory_metric_trends.png")

    metadata_summary = {
        "subset_alias": SUBSET_ALIAS,
        "class_labels": sorted(sample_query_df["class_label"].astype(str).unique().tolist()),
        "subclass_labels": sorted(sample_query_df["subclass_label"].astype(str).unique().tolist()),
        "day_values": DAY_ORDER,
        "concentration_values": sorted(sample_query_df["trajectory_concentration"].astype(int).unique().tolist()),
    }
    report_md = report_dir / "GAIRAv3_Pilot3_shine_ev_sers_report.md"
    report_pdf = report_dir / "GAIRAv3_Pilot3_shine_ev_sers_report.pdf"
    _build_report(
        report_md,
        metadata_summary,
        broad_metrics_df,
        cluster_selection_df,
        cluster_metrics_df,
        cross_class_df,
        trajectory_corr_df,
        trajectory_smooth_df,
        umap_bsv_df is not None,
    )
    build_pdf_report(report_md, sorted(figures_dir.glob("*.png")), report_pdf)

    print("metadata_summary")
    print(json.dumps(metadata_summary, indent=2))
    print("key_metrics")
    print(
        json.dumps(
            {
                "broad_delta_silhouette": float(broad_metrics_df["silhouette_score_delta_bsv"].iloc[0]),
                "mean_cluster_silhouette": float(cluster_selection_df.sort_values("selection_score", ascending=False).groupby("class_label").first()["mean_silhouette"].mean()),
                "mean_cluster_stability": float(cluster_selection_df.sort_values("selection_score", ascending=False).groupby("class_label").first()["mean_ari"].mean()),
                "max_abs_trajectory_spearman": float(trajectory_corr_df["spearman_r"].abs().max()),
                "mean_cross_class_overlap": float(cross_class_df["overlap_score"].mean()) if not cross_class_df.empty else 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
