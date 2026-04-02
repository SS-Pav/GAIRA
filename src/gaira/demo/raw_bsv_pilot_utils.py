from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from gaira.config import get_database_path
from gaira.demo.ev_analysis_utils import align_to_master_grid


PRIMARY_AXES = [
    "protein_peptide",
    "lipid_membrane",
    "nucleic_acid",
    "carbohydrate_glycan",
    "small_molecule_metabolite",
]
CAVEAT_AXES = [
    "matrix_background",
    "substrate_adsorption_bias",
    "protocol_sensitive_signal",
]
ALL_AXES = PRIMARY_AXES + CAVEAT_AXES
UNIVERSAL_PURE_SOURCE_KEYS = {
    "ramanbiolib",
    "adenine_sers_control",
    "amino_acid_raman_grounding",
    "metabolite_sers63_support",
}


@dataclass
class RegistryInputs:
    registry: pd.DataFrame
    target_grounding_map: pd.DataFrame
    exclusions: pd.DataFrame


@dataclass
class SourceSpec:
    dataset_id: str
    subset_id: str
    source_key: str
    source_type: str


def classify_source_bucket(source_key: str) -> str:
    return "universal_pure_grounding" if str(source_key) in UNIVERSAL_PURE_SOURCE_KEYS else "serum_support_grounding"


def load_registry_inputs(
    registry_path: Path,
    grounding_map_path: Path,
    exclusions_path: Path,
) -> RegistryInputs:
    for path in [registry_path, grounding_map_path, exclusions_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required Phase 1 registry input missing: {path}")
    return RegistryInputs(
        registry=pd.read_csv(registry_path),
        target_grounding_map=pd.read_csv(grounding_map_path),
        exclusions=pd.read_csv(exclusions_path),
    )


def get_registry_row_by_alias(registry_df: pd.DataFrame, subset_alias: str) -> pd.Series:
    matches = registry_df[registry_df["subset_alias"].astype(str) == str(subset_alias)].copy()
    if matches.empty:
        raise KeyError(f"Unknown subset alias: {subset_alias}")
    if len(matches) > 1:
        raise RuntimeError(f"Subset alias is not unique: {subset_alias}")
    return matches.iloc[0]


def choose_demo_alias(registry_df: pd.DataFrame) -> str:
    priority = [
        "cspp_metabolite_spike_validation",
        "serum_ag_uricase_validation",
        "serum_erg_calibration_validation",
    ]
    aliases = set(registry_df["subset_alias"].astype(str))
    for alias in priority:
        if alias in aliases:
            return alias
    raise RuntimeError("No preferred validation alias available for the raw/direct BSV pilot.")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def correlation_normalize_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    return normalize_rows(centered)


def choose_processing_version(dataset_id: str, *, source_type: str) -> str:
    table = "grounding_processed_spectra" if source_type == "grounding" else "biosample_processed_spectra"
    con = duckdb.connect(str(get_database_path()), read_only=True)
    try:
        df = con.execute(
            f"""
            select processing_version, count(*) as n
            from {table}
            where dataset_id = ?
            group by 1
            order by 1
            """,
            [dataset_id],
        ).fetchdf()
    finally:
        con.close()
    if df.empty:
        raise RuntimeError(f"No processed spectra found for {dataset_id} in {table}.")
    versions = df["processing_version"].astype(str).tolist()
    poly3 = [v for v in versions if "poly3" in v and "vector" in v]
    if poly3:
        return sorted(poly3)[-1]
    vector = [v for v in versions if "vector" in v]
    if vector:
        return sorted(vector)[-1]
    return sorted(versions)[-1]


def decode_and_align(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_arrays = [np.asarray(json.loads(value), dtype=float) for value in df["wavenumbers_json"]]
    y_arrays = [np.asarray(json.loads(value), dtype=float) for value in df["intensity_json"]]
    master_x = np.unique(np.concatenate(x_arrays)).astype(np.float32)
    matrix = np.vstack(
        [align_to_master_grid(x, y, master_x) for x, y in zip(x_arrays, y_arrays, strict=False)]
    ).astype(np.float32)
    return master_x, matrix


def compute_local_pca(
    df: pd.DataFrame,
    *,
    n_components: int = 3,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    master_x, matrix = decode_and_align(df)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    max_components = min(n_components, vt.shape[0], matrix.shape[0])
    coords = u[:, :max_components] * s[:max_components]
    explained = (s**2) / np.maximum((s**2).sum(), 1e-8)
    coord_df = pd.DataFrame(
        {
            "sample_key": df["sample_key"].astype(str).tolist(),
            "class_label": df["class_label"].astype(str).tolist(),
            "subset_id": df["subclass_label"].astype(str).tolist(),
        }
    )
    for i in range(max_components):
        coord_df[f"pc{i+1}"] = coords[:, i]
        coord_df[f"pc{i+1}_explained_ratio"] = float(explained[i])
    return master_x, matrix, coord_df


def build_group_mean_query_df(
    query_df: pd.DataFrame,
    *,
    group_col: str = "class_label",
) -> pd.DataFrame:
    master_x, matrix = decode_and_align(query_df)
    work = query_df.reset_index(drop=True).copy()
    rows = []
    for group_value, group in work.groupby(group_col, sort=True):
        idx = group.index.to_numpy()
        mean_vector = matrix[idx].mean(axis=0)
        rows.append(
            {
                "sample_key": f"group_mean__{group_col}__{group_value}",
                "dataset_id": str(group["dataset_id"].iloc[0]),
                "processing_version": str(group["processing_version"].iloc[0]),
                "wavenumbers_json": json.dumps(master_x.astype(float).tolist()),
                "intensity_json": json.dumps(mean_vector.astype(float).tolist()),
                "class_label": str(group_value),
                "subclass_label": str(group["subclass_label"].iloc[0]),
                "source_file": "__group_mean__",
                "source_key": str(group["source_key"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def infer_background_label(class_label: str) -> str:
    text = str(class_label)
    if "+Enzyme" in text:
        return text.replace("+Enzyme", "")
    return text


def is_perturbed_label(class_label: str) -> bool:
    return infer_background_label(class_label) != str(class_label)


def build_differential_query_df(
    query_df: pd.DataFrame,
    *,
    baseline_policy: str,
    n_pca_components: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if baseline_policy not in {"mean_background", "nearest_background"}:
        raise ValueError(f"Unsupported baseline policy for local differential build: {baseline_policy}")

    master_x, matrix, pca_df = compute_local_pca(query_df, n_components=n_pca_components)
    meta = query_df.reset_index(drop=True).copy()
    meta["base_label"] = meta["class_label"].astype(str).map(infer_background_label)
    meta["is_perturbed"] = meta["class_label"].astype(str).map(is_perturbed_label)
    for col in [c for c in pca_df.columns if c.startswith("pc")]:
        meta[col] = pca_df[col].values

    treated_idx = meta.index[meta["is_perturbed"]].tolist()
    if not treated_idx:
        raise RuntimeError("No perturbed samples found for differential baseline construction.")

    pair_rows = []
    residual_rows = []
    zero_axis = json.dumps(master_x.astype(float).tolist())
    for idx in treated_idx:
        row = meta.loc[idx]
        pool = meta[(meta["is_perturbed"] == False) & (meta["class_label"].astype(str) == str(row["base_label"]))].copy()
        if pool.empty:
            raise RuntimeError(f"No untreated background pool found for perturbed class {row['class_label']}")
        pool_indices = pool.index.to_numpy()
        if baseline_policy == "nearest_background":
            pca_cols = [c for c in meta.columns if re.fullmatch(r"pc\d+", c)]
            target = row[pca_cols].to_numpy(dtype=float)
            pool_coords = pool[pca_cols].to_numpy(dtype=float)
            distances = np.linalg.norm(pool_coords - target[None, :], axis=1)
            best_pos = int(np.argmin(distances))
            matched_idx = int(pool_indices[best_pos])
            baseline_vector = matrix[matched_idx]
            matched_distance = float(distances[best_pos])
            baseline_kind = "nearest_background"
        else:
            matched_idx = -1
            baseline_vector = matrix[pool_indices].mean(axis=0)
            matched_distance = float("nan")
            baseline_kind = "mean_background"
        residual = matrix[idx] - baseline_vector
        matched_sample_key = str(meta.loc[matched_idx, "sample_key"]) if matched_idx >= 0 else "__mean_background__"
        matched_class_label = str(meta.loc[matched_idx, "class_label"]) if matched_idx >= 0 else str(row["base_label"])
        pair_rows.append(
            {
                "query_sample_key": str(row["sample_key"]),
                "query_class_label": str(row["class_label"]),
                "base_background_label": str(row["base_label"]),
                "baseline_policy": baseline_policy,
                "matched_background_sample_key": matched_sample_key,
                "matched_background_class_label": matched_class_label,
                "background_pool_size": int(len(pool_indices)),
                "pca_distance": matched_distance,
                "pc1": float(row.get("pc1", np.nan)),
                "pc2": float(row.get("pc2", np.nan)),
                "pc3": float(row.get("pc3", np.nan)),
            }
        )
        residual_rows.append(
            {
                "sample_key": f"{row['sample_key']}__residual",
                "dataset_id": str(row["dataset_id"]),
                "processing_version": str(row["processing_version"]),
                "wavenumbers_json": zero_axis,
                "intensity_json": json.dumps(residual.astype(float).tolist()),
                "class_label": str(row["class_label"]),
                "subclass_label": str(row["subclass_label"]),
                "source_file": str(row.get("source_file", "")),
                "source_key": str(row.get("source_key", "")),
                "matched_background_sample_key": matched_sample_key,
                "matched_background_class_label": matched_class_label,
                "baseline_policy": baseline_policy,
            }
        )
    return pd.DataFrame(residual_rows), pd.DataFrame(pair_rows), pca_df


def delta_from_residual_group_means(
    group_means: pd.DataFrame,
    *,
    zero_reference_label: str = "matched_background_zero",
    axis_names: list[str] | None = None,
) -> pd.DataFrame:
    if axis_names is None:
        axis_names = [axis for axis in ALL_AXES if axis in group_means.columns]
    zero_row = {"class_label": zero_reference_label}
    for axis in axis_names + ["unmapped_support"]:
        zero_row[axis] = 0.0
    with_zero = pd.concat([group_means, pd.DataFrame([zero_row])], ignore_index=True)
    return delta_bsv(with_zero, reference_group=zero_reference_label, axis_names=axis_names)


def load_biosample_subset(dataset_id: str, subset_id: str) -> pd.DataFrame:
    processing_version = choose_processing_version(dataset_id, source_type="biosample")
    query = """
        select
          p.processed_id as sample_key,
          p.dataset_id,
          p.processing_version,
          p.wavenumbers_json,
          p.intensity_json,
          m.class_label,
          m.subclass_label,
          m.source_file
        from biosample_processed_spectra p
        join biosample_metadata m
          on p.biosample_id = m.biosample_id
         and p.dataset_id = m.dataset_id
        where p.dataset_id = ?
          and p.processing_version = ?
    """
    params: list[object] = [dataset_id, processing_version]
    if subset_id != "all":
        query += " and m.subclass_label = ?"
        params.append(subset_id)
    query += " order by m.subclass_label, m.class_label, p.processed_id"
    with duckdb.connect(str(get_database_path()), read_only=True) as con:
        df = con.execute(query, params).fetchdf()
    if df.empty:
        raise RuntimeError(f"No biosample processed spectra found for {dataset_id}::{subset_id}")
    df["source_key"] = dataset_id + "::" + df["subclass_label"].astype(str)
    return df


def load_grounding_dataset(dataset_id: str) -> pd.DataFrame:
    processing_version = choose_processing_version(dataset_id, source_type="grounding")
    query = """
        select
          p.processed_id as sample_key,
          p.dataset_id,
          p.processing_version,
          p.wavenumbers_json,
          p.intensity_json,
          m.class_label,
          m.compound_label,
          m.experiment_family,
          m.grounding_role,
          m.source_file
        from grounding_processed_spectra p
        join grounding_metadata m
          on p.grounding_id = m.grounding_id
         and p.dataset_id = m.dataset_id
        where p.dataset_id = ?
          and p.processing_version = ?
        order by p.processed_id
    """
    with duckdb.connect(str(get_database_path()), read_only=True) as con:
        df = con.execute(query, [dataset_id, processing_version]).fetchdf()
    if df.empty:
        raise RuntimeError(f"No grounding processed spectra found for {dataset_id}")
    df["subclass_label"] = "all"
    df["source_key"] = dataset_id
    return df


def parse_source_spec(source: str, registry_df: pd.DataFrame) -> SourceSpec:
    source = str(source).strip()
    if "::" in source:
        dataset_id, subset_id = source.split("::", 1)
        return SourceSpec(dataset_id=dataset_id, subset_id=subset_id, source_key=source, source_type="biosample_support")
    row = registry_df[registry_df["dataset_id"].astype(str) == source].copy()
    if row.empty:
        raise KeyError(f"Grounding source {source} not found in registry.")
    role = row.iloc[0]["proposed_phase1_role"]
    source_type = "grounding" if str(role).startswith("grounding_reference") else "biosample_support"
    return SourceSpec(dataset_id=source, subset_id="all", source_key=source, source_type=source_type)


def derive_grounding_sources_for_alias(alias: str, inputs: RegistryInputs) -> list[str]:
    target_rows = inputs.target_grounding_map[inputs.target_grounding_map["target_alias"].astype(str) == alias].copy()
    if not target_rows.empty:
        raw_sources = str(target_rows.iloc[0]["allowed_grounding_sources"])
        return [part.strip() for part in raw_sources.split(";") if part.strip()]

    row = get_registry_row_by_alias(inputs.registry, alias)
    sample_type = str(row["sample_type"])
    registry = inputs.registry.copy()
    universal = registry[
        registry["proposed_phase1_role"].astype(str) == "grounding_reference_universal_pure"
    ].copy()
    serum_support = registry[
        registry["proposed_phase1_role"].astype(str).isin(
            ["grounding_reference_serum_support", "support_grounding_only_subset"]
        )
    ].copy()
    parts = []
    for _, rec in universal.iterrows():
        parts.append(_registry_source_string(rec))
    if sample_type == "serum":
        for _, rec in serum_support.iterrows():
            parts.append(_registry_source_string(rec))
    return sorted(dict.fromkeys(parts))


def _registry_source_string(row: pd.Series) -> str:
    dataset_id = str(row["dataset_id"])
    subset_id = str(row["subset_id"])
    if subset_id == "all":
        return dataset_id
    return f"{dataset_id}::{subset_id}"


def apply_source_role_policy(
    mapping_df: pd.DataFrame,
    grounding_df: pd.DataFrame,
    *,
    primary_sources: set[str],
    caveat_only_sources: set[str],
    primary_axis_names: Iterable[str] | None = None,
    caveat_axis_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    if primary_axis_names is None:
        primary_axis_names = PRIMARY_AXES
    if caveat_axis_names is None:
        caveat_axis_names = CAVEAT_AXES
    primary_axis_names = list(primary_axis_names)
    caveat_axis_names = list(caveat_axis_names)
    source_lookup = grounding_df.set_index("sample_key")["source_key"].astype(str).to_dict()
    work = mapping_df.copy()
    work["source_key"] = work["sample_key"].astype(str).map(source_lookup)
    work["contribution_role"] = "excluded"
    work.loc[work["source_key"].isin(primary_sources), "contribution_role"] = "primary_biochemical"
    work.loc[work["source_key"].isin(caveat_only_sources), "contribution_role"] = "caveat_only"

    keep_mask = (
        ((work["contribution_role"] == "primary_biochemical") & work["output_axis"].isin(primary_axis_names + caveat_axis_names))
        | ((work["contribution_role"] == "caveat_only") & work["output_axis"].isin(caveat_axis_names))
        | (work["output_axis"] == "unmapped_reference")
    )
    filtered = work[keep_mask].copy()
    return filtered.reset_index(drop=True)


def enforce_grounding_exclusions(alias: str, sources: Iterable[str], exclusions_df: pd.DataFrame) -> None:
    relevant = exclusions_df[
        exclusions_df["experiment_family"].astype(str).isin([alias, "serum_primary_targets" if "serum" in alias else "__none__"])
    ].copy()
    forbidden = {
        (str(r["forbidden_grounding_dataset_id"]), str(r["forbidden_grounding_subset_id"])): str(r["reason"])
        for _, r in relevant.iterrows()
    }
    violations = []
    for source in sources:
        if "::" in source:
            dataset_id, subset_id = source.split("::", 1)
        else:
            dataset_id, subset_id = source, "all"
        reason = forbidden.get((dataset_id, subset_id))
        if reason:
            violations.append((source, reason))
    if violations:
        detail = "; ".join([f"{src}: {reason}" for src, reason in violations])
        raise RuntimeError(f"Grounding exclusion violation for {alias}: {detail}")


def align_query_and_grounding(
    query_df: pd.DataFrame,
    grounding_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_x = [np.asarray(json.loads(v), dtype=float) for v in query_df["wavenumbers_json"]]
    query_y = [np.asarray(json.loads(v), dtype=float) for v in query_df["intensity_json"]]
    ground_x = [np.asarray(json.loads(v), dtype=float) for v in grounding_df["wavenumbers_json"]]
    ground_y = [np.asarray(json.loads(v), dtype=float) for v in grounding_df["intensity_json"]]
    master_x = np.unique(np.concatenate(query_x + ground_x)).astype(np.float32)
    query_matrix = np.vstack(
        [align_to_master_grid(x, y, master_x) for x, y in zip(query_x, query_y, strict=False)]
    ).astype(np.float32)
    ground_matrix = np.vstack(
        [align_to_master_grid(x, y, master_x) for x, y in zip(ground_x, ground_y, strict=False)]
    ).astype(np.float32)
    return master_x, query_matrix, ground_matrix


def cosine_topk(
    query_matrix: np.ndarray,
    grounding_matrix: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_norm = normalize_rows(query_matrix.astype(np.float32))
    ground_norm = normalize_rows(grounding_matrix.astype(np.float32))
    scores = query_norm @ ground_norm.T
    k = min(top_k, scores.shape[1])
    top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    order = np.argsort(top_scores, axis=1)[:, ::-1]
    return np.take_along_axis(top_idx, order, axis=1), np.take_along_axis(top_scores, order, axis=1)


def topk_by_similarity(
    query_matrix: np.ndarray,
    grounding_matrix: np.ndarray,
    *,
    top_k: int,
    similarity_metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    if similarity_metric == "cosine":
        return cosine_topk(query_matrix, grounding_matrix, top_k=top_k)
    if similarity_metric == "correlation":
        query_norm = correlation_normalize_rows(query_matrix.astype(np.float32))
        ground_norm = correlation_normalize_rows(grounding_matrix.astype(np.float32))
        scores = query_norm @ ground_norm.T
        k = min(top_k, scores.shape[1])
        top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(top_scores, axis=1)[:, ::-1]
        return np.take_along_axis(top_idx, order, axis=1), np.take_along_axis(top_scores, order, axis=1)
    raise ValueError(f"Unsupported similarity_metric: {similarity_metric}")


def load_ontology_rules(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ontology rules file missing: {path}")
    return pd.read_csv(path).sort_values(["priority", "output_axis"]).reset_index(drop=True)


def map_references_to_axes(reference_df: pd.DataFrame, rules_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, ref in reference_df.iterrows():
        matched: list[pd.Series] = []
        for _, rule in rules_df.iterrows():
            field = str(rule["match_field"])
            value = str(ref.get(field, ""))
            pattern = str(rule["pattern"])
            match_type = str(rule["match_type"])
            ok = False
            if match_type == "exact":
                ok = value == pattern
            elif match_type == "regex":
                ok = bool(re.search(pattern, value, flags=re.IGNORECASE))
            if ok:
                matched.append(rule)
        if not matched:
            rows.append({"sample_key": ref["sample_key"], "output_axis": "unmapped_reference", "axis_weight": 1.0, "axis_kind": "unmapped"})
            continue
        axis_best: dict[str, tuple[float, str]] = {}
        for rule in matched:
            axis = str(rule["output_axis"])
            weight = float(rule["axis_weight"])
            kind = str(rule["axis_kind"])
            prev = axis_best.get(axis)
            if prev is None or weight > prev[0]:
                axis_best[axis] = (weight, kind)
        for axis, (weight, kind) in axis_best.items():
            rows.append(
                {
                    "sample_key": ref["sample_key"],
                    "output_axis": axis,
                    "axis_weight": weight,
                    "axis_kind": kind,
                }
            )
    return pd.DataFrame(rows)


def _normalize_weights(scores: np.ndarray, mode: str) -> np.ndarray:
    clipped = np.maximum(scores.astype(float), 0.0)
    if mode == "raw_support":
        return clipped
    if mode == "per_spectrum_sum":
        denom = np.maximum(clipped.sum(axis=1, keepdims=True), 1e-8)
        return clipped / denom
    if mode == "softmax_then_sum":
        stable = clipped - clipped.max(axis=1, keepdims=True)
        exp = np.exp(stable)
        denom = np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)
        return exp / denom
    if mode == "delta_zscore_placeholder":
        denom = np.maximum(clipped.sum(axis=1, keepdims=True), 1e-8)
        return clipped / denom
    raise ValueError(f"Unsupported normalization mode: {mode}")


def build_bsv_profiles(
    query_df: pd.DataFrame,
    grounding_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    *,
    top_k: int,
    normalization_mode: str,
    similarity_metric: str = "cosine",
    axis_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if axis_names is None:
        axis_names = ALL_AXES
    _, query_matrix, grounding_matrix = align_query_and_grounding(query_df, grounding_df)
    top_idx, top_scores = topk_by_similarity(
        query_matrix,
        grounding_matrix,
        top_k=top_k,
        similarity_metric=similarity_metric,
    )
    weights = _normalize_weights(top_scores, normalization_mode)

    mapping_grouped = {
        str(sample_key): group[["output_axis", "axis_weight"]].to_dict("records")
        for sample_key, group in mapping_df.groupby("sample_key", sort=False)
    }
    profile_rows = []
    hit_rows = []
    for i, query_row in enumerate(query_df.itertuples(index=False)):
        axis_scores = {axis: 0.0 for axis in axis_names}
        unmapped_support = 0.0
        for rank, (idx, sim, weight) in enumerate(zip(top_idx[i], top_scores[i], weights[i], strict=False), start=1):
            ref = grounding_df.iloc[int(idx)]
            support = float(max(weight, 0.0))
            sample_key = str(ref["sample_key"])
            mappings = mapping_grouped.get(sample_key, [])
            if not mappings:
                unmapped_support += support
            for m in mappings:
                axis = str(m["output_axis"])
                axis_weight = float(m["axis_weight"])
                if axis in axis_scores:
                    axis_scores[axis] += support * axis_weight
                elif axis == "unmapped_reference":
                    unmapped_support += support
            hit_rows.append(
                {
                    "query_sample_key": str(query_row.sample_key),
                    "query_class_label": str(query_row.class_label),
                    "rank": rank,
                    "similarity": float(sim),
                    "support_weight": support,
                    "reference_sample_key": sample_key,
                    "reference_dataset_id": str(ref["dataset_id"]),
                    "reference_source_key": str(ref["source_key"]),
                    "reference_class_label": str(ref["class_label"]),
                    "reference_compound_label": str(ref.get("compound_label", "")),
                }
            )
        total = sum(axis_scores.values())
        if normalization_mode in {"per_spectrum_sum", "softmax_then_sum", "delta_zscore_placeholder"} and total > 0:
            axis_scores = {k: v / total for k, v in axis_scores.items()}
        profile_rows.append(
            {
                "sample_key": str(query_row.sample_key),
                "dataset_id": str(query_row.dataset_id),
                "subset_id": str(query_row.subclass_label),
                "class_label": str(query_row.class_label),
                **axis_scores,
                "unmapped_support": float(unmapped_support),
            }
        )
    return pd.DataFrame(profile_rows), pd.DataFrame(hit_rows)


def group_mean_bsv(
    per_spectrum_df: pd.DataFrame,
    group_col: str = "class_label",
    axis_names: list[str] | None = None,
) -> pd.DataFrame:
    if axis_names is None:
        axes = [axis for axis in ALL_AXES if axis in per_spectrum_df.columns]
    else:
        axes = [axis for axis in axis_names if axis in per_spectrum_df.columns]
    return (
        per_spectrum_df.groupby(group_col, sort=True)[axes + ["unmapped_support"]]
        .mean()
        .reset_index()
    )


def delta_bsv(group_means: pd.DataFrame, *, reference_group: str, axis_names: list[str] | None = None) -> pd.DataFrame:
    if axis_names is None:
        axis_names = [axis for axis in ALL_AXES if axis in group_means.columns]
    ref = group_means[group_means["class_label"].astype(str) == str(reference_group)]
    if ref.empty:
        raise RuntimeError(f"Reference group {reference_group} not found in group means.")
    ref_values = ref.iloc[0]
    rows = []
    for _, row in group_means.iterrows():
        label = str(row["class_label"])
        if label == str(reference_group):
            continue
        record = {"comparison": f"{label}-vs-{reference_group}", "group_label": label, "reference_group": reference_group}
        for axis in axis_names + ["unmapped_support"]:
            record[axis] = float(row[axis] - ref_values[axis])
        rows.append(record)
    return pd.DataFrame(rows)


def resolve_reference_group(query_df: pd.DataFrame) -> str:
    labels = sorted(query_df["class_label"].astype(str).unique().tolist())
    for candidate in ["Bkg", "control", "healthy", "baseline"]:
        for label in labels:
            if label.lower() == candidate.lower():
                return label
    return labels[0]
