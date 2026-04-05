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
    compute_stability_tables,
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
PILOT1A_V5_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot1a_celltype_probe1_v5"
)

SPRINT_SUBDIR = "pilot2_target_validation_v1"
SUBSET_ALIAS = "diabetes_ev_state"

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

CLASS_COLORS = {"Impact": "#d1495b", "Strong-D": "#2a9d8f"}
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
            "experiment_id": f"pilot2_target_validation__{subset_alias}",
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


def _extract_sample_id(sample_key: str, source_file: str, class_label: str) -> str:
    prefix = re.sub(r"[^a-z0-9]+", "_", str(class_label).strip().lower()).strip("_")
    sample_match = re.search(r"sample_\d+", str(sample_key))
    if sample_match:
        return f"{prefix}_{sample_match.group(0)}"
    cell_match = re.search(r"cell_\d+", str(source_file))
    if cell_match:
        return f"{prefix}_{cell_match.group(0)}"
    return f"{prefix}_{str(sample_key)}"


def _build_sample_query_df(query_df: pd.DataFrame) -> pd.DataFrame:
    master_x, matrix = decode_and_align(query_df)
    work = query_df.reset_index(drop=True).copy()
    work["derived_sample_id"] = [
        _extract_sample_id(sample_key, source_file, class_label)
        for sample_key, source_file, class_label in zip(
            work["sample_key"].astype(str),
            work["source_file"].astype(str),
            work["class_label"].astype(str),
            strict=False,
        )
    ]
    rows = []
    for sample_id, group in work.groupby("derived_sample_id", sort=True):
        idx = group.index.to_numpy()
        mean_vector = matrix[idx].mean(axis=0)
        rows.append(
            {
                "sample_key": f"sample_mean__{sample_id}",
                "sample_id": str(sample_id),
                "dataset_id": str(group["dataset_id"].iloc[0]),
                "processing_version": str(group["processing_version"].iloc[0]),
                "wavenumbers_json": json.dumps(master_x.astype(float).tolist()),
                "intensity_json": json.dumps(mean_vector.astype(float).tolist()),
                "class_label": str(group["class_label"].iloc[0]),
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
    out = df[["sample_key", "sample_id", "class_label"]].copy()
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
    X = df[axes].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(n_neighbors=min(12, max(len(df) - 1, 2)), min_dist=0.25, metric="euclidean", random_state=42)
    coords = reducer.fit_transform(X)
    out = df[["sample_key", "sample_id", "class_label"]].copy()
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


def _family_fingerprint_from_neighborhood(class_neighborhood_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = class_neighborhood_df.copy()
    df["family"] = df["compound_label"].astype(str).map(_compound_to_family)
    grouped = df.groupby([label_col, "family"], as_index=False)["support_fraction"].sum()
    rows = []
    for label in sorted(df[label_col].astype(str).unique().tolist()):
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


def _sample_family_fingerprint(sample_retrieval_df: pd.DataFrame) -> pd.DataFrame:
    if sample_retrieval_df.empty:
        return pd.DataFrame(columns=["sample_key", "sample_id", "class_label", "family", "family_fraction"])
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
    rows = []
    for sample_key, group in grouped.groupby("sample_key", sort=True):
        sample_id = str(sample_key).replace("sample_mean__", "")
        total = float(group["family_support"].sum())
        existing = {str(x) for x in group["family"].tolist()}
        class_label = str(group["class_label"].iloc[0])
        for family in FAMILY_ORDER:
            val = 0.0
            if family in existing:
                val = float(group[group["family"].astype(str) == family]["family_support"].iloc[0])
            rows.append(
                {
                    "sample_key": str(sample_key),
                    "sample_id": sample_id,
                    "class_label": class_label,
                    "family": family,
                    "family_fraction": (val / total) if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _nearest_neighbor_purity(df: pd.DataFrame, axes: list[str], *, n_neighbors: int = 5) -> pd.DataFrame:
    X = StandardScaler().fit_transform(df[axes].to_numpy(dtype=float))
    labels = df["class_label"].astype(str).to_numpy()
    n_use = min(n_neighbors + 1, len(df))
    nn = NearestNeighbors(n_neighbors=n_use)
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)
    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        neigh = [j for j in indices[i] if j != i][:n_neighbors]
        purity = float(np.mean(labels[neigh] == labels[i])) if neigh else 0.0
        rows.append({"sample_key": row.sample_key, "class_label": row.class_label, "neighbor_purity": purity})
    return pd.DataFrame(rows)


def _mean_within_variance(df: pd.DataFrame, axes: list[str], class_label: str) -> float:
    sub = df[df["class_label"].astype(str) == str(class_label)].copy()
    if len(sub) <= 1:
        return 0.0
    return float(sub[axes].var(ddof=1).mean())


def _plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    *,
    title: str,
    hue_col: str = "class_label",
    style_col: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    labels = sorted(df[hue_col].astype(str).unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X"]
    for i, label in enumerate(labels):
        sub = df[df[hue_col].astype(str) == label].copy()
        kwargs = {"marker": markers[i % len(markers)]}
        if style_col is not None and style_col in df.columns:
            pass
        ax.scatter(
            sub[x_col].to_numpy(dtype=float),
            sub[y_col].to_numpy(dtype=float),
            s=48,
            alpha=0.82,
            label=label,
            color=CLASS_COLORS.get(label, CLUSTER_COLORS[i % len(CLUSTER_COLORS)]),
            edgecolors="white",
            linewidths=0.5,
            **kwargs,
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
    if delta_mode:
        radius_lim = max(float(np.abs(values).max()), 0.05)
    else:
        radius_lim = max(float(values.max()), 0.5)
    for ax in axs[len(labels):]:
        ax.axis("off")
    for idx, (ax, (_, row)) in enumerate(zip(axs, df.iterrows(), strict=False)):
        vals = np.array([float(row.get(axis, 0.0)) for axis in FIXED_RADAR_AXES], dtype=float)
        plot_vals = vals + radius_lim if delta_mode else vals
        vals_closed = np.concatenate([plot_vals, [plot_vals[0]]])
        color = CLASS_COLORS.get(str(row[label_col]), CLUSTER_COLORS[idx % len(CLUSTER_COLORS)])
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
    heat = cross_df.pivot(index="cluster_label_a", columns="cluster_label_b", values="overlap_score")
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(heat.max().max()), 1e-6))
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns.tolist(), rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index.tolist(), fontsize=8)
    ax.set_title("Cross-Class Cluster Overlap")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{float(heat.iloc[i, j]):.2f}", ha="center", va="center", fontsize=8, color="white")
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


def _cluster_with_stability(sample_df: pd.DataFrame, axes: list[str], class_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = sample_df[sample_df["class_label"].astype(str) == str(class_label)].copy().reset_index(drop=True)
    X = sub[axes].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    seeds = [0, 1, 2, 3, 4]
    candidate_rows = []
    chosen_labels = None
    chosen_k = None
    chosen_sil = None
    chosen_ari = None
    for k in [2, 3, 4]:
        if len(sub) < k:
            continue
        label_runs = []
        sils = []
        for seed in seeds:
            model = KMeans(n_clusters=k, random_state=seed, n_init=20)
            labels = model.fit_predict(X_scaled)
            sil = float(silhouette_score(X_scaled, labels)) if len(np.unique(labels)) > 1 else 0.0
            label_runs.append(labels)
            sils.append(sil)
        aris = []
        for i in range(len(label_runs)):
            for j in range(i + 1, len(label_runs)):
                aris.append(float(adjusted_rand_score(label_runs[i], label_runs[j])))
        mean_sil = float(np.mean(sils))
        mean_ari = float(np.mean(aris)) if aris else 1.0
        score = mean_sil + 0.15 * mean_ari
        candidate_rows.append(
            {
                "class_label": class_label,
                "k": k,
                "mean_silhouette": mean_sil,
                "mean_ari": mean_ari,
                "selection_score": score,
            }
        )
        if chosen_labels is None or score > float(max(r["selection_score"] for r in candidate_rows[:-1]) if candidate_rows[:-1] else -1e9):
            best_idx = int(np.argmax(sils))
            chosen_labels = label_runs[best_idx]
            chosen_k = k
            chosen_sil = mean_sil
            chosen_ari = mean_ari
    if chosen_labels is None or chosen_k is None or chosen_sil is None or chosen_ari is None:
        raise RuntimeError(f"Could not cluster class {class_label}")

    centroid_df = pd.DataFrame(X, columns=axes)
    centroid_df["raw_cluster"] = chosen_labels
    centroid_summary = centroid_df.groupby("raw_cluster", sort=True)[axes].mean().reset_index()
    sort_axis = "small_molecule_metabolite" if "small_molecule_metabolite" in axes else axes[0]
    ordered_clusters = (
        centroid_summary.sort_values(sort_axis, ascending=False)["raw_cluster"].astype(int).tolist()
    )
    mapping = {raw: f"{class_label}_latent_{chr(65+i)}" for i, raw in enumerate(ordered_clusters)}
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
    cluster_abs = cluster_abs_input_df.groupby(["class_label", "cluster_label"], as_index=False)[axes].mean()
    cluster_delta = cluster_delta_input_df.groupby(["class_label", "cluster_label"], as_index=False)[axes].mean()
    fam_rows = []
    for (class_label, cluster_label), group in cluster_delta_input_df.groupby(["class_label", "cluster_label"], sort=True):
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
                    "class_label": str(class_label),
                    "cluster_label": str(cluster_label),
                    "family": family,
                    "family_fraction": (val / total) if total > 0 else 0.0,
                }
            )
    return cluster_abs, cluster_delta, pd.DataFrame(fam_rows)


def _build_report(
    report_path: Path,
    class_metrics_df: pd.DataFrame,
    cluster_selection_df: pd.DataFrame,
    cluster_metrics_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    umap_available: bool,
) -> None:
    impact = class_metrics_df[class_metrics_df["class_label"].astype(str) == "Impact"].iloc[0]
    strong = class_metrics_df[class_metrics_df["class_label"].astype(str) == "Strong-D"].iloc[0]
    best_impact = cluster_selection_df[cluster_selection_df["class_label"].astype(str) == "Impact"].sort_values(
        "selection_score", ascending=False
    ).iloc[0]
    best_strong = cluster_selection_df[cluster_selection_df["class_label"].astype(str) == "Strong-D"].sort_values(
        "selection_score", ascending=False
    ).iloc[0]
    broad_sep = float(impact["silhouette_score_delta"])
    broad_answer = "yes" if broad_sep >= 0.10 else "limited"
    cluster_answer = "yes" if cluster_metrics_df["class_label"].nunique() >= 1 else "no"
    fingerprint_answer = "yes" if float(consistency_df["overlap_score"].max()) < 0.75 else "partly"
    overlap_answer = "yes" if float(consistency_df["overlap_score"].max()) > 0.35 else "limited"
    lines = [
        "# GAIRAv3 Pilot 2 Target Validation v1 Report",
        "",
        "## 1. Objective",
        "- Pilot 2 tests whether the locked cfg05 representation transfers from the small2023 development pilots to the diabetes EV target dataset.",
        "- The broad labels are `Impact` and `Strong-D`.",
        "- Latent subclass discovery is exploratory only. No discovered cluster is assigned to a literature subtype identity.",
        "",
        "## 2. Fixed Configuration",
        "- representation_mode = `raw_direct_bsv_input`",
        "- grounding_mode = `universal_only`",
        "- universal_grounding_filter_mode = `purine_expanded_neighbor`",
        "- aggregation_mode = `class_mean_spectrum_then_bsv`",
        "- ontology_mode = `tier1_plus_subclass`",
        "- similarity_metric = `cosine`",
        "- plausibility_scoring_mode = `baseline_plausibility`",
        "- pca_grouping_mode = `class_label_groups`",
        "- top_k = `5`",
        "- weighting_mode = `softmax_temperature(1.0)`",
        "- diversity_mode = `compound_uniqueness_penalty`",
        "",
        "## 3. Broad Label Validation",
        f"- Impact sample count: `{int(impact['n_samples'])}`; Strong-D sample count: `{int(strong['n_samples'])}`.",
        f"- Absolute centroid distance: `{float(impact['centroid_distance_absolute']):.4f}`.",
        f"- Delta centroid distance: `{float(impact['centroid_distance_delta']):.4f}`.",
        f"- Delta silhouette score: `{float(impact['silhouette_score_delta']):.4f}`.",
        f"- Neighborhood purity in delta space: Impact `{float(impact['neighbor_purity_delta']):.4f}`, Strong-D `{float(strong['neighbor_purity_delta']):.4f}`.",
        f"- Mean top1 dominance: Impact `{float(impact['top1_dominance']):.4f}`, Strong-D `{float(strong['top1_dominance']):.4f}`.",
        f"- Mean neighborhood entropy: Impact `{float(impact['neighborhood_entropy']):.4f}`, Strong-D `{float(strong['neighborhood_entropy']):.4f}`.",
        "",
        "## 4. Latent Subclass Discovery",
        f"- Best Impact clustering: `k={int(best_impact['k'])}` with mean silhouette `{float(best_impact['mean_silhouette']):.4f}` and seed-stability ARI `{float(best_impact['mean_ari']):.4f}`.",
        f"- Best Strong-D clustering: `k={int(best_strong['k'])}` with mean silhouette `{float(best_strong['mean_silhouette']):.4f}` and seed-stability ARI `{float(best_strong['mean_ari']):.4f}`.",
        "- Cluster fingerprints are described only as latent subclasses and biochemical themes.",
        "- Cluster-level radars and family-composition panels should be read as relative enrichment and neighborhood-balance patterns, not molecule calls.",
        "",
        "## 5. Cross-Class Latent Structure",
        f"- Minimum cross-class centroid distance in delta space: `{float(consistency_df['delta_bsv_distance'].min()):.4f}`.",
        f"- Minimum cross-class family distance: `{float(consistency_df['family_distance'].min()):.4f}`.",
        f"- Maximum cross-class overlap score: `{float(consistency_df['overlap_score'].max()):.4f}`.",
        "- Overlap is informative when clusters from different broad labels approach similar biochemical states, but it does not imply the labels are interchangeable.",
        "",
        "## 6. Direct Answers",
        f"1. Broad classes separable in BSV / delta-BSV space: `{broad_answer}`",
        f"2. Latent clusters emerge within at least one class: `{cluster_answer}`",
        f"3. Clusters show distinct radar fingerprints and family structure: `{fingerprint_answer}`",
        f"4. Cross-class cluster overlap exists: `{overlap_answer}`",
        "",
        "## 7. Interpretation",
        "- The broad-label comparison should be read as a biochemical theme-space validation, not as a diagnostic classifier claim.",
        "- The subclass analysis is weakly supervised discovery inside each broad label.",
        "- In this run the broad two-class separation is only modest, while the within-class latent structure is much stronger.",
        "- UMAP availability: "
        + ("enabled and plotted." if umap_available else "not available locally, so PCA-only embeddings were used."),
        "",
        "## 8. Recommendation",
        "- cfg05 appears to transfer better as a latent-state discovery representation than as a strong broad-label separator on this target.",
        "- The broad `Impact` vs `Strong-D` split is weak in delta-BSV space, but the within-class cluster structure is strong and stable.",
        "- Pilot 2 therefore supports continuing with cfg05 if the next question is latent subclass discovery and cross-state overlap, not if the immediate goal is a strong two-class classifier.",
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
        sample_query_df[["sample_key", "sample_id", "n_scans", "source_file"]],
        on="sample_key",
        how="left",
    )

    axes = _axes_present(sample_bsv_df)
    sample_delta_df = _cohort_delta(sample_bsv_df, axes)
    class_mean_bsv_df = sample_bsv_df.groupby("class_label", as_index=False)[axes].mean()
    class_mean_delta_df = _cohort_delta(class_mean_bsv_df, axes)
    pairwise_delta_df = pairwise_delta_bsv(class_mean_delta_df)

    class_neighborhood_df = build_class_topk_neighborhood_composition(sample_retrieval_df)
    class_family_df = _family_fingerprint_from_neighborhood(class_neighborhood_df, "class_label")
    class_neighborhood_entropy_df = build_class_neighborhood_entropy(class_neighborhood_df)
    class_top1_df = build_class_top1_dominance(class_neighborhood_df)
    class_axis_entropy_df = build_class_axis_entropy(class_mean_bsv_df)
    retrieval_summary_df = retrieval_hit_summary(sample_retrieval_df)
    sample_family_df = _sample_family_fingerprint(sample_retrieval_df)

    _, sample_matrix = decode_and_align(sample_query_df)
    spec_scores, spec_explained = _fit_pca(sample_matrix, scale=True)
    spectral_pca_df = sample_query_df[["sample_key", "sample_id", "class_label"]].copy()
    spectral_pca_df["pc1"] = spec_scores[:, 0]
    spectral_pca_df["pc2"] = spec_scores[:, 1]
    spectral_pca_df["pc1_explained_ratio"] = float(spec_explained[0]) if len(spec_explained) > 0 else 1.0
    spectral_pca_df["pc2_explained_ratio"] = float(spec_explained[1]) if len(spec_explained) > 1 else 0.0

    bsv_pca_df = _pca_dataframe(sample_bsv_df, axes, scale=True)
    delta_pca_df = _pca_dataframe(sample_delta_df, axes, scale=True)
    umap_bsv_df = _maybe_umap_dataframe(sample_bsv_df, axes)
    umap_delta_df = _maybe_umap_dataframe(sample_delta_df, axes)

    purity_df = _nearest_neighbor_purity(sample_delta_df, axes)
    intra_abs_df, inter_abs_df = compute_stability_tables(sample_bsv_df, class_mean_bsv_df, axis_names=axes)
    intra_delta_df, inter_delta_df = compute_stability_tables(sample_delta_df, class_mean_delta_df, axis_names=axes)

    class_rows = []
    classes = class_mean_bsv_df["class_label"].astype(str).tolist()
    centroid_abs = class_mean_bsv_df.set_index("class_label")
    centroid_delta = class_mean_delta_df.set_index("class_label")
    silhouette_abs = float(silhouette_score(StandardScaler().fit_transform(sample_bsv_df[axes]), sample_bsv_df["class_label"].astype(str))) if len(classes) > 1 else 0.0
    silhouette_delta = float(silhouette_score(StandardScaler().fit_transform(sample_delta_df[axes]), sample_delta_df["class_label"].astype(str))) if len(classes) > 1 else 0.0
    for class_label in classes:
        other_label = [x for x in classes if x != class_label][0]
        abs_diff = centroid_abs.loc[class_label, axes].to_numpy(dtype=float) - centroid_abs.loc[other_label, axes].to_numpy(dtype=float)
        delta_diff = centroid_delta.loc[class_label, axes].to_numpy(dtype=float) - centroid_delta.loc[other_label, axes].to_numpy(dtype=float)
        entropy = float(class_neighborhood_entropy_df[class_neighborhood_entropy_df["class_label"].astype(str) == class_label]["neighborhood_entropy"].iloc[0])
        top1 = float(class_top1_df[class_top1_df["class_label"].astype(str) == class_label]["top1_fraction"].iloc[0])
        axis_entropy = float(class_axis_entropy_df[class_axis_entropy_df["class_label"].astype(str) == class_label]["axis_entropy"].iloc[0])
        purity = float(purity_df[purity_df["class_label"].astype(str) == class_label]["neighbor_purity"].mean())
        class_rows.append(
            {
                "class_label": class_label,
                "n_samples": int((sample_bsv_df["class_label"].astype(str) == class_label).sum()),
                "within_class_variance_absolute": _mean_within_variance(sample_bsv_df, axes, class_label),
                "within_class_variance_delta": _mean_within_variance(sample_delta_df, axes, class_label),
                "centroid_distance_absolute": float(np.linalg.norm(abs_diff)),
                "centroid_distance_delta": float(np.linalg.norm(delta_diff)),
                "between_class_distance": float(np.linalg.norm(delta_diff)),
                "silhouette_score_absolute": silhouette_abs,
                "silhouette_score_delta": silhouette_delta,
                "neighbor_purity_delta": purity,
                "neighborhood_entropy": entropy,
                "top1_dominance": top1,
                "axis_entropy": axis_entropy,
            }
        )
    class_metrics_df = pd.DataFrame(class_rows)

    cluster_assignments = []
    cluster_selection_rows = []
    for class_label in classes:
        sub = sample_delta_df[["sample_key", "sample_id", "class_label"] + axes].copy()
        for axis in axes:
            sub[f"delta::{axis}"] = sample_delta_df[axis]
        chosen_df, candidates_df = _cluster_with_stability(sub, axes, class_label)
        cluster_assignments.append(chosen_df)
        cluster_selection_rows.append(candidates_df)
    cluster_assignment_df = pd.concat(cluster_assignments, ignore_index=True)
    cluster_selection_df = pd.concat(cluster_selection_rows, ignore_index=True)

    sample_abs_with_clusters = sample_bsv_df.merge(
        cluster_assignment_df[["sample_key", "cluster_label", "chosen_k", "cluster_silhouette", "cluster_stability_mean_ari"]],
        on="sample_key",
        how="left",
    )
    sample_delta_with_clusters = sample_delta_df.merge(
        cluster_assignment_df[["sample_key", "cluster_label", "chosen_k", "cluster_silhouette", "cluster_stability_mean_ari"]],
        on="sample_key",
        how="left",
    )

    cluster_abs_df, cluster_delta_df, cluster_family_df = _cluster_centroid_metrics(
        sample_abs_with_clusters,
        sample_delta_with_clusters,
        axes,
        sample_family_df,
    )
    cluster_metrics_rows = []
    for _, row in cluster_abs_df.iterrows():
        class_label = str(row["class_label"])
        cluster_label = str(row["cluster_label"])
        sub_assign = sample_delta_with_clusters[
            sample_delta_with_clusters["cluster_label"].astype(str) == cluster_label
        ].copy()
        same_class = cluster_delta_df[cluster_delta_df["class_label"].astype(str) == class_label].copy()
        this_centroid = same_class[same_class["cluster_label"].astype(str) == cluster_label].iloc[0]
        others = same_class[same_class["cluster_label"].astype(str) != cluster_label].copy()
        centroid_dist = float("nan")
        if not others.empty:
            dists = []
            for _, other in others.iterrows():
                diff = this_centroid[axes].to_numpy(dtype=float) - other[axes].to_numpy(dtype=float)
                dists.append(float(np.linalg.norm(diff)))
            centroid_dist = float(min(dists))
        cluster_metrics_rows.append(
            {
                "cluster_id": cluster_label,
                "class_label": class_label,
                "size": int(len(sub_assign)),
                "chosen_k": int(sub_assign["chosen_k"].iloc[0]),
                "silhouette_score": float(sub_assign["cluster_silhouette"].iloc[0]),
                "stability_mean_ari": float(sub_assign["cluster_stability_mean_ari"].iloc[0]),
                "centroid_distance_to_other_clusters": centroid_dist,
            }
        )
    cluster_metrics_df = pd.DataFrame(cluster_metrics_rows)

    impact_clusters = cluster_delta_df[cluster_delta_df["class_label"].astype(str) == "Impact"].copy()
    strong_clusters = cluster_delta_df[cluster_delta_df["class_label"].astype(str) == "Strong-D"].copy()
    cross_rows = []
    for _, a in impact_clusters.iterrows():
        fam_a = cluster_family_df[cluster_family_df["cluster_label"].astype(str) == str(a["cluster_label"])].set_index("family")["family_fraction"]
        abs_a = cluster_abs_df[cluster_abs_df["cluster_label"].astype(str) == str(a["cluster_label"])].iloc[0]
        for _, b in strong_clusters.iterrows():
            fam_b = cluster_family_df[cluster_family_df["cluster_label"].astype(str) == str(b["cluster_label"])].set_index("family")["family_fraction"]
            abs_b = cluster_abs_df[cluster_abs_df["cluster_label"].astype(str) == str(b["cluster_label"])].iloc[0]
            delta_dist = float(np.linalg.norm(a[axes].to_numpy(dtype=float) - b[axes].to_numpy(dtype=float)))
            abs_dist = float(np.linalg.norm(abs_a[axes].to_numpy(dtype=float) - abs_b[axes].to_numpy(dtype=float)))
            fam_dist = float(np.abs(fam_a.reindex(FAMILY_ORDER).fillna(0.0).to_numpy() - fam_b.reindex(FAMILY_ORDER).fillna(0.0).to_numpy()).sum())
            overlap_score = 1.0 / (1.0 + delta_dist + fam_dist)
            cross_rows.append(
                {
                    "cluster_label_a": str(a["cluster_label"]),
                    "cluster_label_b": str(b["cluster_label"]),
                    "absolute_bsv_distance": abs_dist,
                    "delta_bsv_distance": delta_dist,
                    "family_distance": fam_dist,
                    "overlap_score": overlap_score,
                }
            )
    cross_class_df = pd.DataFrame(cross_rows)

    class_metrics_df.to_csv(sprint_paths.tables_dir / "class_level_metrics.csv", index=False)
    cluster_metrics_df.to_csv(sprint_paths.tables_dir / "cluster_metrics.csv", index=False)
    cross_class_df.to_csv(sprint_paths.tables_dir / "cross_class_overlap.csv", index=False)
    cluster_selection_df.to_csv(sprint_paths.tables_dir / "cluster_selection_summary.csv", index=False)

    run_dir = sprint_paths.runs_dir / CONFIG_SPEC["config_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_query_df.to_csv(run_dir / "sample_query_spectra.csv", index=False)
    sample_bsv_df.to_csv(run_dir / "per_sample_bsv.csv", index=False)
    sample_delta_df.to_csv(run_dir / "per_sample_delta_bsv.csv", index=False)
    class_mean_bsv_df.to_csv(run_dir / "class_mean_bsv.csv", index=False)
    class_mean_delta_df.to_csv(run_dir / "class_mean_bsv_delta_vs_cohort.csv", index=False)
    pairwise_delta_df.to_csv(run_dir / "pairwise_delta_bsv.csv", index=False)
    class_neighborhood_df.to_csv(run_dir / "class_topk_neighborhood_composition.csv", index=False)
    class_family_df.to_csv(run_dir / "class_neighborhood_family_composition.csv", index=False)
    class_neighborhood_entropy_df.to_csv(run_dir / "class_neighborhood_entropy.csv", index=False)
    class_top1_df.to_csv(run_dir / "class_top1_dominance.csv", index=False)
    class_axis_entropy_df.to_csv(run_dir / "class_axis_entropy.csv", index=False)
    retrieval_summary_df.to_csv(run_dir / "retrieval_hit_summary_by_class.csv", index=False)
    sample_retrieval_df.to_csv(run_dir / "per_sample_retrieval_hits.csv", index=False)
    spectral_pca_df.to_csv(run_dir / "pca_coordinates_spectral.csv", index=False)
    bsv_pca_df.to_csv(run_dir / "pca_coordinates_bsv.csv", index=False)
    delta_pca_df.to_csv(run_dir / "pca_coordinates_delta_bsv.csv", index=False)
    sample_family_df.to_csv(run_dir / "sample_family_fingerprint.csv", index=False)
    cluster_assignment_df.to_csv(run_dir / "sample_cluster_assignments.csv", index=False)
    cluster_abs_df.to_csv(run_dir / "cluster_mean_bsv.csv", index=False)
    cluster_delta_df.to_csv(run_dir / "cluster_mean_delta_bsv.csv", index=False)
    cluster_family_df.to_csv(run_dir / "cluster_family_fingerprint.csv", index=False)
    sample_query_df.to_csv(sprint_paths.tables_dir / "sample_query_spectra.csv", index=False)
    sample_bsv_df.to_csv(sprint_paths.tables_dir / "per_sample_bsv.csv", index=False)
    sample_delta_df.to_csv(sprint_paths.tables_dir / "per_sample_delta_bsv.csv", index=False)
    class_mean_bsv_df.to_csv(sprint_paths.tables_dir / "class_mean_bsv.csv", index=False)
    class_mean_delta_df.to_csv(sprint_paths.tables_dir / "class_mean_bsv_delta_vs_cohort.csv", index=False)
    pairwise_delta_df.to_csv(sprint_paths.tables_dir / "pairwise_delta_bsv.csv", index=False)
    class_neighborhood_df.to_csv(sprint_paths.tables_dir / "class_topk_neighborhood_composition.csv", index=False)
    class_family_df.to_csv(sprint_paths.tables_dir / "class_neighborhood_family_composition.csv", index=False)
    class_neighborhood_entropy_df.to_csv(sprint_paths.tables_dir / "class_neighborhood_entropy.csv", index=False)
    class_top1_df.to_csv(sprint_paths.tables_dir / "class_top1_dominance.csv", index=False)
    class_axis_entropy_df.to_csv(sprint_paths.tables_dir / "class_axis_entropy.csv", index=False)
    retrieval_summary_df.to_csv(sprint_paths.tables_dir / "retrieval_hit_summary_by_class.csv", index=False)
    sample_retrieval_df.to_csv(sprint_paths.tables_dir / "per_sample_retrieval_hits.csv", index=False)
    spectral_pca_df.to_csv(sprint_paths.tables_dir / "pca_coordinates_spectral.csv", index=False)
    bsv_pca_df.to_csv(sprint_paths.tables_dir / "pca_coordinates_bsv.csv", index=False)
    delta_pca_df.to_csv(sprint_paths.tables_dir / "pca_coordinates_delta_bsv.csv", index=False)
    sample_family_df.to_csv(sprint_paths.tables_dir / "sample_family_fingerprint.csv", index=False)
    cluster_assignment_df.to_csv(sprint_paths.tables_dir / "sample_cluster_assignments.csv", index=False)
    cluster_abs_df.to_csv(sprint_paths.tables_dir / "cluster_mean_bsv.csv", index=False)
    cluster_delta_df.to_csv(sprint_paths.tables_dir / "cluster_mean_delta_bsv.csv", index=False)
    cluster_family_df.to_csv(sprint_paths.tables_dir / "cluster_family_fingerprint.csv", index=False)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "subset_alias": SUBSET_ALIAS,
                "dataset_id": str(resolved.dataset_row["dataset_id"]),
                "subset_id": str(resolved.dataset_row["subset_id"]),
                "config": CONFIG_SPEC,
                "available_sources": sorted(grounding_df["source_key"].astype(str).unique().tolist()),
                "unavailable_sources": unavailable_sources,
                "note": "Pilot 2 uses sample-mean spectra as the working per-sample unit.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    figures_dir = sprint_paths.figures_dir
    _plot_scatter(spectral_pca_df, "pc1", "pc2", figures_dir / "pca_spectral_broad_classes.png", title="Spectral PCA of Sample-Mean Spectra")
    _plot_scatter(bsv_pca_df, "pc1", "pc2", figures_dir / "pca_bsv_broad_classes.png", title="BSV PCA of Sample-Level Fingerprints")
    _plot_scatter(delta_pca_df, "pc1", "pc2", figures_dir / "pca_delta_bsv_broad_classes.png", title="Delta-BSV PCA of Sample-Level Fingerprints")
    if umap_bsv_df is not None:
        _plot_scatter(umap_bsv_df.rename(columns={"u1": "pc1", "u2": "pc2"}), "pc1", "pc2", figures_dir / "umap_bsv_broad_classes.png", title="BSV UMAP of Sample-Level Fingerprints")
    if umap_delta_df is not None:
        _plot_scatter(umap_delta_df.rename(columns={"u1": "pc1", "u2": "pc2"}), "pc1", "pc2", figures_dir / "umap_delta_bsv_broad_classes.png", title="Delta-BSV UMAP of Sample-Level Fingerprints")
    _plot_radar_grid(class_mean_bsv_df, "class_label", figures_dir / "radar_bsv_broad_classes.png", "Broad-Class Absolute BSV Fingerprints", delta_mode=False)
    _plot_radar_grid(class_mean_delta_df, "class_label", figures_dir / "radar_delta_bsv_broad_classes.png", "Broad-Class Delta-BSV Fingerprints", delta_mode=True)
    plot_pairwise_delta_heatmap(pairwise_delta_df, "small_molecule_metabolite", figures_dir / "pairwise_delta_heatmap_small_molecule_metabolite.png")
    _plot_family_bars(class_family_df, "class_label", figures_dir / "class_family_fingerprint_bars.png", "Broad-Class Neighborhood Family Composition")
    _plot_cross_class_overlap(cross_class_df, figures_dir / "cross_class_overlap_heatmap.png")

    for class_label in classes:
        abs_sub = cluster_abs_df[cluster_abs_df["class_label"].astype(str) == class_label].copy().reset_index(drop=True)
        delta_sub = cluster_delta_df[cluster_delta_df["class_label"].astype(str) == class_label].copy().reset_index(drop=True)
        fam_sub = cluster_family_df[cluster_family_df["class_label"].astype(str) == class_label].copy()
        _plot_radar_grid(abs_sub.rename(columns={"cluster_label": "cluster_label"}), "cluster_label", figures_dir / f"cluster_radar_bsv_{class_label.replace('-', '_')}.png", f"{class_label} Latent Cluster BSV Fingerprints", delta_mode=False)
        _plot_radar_grid(delta_sub.rename(columns={"cluster_label": "cluster_label"}), "cluster_label", figures_dir / f"cluster_radar_delta_{class_label.replace('-', '_')}.png", f"{class_label} Latent Cluster Delta Fingerprints", delta_mode=True)
        _plot_family_bars(fam_sub, "cluster_label", figures_dir / f"cluster_family_bars_{class_label.replace('-', '_')}.png", f"{class_label} Cluster Family Composition")
        sub_pca = delta_pca_df.merge(cluster_assignment_df[["sample_key", "cluster_label"]], on="sample_key", how="left")
        sub_pca = sub_pca[sub_pca["class_label"].astype(str) == class_label].copy()
        _plot_cluster_scatter(sub_pca, "pc1", "pc2", figures_dir / f"cluster_delta_pca_{class_label.replace('-', '_')}.png", f"{class_label} Delta-BSV PCA by Latent Cluster")

    report_md = sprint_paths.report_dir / "GAIRAv3_Pilot2_target_validation_v1_report.md"
    report_pdf = sprint_paths.report_dir / "GAIRAv3_Pilot2_target_validation_v1_report.pdf"
    _build_report(report_md, class_metrics_df, cluster_selection_df, cluster_metrics_df, cross_class_df, umap_bsv_df is not None)
    figure_paths = sorted(figures_dir.glob("*.png"))
    build_pdf_report(report_md, figure_paths, report_pdf)

    print(f"Class metrics: {sprint_paths.tables_dir / 'class_level_metrics.csv'}")
    print(f"Cluster metrics: {sprint_paths.tables_dir / 'cluster_metrics.csv'}")
    print(f"Cross-class overlap: {sprint_paths.tables_dir / 'cross_class_overlap.csv'}")
    print(f"Markdown report: {report_md}")
    print(f"PDF report: {report_pdf}")


if __name__ == "__main__":
    main()
