from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from gaira.autoresearch_storage import AutoresearchSprintPaths
from gaira.demo.autoresearch_utils import comparator_map_for_alias, compute_delta_by_mapping, score_panel_outputs
from gaira.demo.gaira_pilot_utils import (
    build_class_axis_entropy,
    build_class_neighborhood_entropy,
    build_class_top1_dominance,
    build_class_topk_neighborhood_composition,
    build_mixture_progression_summary,
    compute_stability_tables,
    infer_mixture_order,
)
from gaira.demo.gaira_experiment_runner_utils import retrieval_hit_summary
from gaira.demo.raw_bsv_pilot_utils import (
    ALL_AXES,
    PRIMARY_AXES,
    align_query_and_grounding,
    apply_source_role_policy,
    build_group_mean_query_df,
    correlation_normalize_rows,
    group_mean_bsv,
    load_ontology_rules,
    map_references_to_axes,
    normalize_rows,
    PRIMARY_AXES,
)


PASS5_FILTER_MODE_DEFINITIONS = {
    "purine_focused_universal": "Pass 3 locked universal purine-focused subset.",
    "purine_expanded_neighbor": "Purine neighborhood expanded to include guanine/xanthine/hypoxanthine/urate/inosine-style neighbors when present in current universal grounding metadata.",
    "balanced_metabolite_subset": "Broader metabolite-oriented universal subset using adenine controls plus metabolite support archives rather than a purine-only neighborhood.",
    "purine_capped": "Purine-expanded universal subset intended for anti-dominance caps at weighting time.",
    "diversity_enforced_universal": "All currently available universal pure grounding sources with diversity-aware top-k selection.",
}

PURINE_FOCUSED_TERMS = [
    r"aden",
    r"guan",
    r"methyladenine",
    r"methylguanidine",
]
PURINE_EXPANDED_TERMS = PURINE_FOCUSED_TERMS + [
    r"xanth",
    r"hypox",
    r"inos",
    r"uric",
    r"urate",
]
SULFUR_TERMS = [
    r"glutath",
    r"cyste",
    r"cystine",
    r"cystath",
    r"homocys",
    r"methion",
    r"lipoamide",
    r"seleno",
]
AROMATIC_TERMS = [r"tyros", r"trypt", r"phenyl", r"indole"]


@dataclass(frozen=True)
class Pass5HarnessConfig:
    config_id: str
    universal_grounding_filter_mode: str
    top_k: int
    weighting_mode: str
    weighting_param: float | None
    diversity_mode: str
    family_min_coverage: int = 0


def build_pass5_search_space() -> pd.DataFrame:
    configs = [
        ("cfg01", "purine_focused_universal", 3, "uniform_weighting", None, "none", 0),
        ("cfg02", "purine_focused_universal", 5, "uniform_weighting", None, "none", 0),
        ("cfg03", "purine_focused_universal", 8, "uniform_weighting", None, "none", 0),
        ("cfg04", "purine_expanded_neighbor", 3, "softmax_temperature", 0.5, "compound_uniqueness_penalty", 0),
        ("cfg05", "purine_expanded_neighbor", 5, "softmax_temperature", 1.0, "compound_uniqueness_penalty", 0),
        ("cfg06", "purine_expanded_neighbor", 8, "softmax_temperature", 2.0, "compound_uniqueness_penalty", 0),
        ("cfg07", "balanced_metabolite_subset", 5, "uniform_weighting", None, "family_balance_penalty", 0),
        ("cfg08", "balanced_metabolite_subset", 8, "rank_decay_weighting", 0.75, "family_balance_penalty", 0),
        ("cfg09", "purine_capped", 5, "max_cap_per_compound", 0.5, "compound_uniqueness_penalty", 0),
        ("cfg10", "purine_capped", 8, "max_cap_per_compound", 0.5, "min_family_coverage_constraint", 2),
        ("cfg11", "diversity_enforced_universal", 5, "uniform_weighting", None, "min_family_coverage_constraint", 2),
        ("cfg12", "diversity_enforced_universal", 8, "rank_decay_weighting", 0.75, "min_family_coverage_constraint", 2),
    ]
    rows = []
    for config_id, filter_mode, top_k, weighting_mode, weighting_param, diversity_mode, family_min_coverage in configs:
        rows.append(
            {
                "config_id": config_id,
                "universal_grounding_filter_mode": filter_mode,
                "top_k": top_k,
                "weighting_mode": weighting_mode,
                "weighting_param": weighting_param,
                "diversity_mode": diversity_mode,
                "family_min_coverage": family_min_coverage,
                "status": "valid",
                "exclusion_reason": "",
            }
        )
    return pd.DataFrame(rows)


def _label_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["class_label"].fillna("").astype(str)
        + " "
        + df["compound_label"].fillna("").astype(str)
        + " "
        + df["source_key"].fillna("").astype(str)
    ).str.lower()


def apply_pass5_filter_mode(grounding_df: pd.DataFrame, filter_mode: str) -> pd.DataFrame:
    work = grounding_df.copy()
    text = _label_text(work)
    metabolite_sources = {"adenine_sers_control", "metabolite_sers63_support", "ramanbiolib"}
    if filter_mode == "purine_focused_universal":
        filtered = work[work["source_key"].astype(str).isin(metabolite_sources) & text.str.contains("|".join(PURINE_FOCUSED_TERMS), regex=True)].copy()
    elif filter_mode == "purine_expanded_neighbor":
        filtered = work[work["source_key"].astype(str).isin(metabolite_sources) & text.str.contains("|".join(PURINE_EXPANDED_TERMS), regex=True)].copy()
    elif filter_mode == "balanced_metabolite_subset":
        filtered = work[work["source_key"].astype(str).isin(metabolite_sources)].copy()
    elif filter_mode == "purine_capped":
        filtered = work[work["source_key"].astype(str).isin(metabolite_sources) & text.str.contains("|".join(PURINE_EXPANDED_TERMS), regex=True)].copy()
    elif filter_mode == "diversity_enforced_universal":
        filtered = work.copy()
    else:
        raise ValueError(f"Unsupported pass 5 filter mode: {filter_mode}")
    if filtered.empty:
        raise RuntimeError(f"Pass 5 filter mode {filter_mode} produced an empty universal grounding pool.")
    return filtered.reset_index(drop=True)


def classify_compound_family(reference_compound_label: str, reference_source_key: str) -> str:
    text = f"{reference_compound_label} {reference_source_key}".lower()
    if any(re.search(term, text) for term in PURINE_EXPANDED_TERMS):
        if any(re.search(term, text) for term in [r"uric", r"urate", r"xanth", r"hypox", r"inos"]):
            return "purine_expanded_neighbor"
        return "purine_core"
    if any(re.search(term, text) for term in SULFUR_TERMS):
        return "sulfur_neighbor"
    if any(re.search(term, text) for term in AROMATIC_TERMS):
        return "aromatic_neighbor"
    if "amino_acid" in text or "amino" in text:
        return "amino_acid_like"
    if "ramanbiolib" in text:
        return "biomolecule_library_other"
    return "other_metabolite"


def compute_similarity_matrix(query_matrix: np.ndarray, grounding_matrix: np.ndarray, similarity_metric: str) -> np.ndarray:
    if similarity_metric == "cosine":
        return normalize_rows(query_matrix.astype(np.float32)) @ normalize_rows(grounding_matrix.astype(np.float32)).T
    if similarity_metric == "correlation":
        return correlation_normalize_rows(query_matrix.astype(np.float32)) @ correlation_normalize_rows(grounding_matrix.astype(np.float32)).T
    raise ValueError(f"Unsupported similarity metric: {similarity_metric}")


def _candidate_pool_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    pool_size = min(len(scores), max(top_k * 6, 24))
    order = np.argsort(scores)[::-1]
    return order[:pool_size]


def _select_diverse_indices(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    grounding_df: pd.DataFrame,
    top_k: int,
    diversity_mode: str,
    family_min_coverage: int,
) -> list[int]:
    selected: list[int] = []
    compound_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    candidate_list = list(candidate_indices.tolist())
    for _ in range(min(top_k, len(candidate_list))):
        best_idx = None
        best_score = -1e18
        uncovered_families = family_min_coverage > 0 and len({f for f, c in family_counts.items() if c > 0}) < family_min_coverage
        for idx in candidate_list:
            if idx in selected:
                continue
            ref = grounding_df.iloc[int(idx)]
            compound = str(ref.get("compound_label", ""))
            family = classify_compound_family(compound, str(ref.get("source_key", "")))
            score = float(scores[int(idx)])
            if diversity_mode == "compound_uniqueness_penalty":
                score -= 0.12 * compound_counts.get(compound, 0)
            elif diversity_mode == "family_balance_penalty":
                score -= 0.08 * family_counts.get(family, 0)
            elif diversity_mode == "min_family_coverage_constraint":
                if uncovered_families and family_counts.get(family, 0) == 0:
                    score += 0.5
                elif uncovered_families and family_counts.get(family, 0) > 0:
                    score -= 0.25
            if score > best_score:
                best_score = score
                best_idx = int(idx)
        if best_idx is None:
            break
        selected.append(best_idx)
        ref = grounding_df.iloc[int(best_idx)]
        compound = str(ref.get("compound_label", ""))
        family = classify_compound_family(compound, str(ref.get("source_key", "")))
        compound_counts[compound] = compound_counts.get(compound, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
    return selected


def _weights_uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / max(n, 1), dtype=float)


def _weights_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    stable = (scores - scores.max()) / temp
    exp = np.exp(stable)
    return exp / np.maximum(exp.sum(), 1e-12)


def _weights_rank_decay(n: int, decay: float) -> np.ndarray:
    decay = float(decay)
    ranks = np.arange(n, dtype=float)
    weights = np.exp(-decay * ranks)
    return weights / np.maximum(weights.sum(), 1e-12)


def _weights_max_cap_per_compound(scores: np.ndarray, compounds: list[str], cap: float) -> np.ndarray:
    weights = _weights_softmax(scores, 1.0)
    cap = min(max(float(cap), 0.05), 1.0)
    compound_totals = {}
    for compound, weight in zip(compounds, weights, strict=False):
        compound_totals[compound] = compound_totals.get(compound, 0.0) + float(weight)
    capped = weights.copy()
    released = 0.0
    for i, compound in enumerate(compounds):
        total = compound_totals[compound]
        if total > cap:
            new_weight = capped[i] * (cap / total)
            released += capped[i] - new_weight
            capped[i] = new_weight
    free_mask = np.array([compound_totals[c] <= cap + 1e-12 for c in compounds], dtype=bool)
    if released > 0 and free_mask.any():
        addon = capped[free_mask]
        if addon.sum() <= 1e-12:
            capped[free_mask] += released / free_mask.sum()
        else:
            capped[free_mask] += released * (addon / addon.sum())
    return capped / np.maximum(capped.sum(), 1e-12)


def compute_support_weights(
    selected_scores: np.ndarray,
    selected_compounds: list[str],
    weighting_mode: str,
    weighting_param: float | None,
) -> np.ndarray:
    n = len(selected_scores)
    if n == 0:
        return np.zeros(0, dtype=float)
    if weighting_mode == "uniform_weighting":
        return _weights_uniform(n)
    if weighting_mode == "softmax_temperature":
        return _weights_softmax(selected_scores, 1.0 if weighting_param is None else float(weighting_param))
    if weighting_mode == "max_cap_per_compound":
        return _weights_max_cap_per_compound(selected_scores, selected_compounds, 0.5 if weighting_param is None else float(weighting_param))
    if weighting_mode == "rank_decay_weighting":
        return _weights_rank_decay(n, 0.75 if weighting_param is None else float(weighting_param))
    raise ValueError(f"Unsupported weighting mode: {weighting_mode}")


def build_bsv_profiles_pass5(
    query_df: pd.DataFrame,
    grounding_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    *,
    top_k: int,
    similarity_metric: str,
    weighting_mode: str,
    weighting_param: float | None,
    diversity_mode: str,
    family_min_coverage: int,
    axis_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if axis_names is None:
        axis_names = ALL_AXES
    _, query_matrix, grounding_matrix = align_query_and_grounding(query_df, grounding_df)
    sim_matrix = compute_similarity_matrix(query_matrix, grounding_matrix, similarity_metric)
    mapping_grouped = {
        str(sample_key): group[["output_axis", "axis_weight"]].to_dict("records")
        for sample_key, group in mapping_df.groupby("sample_key", sort=False)
    }
    profile_rows = []
    hit_rows = []
    for i, query_row in enumerate(query_df.itertuples(index=False)):
        row_scores = sim_matrix[i].astype(float)
        candidate_idx = _candidate_pool_indices(row_scores, top_k)
        selected = _select_diverse_indices(candidate_idx, row_scores, grounding_df, top_k, diversity_mode, family_min_coverage)
        selected_scores = np.array([row_scores[idx] for idx in selected], dtype=float)
        selected_compounds = [str(grounding_df.iloc[idx].get("compound_label", "")) for idx in selected]
        weights = compute_support_weights(selected_scores, selected_compounds, weighting_mode, weighting_param)
        axis_scores = {axis: 0.0 for axis in axis_names}
        unmapped_support = 0.0
        for rank, (idx, sim, weight) in enumerate(zip(selected, selected_scores, weights, strict=False), start=1):
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
                    "reference_family": classify_compound_family(str(ref.get("compound_label", "")), str(ref.get("source_key", ""))),
                }
            )
        total = sum(axis_scores.values())
        if total > 0:
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


def compute_mixture_metrics(group_means_df: pd.DataFrame, retrieval_df: pd.DataFrame) -> dict[str, float]:
    class_neighborhood_df = build_class_topk_neighborhood_composition(retrieval_df)
    neighborhood_entropy_df = build_class_neighborhood_entropy(class_neighborhood_df)
    top1_df = build_class_top1_dominance(class_neighborhood_df)
    axis_entropy_df = build_class_axis_entropy(group_means_df)
    progression_df = build_mixture_progression_summary(group_means_df, class_neighborhood_df)
    _, inter_class_distance_df = compute_stability_tables(group_means_df, group_means_df)
    ordered = infer_mixture_order(group_means_df["class_label"].astype(str).tolist())
    work = group_means_df.set_index("class_label").loc[ordered]
    axes = [axis for axis in ALL_AXES if axis in work.columns]
    adj_distances = []
    for left, right in zip(ordered[:-1], ordered[1:], strict=False):
        diff = work.loc[left, axes].to_numpy(dtype=float) - work.loc[right, axes].to_numpy(dtype=float)
        adj_distances.append(float(np.linalg.norm(diff)))
    adjacent_nonzero_ratio = float(np.mean([d > 1e-6 for d in adj_distances])) if adj_distances else 0.0
    unique_profiles = np.unique(np.round(work[axes].to_numpy(dtype=float), 8), axis=0).shape[0]
    noncollapse_ratio = unique_profiles / max(len(ordered), 1)
    order_numeric = pd.Series([int("".join(ch for ch in label if ch.isdigit()) or 0) for label in ordered], dtype=float)
    small_series = pd.Series(work["small_molecule_metabolite"].to_list(), dtype=float)
    nuc_series = pd.Series(work["nucleic_acid"].to_list(), dtype=float)
    small_spearman = max(float(order_numeric.corr(small_series, method="spearman")), 0.0) if len(order_numeric) > 1 else 0.0
    nucleic_inverse_spearman = max(float(-order_numeric.corr(nuc_series, method="spearman")), 0.0) if len(order_numeric) > 1 else 0.0
    mean_neighborhood_entropy = float(neighborhood_entropy_df["neighborhood_entropy"].mean()) if not neighborhood_entropy_df.empty else 0.0
    mean_axis_entropy = float(axis_entropy_df["axis_entropy"].mean()) if not axis_entropy_df.empty else 0.0
    entropy_norm = mean_neighborhood_entropy / max(math.log2(max(int(class_neighborhood_df["compound_label"].nunique()), 2)), 1e-12)
    axis_entropy_norm = mean_axis_entropy / max(math.log2(len(PRIMARY_AXES) + 3), 1e-12)
    mean_top1 = float(top1_df["top1_fraction"].mean()) if not top1_df.empty else 1.0
    collapse_penalty = 1.0 - noncollapse_ratio
    saturation_penalty = max(mean_top1 - 0.85, 0.0) * 2.0 + collapse_penalty + (1.0 - adjacent_nonzero_ratio)
    mixture_progression_score = (
        0.9 * adjacent_nonzero_ratio
        + 0.9 * noncollapse_ratio
        + 0.7 * small_spearman
        + 0.5 * nucleic_inverse_spearman
    )
    diversity_score = 0.7 * entropy_norm + 0.3 * axis_entropy_norm
    return {
        "mean_neighborhood_entropy": mean_neighborhood_entropy,
        "mean_axis_entropy": mean_axis_entropy,
        "mean_top1_dominance": mean_top1,
        "adjacent_nonzero_ratio": adjacent_nonzero_ratio,
        "noncollapse_ratio": noncollapse_ratio,
        "small_molecule_spearman": small_spearman,
        "nucleic_inverse_spearman": nucleic_inverse_spearman,
        "mixture_progression_score": mixture_progression_score,
        "diversity_score": diversity_score,
        "saturation_penalty": saturation_penalty,
        "progression_df": progression_df,
        "class_neighborhood_df": class_neighborhood_df,
        "class_neighborhood_entropy_df": neighborhood_entropy_df,
        "class_top1_df": top1_df,
        "class_axis_entropy_df": axis_entropy_df,
        "inter_class_distance_df": inter_class_distance_df,
    }


def write_pass5_run_artifacts(
    sprint_paths: AutoresearchSprintPaths,
    config: Pass5HarnessConfig,
    panel_name: str,
    outputs: dict[str, object],
) -> Path:
    run_dir = sprint_paths.runs_dir / config.config_id / panel_name
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs["group_means_df"].to_csv(run_dir / "group_mean_bsv.csv", index=False)
    outputs["retrieval_summary_df"].to_csv(run_dir / "retrieval_hit_summary.csv", index=False)
    outputs["retrieval_df"].to_csv(run_dir / "retrieval_hits.csv", index=False)
    if outputs.get("delta_df") is not None:
        outputs["delta_df"].to_csv(run_dir / "delta_bsv.csv", index=False)
    return run_dir


def save_pass5_tables(
    sprint_paths: AutoresearchSprintPaths,
    search_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    best_by_panel_df: pd.DataFrame,
    mixture_metrics_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    dominance_df: pd.DataFrame,
) -> None:
    search_df.to_csv(sprint_paths.tables_dir / "calibration_search_space_used.csv", index=False)
    ranked_df.to_csv(sprint_paths.tables_dir / "calibration_results_ranked.csv", index=False)
    best_by_panel_df.to_csv(sprint_paths.tables_dir / "best_config_by_panel.csv", index=False)
    mixture_metrics_df.to_csv(sprint_paths.tables_dir / "mixture_progression_metrics.csv", index=False)
    entropy_df.to_csv(sprint_paths.tables_dir / "entropy_metrics.csv", index=False)
    dominance_df.to_csv(sprint_paths.tables_dir / "dominance_metrics.csv", index=False)


def _barplot(values: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path, color: str = "#4c78a8") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    order = values.sort_values(y_col, ascending=False).reset_index(drop=True)
    x = np.arange(len(order))
    ax.bar(x, order[y_col], color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(order[x_col].astype(str).tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pass5_figures(
    sprint_paths: AutoresearchSprintPaths,
    ranked_df: pd.DataFrame,
    mixture_metrics_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    dominance_df: pd.DataFrame,
    baseline_progression_df: pd.DataFrame,
    best_progression_df: pd.DataFrame,
) -> list[Path]:
    figure_paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ordered = baseline_progression_df.sort_values("mixture_code_numeric")
    best_ordered = best_progression_df.sort_values("mixture_code_numeric")
    ax.plot(ordered["class_label"], ordered["toward_high_endpoint_score"], marker="o", linewidth=2.0, label="baseline toward-high")
    ax.plot(best_ordered["class_label"], best_ordered["toward_high_endpoint_score"], marker="s", linewidth=2.0, label="best toward-high")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Mixture Progression Alignment: Baseline vs Best")
    ax.set_ylabel("Toward high-endpoint score")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])
    out = sprint_paths.figures_dir / "mixture_progression_alignment.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(out)

    _barplot(dominance_df, "config_id", "mean_top1_dominance", "Top1 Dominance vs Config", sprint_paths.figures_dir / "top1_dominance_vs_config.png", color="#6c757d")
    figure_paths.append(sprint_paths.figures_dir / "top1_dominance_vs_config.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    order = entropy_df.sort_values("diversity_score", ascending=False).reset_index(drop=True)
    x = np.arange(len(order))
    width = 0.38
    ax.bar(x - width / 2, order["mean_neighborhood_entropy"], width=width, label="mean neighborhood entropy", color="#2a9d8f")
    ax.bar(x + width / 2, order["mean_axis_entropy"], width=width, label="mean axis entropy", color="#e9c46a")
    ax.set_xticks(x)
    ax.set_xticklabels(order["config_id"].astype(str).tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_title("Entropy vs Config")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])
    out = sprint_paths.figures_dir / "entropy_vs_config.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(out)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.scatter(ranked_df["validation_score"], ranked_df["mixture_panel_score"], s=55, alpha=0.85, color="#1d3557")
    for row in ranked_df.itertuples(index=False):
        ax.text(float(row.validation_score) + 0.01, float(row.mixture_panel_score) + 0.01, str(row.config_id), fontsize=8)
    ax.set_xlabel("Validation score")
    ax.set_ylabel("Mixture panel score")
    ax.set_title("Validation vs Mixture Tradeoff")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    out = sprint_paths.figures_dir / "validation_vs_mixture_tradeoff.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(out)
    return figure_paths


def build_pass5_markdown_report(
    sprint_paths: AutoresearchSprintPaths,
    ranked_df: pd.DataFrame,
    best_by_panel_df: pd.DataFrame,
    baseline_row: pd.Series,
    best_row: pd.Series,
    holdout_metrics: dict[str, float] | None,
) -> Path:
    lines = [
        "# GAIRA Autoresearch v1 Pass 5 Report",
        "",
        "## Fixed Assumptions",
        "- Canonical preprocessing remained fixed.",
        "- Grounding mode remained `universal_only`.",
        "- Representation remained `raw_direct_bsv_input`.",
        "- Aggregation remained `class_mean_spectrum_then_bsv`.",
        "- Ontology remained `tier1_plus_subclass`.",
        "- Similarity remained `cosine`.",
        "- PCA grouping remained `class_label_groups`.",
        "",
        "## Search Space",
        f"- evaluated configurations: `{len(ranked_df)}`",
        "- validation panels used in scoring: `cspp_metabolite_spike_validation`, `serum_ag_uricase_validation`",
        "- target-side saturation panel used in scoring: `small2023_mixture_probe1`",
        "- holdout excluded from search: `small2023_mixture_probe2`",
        "",
        "## Best Overall Config",
        f"- config_id: `{best_row['config_id']}`",
        f"- filter: `{best_row['universal_grounding_filter_mode']}`",
        f"- top_k: `{int(best_row['top_k'])}`",
        f"- weighting: `{best_row['weighting_mode']}`",
        f"- weighting_param: `{best_row['weighting_param']}`",
        f"- diversity_mode: `{best_row['diversity_mode']}`",
        f"- overall_score: `{best_row['overall_score']:.4f}`",
        "",
        "## Best Config By Panel",
    ]
    for _, row in best_by_panel_df.iterrows():
        lines.append(
            f"- `{row['panel_name']}` -> `{row['config_id']}`: panel_score=`{row['panel_score']:.4f}`"
        )
    lines.extend(
        [
            "",
            "## Baseline vs Best",
            f"- baseline config `{baseline_row['config_id']}` overall=`{baseline_row['overall_score']:.4f}`, validation=`{baseline_row['validation_score']:.4f}`, mixture=`{baseline_row['mixture_panel_score']:.4f}`, saturation_penalty=`{baseline_row['saturation_penalty']:.4f}`",
            f"- best config `{best_row['config_id']}` overall=`{best_row['overall_score']:.4f}`, validation=`{best_row['validation_score']:.4f}`, mixture=`{best_row['mixture_panel_score']:.4f}`, saturation_penalty=`{best_row['saturation_penalty']:.4f}`",
            "",
            "## Interpretation",
            "- Pass 5 was designed to reduce single-compound dominance without breaking the validation panels.",
            "- The critical question is whether mixture diversity improves through anti-saturation behavior rather than by simply flattening chemically meaningful validation signals.",
        ]
    )
    if holdout_metrics:
        lines.extend(
            [
                "",
                "## Holdout Probe2 Check",
                f"- holdout mixture panel score: `{holdout_metrics['mixture_panel_score']:.4f}`",
                f"- holdout mean top1 dominance: `{holdout_metrics['mean_top1_dominance']:.4f}`",
                f"- holdout noncollapse ratio: `{holdout_metrics['noncollapse_ratio']:.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Direct Answers",
            f"1. Did we reduce saturation? `{'yes' if float(best_row['mean_top1_dominance']) < float(baseline_row['mean_top1_dominance']) and float(best_row['saturation_penalty']) < float(baseline_row['saturation_penalty']) else 'no'}`",
            f"2. Did mixture progression become smoother? `{'yes' if float(best_row['mixture_progression_score']) > float(baseline_row['mixture_progression_score']) else 'no'}`",
            f"3. Did validation panels degrade? `{'no' if float(best_row['validation_score']) >= float(baseline_row['validation_score']) else 'yes'}`",
            "4. What is the new recommended deterministic baseline?",
            f"   `filter={best_row['universal_grounding_filter_mode']}; top_k={int(best_row['top_k'])}; weighting={best_row['weighting_mode']}; diversity={best_row['diversity_mode']}`",
        ]
    )
    out = sprint_paths.report_dir / "GAIRA_autoresearch_v1_pass5_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_pdf_report(markdown_path: Path, figure_paths: list[Path], output_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    lines = []
    for raw in text.splitlines():
        if raw.startswith("#"):
            lines.append(raw)
        elif raw.strip():
            lines.extend(textwrap.wrap(raw, width=96))
        else:
            lines.append("")
    with PdfPages(output_path) as pdf:
        chunk_size = 34
        for i in range(0, len(lines), chunk_size):
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.96
            for line in lines[i : i + chunk_size]:
                size = 12 if line.startswith("# ") else 10 if line.startswith("## ") else 8.6
                weight = "bold" if line.startswith("#") else "normal"
                fig.text(0.06, y, line, ha="left", va="top", fontsize=size, fontweight=weight, family="DejaVu Sans Mono")
                y -= 0.026 if line.startswith("#") else 0.023
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        for path in figure_paths:
            img = plt.imread(path)
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(path.name, fontsize=12, y=0.98)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
