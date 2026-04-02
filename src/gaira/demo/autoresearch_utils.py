from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns

from gaira.autoresearch_storage import AutoresearchSprintPaths
from gaira.demo.gaira_experiment_runner_utils import (
    build_source_role_sets,
    load_architecture_registries,
    load_grounding_family_dataframe,
    load_query_dataframe,
    resolve_experiment,
    retrieval_hit_summary,
)
from gaira.demo.raw_bsv_pilot_utils import (
    ALL_AXES,
    PRIMARY_AXES,
    apply_source_role_policy,
    build_bsv_profiles,
    build_group_mean_query_df,
    compute_local_pca,
    group_mean_bsv,
    load_ontology_rules,
    map_references_to_axes,
)


@dataclass(frozen=True)
class HarnessConfig:
    experiment_id: str
    subset_alias: str
    panel_name: str
    grounding_mode: str
    ontology_mode: str
    similarity_metric: str
    aggregation_mode: str
    pca_grouping_mode: str

    @property
    def run_id(self) -> str:
        return "__".join(
            [
                self.panel_name,
                self.grounding_mode,
                self.ontology_mode,
                self.similarity_metric,
                self.aggregation_mode,
                self.pca_grouping_mode,
            ]
        )


def build_search_space() -> pd.DataFrame:
    panels = [
        ("exp_diff_cspp_metabolite_spike", "cspp_metabolite_spike_validation"),
        ("exp_localdiff_serum_uricase", "serum_ag_uricase_validation"),
    ]
    grounding_modes = [
        "universal_only",
        "universal_plus_domain_biochemical",
        "universal_plus_caveat_support",
    ]
    ontology_modes = ["tier1_only", "tier1_plus_subclass"]
    similarity_metrics = ["cosine", "correlation"]
    aggregation_modes = ["per_spectrum_then_group_mean", "class_mean_spectrum_then_bsv"]
    pca_grouping_modes = ["none", "class_label_groups", "local_cluster_summary"]

    rows = []
    for experiment_id, subset_alias in panels:
        panel_name = subset_alias
        for grounding_mode in grounding_modes:
            for ontology_mode in ontology_modes:
                for similarity_metric in similarity_metrics:
                    for aggregation_mode in aggregation_modes:
                        for pca_grouping_mode in pca_grouping_modes:
                            status = "valid"
                            reason = ""
                            if pca_grouping_mode == "none":
                                status = "excluded"
                                reason = "GAIRAv3 v2 treats PCA as always-on local structure rather than an optional off switch."
                            elif pca_grouping_mode == "local_cluster_summary":
                                status = "excluded"
                                reason = "Local cluster summaries are not yet bounded enough for autoresearch v1."
                            rows.append(
                                {
                                    "experiment_id": experiment_id,
                                    "subset_alias": subset_alias,
                                    "panel_name": panel_name,
                                    "grounding_mode": grounding_mode,
                                    "ontology_mode": ontology_mode,
                                    "similarity_metric": similarity_metric,
                                    "aggregation_mode": aggregation_mode,
                                    "pca_grouping_mode": pca_grouping_mode,
                                    "status": status,
                                    "exclusion_reason": reason,
                                }
                            )
    return pd.DataFrame(rows)


def ontology_path_for_mode(project_root: Path, ontology_mode: str) -> Path:
    if ontology_mode == "tier1_only":
        return project_root / "config" / "phase2_bsv_ontology_rules_v1.csv"
    if ontology_mode == "tier1_plus_subclass":
        return project_root / "config" / "phase2_bsv_ontology_rules_v2.csv"
    raise ValueError(f"Unsupported ontology mode: {ontology_mode}")


def families_for_grounding_mode(grounding_mode: str) -> list[str]:
    if grounding_mode == "universal_only":
        return ["universal_biochemical_grounding"]
    if grounding_mode == "universal_plus_domain_biochemical":
        return ["universal_biochemical_grounding", "domain_specific_biochemical_grounding"]
    if grounding_mode == "universal_plus_caveat_support":
        return [
            "universal_biochemical_grounding",
            "domain_specific_biochemical_grounding",
            "domain_specific_caveat_support_grounding",
        ]
    raise ValueError(f"Unsupported grounding_mode: {grounding_mode}")


def comparator_map_for_alias(alias: str, group_means: pd.DataFrame) -> dict[str, str]:
    labels = set(group_means["class_label"].astype(str))
    if alias == "cspp_metabolite_spike_validation":
        return {label: "Bkg" for label in labels if label != "Bkg"}
    if alias == "serum_ag_uricase_validation":
        mapping = {}
        if "SerumSigma+Enzyme" in labels and "SerumSigma" in labels:
            mapping["SerumSigma+Enzyme"] = "SerumSigma"
        if "Serumspiked+Enzyme" in labels and "Serumspiked" in labels:
            mapping["Serumspiked+Enzyme"] = "Serumspiked"
        return mapping
    raise ValueError(f"No comparator mapping defined for alias {alias}")


def compute_delta_by_mapping(group_means: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    rows = []
    lookup = group_means.set_index("class_label")
    axis_names = [axis for axis in group_means.columns if axis != "class_label"]
    for group_label, ref_label in mapping.items():
        row = lookup.loc[group_label]
        ref = lookup.loc[ref_label]
        record = {
            "comparison": f"{group_label}-vs-{ref_label}",
            "group_label": group_label,
            "reference_group": ref_label,
        }
        for axis in axis_names:
            record[axis] = float(row[axis] - ref[axis])
        rows.append(record)
    return pd.DataFrame(rows)


def compute_panel_structural_summary(query_df: pd.DataFrame) -> pd.DataFrame:
    _, _, pca_df = compute_local_pca(query_df, n_components=3)
    pca_cols = [c for c in pca_df.columns if c.startswith("pc") and "_explained_ratio" not in c]
    grouped = pca_df.groupby("class_label")[pca_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "class_label" if col[0] == "class_label" else f"{col[0]}_{col[1]}"
        for col in grouped.columns
    ]
    return grouped


def aggregate_query_df(query_df: pd.DataFrame, aggregation_mode: str) -> pd.DataFrame:
    if aggregation_mode == "per_spectrum_then_group_mean":
        return query_df.copy()
    if aggregation_mode == "class_mean_spectrum_then_bsv":
        return build_group_mean_query_df(query_df, group_col="class_label")
    raise ValueError(f"Unsupported aggregation_mode: {aggregation_mode}")


def compute_run_outputs(
    *,
    resolved,
    registries,
    project_root: Path,
    harness_config: HarnessConfig,
) -> dict[str, object]:
    query_df = load_query_dataframe(resolved.dataset_row)
    structural_summary_df = compute_panel_structural_summary(query_df)

    original_grounding_names = resolved.grounding_family_names
    object.__setattr__(resolved, "grounding_family_names", families_for_grounding_mode(harness_config.grounding_mode))
    try:
        grounding_df, family_to_sources, unavailable_sources = load_grounding_family_dataframe(resolved, registries)
    finally:
        object.__setattr__(resolved, "grounding_family_names", original_grounding_names)

    primary_sources, caveat_only_sources = build_source_role_sets(resolved, family_to_sources)
    ontology_rules = load_ontology_rules(ontology_path_for_mode(project_root, harness_config.ontology_mode))
    raw_mapping_df = map_references_to_axes(grounding_df, ontology_rules)
    mapping_df = apply_source_role_policy(
        raw_mapping_df,
        grounding_df,
        primary_sources=primary_sources,
        caveat_only_sources=caveat_only_sources,
    )
    query_input_df = aggregate_query_df(query_df, harness_config.aggregation_mode)
    per_spectrum_df, retrieval_df = build_bsv_profiles(
        query_input_df,
        grounding_df,
        mapping_df,
        top_k=8,
        normalization_mode="per_spectrum_sum",
        similarity_metric=harness_config.similarity_metric,
    )
    group_means_df = group_mean_bsv(per_spectrum_df, group_col="class_label")
    comparator_map = comparator_map_for_alias(harness_config.subset_alias, group_means_df)
    delta_df = compute_delta_by_mapping(group_means_df, comparator_map)
    retrieval_summary_df = retrieval_hit_summary(retrieval_df)

    return {
        "query_df": query_df,
        "structural_summary_df": structural_summary_df,
        "grounding_df": grounding_df,
        "mapping_df": mapping_df,
        "per_spectrum_df": per_spectrum_df,
        "group_means_df": group_means_df,
        "delta_df": delta_df,
        "retrieval_df": retrieval_df,
        "retrieval_summary_df": retrieval_summary_df,
        "available_sources": sorted(grounding_df["source_key"].astype(str).unique().tolist()),
        "unavailable_sources": unavailable_sources,
        "primary_sources": sorted(primary_sources),
        "caveat_only_sources": sorted(caveat_only_sources),
    }


def _primary_axis_score(delta_df: pd.DataFrame, group_label: str, preferred_axes: list[str]) -> float:
    row = delta_df[delta_df["group_label"].astype(str) == str(group_label)]
    if row.empty:
        return 0.0
    rec = row.iloc[0]
    for axis in preferred_axes:
        if axis in rec.index:
            return float(rec[axis])
    return 0.0


def _top_hit_plausibility_cspp(retrieval_summary_df: pd.DataFrame) -> float:
    expected_map = {
        "Hyp": ["Hypox", "Xanth", "Ade", "Adenine", "Gua", "UA"],
        "Erg": ["Ergo", "Glutathione", "Cysteamine", "Homocysteine", "Homocystine"],
    }
    off_target = ["UA+HSA", "UAbound", "SerumSigma", "Glycerol"]
    score = 0.0
    for label, expected_terms in expected_map.items():
        sub = retrieval_summary_df[retrieval_summary_df["query_class_label"].astype(str) == label].copy()
        total = float(sub["total_support_weight"].sum())
        if total <= 0:
            continue
        expected_weight = float(
            sub[
                sub["reference_compound_label"].astype(str).apply(
                    lambda x: any(term.lower() in x.lower() for term in expected_terms)
                )
            ]["total_support_weight"].sum()
        )
        off_weight = float(
            sub[
                sub["reference_compound_label"].astype(str).apply(
                    lambda x: any(term.lower() in x.lower() for term in off_target)
                )
            ]["total_support_weight"].sum()
        )
        score += (expected_weight - off_weight) / total
    return score / max(len(expected_map), 1)


def _top_hit_plausibility_uricase(retrieval_summary_df: pd.DataFrame) -> float:
    spiked = retrieval_summary_df[retrieval_summary_df["query_class_label"].astype(str) == "Serumspiked+Enzyme"].copy()
    sigma = retrieval_summary_df[retrieval_summary_df["query_class_label"].astype(str) == "SerumSigma+Enzyme"].copy()
    if spiked.empty or sigma.empty:
        return 0.0
    terms = ["Hypox", "Xanth", "UA", "Ade", "Gua"]
    spiked_total = float(spiked["total_support_weight"].sum()) or 1.0
    sigma_total = float(sigma["total_support_weight"].sum()) or 1.0
    spiked_weight = float(
        spiked[spiked["reference_compound_label"].astype(str).apply(lambda x: any(t.lower() in x.lower() for t in terms))][
            "total_support_weight"
        ].sum()
    )
    sigma_weight = float(
        sigma[sigma["reference_compound_label"].astype(str).apply(lambda x: any(t.lower() in x.lower() for t in terms))][
            "total_support_weight"
        ].sum()
    )
    return (spiked_weight / spiked_total) - (sigma_weight / sigma_total)


def score_panel_outputs(panel_name: str, outputs: dict[str, object]) -> dict[str, float]:
    delta_df = outputs["delta_df"]
    retrieval_summary_df = outputs["retrieval_summary_df"]
    metrics: dict[str, float] = {}

    caveat_cols = [axis for axis in ["matrix_background", "substrate_adsorption_bias", "protocol_sensitive_signal"] if axis in delta_df.columns]
    metrics["matrix_collapse_penalty"] = float(delta_df["matrix_background"].abs().mean()) if "matrix_background" in delta_df.columns else 0.0
    metrics["caveat_domination_penalty"] = float(delta_df[caveat_cols].abs().sum(axis=1).mean()) if caveat_cols else 0.0

    if panel_name == "cspp_metabolite_spike_validation":
        hyp_uplift = max(
            _primary_axis_score(delta_df, "Hyp", ["purine_like_metabolite", "small_molecule_metabolite"]),
            0.0,
        )
        erg_uplift = max(
            _primary_axis_score(delta_df, "Erg", ["sulfur_containing_metabolite", "small_molecule_metabolite"]),
            0.0,
        )
        metrics["expected_axis_uplift_score"] = (hyp_uplift + erg_uplift) / 2.0
        metrics["top_hit_plausibility_score"] = _top_hit_plausibility_cspp(retrieval_summary_df)
    elif panel_name == "serum_ag_uricase_validation":
        spiked_score = _primary_axis_score(
            delta_df,
            "Serumspiked+Enzyme",
            ["purine_like_metabolite", "small_molecule_metabolite"],
        )
        sigma_score = _primary_axis_score(
            delta_df,
            "SerumSigma+Enzyme",
            ["purine_like_metabolite", "small_molecule_metabolite"],
        )
        metrics["expected_axis_uplift_score"] = max(spiked_score - sigma_score, 0.0)
        metrics["top_hit_plausibility_score"] = _top_hit_plausibility_uricase(retrieval_summary_df)
    else:
        metrics["expected_axis_uplift_score"] = 0.0
        metrics["top_hit_plausibility_score"] = 0.0

    metrics["stability_proxy_score"] = float(outputs["group_means_df"]["unmapped_support"].max() <= 0.05)
    metrics["overall_score"] = (
        2.5 * metrics["expected_axis_uplift_score"]
        + 1.5 * metrics["top_hit_plausibility_score"]
        + 0.5 * metrics["stability_proxy_score"]
        - 1.0 * metrics["caveat_domination_penalty"]
        - 1.5 * metrics["matrix_collapse_penalty"]
    )
    return metrics


def write_run_artifacts(
    sprint_paths: AutoresearchSprintPaths,
    harness_config: HarnessConfig,
    outputs: dict[str, object],
    scores: dict[str, float],
) -> Path:
    run_dir = sprint_paths.runs_dir / harness_config.panel_name / harness_config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "run_id": harness_config.run_id,
        "experiment_id": harness_config.experiment_id,
        "subset_alias": harness_config.subset_alias,
        "grounding_mode": harness_config.grounding_mode,
        "ontology_mode": harness_config.ontology_mode,
        "similarity_metric": harness_config.similarity_metric,
        "aggregation_mode": harness_config.aggregation_mode,
        "pca_grouping_mode": harness_config.pca_grouping_mode,
        "available_sources": outputs["available_sources"],
        "unavailable_sources": outputs["unavailable_sources"],
        "primary_sources": outputs["primary_sources"],
        "caveat_only_sources": outputs["caveat_only_sources"],
        "scores": scores,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
    outputs["structural_summary_df"].to_csv(run_dir / "pca_structural_summary.csv", index=False)
    outputs["mapping_df"].to_csv(run_dir / "ontology_mapping_applied.csv", index=False)
    outputs["per_spectrum_df"].to_csv(run_dir / "per_spectrum_bsv.csv", index=False)
    outputs["group_means_df"].to_csv(run_dir / "group_mean_bsv.csv", index=False)
    outputs["delta_df"].to_csv(run_dir / "delta_bsv.csv", index=False)
    outputs["retrieval_summary_df"].to_csv(run_dir / "retrieval_hit_summary.csv", index=False)
    outputs["retrieval_df"].to_csv(run_dir / "topk_retrieval_hits.csv", index=False)
    return run_dir


def plot_leaderboards(results_df: pd.DataFrame, sprint_paths: AutoresearchSprintPaths) -> dict[str, Path]:
    paths = {}
    ranked = results_df.sort_values("overall_score", ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=ranked, y="run_id", x="overall_score", hue="panel_name")
    plt.title("GAIRAv3 Autoresearch v1 Overall Leaderboard")
    plt.xlabel("Overall score")
    plt.ylabel("")
    plt.tight_layout()
    out = sprint_paths.figures_dir / "leaderboard_overall.png"
    plt.savefig(out, dpi=220)
    plt.close()
    paths["leaderboard_overall"] = out

    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df.sort_values(["panel_name", "overall_score"], ascending=[True, False]), y="run_id", x="overall_score", hue="panel_name")
    plt.title("GAIRAv3 Autoresearch v1 Leaderboard by Panel")
    plt.xlabel("Overall score")
    plt.ylabel("")
    plt.tight_layout()
    out = sprint_paths.figures_dir / "leaderboard_by_panel.png"
    plt.savefig(out, dpi=220)
    plt.close()
    paths["leaderboard_by_panel"] = out
    return paths


def plot_factor_ablation(results_df: pd.DataFrame, sprint_paths: AutoresearchSprintPaths) -> Path:
    rows = []
    for factor in ["grounding_mode", "ontology_mode", "similarity_metric", "aggregation_mode"]:
        grouped = results_df.groupby(["panel_name", factor], as_index=False)["overall_score"].mean()
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "panel_name": row["panel_name"],
                    "factor_level": f"{factor}={row[factor]}",
                    "mean_overall_score": row["overall_score"],
                }
            )
    df = pd.DataFrame(rows)
    heat = df.pivot(index="factor_level", columns="panel_name", values="mean_overall_score").fillna(0.0)
    plt.figure(figsize=(10.5, max(5, 0.35 * len(heat))))
    sns.heatmap(heat, cmap="mako", annot=True, fmt=".2f")
    plt.title("Factor Ablation Heatmap")
    plt.tight_layout()
    out = sprint_paths.figures_dir / "factor_ablation_heatmap.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_best_vs_baseline(results_df: pd.DataFrame, sprint_paths: AutoresearchSprintPaths, panel_name: str, filename: str) -> Path:
    panel = results_df[results_df["panel_name"] == panel_name].copy().sort_values("overall_score", ascending=False)
    best = panel.iloc[0]
    baseline = panel[
        (panel["grounding_mode"] == "universal_only")
        & (panel["ontology_mode"] == "tier1_only")
        & (panel["similarity_metric"] == "cosine")
        & (panel["aggregation_mode"] == "per_spectrum_then_group_mean")
    ].iloc[0]
    comp = pd.DataFrame(
        [
            {"config_family": "baseline", "overall_score": float(baseline["overall_score"]), "expected_axis_uplift_score": float(baseline["expected_axis_uplift_score"]), "top_hit_plausibility_score": float(baseline["top_hit_plausibility_score"]), "matrix_collapse_penalty": float(baseline["matrix_collapse_penalty"])},
            {"config_family": "best", "overall_score": float(best["overall_score"]), "expected_axis_uplift_score": float(best["expected_axis_uplift_score"]), "top_hit_plausibility_score": float(best["top_hit_plausibility_score"]), "matrix_collapse_penalty": float(best["matrix_collapse_penalty"])},
        ]
    )
    long = comp.melt(id_vars=["config_family"], var_name="metric_name", value_name="metric_value")
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=long, x="metric_name", y="metric_value", hue="config_family")
    plt.xticks(rotation=20, ha="right")
    plt.title(f"Best vs Baseline: {panel_name}")
    plt.tight_layout()
    out = sprint_paths.figures_dir / filename
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def build_markdown_report(
    sprint_paths: AutoresearchSprintPaths,
    results_df: pd.DataFrame,
    search_df: pd.DataFrame,
    best_by_panel_df: pd.DataFrame,
) -> Path:
    total_runs = int((search_df["status"] == "valid").sum())
    completed = int((results_df["status"] == "completed").sum())
    failures = int((results_df["status"] == "failed").sum())
    lines = [
        "# GAIRA Autoresearch v1 Report",
        "",
        "## Fixed Assumptions",
        "- Canonical preprocessing was fixed: cropping, poly3 baseline correction, current common normalization/scaling path.",
        "- Biochemical attribution came from BSV only.",
        "- PCA remained the local structure layer and was not treated as the chemistry space.",
        "- Differential inference meant delta-BSV over absolute BSV outputs.",
        "- Residualized spectra were excluded from the core search.",
        "- Shared embeddings and RAG/context remained out of scope.",
        "",
        "## Search Space",
        f"- valid configurations executed: `{total_runs}`",
        f"- completed runs: `{completed}`",
        f"- failed runs: `{failures}`",
        "- searched factors: grounding_mode, ontology_mode, similarity_metric, aggregation_mode",
        "- pca_grouping_mode was recorded but bounded to `class_label_groups`; `none` and `local_cluster_summary` were explicitly excluded",
        "",
        "## Overall Findings",
    ]
    top_overall = results_df.sort_values("overall_score", ascending=False).head(5)
    for _, row in top_overall.iterrows():
        lines.append(
            f"- `{row['run_id']}`: panel=`{row['panel_name']}`, overall_score=`{row['overall_score']:.4f}`, uplift=`{row['expected_axis_uplift_score']:.4f}`, plausibility=`{row['top_hit_plausibility_score']:.4f}`"
        )
    lines.extend(["", "## Best Config by Panel"])
    for _, row in best_by_panel_df.iterrows():
        lines.append(
            f"- `{row['panel_name']}` best family: grounding=`{row['grounding_mode']}`, ontology=`{row['ontology_mode']}`, similarity=`{row['similarity_metric']}`, aggregation=`{row['aggregation_mode']}`, score=`{row['overall_score']:.4f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- This sprint was bounded and deterministic. It calibrated within the fixed GAIRAv3 framework rather than redesigning the framework.",
            "- The main question was whether one deterministic regime looked consistently stronger across both serum validation panels.",
            "- Results should be interpreted as calibration evidence, not as final proof of target-dataset readiness.",
            "",
            "## Recommendation",
            "- If one regime is consistently near the top across both panels, use it as the deterministic baseline for the next calibration phase.",
            "- If panel optima diverge strongly, keep the framework but prioritize chemistry-aware scoring and similarity calibration before any target-dataset expansion.",
            "- Escalation to a learned local query-grounding alignment encoder is only justified if the deterministic regime plateaus after this bounded sprint.",
        ]
    )
    out = sprint_paths.report_dir / "GAIRA_autoresearch_v1_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_pdf_report(
    markdown_path: Path,
    figure_paths: list[Path],
    output_path: Path,
) -> None:
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
            ax = fig.add_axes([0.03, 0.08, 0.94, 0.86])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(path.name, fontsize=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def save_summary_tables(
    sprint_paths: AutoresearchSprintPaths,
    search_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    search_df.to_csv(sprint_paths.tables_dir / "calibration_search_space_used.csv", index=False)
    ranked = results_df.sort_values(["panel_name", "overall_score"], ascending=[True, False]).copy()
    ranked.to_csv(sprint_paths.tables_dir / "calibration_results_ranked.csv", index=False)
    breakdown_cols = [
        "panel_name",
        "run_id",
        "expected_axis_uplift_score",
        "caveat_domination_penalty",
        "matrix_collapse_penalty",
        "top_hit_plausibility_score",
        "stability_proxy_score",
        "overall_score",
        "status",
    ]
    ranked[breakdown_cols].to_csv(sprint_paths.tables_dir / "panel_metric_breakdown.csv", index=False)
    best = ranked.groupby("panel_name", as_index=False).first()
    best.to_csv(sprint_paths.tables_dir / "best_config_by_panel.csv", index=False)
    return {"ranked": ranked, "best": best}
