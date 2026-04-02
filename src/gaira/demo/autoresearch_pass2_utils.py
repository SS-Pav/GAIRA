from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns

from gaira.autoresearch_storage import AutoresearchSprintPaths
from gaira.demo.autoresearch_utils import (
    comparator_map_for_alias,
    compute_delta_by_mapping,
    compute_panel_structural_summary,
)
from gaira.demo.gaira_experiment_runner_utils import (
    build_source_role_sets,
    load_grounding_family_dataframe,
    load_query_dataframe,
    retrieval_hit_summary,
)
from gaira.demo.raw_bsv_pilot_utils import (
    apply_source_role_policy,
    build_bsv_profiles,
    build_group_mean_query_df,
    group_mean_bsv,
    load_ontology_rules,
    map_references_to_axes,
)


FILTER_MODE_DEFINITIONS = {
    "all_universal": "Use all currently available universal biochemical grounding sources.",
    "metabolite_focused_universal": "Restrict universal grounding to metabolite-oriented sources and adenine controls; exclude the mixed amino-acid archive.",
    "purine_sulfur_neighbor_focused_universal": "Use only explicit universal purine-neighbor and sulfur-neighbor references supported by current grounding metadata.",
}

PURINE_NEIGHBOR_TERMS = [
    r"aden",
    r"guan",
    r"methyladenine",
    r"methylguanidine",
]
SULFUR_NEIGHBOR_TERMS = [
    r"glutath",
    r"cyste",
    r"cystine",
    r"cystath",
    r"homocys",
    r"methion",
    r"lipoamide",
    r"seleno",
]
OFFTARGET_BACKGROUND_TERMS = [
    "Alb",
    "Glucose",
    "Malic Acid",
    "Ala",
    "Arg",
    "Asp",
    "Gly",
    "His",
    "Leu",
    "Phe",
    "Pro",
    "Ser",
    "Trp",
    "Ure",
    "Valine",
]


@dataclass(frozen=True)
class Pass2HarnessConfig:
    experiment_id: str
    subset_alias: str
    panel_name: str
    universal_grounding_filter_mode: str
    top_k: int
    plausibility_scoring_mode: str
    similarity_metric: str = "cosine"
    ontology_mode: str = "tier1_plus_subclass"
    aggregation_mode: str = "class_mean_spectrum_then_bsv"
    grounding_mode: str = "universal_only"
    pca_grouping_mode: str = "class_label_groups"

    @property
    def run_id(self) -> str:
        return "__".join(
            [
                self.panel_name,
                self.universal_grounding_filter_mode,
                f"topk{self.top_k}",
                self.plausibility_scoring_mode,
                self.similarity_metric,
            ]
        )


def build_pass2_search_space() -> pd.DataFrame:
    panels = [
        ("exp_diff_cspp_metabolite_spike", "cspp_metabolite_spike_validation"),
        ("exp_localdiff_serum_uricase", "serum_ag_uricase_validation"),
    ]
    rows = []
    for experiment_id, subset_alias in panels:
        for filter_mode in FILTER_MODE_DEFINITIONS:
            for top_k in [3, 5, 8]:
                for scoring_mode in ["baseline_plausibility", "stricter_background_penalty"]:
                    rows.append(
                        {
                            "experiment_id": experiment_id,
                            "subset_alias": subset_alias,
                            "panel_name": subset_alias,
                            "grounding_mode": "universal_only",
                            "ontology_mode": "tier1_plus_subclass",
                            "similarity_metric": "cosine",
                            "aggregation_mode": "class_mean_spectrum_then_bsv",
                            "pca_grouping_mode": "class_label_groups",
                            "universal_grounding_filter_mode": filter_mode,
                            "top_k": top_k,
                            "plausibility_scoring_mode": scoring_mode,
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


def filter_universal_grounding_df(grounding_df: pd.DataFrame, filter_mode: str) -> pd.DataFrame:
    if filter_mode == "all_universal":
        filtered = grounding_df.copy()
    elif filter_mode == "metabolite_focused_universal":
        keep_sources = {"adenine_sers_control", "metabolite_sers63_support", "ramanbiolib"}
        filtered = grounding_df[grounding_df["source_key"].astype(str).isin(keep_sources)].copy()
    elif filter_mode == "purine_sulfur_neighbor_focused_universal":
        keep_sources = {"adenine_sers_control", "metabolite_sers63_support", "ramanbiolib"}
        work = grounding_df[grounding_df["source_key"].astype(str).isin(keep_sources)].copy()
        text = _label_text(work)
        pattern = "|".join(PURINE_NEIGHBOR_TERMS + SULFUR_NEIGHBOR_TERMS)
        filtered = work[text.str.contains(pattern, regex=True)].copy()
    else:
        raise ValueError(f"Unsupported universal grounding filter mode: {filter_mode}")
    if filtered.empty:
        raise RuntimeError(f"Grounding filter mode {filter_mode} produced an empty universal grounding pool.")
    return filtered.reset_index(drop=True)


def _expected_terms_for_panel(panel_name: str) -> dict[str, list[str]]:
    if panel_name == "cspp_metabolite_spike_validation":
        return {
            "Hyp": ["aden", "guan", "methyladenine", "methylguanidine"],
            "Erg": ["glutath", "cyste", "homocys", "methion", "lipoamide", "seleno"],
        }
    if panel_name == "serum_ag_uricase_validation":
        return {
            "Serumspiked+Enzyme": ["aden", "guan", "methyladenine", "methylguanidine"],
            "SerumSigma+Enzyme": ["aden", "guan", "methyladenine", "methylguanidine"],
        }
    return {}


def _offtarget_fraction(sub: pd.DataFrame) -> float:
    total = float(sub["total_support_weight"].sum())
    if total <= 0:
        return 0.0
    mask = sub["reference_compound_label"].astype(str).apply(
        lambda x: any(term.lower() in x.lower() for term in OFFTARGET_BACKGROUND_TERMS)
    )
    return float(sub.loc[mask, "total_support_weight"].sum()) / total


def _expected_fraction(sub: pd.DataFrame, terms: list[str]) -> float:
    total = float(sub["total_support_weight"].sum())
    if total <= 0:
        return 0.0
    mask = sub["reference_compound_label"].astype(str).apply(
        lambda x: any(term in x.lower() for term in terms)
    )
    return float(sub.loc[mask, "total_support_weight"].sum()) / total


def _plausibility_score(
    panel_name: str,
    retrieval_summary_df: pd.DataFrame,
    scoring_mode: str,
) -> float:
    expected_map = _expected_terms_for_panel(panel_name)
    if not expected_map:
        return 0.0
    values = []
    for query_label, terms in expected_map.items():
        sub = retrieval_summary_df[retrieval_summary_df["query_class_label"].astype(str) == query_label].copy()
        if sub.empty:
            values.append(0.0)
            continue
        expected = _expected_fraction(sub, terms)
        penalty = _offtarget_fraction(sub)
        if scoring_mode == "baseline_plausibility":
            values.append(expected - penalty)
        elif scoring_mode == "stricter_background_penalty":
            amino_acid_penalty = float(
                sub[
                    sub["reference_dataset_id"].astype(str).eq("amino_acid_raman_grounding")
                ]["total_support_weight"].sum()
            ) / max(float(sub["total_support_weight"].sum()), 1e-8)
            values.append(expected - (1.5 * penalty) - (1.25 * amino_acid_penalty))
        else:
            raise ValueError(f"Unsupported plausibility scoring mode: {scoring_mode}")
    return sum(values) / max(len(values), 1)


def _primary_axis_score(delta_df: pd.DataFrame, group_label: str, preferred_axes: list[str]) -> float:
    row = delta_df[delta_df["group_label"].astype(str) == str(group_label)]
    if row.empty:
        return 0.0
    rec = row.iloc[0]
    for axis in preferred_axes:
        if axis in rec.index:
            return float(rec[axis])
    return 0.0


def score_pass2_outputs(harness_config: Pass2HarnessConfig, outputs: dict[str, object]) -> dict[str, float]:
    delta_df = outputs["delta_df"]
    retrieval_summary_df = outputs["retrieval_summary_df"]
    metrics: dict[str, float] = {}

    caveat_cols = [
        axis
        for axis in ["matrix_background", "substrate_adsorption_bias", "protocol_sensitive_signal"]
        if axis in delta_df.columns
    ]
    metrics["matrix_collapse_penalty"] = (
        float(delta_df["matrix_background"].abs().mean()) if "matrix_background" in delta_df.columns else 0.0
    )
    metrics["caveat_domination_penalty"] = (
        float(delta_df[caveat_cols].abs().sum(axis=1).mean()) if caveat_cols else 0.0
    )

    if harness_config.panel_name == "cspp_metabolite_spike_validation":
        hyp_uplift = max(
            _primary_axis_score(delta_df, "Hyp", ["purine_like_metabolite", "small_molecule_metabolite"]),
            0.0,
        )
        erg_uplift = max(
            _primary_axis_score(delta_df, "Erg", ["sulfur_containing_metabolite", "small_molecule_metabolite"]),
            0.0,
        )
        metrics["expected_axis_uplift_score"] = (hyp_uplift + erg_uplift) / 2.0
    elif harness_config.panel_name == "serum_ag_uricase_validation":
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
    else:
        metrics["expected_axis_uplift_score"] = 0.0

    metrics["top_hit_plausibility_score"] = _plausibility_score(
        harness_config.panel_name,
        retrieval_summary_df,
        harness_config.plausibility_scoring_mode,
    )
    metrics["stability_proxy_score"] = float(outputs["group_means_df"]["unmapped_support"].max() <= 0.05)
    metrics["overall_score"] = (
        2.8 * metrics["expected_axis_uplift_score"]
        + 1.8 * metrics["top_hit_plausibility_score"]
        + 0.5 * metrics["stability_proxy_score"]
        - 1.0 * metrics["caveat_domination_penalty"]
        - 1.5 * metrics["matrix_collapse_penalty"]
    )
    return metrics


def compute_pass2_outputs(
    *,
    resolved,
    registries,
    project_root: Path,
    harness_config: Pass2HarnessConfig,
) -> dict[str, object]:
    query_df = load_query_dataframe(resolved.dataset_row)
    structural_summary_df = compute_panel_structural_summary(query_df)

    original_grounding_names = resolved.grounding_family_names
    object.__setattr__(resolved, "grounding_family_names", ["universal_biochemical_grounding"])
    try:
        grounding_df, family_to_sources, unavailable_sources = load_grounding_family_dataframe(resolved, registries)
    finally:
        object.__setattr__(resolved, "grounding_family_names", original_grounding_names)

    grounding_df = filter_universal_grounding_df(grounding_df, harness_config.universal_grounding_filter_mode)
    primary_sources, caveat_only_sources = build_source_role_sets(resolved, family_to_sources)
    primary_sources = {key for key in primary_sources if key in set(grounding_df["source_key"].astype(str))}
    caveat_only_sources = {key for key in caveat_only_sources if key in set(grounding_df["source_key"].astype(str))}

    ontology_rules = load_ontology_rules(project_root / "config" / "phase2_bsv_ontology_rules_v2.csv")
    raw_mapping_df = map_references_to_axes(grounding_df, ontology_rules)
    mapping_df = apply_source_role_policy(
        raw_mapping_df,
        grounding_df,
        primary_sources=primary_sources,
        caveat_only_sources=caveat_only_sources,
    )
    query_input_df = build_group_mean_query_df(query_df, group_col="class_label")
    per_spectrum_df, retrieval_df = build_bsv_profiles(
        query_input_df,
        grounding_df,
        mapping_df,
        top_k=harness_config.top_k,
        normalization_mode="per_spectrum_sum",
        similarity_metric=harness_config.similarity_metric,
    )
    group_means_df = group_mean_bsv(per_spectrum_df, group_col="class_label")
    comparator_map = comparator_map_for_alias(harness_config.subset_alias, group_means_df)
    delta_df = compute_delta_by_mapping(group_means_df, comparator_map)
    retrieval_summary_df = retrieval_hit_summary(retrieval_df)
    filter_summary_df = (
        grounding_df.groupby(["dataset_id", "source_key", "class_label", "compound_label"], dropna=False)
        .size()
        .reset_index(name="reference_count")
        .sort_values(["dataset_id", "class_label", "compound_label"])
        .reset_index(drop=True)
    )

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
        "filter_summary_df": filter_summary_df,
        "available_sources": sorted(grounding_df["source_key"].astype(str).unique().tolist()),
        "unavailable_sources": unavailable_sources,
        "primary_sources": sorted(primary_sources),
        "caveat_only_sources": sorted(caveat_only_sources),
    }


def write_pass2_run_artifacts(
    sprint_paths: AutoresearchSprintPaths,
    harness_config: Pass2HarnessConfig,
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
        "universal_grounding_filter_mode": harness_config.universal_grounding_filter_mode,
        "ontology_mode": harness_config.ontology_mode,
        "similarity_metric": harness_config.similarity_metric,
        "aggregation_mode": harness_config.aggregation_mode,
        "pca_grouping_mode": harness_config.pca_grouping_mode,
        "top_k": harness_config.top_k,
        "plausibility_scoring_mode": harness_config.plausibility_scoring_mode,
        "available_sources": outputs["available_sources"],
        "unavailable_sources": outputs["unavailable_sources"],
        "primary_sources": outputs["primary_sources"],
        "caveat_only_sources": outputs["caveat_only_sources"],
        "scores": scores,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
    outputs["structural_summary_df"].to_csv(run_dir / "pca_structural_summary.csv", index=False)
    outputs["filter_summary_df"].to_csv(run_dir / "grounding_filter_summary.csv", index=False)
    outputs["mapping_df"].to_csv(run_dir / "ontology_mapping_applied.csv", index=False)
    outputs["per_spectrum_df"].to_csv(run_dir / "per_spectrum_bsv.csv", index=False)
    outputs["group_means_df"].to_csv(run_dir / "group_mean_bsv.csv", index=False)
    outputs["delta_df"].to_csv(run_dir / "delta_bsv.csv", index=False)
    outputs["retrieval_summary_df"].to_csv(run_dir / "retrieval_hit_summary.csv", index=False)
    outputs["retrieval_df"].to_csv(run_dir / "topk_retrieval_hits.csv", index=False)
    return run_dir


def save_pass2_summary_tables(
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
        "universal_grounding_filter_mode",
        "top_k",
        "plausibility_scoring_mode",
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


def plot_pass2_leaderboards(results_df: pd.DataFrame, sprint_paths: AutoresearchSprintPaths) -> dict[str, Path]:
    paths = {}
    ranked = results_df.sort_values("overall_score", ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=ranked, y="run_id", x="overall_score", hue="panel_name")
    plt.title("GAIRAv3 Autoresearch v1 Pass 2 Overall Leaderboard")
    plt.xlabel("Overall score")
    plt.ylabel("")
    plt.tight_layout()
    out = sprint_paths.figures_dir / "pass2_leaderboard_overall.png"
    plt.savefig(out, dpi=220)
    plt.close()
    paths["leaderboard_overall"] = out

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=results_df.sort_values(["panel_name", "overall_score"], ascending=[True, False]),
        y="run_id",
        x="overall_score",
        hue="panel_name",
    )
    plt.title("GAIRAv3 Autoresearch v1 Pass 2 Leaderboard by Panel")
    plt.xlabel("Overall score")
    plt.ylabel("")
    plt.tight_layout()
    out = sprint_paths.figures_dir / "pass2_leaderboard_by_panel.png"
    plt.savefig(out, dpi=220)
    plt.close()
    paths["leaderboard_by_panel"] = out
    return paths


def plot_pass2_factor_ablation(results_df: pd.DataFrame, sprint_paths: AutoresearchSprintPaths) -> Path:
    rows = []
    for factor in ["universal_grounding_filter_mode", "top_k", "plausibility_scoring_mode"]:
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
    plt.figure(figsize=(10.5, max(5, 0.4 * len(heat))))
    sns.heatmap(heat, cmap="mako", annot=True, fmt=".2f")
    plt.title("Pass 2 Chemistry-Focus Factor Ablation")
    plt.tight_layout()
    out = sprint_paths.figures_dir / "pass2_factor_ablation_heatmap.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_pass2_best_vs_baseline(
    results_df: pd.DataFrame,
    sprint_paths: AutoresearchSprintPaths,
    panel_name: str,
    filename: str,
) -> Path:
    panel = results_df[results_df["panel_name"] == panel_name].copy().sort_values("overall_score", ascending=False)
    best = panel.iloc[0]
    baseline = panel[
        (panel["universal_grounding_filter_mode"] == "all_universal")
        & (panel["top_k"] == 8)
        & (panel["plausibility_scoring_mode"] == "baseline_plausibility")
        & (panel["similarity_metric"] == "cosine")
    ].iloc[0]
    comp = pd.DataFrame(
        [
            {
                "config_family": "baseline",
                "overall_score": float(baseline["overall_score"]),
                "expected_axis_uplift_score": float(baseline["expected_axis_uplift_score"]),
                "top_hit_plausibility_score": float(baseline["top_hit_plausibility_score"]),
                "matrix_collapse_penalty": float(baseline["matrix_collapse_penalty"]),
            },
            {
                "config_family": "best",
                "overall_score": float(best["overall_score"]),
                "expected_axis_uplift_score": float(best["expected_axis_uplift_score"]),
                "top_hit_plausibility_score": float(best["top_hit_plausibility_score"]),
                "matrix_collapse_penalty": float(best["matrix_collapse_penalty"]),
            },
        ]
    )
    long = comp.melt(id_vars=["config_family"], var_name="metric_name", value_name="metric_value")
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=long, x="metric_name", y="metric_value", hue="config_family")
    plt.xticks(rotation=20, ha="right")
    plt.title(f"Pass 2 Best vs Baseline: {panel_name}")
    plt.tight_layout()
    out = sprint_paths.figures_dir / filename
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def build_pass2_markdown_report(
    sprint_paths: AutoresearchSprintPaths,
    results_df: pd.DataFrame,
    search_df: pd.DataFrame,
    best_by_panel_df: pd.DataFrame,
) -> Path:
    total_runs = int((search_df["status"] == "valid").sum())
    lines = [
        "# GAIRA Autoresearch v1 Pass 2 Report",
        "",
        "## Fixed Assumptions",
        "- Canonical preprocessing remained fixed.",
        "- PCA remained always-on local structure.",
        "- Absolute BSV remained the chemistry layer.",
        "- Differential BSV remained delta-BSV over absolute BSV summaries.",
        "- Grounding mode was fixed to `universal_only`.",
        "- Aggregation mode was fixed to `class_mean_spectrum_then_bsv`.",
        "- PCA grouping semantics were fixed to `class_label_groups`.",
        "- Similarity was held to `cosine` because pass 1 showed no meaningful separation between cosine and correlation.",
        "",
        "## Search Space",
        f"- valid configurations executed: `{total_runs}`",
        "- varied factors: `universal_grounding_filter_mode`, `top_k`, `plausibility_scoring_mode`",
        "- fixed factors: `grounding_mode=universal_only`, `aggregation_mode=class_mean_spectrum_then_bsv`, `pca_grouping_mode=class_label_groups`, `ontology_mode=tier1_plus_subclass`, `similarity_metric=cosine`",
        "",
        "## Best Config by Panel",
    ]
    for _, row in best_by_panel_df.iterrows():
        lines.append(
            f"- `{row['panel_name']}`: filter=`{row['universal_grounding_filter_mode']}`, top_k=`{int(row['top_k'])}`, scoring=`{row['plausibility_scoring_mode']}`, overall=`{row['overall_score']:.4f}`, uplift=`{row['expected_axis_uplift_score']:.4f}`, plausibility=`{row['top_hit_plausibility_score']:.4f}`"
        )
    lines.extend(["", "## Factor Readout"])
    for factor in ["universal_grounding_filter_mode", "top_k", "plausibility_scoring_mode"]:
        grouped = results_df.groupby(factor)["overall_score"].mean().sort_values(ascending=False)
        lines.append(f"- `{factor}` mean overall score:")
        for level, value in grouped.items():
            lines.append(f"  - `{level}`: `{value:.4f}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Pass 2 was a chemistry-layer calibration sweep inside the winning deterministic regime from pass 1.",
            "- The key question was whether filtering the universal pool and tightening chemistry-aware plausibility scoring could improve expected-axis uplift without simply favoring trivial score hacks.",
            "- Any improvement here should be interpreted as chemistry discrimination progress, not as an architecture change.",
            "",
            "## Recommendation",
            "- If one chemistry-focused family is clearly better on both panels, keep deterministic calibration going for one more narrow pass only if the uplift signal is still moving.",
            "- If uplift remains flat while plausibility gains saturate, treat the deterministic regime as plateauing and consider escalation to a dataset-conditioned target+grounding local alignment encoder.",
        ]
    )
    out = sprint_paths.report_dir / "GAIRA_autoresearch_v1_pass2_report.md"
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
