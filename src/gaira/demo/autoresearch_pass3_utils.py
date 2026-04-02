from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns

from gaira.autoresearch_storage import AutoresearchSprintPaths
from gaira.demo.autoresearch_pass2_utils import (
    OFFTARGET_BACKGROUND_TERMS,
    PURINE_NEIGHBOR_TERMS,
    SULFUR_NEIGHBOR_TERMS,
    build_pdf_report as build_pdf_report_common,
    compute_pass2_outputs,
    score_pass2_outputs,
    write_pass2_run_artifacts,
)


@dataclass(frozen=True)
class Pass3HarnessConfig:
    experiment_id: str
    subset_alias: str
    panel_name: str
    universal_grounding_filter_mode: str
    top_k: int
    plausibility_scoring_mode: str = "baseline_plausibility"
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


PASS3_FILTER_MODE_DEFINITIONS = {
    "purine_sulfur_neighbor_focused_universal": "Pass 2 winning universal subset: explicit purine-neighbor plus sulfur-neighbor references only.",
    "purine_focused_universal": "Explicit universal purine-neighbor references only.",
    "sulfur_neighbor_focused_universal": "Explicit universal sulfur-neighbor references only.",
}


def build_pass3_search_space() -> pd.DataFrame:
    panels = [
        ("exp_diff_cspp_metabolite_spike", "cspp_metabolite_spike_validation"),
        ("exp_localdiff_serum_uricase", "serum_ag_uricase_validation"),
    ]
    rows = []
    for experiment_id, subset_alias in panels:
        for filter_mode in PASS3_FILTER_MODE_DEFINITIONS:
            for top_k in [3, 4, 5]:
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
                        "plausibility_scoring_mode": "baseline_plausibility",
                        "status": "valid",
                        "exclusion_reason": "",
                    }
                )
    return pd.DataFrame(rows)


def apply_pass3_filter_mode(grounding_df: pd.DataFrame, filter_mode: str) -> pd.DataFrame:
    work = grounding_df[grounding_df["source_key"].astype(str).isin({"adenine_sers_control", "metabolite_sers63_support", "ramanbiolib"})].copy()
    label_text = (
        work["class_label"].fillna("").astype(str)
        + " "
        + work["compound_label"].fillna("").astype(str)
        + " "
        + work["source_key"].fillna("").astype(str)
    ).str.lower()
    if filter_mode == "purine_sulfur_neighbor_focused_universal":
        pattern = "|".join(PURINE_NEIGHBOR_TERMS + SULFUR_NEIGHBOR_TERMS)
        filtered = work[label_text.str.contains(pattern, regex=True)].copy()
    elif filter_mode == "purine_focused_universal":
        pattern = "|".join(PURINE_NEIGHBOR_TERMS)
        filtered = work[label_text.str.contains(pattern, regex=True)].copy()
    elif filter_mode == "sulfur_neighbor_focused_universal":
        pattern = "|".join(SULFUR_NEIGHBOR_TERMS)
        filtered = work[label_text.str.contains(pattern, regex=True)].copy()
    else:
        raise ValueError(f"Unsupported pass 3 filter mode: {filter_mode}")
    if filtered.empty:
        raise RuntimeError(f"Pass 3 filter mode {filter_mode} produced an empty grounding pool.")
    return filtered.reset_index(drop=True)


def save_pass3_summary_tables(
    sprint_paths: AutoresearchSprintPaths,
    search_df: pd.DataFrame,
    results_df: pd.DataFrame,
    pass2_best_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    search_df.to_csv(sprint_paths.tables_dir / "calibration_search_space_used.csv", index=False)
    ranked = results_df.sort_values(["panel_name", "overall_score"], ascending=[True, False]).copy()
    ranked.to_csv(sprint_paths.tables_dir / "calibration_results_ranked.csv", index=False)
    best = ranked.groupby("panel_name", as_index=False).first()
    best.to_csv(sprint_paths.tables_dir / "best_config_by_panel.csv", index=False)
    compare = pass2_best_df[
        ["panel_name", "run_id", "overall_score", "expected_axis_uplift_score", "top_hit_plausibility_score"]
    ].merge(
        best[
            ["panel_name", "run_id", "overall_score", "expected_axis_uplift_score", "top_hit_plausibility_score"]
        ],
        on="panel_name",
        suffixes=("_pass2", "_pass3"),
    )
    compare.to_csv(sprint_paths.tables_dir / "pass3_vs_pass2_comparison.csv", index=False)
    return {"ranked": ranked, "best": best, "compare": compare}


def plot_pass3_leaderboard(results_df: pd.DataFrame, sprint_paths: AutoresearchSprintPaths) -> Path:
    ranked = results_df.sort_values("overall_score", ascending=False).head(18)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=ranked, y="run_id", x="overall_score", hue="panel_name")
    plt.title("GAIRAv3 Autoresearch v1 Pass 3 Overall Leaderboard")
    plt.xlabel("Overall score")
    plt.ylabel("")
    plt.tight_layout()
    out = sprint_paths.figures_dir / "pass3_leaderboard_overall.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_pass3_best_vs_pass2(
    pass3_best_df: pd.DataFrame,
    pass2_best_df: pd.DataFrame,
    sprint_paths: AutoresearchSprintPaths,
    panel_name: str,
    filename: str,
) -> Path:
    old = pass2_best_df[pass2_best_df["panel_name"] == panel_name].iloc[0]
    new = pass3_best_df[pass3_best_df["panel_name"] == panel_name].iloc[0]
    comp = pd.DataFrame(
        [
            {
                "config_family": "pass2_best",
                "overall_score": float(old["overall_score"]),
                "expected_axis_uplift_score": float(old["expected_axis_uplift_score"]),
                "top_hit_plausibility_score": float(old["top_hit_plausibility_score"]),
            },
            {
                "config_family": "pass3_best",
                "overall_score": float(new["overall_score"]),
                "expected_axis_uplift_score": float(new["expected_axis_uplift_score"]),
                "top_hit_plausibility_score": float(new["top_hit_plausibility_score"]),
            },
        ]
    )
    long = comp.melt(id_vars=["config_family"], var_name="metric_name", value_name="metric_value")
    plt.figure(figsize=(8.5, 5.2))
    sns.barplot(data=long, x="metric_name", y="metric_value", hue="config_family")
    plt.xticks(rotation=20, ha="right")
    plt.title(f"Pass 3 Best vs Pass 2 Best: {panel_name}")
    plt.tight_layout()
    out = sprint_paths.figures_dir / filename
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def build_pass3_markdown_report(
    sprint_paths: AutoresearchSprintPaths,
    results_df: pd.DataFrame,
    search_df: pd.DataFrame,
    best_by_panel_df: pd.DataFrame,
    compare_df: pd.DataFrame,
) -> Path:
    total_runs = int((search_df["status"] == "valid").sum())
    lines = [
        "# GAIRA Autoresearch v1 Pass 3 Report",
        "",
        "## Fixed Assumptions",
        "- Canonical preprocessing remained fixed.",
        "- PCA remained always-on local structure.",
        "- Grounding mode remained `universal_only`.",
        "- Aggregation remained `class_mean_spectrum_then_bsv`.",
        "- PCA grouping remained `class_label_groups`.",
        "- Ontology remained `tier1_plus_subclass`.",
        "- Similarity remained `cosine`.",
        "- Plausibility scoring remained `baseline_plausibility`.",
        "",
        "## Search Space",
        f"- valid configurations executed: `{total_runs}`",
        "- varied factors: `universal_grounding_filter_mode`, `top_k`",
        "",
        "## Best Config by Panel",
    ]
    for _, row in best_by_panel_df.iterrows():
        lines.append(
            f"- `{row['panel_name']}`: filter=`{row['universal_grounding_filter_mode']}`, top_k=`{int(row['top_k'])}`, overall=`{row['overall_score']:.4f}`, uplift=`{row['expected_axis_uplift_score']:.4f}`, plausibility=`{row['top_hit_plausibility_score']:.4f}`"
        )
    lines.extend(["", "## Pass 3 vs Pass 2"])
    for _, row in compare_df.iterrows():
        lines.append(
            f"- `{row['panel_name']}`: pass2 overall=`{row['overall_score_pass2']:.4f}` -> pass3 overall=`{row['overall_score_pass3']:.4f}`; uplift `{row['expected_axis_uplift_score_pass2']:.4f}` -> `{row['expected_axis_uplift_score_pass3']:.4f}`; plausibility `{row['top_hit_plausibility_score_pass2']:.4f}` -> `{row['top_hit_plausibility_score_pass3']:.4f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Pass 3 was designed to answer a single question: whether a final narrow refinement could preserve or improve CSPP without degrading uricase.",
            "- If the same filter family still wins but cross-panel tradeoffs remain, the deterministic regime should be treated as plateaued rather than endlessly retuned.",
            "",
            "## Recommendation",
            "- If pass 3 improves or preserves both panels, the deterministic baseline is stable enough to lock.",
            "- If pass 3 sharpens one panel while still hurting the other, deterministic calibration has reached its useful limit and escalation is justified.",
        ]
    )
    out = sprint_paths.report_dir / "GAIRA_autoresearch_v1_pass3_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_pdf_report(markdown_path: Path, figure_paths: list[Path], output_path: Path) -> None:
    build_pdf_report_common(markdown_path, figure_paths, output_path)
