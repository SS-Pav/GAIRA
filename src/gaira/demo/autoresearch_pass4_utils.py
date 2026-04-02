from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns

from gaira.autoresearch_storage import AutoresearchSprintPaths
from gaira.demo.autoresearch_pass2_utils import (
    OFFTARGET_BACKGROUND_TERMS,
    _expected_terms_for_panel,
)
from gaira.demo.gaira_pilot_utils import pairwise_delta_bsv
from gaira.demo.gaira_experiment_runner_utils import retrieval_hit_summary
from gaira.demo.raw_bsv_pilot_utils import ALL_AXES, PRIMARY_AXES, CAVEAT_AXES


READOUT_MODE_DEFINITIONS = {
    "current_baseline_readout": "Current locked readout with no downstream transformation.",
    "softened_support_readout": "Square-root soften axis support mass before per-row renormalization.",
    "axis_rescaled_readout": "Panel-wise axis rescaling by inverse square-root mean support before renormalization.",
}


@dataclass(frozen=True)
class Pass4HarnessConfig:
    experiment_id: str
    subset_alias: str
    panel_name: str
    readout_mode: str
    grounding_mode: str = "universal_only"
    universal_grounding_filter_mode: str = "purine_focused_universal"
    aggregation_mode: str = "class_mean_spectrum_then_bsv"
    pca_grouping_mode: str = "class_label_groups"
    ontology_mode: str = "tier1_plus_subclass"
    similarity_metric: str = "cosine"
    plausibility_scoring_mode: str = "baseline_plausibility"
    top_k: int = 5

    @property
    def run_id(self) -> str:
        return "__".join([self.panel_name, self.readout_mode])


def build_pass4_search_space() -> pd.DataFrame:
    panels = [
        ("exp_diff_cspp_metabolite_spike", "cspp_metabolite_spike_validation"),
        ("exp_localdiff_serum_uricase", "serum_ag_uricase_validation"),
        ("exp_abs_small2023_cellline_profiles", "small2023_cellline"),
    ]
    rows = []
    for experiment_id, subset_alias in panels:
        for readout_mode in READOUT_MODE_DEFINITIONS:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "subset_alias": subset_alias,
                    "panel_name": subset_alias,
                    "readout_mode": readout_mode,
                    "grounding_mode": "universal_only",
                    "universal_grounding_filter_mode": "purine_focused_universal",
                    "aggregation_mode": "class_mean_spectrum_then_bsv",
                    "pca_grouping_mode": "class_label_groups",
                    "ontology_mode": "tier1_plus_subclass",
                    "similarity_metric": "cosine",
                    "plausibility_scoring_mode": "baseline_plausibility",
                    "top_k": 5,
                    "status": "valid",
                    "exclusion_reason": "",
                }
            )
    return pd.DataFrame(rows)


def _row_normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    work = df.copy()
    values = work[cols].to_numpy(dtype=float)
    denom = np.maximum(values.sum(axis=1, keepdims=True), 1e-8)
    work.loc[:, cols] = values / denom
    return work


def apply_readout_mode(
    class_mean_bsv_df: pd.DataFrame,
    *,
    readout_mode: str,
) -> pd.DataFrame:
    cols = [axis for axis in ALL_AXES if axis in class_mean_bsv_df.columns]
    if "unmapped_support" in class_mean_bsv_df.columns:
        cols = cols + ["unmapped_support"]
    work = class_mean_bsv_df.copy()
    if readout_mode == "current_baseline_readout":
        return work
    if readout_mode == "softened_support_readout":
        values = np.sqrt(np.maximum(work[cols].to_numpy(dtype=float), 0.0))
        work.loc[:, cols] = values
        return _row_normalize(work, cols)
    if readout_mode == "axis_rescaled_readout":
        axis_means = work[cols].mean(axis=0).to_numpy(dtype=float)
        scale = 1.0 / np.sqrt(np.maximum(axis_means, 1e-8))
        values = work[cols].to_numpy(dtype=float) * scale[None, :]
        work.loc[:, cols] = values
        return _row_normalize(work, cols)
    raise ValueError(f"Unsupported readout_mode: {readout_mode}")


def _expected_fraction(sub: pd.DataFrame, terms: list[str]) -> float:
    total = float(sub["total_support_weight"].sum())
    if total <= 0:
        return 0.0
    mask = sub["reference_compound_label"].astype(str).apply(
        lambda x: any(term in x.lower() for term in terms)
    )
    return float(sub.loc[mask, "total_support_weight"].sum()) / total


def _offtarget_fraction(sub: pd.DataFrame) -> float:
    total = float(sub["total_support_weight"].sum())
    if total <= 0:
        return 0.0
    mask = sub["reference_compound_label"].astype(str).apply(
        lambda x: any(term.lower() in x.lower() for term in OFFTARGET_BACKGROUND_TERMS)
    )
    return float(sub.loc[mask, "total_support_weight"].sum()) / total


def top_hit_plausibility_score(panel_name: str, retrieval_summary_df: pd.DataFrame) -> float:
    expected_map = _expected_terms_for_panel(panel_name)
    if not expected_map:
        return 0.0
    values = []
    for query_label, terms in expected_map.items():
        sub = retrieval_summary_df[retrieval_summary_df["query_class_label"].astype(str) == query_label].copy()
        if sub.empty:
            values.append(0.0)
            continue
        values.append(_expected_fraction(sub, terms) - _offtarget_fraction(sub))
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


def _primary_entropy(values: np.ndarray) -> float:
    values = np.maximum(values.astype(float), 0.0)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    p = values / total
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(len(values)))


def compute_axis_visibility_metrics(class_mean_bsv_df: pd.DataFrame) -> dict[str, float]:
    primary_axes = [axis for axis in PRIMARY_AXES if axis in class_mean_bsv_df.columns]
    primary = class_mean_bsv_df[primary_axes].to_numpy(dtype=float)
    entropies = np.array([_primary_entropy(row) for row in primary], dtype=float)
    secondary_mass = np.array([float(max(row.sum() - row.max(), 0.0)) for row in primary], dtype=float)
    return {
        "mean_primary_entropy": float(entropies.mean()) if len(entropies) else 0.0,
        "mean_secondary_axis_mass": float(secondary_mass.mean()) if len(secondary_mass) else 0.0,
    }


def score_pass4_outputs(
    panel_name: str,
    class_mean_bsv_df: pd.DataFrame,
    retrieval_summary_df: pd.DataFrame,
    delta_df: pd.DataFrame | None,
    inter_class_distance_df: pd.DataFrame | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    visibility = compute_axis_visibility_metrics(class_mean_bsv_df)
    metrics.update(visibility)
    metrics["top_hit_plausibility_score"] = top_hit_plausibility_score(panel_name, retrieval_summary_df)
    caveat_cols = [axis for axis in CAVEAT_AXES if axis in class_mean_bsv_df.columns]
    metrics["caveat_domination_penalty"] = float(class_mean_bsv_df[caveat_cols].sum(axis=1).mean()) if caveat_cols else 0.0
    metrics["matrix_collapse_penalty"] = (
        float(class_mean_bsv_df["matrix_background"].abs().mean()) if "matrix_background" in class_mean_bsv_df.columns else 0.0
    )
    metrics["mean_inter_class_distance"] = (
        float(inter_class_distance_df["euclidean_distance"].mean()) if inter_class_distance_df is not None else 0.0
    )

    if panel_name == "cspp_metabolite_spike_validation" and delta_df is not None:
        hyp_uplift = max(_primary_axis_score(delta_df, "Hyp", ["purine_like_metabolite", "small_molecule_metabolite"]), 0.0)
        erg_uplift = max(_primary_axis_score(delta_df, "Erg", ["sulfur_containing_metabolite", "small_molecule_metabolite"]), 0.0)
        metrics["expected_axis_uplift_score"] = (hyp_uplift + erg_uplift) / 2.0
        metrics["overall_score"] = (
            2.8 * metrics["expected_axis_uplift_score"]
            + 1.8 * metrics["top_hit_plausibility_score"]
            + 0.5
            - metrics["caveat_domination_penalty"]
            - 1.5 * metrics["matrix_collapse_penalty"]
        )
    elif panel_name == "serum_ag_uricase_validation" and delta_df is not None:
        spiked = _primary_axis_score(delta_df, "Serumspiked+Enzyme", ["purine_like_metabolite", "small_molecule_metabolite"])
        sigma = _primary_axis_score(delta_df, "SerumSigma+Enzyme", ["purine_like_metabolite", "small_molecule_metabolite"])
        metrics["expected_axis_uplift_score"] = max(spiked - sigma, 0.0)
        metrics["overall_score"] = (
            2.8 * metrics["expected_axis_uplift_score"]
            + 1.8 * metrics["top_hit_plausibility_score"]
            + 0.5
            - metrics["caveat_domination_penalty"]
            - 1.5 * metrics["matrix_collapse_penalty"]
        )
    else:
        metrics["expected_axis_uplift_score"] = 0.0
        metrics["overall_score"] = (
            1.6 * metrics["mean_primary_entropy"]
            + 1.2 * metrics["mean_secondary_axis_mass"]
            + 0.8 * metrics["mean_inter_class_distance"]
            - 0.8 * metrics["caveat_domination_penalty"]
            - 1.2 * metrics["matrix_collapse_penalty"]
        )
    return metrics


def write_pass4_run_artifacts(
    sprint_paths: AutoresearchSprintPaths,
    harness_config: Pass4HarnessConfig,
    *,
    class_mean_bsv_df: pd.DataFrame,
    retrieval_summary_df: pd.DataFrame,
    delta_df: pd.DataFrame | None,
    inter_class_distance_df: pd.DataFrame | None,
    intra_class_variance_df: pd.DataFrame | None,
    scores: dict[str, float],
) -> Path:
    run_dir = sprint_paths.runs_dir / harness_config.panel_name / harness_config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    class_mean_bsv_df.to_csv(run_dir / "class_mean_bsv.csv", index=False)
    retrieval_summary_df.to_csv(run_dir / "retrieval_hit_summary_by_class.csv", index=False)
    if delta_df is not None:
        delta_df.to_csv(run_dir / "pairwise_delta_bsv.csv", index=False)
    if inter_class_distance_df is not None:
        inter_class_distance_df.to_csv(run_dir / "inter_class_bsv_distance.csv", index=False)
    if intra_class_variance_df is not None:
        intra_class_variance_df.to_csv(run_dir / "intra_class_bsv_variance.csv", index=False)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "panel_name": harness_config.panel_name,
                "readout_mode": harness_config.readout_mode,
                "fixed_baseline": {
                    "grounding_mode": harness_config.grounding_mode,
                    "universal_grounding_filter_mode": harness_config.universal_grounding_filter_mode,
                    "aggregation_mode": harness_config.aggregation_mode,
                    "pca_grouping_mode": harness_config.pca_grouping_mode,
                    "ontology_mode": harness_config.ontology_mode,
                    "similarity_metric": harness_config.similarity_metric,
                    "plausibility_scoring_mode": harness_config.plausibility_scoring_mode,
                    "top_k": harness_config.top_k,
                },
                "scores": scores,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def save_pass4_summary_tables(
    sprint_paths: AutoresearchSprintPaths,
    search_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    search_df.to_csv(sprint_paths.tables_dir / "readout_search_space_used.csv", index=False)
    ranked = results_df.sort_values(["panel_name", "overall_score"], ascending=[True, False]).copy()
    ranked.to_csv(sprint_paths.tables_dir / "readout_results_ranked.csv", index=False)
    metric_cols = [
        "panel_name",
        "readout_mode",
        "expected_axis_uplift_score",
        "top_hit_plausibility_score",
        "mean_primary_entropy",
        "mean_secondary_axis_mass",
        "mean_inter_class_distance",
        "caveat_domination_penalty",
        "matrix_collapse_penalty",
        "overall_score",
    ]
    ranked[metric_cols].to_csv(sprint_paths.tables_dir / "readout_metric_breakdown.csv", index=False)
    best = ranked.groupby("panel_name", as_index=False).first()
    best.to_csv(sprint_paths.tables_dir / "best_readout_by_panel.csv", index=False)
    baseline = ranked[ranked["readout_mode"] == "current_baseline_readout"][
        ["panel_name", "overall_score", "expected_axis_uplift_score", "top_hit_plausibility_score", "mean_primary_entropy", "mean_secondary_axis_mass"]
    ]
    compare = baseline.merge(
        best[
            ["panel_name", "readout_mode", "overall_score", "expected_axis_uplift_score", "top_hit_plausibility_score", "mean_primary_entropy", "mean_secondary_axis_mass"]
        ],
        on="panel_name",
        suffixes=("_baseline", "_best"),
    )
    compare.to_csv(sprint_paths.tables_dir / "pass4_vs_baseline_comparison.csv", index=False)
    return {"ranked": ranked, "best": best, "compare": compare}


def plot_pass4_leaderboard(results_df: pd.DataFrame, sprint_paths: AutoresearchSprintPaths) -> Path:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df.sort_values("overall_score", ascending=False), y="panel_name", x="overall_score", hue="readout_mode")
    plt.title("GAIRAv3 Pass 4 Readout Diagnostic Leaderboard")
    plt.tight_layout()
    out = sprint_paths.figures_dir / "pass4_leaderboard_overall.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_pass4_best_vs_baseline(compare_df: pd.DataFrame, sprint_paths: AutoresearchSprintPaths, panel_name: str, filename: str) -> Path:
    row = compare_df[compare_df["panel_name"] == panel_name].iloc[0]
    comp = pd.DataFrame(
        [
            {
                "config_family": "baseline",
                "overall_score": float(row["overall_score_baseline"]),
                "expected_axis_uplift_score": float(row["expected_axis_uplift_score_baseline"]),
                "top_hit_plausibility_score": float(row["top_hit_plausibility_score_baseline"]),
                "mean_primary_entropy": float(row["mean_primary_entropy_baseline"]),
                "mean_secondary_axis_mass": float(row["mean_secondary_axis_mass_baseline"]),
            },
            {
                "config_family": "best",
                "overall_score": float(row["overall_score_best"]),
                "expected_axis_uplift_score": float(row["expected_axis_uplift_score_best"]),
                "top_hit_plausibility_score": float(row["top_hit_plausibility_score_best"]),
                "mean_primary_entropy": float(row["mean_primary_entropy_best"]),
                "mean_secondary_axis_mass": float(row["mean_secondary_axis_mass_best"]),
            },
        ]
    )
    long = comp.melt(id_vars=["config_family"], var_name="metric_name", value_name="metric_value")
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=long, x="metric_name", y="metric_value", hue="config_family")
    plt.xticks(rotation=20, ha="right")
    plt.title(f"Pass 4 Best vs Baseline: {panel_name}")
    plt.tight_layout()
    out = sprint_paths.figures_dir / filename
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_axis_visibility_comparison(results_df: pd.DataFrame, sprint_paths: AutoresearchSprintPaths) -> Path:
    small = results_df[results_df["panel_name"] == "small2023_cellline"].copy()
    long = small.melt(
        id_vars=["readout_mode"],
        value_vars=["mean_primary_entropy", "mean_secondary_axis_mass"],
        var_name="metric_name",
        value_name="metric_value",
    )
    plt.figure(figsize=(8.5, 5.5))
    sns.barplot(data=long, x="metric_name", y="metric_value", hue="readout_mode")
    plt.title("Pass 4 small2023 Axis Visibility Comparison")
    plt.tight_layout()
    out = sprint_paths.figures_dir / "pass4_axis_visibility_comparison.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def build_pass4_markdown_report(
    sprint_paths: AutoresearchSprintPaths,
    results_df: pd.DataFrame,
    search_df: pd.DataFrame,
    best_df: pd.DataFrame,
    compare_df: pd.DataFrame,
) -> Path:
    lines = [
        "# GAIRA Autoresearch v1 Pass 4 Report",
        "",
        "## Fixed Baseline",
        "- canonical preprocessing fixed",
        "- universal_only grounding with purine_focused_universal filter",
        "- class_mean_spectrum_then_bsv aggregation",
        "- cosine similarity",
        "- tier1_plus_subclass ontology",
        "- class_label_groups PCA semantics",
        "- top_k fixed at 5",
        "",
        "## Readout Modes Tested",
        *[f"- `{name}`: {desc}" for name, desc in READOUT_MODE_DEFINITIONS.items()],
        "",
        f"## Search Space\n- readout modes executed: `{int((search_df['status'] == 'valid').sum())}` total runs across 3 panels",
        "",
        "## Best Readout By Panel",
    ]
    for _, row in best_df.iterrows():
        lines.append(
            f"- `{row['panel_name']}`: best=`{row['readout_mode']}`, overall=`{row['overall_score']:.4f}`, uplift=`{row['expected_axis_uplift_score']:.4f}`, entropy=`{row['mean_primary_entropy']:.4f}`, secondary_mass=`{row['mean_secondary_axis_mass']:.4f}`"
        )
    lines.extend(["", "## Baseline Comparison"])
    for _, row in compare_df.iterrows():
        lines.append(
            f"- `{row['panel_name']}`: baseline overall `{row['overall_score_baseline']:.4f}` -> best `{row['overall_score_best']:.4f}`; uplift `{row['expected_axis_uplift_score_baseline']:.4f}` -> `{row['expected_axis_uplift_score_best']:.4f}`; entropy `{row['mean_primary_entropy_baseline']:.4f}` -> `{row['mean_primary_entropy_best']:.4f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Pass 4 tests only downstream readout/normalization style. Upstream retrieval, grounding, ontology, similarity, and top_k remain fixed.",
            "- The key question is whether broader chemistry becomes visible on small2023 without damaging validation behavior.",
            "",
            "## Decision Frame",
            "- If a readout mode improves small2023 broadness while keeping CSPP and uricase intact, adopt it and move to Pilot 2.",
            "- If validation behavior degrades or gains are negligible, keep the current readout and move on.",
        ]
    )
    out = sprint_paths.report_dir / "GAIRA_autoresearch_v1_pass4_report.md"
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
            ax = fig.add_axes([0.03, 0.08, 0.94, 0.86])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(path.name, fontsize=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
