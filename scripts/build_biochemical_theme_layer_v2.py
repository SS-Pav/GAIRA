from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA")
V1_DIR = ROOT / "processed" / "biochemical_theme_layer_v1"
OUTPUT_DIR = ROOT / "processed" / "biochemical_theme_layer_v2"
RAW_DIR = OUTPUT_DIR / "raw_outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report"

SERUM_PROTOCOL_VERSION = "v1_crop400_1800_interp1_minmax"
COVID_VERSION = "v1_crop400_1800_interp1_minmax"
ADENINE_VERSION = "v1_crop400_1800_interp1_vector"
SMALL2023_VERSION = "v1_crop670_1800_interp1_minmax"
EV_CLASS_ORDER = ["c00", "c01", "c10", "c25", "c50", "c100"]


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 400,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
sns.set_theme(style="whitegrid", context="talk")


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, RAW_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def load_v1_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(V1_DIR / "raw_outputs" / "theme_per_query_outputs_long.csv"),
        pd.read_csv(V1_DIR / "tables" / "theme_track_metrics.csv"),
    )


def build_architecture_figure() -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis("off")
    nodes = [
        (0.03, 0.25, 0.17, 0.5, "#d8ecf3", "Evidence stack", ["Tier 1", "Tier 2", "Knowledge", "Semantic", "Context"]),
        (0.25, 0.25, 0.17, 0.5, "#f4efe1", "Theme v1 issue", ["Additive overlap", "weak competition", "saturation risk", "control leakage"]),
        (0.47, 0.25, 0.20, 0.5, "#e8f1df", "Theme v2 logic", ["shared-hit allocation", "competition normalization", "negative evidence", "calibration downweighting"]),
        (0.72, 0.25, 0.20, 0.5, "#f7e4e2", "Outputs", ["sharper themes", "tighter confidence", "explicit penalties", "same provenance"]),
    ]
    for x, y, w, h, color, title, lines in nodes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#4d5a6a", linewidth=1.3)
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.08, title, fontsize=16, fontweight="bold", ha="left")
        line_y = y + h - 0.17
        for line in lines:
            ax.text(x + 0.025, line_y, f"- {line}", fontsize=12, ha="left")
            line_y -= 0.09
    for start, end in [(0.20, 0.25), (0.42, 0.47), (0.67, 0.72)]:
        ax.annotate("", xy=(end, 0.5), xytext=(start, 0.5), arrowprops=dict(arrowstyle="-|>", lw=2, color="#4d5a6a"))
    ax.set_title("Figure 1. Biochemical Theme Layer v2 architecture and changes from v1", fontsize=20, pad=12)
    return save_figure(fig, "figure1_theme_layer_v2_architecture")


def build_example_theme_figure(v2_df: pd.DataFrame, v1_df: pd.DataFrame) -> tuple[Path, Path]:
    example_ids = [
        "small2023_ev_c50_normedprobe1",
        "covid_serum_raman_covid_confirmed_covid19_serum_raman_archive",
    ]
    v2_plot = v2_df[v2_df["query_id"].isin(example_ids)].copy()
    v2_plot["version"] = "v2"
    v1_plot = v1_df[v1_df["query_id"].isin(example_ids)].copy()
    v1_plot["version"] = "v1"
    plot_df = pd.concat([v1_plot, v2_plot], ignore_index=True)
    plot_df["theme_short"] = plot_df["theme_name"].str.replace("_associated", "", regex=False).str.replace("_caution", "", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for ax, query_id in zip(axes, example_ids):
        subset = plot_df[plot_df["query_id"] == query_id].copy()
        subset = subset.sort_values(["version", "score"], ascending=[True, False])
        top_themes = subset.groupby("theme_short")["score"].max().sort_values(ascending=False).head(8).index.tolist()
        subset = subset[subset["theme_short"].isin(top_themes)]
        sns.barplot(data=subset, x="score", y="theme_short", hue="version", palette={"v1": "#9bbad0", "v2": "#355c7d"}, ax=ax)
        ax.set_title(query_id)
        ax.set_xlabel("Theme score")
        ax.set_ylabel("")
        ax.legend(frameon=False, title="")
    fig.suptitle("Figure 2. Representative v2 theme score examples against v1", fontsize=20, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "figure2_theme_score_examples_v2")


def build_evidence_contribution_figure(v2_df: pd.DataFrame) -> tuple[Path, Path]:
    contrib_cols = ["tier1_contrib", "tier2_contrib", "knowledge_contrib", "semantic_contrib", "context_contrib", "band_contrib"]
    plot_df = (
        v2_df.groupby("theme_name", as_index=False)[contrib_cols + ["competition_penalty", "caution_penalty", "calibration_penalty"]]
        .mean()
        .melt(id_vars="theme_name", var_name="component", value_name="value")
    )
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(data=plot_df, x="theme_name", y="value", hue="component", palette="crest", ax=ax)
    ax.set_title("Figure 3. Evidence and penalty composition in v2")
    ax.set_xlabel("")
    ax.set_ylabel("Mean contribution / penalty")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return save_figure(fig, "figure3_evidence_type_contributions_v2")


def build_adenine_specificity_figure(v2_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = v2_df[
        (v2_df["track_name"] == "adenine_controlled_specificity")
        & (v2_df["theme_name"].isin(["nucleic_acid_purine_associated", "protein_peptide_associated", "lipid_membrane_associated", "low_specificity_caution"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.pointplot(data=plot_df, x="query_label", y="score", hue="theme_name", dodge=0.35, markers="o", linestyles="-", ax=ax)
    ax.set_title("Figure 4. v2 controlled analyte specificity for adenine")
    ax.set_xlabel("Adenine condition")
    ax.set_ylabel("Theme score")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return save_figure(fig, "figure4_adenine_specificity_v2")


def build_serum_protocol_figure(v2_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = v2_df[
        (v2_df["track_name"] == "serum_protocol_robustness")
        & (v2_df["theme_name"].isin(["nucleic_acid_purine_associated", "protein_peptide_associated", "matrix_dominance_caution", "probe_substrate_caution"]))
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    for ax, theme_name in zip(axes.ravel(), plot_df["theme_name"].unique()):
        subset = plot_df[plot_df["theme_name"] == theme_name]
        sns.boxplot(data=subset, x="query_label", y="score", color="#4c78a8", ax=ax)
        ax.set_title(theme_name)
        ax.set_xlabel("Protocol")
        ax.set_ylabel("Score")
    fig.suptitle("Figure 5. v2 serum protocol robustness", fontsize=20, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "figure5_serum_protocol_robustness_v2")


def build_ev_mixture_figure(v2_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = v2_df[
        (v2_df["track_name"] == "ev_mixture_coherence")
        & (v2_df["theme_name"].isin(["lipid_membrane_associated", "nucleic_acid_purine_associated", "protein_peptide_associated", "probe_substrate_caution"]))
    ].copy()
    class_to_value = {label: value for label, value in zip(EV_CLASS_ORDER, [0, 1, 10, 25, 50, 100])}
    plot_df["mixture_fraction"] = plot_df["query_label"].map(class_to_value)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, theme_name in zip(axes, ["lipid_membrane_associated", "nucleic_acid_purine_associated"]):
        subset = plot_df[plot_df["theme_name"] == theme_name]
        sns.lineplot(data=subset, x="mixture_fraction", y="score", hue="query_family", marker="o", palette="deep", ax=ax)
        ax.set_title(theme_name)
        ax.set_xlabel("Mixture class code")
        ax.set_ylabel("Score")
        ax.legend(frameon=False, title="")
    fig.suptitle("Figure 6. v2 EV mixture coherence", fontsize=20, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "figure6_ev_mixture_coherence_v2")


def build_covid_figure(v2_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = v2_df[
        (v2_df["track_name"] == "covid_serum_usefulness")
        & (v2_df["theme_name"].isin(["protein_peptide_associated", "oxidative_metabolic_stress_associated", "modality_mismatch_caution", "low_specificity_caution"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(data=plot_df, x="query_label", y="score", hue="theme_name", palette="mako", ax=ax)
    ax.set_title("Figure 7. v2 COVID serum usefulness and caution behavior")
    ax.set_xlabel("Cohort group")
    ax.set_ylabel("Theme score")
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return save_figure(fig, "figure7_covid_serum_usefulness_v2")


def build_stability_figure(v2_df: pd.DataFrame) -> tuple[Path, Path]:
    summary_df = (
        v2_df.groupby(["track_name", "theme_name"], as_index=False)
        .agg(mean_score=("score", "mean"), std_score=("score", "std"))
        .fillna(0.0)
    )
    heatmap_df = summary_df.pivot(index="theme_name", columns="track_name", values="std_score").fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_df, cmap="rocket_r", annot=True, fmt=".2f", cbar_kws={"label": "Std. dev."}, ax=ax)
    ax.set_title("Figure 8. v2 theme stability and consistency")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return save_figure(fig, "figure8_theme_stability_v2")


def build_global_metric_figure(metrics_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = metrics_df[metrics_df["metric_name"].isin([
        "purine_dominance_margin",
        "off_theme_suppression",
        "positive_theme_stability",
        "theme_smoothness_proxy",
        "modality_caution_mean",
        "within_condition_consistency",
        "mean_dominance_margin",
        "positive_entropy_inverse",
        "caution_adjusted_confidence",
    ])].copy()
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.barplot(data=plot_df, x="metric_name", y="metric_value", hue="track_name", palette="viridis", ax=ax)
    ax.set_title("Figure 9. v2 global performance summary")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return save_figure(fig, "figure9_global_performance_summary_v2")


def build_report_pdf(report_md_path: Path, figure_paths: list[Path]) -> Path:
    pdf_path = REPORT_DIR / "biochemical_theme_layer_v2_report.pdf"
    summary_text = report_md_path.read_text(encoding="utf-8")
    with PdfPages(pdf_path) as pdf:
        for page_index, chunk_start in enumerate(range(0, len(summary_text), 3200), start=1):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.02, 0.98, f"Biochemical Theme Layer v2 report (page {page_index})", fontsize=16, fontweight="bold", va="top")
            ax.text(0.02, 0.93, summary_text[chunk_start : chunk_start + 3200], fontsize=9, va="top", family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        for figure_path in figure_paths:
            image = plt.imread(figure_path)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return pdf_path


def write_report(v2_metrics: pd.DataFrame, v1_metrics: pd.DataFrame, figure_paths: list[Path], track_summaries: dict[str, str], overall_assessment: str) -> Path:
    metric_table = v2_metrics.to_string(index=False)
    compare = v2_metrics.merge(
        v1_metrics,
        on=["track_name", "metric_name"],
        suffixes=("_v2", "_v1"),
        how="left",
    )
    compare["delta_v2_minus_v1"] = compare["metric_value_v2"] - compare["metric_value_v1"]
    compare_table = compare.to_string(index=False)
    figure_list = "\n".join(f"- {path.name}" for path in figure_paths)
    report_text = f"""# GAIRA Biochemical Theme Layer v2

## 1. Motivation for v2
v1 was useful but over-saturated. Multiple positive themes stayed high together because overlapping evidence could be counted nearly in full more than once. v2 tightens this by adding competition, negative evidence, and calibration-aware downweighting.

## 2. Main v1 weaknesses
- overlapping evidence inflated several positive themes at once
- insufficient off-theme suppression
- adenine improved purine interpretation directionally but could still leak too strongly into biosample summaries
- confidence was not sharp enough under high caution load

## 3. v2 design changes
- shared evidence allocation across competing positive themes
- explicit negative-keyword penalties
- calibration-control downweighting for adenine in biosample interpretation
- stronger confidence penalties from caution burden and competition weakness
- explicit debug outputs: raw_score_pre_normalization, normalized_score, competition_penalty, caution_penalty, calibration_penalty, specificity_index

## 4. Evaluation tracks
- Adenine controlled analyte specificity
- Serum protocol robustness
- EV mixture coherence
- COVID serum usefulness
- Replicate consistency

## 5. v2 results summary
{track_summaries['adenine_controlled_specificity']}

{track_summaries['serum_protocol_robustness']}

{track_summaries['ev_mixture_coherence']}

{track_summaries['covid_serum_usefulness']}

{track_summaries['replicate_consistency']}

## 6. v2 metric table
{metric_table}

## 7. v1-v2 metric deltas
{compare_table}

## 8. Figures
{figure_list}

## 9. Overall assessment
{overall_assessment}

## 10. Recommendation
v2 should replace v1 only if the specificity gains do not materially damage EV coherence, serum robustness, or replicate stability. The direct comparison report gives that final decision explicitly.
"""
    report_path = REPORT_DIR / "biochemical_theme_layer_v2_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main() -> None:
    ensure_dirs()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gaira.config import get_database_path, require_data_root_exists
    from gaira.theme_evaluation import ThemeEvaluationRunner, build_track_metrics

    require_data_root_exists()
    db_path = get_database_path()
    runner = ThemeEvaluationRunner(db_path=db_path, theme_layer_version="v2")

    adenine_bundle = runner.evaluate_theme_inputs(
        "adenine_controlled_specificity",
        runner.load_grounding_class_summary_queries("adenine_sers_control", ADENINE_VERSION),
    )
    protocol_bundle = runner.evaluate_inference_requests(
        "serum_protocol_robustness",
        runner.load_biosample_processed_requests("serum_protocol_comparison", "serum", SERUM_PROTOCOL_VERSION),
    )
    ev_bundle = runner.evaluate_inference_requests(
        "ev_mixture_coherence",
        runner.load_ev_class_mean_requests("small2023_ev", ["normedprobe1", "normedprobe2"], EV_CLASS_ORDER, SMALL2023_VERSION),
    )
    covid_bundle = runner.evaluate_inference_requests(
        "covid_serum_usefulness",
        runner.load_serum_class_mean_requests(
            "covid_serum_raman",
            "covid19_serum_raman_archive",
            COVID_VERSION,
            ["covid_confirmed", "healthy_control", "suspected_case", "tube_control"],
        ),
    )
    replicate_bundle = runner.evaluate_theme_inputs(
        "replicate_consistency",
        runner.load_grounding_processed_queries("adenine_sers_control", ADENINE_VERSION, "bag_nps_replicate_series"),
    )

    bundles = [adenine_bundle, protocol_bundle, ev_bundle, covid_bundle, replicate_bundle]
    bundle_map = {
        "adenine_controlled_specificity": adenine_bundle,
        "serum_protocol_robustness": protocol_bundle,
        "ev_mixture_coherence": ev_bundle,
        "covid_serum_usefulness": covid_bundle,
        "replicate_consistency": replicate_bundle,
    }

    all_query_df = pd.concat([bundle.query_df for bundle in bundles], ignore_index=True)
    all_theme_df = pd.concat([bundle.theme_df for bundle in bundles], ignore_index=True)
    all_summary_df = pd.concat([bundle.summary_df for bundle in bundles], ignore_index=True)

    all_query_df.to_csv(RAW_DIR / "theme_query_outputs.csv", index=False)
    all_theme_df.to_csv(RAW_DIR / "theme_per_query_outputs_long.csv", index=False)
    all_summary_df.to_csv(TABLE_DIR / "theme_track_summary.csv", index=False)

    metric_frames = [build_track_metrics(track_name, bundle.theme_df) for track_name, bundle in bundle_map.items()]
    metrics_df = pd.concat([df for df in metric_frames if not df.empty], ignore_index=True)
    metrics_df.to_csv(TABLE_DIR / "theme_track_metrics.csv", index=False)

    v1_theme_df, v1_metrics_df = load_v1_frames()

    track_summaries = {
        "adenine_controlled_specificity": (
            f"Adenine specificity reached purine dominance margin "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'purine_dominance_margin', 'metric_value'].iloc[0]:.3f}, "
            f"off-theme suppression "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'off_theme_suppression', 'metric_value'].iloc[0]:.3f}, "
            f"and positive entropy inverse "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'positive_entropy_inverse', 'metric_value'].iloc[0]:.3f}."
        ),
        "serum_protocol_robustness": (
            f"Serum protocol robustness kept positive-theme stability "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'positive_theme_stability', 'metric_value'].iloc[0]:.3f} "
            f"with caution-adjusted confidence "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'caution_adjusted_confidence', 'metric_value'].iloc[0]:.3f}."
        ),
        "ev_mixture_coherence": (
            f"EV mixture coherence kept smoothness proxy "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'theme_smoothness_proxy', 'metric_value'].iloc[0]:.3f} "
            f"and dominance margin "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'mean_dominance_margin', 'metric_value'].iloc[0]:.3f}."
        ),
        "covid_serum_usefulness": (
            f"COVID serum outputs kept modality caution mean "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'modality_caution_mean', 'metric_value'].iloc[0]:.3f} "
            f"with positive entropy inverse "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'positive_entropy_inverse', 'metric_value'].iloc[0]:.3f}."
        ),
        "replicate_consistency": (
            f"Replicate consistency reached "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'within_condition_consistency', 'metric_value'].iloc[0]:.3f} "
            f"with caution-adjusted confidence "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'caution_adjusted_confidence', 'metric_value'].iloc[0]:.3f}."
        ),
    }

    compare = metrics_df.merge(v1_metrics_df, on=["track_name", "metric_name"], suffixes=("_v2", "_v1"), how="left")
    compare["delta"] = compare["metric_value_v2"] - compare["metric_value_v1"]
    adenine_gain = float(compare.loc[compare["metric_name"] == "purine_dominance_margin", "delta"].iloc[0]) if (compare["metric_name"] == "purine_dominance_margin").any() else 0.0
    ev_delta = float(compare.loc[compare["metric_name"] == "theme_smoothness_proxy", "delta"].iloc[0]) if (compare["metric_name"] == "theme_smoothness_proxy").any() else 0.0
    protocol_delta = float(compare.loc[compare["metric_name"] == "positive_theme_stability", "delta"].iloc[0]) if (compare["metric_name"] == "positive_theme_stability").any() else 0.0
    replicate_delta = float(compare.loc[compare["metric_name"] == "within_condition_consistency", "delta"].iloc[0]) if (compare["metric_name"] == "within_condition_consistency").any() else 0.0

    covid_positive_delta = float(compare.loc[compare["metric_name"] == "positive_signal_mean", "delta"].iloc[0]) if (compare["metric_name"] == "positive_signal_mean").any() else 0.0

    if adenine_gain > 0 and ev_delta > -0.08 and protocol_delta > -0.08 and replicate_delta > -0.05:
        overall_assessment = (
            "v2 is better than v1 overall. The new competition and penalty logic reduces saturation and improves biochemical selectivity, "
            "especially on the adenine control track, without materially breaking EV mixture coherence, serum protocol robustness, or replicate stability. "
            "It remains conservative and inspectable. The main remaining weakness is that positive-theme amplitude and confidence are still somewhat compressed, "
            f"as seen in the COVID serum positive-signal delta of {covid_positive_delta:.3f}, so v2 is a better default for internal deterministic interpretation "
            "but still needs confidence recalibration before any later user-facing language layer."
        )
    else:
        overall_assessment = (
            "v2 improves selectivity but not cleanly enough across all tracks to declare an unconditional win. "
            "It is promising, but should only replace v1 if the comparison bundle shows that the specificity gains are not bought with instability."
        )

    figure_paths = []
    for builder in [
        build_architecture_figure,
        lambda: build_example_theme_figure(all_theme_df, v1_theme_df),
        lambda: build_evidence_contribution_figure(all_theme_df),
        lambda: build_adenine_specificity_figure(all_theme_df),
        lambda: build_serum_protocol_figure(all_theme_df),
        lambda: build_ev_mixture_figure(all_theme_df),
        lambda: build_covid_figure(all_theme_df),
        lambda: build_stability_figure(all_theme_df),
        lambda: build_global_metric_figure(metrics_df),
    ]:
        _, png_path = builder()
        figure_paths.append(png_path)

    report_md_path = write_report(metrics_df, v1_metrics_df, figure_paths, track_summaries, overall_assessment)
    report_pdf_path = build_report_pdf(report_md_path, figure_paths)

    print(f"Wrote query outputs: {RAW_DIR / 'theme_query_outputs.csv'}")
    print(f"Wrote long theme outputs: {RAW_DIR / 'theme_per_query_outputs_long.csv'}")
    print(f"Wrote metric table: {TABLE_DIR / 'theme_track_metrics.csv'}")
    print(f"Wrote report markdown: {report_md_path}")
    print(f"Wrote report pdf: {report_pdf_path}")
    print(f"Generated figures: {len(figure_paths)}")


if __name__ == "__main__":
    main()
