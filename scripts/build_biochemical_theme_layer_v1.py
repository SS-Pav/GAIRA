from __future__ import annotations

import math
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
OUTPUT_DIR = ROOT / "processed" / "biochemical_theme_layer_v1"
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


def melt_theme_scores(theme_df: pd.DataFrame) -> pd.DataFrame:
    return theme_df.copy()


def build_architecture_figure() -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    nodes = [
        (0.03, 0.25, 0.17, 0.5, "#d8ecf3", "GAIRA evidence stack", ["Tier 1 direct grounding", "Tier 2 support", "Knowledge support", "Semantic support", "Domain context"]),
        (0.27, 0.25, 0.18, 0.5, "#f4efe1", "Theme layer v1", ["Rule-based ontology", "Evidence weighting", "Band support", "Confidence + caveats"]),
        (0.52, 0.25, 0.18, 0.5, "#e8f1df", "Theme outputs", ["Positive themes", "Caution themes", "Dominant themes", "What not to claim"]),
        (0.77, 0.25, 0.18, 0.5, "#f7e4e2", "Evaluation", ["Adenine specificity", "Serum protocol robustness", "EV coherence", "COVID usefulness"]),
    ]
    for x, y, w, h, color, title, lines in nodes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#4d5a6a", linewidth=1.3)
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.08, title, fontsize=16, fontweight="bold", ha="left")
        line_y = y + h - 0.17
        for line in lines:
            ax.text(x + 0.025, line_y, f"- {line}", fontsize=12, ha="left")
            line_y -= 0.09
    for start, end in [(0.20, 0.27), (0.45, 0.52), (0.70, 0.77)]:
        ax.annotate("", xy=(end, 0.5), xytext=(start, 0.5), arrowprops=dict(arrowstyle="-|>", lw=2, color="#4d5a6a"))
    ax.set_title("Figure 1. GAIRA biochemical theme layer architecture", fontsize=20, pad=12)
    return save_figure(fig, "figure1_theme_layer_architecture")


def build_example_theme_figure(theme_df: pd.DataFrame) -> tuple[Path, Path]:
    example_ids = [
        "small2023_ev_c50_normedprobe1",
        "covid_serum_raman_covid_confirmed_covid19_serum_raman_archive",
    ]
    plot_df = theme_df[theme_df["query_id"].isin(example_ids)].copy()
    plot_df["theme_short"] = plot_df["theme_name"].str.replace("_associated", "", regex=False).str.replace("_caution", "", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, query_id in zip(axes, example_ids):
        subset = plot_df[plot_df["query_id"] == query_id].copy()
        subset = subset.sort_values("score", ascending=False).head(8)
        sns.barplot(data=subset, x="score", y="theme_short", hue="category", palette={"positive": "#355c7d", "caution": "#c06c84"}, dodge=False, ax=ax)
        ax.set_title(query_id)
        ax.set_xlabel("Theme score")
        ax.set_ylabel("")
        ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Figure 2. Theme score composition examples", fontsize=20, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "figure2_theme_score_examples")


def build_evidence_contribution_figure(theme_df: pd.DataFrame) -> tuple[Path, Path]:
    contrib_cols = ["tier1_contrib", "tier2_contrib", "knowledge_contrib", "semantic_contrib", "context_contrib", "band_contrib"]
    plot_df = (
        theme_df.groupby("theme_name", as_index=False)[contrib_cols]
        .mean()
        .melt(id_vars="theme_name", var_name="evidence_type", value_name="contribution")
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=plot_df, x="theme_name", y="contribution", hue="evidence_type", palette="crest", ax=ax)
    ax.set_title("Figure 3. Evidence-type contribution by theme")
    ax.set_xlabel("")
    ax.set_ylabel("Mean contribution")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return save_figure(fig, "figure3_evidence_type_contributions")


def build_adenine_specificity_figure(theme_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = theme_df[
        (theme_df["track_name"] == "adenine_controlled_specificity")
        & (theme_df["theme_name"].isin(["nucleic_acid_purine_associated", "protein_peptide_associated", "lipid_membrane_associated", "low_specificity_caution"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.pointplot(data=plot_df, x="query_label", y="score", hue="theme_name", dodge=0.35, markers="o", linestyles="-", ax=ax)
    ax.set_title("Figure 4. Controlled analyte specificity for adenine grounding")
    ax.set_xlabel("Adenine condition")
    ax.set_ylabel("Theme score")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return save_figure(fig, "figure4_adenine_specificity")


def build_serum_protocol_figure(theme_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = theme_df[
        (theme_df["track_name"] == "serum_protocol_robustness")
        & (theme_df["theme_name"].isin(["nucleic_acid_purine_associated", "protein_peptide_associated", "matrix_dominance_caution", "probe_substrate_caution"]))
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    for ax, theme_name in zip(axes.ravel(), plot_df["theme_name"].unique()):
        subset = plot_df[plot_df["theme_name"] == theme_name]
        sns.boxplot(data=subset, x="query_label", y="score", color="#8ab6d6", ax=ax)
        ax.set_title(theme_name)
        ax.set_xlabel("Protocol")
        ax.set_ylabel("Score")
    fig.suptitle("Figure 5. Serum protocol robustness", fontsize=20, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "figure5_serum_protocol_robustness")


def build_ev_mixture_figure(theme_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = theme_df[
        (theme_df["track_name"] == "ev_mixture_coherence")
        & (theme_df["theme_name"].isin(["lipid_membrane_associated", "nucleic_acid_purine_associated", "protein_peptide_associated", "probe_substrate_caution"]))
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
    fig.suptitle("Figure 6. EV mixture coherence", fontsize=20, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "figure6_ev_mixture_coherence")


def build_covid_figure(theme_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = theme_df[
        (theme_df["track_name"] == "covid_serum_usefulness")
        & (theme_df["theme_name"].isin(["protein_peptide_associated", "oxidative_metabolic_stress_associated", "modality_mismatch_caution", "low_specificity_caution"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(data=plot_df, x="query_label", y="score", hue="theme_name", palette="mako", ax=ax)
    ax.set_title("Figure 7. COVID serum usefulness and modality caution")
    ax.set_xlabel("Cohort group")
    ax.set_ylabel("Theme score")
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return save_figure(fig, "figure7_covid_serum_usefulness")


def build_stability_figure(theme_df: pd.DataFrame) -> tuple[Path, Path]:
    summary_df = (
        theme_df.groupby(["track_name", "theme_name"], as_index=False)
        .agg(mean_score=("score", "mean"), std_score=("score", "std"))
        .fillna(0.0)
    )
    heatmap_df = summary_df.pivot(index="theme_name", columns="track_name", values="std_score").fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_df, cmap="rocket_r", annot=True, fmt=".2f", cbar_kws={"label": "Std. dev."}, ax=ax)
    ax.set_title("Figure 8. Theme stability and consistency")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return save_figure(fig, "figure8_theme_stability")


def build_global_metric_figure(metrics_df: pd.DataFrame) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=metrics_df, x="metric_name", y="metric_value", hue="track_name", palette="viridis", ax=ax)
    ax.set_title("Figure 9. Global performance summary")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return save_figure(fig, "figure9_global_performance_summary")


def build_report_pdf(report_md_path: Path, figure_paths: list[Path]) -> Path:
    pdf_path = REPORT_DIR / "biochemical_theme_layer_v1_report.pdf"
    summary_text = report_md_path.read_text(encoding="utf-8")
    with PdfPages(pdf_path) as pdf:
        for page_index, chunk_start in enumerate(range(0, len(summary_text), 3200), start=1):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.02, 0.98, f"Biochemical Theme Layer v1 report (page {page_index})", fontsize=16, fontweight="bold", va="top")
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


def write_report(
    overall_assessment: str,
    metrics_df: pd.DataFrame,
    figure_paths: list[Path],
    track_summaries: dict[str, str],
) -> Path:
    figure_list = "\n".join(f"- {path.name}" for path in figure_paths)
    metric_table = metrics_df.to_string(index=False)
    report_text = f"""# GAIRA Biochemical Theme Layer v1

## 1. Overview
Biochemical Theme Layer v1 is a rule-based, evidence-weighted interpretation layer built on top of the live SSD_Rad-backed GAIRA inference stack. It reads tier-1 grounding, tier-2 support, knowledge support, semantic-region support, and domain context, then converts those signals into conservative biochemical themes plus explicit cautions.

## 2. Motivation
Raw hit lists are useful, but they are not yet a compact biochemical interpretation layer. The purpose of Theme Layer v1 is to summarize what the current GAIRA evidence stack is pointing toward without hiding provenance or overclaiming certainty.

## 3. Current evidence stack
- Tier 1 direct grounding
- Tier 2 literature and support grounding
- Knowledge-core support
- Semantic-region support
- EV and serum domain-context overlays

## 4. Theme ontology
- Positive themes: lipid_membrane_associated, protein_peptide_associated, nucleic_acid_purine_associated, carbohydrate_glycan_associated, oxidative_metabolic_stress_associated
- Caution themes: matrix_dominance_caution, probe_substrate_caution, modality_mismatch_caution, weak_label_or_cohort_caution, low_specificity_caution

## 5. Scoring logic
Theme scores are computed from explicit evidence contributions.
- tier1 weight = 1.00
- tier2 weight = 0.65
- knowledge weight = 0.60
- semantic weight = 0.50
- context weight = 0.35
- band support weight = 0.45

Confidence is reduced when caution burden is high, when support diversity is low, or when the theme is driven mainly by broad analog or modality-mismatched evidence.

## 6. Evaluation tracks
- Adenine controlled analyte specificity
- Serum protocol robustness
- EV mixture coherence
- COVID serum usefulness
- Replicate consistency

## 7. Results summary
{track_summaries['adenine_controlled_specificity']}

{track_summaries['serum_protocol_robustness']}

{track_summaries['ev_mixture_coherence']}

{track_summaries['covid_serum_usefulness']}

{track_summaries['replicate_consistency']}

## 8. Global metric table
{metric_table}

## 9. Figures
{figure_list}

## 10. Strengths
- The layer is fully inspectable and evidence-linked.
- Controlled analyte specificity is directly testable through adenine.
- Protocol and modality cautions remain explicit rather than hidden.
- EV and serum outputs become easier to compare than raw hit lists alone.

## 11. Failure modes and limitations
- Theme scores are still keyword-driven and depend on the current evidence vocabulary.
- Serum support remains partly serum-ag-colloids-centric.
- EV interpretation remains strongest around small2023_ev.
- Controlled analyte grounding can still appear strongly before reranking and must stay calibration-like.

## 12. Overall assessment
{overall_assessment}

## 13. Recommended next improvements
- expand shared biochemical support so theme mappings are less serum-ag-colloids-heavy
- add more EV-specific literature and band-note coverage
- add explicit negative theme evidence handling
- revisit whether controlled grounding like adenine should receive stronger downweighting for biosample inference summaries
- only consider a future LLM layer after the deterministic theme outputs stay stable under more dataset expansion
"""
    report_path = REPORT_DIR / "biochemical_theme_layer_v1_report.md"
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
    runner = ThemeEvaluationRunner(db_path=db_path)

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

    track_summaries = {
        "adenine_controlled_specificity": (
            f"Adenine specificity showed purine-theme dominance margin "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'purine_dominance_margin', 'metric_value'].iloc[0]:.3f} "
            f"with top-theme fraction "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'purine_top_fraction', 'metric_value'].iloc[0]:.3f}."
        ),
        "serum_protocol_robustness": (
            f"Serum protocol robustness produced positive-theme stability "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'positive_theme_stability', 'metric_value'].iloc[0]:.3f} "
            f"while keeping caution presence "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'caution_presence', 'metric_value'].iloc[0]:.3f}."
        ),
        "ev_mixture_coherence": (
            f"EV mixture coherence produced smoothness proxy "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'theme_smoothness_proxy', 'metric_value'].iloc[0]:.3f} "
            f"and caution variability "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'caution_variability', 'metric_value'].iloc[0]:.3f}."
        ),
        "covid_serum_usefulness": (
            f"COVID serum outputs kept modality caution mean "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'modality_caution_mean', 'metric_value'].iloc[0]:.3f} "
            f"and low-specificity mean "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'low_specificity_mean', 'metric_value'].iloc[0]:.3f}."
        ),
        "replicate_consistency": (
            f"Replicate consistency on adenine replicate series reached "
            f"{metrics_df.loc[metrics_df['metric_name'] == 'within_condition_consistency', 'metric_value'].iloc[0]:.3f}."
        ),
    }

    overall_assessment = (
        "Promising but early. The theme layer is already more interpretable than raw hit lists and is "
        "chemically sensible on controlled analyte and protocol-robustness checks, but it remains partially "
        "vocabulary-driven and still depends on a support base that is stronger for serum and small2023 EV than "
        "for the rest of the stack. It is useful with caveats for internal interpretation and method development, "
        "but not yet externally robust enough for unconstrained explanation."
    )

    figure_paths = []
    for builder in [
        build_architecture_figure,
        lambda: build_example_theme_figure(all_theme_df),
        lambda: build_evidence_contribution_figure(all_theme_df),
        lambda: build_adenine_specificity_figure(all_theme_df),
        lambda: build_serum_protocol_figure(all_theme_df),
        lambda: build_ev_mixture_figure(all_theme_df),
        lambda: build_covid_figure(all_theme_df),
        lambda: build_stability_figure(all_theme_df),
        lambda: build_global_metric_figure(metrics_df),
    ]:
        pdf_path, png_path = builder()
        figure_paths.append(png_path)

    report_md_path = write_report(overall_assessment, metrics_df, figure_paths, track_summaries)
    report_pdf_path = build_report_pdf(report_md_path, figure_paths)

    print(f"Wrote query outputs: {RAW_DIR / 'theme_query_outputs.csv'}")
    print(f"Wrote long theme outputs: {RAW_DIR / 'theme_per_query_outputs_long.csv'}")
    print(f"Wrote metric table: {TABLE_DIR / 'theme_track_metrics.csv'}")
    print(f"Wrote report markdown: {report_md_path}")
    print(f"Wrote report pdf: {report_pdf_path}")
    print(f"Generated figures: {len(figure_paths)}")


if __name__ == "__main__":
    main()
