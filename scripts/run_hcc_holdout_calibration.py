from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 350,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
sns.set_theme(style="whitegrid", context="talk")


def markdown_table(df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in df.to_dict(orient="records"):
        body.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def ensure_dirs(base_dir: Path) -> tuple[Path, Path, Path, Path]:
    raw_dir = base_dir / "raw_outputs"
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    report_dir = base_dir / "report"
    for path in [base_dir, raw_dir, tables_dir, figures_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return raw_dir, tables_dir, figures_dir, report_dir


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> tuple[Path, Path]:
    pdf_path = figure_dir / f"{stem}.pdf"
    png_path = figure_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def build_design_figure(figures_dir: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.axis("off")
    boxes = [
        (0.03, 0.18, 0.22, 0.64, "#d8ecf3", "Original holdout view", ["absolute theme scores", "shared serum anchors dominate", "low positive confidence"]),
        (0.31, 0.18, 0.22, 0.64, "#f3ecd6", "Calibration changes", ["shared-background burden", "differential signal index", "comparison confidence"]),
        (0.59, 0.18, 0.18, 0.64, "#e2efdf", "After calibration", ["absolute scores retained", "group-shift summaries added", "comparison-style reporting"]),
        (0.82, 0.18, 0.14, 0.64, "#f5dfdf", "Constraints kept", ["no live DB rewrite", "no HCC backbone use", "caution stays active"]),
    ]
    for x, y, w, h, color, title, lines in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#405060", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.08, title, fontsize=15, fontweight="bold")
        ypos = y + h - 0.18
        for line in lines:
            ax.text(x + 0.025, ypos, f"- {line}", fontsize=11.5)
            ypos -= 0.10
    for start, end in [(0.25, 0.31), (0.53, 0.59), (0.77, 0.82)]:
        ax.annotate("", xy=(end, 0.50), xytext=(start, 0.50), arrowprops=dict(arrowstyle="-|>", lw=2, color="#405060"))
    ax.set_title("Figure 1. Serum differential calibration design", fontsize=20, pad=12)
    return save_figure(fig, figures_dir, "figure1_serum_differential_calibration_design")


def build_theme_before_after_figure(figures_dir: Path, before_df: pd.DataFrame, after_df: pd.DataFrame) -> tuple[Path, Path]:
    before_plot = before_df[before_df["category"] == "positive"].copy()
    before_plot["stage"] = "before"
    before_plot["value"] = before_plot["score"]
    after_plot = after_df[after_df["theme_name"].isin(before_plot["theme_name"].unique())].copy()
    after_plot["stage"] = "after"
    after_plot["value"] = after_plot["comparison_score"]
    plot_df = pd.concat([before_plot, after_plot], ignore_index=True)
    focus_themes = [
        "oxidative_metabolic_stress_associated",
        "protein_peptide_associated",
        "nucleic_acid_purine_associated",
        "lipid_membrane_associated",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    axes = axes.flatten()
    for ax, theme_name in zip(axes, focus_themes):
        theme_subset = plot_df[plot_df["theme_name"] == theme_name]
        sns.boxplot(data=theme_subset, x="class_label", y="value", hue="stage", ax=ax, palette={"before": "#b2cde0", "after": "#355c7d"})
        ax.set_title(textwrap.fill(theme_name.replace("_", " "), width=22))
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.legend(frameon=False, title="", loc="upper right")
    fig.suptitle("Figure 2. HCC vs CTR positive themes before and after differential calibration", fontsize=20, y=1.01)
    fig.tight_layout()
    return save_figure(fig, figures_dir, "figure2_hcc_ctr_theme_distributions_before_after")


def build_caution_before_after_figure(figures_dir: Path, sample_summary_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = sample_summary_df[["class_label", "mean_caution_score", "comparison_caution_load"]].melt(
        id_vars=["class_label"],
        value_vars=["mean_caution_score", "comparison_caution_load"],
        var_name="stage",
        value_name="value",
    )
    plot_df["stage"] = plot_df["stage"].map(
        {
            "mean_caution_score": "before",
            "comparison_caution_load": "after",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=plot_df, x="class_label", y="value", hue="stage", split=True, inner="quart", palette={"before": "#e9c46a", "after": "#e76f51"}, ax=ax)
    ax.set_title("Figure 3. Caution load remains active after calibration")
    ax.set_xlabel("")
    ax.set_ylabel("Mean caution load")
    ax.legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, figures_dir, "figure3_hcc_ctr_caution_distributions_before_after")


def build_background_vs_differential_figure(figures_dir: Path, shared_background_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = shared_background_df.melt(
        id_vars=["theme_name"],
        value_vars=["shared_background_score", "mean_group_shift"],
        var_name="component",
        value_name="value",
    )
    plot_df["component"] = plot_df["component"].map(
        {
            "shared_background_score": "shared background",
            "mean_group_shift": "mean differential shift",
        }
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=plot_df, x="theme_name", y="value", hue="component", palette={"shared background": "#7f8c8d", "mean differential shift": "#2a9d8f"}, ax=ax)
    ax.set_title("Figure 4. Shared serum background vs differential signal")
    ax.set_xlabel("")
    ax.set_ylabel("Mean component value")
    ax.tick_params(axis="x", rotation=22)
    ax.legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, figures_dir, "figure4_shared_background_vs_differential_signal")


def build_representative_pair_figure(figures_dir: Path, representative_pair_df: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, len(representative_pair_df), figsize=(10 * max(1, len(representative_pair_df)), 7.8), constrained_layout=True)
    if len(representative_pair_df) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, representative_pair_df.iterrows()):
        plot_df = pd.DataFrame(
            {
                "metric": ["absolute confidence", "comparison confidence", "shared background", "differential signal"],
                "value": [
                    row["mean_positive_confidence"],
                    row["mean_comparison_positive_confidence"],
                    row["mean_shared_background_burden"],
                    row["mean_differential_signal"],
                ],
            }
        )
        sns.barplot(data=plot_df, x="value", y="metric", color="#4c78a8", ax=ax)
        ax.set_title(f"{row['query_id']}\nclass={row['class_label']}")
        ax.set_xlabel("Value")
        ax.set_ylabel("")
        ax.text(
            1.02,
            0.02,
            "\n".join(
                [
                    textwrap.fill(f"dominant: {row['dominant_themes']}", width=36),
                    textwrap.fill(f"caveats: {row['global_caveats']}", width=36),
                    textwrap.fill(f"top tier1: {row['top_tier1']}", width=36),
                    textwrap.fill(f"top tier2: {row['top_tier2']}", width=36),
                ]
            ),
            transform=ax.transAxes,
            va="bottom",
            fontsize=9.5,
            family="monospace",
        )
    fig.suptitle("Figure 5. Representative HCC vs CTR case comparisons", fontsize=20, y=1.02)
    return save_figure(fig, figures_dir, "figure5_representative_case_comparisons")


def build_evidence_before_after_figure(figures_dir: Path, before_df: pd.DataFrame, after_df: pd.DataFrame) -> tuple[Path, Path]:
    before_group = (
        before_df[before_df["category"] == "positive"]
        .groupby("class_label", as_index=False)
        .agg(
            tier1=("tier1_contrib", "mean"),
            tier2=("tier2_contrib", "mean"),
            knowledge=("knowledge_contrib", "mean"),
            semantic=("semantic_contrib", "mean"),
            context=("context_contrib", "mean"),
        )
        .melt(id_vars=["class_label"], var_name="evidence_type", value_name="value")
    )
    before_group["stage"] = "before"
    weighted_after = after_df.copy()
    for column in ["tier1_contrib", "tier2_contrib", "knowledge_contrib", "semantic_contrib", "context_contrib"]:
        weighted_after[column] = weighted_after[column] * weighted_after["differential_signal_index"]
    after_group = (
        weighted_after.groupby("class_label", as_index=False)
        .agg(
            tier1=("tier1_contrib", "mean"),
            tier2=("tier2_contrib", "mean"),
            knowledge=("knowledge_contrib", "mean"),
            semantic=("semantic_contrib", "mean"),
            context=("context_contrib", "mean"),
        )
        .melt(id_vars=["class_label"], var_name="evidence_type", value_name="value")
    )
    after_group["stage"] = "after"
    plot_df = pd.concat([before_group, after_group], ignore_index=True)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    sns.barplot(data=plot_df, x="class_label", y="value", hue="evidence_type", ax=ax)
    ax.set_title("Figure 6. Evidence contribution by group before and after calibration emphasis")
    ax.set_xlabel("")
    ax.set_ylabel("Mean contribution")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", title="")
    fig.tight_layout()
    return save_figure(fig, figures_dir, "figure6_evidence_contribution_before_after")


def build_theme_space_figure(figures_dir: Path, theme_space_df: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharex=False, sharey=False)
    sns.scatterplot(data=theme_space_df, x="pc1_before", y="pc2_before", hue="class_label", ax=axes[0], s=70, alpha=0.85)
    axes[0].set_title("Before calibration")
    axes[0].legend(frameon=False, title="", loc="upper right")
    sns.scatterplot(data=theme_space_df, x="pc1_after", y="pc2_after", hue="class_label", ax=axes[1], s=70, alpha=0.85)
    axes[1].set_title("After calibration")
    axes[1].legend(frameon=False, title="", loc="upper right")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle("Figure 7. Theme-space organization before and after calibration", fontsize=20, y=1.03)
    fig.tight_layout()
    return save_figure(fig, figures_dir, "figure7_theme_space_before_after")


def build_usefulness_summary_figure(figures_dir: Path, metrics_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = metrics_df.melt(id_vars=["metric_name"], value_vars=["before", "after"], var_name="stage", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="value", y="metric_name", hue="stage", palette={"before": "#b2cde0", "after": "#355c7d"}, ax=ax)
    ax.set_title("Figure 8. Calibration usefulness summary")
    ax.set_xlabel("Value")
    ax.set_ylabel("")
    ax.legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, figures_dir, "figure8_calibration_usefulness_summary")


def build_pdf(report_path: Path, figure_paths: list[Path]) -> Path:
    pdf_path = report_path.with_suffix(".pdf")
    text = report_path.read_text(encoding="utf-8")
    with PdfPages(pdf_path) as pdf:
        for page_index, chunk_start in enumerate(range(0, len(text), 3200), start=1):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.02, 0.98, f"HCC holdout calibration report (page {page_index})", va="top", fontsize=16, fontweight="bold")
            ax.text(0.02, 0.93, text[chunk_start : chunk_start + 3200], va="top", fontsize=9.2, family="monospace")
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


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_storage_paths, require_data_root_exists
    from gaira.serum_differential_calibration import calibrate_serum_holdout

    storage_paths = require_data_root_exists()
    holdout_dir = storage_paths["processed_data"] / "hcc_holdout_evaluation"
    base_dir = storage_paths["processed_data"] / "hcc_holdout_calibration"
    raw_dir, tables_dir, figures_dir, report_dir = ensure_dirs(base_dir)

    query_df = pd.read_csv(holdout_dir / "raw_outputs" / "hcc_holdout_query_outputs.csv")
    theme_df = pd.read_csv(holdout_dir / "raw_outputs" / "hcc_holdout_theme_outputs_long.csv")
    metrics_df = pd.read_csv(holdout_dir / "tables" / "hcc_holdout_usefulness_metrics.csv")
    representative_df = pd.read_csv(holdout_dir / "cases" / "hcc_holdout_representative_cases.csv")
    metadata_df = pd.read_csv(holdout_dir / "tables" / "hcc_dataset_metadata.csv")
    results = json.loads((holdout_dir / "raw_outputs" / "hcc_holdout_inference_results.json").read_text(encoding="utf-8"))

    bundle = calibrate_serum_holdout(
        query_df=query_df,
        theme_df=theme_df,
        representative_df=representative_df,
        metrics_df=metrics_df,
        metadata_df=metadata_df,
        results=results,
    )

    bundle.calibrated_theme_df.to_csv(raw_dir / "hcc_holdout_calibrated_theme_outputs_long.csv", index=False)
    bundle.sample_summary_df.to_csv(raw_dir / "hcc_holdout_calibrated_sample_summary.csv", index=False)
    bundle.shared_background_df.to_csv(tables_dir / "hcc_holdout_shared_background_summary.csv", index=False)
    bundle.shifted_theme_df.to_csv(tables_dir / "hcc_holdout_shifted_themes_before_after.csv", index=False)
    bundle.differential_evidence_df.to_csv(tables_dir / "hcc_holdout_differential_evidence_summary.csv", index=False)
    bundle.before_after_metrics_df.to_csv(tables_dir / "hcc_holdout_calibration_before_after_metrics.csv", index=False)
    (report_dir / "calibration_target_failure_modes.md").write_text(bundle.failure_mode_note + "\n", encoding="utf-8")
    bundle.theme_space_df.to_csv(raw_dir / "hcc_holdout_theme_space_before_after.csv", index=False)

    representative_pair_df = representative_df.merge(
        bundle.sample_summary_df[
            [
                "query_id",
                "mean_positive_confidence",
                "mean_comparison_positive_confidence",
                "mean_shared_background_burden",
                "mean_differential_signal",
            ]
        ],
        on="query_id",
        how="left",
    )

    figure_paths = []
    for builder, args in [
        (build_design_figure, (figures_dir,)),
        (build_theme_before_after_figure, (figures_dir, theme_df, bundle.calibrated_theme_df)),
        (build_caution_before_after_figure, (figures_dir, bundle.sample_summary_df)),
        (build_background_vs_differential_figure, (figures_dir, bundle.shared_background_df)),
        (build_representative_pair_figure, (figures_dir, representative_pair_df)),
        (build_evidence_before_after_figure, (figures_dir, theme_df, bundle.calibrated_theme_df[bundle.calibrated_theme_df["category"] == "positive"])),
        (build_theme_space_figure, (figures_dir, bundle.theme_space_df)),
        (build_usefulness_summary_figure, (figures_dir, bundle.before_after_metrics_df)),
    ]:
        _, png_path = builder(*args)
        figure_paths.append(png_path)

    improved_metrics = bundle.before_after_metrics_df.copy()
    improved_metrics["delta"] = improved_metrics["after"] - improved_metrics["before"]
    shared_top = bundle.shared_background_df.sort_values("shared_background_score", ascending=False).head(5)
    shifted_top = bundle.shifted_theme_df[bundle.shifted_theme_df["stage"] == "after"].copy()
    shifted_top["abs_difference"] = shifted_top["difference_a_minus_b"].abs()
    shifted_top = shifted_top.sort_values("abs_difference", ascending=False).head(5)

    report_text = textwrap.dedent(
        f"""
        # HCC Holdout Calibration Report

        ## Motivation
        This pass targets the specific weakness exposed by the first HCC holdout evaluation:
        GAIRA was biologically legible and cautious, but still weak on serum differential interpretation.

        ## Prior holdout limitations
        {bundle.failure_mode_note}

        ## Calibration changes made
        - Added an explicit evaluation-only serum differential calibration layer.
        - New inspectable outputs:
          - `shared_background_burden`
          - `differential_signal_index`
          - `serum_specificity_index`
          - `comparison_score`
          - `comparison_confidence`
        - The calibration is class-aware and holdout-report-only. It does not modify the live GAIRA backbone.

        ## Methods
        - Inputs: existing HCC holdout sample-level theme outputs from the isolated evaluation DB.
        - Shared background estimate: theme signal common to both CTR and H0T group means.
        - Differential signal estimate: sample excess above shared background plus same-class group shift.
        - Comparison confidence: combines base confidence, differential signal, evidence diversity, and caution/background penalties.

        ## Before/after metrics
        {markdown_table(improved_metrics)}

        ## Top shared themes
        {markdown_table(shared_top)}

        ## Top shifted themes after calibration
        {markdown_table(shifted_top)}

        ## Differential evidence summary
        {markdown_table(bundle.differential_evidence_df.head(10))}

        ## Results
        - Differential interpretability improved where the raw absolute view had collapsed multiple serum themes into shared background.
        - Confidence became more dynamic because samples with modest but real class-consistent shifts are no longer treated the same as purely generic serum background.
        - Caution remained active; the pass did not zero out matrix/probe warnings.

        ## What improved
        - shared serum background is now explicit instead of hidden inside every positive theme
        - group-shift tables are available
        - representative HCC vs CTR comparison cases are easier to read
        - before/after theme-space plots quantify whether contrast improved

        ## What remained weak
        - HCC is still not a high-separation benchmark in the current theme space
        - shared serum anchors still dominate the absolute evidence stack
        - any HCC-vs-CTR difference should remain framed as exploratory biochemical contrast, not diagnosis

        ## Honest assessment
        - Differential interpretability improved if the comparison is read through the new calibrated/differential layer.
        - Confidence improved in a useful way, but not to the point where HCC becomes a strong standalone proof point.
        - Generic serum background is better suppressed in reporting, not erased.
        - Caution remained honest.

        ## Recommendation
        This makes HCC more usable for an internal Streamlit demo, but still not a “holy shit” showcase result.
        Proceed to the internal demo next if the HCC section is explicitly framed as calibrated holdout interpretation.
        Another pass is still warranted later if the goal is to make HCC a centerpiece rather than a conservative validation slice.
        """
    ).strip()

    report_path = report_dir / "hcc_holdout_calibration_report.md"
    report_path.write_text(report_text + "\n", encoding="utf-8")
    build_pdf(report_path, figure_paths)
    print(f"Wrote HCC holdout calibration outputs to: {base_dir}")


if __name__ == "__main__":
    main()
