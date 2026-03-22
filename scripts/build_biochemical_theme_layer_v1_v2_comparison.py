from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA")
V1_DIR = ROOT / "processed" / "biochemical_theme_layer_v1"
V2_DIR = ROOT / "processed" / "biochemical_theme_layer_v2"
OUTPUT_DIR = ROOT / "processed" / "biochemical_theme_layer_v1_v2_comparison"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report"


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
    for path in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def build_example_comparison(theme_df: pd.DataFrame) -> tuple[Path, Path]:
    example_ids = [
        "small2023_ev_c50_normedprobe1",
        "covid_serum_raman_covid_confirmed_covid19_serum_raman_archive",
    ]
    plot_df = theme_df[theme_df["query_id"].isin(example_ids)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for ax, query_id in zip(axes, example_ids):
        subset = plot_df[plot_df["query_id"] == query_id].copy()
        top_themes = subset.groupby("theme_name")["score"].max().sort_values(ascending=False).head(6).index.tolist()
        subset = subset[subset["theme_name"].isin(top_themes)]
        sns.barplot(data=subset, x="score", y="theme_name", hue="version", palette={"v1": "#9bbad0", "v2": "#355c7d"}, ax=ax)
        ax.set_title(query_id)
        ax.set_xlabel("Theme score")
        ax.set_ylabel("")
        ax.legend(frameon=False, title="")
    fig.suptitle("Figure 1. v1-v2 representative query comparison", fontsize=20, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "figure1_v1_v2_examples")


def build_track_metric_comparison(metrics_df: pd.DataFrame) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(18, 7))
    sns.barplot(data=metrics_df, x="metric_name", y="metric_value", hue="version", palette={"v1": "#9bbad0", "v2": "#355c7d"}, ax=ax)
    ax.set_title("Figure 2. v1-v2 metric comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return save_figure(fig, "figure2_v1_v2_metrics")


def build_track_delta_heatmap(delta_df: pd.DataFrame) -> tuple[Path, Path]:
    heat_df = delta_df.pivot(index="metric_name", columns="track_name", values="delta_v2_minus_v1").fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heat_df, cmap="coolwarm", center=0.0, annot=True, fmt=".2f", cbar_kws={"label": "v2 - v1"}, ax=ax)
    ax.set_title("Figure 3. v1-v2 metric delta heatmap")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return save_figure(fig, "figure3_v1_v2_delta_heatmap")


def build_distribution_comparison(theme_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = theme_df[theme_df["category"] == "positive"].copy()
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.violinplot(data=plot_df, x="theme_name", y="score", hue="version", palette={"v1": "#9bbad0", "v2": "#355c7d"}, split=False, ax=ax)
    ax.set_title("Figure 4. Positive-theme score distribution comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Theme score")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return save_figure(fig, "figure4_v1_v2_score_distributions")


def build_confidence_comparison(theme_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = theme_df[theme_df["category"] == "positive"].copy()
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.scatterplot(data=plot_df, x="score", y="confidence", hue="version", style="theme_name", palette={"v1": "#9bbad0", "v2": "#355c7d"}, ax=ax)
    ax.set_title("Figure 5. Score-confidence relationship in v1 vs v2")
    ax.set_xlabel("Theme score")
    ax.set_ylabel("Confidence")
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return save_figure(fig, "figure5_v1_v2_confidence")


def build_pdf(report_md_path: Path, figure_paths: list[Path]) -> Path:
    pdf_path = REPORT_DIR / "biochemical_theme_layer_v1_v2_comparison_report.pdf"
    text = report_md_path.read_text(encoding="utf-8")
    with PdfPages(pdf_path) as pdf:
        for page_index, chunk_start in enumerate(range(0, len(text), 3200), start=1):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.02, 0.98, f"Biochemical Theme Layer v1-v2 comparison (page {page_index})", fontsize=16, fontweight="bold", va="top")
            ax.text(0.02, 0.93, text[chunk_start : chunk_start + 3200], fontsize=9, va="top", family="monospace")
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
    ensure_dirs()

    v1_theme = pd.read_csv(V1_DIR / "raw_outputs" / "theme_per_query_outputs_long.csv")
    v2_theme = pd.read_csv(V2_DIR / "raw_outputs" / "theme_per_query_outputs_long.csv")
    v1_theme["version"] = "v1"
    v2_theme["version"] = "v2"
    theme_df = pd.concat([v1_theme, v2_theme], ignore_index=True)

    v1_metrics = pd.read_csv(V1_DIR / "tables" / "theme_track_metrics.csv")
    v2_metrics = pd.read_csv(V2_DIR / "tables" / "theme_track_metrics.csv")
    v1_metrics["version"] = "v1"
    v2_metrics["version"] = "v2"
    metrics_df = pd.concat([v1_metrics, v2_metrics], ignore_index=True)

    delta_df = v2_metrics.merge(v1_metrics, on=["track_name", "metric_name"], suffixes=("_v2", "_v1"), how="outer")
    delta_df["delta_v2_minus_v1"] = delta_df["metric_value_v2"] - delta_df["metric_value_v1"]
    delta_df.to_csv(TABLE_DIR / "metric_delta_summary.csv", index=False)

    overall_rows = []
    for metric_name in ["purine_dominance_margin", "off_theme_suppression", "positive_theme_stability", "theme_smoothness_proxy", "within_condition_consistency", "positive_entropy_inverse", "caution_adjusted_confidence"]:
        subset = delta_df[delta_df["metric_name"] == metric_name]
        if subset.empty:
            continue
        overall_rows.append(subset[["track_name", "metric_name", "metric_value_v1", "metric_value_v2", "delta_v2_minus_v1"]])
    summary_df = pd.concat(overall_rows, ignore_index=True) if overall_rows else pd.DataFrame()
    summary_df.to_csv(TABLE_DIR / "comparison_summary.csv", index=False)

    figure_paths = []
    for builder in [
        lambda: build_example_comparison(theme_df),
        lambda: build_track_metric_comparison(metrics_df),
        lambda: build_track_delta_heatmap(delta_df),
        lambda: build_distribution_comparison(theme_df),
        lambda: build_confidence_comparison(theme_df),
    ]:
        _, png_path = builder()
        figure_paths.append(png_path)

    adenine_gain = summary_df.loc[summary_df["metric_name"] == "purine_dominance_margin", "delta_v2_minus_v1"].mean() if not summary_df.empty else 0.0
    ev_delta = summary_df.loc[summary_df["metric_name"] == "theme_smoothness_proxy", "delta_v2_minus_v1"].mean() if not summary_df.empty else 0.0
    protocol_delta = summary_df.loc[summary_df["metric_name"] == "positive_theme_stability", "delta_v2_minus_v1"].mean() if not summary_df.empty else 0.0
    replicate_delta = summary_df.loc[summary_df["metric_name"] == "within_condition_consistency", "delta_v2_minus_v1"].mean() if not summary_df.empty else 0.0

    covid_positive_delta = delta_df.loc[delta_df["metric_name"] == "positive_signal_mean", "delta_v2_minus_v1"].mean() if not delta_df.empty else 0.0

    if adenine_gain > 0 and ev_delta > -0.08 and protocol_delta > -0.08 and replicate_delta > -0.05:
        decision = (
            "v2 is better overall and should replace v1 as the active deterministic theme layer for internal use, "
            f"but with a clear caveat: positive-theme amplitudes are still compressed in some cohort settings (COVID positive-signal delta {covid_positive_delta:.3f}), "
            "so confidence calibration is not finished."
        )
    else:
        decision = "v2 is not a clean enough overall improvement yet to replace v1 automatically."

    report_text = f"""# GAIRA Biochemical Theme Layer v1-v2 comparison

## 1. Purpose
This report compares biochemical theme layer v1 and v2 on the same evaluation tracks.

## 2. What changed in v2
- competitive evidence allocation across positive themes
- explicit negative evidence handling
- calibration-control downweighting for adenine in biosample summaries
- tighter caution- and competition-aware confidence

## 3. Comparison summary table
{summary_df.to_string(index=False) if not summary_df.empty else 'No comparison rows available.'}

## 4. Metric delta table
{delta_df.to_string(index=False)}

## 5. Decision
{decision}

## 6. Interpretation
The key question is whether v2 improves selectivity without breaking stability. Adenine specificity and off-theme suppression matter most for sharpening, while EV smoothness, serum protocol stability, and replicate consistency protect against brittle overfitting.
"""
    report_md = REPORT_DIR / "biochemical_theme_layer_v1_v2_comparison_report.md"
    report_md.write_text(report_text, encoding="utf-8")
    report_pdf = build_pdf(report_md, figure_paths)

    print(f"Wrote comparison summary: {TABLE_DIR / 'comparison_summary.csv'}")
    print(f"Wrote metric deltas: {TABLE_DIR / 'metric_delta_summary.csv'}")
    print(f"Wrote report markdown: {report_md}")
    print(f"Wrote report pdf: {report_pdf}")
    print(f"Generated figures: {len(figure_paths)}")


if __name__ == "__main__":
    main()
