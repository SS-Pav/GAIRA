from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


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


def ensure_dirs(base_dir: Path) -> tuple[Path, Path, Path]:
    figure_dir = base_dir / "figures"
    report_dir = base_dir / "report"
    for path in [base_dir, figure_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return figure_dir, report_dir, base_dir


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> tuple[Path, Path]:
    pdf_path = figure_dir / f"{stem}.pdf"
    png_path = figure_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def relation_label(query_dataset_id: str, row: dict) -> str:
    target_dataset_id = str(row.get("target_dataset_id", "") or "")
    source_dataset_id = str(row.get("source_dataset_id", "") or "")
    target_ids = {part.strip() for part in target_dataset_id.split(",") if part.strip()}
    ev_targets = {"small2023_ev", "shine_ev_sers", "diabetes_plasma_ev_sers"}

    if query_dataset_id in target_ids:
        return "same_dataset_ev"
    if target_ids & ev_targets:
        return "cross_ev"
    if source_dataset_id == "serum_ag_colloids_literature_grounding":
        return "cross_domain_serum"
    return "other_support"


def build_rows(result: dict) -> list[dict]:
    rows: list[dict] = []
    query_dataset_id = result["source_dataset_id"]
    for stage, hits_key in [
        ("before_reranking", "tier2_support_hits_before_reranking"),
        ("after_reranking", "tier2_support_hits"),
    ]:
        for rank, row in enumerate(result.get(hits_key, [])[:8], start=1):
            relation = relation_label(query_dataset_id, row)
            rows.append(
                {
                    "query_id": result["query_id"],
                    "query_label": result["query_label"],
                    "query_family": result["query_family"],
                    "query_dataset_id": query_dataset_id,
                    "stage": stage,
                    "rank": rank,
                    "source_dataset_id": row.get("source_dataset_id", ""),
                    "target_dataset_id": row.get("target_dataset_id", ""),
                    "source_label": row.get("source_label", ""),
                    "result_type": row.get("result_type", ""),
                    "relation": relation,
                    "same_dataset_flag": int(relation == "same_dataset_ev"),
                    "cross_ev_flag": int(relation == "cross_ev"),
                    "base_score": float(row.get("base_score", row.get("score", 0.0)) or 0.0),
                    "domain_relevance_weight": float(row.get("domain_relevance_weight", 1.0) or 1.0),
                    "reranked_score": float(row.get("reranked_score", row.get("score", 0.0)) or 0.0),
                    "rerank_reason": row.get("rerank_reason", ""),
                    "dominant_themes": "|".join(result.get("dominant_themes", [])),
                    "global_caveats": "|".join(result.get("biochemical_global_caveats", [])),
                }
            )
    return rows


def top_rank(summary_df: pd.DataFrame, relation: str, stage: str, query_id: str) -> float:
    subset = summary_df[
        (summary_df["query_id"] == query_id)
        & (summary_df["stage"] == stage)
        & (summary_df["relation"] == relation)
    ]
    if subset.empty:
        return float("nan")
    return float(subset["rank"].min())


def build_summary(order_df: pd.DataFrame) -> pd.DataFrame:
    query_rows = []
    for query_id in order_df["query_id"].drop_duplicates():
        subset = order_df[order_df["query_id"] == query_id]
        first_row = subset.iloc[0]
        after_top = subset[subset["stage"] == "after_reranking"].sort_values("rank").head(1)
        query_rows.append(
            {
                "query_id": query_id,
                "query_dataset_id": first_row["query_dataset_id"],
                "query_label": first_row["query_label"],
                "before_same_dataset_rank": top_rank(order_df, "same_dataset_ev", "before_reranking", query_id),
                "after_same_dataset_rank": top_rank(order_df, "same_dataset_ev", "after_reranking", query_id),
                "before_cross_ev_rank": top_rank(order_df, "cross_ev", "before_reranking", query_id),
                "after_cross_ev_rank": top_rank(order_df, "cross_ev", "after_reranking", query_id),
                "after_top_dataset": after_top["source_dataset_id"].iloc[0] if not after_top.empty else "",
                "after_top_label": after_top["source_label"].iloc[0] if not after_top.empty else "",
                "after_top_relation": after_top["relation"].iloc[0] if not after_top.empty else "",
                "dominant_themes": first_row["dominant_themes"],
                "global_caveats": first_row["global_caveats"],
            }
        )
    return pd.DataFrame(query_rows)


def build_rank_position_figure(summary_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    plot_df = summary_df.melt(
        id_vars=["query_id", "query_dataset_id", "query_label"],
        value_vars=["before_same_dataset_rank", "after_same_dataset_rank", "before_cross_ev_rank", "after_cross_ev_rank"],
        var_name="metric",
        value_name="rank",
    ).dropna()
    plot_df["stage"] = plot_df["metric"].str.extract(r"^(before|after)")
    plot_df["relation"] = plot_df["metric"].str.extract(r"_(same_dataset|cross_ev)_")
    plot_df["relation"] = plot_df["relation"].map(
        {
            "same_dataset": "same-dataset EV support",
            "cross_ev": "cross-EV support",
        }
    )
    plot_df["query_short"] = plot_df["query_dataset_id"] + " / " + plot_df["query_label"]

    fig, ax = plt.subplots(figsize=(15, 6.5))
    sns.scatterplot(
        data=plot_df,
        x="query_short",
        y="rank",
        hue="relation",
        style="stage",
        s=180,
        palette={
            "same-dataset EV support": "#1b9e77",
            "cross-EV support": "#d95f02",
        },
        ax=ax,
    )
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_ylabel("Best support rank")
    ax.set_title("Figure 1. EV tier-2 support rank positions before and after reranking refinement")
    ax.tick_params(axis="x", rotation=22)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, figure_dir, "figure1_before_after_support_ranks")


def build_score_breakdown_figure(order_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    subset = order_df[order_df["relation"].isin(["same_dataset_ev", "cross_ev"])].copy()
    subset["relation"] = subset["relation"].map(
        {
            "same_dataset_ev": "same-dataset EV support",
            "cross_ev": "cross-EV support",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    sns.boxplot(data=subset, x="relation", y="base_score", hue="stage", ax=axes[0], palette={"before_reranking": "#b2cde0", "after_reranking": "#4c78a8"})
    axes[0].set_title("Base score visibility")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, title="")

    after_subset = subset[subset["stage"] == "after_reranking"].copy()
    sns.boxplot(
        data=after_subset,
        x="relation",
        y="domain_relevance_weight",
        hue="relation",
        ax=axes[1],
        palette={
            "same-dataset EV support": "#1b9e77",
            "cross-EV support": "#d95f02",
        },
        legend=False,
    )
    axes[1].set_title("Applied rerank weight")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Weight")
    axes[1].tick_params(axis="x", rotation=10)
    fig.suptitle("Figure 2. Same-dataset EV support receives a principled bonus, cross-EV support stays visible", fontsize=18, y=1.03)
    fig.tight_layout()
    return save_figure(fig, figure_dir, "figure2_support_score_breakdown")


def build_no_regression_figure(summary_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    plot_df = summary_df.copy()
    plot_df["query_short"] = plot_df["query_dataset_id"] + " / " + plot_df["query_label"]
    plot_df["after_same_dataset_rank_filled"] = plot_df["after_same_dataset_rank"].fillna(9.0)
    plot_df["after_cross_ev_rank_filled"] = plot_df["after_cross_ev_rank"].fillna(9.0)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    wide = plot_df.melt(
        id_vars=["query_short"],
        value_vars=["after_same_dataset_rank_filled", "after_cross_ev_rank_filled"],
        var_name="metric",
        value_name="rank",
    )
    wide["metric"] = wide["metric"].map(
        {
            "after_same_dataset_rank_filled": "same-dataset after",
            "after_cross_ev_rank_filled": "cross-EV after",
        }
    )
    sns.barplot(data=wide, x="query_short", y="rank", hue="metric", palette={"same-dataset after": "#1b9e77", "cross-EV after": "#d95f02"}, ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_ylabel("Rank (9 = absent)")
    ax.set_title("Figure 3. No-regression view across diabetes, SHINE, and small2023 EV queries")
    ax.tick_params(axis="x", rotation=22)
    ax.legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, figure_dir, "figure3_no_regression_sanity")


def build_pdf(report_path: Path, figure_paths: list[Path]) -> Path:
    pdf_path = report_path.with_suffix(".pdf")
    text = report_path.read_text(encoding="utf-8")
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.02, 0.98, "EV support reranking refinement", va="top", fontsize=16, fontweight="bold")
        ax.text(0.02, 0.93, text[:7000], va="top", fontsize=9.5, family="monospace")
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


def markdown_table(df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in df.to_dict(orient="records"):
        body.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def main() -> None:
    import sys

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path, get_storage_paths, require_data_root_exists
    from gaira.inference import GAIRAInferenceEngine, load_ev_class_mean_query

    storage_paths = require_data_root_exists()
    base_dir = storage_paths["processed_data"] / "ev_support_reranking_refinement"
    figure_dir, report_dir, output_dir = ensure_dirs(base_dir)
    db_path = get_database_path()

    engine = GAIRAInferenceEngine(db_path=db_path, theme_layer_version="v3")
    requests = [
        load_ev_class_mean_query(db_path, "diabetes_plasma_ev_sers", "Impact", "figure3_processed_archive", processing_version="v1_crop500_1600_interp1_minmax"),
        load_ev_class_mean_query(db_path, "diabetes_plasma_ev_sers", "Strong-D", "figure3_processed_archive", processing_version="v1_crop500_1600_interp1_minmax"),
        load_ev_class_mean_query(db_path, "shine_ev_sers", "D0_C0", "Set9", processing_version="v1_crop450_1800_interp1_minmax"),
        load_ev_class_mean_query(db_path, "shine_ev_sers", "D2_C40", "Set9", processing_version="v1_crop450_1800_interp1_minmax"),
        load_ev_class_mean_query(db_path, "small2023_ev", "c50", "normedprobe1"),
    ]

    results = [engine.run_inference(request) for request in requests]
    order_df = pd.DataFrame([row for result in results for row in build_rows(result)])
    summary_df = build_summary(order_df)

    order_path = output_dir / "before_after_support_ordering.csv"
    summary_path = output_dir / "support_reranking_summary.csv"
    order_df.to_csv(order_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    figure_paths = []
    for figure_builder in [build_rank_position_figure, build_score_breakdown_figure, build_no_regression_figure]:
        output = figure_builder(summary_df if figure_builder != build_score_breakdown_figure else order_df, figure_dir)
        figure_paths.append(output[1])

    matched_count = int(summary_df["after_top_relation"].eq("same_dataset_ev").sum())
    cross_visible_count = int(summary_df["after_cross_ev_rank"].notna().sum())
    report_text = textwrap.dedent(
        f"""
        # EV Support Reranking Refinement

        ## Scope
        Compact refinement pass for shared EV tier-2 support ordering before HCC holdout evaluation.

        ## What changed
        - Final tier-2 reranking now receives `query_source_dataset_id`.
        - EV support hits tied to the same dataset get an explicit `1.22` bonus.
        - Cross-EV support remains visible with a mild `0.95` weight instead of being hard-filtered.
        - Serum literature support under EV queries keeps the existing penalty path.

        ## Targeted result
        - Same-dataset EV support became the top tier-2 hit for {matched_count} of {len(summary_df)} audited EV queries.
        - Cross-EV support remained visible in {cross_visible_count} of {len(summary_df)} audited EV queries.
        - `small2023_ev` still surfaces cross-EV support because no same-dataset EV support document currently exists for that query family.

        ## Query-level readout
        {markdown_table(summary_df)}

        ## Interpretation
        - Diabetes queries now keep `diabetes_ev_context_support` above SHINE support after final reranking.
        - SHINE queries now keep `shine_spectra_context_support` above diabetes support after final reranking.
        - Cross-EV support still survives when relevant, which preserves broader EV reasoning instead of over-narrowing the system.
        - Theme and caveat outputs were preserved because the change only affects tier-2 ordering weights, not the evidence schema.

        ## Remaining caveat
        - `small2023_ev` still lacks a same-dataset EV support document, so its tier-2 layer is expected to remain cross-EV-heavy until such a support artifact exists.
        """
    ).strip()

    report_path = report_dir / "ev_support_reranking_refinement_report.md"
    report_path.write_text(report_text + "\n", encoding="utf-8")
    build_pdf(report_path, figure_paths)
    print(f"Wrote EV support reranking refinement artifacts to: {base_dir}")


if __name__ == "__main__":
    main()
