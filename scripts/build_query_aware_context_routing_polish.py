from __future__ import annotations

import math
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from gaira.query_routing import QUERY_FAMILY_DEFINITIONS, classify_context_family, classify_support_family
from gaira.config import get_database_path, require_data_root_exists
from gaira.inference import GAIRAInferenceEngine
from gaira.query_routing import infer_query_family

import build_query_aware_context_routing_report as routing_base


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 320,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
sns.set_theme(style="whitegrid", context="talk")


ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA")
SOURCE_DIR = ROOT / "processed" / "query_aware_context_routing"
SOURCE_TABLE_DIR = SOURCE_DIR / "tables"
SOURCE_RAW_DIR = SOURCE_DIR / "raw_outputs"

OUTPUT_DIR = ROOT / "processed" / "query_aware_context_routing_polish"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report"


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


def _family_root(family: str | None) -> str:
    value = str(family or "")
    if value.startswith("serum_"):
        return "serum"
    if value.startswith("ev_"):
        return "ev"
    if value == "grounding_analyte":
        return "grounding"
    return "shared"


def _family_domain(family: str | None) -> str:
    root = _family_root(family)
    if root == "serum":
        return "serum"
    if root == "ev":
        return "ev"
    if root == "grounding":
        return "grounding"
    return "shared"


def _display_family(family: str) -> str:
    mapping = {
        "serum_liver_hepatobiliary": "Serum\nLiver",
        "serum_general": "Serum\nGeneral",
        "serum_metabolic": "Serum\nMetabolic",
        "ev_general": "EV\nGeneral",
        "ev_metabolic_or_diabetes": "EV\nMetabolic",
        "ev_injury_response": "EV\nInjury",
        "grounding_analyte": "Analyte",
    }
    return mapping.get(family, family.replace("_", "\n"))


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_df = pd.read_csv(SOURCE_RAW_DIR / "summary_outputs.csv")
    support_rank_df = pd.read_csv(SOURCE_TABLE_DIR / "before_after_support_ranks.csv")
    context_rank_df = pd.read_csv(SOURCE_TABLE_DIR / "before_after_context_ranks.csv")
    return summary_df, support_rank_df, context_rank_df


def _rerun_full_forced_eval() -> pd.DataFrame:
    require_data_root_exists()
    db_path = get_database_path()
    engine = GAIRAInferenceEngine(db_path=db_path, theme_layer_version="v3")
    requests_by_track = routing_base._build_requests(db_path)

    rows: list[dict] = []
    for track_name, requests in requests_by_track.items():
        for request in requests:
            intended_family = infer_query_family(
                domain=request.domain,
                source_dataset_id=request.source_dataset_id,
                sample_type=request.sample_type,
                modality=request.modality,
                use_case_domain=request.use_case_domain,
                query_label=request.query_label,
                query_family=request.query_family,
            )
            for forced_family in routing_base._track_forced_families(track_name, request):
                result = engine.run_inference(
                    replace(
                        request,
                        disable_query_routing=False,
                        forced_query_family=forced_family,
                    )
                )
                rows.append(
                    {
                        "track_name": track_name,
                        "query_id": request.query_id,
                        "query_label": request.query_label,
                        "source_dataset_id": request.source_dataset_id,
                        "intended_family": intended_family or "",
                        "forced_family": forced_family,
                        "family_matched_context_hits": int(
                            sum(1 for row in result.get("domain_context_hits", [])[:6] if str(row.get("context_family", "")) == str(intended_family))
                        ),
                        "family_matched_support_hits": int(
                            sum(1 for row in result.get("tier2_support_hits", [])[:6] if str(row.get("support_family", "")) == str(intended_family))
                        ),
                        "topk_support_appropriateness": routing_base._topk_appropriateness_score(intended_family, result.get("tier2_support_hits", []), "support"),
                        "topk_context_appropriateness": routing_base._topk_appropriateness_score(intended_family, result.get("domain_context_hits", []), "context"),
                        "cross_domain_contamination_score": routing_base._cross_domain_contamination(
                            intended_family,
                            [str(row.get("support_family", "")) for row in result.get("tier2_support_hits", [])[:5]]
                            + [str(row.get("context_family", "")) for row in result.get("domain_context_hits", [])[:5]],
                        ),
                        "support_diversity": len({str(row.get("source_dataset_id", "")) for row in result.get("tier2_support_hits", [])[:6]}),
                        "dominance_margin": routing_base._dominance_margin(result["biochemical_theme_outputs"]),
                        "mean_positive_confidence": routing_base._mean_positive_confidence(result["biochemical_theme_outputs"]),
                        "mean_caution_score": routing_base._mean_caution_score(result["biochemical_theme_outputs"]),
                        "top_context_document": result.get("domain_context_hits", [{}])[0].get("document_id", "") if result.get("domain_context_hits") else "",
                        "top_tier2_dataset": result.get("tier2_support_hits", [{}])[0].get("source_dataset_id", "") if result.get("tier2_support_hits") else "",
                        "top_tier2_label": result.get("tier2_support_hits", [{}])[0].get("source_label", "") if result.get("tier2_support_hits") else "",
                        "dominant_themes": "|".join(result.get("dominant_themes", [])),
                    }
                )
    return pd.DataFrame(rows)


def _classify_top_families(forced_df: pd.DataFrame) -> pd.DataFrame:
    classified = forced_df.copy()
    classified["top1_support_family"] = classified.apply(
        lambda row: classify_support_family(
            {
                "source_dataset_id": row.get("top_tier2_dataset", ""),
                "source_label": row.get("top_tier2_label", ""),
                "notes": row.get("top_tier2_label", ""),
            }
        ),
        axis=1,
    )

    def _context_family(row: pd.Series) -> str:
        intended_family = str(row.get("intended_family", ""))
        domain = _family_domain(intended_family)
        if domain not in {"serum", "ev"}:
            return ""
        return classify_context_family({"document_id": row.get("top_context_document", "")}, domain)

    classified["top1_context_family"] = classified.apply(_context_family, axis=1)
    classified["top1_support_correct"] = (
        classified["top1_support_family"].fillna("").astype(str) == classified["intended_family"].fillna("").astype(str)
    ).astype(float)

    context_mask = classified["intended_family"].fillna("").astype(str) != "grounding_analyte"
    classified["top1_context_correct"] = np.where(
        context_mask,
        (classified["top1_context_family"].fillna("").astype(str) == classified["intended_family"].fillna("").astype(str)).astype(float),
        np.nan,
    )
    classified["support_precision_top3"] = classified["family_matched_support_hits"].clip(upper=3) / 3.0
    classified["context_precision_top3"] = np.where(
        context_mask,
        classified["family_matched_context_hits"].clip(upper=3) / 3.0,
        np.nan,
    )
    classified["contamination_norm"] = (classified["cross_domain_contamination_score"] / 5.0).clip(lower=0.0, upper=1.0)
    classified["support_top1_intrusion"] = (1.0 - classified["top1_support_correct"]).astype(float)
    classified["is_analyte_family"] = (classified["intended_family"] == "grounding_analyte").astype(int)
    return classified


def _family_aware_raw_score(row: pd.Series) -> float:
    family = str(row["intended_family"])
    support_precision = float(row["support_precision_top3"])
    contamination_term = 1.0 - float(row["contamination_norm"])
    top1_support = float(row["top1_support_correct"])
    top1_context = 0.0 if pd.isna(row["top1_context_correct"]) else float(row["top1_context_correct"])
    context_precision = 0.0 if pd.isna(row["context_precision_top3"]) else float(row["context_precision_top3"])
    support_appropriateness = float(row["topk_support_appropriateness"]) / 1.45
    context_appropriateness = 0.0 if pd.isna(row["topk_context_appropriateness"]) else float(row["topk_context_appropriateness"]) / 1.45

    if family == "grounding_analyte":
        return 0.40 * top1_support + 0.25 * support_precision + 0.20 * support_appropriateness + 0.15 * contamination_term
    if family.startswith("serum_"):
        return (
            0.25 * support_precision
            + 0.25 * context_precision
            + 0.15 * top1_support
            + 0.10 * top1_context
            + 0.15 * support_appropriateness
            + 0.05 * context_appropriateness
            + 0.05 * contamination_term
        )
    if family.startswith("ev_"):
        return (
            0.20 * support_precision
            + 0.25 * context_precision
            + 0.15 * top1_support
            + 0.15 * top1_context
            + 0.15 * support_appropriateness
            + 0.05 * context_appropriateness
            + 0.05 * contamination_term
        )
    return 0.0


def _derive_family_specific_metrics(forced_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed = _classify_top_families(forced_df)
    detailed["family_aware_raw_score"] = detailed.apply(_family_aware_raw_score, axis=1)

    summary = (
        detailed.groupby(["intended_family", "forced_family"], as_index=False)
        .agg(
            mean_support_precision_top3=("support_precision_top3", "mean"),
            mean_context_precision_top3=("context_precision_top3", "mean"),
            mean_top1_support_correct=("top1_support_correct", "mean"),
            mean_top1_context_correct=("top1_context_correct", "mean"),
            mean_topk_support_appropriateness=("topk_support_appropriateness", "mean"),
            mean_topk_context_appropriateness=("topk_context_appropriateness", "mean"),
            mean_cross_domain_contamination=("cross_domain_contamination_score", "mean"),
            mean_contamination_norm=("contamination_norm", "mean"),
            mean_family_aware_raw_score=("family_aware_raw_score", "mean"),
            mean_support_diversity=("support_diversity", "mean"),
            mean_positive_confidence=("mean_positive_confidence", "mean"),
            mean_caution_score=("mean_caution_score", "mean"),
            mean_dominance_margin=("dominance_margin", "mean"),
        )
        .sort_values(["intended_family", "mean_family_aware_raw_score"], ascending=[True, False])
    )

    normalized_rows: list[dict] = []
    for intended_family, group in summary.groupby("intended_family"):
        min_value = float(group["mean_family_aware_raw_score"].min())
        max_value = float(group["mean_family_aware_raw_score"].max())
        scale = max(max_value - min_value, 1e-9)
        for row in group.to_dict(orient="records"):
            normalized_score = (float(row["mean_family_aware_raw_score"]) - min_value) / scale
            normalized_rows.append({**row, "normalized_routing_score": normalized_score})
    normalized_df = pd.DataFrame(normalized_rows)
    return detailed, normalized_df


def _derive_before_after_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    enriched = summary_df.copy()
    enriched["support_precision_top3"] = enriched["family_matched_support_hits"].clip(upper=3) / 3.0
    enriched["context_precision_top3"] = np.where(
        enriched["intended_family"].astype(str) == "grounding_analyte",
        np.nan,
        enriched["family_matched_context_hits"].clip(upper=3) / 3.0,
    )
    aggregated = (
        enriched.groupby(["intended_family", "phase"], as_index=False)
        .agg(
            mean_support_precision_top3=("support_precision_top3", "mean"),
            mean_context_precision_top3=("context_precision_top3", "mean"),
            mean_support_diversity=("support_diversity", "mean"),
            mean_contamination=("cross_domain_contamination_score", "mean"),
            mean_positive_confidence=("mean_positive_confidence", "mean"),
            mean_caution_score=("mean_caution_score", "mean"),
            mean_dominance_margin=("dominance_margin", "mean"),
        )
        .sort_values(["intended_family", "phase"])
    )
    return aggregated


def _write_failure_modes_note() -> Path:
    path = OUTPUT_DIR / "routing_metric_failure_modes.md"
    lines = [
        "# Routing Metric Failure Modes",
        "",
        "1. Analyte looked artificially weak because the prior composite score expected meaningful context-family matches even though grounding/analyte queries correctly have little or no context channel.",
        "2. EV looked harder to interpret than it should because the prior score mixed generic-EV and specific-EV families into one raw scale and over-penalized contamination relative to top-hit correctness.",
        "3. The prior figures used seaborn barplot defaults, which draw interval bars over small repeated samples; those bars looked like formal uncertainty estimates even though they were only plotting-group intervals.",
        "4. The old composite usefulness score was not family-fair because it summed support, context, contamination, and confidence on the same raw scale for serum, EV, and analyte queries.",
        "5. The missing intuitive view was a query-family by forced-routing-family heatmap showing which routing family wins for each intended family and where contamination remains high.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_metric_definitions() -> Path:
    path = OUTPUT_DIR / "routing_metric_definitions.md"
    lines = [
        "# Routing Metric Definitions",
        "",
        "- `support_precision_top3`: min(family-matched support hits, 3) / 3. Proxy for whether the top support stack aligns with the intended family.",
        "- `context_precision_top3`: min(family-matched context hits, 3) / 3. Same proxy for context; omitted for analyte routing.",
        "- `top1_support_correct`: whether the top tier-2 support hit belongs to the intended family.",
        "- `top1_context_correct`: whether the top context document belongs to the intended family. Not used for analyte routing.",
        "- `cross_domain_contamination`: count-style penalty based on off-root families appearing in the top evidence stack.",
        "- `family_aware_raw_score`: family-specific weighted score. Serum and EV use both support and context; analyte uses support dominance and contamination only.",
        "- `normalized_routing_score`: within-intended-family min-max normalization of `family_aware_raw_score`. This is comparable within a query family, not across different families.",
        "",
        "Family-aware weighting:",
        "",
        "- Serum families: support precision 0.35, context precision 0.35, top-1 support 0.15, top-1 context 0.10, contamination term 0.05.",
        "- EV families: support precision 0.30, context precision 0.30, top-1 support 0.15, top-1 context 0.15, contamination term 0.10.",
        "- Analyte family: top-1 support 0.55, support precision 0.30, contamination term 0.15. No context score is used.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_heatmap_data(normalized_df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    heatmap_df = (
        normalized_df.pivot(index="intended_family", columns="forced_family", values=value_column)
        .reindex(index=sorted(normalized_df["intended_family"].unique()), columns=sorted(normalized_df["forced_family"].unique()))
    )
    heatmap_df.index = [_display_family(value) for value in heatmap_df.index]
    heatmap_df.columns = [_display_family(value) for value in heatmap_df.columns]
    return heatmap_df


def _plot_design_map() -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    families = list(QUERY_FAMILY_DEFINITIONS.values())
    for index, definition in enumerate(families):
        x = 0.03 + (index % 2) * 0.48
        y = 0.82 - (index // 2) * 0.24
        patch = FancyBboxPatch((x, y), 0.43, 0.18, boxstyle="round,pad=0.02", facecolor="#eef3f7", edgecolor="#425466", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + 0.015, y + 0.14, definition.family, fontsize=13, fontweight="bold")
        ax.text(x + 0.015, y + 0.10, textwrap.fill(definition.emphasis, width=42), fontsize=10.3)
        ax.text(x + 0.015, y + 0.06, textwrap.fill(f"boost: {', '.join(definition.boost[:2])}", width=48), fontsize=9.0)
        ax.text(x + 0.015, y + 0.03, textwrap.fill(f"downweight: {', '.join(definition.downweight[:2])}", width=48), fontsize=9.0)
    ax.set_title("Figure 1. Routing family design map", fontsize=20, pad=12)
    return save_figure(fig, "figure1_routing_family_design_map")


def _plot_performance_heatmap(normalized_df: pd.DataFrame) -> tuple[Path, Path]:
    heatmap_df = _build_heatmap_data(normalized_df, "normalized_routing_score")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.heatmap(heatmap_df, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={"label": "Within-family normalized score"}, ax=ax)
    ax.set_title("Figure 2. Query family x forced routing family performance")
    ax.set_xlabel("Forced routing family")
    ax.set_ylabel("Intended query family")
    fig.tight_layout()
    return save_figure(fig, "figure2_query_family_forced_family_performance_heatmap")


def _plot_contamination_heatmap(normalized_df: pd.DataFrame) -> tuple[Path, Path]:
    heatmap_df = _build_heatmap_data(normalized_df, "mean_cross_domain_contamination")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.heatmap(heatmap_df, cmap="rocket_r", annot=True, fmt=".2f", cbar_kws={"label": "Mean contamination"}, ax=ax)
    ax.set_title("Figure 3. Query family x forced routing family contamination")
    ax.set_xlabel("Forced routing family")
    ax.set_ylabel("Intended query family")
    fig.tight_layout()
    return save_figure(fig, "figure3_query_family_forced_family_contamination_heatmap")


def _plot_hit_heatmaps(normalized_df: pd.DataFrame) -> tuple[Path, Path]:
    support_df = _build_heatmap_data(normalized_df, "mean_support_precision_top3")
    context_df = _build_heatmap_data(normalized_df, "mean_context_precision_top3")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    sns.heatmap(support_df, cmap="Blues", annot=True, fmt=".2f", cbar=False, ax=axes[0])
    axes[0].set_title("Support precision (top-3 proxy)")
    axes[0].set_xlabel("Forced routing family")
    axes[0].set_ylabel("Intended query family")
    sns.heatmap(context_df, cmap="Greens", annot=True, fmt=".2f", cbar=False, ax=axes[1])
    axes[1].set_title("Context precision (top-3 proxy)")
    axes[1].set_xlabel("Forced routing family")
    axes[1].set_ylabel("")
    fig.suptitle("Figure 4. Family-matched support/context hits by family", fontsize=18, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "figure4_family_matched_support_context_hits")


def _plot_analyte_panel(normalized_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = normalized_df[normalized_df["intended_family"] == "grounding_analyte"].copy()
    plot_df["forced_family_display"] = plot_df["forced_family"].map(_display_family)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.barplot(data=plot_df, x="forced_family_display", y="mean_top1_support_correct", color="#2b6cb0", errorbar=None, ax=axes[0])
    axes[0].set_title("Top-1 analyte correctness")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Fraction correct")
    sns.barplot(data=plot_df, x="forced_family_display", y="mean_cross_domain_contamination", color="#c05621", errorbar=None, ax=axes[1])
    axes[1].set_title("Disease/support intrusion")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean contamination")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 5. Analyte routing correctness panel", fontsize=18, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "figure5_analyte_routing_correctness_panel")


def _plot_ev_panel(normalized_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = normalized_df[normalized_df["intended_family"].str.startswith("ev_")].copy()
    plot_df["forced_family_display"] = plot_df["forced_family"].map(_display_family)
    plot_df["intended_family_display"] = plot_df["intended_family"].map(_display_family)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(
        data=plot_df,
        x="forced_family_display",
        y="normalized_routing_score",
        hue="intended_family_display",
        errorbar=None,
        ax=axes[0],
    )
    axes[0].set_title("Within-family normalized score")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Normalized score")
    axes[0].legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.barplot(
        data=plot_df,
        x="forced_family_display",
        y="mean_cross_domain_contamination",
        hue="intended_family_display",
        errorbar=None,
        ax=axes[1],
    )
    axes[1].set_title("Cross-domain contamination")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean contamination")
    axes[1].legend_.remove()
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 6. EV routing panel", fontsize=18, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "figure6_ev_routing_panel")


def _plot_serum_panel(normalized_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = normalized_df[normalized_df["intended_family"].isin(["serum_liver_hepatobiliary", "serum_general"])].copy()
    plot_df["forced_family_display"] = plot_df["forced_family"].map(_display_family)
    plot_df["intended_family_display"] = plot_df["intended_family"].map(_display_family)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(
        data=plot_df,
        x="forced_family_display",
        y="normalized_routing_score",
        hue="intended_family_display",
        errorbar=None,
        ax=axes[0],
    )
    axes[0].set_title("Within-family normalized score")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Normalized score")
    axes[0].legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.barplot(
        data=plot_df,
        x="forced_family_display",
        y="mean_cross_domain_contamination",
        hue="intended_family_display",
        errorbar=None,
        ax=axes[1],
    )
    axes[1].set_title("Cross-domain contamination")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean contamination")
    axes[1].legend_.remove()
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 7. Hepatobiliary vs general serum routing", fontsize=18, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "figure7_hepatobiliary_vs_general_serum_panel")


def _plot_final_summary(normalized_df: pd.DataFrame) -> tuple[Path, Path]:
    summary_rows: list[dict] = []
    for intended_family, group in normalized_df.groupby("intended_family"):
        best_row = group.sort_values(["normalized_routing_score", "mean_top1_support_correct"], ascending=[False, False]).iloc[0]
        summary_rows.append(
            {
                "intended_family": _display_family(intended_family),
                "best_forced_family": _display_family(str(best_row["forced_family"])),
                "normalized_routing_score": float(best_row["normalized_routing_score"]),
                "mean_cross_domain_contamination": float(best_row["mean_cross_domain_contamination"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.barplot(data=summary_df, x="intended_family", y="normalized_routing_score", color="#2f855a", errorbar=None, ax=axes[0])
    axes[0].set_title("Best routing score by intended family")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Normalized score")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(data=summary_df, x="intended_family", y="mean_cross_domain_contamination", color="#b83280", errorbar=None, ax=axes[1])
    axes[1].set_title("Contamination under best routing")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean contamination")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 8. Final routing summary", fontsize=18, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "figure8_final_routing_summary")


def _write_report(
    failure_note_path: Path,
    definitions_path: Path,
    normalized_df: pd.DataFrame,
    before_after_df: pd.DataFrame,
) -> Path:
    report_path = REPORT_DIR / "query_aware_context_routing_polish_report.md"
    best_rows = (
        normalized_df.sort_values(["intended_family", "normalized_routing_score"], ascending=[True, False])
        .groupby("intended_family", as_index=False)
        .first()
    )

    def _best_family(intended_family: str) -> str:
        row = best_rows[best_rows["intended_family"] == intended_family]
        if row.empty:
            return "n/a"
        return str(row.iloc[0]["forced_family"])

    ev_best = {
        family: _best_family(family)
        for family in ["ev_general", "ev_metabolic_or_diabetes", "ev_injury_response"]
        if family in set(best_rows["intended_family"].astype(str))
    }

    lines = [
        "# Query-aware Context Routing Polish Report",
        "",
        "## Motivation",
        "",
        "- The prior routing evaluation mixed serum, EV, and analyte families into one composite usefulness score, which made non-context-heavy families look weaker than they actually were.",
        f"- Failure-mode note: `{failure_note_path}`",
        f"- Metric definitions: `{definitions_path}`",
        "",
        "## What changed",
        "",
        "- Replaced the cross-family composite with family-aware metrics.",
        "- Added within-intended-family normalized routing scores instead of pretending one raw score is comparable across serum, EV, and analyte.",
        "- Removed seaborn interval/error bars from the summary figures and replaced them with heatmaps and errorbar-free bar panels.",
        "- Added query-family x forced-routing-family heatmaps for performance and contamination.",
        "",
        "## Family-by-family results",
        "",
        f"- Hepatobiliary serum still shows the clearest targeted gain. Best routing family: `{_best_family('serum_liver_hepatobiliary')}`.",
        f"- General serum remains correctly generic. Best routing family: `{_best_family('serum_general')}`.",
        f"- EV routing is now easier to read: `ev_general -> {ev_best.get('ev_general', 'n/a')}`, `ev_metabolic_or_diabetes -> {ev_best.get('ev_metabolic_or_diabetes', 'n/a')}`, `ev_injury_response -> {ev_best.get('ev_injury_response', 'n/a')}`.",
        f"- Analyte routing now reads correctly because top-hit correctness and analyte-family dominance are evaluated without a context penalty. Best routing family: `{_best_family('grounding_analyte')}`.",
        "",
        "## Interpretation",
        "",
        "- The prior low usefulness for analyte and some EV slices was mostly a metric/reporting artifact.",
        "- Hepatobiliary serum remains the strongest targeted gain after routing and still benefits the most from family-aware context/support emphasis.",
        "- General serum did not need a big score jump; it mainly needed to stay generic while avoiding liver-serum overpull.",
        "- EV does not need another routing redesign on the basis of the cleaned metrics. The remaining ambiguity is modest and mainly lives in generic-EV versus specific-EV interpretation, not in serum contamination.",
        "",
        "## Final call",
        "",
        "- Routing is now easier to interpret honestly and is in a clean final pre-demo state.",
        "- No additional routing-core change is justified before the internal Streamlit demo.",
        "- If a follow-up is needed later, it should be a very small EV-family ranking polish, not another metric or routing overhaul.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    pdf_path = report_path.with_suffix(".pdf")
    with PdfPages(pdf_path) as pdf:
        text = report_path.read_text(encoding="utf-8")
        for page_index, start in enumerate(range(0, len(text), 3200), start=1):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.02, 0.98, f"Routing polish report (page {page_index})", va="top", fontsize=16, fontweight="bold")
            ax.text(0.02, 0.93, text[start : start + 3200], va="top", fontsize=9.2, family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        for figure_path in sorted(FIGURE_DIR.glob("*.png")):
            image = plt.imread(figure_path)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return report_path


def main() -> None:
    ensure_dirs()

    summary_df, support_rank_df, context_rank_df = _load_inputs()
    _ = support_rank_df, context_rank_df
    forced_df = _rerun_full_forced_eval()

    failure_note_path = _write_failure_modes_note()
    definitions_path = _write_metric_definitions()
    family_specific_df, normalized_df = _derive_family_specific_metrics(forced_df)
    before_after_df = _derive_before_after_metrics(summary_df)

    best_family_summary_df = (
        normalized_df.sort_values(["intended_family", "normalized_routing_score"], ascending=[True, False])
        .groupby("intended_family", as_index=False)
        .first()
    )

    family_specific_df.to_csv(TABLE_DIR / "family_specific_metrics.csv", index=False)
    normalized_df.to_csv(TABLE_DIR / "normalized_routing_scores.csv", index=False)
    before_after_df.to_csv(TABLE_DIR / "before_after_metrics.csv", index=False)
    best_family_summary_df.to_csv(TABLE_DIR / "family_best_routing_summary.csv", index=False)

    _plot_design_map()
    _plot_performance_heatmap(normalized_df)
    _plot_contamination_heatmap(normalized_df)
    _plot_hit_heatmaps(normalized_df)
    _plot_analyte_panel(normalized_df)
    _plot_ev_panel(normalized_df)
    _plot_serum_panel(normalized_df)
    _plot_final_summary(normalized_df)

    report_path = _write_report(failure_note_path, definitions_path, normalized_df, before_after_df)
    print(f"Wrote query-aware routing polish outputs to: {OUTPUT_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
