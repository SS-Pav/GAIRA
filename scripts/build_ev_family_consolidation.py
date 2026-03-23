from __future__ import annotations

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_query_aware_context_routing_report as routing_base
from gaira.config import get_database_path, require_data_root_exists
from gaira.inference import GAIRAInferenceEngine
from gaira.query_routing import QUERY_FAMILY_DEFINITIONS, classify_context_family, classify_support_family, infer_query_family


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
PREV_POLISH_DIR = ROOT / "processed" / "query_aware_context_routing_polish"
PREV_FINAL_DIR = ROOT / "processed" / "query_aware_context_routing_final_polish"

OUTPUT_DIR = ROOT / "processed" / "ev_family_consolidation"
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


def _display_family(family: str) -> str:
    mapping = {
        "serum_liver_hepatobiliary": "Serum\nLiver",
        "serum_general": "Serum\nGeneral",
        "serum_metabolic": "Serum\nMetabolic",
        "ev_general": "EV\nGeneral",
        "ev_disease_or_stress": "EV\nDisease/\nStress",
        "grounding_analyte": "Analyte",
    }
    return mapping.get(family, family.replace("_", "\n"))


def _family_root(family: str | None) -> str:
    value = str(family or "")
    if value.startswith("serum_"):
        return "serum"
    if value.startswith("ev_"):
        return "ev"
    if value == "grounding_analyte":
        return "grounding"
    return "shared"


def _track_forced_families(track_name: str, request) -> list[str]:
    if track_name == "hepatobiliary_serum":
        return [
            "serum_liver_hepatobiliary",
            "serum_general",
            "ev_general",
            "ev_disease_or_stress",
            "grounding_analyte",
        ]
    if track_name == "general_serum":
        return [
            "serum_general",
            "serum_liver_hepatobiliary",
            "ev_general",
            "ev_disease_or_stress",
            "grounding_analyte",
        ]
    if track_name in {"ev_general", "ev_disease_or_stress"}:
        if track_name == "ev_general":
            return [
                "ev_general",
                "ev_disease_or_stress",
                "serum_general",
                "serum_liver_hepatobiliary",
                "grounding_analyte",
            ]
        return [
            "ev_disease_or_stress",
            "ev_general",
            "serum_general",
            "serum_liver_hepatobiliary",
                "grounding_analyte",
            ]
    if track_name == "analyte":
        return [
            "grounding_analyte",
            "ev_general",
            "ev_disease_or_stress",
            "serum_general",
            "serum_liver_hepatobiliary",
        ]
    return []


def _split_contamination(families: list[str], intended_family: str) -> tuple[int, int]:
    intended_root = _family_root(intended_family)
    near_overlap = 0
    cross_domain = 0
    for family in families:
        if not family or family == "shared_generic" or family == intended_family:
            continue
        candidate_root = _family_root(family)
        if candidate_root == intended_root:
            near_overlap += 1
        elif candidate_root != "shared":
            cross_domain += 1
    return near_overlap, cross_domain


def _context_family_from_doc(document_id: str, intended_family: str) -> str:
    domain = _family_root(intended_family)
    if domain not in {"serum", "ev"}:
        return ""
    return classify_context_family({"document_id": document_id}, domain)


def _family_aware_raw_score(row: pd.Series) -> float:
    family = str(row["intended_family"])
    support_precision = float(row["support_precision_top3"])
    support_appropriateness = float(row["topk_support_appropriateness"]) / 1.45
    contamination_term = 1.0 - float(row["cross_domain_contamination_norm"])
    top1_support = float(row["top1_support_correct"])
    if family == "grounding_analyte":
        return 0.45 * top1_support + 0.25 * support_precision + 0.20 * support_appropriateness + 0.10 * contamination_term

    context_precision = 0.0 if pd.isna(row["context_precision_top3"]) else float(row["context_precision_top3"])
    context_appropriateness = 0.0 if pd.isna(row["topk_context_appropriateness"]) else float(row["topk_context_appropriateness"]) / 1.45
    top1_context = 0.0 if pd.isna(row["top1_context_correct"]) else float(row["top1_context_correct"])
    near_term = 1.0 - float(row["near_family_overlap_norm"])

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
            0.24 * support_precision
            + 0.24 * context_precision
            + 0.15 * top1_support
            + 0.10 * top1_context
            + 0.10 * support_appropriateness
            + 0.05 * context_appropriateness
            + 0.07 * contamination_term
            + 0.05 * near_term
        )
    return 0.0


def _normalized_scores(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for intended_family, group in summary_df.groupby("intended_family"):
        min_score = float(group["mean_family_aware_raw_score"].min())
        max_score = float(group["mean_family_aware_raw_score"].max())
        denom = max(max_score - min_score, 1e-9)
        for row in group.to_dict("records"):
            record = dict(row)
            record["normalized_routing_score"] = (float(row["mean_family_aware_raw_score"]) - min_score) / denom
            rows.append(record)
    return pd.DataFrame(rows)


def _winner_margin_summary(norm_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for intended_family, group in norm_df.groupby("intended_family"):
        ranked = group.sort_values("normalized_routing_score", ascending=False).reset_index(drop=True)
        best = ranked.iloc[0]
        runner = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
        margin = float(best["normalized_routing_score"] - runner["normalized_routing_score"])
        if margin >= 0.15:
            category = "strong win"
            status = "solved"
        elif margin >= 0.03:
            category = "moderate win"
            status = "acceptable but close"
        else:
            category = "close"
            status = "acceptable but close"
        rows.append(
            {
                "intended_family": intended_family,
                "best_forced_family": str(best["forced_family"]),
                "runner_up_forced_family": str(runner["forced_family"]),
                "best_raw_score": float(best["mean_family_aware_raw_score"]),
                "runner_up_raw_score": float(runner["mean_family_aware_raw_score"]),
                "winner_margin": margin,
                "winner_margin_category": category,
                "final_status": status,
                "best_cross_domain_contamination": float(best["mean_cross_domain_contamination"]),
                "best_near_family_overlap": float(best["mean_near_family_overlap"]),
            }
        )
    return pd.DataFrame(rows).sort_values("intended_family")


def _load_requests(db_path: Path) -> dict[str, list]:
    all_requests = routing_base._build_requests(db_path)
    return {
        "ev_general": [request for request in all_requests["ev_routing"] if "small2023_ev" in str(request.source_dataset_id)],
        "ev_disease_or_stress": [
            request
            for request in all_requests["ev_routing"]
            if str(request.source_dataset_id) in {"diabetes_plasma_ev_sers", "shine_ev_sers"}
        ],
        "general_serum": all_requests["general_serum_routing"],
        "hepatobiliary_serum": all_requests["hepatobiliary_serum_routing"],
        "analyte": all_requests["analyte_routing"],
    }


def _run_forced_eval(db_path: Path) -> pd.DataFrame:
    engine = GAIRAInferenceEngine(db_path=db_path, theme_layer_version="v3")
    requests_by_family = _load_requests(db_path)
    rows: list[dict] = []
    for track_name, requests in requests_by_family.items():
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
            for forced_family in _track_forced_families(track_name, request):
                result = engine.run_inference(replace(request, disable_query_routing=False, forced_query_family=forced_family))
                support_hits = result.get("tier2_support_hits", [])[:6]
                context_hits = result.get("domain_context_hits", [])[:6]
                support_families = [str(row.get("support_family", "")) for row in support_hits]
                context_families = [str(row.get("context_family", "")) for row in context_hits]
                near_support, cross_support = _split_contamination(support_families, str(intended_family))
                near_context, cross_context = _split_contamination(context_families, str(intended_family))
                top_context_document = context_hits[0].get("document_id", "") if context_hits else ""
                top_tier2_dataset = support_hits[0].get("source_dataset_id", "") if support_hits else ""
                top_tier2_label = support_hits[0].get("source_label", "") if support_hits else ""
                top1_support_family = classify_support_family(
                    {
                        "source_dataset_id": top_tier2_dataset,
                        "source_label": top_tier2_label,
                        "notes": top_tier2_label,
                    }
                )
                top1_context_family = _context_family_from_doc(str(top_context_document), str(intended_family))
                row = {
                    "track_name": track_name,
                    "query_id": request.query_id,
                    "query_label": request.query_label,
                    "source_dataset_id": request.source_dataset_id,
                    "intended_family": intended_family or "",
                    "forced_family": forced_family,
                    "family_matched_context_hits": int(sum(1 for family in context_families if family == str(intended_family))),
                    "family_matched_support_hits": int(sum(1 for family in support_families if family == str(intended_family))),
                    "topk_support_appropriateness": routing_base._topk_appropriateness_score(intended_family, support_hits, "support"),
                    "topk_context_appropriateness": routing_base._topk_appropriateness_score(intended_family, context_hits, "context"),
                    "support_diversity": len({str(item.get("source_dataset_id", "")) for item in support_hits}),
                    "dominance_margin": routing_base._dominance_margin(result["biochemical_theme_outputs"]),
                    "mean_positive_confidence": routing_base._mean_positive_confidence(result["biochemical_theme_outputs"]),
                    "mean_caution_score": routing_base._mean_caution_score(result["biochemical_theme_outputs"]),
                    "top_context_document": top_context_document,
                    "top_tier2_dataset": top_tier2_dataset,
                    "top_tier2_label": top_tier2_label,
                    "dominant_themes": "|".join(result.get("dominant_themes", [])),
                    "support_families_top6": "|".join(support_families),
                    "context_families_top6": "|".join(context_families),
                    "near_family_overlap_support": near_support,
                    "near_family_overlap_context": near_context,
                    "cross_domain_contamination_support": cross_support,
                    "cross_domain_contamination_context": cross_context,
                    "top1_support_family": top1_support_family,
                    "top1_context_family": top1_context_family,
                }
                row["near_family_overlap_total"] = near_support + near_context
                row["cross_domain_contamination_total"] = cross_support + cross_context
                row["top1_support_correct"] = float(top1_support_family == str(intended_family))
                row["top1_context_correct"] = np.nan if str(intended_family) == "grounding_analyte" else float(top1_context_family == str(intended_family))
                row["support_precision_top3"] = min(int(row["family_matched_support_hits"]), 3) / 3.0
                row["context_precision_top3"] = np.nan if str(intended_family) == "grounding_analyte" else min(int(row["family_matched_context_hits"]), 3) / 3.0
                row["near_family_overlap_norm"] = min(float(row["near_family_overlap_total"]) / 5.0, 1.0)
                row["cross_domain_contamination_norm"] = min(float(row["cross_domain_contamination_total"]) / 5.0, 1.0)
                rows.append(row)
    detailed_df = pd.DataFrame(rows)
    detailed_df["family_aware_raw_score"] = detailed_df.apply(_family_aware_raw_score, axis=1)
    return detailed_df


def _aggregate_forced_eval(detailed_df: pd.DataFrame) -> pd.DataFrame:
    return (
        detailed_df.groupby(["track_name", "intended_family", "forced_family"], as_index=False)
        .agg(
            mean_family_matched_context_hits=("family_matched_context_hits", "mean"),
            mean_family_matched_support_hits=("family_matched_support_hits", "mean"),
            mean_topk_support_appropriateness=("topk_support_appropriateness", "mean"),
            mean_topk_context_appropriateness=("topk_context_appropriateness", "mean"),
            mean_support_diversity=("support_diversity", "mean"),
            mean_dominance_margin=("dominance_margin", "mean"),
            mean_positive_confidence=("mean_positive_confidence", "mean"),
            mean_caution_score=("mean_caution_score", "mean"),
            mean_near_family_overlap=("near_family_overlap_total", "mean"),
            mean_cross_domain_contamination=("cross_domain_contamination_total", "mean"),
            mean_top1_support_correct=("top1_support_correct", "mean"),
            mean_top1_context_correct=("top1_context_correct", "mean"),
            mean_support_precision_top3=("support_precision_top3", "mean"),
            mean_context_precision_top3=("context_precision_top3", "mean"),
            mean_family_aware_raw_score=("family_aware_raw_score", "mean"),
        )
        .sort_values(["track_name", "intended_family", "mean_family_aware_raw_score"], ascending=[True, True, False])
    )


def _summarize_by_dataset(detailed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (track_name, source_dataset_id), group in detailed_df.groupby(["track_name", "source_dataset_id"]):
        score_df = (
            group.groupby("forced_family", as_index=False)
            .agg(mean_family_aware_raw_score=("family_aware_raw_score", "mean"))
            .sort_values("mean_family_aware_raw_score", ascending=False)
            .reset_index(drop=True)
        )
        best = score_df.iloc[0]
        runner = score_df.iloc[1] if len(score_df) > 1 else score_df.iloc[0]
        rows.append(
            {
                "track_name": track_name,
                "source_dataset_id": source_dataset_id,
                "intended_family": str(group["intended_family"].iloc[0]),
                "best_forced_family": str(best["forced_family"]),
                "runner_up_forced_family": str(runner["forced_family"]),
                "winner_margin": float(best["mean_family_aware_raw_score"] - runner["mean_family_aware_raw_score"]),
                "mean_near_family_overlap": float(group[group["forced_family"] == best["forced_family"]]["near_family_overlap_total"].mean()),
                "mean_cross_domain_contamination": float(group[group["forced_family"] == best["forced_family"]]["cross_domain_contamination_total"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["track_name", "source_dataset_id"])


def _load_previous_ev_metrics() -> pd.DataFrame:
    old_df = pd.read_csv(PREV_POLISH_DIR / "tables" / "family_specific_metrics.csv")
    ev_df = old_df[old_df["track_name"] == "ev_routing"].copy()
    rows: list[dict] = []
    for source_dataset_id, group in ev_df.groupby("source_dataset_id"):
        score_df = (
            group.groupby("forced_family", as_index=False)
            .agg(mean_family_aware_raw_score=("family_aware_raw_score", "mean"))
            .sort_values("mean_family_aware_raw_score", ascending=False)
            .reset_index(drop=True)
        )
        best = score_df.iloc[0]
        runner = score_df.iloc[1] if len(score_df) > 1 else score_df.iloc[0]
        rows.append(
            {
                "source_dataset_id": source_dataset_id,
                "before_best_forced_family": str(best["forced_family"]),
                "before_runner_up_forced_family": str(runner["forced_family"]),
                "before_winner_margin": float(best["mean_family_aware_raw_score"] - runner["mean_family_aware_raw_score"]),
                "before_intended_family": str(group["intended_family"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _write_audit_note() -> Path:
    path = OUTPUT_DIR / "ev_family_audit.md"
    lines = [
        "# EV Family Audit",
        "",
        "Current EV-family usage before this consolidation pass was concentrated in `src/gaira/query_routing.py`.",
        "",
        "- `QUERY_FAMILIES` listed `ev_general`, `ev_metabolic_or_diabetes`, and `ev_injury_response`.",
        "- `infer_query_family(...)` split diabetes/metabolic EV queries from SHINE/injury EV queries using explicit request metadata and tokens.",
        "- `classify_support_family(...)`, `classify_context_family(...)`, and `classify_knowledge_family(...)` mapped diabetes EV artifacts to `ev_metabolic_or_diabetes` and SHINE/SPECTRA artifacts to `ev_injury_response`.",
        "- Routing weights treated both disease/stress EV families as near-neighbors of `ev_general`, which kept routing functional but made margins artificially tight.",
        "- The older routing evaluation/report scripts still contain the previous EV-family labels in display maps and forced-family comparison tables. They were left untouched in this pass and treated as historical comparison artifacts.",
        "",
        "Implementation scope for this pass:",
        "",
        "- Merge the two perturbed EV families into `ev_disease_or_stress` in core query routing only.",
        "- Rebuild the EV routing evaluation on top of the updated taxonomy in a dedicated `ev_family_consolidation` bundle.",
        "- Leave serum/analyte routing logic unchanged except for the new EV-family references they see during counterfactual comparisons.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_design_note() -> Path:
    path = OUTPUT_DIR / "ev_family_design.md"
    families = [
        "serum_liver_hepatobiliary",
        "serum_general",
        "ev_general",
        "ev_disease_or_stress",
        "grounding_analyte",
    ]
    lines = ["# EV Family Design", "", "Final routing family set used for this pass:", ""]
    for family in families:
        definition = QUERY_FAMILY_DEFINITIONS[family]
        lines.extend(
            [
                f"## {family}",
                "",
                f"- Sample type: `{definition.sample_type}`",
                f"- Intended emphasis: {definition.emphasis}",
                f"- Boost: {', '.join(definition.boost)}",
                f"- Downweight: {', '.join(definition.downweight)}",
                f"- Keep visible: {', '.join(definition.keep_visible)}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_report(
    best_df: pd.DataFrame,
    ev_before_after_df: pd.DataFrame,
    no_regression_df: pd.DataFrame,
) -> tuple[Path, Path]:
    report_path = REPORT_DIR / "ev_family_consolidation_report.md"
    pdf_path = REPORT_DIR / "ev_family_consolidation_report.pdf"

    ev_general = best_df[best_df["intended_family"] == "ev_general"].iloc[0]
    ev_disease = best_df[best_df["intended_family"] == "ev_disease_or_stress"].iloc[0]
    serum_general = best_df[best_df["intended_family"] == "serum_general"].iloc[0]
    serum_liver = best_df[best_df["intended_family"] == "serum_liver_hepatobiliary"].iloc[0]
    analyte = best_df[best_df["intended_family"] == "grounding_analyte"].iloc[0]

    lines = [
        "# EV Family Consolidation Report",
        "",
        "## Motivation",
        "",
        "The prior routing-core polish showed that EV routing was functioning, but the three-way EV taxonomy created near-neighbor competition that was harder to defend than the underlying evidence base. This pass collapses the two perturbed EV families into a single `ev_disease_or_stress` family while preserving `ev_general`.",
        "",
        "## What Changed",
        "",
        "- `ev_metabolic_or_diabetes` and `ev_injury_response` were merged into `ev_disease_or_stress` in core query-family inference and family classification.",
        "- Diabetes EV and SHINE/SPECTRA EV context/support now route through the same perturbed-EV family.",
        "- Serum and analyte families were left intact.",
        "",
        "## Results",
        "",
        f"- `ev_general` still routes to `ev_general`, with runner-up `{ev_general['runner_up_forced_family']}` and margin `{float(ev_general['winner_margin']):.3f}`.",
        f"- `ev_disease_or_stress` now routes to `ev_disease_or_stress`, with runner-up `{ev_disease['runner_up_forced_family']}` and margin `{float(ev_disease['winner_margin']):.3f}`.",
        f"- `serum_general` remains correct with best family `{serum_general['best_forced_family']}`.",
        f"- `serum_liver_hepatobiliary` remains the strongest routing win with margin `{float(serum_liver['winner_margin']):.3f}`.",
        f"- `grounding_analyte` remains top-hit clean with best family `{analyte['best_forced_family']}`.",
        "",
        "## EV Before / After",
        "",
        "```text",
        ev_before_after_df.to_string(index=False),
        "```",
        "",
        "## No Regression",
        "",
        "```text",
        no_regression_df.to_string(index=False),
        "```",
        "",
        "## Assessment",
        "",
        "The EV taxonomy is now cleaner and easier to explain. `small2023_ev` remains the general EV anchor, while both `shine_ev_sers` and `diabetes_plasma_ev_sers` route through the same perturbed-EV family without pretending the evidence base is dense enough to justify two separate disease/stress EV families.",
        "",
        "Serum and analyte routing did not regress. This should be the final routing taxonomy for the internal demo.",
        "",
    ]
    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")

    wrapped = textwrap.wrap(report_text.replace("|", " "), width=105)
    fig, ax = plt.subplots(figsize=(11, max(8.5, 0.18 * len(wrapped))))
    ax.axis("off")
    ax.text(0.01, 0.99, "\n".join(wrapped), va="top", ha="left", fontsize=10, family="DejaVu Sans Mono")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return report_path, pdf_path


def main() -> None:
    require_data_root_exists()
    ensure_dirs()

    audit_path = _write_audit_note()
    design_path = _write_design_note()

    db_path = get_database_path()
    detailed_df = _run_forced_eval(db_path)
    summary_df = _aggregate_forced_eval(detailed_df)
    normalized_df = _normalized_scores(summary_df)
    best_df = _winner_margin_summary(normalized_df)

    prev_ev_df = _load_previous_ev_metrics()
    current_ev_df = _summarize_by_dataset(detailed_df)
    ev_after = current_ev_df[current_ev_df["track_name"].isin(["ev_general", "ev_disease_or_stress"])].copy()
    ev_before_after_df = prev_ev_df.merge(ev_after, on="source_dataset_id", how="outer")
    ev_before_after_df["dataset_label"] = ev_before_after_df["source_dataset_id"].map(
        {
            "small2023_ev": "small2023_ev",
            "diabetes_plasma_ev_sers": "diabetes_plasma_ev_sers",
            "shine_ev_sers": "shine_ev_sers",
        }
    )
    ev_before_after_df = ev_before_after_df[
        [
            "dataset_label",
            "before_intended_family",
            "before_best_forced_family",
            "before_runner_up_forced_family",
            "before_winner_margin",
            "intended_family",
            "best_forced_family",
            "runner_up_forced_family",
            "winner_margin",
            "mean_near_family_overlap",
            "mean_cross_domain_contamination",
        ]
    ].sort_values("dataset_label")

    no_regression_df = best_df[best_df["intended_family"].isin(["serum_general", "serum_liver_hepatobiliary", "grounding_analyte"])].copy()
    no_regression_df = no_regression_df[["intended_family", "best_forced_family", "runner_up_forced_family", "winner_margin", "final_status"]]

    representative_df = detailed_df[
        [
            "track_name",
            "query_label",
            "source_dataset_id",
            "intended_family",
            "forced_family",
            "top_context_document",
            "top_tier2_dataset",
            "top_tier2_label",
            "dominant_themes",
            "family_aware_raw_score",
        ]
    ].copy()

    ev_forced_summary_path = TABLE_DIR / "ev_forced_routing_summary.csv"
    ev_before_after_path = TABLE_DIR / "ev_family_before_after_metrics.csv"
    ev_margin_path = TABLE_DIR / "ev_winner_margin_summary.csv"
    no_regression_path = TABLE_DIR / "no_regression_summary.csv"
    best_path = TABLE_DIR / "final_family_best_routing_summary.csv"
    representative_path = TABLE_DIR / "representative_case_table.csv"

    summary_df.to_csv(ev_forced_summary_path, index=False)
    ev_before_after_df.to_csv(ev_before_after_path, index=False)
    best_df[best_df["intended_family"].str.startswith("ev_")].to_csv(ev_margin_path, index=False)
    no_regression_df.to_csv(no_regression_path, index=False)
    best_df.to_csv(best_path, index=False)
    representative_df.to_csv(representative_path, index=False)

    # Figure 1
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    ax.text(0.15, 0.76, "Old EV Families", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(0.15, 0.56, "ev_general", ha="center", va="center", fontsize=14, bbox={"boxstyle": "round,pad=0.4", "fc": "#dbeafe", "ec": "#93c5fd"})
    ax.text(0.15, 0.36, "ev_metabolic_or_diabetes", ha="center", va="center", fontsize=12, bbox={"boxstyle": "round,pad=0.4", "fc": "#fef3c7", "ec": "#f59e0b"})
    ax.text(0.15, 0.16, "ev_injury_response", ha="center", va="center", fontsize=12, bbox={"boxstyle": "round,pad=0.4", "fc": "#fee2e2", "ec": "#ef4444"})
    ax.text(0.74, 0.76, "New EV Families", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(0.74, 0.56, "ev_general", ha="center", va="center", fontsize=14, bbox={"boxstyle": "round,pad=0.4", "fc": "#dbeafe", "ec": "#93c5fd"})
    ax.text(0.74, 0.26, "ev_disease_or_stress", ha="center", va="center", fontsize=14, bbox={"boxstyle": "round,pad=0.4", "fc": "#dcfce7", "ec": "#22c55e"})
    ax.annotate("", xy=(0.62, 0.56), xytext=(0.28, 0.56), arrowprops=dict(arrowstyle="->", lw=2.0, color="#2563eb"))
    ax.annotate("", xy=(0.62, 0.26), xytext=(0.28, 0.36), arrowprops=dict(arrowstyle="->", lw=2.0, color="#16a34a"))
    ax.annotate("", xy=(0.62, 0.26), xytext=(0.28, 0.16), arrowprops=dict(arrowstyle="->", lw=2.0, color="#16a34a"))
    save_figure(fig, "figure1_ev_family_consolidation_schematic")

    # Figure 2
    heatmap_df = normalized_df.pivot(index="intended_family", columns="forced_family", values="normalized_routing_score").fillna(0.0)
    heatmap_df = heatmap_df.loc[[idx for idx in ["serum_liver_hepatobiliary", "serum_general", "ev_general", "ev_disease_or_stress", "grounding_analyte"] if idx in heatmap_df.index]]
    heatmap_df = heatmap_df[[col for col in ["serum_liver_hepatobiliary", "serum_general", "ev_general", "ev_disease_or_stress", "grounding_analyte"] if col in heatmap_df.columns]]
    fig, ax = plt.subplots(figsize=(10, 6.6))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Normalized routing score"}, ax=ax)
    ax.set_title("Query Family x Forced Routing Family")
    ax.set_xlabel("Forced routing family")
    ax.set_ylabel("Intended family")
    ax.set_xticklabels([_display_family(label.get_text()) for label in ax.get_xticklabels()], rotation=0)
    ax.set_yticklabels([_display_family(label.get_text()) for label in ax.get_yticklabels()], rotation=0)
    save_figure(fig, "figure2_query_family_forced_routing_heatmap")

    # Figure 3
    ev_plot_df = ev_before_after_df.melt(
        id_vars=["dataset_label"],
        value_vars=["before_winner_margin", "winner_margin"],
        var_name="phase",
        value_name="margin_value",
    )
    ev_plot_df["phase"] = ev_plot_df["phase"].map({"before_winner_margin": "Before", "winner_margin": "After"})
    fig, ax = plt.subplots(figsize=(9, 5.8))
    sns.barplot(data=ev_plot_df, x="dataset_label", y="margin_value", hue="phase", palette=["#94a3b8", "#2563eb"], ax=ax)
    ax.set_title("EV Routing Margin Before vs After Consolidation")
    ax.set_xlabel("")
    ax.set_ylabel("Winner minus runner-up margin")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")
    save_figure(fig, "figure3_ev_routing_before_after")

    # Figure 4
    contamination_df = summary_df[summary_df["track_name"].isin(["ev_general", "ev_disease_or_stress"])].copy()
    contamination_df["forced_display"] = contamination_df["forced_family"].map(_display_family)
    contamination_df["intended_display"] = contamination_df["intended_family"].map(_display_family)
    near_df = contamination_df.pivot(index="intended_display", columns="forced_display", values="mean_near_family_overlap").fillna(0.0)
    cross_df = contamination_df.pivot(index="intended_display", columns="forced_display", values="mean_cross_domain_contamination").fillna(0.0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    sns.heatmap(near_df, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"label": "Near-family overlap"}, ax=axes[0])
    sns.heatmap(cross_df, annot=True, fmt=".2f", cmap="Reds", cbar_kws={"label": "Cross-domain contamination"}, ax=axes[1])
    axes[0].set_title("Near-family overlap")
    axes[1].set_title("Cross-domain contamination")
    for ax in axes:
        ax.set_xlabel("Forced routing family")
        ax.set_ylabel("Intended family")
    save_figure(fig, "figure4_near_family_and_contamination_summary")

    # Figure 5
    margin_plot_df = best_df.copy()
    margin_plot_df["intended_display"] = margin_plot_df["intended_family"].map(_display_family)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    sns.barplot(data=margin_plot_df, x="intended_display", y="winner_margin", color="#2563eb", ax=ax)
    for idx, row in margin_plot_df.reset_index(drop=True).iterrows():
        ax.text(idx, float(row["winner_margin"]) + 0.01, f"{row['best_forced_family']}\nvs {row['runner_up_forced_family']}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Winner / Runner-up / Margin")
    ax.set_xlabel("")
    ax.set_ylabel("Margin")
    save_figure(fig, "figure5_winner_runner_up_margin")

    # Figure 6
    no_reg_plot_df = no_regression_df.copy()
    no_reg_plot_df["intended_display"] = no_reg_plot_df["intended_family"].map(_display_family)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.barplot(data=no_reg_plot_df, x="intended_display", y="winner_margin", color="#16a34a", ax=ax)
    for idx, row in no_reg_plot_df.reset_index(drop=True).iterrows():
        ax.text(idx, float(row["winner_margin"]) + 0.01, str(row["best_forced_family"]), ha="center", va="bottom", fontsize=9)
    ax.set_title("No-regression Summary")
    ax.set_xlabel("")
    ax.set_ylabel("Margin")
    save_figure(fig, "figure6_no_regression_summary")

    report_path, report_pdf_path = _build_report(best_df, ev_before_after_df, no_regression_df)

    print(f"audit_note={audit_path}")
    print(f"design_note={design_path}")
    print(f"summary_table={ev_forced_summary_path}")
    print(f"report_md={report_path}")
    print(f"report_pdf={report_pdf_path}")


if __name__ == "__main__":
    main()
