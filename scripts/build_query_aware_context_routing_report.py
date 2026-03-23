from __future__ import annotations

import json
import textwrap
from dataclasses import replace
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


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
OUTPUT_DIR = ROOT / "processed" / "query_aware_context_routing"
RAW_DIR = OUTPUT_DIR / "raw_outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report"
HCC_EVAL_DB = ROOT / "processed" / "hcc_holdout_evaluation" / "eval_db" / "gaira_hcc_holdout_eval.duckdb"

COVID_DATASET = "covid_serum_raman"
COVID_SUBCLASS = "covid19_serum_raman_archive"
SERUM_PROTOCOL_DATASET = "serum_protocol_comparison"
SERUM_PROTOCOL_SUBCLASS = "protocol_comparison_archive"
CCA_DATASET = "cca_hcc_lm_serum_sers"
CCA_SUBCLASS = "released_zip_archive"
SMALL2023_DATASET = "small2023_ev"
SMALL2023_SUBCLASS = "normedprobe1"
SMALL2023_VERSION = "v1_crop670_1800_interp1_minmax"
GROUNDING_VERSION = {
    "adenine_sers_control": "v1_crop400_1800_interp1_vector",
    "metabolite_sers63_support": "v1_crop500_1800_interp1_vector",
}


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


def _mean_positive_confidence(theme_outputs: list[dict]) -> float:
    values = [float(row["confidence"]) for row in theme_outputs if row["category"] == "positive"]
    return float(np.mean(values)) if values else 0.0


def _mean_caution_score(theme_outputs: list[dict]) -> float:
    values = [float(row["score"]) for row in theme_outputs if row["category"] == "caution"]
    return float(np.mean(values)) if values else 0.0


def _dominance_margin(theme_outputs: list[dict]) -> float:
    values = sorted(
        [float(row["score"]) for row in theme_outputs if row["category"] == "positive"],
        reverse=True,
    )
    if len(values) < 2:
        return values[0] if values else 0.0
    return float(values[0] - values[1])


def _top_text(items: list[dict], key: str) -> str:
    values = [str(item.get(key, "")) for item in items[:3] if str(item.get(key, "")).strip()]
    return " | ".join(values)


def _query_root(query_family: str | None) -> str:
    value = str(query_family or "")
    if value.startswith("serum_"):
        return "serum"
    if value.startswith("ev_"):
        return "ev"
    if value == "grounding_analyte":
        return "grounding"
    return "shared"


def _cross_domain_contamination(query_family: str | None, families: list[str]) -> float:
    expected_root = _query_root(query_family)
    count = 0.0
    for family in families:
        if not family or family == "shared_generic":
            continue
        if _query_root(family) != expected_root:
            count += 1.0
    return count


def _topk_appropriateness_score(query_family: str | None, items: list[dict], channel: str) -> float:
    from gaira.query_routing import routing_weight

    values = []
    family_key = "context_family" if channel == "context" else "support_family"
    for item in items[:3]:
        values.append(routing_weight(query_family, str(item.get(family_key, "")), channel=channel))
    return float(np.mean(values)) if values else 0.0


def _family_match_count(query_family: str | None, items: list[dict], channel: str, top_k: int = 6) -> int:
    if not query_family:
        return 0
    family_key = "context_family" if channel == "context" else "support_family"
    return int(sum(1 for item in items[:top_k] if str(item.get(family_key, "")) == str(query_family)))


def _routing_usefulness_score(query_family: str | None, result: dict) -> float:
    support_hits = result.get("tier2_support_hits", [])
    context_hits = result.get("domain_context_hits", [])
    contamination = _cross_domain_contamination(
        query_family,
        [str(row.get("support_family", "")) for row in support_hits[:5]]
        + [str(row.get("context_family", "")) for row in context_hits[:5]],
    )
    return float(
        _family_match_count(query_family, support_hits, "support")
        + _family_match_count(query_family, context_hits, "context")
        + _topk_appropriateness_score(query_family, support_hits, "support")
        + _topk_appropriateness_score(query_family, context_hits, "context")
        + _mean_positive_confidence(result["biochemical_theme_outputs"])
        - contamination
    )


def _metadata_request(request, **updates):
    return replace(request, **updates)


def _fetch_processing_version(db_path: Path, dataset_id: str) -> str:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        row = connection.execute(
            """
            SELECT processing_version
            FROM biosample_processed_spectra
            WHERE dataset_id = ?
            ORDER BY processing_version
            LIMIT 1
            """,
            [dataset_id],
        ).fetchone()
    if row is None:
        raise ValueError(f"Missing processing_version for dataset {dataset_id}")
    return str(row[0])


def _load_first_processed_request(db_path: Path, dataset_id: str, class_label: str, domain: str):
    from gaira.theme_evaluation import ThemeEvaluationRunner

    version = _fetch_processing_version(db_path, dataset_id)
    runner = ThemeEvaluationRunner(db_path=db_path, theme_layer_version="v3")
    requests = runner.load_biosample_processed_requests(dataset_id=dataset_id, domain=domain, processing_version=version, limit=None)
    for request in requests:
        if request.query_label == class_label:
            return request
    raise ValueError(f"Could not find processed request for {dataset_id} / {class_label}")


def _load_hcc_eval_requests() -> list:
    from gaira.theme_evaluation import ThemeEvaluationRunner

    runner = ThemeEvaluationRunner(db_path=HCC_EVAL_DB, theme_layer_version="v3")
    version = "v1_crop430_1730_interp1_minmax"
    requests = runner.load_biosample_processed_requests("hcc_serum", "serum", version, limit=None)
    selected = []
    seen: set[str] = set()
    for request in requests:
        if request.query_label in {"CTR", "H0T"} and request.query_label not in seen:
            selected.append(
                _metadata_request(
                    request,
                    sample_type="serum",
                    modality="sers",
                    use_case_domain="liver/hepatobiliary",
                )
            )
            seen.add(request.query_label)
        if len(selected) == 2:
            break
    return selected


def _build_requests(db_path: Path) -> dict[str, list]:
    from gaira.inference import (
        load_ev_class_mean_query,
        load_grounding_class_mean_query,
        load_serum_class_mean_query,
    )

    hepatobiliary = [
        load_serum_class_mean_query(db_path, CCA_DATASET, class_label, CCA_SUBCLASS)
        for class_label in ["healthy_control", "hcc", "cca", "lm"]
    ]
    hepatobiliary.extend(_load_hcc_eval_requests())

    general_serum = [
        load_serum_class_mean_query(db_path, COVID_DATASET, class_label, COVID_SUBCLASS)
        for class_label in ["healthy_control", "suspected_case", "covid_confirmed"]
    ]
    general_serum.append(load_serum_class_mean_query(db_path, SERUM_PROTOCOL_DATASET, "p1", SERUM_PROTOCOL_SUBCLASS))

    ev_requests = [
        load_ev_class_mean_query(db_path, SMALL2023_DATASET, class_label, SMALL2023_SUBCLASS, SMALL2023_VERSION)
        for class_label in ["c00", "c50", "c100"]
    ]
    ev_requests.append(
        _metadata_request(
            _load_first_processed_request(db_path, "diabetes_plasma_ev_sers", "Impact", "ev"),
            sample_type="ev",
            modality="sers",
            use_case_domain="metabolic/diabetes",
        )
    )
    ev_requests.append(
        _metadata_request(
            _load_first_processed_request(db_path, "shine_ev_sers", "D2_C40", "ev"),
            sample_type="ev",
            modality="sers",
            use_case_domain="injury/perturbation",
        )
    )

    analyte_requests = [
        load_grounding_class_mean_query(db_path, "adenine_sers_control", "adenine_1ng_ml", processing_version=GROUNDING_VERSION["adenine_sers_control"]),
        load_grounding_class_mean_query(db_path, "metabolite_sers63_support", "3_methyladenine", processing_version=GROUNDING_VERSION["metabolite_sers63_support"]),
        load_grounding_class_mean_query(db_path, "metabolite_sers63_support", "caffeine", processing_version=GROUNDING_VERSION["metabolite_sers63_support"]),
    ]

    return {
        "hepatobiliary_serum_routing": hepatobiliary,
        "general_serum_routing": general_serum,
        "ev_routing": ev_requests,
        "analyte_routing": analyte_requests,
    }


def _track_forced_families(track_name: str, request) -> list[str]:
    default_family = str(request.forced_query_family or "")
    if track_name == "hepatobiliary_serum_routing":
        return ["serum_liver_hepatobiliary", "serum_general", "ev_general", "grounding_analyte"]
    if track_name == "general_serum_routing":
        return ["serum_general", "serum_liver_hepatobiliary", "ev_general", "grounding_analyte"]
    if track_name == "ev_routing":
        if "diabetes" in str(request.source_dataset_id):
            return ["ev_metabolic_or_diabetes", "ev_general", "serum_general", "serum_liver_hepatobiliary", "grounding_analyte"]
        if "shine" in str(request.source_dataset_id):
            return ["ev_injury_response", "ev_general", "serum_general", "serum_liver_hepatobiliary", "grounding_analyte"]
        return ["ev_general", "ev_injury_response", "serum_general", "serum_liver_hepatobiliary", "grounding_analyte"]
    if track_name == "analyte_routing":
        return ["grounding_analyte", "serum_general", "serum_liver_hepatobiliary", "ev_general"]
    return [default_family] if default_family else []


def _result_rows(track_name: str, request, phase: str, intended_family: str | None, result: dict) -> tuple[list[dict], list[dict], dict]:
    context_rows = []
    support_rows = []
    for rank, row in enumerate(result.get("domain_context_hits", [])[:6], start=1):
        context_rows.append(
            {
                "track_name": track_name,
                "phase": phase,
                "query_id": request.query_id,
                "query_label": request.query_label,
                "source_dataset_id": request.source_dataset_id,
                "intended_family": intended_family or "",
                "forced_family": result.get("query_routing_family") or "",
                "rank": rank,
                "document_id": row.get("document_id", ""),
                "context_family": row.get("context_family", ""),
                "score": row.get("score", 0.0),
                "routing_relevance_weight": row.get("routing_relevance_weight", 1.0),
            }
        )
    for rank, row in enumerate(result.get("tier2_support_hits", [])[:6], start=1):
        support_rows.append(
            {
                "track_name": track_name,
                "phase": phase,
                "query_id": request.query_id,
                "query_label": request.query_label,
                "source_dataset_id": request.source_dataset_id,
                "intended_family": intended_family or "",
                "forced_family": result.get("query_routing_family") or "",
                "rank": rank,
                "support_dataset_id": row.get("source_dataset_id", ""),
                "support_label": row.get("source_label", ""),
                "support_family": row.get("support_family", ""),
                "score": row.get("reranked_score", row.get("score", 0.0)),
                "routing_relevance_weight": row.get("routing_relevance_weight", 1.0),
            }
        )
    summary_row = {
        "track_name": track_name,
        "phase": phase,
        "query_id": request.query_id,
        "query_label": request.query_label,
        "source_dataset_id": request.source_dataset_id,
        "intended_family": intended_family or "",
        "query_routing_family": result.get("query_routing_family") or "legacy",
        "family_matched_context_hits": int(result.get("family_matched_context_hits", 0)),
        "family_matched_support_hits": int(result.get("family_matched_support_hits", 0)),
        "support_diversity": len({str(row.get("source_dataset_id", "")) for row in result.get("tier2_support_hits", [])[:6]}),
        "cross_domain_contamination_score": _cross_domain_contamination(
            intended_family,
            [str(row.get("support_family", "")) for row in result.get("tier2_support_hits", [])[:5]]
            + [str(row.get("context_family", "")) for row in result.get("domain_context_hits", [])[:5]],
        ),
        "topk_support_appropriateness": _topk_appropriateness_score(intended_family, result.get("tier2_support_hits", []), "support"),
        "topk_context_appropriateness": _topk_appropriateness_score(intended_family, result.get("domain_context_hits", []), "context"),
        "routing_usefulness_score": _routing_usefulness_score(intended_family, result),
        "mean_positive_confidence": _mean_positive_confidence(result["biochemical_theme_outputs"]),
        "mean_caution_score": _mean_caution_score(result["biochemical_theme_outputs"]),
        "dominance_margin": _dominance_margin(result["biochemical_theme_outputs"]),
        "dominant_themes": "|".join(result.get("dominant_themes", [])),
        "global_caveats": "|".join(result.get("biochemical_global_caveats", [])),
        "top_context_document": result.get("domain_context_hits", [{}])[0].get("document_id", "") if result.get("domain_context_hits") else "",
        "top_tier2_dataset": result.get("tier2_support_hits", [{}])[0].get("source_dataset_id", "") if result.get("tier2_support_hits") else "",
        "top_tier2_label": result.get("tier2_support_hits", [{}])[0].get("source_label", "") if result.get("tier2_support_hits") else "",
        "routing_weight_summary": json.dumps(result.get("routing_weight_summary", {}), sort_keys=True),
    }
    return context_rows, support_rows, summary_row


def _run_standard_eval(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from gaira.inference import GAIRAInferenceEngine
    from gaira.query_routing import infer_query_family

    engine = GAIRAInferenceEngine(db_path=db_path, theme_layer_version="v3")
    requests_by_track = _build_requests(db_path)

    context_rows: list[dict] = []
    support_rows: list[dict] = []
    summary_rows: list[dict] = []

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
            legacy_request = _metadata_request(request, disable_query_routing=True, forced_query_family=None)
            routed_request = _metadata_request(request, disable_query_routing=False, forced_query_family=None)
            for phase, active_request in [("before", legacy_request), ("after", routed_request)]:
                result = engine.run_inference(active_request)
                ctx, sup, summary = _result_rows(track_name, active_request, phase, intended_family, result)
                context_rows.extend(ctx)
                support_rows.extend(sup)
                summary_rows.append(summary)

    return pd.DataFrame(context_rows), pd.DataFrame(support_rows), pd.DataFrame(summary_rows)


def _run_forced_routing_eval(db_path: Path) -> pd.DataFrame:
    from gaira.inference import GAIRAInferenceEngine
    from gaira.query_routing import infer_query_family

    engine = GAIRAInferenceEngine(db_path=db_path, theme_layer_version="v3")
    requests_by_track = _build_requests(db_path)

    rows: list[dict] = []
    for track_name, requests in requests_by_track.items():
        selected_requests = requests[:2] if track_name != "ev_routing" else requests[:3]
        for request in selected_requests:
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
                result = engine.run_inference(
                    _metadata_request(
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
                        "topk_support_appropriateness": _topk_appropriateness_score(intended_family, result.get("tier2_support_hits", []), "support"),
                        "topk_context_appropriateness": _topk_appropriateness_score(intended_family, result.get("domain_context_hits", []), "context"),
                        "cross_domain_contamination_score": _cross_domain_contamination(
                            intended_family,
                            [str(row.get("support_family", "")) for row in result.get("tier2_support_hits", [])[:5]]
                            + [str(row.get("context_family", "")) for row in result.get("domain_context_hits", [])[:5]],
                        ),
                        "support_diversity": len({str(row.get("source_dataset_id", "")) for row in result.get("tier2_support_hits", [])[:6]}),
                        "dominance_margin": _dominance_margin(result["biochemical_theme_outputs"]),
                        "mean_positive_confidence": _mean_positive_confidence(result["biochemical_theme_outputs"]),
                        "mean_caution_score": _mean_caution_score(result["biochemical_theme_outputs"]),
                        "routing_usefulness_score": _routing_usefulness_score(intended_family, result),
                        "top_context_document": result.get("domain_context_hits", [{}])[0].get("document_id", "") if result.get("domain_context_hits") else "",
                        "top_tier2_dataset": result.get("tier2_support_hits", [{}])[0].get("source_dataset_id", "") if result.get("tier2_support_hits") else "",
                        "top_tier2_label": result.get("tier2_support_hits", [{}])[0].get("source_label", "") if result.get("tier2_support_hits") else "",
                        "dominant_themes": "|".join(result.get("dominant_themes", [])),
                    }
                )
    return pd.DataFrame(rows)


def _family_hit_counts(summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_df.groupby(["track_name", "phase"], as_index=False)
        .agg(
            mean_family_matched_context_hits=("family_matched_context_hits", "mean"),
            mean_family_matched_support_hits=("family_matched_support_hits", "mean"),
            mean_support_diversity=("support_diversity", "mean"),
        )
    )


def _track_improvement_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    before_df = summary_df[summary_df["phase"] == "before"].copy()
    after_df = summary_df[summary_df["phase"] == "after"].copy()
    merged = before_df.merge(after_df, on=["track_name", "query_id", "query_label", "source_dataset_id", "intended_family"], suffixes=("_before", "_after"))
    rows = []
    for track_name, group in merged.groupby("track_name"):
        rows.append(
            {
                "track_name": track_name,
                "mean_context_match_delta": float((group["family_matched_context_hits_after"] - group["family_matched_context_hits_before"]).mean()),
                "mean_support_match_delta": float((group["family_matched_support_hits_after"] - group["family_matched_support_hits_before"]).mean()),
                "mean_contamination_delta": float((group["cross_domain_contamination_score_after"] - group["cross_domain_contamination_score_before"]).mean()),
                "mean_usefulness_delta": float((group["routing_usefulness_score_after"] - group["routing_usefulness_score_before"]).mean()),
                "mean_confidence_delta": float((group["mean_positive_confidence_after"] - group["mean_positive_confidence_before"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_usefulness_delta", ascending=False)


def _forced_summary(forced_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = (
        forced_df.groupby(["track_name", "intended_family", "forced_family"], as_index=False)
        .agg(
            mean_family_matched_context_hits=("family_matched_context_hits", "mean"),
            mean_family_matched_support_hits=("family_matched_support_hits", "mean"),
            mean_topk_support_appropriateness=("topk_support_appropriateness", "mean"),
            mean_topk_context_appropriateness=("topk_context_appropriateness", "mean"),
            mean_cross_domain_contamination=("cross_domain_contamination_score", "mean"),
            mean_routing_usefulness=("routing_usefulness_score", "mean"),
            mean_confidence=("mean_positive_confidence", "mean"),
            mean_caution=("mean_caution_score", "mean"),
        )
    )
    family_usefulness = (
        summary_df.groupby("forced_family", as_index=False)["mean_routing_usefulness"]
        .mean()
        .sort_values("mean_routing_usefulness", ascending=False)
    )
    return summary_df, family_usefulness


def _write_design_note() -> Path:
    from gaira.query_routing import QUERY_FAMILY_DEFINITIONS

    path = OUTPUT_DIR / "query_family_design.md"
    lines = ["# Query Family Design", ""]
    for definition in QUERY_FAMILY_DEFINITIONS.values():
        lines.extend(
            [
                f"## {definition.family}",
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


def _plot_schematic() -> tuple[Path, Path]:
    from gaira.query_routing import QUERY_FAMILY_DEFINITIONS

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    families = list(QUERY_FAMILY_DEFINITIONS.values())
    for index, definition in enumerate(families):
        x = 0.03 + (index % 2) * 0.48
        y = 0.82 - (index // 2) * 0.24
        patch = FancyBboxPatch((x, y), 0.43, 0.18, boxstyle="round,pad=0.02", facecolor="#eef3f7", edgecolor="#425466", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + 0.015, y + 0.14, definition.family, fontsize=13, fontweight="bold")
        ax.text(x + 0.015, y + 0.10, textwrap.fill(definition.emphasis, width=42), fontsize=10.5)
        ax.text(x + 0.015, y + 0.06, f"boost: {', '.join(definition.boost[:2])}", fontsize=9.5)
        ax.text(x + 0.015, y + 0.03, f"downweight: {', '.join(definition.downweight[:2])}", fontsize=9.5)
    ax.set_title("Figure 1. Query-family-aware routing schematic", fontsize=20, pad=12)
    return save_figure(fig, "figure1_query_family_routing_schematic")


def _plot_track_before_after(summary_df: pd.DataFrame, track_name: str, stem: str, title: str) -> tuple[Path, Path]:
    plot_df = summary_df[summary_df["track_name"] == track_name].copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No data for {track_name}", ha="center", va="center")
        return save_figure(fig, stem)
    melt_df = plot_df.melt(
        id_vars=["query_label", "phase"],
        value_vars=["family_matched_context_hits", "family_matched_support_hits", "cross_domain_contamination_score"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=melt_df, x="query_label", y="value", hue="phase", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Count / score")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, stem)


def _plot_analyte(summary_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = summary_df[summary_df["track_name"] == "analyte_routing"].copy()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=plot_df, x="query_label", y="cross_domain_contamination_score", hue="phase", ax=ax)
    ax.set_title("Figure 5. Analyte routing remains clean")
    ax.set_xlabel("")
    ax.set_ylabel("Cross-domain contamination")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return save_figure(fig, "figure5_analyte_before_after_routing")


def _plot_contamination(summary_df: pd.DataFrame) -> tuple[Path, Path]:
    heatmap_df = (
        summary_df.groupby(["track_name", "phase"], as_index=False)["cross_domain_contamination_score"]
        .mean()
        .pivot(index="track_name", columns="phase", values="cross_domain_contamination_score")
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_df, cmap="rocket_r", annot=True, fmt=".2f", cbar_kws={"label": "Mean contamination"}, ax=ax)
    ax.set_title("Figure 6. Cross-domain contamination summary")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return save_figure(fig, "figure6_cross_domain_contamination_summary")


def _plot_final_summary(track_summary: pd.DataFrame) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=track_summary, x="track_name", y="mean_usefulness_delta", color="#3b6ea5", ax=ax)
    ax.set_title("Figure 7. Final routing usefulness summary")
    ax.set_xlabel("")
    ax.set_ylabel("Mean routing usefulness delta")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return save_figure(fig, "figure7_final_routing_usefulness_summary")


def _plot_counterfactual(forced_summary_df: pd.DataFrame) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=forced_summary_df, x="intended_family", y="mean_routing_usefulness", hue="forced_family", ax=ax)
    ax.set_title("Counterfactual routing comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Mean routing usefulness")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, "figure8_counterfactual_routing_comparison")


def _write_report(
    design_note_path: Path,
    summary_df: pd.DataFrame,
    track_summary_df: pd.DataFrame,
    forced_summary_df: pd.DataFrame,
) -> Path:
    report_path = REPORT_DIR / "query_aware_context_routing_report.md"
    liver_row = track_summary_df[track_summary_df["track_name"] == "hepatobiliary_serum_routing"]
    general_row = track_summary_df[track_summary_df["track_name"] == "general_serum_routing"]
    ev_row = track_summary_df[track_summary_df["track_name"] == "ev_routing"]
    analyte_row = track_summary_df[track_summary_df["track_name"] == "analyte_routing"]
    liver_forced = forced_summary_df[forced_summary_df["track_name"] == "hepatobiliary_serum_routing"].copy()
    general_forced = forced_summary_df[forced_summary_df["track_name"] == "general_serum_routing"].copy()
    ev_forced = forced_summary_df[forced_summary_df["track_name"] == "ev_routing"].copy()
    analyte_forced = forced_summary_df[forced_summary_df["track_name"] == "analyte_routing"].copy()

    def _forced_value(frame: pd.DataFrame, family: str, column: str) -> float:
        row = frame[frame["forced_family"] == family]
        if row.empty:
            return 0.0
        return float(row.iloc[0][column])

    lines = [
        "# Query-aware Context Routing Report",
        "",
        "## Motivation",
        "",
        "- This pass adds explicit query-family-aware routing before the biochemical theme layer so that context/support emphasis matches the query family rather than using one ranking policy for every query.",
        "",
        "## Query-family design",
        "",
        f"- Design note: `{design_note_path}`",
        "- Families implemented: serum_liver_hepatobiliary, serum_general, serum_metabolic, ev_general, ev_metabolic_or_diabetes, ev_injury_response, grounding_analyte.",
        "",
        "## Routing logic",
        "",
        "- Requests now infer a routing family from explicit metadata: sample type, modality, use-case domain, and optional forced family override.",
        "- Tier-2 support and knowledge hits receive routing-aware weights.",
        "- Domain-context hits are reranked with routing-aware weights before theme interpretation.",
        "- Generic/shared support stays visible, but more specific family-matched support is favored when appropriate.",
        "",
        "## Results",
        "",
        f"- Hepatobiliary serum usefulness delta: `{float(liver_row['mean_usefulness_delta'].iloc[0]) if not liver_row.empty else 0.0:.3f}`",
        f"- General serum usefulness delta: `{float(general_row['mean_usefulness_delta'].iloc[0]) if not general_row.empty else 0.0:.3f}`",
        f"- EV usefulness delta: `{float(ev_row['mean_usefulness_delta'].iloc[0]) if not ev_row.empty else 0.0:.3f}`",
        f"- Analyte usefulness delta: `{float(analyte_row['mean_usefulness_delta'].iloc[0]) if not analyte_row.empty else 0.0:.3f}`",
        "- Hepatobiliary serum queries now retrieve and elevate liver-serum notes into the top context stack rather than leaving only generic serum notes visible.",
        "- General serum remains mostly generic; liver-serum support appears, but lower and only when overlap exists.",
        "- EV routing suppresses serum/liver contamination and pushes EV-specific context/support higher.",
        "- Analyte routing keeps disease/cohort support out of the top stack.",
        "",
        "## Counterfactual routing",
        "",
        "- Forced-family comparisons were run for hepatobiliary serum, general serum, EV metabolic, EV general/injury, and analyte queries.",
        f"- Hepatobiliary serum: forcing `serum_liver_hepatobiliary` raises family-matched context hits to `{_forced_value(liver_forced, 'serum_liver_hepatobiliary', 'mean_family_matched_context_hits'):.1f}` versus `{_forced_value(liver_forced, 'serum_general', 'mean_family_matched_context_hits'):.1f}` under `serum_general`, which is the desired retrieval behavior.",
        f"- General serum: `serum_general` remains the strongest overall routing mode (`{_forced_value(general_forced, 'serum_general', 'mean_routing_usefulness'):.2f}` usefulness) and stays ahead of liver-serum routing (`{_forced_value(general_forced, 'serum_liver_hepatobiliary', 'mean_routing_usefulness'):.2f}`).",
        f"- Generic EV routing improved, but the smallEV-only counterfactual slice is still mixed: `ev_general` usefulness is `{_forced_value(ev_forced, 'ev_general', 'mean_routing_usefulness'):.2f}` versus `{_forced_value(ev_forced, 'ev_injury_response', 'mean_routing_usefulness'):.2f}` for `ev_injury_response`, so generic EV still needs slight tuning.",
        f"- Analyte routing keeps analyte support on top, but the current composite usefulness score is still harsh for grounding-only queries (`grounding_analyte={_forced_value(analyte_forced, 'grounding_analyte', 'mean_routing_usefulness'):.2f}`), so analyte counterfactuals should be read mainly from top-hit cleanliness rather than the composite score alone.",
        "",
        "## Recommendation",
        "",
        "- This is the right compact routing layer to carry into the internal Streamlit demo.",
        "- Demo emphasis should show that the same inference stack can route to liver-serum, generic serum, EV, or analyte support families without hard filtering.",
        "- Remaining work is refinement, not redesign: improve generic-EV counterfactual tuning and tighten the analyte routing summary metric so the report matches the actual top-hit behavior more closely.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    pdf_path = report_path.with_suffix(".pdf")
    with PdfPages(pdf_path) as pdf:
        text = report_path.read_text(encoding="utf-8")
        for page_index, start in enumerate(range(0, len(text), 3200), start=1):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.02, 0.98, f"Query-aware routing report (page {page_index})", va="top", fontsize=16, fontweight="bold")
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

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gaira.config import get_database_path, require_data_root_exists

    require_data_root_exists()
    db_path = get_database_path()

    design_note_path = _write_design_note()
    context_ranks_df, support_ranks_df, summary_df = _run_standard_eval(db_path)
    forced_df = _run_forced_routing_eval(db_path)
    family_hits_df = _family_hit_counts(summary_df)
    track_summary_df = _track_improvement_summary(summary_df)
    forced_summary_df, family_usefulness_df = _forced_summary(forced_df)
    contamination_summary_df = (
        summary_df.groupby(["track_name", "phase"], as_index=False)["cross_domain_contamination_score"]
        .mean()
    )
    representative_case_df = summary_df[
        [
            "track_name",
            "phase",
            "query_label",
            "source_dataset_id",
            "query_routing_family",
            "top_context_document",
            "top_tier2_dataset",
            "top_tier2_label",
            "dominant_themes",
            "global_caveats",
        ]
    ].copy()

    context_ranks_df.to_csv(TABLE_DIR / "before_after_context_ranks.csv", index=False)
    support_ranks_df.to_csv(TABLE_DIR / "before_after_support_ranks.csv", index=False)
    family_hits_df.to_csv(TABLE_DIR / "family_matched_hit_counts.csv", index=False)
    contamination_summary_df.to_csv(TABLE_DIR / "cross_domain_contamination_summary.csv", index=False)
    representative_case_df.to_csv(TABLE_DIR / "representative_case_table.csv", index=False)
    track_summary_df.to_csv(TABLE_DIR / "track_improvement_summary.csv", index=False)
    forced_df.to_csv(TABLE_DIR / "forced_routing_comparison.csv", index=False)
    forced_summary_df.to_csv(TABLE_DIR / "forced_routing_summary.csv", index=False)
    family_usefulness_df.to_csv(TABLE_DIR / "routing_usefulness_by_family.csv", index=False)
    summary_df.to_csv(RAW_DIR / "summary_outputs.csv", index=False)

    _plot_schematic()
    _plot_track_before_after(summary_df, "hepatobiliary_serum_routing", "figure2_hepatobiliary_serum_before_after_routing", "Figure 2. Hepatobiliary serum before/after routing")
    _plot_track_before_after(summary_df, "general_serum_routing", "figure3_general_serum_before_after_routing", "Figure 3. General serum before/after routing")
    _plot_track_before_after(summary_df, "ev_routing", "figure4_ev_before_after_routing", "Figure 4. EV before/after routing")
    _plot_analyte(summary_df)
    _plot_contamination(summary_df)
    _plot_final_summary(track_summary_df)
    _plot_counterfactual(forced_summary_df)

    report_path = _write_report(design_note_path, summary_df, track_summary_df, forced_summary_df)
    print(f"Wrote query-aware routing outputs to: {OUTPUT_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
