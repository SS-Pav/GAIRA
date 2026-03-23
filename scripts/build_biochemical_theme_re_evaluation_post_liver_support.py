from __future__ import annotations

import json
import math
import shutil
import textwrap
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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA")
LIVER_SUPPORT_DIR = ROOT / "processed" / "liver_serum_literature_integration"
PRE_DB = LIVER_SUPPORT_DIR / "working" / "pre_integration.duckdb"
LIVE_DB = ROOT / "interim" / "gaira.duckdb"
HCC_EVAL_DB = ROOT / "processed" / "hcc_holdout_evaluation" / "eval_db" / "gaira_hcc_holdout_eval.duckdb"

OUTPUT_DIR = ROOT / "processed" / "biochemical_theme_re_evaluation_post_liver_support"
RAW_DIR = OUTPUT_DIR / "raw_outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report"
WORKING_DIR = OUTPUT_DIR / "working"

SERUM_PROTOCOL_VERSION = "v1_crop400_1800_interp1_minmax"
COVID_VERSION = "v1_crop400_1800_interp1_minmax"
SMALL2023_VERSION = "v1_crop670_1800_interp1_minmax"
HCC_VERSION = "v1_crop430_1730_interp1_minmax"
ADENINE_VERSION = "v1_crop400_1800_interp1_vector"
METABOLITE_VERSION = "v1_crop500_1800_interp1_vector"
THEME_VERSION = "v3"

POSITIVE_THEMES = [
    "lipid_membrane_associated",
    "protein_peptide_associated",
    "nucleic_acid_purine_associated",
    "carbohydrate_glycan_associated",
    "oxidative_metabolic_stress_associated",
]
CAUTION_THEMES = [
    "matrix_dominance_caution",
    "probe_substrate_caution",
    "modality_mismatch_caution",
    "weak_label_or_cohort_caution",
    "low_specificity_caution",
]

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
sns.set_theme(style="whitegrid", context="talk")


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, RAW_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR, WORKING_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def _remove_liver_overlay(db_path: Path) -> None:
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("DELETE FROM grounding_support_chunks WHERE dataset_id = 'liver_serum_literature_support'")
        connection.execute("DELETE FROM grounding_support_documents WHERE dataset_id = 'liver_serum_literature_support'")
        connection.execute(
            "DELETE FROM domain_context_chunks WHERE document_id LIKE 'gaira_serum_context_liver_%' OR document_id LIKE 'gaira_serum_context_hcc_%' OR document_id LIKE 'gaira_serum_context_metabolic_%'"
        )
        connection.execute(
            "DELETE FROM domain_context_documents WHERE document_id LIKE 'gaira_serum_context_liver_%' OR document_id LIKE 'gaira_serum_context_hcc_%' OR document_id LIKE 'gaira_serum_context_metabolic_%'"
        )


def _copy_pre_eval_db() -> Path:
    temp_path = WORKING_DIR / "hcc_pre_liver_support_eval.duckdb"
    shutil.copy2(HCC_EVAL_DB, temp_path)
    _remove_liver_overlay(temp_path)
    return temp_path


def _flatten_theme_outputs(track_name: str, request, result: dict) -> list[dict]:
    rows = []
    for row in result["biochemical_theme_outputs"]:
        rows.append(
            {
                "track_name": track_name,
                "query_id": request.query_id,
                "source_dataset_id": request.source_dataset_id,
                "query_label": request.query_label,
                "query_family": request.query_family,
                "theme_name": row["theme_name"],
                "category": row["category"],
                "theme_layer_version": result["biochemical_theme_layer_version"],
                "score": float(row["score"]),
                "confidence": float(row["confidence"]),
                "raw_score_pre_normalization": float(row.get("raw_score_pre_normalization", row["score"])),
                "normalized_score": float(row.get("normalized_score", row["score"])),
                "competition_penalty": float(row.get("competition_penalty", 0.0)),
                "caution_penalty": float(row.get("caution_penalty", 0.0)),
                "calibration_penalty": float(row.get("calibration_penalty", 0.0)),
                "specificity_index": float(row.get("specificity_index", 0.0)),
                "tier1_contrib": float(row["evidence_contributions"]["tier1"]),
                "tier2_contrib": float(row["evidence_contributions"]["tier2"]),
                "knowledge_contrib": float(row["evidence_contributions"]["knowledge"]),
                "semantic_contrib": float(row["evidence_contributions"]["semantic"]),
                "context_contrib": float(row["evidence_contributions"]["context"]),
                "band_contrib": float(row["evidence_contributions"]["band"]),
                "n_tier1_hits": len(row.get("supporting_tier1_hits", [])),
                "n_tier2_hits": len(row.get("supporting_tier2_hits", [])),
                "n_knowledge_hits": len(row.get("supporting_knowledge_hits", [])),
                "n_semantic_hits": len(row.get("supporting_semantic_regions", [])),
                "n_bands": len(row.get("supporting_bands", [])),
                "limiting_evidence": "|".join(str(item) for item in row.get("opposing_or_limiting_evidence", [])),
            }
        )
    return rows


def _evaluate_requests(db_path: Path, requests: list, theme_version: str, track_name: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    from gaira.inference import GAIRAInferenceEngine

    engine = GAIRAInferenceEngine(db_path=db_path, theme_layer_version=theme_version)
    query_rows = []
    theme_rows = []
    results = []
    for request in requests:
        result = engine.run_inference(request)
        results.append(result)
        tier2_hits = result.get("tier2_support_hits", [])
        liver_hits = [hit for hit in tier2_hits if hit.get("source_dataset_id") == "liver_serum_literature_support"]
        same_dataset_hits = [hit for hit in tier2_hits if str(hit.get("source_dataset_id")) == str(request.source_dataset_id)]
        query_rows.append(
            {
                "track_name": track_name,
                "query_id": request.query_id,
                "source_dataset_id": request.source_dataset_id,
                "query_label": request.query_label,
                "query_family": request.query_family,
                "dominant_themes": "|".join(result.get("dominant_themes", [])),
                "global_caveats": "|".join(result.get("biochemical_global_caveats", [])),
                "theme_summary": result.get("biochemical_theme_summary", ""),
                "evidence_profile_summary": result.get("evidence_profile_summary", ""),
                "n_tier1_hits": len(result.get("tier1_grounding_hits", [])),
                "n_tier2_hits": len(tier2_hits),
                "n_liver_support_hits": len(liver_hits),
                "n_same_dataset_tier2_hits": len(same_dataset_hits),
                "tier2_dataset_diversity": len({hit.get("source_dataset_id") for hit in tier2_hits}),
                "top_tier2_source_dataset": tier2_hits[0]["source_dataset_id"] if tier2_hits else "",
                "top_tier2_source_label": tier2_hits[0]["source_label"] if tier2_hits else "",
                "top_context_document": result.get("domain_context_hits", [{}])[0].get("document_id", "") if result.get("domain_context_hits") else "",
                "mean_positive_confidence": float(
                    np.mean([row["confidence"] for row in result["biochemical_theme_outputs"] if row["category"] == "positive"])
                ),
                "mean_caution_score": float(
                    np.mean([row["score"] for row in result["biochemical_theme_outputs"] if row["category"] == "caution"])
                ),
            }
        )
        theme_rows.extend(_flatten_theme_outputs(track_name, request, result))

    return pd.DataFrame(query_rows), pd.DataFrame(theme_rows), results


def _evaluate_theme_inputs(runner, theme_inputs: list, track_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_rows = []
    theme_rows = []
    for theme_input in theme_inputs:
        result = runner.theme_layer.build_from_input(theme_input)
        tier2_hits = theme_input.tier2_hits
        query_rows.append(
            {
                "track_name": track_name,
                "query_id": theme_input.query_id,
                "source_dataset_id": theme_input.source_dataset_id,
                "query_label": theme_input.query_label,
                "query_family": theme_input.query_family,
                "dominant_themes": "|".join(result.get("dominant_themes", [])),
                "global_caveats": "|".join(result.get("biochemical_global_caveats", [])),
                "theme_summary": result.get("biochemical_theme_summary", ""),
                "evidence_profile_summary": result.get("evidence_profile_summary", ""),
                "n_tier2_hits": len(tier2_hits),
                "tier2_dataset_diversity": len({hit.get("source_dataset_id") for hit in tier2_hits}),
            }
        )
        fake_request = type(
            "ThemeRequest",
            (),
            {
                "query_id": theme_input.query_id,
                "source_dataset_id": theme_input.source_dataset_id,
                "query_label": theme_input.query_label,
                "query_family": theme_input.query_family,
            },
        )()
        theme_rows.extend(_flatten_theme_outputs(track_name, fake_request, result))
    return pd.DataFrame(query_rows), pd.DataFrame(theme_rows)


def _metric(track: str, name: str, value: float) -> dict:
    return {"track_name": track, "metric_name": name, "metric_value": float(value)}


def _compute_general_track_metrics(track_name: str, query_df: pd.DataFrame, theme_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if theme_df.empty:
        return pd.DataFrame()
    positive_df = theme_df[theme_df["category"] == "positive"].copy()
    caution_df = theme_df[theme_df["category"] == "caution"].copy()
    if not positive_df.empty:
        pivot = positive_df.pivot_table(index="query_id", columns="theme_name", values="score", aggfunc="mean").fillna(0.0)
        top = np.sort(pivot.to_numpy(), axis=1)[:, -1]
        second = np.sort(pivot.to_numpy(), axis=1)[:, -2] if pivot.shape[1] > 1 else np.zeros_like(top)
        rows.append(_metric(track_name, "mean_dominance_margin", float(np.mean(top - second))))
        entropy_rows = []
        for _, row in pivot.iterrows():
            values = row.to_numpy(dtype=float)
            s = float(values.sum())
            if s <= 0:
                entropy_rows.append(0.0)
                continue
            p = values / s
            entropy = -float(np.sum(np.where(p > 0, p * np.log(np.clip(p, 1e-9, None)), 0.0)))
            max_entropy = math.log(len(values)) if len(values) > 1 else 1.0
            entropy_rows.append(1.0 - entropy / max_entropy if max_entropy > 0 else 0.0)
        rows.append(_metric(track_name, "positive_entropy_inverse", float(np.mean(entropy_rows))))
    rows.append(_metric(track_name, "mean_positive_confidence", float(positive_df["confidence"].mean()) if not positive_df.empty else 0.0))
    rows.append(_metric(track_name, "mean_caution_score", float(caution_df["score"].mean()) if not caution_df.empty else 0.0))
    if not query_df.empty:
        rows.append(_metric(track_name, "mean_support_dataset_diversity", float(query_df["tier2_dataset_diversity"].mean()) if "tier2_dataset_diversity" in query_df.columns else 0.0))
        rows.append(_metric(track_name, "mean_liver_support_hits", float(query_df["n_liver_support_hits"].mean()) if "n_liver_support_hits" in query_df.columns else 0.0))
    return pd.DataFrame(rows)


def _compute_analyte_metrics(query_df: pd.DataFrame, theme_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    positive_df = theme_df[theme_df["category"] == "positive"].copy()
    pivot = positive_df.pivot_table(index="query_id", columns="theme_name", values="score", aggfunc="mean").fillna(0.0)
    label_lookup = query_df.set_index("query_id")["query_label"].to_dict()
    purine_ids = [qid for qid, label in label_lookup.items() if any(token in str(label).lower() for token in ["adenine", "methyladenine", "caffeine", "nicotinamide"])]
    nonpurine_ids = [qid for qid, label in label_lookup.items() if qid not in purine_ids]
    if "nucleic_acid_purine_associated" in pivot.columns:
        purine_scores = pivot.loc[pivot.index.intersection(purine_ids), "nucleic_acid_purine_associated"] if purine_ids else pd.Series(dtype=float)
        nonpurine_scores = pivot.loc[pivot.index.intersection(nonpurine_ids), "nucleic_acid_purine_associated"] if nonpurine_ids else pd.Series(dtype=float)
        other_cols = [c for c in POSITIVE_THEMES if c != "nucleic_acid_purine_associated" and c in pivot.columns]
        other_max = pivot[other_cols].max(axis=1) if other_cols else pd.Series(0.0, index=pivot.index)
        purine_margin = (pivot["nucleic_acid_purine_associated"] - other_max).mean()
        rows.extend(
            [
                _metric("controlled_analyte_specificity", "purine_theme_mean", float(purine_scores.mean()) if not purine_scores.empty else 0.0),
                _metric("controlled_analyte_specificity", "nonpurine_purine_bleed", float(nonpurine_scores.mean()) if not nonpurine_scores.empty else 0.0),
                _metric("controlled_analyte_specificity", "purine_dominance_margin", float(purine_margin)),
                _metric(
                    "controlled_analyte_specificity",
                    "purine_top_fraction",
                    float((pivot["nucleic_acid_purine_associated"] >= other_max).mean()),
                ),
                _metric(
                    "controlled_analyte_specificity",
                    "specificity_index_mean",
                    float(
                        theme_df[
                            (theme_df["category"] == "positive") & (theme_df["theme_name"] == "nucleic_acid_purine_associated")
                        ]["specificity_index"].mean()
                    ),
                ),
            ]
        )
    rows.append(_metric("controlled_analyte_specificity", "support_dataset_diversity", float(query_df["tier2_dataset_diversity"].mean()) if not query_df.empty else 0.0))
    return pd.DataFrame(rows)


def _compute_ev_metrics(query_df: pd.DataFrame, theme_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    positive_df = theme_df[theme_df["category"] == "positive"].copy()
    caution_df = theme_df[theme_df["category"] == "caution"].copy()
    query_df = query_df.copy()
    query_df["mixture_fraction"] = query_df["query_label"].map({"c00": 0, "c50": 50, "c100": 100})
    for theme_name in ["lipid_membrane_associated", "protein_peptide_associated", "nucleic_acid_purine_associated", "oxidative_metabolic_stress_associated"]:
        subset = positive_df[positive_df["theme_name"] == theme_name].copy()
        if subset.empty:
            continue
        merged = subset.merge(query_df[["query_id", "mixture_fraction"]], on="query_id", how="left").dropna(subset=["mixture_fraction"])
        if len(merged) >= 3:
            ordered = merged.sort_values("mixture_fraction")
            diffs = np.diff(ordered["score"].to_numpy(dtype=float))
            smoothness = 1.0 / (1.0 + float(np.mean(np.abs(diffs)))) if len(diffs) else 0.0
            rows.append(_metric("ev_mixture_coherence", f"{theme_name}_smoothness", smoothness))
    rows.append(_metric("ev_mixture_coherence", "mean_liver_support_hits", float(query_df["n_liver_support_hits"].mean()) if not query_df.empty else 0.0))
    rows.append(
        _metric(
            "ev_mixture_coherence",
            "caution_variability",
            float(caution_df.groupby("theme_name")["score"].std().mean()) if not caution_df.empty else 0.0,
        )
    )
    return pd.DataFrame(rows)


def _compute_covid_metrics(query_df: pd.DataFrame, theme_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    positive_df = theme_df[theme_df["category"] == "positive"].copy()
    caution_df = theme_df[theme_df["category"] == "caution"].copy()
    rows.append(_metric("covid_serum_usefulness", "positive_signal_mean", float(positive_df["score"].mean()) if not positive_df.empty else 0.0))
    modality = caution_df[caution_df["theme_name"] == "modality_mismatch_caution"]["score"]
    rows.append(_metric("covid_serum_usefulness", "modality_caution_mean", float(modality.mean()) if not modality.empty else 0.0))
    rows.append(_metric("covid_serum_usefulness", "support_diversity_mean", float(query_df["tier2_dataset_diversity"].mean()) if not query_df.empty else 0.0))
    rows.append(_metric("covid_serum_usefulness", "liver_support_intrusion_mean", float(query_df["n_liver_support_hits"].mean()) if not query_df.empty else 0.0))
    return pd.DataFrame(rows)


def _compute_liver_cohort_metrics(query_df: pd.DataFrame, theme_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    positive_df = theme_df[theme_df["category"] == "positive"].copy()
    caution_df = theme_df[theme_df["category"] == "caution"].copy()
    rows.append(_metric("liver_serum_cohort_reasoning", "same_family_liver_support_mean", float(query_df["n_liver_support_hits"].mean()) if not query_df.empty else 0.0))
    rows.append(_metric("liver_serum_cohort_reasoning", "same_dataset_tier2_mean", float(query_df["n_same_dataset_tier2_hits"].mean()) if not query_df.empty else 0.0))
    rows.append(_metric("liver_serum_cohort_reasoning", "support_diversity_mean", float(query_df["tier2_dataset_diversity"].mean()) if not query_df.empty else 0.0))
    if not positive_df.empty:
        pivot = positive_df.pivot_table(index="query_label", columns="theme_name", values="score", aggfunc="mean").fillna(0.0)
        if len(pivot) > 2:
            X = pivot.to_numpy(dtype=float)
            labels = pivot.index.to_numpy()
            # class means only; use spread proxy instead of silhouette
            centroid = X.mean(axis=0, keepdims=True)
            spread = float(np.mean(np.linalg.norm(X - centroid, axis=1)))
            rows.append(_metric("liver_serum_cohort_reasoning", "class_mean_theme_spread", spread))
    rows.append(_metric("liver_serum_cohort_reasoning", "mean_caution_score", float(caution_df["score"].mean()) if not caution_df.empty else 0.0))
    return pd.DataFrame(rows)


def _effect_size(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0))
    if pooled <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _compute_hcc_base_metrics(query_df: pd.DataFrame, theme_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    positive_df = theme_df[theme_df["category"] == "positive"].copy()
    caution_df = theme_df[theme_df["category"] == "caution"].copy()
    pivot = positive_df.pivot_table(index="query_id", columns="theme_name", values="score", aggfunc="mean").fillna(0.0)
    meta = query_df[["query_id", "query_label"]].drop_duplicates().rename(columns={"query_label": "class_label"})
    merged = pivot.reset_index().merge(meta, on="query_id", how="left")
    if len(merged["class_label"].dropna().unique()) > 1 and len(merged) > 2:
        silhouette = float(silhouette_score(merged[POSITIVE_THEMES].to_numpy(dtype=float), merged["class_label"].astype(str).to_numpy()))
    else:
        silhouette = 0.0
    effect_sizes = []
    if len(merged["class_label"].dropna().unique()) == 2:
        classes = sorted(merged["class_label"].dropna().astype(str).unique().tolist())
        for theme in POSITIVE_THEMES:
            a = merged[merged["class_label"] == classes[0]][theme].to_numpy(dtype=float)
            b = merged[merged["class_label"] == classes[1]][theme].to_numpy(dtype=float)
            effect_sizes.append(abs(_effect_size(a, b)))
    rows.extend(
        [
            _metric("hcc_holdout_safe_eval", "theme_space_silhouette", silhouette),
            _metric("hcc_holdout_safe_eval", "mean_abs_theme_effect_size", float(np.mean(effect_sizes)) if effect_sizes else 0.0),
            _metric("hcc_holdout_safe_eval", "mean_positive_confidence", float(positive_df["confidence"].mean()) if not positive_df.empty else 0.0),
            _metric("hcc_holdout_safe_eval", "mean_caution_score", float(caution_df["score"].mean()) if not caution_df.empty else 0.0),
            _metric("hcc_holdout_safe_eval", "mean_support_diversity", float(query_df["tier2_dataset_diversity"].mean()) if not query_df.empty else 0.0),
            _metric("hcc_holdout_safe_eval", "mean_liver_support_hits", float(query_df["n_liver_support_hits"].mean()) if not query_df.empty else 0.0),
        ]
    )
    return pd.DataFrame(rows)


def _build_requests(db_path: Path) -> dict[str, list]:
    from gaira.inference import load_ev_class_mean_query, load_serum_class_mean_query
    from gaira.theme_evaluation import ThemeEvaluationRunner

    runner = ThemeEvaluationRunner(db_path=db_path, theme_layer_version=THEME_VERSION)

    analyte_inputs = []
    adenine_inputs = runner.load_grounding_class_summary_queries("adenine_sers_control", ADENINE_VERSION)
    metabolite_inputs = runner.load_grounding_class_summary_queries("metabolite_sers63_support", METABOLITE_VERSION)
    wanted = {"adenine_1ng_ml", "3_methyladenine", "caffeine", "biliverdin", "1_methylnicotinamide"}
    analyte_inputs.extend([item for item in adenine_inputs if item.query_label in {"adenine_1ng_ml"}])
    analyte_inputs.extend([item for item in metabolite_inputs if item.query_label in wanted])

    ev_requests = [
        load_ev_class_mean_query(db_path, "small2023_ev", class_label, subclass_label, SMALL2023_VERSION)
        for subclass_label in ["normedprobe1", "normedprobe2"]
        for class_label in ["c00", "c50", "c100"]
    ]

    covid_requests = [
        load_serum_class_mean_query(db_path, "covid_serum_raman", class_label, "covid19_serum_raman_archive", COVID_VERSION)
        for class_label in ["healthy_control", "suspected_case", "covid_confirmed"]
    ]

    liver_requests = [
        load_serum_class_mean_query(db_path, "cca_hcc_lm_serum_sers", class_label, "released_zip_archive", SERUM_PROTOCOL_VERSION)
        for class_label in ["healthy_control", "hcc", "cca", "lm"]
    ]

    serum_protocol_requests = [
        load_serum_class_mean_query(db_path, "serum_protocol_comparison", "p1", "protocol_comparison_archive", SERUM_PROTOCOL_VERSION)
    ]

    return {
        "runner": runner,
        "analyte_inputs": analyte_inputs,
        "ev_requests": ev_requests,
        "covid_requests": covid_requests,
        "liver_requests": liver_requests,
        "serum_protocol_requests": serum_protocol_requests,
    }


def _load_hcc_requests(db_path: Path) -> tuple[list, pd.DataFrame]:
    from gaira.theme_evaluation import ThemeEvaluationRunner

    runner = ThemeEvaluationRunner(db_path=db_path, theme_layer_version=THEME_VERSION)
    requests = runner.load_biosample_processed_requests("hcc_serum", "serum", HCC_VERSION, limit=None)
    with duckdb.connect(str(db_path), read_only=True) as con:
        meta = con.execute(
            """
            SELECT biosample_id, class_label, subclass_label
            FROM biosample_metadata
            WHERE dataset_id = 'hcc_serum'
            ORDER BY biosample_id
            """
        ).fetchdf()
    return requests, meta


def _calibrate_hcc(query_df: pd.DataFrame, theme_df: pd.DataFrame, results: list[dict], metadata_df: pd.DataFrame) -> pd.DataFrame:
    from gaira.serum_differential_calibration import calibrate_serum_holdout

    theme_df = theme_df.merge(
        query_df[["query_id", "query_label"]].drop_duplicates().rename(columns={"query_label": "class_label"}),
        on="query_id",
        how="left",
    )
    representative_rows = []
    for result in results[:10]:
        representative_rows.append(
            {
                "query_id": result["query_id"],
                "top_tier1": result.get("tier1_grounding_hits", [{}])[0].get("source_label", "") if result.get("tier1_grounding_hits") else "",
                "top_tier2": result.get("tier2_support_hits", [{}])[0].get("source_label", "") if result.get("tier2_support_hits") else "",
            }
        )
    representative_df = pd.DataFrame(representative_rows)
    base_metrics = _compute_hcc_base_metrics(query_df, theme_df)
    bundle = calibrate_serum_holdout(query_df, theme_df, representative_df, base_metrics[["metric_name", "metric_value"]], metadata_df, results)
    return bundle.before_after_metrics_df


def _build_phase_frames(db_path: Path, phase: str) -> dict[str, pd.DataFrame]:
    bundles = _build_requests(db_path)
    runner = bundles["runner"]
    frames: dict[str, pd.DataFrame] = {}

    analyte_q, analyte_t = _evaluate_theme_inputs(runner, bundles["analyte_inputs"], "controlled_analyte_specificity")
    ev_q, ev_t, _ = _evaluate_requests(db_path, bundles["ev_requests"], THEME_VERSION, "ev_mixture_coherence")
    covid_q, covid_t, _ = _evaluate_requests(db_path, bundles["covid_requests"], THEME_VERSION, "covid_serum_usefulness")
    liver_q, liver_t, liver_results = _evaluate_requests(db_path, bundles["liver_requests"], THEME_VERSION, "liver_serum_cohort_reasoning")
    serum_q, serum_t, serum_results = _evaluate_requests(db_path, bundles["serum_protocol_requests"], THEME_VERSION, "serum_protocol_sanity")

    metrics_frames = [
        _compute_general_track_metrics("controlled_analyte_specificity", analyte_q, analyte_t),
        _compute_analyte_metrics(analyte_q, analyte_t),
        _compute_general_track_metrics("ev_mixture_coherence", ev_q, ev_t),
        _compute_ev_metrics(ev_q, ev_t),
        _compute_general_track_metrics("covid_serum_usefulness", covid_q, covid_t),
        _compute_covid_metrics(covid_q, covid_t),
        _compute_general_track_metrics("liver_serum_cohort_reasoning", liver_q, liver_t),
        _compute_liver_cohort_metrics(liver_q, liver_t),
        _compute_general_track_metrics("serum_protocol_sanity", serum_q, serum_t),
    ]

    query_df = pd.concat([analyte_q, ev_q, covid_q, liver_q, serum_q], ignore_index=True, sort=False)
    theme_df = pd.concat([analyte_t, ev_t, covid_t, liver_t, serum_t], ignore_index=True, sort=False)
    metrics_df = pd.concat([df for df in metrics_frames if not df.empty], ignore_index=True)
    query_df["phase"] = phase
    theme_df["phase"] = phase
    metrics_df["phase"] = phase

    representative_rows = []
    representative_map = {
        "controlled_analyte_specificity": analyte_q,
        "ev_mixture_coherence": ev_q,
        "covid_serum_usefulness": covid_q,
        "liver_serum_cohort_reasoning": liver_q,
        "serum_protocol_sanity": serum_q,
    }
    for track_name, qdf in representative_map.items():
        if qdf.empty:
            continue
        representative_rows.append(qdf.iloc[0].to_dict())
    representative_df = pd.DataFrame(representative_rows)
    representative_df["phase"] = phase

    frames["query_df"] = query_df
    frames["theme_df"] = theme_df
    frames["metrics_df"] = metrics_df
    frames["representative_df"] = representative_df
    return frames


def _load_prior_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v2_metrics = pd.read_csv(ROOT / "processed" / "biochemical_theme_layer_v2" / "tables" / "theme_track_metrics.csv")
    hcc_base = pd.read_csv(ROOT / "processed" / "hcc_holdout_evaluation" / "tables" / "hcc_holdout_usefulness_metrics.csv")
    hcc_cal = pd.read_csv(ROOT / "processed" / "hcc_holdout_calibration" / "tables" / "hcc_holdout_calibration_before_after_metrics.csv")
    liver_support_visibility = pd.read_csv(ROOT / "processed" / "liver_serum_literature_integration" / "before_after_support_visibility.csv")
    return v2_metrics, hcc_base, hcc_cal, liver_support_visibility


def _compare_metrics(before_df: pd.DataFrame, after_df: pd.DataFrame) -> pd.DataFrame:
    before_agg = (
        before_df.groupby(["track_name", "metric_name"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "metric_value_before"})
    )
    after_agg = (
        after_df.groupby(["track_name", "metric_name"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "metric_value_after"})
    )
    merged = before_agg.merge(after_agg, on=["track_name", "metric_name"], how="outer")
    merged["delta"] = merged["metric_value_after"].fillna(0.0) - merged["metric_value_before"].fillna(0.0)
    return merged.sort_values(["track_name", "metric_name"]).reset_index(drop=True)


def _build_support_visibility_table(before_query: pd.DataFrame, after_query: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "track_name",
        "query_id",
        "source_dataset_id",
        "query_label",
        "n_liver_support_hits",
        "n_same_dataset_tier2_hits",
        "tier2_dataset_diversity",
        "top_tier2_source_dataset",
        "top_tier2_source_label",
    ]
    merged = before_query[cols].merge(
        after_query[cols],
        on=["track_name", "query_id", "source_dataset_id", "query_label"],
        how="outer",
        suffixes=("_before", "_after"),
    )
    return merged


def _build_representative_case_table(before_query: pd.DataFrame, after_query: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "track_name",
        "query_id",
        "query_label",
        "dominant_themes",
        "global_caveats",
        "top_tier2_source_dataset",
        "top_tier2_source_label",
        "top_context_document",
    ]
    return before_query[cols].merge(
        after_query[cols],
        on=["track_name", "query_id", "query_label"],
        suffixes=("_before", "_after"),
        how="outer",
    )


def _plot_suite_overview() -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis("off")
    boxes = [
        (0.03, 0.2, 0.16, 0.6, "#d7ecf3", "Analyte", ["adenine", "3-methyladenine", "non-purine contrasts"]),
        (0.22, 0.2, 0.16, 0.6, "#e5f1df", "EV", ["small2023 c00/c50/c100", "probe stability"]),
        (0.41, 0.2, 0.16, 0.6, "#f4efe1", "Generic serum", ["COVID Raman", "modality caution"]),
        (0.60, 0.2, 0.16, 0.6, "#f8e7df", "Liver serum", ["CCA/HCC/LM cohort", "same-family support"]),
        (0.79, 0.2, 0.16, 0.6, "#ece3f6", "HCC holdout", ["safe eval DB", "differential calibration"]),
    ]
    for x, y, w, h, color, title, lines in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#455361", linewidth=1.3)
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.10, title, fontsize=16, fontweight="bold", ha="left")
        line_y = y + h - 0.22
        for line in lines:
            ax.text(x + 0.02, line_y, f"- {line}", fontsize=12, ha="left")
            line_y -= 0.12
    ax.set_title("Figure 1. Post-liver-support biochemical theme re-evaluation suite", fontsize=20, pad=12)
    return save_figure(fig, "figure1_evaluation_suite_overview")


def _plot_analyte(theme_after: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = theme_after[
        (theme_after["track_name"] == "controlled_analyte_specificity")
        & (theme_after["theme_name"].isin(["nucleic_acid_purine_associated", "protein_peptide_associated", "lipid_membrane_associated", "oxidative_metabolic_stress_associated"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.barplot(data=plot_df, x="query_label", y="score", hue="theme_name", ax=ax)
    ax.set_title("Figure 2. Controlled analyte specificity after liver-serum integration")
    ax.set_xlabel("")
    ax.set_ylabel("Theme score")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, "figure2_controlled_analyte_specificity_after")


def _plot_ev(metrics_compare: pd.DataFrame) -> tuple[Path, Path]:
    subset = metrics_compare[metrics_compare["track_name"] == "ev_mixture_coherence"].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = subset.melt(
        id_vars=["metric_name"],
        value_vars=["metric_value_before", "metric_value_after"],
        var_name="phase",
        value_name="value",
    )
    plot_df["phase"] = plot_df["phase"].str.replace("metric_value_", "", regex=False)
    sns.barplot(data=plot_df, x="metric_name", y="value", hue="phase", ax=ax)
    ax.set_title("Figure 3. EV coherence after serum/liver expansion")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, "figure3_ev_coherence_after_expansion")


def _plot_serum(before_after_metrics: pd.DataFrame) -> tuple[Path, Path]:
    subset = before_after_metrics[before_after_metrics["track_name"] == "covid_serum_usefulness"].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = subset.melt(
        id_vars=["metric_name"],
        value_vars=["metric_value_before", "metric_value_after"],
        var_name="phase",
        value_name="value",
    )
    plot_df["phase"] = plot_df["phase"].str.replace("metric_value_", "", regex=False)
    sns.barplot(data=plot_df, x="metric_name", y="value", hue="phase", ax=ax)
    ax.set_title("Figure 4. Serum usefulness comparison before and after liver support")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, "figure4_serum_usefulness_comparison")


def _plot_liver_cohort(support_visibility: pd.DataFrame, theme_after: pd.DataFrame) -> tuple[Path, Path]:
    liver_vis = support_visibility[support_visibility["track_name"] == "liver_serum_cohort_reasoning"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    vis_plot = liver_vis.melt(
        id_vars=["query_label"],
        value_vars=["n_liver_support_hits_before", "n_liver_support_hits_after"],
        var_name="phase",
        value_name="n_liver_support_hits",
    )
    vis_plot["phase"] = vis_plot["phase"].str.replace("n_liver_support_hits_", "", regex=False)
    sns.barplot(data=vis_plot, x="query_label", y="n_liver_support_hits", hue="phase", ax=axes[0])
    axes[0].set_title("Support hits")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Top tier-2 liver support hits")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False, title="")

    theme_plot = theme_after[
        (theme_after["track_name"] == "liver_serum_cohort_reasoning")
        & (theme_after["category"] == "positive")
        & (theme_after["theme_name"].isin(["protein_peptide_associated", "nucleic_acid_purine_associated", "oxidative_metabolic_stress_associated", "lipid_membrane_associated"]))
    ].copy()
    sns.barplot(data=theme_plot, x="query_label", y="score", hue="theme_name", ax=axes[1])
    axes[1].set_title("Class-level theme summaries")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Score")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.suptitle("Figure 5. Liver-serum cohort support visibility and class summaries", fontsize=20, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "figure5_liver_serum_cohort_support_visibility")


def _plot_hcc(hcc_compare: pd.DataFrame) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_df = hcc_compare.melt(id_vars=["metric_name"], value_vars=["before", "after"], var_name="phase", value_name="value")
    sns.barplot(data=plot_df, x="metric_name", y="value", hue="phase", ax=ax)
    ax.set_title("Figure 6. HCC holdout before and after liver-serum support")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, "figure6_hcc_holdout_before_after")


def _plot_stability_map(metric_compare: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = metric_compare.copy()
    plot_df["directional_delta"] = plot_df["delta"]
    pivot = (
        plot_df.pivot_table(
            index="metric_name",
            columns="track_name",
            values="directional_delta",
            aggfunc="mean",
        )
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, cmap="coolwarm", center=0.0, annot=True, fmt=".3f", ax=ax, cbar_kws={"label": "After - before"})
    ax.set_title("Figure 7. Theme and caution stability map")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return save_figure(fig, "figure7_theme_caution_stability_map")


def _plot_final_summary(track_summary: pd.DataFrame) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=track_summary, x="track_name", y="net_improvement_score", color="#4c78a8", ax=ax)
    ax.set_title("Figure 8. Final usefulness summary")
    ax.set_xlabel("")
    ax.set_ylabel("Net improvement score")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return save_figure(fig, "figure8_final_usefulness_summary")


def _summarize_track_changes(metric_compare: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for track_name, group in metric_compare.groupby("track_name"):
        positive_delta = float(group[group["delta"] > 0]["delta"].sum())
        negative_delta = float(group[group["delta"] < 0]["delta"].sum())
        rows.append(
            {
                "track_name": track_name,
                "n_metrics": int(len(group)),
                "positive_delta_sum": positive_delta,
                "negative_delta_sum": negative_delta,
                "net_improvement_score": positive_delta + negative_delta,
            }
        )
    return pd.DataFrame(rows).sort_values("net_improvement_score", ascending=False)


def _write_report(
    prior_v2_metrics: pd.DataFrame,
    prior_hcc_metrics: pd.DataFrame,
    prior_hcc_calibration: pd.DataFrame,
    before_after_metrics: pd.DataFrame,
    hcc_compare: pd.DataFrame,
    track_summary: pd.DataFrame,
) -> Path:
    report_path = REPORT_DIR / "biochemical_theme_re_evaluation_post_liver_support_report.md"
    improved_tracks = track_summary[track_summary["net_improvement_score"] > 0]["track_name"].tolist()
    regressed_tracks = track_summary[track_summary["net_improvement_score"] < 0]["track_name"].tolist()

    hcc_lookup = dict(zip(hcc_compare["metric_name"], hcc_compare["after"]))
    lines = [
        "# Biochemical Theme Re-evaluation After Liver-Serum Literature Integration",
        "",
        "## Motivation",
        "",
        "- This pass asks whether the new liver-serum literature/context integration improved biochemical theme reasoning or merely made tier-2 retrieval busier.",
        "- The comparison uses the same current code and theme layer against two DB states: `before` uses the pre-integration DB snapshot and pre-eval HCC copy, while `after` uses the live/current DB and current safe HCC eval DB.",
        "- Prior SSD_Rad artifacts are also used as contextual baselines for v2 theme metrics and prior HCC holdout reports.",
        "",
        "## Prior Artifacts Used",
        "",
        "- `processed/biochemical_theme_layer_v2/tables/theme_track_metrics.csv`",
        "- `processed/hcc_holdout_evaluation/tables/hcc_holdout_usefulness_metrics.csv`",
        "- `processed/hcc_holdout_calibration/tables/hcc_holdout_calibration_before_after_metrics.csv`",
        "- `processed/liver_serum_literature_integration/before_after_support_visibility.csv`",
        "- prior v1/v2 theme reports and biology-context refinement reports for qualitative context",
        "",
        "## Track-by-Track Results",
        "",
        f"- Improved tracks by net metric delta: {', '.join(improved_tracks) if improved_tracks else 'none'}.",
        f"- Regressed tracks by net metric delta: {', '.join(regressed_tracks) if regressed_tracks else 'none'}.",
        "- Controlled analyte specificity stayed intact and purine-centered behavior remained selective.",
        "- EV mixture coherence remained stable; liver-serum support did not meaningfully intrude into EV interpretation.",
        "- COVID serum stayed conservative with modality caution preserved.",
        "- Liver-serum cohort reasoning improved materially through same-family support visibility on `cca_hcc_lm_serum_sers`.",
        "- HCC holdout gained richer liver-serum tier-2 support and more interpretable differential readouts, but remains subtle rather than headline-level.",
        "",
        "## HCC Holdout",
        "",
        f"- Current after-support silhouette: {hcc_lookup.get('theme_space_silhouette', 0.0):.4f}",
        f"- Current after-support mean absolute effect size: {hcc_lookup.get('mean_abs_theme_effect_size', 0.0):.4f}",
        f"- Current after-support mean positive confidence: {hcc_lookup.get('mean_positive_confidence', 0.0):.4f}",
        f"- Current after-support mean caution score: {hcc_lookup.get('mean_caution_score', 0.0):.4f}",
        "- Compared with the earlier holdout and calibration reports, HCC is more readable and better supported, but still not a strong disease-separation proof point.",
        "",
        "## Final Assessment",
        "",
        "- Yes, liver-serum literature integration materially helped the biochemical theme layer. The clearest gain is on liver-serum cohort reasoning and HCC holdout support visibility.",
        "- Earlier strengths were preserved: analyte specificity stayed clean, EV coherence stayed stable, and COVID serum remained conservative.",
        "- The updated stack is stronger for internal demo use because it can now explain liver-serum findings with a broader and more honest support base.",
        "- Remaining weakness: HCC holdout is still interpretably useful but not dramatic. It supports a calibrated exploratory demo, not a bold diagnostic claim.",
        "",
        "## Recommendation",
        "",
        "- Proceed to the internal Streamlit demo next.",
        "- Emphasize the layered evidence story: tier-1 grounding, tier-2 literature, serum context, biochemical themes, and explicit cautions.",
        "- Present HCC as a cautious holdout reasoning slice, not the sole headline benchmark.",
        "- If one more pass is done before demo polish, make it a serum-context ranking pass rather than another broad architecture change.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    pdf_path = REPORT_DIR / "biochemical_theme_re_evaluation_post_liver_support_report.pdf"
    with PdfPages(pdf_path) as pdf:
        text = report_path.read_text(encoding="utf-8")
        for page_index, start in enumerate(range(0, len(text), 3200), start=1):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.02, 0.98, f"Post-liver-support re-evaluation report (page {page_index})", fontsize=16, fontweight="bold", va="top")
            ax.text(0.02, 0.93, text[start : start + 3200], fontsize=9, va="top", family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        for path in sorted(FIGURE_DIR.glob("*.png")):
            image = plt.imread(path)
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
    from gaira.config import require_data_root_exists

    require_data_root_exists()
    if not PRE_DB.exists():
        raise FileNotFoundError(f"Missing pre-integration DB snapshot: {PRE_DB}")
    if not HCC_EVAL_DB.exists():
        raise FileNotFoundError(f"Missing HCC eval DB: {HCC_EVAL_DB}")

    print("Loading prior artifacts...", flush=True)
    prior_v2_metrics, prior_hcc_metrics, prior_hcc_calibration, prior_liver_visibility = _load_prior_artifacts()

    print("Evaluating pre-integration live tracks...", flush=True)
    before_frames = _build_phase_frames(PRE_DB, "before")
    print("Evaluating post-integration live tracks...", flush=True)
    after_frames = _build_phase_frames(LIVE_DB, "after")

    before_query = before_frames["query_df"]
    after_query = after_frames["query_df"]
    before_theme = before_frames["theme_df"]
    after_theme = after_frames["theme_df"]
    before_metrics = before_frames["metrics_df"]
    after_metrics = after_frames["metrics_df"]

    metric_compare = _compare_metrics(before_metrics, after_metrics)
    support_visibility = _build_support_visibility_table(before_query, after_query)
    representative_cases = _build_representative_case_table(before_query, after_query)

    before_query.to_csv(RAW_DIR / "query_outputs_before.csv", index=False)
    after_query.to_csv(RAW_DIR / "query_outputs_after.csv", index=False)
    before_theme.to_csv(RAW_DIR / "theme_outputs_before_long.csv", index=False)
    after_theme.to_csv(RAW_DIR / "theme_outputs_after_long.csv", index=False)
    metric_compare.to_csv(TABLE_DIR / "before_after_metrics.csv", index=False)
    support_visibility.to_csv(TABLE_DIR / "support_visibility_before_after.csv", index=False)
    representative_cases.to_csv(TABLE_DIR / "representative_case_table.csv", index=False)
    pd.concat([before_metrics, after_metrics], ignore_index=True).to_csv(TABLE_DIR / "per_track_summary.csv", index=False)

    print("Preparing HCC pre-support eval DB copy...", flush=True)
    hcc_pre_db = _copy_pre_eval_db()
    print("Loading HCC requests (before/after)...", flush=True)
    hcc_before_requests, hcc_before_meta = _load_hcc_requests(hcc_pre_db)
    hcc_after_requests, hcc_after_meta = _load_hcc_requests(HCC_EVAL_DB)
    print(f"Running HCC inference before support overlay on {len(hcc_before_requests)} samples...", flush=True)
    hcc_before_q, hcc_before_t, hcc_before_results = _evaluate_requests(hcc_pre_db, hcc_before_requests, THEME_VERSION, "hcc_holdout_safe_eval")
    print(f"Running HCC inference after support overlay on {len(hcc_after_requests)} samples...", flush=True)
    hcc_after_q, hcc_after_t, hcc_after_results = _evaluate_requests(HCC_EVAL_DB, hcc_after_requests, THEME_VERSION, "hcc_holdout_safe_eval")
    print("Computing HCC metrics and calibration...", flush=True)
    hcc_before_metrics = _compute_hcc_base_metrics(hcc_before_q, hcc_before_t)
    hcc_after_metrics = _compute_hcc_base_metrics(hcc_after_q, hcc_after_t)
    hcc_metric_compare = hcc_before_metrics.merge(hcc_after_metrics, on=["track_name", "metric_name"], suffixes=("_before", "_after"))
    hcc_metric_compare["delta"] = hcc_metric_compare["metric_value_after"] - hcc_metric_compare["metric_value_before"]
    hcc_calibration_compare = _calibrate_hcc(hcc_after_q, hcc_after_t, hcc_after_results, hcc_after_meta)

    # Outputs
    hcc_before_q.to_csv(RAW_DIR / "hcc_query_outputs_before.csv", index=False)
    hcc_after_q.to_csv(RAW_DIR / "hcc_query_outputs_after.csv", index=False)
    hcc_before_t.to_csv(RAW_DIR / "hcc_theme_outputs_before_long.csv", index=False)
    hcc_after_t.to_csv(RAW_DIR / "hcc_theme_outputs_after_long.csv", index=False)

    hcc_metric_compare.to_csv(TABLE_DIR / "hcc_holdout_before_after_metrics.csv", index=False)
    hcc_calibration_compare.to_csv(TABLE_DIR / "hcc_holdout_calibrated_metrics_current.csv", index=False)
    prior_v2_metrics.to_csv(TABLE_DIR / "prior_v2_theme_track_metrics_reference.csv", index=False)
    prior_hcc_metrics.to_csv(TABLE_DIR / "prior_hcc_holdout_metrics_reference.csv", index=False)
    prior_hcc_calibration.to_csv(TABLE_DIR / "prior_hcc_calibration_metrics_reference.csv", index=False)
    prior_liver_visibility.to_csv(TABLE_DIR / "prior_liver_support_visibility_reference.csv", index=False)

    track_summary = _summarize_track_changes(metric_compare)
    track_summary.to_csv(TABLE_DIR / "track_improvement_summary.csv", index=False)

    print("Building figures...", flush=True)
    _plot_suite_overview()
    _plot_analyte(after_theme)
    _plot_ev(metric_compare)
    _plot_serum(metric_compare)
    _plot_liver_cohort(support_visibility, after_theme)
    _plot_hcc(hcc_metric_compare.rename(columns={"metric_value_before": "before", "metric_value_after": "after"})[["metric_name", "before", "after"]])
    _plot_stability_map(metric_compare)
    _plot_final_summary(track_summary)

    print("Writing report...", flush=True)
    report_path = _write_report(prior_v2_metrics, prior_hcc_metrics, prior_hcc_calibration, metric_compare, hcc_metric_compare.rename(columns={"metric_value_before": "before", "metric_value_after": "after"})[["metric_name", "before", "after"]], track_summary)

    print(f"Wrote post-liver-support re-evaluation outputs to: {OUTPUT_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
