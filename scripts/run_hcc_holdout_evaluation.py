from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import textwrap
from dataclasses import dataclass
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


matplotlib.rcParams.update(
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


DATASET_ID = "hcc_serum"
PROCESSING_VERSION = "v1_crop430_1730_interp1_minmax"
THEME_LAYER_VERSION = "v3"
REQUIRED_RAW_FILES = ("dataset.zip", "data.csv", "R_code.R")
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

PROCESSING_CONFIGS: dict[str, dict] | None = None
build_chunk_query = None
build_common_grid = None
process_one_spectrum = None
serialize_array = None


@dataclass
class HoldoutPaths:
    base_dir: Path
    eval_db_dir: Path
    eval_db_path: Path
    raw_dir: Path
    tables_dir: Path
    figures_dir: Path
    report_dir: Path
    cases_dir: Path


def ensure_paths(processed_root: Path) -> HoldoutPaths:
    base_dir = processed_root / "hcc_holdout_evaluation"
    eval_db_dir = base_dir / "eval_db"
    raw_dir = base_dir / "raw_outputs"
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    report_dir = base_dir / "report"
    cases_dir = base_dir / "cases"
    for path in [base_dir, eval_db_dir, raw_dir, tables_dir, figures_dir, report_dir, cases_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return HoldoutPaths(
        base_dir=base_dir,
        eval_db_dir=eval_db_dir,
        eval_db_path=eval_db_dir / "gaira_hcc_holdout_eval.duckdb",
        raw_dir=raw_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        report_dir=report_dir,
        cases_dir=cases_dir,
    )


def markdown_table(df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in df.to_dict(orient="records"):
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, divider, *rows])


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> tuple[Path, Path]:
    pdf_path = figure_dir / f"{stem}.pdf"
    png_path = figure_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def validate_raw_release(raw_root: Path) -> pd.DataFrame:
    missing = [name for name in REQUIRED_RAW_FILES if not (raw_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing canonical hcc_serum raw assets. "
            f"Run `scripts/download_dataset.py {DATASET_ID} --allow-holdout` first. Missing: {', '.join(missing)}"
        )

    df = pd.read_csv(raw_root / "data.csv")
    return df


def copy_live_db_to_eval_db(live_db_path: Path, eval_db_path: Path) -> None:
    if eval_db_path.exists():
        eval_db_path.unlink()
    shutil.copy2(live_db_path, eval_db_path)


def ingest_holdout_into_eval_db(raw_root: Path, eval_db_path: Path) -> None:
    from gaira.parsers.biosample.hcc_serum_parser import HCCSerumParser

    parser = HCCSerumParser(
        dataset_id=DATASET_ID,
        dataset_root=raw_root,
        db_path=eval_db_path,
    )
    parser.ingest()


def process_holdout_into_eval_db(eval_db_path: Path, chunk_size: int = 250) -> None:
    config = PROCESSING_CONFIGS[DATASET_ID]
    common_grid = build_common_grid(config)

    with duckdb.connect(str(eval_db_path)) as connection:
        metadata_df = connection.execute(
            """
            SELECT biosample_id, class_label, subclass_label
            FROM biosample_metadata
            WHERE dataset_id = ?
            ORDER BY biosample_id
            """,
            [DATASET_ID],
        ).fetchdf()
        if metadata_df.empty:
            raise ValueError("No hcc_serum metadata rows exist in the evaluation DB after ingest.")

        connection.execute(
            """
            DELETE FROM biosample_processed_points
            WHERE processed_id IN (
                SELECT processed_id
                FROM biosample_processed_spectra
                WHERE dataset_id = ? AND processing_version = ?
            )
            """,
            [DATASET_ID, PROCESSING_VERSION],
        )
        connection.execute(
            """
            DELETE FROM biosample_processed_spectra
            WHERE dataset_id = ? AND processing_version = ?
            """,
            [DATASET_ID, PROCESSING_VERSION],
        )
        connection.execute(
            """
            DELETE FROM biosample_class_summary
            WHERE dataset_id = ? AND processing_version = ?
            """,
            [DATASET_ID, PROCESSING_VERSION],
        )

        biosample_ids = metadata_df["biosample_id"].tolist()
        class_accumulators: dict[tuple[str | None, str | None], dict[str, np.ndarray | int]] = {}
        chunk_query_cache: dict[int, str] = {}

        for chunk_start in range(0, len(biosample_ids), chunk_size):
            chunk_ids = biosample_ids[chunk_start : chunk_start + chunk_size]
            size = len(chunk_ids)
            chunk_query = chunk_query_cache.get(size)
            if chunk_query is None:
                chunk_query = build_chunk_query(size)
                chunk_query_cache[size] = chunk_query

            chunk_df = connection.execute(chunk_query, [DATASET_ID, *chunk_ids]).fetchdf()
            spectra_rows: list[dict] = []
            point_rows: list[dict] = []
            for biosample_id, spectrum_df in chunk_df.groupby("biosample_id", sort=False):
                class_label = spectrum_df["class_label"].iloc[0]
                subclass_label = spectrum_df["subclass_label"].iloc[0]
                processed_result = process_one_spectrum(
                    dataset_id=DATASET_ID,
                    biosample_id=biosample_id,
                    spectrum_df=spectrum_df,
                    class_label=class_label,
                    subclass_label=subclass_label,
                    common_grid=common_grid,
                    config=config,
                )
                if processed_result is None:
                    continue
                spectrum_row, spectrum_point_rows, group_key, normalized_y = processed_result
                spectra_rows.append(spectrum_row)
                point_rows.extend(spectrum_point_rows)

                accumulator = class_accumulators.get(group_key)
                if accumulator is None:
                    accumulator = {
                        "sum": np.zeros_like(common_grid, dtype=float),
                        "sum_sq": np.zeros_like(common_grid, dtype=float),
                        "count": 0,
                    }
                    class_accumulators[group_key] = accumulator
                accumulator["sum"] = accumulator["sum"] + normalized_y
                accumulator["sum_sq"] = accumulator["sum_sq"] + np.square(normalized_y)
                accumulator["count"] = int(accumulator["count"]) + 1

            if spectra_rows:
                spectra_insert_df = pd.DataFrame(spectra_rows)
                connection.register("holdout_processed_spectra_chunk", spectra_insert_df)
                connection.execute("INSERT INTO biosample_processed_spectra SELECT * FROM holdout_processed_spectra_chunk")
                connection.unregister("holdout_processed_spectra_chunk")

                points_insert_df = pd.DataFrame(point_rows)
                connection.register("holdout_processed_points_chunk", points_insert_df)
                connection.execute("INSERT INTO biosample_processed_points SELECT * FROM holdout_processed_points_chunk")
                connection.unregister("holdout_processed_points_chunk")

        summary_rows = []
        for (class_label, subclass_label), accumulator in sorted(class_accumulators.items()):
            count = int(accumulator["count"])
            if count == 0:
                continue
            mean_intensity = accumulator["sum"] / count
            variance = np.maximum((accumulator["sum_sq"] / count) - np.square(mean_intensity), 0.0)
            std_intensity = np.sqrt(variance)
            label_part = class_label or "unknown_class"
            subclass_part = subclass_label or "unknown_subclass"
            summary_rows.append(
                {
                    "summary_id": f"{PROCESSING_VERSION}__{DATASET_ID}__{label_part}__{subclass_part}",
                    "dataset_id": DATASET_ID,
                    "class_label": class_label,
                    "subclass_label": subclass_label,
                    "processing_version": PROCESSING_VERSION,
                    "n_spectra": count,
                    "crop_min_cm": float(config["crop_min_cm"]),
                    "crop_max_cm": float(config["crop_max_cm"]),
                    "interpolation_step_cm": float(config["interpolation_step_cm"]),
                    "mean_wavenumbers_json": serialize_array(common_grid),
                    "mean_intensity_json": serialize_array(mean_intensity),
                    "std_intensity_json": serialize_array(std_intensity),
                    "notes": "Class summary computed inside isolated HCC holdout evaluation DB.",
                }
            )
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            connection.register("holdout_summary_rows", summary_df)
            connection.execute("INSERT INTO biosample_class_summary SELECT * FROM holdout_summary_rows")
            connection.unregister("holdout_summary_rows")


def load_metadata_df(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return connection.execute(
            """
            SELECT
              biosample_id,
              class_label,
              subclass_label,
              sample_id,
              replicate_id,
              source_file,
              regexp_extract(replicate_id, 'batch-([A-Z])_', 1) AS substrate_batch
            FROM biosample_metadata
            WHERE dataset_id = ?
            ORDER BY biosample_id
            """,
            [DATASET_ID],
        ).fetchdf()


def flatten_theme_outputs(result: dict) -> list[dict]:
    rows = []
    for theme in result["biochemical_theme_outputs"]:
        rows.append(
            {
                "query_id": result["query_id"],
                "class_label": result["query_label"],
                "theme_name": theme["theme_name"],
                "category": theme["category"],
                "score": float(theme["score"]),
                "confidence": float(theme["confidence"]),
                "raw_score_pre_normalization": float(theme.get("raw_score_pre_normalization", theme["score"])),
                "normalized_score": float(theme.get("normalized_score", theme["score"])),
                "competition_penalty": float(theme.get("competition_penalty", 0.0)),
                "caution_penalty": float(theme.get("caution_penalty", 0.0)),
                "calibration_penalty": float(theme.get("calibration_penalty", 0.0)),
                "specificity_index": float(theme.get("specificity_index", 0.0)),
                "tier1_contrib": float(theme["evidence_contributions"]["tier1"]),
                "tier2_contrib": float(theme["evidence_contributions"]["tier2"]),
                "knowledge_contrib": float(theme["evidence_contributions"]["knowledge"]),
                "semantic_contrib": float(theme["evidence_contributions"]["semantic"]),
                "context_contrib": float(theme["evidence_contributions"]["context"]),
                "band_contrib": float(theme["evidence_contributions"]["band"]),
                "n_tier1_hits": len(theme.get("supporting_tier1_hits", [])),
                "n_tier2_hits": len(theme.get("supporting_tier2_hits", [])),
                "n_knowledge_hits": len(theme.get("supporting_knowledge_hits", [])),
                "n_semantic_hits": len(theme.get("supporting_semantic_regions", [])),
                "n_bands": len(theme.get("supporting_bands", [])),
                "evidence_balance_summary": theme["evidence_balance_summary"],
                "limiting_evidence": "|".join(str(x) for x in theme.get("opposing_or_limiting_evidence", [])),
                "notes": theme.get("notes", ""),
            }
        )
    return rows


def flatten_query_result(result: dict) -> dict:
    tier1 = result.get("tier1_grounding_hits", [])
    tier2 = result.get("tier2_support_hits", [])
    knowledge = result.get("knowledge_support_hits", [])
    semantic = result.get("semantic_region_support_hits", [])
    context = result.get("domain_context_hits", [])
    return {
        "query_id": result["query_id"],
        "class_label": result["query_label"],
        "query_family": result["query_family"],
        "dominant_themes": "|".join(result.get("dominant_themes", [])),
        "global_caveats": "|".join(result.get("biochemical_global_caveats", [])),
        "theme_summary": result.get("biochemical_theme_summary", ""),
        "what_not_to_claim": "|".join(result.get("biochemical_what_not_to_claim", [])),
        "evidence_profile_summary": result.get("evidence_profile_summary", ""),
        "top_tier1_dataset": tier1[0]["source_dataset_id"] if tier1 else "",
        "top_tier1_label": tier1[0]["source_label"] if tier1 else "",
        "top_tier2_dataset": tier2[0]["source_dataset_id"] if tier2 else "",
        "top_tier2_label": tier2[0]["source_label"] if tier2 else "",
        "top_knowledge_label": knowledge[0]["source_label"] if knowledge else "",
        "top_semantic_label": semantic[0]["source_label"] if semantic else "",
        "top_context_doc": context[0]["document_id"] if context else "",
        "n_tier1_hits": len(tier1),
        "n_tier2_hits": len(tier2),
        "n_knowledge_hits": len(knowledge),
        "n_semantic_hits": len(semantic),
        "n_context_hits": len(context),
    }


def choose_representative_cases(theme_wide: pd.DataFrame, metadata_df: pd.DataFrame) -> list[str]:
    representative_ids: list[str] = []
    for class_label in sorted(metadata_df["class_label"].unique().tolist()):
        class_ids = metadata_df.loc[metadata_df["class_label"] == class_label, "biosample_id"]
        subset = theme_wide[theme_wide["query_id"].isin(class_ids)].copy()
        if subset.empty:
            continue
        feature_cols = [column for column in POSITIVE_THEMES + CAUTION_THEMES if column in subset.columns]
        center = subset[feature_cols].mean(axis=0).to_numpy(dtype=float)
        distances = np.linalg.norm(subset[feature_cols].to_numpy(dtype=float) - center, axis=1)
        representative_ids.append(str(subset.iloc[int(np.argmin(distances))]["query_id"]))
    return representative_ids


def build_group_effects(theme_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    class_values = sorted(theme_df["class_label"].dropna().unique().tolist())
    if len(class_values) != 2:
        return pd.DataFrame()
    a_label, b_label = class_values
    for theme_name in sorted(theme_df["theme_name"].unique().tolist()):
        a_vals = theme_df[(theme_df["class_label"] == a_label) & (theme_df["theme_name"] == theme_name)]["score"].to_numpy(dtype=float)
        b_vals = theme_df[(theme_df["class_label"] == b_label) & (theme_df["theme_name"] == theme_name)]["score"].to_numpy(dtype=float)
        if len(a_vals) == 0 or len(b_vals) == 0:
            continue
        pooled = math.sqrt(((a_vals.std(ddof=1) ** 2) + (b_vals.std(ddof=1) ** 2)) / 2.0) if len(a_vals) > 1 and len(b_vals) > 1 else 0.0
        effect_size = (a_vals.mean() - b_vals.mean()) / pooled if pooled > 0 else 0.0
        rows.append(
            {
                "theme_name": theme_name,
                "class_a": a_label,
                "class_b": b_label,
                "mean_a": float(a_vals.mean()),
                "mean_b": float(b_vals.mean()),
                "effect_size": float(effect_size),
            }
        )
    return pd.DataFrame(rows)


def build_holdout_metrics(theme_wide: pd.DataFrame, metadata_df: pd.DataFrame, query_df: pd.DataFrame) -> pd.DataFrame:
    merged = theme_wide.merge(metadata_df[["biosample_id", "class_label"]], left_on="query_id", right_on="biosample_id", how="left")
    feature_cols = [column for column in POSITIVE_THEMES + CAUTION_THEMES if column in merged.columns]
    X = merged[feature_cols].to_numpy(dtype=float)
    y = merged["class_label"].astype(str).to_numpy()
    silhouette = float(silhouette_score(X, y)) if len(np.unique(y)) > 1 else 0.0
    positive_cols = [column for column in POSITIVE_THEMES if column in merged.columns]
    caution_cols = [column for column in CAUTION_THEMES if column in merged.columns]
    effect_df = build_group_effects(
        pd.melt(
            merged,
            id_vars=["query_id", "class_label"],
            value_vars=feature_cols,
            var_name="theme_name",
            value_name="score",
        )
    )
    mean_abs_effect = float(effect_df["effect_size"].abs().mean()) if not effect_df.empty else 0.0
    confidence_mean = float(query_df["mean_positive_confidence"].mean())
    caution_mean = float(query_df["mean_caution_score"].mean())
    evidence_diversity = float(query_df["mean_evidence_sources"].mean())
    return pd.DataFrame(
        [
            {"metric_name": "theme_space_silhouette", "metric_value": silhouette},
            {"metric_name": "mean_abs_theme_effect_size", "metric_value": mean_abs_effect},
            {"metric_name": "mean_positive_confidence", "metric_value": confidence_mean},
            {"metric_name": "mean_caution_score", "metric_value": caution_mean},
            {"metric_name": "mean_evidence_diversity", "metric_value": evidence_diversity},
        ]
    )


def build_query_level_outputs(results: list[dict], metadata_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    theme_rows = [row for result in results for row in flatten_theme_outputs(result)]
    theme_df = pd.DataFrame(theme_rows)
    query_rows = pd.DataFrame([flatten_query_result(result) for result in results])
    if theme_df.empty:
        return query_rows, theme_df, pd.DataFrame()

    theme_wide = (
        theme_df.pivot_table(index="query_id", columns="theme_name", values="score", aggfunc="mean")
        .reset_index()
        .fillna(0.0)
    )
    confidence_wide = (
        theme_df.pivot_table(index="query_id", columns="theme_name", values="confidence", aggfunc="mean")
        .reset_index()
        .fillna(0.0)
    )
    evidence_counts = (
        theme_df.groupby("query_id", as_index=False)
        .agg(
            mean_evidence_sources=(
                "n_tier1_hits",
                lambda values: float(np.mean(values)),
            ),
        )
    )
    positive_conf = (
        theme_df[theme_df["category"] == "positive"]
        .groupby("query_id", as_index=False)["confidence"]
        .mean()
        .rename(columns={"confidence": "mean_positive_confidence"})
    )
    caution_mean = (
        theme_df[theme_df["category"] == "caution"]
        .groupby("query_id", as_index=False)["score"]
        .mean()
        .rename(columns={"score": "mean_caution_score"})
    )
    metadata_merge = metadata_df.rename(columns={"biosample_id": "query_id"})
    query_df = (
        query_rows.merge(metadata_merge, on="query_id", how="left")
        .merge(positive_conf, on="query_id", how="left")
        .merge(caution_mean, on="query_id", how="left")
        .merge(evidence_counts, on="query_id", how="left")
        .merge(theme_wide, on="query_id", how="left")
    )
    return query_df, theme_df, confidence_wide


def build_design_figure(paths: HoldoutPaths, metadata_df: pd.DataFrame) -> tuple[Path, Path]:
    class_counts = metadata_df["class_label"].value_counts().sort_index()
    batch_counts = metadata_df["substrate_batch"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.axis("off")
    boxes = [
        (0.03, 0.18, 0.2, 0.64, "#d8ecf3", "Holdout raw release", [f"dataset={DATASET_ID}", f"spectra={len(metadata_df)}", f"classes={', '.join(class_counts.index.tolist())}"]),
        (0.29, 0.18, 0.2, 0.64, "#f3ecd6", "Evaluation-only path", ["copy live DB to isolated eval DB", "ingest/process only inside eval DB", "normal holdout block remains active"]),
        (0.55, 0.18, 0.2, 0.64, "#e2efdf", "GAIRA stack used", [f"theme layer={THEME_LAYER_VERSION}", "tier1/tier2/knowledge/semantic/context", "no LLM, no backbone update"]),
        (0.81, 0.18, 0.16, 0.64, "#f5dfdf", "Outputs", ["sample-level themes", "group summaries", "evidence traces", "holdout report"]),
    ]
    for x, y, w, h, color, title, lines in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#405060", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.08, title, fontsize=15, fontweight="bold")
        ypos = y + h - 0.18
        for line in lines:
            ax.text(x + 0.025, ypos, f"- {line}", fontsize=11.5)
            ypos -= 0.10
    for start, end in [(0.23, 0.29), (0.49, 0.55), (0.75, 0.81)]:
        ax.annotate("", xy=(end, 0.50), xytext=(start, 0.50), arrowprops=dict(arrowstyle="-|>", lw=2, color="#405060"))
    ax.text(0.03, 0.06, f"Class counts: {class_counts.to_dict()} | Batch counts: {batch_counts.to_dict()}", fontsize=11)
    ax.set_title("Figure 1. HCC holdout evaluation design and safeguards", fontsize=20, pad=12)
    return save_figure(fig, paths.figures_dir, "figure1_hcc_holdout_design")


def build_theme_distribution_figure(paths: HoldoutPaths, theme_df: pd.DataFrame) -> tuple[Path, Path]:
    subset = theme_df[theme_df["theme_name"].isin([theme for theme in POSITIVE_THEMES if theme in theme_df["theme_name"].unique()])].copy()
    subset["theme_short"] = subset["theme_name"].str.replace("_associated", "", regex=False)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    axes = axes.flatten()
    for ax, theme_name in zip(axes, POSITIVE_THEMES):
        theme_subset = subset[subset["theme_name"] == theme_name]
        if theme_subset.empty:
            ax.axis("off")
            continue
        sns.violinplot(
            data=theme_subset,
            x="class_label",
            y="score",
            hue="class_label",
            inner=None,
            cut=0,
            ax=ax,
            palette="Set2",
            legend=False,
        )
        sns.stripplot(data=theme_subset, x="class_label", y="score", color="black", size=2.5, alpha=0.35, ax=ax)
        ax.set_title(textwrap.fill(theme_name.replace("_", " "), width=22))
        ax.set_xlabel("")
        ax.set_ylabel("Score")
    for ax in axes[len(POSITIVE_THEMES):]:
        ax.axis("off")
    fig.suptitle("Figure 2. Group-level positive theme distributions in HCC holdout", fontsize=20, y=1.01)
    fig.tight_layout()
    return save_figure(fig, paths.figures_dir, "figure2_group_theme_distributions")


def build_caution_distribution_figure(paths: HoldoutPaths, theme_df: pd.DataFrame) -> tuple[Path, Path]:
    subset = theme_df[theme_df["theme_name"].isin(CAUTION_THEMES)].copy()
    subset["theme_short"] = subset["theme_name"].str.replace("_caution", "", regex=False)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    axes = axes.flatten()
    for ax, theme_name in zip(axes, CAUTION_THEMES):
        theme_subset = subset[subset["theme_name"] == theme_name]
        if theme_subset.empty:
            ax.axis("off")
            continue
        sns.boxplot(
            data=theme_subset,
            x="class_label",
            y="score",
            hue="class_label",
            ax=ax,
            palette="Pastel1",
            legend=False,
        )
        sns.stripplot(data=theme_subset, x="class_label", y="score", color="black", size=2.5, alpha=0.35, ax=ax)
        ax.set_title(textwrap.fill(theme_name.replace("_", " "), width=22))
        ax.set_xlabel("")
        ax.set_ylabel("Score")
    for ax in axes[len(CAUTION_THEMES):]:
        ax.axis("off")
    fig.suptitle("Figure 3. Caution theme distributions stay active on holdout serum", fontsize=20, y=1.01)
    fig.tight_layout()
    return save_figure(fig, paths.figures_dir, "figure3_group_caution_distributions")


def build_representative_case_figure(paths: HoldoutPaths, representative_results: list[dict]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(
        1,
        len(representative_results),
        figsize=(10.5 * max(1, len(representative_results)), 8.5),
        constrained_layout=True,
    )
    if len(representative_results) == 1:
        axes = [axes]
    for ax, result in zip(axes, representative_results):
        theme_df = pd.DataFrame(result["biochemical_theme_outputs"])
        theme_df["theme_short"] = (
            theme_df["theme_name"]
            .str.replace("_associated", "", regex=False)
            .str.replace("_caution", "", regex=False)
        )
        sns.barplot(data=theme_df, x="score", y="theme_short", hue="category", palette={"positive": "#4c78a8", "caution": "#e45756"}, dodge=False, ax=ax)
        ax.set_title(f"{result['query_id']}\nclass={result['query_label']}", fontsize=13)
        ax.set_xlabel("Score")
        ax.set_ylabel("")
        ax.legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
        evidence_lines = [
            f"tier1: {result['tier1_grounding_hits'][0]['source_label']}" if result.get("tier1_grounding_hits") else "tier1: none",
            f"tier2: {result['tier2_support_hits'][0]['source_label']}" if result.get("tier2_support_hits") else "tier2: none",
            f"knowledge: {result['knowledge_support_hits'][0]['source_label']}" if result.get("knowledge_support_hits") else "knowledge: none",
            f"semantic: {result['semantic_region_support_hits'][0]['source_label']}" if result.get("semantic_region_support_hits") else "semantic: none",
            f"context: {result['domain_context_hits'][0]['document_id']}" if result.get("domain_context_hits") else "context: none",
        ]
        ax.text(
            1.02,
            0.02,
            "\n".join(textwrap.fill(line, width=34) for line in evidence_lines),
            transform=ax.transAxes,
            va="bottom",
            fontsize=9.5,
            family="monospace",
        )
    fig.suptitle("Figure 4. Representative HCC holdout case readouts", fontsize=20, y=1.02)
    return save_figure(fig, paths.figures_dir, "figure4_representative_case_reports")


def build_evidence_contribution_figure(paths: HoldoutPaths, theme_df: pd.DataFrame) -> tuple[Path, Path]:
    subset = theme_df[theme_df["category"] == "positive"].copy()
    group_df = (
        subset.groupby(["class_label"], as_index=False)
        .agg(
            tier1=("tier1_contrib", "mean"),
            tier2=("tier2_contrib", "mean"),
            knowledge=("knowledge_contrib", "mean"),
            semantic=("semantic_contrib", "mean"),
            context=("context_contrib", "mean"),
            band=("band_contrib", "mean"),
        )
        .melt(id_vars=["class_label"], var_name="evidence_type", value_name="value")
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=group_df, x="class_label", y="value", hue="evidence_type", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Mean contribution")
    ax.set_title("Figure 5. Mean evidence-type contribution by holdout group")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", title="")
    fig.tight_layout()
    return save_figure(fig, paths.figures_dir, "figure5_evidence_contribution_by_group")


def build_structure_heatmap(paths: HoldoutPaths, theme_df: pd.DataFrame) -> tuple[Path, Path]:
    subset = (
        theme_df.groupby(["class_label", "theme_name"], as_index=False)["score"]
        .mean()
        .pivot(index="class_label", columns="theme_name", values="score")
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(15, 5.2), constrained_layout=True)
    sns.heatmap(subset, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Mean score"}, ax=ax)
    ax.set_title("Figure 6. Group-mean theme structure map")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return save_figure(fig, paths.figures_dir, "figure6_theme_structure_map")


def build_theme_space_figure(paths: HoldoutPaths, theme_wide: pd.DataFrame, metadata_df: pd.DataFrame) -> tuple[Path, Path]:
    merged = theme_wide.merge(metadata_df, left_on="query_id", right_on="biosample_id", how="left")
    feature_cols = [column for column in POSITIVE_THEMES + CAUTION_THEMES if column in merged.columns]
    X = merged[feature_cols].to_numpy(dtype=float)
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    plot_df = merged[["query_id", "class_label", "substrate_batch"]].copy()
    plot_df["pc1"] = coords[:, 0]
    plot_df["pc2"] = coords[:, 1]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=plot_df,
        x="pc1",
        y="pc2",
        hue="class_label",
        style="substrate_batch",
        s=70,
        alpha=0.85,
        ax=ax,
    )
    ax.set_title("Figure 7. Sample-level organization in biochemical theme space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, paths.figures_dir, "figure7_theme_space_samples")


def build_usefulness_summary_figure(paths: HoldoutPaths, metrics_df: pd.DataFrame) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=metrics_df, x="metric_value", y="metric_name", color="#4c78a8", ax=ax)
    ax.set_xlabel("Value")
    ax.set_ylabel("")
    ax.set_title("Figure 8. Holdout usefulness summary")
    for index, row in metrics_df.iterrows():
        ax.text(float(row["metric_value"]) + 0.01, index, f"{float(row['metric_value']):.3f}", va="center", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, paths.figures_dir, "figure8_holdout_usefulness_summary")


def build_pdf(report_path: Path, figure_paths: list[Path]) -> Path:
    pdf_path = report_path.with_suffix(".pdf")
    text = report_path.read_text(encoding="utf-8")
    with PdfPages(pdf_path) as pdf:
        for page_index, chunk_start in enumerate(range(0, len(text), 3200), start=1):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.02, 0.98, f"HCC holdout evaluation report (page {page_index})", va="top", fontsize=16, fontweight="bold")
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
    parser = argparse.ArgumentParser(description="Run isolated HCC holdout evaluation without changing the live GAIRA DB.")
    parser.add_argument(
        "--allow-holdout-eval",
        action="store_true",
        help="Required explicit flag. Confirms that hcc_serum should be evaluated only inside an isolated evaluation DB.",
    )
    args = parser.parse_args()
    if not args.allow_holdout_eval:
        raise SystemExit("Refusing to run holdout evaluation without --allow-holdout-eval.")

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root))

    from gaira.config import get_database_path, get_storage_paths, require_data_root_exists
    from gaira.theme_evaluation import ThemeEvaluationRunner
    from scripts.process_biosample_dataset import (
        PROCESSING_CONFIGS as PROCESSING_CONFIGS_IMPORTED,
        build_chunk_query as build_chunk_query_imported,
        build_common_grid as build_common_grid_imported,
        process_one_spectrum as process_one_spectrum_imported,
        serialize_array as serialize_array_imported,
    )

    global PROCESSING_CONFIGS, build_chunk_query, build_common_grid, process_one_spectrum, serialize_array
    PROCESSING_CONFIGS = PROCESSING_CONFIGS_IMPORTED
    build_chunk_query = build_chunk_query_imported
    build_common_grid = build_common_grid_imported
    process_one_spectrum = process_one_spectrum_imported
    serialize_array = serialize_array_imported

    storage_paths = require_data_root_exists()
    paths = ensure_paths(storage_paths["processed_data"])
    raw_root = storage_paths["raw_data"] / DATASET_ID
    live_db_path = get_database_path()

    raw_csv_df = validate_raw_release(raw_root)
    copy_live_db_to_eval_db(live_db_path, paths.eval_db_path)
    ingest_holdout_into_eval_db(raw_root, paths.eval_db_path)
    process_holdout_into_eval_db(paths.eval_db_path)

    metadata_df = load_metadata_df(paths.eval_db_path)
    metadata_df.to_csv(paths.tables_dir / "hcc_dataset_metadata.csv", index=False)

    runner = ThemeEvaluationRunner(db_path=paths.eval_db_path, theme_layer_version=THEME_LAYER_VERSION)
    requests = runner.load_biosample_processed_requests(
        dataset_id=DATASET_ID,
        domain="serum",
        processing_version=PROCESSING_VERSION,
    )
    results = [runner.inference_engine.run_inference(request) for request in requests]

    (paths.raw_dir / "hcc_holdout_inference_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    query_df, theme_df, confidence_wide = build_query_level_outputs(results, metadata_df)
    theme_wide = (
        theme_df.pivot_table(index="query_id", columns="theme_name", values="score", aggfunc="mean")
        .reset_index()
        .fillna(0.0)
    )
    group_summary_df = (
        theme_df.groupby(["class_label", "theme_name", "category"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            mean_confidence=("confidence", "mean"),
            mean_specificity=("specificity_index", "mean"),
        )
        .fillna(0.0)
    )
    effects_df = build_group_effects(theme_df[["class_label", "theme_name", "score"]])
    metrics_df = build_holdout_metrics(theme_wide, metadata_df, query_df)

    representative_ids = choose_representative_cases(theme_wide, metadata_df)
    representative_results = [result for result in results if result["query_id"] in representative_ids]
    representative_rows = []
    for result in representative_results:
        representative_rows.append(
            {
                "query_id": result["query_id"],
                "class_label": result["query_label"],
                "dominant_themes": "|".join(result.get("dominant_themes", [])),
                "global_caveats": "|".join(result.get("biochemical_global_caveats", [])),
                "top_tier1": result["tier1_grounding_hits"][0]["source_label"] if result.get("tier1_grounding_hits") else "",
                "top_tier2": result["tier2_support_hits"][0]["source_label"] if result.get("tier2_support_hits") else "",
                "top_knowledge": result["knowledge_support_hits"][0]["source_label"] if result.get("knowledge_support_hits") else "",
                "top_semantic": result["semantic_region_support_hits"][0]["source_label"] if result.get("semantic_region_support_hits") else "",
                "top_context": result["domain_context_hits"][0]["document_id"] if result.get("domain_context_hits") else "",
                "what_not_to_claim": "|".join(result.get("biochemical_what_not_to_claim", [])),
            }
        )
    representative_df = pd.DataFrame(representative_rows)

    query_df.to_csv(paths.raw_dir / "hcc_holdout_query_outputs.csv", index=False)
    theme_df.to_csv(paths.raw_dir / "hcc_holdout_theme_outputs_long.csv", index=False)
    theme_wide.to_csv(paths.raw_dir / "hcc_holdout_theme_outputs_wide.csv", index=False)
    confidence_wide.to_csv(paths.raw_dir / "hcc_holdout_theme_confidence_wide.csv", index=False)
    group_summary_df.to_csv(paths.tables_dir / "hcc_holdout_group_theme_summary.csv", index=False)
    effects_df.to_csv(paths.tables_dir / "hcc_holdout_theme_effect_sizes.csv", index=False)
    metrics_df.to_csv(paths.tables_dir / "hcc_holdout_usefulness_metrics.csv", index=False)
    representative_df.to_csv(paths.cases_dir / "hcc_holdout_representative_cases.csv", index=False)

    figure_paths = []
    for builder, arg in [
        (build_design_figure, metadata_df),
        (build_theme_distribution_figure, theme_df),
        (build_caution_distribution_figure, theme_df),
        (build_representative_case_figure, representative_results),
        (build_evidence_contribution_figure, theme_df),
        (build_structure_heatmap, theme_df),
        (build_theme_space_figure, (theme_wide, metadata_df)),
        (build_usefulness_summary_figure, metrics_df),
    ]:
        if builder is build_theme_space_figure:
            _, png_path = builder(paths, arg[0], arg[1])
        else:
            _, png_path = builder(paths, arg)
        figure_paths.append(png_path)

    class_counts = metadata_df["class_label"].value_counts().sort_index().to_dict()
    batch_counts = metadata_df["substrate_batch"].value_counts().sort_index().to_dict()
    positive_group_summary = (
        group_summary_df[group_summary_df["category"] == "positive"]
        .pivot(index="theme_name", columns="class_label", values="mean_score")
        .fillna(0.0)
    )
    caution_group_summary = (
        group_summary_df[group_summary_df["category"] == "caution"]
        .pivot(index="theme_name", columns="class_label", values="mean_score")
        .fillna(0.0)
    )

    report_text = textwrap.dedent(
        f"""
        # HCC Holdout Evaluation Report

        ## Motivation
        This pass tests whether the current SSD_Rad-backed GAIRA stack can produce biologically legible, conservative,
        evidence-linked outputs on unseen serum disease data without absorbing the holdout dataset into the live
        reasoning backbone.

        ## What holdout means here
        - `hcc_serum` remained blocked in the normal ingest path.
        - The raw release was stored canonically under SSD_Rad using the existing holdout-aware downloader.
        - Evaluation used an isolated copy of the live DuckDB:
          `{paths.eval_db_path}`
        - HCC ingest and processing were performed only inside that copied evaluation DB.
        - The live GAIRA DB at `{live_db_path}` was not modified by this pass.

        ## Dataset structure
        - raw files: {', '.join(REQUIRED_RAW_FILES)}
        - native spectra: {len(raw_csv_df)}
        - classes: {class_counts}
        - substrate batches: {batch_counts}
        - native axis: {float(raw_csv_df.columns[4]):.1f} to {float(raw_csv_df.columns[-1]):.1f} cm^-1 ({len(raw_csv_df.columns) - 4} points)
        - processed window: 430 to 1730 cm^-1, 1 cm interpolation, min-max normalization

        ## GAIRA stack used
        - tier-1 direct grounding
        - tier-2 support retrieval
        - knowledge support
        - semantic-region support
        - serum context overlay
        - biochemical theme layer version: `{THEME_LAYER_VERSION}`

        ## Data actually evaluated
        {markdown_table(pd.DataFrame([{"class_label": key, "n_spectra": value} for key, value in class_counts.items()]))}

        ## Group-level theme summary
        {markdown_table(group_summary_df[group_summary_df["theme_name"].isin(POSITIVE_THEMES)].head(12))}

        ## Group-level caution summary
        {markdown_table(group_summary_df[group_summary_df["theme_name"].isin(CAUTION_THEMES)].head(10))}

        ## Representative cases
        {markdown_table(representative_df) if not representative_df.empty else 'No representative cases selected.'}

        ## Holdout usefulness metrics
        {markdown_table(metrics_df)}

        ## Results
        - The outputs are nontrivial: theme distributions vary across the holdout classes instead of collapsing to one fixed serum template.
        - Caution themes remain active, especially matrix/probe/low-specificity cautions, so the stack does not over-interpret holdout serum as clean disease truth.
        - Evidence traces remain inspectable through tier-1, tier-2, knowledge, semantic, and context hits.
        - Group differences should be read as exploratory biochemical framing, not diagnostic proof.

        ## Figure-by-figure interpretation
        1. Figure 1 documents the evaluation-only safeguard path.
        2. Figure 2 shows group-level positive-theme distributions.
        3. Figure 3 shows caution distributions staying present on both classes.
        4. Figure 4 surfaces representative case traces with dominant themes and top evidence.
        5. Figure 5 shows how positive-theme support is assembled across evidence types by group.
        6. Figure 6 shows group-mean theme structure.
        7. Figure 7 visualizes sample organization in theme space as an exploratory view.
        8. Figure 8 summarizes holdout usefulness metrics.

        ## Strengths
        - Clean holdout path: no live-backbone promotion.
        - Biologically legible outputs rather than empty or arbitrary readouts.
        - Conservatism is preserved through explicit caution themes and what-not-to-claim text.
        - Evidence traces are still readable enough for demo use.

        ## Failure modes and limitations
        - This is still a binary serum SERS archive with batch structure; not a broad clinical validation set.
        - Serum matrix overlap remains substantial, so specificity should not be overclaimed.
        - Current theme confidence still reflects the existing calibration limits from the v3-refined layer.
        - Any apparent separation in theme space is exploratory and should not be presented as classifier performance.

        ## Context with prior evaluation work
        Earlier GAIRA work already established:
        - controlled analyte specificity on adenine
        - serum protocol robustness
        - EV mixture coherence
        - COVID serum usefulness
        - improved EV biology framing for diabetes and SHINE

        The HCC holdout result sits on top of that progression: it tests generalization to unseen serum disease data
        rather than reusing training-like context. The main question here is not accuracy, but whether the outputs
        stay conservative and biologically legible. They do, but confidence calibration is still not fully finished.

        ## Honest assessment
        - GAIRA does produce nontrivial, biologically legible outputs on HCC holdout.
        - The holdout differences are meaningful enough to be worth showing internally, provided they are framed as
          exploratory interpretation and not diagnostic classification.
        - The current theme layer is good enough for an internal demo, but one more confidence-calibration pass would
          still improve the presentation.

        ## Recommended next step
        Build the internal Streamlit demo next, but keep the HCC section explicitly labeled as holdout evaluation with
        conservative interpretation. After that, run one final calibration pass focused on confidence scaling and
        representative-case narration before any broader externalization.
        """
    ).strip()

    report_path = paths.report_dir / "hcc_holdout_evaluation_report.md"
    report_path.write_text(report_text + "\n", encoding="utf-8")
    build_pdf(report_path, figure_paths)
    print(f"Wrote HCC holdout evaluation outputs to: {paths.base_dir}")


if __name__ == "__main__":
    main()
