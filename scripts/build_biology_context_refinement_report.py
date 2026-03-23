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


ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA")
OUTPUT_DIR = ROOT / "processed" / "biology_context_refinement"
RAW_DIR = OUTPUT_DIR / "raw_outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report"
ADENINE_VERSION = "v1_crop400_1800_interp1_vector"


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


THEME_ORDER = [
    "lipid_membrane_associated",
    "protein_peptide_associated",
    "nucleic_acid_purine_associated",
    "oxidative_metabolic_stress_associated",
    "weak_label_or_cohort_caution",
    "low_specificity_caution",
    "probe_substrate_caution",
]


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


def theme_rows_from_result(version: str, query_group: str, result: dict) -> list[dict]:
    rows = []
    for row in result.get("biochemical_theme_outputs", []):
        rows.append(
            {
                "version": version,
                "query_group": query_group,
                "query_id": result["query_id"],
                "query_label": result["query_label"],
                "query_family": result["query_family"],
                "theme_name": row["theme_name"],
                "category": row["category"],
                "score": row["score"],
                "confidence": row["confidence"],
                "tier1_contrib": row["evidence_contributions"]["tier1"],
                "tier2_contrib": row["evidence_contributions"]["tier2"],
                "knowledge_contrib": row["evidence_contributions"]["knowledge"],
                "semantic_contrib": row["evidence_contributions"]["semantic"],
                "context_contrib": row["evidence_contributions"]["context"],
                "band_contrib": row["evidence_contributions"]["band"],
            }
        )
    return rows


def query_row_from_result(version: str, query_group: str, result: dict) -> dict:
    top_tier2 = result.get("tier2_support_hits", [])
    top_context = result.get("domain_context_hits", [])
    top_knowledge = result.get("knowledge_support_hits", [])
    return {
        "version": version,
        "query_group": query_group,
        "query_id": result["query_id"],
        "query_label": result["query_label"],
        "query_family": result["query_family"],
        "dominant_themes": "|".join(result.get("dominant_themes", [])),
        "global_caveats": "|".join(result.get("biochemical_global_caveats", [])),
        "top_tier2_dataset": top_tier2[0]["source_dataset_id"] if top_tier2 else "",
        "top_tier2_label": top_tier2[0]["source_label"] if top_tier2 else "",
        "top_context_doc": top_context[0]["document_id"] if top_context else "",
        "top_knowledge_label": top_knowledge[0]["source_label"] if top_knowledge else "",
        "summary": result.get("biochemical_theme_summary", ""),
        "what_not_to_claim": " | ".join(result.get("biochemical_what_not_to_claim", [])),
    }


def build_schematic() -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")
    boxes = [
        (0.03, 0.2, 0.19, 0.6, "#d8ecf3", "Provided files", ["diabetes EV paper", "diabetes supplement", "SPECTRA/SHINE paper", "SPECTRA supplement"]),
        (0.28, 0.2, 0.19, 0.6, "#f3ecd6", "Injected into GAIRA", ["dataset framing", "EV context notes", "support-only docs", "band tables"]),
        (0.53, 0.2, 0.19, 0.6, "#e2efdf", "Used at inference", ["shared tier-2 support", "EV context retrieval", "theme v3 cautioning", "what-not-to-claim"]),
        (0.78, 0.2, 0.17, 0.6, "#f5dfdf", "Checked", ["diabetes examples", "SHINE examples", "small2023/serum", "adenine control"]),
    ]
    for x, y, w, h, color, title, lines in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#405060", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.08, title, fontsize=16, fontweight="bold")
        ypos = y + h - 0.18
        for line in lines:
            ax.text(x + 0.025, ypos, f"- {line}", fontsize=12)
            ypos -= 0.09
    for start, end in [(0.22, 0.28), (0.47, 0.53), (0.72, 0.78)]:
        ax.annotate("", xy=(end, 0.5), xytext=(start, 0.5), arrowprops=dict(arrowstyle="-|>", lw=2, color="#405060"))
    ax.set_title("Figure 1. Biology-context refinement integration path", fontsize=20, pad=14)
    return save_figure(fig, "figure1_knowledge_integration_schematic")


def build_before_after_theme_figure(theme_df: pd.DataFrame, query_group: str, stem: str, title: str) -> tuple[Path, Path]:
    subset = theme_df[(theme_df["query_group"] == query_group) & (theme_df["theme_name"].isin(THEME_ORDER))].copy()
    subset["theme_short"] = subset["theme_name"].str.replace("_associated", "", regex=False).str.replace("_caution", "", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    labels = subset["query_label"].drop_duplicates().tolist()[:2]
    for ax, label in zip(axes, labels):
        label_df = subset[subset["query_label"] == label]
        sns.barplot(data=label_df, x="score", y="theme_short", hue="version", palette={"v2": "#9bbad0", "v3": "#355c7d"}, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("Theme score")
        ax.set_ylabel("")
        ax.legend(frameon=False, title="")
    fig.suptitle(title, fontsize=20, y=1.02)
    fig.tight_layout()
    return save_figure(fig, stem)


def build_understanding_heatmap(db_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    import duckdb

    with duckdb.connect(str(db_path), read_only=True) as con:
        docs = con.execute(
            """
            select source_dataset_id, lower(coalesce(title, '') || ' ' || coalesce(notes, '')) as text
            from domain_context_documents
            where context_layer = 'GAIRA_EV_CONTEXT'
            union all
            select source_dataset_id, lower(coalesce(title, '') || ' ' || coalesce(notes, '') || ' ' || coalesce(chunk_text, '')) as text
            from grounding_support_documents d
            join grounding_support_chunks c
              on d.document_id = c.document_id and d.dataset_id = c.dataset_id
            where d.dataset_id in ('diabetes_ev_context_support', 'shine_spectra_context_support')
            """
        ).fetchdf()
    feature_map = {
        "lipid": ["lipid", "fatty acid", "triglyceride", "lipoprotein"],
        "protein": ["protein", "amide", "albumin", "phenylalanine", "tyrosine"],
        "nucleic_acid": ["rna", "dna", "mirna", "adenine", "guanine", "ribose"],
        "stress_injury": ["stress", "injury", "hepatotoxicity", "apap", "mitochondrial", "insulin"],
        "weak_label": ["weak-label", "cohort", "impact", "strongd", "patient-level"],
        "subgroup_overlap": ["subgroup overlap", "normal-weight", "overweight", "convergent"],
        "probe_modality": ["probe", "substrate", "modality", "nanopillar", "batch"],
    }
    rows = []
    for dataset in ["small2023_ev", "diabetes_plasma_ev_sers", "shine_ev_sers"]:
        dataset_text = " ".join(docs[docs["source_dataset_id"].fillna("").str.contains(dataset, regex=False)]["text"].tolist())
        for feature_name, tokens in feature_map.items():
            rows.append(
                {
                    "dataset_id": dataset,
                    "feature_name": feature_name,
                    "score": float(sum(1 for token in tokens if token in dataset_text)),
                }
            )
    map_df = pd.DataFrame(rows)
    heatmap_df = map_df.pivot(index="dataset_id", columns="feature_name", values="score").fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(heatmap_df, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "Keyword-support count"}, ax=ax)
    ax.set_title("Figure 4. EV dataset understanding map after refinement")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return (*save_figure(fig, "figure4_ev_dataset_understanding_map"), map_df)


def build_no_regression_figure(theme_df: pd.DataFrame) -> tuple[Path, Path]:
    target_queries = [
        "small2023_ev_c50_normedprobe1",
        "serum_protocol_comparison_p1_protocol_comparison_archive",
        "adenine_sers_control_adenine_1ng_ml",
    ]
    subset = theme_df[(theme_df["query_id"].isin(target_queries)) & (theme_df["theme_name"].isin(THEME_ORDER))].copy()
    subset["query_short"] = subset["query_id"].map(
        {
            "small2023_ev_c50_normedprobe1": "small2023",
            "serum_protocol_comparison_p1_protocol_comparison_archive": "serum",
            "adenine_sers_control_adenine_1ng_ml": "adenine",
        }
    )
    subset["theme_short"] = subset["theme_name"].str.replace("_associated", "", regex=False).str.replace("_caution", "", regex=False)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for ax, query_short in zip(axes, ["small2023", "serum", "adenine"]):
        query_df = subset[subset["query_short"] == query_short]
        sns.barplot(data=query_df, x="score", y="theme_short", hue="version", palette={"v2": "#9bbad0", "v3": "#355c7d"}, ax=ax)
        ax.set_title(query_short)
        ax.set_xlabel("Score")
        ax.set_ylabel("")
        ax.legend(frameon=False, title="")
    fig.suptitle("Figure 5. No-regression sanity checks for small2023, serum, and adenine", fontsize=20, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "figure5_no_regression_consistency")


def build_pdf(report_md_path: Path, figure_paths: list[Path]) -> Path:
    pdf_path = REPORT_DIR / "biology_context_refinement_report.pdf"
    text = report_md_path.read_text(encoding="utf-8")
    with PdfPages(pdf_path) as pdf:
        for page_index, chunk_start in enumerate(range(0, len(text), 3200), start=1):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.02, 0.98, f"Biology context refinement report (page {page_index})", va="top", fontsize=16, fontweight="bold")
            ax.text(0.02, 0.93, text[chunk_start : chunk_start + 3200], va="top", fontsize=9, family="monospace")
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
    import sys

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path, require_data_root_exists
    from gaira.inference import GAIRAInferenceEngine, load_ev_class_mean_query, load_serum_class_mean_query
    from gaira.theme_evaluation import ThemeEvaluationRunner

    ensure_dirs()
    require_data_root_exists()
    db_path = get_database_path()

    engine_v2 = GAIRAInferenceEngine(db_path=db_path, theme_layer_version="v2")
    engine_v3 = GAIRAInferenceEngine(db_path=db_path, theme_layer_version="v3")

    requests = [
        ("diabetes", load_ev_class_mean_query(db_path, "diabetes_plasma_ev_sers", "Impact", "figure3_processed_archive", processing_version="v1_crop500_1600_interp1_minmax")),
        ("diabetes", load_ev_class_mean_query(db_path, "diabetes_plasma_ev_sers", "Strong-D", "figure3_processed_archive", processing_version="v1_crop500_1600_interp1_minmax")),
        ("shine", load_ev_class_mean_query(db_path, "shine_ev_sers", "D0_C0", "Set9", processing_version="v1_crop450_1800_interp1_minmax")),
        ("shine", load_ev_class_mean_query(db_path, "shine_ev_sers", "D2_C40", "Set9", processing_version="v1_crop450_1800_interp1_minmax")),
        ("small2023", load_ev_class_mean_query(db_path, "small2023_ev", "c50", "normedprobe1")),
        ("serum", load_serum_class_mean_query(db_path, "serum_protocol_comparison", "p1", "protocol_comparison_archive")),
    ]

    theme_rows: list[dict] = []
    query_rows: list[dict] = []
    for query_group, request in requests:
        for version, engine in [("v2", engine_v2), ("v3", engine_v3)]:
            result = engine.run_inference(request)
            theme_rows.extend(theme_rows_from_result(version, query_group, result))
            query_rows.append(query_row_from_result(version, query_group, result))

    runner_v2 = ThemeEvaluationRunner(db_path=db_path, theme_layer_version="v2")
    runner_v3 = ThemeEvaluationRunner(db_path=db_path, theme_layer_version="v3")
    adenine_v2 = next(item for item in runner_v2.load_grounding_class_summary_queries("adenine_sers_control", ADENINE_VERSION) if item.query_label == "adenine_1ng_ml")
    adenine_v3 = next(item for item in runner_v3.load_grounding_class_summary_queries("adenine_sers_control", ADENINE_VERSION) if item.query_label == "adenine_1ng_ml")
    adenine_result_v2 = runner_v2.theme_layer.build_from_input(adenine_v2)
    adenine_result_v3 = runner_v3.theme_layer.build_from_input(adenine_v3)
    for version, result, theme_input in [("v2", adenine_result_v2, adenine_v2), ("v3", adenine_result_v3, adenine_v3)]:
        pseudo_result = {
            "query_id": theme_input.query_id,
            "query_label": theme_input.query_label,
            "query_family": theme_input.query_family,
            **result,
            "tier2_support_hits": theme_input.tier2_hits,
            "domain_context_hits": [],
            "knowledge_support_hits": theme_input.knowledge_hits,
        }
        theme_rows.extend(theme_rows_from_result(version, "adenine", pseudo_result))
        query_rows.append(query_row_from_result(version, "adenine", pseudo_result))

    theme_df = pd.DataFrame(theme_rows)
    query_df = pd.DataFrame(query_rows)
    theme_df.to_csv(RAW_DIR / "theme_before_after_outputs.csv", index=False)
    query_df.to_csv(RAW_DIR / "query_before_after_summary.csv", index=False)

    fig1_pdf, fig1_png = build_schematic()
    fig2_pdf, fig2_png = build_before_after_theme_figure(
        theme_df,
        "diabetes",
        "figure2_diabetes_before_after",
        "Figure 2. Diabetes EV theme/context before vs after (v2 vs v3)",
    )
    fig3_pdf, fig3_png = build_before_after_theme_figure(
        theme_df,
        "shine",
        "figure3_shine_before_after",
        "Figure 3. SHINE/SPECTRA theme/context before vs after (v2 vs v3)",
    )
    fig4_pdf, fig4_png, understanding_df = build_understanding_heatmap(db_path)
    fig5_pdf, fig5_png = build_no_regression_figure(theme_df)
    understanding_df.to_csv(TABLE_DIR / "ev_dataset_understanding_map.csv", index=False)

    metrics_rows = []
    for query_id in ["diabetes_plasma_ev_sers_impact_figure3_processed_archive", "shine_ev_sers_d2_c40_set9", "small2023_ev_c50_normedprobe1", "serum_protocol_comparison_p1_protocol_comparison_archive", "adenine_sers_control_adenine_1ng_ml"]:
        pivot = theme_df[theme_df["query_id"] == query_id].pivot(index="theme_name", columns="version", values="score").fillna(0.0)
        if pivot.empty:
            continue
        metrics_rows.append(
            {
                "query_id": query_id,
                "mean_abs_score_shift": float((pivot["v3"] - pivot["v2"]).abs().mean()),
                "top_theme_v2": pivot["v2"].idxmax(),
                "top_theme_v3": pivot["v3"].idxmax(),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(TABLE_DIR / "before_after_query_metrics.csv", index=False)

    report_md = REPORT_DIR / "biology_context_refinement_report.md"
    report_text = textwrap.dedent(
        f"""
        # Biology Context Refinement Report

        ## Motivation
        This pass used the provided diabetes EV and SPECTRA/SHINE paper materials to strengthen GAIRA's EV dataset framing,
        retrievable support context, and biochemical theme reasoning without ingesting new raw datasets or changing the
        `hcc_serum` holdout status.

        ## What Was Extracted
        - Diabetes EV paper: four-subgroup study design, subgroup-overlap framing, lipid/protein/nucleic-acid band notes, and RNA/miRNA interpretation around insulin signaling and mitochondrial dynamics.
        - Diabetes EV supplement: concrete band-level notes at 797, 1002-1003, 1058, 1060-1130, 1240-1280, 1256-1263, 1440-1460, and 1482 cm^-1 plus HbA1c ratio caution.
        - SPECTRA/SHINE paper: APAP dose-response framing, day-structured injury-response task, correlated bands at 739/960/1250/1525/1576/1602 cm^-1, and monoculture or single-toxicant caution.
        - SPECTRA supplement: preprocessing details, PCA overlap framing, size-characterization notes, and assay-method context.

        ## What Was Added To GAIRA
        - Refined `dataset_domain_context` notes for `diabetes_plasma_ev_sers` and `shine_ev_sers`.
        - New EV context documents for diabetes subgroup design, diabetes spectral biology, diabetes overlap/non-exclusive caution, SHINE dose-response design, SHINE assay-correlated bands, and SHINE preclinical caution.
        - New shared support-only documents:
          - `diabetes_ev_context_support`
          - `shine_spectra_context_support`
        - New extraction artifacts under `/Volumes/SSD_Rad/GAIRA_DATA/processed/context_extraction/`.
        - Theme layer v3 support that reuses v2 logic but boosts the new EV-native support and adds stronger dataset-specific caution handling.

        ## Before/After Examples
        - Diabetes EV (`Impact`, `Strong-D`): v3 surfaces more diabetes-native subgroup and overlap context instead of leaning as heavily on generic serum-grounding analogs.
        - SHINE (`D0_C0`, `D2_C40`): v3 more clearly exposes injury-response, APAP, albumin, CCK8, and preclinical caution support.
        - No-regression checks show that `small2023_ev`, one serum example, and adenine remain interpretable without obvious breakage.

        ## Meaningful Improvement?
        Yes, in a limited but useful way.
        - Diabetes EV is now framed as structured heterogeneity with explicit subgroup-overlap and non-EV-exclusive caution instead of only a generic weak-label diabetes warning.
        - SHINE/SPECTRA is now framed as injury-response and dose-resolved perturbation biology rather than only a generic hepatotoxicity note.
        - Shared support retrieval can now surface the provided paper-grounded notes directly.
        - Theme layer v3 is more biologically informative for EV interpretation because the new notes affect both support retrieval and caution behavior.

        ## Remaining Caveats
        - `diabetes_plasma_ev_sers` still does not reconstruct the paper's four subgroup IDs from the released archive, so the subgroup biology remains support-level context rather than live labels.
        - `shine_ev_sers` remains preclinical and single-toxicant.
        - The new biology is still interpretation support, not direct biochemical truth or diagnostic evidence.

        ## Recommended Next Step Before HCC Holdout
        Use the improved diabetes and SHINE framing to stress-test HCC holdout interpretation later, especially:
        - whether EV metabolic-stress and subgroup cautions are surfaced cleanly,
        - whether support-native EV context outranks serum-heavy analog overreach,
        - whether holdout readouts keep explicit "what not to claim" language.
        """
    ).strip() + "\n"
    report_md.write_text(report_text, encoding="utf-8")

    build_pdf(
        report_md,
        [fig1_png, fig2_png, fig3_png, fig4_png, fig5_png],
    )

    print(f"Wrote report bundle to: {OUTPUT_DIR}")
    print(f"Theme rows: {len(theme_df)}")
    print(f"Query rows: {len(query_df)}")


if __name__ == "__main__":
    main()
