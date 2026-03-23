import json
import textwrap
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_sampled_df(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if df.empty:
        return df
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=42).sort_values(df.columns[0]).reset_index(drop=True)


def wrap(value: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width)) if value else ""


def parse_json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def summarize_discovery(download_root: Path, output_path: Path) -> None:
    files = sorted(path for path in download_root.rglob("*") if path.is_file())
    ext_counts: dict[str, int] = {}
    for path in files:
        ext = path.suffix.lower() or "<no_ext>"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    metabolite_root = download_root / "Metabolite SERS fingerprints Fityk .fit and .peaks files" / "Supplementary-material"
    cholangio_zip = download_root / "Combination of label-free SERS-based nanosensor an.zip"
    relevant = [
        (
            "metabolite_sers63_support",
            sorted((metabolite_root / "fit").glob("*.fit")) + sorted((metabolite_root / "peaks").glob("*.peaks")),
        ),
        ("cca_hcc_lm_serum_sers", [cholangio_zip]),
    ]

    lines = [
        "# New Dataset File Discovery Summary",
        "",
        f"Root scanned: `{download_root}`",
        "",
        "## Extension counts",
    ]
    for ext, count in sorted(ext_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{ext}`: {count}")

    lines.append("")
    lines.append("## Task-relevant dataset grouping")
    for dataset_id, paths in relevant:
        total_size = sum(path.stat().st_size for path in paths if path.exists())
        lines.append(f"- `{dataset_id}`: {len(paths)} task-relevant file(s), {total_size / (1024 * 1024):.2f} MB")
        for path in paths[:8]:
            rel = path.relative_to(download_root)
            lines.append(f"  - `{rel}`")
        if len(paths) > 8:
            lines.append(f"  - ... plus {len(paths) - 8} additional task-relevant files")

    lines.append("")
    lines.append("## Placement decisions")
    lines.append("- `metabolite_sers63_support`: grounding spectra reconstructed from Fityk `.fit` files with released `.peaks` support")
    lines.append("- `cca_hcc_lm_serum_sers`: biosample serum cohort in `GAIRA_SERUM`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_validation_text(dataset_id: str, connection: duckdb.DuckDBPyConnection) -> str:
    registry = connection.execute(
        "SELECT dataset_family FROM read_csv_auto(?) WHERE dataset_id = ?",
        ["data/registry/datasets.csv", dataset_id],
    ).fetchone()
    family = str(registry[0])
    lines = [f"dataset_id: {dataset_id}", f"dataset_family: {family}"]

    if family == "biosample":
        counts = connection.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM biosample_metadata WHERE dataset_id = ?) AS metadata_count,
              (SELECT COUNT(*) FROM biosample_spectra WHERE dataset_id = ?) AS spectra_count,
              (SELECT COUNT(*) FROM biosample_processed_spectra WHERE dataset_id = ?) AS processed_count,
              (SELECT COUNT(*) FROM biosample_peaks WHERE dataset_id = ?) AS peak_count
            """,
            [dataset_id, dataset_id, dataset_id, dataset_id],
        ).fetchone()
        lines.extend(
            [
                f"biosample_metadata: {counts[0]}",
                f"biosample_spectra: {counts[1]}",
                f"biosample_processed_spectra: {counts[2]}",
                f"biosample_peaks: {counts[3]}",
                f"processed_coverage_ratio: {counts[2] / counts[1]:.4f}" if counts[1] else "processed_coverage_ratio: 0.0",
                "",
                "class_counts:",
                connection.execute(
                    """
                    SELECT class_label, COUNT(*) AS n
                    FROM biosample_metadata
                    WHERE dataset_id = ?
                    GROUP BY 1
                    ORDER BY 1
                    """,
                    [dataset_id],
                ).fetchdf().to_string(index=False),
                "",
                "subclass_counts:",
                connection.execute(
                    """
                    SELECT subclass_label, COUNT(*) AS n
                    FROM biosample_metadata
                    WHERE dataset_id = ?
                    GROUP BY 1
                    ORDER BY 1
                    """,
                    [dataset_id],
                ).fetchdf().to_string(index=False),
                "",
                "raw_axis_summary:",
                connection.execute(
                    """
                    SELECT MIN(x_min) AS min_x, MAX(x_max) AS max_x, MIN(n_points) AS min_points, MAX(n_points) AS max_points
                    FROM biosample_spectra WHERE dataset_id = ?
                    """,
                    [dataset_id],
                ).fetchdf().to_string(index=False),
                "",
                "processed_axis_summary:",
                connection.execute(
                    """
                    SELECT MIN(x_min) AS min_x, MAX(x_max) AS max_x, MIN(n_points) AS min_points, MAX(n_points) AS max_points
                    FROM biosample_processed_spectra WHERE dataset_id = ?
                    """,
                    [dataset_id],
                ).fetchdf().to_string(index=False),
                "",
                "processed_nonzero_summary:",
                connection.execute(
                    """
                    SELECT
                      COUNT(*) AS n_spectra,
                      AVG(CASE WHEN json_array_length(intensity_json) > 0 THEN 1 ELSE 0 END) AS nonempty_ratio
                    FROM biosample_processed_spectra
                    WHERE dataset_id = ?
                    """,
                    [dataset_id],
                ).fetchdf().to_string(index=False),
            ]
        )
    else:
        counts = connection.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM grounding_metadata WHERE dataset_id = ?) AS metadata_count,
              (SELECT COUNT(*) FROM grounding_spectra WHERE dataset_id = ?) AS spectra_count,
              (SELECT COUNT(*) FROM grounding_processed_spectra WHERE dataset_id = ?) AS processed_count,
              (SELECT COUNT(*) FROM grounding_support_documents WHERE dataset_id = ?) AS support_doc_count,
              (SELECT COUNT(*) FROM grounding_peaks WHERE dataset_id = ?) AS peak_count
            """,
            [dataset_id, dataset_id, dataset_id, dataset_id, dataset_id],
        ).fetchone()
        lines.extend(
            [
                f"grounding_metadata: {counts[0]}",
                f"grounding_spectra: {counts[1]}",
                f"grounding_processed_spectra: {counts[2]}",
                f"grounding_support_documents: {counts[3]}",
                f"grounding_peaks: {counts[4]}",
                f"processed_coverage_ratio: {counts[2] / counts[1]:.4f}" if counts[1] else "processed_coverage_ratio: skipped",
                "",
                "class_counts:",
                connection.execute(
                    """
                    SELECT class_label, COUNT(*) AS n
                    FROM grounding_metadata
                    WHERE dataset_id = ?
                    GROUP BY 1
                    ORDER BY 1
                    """,
                    [dataset_id],
                ).fetchdf().head(20).to_string(index=False),
                "",
                "raw_axis_summary:",
                connection.execute(
                    """
                    SELECT MIN(x_min) AS min_x, MAX(x_max) AS max_x, MIN(n_points) AS min_points, MAX(n_points) AS max_points
                    FROM grounding_spectra WHERE dataset_id = ?
                    """,
                    [dataset_id],
                ).fetchdf().to_string(index=False),
                "",
                "processed_axis_summary:",
                connection.execute(
                    """
                    SELECT MIN(x_min) AS min_x, MAX(x_max) AS max_x, MIN(n_points) AS min_points, MAX(n_points) AS max_points
                    FROM grounding_processed_spectra WHERE dataset_id = ?
                    """,
                    [dataset_id],
                ).fetchdf().to_string(index=False),
            ]
        )

    return "\n".join(lines) + "\n"


def load_raw_spectra(connection: duckdb.DuckDBPyConnection, dataset_id: str, family: str) -> pd.DataFrame:
    if family == "biosample":
        return connection.execute(
            """
            SELECT m.class_label, s.biosample_id AS spectrum_id, s.wavenumbers_json, s.intensity_json
            FROM biosample_spectra s
            JOIN biosample_metadata m USING (biosample_id, dataset_id)
            WHERE s.dataset_id = ?
            ORDER BY s.biosample_id
            """,
            [dataset_id],
        ).fetchdf()
    return connection.execute(
        """
        SELECT class_label, grounding_id AS spectrum_id, wavenumbers_json, intensity_json
        FROM grounding_spectra
        JOIN grounding_metadata USING (grounding_id, dataset_id)
        WHERE dataset_id = ?
        ORDER BY grounding_id
        """,
        [dataset_id],
    ).fetchdf()


def load_processed_spectra(connection: duckdb.DuckDBPyConnection, dataset_id: str, family: str) -> pd.DataFrame:
    if family == "biosample":
        return connection.execute(
            """
            SELECT m.class_label, p.biosample_id AS spectrum_id, p.wavenumbers_json, p.intensity_json
            FROM biosample_processed_spectra p
            JOIN biosample_metadata m USING (biosample_id, dataset_id)
            WHERE p.dataset_id = ?
            ORDER BY p.biosample_id
            """,
            [dataset_id],
        ).fetchdf()
    return connection.execute(
        """
        SELECT m.class_label, p.grounding_id AS spectrum_id, p.wavenumbers_json, p.intensity_json
        FROM grounding_processed_spectra p
        JOIN grounding_metadata m USING (grounding_id, dataset_id)
        WHERE p.dataset_id = ?
        ORDER BY p.grounding_id
        """,
        [dataset_id],
    ).fetchdf()


def load_class_summary(connection: duckdb.DuckDBPyConnection, dataset_id: str, family: str) -> pd.DataFrame:
    if family == "biosample":
        return connection.execute(
            """
            SELECT class_label, mean_wavenumbers_json AS wavenumbers_json, mean_intensity_json AS intensity_json
            FROM biosample_class_summary
            WHERE dataset_id = ?
            ORDER BY class_label
            """,
            [dataset_id],
        ).fetchdf()
    return connection.execute(
        """
        SELECT class_label, mean_wavenumbers_json AS wavenumbers_json, mean_intensity_json AS intensity_json
        FROM grounding_class_summary
        WHERE dataset_id = ?
        ORDER BY class_label
        """,
        [dataset_id],
    ).fetchdf()


def plot_overlays(df: pd.DataFrame, title: str, output_base: Path) -> None:
    if df.empty:
        return
    sampled = load_sampled_df(df, n=5)
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in sampled.iterrows():
        ax.plot(parse_json_array(row["wavenumbers_json"]), parse_json_array(row["intensity_json"]), alpha=0.8, label=wrap(row["spectrum_id"], 18))
    ax.set_title(title)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_means(df: pd.DataFrame, title: str, output_base: Path, max_classes: int = 8) -> None:
    if df.empty:
        return
    plot_df = df.head(max_classes).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=len(plot_df))
    for color, (_, row) in zip(palette, plot_df.iterrows()):
        ax.plot(parse_json_array(row["wavenumbers_json"]), parse_json_array(row["intensity_json"]), label=wrap(row["class_label"], 16), color=color)
    ax.set_title(title)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Mean processed intensity")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_serum_inference_examples(db_path: Path) -> list[dict]:
    from gaira.inference import GAIRAInferenceEngine, InferenceRequest
    from gaira.grounding_search import SpectrumQuery

    engine = GAIRAInferenceEngine(db_path=db_path, theme_layer_version="v3")
    queries: list[InferenceRequest] = []
    with duckdb.connect(str(db_path), read_only=True) as connection:
        rows = connection.execute(
            """
            SELECT p.biosample_id, m.class_label, m.subclass_label, p.wavenumbers_json, p.intensity_json
            FROM biosample_processed_spectra p
            JOIN biosample_metadata m USING (biosample_id, dataset_id)
            WHERE p.dataset_id = 'cca_hcc_lm_serum_sers'
              AND m.class_label IN ('cca', 'healthy_control')
            QUALIFY ROW_NUMBER() OVER (PARTITION BY m.class_label ORDER BY p.biosample_id) = 1
            ORDER BY m.class_label
            """
        ).fetchdf()
    for _, row in rows.iterrows():
        queries.append(
            InferenceRequest(
                domain="serum",
                query_id=str(row["biosample_id"]),
                query_label=str(row["class_label"]),
                query_family=str(row["subclass_label"]),
                source_dataset_id="cca_hcc_lm_serum_sers",
                spectrum_query=SpectrumQuery(
                    query_id=str(row["biosample_id"]),
                    query_label=str(row["class_label"]),
                    query_family=str(row["subclass_label"]),
                    source_dataset_id="cca_hcc_lm_serum_sers",
                    x=parse_json_array(row["wavenumbers_json"]),
                    y=parse_json_array(row["intensity_json"]),
                    notes="Representative processed serum sample from cca_hcc_lm_serum_sers",
                ),
            )
        )
    return [engine.run_inference(query) for query in queries]


def build_grounding_examples(db_path: Path, dataset_id: str, n_examples: int = 2) -> list[dict]:
    from gaira.grounding_search import GroundingSearchEngine, SpectrumQuery

    engine = GroundingSearchEngine(db_path=db_path)
    examples: list[dict] = []
    with duckdb.connect(str(db_path), read_only=True) as connection:
        rows = connection.execute(
            """
            SELECT m.class_label, p.grounding_id, p.wavenumbers_json, p.intensity_json
            FROM grounding_processed_spectra p
            JOIN grounding_metadata m USING (grounding_id, dataset_id)
            WHERE p.dataset_id = ?
            ORDER BY p.grounding_id
            LIMIT ?
            """,
            [dataset_id, n_examples],
        ).fetchdf()
    for _, row in rows.iterrows():
        query = SpectrumQuery(
            query_id=str(row["grounding_id"]),
            query_label=str(row["class_label"]),
            query_family=dataset_id,
            source_dataset_id=dataset_id,
            x=parse_json_array(row["wavenumbers_json"]),
            y=parse_json_array(row["intensity_json"]),
            notes=f"Representative processed grounding spectrum from {dataset_id}",
        )
        direct_df = engine.search_direct_spectral_evidence(query, top_n_per_source=5)
        tier2_df = engine.search_supporting_literature_for_spectrum(query, seed_labels=direct_df["source_label"].head(6).astype(str).tolist(), domain="serum", top_n=5)
        examples.append(
            {
                "dataset_id": dataset_id,
                "query_id": row["grounding_id"],
                "query_label": row["class_label"],
                "top_same_dataset_tier1": direct_df[direct_df["source_dataset_id"] == dataset_id].head(3).to_dict(orient="records"),
                "top_tier1": direct_df.head(3).to_dict(orient="records"),
                "top_tier2": tier2_df.head(3).to_dict(orient="records") if not tier2_df.empty else [],
            }
        )
    return examples


def write_inference_summary(output_path: Path, serum_results: list[dict], grounding_results: list[dict]) -> None:
    lines = ["# Inference Sanity Summary", ""]
    lines.append("## Biosample inference checks")
    for result in serum_results:
        lines.append(f"### {result['query_id']}")
        lines.append(f"- dominant themes: {', '.join(result.get('dominant_themes', []))}")
        lines.append(f"- top tier-1: {result['tier1_grounding_hits'][0]['source_dataset_id']} / {result['tier1_grounding_hits'][0]['source_label']}" if result.get("tier1_grounding_hits") else "- top tier-1: none")
        lines.append(f"- top tier-2: {result['tier2_support_hits'][0]['source_dataset_id']} / {result['tier2_support_hits'][0]['source_label']}" if result.get("tier2_support_hits") else "- top tier-2: none")
        lines.append(f"- top context: {result['domain_context_hits'][0]['document_id']}" if result.get("domain_context_hits") else "- top context: none")
        lines.append(f"- caveats: {', '.join(result.get('biochemical_global_caveats', []))}")
        lines.append("")

    lines.append("## Grounding/support retrieval checks")
    for result in grounding_results:
        lines.append(f"### {result['query_id']} ({result['dataset_id']})")
        if result.get("top_same_dataset_tier1"):
            top = result["top_same_dataset_tier1"][0]
            lines.append(f"- top same-dataset tier-1: {top['source_dataset_id']} / {top['source_label']} / {top['result_type']}")
        if result["top_tier1"]:
            top = result["top_tier1"][0]
            lines.append(f"- top tier-1: {top['source_dataset_id']} / {top['source_label']} / {top['result_type']}")
        if result["top_tier2"]:
            top = result["top_tier2"][0]
            lines.append(f"- top tier-2: {top['source_dataset_id']} / {top['source_label']} / {top['result_type']}")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_final_report(output_path: Path, validation_dir: Path, plot_dir: Path, inference_summary_path: Path) -> None:
    lines = [
        "# New Dataset Ingestion Report",
        "",
        "## Dataset placement decisions",
        "- `metabolite_sers63_support`: `GAIRA_GROUNDING` reconstructable metabolite SERS spectra with peak support",
        "- `cca_hcc_lm_serum_sers`: `GAIRA_SERUM` multi-class serum biosample cohort",
        "",
        "## Validation artifacts",
    ]
    for path in sorted(validation_dir.glob("*_validation.txt")):
        lines.append(f"- `{path.name}`")
    lines.append("")
    lines.append("## Sanity plots")
    for path in sorted(plot_dir.glob("*.png")):
        lines.append(f"- `{path.name}`")
    lines.append("")
    lines.append("## Inference summary")
    lines.append(f"- `{inference_summary_path.name}`")
    lines.append("")
    lines.append("## Caveats")
    lines.append("- `metabolite_sers63_support` includes one fit-only spectrum (`spermicide`) without a released peak list; GAIRA detected peaks for it directly.")
    lines.append("- `cca_hcc_lm_serum_sers` required explicit filtering of non-spectral spatial-matrix TXT files and the auxiliary `Try dilute condition` branch.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("- Cleanest and highest-value downstream serum reasoning dataset: `cca_hcc_lm_serum_sers`")
    lines.append("- Highest-value shared grounding addition: `metabolite_sers63_support`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path, get_storage_paths, require_data_root_exists

    sns.set_theme(style="whitegrid", context="talk")
    storage_paths = require_data_root_exists()
    db_path = get_database_path()
    output_root = storage_paths["processed_data"] / "new_dataset_ingestion_report"
    validation_dir = output_root / "validation"
    plot_dir = output_root / "sanity_plots"
    ensure_dir(output_root)
    ensure_dir(validation_dir)
    ensure_dir(plot_dir)

    download_root = Path.home() / "Downloads" / "New_Set_SERS_Papers_Data"
    summarize_discovery(download_root, output_root / "file_discovery_summary.md")

    datasets = {
        "metabolite_sers63_support": "grounding",
        "cca_hcc_lm_serum_sers": "biosample",
    }

    with duckdb.connect(str(db_path), read_only=True) as connection:
        for dataset_id, family in datasets.items():
            (validation_dir / f"{dataset_id}_validation.txt").write_text(
                build_validation_text(dataset_id, connection),
                encoding="utf-8",
            )

            raw_df = load_raw_spectra(connection, dataset_id, family)
            processed_df = load_processed_spectra(connection, dataset_id, family)
            class_df = load_class_summary(connection, dataset_id, family)

            plot_overlays(raw_df, f"{dataset_id} raw spectra overlays", plot_dir / f"{dataset_id}_raw_overlay")
            plot_overlays(processed_df, f"{dataset_id} processed spectra overlays", plot_dir / f"{dataset_id}_processed_overlay")
            plot_class_means(class_df, f"{dataset_id} class means", plot_dir / f"{dataset_id}_class_means")

    serum_results = build_serum_inference_examples(db_path)
    grounding_results = build_grounding_examples(db_path, "metabolite_sers63_support", n_examples=2)
    write_inference_summary(output_root / "inference_sanity_summary.md", serum_results, grounding_results)
    build_final_report(output_root / "ingestion_report.md", validation_dir, plot_dir, output_root / "inference_sanity_summary.md")


if __name__ == "__main__":
    main()
