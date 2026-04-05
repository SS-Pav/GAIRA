#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from gaira.demo.v8_report_utils import add_text_page, image_grid_page, image_page, table_page


DEFAULT_SHARED_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_shared_backbone_diagnostics_v1")
DEFAULT_EV_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_ev_stress_analysis_v1")
DEFAULT_SMALL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_small2023_benchmark_v1")
DEFAULT_SERUM_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_serum_stress_analysis_v1")
DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_diagnostics_report_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Nature-style PDF report for v8 CPU diagnostics.")
    parser.add_argument("--shared-dir", type=Path, default=DEFAULT_SHARED_DIR)
    parser.add_argument("--ev-dir", type=Path, default=DEFAULT_EV_DIR)
    parser.add_argument("--small-dir", type=Path, default=DEFAULT_SMALL_DIR)
    parser.add_argument("--serum-dir", type=Path, default=DEFAULT_SERUM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--title", default="GAIRAM v8 CPU Diagnostics Report")
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def extract_bullets(text: str, limit: int = 6) -> list[str]:
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return bullets[:limit]


def safe_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_title_page(pdf: PdfPages, title: str, output_dir: Path) -> None:
    add_text_page(
        pdf,
        title,
        [
            "Decision-grade CPU diagnostics comparing the current shared GAIRAM backbones, stress/disease EV structure, the small2023 probe-invariance benchmark, and a broad serum stress/inflammation slice before any v8 GPU retraining.",
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
            f"Repository: {Path.cwd()}",
            f"Report output directory: {output_dir}",
        ],
        subtitle="Nature-style internal methods and architecture memo",
        footer="All analyses were run non-destructively against existing local GAIRAM processed outputs.",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shared_metrics = safe_csv(args.shared_dir / "shared_backbone_metrics.csv")
    ev_metrics = safe_csv(args.ev_dir / "ev_stress_metrics.csv")
    ev_linkage = safe_csv(args.ev_dir / "state_composition_linkage_metrics.csv")
    small_metrics = safe_csv(args.small_dir / "small2023_benchmark_metrics.csv")
    small_compare = safe_csv(args.small_dir / "small2023_direct_vs_embedding_comparison.csv")
    serum_metrics = safe_csv(args.serum_dir / "serum_metrics.csv")

    shared_summary = read_text(args.shared_dir / "shared_backbone_summary.md")
    ev_summary = read_text(args.ev_dir / "ev_stress_summary.md")
    small_summary = read_text(args.small_dir / "small2023_summary.md")
    serum_summary = read_text(args.serum_dir / "serum_summary.md")

    executive_summary = [
        "The diagnostics converge on a clean v8 architecture decision. v7 remains the best available shared backbone for EV-facing work because it improves cross-dataset EV mixing without breaking the global sample-type scaffold, but it is not yet a universal answer for every domain.",
        "The strongest biological story is still EV stress/disease. It shows meaningful unsupervised latent structure, cautious grounding-derived composition profiles, and usable broad-state overlays even though dataset identity remains stronger than broad-state identity.",
        "small2023 does not look solved by the shared encoder. The direct-versus-embedding comparison and the older specialized invariant benchmark together indicate that a dedicated small2023 head remains justified if that benchmark matters scientifically.",
        "Serum is promising but not ready. Its broad stress/inflammation story is real only in a limited, conservative sense, and current cohort/protocol effects remain too strong for an immediate dedicated serum invariance retrain.",
    ]

    report_pdf = args.output_dir / "v8_diagnostics_report.pdf"
    figure_manifest_rows = []

    with PdfPages(report_pdf) as pdf:
        build_title_page(pdf, args.title, args.output_dir)

        add_text_page(
            pdf,
            "Executive Summary",
            executive_summary,
            footer="Core architecture readout: keep v7 shared, lead with EV stress, specialize small2023, delay serum specialization.",
        )

        add_text_page(
            pdf,
            "Shared Backbone Comparison",
            extract_bullets(shared_summary, limit=8),
            subtitle="Comparing v5 full-corpus shared, v6 within-type, and v7 anchor-invariance under the locally available artifacts.",
        )
        if not shared_metrics.empty:
            cols = [
                "backbone",
                "status",
                "nn_purity_sample_type",
                "nn_purity_dataset",
                "nn_purity_family",
                "ev_dataset_purity",
                "serum_dataset_purity",
                "ev_cross_dataset_mixed_clusters",
                "serum_cross_dataset_mixed_clusters",
            ]
            table_page(
                pdf,
                "Shared Backbone Metrics",
                shared_metrics[[c for c in cols if c in shared_metrics.columns]].round(4),
                subtitle="Global and within-type neighborhood metrics used to decide the shared starting point for v8.",
            )
        shared_images = [
            ("backbone_comparison_metrics.png", "Shared global metrics"),
            ("ev_within_type_metric_comparison.png", "Within-EV metrics"),
            ("serum_within_type_metric_comparison.png", "Within-serum metrics"),
        ]
        existing_shared = [(args.shared_dir / name, caption) for name, caption in shared_images if (args.shared_dir / name).exists()]
        if existing_shared:
            image_grid_page(pdf, "Shared Backbone Figures", existing_shared, subtitle="Seaborn figures comparing the available shared backbones.")
            for path, caption in existing_shared:
                figure_manifest_rows.append({"section": "shared_backbone", "file": str(path), "caption": caption})

        add_text_page(
            pdf,
            "EV Stress / Disease Analysis",
            extract_bullets(ev_summary, limit=8),
            subtitle="SHINE EV hepatotoxicity plus diabetes EV under broad low-vs-high metabolic stress harmonization.",
        )
        if not ev_metrics.empty:
            table_page(
                pdf,
                "EV Stress Metrics",
                ev_metrics.round(4),
                subtitle="Latent structure, neighborhood purity, and cluster counts for the stress/disease slice.",
            )
        ev_images = [
            ("neutral_latent_map.png", "Unsupervised EV latent structure"),
            ("latent_map_by_state.png", "Broad stress-state overlay"),
            ("latent_map_by_dominant_biochemical_theme.png", "Dominant grounding-derived biochemical theme overlay"),
            ("cluster_composition_heatmap.png", "Cluster biochemical composition profiles"),
            ("cluster_state_heatmap.png", "Cluster state enrichment"),
            ("composition_vs_state_scatter.png", "Theme-vs-state linkage summary"),
        ]
        existing_ev = [(args.ev_dir / name, caption) for name, caption in ev_images if (args.ev_dir / name).exists()]
        if existing_ev:
            image_grid_page(pdf, "EV Stress Figures", existing_ev[:3], subtitle="Latent structure first, biological and biochemical meaning second.")
            if len(existing_ev) > 3:
                image_grid_page(pdf, "EV Stress Figures II", existing_ev[3:], subtitle="Composition structure and state-enrichment diagnostics.")
            for path, caption in existing_ev:
                figure_manifest_rows.append({"section": "ev_stress", "file": str(path), "caption": caption})
        if not ev_linkage.empty:
            top_linkage = ev_linkage.copy()
            top_linkage["abs_value"] = top_linkage["value"].abs()
            top_linkage = top_linkage.sort_values("abs_value", ascending=False).drop(columns=["abs_value"]).head(6)
            table_page(
                pdf,
                "EV State–Composition Linkage",
                top_linkage.round(4),
                subtitle="Correlations between grounding-derived theme weights and cluster-level stress enrichment.",
            )

        add_text_page(
            pdf,
            "small2023 EV Benchmark",
            extract_bullets(small_summary, limit=8),
            subtitle="Direct normalized poly3 spectra versus shared backbones and the older specialized invariant benchmark.",
        )
        if not small_compare.empty:
            table_page(
                pdf,
                "small2023 Scorecard",
                small_compare.round(4),
                subtitle="Direct versus v5/v6/v7 shared embeddings, plus v2 specialized invariant metrics where locally available.",
                font_size=7.8,
            )
        small_images = [
            ("direct_map_by_class.png", "Direct spectra by class"),
            ("direct_map_by_probe.png", "Direct spectra by probe"),
            ("v5_embedding_map_by_class.png", "v5 shared embedding by class"),
            ("v7_embedding_map_by_class.png", "v7 shared embedding by class"),
            ("direct_vs_embedding_metric_bars.png", "Direct-vs-embedding metric comparison"),
            ("class_composition_heatmap.png", "small2023 class-level biochemical composition"),
        ]
        existing_small = [(args.small_dir / name, caption) for name, caption in small_images if (args.small_dir / name).exists()]
        if existing_small:
            image_grid_page(pdf, "small2023 Figures", existing_small[:3], subtitle="Direct and shared latent geometry on the benchmark slice.")
            if len(existing_small) > 3:
                image_grid_page(pdf, "small2023 Figures II", existing_small[3:], subtitle="Metric comparison and cautious composition interpretation.")
            for path, caption in existing_small:
                figure_manifest_rows.append({"section": "small2023", "file": str(path), "caption": caption})

        add_text_page(
            pdf,
            "Serum Stress / Inflammation Analysis",
            extract_bullets(serum_summary, limit=8),
            subtitle="Broad, conservative serum harmonization across liver, COVID, calibration, protocol, and spiked-serum archives.",
        )
        if not serum_metrics.empty:
            table_page(
                pdf,
                "Serum Metrics",
                serum_metrics.round(4),
                subtitle="Latent structure, neighborhood purity, and cross-dataset mixing readout for the serum slice.",
            )
        serum_images = [
            ("serum_neutral_latent_map.png", "Serum latent structure"),
            ("serum_latent_map_by_state.png", "Broad serum state overlay"),
            ("serum_latent_map_by_theme.png", "Dominant biochemical theme overlay"),
            ("serum_cluster_composition_heatmap.png", "Serum cluster composition profiles"),
            ("serum_cluster_state_heatmap.png", "Serum cluster state enrichment"),
            ("serum_composition_vs_state_scatter.png", "Serum composition versus high-stress enrichment"),
        ]
        existing_serum = [(args.serum_dir / name, caption) for name, caption in serum_images if (args.serum_dir / name).exists()]
        if existing_serum:
            image_grid_page(pdf, "Serum Figures", existing_serum[:3], subtitle="Serum geometry and broad-state overlays.")
            if len(existing_serum) > 3:
                image_grid_page(pdf, "Serum Figures II", existing_serum[3:], subtitle="Composition diagnostics and cohort caveats.")
            for path, caption in existing_serum:
                figure_manifest_rows.append({"section": "serum", "file": str(path), "caption": caption})

        architecture_decision = [
            "v7 should remain the shared backbone for the next GPU iteration. It is the best current compromise between preserving sample-type structure and improving cross-dataset EV mixing.",
            "EV stress/disease should be the lead v8 GPU target. It is the strongest present biological page and the one most clearly supported by unsupervised structure plus grounding-derived composition profiles.",
            "small2023 should get a dedicated specialized head. The current shared encoder does not reproduce the old probe-invariant class benchmark strongly enough to justify treating it as solved by a universal backbone.",
            "Serum should not yet get a full dedicated training track. The right next serum step is better harmonization and anchor construction, not immediate large-scale retraining.",
        ]
        add_text_page(
            pdf,
            "Integrated v8 Architecture Decision",
            architecture_decision,
            subtitle="CPU diagnostics converted into the recommended CPU-to-GPU plan.",
        )

        next_steps = [
            "1. Lock v7 as the shared initialization point for any v8 EV-facing GPU work.",
            "2. Build the next GPU experiment around EV stress/disease supervision and evaluation, preserving composition-profile reporting rather than collapsing everything to one-hot themes.",
            "3. Run a dedicated small2023 specialized-head experiment in parallel if probe invariance remains a scientific requirement.",
            "4. Treat serum as a metadata and harmonization problem first. Improve cohort bridges, then revisit whether a serum-specific head is warranted.",
            "5. Keep all downstream demo and interpretation layers explicit about the distinction between unsupervised structure and meaning painted on after grounding retrieval.",
        ]
        add_text_page(
            pdf,
            "Recommended CPU → GPU Next Steps",
            next_steps,
            footer="The strongest next move is not another generic shared retrain; it is a targeted EV stress GPU plan plus specialized follow-up for small2023.",
        )

        appendix_lines = [
            "Direct spectra analyses use the normalized poly3 baseline-corrected processing branch.",
            "Grounding-derived composition profiles are generated by retrieving nearest grounding embeddings in latent space and aggregating broad biochemical themes into normalized support vectors.",
            "Broad stress/state harmonization is intentionally conservative. Ambiguous or protocol-heavy cohorts are retained as intermediate rather than forced into a biological label.",
            "The local decision-grade limitation is that a full local v6 within-type artifact folder was not present, so the shared-backbone comparison treats v6 as a missing local run rather than fabricating metrics.",
        ]
        add_text_page(
            pdf,
            "Appendix: Assumptions and Metric Notes",
            appendix_lines,
            subtitle="Metric definitions, composition profiling, and harmonization caveats used throughout the report.",
        )

    sections_md = [
        "# v8 Diagnostics Report Sections",
        "",
        "## Executive summary",
        "",
        *[f"- {line}" for line in executive_summary],
        "",
        "## Integrated architecture decision",
        "",
        *[f"- {line}" for line in architecture_decision],
    ]
    (args.output_dir / "report_sections.md").write_text("\n".join(sections_md) + "\n", encoding="utf-8")
    pd.DataFrame(figure_manifest_rows).to_csv(args.output_dir / "figure_manifest.csv", index=False)


if __name__ == "__main__":
    main()
