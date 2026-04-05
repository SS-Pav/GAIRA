#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from gaira.demo.v8_master_utils import MASTER_EV_DIR, MASTER_REPORT_DIR, MASTER_SERUM_DIR, MASTER_SHARED_DIR, MASTER_SMALL_DIR, ensure_dir, safe_csv, safe_text
from gaira.demo.v8_report_layout import add_figure_manifest_rows, add_text_page, maybe_image_grid, maybe_table_page


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the v8 master CPU-first report.")
    parser.add_argument("--shared-dir", type=Path, default=MASTER_SHARED_DIR)
    parser.add_argument("--ev-dir", type=Path, default=MASTER_EV_DIR)
    parser.add_argument("--small-dir", type=Path, default=MASTER_SMALL_DIR)
    parser.add_argument("--serum-dir", type=Path, default=MASTER_SERUM_DIR)
    parser.add_argument("--output-dir", type=Path, default=MASTER_REPORT_DIR)
    parser.add_argument("--title", default="GAIRAM v8 Master CPU Diagnostics and Training-Prep Report")
    return parser.parse_args()


def extract_bullets(text: str, limit: int = 8) -> list[str]:
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return bullets[:limit]


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    shared_metrics = safe_csv(args.shared_dir / "shared_backbone_metrics.csv")
    shared_compare = safe_csv(args.shared_dir / "shared_backbone_comparison.csv")
    ev_metrics = safe_csv(args.ev_dir / "ev_latent_structure_metrics.csv")
    ev_linkage = safe_csv(args.ev_dir / "ev_state_theme_linkage_metrics.csv")
    small_compare = safe_csv(args.small_dir / "small2023_comparison_table.csv")
    serum_metrics = safe_csv(args.serum_dir / "serum_dataset_latent_metrics.csv")
    serum_delta = safe_csv(args.serum_dir / "serum_delta_metrics.csv")
    serum_similarity = safe_csv(args.serum_dir / "serum_delta_similarity.csv")

    shared_summary = safe_text(args.shared_dir / "shared_backbone_summary.md")
    ev_summary = safe_text(args.ev_dir / "ev_stress_summary.md")
    small_summary = safe_text(args.small_dir / "small2023_summary.md")
    serum_summary = safe_text(args.serum_dir / "serum_summary.md")

    scorecard_rows = [
        {"decision_area": "shared_backbone", "decision": "freeze_v7_shared", "evidence": "best EV-facing shared compromise; v6 full artifacts missing locally"},
        {"decision_area": "ev_stress", "decision": "lead_v8_gpu_branch", "evidence": "strongest current biology-learning story"},
        {"decision_area": "small2023", "decision": "dedicated_specialized_head", "evidence": "shared encoder still below old specialized invariant benchmark"},
        {"decision_area": "serum", "decision": "cohort_only_not_head", "evidence": "within-dataset shifts exist but cross-dataset delta alignment remains weak"},
        {"decision_area": "prototype_timing", "decision": "deterministic_inference_before_rag", "evidence": "complete v8 EV outputs, then inference objects, then RAG/context layers"},
    ]
    scorecard_df = pd.DataFrame(scorecard_rows)
    scorecard_df.to_csv(args.output_dir / "v8_master_scorecard.csv", index=False)

    figure_manifest_rows: list[dict[str, str]] = []
    report_pdf = args.output_dir / "v8_master_report.pdf"

    with PdfPages(report_pdf) as pdf:
        add_text_page(
            pdf,
            args.title,
            [
                "CPU-first master decision memo for the next GAIRA v8 rollout. This report freezes the shared backbone decision, specifies the EV stress branch, separates the small2023 specialized benchmark head, and keeps serum in cohort-mode interpretation rather than forcing a premature serum manifold claim.",
                f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
                f"Repo root: {Path.cwd()}",
                f"Output directory: {args.output_dir}",
            ],
            subtitle="Nature-style internal technical report",
        )

        add_text_page(
            pdf,
            "Executive Summary",
            [
                "Freeze the shared backbone at v7 anchor-invariance behavior.",
                "Make EV stress/state the main biology-learning GPU branch.",
                "Keep small2023 separate as a specialized head problem rather than folding it into the shared EV branch.",
                "Keep serum in cohort-mode interpretation for now, because stable within-dataset shifts exist but cross-dataset serum delta alignment still does not support a shared serum manifold.",
                "Do not add context RAG yet. First complete deterministic v8 EV outputs, then build inference objects, then layer context graphs and LLM explanations.",
            ],
        )

        add_text_page(pdf, "Current Shared Backbone Status", extract_bullets(shared_summary))
        maybe_table_page(
            pdf,
            "Shared Backbone Metrics",
            shared_metrics.round(4),
            subtitle="Global and within-type purity metrics for the shared backbone candidates.",
            font_size=7.8,
        )
        maybe_table_page(
            pdf,
            "Shared Backbone Comparison Table",
            shared_compare.round(4),
            subtitle="Legacy v8 shared outputs compared against the new master re-exported schema.",
            font_size=6.8,
        )
        shared_images = [
            (args.shared_dir / "backbone_comparison_metrics.png", "Global comparison metrics"),
            (args.shared_dir / "sample_type_purity_comparison.png", "Sample-type purity comparison"),
            (args.shared_dir / "ev_within_type_metric_comparison.png", "Within-EV metrics"),
            (args.shared_dir / "serum_within_type_metric_comparison.png", "Within-serum metrics"),
            (args.shared_dir / "grounding_within_type_metric_comparison.png", "Within-grounding metrics"),
            (args.shared_dir / "cross_dataset_mixed_cluster_counts.png", "Cross-dataset mixed clusters"),
            (args.shared_dir / "cluster_size_distribution_comparison.png", "Cluster-count comparison"),
        ]
        maybe_image_grid(pdf, "Shared Backbone Figures", shared_images[:4], subtitle="Comparison figures for the frozen shared baseline decision.")
        maybe_image_grid(pdf, "Shared Backbone Figures II", shared_images[4:], subtitle="Grounding, mixing, and cluster-count comparisons.")
        add_figure_manifest_rows(figure_manifest_rows, "shared_backbone", [(p, c) for p, c in shared_images if p.exists()])

        add_text_page(pdf, "EV Stress / State Diagnostics", extract_bullets(ev_summary))
        maybe_table_page(pdf, "EV Stress Metrics", ev_metrics.round(4), subtitle="Latent structure metrics for the EV stress/state branch.")
        if not ev_linkage.empty:
            maybe_table_page(
                pdf,
                "EV Theme–State Linkage",
                ev_linkage.round(4),
                subtitle="Grounding-derived theme correlations with cluster-level stress enrichment.",
            )
        ev_images = [
            (args.ev_dir / "neutral_latent_map.png", "Unsupervised EV cluster structure"),
            (args.ev_dir / "latent_map_by_state.png", "State overlay"),
            (args.ev_dir / "latent_map_by_dominant_biochemical_theme.png", "Theme overlay"),
            (args.ev_dir / "cluster_composition_heatmap.png", "Cluster composition profiles"),
            (args.ev_dir / "cluster_state_heatmap.png", "Cluster state enrichment"),
            (args.ev_dir / "composition_vs_state_scatter.png", "Theme versus stress linkage"),
            (args.ev_dir / "dataset_vs_state_cluster_map.png", "Dataset/state cluster map"),
            (args.ev_dir / "state_enrichment_ranked_clusters.png", "Ranked stress-enriched clusters"),
        ]
        maybe_image_grid(pdf, "EV Stress Figures", ev_images[:4], subtitle="Latent structure first, biological meaning painted on second.")
        maybe_image_grid(pdf, "EV Stress Figures II", ev_images[4:], subtitle="State linkage and cluster-level interpretability.")
        add_figure_manifest_rows(figure_manifest_rows, "ev_stress", [(p, c) for p, c in ev_images if p.exists()])
        add_text_page(pdf, "EV Stress GPU Spec", safe_text(args.ev_dir / "ev_v8_training_spec.md").splitlines())

        add_text_page(pdf, "small2023 Specialized Benchmark", extract_bullets(small_summary))
        maybe_table_page(
            pdf,
            "small2023 Comparison Scorecard",
            small_compare.round(4),
            subtitle="Direct spectra, frozen shared backbones, and the old specialized v2 benchmark.",
            font_size=7.2,
        )
        small_images = [
            (args.small_dir / "direct_map_by_class.png", "Direct spectra by class"),
            (args.small_dir / "direct_map_by_probe.png", "Direct spectra by probe"),
            (args.small_dir / "v5_embedding_map_by_class.png", "v5 shared by class"),
            (args.small_dir / "v7_embedding_map_by_class.png", "v7 shared by class"),
            (args.small_dir / "v2_embedding_map_by_class.png", "v2 specialized by class"),
            (args.small_dir / "direct_vs_embedding_metric_bars.png", "Metric comparison"),
            (args.small_dir / "class_composition_heatmap.png", "Class composition heatmap"),
            (args.small_dir / "probe_transfer_scorecard.png", "Probe transfer scorecard"),
        ]
        maybe_image_grid(pdf, "small2023 Figures", small_images[:4], subtitle="Direct versus shared latent geometry.")
        maybe_image_grid(pdf, "small2023 Figures II", small_images[4:], subtitle="Specialized benchmark readout and class composition.")
        add_figure_manifest_rows(figure_manifest_rows, "small2023", [(p, c) for p, c in small_images if p.exists()])
        add_text_page(pdf, "small2023 GPU Spec", safe_text(args.small_dir / "small2023_v8_training_spec.md").splitlines())

        add_text_page(pdf, "Serum Cohort-Mode Diagnostics", extract_bullets(serum_summary))
        maybe_table_page(pdf, "Serum Dataset Latent Metrics", serum_metrics.round(4), subtitle="Per-dataset cohort-mode readout rather than a forced shared manifold table.")
        maybe_table_page(pdf, "Serum Delta Metrics", serum_delta.round(4), subtitle="Within-dataset low-to-high state delta stability and theme shifts.", font_size=7.0)
        maybe_table_page(pdf, "Serum Delta Similarity", serum_similarity.round(4), subtitle="Cross-dataset delta alignment for the included biological contrast cohorts.", font_size=7.0)
        serum_images = [
            (args.serum_dir / "serum_dataset_latent_maps.png", "Per-dataset serum latent maps"),
            (args.serum_dir / "serum_dataset_state_maps.png", "Per-dataset state overlays"),
            (args.serum_dir / "serum_dataset_composition_heatmaps.png", "Per-dataset composition structure"),
            (args.serum_dir / "serum_delta_similarity_heatmap.png", "Cross-dataset serum delta similarity"),
            (args.serum_dir / "serum_theme_shift_summary.png", "Cross-dataset theme-shift consistency"),
            (args.serum_dir / "serum_within_dataset_state_scatter.png", "Within-dataset state composition scatter"),
        ]
        maybe_image_grid(pdf, "Serum Figures", serum_images[:3], subtitle="Cohort-mode serum structure and composition.")
        maybe_image_grid(pdf, "Serum Figures II", serum_images[3:], subtitle="Delta alignment diagnostics across the usable biological contrast cohorts.")
        add_figure_manifest_rows(figure_manifest_rows, "serum", [(p, c) for p, c in serum_images if p.exists()])
        add_text_page(pdf, "Serum Recommendation", safe_text(args.serum_dir / "serum_v8_recommendation.md").splitlines())

        add_text_page(
            pdf,
            "Recommended v8 GPU Training Plan",
            [
                "1. Freeze the shared initialization at v7 anchor-invariance outputs.",
                "2. Run the EV stress/state GPU branch first.",
                "3. Run the small2023 specialized head as a separate benchmark branch rather than mixing it into the EV stress training loop.",
                "4. Keep serum in cohort-only interpretation mode while harmonization and anchor work continue.",
            ],
        )

        rag_lines = [
            "Recommendation:",
            "- do not add context RAG before the first v8 EV GPU runs",
            "- first produce deterministic EV stress outputs and benchmark them",
            "- then build deterministic inference objects",
            "- then add context RAG and richer context graphs",
            "- only after deterministic objects exist should an LLM explanation layer become a first-class product layer",
        ]
        add_text_page(pdf, "Post-v8 Prototype Timing", rag_lines, subtitle="RAG and explanation timing should follow deterministic outputs, not precede them.")

    sections = [
        "# v8 Master Report Sections",
        "",
        "## Executive summary",
        "- freeze v7 shared backbone",
        "- launch EV stress/state as the lead GPU branch",
        "- keep small2023 as a specialized head problem",
        "- keep serum as cohort-only interpretation for now",
        "- delay context RAG until deterministic EV outputs exist",
    ]
    (args.output_dir / "report_sections.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    pd.DataFrame(figure_manifest_rows).to_csv(args.output_dir / "figure_manifest.csv", index=False)
    (args.output_dir / "rag_timing_recommendation.md").write_text(
        "# RAG / Inference Timing Recommendation\n\n"
        "- complete deterministic v8 EV outputs first\n"
        "- then build deterministic inference objects\n"
        "- then add context RAG\n"
        "- then build richer context graph and explanation UI layers\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
