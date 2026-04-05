from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from gaira.demo.gaira_pilot_utils import build_pdf_report


SSD_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1")
OUT_ROOT = SSD_ROOT / "pilot1_abc_summary_v1"

PILOT1A_ROOT = SSD_ROOT / "pilot1a_celltype_probe1_v5"
PILOT1B_ROOT = SSD_ROOT / "pilot1b_mixture_probe1_v1"
PILOT1C_ROOT = SSD_ROOT / "pilot1c_probe_consistency_v1"
PASS5_ROOT = SSD_ROOT / "pass5_saturation_fix"


def _copy_figure(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _fmt(x: float, digits: int = 4) -> str:
    return f"{float(x):.{digits}f}"


def _load_single_row(df: pd.DataFrame, config_id: str) -> pd.Series:
    row = df[df["config_id"] == config_id]
    if row.empty:
        raise KeyError(f"Missing config_id={config_id}")
    return row.iloc[0]


def main() -> None:
    figures_dir = OUT_ROOT / "figures"
    tables_dir = OUT_ROOT / "tables"
    report_dir = OUT_ROOT / "report"
    for path in [figures_dir, tables_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)

    pilot1a = pd.read_csv(PILOT1A_ROOT / "tables" / "pilot1a_v5_comparator.csv")
    pilot1b = pd.read_csv(PILOT1B_ROOT / "tables" / "pilot1b_cfg05_vs_cfg08_comparison.csv")
    pilot1c = pd.read_csv(PILOT1C_ROOT / "tables" / "probe_consistency_metrics.csv")
    pass5 = pd.read_csv(PASS5_ROOT / "tables" / "calibration_results_ranked.csv")

    baseline_1a = _load_single_row(pilot1a, "baseline_v1_locked_purine")
    cfg05_1a = _load_single_row(pilot1a, "candidate_v2_cfg05_max_desaturation")
    cfg08_1a = _load_single_row(pilot1a, "candidate_v2_cfg08_balanced_update")
    cfg05_1b = _load_single_row(pilot1b, "candidate_v2_cfg05_max_desaturation")
    cfg08_1b = _load_single_row(pilot1b, "candidate_v2_cfg08_balanced_update")
    cfg05_1c = _load_single_row(pilot1c, "candidate_v2_cfg05_max_desaturation")
    cfg08_1c = _load_single_row(pilot1c, "candidate_v2_cfg08_balanced_update")
    cfg05_p5 = _load_single_row(pass5, "cfg05")
    cfg08_p5 = _load_single_row(pass5, "cfg08")
    baseline_p5 = _load_single_row(pass5, "cfg02")

    summary_rows = [
        {
            "config_id": "baseline_v1_locked_purine",
            "role": "narrow_reference",
            "pass5_validation_score": float(baseline_p5["validation_score"]),
            "pilot1a_delta_accuracy": float(baseline_1a["classification_accuracy_delta"]),
            "pilot1a_mean_top1_dominance": float(baseline_1a["mean_top1_dominance"]),
            "pilot1a_mean_neighborhood_entropy": float(baseline_1a["mean_neighborhood_entropy"]),
            "pilot1a_mean_inter_class_distance_delta": float(baseline_1a["mean_inter_class_distance_delta"]),
            "pilot1b_progression_combined_spearman": None,
            "pilot1b_noncollapse_ratio": None,
            "pilot1b_intermediate_distinct_count": None,
            "pilot1c_probe2_progression_spearman": None,
            "pilot1c_pairwise_distance_spearman": None,
            "pilot1c_mean_family_drift": None,
            "decision_note": "Narrow purine-heavy reference only.",
        },
        {
            "config_id": "candidate_v2_cfg05_max_desaturation",
            "role": "working_default",
            "pass5_validation_score": float(cfg05_p5["validation_score"]),
            "pilot1a_delta_accuracy": float(cfg05_1a["classification_accuracy_delta"]),
            "pilot1a_mean_top1_dominance": float(cfg05_1a["mean_top1_dominance"]),
            "pilot1a_mean_neighborhood_entropy": float(cfg05_1a["mean_neighborhood_entropy"]),
            "pilot1a_mean_inter_class_distance_delta": float(cfg05_1a["mean_inter_class_distance_delta"]),
            "pilot1b_progression_combined_spearman": float(cfg05_1b["progression_combined_spearman"]),
            "pilot1b_noncollapse_ratio": float(cfg05_1b["noncollapse_ratio"]),
            "pilot1b_intermediate_distinct_count": float(cfg05_1b["intermediate_distinct_count"]),
            "pilot1c_probe2_progression_spearman": float(cfg05_1c["probe2_progression_spearman"]),
            "pilot1c_pairwise_distance_spearman": float(cfg05_1c["pairwise_distance_spearman"]),
            "pilot1c_mean_family_drift": float(cfg05_1c["mean_family_drift"]),
            "decision_note": "Best interpretability and probe stability tradeoff.",
        },
        {
            "config_id": "candidate_v2_cfg08_balanced_update",
            "role": "broadness_comparator",
            "pass5_validation_score": float(cfg08_p5["validation_score"]),
            "pilot1a_delta_accuracy": float(cfg08_1a["classification_accuracy_delta"]),
            "pilot1a_mean_top1_dominance": float(cfg08_1a["mean_top1_dominance"]),
            "pilot1a_mean_neighborhood_entropy": float(cfg08_1a["mean_neighborhood_entropy"]),
            "pilot1a_mean_inter_class_distance_delta": float(cfg08_1a["mean_inter_class_distance_delta"]),
            "pilot1b_progression_combined_spearman": float(cfg08_1b["progression_combined_spearman"]),
            "pilot1b_noncollapse_ratio": float(cfg08_1b["noncollapse_ratio"]),
            "pilot1b_intermediate_distinct_count": float(cfg08_1b["intermediate_distinct_count"]),
            "pilot1c_probe2_progression_spearman": float(cfg08_1c["probe2_progression_spearman"]),
            "pilot1c_pairwise_distance_spearman": float(cfg08_1c["pairwise_distance_spearman"]),
            "pilot1c_mean_family_drift": float(cfg08_1c["mean_family_drift"]),
            "decision_note": "Broadest vocabulary, but too diffuse and probe-unstable.",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / "pilot1_abc_summary_metrics.csv", index=False)

    figure_specs = [
        ("01_pilot1a_spectral_pca.png", PILOT1A_ROOT / "figures" / "pca_spectral_original_dataset.png"),
        ("02_pilot1a_delta_radar_baseline.png", PILOT1A_ROOT / "figures" / "radar_delta_fixed_baseline_v1_locked_purine.png"),
        ("03_pilot1a_delta_radar_cfg05.png", PILOT1A_ROOT / "figures" / "radar_delta_fixed_candidate_v2_cfg05_max_desaturation.png"),
        ("04_pilot1a_delta_radar_cfg08.png", PILOT1A_ROOT / "figures" / "radar_delta_fixed_candidate_v2_cfg08_balanced_update.png"),
        ("05_pilot1a_pca_bsv_cfg05.png", PILOT1A_ROOT / "figures" / "pca_bsv_candidate_v2_cfg05_max_desaturation.png"),
        ("06_pilot1a_pca_bsv_cfg08.png", PILOT1A_ROOT / "figures" / "pca_bsv_candidate_v2_cfg08_balanced_update.png"),
        ("07_pilot1a_tradeoff.png", PILOT1A_ROOT / "figures" / "pilot1a_config_tradeoff_summary.png"),
        ("08_pilot1b_endpoint_reference.png", PILOT1B_ROOT / "figures" / "endpoint_fingerprint_reference.png"),
        ("09_pilot1b_progression_cfg05.png", PILOT1B_ROOT / "figures" / "mixture_progression_alignment_cfg05.png"),
        ("10_pilot1b_progression_cfg08.png", PILOT1B_ROOT / "figures" / "mixture_progression_alignment_cfg08.png"),
        ("11_pilot1b_delta_grid_cfg05.png", PILOT1B_ROOT / "figures" / "delta_fingerprint_grid_cfg05.png"),
        ("12_pilot1b_delta_grid_cfg08.png", PILOT1B_ROOT / "figures" / "delta_fingerprint_grid_cfg08.png"),
        ("13_pilot1b_tradeoff.png", PILOT1B_ROOT / "figures" / "pilot1b_cfg05_vs_cfg08_tradeoff.png"),
        ("14_pilot1c_bsv_overlay_cfg05.png", PILOT1C_ROOT / "figures" / "pca_bsv_probe1_vs_probe2_overlay_cfg05.png"),
        ("15_pilot1c_bsv_overlay_cfg08.png", PILOT1C_ROOT / "figures" / "pca_bsv_probe1_vs_probe2_overlay_cfg08.png"),
        ("16_pilot1c_progression_cfg05.png", PILOT1C_ROOT / "figures" / "progression_probe1_vs_probe2_cfg05.png"),
        ("17_pilot1c_progression_cfg08.png", PILOT1C_ROOT / "figures" / "progression_probe1_vs_probe2_cfg08.png"),
        ("18_pilot1c_family_cfg05.png", PILOT1C_ROOT / "figures" / "family_fingerprint_probe1_vs_probe2_cfg05.png"),
        ("19_pilot1c_family_cfg08.png", PILOT1C_ROOT / "figures" / "family_fingerprint_probe1_vs_probe2_cfg08.png"),
        ("20_pilot1c_drift_cfg05.png", PILOT1C_ROOT / "figures" / "class_drift_barplot_cfg05.png"),
        ("21_pilot1c_drift_cfg08.png", PILOT1C_ROOT / "figures" / "class_drift_barplot_cfg08.png"),
        ("22_pilot1c_tradeoff.png", PILOT1C_ROOT / "figures" / "cfg05_vs_cfg08_probe_consistency_tradeoff.png"),
    ]
    copied_figures: list[Path] = []
    for name, src in figure_specs:
        copied_figures.append(_copy_figure(src, figures_dir / name))

    report_md = report_dir / "GAIRAv3_Pilot1_ABC_summary_report.md"
    report_pdf = report_dir / "GAIRAv3_Pilot1_ABC_summary_report.pdf"

    lines = [
        "# GAIRAv3 Pilot 1 Summary Report",
        "",
        "## 1. Introduction",
        "Raman and SERS measurements are mixture problems: many biochemical contributors can support the same observed spectrum, and static peak-level interpretation is underdetermined.",
        "GAIRA addresses that by building Biochemical Signature Vectors instead of forcing direct molecule labels. The practical goal is to recover stable biochemical themes, compositional structure, and probe-robust reasoning signals rather than optimize for the single largest class separation.",
        "",
        "## 2. How Configurations Were Chosen",
        "The three fixed Pilot 1 configurations came from the pass 5 saturation-fix harness, which optimized a deterministic objective combining validation performance, mixture progression recovery, entropy gain, and saturation penalties.",
        f"Pass 5 reference scores: baseline validation `{_fmt(baseline_p5['validation_score'])}`, cfg05 validation `{_fmt(cfg05_p5['validation_score'])}`, cfg08 validation `{_fmt(cfg08_p5['validation_score'])}`.",
        f"Pass 5 mixture-panel scores: cfg05 `{_fmt(cfg05_p5['mixture_panel_score'])}`, cfg08 `{_fmt(cfg08_p5['mixture_panel_score'])}`.",
        "Interpretation of the three configurations:",
        "- `baseline_v1_locked_purine`: narrow purine-heavy reference with high dominance and compressed chemistry vocabulary.",
        "- `candidate_v2_cfg05_max_desaturation`: controlled desaturation while staying anchored in the purine-adjacent chemistry neighborhood.",
        "- `candidate_v2_cfg08_balanced_update`: broader vocabulary with lower dominance and stronger raw validation-side performance, but at the risk of diffusion.",
        "The selection problem was therefore not just accuracy. It was to balance desaturation, biochemical coherence, validation behavior, and cross-panel stability.",
        "",
        "## 3. Pilot 1a: Cell-Type Fingerprinting on Probe 1",
        "Spectral PCA is shown once because it does not change across configs: the spectra are fixed, so the raw spectral geometry is unchanged.",
        "What changes is the BSV geometry and the fingerprint representation built on top of it.",
        f"Baseline remained narrow and purine-dominant with mean top1 dominance `{_fmt(baseline_1a['mean_top1_dominance'])}` and mean neighborhood entropy `{_fmt(baseline_1a['mean_neighborhood_entropy'])}`.",
        f"cfg05 broadened the fingerprint in a controlled way: mean top1 dominance fell to `{_fmt(cfg05_1a['mean_top1_dominance'])}` while mean neighborhood entropy rose to `{_fmt(cfg05_1a['mean_neighborhood_entropy'])}`.",
        f"cfg08 broadened further: mean top1 dominance `{_fmt(cfg08_1a['mean_top1_dominance'])}`, mean neighborhood entropy `{_fmt(cfg08_1a['mean_neighborhood_entropy'])}`, and the strongest class-label recovery in 1a with delta accuracy `{_fmt(cfg08_1a['classification_accuracy_delta'])}`.",
        "The fixed-axis delta radars are the most informative view. They show that the baseline exaggerates differences inside a narrow chemistry vocabulary, cfg05 introduces more controlled multi-axis variation, and cfg08 broadens the support further but makes the class fingerprints less tightly anchored to a single biochemical interpretation.",
        "On the cell-type panel alone, cfg05 was the best interpretability compromise: broader than baseline, but less diffuse than cfg08.",
        "",
        "## 4. Pilot 1b: Mixture Progression on Probe 1",
        "Pilot 1b asked whether the fingerprint stack behaves compositionally on the ordered mixture series `c00 -> c01 -> c10 -> c25 -> c50 -> c100`.",
        f"cfg05 produced a strong ordered progression with combined Spearman `{_fmt(cfg05_1b['progression_combined_spearman'])}`, noncollapse ratio `{_fmt(cfg05_1b['noncollapse_ratio'])}`, and intermediate distinct count `{int(cfg05_1b['intermediate_distinct_count'])}`.",
        f"cfg08 matched cfg05 on the headline combined Spearman `{_fmt(cfg08_1b['progression_combined_spearman'])}`, but it merged more of the middle of the series: noncollapse ratio `{_fmt(cfg08_1b['noncollapse_ratio'])}` and intermediate distinct count `{int(cfg08_1b['intermediate_distinct_count'])}`.",
        f"cfg05 also preserved a more coherent endpoint-directed structure with endpoint combined separation `{_fmt(cfg05_1b['endpoint_combined_separation'])}` versus `{_fmt(cfg08_1b['endpoint_combined_separation'])}` for cfg08.",
        "The key scientific point is that GAIRA is not demanding perfect linearity. The question is whether the mixture fingerprints move coherently and whether intermediate states remain interpretable rather than collapsing into one endpoint-heavy regime.",
        "On that criterion, cfg05 was better: it preserved intermediate states and yielded the cleaner progression story.",
        "",
        "## 5. Pilot 1c: Probe Consistency Between Probe 1 and Probe 2",
        "The originally intended 1c cell-type Probe 2 study was not possible because a true `small2023_celltype_probe2` subset is not present locally. The available decision-grade substitute is mixture probe consistency, comparing `small2023_mixture_probe1` and `small2023_mixture_probe2` under the same fixed configs.",
        f"cfg05 retained ordered progression on Probe 2 with progression Spearman `{_fmt(cfg05_1c['probe2_progression_spearman'])}`, pairwise distance Spearman `{_fmt(cfg05_1c['pairwise_distance_spearman'])}`, and mean family drift `{_fmt(cfg05_1c['mean_family_drift'])}`.",
        f"cfg08 degraded sharply across probes: Probe 2 progression Spearman `{_fmt(cfg08_1c['probe2_progression_spearman'])}`, pairwise distance Spearman `{_fmt(cfg08_1c['pairwise_distance_spearman'])}`, and mean family drift `{_fmt(cfg08_1c['mean_family_drift'])}`.",
        f"Neighborhood overlap stayed perfect for cfg05 (`{_fmt(cfg05_1c['neighborhood_overlap_score'])}`) and nearly vanished for cfg08 (`{_fmt(cfg08_1c['neighborhood_overlap_score'])}`).",
        "This is the decisive stability result. cfg08 is broader, but that broader vocabulary is not probe-stable enough for the current scientific objective. cfg05 keeps a chemically coherent support structure while preserving more of the ordered trajectory across probes.",
        "",
        "## 6. Final Decision",
        "The working GAIRAv3 Pilot 1 configuration is `candidate_v2_cfg05_max_desaturation`.",
        "Why cfg05 is selected:",
        "- It broadens the narrow baseline without losing biochemical coherence.",
        "- It supports mixture reasoning better than cfg08 on Probe 1 because intermediate states remain distinguishable.",
        "- It remains materially more stable across probes than cfg08, with lower drift and better preservation of ordered progression.",
        "How to treat the other configurations:",
        "- `baseline_v1_locked_purine` remains the narrow reference.",
        "- `candidate_v2_cfg08_balanced_update` remains the broadness comparator and upper-bound vocabulary expansion reference.",
        "",
        "## 7. Key Insight",
        "GAIRA is not optimizing for the largest raw separation or the highest single classification score.",
        "It is optimizing for stable, interpretable biochemical structure: support patterns that preserve mixture reasoning, resist collapse, and remain coherent across probes.",
        "That is why cfg05 wins the Pilot 1 decision even though cfg08 is broader and stronger on some raw classification-oriented metrics.",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    build_pdf_report(report_md, copied_figures, report_pdf)

    print(f"Summary metrics: {tables_dir / 'pilot1_abc_summary_metrics.csv'}")
    print(f"Markdown report: {report_md}")
    print(f"PDF report: {report_pdf}")


if __name__ == "__main__":
    main()
