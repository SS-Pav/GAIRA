#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_DATA_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed")
DEFAULT_OUTPUT_PDF = Path("reports/GAIRAM_embedding_evolution_report.pdf")
DEFAULT_PREVIOUS_REPORT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v2_gpu_run1_final/GAIRAM_v2_GPU_Run1_Report.pdf"
)

EXPECTED_STAGE_SPECS = [
    {
        "key": "v2_gpu_run1_final",
        "title": "GAIRAM v2 GPU Run 1 Final",
        "run_dir_name": "embedding_v2_gpu_run1_final",
        "eval_dir_name": None,
        "stage_order": 2,
        "narrative": (
            "First credible GPU-scale contrastive result on a balanced subset. This run established "
            "that sample type can be learned cleanly from spectra alone, but also showed that dataset "
            "identity remained a dominant organizing factor."
        ),
    },
    {
        "key": "v3_pass3_gpu_run2",
        "title": "GAIRAM v3 Pass 3 GPU Run 2",
        "run_dir_name": "embedding_v3_pass3_gpu_run2",
        "eval_dir_name": None,
        "stage_order": 3,
        "narrative": (
            "Intermediate biology-aware pass using tempered semantic positives, hard negatives, and "
            "region-aware augmentations. This stage tested whether similarity could shift away from "
            "pure dataset memorization without losing sample-type structure."
        ),
    },
    {
        "key": "v5_full_true_gpu_run1",
        "title": "GAIRAM v5 Full True GPU Run 1",
        "run_dir_name": "embedding_v5_full_true_gpu_run1",
        "eval_dir_name": "embedding_eval_v2/embedding_v5_full_true_gpu_run1_eval_v2",
        "stage_order": 5,
        "narrative": (
            "First true full-corpus training run. This stage separated large-scale training from a new "
            "scalable evaluation design and revealed the current hierarchy of learned structure: "
            "sample type first, dataset identity second, and biology emerging only after scale."
        ),
    },
]

PRIOR_REPORT_SUMMARY = [
    (
        "The previous GPU report established that the embedding model had already learned a physically "
        "credible sample-type manifold: serum, EV, and grounding spectra were not randomly mixed."
    ),
    (
        "At the same time, the dominant limitation was clear. Dataset identity remained stronger than "
        "coherent biological class structure, implying that the model was still using acquisition- and "
        "archive-specific shortcuts rather than reliably learning within-domain biochemical invariants."
    ),
    (
        "That report therefore set the next methodological hypothesis: preserve the useful sample-type "
        "separation, but alter the training objective and sampling regime so the encoder is pressured "
        "to stop memorizing dataset boundaries and begin learning biologically meaningful similarity."
    ),
]


@dataclass
class StageArtifacts:
    key: str
    title: str
    stage_order: int
    narrative: str
    run_dir: Path | None = None
    eval_dir: Path | None = None
    run_config: dict[str, Any] = field(default_factory=dict)
    train_log: pd.DataFrame | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    probe_metrics: pd.DataFrame | None = None
    report_excerpt: str = ""
    probe_excerpt: str = ""
    images: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int | None:
        value = self.run_config.get("samples")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a multi-page PDF report covering GAIRAM embedding evolution."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--previous-report-pdf", type=Path, default=DEFAULT_PREVIOUS_REPORT)
    parser.add_argument(
        "--title",
        default="GAIRAM Embedding Evolution Report",
    )
    parser.add_argument("--author", default="Codex for Suraj / GAIRAM")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def clean_existing(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.exists() else None


def read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def read_csv(path: Path | None) -> pd.DataFrame | None:
    if not path or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def metric_dict_from_df(df: pd.DataFrame | None) -> dict[str, float]:
    if df is None or df.empty or "metric" not in df.columns or "value" not in df.columns:
        return {}
    output: dict[str, float] = {}
    for _, row in df.iterrows():
        metric = str(row["metric"])
        try:
            output[metric] = float(row["value"])
        except (TypeError, ValueError):
            continue
    return output


def read_md_excerpt(path: Path | None, max_chars: int = 1200) -> str:
    if not path or not path.exists():
        return ""
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return ""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("._"):
            continue
        lines.append(stripped)
    return " ".join(lines)[:max_chars]


def find_image(dir_path: Path | None, stem: str) -> Path | None:
    if not dir_path or not dir_path.exists():
        return None
    candidate = dir_path / stem
    if candidate.exists():
        return candidate
    return None


def discover_stage(data_root: Path, spec: dict[str, Any], verbose: bool) -> StageArtifacts:
    run_dir = clean_existing(data_root / spec["run_dir_name"])
    eval_dir = clean_existing(data_root / spec["eval_dir_name"]) if spec["eval_dir_name"] else None
    stage = StageArtifacts(
        key=spec["key"],
        title=spec["title"],
        stage_order=spec["stage_order"],
        narrative=spec["narrative"],
        run_dir=run_dir,
        eval_dir=eval_dir,
    )
    if not run_dir:
        stage.notes.append(f"Run folder `{spec['run_dir_name']}` was not found under data root.")
        return stage

    run_config = read_json(run_dir / "run_config.json")
    train_log = read_csv(run_dir / "training_log.csv")
    metrics_df = read_csv(run_dir / "embedding_metrics.csv")
    probe_df = read_csv(run_dir / "probe_metrics.csv")
    md_excerpt = read_md_excerpt(run_dir / "embedding_report.md")
    probe_excerpt = read_md_excerpt(run_dir / "probe_report.md")

    # Prefer eval_v2 metrics when present for the full-corpus run.
    if eval_dir:
        eval_metrics_df = read_csv(eval_dir / "embedding_metrics_v2.csv")
        eval_probe_df = read_csv(eval_dir / "probe_metrics.csv")
        if eval_metrics_df is not None:
            metrics_df = eval_metrics_df
            md_excerpt = read_md_excerpt(eval_dir / "embedding_report_v2.md") or md_excerpt
        if eval_probe_df is not None:
            probe_df = eval_probe_df
            probe_excerpt = read_md_excerpt(eval_dir / "probe_report.md") or probe_excerpt

    stage.run_config = run_config
    stage.train_log = train_log
    stage.metrics = metric_dict_from_df(metrics_df)
    stage.probe_metrics = probe_df
    stage.report_excerpt = md_excerpt
    stage.probe_excerpt = probe_excerpt

    image_dir = eval_dir if eval_dir else run_dir
    for name in ("umap_sample_type.png", "umap_dataset.png", "umap_class.png"):
        image_path = find_image(image_dir, name)
        if image_path:
            stage.images[name] = image_path
    if not stage.images:
        stage.notes.append("No UMAP images were found for this stage.")
    if not stage.metrics:
        stage.notes.append("No embedding metrics were found for this stage.")
    if stage.probe_metrics is None:
        stage.notes.append("No probe metrics were found for this stage.")
    log(verbose, f"Discovered stage {stage.key}: run={run_dir} eval={eval_dir}")
    return stage


def wrap_paragraphs(paragraphs: Iterable[str], width: int = 100) -> str:
    output = []
    for paragraph in paragraphs:
        output.append(textwrap.fill(paragraph, width=width))
    return "\n\n".join(output)


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str], footer: str | None = None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.07, 0.95, title, fontsize=20, fontweight="bold", va="top")
    fig.text(0.07, 0.91, wrap_paragraphs(paragraphs, width=95), fontsize=10.5, va="top")
    if footer:
        fig.text(0.07, 0.04, footer, fontsize=8, color="#666666")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def table_page(
    pdf: PdfPages,
    title: str,
    dataframe: pd.DataFrame,
    subtitle: str | None = None,
    footer: str | None = None,
    font_size: float = 8.0,
    scale_y: float = 1.3,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.82])
    ax.axis("off")
    fig.text(0.05, 0.96, title, fontsize=18, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.05, 0.93, textwrap.fill(subtitle, 110), fontsize=9.5, va="top")
    table = ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc="center",
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, scale_y)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#d9e2f3")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f7fa")
    if footer:
        fig.text(0.05, 0.03, footer, fontsize=8, color="#666666")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_title_page(
    pdf: PdfPages,
    report_title: str,
    author: str,
    data_root: Path,
    output_pdf: Path,
    stages: list[StageArtifacts],
    previous_report_pdf: Path,
) -> None:
    found_stage_names = [stage.title for stage in stages if stage.run_dir]
    summary_text = [
        "Technical progress report tracing the GAIRAM spectral embedding programme from the earlier "
        "GPU baseline through the subsequent biology-aware and full-corpus runs.",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Repo: {Path.cwd()}",
        f"Data root: {data_root}",
        f"Output PDF: {output_pdf}",
        f"Prior continuity anchor: {previous_report_pdf if previous_report_pdf.exists() else 'not found'}",
        "Stages discovered locally: " + (", ".join(found_stage_names) if found_stage_names else "none"),
    ]
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.07, 0.90, report_title, fontsize=26, fontweight="bold", va="top")
    fig.text(0.07, 0.85, "GAIRAM embedding evolution: methods, scaling, failure modes, and next constraints", fontsize=13)
    fig.text(0.07, 0.79, f"Author: {author}", fontsize=11)
    fig.text(0.07, 0.75, wrap_paragraphs(summary_text, width=95), fontsize=10.5, va="top")
    fig.text(0.07, 0.08, "Internal technical memo. Figures and metrics were assembled automatically from local GAIRAM artifacts.", fontsize=8.5, color="#666666")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def summarize_training(stage: StageArtifacts) -> str:
    cfg = stage.run_config or {}
    parts = []
    if stage.sample_count is not None:
        parts.append(f"{stage.sample_count:,} spectra")
    if cfg.get("epochs") is not None:
        parts.append(f"{cfg['epochs']} epochs")
    if cfg.get("batch_size") is not None:
        parts.append(f"batch {cfg['batch_size']}")
    if cfg.get("device"):
        parts.append(str(cfg["device"]).upper())
    if cfg.get("preset"):
        parts.append(f"preset={cfg['preset']}")
    elif cfg.get("positive_pair_mode"):
        parts.append(f"objective={cfg['positive_pair_mode']}")
    return ", ".join(parts)


def training_curve_summary(train_log: pd.DataFrame | None) -> str:
    if train_log is None or train_log.empty or "loss" not in train_log.columns:
        return "Training log not available."
    start = float(train_log["loss"].iloc[0])
    end = float(train_log["loss"].iloc[-1])
    epochs = len(train_log)
    return f"Loss declined from {start:.3f} to {end:.3f} over {epochs} epochs."


def stage_metric(stage: StageArtifacts, key: str) -> str:
    value = stage.metrics.get(key)
    return f"{value:.3f}" if value is not None else "NA"


def probe_lookup(stage: StageArtifacts, task_name: str, field: str = "macro_f1") -> str:
    if stage.probe_metrics is None or stage.probe_metrics.empty:
        return "NA"
    subset = stage.probe_metrics[stage.probe_metrics["task_name"] == task_name]
    if subset.empty or field not in subset.columns:
        return "NA"
    try:
        return f"{float(subset.iloc[0][field]):.3f}"
    except (TypeError, ValueError):
        return "NA"


def comparative_metrics_table(stages: list[StageArtifacts]) -> pd.DataFrame:
    rows = []
    for stage in stages:
        if not stage.run_dir:
            continue
        rows.append(
            {
                "Run": stage.title,
                "N spectra": f"{stage.sample_count:,}" if stage.sample_count is not None else "NA",
                "NN sample_type": stage_metric(stage, "nn_consistency_sample_type"),
                "NN dataset_id": stage_metric(stage, "nn_consistency_dataset_id"),
                "NN class": stage_metric(stage, "nn_consistency_class"),
                "NN family": stage_metric(stage, "nn_consistency_family"),
                "Sil sample_type": stage_metric(stage, "silhouette_sample_type"),
                "Sil dataset_id": stage_metric(stage, "silhouette_dataset_id"),
                "Sil class": stage_metric(stage, "silhouette_class"),
                "Sil family": stage_metric(stage, "silhouette_family"),
            }
        )
    return pd.DataFrame(rows)


def comparative_probe_table(stages: list[StageArtifacts]) -> pd.DataFrame:
    rows = []
    for stage in stages:
        if not stage.run_dir:
            continue
        rows.append(
            {
                "Run": stage.title,
                "Probe sample_type F1": probe_lookup(stage, "sample_type"),
                "Probe family F1": probe_lookup(stage, "family_label"),
                "Probe CCA/HCC/LM F1": probe_lookup(stage, "cca_hcc_lm_serum_sers_class"),
                "Probe EV family F1": probe_lookup(stage, "ev_dataset_family"),
            }
        )
    return pd.DataFrame(rows)


def methods_table(stages: list[StageArtifacts]) -> pd.DataFrame:
    rows = []
    for stage in stages:
        if not stage.run_dir:
            continue
        cfg = stage.run_config
        rows.append(
            {
                "Run": stage.title,
                "Samples": f"{stage.sample_count:,}" if stage.sample_count is not None else "NA",
                "Device": str(cfg.get("device", "NA")).upper(),
                "Epochs": cfg.get("epochs", "NA"),
                "Batch": cfg.get("batch_size", "NA"),
                "Positive pairs": cfg.get("positive_pair_mode", "instance_only"),
                "Semantic wt": cfg.get("semantic_positive_weight", "NA"),
                "Hard negatives": cfg.get("hard_negative_mode", "none"),
                "Hard-neg wt": cfg.get("hard_negative_weight", "NA"),
                "Augmentation": cfg.get("augmentation_mode", "baseline"),
                "Aug strength": cfg.get("augmentation_strength", "NA"),
            }
        )
    return pd.DataFrame(rows)


def add_stage_result_pages(pdf: PdfPages, stage: StageArtifacts) -> None:
    if not stage.run_dir:
        add_text_page(
            pdf,
            stage.title,
            [
                stage.narrative,
                "This stage was expected in the report sequence but its folder was not found locally, "
                "so it could not be analyzed directly.",
            ]
            + stage.notes,
        )
        return

    cfg = stage.run_config
    paragraphs = [
        stage.narrative,
        f"Training summary: {summarize_training(stage)}. {training_curve_summary(stage.train_log)}",
        (
            f"Core neighborhood metrics: sample_type {stage_metric(stage, 'nn_consistency_sample_type')}, "
            f"dataset_id {stage_metric(stage, 'nn_consistency_dataset_id')}, "
            f"class {stage_metric(stage, 'nn_consistency_class')}, "
            f"family {stage_metric(stage, 'nn_consistency_family')}."
        ),
        (
            f"Probe macro-F1: sample_type {probe_lookup(stage, 'sample_type')}, "
            f"family {probe_lookup(stage, 'family_label')}, "
            f"CCA/HCC/LM {probe_lookup(stage, 'cca_hcc_lm_serum_sers_class')}, "
            f"EV family {probe_lookup(stage, 'ev_dataset_family')}."
        ),
    ]
    if cfg:
        method_bits = []
        if cfg.get("dataset_path"):
            method_bits.append(f"dataset artifact = {cfg['dataset_path']}")
        if cfg.get("learning_rate"):
            method_bits.append(f"learning rate = {cfg['learning_rate']}")
        if cfg.get("temperature"):
            method_bits.append(f"temperature = {cfg['temperature']}")
        if cfg.get("preset"):
            method_bits.append(f"preset = {cfg['preset']}")
        if method_bits:
            paragraphs.append("Method details: " + "; ".join(method_bits) + ".")
    if stage.report_excerpt:
        paragraphs.append("Recorded evaluation excerpt: " + stage.report_excerpt[:900])
    if stage.notes:
        paragraphs.append("Artifact notes: " + " ".join(stage.notes))
    add_text_page(pdf, stage.title, paragraphs)

    image_items = []
    for image_name, caption in [
        ("umap_sample_type.png", "UMAP coloured by sample type"),
        ("umap_dataset.png", "UMAP coloured by dataset identity"),
        ("umap_class.png", "UMAP coloured by class label where available"),
    ]:
        if image_name in stage.images:
            image_items.append((stage.images[image_name], caption))
    if image_items:
        add_image_grid_page(
            pdf,
            title=f"{stage.title} — Visual geometry",
            image_items=image_items,
            subtitle="Local UMAP visualizations discovered in the run folder or its evaluation companion.",
        )


def add_image_grid_page(
    pdf: PdfPages,
    title: str,
    image_items: list[tuple[Path, str]],
    subtitle: str | None = None,
) -> None:
    cols = 1
    rows = len(image_items)
    fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69))
    if rows == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    if subtitle:
        fig.text(0.05, 0.945, textwrap.fill(subtitle, 110), fontsize=9.5, va="top")
    top = 0.89 if subtitle else 0.93
    bottom = 0.05
    fig.subplots_adjust(top=top, bottom=bottom, hspace=0.18)
    for ax, (image_path, caption) in zip(axes, image_items):
        ax.axis("off")
        try:
            img = mpimg.imread(image_path)
            ax.imshow(img)
            ax.set_title(caption, fontsize=10, pad=8)
            ax.text(
                0.5,
                -0.04,
                str(image_path),
                fontsize=7,
                color="#666666",
                ha="center",
                va="top",
                transform=ax.transAxes,
            )
        except Exception as exc:
            ax.text(0.5, 0.5, f"Could not load image:\n{image_path}\n{exc}", ha="center", va="center")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_visual_evolution_pages(pdf: PdfPages, stages: list[StageArtifacts]) -> None:
    for image_name, title in [
        ("umap_sample_type.png", "Visual evolution — sample type"),
        ("umap_dataset.png", "Visual evolution — dataset identity"),
        ("umap_class.png", "Visual evolution — class structure"),
    ]:
        available = [(stage.title, stage.images[image_name]) for stage in stages if image_name in stage.images]
        if not available:
            continue
        fig, axes = plt.subplots(len(available), 1, figsize=(8.27, 11.69))
        if len(available) == 1:
            axes = [axes]
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
        fig.text(
            0.05,
            0.945,
            textwrap.fill(
                "Figures are shown in chronological order to make the change in embedding geometry legible across "
                "method changes and scale. For the full-corpus run, the projection is the sampled evaluation_v2 view.",
                110,
            ),
            fontsize=9.5,
            va="top",
        )
        fig.subplots_adjust(top=0.90, bottom=0.04, hspace=0.16)
        for ax, (stage_title, image_path) in zip(axes, available):
            ax.axis("off")
            try:
                ax.imshow(mpimg.imread(image_path))
            except Exception as exc:
                ax.text(0.5, 0.5, f"Could not load {image_path}\n{exc}", ha="center", va="center")
            ax.set_title(stage_title, fontsize=10, pad=8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_interpretation(stages: list[StageArtifacts]) -> list[str]:
    paragraphs = [
        (
            "Across all discovered runs, the same high-level structure persists: sample type is the first "
            "and easiest axis of organization. Even the smaller GPU run already separated serum, EV, and "
            "grounding with high nearest-neighbor purity."
        ),
        (
            "However, scaling did not simply make the manifold more biological. The full-corpus run showed "
            "that increasing data volume strengthened two things at once: local biological coherence and "
            "dataset identity. The v5 full-corpus evaluation is therefore an important diagnostic result: "
            "more data alone is not enough to remove dataset shortcuts."
        ),
        (
            "The intermediate pass-3 run is useful precisely because it did not magically solve the problem. "
            "It shows that biology-aware positives and same-scope repulsion can change the geometry, but "
            "not yet enough to dominate the extremely strong archive-specific signal present in a multi-dataset Raman/SERS corpus."
        ),
    ]
    if len(stages) >= 3:
        v2 = next((s for s in stages if s.key == "v2_gpu_run1_final" and s.run_dir), None)
        v3 = next((s for s in stages if s.key == "v3_pass3_gpu_run2" and s.run_dir), None)
        v5 = next((s for s in stages if s.key == "v5_full_true_gpu_run1" and s.run_dir), None)
        if v2 and v3 and v5:
            paragraphs.append(
                "The hierarchy visible in the current evidence is: sample type → dataset identity → biology. "
                f"In the small GPU baseline, sample-type probe macro-F1 was {probe_lookup(v2, 'sample_type')} and "
                f"family macro-F1 was {probe_lookup(v2, 'family_label')}. In the full-corpus run, those rose to "
                f"{probe_lookup(v5, 'sample_type')} and {probe_lookup(v5, 'family_label')}, while class-level neighbor "
                f"consistency climbed from {stage_metric(v2, 'nn_consistency_class')} to {stage_metric(v5, 'nn_consistency_class')}. "
                "That is evidence that biology really does emerge with scale. But dataset purity remained nearly perfect in v5, "
                "which means the next bottleneck is disentanglement, not raw data volume."
            )
    paragraphs.append(
        "The practical implication is that within-sample-type invariance now matters more than forcing all spectra into "
        "one universal mixed manifold. The model should probably keep the physically meaningful separation between serum, EV, "
        "and grounding, while learning to suppress acquisition- and dataset-specific shortcuts inside each domain."
    )
    return paragraphs


def build_recommendations() -> list[str]:
    return [
        (
            "The current technical conclusion is not that the embedding programme failed, but that it has reached a more "
            "difficult stage. Sample-type structure is reliable; the remaining work is to preserve that structure while "
            "reducing dataset leakage and increasing within-sample-type invariance."
        ),
        (
            "The next method step should therefore focus on objective design and evaluation inside coherent sample-type slices: "
            "dataset-invariant positives and negatives, stronger within-domain retrieval tests, and explicit measurement of "
            "whether spectra from different datasets but similar biochemical regimes are being pulled together."
        ),
        (
            "This also means the full-corpus result changes the interpretation of the problem. Biology is no longer absent. "
            "Rather, it is present but embedded beneath a stronger layer of dataset identity. That is a modeling problem, "
            "not just a data quantity problem."
        ),
    ]


def build_next_step() -> list[str]:
    return [
        (
            "The recommended next direction is not to chase another indiscriminate scale jump. The next experiment should "
            "hold the physically correct sample-type manifold in place while explicitly training for invariance inside each "
            "sample type."
        ),
        (
            "Concretely, that means within-sample-type objective design: cross-dataset positives inside coherent domains, "
            "harder same-domain negatives when label trust is high, and evaluation that asks whether spectra from different "
            "datasets but similar biological states actually become neighbors."
        ),
        (
            "The full-corpus v5 result justifies this focus. Data scale is no longer the limiting factor. The next bottleneck "
            "is disentanglement of dataset identity from biochemically meaningful structure, especially inside EV and serum."
        ),
    ]


def discovered_and_missing_notes(stages: list[StageArtifacts], data_root: Path) -> tuple[list[str], list[str]]:
    found = []
    missing = []
    for stage in stages:
        if stage.run_dir:
            found.append(f"{stage.title}: {stage.run_dir}")
            if stage.eval_dir:
                found.append(f"{stage.title} evaluation companion: {stage.eval_dir}")
        else:
            missing.append(stage.title)

    extras = []
    for extra_name in ("embedding_v4_medium", "embedding_v5_full", "embedding_v5_full_true"):
        p = data_root / extra_name
        if p.exists():
            extras.append(
                f"{extra_name} was found but only contains dataset build artifacts (dataset_summary/embedding_dataset), "
                "so it was not treated as a reportable evaluated run."
            )
    found.extend(extras)
    return found, missing


def generate_report(args: argparse.Namespace) -> tuple[list[StageArtifacts], list[str], list[str]]:
    stages = [discover_stage(args.data_root, spec, args.verbose) for spec in EXPECTED_STAGE_SPECS]
    stages.sort(key=lambda stage: stage.stage_order)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)

    found_notes, missing_notes = discovered_and_missing_notes(stages, args.data_root)

    with PdfPages(args.output_pdf) as pdf:
        add_title_page(
            pdf,
            report_title=args.title,
            author=args.author,
            data_root=args.data_root,
            output_pdf=args.output_pdf,
            stages=stages,
            previous_report_pdf=args.previous_report_pdf,
        )

        add_text_page(
            pdf,
            "Executive Summary",
            [
                (
                    "GAIRAM is attempting to learn a spectra-only embedding geometry that preserves physically real "
                    "sample-type structure while eventually enabling useful biochemical similarity inside each domain."
                ),
                (
                    "The earlier GPU baseline already showed that this is possible at the coarse level: serum, EV, and "
                    "grounding spectra organize non-randomly. The central weakness was that dataset identity remained too strong."
                ),
                (
                    "The subsequent work documented here moved through three technical steps: a smaller GPU baseline "
                    "evaluated with better metrics, a pass-3 biology-aware contrastive regime, and a full-corpus run "
                    "paired with a scalable evaluation redesign."
                ),
                (
                    "The main takeaway is now sharper than before. Scaling improved biological signal, but it also amplified "
                    "dataset memorization. The embedding problem is therefore no longer 'get more data'; it is 'learn within-sample-type "
                    "invariance without destroying physically meaningful sample-type separation.'"
                ),
            ],
        )

        continuity_paragraphs = PRIOR_REPORT_SUMMARY.copy()
        if args.previous_report_pdf.exists():
            continuity_paragraphs.append(
                f"The prior PDF baseline was found locally at: {args.previous_report_pdf}. "
                "This report uses it as the continuity anchor even though the PDF body was not parsed directly."
            )
        else:
            continuity_paragraphs.append(
                f"The prior PDF baseline was not found at the requested path ({args.previous_report_pdf}), "
                "so this continuity section is seeded from the known prior conclusion."
            )
        add_text_page(pdf, "Prior Report Continuity", continuity_paragraphs)

        table_page(
            pdf,
            "Methods Evolution",
            methods_table(stages),
            subtitle=(
                "Run-by-run method progression. The v2 baseline is essentially instance-only contrastive training, "
                "whereas the pass-3 family introduces tempered semantic attraction, same-scope repulsion, and "
                "region-aware augmentations. The full-corpus v5 run scales that tempered objective rather than inventing a new encoder."
            ),
            font_size=7.5,
        )

        table_page(
            pdf,
            "Comparative Embedding Metrics",
            comparative_metrics_table(stages),
            subtitle=(
                "Nearest-neighbor metrics indicate local neighborhood purity; silhouette metrics indicate coarse global "
                "geometry. For v5, neighborhood metrics are full-corpus and silhouette values are sampled-global from evaluation_v2."
            ),
            font_size=8.0,
        )

        table_page(
            pdf,
            "Comparative Frozen Probe Metrics",
            comparative_probe_table(stages),
            subtitle=(
                "Frozen linear probes estimate how much useful downstream signal is already present in the learned embeddings. "
                "Tasks are restricted to coherent label spaces rather than fabricated global class problems."
            ),
            font_size=8.0,
        )

        for stage in stages:
            add_stage_result_pages(pdf, stage)

        add_visual_evolution_pages(pdf, stages)

        add_text_page(pdf, "Interpretation", build_interpretation(stages))
        add_text_page(pdf, "Main Technical Conclusion", build_recommendations())
        add_text_page(pdf, "Recommended Next Step", build_next_step())

        found_lines = found_notes or ["No expected run folders were discovered."]
        missing_lines = (
            ["Missing expected stage folders: " + ", ".join(missing_notes)] if missing_notes else ["All expected major stage folders were found."]
        )
        add_text_page(
            pdf,
            "Artifact Provenance and Coverage",
            [
                "Run folders used in this report:",
                *found_lines,
                *missing_lines,
                "The report intentionally skipped dataset-build staging directories that did not contain evaluation artifacts.",
            ],
        )

    return stages, found_notes, missing_notes


def main() -> None:
    args = parse_args()
    stages, found_notes, missing_notes = generate_report(args)
    print(f"Wrote PDF report to {args.output_pdf}")
    print("Stages used:")
    for stage in stages:
        status = "FOUND" if stage.run_dir else "MISSING"
        print(f"- {status}: {stage.title}")
    if missing_notes:
        print("Missing expected stages: " + ", ".join(missing_notes))


if __name__ == "__main__":
    main()
