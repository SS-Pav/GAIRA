import shutil
import textwrap
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


BASE_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/small2023_ev_invariant_embedding")
REPORT_DIR = BASE_DIR / "report_v1"
FIGURES_DIR = REPORT_DIR / "figures"
TABLES_DIR = REPORT_DIR / "tables"
REPORT_SUBDIR = REPORT_DIR / "report"
SOURCE_PLOTS_DIR = REPORT_DIR / "source_plots"
MPLCONFIGDIR = REPORT_DIR / ".mplconfig"

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "savefig.dpi": 400,
    }
)

COLORS = {
    "baseline": "#8c8c8c",
    "embedding": "#355c7d",
    "class_before": "#b9c0c9",
    "class_after": "#355c7d",
    "probe_before": "#d18f77",
    "probe_after": "#7a8e5a",
}


def ensure_dirs() -> None:
    for path in (REPORT_DIR, FIGURES_DIR, TABLES_DIR, REPORT_SUBDIR, SOURCE_PLOTS_DIR, MPLCONFIGDIR):
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> dict[str, pd.DataFrame | str]:
    return {
        "counts": pd.read_csv(BASE_DIR / "benchmark_sample_counts.csv"),
        "baseline": pd.read_csv(BASE_DIR / "baseline_cross_probe_metrics.csv"),
        "embedding": pd.read_csv(BASE_DIR / "embedding_cross_probe_metrics.csv"),
        "distances": pd.read_csv(BASE_DIR / "class_probe_distance_summary.csv"),
        "geometry": pd.read_csv(BASE_DIR / "geometry_metrics.csv"),
        "summary": (BASE_DIR / "embedding_summary.txt").read_text(encoding="utf-8"),
    }


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def figure1_method_flowchart(counts_df: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    panel_specs = [
        (
            "A",
            "Dataset structure",
            [
                "small2023_ev",
                "Processed benchmark subset",
                "Probe1: 12,000 spectra",
                "Probe2: 12,000 spectra",
                "Classes: c00, c01, c10, c25, c50, c100",
            ],
        ),
        (
            "B",
            "Processing pipeline",
            [
                "Raw Probe1: 670–1800 cm^-1",
                "Raw Probe2: 401–1800 cm^-1",
                "Crop to 670–1800",
                "Interpolate to 1 cm^-1 grid",
                "Min-max normalize to 1131 points",
            ],
        ),
        (
            "C",
            "Benchmark design",
            [
                "Balanced subset per probe/class",
                f"{int(counts_df['n_used'].iloc[0])} spectra per group",
                "Total benchmark size: 24,000",
                "Exclude fig3_norm_archive",
                "Use only normedprobe1 and normedprobe2",
            ],
        ),
        (
            "D",
            "Evaluation strategy",
            [
                "Raw logistic baseline",
                "Train Probe1 → test Probe2",
                "Train Probe2 → test Probe1",
                "MLP embedding + linear probe",
                "Geometry and centroid checks",
            ],
        ),
    ]

    for ax, (panel_label, title, lines) in zip(axes, panel_specs):
        add_panel_label(ax, panel_label)
        ax.axis("off")
        box = patches.FancyBboxPatch(
            (0.05, 0.08),
            0.9,
            0.82,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#4f5d75",
            facecolor="#f7f8fa",
        )
        ax.add_patch(box)
        ax.text(0.09, 0.84, title, fontsize=12, fontweight="bold", transform=ax.transAxes)
        y = 0.72
        for line in lines:
            ax.text(0.1, y, line, fontsize=10, transform=ax.transAxes)
            y -= 0.12
            if y > 0.12:
                ax.annotate(
                    "",
                    xy=(0.12, y + 0.03),
                    xytext=(0.12, y + 0.09),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#7f8c8d"),
                    transform=ax.transAxes,
                )

    fig.tight_layout()
    return save_figure(fig, "figure1_method_flowchart")


def figure2_cross_probe_accuracy(baseline_df: pd.DataFrame, embedding_df: pd.DataFrame) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    directions = baseline_df["direction"].tolist()
    x = np.arange(len(directions))
    width = 0.34

    ax.bar(x - width / 2, baseline_df["accuracy"], width=width, color=COLORS["baseline"], label="Baseline")
    ax.bar(x + width / 2, embedding_df["accuracy"], width=width, color=COLORS["embedding"], label="Embedding")
    ax.set_xticks(x)
    ax.set_xticklabels(directions, rotation=0)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-probe classification transfer")
    ax.legend(frameon=False)
    add_panel_label(ax, "A")
    fig.tight_layout()
    return save_figure(fig, "figure2_cross_probe_accuracy")


def figure3_separability(geometry_df: pd.DataFrame) -> tuple[Path, Path]:
    row = geometry_df.iloc[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    categories = ["Class separability", "Probe separability"]
    before = [row["raw_class_silhouette"], row["raw_probe_silhouette"]]
    after = [row["embedding_class_silhouette"], row["embedding_probe_silhouette"]]
    x = np.arange(len(categories))
    width = 0.34

    ax.bar(x - width / 2, before, width=width, color="#b3bbc7", label="Before embedding")
    ax.bar(x + width / 2, after, width=width, color=COLORS["embedding"], label="After embedding")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Silhouette score")
    ax.set_title("Class and probe separability")
    ax.legend(frameon=False)
    add_panel_label(ax, "A")
    fig.tight_layout()
    return save_figure(fig, "figure3_separability")


def figure4_centroid_shift(distance_df: pd.DataFrame) -> tuple[Path, Path]:
    overall = distance_df[distance_df["class_label"] == "overall_mean"].iloc[0]
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    labels = ["Before", "After"]
    values = [
        overall["raw_cross_probe_centroid_distance"],
        overall["embedding_cross_probe_centroid_distance"],
    ]
    ax.bar(labels, values, color=["#b3bbc7", COLORS["embedding"]], width=0.55)
    ax.set_ylabel("Mean same-class cross-probe centroid distance")
    ax.set_title("Cross-probe centroid alignment")
    add_panel_label(ax, "A")
    fig.tight_layout()
    return save_figure(fig, "figure4_centroid_shift")


def figure5_mixture_ordering(geometry_df: pd.DataFrame) -> tuple[Path, Path]:
    row = geometry_df.iloc[0]
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    labels = ["Before", "After"]
    values = [
        row["raw_mixture_order_correlation"],
        row["embedding_mixture_order_correlation"],
    ]
    ax.bar(labels, values, color=["#b3bbc7", COLORS["embedding"]], width=0.55)
    ax.set_ylabel("Correlation with mixture ordering")
    ax.set_title("Mixture ordering sanity check")
    add_panel_label(ax, "A")
    fig.tight_layout()
    return save_figure(fig, "figure5_mixture_ordering")


def copy_source_plots() -> list[Path]:
    copied = []
    for name in [
        "raw_tsne_by_class.png",
        "raw_tsne_by_probe.png",
        "embedding_tsne_by_class.png",
        "embedding_tsne_by_probe.png",
    ]:
        src = BASE_DIR / name
        dst = SOURCE_PLOTS_DIR / name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def figure6_tsne_geometry() -> tuple[Path, Path]:
    image_specs = [
        ("A", "Raw t-SNE by class", SOURCE_PLOTS_DIR / "raw_tsne_by_class.png"),
        ("B", "Raw t-SNE by probe", SOURCE_PLOTS_DIR / "raw_tsne_by_probe.png"),
        ("C", "Embedding t-SNE by class", SOURCE_PLOTS_DIR / "embedding_tsne_by_class.png"),
        ("D", "Embedding t-SNE by probe", SOURCE_PLOTS_DIR / "embedding_tsne_by_probe.png"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (panel, title, path) in zip(axes.ravel(), image_specs):
        with Image.open(path) as img:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
        add_panel_label(ax, panel)
    fig.tight_layout()
    return save_figure(fig, "figure6_tsne_geometry")


def build_metrics_table(
    baseline_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    distance_df: pd.DataFrame,
) -> Path:
    overall = distance_df[distance_df["class_label"] == "overall_mean"].iloc[0]
    row = geometry_df.iloc[0]
    metrics = [
        ("Baseline Probe1→Probe2 accuracy", baseline_df.loc[0, "accuracy"]),
        ("Baseline Probe2→Probe1 accuracy", baseline_df.loc[1, "accuracy"]),
        ("Embedding Probe1→Probe2 accuracy", embedding_df.loc[0, "accuracy"]),
        ("Embedding Probe2→Probe1 accuracy", embedding_df.loc[1, "accuracy"]),
        ("Class separability before", row["raw_class_silhouette"]),
        ("Class separability after", row["embedding_class_silhouette"]),
        ("Probe separability before", row["raw_probe_silhouette"]),
        ("Probe separability after", row["embedding_probe_silhouette"]),
        ("Centroid distance before", overall["raw_cross_probe_centroid_distance"]),
        ("Centroid distance after", overall["embedding_cross_probe_centroid_distance"]),
        ("Mixture ordering correlation before", row["raw_mixture_order_correlation"]),
        ("Mixture ordering correlation after", row["embedding_mixture_order_correlation"]),
    ]
    table_df = pd.DataFrame(metrics, columns=["metric", "value"])
    out_path = TABLES_DIR / "benchmark_metrics_table.csv"
    table_df.to_csv(out_path, index=False)
    return out_path


def write_markdown(
    counts_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
) -> Path:
    row = geometry_df.iloc[0]
    baseline_mean = baseline_df["accuracy"].mean()
    embedding_mean = embedding_df["accuracy"].mean()
    md = f"""# Substrate-invariant spectral embedding benchmark for small EV SERS dataset

## 1. Study motivation

This report summarizes the first GAIRA substrate-invariant embedding benchmark on `small2023_ev`, using only processed mixture-class spectra from `normedprobe1` and `normedprobe2`. The aim is to reduce probe-domain separation while preserving class-level structure across the common 670–1800 cm^-1 processed grid.

## 2. Dataset structure

The benchmark uses six mixture classes (`c00`, `c01`, `c10`, `c25`, `c50`, `c100`) and two probe families. A balanced subset of 2,000 spectra per probe/class group was used.

![Figure 1](../figures/figure1_method_flowchart.png)

## 3. Benchmark experiment design

- Raw baseline: logistic regression on processed spectra
- Learned embedding: compact MLP encoder with 64-dimensional embedding
- Transfer directions: Probe1→Probe2 and Probe2→Probe1
- Geometry checks: class separability, probe separability, cross-probe centroid shift, mixture ordering

## 4. Embedding method

The v1 embedding benchmark uses a deterministic sklearn MLP encoder because PyTorch was not available in the environment. This should be treated as a compact representation-learning baseline rather than a final invariant model.

## 5. Results

Mean cross-probe accuracy improved from {baseline_mean:.4f} to {embedding_mean:.4f}. Class separability rose from {row['raw_class_silhouette']:.4f} to {row['embedding_class_silhouette']:.4f}, while probe separability fell from {row['raw_probe_silhouette']:.4f} to {row['embedding_probe_silhouette']:.4f}.

![Figure 2](../figures/figure2_cross_probe_accuracy.png)

![Figure 3](../figures/figure3_separability.png)

![Figure 4](../figures/figure4_centroid_shift.png)

![Figure 5](../figures/figure5_mixture_ordering.png)

![Figure 6](../figures/figure6_tsne_geometry.png)

## 6. Interpretation

The embedding improves cross-probe transfer substantially and reduces probe-domain clustering, while preserving a usable amount of mixture-class structure. The benchmark therefore supports the feasibility of a more explicit invariant embedding program for `small2023_ev`.

## 7. Limitations

- The benchmark uses an sklearn MLP rather than a true supervised-contrastive deep encoder.
- Probe separation remains present after embedding.
- Mixture ordering correlation declines somewhat after embedding.
- This is a benchmark, not a production inference model.

## 8. Next steps for GAIRA

- Replace the MLP baseline with an explicit supervised-contrastive or domain-adversarial encoder.
- Add stronger probe-invariance objectives while monitoring mixture-order coherence.
- Extend the benchmark to held-out class summaries and downstream analog-matching analyses.
"""
    out_path = REPORT_SUBDIR / "small2023_ev_invariant_embedding_v1.md"
    out_path.write_text(md, encoding="utf-8")
    return out_path


def _draw_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, title, fontsize=18, fontweight="bold", va="top")
    wrapped = "\n".join(textwrap.wrap(body, width=95, break_long_words=False))
    fig.text(0.08, 0.9, wrapped, fontsize=11, va="top", linespacing=1.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_image_page(pdf: PdfPages, title: str, image_path: Path, caption: str = "") -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.96, title, fontsize=16, fontweight="bold", va="top")
    ax = fig.add_axes([0.08, 0.18, 0.84, 0.72])
    with Image.open(image_path) as img:
        ax.imshow(img)
    ax.axis("off")
    if caption:
        fig.text(0.08, 0.1, textwrap.fill(caption, width=110), fontsize=10, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf_report(
    baseline_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    distance_df: pd.DataFrame,
) -> Path:
    row = geometry_df.iloc[0]
    overall = distance_df[distance_df["class_label"] == "overall_mean"].iloc[0]
    out_path = REPORT_SUBDIR / "small2023_ev_invariant_embedding_v1.pdf"
    with PdfPages(out_path) as pdf:
        _draw_text_page(
            pdf,
            "Substrate-invariant spectral embedding benchmark for small EV SERS dataset",
            (
                "This report summarizes the completed v1 invariant embedding benchmark for small2023_ev. "
                "The analysis compares a raw processed-spectrum logistic baseline against a compact learned embedding "
                "using only normedprobe1 and normedprobe2 mixture-class spectra on the common 670–1800 cm^-1 grid."
            ),
        )
        _draw_image_page(
            pdf,
            "Figure 1. Method overview",
            FIGURES_DIR / "figure1_method_flowchart.png",
            "Dataset structure, processing pipeline, benchmark design, and evaluation strategy.",
        )
        _draw_image_page(
            pdf,
            "Figure 2. Cross-probe classification",
            FIGURES_DIR / "figure2_cross_probe_accuracy.png",
            (
                f"Embedding transfer improves from {baseline_df['accuracy'].mean():.3f} mean accuracy to "
                f"{embedding_df['accuracy'].mean():.3f}."
            ),
        )
        _draw_image_page(
            pdf,
            "Figure 3. Separability comparison",
            FIGURES_DIR / "figure3_separability.png",
            (
                f"Class silhouette improves from {row['raw_class_silhouette']:.3f} to "
                f"{row['embedding_class_silhouette']:.3f}, while probe silhouette drops from "
                f"{row['raw_probe_silhouette']:.3f} to {row['embedding_probe_silhouette']:.3f}."
            ),
        )
        _draw_image_page(
            pdf,
            "Figure 4. Cross-probe centroid shift",
            FIGURES_DIR / "figure4_centroid_shift.png",
            (
                f"Mean same-class cross-probe centroid distance contracts from "
                f"{overall['raw_cross_probe_centroid_distance']:.3f} to "
                f"{overall['embedding_cross_probe_centroid_distance']:.3f}."
            ),
        )
        _draw_image_page(
            pdf,
            "Figure 5. Mixture ordering",
            FIGURES_DIR / "figure5_mixture_ordering.png",
            (
                f"Mixture ordering correlation changes from {row['raw_mixture_order_correlation']:.3f} to "
                f"{row['embedding_mixture_order_correlation']:.3f}."
            ),
        )
        _draw_image_page(
            pdf,
            "Figure 6. Spectral geometry",
            FIGURES_DIR / "figure6_tsne_geometry.png",
            "Raw and embedded geometry colored by class and by probe.",
        )
        _draw_text_page(
            pdf,
            "Interpretation and limitations",
            (
                "The learned embedding is promising as a GAIRA benchmark because it improves cross-probe transfer, "
                "increases class separability, reduces probe separability, and moves same-class probe centroids closer. "
                "Its main limitations are that probe structure remains visible, mixture ordering weakens somewhat, and "
                "the method is still a compact sklearn baseline rather than a dedicated invariant deep representation."
            ),
        )
    return out_path


def main() -> None:
    ensure_dirs()
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
    inputs = load_inputs()
    counts_df = inputs["counts"]
    baseline_df = inputs["baseline"]
    embedding_df = inputs["embedding"]
    distance_df = inputs["distances"]
    geometry_df = inputs["geometry"]

    copy_source_plots()
    figure1_method_flowchart(counts_df)
    figure2_cross_probe_accuracy(baseline_df, embedding_df)
    figure3_separability(geometry_df)
    figure4_centroid_shift(distance_df)
    figure5_mixture_ordering(geometry_df)
    figure6_tsne_geometry()
    build_metrics_table(baseline_df, embedding_df, geometry_df, distance_df)
    write_markdown(counts_df, baseline_df, embedding_df, geometry_df)
    pdf_path = build_pdf_report(baseline_df, embedding_df, geometry_df, distance_df)

    print(f"Report outputs written to: {REPORT_DIR}")
    print(f"PDF report: {pdf_path}")
    print(f"Figures folder: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
