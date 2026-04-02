from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from gaira.autoresearch_storage import AutoresearchSprintPaths
from gaira.demo.raw_bsv_pilot_utils import ALL_AXES


def pairwise_delta_bsv(class_mean_bsv: pd.DataFrame, axis_names: list[str] | None = None) -> pd.DataFrame:
    axes = [axis for axis in (axis_names or ALL_AXES) if axis in class_mean_bsv.columns]
    rows = []
    work = class_mean_bsv.set_index("class_label")
    labels = list(work.index.astype(str))
    for source_label in labels:
        for target_label in labels:
            if source_label == target_label:
                continue
            source = work.loc[source_label]
            target = work.loc[target_label]
            record = {
                "comparison": f"{source_label}-vs-{target_label}",
                "group_label": source_label,
                "reference_group": target_label,
            }
            for axis in axes + ["unmapped_support"]:
                if axis in work.columns:
                    record[axis] = float(source[axis] - target[axis])
            rows.append(record)
    return pd.DataFrame(rows)


def compute_stability_tables(
    per_spectrum_bsv: pd.DataFrame,
    class_mean_bsv: pd.DataFrame,
    axis_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    axes = [axis for axis in (axis_names or ALL_AXES) if axis in per_spectrum_bsv.columns]
    var_df = (
        per_spectrum_bsv.groupby("class_label", sort=True)[axes]
        .var(ddof=1)
        .fillna(0.0)
        .reset_index()
    )
    distance_rows = []
    work = class_mean_bsv.set_index("class_label")
    labels = list(work.index.astype(str))
    for left in labels:
        for right in labels:
            diff = work.loc[left, axes].to_numpy(dtype=float) - work.loc[right, axes].to_numpy(dtype=float)
            distance_rows.append(
                {
                    "class_label_a": left,
                    "class_label_b": right,
                    "euclidean_distance": float(np.linalg.norm(diff)),
                }
            )
    distance_df = pd.DataFrame(distance_rows)
    return var_df, distance_df


def _entropy_from_fractions(values: list[float]) -> float:
    safe = np.asarray([float(v) for v in values if float(v) > 0], dtype=float)
    if safe.size == 0:
        return 0.0
    return float(-(safe * np.log2(safe)).sum())


def build_class_topk_neighborhood_composition(retrieval_df: pd.DataFrame) -> pd.DataFrame:
    if retrieval_df.empty:
        return pd.DataFrame(
            columns=["class_label", "compound_label", "support_weight_sum", "support_fraction"]
        )
    grouped = (
        retrieval_df.groupby(
            ["query_class_label", "reference_compound_label"],
            sort=True,
            dropna=False,
        )["support_weight"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "query_class_label": "class_label",
                "reference_compound_label": "compound_label",
                "support_weight": "support_weight_sum",
            }
        )
    )
    grouped["support_fraction"] = grouped.groupby("class_label")["support_weight_sum"].transform(
        lambda s: s / max(float(s.sum()), 1e-12)
    )
    return grouped.sort_values(["class_label", "support_weight_sum"], ascending=[True, False]).reset_index(drop=True)


def build_class_neighborhood_entropy(class_neighborhood_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_label, group in class_neighborhood_df.groupby("class_label", sort=True):
        rows.append(
            {
                "class_label": str(class_label),
                "neighborhood_entropy": _entropy_from_fractions(group["support_fraction"].tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("class_label").reset_index(drop=True)


def build_class_top1_dominance(class_neighborhood_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_label, group in class_neighborhood_df.groupby("class_label", sort=True):
        rows.append(
            {
                "class_label": str(class_label),
                "top1_fraction": float(group["support_fraction"].max()) if not group.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("class_label").reset_index(drop=True)


def build_class_axis_entropy(class_mean_bsv: pd.DataFrame) -> pd.DataFrame:
    axes = [axis for axis in ALL_AXES if axis in class_mean_bsv.columns]
    rows = []
    for _, row in class_mean_bsv.iterrows():
        values = [float(row[axis]) for axis in axes]
        total = sum(values)
        fractions = [v / total for v in values] if total > 0 else values
        rows.append(
            {
                "class_label": str(row["class_label"]),
                "axis_entropy": _entropy_from_fractions(fractions),
            }
        )
    return pd.DataFrame(rows).sort_values("class_label").reset_index(drop=True)


def _class_colors(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % 10) for i, label in enumerate(labels)}


def plot_pca_by_class(
    pca_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "PCA of Canonical Spectra",
    legend_title: str = "Class",
) -> None:
    labels = sorted(pca_df["class_label"].astype(str).unique().tolist())
    colors = _class_colors(labels)
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    for label in labels:
        sub = pca_df[pca_df["class_label"].astype(str) == label]
        ax.scatter(
            sub["pc1"],
            sub["pc2"],
            s=22,
            alpha=0.82,
            label=label,
            color=colors[label],
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title=legend_title)
    fig.tight_layout(rect=[0.0, 0.0, 0.83, 1.0])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_heatmap(
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    output_path: Path,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
) -> None:
    fig_width = max(8.0, 0.8 * len(col_labels) + 3.6)
    fig_height = max(4.8, 0.62 * len(row_labels) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if center is not None:
        max_abs = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), abs(center))
        vmin = -max_abs
        vmax = max_abs
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = float(matrix[i, j])
            text_color = "white" if abs(val) > 0.55 * max(abs(image.norm.vmin), abs(image.norm.vmax), 1e-8) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    cbar = fig.colorbar(image, ax=ax, shrink=0.9)
    cbar.ax.set_ylabel("Support", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_bsv_heatmap(class_mean_bsv: pd.DataFrame, output_path: Path, title: str) -> None:
    axes = [axis for axis in ALL_AXES if axis in class_mean_bsv.columns]
    heat = class_mean_bsv.set_index("class_label")[axes]
    _draw_heatmap(
        heat.to_numpy(dtype=float),
        row_labels=heat.index.astype(str).tolist(),
        col_labels=axes,
        output_path=output_path,
        title=title,
        cmap="viridis",
        vmin=0.0,
        vmax=max(1.0, float(np.nanmax(heat.to_numpy(dtype=float)))),
    )


def plot_pairwise_delta_heatmap(pairwise_delta_df: pd.DataFrame, axis: str, output_path: Path) -> None:
    heat = pairwise_delta_df.pivot(index="group_label", columns="reference_group", values=axis)
    heat = heat.reindex(sorted(heat.index), axis=0).reindex(sorted(heat.columns), axis=1)
    _draw_heatmap(
        heat.to_numpy(dtype=float),
        row_labels=heat.index.astype(str).tolist(),
        col_labels=heat.columns.astype(str).tolist(),
        output_path=output_path,
        title=f"Pairwise Delta BSV Heatmap: {axis}",
        cmap="coolwarm",
        center=0.0,
    )


def plot_top_hit_distribution_by_class(retrieval_summary_df: pd.DataFrame, output_path: Path, top_n: int = 5) -> None:
    labels = sorted(retrieval_summary_df["query_class_label"].astype(str).unique().tolist())
    if not labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No retrieval hits", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return
    ncols = 2
    nrows = math.ceil(len(labels) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13.0, 3.7 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat[len(labels) :]:
        ax.axis("off")
    for ax, label in zip(axes_flat, labels, strict=False):
        sub = retrieval_summary_df[retrieval_summary_df["query_class_label"].astype(str) == label].copy()
        sub = sub.sort_values("total_support_weight", ascending=False).head(top_n)
        y = np.arange(len(sub))
        ax.barh(y, sub["total_support_weight"], color="#4c78a8")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["reference_compound_label"].astype(str).tolist(), fontsize=8)
        ax.invert_yaxis()
        ax.set_title(label)
        ax.set_xlabel("Total support weight")
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    fig.suptitle("Top Retrieval Hit Distribution by Class", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_radar_for_class(class_mean_bsv: pd.DataFrame, class_label: str, output_path: Path) -> None:
    axes = [axis for axis in ALL_AXES if axis in class_mean_bsv.columns]
    row = class_mean_bsv[class_mean_bsv["class_label"].astype(str) == str(class_label)].iloc[0]
    values = [float(row[axis]) for axis in axes]
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]
    fig = plt.figure(figsize=(8.6, 8.0))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_closed, values_closed, color="#1f77b4", linewidth=2.0)
    ax.fill(angles_closed, values_closed, color="#1f77b4", alpha=0.2)
    ax.set_xticks(angles)
    ax.set_xticklabels(axes, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(f"{class_label} BSV Fingerprint", pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_neighborhood_composition_for_class(
    class_neighborhood_df: pd.DataFrame,
    class_label: str,
    output_path: Path,
    top_n: int = 10,
) -> None:
    sub = class_neighborhood_df[class_neighborhood_df["class_label"].astype(str) == str(class_label)].copy()
    sub = sub.sort_values("support_fraction", ascending=False).head(top_n)
    fig_height = max(4.8, 0.52 * len(sub) + 2.2)
    fig, ax = plt.subplots(figsize=(9.8, fig_height))
    y = np.arange(len(sub))
    ax.barh(y, sub["support_fraction"], color="#2a9d8f")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["compound_label"].astype(str).tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Support fraction")
    ax.set_title(f"{class_label} Neighborhood Composition")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bar(metric_df: pd.DataFrame, value_col: str, title: str, output_path: Path) -> None:
    work = metric_df.sort_values("class_label").copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = np.arange(len(work))
    ax.bar(x, work[value_col], color="#6c757d")
    ax.set_xticks(x)
    ax.set_xticklabels(work["class_label"].astype(str).tolist(), rotation=0)
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def infer_mixture_order(class_labels: list[str]) -> list[str]:
    def _score(label: str) -> tuple[int, str]:
        digits = "".join(ch for ch in str(label) if ch.isdigit())
        return (int(digits) if digits else 10**9, str(label))

    return sorted([str(label) for label in class_labels], key=_score)


def build_mixture_progression_summary(
    class_mean_bsv: pd.DataFrame,
    class_neighborhood_df: pd.DataFrame,
) -> pd.DataFrame:
    ordered = infer_mixture_order(class_mean_bsv["class_label"].astype(str).tolist())
    work = class_mean_bsv.set_index("class_label").loc[ordered]
    endpoint_low = ordered[0]
    endpoint_high = ordered[-1]
    low_vec = work.loc[endpoint_low, [axis for axis in ALL_AXES if axis in work.columns]].to_numpy(dtype=float)
    high_vec = work.loc[endpoint_high, [axis for axis in ALL_AXES if axis in work.columns]].to_numpy(dtype=float)
    endpoint_compounds = {}
    for endpoint in [endpoint_low, endpoint_high]:
        sub = class_neighborhood_df[class_neighborhood_df["class_label"].astype(str) == endpoint].copy()
        if sub.empty:
            endpoint_compounds[endpoint] = ""
        else:
            endpoint_compounds[endpoint] = str(
                sub.sort_values("support_fraction", ascending=False).iloc[0]["compound_label"]
            )
    rows = []
    for label in ordered:
        vec = work.loc[label, [axis for axis in ALL_AXES if axis in work.columns]].to_numpy(dtype=float)
        dist_low = float(np.linalg.norm(vec - low_vec))
        dist_high = float(np.linalg.norm(vec - high_vec))
        denom = max(dist_low + dist_high, 1e-12)
        toward_high = dist_low / denom
        hood = class_neighborhood_df[class_neighborhood_df["class_label"].astype(str) == label].copy()
        hood_total = max(float(hood["support_weight_sum"].sum()), 1e-12)
        low_comp = endpoint_compounds[endpoint_low]
        high_comp = endpoint_compounds[endpoint_high]
        low_frac = (
            float(hood[hood["compound_label"].astype(str) == low_comp]["support_weight_sum"].sum()) / hood_total
            if low_comp
            else 0.0
        )
        high_frac = (
            float(hood[hood["compound_label"].astype(str) == high_comp]["support_weight_sum"].sum()) / hood_total
            if high_comp
            else 0.0
        )
        digits = "".join(ch for ch in str(label) if ch.isdigit())
        rows.append(
            {
                "class_label": label,
                "mixture_code_numeric": int(digits) if digits else np.nan,
                "distance_to_low_endpoint": dist_low,
                "distance_to_high_endpoint": dist_high,
                "toward_high_endpoint_score": toward_high,
                "low_endpoint_class": endpoint_low,
                "high_endpoint_class": endpoint_high,
                "low_endpoint_top_compound": low_comp,
                "high_endpoint_top_compound": high_comp,
                "low_endpoint_compound_fraction": low_frac,
                "high_endpoint_compound_fraction": high_frac,
            }
        )
    return pd.DataFrame(rows)


def plot_mixture_progression(
    progression_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    work = progression_df.sort_values("mixture_code_numeric").copy()
    x = np.arange(len(work))
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.plot(x, work["toward_high_endpoint_score"], marker="o", linewidth=2.0, label="Toward high-endpoint score")
    ax.plot(
        x,
        work["high_endpoint_compound_fraction"],
        marker="s",
        linewidth=1.8,
        label=f"High-endpoint compound fraction ({work['high_endpoint_top_compound'].iloc[0]})",
    )
    ax.plot(
        x,
        work["low_endpoint_compound_fraction"],
        marker="^",
        linewidth=1.8,
        label=f"Low-endpoint compound fraction ({work['low_endpoint_top_compound'].iloc[0]})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(work["class_label"].astype(str).tolist())
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Relative progression / fraction")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_pilot_markdown_report(
    sprint_paths: AutoresearchSprintPaths,
    *,
    subset_alias: str,
    config_summary: dict[str, object],
    class_mean_bsv: pd.DataFrame,
    retrieval_summary_df: pd.DataFrame,
    intra_class_variance_df: pd.DataFrame,
    inter_class_distance_df: pd.DataFrame,
) -> Path:
    mean_variance = float(intra_class_variance_df.drop(columns=["class_label"]).mean(axis=1).mean())
    mean_inter_distance = float(inter_class_distance_df["euclidean_distance"].mean())
    top_lines = []
    for _, row in class_mean_bsv.iterrows():
        strongest = sorted(
            [(axis, float(row[axis])) for axis in ALL_AXES if axis in row.index],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        top_lines.append(
            f"- `{row['class_label']}`: top axes " + ", ".join([f"`{axis}={value:.3f}`" for axis, value in strongest])
        )
    retrieval_lines = []
    for class_label, group in retrieval_summary_df.groupby("query_class_label", sort=True):
        top = group.sort_values("total_support_weight", ascending=False).head(3)
        retrieval_lines.append(
            f"- `{class_label}`: "
            + "; ".join(
                [
                    f"`{r.reference_compound_label}` ({float(r.total_support_weight):.3f})"
                    for r in top.itertuples(index=False)
                ]
            )
        )
    lines = [
        "# GAIRAv3 Pilot 1: small2023_cellline",
        "",
        "## Locked Configuration",
        f"- subset alias: `{subset_alias}`",
        *[f"- {key}: `{value}`" for key, value in config_summary.items()],
        "",
        "## Summary",
        f"- mean intra-class BSV variance: `{mean_variance:.4f}`",
        f"- mean inter-class BSV distance: `{mean_inter_distance:.4f}`",
        "",
        "## Class Mean BSV Profiles",
        *top_lines,
        "",
        "## Retrieval Sanity Assessment",
        *retrieval_lines,
        "",
        "## Interpretation",
        "- This pilot uses the locked deterministic baseline from autoresearch pass 3 without parameter changes.",
        "- Per-spectrum BSV is included for stability inspection; class mean BSV is the primary absolute-BSV summary because the locked aggregation mode is `class_mean_spectrum_then_bsv`.",
        "- Pairwise delta BSV is descriptive here, not a separate tuned inference lane.",
        "",
        "## Stability And Interpretability",
        f"- Signal stability looks `{'acceptable' if mean_inter_distance > mean_variance else 'weak'}` under the simple variance-vs-distance check.",
        "- Interpretability should be judged jointly from the class-level axis structure and the retrieval sanity profile, not from PCA alone.",
    ]
    output_path = sprint_paths.report_dir / "GAIRAv3_Pilot1_small2023_cellline_report.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_fingerprint_markdown_report(
    sprint_paths: AutoresearchSprintPaths,
    *,
    subset_alias: str,
    config_summary: dict[str, object],
    class_mean_bsv: pd.DataFrame,
    class_neighborhood_df: pd.DataFrame,
    class_neighborhood_entropy_df: pd.DataFrame,
    class_top1_dominance_df: pd.DataFrame,
    class_axis_entropy_df: pd.DataFrame,
    intra_class_variance_df: pd.DataFrame,
    inter_class_distance_df: pd.DataFrame,
) -> Path:
    axes = [axis for axis in ALL_AXES if axis in class_mean_bsv.columns]
    mean_variance_by_class = intra_class_variance_df.copy()
    mean_variance_by_class["mean_variance"] = mean_variance_by_class[axes].mean(axis=1)
    top_distances = (
        inter_class_distance_df[inter_class_distance_df["class_label_a"] != inter_class_distance_df["class_label_b"]]
        .sort_values("euclidean_distance", ascending=False)
        .head(5)
    )
    report_lines = [
        "# GAIRAv3 Pilot 1a: small2023_cellline Fingerprint Report",
        "",
        "## 1. Overview",
        "- Fingerprint here means the combination of class-level BSV profile, local top-k neighborhood composition, and stability metrics.",
        "- The chemistry layer is unchanged from Pilot 1. This is a readout/reporting enhancement only.",
        f"- subset alias: `{subset_alias}`",
        *[f"- {key}: `{value}`" for key, value in config_summary.items()],
        "",
        "## 2. Class-level Fingerprints",
    ]
    for _, row in class_mean_bsv.sort_values("class_label").iterrows():
        label = str(row["class_label"])
        strongest = sorted([(axis, float(row[axis])) for axis in axes], key=lambda x: x[1], reverse=True)[:4]
        hood = class_neighborhood_df[class_neighborhood_df["class_label"].astype(str) == label].copy()
        hood_top = hood.sort_values("support_fraction", ascending=False).head(5)
        hood_entropy = float(
            class_neighborhood_entropy_df[class_neighborhood_entropy_df["class_label"].astype(str) == label][
                "neighborhood_entropy"
            ].iloc[0]
        )
        top1 = float(
            class_top1_dominance_df[class_top1_dominance_df["class_label"].astype(str) == label]["top1_fraction"].iloc[0]
        )
        axis_entropy = float(
            class_axis_entropy_df[class_axis_entropy_df["class_label"].astype(str) == label]["axis_entropy"].iloc[0]
        )
        report_lines.extend(
            [
                f"### {label}",
                f"- BSV profile: "
                + ", ".join([f"`{axis}={value:.3f}`" for axis, value in strongest]),
                f"- Axis entropy: `{axis_entropy:.3f}`",
                f"- Neighborhood entropy: `{hood_entropy:.3f}`",
                f"- Top1 dominance: `{top1:.3f}`",
                f"- Radar figure: `figures/radar_{label}.png`",
                f"- Neighborhood figure: `figures/neighborhood_{label}.png`",
                "- Neighborhood composition: "
                + "; ".join(
                    [
                        f"`{r.compound_label}` ({float(r.support_fraction):.3f})"
                        for r in hood_top.itertuples(index=False)
                    ]
                ),
                "- Interpretation: consistent with a relative shift within a purine-adjacent neighborhood rather than a broad molecular assignment.",
                "",
            ]
        )
    report_lines.extend(
        [
            "## 3. Stability",
            "- Intra-class variance is summarized from per-spectrum BSV dispersion within each class.",
            "- Inter-class distance is summarized as Euclidean distance between class mean BSV vectors.",
            "- Lowest-variance classes:",
        ]
    )
    for row in mean_variance_by_class.sort_values("mean_variance").head(5).itertuples(index=False):
        report_lines.append(f"- `{row.class_label}` mean variance `{float(row.mean_variance):.4f}`")
    report_lines.append("- Largest class separations:")
    for row in top_distances.itertuples(index=False):
        report_lines.append(
            f"- `{row.class_label_a}` vs `{row.class_label_b}` distance `{float(row.euclidean_distance):.4f}`"
        )
    report_lines.extend(
        [
            "",
            "## 4. Neighborhood Behavior",
            "- The retrieval layer is inspected here as a class-level neighborhood fingerprint, not as a literal molecular call.",
            "- Key question: do classes differ mainly by the same compound ratios, or by distinct neighborhoods?",
        ]
    )
    for class_label, group in class_neighborhood_df.groupby("class_label", sort=True):
        top = group.sort_values("support_fraction", ascending=False).head(3)
        report_lines.append(
            f"- `{class_label}`: "
            + "; ".join([f"`{r.compound_label}` ({float(r.support_fraction):.3f})" for r in top.itertuples(index=False)])
        )
    report_lines.extend(
        [
            "",
            "## 5. Interpretation",
            "- Language is intentionally conservative.",
            "- These fingerprints are best read as relative support shifts consistent with purine-adjacent contributions and methyladenine-like neighborhood changes.",
            "- The current locked universal grounding filter is narrow, so chemistry is expected to be narrow rather than broad.",
            "",
            "## 6. Key Observations",
            f"- Fingerprints are `{'stable' if float(mean_variance_by_class['mean_variance'].mean()) < 0.02 else 'unstable'}` by the simple intra-class variance check.",
            f"- Classes `{'do' if float(top_distances['euclidean_distance'].max()) > 0.2 else 'do not'}` separate in BSV space.",
            "- Chemistry remains narrow rather than broad under the locked purine-focused grounding pool.",
            "- Dominance is class-dependent: some classes are nearly single-neighborhood dominated while others show more mixed support.",
        ]
    )
    output_path = sprint_paths.report_dir / "GAIRAv3_Pilot1a_small2023_cellline_fingerprint_report.md"
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return output_path


def build_mixture_fingerprint_markdown_report(
    sprint_paths: AutoresearchSprintPaths,
    *,
    subset_alias: str,
    config_summary: dict[str, object],
    class_mean_bsv: pd.DataFrame,
    class_neighborhood_df: pd.DataFrame,
    class_neighborhood_entropy_df: pd.DataFrame,
    class_top1_dominance_df: pd.DataFrame,
    class_axis_entropy_df: pd.DataFrame,
    intra_class_variance_df: pd.DataFrame,
    inter_class_distance_df: pd.DataFrame,
    progression_df: pd.DataFrame | None,
) -> Path:
    axes = [axis for axis in ALL_AXES if axis in class_mean_bsv.columns]
    mean_variance_by_class = intra_class_variance_df.copy()
    mean_variance_by_class["mean_variance"] = mean_variance_by_class[axes].mean(axis=1)
    top_distances = (
        inter_class_distance_df[inter_class_distance_df["class_label_a"] != inter_class_distance_df["class_label_b"]]
        .sort_values("euclidean_distance", ascending=False)
        .head(6)
    )
    ordered_labels = infer_mixture_order(class_mean_bsv["class_label"].astype(str).tolist())
    report_lines = [
        "# GAIRAv3 Pilot 1b: small2023_mixture Fingerprint Report",
        "",
        "## 1. Overview",
        "- Pilot 1b tests whether fingerprints behave compositionally on the small2023 mixture panel under the locked deterministic baseline.",
        "- The analysis is probe-local and uses the clean Probe2 path only.",
        "- Interpretation is restricted to relative support shifts within the current purine-adjacent biochemical vocabulary.",
        f"- subset alias: `{subset_alias}`",
        *[f"- {key}: `{value}`" for key, value in config_summary.items()],
        "",
        "## 2. Mixture-class Fingerprints",
    ]
    for label in ordered_labels:
        row = class_mean_bsv[class_mean_bsv["class_label"].astype(str) == label].iloc[0]
        strongest = sorted([(axis, float(row[axis])) for axis in axes], key=lambda x: x[1], reverse=True)[:4]
        hood = class_neighborhood_df[class_neighborhood_df["class_label"].astype(str) == label].copy()
        hood_top = hood.sort_values("support_fraction", ascending=False).head(5)
        hood_entropy = float(
            class_neighborhood_entropy_df[class_neighborhood_entropy_df["class_label"].astype(str) == label][
                "neighborhood_entropy"
            ].iloc[0]
        )
        top1 = float(
            class_top1_dominance_df[class_top1_dominance_df["class_label"].astype(str) == label]["top1_fraction"].iloc[0]
        )
        axis_entropy = float(
            class_axis_entropy_df[class_axis_entropy_df["class_label"].astype(str) == label]["axis_entropy"].iloc[0]
        )
        report_lines.extend(
            [
                f"### {label}",
                "- BSV profile: " + ", ".join([f"`{axis}={value:.3f}`" for axis, value in strongest]),
                f"- Axis entropy: `{axis_entropy:.3f}`",
                f"- Neighborhood entropy: `{hood_entropy:.3f}`",
                f"- Top1 dominance: `{top1:.3f}`",
                f"- Radar figure: `figures/radar_{label}.png`",
                f"- Neighborhood figure: `figures/neighborhood_{label}.png`",
                "- Neighborhood composition: "
                + "; ".join([f"`{r.compound_label}` ({float(r.support_fraction):.3f})" for r in hood_top.itertuples(index=False)]),
                "",
            ]
        )
    report_lines.extend(
        [
            "## 3. Stability",
            "- Intra-class variance summarizes per-spectrum BSV dispersion within each mixture class.",
            "- Inter-class distance summarizes separation between mixture-class mean BSV vectors.",
            "- Lowest-variance classes:",
        ]
    )
    for row in mean_variance_by_class.sort_values("mean_variance").head(6).itertuples(index=False):
        report_lines.append(f"- `{row.class_label}` mean variance `{float(row.mean_variance):.4f}`")
    report_lines.append("- Largest class separations:")
    for row in top_distances.itertuples(index=False):
        report_lines.append(
            f"- `{row.class_label_a}` vs `{row.class_label_b}` distance `{float(row.euclidean_distance):.4f}`"
        )
    report_lines.extend(
        [
            "",
            "## 4. Mixture Behavior",
            "- Mixture codes are treated as an ordered class series only because the labels explicitly encode increasing numeric mixture levels (`c00` ... `c100`).",
            "- No hidden component metadata were invented. Dominant-component alignment is therefore judged only relative to the observed endpoint classes.",
        ]
    )
    if progression_df is not None and not progression_df.empty:
        for row in progression_df.sort_values("mixture_code_numeric").itertuples(index=False):
            report_lines.append(
                f"- `{row.class_label}`: toward-high-endpoint score `{float(row.toward_high_endpoint_score):.3f}`, "
                f"`{row.low_endpoint_top_compound}` fraction `{float(row.low_endpoint_compound_fraction):.3f}`, "
                f"`{row.high_endpoint_top_compound}` fraction `{float(row.high_endpoint_compound_fraction):.3f}`"
            )
    report_lines.extend(
        [
            "",
            "## 5. Interpretation",
            "- The key question is whether class fingerprints move plausibly along the ordered mixture path.",
            "- Strong claims about exact molecules remain inappropriate here. The correct reading is relative movement within a narrow purine-adjacent neighborhood.",
            "",
            "## 6. Key Observations",
            f"- Fingerprints are `{'stable' if float(mean_variance_by_class['mean_variance'].mean()) < 0.02 else 'unstable'}` by the simple intra-class variance check.",
            f"- Chemistry remains narrow under the locked purine-focused universal grounding pool.",
            "- The main evidence for compositional behavior is whether intermediate classes interpolate between endpoint fingerprints and endpoint neighborhoods.",
            "- If that interpolation is visible and stable, Pilot 1c cross-probe comparison becomes justified.",
        ]
    )
    output_path = sprint_paths.report_dir / "GAIRAv3_Pilot1b_small2023_mixture_fingerprint_report.md"
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return output_path


def build_pdf_report(markdown_path: Path, figure_paths: list[Path], output_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    lines = []
    for raw in text.splitlines():
        if raw.startswith("#"):
            lines.append(raw)
        elif raw.strip():
            lines.extend(textwrap.wrap(raw, width=96))
        else:
            lines.append("")
    with PdfPages(output_path) as pdf:
        chunk_size = 34
        for i in range(0, len(lines), chunk_size):
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.96
            for line in lines[i : i + chunk_size]:
                size = 12 if line.startswith("# ") else 10 if line.startswith("## ") else 9 if line.startswith("### ") else 8.6
                weight = "bold" if line.startswith("#") else "normal"
                fig.text(0.06, y, line, ha="left", va="top", fontsize=size, fontweight=weight, family="DejaVu Sans Mono")
                y -= 0.026 if line.startswith("#") else 0.023
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        for path in figure_paths:
            img = plt.imread(path)
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(path.name, fontsize=12, y=0.98)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
