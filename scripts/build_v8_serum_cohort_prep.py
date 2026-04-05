#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from gaira.demo.serum_delta_utils import (
    DELTA_AMBIGUOUS,
    DELTA_HIGH,
    DELTA_LOW,
    bootstrap_delta_stability,
    cosine_similarity,
    dataset_mapping_audit,
    load_serum_delta_inputs,
)
from gaira.demo.v8_analysis_utils import save_heatmap, save_scatter, sampled_silhouette
from gaira.demo.v8_master_utils import MASTER_SERUM_DIR, ensure_dir, safe_csv, safe_text
from gaira.demo.v8_theme_utils import MASTER_THEME_ORDER, compute_split_theme_profiles, split_grounding_theme_table, write_theme_split_note

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SERUM_STRESS_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_serum_stress_analysis_v1")
DEFAULT_SERUM_DELTA_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_serum_delta_analysis_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v8 serum cohort-mode prep outputs.")
    parser.add_argument("--output-dir", type=Path, default=MASTER_SERUM_DIR)
    parser.add_argument("--serum-stress-dir", type=Path, default=DEFAULT_SERUM_STRESS_DIR)
    parser.add_argument("--serum-delta-dir", type=Path, default=DEFAULT_SERUM_DELTA_DIR)
    parser.add_argument("--top-k-grounding", type=int, default=12)
    parser.add_argument("--bootstrap-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-group-size", type=int, default=40)
    return parser.parse_args()


def save_dataset_panels(df: pd.DataFrame, *, hue: str, title: str, output_path: Path, palette: dict | str = "deep") -> None:
    datasets = sorted(df["dataset_id"].dropna().unique().tolist())
    if not datasets:
        return
    n_cols = 2
    n_rows = int(np.ceil(len(datasets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, 4.2 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, dataset_id in zip(axes.ravel(), datasets, strict=False):
        subset = df[df["dataset_id"] == dataset_id].copy()
        ax.axis("on")
        sns.scatterplot(data=subset, x="dim1", y="dim2", hue=hue, ax=ax, palette=palette, s=20, alpha=0.45, linewidth=0)
        ax.set_title(dataset_id)
        legend = ax.get_legend()
        if legend:
            legend.remove()
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    common = load_serum_delta_inputs(serum_stress_dir=args.serum_stress_dir)
    metadata = common["metadata"].copy()  # type: ignore[assignment]
    embeddings = common["embeddings"]  # type: ignore[assignment]
    projection = common["projection"].copy()  # type: ignore[assignment]
    cluster_assignments = common["cluster_assignments"].copy()  # type: ignore[assignment]
    grounding_theme_table = common["grounding_theme_table"].copy()  # type: ignore[assignment]
    state_table = common["state_table"].copy()  # type: ignore[assignment]

    mapping_df = dataset_mapping_audit(state_table, min_group_size=args.min_group_size)
    mapping_df.to_csv(args.output_dir / "serum_dataset_state_mapping.csv", index=False)
    note = safe_text(args.serum_delta_dir / "serum_dataset_state_mapping_report.md")
    (args.output_dir / "serum_dataset_state_mapping_report.md").write_text(note, encoding="utf-8")

    serum = metadata[(metadata["sample_type"] == "serum") & (metadata["sample_key"].astype(str).isin(state_table["sample_key"].astype(str)))].copy()
    serum["sample_key"] = serum["sample_key"].astype(str)
    serum = serum.merge(state_table[["sample_key", "dataset_id", "delta_state"]], on=["sample_key", "dataset_id"], how="left")
    serum_assign = cluster_assignments[cluster_assignments["sample_key"].astype(str).isin(serum["sample_key"])][["sample_key", "within_type_cluster_id"]].copy()
    serum_assign["sample_key"] = serum_assign["sample_key"].astype(str)
    serum = serum.merge(serum_assign, on="sample_key", how="left")

    dataset_metric_rows = []
    for dataset_id, group in serum.groupby("dataset_id", sort=True):
        clear = group[group["delta_state"].isin([DELTA_LOW, DELTA_HIGH])].copy()
        if clear.empty:
            dataset_metric_rows.append(
                {
                    "dataset_id": dataset_id,
                    "n_total": int(len(group)),
                    "n_clear": 0,
                    "state_nn_purity": float("nan"),
                    "state_silhouette": float("nan"),
                    "cluster_count": int(group["within_type_cluster_id"].nunique()),
                    "cluster_state_entropy": float("nan"),
                }
            )
            continue
        Z = embeddings[clear.index.to_numpy()]
        low = clear[clear["delta_state"] == DELTA_LOW]
        high = clear[clear["delta_state"] == DELTA_HIGH]
        state_labels = clear["delta_state"].to_numpy()
        state_purity = float(
            (
                pd.Series(state_labels)
                .groupby(clear["within_type_cluster_id"].astype(str))
                .transform(lambda s: s.value_counts(normalize=True).iloc[0])
                .mean()
            )
        )
        cluster_state_entropy = float(
            clear.groupby("within_type_cluster_id")["delta_state"]
            .apply(lambda s: float(-(s.value_counts(normalize=True) * np.log2(s.value_counts(normalize=True) + 1e-12)).sum()))
            .mean()
        )
        dataset_metric_rows.append(
            {
                "dataset_id": dataset_id,
                "n_total": int(len(group)),
                "n_low": int(len(low)),
                "n_high": int(len(high)),
                "n_ambiguous": int((group["delta_state"] == DELTA_AMBIGUOUS).sum()),
                "n_clear": int(len(clear)),
                "state_silhouette": sampled_silhouette(Z, state_labels, seed=args.seed),
                "state_nn_purity": state_purity,
                "cluster_count": int(group["within_type_cluster_id"].nunique()),
                "cluster_state_entropy": cluster_state_entropy,
            }
        )
    dataset_metric_df = pd.DataFrame(dataset_metric_rows).sort_values("dataset_id").reset_index(drop=True)
    dataset_metric_df.to_csv(args.output_dir / "serum_dataset_latent_metrics.csv", index=False)

    split_grounding = split_grounding_theme_table(grounding_theme_table)
    split_grounding.to_csv(args.output_dir / "grounding_theme_table_split.csv", index=False)
    write_theme_split_note(args.output_dir / "theme_split_note.md", split_grounding)
    grounding_idx = metadata.index[metadata["sample_key"].astype(str).isin(split_grounding["sample_key"].astype(str))].to_numpy()
    serum_profiles = compute_split_theme_profiles(
        embeddings[serum.index.to_numpy()] / np.maximum(np.linalg.norm(embeddings[serum.index.to_numpy()], axis=1, keepdims=True), 1e-8),
        embeddings[grounding_idx] / np.maximum(np.linalg.norm(embeddings[grounding_idx], axis=1, keepdims=True), 1e-8),
        split_grounding,
        top_k=args.top_k_grounding,
    )
    serum_profiles.insert(0, "sample_key", serum["sample_key"].to_numpy())
    serum_profiles.insert(1, "dataset_id", serum["dataset_id"].to_numpy())
    serum_profiles.insert(2, "delta_state", serum["delta_state"].to_numpy())
    serum_profiles.insert(3, "within_type_cluster_id", serum["within_type_cluster_id"].to_numpy())
    serum_profiles.to_csv(args.output_dir / "serum_spectrum_composition_profiles.csv", index=False)

    dataset_comp = serum_profiles.groupby(["dataset_id", "delta_state"], as_index=False)[MASTER_THEME_ORDER].mean()
    dataset_comp.to_csv(args.output_dir / "serum_dataset_composition_profiles.csv", index=False)

    included = mapping_df[mapping_df["include_in_delta"]]["dataset_id"].tolist()
    delta_rows = []
    delta_similarity_rows = []
    theme_consensus_rows = []
    latent_vectors: dict[str, np.ndarray] = {}
    comp_vectors: dict[str, np.ndarray] = {}

    for dataset_id in included:
        clear = serum[serum["dataset_id"] == dataset_id].copy()
        clear = clear[clear["delta_state"].isin([DELTA_LOW, DELTA_HIGH])].copy()
        low = clear[clear["delta_state"] == DELTA_LOW]
        high = clear[clear["delta_state"] == DELTA_HIGH]
        low_values = embeddings[low.index.to_numpy()]
        high_values = embeddings[high.index.to_numpy()]
        latent_delta = high_values.mean(axis=0) - low_values.mean(axis=0)
        latent_vectors[dataset_id] = latent_delta
        stability = bootstrap_delta_stability(low_values, high_values, seed=args.seed, n_bootstrap=args.bootstrap_iters)

        low_comp = serum_profiles[(serum_profiles["dataset_id"] == dataset_id) & (serum_profiles["delta_state"] == DELTA_LOW)][MASTER_THEME_ORDER].mean()
        high_comp = serum_profiles[(serum_profiles["dataset_id"] == dataset_id) & (serum_profiles["delta_state"] == DELTA_HIGH)][MASTER_THEME_ORDER].mean()
        comp_delta = (high_comp - low_comp).fillna(0.0)
        comp_vectors[dataset_id] = comp_delta.to_numpy(dtype=float)
        delta_row = {"dataset_id": dataset_id, **stability}
        delta_row.update({theme: float(comp_delta[theme]) for theme in MASTER_THEME_ORDER})
        delta_rows.append(delta_row)

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(args.output_dir / "serum_delta_metrics.csv", index=False)

    for left in included:
        for right in included:
            delta_similarity_rows.append(
                {
                    "left_dataset": left,
                    "right_dataset": right,
                    "latent_delta_cosine": cosine_similarity(latent_vectors[left], latent_vectors[right]),
                    "composition_delta_cosine": cosine_similarity(comp_vectors[left], comp_vectors[right]),
                }
            )
    delta_similarity_df = pd.DataFrame(delta_similarity_rows)
    delta_similarity_df.to_csv(args.output_dir / "serum_delta_similarity.csv", index=False)

    if not delta_df.empty:
        for theme in MASTER_THEME_ORDER:
            values = delta_df[theme].to_numpy(dtype=float)
            inc = int((values > 0).sum())
            dec = int((values < 0).sum())
            theme_consensus_rows.append(
                {
                    "theme": theme,
                    "increase_count": inc,
                    "decrease_count": dec,
                    "mean_delta": float(values.mean()),
                    "abs_mean_delta": float(np.abs(values).mean()),
                    "sign_consistency": float(max(inc, dec) / max(len(values), 1)),
                }
            )
    theme_consensus_df = pd.DataFrame(theme_consensus_rows).sort_values(["sign_consistency", "abs_mean_delta"], ascending=[False, False])
    theme_consensus_df.to_csv(args.output_dir / "serum_theme_consensus.csv", index=False)

    projection_serum = projection[(projection["sample_type"] == "serum") & (projection["sample_key"].astype(str).isin(serum["sample_key"]))].copy()
    projection_serum["sample_key"] = projection_serum["sample_key"].astype(str)
    projection_serum = projection_serum.merge(serum[["sample_key", "delta_state", "within_type_cluster_id"]], on="sample_key", how="left")
    save_dataset_panels(projection_serum, hue="within_type_cluster_id", title="Serum dataset latent maps", output_path=args.output_dir / "serum_dataset_latent_maps.png", palette="tab20")
    save_dataset_panels(projection_serum.fillna({"delta_state": DELTA_AMBIGUOUS}), hue="delta_state", title="Serum dataset state maps", output_path=args.output_dir / "serum_dataset_state_maps.png", palette="deep")

    cluster_comp = serum_profiles.groupby(["dataset_id", "within_type_cluster_id"], as_index=False)[MASTER_THEME_ORDER].mean()
    if not cluster_comp.empty:
        cluster_comp["cluster_label"] = cluster_comp["dataset_id"].astype(str) + "::" + cluster_comp["within_type_cluster_id"].astype(str)
        save_heatmap(
            cluster_comp.set_index("cluster_label")[MASTER_THEME_ORDER],
            title="Serum dataset composition heatmaps",
            output_path=args.output_dir / "serum_dataset_composition_heatmaps.png",
        )

    if not delta_similarity_df.empty:
        latent_pivot = delta_similarity_df.pivot(index="left_dataset", columns="right_dataset", values="latent_delta_cosine").fillna(0.0)
        save_heatmap(latent_pivot, title="Serum delta similarity heatmap", output_path=args.output_dir / "serum_delta_similarity_heatmap.png", cmap="vlag")

    if not theme_consensus_df.empty:
        plt.figure(figsize=(9.2, 5.8))
        ax = sns.barplot(data=theme_consensus_df, x="theme", y="sign_consistency", color="#4c78a8")
        ax.set_title("Serum theme shift consistency")
        ax.tick_params(axis="x", rotation=28)
        plt.tight_layout()
        plt.savefig(args.output_dir / "serum_theme_shift_summary.png", dpi=220)
        plt.close()

    clear_profiles = serum_profiles[serum_profiles["delta_state"].isin([DELTA_LOW, DELTA_HIGH])].copy()
    if not clear_profiles.empty:
        plot_df = clear_profiles.groupby(["dataset_id", "delta_state"], as_index=False)[MASTER_THEME_ORDER].mean()
        melted = plot_df.melt(id_vars=["dataset_id", "delta_state"], value_vars=MASTER_THEME_ORDER, var_name="theme", value_name="weight")
        plt.figure(figsize=(10.8, 6.0))
        ax = sns.scatterplot(data=melted, x="theme", y="weight", hue="delta_state", style="dataset_id", s=85)
        ax.set_title("Serum within-dataset state composition scatter")
        ax.tick_params(axis="x", rotation=28)
        plt.tight_layout()
        plt.savefig(args.output_dir / "serum_within_dataset_state_scatter.png", dpi=220)
        plt.close()

    summary_lines = [
        "# Serum Cohort Prep Summary",
        "",
        "Readout:",
        "- Serum currently supports dataset-level biochemical composition analysis and within-dataset state-shift analysis.",
        "- Serum does not yet support a shared cross-dataset disease manifold analysis strong enough to justify a dedicated head.",
        "- The master prep therefore treats serum as cohort-mode interpretation only.",
        "",
        "Delta alignment:",
        f"- included biological contrast datasets: {', '.join(included) if included else 'none'}",
        f"- mean latent delta cosine across included datasets: {delta_similarity_df[delta_similarity_df['left_dataset'] != delta_similarity_df['right_dataset']]['latent_delta_cosine'].mean():.4f}" if len(included) > 1 else "- mean latent delta cosine across included datasets: not computable",
        f"- mean composition delta cosine across included datasets: {delta_similarity_df[delta_similarity_df['left_dataset'] != delta_similarity_df['right_dataset']]['composition_delta_cosine'].mean():.4f}" if len(included) > 1 else "- mean composition delta cosine across included datasets: not computable",
    ]
    (args.output_dir / "serum_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    recommendation_lines = [
        "# Serum v8 Recommendation",
        "",
        "Recommendation:",
        "- no dedicated serum head yet",
        "- cohort-only inference / interpretation mode is justified now",
        "- revisit a dedicated serum head only after more harmonized disease cohorts and cleaner cross-cohort bridge anchors exist",
        "",
        "Why:",
        "- per-dataset state shifts are stable enough to interpret locally",
        "- cross-dataset delta alignment remains weak",
        "- protocol, spiking, and calibration archives still dominate too much of the serum universe to justify a shared serum manifold claim",
    ]
    (args.output_dir / "serum_v8_recommendation.md").write_text("\n".join(recommendation_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
