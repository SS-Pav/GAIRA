#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from gaira.demo.ev_analysis_utils import (
    DEFAULT_ANCHOR_AUDIT_DIR,
    DEFAULT_CLUSTER_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_GROUNDING_DIR,
    DEFAULT_RUN_DIR,
    STATE_COLORS,
    THEME_COLORS,
    THEME_ORDER,
    cluster_composition_summary,
    cluster_label_enrichment,
    compute_theme_profiles,
    entropy_normalized,
    knn_label_metrics,
    load_common_artifacts,
    normalize_rows,
    sampled_global_metrics,
    save_heatmap,
    save_scatter,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/ev_stress_disease_analysis_v1")
TARGET_DATASETS = ["shine_ev_sers", "diabetes_plasma_ev_sers"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EV stress/disease analysis for GAIRAM.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--grounding-dir", type=Path, default=DEFAULT_GROUNDING_DIR)
    parser.add_argument("--anchor-audit-dir", type=Path, default=DEFAULT_ANCHOR_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k-grounding", type=int, default=12)
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def harmonized_state_for_row(dataset_id: str, label: str) -> tuple[str, str]:
    dataset_id = str(dataset_id)
    label = str(label).strip()
    if dataset_id == "diabetes_plasma_ev_sers":
        if label == "Strong-D":
            return "control_like", "Dataset release treats Strong-D as the lower-stress baseline arm."
        if label == "Impact":
            return "stress_or_toxicity_like", "Dataset release treats Impact as the perturbed arm."
        return "intermediate_or_ambiguous", "Unexpected diabetes EV label; left broad and cautious."

    if dataset_id == "shine_ev_sers":
        try:
            day_token, conc_token = label.split("_")
            day = int(day_token.replace("D", ""))
            conc = int(conc_token.replace("C", ""))
        except Exception:
            return "intermediate_or_ambiguous", "SHINE label could not be parsed cleanly."

        if day == 0 and conc <= 10:
            return "control_like", "Earliest SHINE exposure and low concentration were treated as control-like."
        if day >= 2 or conc >= 20:
            return "stress_or_toxicity_like", "Later SHINE exposure or higher concentration was treated as stress/toxicity-like."
        return "intermediate_or_ambiguous", "Middle SHINE exposure regime kept broad rather than forced."

    return "unmapped", "Dataset not part of the stress/disease EV analysis."


def build_harmonized_state_table(metadata: pd.DataFrame) -> pd.DataFrame:
    subset = metadata[
        (metadata["sample_type"] == "ev")
        & (metadata["record_kind"] == "processed_spectrum")
        & (metadata["dataset_id"].isin(TARGET_DATASETS))
    ].copy()
    states = subset.apply(
        lambda row: harmonized_state_for_row(row["dataset_id"], row["label_optional"]),
        axis=1,
        result_type="expand",
    )
    subset["harmonized_state"] = states[0]
    subset["harmonization_note"] = states[1]
    return subset


def cluster_entropy_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for cluster_id, group in df.groupby("within_type_cluster_id", sort=True):
        valid = group[label_col].fillna("").astype(str)
        valid = valid[valid != ""]
        rows.append(
            {
                "cluster_id": cluster_id,
                f"{label_col}_entropy": entropy_normalized(valid),
                f"{label_col}_unique_count": int(valid.nunique()),
            }
        )
    return pd.DataFrame(rows)


def cluster_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, group in df.groupby("within_type_cluster_id", sort=True):
        counts = group["harmonized_state"].value_counts(normalize=True)
        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": int(len(group)),
                "control_like_fraction": float(counts.get("control_like", 0.0)),
                "stress_or_toxicity_like_fraction": float(counts.get("stress_or_toxicity_like", 0.0)),
                "intermediate_or_ambiguous_fraction": float(counts.get("intermediate_or_ambiguous", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def summarize_latent_structure(
    state_df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    knn_k: int,
    seed: int,
) -> pd.DataFrame:
    idx = state_df.index.to_numpy()
    X = normalize_rows(embeddings[idx])
    metrics = []

    dataset_metrics = knn_label_metrics(X, state_df["dataset_id"].to_numpy(), k=knn_k)
    state_metrics = knn_label_metrics(X, state_df["harmonized_state"].to_numpy(), k=knn_k)
    cluster_metrics = knn_label_metrics(X, state_df["within_type_cluster_id"].to_numpy(), k=knn_k)
    dataset_global = sampled_global_metrics(X, state_df["dataset_id"].to_numpy(), seed=seed)
    state_global = sampled_global_metrics(X, state_df["harmonized_state"].to_numpy(), seed=seed)

    metrics.extend(
        [
            {"metric": "n_spectra", "value": float(len(state_df)), "scope": "full_corpus"},
            {"metric": "n_clusters_represented", "value": float(state_df["within_type_cluster_id"].nunique()), "scope": "full_corpus"},
            {"metric": "mean_cluster_size", "value": float(state_df.groupby("within_type_cluster_id").size().mean()), "scope": "full_corpus"},
            {"metric": "median_cluster_size", "value": float(state_df.groupby("within_type_cluster_id").size().median()), "scope": "full_corpus"},
            {"metric": "cross_dataset_mixed_cluster_count", "value": float(state_df.groupby("within_type_cluster_id")["dataset_id"].nunique().ge(2).sum()), "scope": "full_corpus"},
            {"metric": "nn_purity_dataset", "value": dataset_metrics["nn_purity"], "scope": "full_corpus"},
            {"metric": "nn_purity_harmonized_state", "value": state_metrics["nn_purity"], "scope": "full_corpus"},
            {"metric": "nn_purity_cluster", "value": cluster_metrics["nn_purity"], "scope": "full_corpus"},
            {"metric": "neighbor_entropy_dataset", "value": dataset_metrics["neighbor_entropy"], "scope": "full_corpus"},
            {"metric": "neighbor_entropy_harmonized_state", "value": state_metrics["neighbor_entropy"], "scope": "full_corpus"},
            {"metric": "top1_match_dataset", "value": dataset_metrics["top1_match"], "scope": "full_corpus"},
            {"metric": "top1_match_harmonized_state", "value": state_metrics["top1_match"], "scope": "full_corpus"},
            {"metric": "silhouette_dataset", "value": dataset_global["silhouette"], "scope": "sampled_global"},
            {"metric": "silhouette_harmonized_state", "value": state_global["silhouette"], "scope": "sampled_global"},
            {"metric": "davies_bouldin_dataset", "value": dataset_global["davies_bouldin"], "scope": "sampled_global"},
            {"metric": "davies_bouldin_harmonized_state", "value": state_global["davies_bouldin"], "scope": "sampled_global"},
            {"metric": "calinski_harabasz_dataset", "value": dataset_global["calinski_harabasz"], "scope": "sampled_global"},
            {"metric": "calinski_harabasz_harmonized_state", "value": state_global["calinski_harabasz"], "scope": "sampled_global"},
        ]
    )
    return pd.DataFrame(metrics)


def build_projection_table(
    projection: pd.DataFrame,
    state_df: pd.DataFrame,
    cluster_interp: pd.DataFrame,
) -> pd.DataFrame:
    sampled = projection[
        (projection["sample_type"] == "ev")
        & (projection["dataset_id"].isin(TARGET_DATASETS))
    ].copy()
    merge_cols = ["sample_key", "harmonized_state", "within_type_cluster_id", "dataset_id"]
    sampled = sampled.merge(state_df[merge_cols], on=["sample_key", "dataset_id"], how="left")
    sampled = sampled.merge(
        cluster_interp[
            [
                "cluster_id",
                "top_biochemical_theme",
                "secondary_biochemical_theme",
                "theme_support_strength",
                "cross_dataset_mixed",
            ]
        ],
        left_on="within_type_cluster_id",
        right_on="cluster_id",
        how="left",
    )
    return sampled


def save_stress_figures(
    sampled_projection: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    cluster_state: pd.DataFrame,
    cluster_metrics: pd.DataFrame,
    state_linkage: pd.DataFrame,
    output_dir: Path,
) -> None:
    centroid_map = (
        sampled_projection.groupby("within_type_cluster_id", as_index=False)
        .agg(
            dim1=("dim1", "mean"),
            dim2=("dim2", "mean"),
            cluster_size=("sample_key", "count"),
            cross_dataset_mixed=("cross_dataset_mixed", "max"),
            top_biochemical_theme=("top_biochemical_theme", "first"),
            harmonized_state=("harmonized_state", lambda s: s.value_counts().index[0] if len(s) else "unmapped"),
            theme_support_strength=("theme_support_strength", "first"),
        )
    )
    centroid_map["mixing_category"] = np.where(centroid_map["cross_dataset_mixed"].fillna(False), "mixed", "single_dataset")
    save_scatter(
        centroid_map,
        x_col="dim1",
        y_col="dim2",
        color_col="mixing_category",
        output_path=output_dir / "neutral_latent_map.png",
        title="EV stress/disease latent map (neutral structure view)",
        color_map={"mixed": "#d07a37", "single_dataset": "#6f7a85"},
        size_col="cluster_size",
        marker_col="mixing_category",
    )
    save_scatter(
        sampled_projection.dropna(subset=["harmonized_state"]),
        x_col="dim1",
        y_col="dim2",
        color_col="harmonized_state",
        output_path=output_dir / "latent_map_by_state.png",
        title="EV stress/disease latent map painted by harmonized broad state",
        color_map=STATE_COLORS,
    )
    save_scatter(
        centroid_map.fillna({"top_biochemical_theme": "unresolved"}),
        x_col="dim1",
        y_col="dim2",
        color_col="top_biochemical_theme",
        output_path=output_dir / "latent_map_by_dominant_biochemical_theme.png",
        title="EV stress/disease latent map painted by dominant biochemical theme",
        color_map=THEME_COLORS,
        size_col="cluster_size",
        marker_col="mixing_category",
    )
    heatmap_profiles = cluster_profiles.set_index("cluster_id")[THEME_ORDER]
    save_heatmap(
        heatmap_profiles,
        output_path=output_dir / "cluster_composition_heatmap.png",
        title="Cluster biochemical composition profiles",
        figsize=(9.5, max(6.0, len(heatmap_profiles) * 0.18)),
    )
    heatmap_state = cluster_state.set_index("cluster_id")[
        ["control_like_fraction", "stress_or_toxicity_like_fraction", "intermediate_or_ambiguous_fraction"]
    ]
    save_heatmap(
        heatmap_state,
        output_path=output_dir / "cluster_state_heatmap.png",
        title="Cluster state fractions",
        figsize=(7.5, max(6.0, len(heatmap_state) * 0.18)),
        cmap="magma",
    )

    merged = state_linkage.merge(cluster_metrics, on="cluster_id", how="left", suffixes=("", "_metric"))
    size_col = "cluster_size"
    if size_col not in merged.columns and "cluster_size_metric" in merged.columns:
        size_col = "cluster_size_metric"
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
    axes[0].scatter(
        merged["protein_peptide_associated"],
        merged["stress_or_toxicity_like_log2_odds"],
        s=np.clip(merged[size_col].to_numpy(dtype=float) / 8.0, 20, 180),
        c="#3e6ea1",
        alpha=0.76,
    )
    axes[0].set_xlabel("protein/peptide composition weight")
    axes[0].set_ylabel("stress-state enrichment log2 odds")
    axes[0].set_title("Protein/peptide vs stress enrichment")
    axes[1].scatter(
        merged["oxidative_redox_associated"],
        merged["stress_or_toxicity_like_log2_odds"],
        s=np.clip(merged[size_col].to_numpy(dtype=float) / 8.0, 20, 180),
        c="#7a4b9d",
        alpha=0.76,
    )
    axes[1].set_xlabel("oxidative/redox composition weight")
    axes[1].set_ylabel("stress-state enrichment log2 odds")
    axes[1].set_title("Oxidative/redox vs stress enrichment")
    fig.tight_layout()
    fig.savefig(output_dir / "composition_vs_state_scatter.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_common_artifacts(
        run_dir=args.run_dir,
        eval_dir=args.eval_dir,
        cluster_dir=args.cluster_dir,
        grounding_dir=args.grounding_dir,
        anchor_audit_dir=args.anchor_audit_dir,
    )

    metadata = artifacts["metadata"].copy()  # type: ignore[assignment]
    embeddings = artifacts["embeddings"]  # type: ignore[assignment]
    projection = artifacts["projection"].copy()  # type: ignore[assignment]
    cluster_assignments = artifacts["cluster_assignments"].copy()  # type: ignore[assignment]
    cluster_interp = artifacts["ev_cluster_interpretation"].copy()  # type: ignore[assignment]
    grounding_theme_table = artifacts["grounding_theme_table"].copy()  # type: ignore[assignment]

    state_df = build_harmonized_state_table(metadata)
    state_df = state_df.merge(
        cluster_assignments[["sample_key", "within_type_cluster_id"]],
        on="sample_key",
        how="left",
    )
    state_df.to_csv(args.output_dir / "harmonized_state_table.csv", index=False)

    harmonized_report = textwrap.dedent(
        """\
        # Harmonized EV Stress/Disease State Table

        This analysis uses only `shine_ev_sers` and `diabetes_plasma_ev_sers`.

        Mapping rules:
        - `diabetes_plasma_ev_sers`: `Strong-D -> control_like`, `Impact -> stress_or_toxicity_like`.
        - `shine_ev_sers`: earliest + low-exposure conditions (`D0` and concentration <= 10) were treated as `control_like`.
        - `shine_ev_sers`: later or higher-exposure conditions (`day >= 2` or concentration >= 20) were treated as `stress_or_toxicity_like`.
        - SHINE middle-regime cases were left as `intermediate_or_ambiguous` rather than forced.

        This is a broad-state harmonization only. It does not claim disease equivalence across datasets.
        """
    )
    (args.output_dir / "harmonized_state_report.md").write_text(harmonized_report, encoding="utf-8")

    latent_metrics = summarize_latent_structure(state_df, embeddings, knn_k=args.knn_k, seed=args.seed)
    latent_metrics.to_csv(args.output_dir / "latent_structure_metrics.csv", index=False)

    dataset_entropy = cluster_entropy_table(state_df, "dataset_id")
    state_entropy = cluster_entropy_table(state_df, "harmonized_state")
    cluster_state = cluster_state_summary(state_df)
    cluster_state_enrichment = cluster_label_enrichment(state_df, cluster_col="within_type_cluster_id", label_col="harmonized_state")
    cluster_state_enrichment.to_csv(args.output_dir / "cluster_state_enrichment.csv", index=False)

    grounding_mask = grounding_theme_table["sample_key"].isin(
        metadata[metadata["sample_type"] == "grounding"]["sample_key"]
    )
    grounding_metadata = grounding_theme_table.loc[grounding_mask].copy().reset_index(drop=True)
    grounding_keys = set(grounding_metadata["sample_key"].astype(str))
    grounding_idx = metadata.index[metadata["sample_key"].astype(str).isin(grounding_keys)].to_numpy()
    grounding_embeddings = embeddings[grounding_idx]
    grounding_themes = grounding_metadata["grounding_theme"].astype(str).to_numpy()

    per_spectrum_path = args.output_dir / "per_spectrum_composition_profiles.csv"
    cluster_profiles_path = args.output_dir / "per_cluster_composition_profiles.csv"
    cluster_metrics_path = args.output_dir / "cluster_composition_metrics.csv"
    if per_spectrum_path.exists() and cluster_profiles_path.exists() and cluster_metrics_path.exists():
        per_spectrum = pd.read_csv(per_spectrum_path)
        cluster_profiles = pd.read_csv(cluster_profiles_path)
        cluster_metrics = pd.read_csv(cluster_metrics_path)
    else:
        composition_profiles = compute_theme_profiles(
            normalize_rows(embeddings[state_df.index.to_numpy()]),
            normalize_rows(grounding_embeddings),
            grounding_themes,
            top_k=args.top_k_grounding,
        )
        per_spectrum = state_df.reset_index(drop=True)[
            ["sample_key", "dataset_id", "label_optional", "harmonized_state", "within_type_cluster_id"]
        ].join(composition_profiles)
        per_spectrum.to_csv(per_spectrum_path, index=False)

        cluster_profiles, cluster_metrics = cluster_composition_summary(
            per_spectrum,
            cluster_col="within_type_cluster_id",
            theme_cols=THEME_ORDER,
        )
        cluster_profiles = cluster_profiles.rename(columns={"within_type_cluster_id": "cluster_id"})
        cluster_metrics = cluster_metrics.rename(columns={"within_type_cluster_id": "cluster_id"})
        cluster_profiles.to_csv(cluster_profiles_path, index=False)
        cluster_metrics.to_csv(cluster_metrics_path, index=False)

    state_linkage = cluster_profiles.merge(cluster_state, on="cluster_id", how="left")
    stress_rows = cluster_state_enrichment[cluster_state_enrichment["harmonized_state"] == "stress_or_toxicity_like"][
        ["within_type_cluster_id", "log2_odds"]
    ].rename(columns={"within_type_cluster_id": "cluster_id", "log2_odds": "stress_or_toxicity_like_log2_odds"})
    control_rows = cluster_state_enrichment[cluster_state_enrichment["harmonized_state"] == "control_like"][
        ["within_type_cluster_id", "log2_odds"]
    ].rename(columns={"within_type_cluster_id": "cluster_id", "log2_odds": "control_like_log2_odds"})
    state_linkage = state_linkage.merge(stress_rows, on="cluster_id", how="left").merge(control_rows, on="cluster_id", how="left")

    linkage_rows = []
    for theme in THEME_ORDER:
        subset = state_linkage[[theme, "stress_or_toxicity_like_log2_odds"]].dropna()
        corr = subset[theme].corr(subset["stress_or_toxicity_like_log2_odds"]) if len(subset) >= 3 else np.nan
        linkage_rows.append({"metric": f"{theme}_vs_stress_log2_odds_corr", "value": float(corr) if pd.notna(corr) else np.nan})
    state_linkage_metrics = pd.DataFrame(linkage_rows)
    state_linkage_metrics.to_csv(args.output_dir / "state_composition_linkage_metrics.csv", index=False)

    sampled_projection = build_projection_table(projection, state_df, cluster_interp)
    save_stress_figures(sampled_projection, cluster_profiles, cluster_state, cluster_metrics, state_linkage, args.output_dir)

    latent_report = textwrap.dedent(
        f"""\
        # EV Stress / Disease Latent Structure Report

        Records analyzed: {len(state_df):,}
        Clusters represented: {state_df['within_type_cluster_id'].nunique():,}

        Key full-corpus neighborhood metrics:
        - dataset nn purity: {latent_metrics.loc[latent_metrics.metric == 'nn_purity_dataset', 'value'].iloc[0]:.4f}
        - broad-state nn purity: {latent_metrics.loc[latent_metrics.metric == 'nn_purity_harmonized_state', 'value'].iloc[0]:.4f}
        - sampled broad-state silhouette: {latent_metrics.loc[latent_metrics.metric == 'silhouette_harmonized_state', 'value'].iloc[0]:.4f}

        Interpretation:
        - The structure remains strongly clustered by local neighborhood.
        - Broad-state organization is present, but weaker than dataset organization.
        - Cross-dataset mixing exists in a minority of EV clusters rather than across the whole map.
        """
    )
    (args.output_dir / "latent_structure_report.md").write_text(latent_report, encoding="utf-8")

    composition_report = textwrap.dedent(
        f"""\
        # EV Stress / Disease Composition Report

        Per-spectrum biochemical theme profiles were computed by retrieving the top {args.top_k_grounding} nearest grounding embeddings
        and aggregating their broad theme weights into normalized composition vectors.

        Key observations:
        - dominant theme counts by cluster are stored in `per_cluster_composition_profiles.csv`
        - coherence metrics are stored in `cluster_composition_metrics.csv`
        - serum_matrix_associated is retained as a caveat-like theme rather than an EV headline narrative
        """
    )
    (args.output_dir / "composition_report.md").write_text(composition_report, encoding="utf-8")

    linkage_report = textwrap.dedent(
        """\
        # EV Stress / Disease State-Composition Linkage Report

        Cluster-level broad-state enrichment was computed from harmonized state proportions and log2 enrichment against the
        global control-like / stress-like distribution. These enrichments were then compared with grounding-derived
        biochemical composition weights.

        Read the linkage as broad-pattern evidence only. It is not a molecule-level attribution layer.
        """
    )
    (args.output_dir / "state_composition_linkage_report.md").write_text(linkage_report, encoding="utf-8")

    control_enriched = state_linkage[state_linkage["control_like_fraction"] >= 0.65]
    stress_enriched = state_linkage[state_linkage["stress_or_toxicity_like_fraction"] >= 0.65]
    summary_lines = [
        "# EV Stress / Disease Summary",
        "",
        f"- total spectra analyzed: {len(state_df):,}",
        f"- sampled latent-map points available: {len(sampled_projection):,}",
        f"- cross-dataset mixed clusters in this subset: {int(state_df.groupby('within_type_cluster_id')['dataset_id'].nunique().ge(2).sum())}",
        f"- control-enriched clusters (>= 0.65 fraction): {len(control_enriched)}",
        f"- stress-enriched clusters (>= 0.65 fraction): {len(stress_enriched)}",
        "",
        "Assessment:",
        "- Meaningful unsupervised EV latent structure is present.",
        "- Broad-state organization is detectable, but it is weaker than dataset identity.",
        "- Grounding-derived composition profiles are more informative as broad mixtures than as single-label cluster tags.",
        "- The stress/disease page is supportable for a cautious demo if it is framed as broad-state and composition structure, not as diagnosis.",
    ]
    (args.output_dir / "ev_stress_disease_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
