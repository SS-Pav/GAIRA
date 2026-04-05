#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gaira.demo.ev_analysis_utils import THEME_ORDER, compute_theme_profiles, normalize_rows
from gaira.demo.serum_delta_utils import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SERUM_STRESS_DIR,
    DELTA_AMBIGUOUS,
    DELTA_HIGH,
    DELTA_LOW,
    bootstrap_delta_stability,
    dataset_mapping_audit,
    load_serum_delta_inputs,
    pairwise_similarity_matrix,
    save_centroid_shift_panels,
    save_delta_magnitude_vs_stability,
    save_similarity_heatmap,
    save_state_count_figure,
    save_theme_consensus_bars,
    save_theme_delta_heatmap,
    save_composition_profile_lines,
    theme_consensus_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v8 serum delta analysis.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--serum-stress-dir", type=Path, default=DEFAULT_SERUM_STRESS_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-group-size", type=int, default=40)
    parser.add_argument("--top-k-grounding", type=int, default=12)
    parser.add_argument("--bootstrap-iters", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[serum-delta] loading inputs")
    common = load_serum_delta_inputs(serum_stress_dir=args.serum_stress_dir)
    metadata = common["metadata"].copy()  # type: ignore[assignment]
    embeddings = common["embeddings"]  # type: ignore[assignment]
    grounding_theme_table = common["grounding_theme_table"].copy()  # type: ignore[assignment]
    state_table = common["state_table"].copy()  # type: ignore[assignment]

    print("[serum-delta] auditing dataset mappings")
    mapping_df = dataset_mapping_audit(state_table, min_group_size=args.min_group_size)
    mapping_df.to_csv(args.output_dir / "serum_dataset_state_mapping.csv", index=False)

    included = mapping_df[mapping_df["include_in_delta"]].copy()
    included_datasets = included["dataset_id"].tolist()

    report_lines = [
        "# Serum Dataset State Mapping Report",
        "",
        "This audit reuses the conservative serum harmonization from the existing v8 serum stress analysis and converts it into a delta-analysis inclusion decision.",
        "",
        "Included datasets:",
    ]
    if included_datasets:
        report_lines.extend([f"- {ds}" for ds in included_datasets])
    else:
        report_lines.append("- none")
    report_lines.extend(["", "Excluded datasets:"])
    excluded_rows = mapping_df[~mapping_df["include_in_delta"]]
    if excluded_rows.empty:
        report_lines.append("- none")
    else:
        report_lines.extend([f"- {row.dataset_id}: {row.decision_reason}" for row in excluded_rows.itertuples(index=False)])
    (args.output_dir / "serum_dataset_state_mapping_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    save_state_count_figure(mapping_df, args.output_dir / "dataset_state_counts.png")

    print("[serum-delta] building latent deltas")
    latent_rows = []
    latent_metric_rows = []
    composition_delta_rows = []
    composition_metric_rows = []
    included_state_metadata = []
    latent_delta_vectors: dict[str, np.ndarray] = {}
    composition_delta_vectors: dict[str, np.ndarray] = {}

    grounding_idx = metadata.index[metadata["sample_key"].astype(str).isin(grounding_theme_table["sample_key"].astype(str))].to_numpy()
    grounding_embeddings = normalize_rows(embeddings[grounding_idx])
    grounding_themes = grounding_theme_table["grounding_theme"].astype(str).to_numpy()

    for dataset_id in included_datasets:
        subset = state_table[state_table["dataset_id"] == dataset_id].copy()
        subset["sample_key"] = subset["sample_key"].astype(str)
        subset = subset.merge(
            metadata[["sample_key", "dataset_id", "label_optional"]].assign(sample_key=lambda df: df["sample_key"].astype(str)),
            on=["sample_key", "dataset_id", "label_optional"],
            how="left",
        )
        subset_meta = metadata[metadata["sample_key"].astype(str).isin(subset["sample_key"])].copy()
        subset_meta = subset_meta.merge(subset[["sample_key", "delta_state"]], on="sample_key", how="left")
        subset_meta = subset_meta[subset_meta["delta_state"].isin([DELTA_LOW, DELTA_HIGH])].copy()
        included_state_metadata.append(subset_meta)

        low_meta = subset_meta[subset_meta["delta_state"] == DELTA_LOW]
        high_meta = subset_meta[subset_meta["delta_state"] == DELTA_HIGH]
        low_values = normalize_rows(embeddings[low_meta.index.to_numpy()])
        high_values = normalize_rows(embeddings[high_meta.index.to_numpy()])
        delta = high_values.mean(axis=0) - low_values.mean(axis=0)
        latent_delta_vectors[dataset_id] = delta.astype(float)
        latent_rows.append(pd.DataFrame({"dataset_id": dataset_id, "dimension": np.arange(len(delta)), "delta_value": delta}))

        stability = bootstrap_delta_stability(low_values, high_values, seed=args.seed, n_bootstrap=args.bootstrap_iters)
        latent_metric_rows.append(
            {
                "dataset_id": dataset_id,
                "low_n": int(len(low_meta)),
                "high_n": int(len(high_meta)),
                "total_n": int(len(subset_meta)),
                **stability,
            }
        )

        subset_profiles = compute_theme_profiles(
            normalize_rows(embeddings[subset_meta.index.to_numpy()]),
            grounding_embeddings,
            grounding_themes,
            top_k=args.top_k_grounding,
        )
        subset_profiles.insert(0, "sample_key", subset_meta["sample_key"].to_numpy())
        subset_profiles.insert(1, "dataset_id", dataset_id)
        subset_profiles.insert(2, "delta_state", subset_meta["delta_state"].to_numpy())
        low_comp = subset_profiles[subset_profiles["delta_state"] == DELTA_LOW][THEME_ORDER].mean()
        high_comp = subset_profiles[subset_profiles["delta_state"] == DELTA_HIGH][THEME_ORDER].mean()
        comp_delta = (high_comp - low_comp).astype(float)
        composition_delta_vectors[dataset_id] = comp_delta.to_numpy(dtype=float)

        comp_row = {"dataset_id": dataset_id}
        comp_row.update({theme: float(comp_delta[theme]) for theme in THEME_ORDER})
        composition_delta_rows.append(comp_row)

        composition_metric_rows.append(
            {
                "dataset_id": dataset_id,
                "low_n": int(len(low_meta)),
                "high_n": int(len(high_meta)),
                "delta_norm": float(np.linalg.norm(comp_delta.to_numpy(dtype=float))),
                "dominant_positive_theme": str(comp_delta.sort_values(ascending=False).index[0]),
                "dominant_negative_theme": str(comp_delta.sort_values(ascending=True).index[0]),
                "low_profile_entropy": float(-(low_comp[low_comp > 0] * np.log2(low_comp[low_comp > 0])).sum()) if (low_comp > 0).any() else 0.0,
                "high_profile_entropy": float(-(high_comp[high_comp > 0] * np.log2(high_comp[high_comp > 0])).sum()) if (high_comp > 0).any() else 0.0,
            }
        )

    dataset_latent_deltas = pd.concat(latent_rows, ignore_index=True) if latent_rows else pd.DataFrame(columns=["dataset_id", "dimension", "delta_value"])
    dataset_latent_deltas.to_csv(args.output_dir / "dataset_latent_deltas.csv", index=False)
    latent_metric_df = pd.DataFrame(latent_metric_rows)
    latent_metric_df.to_csv(args.output_dir / "dataset_latent_delta_metrics.csv", index=False)

    dataset_comp_deltas = pd.DataFrame(composition_delta_rows)
    dataset_comp_deltas.to_csv(args.output_dir / "dataset_composition_deltas.csv", index=False)
    pd.DataFrame(composition_metric_rows).to_csv(args.output_dir / "dataset_composition_delta_metrics.csv", index=False)

    print("[serum-delta] computing cross-dataset similarity")
    latent_cosine_df, _ = pairwise_similarity_matrix(latent_delta_vectors)
    comp_cosine_df, comp_corr_df = pairwise_similarity_matrix(composition_delta_vectors)
    latent_cosine_df.to_csv(args.output_dir / "cross_dataset_latent_delta_similarity.csv", index=False)
    comp_cosine_df.to_csv(args.output_dir / "cross_dataset_composition_delta_similarity.csv", index=False)

    consensus_rows = []
    if included_datasets:
        latent_stack = np.vstack([latent_delta_vectors[d] for d in included_datasets])
        comp_stack = np.vstack([composition_delta_vectors[d] for d in included_datasets])
        latent_mean = latent_stack.mean(axis=0)
        comp_mean = comp_stack.mean(axis=0)
        consensus_rows.extend(
            [
                {"metric": "included_dataset_count", "value": float(len(included_datasets))},
                {"metric": "latent_consensus_norm", "value": float(np.linalg.norm(latent_mean))},
                {"metric": "composition_consensus_norm", "value": float(np.linalg.norm(comp_mean))},
            ]
        )
        offdiag_latent = latent_cosine_df.set_index("dataset_id").to_numpy(dtype=float)
        offdiag_comp = comp_cosine_df.set_index("dataset_id").to_numpy(dtype=float)
        if len(included_datasets) > 1:
            mask = ~np.eye(len(included_datasets), dtype=bool)
            consensus_rows.extend(
                [
                    {"metric": "latent_mean_offdiag_cosine", "value": float(np.nanmean(offdiag_latent[mask]))},
                    {"metric": "composition_mean_offdiag_cosine", "value": float(np.nanmean(offdiag_comp[mask]))},
                    {"metric": "composition_mean_offdiag_pearson", "value": float(np.nanmean(comp_corr_df.set_index("dataset_id").to_numpy(dtype=float)[mask]))},
                ]
            )
    consensus_df = pd.DataFrame(consensus_rows)
    consensus_df.to_csv(args.output_dir / "cross_dataset_delta_consensus.csv", index=False)

    print("[serum-delta] computing theme consistency")
    theme_consensus = theme_consensus_table(dataset_comp_deltas)
    theme_consensus.to_csv(args.output_dir / "theme_consensus_table.csv", index=False)
    theme_report = [
        "# Theme Consensus Report",
        "",
        "Theme deltas are computed as high-state mean minus low-state mean within each included serum dataset, then compared across datasets.",
        "",
        "Legacy note:",
        "- `purine_metabolite_associated` is retained for compatibility with existing outputs, but it is a coarse legacy bucket and should later be split into purine-specific versus broader metabolite themes.",
        "",
        "Highest-consistency themes:",
    ]
    theme_report.extend([f"- {row.theme}: {row.dominant_direction}, consistency={row.sign_consistency:.2f}, mean_delta={row.mean_delta:.4f}" for row in theme_consensus.head(5).itertuples(index=False)])
    (args.output_dir / "theme_consensus_report.md").write_text("\n".join(theme_report) + "\n", encoding="utf-8")

    print("[serum-delta] generating figures")
    save_similarity_heatmap(latent_cosine_df, "Latent delta cosine similarity", args.output_dir / "latent_delta_similarity_heatmap.png")
    save_similarity_heatmap(comp_cosine_df, "Composition delta cosine similarity", args.output_dir / "composition_delta_similarity_heatmap.png")
    save_theme_delta_heatmap(dataset_comp_deltas, args.output_dir / "theme_delta_heatmap.png")
    save_theme_consensus_bars(theme_consensus, args.output_dir / "theme_consensus_bars.png")
    save_delta_magnitude_vs_stability(latent_metric_df, args.output_dir / "latent_delta_magnitude_vs_stability.png")
    if included_state_metadata:
        all_included_meta = pd.concat(included_state_metadata, ignore_index=False)
        save_centroid_shift_panels(embeddings, all_included_meta, included_datasets, args.output_dir / "dataset_latent_centroid_shift_panels.png")
    save_composition_profile_lines(dataset_comp_deltas, args.output_dir / "composition_delta_profile_lines.png")

    print("[serum-delta] writing summary and decision memo")
    if len(included_datasets) > 1:
        latent_align = float(consensus_df.loc[consensus_df["metric"] == "latent_mean_offdiag_cosine", "value"].iloc[0])
        comp_align = float(consensus_df.loc[consensus_df["metric"] == "composition_mean_offdiag_cosine", "value"].iloc[0])
    else:
        latent_align = float("nan")
        comp_align = float("nan")

    aligned_pairs = []
    for left in included_datasets:
        for right in included_datasets:
            if left >= right:
                continue
            latent_val = float(latent_cosine_df.set_index("dataset_id").loc[left, right])
            comp_val = float(comp_cosine_df.set_index("dataset_id").loc[left, right])
            aligned_pairs.append((left, right, latent_val, comp_val))

    summary_lines = [
        "# Serum Delta Summary",
        "",
        f"- included datasets: {', '.join(included_datasets) if included_datasets else 'none'}",
        f"- excluded datasets: {', '.join(mapping_df.loc[~mapping_df['include_in_delta'], 'dataset_id'].tolist()) if (~mapping_df['include_in_delta']).any() else 'none'}",
        f"- latent mean off-diagonal cosine: {latent_align:.4f}" if len(included_datasets) > 1 else "- latent mean off-diagonal cosine: not computable",
        f"- composition mean off-diagonal cosine: {comp_align:.4f}" if len(included_datasets) > 1 else "- composition mean off-diagonal cosine: not computable",
        "",
        "Readout:",
        "- This module asks whether serum disease biology aligns as a direction of change within datasets, not whether all serum datasets already occupy one shared manifold.",
        "- Only datasets with a defensible within-dataset control-versus-disease contrast are included.",
        "- The purine/metabolite bucket remains a legacy coarse theme and should not be over-interpreted.",
    ]
    (args.output_dir / "serum_delta_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    memo_lines = [
        "# Serum Delta Decision Memo",
        "",
        "1. Do serum disease/control differences move in a similar direction across datasets?",
        f"- Included datasets: {', '.join(included_datasets) if included_datasets else 'none'}",
    ]
    if aligned_pairs:
        memo_lines.extend(
            [f"- {left} vs {right}: latent cosine={latent_val:.4f}, composition cosine={comp_val:.4f}" for left, right, latent_val, comp_val in aligned_pairs]
        )
    else:
        memo_lines.append("- Not enough included datasets for pairwise alignment.")
    memo_lines.extend(
        [
            "",
            "2. Is there evidence for a shared serum stress manifold?",
            "- Use the latent delta alignment, not the raw global serum manifold, as the decision signal.",
            f"- Mean latent delta alignment = {latent_align:.4f}" if len(included_datasets) > 1 else "- Mean latent delta alignment unavailable",
            f"- Mean composition delta alignment = {comp_align:.4f}" if len(included_datasets) > 1 else "- Mean composition delta alignment unavailable",
            "",
            "3. Which datasets align with each other?",
        ]
    )
    if aligned_pairs:
        memo_lines.extend(
            [
                f"- {left} and {right} align {'weakly' if max(latent_val, comp_val) < 0.5 else 'moderately'}"
                for left, right, latent_val, comp_val in aligned_pairs
            ]
        )
    else:
        memo_lines.append("- none")
    memo_lines.extend(["", "4. Which datasets should be excluded from future serum-specific training?"])
    excluded_lines = [f"- {row.dataset_id}: {row.decision_reason}" for row in mapping_df[~mapping_df["include_in_delta"]].itertuples(index=False)]
    memo_lines.extend(excluded_lines if excluded_lines else ["- none"])
    memo_lines.extend(
        [
            "",
            "5. Is serum ready for a dedicated v8/v9 head, or still mainly a harmonization problem?",
            "- If only a small set of datasets show aligned deltas and the rest remain protocol-heavy, serum is still mainly a harmonization problem.",
            "- The legacy `purine_metabolite_associated` theme remains too coarse to be treated as a precise mechanistic readout.",
        ]
    )
    (args.output_dir / "serum_delta_decision_memo.md").write_text("\n".join(memo_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
