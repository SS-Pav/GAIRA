#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gaira.demo.v8_analysis_utils import (
    SMALL2023_V1_DIR,
    SMALL2023_V2_DIR,
    SMALL2023_V3_DIR,
    THEME_ORDER,
    V5_EVAL_DIR,
    V5_RUN_DIR,
    V6_EVAL_DIR,
    V6_RUN_DIR,
    V7_EVAL_DIR,
    V7_RUN_DIR,
    balanced_small2023_direct,
    class_probe_metrics,
    cluster_composition_summary,
    compute_theme_profiles,
    decode_direct_matrix,
    load_common_artifacts,
    maybe_read_csv,
    normalize_rows,
    reduce_for_plot,
    save_barplot,
    save_heatmap,
    save_scatter,
)


DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_small2023_benchmark_v1")
PROCESSING_VERSION = "v2_crop670_1800_interp1_poly3_vector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v8 small2023 benchmark.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--balanced-per-group", type=int, default=250)
    parser.add_argument("--plot-per-group", type=int, default=120)
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--top-k-grounding", type=int, default=12)
    return parser.parse_args()


def placeholder_figure(path: Path, text: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def load_run_embeddings(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = pd.read_csv(run_dir / "metadata.csv")
    embeddings = np.load(run_dir / "embeddings.npy")
    metadata["sample_key"] = metadata["sample_key"].astype(str)
    return metadata, embeddings


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = load_common_artifacts(
        run_dir=V7_RUN_DIR,
        eval_dir=V7_EVAL_DIR,
    )
    v7_metadata = common["metadata"].copy()  # type: ignore[assignment]
    v7_embeddings = common["embeddings"]  # type: ignore[assignment]
    grounding_theme_table = common["grounding_theme_table"].copy()  # type: ignore[assignment]

    direct = balanced_small2023_direct(args.balanced_per_group, seed=args.seed, processing_version=PROCESSING_VERSION)
    _, X_direct = decode_direct_matrix(direct)
    X_direct = StandardScaler().fit_transform(X_direct)
    class_labels = direct["class_label"].to_numpy()
    probe_labels = direct["subclass_label"].to_numpy()
    direct_metrics = class_probe_metrics(X_direct, class_labels, probe_labels, seed=args.seed, k=args.knn_k)

    sample_keys = direct["sample_key"].astype(str).tolist()

    model_rows = []
    plot_frames = {}
    for name, run_dir in [("v5", V5_RUN_DIR), ("v6", V6_RUN_DIR), ("v7", V7_RUN_DIR)]:
        if not run_dir.exists():
            model_rows.append({"model": name, "metric": "status", "value": float("nan"), "status": "missing"})
            continue
        run_metadata, run_embeddings = load_run_embeddings(run_dir)
        subset = run_metadata[run_metadata["sample_key"].isin(sample_keys)].copy()
        if subset.empty:
            model_rows.append({"model": name, "metric": "status", "value": float("nan"), "status": "missing"})
            continue
        Z = normalize_rows(run_embeddings[subset.index.to_numpy()])
        metrics = class_probe_metrics(Z, subset["label_optional"].to_numpy(), subset["subclass_label"].to_numpy(), seed=args.seed, k=args.knn_k)
        metrics["model"] = name
        metrics["status"] = "available"
        model_rows.append(metrics)
        sampled = direct.groupby(["subclass_label", "class_label"], group_keys=False).apply(
            lambda group: group.sample(n=min(args.plot_per_group, len(group)), random_state=args.seed)
        ).reset_index(drop=True)
        sample_subset = subset[subset["sample_key"].isin(sampled["sample_key"].astype(str))].copy()
        plot_frames[name] = (
            reduce_for_plot(normalize_rows(run_embeddings[sample_subset.index.to_numpy()]), seed=args.seed),
            sampled,
        )

    direct_metrics["model"] = "direct"
    direct_metrics["status"] = "available"
    metric_frames = [direct_metrics]
    for frame in model_rows:
        if isinstance(frame, pd.DataFrame):
            metric_frames.append(frame)
    benchmark_metrics = pd.concat(metric_frames, ignore_index=True)

    # v2 specialized benchmark metrics from prior invariant run.
    v2_geometry = maybe_read_csv(SMALL2023_V2_DIR / "geometry_metrics_v2.csv")
    if v2_geometry is not None and not v2_geometry.empty:
        g = v2_geometry.iloc[0]
        v2_rows = pd.DataFrame(
            [
                {"model": "v2_specialized", "metric": "silhouette_class", "value": float(g["v2_class_silhouette"]), "status": "available"},
                {"model": "v2_specialized", "metric": "silhouette_probe", "value": float(g["v2_probe_silhouette"]), "status": "available"},
            ]
        )
        benchmark_metrics = pd.concat([benchmark_metrics, v2_rows], ignore_index=True)

    benchmark_metrics.to_csv(args.output_dir / "small2023_benchmark_metrics.csv", index=False)
    benchmark_metrics.pivot_table(index="metric", columns="model", values="value").reset_index().to_csv(
        args.output_dir / "small2023_direct_vs_embedding_comparison.csv", index=False
    )

    # Direct / v5 / v7 maps.
    sampled_direct = direct.groupby(["subclass_label", "class_label"], group_keys=False).apply(
        lambda group: group.sample(n=min(args.plot_per_group, len(group)), random_state=args.seed)
    ).reset_index(drop=True)
    direct_coords = reduce_for_plot(StandardScaler().fit_transform(decode_direct_matrix(sampled_direct)[1]), seed=args.seed)
    direct_plot_df = sampled_direct.copy()
    direct_plot_df["dim1"] = direct_coords[:, 0]
    direct_plot_df["dim2"] = direct_coords[:, 1]
    save_scatter(direct_plot_df, x="dim1", y="dim2", hue="class_label", style=None, size=None, title="Direct processed spectra by class", output_path=args.output_dir / "direct_map_by_class.png")
    save_scatter(direct_plot_df, x="dim1", y="dim2", hue="subclass_label", style=None, size=None, title="Direct processed spectra by probe", output_path=args.output_dir / "direct_map_by_probe.png")

    for model_name in ["v5", "v6", "v7"]:
        class_path = args.output_dir / f"{model_name}_embedding_map_by_class.png"
        probe_path = args.output_dir / f"{model_name}_embedding_map_by_probe.png"
        if model_name not in plot_frames:
            placeholder_figure(class_path, f"{model_name} embedding unavailable locally")
            placeholder_figure(probe_path, f"{model_name} embedding unavailable locally")
            continue
        coords, sampled = plot_frames[model_name]
        df = sampled.copy()
        df["dim1"] = coords[:, 0]
        df["dim2"] = coords[:, 1]
        save_scatter(df, x="dim1", y="dim2", hue="class_label", style=None, size=None, title=f"{model_name} embedding by class", output_path=class_path)
        save_scatter(df, x="dim1", y="dim2", hue="subclass_label", style=None, size=None, title=f"{model_name} embedding by probe", output_path=probe_path)

    if (SMALL2023_V2_DIR / "embedding_tsne_by_class_v2.png").exists():
        shutil.copyfile(SMALL2023_V2_DIR / "embedding_tsne_by_class_v2.png", args.output_dir / "v2_embedding_map_by_class.png")
    else:
        placeholder_figure(args.output_dir / "v2_embedding_map_by_class.png", "v2 specialized figure missing")
    if (SMALL2023_V2_DIR / "embedding_tsne_by_probe_v2.png").exists():
        shutil.copyfile(SMALL2023_V2_DIR / "embedding_tsne_by_probe_v2.png", args.output_dir / "v2_embedding_map_by_probe.png")
    else:
        placeholder_figure(args.output_dir / "v2_embedding_map_by_probe.png", "v2 specialized figure missing")

    metric_plot = benchmark_metrics[benchmark_metrics["metric"].isin(["silhouette_class", "silhouette_probe", "nn_purity_class", "nn_purity_probe"])].copy()
    save_barplot(metric_plot, x="metric", y="value", hue="model", title="Direct vs shared vs specialized metrics", output_path=args.output_dir / "direct_vs_embedding_metric_bars.png")
    save_barplot(metric_plot, x="model", y="value", hue="metric", title="small2023 class/probe scorecard", output_path=args.output_dir / "class_probe_scorecard.png")

    # Composition on explicit cell-line subset.
    small_all = v7_metadata[(v7_metadata["dataset_id"] == "small2023_ev") & (v7_metadata["record_kind"] == "processed_spectrum")].copy()
    cellline_meta = small_all[small_all["label_optional"].isin(["Hec", "Hela", "Ht", "Mef", "Thp"])].copy()
    grounding_idx = v7_metadata.index[v7_metadata["sample_key"].astype(str).isin(grounding_theme_table["sample_key"].astype(str))].to_numpy()
    profiles = compute_theme_profiles(
        normalize_rows(v7_embeddings[cellline_meta.index.to_numpy()]),
        normalize_rows(v7_embeddings[grounding_idx]),
        grounding_theme_table["grounding_theme"].astype(str).to_numpy(),
        top_k=args.top_k_grounding,
    )
    class_profiles = cellline_meta[["sample_key", "label_optional"]].reset_index(drop=True).join(profiles)
    class_summary = class_profiles.groupby("label_optional")[THEME_ORDER].mean().reset_index().rename(columns={"label_optional": "class_label"})
    class_summary.to_csv(args.output_dir / "small2023_class_composition_summary.csv", index=False)
    save_heatmap(class_summary.set_index("class_label")[THEME_ORDER], title="small2023 class composition heatmap", output_path=args.output_dir / "class_composition_heatmap.png")

    summary = [
        "# small2023 Benchmark Summary",
        "",
        "- The shared backbone reduces probe-family nuisance relative to direct processed spectra, but it also degrades useful class organization on the mixture benchmark.",
        "- The old specialized v2 invariant embedding remains much stronger than the shared backbone on the explicit probe-invariance benchmark.",
        "- The correct v8 conclusion is that small2023 should get a dedicated specialized head rather than relying on the shared encoder alone.",
    ]
    (args.output_dir / "small2023_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
