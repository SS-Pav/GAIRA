#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gaira.demo.small2023_branch_utils import load_branch_run, mode_labels
from gaira.demo.v8_analysis_utils import V7_GROUNDING_DIR, save_heatmap, save_scatter
from gaira.demo.v8_theme_utils import MASTER_THEME_ORDER, compute_split_theme_profiles, split_grounding_theme_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small2023 branch composition analysis.")
    parser.add_argument("--mode", choices=["cellline", "mixture"], required=True)
    parser.add_argument("--branch-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-grounding", type=int, default=12)
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def encode_grounding_embeddings(run_dir: Path, dataset_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    from gaira.embedding.model import RamanEncoder

    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu")
    dataset = np.load(dataset_path, allow_pickle=True)
    mask = dataset["sample_types"].astype(str) == "grounding"
    X = torch.from_numpy(dataset["X"][mask].astype(np.float32))
    metadata = pd.DataFrame(
        {
            "sample_key": dataset["sample_keys"][mask].astype(str),
            "dataset_id": dataset["dataset_ids"][mask].astype(str),
            "label_optional": dataset["labels_optional"][mask].astype(str),
            "semantic_group": dataset["semantic_groups"][mask].astype(str) if "semantic_groups" in dataset.files else np.asarray([""] * int(mask.sum()), dtype=object),
        }
    )
    model = RamanEncoder(input_len=int(checkpoint["input_len"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        embeddings = model(X).cpu().numpy().astype(np.float32)
    return embeddings, metadata


def bootstrap_ci(values: np.ndarray, *, reps: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(reps):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(sample.mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta, branch_embeddings, run_config = load_branch_run(args.branch_run_dir, mode=args.mode)
    meta["class_label"] = meta["branch_primary_label"].astype(str)
    meta["probe_label"] = meta["branch_secondary_label"].astype(str)
    dataset_path = Path(run_config["dataset_path"]).expanduser().resolve()

    grounding_theme_table = pd.read_csv(V7_GROUNDING_DIR / "grounding_theme_table.csv")
    split_grounding = split_grounding_theme_table(grounding_theme_table)
    grounding_embeddings, grounding_meta = encode_grounding_embeddings(args.branch_run_dir, dataset_path)
    grounding_index = grounding_meta["sample_key"].astype(str).isin(split_grounding["sample_key"].astype(str))
    grounding_embeddings = grounding_embeddings[grounding_index.to_numpy()]
    split_grounding = split_grounding[split_grounding["sample_key"].astype(str).isin(grounding_meta.loc[grounding_index, "sample_key"].astype(str))].copy()
    split_grounding = split_grounding.reset_index(drop=True)

    profiles = compute_split_theme_profiles(
        branch_embeddings,
        grounding_embeddings,
        split_grounding,
        top_k=args.top_k_grounding,
    )
    profile_df = meta[["sample_key", "class_label", "probe_label"]].reset_index(drop=True).join(profiles)
    profile_df.to_csv(args.output_dir / "per_spectrum_composition_profiles.csv", index=False)

    class_rows = []
    for class_label, group in profile_df.groupby("class_label", sort=True):
        profile = group[MASTER_THEME_ORDER].mean()
        ordered = profile.sort_values(ascending=False)
        row = {
            "class_label": class_label,
            "n_samples": int(len(group)),
            "top_theme": str(ordered.index[0]),
            "secondary_theme": str(ordered.index[1]) if len(ordered) > 1 else "none",
        }
        for theme in MASTER_THEME_ORDER:
            row[theme] = float(profile[theme])
            low, high = bootstrap_ci(group[theme].to_numpy(dtype=float), reps=args.bootstrap_reps, seed=args.seed)
            row[f"{theme}_ci_low"] = low
            row[f"{theme}_ci_high"] = high
        class_rows.append(row)
    class_summary = pd.DataFrame(class_rows).sort_values("class_label").reset_index(drop=True)
    class_summary.to_csv(args.output_dir / "class_composition_summary.csv", index=False)

    heatmap_df = class_summary.set_index("class_label")[MASTER_THEME_ORDER]
    save_heatmap(heatmap_df, title=f"{args.mode} class composition heatmap", output_path=args.output_dir / "class_composition_heatmap.png")

    variance_order = profile_df[MASTER_THEME_ORDER].var().sort_values(ascending=False).index.tolist()
    x_theme = variance_order[0]
    y_theme = variance_order[1] if len(variance_order) > 1 else MASTER_THEME_ORDER[1]
    scatter_df = class_summary[["class_label", x_theme, y_theme, "n_samples"]].copy()
    save_scatter(
        scatter_df,
        x=x_theme,
        y=y_theme,
        hue="class_label",
        style=None,
        size="n_samples",
        title=f"{args.mode} class composition scatter",
        output_path=args.output_dir / "class_composition_scatter.png",
    )

    distances = []
    for left in class_summary["class_label"]:
        left_row = class_summary[class_summary["class_label"] == left].iloc[0][MASTER_THEME_ORDER].to_numpy(dtype=float)
        for right in class_summary["class_label"]:
            right_row = class_summary[class_summary["class_label"] == right].iloc[0][MASTER_THEME_ORDER].to_numpy(dtype=float)
            distances.append({"class_left": left, "class_right": right, "distance": float(np.linalg.norm(left_row - right_row))})
    distance_df = pd.DataFrame(distances)
    distance_df.to_csv(args.output_dir / "class_pairwise_composition_distance.csv", index=False)
    save_heatmap(
        distance_df.pivot(index="class_left", columns="class_right", values="distance"),
        title=f"{args.mode} class composition distance heatmap",
        output_path=args.output_dir / "class_composition_distance_heatmap.png",
        cmap="rocket",
    )

    lines = [
        f"# small2023 {args.mode} Composition Summary",
        "",
        f"- Classes analyzed: {', '.join(mode_labels(args.mode))}",
        "- Composition is broad grounding-derived theme support, not molecule certainty.",
        "- `serum_matrix_associated` is retained only as a caveat/support channel and should not drive the headline interpretation.",
        "",
        "The class heatmap and distance matrix show whether the specialized branch produces chemically interpretable class tendencies rather than only geometric separation.",
    ]
    (args.output_dir / "composition_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
