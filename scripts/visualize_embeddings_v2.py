from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 220,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
sns.set_theme(style="whitegrid", context="talk")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.runtime import REMOTE_OUTPUT_ROOT, add_common_io_args

    parser = argparse.ArgumentParser(description="Sampled v2 embedding visualization.")
    add_common_io_args(parser, default_run_name="embedding_v2", default_root=REMOTE_OUTPUT_ROOT)
    parser.add_argument("--report-dir", default=None, help="Directory containing sample_manifest.csv and target outputs.")
    parser.add_argument("--projection-backend", choices=["auto", "umap", "tsne", "pca"], default="auto")
    return parser.parse_args()


def compute_projection(embeddings: np.ndarray, backend: str) -> tuple[np.ndarray, str]:
    if backend in {"auto", "umap"}:
        try:
            import umap

            reducer = umap.UMAP(n_neighbors=20, min_dist=0.2, metric="cosine", random_state=7)
            coords = reducer.fit_transform(embeddings)
            return coords, "UMAP"
        except Exception:
            if backend == "umap":
                raise
    if backend in {"auto", "tsne"}:
        reducer = TSNE(n_components=2, perplexity=30, init="pca", random_state=7)
        return reducer.fit_transform(embeddings), "t-SNE"
    if backend == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=7)
        return reducer.fit_transform(embeddings), "PCA"
    raise ValueError(f"Unsupported projection backend: {backend}")


def plot_projection(df: pd.DataFrame, color_column: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    categories = df[color_column].fillna("").astype(str)
    unique_values = categories.value_counts().index.tolist()[:12]
    plot_df = df.copy()
    plot_df[color_column] = plot_df[color_column].where(plot_df[color_column].isin(unique_values), "other")
    sns.scatterplot(
        data=plot_df,
        x="dim1",
        y="dim2",
        hue=color_column,
        s=28,
        alpha=0.8,
        linewidth=0.0,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title=color_column)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.runtime import resolve_output_dir

    args = parse_args()
    output_dir = resolve_output_dir(args)
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else output_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(output_dir / "embeddings.npy")
    metadata_df = pd.read_csv(output_dir / "metadata.csv")
    sample_manifest = pd.read_csv(report_dir / "sample_manifest.csv")
    sample_indices = sample_manifest["original_index"].astype(int).to_numpy()
    sampled_embeddings = embeddings[sample_indices]
    sampled_df = metadata_df.loc[sample_indices].copy().reset_index(drop=True)
    coords, method_name = compute_projection(sampled_embeddings, args.projection_backend)
    sampled_df["dim1"] = coords[:, 0]
    sampled_df["dim2"] = coords[:, 1]
    sampled_df.to_csv(report_dir / "embedding_projection_v2.csv", index=False)

    suffix = f"{method_name} sampled/stratified"
    plot_projection(sampled_df, "sample_type", f"{suffix} by sample type", report_dir / "umap_sample_type.png")
    plot_projection(sampled_df, "dataset_id", f"{suffix} by dataset", report_dir / "umap_dataset.png")
    plot_projection(sampled_df, "label_optional", f"{suffix} by class", report_dir / "umap_class.png")
    print(f"Saved sampled embedding projection using {method_name}")


if __name__ == "__main__":
    main()
