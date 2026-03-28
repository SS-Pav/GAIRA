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

    parser = argparse.ArgumentParser(description="Visualize learned spectral embeddings.")
    add_common_io_args(parser, default_run_name="embedding_v2", default_root=REMOTE_OUTPUT_ROOT)
    return parser.parse_args()


def compute_projection(embeddings: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import umap

        reducer = umap.UMAP(n_neighbors=20, min_dist=0.2, metric="cosine", random_state=7)
        coords = reducer.fit_transform(embeddings)
        return coords, "UMAP"
    except Exception:
        reducer = TSNE(n_components=2, perplexity=30, init="pca", random_state=7)
        return reducer.fit_transform(embeddings), "t-SNE"


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
    embeddings = np.load(output_dir / "embeddings.npy")
    metadata_df = pd.read_csv(output_dir / "metadata.csv")
    coords, method_name = compute_projection(embeddings)
    plot_df = metadata_df.copy()
    plot_df["dim1"] = coords[:, 0]
    plot_df["dim2"] = coords[:, 1]
    plot_df.to_csv(output_dir / "embedding_projection.csv", index=False)

    plot_projection(plot_df, "sample_type", f"{method_name} by sample type", output_dir / "umap_sample_type.png")
    plot_projection(plot_df, "dataset_id", f"{method_name} by dataset", output_dir / "umap_dataset.png")
    plot_projection(plot_df, "label_optional", f"{method_name} by class", output_dir / "umap_class.png")
    print(f"Saved embedding projection using {method_name}")


if __name__ == "__main__":
    main()
