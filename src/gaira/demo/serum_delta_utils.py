from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from gaira.demo.ev_analysis_utils import THEME_ORDER, compute_theme_profiles, normalize_rows
from gaira.demo.v8_analysis_utils import load_v7_common

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_serum_delta_analysis_v1")
DEFAULT_SERUM_STRESS_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/v8_serum_stress_analysis_v1")
DELTA_LOW = "low_stress_or_control"
DELTA_HIGH = "high_stress_or_disease"
DELTA_AMBIGUOUS = "intermediate_or_ambiguous"
SERUM_BIOLOGY_DATASETS = {"cca_hcc_lm_serum_sers", "covid_serum_raman"}
SERUM_DATASETS = {
    "cca_hcc_lm_serum_sers",
    "covid_serum_raman",
    "cspp_serum",
    "ergothioneine_serum",
    "serum_ag_colloids",
    "serum_protocol_comparison",
}


def harmonize_serum_state(row: pd.Series) -> tuple[str, str, str]:
    dataset_id = str(row.get("dataset_id", ""))
    label = str(row.get("label_optional", ""))
    subclass = str(row.get("subclass_label", ""))
    record_kind = str(row.get("record_kind", ""))
    if record_kind == "class_summary":
        return (DELTA_AMBIGUOUS, "low", "summary artifact retained for completeness")

    if dataset_id == "cca_hcc_lm_serum_sers":
        if label == "healthy_control":
            return (DELTA_LOW, "high", "explicit healthy control in liver serum cohort")
        if label in {"cca", "hcc", "lm"}:
            return (DELTA_HIGH, "high", "explicit hepatobiliary disease cohort label")

    if dataset_id == "covid_serum_raman":
        if label in {"healthy_control", "tube_control"}:
            return (DELTA_LOW, "high", "explicit healthy or tube-control serum state")
        if label == "covid_confirmed":
            return (DELTA_HIGH, "high", "explicit confirmed inflammatory infection cohort")
        if label == "suspected_case":
            return (DELTA_AMBIGUOUS, "medium", "suspected inflammatory state retained as ambiguous")

    if dataset_id == "serum_ag_colloids":
        if label in {"Serum", "SerumMerck", "SerumSigma"}:
            return (DELTA_LOW, "medium", "commercial or donor serum reference")
        return (DELTA_AMBIGUOUS, "low", "spiked serum or component reference, not a direct biological state")

    if dataset_id in {"cspp_serum", "serum_protocol_comparison", "ergothioneine_serum"}:
        if label in {"Bkg", "standard", "unprocessed"}:
            return (DELTA_LOW, "low", "process-control spectrum, kept as low-stress reference only")
        return (DELTA_AMBIGUOUS, "low", "protocol, spiking, calibration, or process-variation archive")

    if subclass == "released_zip_archive" and label == "healthy_control":
        return (DELTA_LOW, "medium", "healthy serum label")
    return (DELTA_AMBIGUOUS, "low", "no defensible broad serum stress mapping")


def load_serum_delta_inputs(
    *,
    serum_stress_dir: Path = DEFAULT_SERUM_STRESS_DIR,
) -> dict[str, pd.DataFrame | np.ndarray]:
    common = load_v7_common()
    metadata = common["metadata"].copy()  # type: ignore[assignment]
    serum = metadata[(metadata["sample_type"] == "serum") & (metadata["dataset_id"].isin(SERUM_DATASETS))].copy()
    harmonized = serum.apply(harmonize_serum_state, axis=1, result_type="expand")
    harmonized.columns = ["delta_state", "state_confidence", "state_notes"]
    state_table = pd.concat([serum.reset_index(drop=True), harmonized.reset_index(drop=True)], axis=1)
    state_table["sample_key"] = state_table["sample_key"].astype(str)
    return {
        **common,
        "state_table": state_table,
    }


def dataset_mapping_audit(state_table: pd.DataFrame, *, min_group_size: int = 40) -> pd.DataFrame:
    rows = []
    grouped = state_table.groupby("dataset_id", sort=True)
    for dataset_id, group in grouped:
        counts = group["delta_state"].value_counts()
        low_n = int(counts.get(DELTA_LOW, 0))
        high_n = int(counts.get(DELTA_HIGH, 0))
        amb_n = int(counts.get(DELTA_AMBIGUOUS, 0))
        biologically_defensible = dataset_id in SERUM_BIOLOGY_DATASETS
        include = biologically_defensible and low_n >= min_group_size and high_n >= min_group_size
        if not biologically_defensible:
            reason = "excluded: protocol, spiking, calibration, or non-disease serum reference"
        elif low_n < min_group_size or high_n < min_group_size:
            reason = "excluded: low/high groups too small for stable delta estimate"
        else:
            reason = "included: explicit within-dataset control-versus-disease or inflammatory contrast"
        rows.append(
            {
                "dataset_id": dataset_id,
                "low_stress_or_control_n": low_n,
                "high_stress_or_disease_n": high_n,
                "intermediate_or_ambiguous_n": amb_n,
                "biologically_defensible": biologically_defensible,
                "include_in_delta": include,
                "decision_reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["include_in_delta", "dataset_id"], ascending=[False, True]).reset_index(drop=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def bootstrap_delta_stability(
    low_values: np.ndarray,
    high_values: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = 100,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    full_delta = high_values.mean(axis=0) - low_values.mean(axis=0)
    full_norm = float(np.linalg.norm(full_delta))
    if full_norm <= 1e-12:
        return {
            "delta_norm": 0.0,
            "bootstrap_mean_cosine": float("nan"),
            "bootstrap_std_cosine": float("nan"),
            "bootstrap_min_cosine": float("nan"),
        }
    cosines = []
    for _ in range(n_bootstrap):
        low_idx = rng.integers(0, len(low_values), size=len(low_values))
        high_idx = rng.integers(0, len(high_values), size=len(high_values))
        delta = high_values[high_idx].mean(axis=0) - low_values[low_idx].mean(axis=0)
        cosines.append(cosine_similarity(full_delta, delta))
    cosines_arr = np.asarray(cosines, dtype=float)
    return {
        "delta_norm": full_norm,
        "bootstrap_mean_cosine": float(np.nanmean(cosines_arr)),
        "bootstrap_std_cosine": float(np.nanstd(cosines_arr)),
        "bootstrap_min_cosine": float(np.nanmin(cosines_arr)),
    }


def pairwise_similarity_matrix(vectors: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = list(vectors)
    cosine_rows = []
    corr_rows = []
    for left in labels:
        cosine_row = {"dataset_id": left}
        corr_row = {"dataset_id": left}
        for right in labels:
            cosine_row[right] = cosine_similarity(vectors[left], vectors[right])
            if len(vectors[left]) > 1 and len(vectors[right]) > 1:
                corr_row[right] = float(pd.Series(vectors[left]).corr(pd.Series(vectors[right]), method="pearson"))
            else:
                corr_row[right] = float("nan")
        cosine_rows.append(cosine_row)
        corr_rows.append(corr_row)
    return pd.DataFrame(cosine_rows), pd.DataFrame(corr_rows)


def theme_consensus_table(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_datasets = max(int(delta_df["dataset_id"].nunique()), 1)
    for theme in THEME_ORDER:
        values = delta_df[["dataset_id", theme]].dropna()
        positive = int((values[theme] > 0).sum())
        negative = int((values[theme] < 0).sum())
        zeroish = int((values[theme].abs() <= 1e-9).sum())
        sign_consistency = max(positive, negative) / n_datasets
        rows.append(
            {
                "theme": theme,
                "increase_count": positive,
                "decrease_count": negative,
                "near_zero_count": zeroish,
                "mean_delta": float(values[theme].mean()),
                "abs_mean_delta": float(values[theme].abs().mean()),
                "sign_consistency": float(sign_consistency),
                "dominant_direction": "increase" if positive >= negative else "decrease",
                "legacy_note": "legacy coarse bucket" if theme == "purine_metabolite_associated" else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["sign_consistency", "abs_mean_delta"], ascending=[False, False]).reset_index(drop=True)


def save_state_count_figure(mapping_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = mapping_df.melt(
        id_vars=["dataset_id"],
        value_vars=["low_stress_or_control_n", "high_stress_or_disease_n", "intermediate_or_ambiguous_n"],
        var_name="state_group",
        value_name="count",
    )
    plt.figure(figsize=(9.2, 5.8))
    ax = sns.barplot(data=plot_df, x="dataset_id", y="count", hue="state_group", palette="deep")
    ax.set_title("Serum dataset state counts")
    ax.set_xlabel("dataset")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, title="state group")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_similarity_heatmap(df: pd.DataFrame, title: str, output_path: Path, *, cmap: str = "vlag") -> None:
    table = df.set_index("dataset_id")
    plt.figure(figsize=(6.8, 5.8))
    ax = sns.heatmap(table, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_theme_delta_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    table = df.set_index("dataset_id")[THEME_ORDER]
    plt.figure(figsize=(9.0, 4.8))
    ax = sns.heatmap(table, annot=True, fmt=".2f", cmap="vlag", center=0.0)
    ax.set_title("Serum composition delta by dataset")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_theme_consensus_bars(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9.0, 5.8))
    ax = sns.barplot(data=df, x="theme", y="sign_consistency", hue="dominant_direction", dodge=False, palette="deep")
    ax.set_title("Theme consistency across serum delta datasets")
    ax.set_xlabel("theme")
    ax.set_ylabel("sign consistency")
    ax.tick_params(axis="x", rotation=28)
    ax.legend(frameon=False, title="dominant direction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_delta_magnitude_vs_stability(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7.2, 5.6))
    ax = sns.scatterplot(
        data=df,
        x="delta_norm",
        y="bootstrap_mean_cosine",
        hue="dataset_id",
        size="total_n",
        sizes=(90, 260),
        palette="deep",
    )
    ax.set_title("Latent delta magnitude versus bootstrap stability")
    ax.set_xlabel("latent delta norm")
    ax.set_ylabel("bootstrap mean cosine")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_centroid_shift_panels(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    included_datasets: list[str],
    output_path: Path,
) -> None:
    n_cols = 2
    n_rows = int(np.ceil(len(included_datasets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, 4.5 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, dataset_id in zip(axes.ravel(), included_datasets, strict=False):
        group = metadata[metadata["dataset_id"] == dataset_id].copy()
        coords = PCA(n_components=2, random_state=7).fit_transform(normalize_rows(embeddings[group.index.to_numpy()]))
        group["dim1"] = coords[:, 0]
        group["dim2"] = coords[:, 1]
        for state, color in [(DELTA_LOW, "#3e6ea1"), (DELTA_HIGH, "#c24d67")]:
            subset = group[group["delta_state"] == state]
            ax.scatter(subset["dim1"], subset["dim2"], s=14, alpha=0.32, color=color, label=state)
            if not subset.empty:
                center = subset[["dim1", "dim2"]].mean()
                ax.scatter(center["dim1"], center["dim2"], s=140, color=color, edgecolor="black", linewidth=0.8)
        low_center = group[group["delta_state"] == DELTA_LOW][["dim1", "dim2"]].mean()
        high_center = group[group["delta_state"] == DELTA_HIGH][["dim1", "dim2"]].mean()
        if low_center.notna().all() and high_center.notna().all():
            ax.annotate(
                "",
                xy=(high_center["dim1"], high_center["dim2"]),
                xytext=(low_center["dim1"], low_center["dim2"]),
                arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#222222"},
            )
        ax.set_title(dataset_id)
        ax.legend(frameon=False, fontsize=8)
        ax.axis("on")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle("Serum low-to-high centroid shifts on local PCA projections", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_composition_profile_lines(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.melt(id_vars=["dataset_id"], value_vars=THEME_ORDER, var_name="theme", value_name="delta")
    plt.figure(figsize=(10.2, 5.8))
    ax = sns.lineplot(data=plot_df, x="theme", y="delta", hue="dataset_id", marker="o", palette="deep")
    ax.axhline(0.0, color="#666666", lw=1.0, ls="--")
    ax.set_title("Serum composition delta profiles")
    ax.set_xlabel("theme")
    ax.set_ylabel("high minus low composition weight")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
