from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaira.autoresearch_storage import (
    DEFAULT_STORAGE_CONFIG_PATH,
    initialize_autoresearch_sprint,
    load_autoresearch_storage_config,
)
from gaira.demo.autoresearch_pass5_utils import build_bsv_profiles_pass5
from gaira.demo.gaira_experiment_runner_utils import load_architecture_registries, load_query_dataframe
from gaira.demo.gaira_pilot_utils import ALL_AXES, build_pdf_report
from gaira.demo.raw_bsv_pilot_utils import decode_and_align
from scripts.run_gaira_pilot3_shine_ev_sers_fullspectra import (
    ARCH_DIR,
    CLUSTER_COLORS,
    CONFIG_SPEC,
    DAY_COLORS,
    FAMILY_COLORS,
    FAMILY_ORDER,
    FIXED_RADAR_AXES,
    PHASE1_DIR,
    ROOT,
    _cohort_delta,
    _compound_to_family,
    _ensure_fixed_axes,
    _extract_sample_id,
    _fit_pca,
    _parse_concentration,
    _plot_family_bars,
    _plot_radar_grid,
    _plot_scatter,
    _prepare_grounding_and_mapping,
    _resolve_alias,
    _trajectory_index,
)


SPRINT_SUBDIR = "pilot3_shine_day2_controlanchored"
SUBSET_ALIAS = "shine_ev_stress"
DAY2_CLASS_ORDER = ["D2_C0", "D2_C10", "D2_C20", "D2_C40"]
CONCENTRATION_ORDER = [0, 10, 20, 40]
PREVIOUS_SAMPLEMEAN_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot3_shine_ev_sers"
)
PREVIOUS_FULLSPECTRA_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot3_shine_ev_sers_fullspectra"
)


def _prepare_day2_query_df(query_df: pd.DataFrame) -> pd.DataFrame:
    work = query_df.reset_index(drop=True).copy()
    work["sample_id"] = work.apply(_extract_sample_id, axis=1)
    work["trajectory_concentration"] = work["class_label"].astype(str).map(_parse_concentration)
    work = work[work["class_label"].astype(str).isin(DAY2_CLASS_ORDER)].copy().reset_index(drop=True)
    work["day_label"] = "D2"
    work["trajectory_index"] = [
        _trajectory_index("D2", concentration)
        for concentration in work["trajectory_concentration"].astype(int).tolist()
    ]
    work["n_scans"] = 1
    return work


def _axes_present(df: pd.DataFrame) -> list[str]:
    return [axis for axis in ALL_AXES if axis in df.columns]


def _fit_pca_df(df: pd.DataFrame, axes: list[str], *, scale: bool) -> pd.DataFrame:
    scores, explained = _fit_pca(df[axes].to_numpy(dtype=float), scale=scale)
    out = df[
        [
            "sample_key",
            "sample_id",
            "class_label",
            "trajectory_concentration",
            "trajectory_index",
        ]
    ].copy()
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1]
    out["pc1_explained_ratio"] = float(explained[0]) if len(explained) > 0 else 1.0
    out["pc2_explained_ratio"] = float(explained[1]) if len(explained) > 1 else 0.0
    return out


def _family_fingerprint_from_retrieval(
    retrieval_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    work = retrieval_df.copy()
    work["family"] = work["reference_compound_label"].astype(str).map(_compound_to_family)
    grouped = (
        work.groupby(["query_sample_key", "query_class_label", "family"], as_index=False)["support_weight"]
        .sum()
        .rename(
            columns={
                "query_sample_key": "sample_key",
                "query_class_label": "class_label",
                "support_weight": "family_support",
            }
        )
    )
    sample_map = meta_df.set_index("sample_key")[
        ["sample_id", "trajectory_concentration", "trajectory_index"]
    ]
    rows: list[dict[str, object]] = []
    for sample_key, sub in grouped.groupby("sample_key", sort=True):
        total = float(sub["family_support"].sum())
        meta = sample_map.loc[str(sample_key)]
        existing = {str(x) for x in sub["family"].tolist()}
        for family in FAMILY_ORDER:
            value = 0.0
            if family in existing:
                value = float(sub[sub["family"].astype(str) == family]["family_support"].iloc[0])
            rows.append(
                {
                    "sample_key": str(sample_key),
                    "sample_id": str(meta["sample_id"]),
                    "class_label": str(sub["class_label"].iloc[0]),
                    "trajectory_concentration": int(meta["trajectory_concentration"]),
                    "trajectory_index": int(meta["trajectory_index"]),
                    "family": family,
                    "family_fraction": (value / total) if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _class_family_means(family_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        family_df.groupby(["class_label", "family"], as_index=False)["family_fraction"].mean()
    )
    rows: list[dict[str, object]] = []
    for class_label in DAY2_CLASS_ORDER:
        sub = grouped[grouped["class_label"].astype(str) == class_label].copy()
        existing = {str(x) for x in sub["family"].tolist()}
        total = float(sub["family_fraction"].sum())
        for family in FAMILY_ORDER:
            value = 0.0
            if family in existing:
                value = float(sub[sub["family"].astype(str) == family]["family_fraction"].iloc[0])
            rows.append(
                {
                    "class_label": class_label,
                    "family": family,
                    "family_fraction": (value / total) if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _mean_by_class(df: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    return (
        df.groupby("class_label", as_index=False)[axes]
        .mean()
        .assign(
            trajectory_concentration=lambda frame: frame["class_label"].astype(str).map(_parse_concentration)
        )
        .sort_values("trajectory_concentration")
        .reset_index(drop=True)
    )


def _control_delta(df: pd.DataFrame, axes: list[str], *, control_label: str) -> pd.DataFrame:
    control_mean = (
        df[df["class_label"].astype(str) == str(control_label)][axes]
        .mean(axis=0)
        .astype(float)
    )
    out = df.copy()
    for axis in axes:
        out[axis] = df[axis].to_numpy(dtype=float) - float(control_mean[axis])
    return out


def _mean_within_variance(df: pd.DataFrame, axes: list[str]) -> float:
    rows = []
    for class_label in DAY2_CLASS_ORDER:
        sub = df[df["class_label"].astype(str) == class_label].copy()
        if len(sub) <= 1:
            rows.append(0.0)
            continue
        rows.append(float(sub[axes].var(ddof=1).mean()))
    return float(np.mean(rows)) if rows else 0.0


def _mean_between_distance(class_mean_df: pd.DataFrame, axes: list[str]) -> float:
    arr = class_mean_df.sort_values("trajectory_concentration")[axes].to_numpy(dtype=float)
    distances = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            distances.append(float(np.linalg.norm(arr[i] - arr[j])))
    return float(np.mean(distances)) if distances else 0.0


def _nearest_neighbor_purity(df: pd.DataFrame, axes: list[str], *, n_neighbors: int = 5) -> float:
    X = StandardScaler().fit_transform(df[axes].to_numpy(dtype=float))
    labels = df["class_label"].astype(str).to_numpy()
    n_use = min(n_neighbors + 1, len(df))
    nn = NearestNeighbors(n_neighbors=n_use)
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)
    purities = []
    for i in range(len(df)):
        neigh = [j for j in indices[i] if j != i][:n_neighbors]
        purities.append(float(np.mean(labels[neigh] == labels[i])) if neigh else 0.0)
    return float(np.mean(purities)) if purities else 0.0


def _representation_separation_metrics(
    df: pd.DataFrame,
    axes: list[str],
    *,
    representation: str,
) -> pd.DataFrame:
    silhouette = float(
        silhouette_score(StandardScaler().fit_transform(df[axes]), df["class_label"].astype(str))
    )
    class_mean = _mean_by_class(df, axes)
    return pd.DataFrame(
        [
            {
                "representation": representation,
                "silhouette_by_concentration": silhouette,
                "mean_centroid_distance": _mean_between_distance(class_mean, axes),
                "mean_within_class_variance": _mean_within_variance(df, axes),
                "mean_between_class_distance": _mean_between_distance(class_mean, axes),
                "nearest_neighbor_purity": _nearest_neighbor_purity(df, axes),
            }
        ]
    )


def _entropy_from_values(values: np.ndarray) -> float:
    safe = values[values > 0]
    if safe.size == 0:
        return 0.0
    return float(-(safe * np.log(safe)).sum())


def _build_metric_frame(
    rep_name: str,
    rep_df: pd.DataFrame,
    axes: list[str],
    family_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pca_df = _fit_pca_df(rep_df, axes, scale=True)
    family_pivot = (
        family_df.pivot_table(
            index=["sample_key", "sample_id", "class_label", "trajectory_concentration", "trajectory_index"],
            columns="family",
            values="family_fraction",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reset_index()
    )
    family_pivot.columns.name = None
    for family in FAMILY_ORDER:
        if family not in family_pivot.columns:
            family_pivot[family] = 0.0
    family_pivot["family_entropy"] = family_pivot[FAMILY_ORDER].apply(
        lambda row: _entropy_from_values(row.to_numpy(dtype=float)),
        axis=1,
    )
    family_pivot["top1_dominance"] = family_pivot[FAMILY_ORDER].max(axis=1)
    merged = rep_df.merge(
        pca_df[["sample_key", "pc1", "pc2"]].rename(columns={"pc1": "pc1_rep", "pc2": "pc2_rep"}),
        on="sample_key",
        how="left",
    ).merge(
        family_pivot[
            ["sample_key", "family_entropy", "top1_dominance"] + FAMILY_ORDER
        ],
        on="sample_key",
        how="left",
    )
    rows = []
    x = merged["trajectory_concentration"].to_numpy(dtype=float)
    metric_cols = ["pc1_rep", "family_entropy", "top1_dominance"] + FIXED_RADAR_AXES + FAMILY_ORDER
    for metric in metric_cols:
        if metric not in merged.columns:
            continue
        y = merged[metric].to_numpy(dtype=float)
        pearson = float(np.corrcoef(x, y)[0, 1]) if np.nanstd(y) > 0 else math.nan
        spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman")) if np.nanstd(y) > 0 else math.nan
        rows.append(
            {
                "representation": rep_name,
                "level": "per_spectrum",
                "metric_name": metric,
                "pearson_r": pearson,
                "spearman_r": spearman,
                "monotonicity_score": abs(spearman) if pd.notna(spearman) else math.nan,
            }
        )
    cond_mean = merged.groupby(["class_label", "trajectory_concentration"], as_index=False)[metric_cols].mean()
    x_cond = cond_mean["trajectory_concentration"].to_numpy(dtype=float)
    for metric in metric_cols:
        y = cond_mean[metric].to_numpy(dtype=float)
        pearson = float(np.corrcoef(x_cond, y)[0, 1]) if np.nanstd(y) > 0 else math.nan
        spearman = float(pd.Series(x_cond).corr(pd.Series(y), method="spearman")) if np.nanstd(y) > 0 else math.nan
        rows.append(
            {
                "representation": rep_name,
                "level": "condition_mean",
                "metric_name": metric,
                "pearson_r": pearson,
                "spearman_r": spearman,
                "monotonicity_score": abs(spearman) if pd.notna(spearman) else math.nan,
            }
        )
    return pd.DataFrame(rows), pca_df


def _adjacent_distance_metrics(
    rep_name: str,
    df: pd.DataFrame,
    axes: list[str],
) -> pd.DataFrame:
    class_mean = _mean_by_class(df, axes).sort_values("trajectory_concentration").reset_index(drop=True)
    rows = []
    for i in range(len(class_mean) - 1):
        left = class_mean.iloc[i]
        right = class_mean.iloc[i + 1]
        lv = left[axes].to_numpy(dtype=float)
        rv = right[axes].to_numpy(dtype=float)
        rows.append(
            {
                "representation": rep_name,
                "from_class": str(left["class_label"]),
                "to_class": str(right["class_label"]),
                "from_concentration": int(left["trajectory_concentration"]),
                "to_concentration": int(right["trajectory_concentration"]),
                "adjacent_distance": float(np.linalg.norm(rv - lv)),
            }
        )
    return pd.DataFrame(rows)


def _fit_response_axis(
    df: pd.DataFrame,
    axes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = df[axes].to_numpy(dtype=float)
    y = df["trajectory_concentration"].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    y_centered = y - y.mean()
    coef, *_ = np.linalg.lstsq(Xs, y_centered, rcond=None)
    score = Xs @ coef
    out = df[
        ["sample_key", "sample_id", "class_label", "trajectory_concentration", "trajectory_index"]
    ].copy()
    out["response_axis_score"] = score
    axis_metrics = pd.DataFrame(
        [
            {
                "axis_name": "delta_day2_control_response_axis",
                "pearson_r": float(np.corrcoef(y, score)[0, 1]) if np.nanstd(score) > 0 else math.nan,
                "spearman_r": float(pd.Series(y).corr(pd.Series(score), method="spearman")) if np.nanstd(score) > 0 else math.nan,
                "silhouette_by_concentration": float(
                    silhouette_score(score.reshape(-1, 1), df["class_label"].astype(str))
                ),
                "condition_mean_spearman": float(
                    pd.Series(out.groupby("class_label")["trajectory_concentration"].mean())
                    .corr(out.groupby("class_label")["response_axis_score"].mean(), method="spearman")
                ),
            }
        ]
    )
    abs_sum = float(np.abs(coef).sum())
    contrib_rows = []
    for axis, value in zip(axes, coef, strict=False):
        contrib_rows.append(
            {
                "axis_name": axis,
                "coefficient": float(value),
                "abs_fraction": abs(float(value)) / abs_sum if abs_sum > 0 else 0.0,
                "direction": "positive" if float(value) >= 0 else "negative",
            }
        )
    return out, axis_metrics, pd.DataFrame(contrib_rows).sort_values("abs_fraction", ascending=False)


def _family_response_contributions(family_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        family_df.pivot_table(
            index=["sample_key", "trajectory_concentration"],
            columns="family",
            values="family_fraction",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reset_index()
    )
    pivot.columns.name = None
    for family in FAMILY_ORDER:
        if family not in pivot.columns:
            pivot[family] = 0.0
    X = pivot[FAMILY_ORDER].to_numpy(dtype=float)
    y = pivot["trajectory_concentration"].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    coef, *_ = np.linalg.lstsq(Xs, y - y.mean(), rcond=None)
    abs_sum = float(np.abs(coef).sum())
    rows = []
    for family, value in zip(FAMILY_ORDER, coef, strict=False):
        rows.append(
            {
                "family": family,
                "coefficient": float(value),
                "abs_fraction": abs(float(value)) / abs_sum if abs_sum > 0 else 0.0,
                "direction": "positive" if float(value) >= 0 else "negative",
            }
        )
    return pd.DataFrame(rows).sort_values("abs_fraction", ascending=False)


def _plot_family_shift_vs_control(class_family_df: pd.DataFrame, output_path: Path) -> None:
    control = class_family_df[class_family_df["class_label"].astype(str) == "D2_C0"].set_index("family")[
        "family_fraction"
    ]
    labels = [label for label in DAY2_CLASS_ORDER if label != "D2_C0"]
    x = np.arange(len(FAMILY_ORDER))
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    width = 0.22
    for i, label in enumerate(labels):
        sub = class_family_df[class_family_df["class_label"].astype(str) == label].set_index("family")[
            "family_fraction"
        ].reindex(FAMILY_ORDER, fill_value=0.0)
        diff = sub.to_numpy(dtype=float) - control.reindex(FAMILY_ORDER, fill_value=0.0).to_numpy(dtype=float)
        ax.bar(x + (i - 1) * width, diff, width=width, label=label, color=CLUSTER_COLORS[i + 1], alpha=0.92)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILY_ORDER, rotation=25, ha="right")
    ax.set_ylabel("Family fraction shift vs D2_C0")
    ax.set_title("Day 2 Family Shift vs Control")
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_response_axis_boxplot(score_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    data = [
        score_df[score_df["class_label"].astype(str) == label]["response_axis_score"].to_numpy(dtype=float)
        for label in DAY2_CLASS_ORDER
    ]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=DAY2_CLASS_ORDER)
    for patch, color in zip(bp["boxes"], CLUSTER_COLORS[: len(DAY2_CLASS_ORDER)], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax.set_title("Day 2 Response Axis by Concentration")
    ax.set_ylabel("Response axis score")
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_response_axis_trend(score_df: pd.DataFrame, output_path: Path) -> None:
    trend = (
        score_df.groupby(["class_label", "trajectory_concentration"], as_index=False)["response_axis_score"]
        .mean()
        .sort_values("trajectory_concentration")
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(
        trend["trajectory_concentration"].to_numpy(dtype=float),
        trend["response_axis_score"].to_numpy(dtype=float),
        marker="o",
        linewidth=2.2,
        color="#355070",
    )
    ax.set_xlabel("APAP concentration")
    ax.set_ylabel("Mean response axis score")
    ax.set_title("Day 2 Response Axis Trend")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_response_axis_scatter(score_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for i, label in enumerate(DAY2_CLASS_ORDER):
        sub = score_df[score_df["class_label"].astype(str) == label].copy()
        ax.scatter(
            sub["trajectory_concentration"].to_numpy(dtype=float),
            sub["response_axis_score"].to_numpy(dtype=float),
            s=24,
            alpha=0.45,
            color=CLUSTER_COLORS[i],
            label=label,
        )
    ax.set_xlabel("APAP concentration")
    ax.set_ylabel("Response axis score")
    ax.set_title("Day 2 Response Axis Scatter")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_adjacent_distance(adjacent_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    step_labels = adjacent_df["from_class"].astype(str) + "→" + adjacent_df["to_class"].astype(str)
    reps = adjacent_df["representation"].astype(str).drop_duplicates().tolist()
    x = np.arange(len(step_labels.drop_duplicates()))
    width = 0.24
    unique_steps = adjacent_df[["from_class", "to_class"]].drop_duplicates().reset_index(drop=True)
    for i, rep in enumerate(reps):
        vals = []
        sub = adjacent_df[adjacent_df["representation"].astype(str) == rep].copy()
        for row in unique_steps.itertuples(index=False):
            hit = sub[(sub["from_class"].astype(str) == row.from_class) & (sub["to_class"].astype(str) == row.to_class)]
            vals.append(float(hit["adjacent_distance"].iloc[0]) if not hit.empty else 0.0)
        ax.bar(x + (i - 1) * width, vals, width=width, label=rep, color=CLUSTER_COLORS[i], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row.from_class}→{row.to_class}" for row in unique_steps.itertuples(index=False)],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Adjacent centroid distance")
    ax.set_title("Day 2 Adjacent Concentration Distance")
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_condition_separation(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.5))
    metric_cols = [
        "silhouette_by_concentration",
        "mean_centroid_distance",
        "nearest_neighbor_purity",
    ]
    titles = ["Silhouette", "Mean Centroid Distance", "NN Purity"]
    for ax, metric, title in zip(axs, metric_cols, titles, strict=False):
        vals = metrics_df[metric].to_numpy(dtype=float)
        ax.bar(
            np.arange(len(metrics_df)),
            vals,
            color=CLUSTER_COLORS[: len(metrics_df)],
            alpha=0.9,
        )
        ax.set_xticks(np.arange(len(metrics_df)))
        ax.set_xticklabels(metrics_df["representation"].astype(str).tolist(), rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    fig.suptitle("Day 2 Condition Separation Comparison")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _prior_run_summary(
    run_name: str,
    root: Path,
) -> dict[str, object]:
    bsv_path = root / "tables" / "per_sample_bsv.csv"
    delta_path = root / "tables" / "per_sample_delta_bsv.csv"
    family_path = root / "tables" / "sample_family_fingerprint.csv"
    if not (bsv_path.exists() and delta_path.exists() and family_path.exists()):
        return {
            "run_name": run_name,
            "strongest_concentration_silhouette": math.nan,
            "strongest_monotonicity_correlation": math.nan,
            "mean_adjacent_concentration_distance": math.nan,
            "best_response_axis_correlation": math.nan,
        }
    bsv_df = pd.read_csv(bsv_path)
    delta_df = pd.read_csv(delta_path)
    family_df = pd.read_csv(family_path)
    bsv_df = bsv_df[bsv_df["class_label"].astype(str).isin(DAY2_CLASS_ORDER)].copy()
    delta_df = delta_df[delta_df["class_label"].astype(str).isin(DAY2_CLASS_ORDER)].copy()
    family_df = family_df[family_df["class_label"].astype(str).isin(DAY2_CLASS_ORDER)].copy()
    axes = _axes_present(bsv_df)
    if bsv_df.empty or delta_df.empty or not axes:
        return {
            "run_name": run_name,
            "strongest_concentration_silhouette": math.nan,
            "strongest_monotonicity_correlation": math.nan,
            "mean_adjacent_concentration_distance": math.nan,
            "best_response_axis_correlation": math.nan,
        }
    sils = []
    for rep_df in [bsv_df, delta_df]:
        sils.append(float(silhouette_score(StandardScaler().fit_transform(rep_df[axes]), rep_df["class_label"].astype(str))))
    family_pivot = (
        family_df.pivot_table(
            index=["sample_key", "class_label", "trajectory_concentration"],
            columns="family",
            values="family_fraction",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reset_index()
    )
    family_pivot.columns.name = None
    for family in FAMILY_ORDER:
        if family not in family_pivot.columns:
            family_pivot[family] = 0.0
    family_pivot["family_entropy"] = family_pivot[FAMILY_ORDER].apply(
        lambda row: _entropy_from_values(row.to_numpy(dtype=float)),
        axis=1,
    )
    y = delta_df["trajectory_concentration"].to_numpy(dtype=float)
    top_corrs = []
    for metric in axes:
        top_corrs.append(abs(float(pd.Series(y).corr(delta_df[metric], method="spearman"))))
    top_corrs.append(abs(float(pd.Series(y).corr(family_pivot["family_entropy"], method="spearman"))))
    class_mean = delta_df.groupby(["class_label", "trajectory_concentration"], as_index=False)[axes].mean().sort_values("trajectory_concentration")
    adj = []
    for i in range(len(class_mean) - 1):
        adj.append(
            float(
                np.linalg.norm(
                    class_mean.iloc[i + 1][axes].to_numpy(dtype=float)
                    - class_mean.iloc[i][axes].to_numpy(dtype=float)
                )
            )
        )
    axis_scores, axis_metrics, _ = _fit_response_axis(delta_df, axes)
    return {
        "run_name": run_name,
        "strongest_concentration_silhouette": float(max(sils)),
        "strongest_monotonicity_correlation": float(max(top_corrs)),
        "mean_adjacent_concentration_distance": float(np.mean(adj)) if adj else math.nan,
        "best_response_axis_correlation": float(axis_metrics["spearman_r"].iloc[0]),
    }


def _build_report(
    report_path: Path,
    verification_df: pd.DataFrame,
    separation_df: pd.DataFrame,
    monotonicity_df: pd.DataFrame,
    adjacent_df: pd.DataFrame,
    response_axis_metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    contribution_bsv_df: pd.DataFrame,
    contribution_family_df: pd.DataFrame,
) -> None:
    header = "| " + " | ".join(comparison_df.columns.astype(str).tolist()) + " |"
    divider = "| " + " | ".join(["---"] * len(comparison_df.columns)) + " |"
    body = [
        "| " + " | ".join(
            f"{value:.4f}" if isinstance(value, float) and pd.notna(value) else str(value)
            for value in row
        ) + " |"
        for row in comparison_df.itertuples(index=False, name=None)
    ]
    best_sep = separation_df.sort_values("silhouette_by_concentration", ascending=False).iloc[0]
    best_mono = monotonicity_df.sort_values("monotonicity_score", ascending=False).iloc[0]
    lines = [
        "# GAIRAv3 Pilot 3 SHINE Day 2 Control-Anchored Report",
        "",
        "## 1. Why This Follow-Up Was Needed",
        "- Previous SHINE analyses were dominated by all-day structure.",
        "- The SHINE paper reports its clearest APAP signal on Day 2.",
        "- This pass isolates Day 2 and tests whether a D2_C0-anchored delta reveals a cleaner concentration-response direction.",
        "",
        "## 2. Day 2 Input Verification",
        f"- Total Day 2 rows: `{int(verification_df['day2_rows'].iloc[0])}`.",
        f"- Total unique Day 2 sample IDs: `{int(verification_df['day2_unique_sample_ids'].iloc[0])}`.",
        f"- Rows passed into cfg05 BSV generation: `{int(verification_df['rows_passed_to_bsv_generation'].iloc[0])}`.",
        f"- Day 2 class counts: D2_C0 `{int(verification_df['rows_D2_C0'].iloc[0])}`, D2_C10 `{int(verification_df['rows_D2_C10'].iloc[0])}`, D2_C20 `{int(verification_df['rows_D2_C20'].iloc[0])}`, D2_C40 `{int(verification_df['rows_D2_C40'].iloc[0])}`.",
        "",
        "## 3. Concentration Separation",
        f"- Best concentration silhouette came from `{best_sep['representation']}` at `{float(best_sep['silhouette_by_concentration']):.4f}`.",
        "- Raw BSV, Day-2 cohort-relative delta, and Day-2 control-anchored delta were all evaluated on the same Day 2 full-spectrum rows.",
        "- Control anchoring is most useful if it increases concentration ordering without making the representation diffuse or unstable.",
        "",
        "## 4. Monotonicity",
        f"- Strongest monotonicity signal was `{best_mono['representation']}` / `{best_mono['metric_name']}` / `{best_mono['level']}` with Spearman `{float(best_mono['spearman_r']):.4f}`.",
        f"- Mean adjacent concentration distance for control-anchored delta: `{float(adjacent_df[adjacent_df['representation'].astype(str) == 'delta_day2_control']['adjacent_distance'].mean()):.4f}`.",
        "- The central question is whether D2_C0→D2_C40 shows a coherent ordered movement rather than only broad latent-state variation.",
        "",
        "## 5. Response Axis",
        f"- The Day 2 response axis was fit in control-anchored delta-BSV space.",
        f"- Response-axis Spearman vs concentration: `{float(response_axis_metrics_df['spearman_r'].iloc[0]):.4f}`.",
        f"- Response-axis silhouette by concentration: `{float(response_axis_metrics_df['silhouette_by_concentration'].iloc[0]):.4f}`.",
        f"- Condition-mean ordering Spearman: `{float(response_axis_metrics_df['condition_mean_spearman'].iloc[0]):.4f}`.",
        "- If this one-axis readout is useful, it should preserve concentration ordering and spread adjacent concentrations apart along a single biochemical trend.",
        "",
        "## 6. Biochemical Interpretation",
        "- Interpret the response axis only in broad biochemical themes.",
        "- Most influential BSV axes: " + ", ".join(
            f"`{row.axis_name}` ({row.direction}, {row.abs_fraction:.3f})"
            for row in contribution_bsv_df.head(4).itertuples(index=False)
        )
        + ".",
        "- Most influential family themes: " + ", ".join(
            f"`{row.family}` ({row.direction}, {row.abs_fraction:.3f})"
            for row in contribution_family_df.head(4).itertuples(index=False)
        )
        + ".",
        "",
        "## 7. Comparison to Previous SHINE Runs",
        header,
        divider,
        *body,
        "",
        "## 8. Final Conclusion",
        "- This pass should be judged against the SHINE paper's Day-2 concentration framing, not the broader all-day latent-state framing alone.",
        "- If the control-anchored delta and response axis improve ordering relative to the previous runs, cfg05 is recovering a SHINE-specific Day 2 readout rather than only latent states.",
        "- If the improvement is still weak, SHINE remains primarily a latent-state dataset under GAIRA even after Day-2 control anchoring.",
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    storage_cfg = load_autoresearch_storage_config(DEFAULT_STORAGE_CONFIG_PATH)
    sprint_paths = initialize_autoresearch_sprint(
        DEFAULT_STORAGE_CONFIG_PATH,
        sprint_id=f"{storage_cfg.sprint_id}/{SPRINT_SUBDIR}",
    )
    for directory in [sprint_paths.tables_dir, sprint_paths.figures_dir, sprint_paths.report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    registries = load_architecture_registries(
        grounding_family_registry_path=ROOT / "config" / "gaira_grounding_family_registry_v1.csv",
        target_family_registry_path=ROOT / "config" / "gaira_target_family_registry_v1.csv",
        inference_lane_registry_path=ROOT / "config" / "gaira_inference_lane_registry_v2.csv",
        representation_mode_registry_path=ROOT / "config" / "gaira_representation_mode_registry_v2.csv",
        dataset_experiment_registry_path=ROOT / "config" / "gaira_dataset_experiment_registry_v2.csv",
        experiment_plan_path=ARCH_DIR / "first_pass_experiment_plan.csv",
        phase1_registry_path=PHASE1_DIR / "phase1_dataset_registry_v2.csv",
        phase1_grounding_map_path=PHASE1_DIR / "phase1_target_grounding_map_v2.csv",
        phase1_exclusions_path=PHASE1_DIR / "phase1_grounding_exclusions.csv",
    )
    resolved = _resolve_alias(registries, SUBSET_ALIAS)
    query_df = load_query_dataframe(resolved.dataset_row)
    day2_query_df = _prepare_day2_query_df(query_df)

    verification_df = pd.DataFrame(
        [
            {
                "dataset_id": "shine_ev_sers",
                "subset_alias": SUBSET_ALIAS,
                "day2_rows": int(len(day2_query_df)),
                "day2_unique_sample_ids": int(day2_query_df["sample_id"].astype(str).nunique()),
                "day2_unique_biosample_ids": int(len(day2_query_df)),
                "rows_D2_C0": int((day2_query_df["class_label"].astype(str) == "D2_C0").sum()),
                "rows_D2_C10": int((day2_query_df["class_label"].astype(str) == "D2_C10").sum()),
                "rows_D2_C20": int((day2_query_df["class_label"].astype(str) == "D2_C20").sum()),
                "rows_D2_C40": int((day2_query_df["class_label"].astype(str) == "D2_C40").sum()),
                "rows_passed_to_bsv_generation": int(len(day2_query_df)),
            }
        ]
    )
    verification_df.to_csv(sprint_paths.tables_dir / "day2_input_verification.csv", index=False)

    grounding_df, mapping_df, harness_config, _ = _prepare_grounding_and_mapping(
        registries, resolved, CONFIG_SPEC
    )
    spectrum_bsv_df, retrieval_df = build_bsv_profiles_pass5(
        day2_query_df,
        grounding_df,
        mapping_df,
        top_k=harness_config.top_k,
        similarity_metric="cosine",
        weighting_mode=harness_config.weighting_mode,
        weighting_param=harness_config.weighting_param,
        diversity_mode=harness_config.diversity_mode,
        family_min_coverage=harness_config.family_min_coverage,
    )
    axes = _axes_present(spectrum_bsv_df)
    if not axes:
        raise RuntimeError("No BSV axes found for Day 2 SHINE analysis.")

    spectrum_bsv_df = spectrum_bsv_df.copy()
    spectrum_bsv_df["sample_id"] = spectrum_bsv_df["sample_key"].astype(str).map(
        day2_query_df.set_index("sample_key")["sample_id"].astype(str).to_dict()
    )
    spectrum_bsv_df["trajectory_concentration"] = spectrum_bsv_df["class_label"].astype(str).map(_parse_concentration)
    spectrum_bsv_df["trajectory_index"] = spectrum_bsv_df["trajectory_concentration"].astype(int)

    spectrum_bsv_df.to_csv(sprint_paths.tables_dir / "per_spectrum_bsv_day2.csv", index=False)
    retrieval_df.to_csv(sprint_paths.tables_dir / "per_spectrum_retrieval_hits_day2.csv", index=False)

    spectrum_family_df = _family_fingerprint_from_retrieval(
        retrieval_df,
        spectrum_bsv_df[["sample_key", "sample_id", "trajectory_concentration", "trajectory_index"]].assign(
            class_label=spectrum_bsv_df["class_label"].astype(str).values
        ),
    )
    spectrum_family_df.to_csv(sprint_paths.tables_dir / "per_spectrum_family_fingerprint_day2.csv", index=False)

    class_mean_bsv_df = _mean_by_class(spectrum_bsv_df, axes)
    class_mean_bsv_df.to_csv(sprint_paths.tables_dir / "class_mean_bsv_day2.csv", index=False)
    class_mean_family_df = _class_family_means(spectrum_family_df)
    class_mean_family_df.to_csv(sprint_paths.tables_dir / "class_mean_family_fingerprint_day2.csv", index=False)

    delta_cohort_df = _cohort_delta(spectrum_bsv_df, axes)
    delta_control_df = _control_delta(spectrum_bsv_df, axes, control_label="D2_C0")
    delta_cohort_df.to_csv(sprint_paths.tables_dir / "per_spectrum_delta_bsv_day2_cohort.csv", index=False)
    delta_control_df.to_csv(sprint_paths.tables_dir / "per_spectrum_delta_bsv_day2_control.csv", index=False)
    _mean_by_class(delta_cohort_df, axes).to_csv(
        sprint_paths.tables_dir / "class_mean_delta_bsv_day2_cohort.csv", index=False
    )
    _mean_by_class(delta_control_df, axes).to_csv(
        sprint_paths.tables_dir / "class_mean_delta_bsv_day2_control.csv", index=False
    )

    sample_bsv_day2 = (
        spectrum_bsv_df.groupby(["sample_id", "class_label", "trajectory_concentration", "trajectory_index"], as_index=False)[axes]
        .mean()
    )
    sample_bsv_day2["sample_key"] = "sample_mean__" + sample_bsv_day2["sample_id"].astype(str)
    sample_bsv_day2 = sample_bsv_day2[
        ["sample_key", "sample_id", "class_label", "trajectory_concentration", "trajectory_index"] + axes
    ]
    sample_delta_cohort_day2 = _cohort_delta(sample_bsv_day2, axes)
    sample_delta_control_day2 = _control_delta(sample_bsv_day2, axes, control_label="D2_C0")
    sample_family_day2 = (
        spectrum_family_df.groupby(["sample_id", "class_label", "trajectory_concentration", "family"], as_index=False)["family_fraction"]
        .mean()
    )
    sample_family_day2["sample_key"] = "sample_mean__" + sample_family_day2["sample_id"].astype(str)
    sample_family_day2["trajectory_index"] = sample_family_day2["trajectory_concentration"].astype(int)
    sample_bsv_day2.to_csv(sprint_paths.tables_dir / "per_sample_bsv_day2.csv", index=False)
    sample_delta_cohort_day2.to_csv(
        sprint_paths.tables_dir / "per_sample_delta_bsv_day2_cohort.csv", index=False
    )
    sample_delta_control_day2.to_csv(
        sprint_paths.tables_dir / "per_sample_delta_bsv_day2_control.csv", index=False
    )
    sample_family_day2.to_csv(
        sprint_paths.tables_dir / "sample_family_fingerprint_day2.csv", index=False
    )

    separation_df = pd.concat(
        [
            _representation_separation_metrics(spectrum_bsv_df, axes, representation="bsv"),
            _representation_separation_metrics(delta_cohort_df, axes, representation="delta_day2_cohort"),
            _representation_separation_metrics(delta_control_df, axes, representation="delta_day2_control"),
        ],
        ignore_index=True,
    )
    separation_df.to_csv(sprint_paths.tables_dir / "day2_concentration_separation_metrics.csv", index=False)

    monotonicity_bsv_df, bsv_pca_df = _build_metric_frame("bsv", spectrum_bsv_df, axes, spectrum_family_df)
    monotonicity_cohort_df, cohort_pca_df = _build_metric_frame("delta_day2_cohort", delta_cohort_df, axes, spectrum_family_df)
    monotonicity_control_df, control_pca_df = _build_metric_frame("delta_day2_control", delta_control_df, axes, spectrum_family_df)
    monotonicity_df = pd.concat(
        [monotonicity_bsv_df, monotonicity_cohort_df, monotonicity_control_df],
        ignore_index=True,
    )
    monotonicity_df.to_csv(sprint_paths.tables_dir / "day2_monotonicity_metrics.csv", index=False)

    adjacent_df = pd.concat(
        [
            _adjacent_distance_metrics("bsv", spectrum_bsv_df, axes),
            _adjacent_distance_metrics("delta_day2_cohort", delta_cohort_df, axes),
            _adjacent_distance_metrics("delta_day2_control", delta_control_df, axes),
        ],
        ignore_index=True,
    )
    adjacent_df.to_csv(sprint_paths.tables_dir / "day2_adjacent_distance_metrics.csv", index=False)

    response_axis_scores_df, response_axis_metrics_df, response_axis_bsv_df = _fit_response_axis(
        delta_control_df, axes
    )
    response_axis_scores_df.to_csv(sprint_paths.tables_dir / "day2_response_axis_scores.csv", index=False)
    response_axis_metrics_df.to_csv(sprint_paths.tables_dir / "day2_response_axis_metrics.csv", index=False)
    response_axis_bsv_df.to_csv(
        sprint_paths.tables_dir / "day2_response_axis_bsv_contributions.csv", index=False
    )
    response_axis_family_df = _family_response_contributions(spectrum_family_df)
    response_axis_family_df.to_csv(
        sprint_paths.tables_dir / "day2_response_axis_family_contributions.csv",
        index=False,
    )

    comparison_df = pd.DataFrame(
        [
            _prior_run_summary("samplemean_pilot3", PREVIOUS_SAMPLEMEAN_ROOT),
            _prior_run_summary("all_day_fullspectra_pilot3", PREVIOUS_FULLSPECTRA_ROOT),
            {
                "run_name": "day2_controlanchored_pass",
                "strongest_concentration_silhouette": float(
                    separation_df["silhouette_by_concentration"].max()
                ),
                "strongest_monotonicity_correlation": float(
                    monotonicity_df["monotonicity_score"].fillna(0.0).max()
                ),
                "mean_adjacent_concentration_distance": float(
                    adjacent_df[adjacent_df["representation"].astype(str) == "delta_day2_control"]["adjacent_distance"].mean()
                ),
                "best_response_axis_correlation": float(response_axis_metrics_df["spearman_r"].iloc[0]),
            },
        ]
    )
    comparison_df.to_csv(sprint_paths.tables_dir / "day2_vs_prior_shine_comparison.csv", index=False)

    _, spectral_matrix = decode_and_align(day2_query_df)
    spectral_scores, spectral_explained = _fit_pca(spectral_matrix, scale=False)
    spectral_pca_df = day2_query_df[
        ["sample_key", "sample_id", "class_label", "trajectory_concentration", "trajectory_index"]
    ].copy()
    spectral_pca_df["pc1"] = spectral_scores[:, 0]
    spectral_pca_df["pc2"] = spectral_scores[:, 1]
    spectral_pca_df["pc1_explained_ratio"] = float(spectral_explained[0])
    spectral_pca_df["pc2_explained_ratio"] = float(spectral_explained[1])

    _plot_scatter(
        spectral_pca_df,
        "pc1",
        "pc2",
        sprint_paths.figures_dir / "day2_pca_spectral_by_concentration.png",
        title="Day 2 Spectral PCA by Concentration",
        hue_col="class_label",
    )
    _plot_scatter(
        bsv_pca_df,
        "pc1",
        "pc2",
        sprint_paths.figures_dir / "day2_pca_bsv_by_concentration.png",
        title="Day 2 BSV PCA by Concentration",
        hue_col="class_label",
    )
    _plot_scatter(
        cohort_pca_df,
        "pc1",
        "pc2",
        sprint_paths.figures_dir / "day2_pca_delta_cohort_by_concentration.png",
        title="Day 2 Cohort-Delta PCA by Concentration",
        hue_col="class_label",
    )
    _plot_scatter(
        control_pca_df,
        "pc1",
        "pc2",
        sprint_paths.figures_dir / "day2_pca_delta_control_by_concentration.png",
        title="Day 2 Control-Delta PCA by Concentration",
        hue_col="class_label",
    )

    _plot_radar_grid(
        _ensure_fixed_axes(class_mean_bsv_df)[["class_label"] + FIXED_RADAR_AXES],
        "class_label",
        sprint_paths.figures_dir / "day2_radar_bsv_by_concentration.png",
        "Day 2 Absolute BSV by Concentration",
        delta_mode=False,
    )
    _plot_radar_grid(
        _ensure_fixed_axes(_mean_by_class(delta_cohort_df, axes))[["class_label"] + FIXED_RADAR_AXES],
        "class_label",
        sprint_paths.figures_dir / "day2_radar_delta_cohort_by_concentration.png",
        "Day 2 Cohort-Relative Delta by Concentration",
        delta_mode=True,
    )
    _plot_radar_grid(
        _ensure_fixed_axes(_mean_by_class(delta_control_df, axes))[["class_label"] + FIXED_RADAR_AXES],
        "class_label",
        sprint_paths.figures_dir / "day2_radar_delta_control_by_concentration.png",
        "Day 2 Control-Anchored Delta by Concentration",
        delta_mode=True,
    )

    _plot_family_bars(
        class_mean_family_df,
        "class_label",
        sprint_paths.figures_dir / "day2_family_fingerprint_bars.png",
        "Day 2 Family Fingerprints by Concentration",
    )
    _plot_family_shift_vs_control(
        class_mean_family_df,
        sprint_paths.figures_dir / "day2_family_shift_vs_control.png",
    )
    _plot_response_axis_boxplot(
        response_axis_scores_df,
        sprint_paths.figures_dir / "day2_response_axis_boxplot.png",
    )
    _plot_response_axis_trend(
        response_axis_scores_df,
        sprint_paths.figures_dir / "day2_response_axis_trend.png",
    )
    _plot_response_axis_scatter(
        response_axis_scores_df,
        sprint_paths.figures_dir / "day2_response_axis_vs_concentration_scatter.png",
    )
    _plot_adjacent_distance(
        adjacent_df,
        sprint_paths.figures_dir / "day2_adjacent_distance_plot.png",
    )
    _plot_condition_separation(
        separation_df,
        sprint_paths.figures_dir / "day2_condition_separation_comparison.png",
    )

    report_md = sprint_paths.report_dir / "GAIRAv3_Pilot3_SHINE_day2_controlanchored_report.md"
    report_pdf = sprint_paths.report_dir / "GAIRAv3_Pilot3_SHINE_day2_controlanchored_report.pdf"
    _build_report(
        report_md,
        verification_df,
        separation_df,
        monotonicity_df,
        adjacent_df,
        response_axis_metrics_df,
        comparison_df,
        response_axis_bsv_df,
        response_axis_family_df,
    )
    build_pdf_report(report_md, sorted(sprint_paths.figures_dir.glob("*.png")), report_pdf)


if __name__ == "__main__":
    main()
