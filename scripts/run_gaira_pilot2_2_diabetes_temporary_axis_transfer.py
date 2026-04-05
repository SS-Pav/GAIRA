from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaira.demo.gaira_pilot_utils import build_pdf_report
from scripts.run_gaira_pilot2_1_latent_state_interpretation import FAMILY_ORDER
from scripts.run_gaira_pilot3_shine_ev_sers_fullspectra import FIXED_RADAR_AXES, _fit_pca


PILOT2_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot2_target_validation_v1"
)
PILOT21_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot2_1_latent_state_interpretation"
)
EXT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/shine_bsv_extension_test"
)
OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot2_2_diabetes_temporary_axis_transfer"
)
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "report"

TEMP_AXES = ["guanidino_response", "amide_stress_response"]
BROAD_ORDER = ["Impact", "Strong-D"]
BROAD_COLORS = {"Impact": "#d1495b", "Strong-D": "#2a9d8f"}
TEMP_COLORS = {"guanidino_response": "#6d597a", "amide_stress_response": "#e76f51"}


def _ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, TABLES_DIR, FIGURES_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _df_to_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _load_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / "tables" / name
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Required table is empty: {path}")
    return df


def _parse_regions(text: str) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    for token in str(text).split(";"):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            regions.append((int(a), int(b)))
        else:
            val = int(token)
            regions.append((val, val))
    return regions


def _region_mean(matrix: np.ndarray, wns: np.ndarray, start: int, end: int) -> np.ndarray:
    mask = (wns >= start) & (wns <= end)
    if not np.any(mask):
        return np.zeros(matrix.shape[0], dtype=float)
    return matrix[:, mask].mean(axis=1)


def _compute_temp_axes_from_spectra(sample_query_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["sample_key", "sample_id", "class_label", "subclass_label", "source_file", "n_scans"]
    out = sample_query_df[meta_cols].copy()
    out = out.rename(columns={"class_label": "broad_class_label"})
    wns = np.array(json.loads(sample_query_df.iloc[0]["wavenumbers_json"]), dtype=int)
    matrix = np.vstack(sample_query_df["intensity_json"].map(lambda x: np.array(json.loads(x), dtype=float)))
    for axis_name in TEMP_AXES:
        row = candidate_df[candidate_df["candidate_axis_name"].astype(str) == axis_name]
        if row.empty:
            raise RuntimeError(f"Missing temporary axis definition for {axis_name}")
        spec = row.iloc[0]
        pos_regions = _parse_regions(spec["positive_regions"])
        neg_regions = _parse_regions(spec["negative_regions"])
        pos_stack = np.vstack([_region_mean(matrix, wns, a, b) for a, b in pos_regions]) if pos_regions else np.zeros((1, len(out)))
        neg_stack = np.vstack([_region_mean(matrix, wns, a, b) for a, b in neg_regions]) if neg_regions else np.zeros((1, len(out)))
        out[axis_name] = pos_stack.mean(axis=0) - neg_stack.mean(axis=0)
    return out


def _family_wide(family_df: pd.DataFrame, sample_keys: pd.Series) -> pd.DataFrame:
    wide = family_df.pivot(index="sample_key", columns="family", values="family_fraction")
    wide = wide.reindex(sample_keys.astype(str)).fillna(0.0)
    for family in FAMILY_ORDER:
        if family not in wide.columns:
            wide[family] = 0.0
    return wide[FAMILY_ORDER].reset_index()


def _cluster_sample_counts(assign_df: pd.DataFrame) -> pd.DataFrame:
    return (
        assign_df.groupby(["class_label", "cluster_label"], as_index=False)
        .size()
        .rename(columns={"size": "n_samples"})
        .sort_values(["class_label", "cluster_label"])
        .reset_index(drop=True)
    )


def _write_input_verification(
    bsv_df: pd.DataFrame,
    assign_df: pd.DataFrame,
    reused_tables: list[str],
) -> pd.DataFrame:
    broad_counts = (
        bsv_df.groupby("class_label", as_index=False)
        .size()
        .rename(columns={"size": "count", "class_label": "group_label"})
        .assign(group_type="broad_class")
    )
    cluster_counts = (
        assign_df.groupby("cluster_label", as_index=False)
        .size()
        .rename(columns={"size": "count", "cluster_label": "group_label"})
        .assign(group_type="latent_cluster")
    )
    reuse_rows = pd.DataFrame(
        {
            "group_type": "reused_table",
            "group_label": reused_tables,
            "count": 1,
        }
    )
    totals = pd.DataFrame(
        [
            {"group_type": "summary", "group_label": "total_sample_count", "count": int(len(bsv_df))},
            {"group_type": "summary", "group_label": "core_bsv_recomputed", "count": 0},
        ]
    )
    out = pd.concat([totals, broad_counts, cluster_counts, reuse_rows], ignore_index=True)
    out.to_csv(TABLES_DIR / "pilot2_2_input_verification.csv", index=False)
    return out


def _linear_cv_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return float("nan")
    min_count = int(counts.min())
    if min_count < 2:
        return float("nan")
    n_splits = min(5, min_count)
    clf = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return float(cross_val_score(clf, X, y, cv=cv).mean())


def _nn_purity(X: np.ndarray, y: np.ndarray) -> float:
    n_neighbors = min(6, len(X))
    if n_neighbors <= 1:
        return float("nan")
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, y)
    neighbors = clf.kneighbors(X, return_distance=False)
    scores = []
    for idx, nbrs in enumerate(neighbors):
        nbrs = [n for n in nbrs if n != idx]
        if not nbrs:
            continue
        scores.append(np.mean(y[nbrs] == y[idx]))
    return float(np.mean(scores)) if scores else float("nan")


def _mean_pairwise_centroid_distance(X: np.ndarray, labels: np.ndarray) -> float:
    unique = list(pd.unique(labels))
    if len(unique) < 2:
        return float("nan")
    cents = []
    for label in unique:
        cents.append(X[labels == label].mean(axis=0))
    cents = np.vstack(cents)
    dists = []
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            dists.append(float(np.linalg.norm(cents[i] - cents[j])))
    return float(np.mean(dists)) if dists else float("nan")


def _representation_matrix(
    kind: str,
    merged_df: pd.DataFrame,
    family_wide_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    if kind == "current_bsv":
        cols = [axis for axis in FIXED_RADAR_AXES if axis in merged_df.columns]
        return merged_df[["sample_key"] + cols].copy(), cols
    if kind == "family":
        cols = FAMILY_ORDER.copy()
        return family_wide_df[["sample_key"] + cols].copy(), cols
    if kind == "temporary_axes":
        cols = TEMP_AXES.copy()
        return merged_df[["sample_key"] + cols].copy(), cols
    if kind == "bsv_family":
        cols = [axis for axis in FIXED_RADAR_AXES if axis in merged_df.columns] + FAMILY_ORDER
        df = merged_df[["sample_key"] + [axis for axis in FIXED_RADAR_AXES if axis in merged_df.columns]].merge(
            family_wide_df[["sample_key"] + FAMILY_ORDER], on="sample_key", how="left"
        )
        return df, cols
    if kind == "bsv_family_temp":
        cols = [axis for axis in FIXED_RADAR_AXES if axis in merged_df.columns] + FAMILY_ORDER + TEMP_AXES
        df = merged_df[
            ["sample_key"] + [axis for axis in FIXED_RADAR_AXES if axis in merged_df.columns] + TEMP_AXES
        ].merge(family_wide_df[["sample_key"] + FAMILY_ORDER], on="sample_key", how="left")
        return df, cols
    raise ValueError(f"Unknown representation kind: {kind}")


def _pca_frame(rep_df: pd.DataFrame, cols: list[str], meta_df: pd.DataFrame) -> pd.DataFrame:
    scores, explained = _fit_pca(rep_df[cols].to_numpy(dtype=float), scale=True)
    out = meta_df.copy()
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1]
    out["pc1_explained_ratio"] = float(explained[0])
    out["pc2_explained_ratio"] = float(explained[1])
    return out


def _plot_pca(
    pca_df: pd.DataFrame,
    label_col: str,
    color_map: dict[str, str],
    path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(7.2, 5.8))
    for label, sub in pca_df.groupby(label_col, sort=False):
        color = color_map.get(str(label), "#355070")
        plt.scatter(sub["pc1"], sub["pc2"], s=46, alpha=0.82, label=str(label), color=color)
    plt.xlabel(f"PC1 ({pca_df['pc1_explained_ratio'].iloc[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca_df['pc2_explained_ratio'].iloc[0] * 100:.1f}%)")
    plt.title(title)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _broad_label_comparison(
    merged_df: pd.DataFrame,
    family_wide_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    pca_frames: dict[str, pd.DataFrame] = {}
    labels = merged_df["class_label"].astype(str).to_numpy()
    for kind in ["current_bsv", "family", "temporary_axes", "bsv_family", "bsv_family_temp"]:
        rep_df, cols = _representation_matrix(kind, merged_df, family_wide_df)
        X = StandardScaler().fit_transform(rep_df[cols].to_numpy(dtype=float))
        rows.append(
            {
                "representation_name": kind,
                "silhouette_by_broad_class": float(silhouette_score(X, labels)),
                "centroid_distance": _mean_pairwise_centroid_distance(X, labels),
                "nearest_neighbor_purity": _nn_purity(X, labels),
                "linear_cv_accuracy": _linear_cv_accuracy(X, labels),
            }
        )
        pca_frames[kind] = _pca_frame(rep_df, cols, merged_df[["sample_key", "class_label"]].copy())
    out = pd.DataFrame(rows).sort_values("silhouette_by_broad_class", ascending=False).reset_index(drop=True)
    out.to_csv(TABLES_DIR / "diabetes_broad_label_transfer_comparison.csv", index=False)
    return out, pca_frames


def _effect_size(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 1e-12 else 0.0


def _latent_cluster_enrichment(merged_df: pd.DataFrame) -> pd.DataFrame:
    cohort_means = {axis: float(merged_df[axis].mean()) for axis in TEMP_AXES}
    rows = []
    for (broad, cluster), sub in merged_df.groupby(["class_label", "cluster_label"], sort=True):
        rows.append(
            {
                "cluster_label": str(cluster),
                "broad_class_label": str(broad),
                "n_samples": int(len(sub)),
                "mean_guanidino_response": float(sub["guanidino_response"].mean()),
                "mean_amide_stress_response": float(sub["amide_stress_response"].mean()),
                "cluster_vs_cohort_delta_guanidino": float(sub["guanidino_response"].mean() - cohort_means["guanidino_response"]),
                "cluster_vs_cohort_delta_amide": float(sub["amide_stress_response"].mean() - cohort_means["amide_stress_response"]),
                "interpretation_note": (
                    "guanidino-enriched"
                    if float(sub["guanidino_response"].mean() - cohort_means["guanidino_response"]) > 0
                    else "amide-shifted"
                    if float(sub["amide_stress_response"].mean() - cohort_means["amide_stress_response"]) > 0
                    else "cohort-like"
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values(["broad_class_label", "cluster_label"]).reset_index(drop=True)
    out.to_csv(TABLES_DIR / "diabetes_latent_cluster_axis_enrichment.csv", index=False)
    return out


def _latent_cluster_comparison(
    merged_df: pd.DataFrame,
    family_wide_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    pca_frames: dict[str, pd.DataFrame] = {}
    labels = merged_df["cluster_label"].astype(str).to_numpy()
    for kind in ["current_bsv", "family", "temporary_axes", "bsv_family", "bsv_family_temp"]:
        rep_df, cols = _representation_matrix(kind, merged_df, family_wide_df)
        X = StandardScaler().fit_transform(rep_df[cols].to_numpy(dtype=float))
        rows.append(
            {
                "representation_name": kind,
                "silhouette_by_latent_cluster": float(silhouette_score(X, labels)),
                "mean_centroid_distance": _mean_pairwise_centroid_distance(X, labels),
                "nearest_neighbor_agreement": _nn_purity(X, labels),
                "linear_cv_accuracy": _linear_cv_accuracy(X, labels),
            }
        )
        pca_frames[kind] = _pca_frame(
            rep_df,
            cols,
            merged_df[["sample_key", "class_label", "cluster_label"]].copy(),
        )
    out = pd.DataFrame(rows).sort_values("silhouette_by_latent_cluster", ascending=False).reset_index(drop=True)
    out.to_csv(TABLES_DIR / "diabetes_latent_cluster_representation_comparison.csv", index=False)
    return out, pca_frames


def _plot_cluster_axis_heatmap(enrich_df: pd.DataFrame) -> None:
    plot_df = enrich_df.set_index("cluster_label")[["mean_guanidino_response", "mean_amide_stress_response"]]
    matrix = plot_df.to_numpy(dtype=float)
    plt.figure(figsize=(6.2, 3.8))
    im = plt.imshow(matrix, aspect="auto", cmap="RdBu_r")
    plt.yticks(range(len(plot_df.index)), plot_df.index)
    plt.xticks(range(len(plot_df.columns)), ["guanidino", "amide"], rotation=25, ha="right")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Diabetes Latent Cluster Temporary-Axis Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "diabetes_latent_cluster_axis_heatmap.png", dpi=220)
    plt.close()


def _cross_class_alignment(
    merged_df: pd.DataFrame,
    match_df: pd.DataFrame,
) -> pd.DataFrame:
    cluster_means = (
        merged_df.groupby("cluster_label", as_index=False)[TEMP_AXES]
        .mean()
        .rename(columns={"cluster_label": "cluster"})
    )
    rows = []
    all_clusters = cluster_means["cluster"].astype(str).tolist()
    matched_pairs = set()
    for row in match_df.itertuples(index=False):
        matched_pairs.add((str(row.impact_cluster), str(row.strong_d_cluster)))
    for impact in [c for c in all_clusters if c.startswith("Impact_")]:
        irow = cluster_means[cluster_means["cluster"].astype(str) == impact].iloc[0]
        ivec = irow[TEMP_AXES].to_numpy(dtype=float)
        for strong in [c for c in all_clusters if c.startswith("Strong-D_")]:
            srow = cluster_means[cluster_means["cluster"].astype(str) == strong].iloc[0]
            svec = srow[TEMP_AXES].to_numpy(dtype=float)
            rows.append(
                {
                    "impact_cluster": impact,
                    "strong_d_cluster": strong,
                    "mean_guanidino_diff": float(abs(ivec[0] - svec[0])),
                    "mean_amide_diff": float(abs(ivec[1] - svec[1])),
                    "temp_axis_distance": float(np.linalg.norm(ivec - svec)),
                    "matched_pair": (impact, strong) in matched_pairs,
                }
            )
    out = pd.DataFrame(rows).sort_values(["matched_pair", "temp_axis_distance"], ascending=[False, True]).reset_index(drop=True)
    matched_mean = float(out.loc[out["matched_pair"], "temp_axis_distance"].mean())
    unmatched_mean = float(out.loc[~out["matched_pair"], "temp_axis_distance"].mean())
    out["matched_pair_mean_distance"] = matched_mean
    out["unmatched_pair_mean_distance"] = unmatched_mean
    out.to_csv(TABLES_DIR / "diabetes_cross_class_axis_alignment.csv", index=False)
    return out


def _plot_extended_radar(
    mean_df: pd.DataFrame,
    label_col: str,
    path: Path,
    title: str,
) -> None:
    plot_axes = FIXED_RADAR_AXES + TEMP_AXES
    plot_df = mean_df.copy()
    for axis in plot_axes:
        if axis not in plot_df.columns:
            plot_df[axis] = 0.0
    norm_df = plot_df.copy()
    for axis in plot_axes:
        vals = plot_df[axis].to_numpy(dtype=float)
        vmin, vmax = float(vals.min()), float(vals.max())
        if abs(vmax - vmin) < 1e-12:
            norm_df[axis] = 0.5
        else:
            norm_df[axis] = (vals - vmin) / (vmax - vmin)

    n = len(plot_axes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(len(norm_df), 1, subplot_kw={"polar": True}, figsize=(8.2, max(3.2, 2.8 * len(norm_df))))
    if len(norm_df) == 1:
        axes = [axes]
    for ax, row in zip(axes, norm_df.itertuples(index=False), strict=False):
        values = [float(getattr(row, axis)) for axis in plot_axes]
        values += values[:1]
        color = "#355070"
        ax.plot(angles, values, color=color, linewidth=1.8)
        ax.fill(angles, values, color=color, alpha=0.24)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(plot_axes, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticklabels([])
        ax.set_title(str(getattr(row, label_col)), y=1.08, fontsize=11)
    fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_report(
    verification_df: pd.DataFrame,
    broad_df: pd.DataFrame,
    enrich_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    align_df: pd.DataFrame,
    decision_label: str,
) -> Path:
    matched_rows = align_df[align_df["matched_pair"]].copy()
    matched_better = False
    if not matched_rows.empty:
        matched_mean = float(matched_rows["temp_axis_distance"].mean())
        unmatched_mean = float(align_df.loc[~align_df["matched_pair"], "temp_axis_distance"].mean())
        matched_better = matched_mean < unmatched_mean
    best_axis = "guanidino_response"
    if abs(float(enrich_df["mean_amide_stress_response"].mean())) > abs(float(enrich_df["mean_guanidino_response"].mean())):
        best_axis = "amide_stress_response"

    lines = [
        "# GAIRAv3 Pilot2.2 Diabetes Temporary Axis Transfer Report",
        "",
        "## 1. Why this test was needed",
        "",
        "- Diabetes EV is primarily a latent-state target rather than a strong broad-label separator.",
        "- SHINE extension testing suggested `guanidino_response` as the strongest temporary watchlist axis, with secondary `amide_stress_response` support.",
        "- This pass tests whether those temporary SHINE-derived axes add transfer value on the existing diabetes Pilot 2 / 2.1 sample and latent-state structure.",
        "",
        "## 2. Broad-label transfer result",
        "",
        _df_to_md(broad_df),
        "",
        "## 3. Latent-cluster transfer result",
        "",
        _df_to_md(latent_df),
        "",
        "### Cluster axis enrichment",
        "",
        _df_to_md(enrich_df),
        "",
        "## 4. Cross-class matched-state interpretation",
        "",
        _df_to_md(align_df[["impact_cluster", "strong_d_cluster", "mean_guanidino_diff", "mean_amide_diff", "temp_axis_distance", "matched_pair"]]),
        "",
        "## 5. Temporary extended radars",
        "",
        "- Broad-class and latent-cluster radars include the six cfg05 BSV axes plus temporary `guanidino_response` and `amide_stress_response` overlays.",
        "",
        "## 6. Final conclusion",
        "",
        f"- decision_label: `{decision_label}`",
        f"- most convincing temporary axis: `{best_axis}`",
        f"- broad-label gain remains {'weak' if float(broad_df['silhouette_by_broad_class'].max()) < 0.08 else 'present'}",
        f"- latent-cluster gain is {'present' if float(latent_df.iloc[0]['silhouette_by_latent_cluster']) > float(latent_df[latent_df['representation_name'] == 'current_bsv']['silhouette_by_latent_cluster'].iloc[0]) else 'not evident'}",
        f"- matched cross-class states are {'more similar' if matched_better else 'not clearly more similar'} in temporary-axis space than unmatched states",
        "",
        "Direct answers:",
        f"1. Is guanidino_response transferable? `{'yes, partially' if best_axis == 'guanidino_response' else 'weakly'}`",
        f"2. Is amide_stress_response transferable? `{'yes, partially' if best_axis == 'amide_stress_response' else 'weakly'}`",
        f"3. Should these remain temporary watchlist axes or be considered candidate GAIRA extension leads? `{'temporary watchlist axes' if decision_label != 'broad_and_latent_transfer' else 'candidate extension leads'}`",
        "",
        "## Appendix: Input verification",
        "",
        _df_to_md(verification_df),
    ]
    report_md = REPORT_DIR / "GAIRAv3_Pilot2_2_diabetes_temporary_axis_transfer_report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")
    decision_md = REPORT_DIR / "pilot2_2_transfer_decision.md"
    decision_md.write_text(
        "\n".join(
            [
                "# Pilot2.2 Transfer Decision",
                "",
                f"- decision_label: `{decision_label}`",
                f"- best broad-label representation: `{broad_df.iloc[0]['representation_name']}`",
                f"- best latent-cluster representation: `{latent_df.iloc[0]['representation_name']}`",
                f"- matched-state temporary-axis distance mean: `{float(align_df.loc[align_df['matched_pair'], 'temp_axis_distance'].mean()):.4f}`",
                f"- unmatched-state temporary-axis distance mean: `{float(align_df.loc[~align_df['matched_pair'], 'temp_axis_distance'].mean()):.4f}`",
            ]
        ),
        encoding="utf-8",
    )
    return report_md


def main() -> None:
    _ensure_dirs()

    bsv_df = _load_csv(PILOT2_ROOT, "per_sample_bsv.csv")
    delta_df = _load_csv(PILOT2_ROOT, "per_sample_delta_bsv.csv")
    family_df = _load_csv(PILOT2_ROOT, "sample_family_fingerprint.csv")
    sample_query_df = _load_csv(PILOT2_ROOT, "sample_query_spectra.csv")
    assign_df = _load_csv(PILOT21_ROOT, "latent_cluster_assignments.csv")
    match_df = _load_csv(PILOT21_ROOT, "cluster_cross_class_alignment.csv")
    candidate_df = pd.read_csv(EXT_ROOT / "tables" / "candidate_extension_axes.csv")

    temp_df = _compute_temp_axes_from_spectra(sample_query_df, candidate_df)
    temp_df = temp_df.merge(
        assign_df[["sample_key", "cluster_label"]].rename(columns={"cluster_label": "latent_cluster_label"}),
        on="sample_key",
        how="left",
    )
    temp_df.to_csv(TABLES_DIR / "diabetes_temporary_axes_per_sample.csv", index=False)

    cluster_means = (
        temp_df.groupby(["latent_cluster_label", "broad_class_label"], as_index=False)[TEMP_AXES]
        .mean()
        .sort_values(["broad_class_label", "latent_cluster_label"])
        .reset_index(drop=True)
    )
    cluster_means.to_csv(TABLES_DIR / "diabetes_temporary_axes_cluster_means.csv", index=False)

    reused_tables = [
        "pilot2_target_validation_v1/per_sample_bsv.csv",
        "pilot2_target_validation_v1/per_sample_delta_bsv.csv",
        "pilot2_target_validation_v1/sample_family_fingerprint.csv",
        "pilot2_target_validation_v1/sample_query_spectra.csv",
        "pilot2_1_latent_state_interpretation/latent_cluster_assignments.csv",
        "pilot2_1_latent_state_interpretation/cluster_cross_class_alignment.csv",
    ]
    verification_df = _write_input_verification(bsv_df, assign_df, reused_tables)

    merged_df = (
        bsv_df.merge(
            delta_df[["sample_key"] + [axis for axis in FIXED_RADAR_AXES if axis in delta_df.columns]].add_prefix("delta_"),
            left_on="sample_key",
            right_on="delta_sample_key",
            how="left",
        )
        .drop(columns=["delta_sample_key"])
        .merge(temp_df[["sample_key", "latent_cluster_label"] + TEMP_AXES], on="sample_key", how="left")
        .merge(assign_df[["sample_key", "cluster_label"]], on="sample_key", how="left")
    )
    merged_df["cluster_label"] = merged_df["cluster_label"].fillna(merged_df["latent_cluster_label"])
    family_wide_df = _family_wide(family_df, merged_df["sample_key"])

    broad_df, broad_pcas = _broad_label_comparison(merged_df, family_wide_df)
    _plot_pca(
        broad_pcas["temporary_axes"],
        "class_label",
        BROAD_COLORS,
        FIGURES_DIR / "diabetes_broad_label_axes_only_pca.png",
        "Diabetes Broad Labels: Temporary Axes PCA",
    )
    _plot_pca(
        broad_pcas["bsv_family_temp"],
        "class_label",
        BROAD_COLORS,
        FIGURES_DIR / "diabetes_broad_label_combined_pca.png",
        "Diabetes Broad Labels: BSV + Family + Temporary Axes PCA",
    )

    enrich_df = _latent_cluster_enrichment(merged_df)
    latent_df, latent_pcas = _latent_cluster_comparison(merged_df, family_wide_df)
    _plot_pca(
        latent_pcas["temporary_axes"],
        "cluster_label",
        {
            "Impact_latent_A": "#355070",
            "Impact_latent_B": "#b56576",
            "Strong-D_latent_A": "#2a9d8f",
            "Strong-D_latent_B": "#e76f51",
        },
        FIGURES_DIR / "diabetes_latent_cluster_axes_only_pca.png",
        "Diabetes Latent Clusters: Temporary Axes PCA",
    )
    _plot_pca(
        latent_pcas["bsv_family_temp"],
        "cluster_label",
        {
            "Impact_latent_A": "#355070",
            "Impact_latent_B": "#b56576",
            "Strong-D_latent_A": "#2a9d8f",
            "Strong-D_latent_B": "#e76f51",
        },
        FIGURES_DIR / "diabetes_latent_cluster_combined_pca.png",
        "Diabetes Latent Clusters: BSV + Family + Temporary Axes PCA",
    )
    _plot_cluster_axis_heatmap(enrich_df)

    align_df = _cross_class_alignment(merged_df, match_df)

    cluster_mean_extended = (
        merged_df.groupby("cluster_label", as_index=False)[[axis for axis in FIXED_RADAR_AXES if axis in merged_df.columns] + TEMP_AXES]
        .mean()
        .sort_values("cluster_label")
        .reset_index(drop=True)
    )
    broad_mean_extended = (
        merged_df.groupby("class_label", as_index=False)[[axis for axis in FIXED_RADAR_AXES if axis in merged_df.columns] + TEMP_AXES]
        .mean()
        .sort_values("class_label")
        .reset_index(drop=True)
    )
    _plot_extended_radar(
        cluster_mean_extended.rename(columns={"cluster_label": "label"}),
        "label",
        FIGURES_DIR / "diabetes_latent_cluster_extended_radars.png",
        "Diabetes Latent Cluster Temporary Extended Radars",
    )
    _plot_extended_radar(
        broad_mean_extended.rename(columns={"class_label": "label"}),
        "label",
        FIGURES_DIR / "diabetes_broad_class_extended_radars.png",
        "Diabetes Broad Class Temporary Extended Radars",
    )

    broad_best = float(broad_df.iloc[0]["silhouette_by_broad_class"])
    latent_base = float(latent_df[latent_df["representation_name"] == "current_bsv"]["silhouette_by_latent_cluster"].iloc[0])
    latent_best = float(latent_df.iloc[0]["silhouette_by_latent_cluster"])
    matched_mean = float(align_df.loc[align_df["matched_pair"], "temp_axis_distance"].mean())
    unmatched_mean = float(align_df.loc[~align_df["matched_pair"], "temp_axis_distance"].mean())

    if latent_best > latent_base + 0.015 and matched_mean < unmatched_mean:
        decision_label = "latent_only_transfer" if broad_best < 0.08 else "broad_and_latent_transfer"
    else:
        decision_label = "no_transfer"

    report_md = _build_report(verification_df, broad_df, enrich_df, latent_df, align_df, decision_label)
    figure_paths = sorted(FIGURES_DIR.glob("*.png"))
    build_pdf_report(
        report_md,
        figure_paths,
        REPORT_DIR / "GAIRAv3_Pilot2_2_diabetes_temporary_axis_transfer_report.pdf",
    )


if __name__ == "__main__":
    main()
