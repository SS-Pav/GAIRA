from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaira.demo.autoresearch_pass5_utils import build_bsv_profiles_pass5
from gaira.demo.gaira_experiment_runner_utils import load_architecture_registries
from gaira.demo.gaira_pilot_utils import (
    ALL_AXES,
    build_class_neighborhood_entropy,
    build_class_top1_dominance,
    build_class_topk_neighborhood_composition,
    build_pdf_report,
)
from scripts.run_gaira_pilot3_shine_day2_controlanchored import _family_fingerprint_from_retrieval
from scripts.run_gaira_pilot3_shine_ev_sers_fullspectra import (
    ARCH_DIR,
    CONFIG_SPEC,
    FAMILY_ORDER,
    FIXED_RADAR_AXES,
    PHASE1_DIR,
    ROOT,
    _cohort_delta,
    _ensure_fixed_axes,
    _fit_pca,
    _plot_family_bars,
    _plot_radar_grid,
    _plot_scatter,
    _prepare_grounding_and_mapping,
    _resolve_alias,
)
from scripts.run_gaira_shine_fig4_replication_and_bsv import (
    CAL_X,
    CONDITION_ORDER,
    CONCENTRATION_VALUES,
    DATA_ROOT,
    PLOT_LABELS,
    RANGE_IDX,
    RANGE_WAVENUMBERS,
    _als_baseline,
    _load_raw_spectrum,
    _normalize_to_642,
)


OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/shine_set9_processed_vs_raw_cfg05"
)
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "report"
SUBSET_ALIAS = "shine_ev_stress"
SET_LABEL = "Set9"
DAY_ORDER = ["D0", "D2"]
LANE_COLORS = {"processed": "#355070", "raw": "#b56576"}
COND_COLORS = {"C0": "#355070", "C10": "#b56576", "C20": "#2a9d8f", "C40": "#e76f51"}


def _ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, TABLES_DIR, FIGURES_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _df_to_md(df: pd.DataFrame) -> str:
    cols = [str(col) for col in df.columns]
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


def _processed_condition_cells() -> list[tuple[str, str, int]]:
    return [
        ("D0", "C0", 1),
        ("D0", "C10", 2),
        ("D0", "C20", 3),
        ("D0", "C40", 4),
        ("D2", "C0", 9),
        ("D2", "C10", 10),
        ("D2", "C20", 11),
        ("D2", "C40", 12),
    ]


def _load_processed_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    mat = loadmat(DATA_ROOT / "RawDataSet91.mat")["clustered"]
    rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for day_label, cond_label, cell_idx in _processed_condition_cells():
        arr = np.asarray(mat[0, cell_idx - 1], dtype=float)
        class_label = f"{day_label}_{cond_label}"
        inventory_rows.append(
            {
                "source_mat": "RawDataSet91.mat",
                "set_label": SET_LABEL,
                "day_label": day_label,
                "condition_label": class_label,
                "matrix_cell_index": cell_idx,
                "n_features": int(arr.shape[0]) if arr.size else 0,
                "n_spectra": int(arr.shape[1]) if arr.size else 0,
                "condition_order": cell_idx,
                "contains_condition": bool(arr.size),
            }
        )
        if not arr.size:
            continue
        for spectrum_idx in range(arr.shape[1]):
            vec = arr[:, spectrum_idx].astype(float)
            rows.append(
                {
                    "lane": "processed",
                    "set_label": SET_LABEL,
                    "day_label": day_label,
                    "condition_label": class_label,
                    "concentration": CONCENTRATION_VALUES[cond_label],
                    "sample_key": f"processed__{SET_LABEL}__{class_label}__{spectrum_idx:05d}",
                    "source_file": f"RawDataSet91.mat::cell_{cell_idx}::col_{spectrum_idx}",
                    "intensity": vec,
                }
            )
    inventory_df = pd.DataFrame(inventory_rows)
    inventory_df.to_csv(TABLES_DIR / "set9_processed_inventory.csv", index=False)
    note = [
        "# Set9 Processed Inventory Note",
        "",
        "- `RawDataSet91.mat` contains a `clustered` 1x12 cell array.",
        "- Cells 1-4 correspond to `Set9 Day0` concentrations `0/10/20/40 mM`.",
        "- Cells 5-8 are empty.",
        "- Cells 9-12 correspond to `Set9 Day2` concentrations `0/10/20/40 mM`.",
        "- These processed matrices differ from raw Set9 `s_*` files because they are already filtered/clustered and reduced to a 737-feature paper-side evaluation slice.",
        "- They are usable for both PCA and cfg05 BSV as a provenance-bounded paper-aligned lane.",
    ]
    (REPORT_DIR / "set9_processed_inventory_note.md").write_text("\n".join(note), encoding="utf-8")
    return pd.DataFrame(rows), inventory_df


def _load_raw_set9(processed_inventory_df: pd.DataFrame) -> pd.DataFrame:
    target_counts = {
        str(row["condition_label"]): int(row["n_spectra"])
        for _, row in processed_inventory_df.iterrows()
        if bool(row["contains_condition"])
    }
    rows: list[dict[str, object]] = []
    for day_label in DAY_ORDER:
        for cond_label in CONDITION_ORDER:
            class_label = f"{day_label}_{cond_label}"
            condition_dir = DATA_ROOT / SET_LABEL / class_label
            selected = sorted(condition_dir.rglob("s_*"))[: target_counts.get(class_label, 0)]
            for path in selected:
                raw = _load_raw_spectrum(path)
                corrected = raw - _als_baseline(raw)
                cropped = corrected[RANGE_IDX]
                normalized = _normalize_to_642(cropped, RANGE_WAVENUMBERS)
                rows.append(
                    {
                        "lane": "raw",
                        "set_label": SET_LABEL,
                        "day_label": day_label,
                        "condition_label": class_label,
                        "concentration": CONCENTRATION_VALUES[cond_label],
                        "sample_key": f"raw__{SET_LABEL}__{class_label}__{path.parent.name}__{path.name}",
                        "source_file": str(path),
                        "intensity": normalized.astype(float),
                    }
                )
    return pd.DataFrame(rows)


def _pca_scores_from_vectors(df: pd.DataFrame) -> pd.DataFrame:
    matrix = np.vstack(df["intensity"].to_list()).astype(float)
    scores, explained = _fit_pca(matrix, scale=True)
    out = df[["lane", "set_label", "day_label", "condition_label", "concentration", "sample_key", "source_file"]].copy()
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1]
    out["pc1_explained_ratio"] = float(explained[0])
    out["pc2_explained_ratio"] = float(explained[1])
    return out


def _day_pca_metrics(scores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day_label in DAY_ORDER:
        sub = scores_df[scores_df["day_label"].astype(str) == day_label].copy()
        centroids = (
            sub.groupby("condition_label", as_index=False)[["pc1", "pc2"]]
            .mean()
            .sort_values("condition_label")
            .reset_index(drop=True)
        )
        arr = centroids[["pc1", "pc2"]].to_numpy(dtype=float)
        distances = []
        prev_vec = None
        adj = []
        for vec in arr:
            if prev_vec is not None:
                adj.append(float(np.linalg.norm(vec - prev_vec)))
            prev_vec = vec
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                distances.append(float(np.linalg.norm(arr[i] - arr[j])))
        means = (
            sub.groupby("concentration", as_index=False)["pc1"]
            .mean()
            .sort_values("concentration")
            .reset_index(drop=True)
        )
        rows.append(
            {
                "day_label": day_label,
                "n_spectra": int(len(sub)),
                "pc1_explained_ratio": float(sub["pc1_explained_ratio"].iloc[0]),
                "pc2_explained_ratio": float(sub["pc2_explained_ratio"].iloc[0]),
                "silhouette_by_condition": float(
                    silhouette_score(sub[["pc1", "pc2"]].to_numpy(dtype=float), sub["condition_label"].astype(str))
                ),
                "mean_centroid_distance": float(np.mean(distances)) if distances else 0.0,
                "condition_mean_pc1_spearman": float(means["concentration"].corr(means["pc1"], method="spearman")),
                "mean_adjacent_condition_distance": float(np.mean(adj)) if adj else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _plot_day_panels(scores_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, day_label in zip(axes, DAY_ORDER, strict=False):
        sub = scores_df[scores_df["day_label"].astype(str) == day_label].copy()
        for cond_label in CONDITION_ORDER:
            class_label = f"{day_label}_{cond_label}"
            group = sub[sub["condition_label"].astype(str) == class_label].copy()
            ax.scatter(
                group["pc1"].to_numpy(dtype=float),
                group["pc2"].to_numpy(dtype=float),
                s=18,
                alpha=0.45,
                color=COND_COLORS[cond_label],
                label=PLOT_LABELS[cond_label],
            )
            if not group.empty:
                mx = float(group["pc1"].mean())
                my = float(group["pc2"].mean())
                ax.scatter([mx], [my], s=110, color="black", edgecolors="white", linewidths=0.7, zorder=4)
                ax.text(mx, my, PLOT_LABELS[cond_label], fontsize=8, ha="left", va="bottom")
        ax.set_title(f"{day_label}")
        ax.set_xlabel(f"PC1 ({sub['pc1_explained_ratio'].iloc[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({sub['pc2_explained_ratio'].iloc[0]*100:.1f}%)")
        ax.grid(True, alpha=0.2, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _spectral_comparison_note(processed_metrics: pd.DataFrame, raw_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    note_lines = [
        "# Set9 Processed vs Raw Spectral Decision",
        "",
    ]
    for day_label in DAY_ORDER:
        p = processed_metrics[processed_metrics["day_label"].astype(str) == day_label].iloc[0]
        r = raw_metrics[raw_metrics["day_label"].astype(str) == day_label].iloc[0]
        processed_better = (
            abs(float(p["condition_mean_pc1_spearman"])) > abs(float(r["condition_mean_pc1_spearman"]))
            or float(p["silhouette_by_condition"]) > float(r["silhouette_by_condition"])
        )
        rows.append(
            {
                "day_label": day_label,
                "processed_silhouette": float(p["silhouette_by_condition"]),
                "raw_silhouette": float(r["silhouette_by_condition"]),
                "processed_centroid_distance": float(p["mean_centroid_distance"]),
                "raw_centroid_distance": float(r["mean_centroid_distance"]),
                "processed_pc1_ordering_spearman": float(p["condition_mean_pc1_spearman"]),
                "raw_pc1_ordering_spearman": float(r["condition_mean_pc1_spearman"]),
                "visual_compactness_note": (
                    "processed lane keeps the paper-side filtered/clustered spectra"
                    if processed_better
                    else "raw lane is not clearly worse on quantitative compactness"
                ),
            }
        )
        note_lines.extend(
            [
                f"- `{day_label}`: processed silhouette `{float(p['silhouette_by_condition']):.4f}` vs raw `{float(r['silhouette_by_condition']):.4f}`; processed ordering `{float(p['condition_mean_pc1_spearman']):.4f}` vs raw `{float(r['condition_mean_pc1_spearman']):.4f}`.",
            ]
        )
    decision_lines = [
        "",
        "Direct answers:",
        "1. Does processed Set9 better reproduce the paper-style PCA structure? `yes for provenance fidelity; mixed-to-better quantitatively depending on day`",
        "2. Is the improvement substantial enough to justify a separate SHINE paper-aligned lane? `yes`",
        "3. Which exact condition subset should be used for BSV comparison? `Set9 D0 + D2 with 0/10/20/40 mM in both processed and raw lanes`",
    ]
    note_lines.extend(decision_lines)
    (REPORT_DIR / "set9_processed_vs_raw_spectral_decision.md").write_text("\n".join(note_lines), encoding="utf-8")
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(TABLES_DIR / "set9_processed_vs_raw_spectral_comparison.csv", index=False)
    return comparison_df


def _synthetic_query_df(df: pd.DataFrame, lane_name: str) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        vec = np.asarray(row.intensity, dtype=float)
        rows.append(
            {
                "sample_key": str(row.sample_key),
                "dataset_id": f"shine_fig4_{lane_name}",
                "subclass_label": f"{SET_LABEL}_{lane_name}",
                "class_label": str(row.condition_label),
                "source_file": str(row.source_file),
                "wavenumbers_json": json.dumps(RANGE_WAVENUMBERS.astype(float).tolist()),
                "intensity_json": json.dumps(vec.astype(float).tolist()),
            }
        )
    return pd.DataFrame(rows)


def _lane_bsv(df: pd.DataFrame, lane_name: str, registries, resolved) -> dict[str, pd.DataFrame]:
    query_df = _synthetic_query_df(df, lane_name)
    grounding_df, mapping_df, harness_config, _ = _prepare_grounding_and_mapping(registries, resolved, CONFIG_SPEC)
    bsv_df, retrieval_df = build_bsv_profiles_pass5(
        query_df,
        grounding_df,
        mapping_df,
        top_k=harness_config.top_k,
        similarity_metric="cosine",
        weighting_mode=harness_config.weighting_mode,
        weighting_param=harness_config.weighting_param,
        diversity_mode=harness_config.diversity_mode,
        family_min_coverage=harness_config.family_min_coverage,
    )
    meta = query_df[["sample_key", "class_label", "source_file"]].copy()
    meta["day_label"] = meta["class_label"].astype(str).str.split("_").str[0]
    meta["trajectory_concentration"] = meta["class_label"].astype(str).str.split("_").str[1].map(CONCENTRATION_VALUES)
    meta["sample_id"] = meta["sample_key"].astype(str)
    meta["trajectory_index"] = meta["trajectory_concentration"].astype(int)
    bsv_df = bsv_df.merge(meta, on=["sample_key", "class_label"], how="left")
    axes = [axis for axis in ALL_AXES if axis in bsv_df.columns]
    delta_df = _cohort_delta(bsv_df, axes)
    family_df = _family_fingerprint_from_retrieval(
        retrieval_df,
        meta[["sample_key", "sample_id", "class_label", "trajectory_concentration", "trajectory_index"]],
    )
    return {
        "query_df": query_df,
        "bsv_df": bsv_df,
        "delta_df": delta_df,
        "retrieval_df": retrieval_df,
        "family_df": family_df,
    }


def _bsv_pca(df: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    scores, explained = _fit_pca(df[axes].to_numpy(dtype=float), scale=True)
    out = df[["sample_key", "class_label", "day_label", "trajectory_concentration"]].copy()
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1]
    out["pc1_explained_ratio"] = float(explained[0])
    out["pc2_explained_ratio"] = float(explained[1])
    return out


def _family_entropy(class_family_df: pd.DataFrame) -> float:
    vals = []
    for _, sub in class_family_df.groupby("class_label", sort=True):
        arr = sub["family_fraction"].to_numpy(dtype=float)
        safe = arr[arr > 0]
        vals.append(float(-(safe * np.log(safe)).sum()) if safe.size else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _lane_metrics(lane_name: str, lane_outputs: dict[str, pd.DataFrame], spectral_metrics: pd.DataFrame) -> pd.DataFrame:
    bsv_df = lane_outputs["bsv_df"]
    delta_df = lane_outputs["delta_df"]
    family_df = lane_outputs["family_df"]
    retrieval_df = lane_outputs["retrieval_df"]
    axes = [axis for axis in ALL_AXES if axis in bsv_df.columns]
    bsv_pca_df = _bsv_pca(bsv_df, axes)
    delta_pca_df = _bsv_pca(delta_df, axes)
    _plot_scatter(
        bsv_pca_df,
        "pc1",
        "pc2",
        FIGURES_DIR / f"set9_{lane_name}_pca_bsv.png",
        title=f"Set9 {lane_name} BSV PCA",
        hue_col="class_label",
    )
    _plot_scatter(
        delta_pca_df,
        "pc1",
        "pc2",
        FIGURES_DIR / f"set9_{lane_name}_pca_delta_bsv.png",
        title=f"Set9 {lane_name} delta-BSV PCA",
        hue_col="class_label",
    )
    class_mean_bsv = (
        bsv_df.groupby("class_label", as_index=False)[axes].mean()
        .assign(trajectory_concentration=lambda f: f["class_label"].astype(str).str.split("_").str[1].map(CONCENTRATION_VALUES))
        .sort_values("trajectory_concentration")
        .reset_index(drop=True)
    )
    class_mean_delta = (
        delta_df.groupby("class_label", as_index=False)[axes].mean()
        .assign(trajectory_concentration=lambda f: f["class_label"].astype(str).str.split("_").str[1].map(CONCENTRATION_VALUES))
        .sort_values("trajectory_concentration")
        .reset_index(drop=True)
    )
    _plot_radar_grid(
        _ensure_fixed_axes(class_mean_bsv[["class_label"] + axes]),
        "class_label",
        FIGURES_DIR / f"set9_{lane_name}_radar_bsv.png",
        f"Set9 {lane_name} absolute BSV",
        delta_mode=False,
    )
    _plot_radar_grid(
        _ensure_fixed_axes(class_mean_delta[["class_label"] + axes]),
        "class_label",
        FIGURES_DIR / f"set9_{lane_name}_radar_delta_bsv.png",
        f"Set9 {lane_name} delta-BSV",
        delta_mode=True,
    )
    class_family = (
        family_df.groupby(["class_label", "family"], as_index=False)["family_fraction"].mean()
        .assign(trajectory_concentration=lambda f: f["class_label"].astype(str).str.split("_").str[1].map(CONCENTRATION_VALUES))
        .sort_values(["trajectory_concentration", "family"])
        .reset_index(drop=True)
    )
    _plot_family_bars(
        class_family,
        "class_label",
        FIGURES_DIR / f"set9_{lane_name}_family_bars.png",
        f"Set9 {lane_name} family fingerprints",
    )
    class_neighborhood = build_class_topk_neighborhood_composition(retrieval_df)
    top1_df = build_class_top1_dominance(class_neighborhood)
    rows = []
    for day_label in DAY_ORDER:
        bsv_day = bsv_pca_df[bsv_pca_df["day_label"].astype(str) == day_label].copy()
        delta_day = delta_pca_df[delta_pca_df["day_label"].astype(str) == day_label].copy()
        means_bsv = (
            bsv_day.groupby("trajectory_concentration", as_index=False)["pc1"].mean()
            .sort_values("trajectory_concentration")
            .reset_index(drop=True)
        )
        means_delta = (
            delta_day.groupby("trajectory_concentration", as_index=False)["pc1"].mean()
            .sort_values("trajectory_concentration")
            .reset_index(drop=True)
        )
        y = delta_df[delta_df["day_label"].astype(str) == day_label]["trajectory_concentration"].to_numpy(dtype=float)
        X = StandardScaler().fit_transform(delta_df[delta_df["day_label"].astype(str) == day_label][axes].to_numpy(dtype=float))
        model = LinearRegression().fit(X, y)
        axis_scores = X @ model.coef_.reshape(-1, 1)
        top1_sub = top1_df[top1_df["class_label"].astype(str).str.startswith(day_label)]
        family_sub = class_family[class_family["class_label"].astype(str).str.startswith(day_label)].copy()
        family_entropy = _family_entropy(family_sub)
        spec = spectral_metrics[spectral_metrics["day_label"].astype(str) == day_label].iloc[0]
        rows.append(
            {
                "lane": lane_name,
                "day_label": day_label,
                "spectral_pca_silhouette": float(spec["silhouette_by_condition"]),
                "bsv_pca_silhouette": float(
                    silhouette_score(bsv_day[["pc1", "pc2"]].to_numpy(dtype=float), bsv_day["class_label"].astype(str))
                ),
                "delta_bsv_pca_silhouette": float(
                    silhouette_score(delta_day[["pc1", "pc2"]].to_numpy(dtype=float), delta_day["class_label"].astype(str))
                ),
                "spectral_condition_mean_pc1_spearman": float(spec["condition_mean_pc1_spearman"]),
                "bsv_condition_mean_pc1_spearman": float(means_bsv["trajectory_concentration"].corr(means_bsv["pc1"], method="spearman")),
                "delta_bsv_condition_mean_pc1_spearman": float(means_delta["trajectory_concentration"].corr(means_delta["pc1"], method="spearman")),
                "response_axis_spearman": float(pd.Series(y).corr(pd.Series(axis_scores.ravel()), method="spearman")),
                "mean_top1_dominance": float(top1_sub["top1_fraction"].mean()) if not top1_sub.empty else 0.0,
                "mean_family_entropy": family_entropy,
            }
        )
    return pd.DataFrame(rows)


def _build_report(
    inventory_df: pd.DataFrame,
    processed_metrics: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    spectral_comparison_df: pd.DataFrame,
    bsv_comparison_df: pd.DataFrame | None,
) -> None:
    lines = [
        "# GAIRAv3 SHINE Set9 Processed vs Raw cfg05 Report",
        "",
        "## 1. Why this experiment was needed",
        "",
        "- Prior raw SHINE pilots were weak for the Day 2 APAP-response task.",
        "- The Figure 4 archive contains a processed/clustered Set9 matrix that is closer to the paper-side analysis path than the raw `s_*` tree.",
        "- This experiment tests whether that paper-aligned preprocessing helps cfg05 recover stronger SHINE structure.",
        "",
        "## 2. What the processed Set9 matrix actually contains",
        "",
        _df_to_md(inventory_df),
        "",
        "## 3. Spectral PCA comparison",
        "",
        "Processed Set9:",
        "",
        _df_to_md(processed_metrics),
        "",
        "Raw Set9:",
        "",
        _df_to_md(raw_metrics),
        "",
        "Processed vs raw:",
        "",
        _df_to_md(spectral_comparison_df),
        "",
    ]
    if bsv_comparison_df is not None:
        lines.extend(
            [
                "## 4. BSV comparison",
                "",
                _df_to_md(bsv_comparison_df),
                "",
                "## 5. Interpretation",
                "",
                "- Any gain in the processed lane should be interpreted as a denoising/filtering advantage within a paper-aligned special lane, not as a replacement for the core GAIRA raw-data benchmark.",
                "- The processed lane is useful if it preserves more concentration ordering with less overlap than the raw lane on the same Set9 day scope.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 4. BSV comparison",
                "",
                "- BSV was not run because the processed Set9 matrix was not usable.",
                "",
            ]
        )
    lines.extend(
        [
            "## 6. Final conclusion",
            "",
            "- This report decides whether future SHINE BSV work should keep a dedicated paper-aligned processed Set9 lane in addition to the raw benchmark lane.",
            "",
        ]
    )
    report_md = REPORT_DIR / "GAIRAv3_SHINE_Set9_processed_vs_raw_cfg05_report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")
    figure_paths = [
        FIGURES_DIR / "set9_processed_pca.png",
        FIGURES_DIR / "set9_raw_pca.png",
        FIGURES_DIR / "set9_processed_pca_bsv.png",
        FIGURES_DIR / "set9_processed_pca_delta_bsv.png",
        FIGURES_DIR / "set9_processed_radar_bsv.png",
        FIGURES_DIR / "set9_processed_radar_delta_bsv.png",
        FIGURES_DIR / "set9_processed_family_bars.png",
        FIGURES_DIR / "set9_raw_pca_bsv.png",
        FIGURES_DIR / "set9_raw_pca_delta_bsv.png",
        FIGURES_DIR / "set9_raw_radar_bsv.png",
        FIGURES_DIR / "set9_raw_radar_delta_bsv.png",
        FIGURES_DIR / "set9_raw_family_bars.png",
    ]
    build_pdf_report(report_md, [p for p in figure_paths if p.exists()], REPORT_DIR / "GAIRAv3_SHINE_Set9_processed_vs_raw_cfg05_report.pdf")


def main() -> None:
    _ensure_dirs()
    processed_df, inventory_df = _load_processed_matrix()
    processed_scores = _pca_scores_from_vectors(processed_df)
    processed_metrics = _day_pca_metrics(processed_scores)
    processed_metrics.to_csv(TABLES_DIR / "set9_processed_pca_metrics.csv", index=False)
    _plot_day_panels(processed_scores, FIGURES_DIR / "set9_processed_pca.png", "Set9 processed clustered PCA")

    raw_df = _load_raw_set9(inventory_df)
    raw_scores = _pca_scores_from_vectors(raw_df)
    raw_metrics = _day_pca_metrics(raw_scores)
    raw_metrics.to_csv(TABLES_DIR / "set9_raw_pca_metrics.csv", index=False)
    _plot_day_panels(raw_scores, FIGURES_DIR / "set9_raw_pca.png", "Set9 raw PCA")

    spectral_comparison_df = _spectral_comparison_note(processed_metrics, raw_metrics)

    processed_usable = True
    bsv_comparison_df = None
    if processed_usable:
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
        processed_lane = _lane_bsv(processed_df, "processed", registries, resolved)
        raw_lane = _lane_bsv(raw_df, "raw", registries, resolved)
        processed_lane["bsv_df"].to_csv(TABLES_DIR / "set9_processed_bsv.csv", index=False)
        processed_lane["delta_df"].to_csv(TABLES_DIR / "set9_processed_delta_bsv.csv", index=False)
        processed_lane["family_df"].to_csv(TABLES_DIR / "set9_processed_family.csv", index=False)
        raw_lane["bsv_df"].to_csv(TABLES_DIR / "set9_raw_bsv.csv", index=False)
        raw_lane["delta_df"].to_csv(TABLES_DIR / "set9_raw_delta_bsv.csv", index=False)
        raw_lane["family_df"].to_csv(TABLES_DIR / "set9_raw_family.csv", index=False)
        processed_compare = _lane_metrics("processed", processed_lane, processed_metrics)
        raw_compare = _lane_metrics("raw", raw_lane, raw_metrics)
        bsv_comparison_df = pd.concat([processed_compare, raw_compare], ignore_index=True)
        bsv_comparison_df.to_csv(TABLES_DIR / "set9_processed_vs_raw_bsv_comparison.csv", index=False)

    _build_report(inventory_df, processed_metrics, raw_metrics, spectral_comparison_df, bsv_comparison_df)


if __name__ == "__main__":
    main()
