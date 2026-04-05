from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaira.demo.autoresearch_pass5_utils import build_bsv_profiles_pass5
from gaira.demo.gaira_experiment_runner_utils import load_architecture_registries
from gaira.demo.gaira_pilot_utils import ALL_AXES, build_pdf_report
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
    _prepare_grounding_and_mapping,
    _resolve_alias,
)
from scripts.run_gaira_shine_fig4_replication_and_bsv import (
    CONDITION_ORDER,
    CONCENTRATION_VALUES,
    DATA_ROOT,
    OUTPUT_ROOT as FIG4_ROOT,
    RANGE_WAVENUMBERS,
    _build_raw_sampled_subset,
)


OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/shine_set10_day2_spectral_axis_to_bsv"
)
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "report"
SUBSET_ALIAS = "shine_ev_stress"
VALIDATED_SUBSET = "Set10_D2_raw_sampled"
DAY_LABEL = "D2"
COND_ORDER = ["C0", "C10", "C20", "C40"]
COND_LABELS = [f"{DAY_LABEL}_{cond}" for cond in COND_ORDER]
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


def _load_validated_subset() -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_text = (FIG4_ROOT / "report" / "fig4_pca_replication_decision.md").read_text(encoding="utf-8")
    if VALIDATED_SUBSET.replace("_raw_sampled", "") not in decision_text:
        raise RuntimeError("Validated subset mismatch: Set10_D2 was not found in the prior decision note")
    spectral_df, used_df = _build_raw_sampled_subset("Set10", "D2")
    spectral_df = spectral_df.copy()
    spectral_df["class_label"] = f"{DAY_LABEL}_" + spectral_df["condition_label"].astype(str)
    spectral_df["source_file"] = used_df["source_file"].astype(str).tolist()
    spectral_df["sample_key"] = [
        f"set10_day2_raw_sampled__{Path(path).relative_to(DATA_ROOT).as_posix().replace('/', '__')}"
        for path in spectral_df["source_file"].astype(str)
    ]
    spectral_df["trajectory_concentration"] = spectral_df["concentration"].astype(int)
    return spectral_df, used_df


def _write_input_verification(spectral_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in COND_LABELS:
        sub = spectral_df[spectral_df["class_label"].astype(str) == cond].copy()
        rows.append(
            {
                "validated_subset": VALIDATED_SUBSET,
                "class_label": cond,
                "n_spectra": int(len(sub)),
                "n_unique_source_paths": int(sub["source_file"].astype(str).nunique()),
                "example_source_file": str(sub["source_file"].astype(str).iloc[0]) if not sub.empty else "",
            }
        )
    verification_df = pd.DataFrame(rows)
    verification_df.to_csv(TABLES_DIR / "set10_day2_input_verification.csv", index=False)
    return verification_df


def _build_spectral_matrix(spectral_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feature_cols = [col for col in spectral_df.columns if str(col).startswith("f")]
    matrix_df = spectral_df[["sample_key"] + feature_cols].copy()
    matrix_df = matrix_df.rename(columns={col: f"wn_{int(round(float(RANGE_WAVENUMBERS[int(col[1:])])))}" for col in feature_cols})
    spectral_cols = [col for col in matrix_df.columns if col.startswith("wn_")]
    matrix_df.to_csv(TABLES_DIR / "set10_day2_spectral_matrix.csv", index=False)
    metadata_df = spectral_df[
        ["sample_key", "class_label", "trajectory_concentration", "source_file"]
    ].copy()
    metadata_df.to_csv(TABLES_DIR / "set10_day2_metadata.csv", index=False)
    return matrix_df, metadata_df, spectral_cols


def _spectral_pca(scores_input: np.ndarray, metadata_df: pd.DataFrame) -> pd.DataFrame:
    scores, explained = _fit_pca(scores_input, scale=True)
    out = metadata_df.copy()
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1]
    out["pc1_explained_ratio"] = float(explained[0])
    out["pc2_explained_ratio"] = float(explained[1])
    return out


def _axis_metrics(scores: np.ndarray, concentrations: np.ndarray, class_labels: np.ndarray) -> dict[str, float]:
    means = (
        pd.DataFrame({"concentration": concentrations, "score": scores})
        .groupby("concentration", as_index=False)["score"]
        .mean()
        .sort_values("concentration")
        .reset_index(drop=True)
    )
    adjacent = np.diff(means["score"].to_numpy(dtype=float))
    return {
        "spearman_concentration": float(pd.Series(concentrations).corr(pd.Series(scores), method="spearman")),
        "pearson_concentration": float(pd.Series(concentrations).corr(pd.Series(scores), method="pearson")),
        "condition_mean_ordering_spearman": float(means["concentration"].corr(means["score"], method="spearman")),
        "silhouette_by_concentration": float(
            silhouette_score(np.column_stack([scores, np.zeros_like(scores)]), class_labels.astype(str))
        ),
        "mean_adjacent_distance": float(np.mean(np.abs(adjacent))) if len(adjacent) else 0.0,
        "min_adjacent_distance": float(np.min(np.abs(adjacent))) if len(adjacent) else 0.0,
        "monotonicity_score": float(np.mean(np.sign(adjacent) == np.sign(np.mean(adjacent)))) if len(adjacent) else 0.0,
    }


def _candidate_axes(matrix: np.ndarray, concentrations: np.ndarray, class_labels: np.ndarray, spectral_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = StandardScaler().fit_transform(matrix)
    pca_scores, _ = _fit_pca(X, scale=False)
    pca_scores_1d = pca_scores[:, 0]
    u, s, vh = np.linalg.svd(X - X.mean(axis=0, keepdims=True), full_matrices=False)
    pca_loading = vh[0]

    reg = LinearRegression().fit(X, concentrations)
    reg_scores = reg.predict(X)
    reg_loading = reg.coef_.astype(float)

    c40 = X[concentrations == 40].mean(axis=0)
    c0 = X[concentrations == 0].mean(axis=0)
    diff_loading = (c40 - c0).astype(float)
    diff_scores = X @ diff_loading

    axis_specs = [
        ("pca_pc1", pca_scores_1d, pca_loading),
        ("linear_regression", reg_scores, reg_loading),
        ("control_anchored_difference", diff_scores, diff_loading),
    ]
    metric_rows = []
    score_rows = []
    loading_rows = []
    for axis_name, scores, loadings in axis_specs:
        metrics = _axis_metrics(np.asarray(scores, dtype=float), concentrations, class_labels)
        metric_rows.append({"axis_name": axis_name, **metrics})
        for idx, score in enumerate(scores):
            score_rows.append(
                {
                    "axis_name": axis_name,
                    "sample_index": idx,
                    "class_label": str(class_labels[idx]),
                    "trajectory_concentration": int(concentrations[idx]),
                    "axis_score": float(score),
                }
            )
        for feature_name, loading in zip(spectral_cols, loadings, strict=False):
            wn = int(feature_name.replace("wn_", ""))
            loading_rows.append(
                {
                    "axis_name": axis_name,
                    "feature_name": feature_name,
                    "wavenumber": wn,
                    "loading": float(loading),
                }
            )
    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["spearman_concentration", "condition_mean_ordering_spearman", "mean_adjacent_distance"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    scores_df = pd.DataFrame(score_rows)
    loadings_df = pd.DataFrame(loading_rows)
    metrics_df.to_csv(TABLES_DIR / "spectral_response_axis_metrics.csv", index=False)
    scores_df.to_csv(TABLES_DIR / "spectral_response_axis_scores.csv", index=False)
    loadings_df.to_csv(TABLES_DIR / "spectral_response_axis_loadings.csv", index=False)
    return metrics_df, scores_df


def _choose_best_axis(metrics_df: pd.DataFrame) -> str:
    metrics_df = metrics_df.copy()
    best = metrics_df.sort_values(
        [
            "spearman_concentration",
            "condition_mean_ordering_spearman",
            "silhouette_by_concentration",
            "mean_adjacent_distance",
        ],
        ascending=[False, False, False, False],
    ).iloc[0]
    lines = [
        "# Set10 Day2 Best Spectral Axis Decision",
        "",
        f"- winning axis: `{best['axis_name']}`",
        f"- concentration Spearman: `{best['spearman_concentration']:.4f}`",
        f"- condition-mean ordering Spearman: `{best['condition_mean_ordering_spearman']:.4f}`",
        f"- silhouette by concentration: `{best['silhouette_by_concentration']:.4f}`",
        "",
        "Decision:",
        "- The winning axis was chosen primarily by concentration correlation, then condition ordering, then separation quality, and only after that by adjacent spacing.",
        "- This confirms whether a Day-2 APAP signal is actually present in spectral space on the validated subset.",
    ]
    (REPORT_DIR / "set10_day2_best_spectral_axis_decision.md").write_text("\n".join(lines), encoding="utf-8")
    return str(best["axis_name"])


def _build_query_df(spectral_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in spectral_df.columns if str(col).startswith("f")]
    rows = []
    for row in spectral_df.itertuples(index=False):
        vec = np.array([getattr(row, col) for col in feature_cols], dtype=float)
        rows.append(
            {
                "sample_key": str(row.sample_key),
                "dataset_id": "shine_ev_sers",
                "subclass_label": "Set10",
                "class_label": str(row.class_label),
                "source_file": str(row.source_file),
                "wavenumbers_json": json.dumps(RANGE_WAVENUMBERS.astype(float).tolist()),
                "intensity_json": json.dumps(vec.tolist()),
            }
        )
    return pd.DataFrame(rows)


def _run_bsv(spectral_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    query_df = _build_query_df(spectral_df)
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
    meta = spectral_df[["sample_key", "class_label", "trajectory_concentration", "source_file"]].copy()
    meta["sample_id"] = meta["sample_key"].astype(str)
    meta["trajectory_index"] = meta["trajectory_concentration"].astype(int)
    bsv_df = bsv_df.merge(meta, on=["sample_key", "class_label"], how="left")
    axes = [axis for axis in ALL_AXES if axis in bsv_df.columns]
    delta_df = _cohort_delta(bsv_df, axes)
    family_df = _family_fingerprint_from_retrieval(
        retrieval_df,
        meta[["sample_key", "sample_id", "class_label", "trajectory_concentration", "trajectory_index"]],
    )
    bsv_df.to_csv(TABLES_DIR / "set10_day2_bsv.csv", index=False)
    delta_df.to_csv(TABLES_DIR / "set10_day2_delta_bsv.csv", index=False)
    family_df.to_csv(TABLES_DIR / "set10_day2_family.csv", index=False)
    return bsv_df, delta_df, family_df


def _axis_correlations(
    chosen_axis: str,
    scores_df: pd.DataFrame,
    bsv_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    family_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    axis_scores = scores_df[scores_df["axis_name"].astype(str) == chosen_axis][
        ["sample_index", "class_label", "trajectory_concentration", "axis_score"]
    ].copy()
    axis_scores["sample_key"] = bsv_df["sample_key"].astype(str).tolist()
    merged = axis_scores.merge(
        bsv_df[["sample_key"] + [axis for axis in FIXED_RADAR_AXES if axis in bsv_df.columns]],
        on="sample_key",
        how="left",
    ).merge(
        delta_df[["sample_key"] + [axis for axis in FIXED_RADAR_AXES if axis in delta_df.columns]].rename(
            columns={axis: f"delta_{axis}" for axis in FIXED_RADAR_AXES if axis in delta_df.columns}
        ),
        on="sample_key",
        how="left",
    )
    bsv_rows = []
    for axis in FIXED_RADAR_AXES:
        if axis in merged.columns:
            bsv_rows.append(
                {
                    "feature_name": axis,
                    "feature_group": "bsv",
                    "pearson_r": float(merged["axis_score"].corr(merged[axis], method="pearson")),
                    "spearman_r": float(merged["axis_score"].corr(merged[axis], method="spearman")),
                }
            )
        delta_axis = f"delta_{axis}"
        if delta_axis in merged.columns:
            bsv_rows.append(
                {
                    "feature_name": delta_axis,
                    "feature_group": "delta_bsv",
                    "pearson_r": float(merged["axis_score"].corr(merged[delta_axis], method="pearson")),
                    "spearman_r": float(merged["axis_score"].corr(merged[delta_axis], method="spearman")),
                }
            )

    family_wide = family_df.pivot(index="sample_key", columns="family", values="family_fraction").reset_index()
    family_merged = axis_scores.merge(family_wide, on="sample_key", how="left")
    family_top1 = family_df.groupby("sample_key", as_index=False)["family_fraction"].max().rename(columns={"family_fraction": "top1_dominance"})
    family_entropy = (
        family_df.groupby("sample_key", as_index=False)
        .apply(lambda sub: float(-(sub["family_fraction"].to_numpy(dtype=float)[sub["family_fraction"].to_numpy(dtype=float) > 0] * np.log(sub["family_fraction"].to_numpy(dtype=float)[sub["family_fraction"].to_numpy(dtype=float) > 0])).sum()))
        .reset_index()
    )
    family_entropy.columns = ["_drop", "sample_key", "family_entropy"]
    family_entropy = family_entropy.drop(columns="_drop")
    family_merged = family_merged.merge(family_top1, on="sample_key", how="left").merge(family_entropy, on="sample_key", how="left")

    family_rows = []
    for family in FAMILY_ORDER + ["top1_dominance", "family_entropy"]:
        if family in family_merged.columns:
            family_rows.append(
                {
                    "feature_name": family,
                    "pearson_r": float(family_merged["axis_score"].corr(family_merged[family], method="pearson")),
                    "spearman_r": float(family_merged["axis_score"].corr(family_merged[family], method="spearman")),
                }
            )
    bsv_corr_df = pd.DataFrame(bsv_rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    family_corr_df = pd.DataFrame(family_rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    bsv_corr_df.to_csv(TABLES_DIR / "spectral_axis_to_bsv_correlations.csv", index=False)
    family_corr_df.to_csv(TABLES_DIR / "spectral_axis_to_family_correlations.csv", index=False)
    return bsv_corr_df, family_corr_df


def _top_regions(loadings_df: pd.DataFrame, chosen_axis: str, bsv_corr_df: pd.DataFrame, family_corr_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = loadings_df[loadings_df["axis_name"].astype(str) == chosen_axis].copy().sort_values("wavenumber").reset_index(drop=True)
    abs_cutoff = float(sub["loading"].abs().quantile(0.95))
    region_rows = []
    for sign_name, sign in [("positive", 1), ("negative", -1)]:
        sign_df = sub[sub["loading"] * sign > 0].copy()
        sign_df = sign_df[sign_df["loading"].abs() >= abs_cutoff].copy()
        if sign_df.empty:
            continue
        groups = []
        current = [int(sign_df["wavenumber"].iloc[0])]
        for wn in sign_df["wavenumber"].astype(int).tolist()[1:]:
            if wn - current[-1] <= 4:
                current.append(int(wn))
            else:
                groups.append(current)
                current = [int(wn)]
        groups.append(current)
        for idx, group in enumerate(groups, start=1):
            group_df = sign_df[sign_df["wavenumber"].isin(group)].copy()
            peak_row = group_df.iloc[group_df["loading"].abs().argmax()]
            region_rows.append(
                {
                    "axis_name": chosen_axis,
                    "region_id": f"{sign_name}_{idx}",
                    "sign": sign_name,
                    "start_wavenumber": min(group),
                    "end_wavenumber": max(group),
                    "peak_wavenumber": int(peak_row["wavenumber"]),
                    "mean_loading": float(group_df["loading"].mean()),
                    "max_abs_loading": float(group_df["loading"].abs().max()),
                }
            )
    regions_df = pd.DataFrame(region_rows).sort_values(["sign", "peak_wavenumber"]).reset_index(drop=True)
    regions_df.to_csv(TABLES_DIR / "spectral_axis_key_regions.csv", index=False)

    top_bsv = bsv_corr_df.iloc[0]["feature_name"] if not bsv_corr_df.empty else "unknown"
    top_family = family_corr_df.iloc[0]["feature_name"] if not family_corr_df.empty else "unknown"
    mapping_rows = []
    for _, row in regions_df.iterrows():
        mapping_rows.append(
            {
                "region_id": row["region_id"],
                "sign": row["sign"],
                "peak_wavenumber": row["peak_wavenumber"],
                "suggested_bsv_theme": top_bsv,
                "suggested_family_theme": top_family,
                "notes": "broad-theme mapping based on global spectral-axis correlations; molecule-specific attribution not claimed",
            }
        )
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(TABLES_DIR / "spectral_axis_region_bsv_mapping.csv", index=False)
    return regions_df, mapping_df


def _explainability(
    chosen_axis: str,
    scores_df: pd.DataFrame,
    bsv_df: pd.DataFrame,
    family_df: pd.DataFrame,
) -> pd.DataFrame:
    target = scores_df[scores_df["axis_name"].astype(str) == chosen_axis]["axis_score"].to_numpy(dtype=float)
    axes = [axis for axis in FIXED_RADAR_AXES if axis in bsv_df.columns]
    X_bsv = bsv_df[axes].to_numpy(dtype=float)
    family_wide = family_df.pivot(index="sample_key", columns="family", values="family_fraction").reindex(
        bsv_df["sample_key"].astype(str)
    )[FAMILY_ORDER].to_numpy(dtype=float)
    specs = [
        ("bsv_only", X_bsv),
        ("family_only", family_wide),
        ("bsv_plus_family", np.hstack([X_bsv, family_wide])),
    ]
    rows = []
    residual_best = None
    best_name = None
    for name, X in specs:
        Xs = StandardScaler().fit_transform(X)
        model = LinearRegression().fit(Xs, target)
        pred = model.predict(Xs)
        r2 = float(r2_score(target, pred))
        spearman = float(pd.Series(target).corr(pd.Series(pred), method="spearman"))
        rmse = float(np.sqrt(np.mean((target - pred) ** 2)))
        rows.append(
            {
                "model_name": name,
                "r2": r2,
                "spearman_r": spearman,
                "rmse": rmse,
            }
        )
        if residual_best is None or r2 > max(r["r2"] for r in rows[:-1] or [{"r2": -np.inf}]):
            residual_best = target - pred
            best_name = name
    explain_df = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    explain_df.to_csv(TABLES_DIR / "spectral_axis_explainability_by_bsv.csv", index=False)
    return explain_df, residual_best, best_name


def _refinement_hints(explain_df: pd.DataFrame, regions_df: pd.DataFrame, bsv_corr_df: pd.DataFrame, family_corr_df: pd.DataFrame) -> None:
    best_r2 = float(explain_df["r2"].max())
    hints = ["# Set10 Day2 BSV Refinement Hints", ""]
    if best_r2 >= 0.7:
        hints.append("- Current BSV vocabulary already explains most of the spectral response axis; no urgent refinement is indicated from SHINE alone.")
    else:
        hints.append("- Current BSV/family space only partially explains the spectral response axis, so SHINE is likely pointing to a missing finer-grained response theme.")
        top_bsv = bsv_corr_df.iloc[0]["feature_name"] if not bsv_corr_df.empty else "small_molecule_metabolite"
        top_family = family_corr_df.iloc[0]["feature_name"] if not family_corr_df.empty else "generic_other_metabolite"
        hints.append(f"- Strongest current alignment is with `{top_bsv}` and family signal `{top_family}`, but residual structure remains.")
        if any((regions_df["peak_wavenumber"] >= 1200) & (regions_df["peak_wavenumber"] <= 1700)):
            hints.append("- Candidate refinement hint: create a finer protein/amide-stress sub-axis rather than relying only on the coarse `protein_peptide` bucket.")
        if any((regions_df["peak_wavenumber"] >= 600) & (regions_df["peak_wavenumber"] <= 900)):
            hints.append("- Candidate refinement hint: split current nucleic-acid / aromatic / guanidino response themes where low-wavenumber stress bands dominate the axis.")
        hints.append("- Candidate refinement hint: subdivide `small_molecule_metabolite` into response-linked subfamilies if future datasets repeat this residual structure.")
    (REPORT_DIR / "set10_day2_bsv_refinement_hints.md").write_text("\n".join(hints), encoding="utf-8")


def _plot_spectral_pca(pca_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for cond in COND_ORDER:
        label = f"{DAY_LABEL}_{cond}"
        sub = pca_df[pca_df["class_label"].astype(str) == label].copy()
        ax.scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=24,
            alpha=0.5,
            color=COND_COLORS[cond],
            label=f"{CONCENTRATION_VALUES[cond]} mM",
        )
    ax.set_xlabel(f"PC1 ({pca_df['pc1_explained_ratio'].iloc[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_df['pc2_explained_ratio'].iloc[0]*100:.1f}%)")
    ax.set_title("Set10 Day2 spectral PCA")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=[0.0, 0.0, 0.84, 1.0])
    fig.savefig(FIGURES_DIR / "set10_day2_spectral_pca.png", dpi=240)
    plt.close(fig)


def _plot_best_axis(
    chosen_axis: str,
    scores_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    bsv_corr_df: pd.DataFrame,
    family_corr_df: pd.DataFrame,
    bsv_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    family_df: pd.DataFrame,
    explain_df: pd.DataFrame,
    residuals: np.ndarray,
    spectral_pca_df: pd.DataFrame,
) -> None:
    sub = scores_df[scores_df["axis_name"].astype(str) == chosen_axis].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    data = [sub[sub["trajectory_concentration"] == c]["axis_score"].to_numpy(dtype=float) for c in [0, 10, 20, 40]]
    ax.boxplot(data, tick_labels=["0", "10", "20", "40"])
    ax.set_xlabel("Concentration (mM)")
    ax.set_ylabel("Axis score")
    ax.set_title(f"{chosen_axis} scores by concentration")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "set10_day2_best_response_axis_scores.png", dpi=240)
    plt.close(fig)

    means = sub.groupby("trajectory_concentration", as_index=False)["axis_score"].mean().sort_values("trajectory_concentration")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(means["trajectory_concentration"], means["axis_score"], marker="o", linewidth=2.2, color="#355070")
    ax.set_xlabel("Concentration (mM)")
    ax.set_ylabel("Mean axis score")
    ax.set_title(f"{chosen_axis} concentration trend")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "set10_day2_best_response_axis_trend.png", dpi=240)
    plt.close(fig)

    load_sub = loadings_df[loadings_df["axis_name"].astype(str) == chosen_axis].copy()
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(load_sub["wavenumber"].to_numpy(dtype=float), load_sub["loading"].to_numpy(dtype=float), color="#2a9d8f", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Axis loading")
    ax.set_title(f"{chosen_axis} spectral loadings")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "set10_day2_best_response_axis_loadings.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    top = bsv_corr_df.head(10).copy()
    ax.barh(top["feature_name"], top["spearman_r"], color="#355070")
    ax.invert_yaxis()
    ax.set_xlabel("Spearman r")
    ax.set_title("Spectral axis vs BSV axes")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "set10_day2_spectral_axis_vs_bsv_axes.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    topf = family_corr_df.head(10).copy()
    ax.barh(topf["feature_name"], topf["spearman_r"], color="#b56576")
    ax.invert_yaxis()
    ax.set_xlabel("Spearman r")
    ax.set_title("Spectral axis vs family features")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "set10_day2_spectral_axis_vs_family_axes.png", dpi=240)
    plt.close(fig)

    axes = [axis for axis in FIXED_RADAR_AXES if axis in bsv_df.columns]
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
        FIGURES_DIR / "set10_day2_bsv_radar_by_concentration.png",
        "Set10 Day2 absolute BSV by concentration",
        delta_mode=False,
    )
    _plot_radar_grid(
        _ensure_fixed_axes(class_mean_delta[["class_label"] + axes]),
        "class_label",
        FIGURES_DIR / "set10_day2_delta_bsv_radar_by_concentration.png",
        "Set10 Day2 delta-BSV by concentration",
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
        FIGURES_DIR / "set10_day2_family_bars_by_concentration.png",
        "Set10 Day2 family fingerprints by concentration",
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(explain_df["model_name"], explain_df["r2"], color=["#355070", "#b56576", "#2a9d8f"])
    ax.set_ylabel("R^2")
    ax.set_title("Explainability of spectral axis by current GAIRA space")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "set10_day2_spectral_axis_explainability.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    scatter = ax.scatter(
        spectral_pca_df["pc1"].to_numpy(dtype=float),
        spectral_pca_df["pc2"].to_numpy(dtype=float),
        c=residuals,
        cmap="coolwarm",
        s=24,
        alpha=0.6,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Residual structure after BSV explainability")
    ax.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Residual")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "set10_day2_residual_structure.png", dpi=240)
    plt.close(fig)


def _build_report(
    verification_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    chosen_axis: str,
    bsv_corr_df: pd.DataFrame,
    family_corr_df: pd.DataFrame,
    explain_df: pd.DataFrame,
    regions_df: pd.DataFrame,
) -> None:
    report_lines = [
        "# GAIRAv3 SHINE Set10 D2 Spectral Axis to BSV Report",
        "",
        "## 1. Why this experiment was needed",
        "",
        "- The prior Figure 4 replication was only partial, but Set10 Day2 gave the strongest paper-aligned subset.",
        "- BSV preserved some concentration ordering without producing strong separation.",
        "- This experiment tests whether the Day-2 SHINE signal is already inside current GAIRA vocabulary or points to a missing finer-grained response axis.",
        "",
        "## 2. Verified subset",
        "",
        _df_to_md(verification_df),
        "",
        "## 3. Spectral response axis",
        "",
        _df_to_md(metrics_df),
        "",
        f"- Winning spectral axis: `{chosen_axis}`",
        "",
        "## 4. Mapping into current BSV space",
        "",
        "Top BSV correlations:",
        "",
        _df_to_md(bsv_corr_df.head(10)),
        "",
        "Top family correlations:",
        "",
        _df_to_md(family_corr_df.head(10)),
        "",
        "## 5. Explainability result",
        "",
        _df_to_md(explain_df),
        "",
        "## 6. Candidate refinement hints",
        "",
        (REPORT_DIR / "set10_day2_bsv_refinement_hints.md").read_text(encoding="utf-8"),
        "",
        "## 7. Final conclusion",
        "",
        f"- Is the SHINE Day-2 APAP signal real in spectral space? `{'yes' if metrics_df.iloc[0]['spearman_concentration'] and abs(metrics_df.iloc[0]['spearman_concentration']) > 0.3 else 'weak but present'}`",
        f"- Does current BSV capture it well, partially, or poorly? `{'well' if explain_df['r2'].max() >= 0.7 else 'partially' if explain_df['r2'].max() >= 0.35 else 'poorly'}`",
        f"- Does SHINE justify a future refinement of GAIRA vocabulary? `{'yes' if explain_df['r2'].max() < 0.7 else 'not strongly from this subset alone'}`",
        "",
        "Key spectral regions:",
        "",
        _df_to_md(regions_df),
    ]
    report_md = REPORT_DIR / "GAIRAv3_SHINE_Set10_D2_spectral_axis_to_bsv_report.md"
    report_md.write_text("\n".join(report_lines), encoding="utf-8")
    figure_paths = [
        FIGURES_DIR / "set10_day2_spectral_pca.png",
        FIGURES_DIR / "set10_day2_best_response_axis_scores.png",
        FIGURES_DIR / "set10_day2_best_response_axis_trend.png",
        FIGURES_DIR / "set10_day2_best_response_axis_loadings.png",
        FIGURES_DIR / "set10_day2_spectral_axis_vs_bsv_axes.png",
        FIGURES_DIR / "set10_day2_spectral_axis_vs_family_axes.png",
        FIGURES_DIR / "set10_day2_bsv_radar_by_concentration.png",
        FIGURES_DIR / "set10_day2_delta_bsv_radar_by_concentration.png",
        FIGURES_DIR / "set10_day2_family_bars_by_concentration.png",
        FIGURES_DIR / "set10_day2_spectral_axis_explainability.png",
        FIGURES_DIR / "set10_day2_residual_structure.png",
    ]
    build_pdf_report(report_md, [p for p in figure_paths if p.exists()], REPORT_DIR / "GAIRAv3_SHINE_Set10_D2_spectral_axis_to_bsv_report.pdf")


def main() -> None:
    _ensure_dirs()
    spectral_df, used_df = _load_validated_subset()
    verification_df = _write_input_verification(spectral_df)
    matrix_df, metadata_df, spectral_cols = _build_spectral_matrix(spectral_df)
    matrix = matrix_df[spectral_cols].to_numpy(dtype=float)
    concentrations = metadata_df["class_label"].astype(str).str.split("_").str[1].map(CONCENTRATION_VALUES).to_numpy(dtype=float)
    class_labels = metadata_df["class_label"].astype(str).to_numpy()
    pca_df = _spectral_pca(matrix, metadata_df.copy())
    _plot_spectral_pca(pca_df)
    metrics_df, scores_df = _candidate_axes(matrix, concentrations, class_labels, spectral_cols)
    chosen_axis = _choose_best_axis(metrics_df)
    bsv_df, delta_df, family_df = _run_bsv(spectral_df)
    bsv_corr_df, family_corr_df = _axis_correlations(chosen_axis, scores_df, bsv_df, delta_df, family_df)
    loadings_df = pd.read_csv(TABLES_DIR / "spectral_response_axis_loadings.csv")
    regions_df, mapping_df = _top_regions(loadings_df, chosen_axis, bsv_corr_df, family_corr_df)
    explain_df, residuals, best_name = _explainability(chosen_axis, scores_df, bsv_df, family_df)
    _refinement_hints(explain_df, regions_df, bsv_corr_df, family_corr_df)
    _plot_best_axis(
        chosen_axis,
        scores_df,
        loadings_df,
        bsv_corr_df,
        family_corr_df,
        bsv_df,
        delta_df,
        family_df,
        explain_df,
        residuals,
        pca_df,
    )
    _build_report(verification_df, metrics_df, chosen_axis, bsv_corr_df, family_corr_df, explain_df, regions_df)


if __name__ == "__main__":
    main()
