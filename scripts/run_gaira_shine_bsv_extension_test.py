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
    RANGE_WAVENUMBERS,
    _build_raw_sampled_subset,
)


OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/shine_bsv_extension_test"
)
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "report"

SET10_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/shine_set10_day2_spectral_axis_to_bsv"
)
SET10_VALIDATED_LABEL = "Set10_D2_raw_sampled"
SUBSET_ALIAS = "shine_ev_stress"
WINNING_AXIS = "linear_regression"
SET10_DAY_LABEL = "D2"
SET9_MATCHED_LABEL = "Set9_D2_matched"
SET9_D0_LABEL = "Set9_D0_matched"


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


def _load_set10_discovery_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_df = pd.read_csv(SET10_ROOT / "tables" / "spectral_response_axis_metrics.csv")
    best_axis = str(metrics_df.sort_values("spearman_concentration", ascending=False).iloc[0]["axis_name"])
    if best_axis != WINNING_AXIS:
        raise RuntimeError(f"Expected winning spectral axis {WINNING_AXIS}, found {best_axis}")
    matrix_df = pd.read_csv(SET10_ROOT / "tables" / "set10_day2_spectral_matrix.csv")
    metadata_df = pd.read_csv(SET10_ROOT / "tables" / "set10_day2_metadata.csv")
    scores_df = pd.read_csv(SET10_ROOT / "tables" / "spectral_response_axis_scores.csv")
    scores_df = scores_df[scores_df["axis_name"].astype(str) == WINNING_AXIS].copy().reset_index(drop=True)
    scores_df["sample_key"] = metadata_df["sample_key"].astype(str).tolist()
    bsv_df = pd.read_csv(SET10_ROOT / "tables" / "set10_day2_bsv.csv")
    family_df = pd.read_csv(SET10_ROOT / "tables" / "set10_day2_family.csv")
    return matrix_df, metadata_df, scores_df, bsv_df, family_df


def _candidate_axis_specs() -> pd.DataFrame:
    rows = [
        {
            "candidate_axis_name": "purine_methylation_response",
            "positive_regions": "1490-1497;1502-1506;1524-1528",
            "negative_regions": "1486-1488;1492-1494;1529-1531",
            "anchor_peaks": "1491,1496,1503,1528,1530",
            "rationale": "High-wavenumber methylation-like contrast derived from the strongest Set10 Day2 linear spectral loadings around 1490-1530.",
            "overlaps_existing_bsv": "nucleic_acid;small_molecule_metabolite",
            "overlaps_existing_family": "methylated_purine_like;purine_core_like",
        },
        {
            "candidate_axis_name": "aromatic_stress_response",
            "positive_regions": "904-907;945-948;990-993",
            "negative_regions": "888-892",
            "anchor_peaks": "890,906,947,992",
            "rationale": "Mid-wavenumber contrast grounded in the 890 vs 906/947/992 loading structure, treated as a temporary aromatic-stress proxy.",
            "overlaps_existing_bsv": "small_molecule_metabolite",
            "overlaps_existing_family": "aromatic_small_molecule_like",
        },
        {
            "candidate_axis_name": "guanidino_response",
            "positive_regions": "1157-1159;1370-1373",
            "negative_regions": "1154-1156",
            "anchor_peaks": "1155,1158,1372",
            "rationale": "Narrow local contrast around 1155-1158 with an additional 1372 anchor, aligned to the weak guanidino family hint without changing ontology.",
            "overlaps_existing_bsv": "small_molecule_metabolite;nucleic_acid",
            "overlaps_existing_family": "guanidine_like",
        },
        {
            "candidate_axis_name": "amide_stress_response",
            "positive_regions": "1543-1545;1558-1560;1578-1580;1604-1606",
            "negative_regions": "1515-1517;1551-1553;1574-1576",
            "anchor_peaks": "1544,1552,1559,1576,1580,1605",
            "rationale": "High-wavenumber stress-like contour spanning 1544-1605, derived from the strongest positive and negative Day2 loadings in that band.",
            "overlaps_existing_bsv": "protein_peptide",
            "overlaps_existing_family": "generic_other_metabolite;guanidine_like",
        },
        {
            "candidate_axis_name": "low_wavenumber_stress_response",
            "positive_regions": "513-515",
            "negative_regions": "484-487;511-512",
            "anchor_peaks": "486,512,513",
            "rationale": "Derivative-like low-wavenumber contrast around 486-513 taken directly from the top opposing Day2 loadings.",
            "overlaps_existing_bsv": "nucleic_acid;small_molecule_metabolite;substrate_adsorption_bias",
            "overlaps_existing_family": "purine_core_like;methylated_purine_like",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "candidate_extension_axes.csv", index=False)
    lines = [
        "# Candidate Extension Axis Design",
        "",
        "- These are temporary SHINE extension axes derived from the validated Set10 Day2 linear spectral-response loadings.",
        "- They are implemented as simple signed band-contrast features and do not modify the base GAIRA ontology.",
        "",
        _df_to_md(df),
    ]
    (REPORT_DIR / "candidate_extension_axis_design.md").write_text("\n".join(lines), encoding="utf-8")
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


def _spectral_feature_columns(matrix_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    spectral_cols = [c for c in matrix_df.columns if str(c).startswith("wn_")]
    wns = np.array([int(str(c).replace("wn_", "")) for c in spectral_cols], dtype=int)
    return spectral_cols, wns


def _region_mean(matrix: np.ndarray, wns: np.ndarray, start: int, end: int) -> np.ndarray:
    mask = (wns >= start) & (wns <= end)
    if not np.any(mask):
        return np.zeros(matrix.shape[0], dtype=float)
    return matrix[:, mask].mean(axis=1)


def _compute_extension_features(
    matrix_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    spectral_cols, wns = _spectral_feature_columns(matrix_df)
    matrix = matrix_df[spectral_cols].to_numpy(dtype=float)
    out = metadata_df.copy()
    for row in candidate_df.itertuples(index=False):
        pos_regions = _parse_regions(row.positive_regions)
        neg_regions = _parse_regions(row.negative_regions)
        pos_stack = np.vstack([_region_mean(matrix, wns, a, b) for a, b in pos_regions]) if pos_regions else np.zeros((1, len(out)))
        neg_stack = np.vstack([_region_mean(matrix, wns, a, b) for a, b in neg_regions]) if neg_regions else np.zeros((1, len(out)))
        pos_score = pos_stack.mean(axis=0)
        neg_score = neg_stack.mean(axis=0)
        out[row.candidate_axis_name] = pos_score - neg_score
    out.to_csv(TABLES_DIR / f"{prefix}_extension_features.csv", index=False)
    mean_df = (
        out.groupby("class_label", as_index=False)[candidate_df["candidate_axis_name"].tolist()]
        .mean()
        .assign(
            trajectory_concentration=lambda f: f["class_label"].astype(str).str.split("_").str[1].str.replace("C", "").astype(int)
        )
        .sort_values("trajectory_concentration")
        .reset_index(drop=True)
    )
    mean_df.to_csv(TABLES_DIR / f"{prefix}_extension_feature_means.csv", index=False)
    return out


def _family_wide(family_df: pd.DataFrame, sample_keys: pd.Series) -> pd.DataFrame:
    wide = family_df.pivot(index="sample_key", columns="family", values="family_fraction")
    wide = wide.reindex(sample_keys.astype(str)).fillna(0.0)
    for family in FAMILY_ORDER:
        if family not in wide.columns:
            wide[family] = 0.0
    return wide[FAMILY_ORDER].reset_index()


def _dominance_entropy(family_df: pd.DataFrame, sample_keys: pd.Series) -> pd.DataFrame:
    rows = []
    for sample_key in sample_keys.astype(str):
        sub = family_df[family_df["sample_key"].astype(str) == sample_key].copy()
        vals = sub["family_fraction"].to_numpy(dtype=float)
        vals = vals[vals > 0]
        rows.append(
            {
                "sample_key": sample_key,
                "top1_dominance": float(sub["family_fraction"].max()) if len(sub) else 0.0,
                "family_entropy": float(-(vals * np.log(vals)).sum()) if len(vals) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _explainability_table(
    target_scores: np.ndarray,
    bsv_df: pd.DataFrame,
    family_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    *,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    axes = [axis for axis in FIXED_RADAR_AXES if axis in bsv_df.columns]
    bsv_features = bsv_df[["sample_key"] + axes].copy()
    family_wide = _family_wide(family_df, bsv_df["sample_key"])
    ext_cols = [c for c in ext_df.columns if c not in {"sample_key", "class_label", "trajectory_concentration", "source_file"}]
    ext_features = ext_df[["sample_key"] + ext_cols].copy()

    merged = (
        bsv_features.merge(family_wide, on="sample_key", how="left")
        .merge(ext_features, on="sample_key", how="left")
        .fillna(0.0)
    )
    feature_blocks = {
        "bsv_only": axes,
        "family_only": FAMILY_ORDER,
        "bsv_plus_family": axes + FAMILY_ORDER,
        "extension_only": ext_cols,
        "bsv_plus_family_plus_extension": axes + FAMILY_ORDER + ext_cols,
    }
    rows = []
    pred_cols = {"sample_key": merged["sample_key"].astype(str).tolist()}
    for model_name, cols in feature_blocks.items():
        X = merged[cols].to_numpy(dtype=float)
        Xs = StandardScaler().fit_transform(X)
        model = LinearRegression().fit(Xs, target_scores)
        pred = model.predict(Xs)
        rows.append(
            {
                "model_name": model_name,
                "r2": float(r2_score(target_scores, pred)),
                "spearman_r": float(pd.Series(target_scores).corr(pd.Series(pred), method="spearman")),
                "pearson_r": float(pd.Series(target_scores).corr(pd.Series(pred), method="pearson")),
                "rmse": float(np.sqrt(np.mean((target_scores - pred) ** 2))),
            }
        )
        pred_cols[model_name] = pred.tolist()
    explain_df = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    pred_df = pd.DataFrame(pred_cols)
    explain_df.to_csv(TABLES_DIR / f"{prefix}_extension_explainability.csv", index=False)
    return explain_df, pred_df


def _representation_metrics(
    target_scores: np.ndarray,
    concentrations: np.ndarray,
    class_labels: np.ndarray,
    bsv_df: pd.DataFrame,
    family_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    *,
    prefix: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    axes = [axis for axis in FIXED_RADAR_AXES if axis in bsv_df.columns]
    family_wide = _family_wide(family_df, bsv_df["sample_key"])
    ext_cols = [c for c in ext_df.columns if c not in {"sample_key", "class_label", "trajectory_concentration", "source_file"}]
    merged = (
        bsv_df[["sample_key"] + axes]
        .merge(family_wide, on="sample_key", how="left")
        .merge(ext_df[["sample_key"] + ext_cols], on="sample_key", how="left")
        .fillna(0.0)
    )
    specs = {
        "current_bsv": axes,
        "current_bsv_plus_family": axes + FAMILY_ORDER,
        "extension_only": ext_cols,
        "current_bsv_plus_family_plus_extension": axes + FAMILY_ORDER + ext_cols,
    }
    rows = []
    score_frames: dict[str, pd.DataFrame] = {}
    for name, cols in specs.items():
        X = merged[cols].to_numpy(dtype=float)
        scores, explained = _fit_pca(X, scale=True)
        score_df = pd.DataFrame(
            {
                "sample_key": merged["sample_key"].astype(str).tolist(),
                "class_label": class_labels.astype(str),
                "trajectory_concentration": concentrations.astype(int),
                "pc1": scores[:, 0],
                "pc2": scores[:, 1],
                "pc1_explained_ratio": float(explained[0]),
                "pc2_explained_ratio": float(explained[1]),
            }
        )
        score_frames[name] = score_df
        silhouette = float(silhouette_score(score_df[["pc1", "pc2"]].to_numpy(dtype=float), score_df["class_label"].astype(str)))
        means = (
            score_df.groupby("trajectory_concentration", as_index=False)["pc1"].mean().sort_values("trajectory_concentration")
        )
        ordering = float(means["trajectory_concentration"].corr(means["pc1"], method="spearman"))
        response_corr = float(pd.Series(target_scores).corr(score_df["pc1"], method="spearman"))
        centroids = (
            score_df.groupby("trajectory_concentration", as_index=False)[["pc1", "pc2"]]
            .mean()
            .sort_values("trajectory_concentration")
            .reset_index(drop=True)
        )
        adj = []
        arr = centroids[["pc1", "pc2"]].to_numpy(dtype=float)
        for i in range(1, len(arr)):
            adj.append(float(np.linalg.norm(arr[i] - arr[i - 1])))
        rows.append(
            {
                "representation_name": name,
                "pca_silhouette_by_concentration": silhouette,
                "condition_mean_ordering_spearman": ordering,
                "response_axis_spearman": response_corr,
                "mean_adjacent_distance": float(np.mean(adj)) if adj else 0.0,
                "min_adjacent_distance": float(np.min(adj)) if adj else 0.0,
            }
        )
    rep_df = pd.DataFrame(rows).sort_values("response_axis_spearman", ascending=False).reset_index(drop=True)
    rep_df.to_csv(TABLES_DIR / f"{prefix}_representation_comparison.csv", index=False)
    return rep_df, score_frames


def _plot_representation_pca(score_frames: dict[str, pd.DataFrame], output_path: Path, titles: list[str]) -> None:
    fig, axes = plt.subplots(1, len(titles), figsize=(7.2 * len(titles), 5.6), squeeze=False)
    for ax, name in zip(axes.ravel(), titles, strict=False):
        score_df = score_frames[name]
        for cond_label in CONDITION_ORDER:
            class_label = f"D2_{cond_label}"
            sub = score_df[score_df["class_label"].astype(str) == class_label].copy()
            ax.scatter(
                sub["pc1"].to_numpy(dtype=float),
                sub["pc2"].to_numpy(dtype=float),
                s=18,
                alpha=0.55,
                label=f"{CONCENTRATION_VALUES[cond_label]} mM",
            )
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center right")
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_extension_trends(mean_df: pd.DataFrame, axis_names: list[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = mean_df["trajectory_concentration"].to_numpy(dtype=float)
    for axis_name in axis_names:
        ax.plot(x, mean_df[axis_name].to_numpy(dtype=float), marker="o", linewidth=2.0, label=axis_name)
    ax.set_xlabel("Concentration (mM)")
    ax.set_ylabel("Mean extension score")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_explainability(explain_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.bar(explain_df["model_name"], explain_df["r2"], color=["#355070", "#b56576", "#6d597a", "#2a9d8f", "#e76f51"])
    ax.set_ylabel("R^2")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _build_query_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    fcols = [c for c in feature_df.columns if str(c).startswith("f")]
    rows = []
    for row in feature_df.itertuples(index=False):
        vec = np.array([getattr(row, c) for c in fcols], dtype=float)
        rows.append(
            {
                "sample_key": str(row.sample_key),
                "dataset_id": "shine_ev_sers",
                "subclass_label": str(row.subclass_label),
                "class_label": str(row.class_label),
                "source_file": str(row.source_file),
                "wavenumbers_json": json.dumps(RANGE_WAVENUMBERS.astype(float).tolist()),
                "intensity_json": json.dumps(vec.tolist()),
            }
        )
    return pd.DataFrame(rows)


def _run_bsv_on_feature_df(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    query_df = _build_query_df(feature_df)
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
    meta = feature_df[["sample_key", "class_label", "trajectory_concentration", "source_file"]].copy()
    meta["sample_id"] = meta["sample_key"].astype(str)
    meta["trajectory_index"] = meta["trajectory_concentration"].astype(int)
    bsv_df = bsv_df.merge(meta, on=["sample_key", "class_label"], how="left")
    family_df = _family_fingerprint_from_retrieval(
        retrieval_df,
        meta[["sample_key", "sample_id", "class_label", "trajectory_concentration", "trajectory_index"]],
    )
    return bsv_df, family_df


def _build_set9_matched_subsets(set10_counts: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    set9_d2, _ = _build_raw_sampled_subset("Set9", "D2")
    set9_d0, _ = _build_raw_sampled_subset("Set9", "D0")
    source_counts_d2 = set9_d2.groupby("condition_label").size().to_dict()
    matched_counts = {cond: min(int(set10_counts.get(f"D2_{cond}", 0)), int(source_counts_d2.get(cond, 0))) for cond in CONDITION_ORDER}
    d2_rows = []
    for cond in CONDITION_ORDER:
        sub = set9_d2[set9_d2["condition_label"].astype(str) == cond].copy().sort_values("source_file").head(matched_counts[cond])
        sub["sample_key"] = [
            f"set9_d2_matched__{cond}__{Path(path).name}__{idx}" for idx, path in enumerate(sub["source_file"].astype(str).tolist())
        ]
        sub["class_label"] = f"D2_{cond}"
        sub["trajectory_concentration"] = CONCENTRATION_VALUES[cond]
        sub["subclass_label"] = "Set9"
        d2_rows.append(sub)
    matched_d2 = pd.concat(d2_rows, ignore_index=True)

    d0_rows = []
    for cond in CONDITION_ORDER:
        target_n = min(int(matched_counts.get(cond, 0)), int((set9_d0["condition_label"].astype(str) == cond).sum()))
        sub = set9_d0[set9_d0["condition_label"].astype(str) == cond].copy().sort_values("source_file").head(target_n)
        sub["sample_key"] = [
            f"set9_d0_matched__{cond}__{Path(path).name}__{idx}" for idx, path in enumerate(sub["source_file"].astype(str).tolist())
        ]
        sub["class_label"] = f"D0_{cond}"
        sub["trajectory_concentration"] = CONCENTRATION_VALUES[cond]
        sub["subclass_label"] = "Set9"
        d0_rows.append(sub)
    matched_d0 = pd.concat(d0_rows, ignore_index=True)
    return matched_d2, matched_d0


def _feature_df_to_matrix_and_meta(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fcols = [c for c in feature_df.columns if str(c).startswith("f")]
    matrix_df = feature_df[["sample_key"] + fcols].copy()
    matrix_df = matrix_df.rename(columns={c: f"wn_{int(round(float(RANGE_WAVENUMBERS[int(c[1:])])))}" for c in fcols})
    meta_df = feature_df[["sample_key", "class_label", "trajectory_concentration", "source_file"]].copy()
    return matrix_df, meta_df


def _spectral_linear_axis(matrix_df: pd.DataFrame, concentrations: np.ndarray) -> np.ndarray:
    spectral_cols, _ = _spectral_feature_columns(matrix_df)
    X = StandardScaler().fit_transform(matrix_df[spectral_cols].to_numpy(dtype=float))
    model = LinearRegression().fit(X, concentrations)
    return model.predict(X)


def _transfer_metrics(
    target_scores: np.ndarray,
    ext_df: pd.DataFrame,
    bsv_df: pd.DataFrame,
    family_df: pd.DataFrame,
    concentrations: np.ndarray,
    class_labels: np.ndarray,
    *,
    prefix: str,
) -> pd.DataFrame:
    explain_df, _ = _explainability_table(target_scores, bsv_df, family_df, ext_df, prefix=prefix)
    rep_df, _ = _representation_metrics(target_scores, concentrations, class_labels, bsv_df, family_df, ext_df, prefix=prefix)
    ext_cols = [c for c in ext_df.columns if c not in {"sample_key", "class_label", "trajectory_concentration", "source_file"}]
    feature_rows = []
    for col in ext_cols:
        feature_rows.append(
            {
                "candidate_axis_name": col,
                "concentration_spearman": float(pd.Series(concentrations).corr(ext_df[col], method="spearman")),
                "concentration_pearson": float(pd.Series(concentrations).corr(ext_df[col], method="pearson")),
            }
        )
    best_feature = pd.DataFrame(feature_rows).sort_values("concentration_spearman", key=lambda s: s.abs(), ascending=False).iloc[0]
    rows = [
        {
            "subset_label": prefix,
            "n_spectra": int(len(ext_df)),
            "best_extension_feature": str(best_feature["candidate_axis_name"]),
            "best_extension_feature_spearman": float(best_feature["concentration_spearman"]),
            "extension_only_pca_silhouette": float(
                rep_df.loc[rep_df["representation_name"].astype(str) == "extension_only", "pca_silhouette_by_concentration"].iloc[0]
            ),
            "extension_only_ordering_spearman": float(
                rep_df.loc[rep_df["representation_name"].astype(str) == "extension_only", "condition_mean_ordering_spearman"].iloc[0]
            ),
            "baseline_bsv_family_r2": float(
                explain_df.loc[explain_df["model_name"].astype(str) == "bsv_plus_family", "r2"].iloc[0]
            ),
            "extended_r2": float(
                explain_df.loc[explain_df["model_name"].astype(str) == "bsv_plus_family_plus_extension", "r2"].iloc[0]
            ),
        }
    ]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(TABLES_DIR / f"{prefix}_transfer_metrics.csv", index=False)
    return metrics_df


def _plot_temporary_extended_radar(
    class_df: pd.DataFrame,
    axis_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    features = FIXED_RADAR_AXES + axis_names
    plot_df = class_df.copy()
    for axis in axis_names:
        values = plot_df[axis].to_numpy(dtype=float)
        lo = float(values.min())
        hi = float(values.max())
        if hi - lo > 1e-12:
            plot_df[axis] = (values - lo) / (hi - lo)
        else:
            plot_df[axis] = 0.5

    angles = np.linspace(0.0, 2.0 * math.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8.4, 8.0), subplot_kw={"projection": "polar"})
    for idx, row in plot_df.iterrows():
        vals = row[features].to_numpy(dtype=float).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.0, label=str(row["class_label"]))
        ax.fill(angles, vals, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_yticklabels([])
    ax.set_title(title, pad=24)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.05, 0.5))
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main() -> None:
    _ensure_dirs()

    set10_matrix_df, set10_meta_df, set10_scores_df, set10_bsv_df, set10_family_df = _load_set10_discovery_inputs()
    candidate_df = _candidate_axis_specs()
    set10_ext_df = _compute_extension_features(set10_matrix_df, set10_meta_df, candidate_df, prefix="set10_day2")
    set10_mean_df = pd.read_csv(TABLES_DIR / "set10_day2_extension_feature_means.csv")

    set10_target = set10_scores_df["axis_score"].to_numpy(dtype=float)
    set10_explain_df, _ = _explainability_table(
        set10_target,
        set10_bsv_df,
        set10_family_df,
        set10_ext_df,
        prefix="set10_day2",
    )
    set10_rep_df, set10_score_frames = _representation_metrics(
        set10_target,
        set10_meta_df["class_label"].astype(str).str.split("_").str[1].str.replace("C", "").astype(int).to_numpy(),
        set10_meta_df["class_label"].astype(str).to_numpy(),
        set10_bsv_df,
        set10_family_df,
        set10_ext_df,
        prefix="set10_day2",
    )

    _plot_representation_pca(
        set10_score_frames,
        FIGURES_DIR / "set10_day2_extension_pca.png",
        ["extension_only", "current_bsv_plus_family_plus_extension"],
    )
    _plot_extension_trends(
        set10_mean_df,
        candidate_df["candidate_axis_name"].tolist(),
        FIGURES_DIR / "set10_day2_extension_feature_trends.png",
        "Set10 Day2 extension feature trends",
    )
    _plot_explainability(
        set10_explain_df,
        FIGURES_DIR / "set10_day2_extension_vs_bsv_explainability.png",
        "Set10 Day2 explainability with temporary extension axes",
    )

    set10_counts = set10_meta_df.groupby("class_label").size().to_dict()
    set9_d2_df, set9_d0_df = _build_set9_matched_subsets(set10_counts)

    set9_d2_matrix_df, set9_d2_meta_df = _feature_df_to_matrix_and_meta(set9_d2_df)
    set9_d2_ext_df = _compute_extension_features(set9_d2_matrix_df, set9_d2_meta_df, candidate_df, prefix="set9_d2")
    set9_d2_bsv_df, set9_d2_family_df = _run_bsv_on_feature_df(set9_d2_df)
    set9_d2_bsv_df.to_csv(TABLES_DIR / "set9_d2_bsv.csv", index=False)
    _cohort_delta(set9_d2_bsv_df, [axis for axis in FIXED_RADAR_AXES if axis in set9_d2_bsv_df.columns]).to_csv(
        TABLES_DIR / "set9_d2_delta_bsv.csv", index=False
    )
    set9_d2_family_df.to_csv(TABLES_DIR / "set9_d2_family.csv", index=False)

    set9_d2_target = _spectral_linear_axis(
        set9_d2_matrix_df,
        set9_d2_meta_df["class_label"].astype(str).str.split("_").str[1].str.replace("C", "").astype(int).to_numpy(),
    )
    set9_transfer_df = _transfer_metrics(
        set9_d2_target,
        set9_d2_ext_df,
        set9_d2_bsv_df,
        set9_d2_family_df,
        set9_d2_meta_df["class_label"].astype(str).str.split("_").str[1].str.replace("C", "").astype(int).to_numpy(),
        set9_d2_meta_df["class_label"].astype(str).to_numpy(),
        prefix="set9_d2_extension",
    )

    set9_d0_matrix_df, set9_d0_meta_df = _feature_df_to_matrix_and_meta(set9_d0_df)
    set9_d0_ext_df = _compute_extension_features(set9_d0_matrix_df, set9_d0_meta_df, candidate_df, prefix="set9_d0")
    day_rows = []
    for day_label, meta_df, ext_df in [("D0", set9_d0_meta_df, set9_d0_ext_df), ("D2", set9_d2_meta_df, set9_d2_ext_df)]:
        ext_cols = candidate_df["candidate_axis_name"].tolist()
        best = max(
            (
                (col, float(pd.Series(meta_df["class_label"].astype(str).str.split("_").str[1].str.replace("C", "").astype(int)).corr(ext_df[col], method="spearman")))
                for col in ext_cols
            ),
            key=lambda item: abs(item[1]),
        )
        day_rows.append(
            {
                "day_label": day_label,
                "n_spectra": int(len(ext_df)),
                "best_extension_feature": best[0],
                "best_extension_feature_spearman": best[1],
            }
        )
    set9_day_compare_df = pd.DataFrame(day_rows)
    set9_day_compare_df.to_csv(TABLES_DIR / "set9_day0_day2_extension_comparison.csv", index=False)

    baseline_r2 = float(set10_explain_df.loc[set10_explain_df["model_name"].astype(str) == "bsv_plus_family", "r2"].iloc[0])
    extended_r2 = float(
        set10_explain_df.loc[set10_explain_df["model_name"].astype(str) == "bsv_plus_family_plus_extension", "r2"].iloc[0]
    )
    set10_delta_r2 = extended_r2 - baseline_r2
    baseline_order = float(
        set10_rep_df.loc[set10_rep_df["representation_name"].astype(str) == "current_bsv_plus_family", "condition_mean_ordering_spearman"].iloc[0]
    )
    extended_order = float(
        set10_rep_df.loc[
            set10_rep_df["representation_name"].astype(str) == "current_bsv_plus_family_plus_extension", "condition_mean_ordering_spearman"
        ].iloc[0]
    )
    set9_baseline_r2 = float(set9_transfer_df["baseline_bsv_family_r2"].iloc[0])
    set9_extended_r2 = float(set9_transfer_df["extended_r2"].iloc[0])
    set9_delta_r2 = set9_extended_r2 - set9_baseline_r2
    set10_structure_improved = abs(extended_order) > abs(baseline_order) + 0.10
    set9_extension_ordering = float(set9_transfer_df["extension_only_ordering_spearman"].iloc[0])

    if set10_delta_r2 >= 0.08 and set9_delta_r2 >= 0.10 and (set10_structure_improved or abs(set9_extension_ordering) >= 0.60):
        decision_label = "transferable_support"
    elif set10_delta_r2 >= 0.03 and set9_delta_r2 >= 0.10:
        decision_label = "local_support_only"
    else:
        decision_label = "no_support"

    supported_axes = candidate_df["candidate_axis_name"].tolist()
    ext_corr_rows = []
    for axis_name in supported_axes:
        ext_corr_rows.append(
            {
                "candidate_axis_name": axis_name,
                "spearman_to_spectral_axis": float(set10_ext_df[axis_name].corr(pd.Series(set10_target), method="spearman")),
                "pearson_to_spectral_axis": float(set10_ext_df[axis_name].corr(pd.Series(set10_target), method="pearson")),
                "set10_spearman_to_concentration": float(
                    set10_ext_df[axis_name].corr(set10_ext_df["trajectory_concentration"], method="spearman")
                ),
                "set9_spearman_to_concentration": float(
                    set9_d2_ext_df[axis_name].corr(set9_d2_ext_df["trajectory_concentration"], method="spearman")
                ),
            }
        )
    ext_corr_df = pd.DataFrame(ext_corr_rows)
    ext_corr_df["combined_support_score"] = ext_corr_df["spearman_to_spectral_axis"].abs() + 0.5 * ext_corr_df["set9_spearman_to_concentration"].abs()
    ext_corr_df = ext_corr_df.sort_values("combined_support_score", ascending=False).reset_index(drop=True)
    ext_corr_df.to_csv(TABLES_DIR / "extension_axis_support_summary.csv", index=False)
    top_supported = str(ext_corr_df.iloc[0]["candidate_axis_name"])

    if decision_label != "no_support":
        set10_class_ext = (
            set10_ext_df.groupby("class_label", as_index=False)[supported_axes].mean()
            .assign(trajectory_concentration=lambda f: f["class_label"].astype(str).str.split("_").str[1].str.replace("C", "").astype(int))
            .sort_values("trajectory_concentration")
            .reset_index(drop=True)
        )
        set10_class_bsv = (
            set10_bsv_df.groupby("class_label", as_index=False)[[axis for axis in FIXED_RADAR_AXES if axis in set10_bsv_df.columns]].mean()
        )
        _plot_temporary_extended_radar(
            set10_class_bsv.merge(set10_class_ext, on="class_label"),
            ext_corr_df["candidate_axis_name"].head(2).astype(str).tolist(),
            FIGURES_DIR / "set10_day2_extended_radar_by_concentration.png",
            "Set10 Day2 temporary extension radar",
        )
        set9_class_ext = (
            set9_d2_ext_df.groupby("class_label", as_index=False)[supported_axes].mean()
            .assign(trajectory_concentration=lambda f: f["class_label"].astype(str).str.split("_").str[1].str.replace("C", "").astype(int))
            .sort_values("trajectory_concentration")
            .reset_index(drop=True)
        )
        set9_class_bsv = (
            set9_d2_bsv_df.groupby("class_label", as_index=False)[[axis for axis in FIXED_RADAR_AXES if axis in set9_d2_bsv_df.columns]].mean()
        )
        _plot_temporary_extended_radar(
            set9_class_bsv.merge(set9_class_ext, on="class_label"),
            ext_corr_df["candidate_axis_name"].head(2).astype(str).tolist(),
            FIGURES_DIR / "set9_d2_extended_radar_by_concentration.png",
            "Set9 D2 temporary extension radar",
        )

    set10_compare_df = set10_rep_df.copy()
    set10_compare_df.to_csv(TABLES_DIR / "set10_day2_representation_comparison.csv", index=False)
    set9_transfer_df.to_csv(TABLES_DIR / "set9_d2_extension_transfer_metrics.csv", index=False)

    decision_lines = [
        "# Extension Test Decision",
        "",
        f"- decision_label: `{decision_label}`",
        f"- baseline Set10 BSV+family R^2: `{baseline_r2:.4f}`",
        f"- extended Set10 BSV+family+extension R^2: `{extended_r2:.4f}`",
        f"- baseline Set9 D2 BSV+family R^2: `{set9_baseline_r2:.4f}`",
        f"- extended Set9 D2 BSV+family+extension R^2: `{set9_extended_r2:.4f}`",
        "",
        "Direct answers:",
        f"1. Did extension features improve Set10 Day2 explainability meaningfully? `{'yes' if set10_delta_r2 >= 0.03 else 'no'}`",
        f"2. Did extension features improve concentration ordering / structure? `{'yes' if abs(extended_order) > abs(baseline_order) else 'no clear improvement'}`",
        f"3. Did the extension transfer to Set9 at all? `{'yes' if set9_delta_r2 > 0.05 else 'minimal'}`",
        f"4. Most supported temporary axis/theme: `{top_supported}`",
    ]
    (REPORT_DIR / "extension_test_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")

    report_lines = [
        "# GAIRAv3 SHINE BSV Extension Test Report",
        "",
        "## 1. Why this experiment was needed",
        "",
        "- The validated Set10 Day2 spectral response axis was real in spectral space, but current GAIRA BSV axes explained it poorly.",
        "- This experiment adds a temporary SHINE extension layer without changing the locked cfg05 base representation.",
        "",
        "## 2. Candidate extension axes",
        "",
        _df_to_md(candidate_df),
        "",
        "## 3. Set10 Day2 explainability test",
        "",
        _df_to_md(set10_explain_df),
        "",
        "## 4. Set10 Day2 structure test",
        "",
        _df_to_md(set10_rep_df),
        "",
        "## 5. Set9 transfer test",
        "",
        _df_to_md(set9_transfer_df),
        "",
        "Optional Day0 vs Day2 extension contrast:",
        "",
        _df_to_md(set9_day_compare_df),
        "",
        "## 6. Decision",
        "",
        (REPORT_DIR / "extension_test_decision.md").read_text(encoding="utf-8"),
        "",
        "## 7. Final concise conclusion",
        "",
        f"- Did the extension help? `{'yes' if extended_r2 > baseline_r2 else 'no'}`",
        f"- Did it transfer? `{'yes' if set9_extended_r2 > set9_baseline_r2 else 'no'}`",
        f"- What exact axis/theme is most supported? `{top_supported}`",
    ]
    report_md = REPORT_DIR / "GAIRAv3_SHINE_BSV_extension_test_report.md"
    report_md.write_text("\n".join(report_lines), encoding="utf-8")
    figure_paths = [
        FIGURES_DIR / "set10_day2_extension_pca.png",
        FIGURES_DIR / "set10_day2_extension_feature_trends.png",
        FIGURES_DIR / "set10_day2_extension_vs_bsv_explainability.png",
        FIGURES_DIR / "set10_day2_extended_radar_by_concentration.png",
        FIGURES_DIR / "set9_d2_extended_radar_by_concentration.png",
    ]
    build_pdf_report(report_md, [p for p in figure_paths if p.exists()], REPORT_DIR / "GAIRAv3_SHINE_BSV_extension_test_report.pdf")


if __name__ == "__main__":
    main()
