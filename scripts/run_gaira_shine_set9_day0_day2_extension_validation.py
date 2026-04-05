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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaira.demo.autoresearch_pass5_utils import build_bsv_profiles_pass5
from gaira.demo.gaira_experiment_runner_utils import load_architecture_registries
from gaira.demo.gaira_pilot_utils import build_pdf_report
from scripts.run_gaira_pilot3_shine_day2_controlanchored import _family_fingerprint_from_retrieval
from scripts.run_gaira_pilot3_shine_ev_sers_fullspectra import (
    ARCH_DIR,
    CONFIG_SPEC,
    FAMILY_ORDER,
    FIXED_RADAR_AXES,
    PHASE1_DIR,
    ROOT,
    _cohort_delta,
    _fit_pca,
    _prepare_grounding_and_mapping,
    _resolve_alias,
)
from scripts.run_gaira_shine_fig4_replication_and_bsv import (
    CONDITION_ORDER,
    CONCENTRATION_VALUES,
    DATA_ROOT,
    RANGE_WAVENUMBERS,
    _preprocess_raw_spectrum,
)


OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/shine_set9_day0_day2_extension_validation"
)
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "report"
EXT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/shine_bsv_extension_test"
)
SUBSET_ALIAS = "shine_ev_stress"
SET_LABEL = "Set9"
DAYS = ["D0", "D2"]
CLASS_LABELS = [f"{day}_{cond}" for day in DAYS for cond in CONDITION_ORDER]
DAY_COLORS = {"D0": "#355070", "D2": "#e76f51"}
COND_COLORS = {"C0": "#355070", "C10": "#b56576", "C20": "#2a9d8f", "C40": "#e76f51"}


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


def _load_candidate_axes() -> pd.DataFrame:
    return pd.read_csv(EXT_ROOT / "tables" / "candidate_extension_axes.csv")


def _build_full_set9_feature_df() -> pd.DataFrame:
    rows = []
    for day in DAYS:
        for cond in CONDITION_ORDER:
            cond_dir = DATA_ROOT / SET_LABEL / f"{day}_{cond}"
            for idx, path in enumerate(sorted(cond_dir.rglob("s_*"))):
                vec = _preprocess_raw_spectrum(path)
                record = {f"f{i:03d}": float(v) for i, v in enumerate(vec)}
                record["sample_key"] = f"set9_full__{day}_{cond}__{path.parent.name}__{path.name}"
                record["class_label"] = f"{day}_{cond}"
                record["day_label"] = day
                record["trajectory_concentration"] = CONCENTRATION_VALUES[cond]
                record["source_file"] = str(path)
                rows.append(record)
    return pd.DataFrame(rows)


def _write_input_verification(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_label in CLASS_LABELS:
        sub = feature_df[feature_df["class_label"].astype(str) == class_label].copy()
        rows.append(
            {
                "class_label": class_label,
                "n_spectra": int(len(sub)),
                "n_unique_source_files": int(sub["source_file"].astype(str).nunique()),
            }
        )
    out = pd.DataFrame(rows)
    totals = pd.DataFrame(
        [
            {"class_label": "TOTAL_D0", "n_spectra": int((feature_df["day_label"] == "D0").sum()), "n_unique_source_files": int(feature_df.loc[feature_df["day_label"] == "D0", "source_file"].astype(str).nunique())},
            {"class_label": "TOTAL_D2", "n_spectra": int((feature_df["day_label"] == "D2").sum()), "n_unique_source_files": int(feature_df.loc[feature_df["day_label"] == "D2", "source_file"].astype(str).nunique())},
        ]
    )
    final = pd.concat([out, totals], ignore_index=True)
    final.to_csv(TABLES_DIR / "set9_day0_day2_input_verification.csv", index=False)
    return final


def _matrix_and_meta(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fcols = [c for c in feature_df.columns if str(c).startswith("f")]
    matrix_df = feature_df[["sample_key"] + fcols].copy()
    matrix_df = matrix_df.rename(columns={c: f"wn_{int(round(float(RANGE_WAVENUMBERS[int(c[1:])])))}" for c in fcols})
    meta_df = feature_df[["sample_key", "class_label", "day_label", "trajectory_concentration", "source_file"]].copy()
    return matrix_df, meta_df


def _compute_extension_features(matrix_df: pd.DataFrame, meta_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    spectral_cols = [c for c in matrix_df.columns if str(c).startswith("wn_")]
    wns = np.array([int(str(c).replace("wn_", "")) for c in spectral_cols], dtype=int)
    matrix = matrix_df[spectral_cols].to_numpy(dtype=float)
    out = meta_df.copy()
    for row in candidate_df.itertuples(index=False):
        pos_regions = _parse_regions(row.positive_regions)
        neg_regions = _parse_regions(row.negative_regions)
        pos_stack = np.vstack([_region_mean(matrix, wns, a, b) for a, b in pos_regions]) if pos_regions else np.zeros((1, len(out)))
        neg_stack = np.vstack([_region_mean(matrix, wns, a, b) for a, b in neg_regions]) if neg_regions else np.zeros((1, len(out)))
        out[row.candidate_axis_name] = pos_stack.mean(axis=0) - neg_stack.mean(axis=0)
    out.to_csv(TABLES_DIR / "set9_day0_day2_extension_features.csv", index=False)
    return out


def _build_query_df(matrix_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    spectral_cols = [c for c in matrix_df.columns if str(c).startswith("wn_")]
    wns = [int(c.replace("wn_", "")) for c in spectral_cols]
    rows = []
    for i, row in enumerate(meta_df.itertuples(index=False)):
        vec = matrix_df.loc[i, spectral_cols].to_numpy(dtype=float)
        rows.append(
            {
                "sample_key": str(row.sample_key),
                "dataset_id": "shine_ev_sers",
                "subclass_label": SET_LABEL,
                "class_label": str(row.class_label),
                "source_file": str(row.source_file),
                "wavenumbers_json": json.dumps(wns),
                "intensity_json": json.dumps(vec.tolist()),
            }
        )
    return pd.DataFrame(rows)


def _run_bsv(matrix_df: pd.DataFrame, meta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    query_df = _build_query_df(matrix_df, meta_df)
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
    meta = meta_df.copy()
    meta["sample_id"] = meta["sample_key"].astype(str)
    meta["trajectory_index"] = meta["trajectory_concentration"].astype(int)
    bsv_df = bsv_df.merge(meta[["sample_key", "class_label", "trajectory_concentration", "source_file", "sample_id", "trajectory_index", "day_label"]], on=["sample_key", "class_label"], how="left")
    axes = [axis for axis in FIXED_RADAR_AXES if axis in bsv_df.columns]
    delta_df = _cohort_delta(bsv_df[["sample_key"] + axes].copy(), axes).merge(
        bsv_df[["sample_key", "class_label", "trajectory_concentration", "source_file", "day_label"]],
        on="sample_key",
        how="left",
    )
    family_df = _family_fingerprint_from_retrieval(
        retrieval_df,
        meta[["sample_key", "sample_id", "class_label", "trajectory_concentration", "trajectory_index"]],
    ).merge(meta[["sample_key", "day_label"]], on="sample_key", how="left")
    return bsv_df, delta_df, family_df


def _family_wide(family_df: pd.DataFrame, sample_keys: pd.Series) -> pd.DataFrame:
    wide = family_df.pivot(index="sample_key", columns="family", values="family_fraction")
    wide = wide.reindex(sample_keys.astype(str)).fillna(0.0)
    for family in FAMILY_ORDER:
        if family not in wide.columns:
            wide[family] = 0.0
    return wide[FAMILY_ORDER].reset_index()


def _fit_pca_df(feature_df: pd.DataFrame, cols: list[str], meta_df: pd.DataFrame) -> pd.DataFrame:
    scores, explained = _fit_pca(feature_df[cols].to_numpy(dtype=float), scale=True)
    out = meta_df.copy()
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1]
    out["pc1_explained_ratio"] = float(explained[0])
    out["pc2_explained_ratio"] = float(explained[1])
    return out


def _nn_purity(X: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.0
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    pred = model.predict(X)
    return float(np.mean(pred == y))


def _linear_cv_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.0
    min_count = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, X, y, cv=cv)
    return float(np.mean(scores))


def _representation_matrix(name: str, bsv_df: pd.DataFrame, family_df: pd.DataFrame, ext_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if name == "current_bsv":
        cols = [c for c in FIXED_RADAR_AXES if c in bsv_df.columns]
        return bsv_df[["sample_key"] + cols].copy(), cols
    if name == "family":
        wide = _family_wide(family_df, bsv_df["sample_key"])
        cols = FAMILY_ORDER.copy()
        return wide.rename(columns={"index": "sample_key"}), cols
    if name == "extension":
        cols = [c for c in ext_df.columns if c not in {"sample_key", "class_label", "day_label", "trajectory_concentration", "source_file"}]
        return ext_df[["sample_key"] + cols].copy(), cols
    if name == "combined":
        bsv_cols = [c for c in FIXED_RADAR_AXES if c in bsv_df.columns]
        fam = _family_wide(family_df, bsv_df["sample_key"])
        ext_cols = [c for c in ext_df.columns if c not in {"sample_key", "class_label", "day_label", "trajectory_concentration", "source_file"}]
        merged = bsv_df[["sample_key"] + bsv_cols].copy().merge(fam, on="sample_key", how="left").merge(
            ext_df[["sample_key"] + ext_cols], on="sample_key", how="left"
        )
        cols = bsv_cols + FAMILY_ORDER + ext_cols
        return merged.fillna(0.0), cols
    raise ValueError(name)


def _day_contrast_metrics(feature_df: pd.DataFrame, cols: list[str], meta_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    X = StandardScaler().fit_transform(feature_df[cols].to_numpy(dtype=float))
    day_labels = meta_df["day_label"].astype(str).to_numpy()
    day_binary = (meta_df["day_label"].astype(str) == "D2").astype(int).to_numpy()
    silhouette = float(silhouette_score(X, day_labels))
    centroids = pd.DataFrame(X).assign(day_label=day_labels).groupby("day_label", as_index=False).mean()
    centroid_dist = float(np.linalg.norm(centroids.iloc[0, 1:].to_numpy(dtype=float) - centroids.iloc[1, 1:].to_numpy(dtype=float)))
    nn_purity = _nn_purity(X, day_labels)
    cv_acc = _linear_cv_accuracy(X, day_binary)
    pca_df = _fit_pca_df(pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]), [f"x{i}" for i in range(X.shape[1])], meta_df.copy())
    return pca_df, {
        "silhouette_by_day": silhouette,
        "centroid_distance_day0_day2": centroid_dist,
        "nearest_neighbor_purity_by_day": nn_purity,
        "linear_cv_accuracy_by_day": cv_acc,
    }


def _plot_day_pca(pca_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for day in DAYS:
        sub = pca_df[pca_df["day_label"].astype(str) == day].copy()
        ax.scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=16,
            alpha=0.45,
            color=DAY_COLORS[day],
            label=day,
        )
    ax.set_xlabel(f"PC1 ({pca_df['pc1_explained_ratio'].iloc[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_df['pc2_explained_ratio'].iloc[0]*100:.1f}%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _effect_size(day0: np.ndarray, day2: np.ndarray) -> float:
    if len(day0) < 2 or len(day2) < 2:
        return 0.0
    s0 = float(np.var(day0, ddof=1))
    s2 = float(np.var(day2, ddof=1))
    pooled = ((len(day0) - 1) * s0 + (len(day2) - 1) * s2) / max(1, (len(day0) + len(day2) - 2))
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(day2) - np.mean(day0)) / math.sqrt(pooled))


def _feature_shift_summary(ext_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    day_binary = (ext_df["day_label"].astype(str) == "D2").astype(int)
    for feature_name in candidate_df["candidate_axis_name"].astype(str).tolist():
        day0 = ext_df.loc[ext_df["day_label"].astype(str) == "D0", feature_name].to_numpy(dtype=float)
        day2 = ext_df.loc[ext_df["day_label"].astype(str) == "D2", feature_name].to_numpy(dtype=float)
        delta = float(day2.mean() - day0.mean())
        effect = _effect_size(day0, day2)
        note = "higher on Day2" if delta > 0 else "lower on Day2"
        rows.append(
            {
                "feature_name": feature_name,
                "mean_day0": float(day0.mean()),
                "mean_day2": float(day2.mean()),
                "delta_day2_minus_day0": delta,
                "effect_size": effect,
                "spearman_with_day_label": float(ext_df[feature_name].corr(day_binary, method="spearman")),
                "interpretation_note": note,
            }
        )
    out = pd.DataFrame(rows).sort_values("effect_size", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    out.to_csv(TABLES_DIR / "set9_day_feature_shift_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.barh(out["feature_name"], out["delta_day2_minus_day0"], color="#b56576")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Day2 minus Day0 mean")
    ax.set_title("Set9 day feature shifts")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "set9_day_feature_shift_bars.png", dpi=240)
    plt.close(fig)
    return out


def _within_day_metrics(feature_df: pd.DataFrame, cols: list[str], meta_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    X = StandardScaler().fit_transform(feature_df[cols].to_numpy(dtype=float))
    concentrations = meta_df["trajectory_concentration"].to_numpy(dtype=int)
    class_labels = meta_df["class_label"].astype(str).to_numpy()
    silhouette = float(silhouette_score(X, class_labels))
    reg = LinearRegression().fit(X, concentrations)
    pred = reg.predict(X)
    pred_spearman = float(pd.Series(concentrations).corr(pd.Series(pred), method="spearman"))
    pca_df = _fit_pca_df(pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]), [f"x{i}" for i in range(X.shape[1])], meta_df.copy())
    means = pca_df.groupby("trajectory_concentration", as_index=False)["pc1"].mean().sort_values("trajectory_concentration")
    ordering = float(means["trajectory_concentration"].corr(means["pc1"], method="spearman"))
    arr = (
        pca_df.groupby("trajectory_concentration", as_index=False)[["pc1", "pc2"]]
        .mean()
        .sort_values("trajectory_concentration")[["pc1", "pc2"]]
        .to_numpy(dtype=float)
    )
    adjacent = [float(np.linalg.norm(arr[i] - arr[i - 1])) for i in range(1, len(arr))]
    return pca_df, {
        "silhouette_by_concentration": silhouette,
        "condition_mean_ordering_spearman": ordering,
        "response_axis_correlation": pred_spearman,
        "mean_adjacent_distance": float(np.mean(adjacent)) if adjacent else 0.0,
        "min_adjacent_distance": float(np.min(adjacent)) if adjacent else 0.0,
    }


def _plot_concentration_pca(pca_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for cond in CONDITION_ORDER:
        class_label = f"{pca_df['day_label'].iloc[0]}_{cond}"
        sub = pca_df[pca_df["class_label"].astype(str) == class_label].copy()
        ax.scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=16,
            alpha=0.45,
            color=COND_COLORS[cond],
            label=f"{CONCENTRATION_VALUES[cond]} mM",
        )
    ax.set_xlabel(f"PC1 ({pca_df['pc1_explained_ratio'].iloc[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_df['pc2_explained_ratio'].iloc[0]*100:.1f}%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _within_day_table(day_label: str, bsv_df: pd.DataFrame, family_df: pd.DataFrame, ext_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    sub_meta = bsv_df[bsv_df["day_label"].astype(str) == day_label][["sample_key", "class_label", "day_label", "trajectory_concentration", "source_file"]].copy().reset_index(drop=True)
    sub_bsv = bsv_df[bsv_df["day_label"].astype(str) == day_label].copy().reset_index(drop=True)
    sub_family = family_df[family_df["day_label"].astype(str) == day_label].copy().reset_index(drop=True)
    sub_ext = ext_df[ext_df["day_label"].astype(str) == day_label].copy().reset_index(drop=True)

    rows = []
    pca_frames: dict[str, pd.DataFrame] = {}
    for rep in ["current_bsv", "family", "extension", "combined"]:
        feat_df, cols = _representation_matrix(rep, sub_bsv, sub_family, sub_ext)
        pca_df, metrics = _within_day_metrics(feat_df, cols, sub_meta)
        rows.append({"representation_name": rep, **metrics})
        pca_frames[rep] = pca_df
    out = pd.DataFrame(rows).sort_values("response_axis_correlation", ascending=False).reset_index(drop=True)
    out.to_csv(TABLES_DIR / f"set9_{day_label.lower()}_extension_concentration_metrics.csv", index=False)
    return out, pca_frames


def _trend_plot(ext_df: pd.DataFrame, day_label: str, output_path: Path) -> None:
    features = ["guanidino_response", "aromatic_stress_response", "amide_stress_response"]
    means = ext_df.groupby("trajectory_concentration", as_index=False)[features].mean().sort_values("trajectory_concentration")
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for feat in features:
        ax.plot(means["trajectory_concentration"], means[feat], marker="o", linewidth=2.0, label=feat)
    ax.set_xlabel("Concentration (mM)")
    ax.set_ylabel("Mean feature score")
    ax.set_title(f"Set9 {day_label} extension feature trends")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _build_stress_readout(ext_df: pd.DataFrame, shift_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_features = ["guanidino_response", "aromatic_stress_response", "amide_stress_response"]
    weight_map = {}
    for feat in candidate_features:
        delta = float(shift_df.loc[shift_df["feature_name"] == feat, "delta_day2_minus_day0"].iloc[0])
        effect = float(shift_df.loc[shift_df["feature_name"] == feat, "effect_size"].iloc[0])
        sign = 1.0 if delta >= 0 else -1.0
        weight_map[feat] = sign * max(abs(effect), 0.5)
    z_df = ext_df.copy()
    for feat in candidate_features:
        vals = z_df[feat].to_numpy(dtype=float)
        z_df[feat] = (vals - vals.mean()) / (vals.std(ddof=0) + 1e-12)
    z_df["temporary_stress_readout"] = sum(weight_map[feat] * z_df[feat] for feat in candidate_features) / sum(abs(v) for v in weight_map.values())
    score_df = z_df[["sample_key", "class_label", "day_label", "trajectory_concentration", "temporary_stress_readout"]].copy()
    score_df.to_csv(TABLES_DIR / "set9_temporary_stress_readout_scores.csv", index=False)

    day_binary = (score_df["day_label"].astype(str) == "D2").astype(int)
    day_sep = float(score_df["temporary_stress_readout"].corr(day_binary, method="spearman"))
    day0 = score_df[score_df["day_label"].astype(str) == "D0"].copy()
    day2 = score_df[score_df["day_label"].astype(str) == "D2"].copy()
    day0_mon = float(day0["temporary_stress_readout"].corr(day0["trajectory_concentration"], method="spearman"))
    day2_mon = float(day2["temporary_stress_readout"].corr(day2["trajectory_concentration"], method="spearman"))
    rows = [
        {
            "day_separation_spearman": day_sep,
            "mean_day0": float(day0["temporary_stress_readout"].mean()),
            "mean_day2": float(day2["temporary_stress_readout"].mean()),
            "delta_day2_minus_day0": float(day2["temporary_stress_readout"].mean() - day0["temporary_stress_readout"].mean()),
            "day0_concentration_spearman": day0_mon,
            "day2_concentration_spearman": day2_mon,
        }
    ]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(TABLES_DIR / "set9_temporary_stress_readout_metrics.csv", index=False)
    return score_df, metrics_df


def _plot_stress_readout(score_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    data = [
        score_df.loc[score_df["day_label"].astype(str) == "D0", "temporary_stress_readout"].to_numpy(dtype=float),
        score_df.loc[score_df["day_label"].astype(str) == "D2", "temporary_stress_readout"].to_numpy(dtype=float),
    ]
    ax.boxplot(data, tick_labels=["Day0", "Day2"])
    ax.set_ylabel("Temporary stress-readout score")
    ax.set_title("Set9 temporary stress-readout by day")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "set9_temporary_stress_readout_day_comparison.png", dpi=240)
    plt.close(fig)

    day2 = score_df[score_df["day_label"].astype(str) == "D2"].copy()
    means = day2.groupby("trajectory_concentration", as_index=False)["temporary_stress_readout"].mean().sort_values("trajectory_concentration")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(means["trajectory_concentration"], means["temporary_stress_readout"], marker="o", linewidth=2.0, color="#e76f51")
    ax.set_xlabel("Concentration (mM)")
    ax.set_ylabel("Mean stress-readout")
    ax.set_title("Set9 Day2 temporary stress-readout trend")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "set9_temporary_stress_readout_day2_trend.png", dpi=240)
    plt.close(fig)


def _plot_extended_radar(class_df: pd.DataFrame, ext_axes: list[str], output_path: Path, title: str) -> None:
    features = FIXED_RADAR_AXES + ext_axes
    plot_df = class_df.copy()
    for feat in ext_axes:
        vals = plot_df[feat].to_numpy(dtype=float)
        lo = float(vals.min())
        hi = float(vals.max())
        plot_df[feat] = (vals - lo) / (hi - lo) if hi - lo > 1e-12 else 0.5
    angles = np.linspace(0.0, 2.0 * math.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8.4, 8.0), subplot_kw={"projection": "polar"})
    for _, row in plot_df.iterrows():
        vals = row[features].to_numpy(dtype=float).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.0, label=str(row["class_label"]))
        ax.fill(angles, vals, alpha=0.14)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_yticklabels([])
    ax.set_title(title, pad=24)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.05, 0.5))
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_day_summary_radar(day0_df: pd.DataFrame, day2_df: pd.DataFrame, ext_axes: list[str], output_path: Path) -> None:
    def _mean_row(df: pd.DataFrame, label: str) -> pd.Series:
        row = {"class_label": label}
        for feat in FIXED_RADAR_AXES + ext_axes:
            row[feat] = float(df[feat].mean())
        return pd.Series(row)

    summary = pd.DataFrame([_mean_row(day0_df, "Day0_mean"), _mean_row(day2_df, "Day2_mean")])
    _plot_extended_radar(summary, ext_axes, output_path, "Set9 Day0 vs Day2 extended radar summary")


def main() -> None:
    _ensure_dirs()
    candidate_df = _load_candidate_axes()
    feature_df = _build_full_set9_feature_df()
    verification_df = _write_input_verification(feature_df)
    matrix_df, meta_df = _matrix_and_meta(feature_df)
    ext_df = _compute_extension_features(matrix_df, meta_df, candidate_df)
    bsv_df, delta_df, family_df = _run_bsv(matrix_df, meta_df)
    bsv_df.to_csv(TABLES_DIR / "set9_day0_day2_bsv.csv", index=False)
    delta_df.to_csv(TABLES_DIR / "set9_day0_day2_delta_bsv.csv", index=False)
    family_df.to_csv(TABLES_DIR / "set9_day0_day2_family.csv", index=False)

    # Step 3 day contrast
    day_rows = []
    pca_outputs = {}
    spectral_cols = [c for c in matrix_df.columns if str(c).startswith("wn_")]
    spectral_pca = _fit_pca_df(matrix_df, spectral_cols, meta_df.copy())
    _plot_day_pca(spectral_pca, FIGURES_DIR / "set9_day0_vs_day2_spectral_pca.png", "Set9 Day0 vs Day2 spectral PCA")
    for rep_name, fig_name in [
        ("current_bsv", "set9_day0_vs_day2_bsv_pca.png"),
        ("family", "set9_day0_vs_day2_family_pca.png"),
        ("extension", "set9_day0_vs_day2_extension_pca.png"),
        ("combined", "set9_day0_vs_day2_combined_pca.png"),
    ]:
        feat_df, cols = _representation_matrix(rep_name, bsv_df, family_df, ext_df)
        pca_df, metrics = _day_contrast_metrics(feat_df, cols, meta_df)
        pca_outputs[rep_name] = pca_df
        day_rows.append({"representation_name": rep_name, **metrics})
        _plot_day_pca(pca_df, FIGURES_DIR / fig_name, f"Set9 Day0 vs Day2 {rep_name.replace('_', ' ')} PCA")
    day_compare_df = pd.DataFrame(day_rows).sort_values("silhouette_by_day", ascending=False).reset_index(drop=True)
    day_compare_df.to_csv(TABLES_DIR / "set9_day0_vs_day2_representation_comparison.csv", index=False)

    # Step 4
    shift_df = _feature_shift_summary(ext_df, candidate_df)

    # Step 5 and 6
    day2_metrics_df, day2_pcas = _within_day_table("D2", bsv_df, family_df, ext_df)
    _plot_concentration_pca(day2_pcas["extension"], FIGURES_DIR / "set9_day2_extension_concentration_pca.png", "Set9 Day2 extension concentration PCA")
    _trend_plot(ext_df[ext_df["day_label"].astype(str) == "D2"].copy(), "Day2", FIGURES_DIR / "set9_day2_extension_concentration_trends.png")

    day0_metrics_df, day0_pcas = _within_day_table("D0", bsv_df, family_df, ext_df)
    _plot_concentration_pca(day0_pcas["extension"], FIGURES_DIR / "set9_day0_extension_concentration_pca.png", "Set9 Day0 extension concentration PCA")
    _trend_plot(ext_df[ext_df["day_label"].astype(str) == "D0"].copy(), "Day0", FIGURES_DIR / "set9_day0_extension_concentration_trends.png")

    # Step 7
    stress_scores_df, stress_metrics_df = _build_stress_readout(ext_df, shift_df)
    _plot_stress_readout(stress_scores_df)

    # Step 8
    supported_axes = ["guanidino_response", "aromatic_stress_response", "amide_stress_response"]
    day0_class_means = (
        bsv_df[bsv_df["day_label"].astype(str) == "D0"]
        .groupby("class_label", as_index=False)[[c for c in FIXED_RADAR_AXES if c in bsv_df.columns]]
        .mean()
        .merge(
            ext_df[ext_df["day_label"].astype(str) == "D0"].groupby("class_label", as_index=False)[supported_axes].mean(),
            on="class_label",
            how="left",
        )
        .assign(trajectory_concentration=lambda f: f["class_label"].astype(str).str.split("_").str[1].str.replace("C", "").astype(int))
        .sort_values("trajectory_concentration")
        .reset_index(drop=True)
    )
    day2_class_means = (
        bsv_df[bsv_df["day_label"].astype(str) == "D2"]
        .groupby("class_label", as_index=False)[[c for c in FIXED_RADAR_AXES if c in bsv_df.columns]]
        .mean()
        .merge(
            ext_df[ext_df["day_label"].astype(str) == "D2"].groupby("class_label", as_index=False)[supported_axes].mean(),
            on="class_label",
            how="left",
        )
        .assign(trajectory_concentration=lambda f: f["class_label"].astype(str).str.split("_").str[1].str.replace("C", "").astype(int))
        .sort_values("trajectory_concentration")
        .reset_index(drop=True)
    )
    _plot_extended_radar(day0_class_means, supported_axes, FIGURES_DIR / "set9_day0_extended_radar_by_concentration.png", "Set9 Day0 extended radar")
    _plot_extended_radar(day2_class_means, supported_axes, FIGURES_DIR / "set9_day2_extended_radar_by_concentration.png", "Set9 Day2 extended radar")
    _plot_day_summary_radar(day0_class_means, day2_class_means, supported_axes, FIGURES_DIR / "set9_day0_vs_day2_extended_radar_summary.png")

    # Decision
    best_day_rep = day_compare_df.iloc[0]
    best_day2_ext = day2_metrics_df.loc[day2_metrics_df["representation_name"].astype(str) == "extension"].iloc[0]
    best_day0_ext = day0_metrics_df.loc[day0_metrics_df["representation_name"].astype(str) == "extension"].iloc[0]
    strong_day_contrast = float(best_day_rep["silhouette_by_day"]) > 0.05 and float(stress_metrics_df["delta_day2_minus_day0"].iloc[0]) > 0
    strong_day2_conc = (
        float(best_day2_ext["response_axis_correlation"]) > float(best_day0_ext["response_axis_correlation"]) + 0.10
        and float(best_day2_ext["condition_mean_ordering_spearman"]) >= float(best_day0_ext["condition_mean_ordering_spearman"])
    )
    if strong_day_contrast and strong_day2_conc:
        decision_label = "day_and_concentration_support"
    elif strong_day_contrast:
        decision_label = "day_contrast_only"
    else:
        decision_label = "no_support"

    decision_lines = [
        "# Set9 Day0 Day2 Extension Decision",
        "",
        f"- decision_label: `{decision_label}`",
        f"- best day-contrast representation: `{best_day_rep['representation_name']}`",
        f"- best day-contrast silhouette: `{best_day_rep['silhouette_by_day']:.4f}`",
        f"- extension Day2 response-axis correlation: `{best_day2_ext['response_axis_correlation']:.4f}`",
        f"- extension Day0 response-axis correlation: `{best_day0_ext['response_axis_correlation']:.4f}`",
        f"- temporary stress-readout day separation Spearman: `{float(stress_metrics_df['day_separation_spearman'].iloc[0]):.4f}`",
        "",
        "Direct answers:",
        f"1. Do the temporary SHINE axes distinguish Day2 from Day0? `{'yes' if strong_day_contrast else 'weakly'}`",
        f"2. Is Day2 more strongly aligned with these axes than Day0? `{'yes' if float(best_day2_ext['response_axis_correlation']) > float(best_day0_ext['response_axis_correlation']) else 'no'}`",
        f"3. Do these axes improve Day2 concentration structure? `{'yes' if strong_day2_conc else 'partially or weakly'}`",
        "4. Are these axes behaving like plausible stress-response features rather than random Set10 artifacts? `yes`" if decision_label != "no_support" else "4. Are these axes behaving like plausible stress-response features rather than random Set10 artifacts? `not clearly`",
    ]
    (REPORT_DIR / "set9_day0_day2_extension_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")

    report_lines = [
        "# GAIRAv3 SHINE Set9 Day0 Day2 Extension Validation Report",
        "",
        "## 1. Why this experiment was needed",
        "",
        "- Set10 showed a real spectral Day2 signal and a partial temporary extension benefit.",
        "- This experiment tests whether those temporary axes behave meaningfully on archive-side Set9 Day0 vs Day2 biology.",
        "",
        "## 2. Set9 Day0 vs Day2 contrast",
        "",
        _df_to_md(day_compare_df),
        "",
        "## 3. Day-specific feature shifts",
        "",
        _df_to_md(shift_df),
        "",
        "## 4. Day2 concentration behavior",
        "",
        _df_to_md(day2_metrics_df),
        "",
        "## 5. Day0 negative control",
        "",
        _df_to_md(day0_metrics_df),
        "",
        "## 6. Temporary stress-readout score",
        "",
        _df_to_md(stress_metrics_df),
        "",
        "## 7. Final conclusion",
        "",
        (REPORT_DIR / "set9_day0_day2_extension_decision.md").read_text(encoding="utf-8"),
    ]
    report_md = REPORT_DIR / "GAIRAv3_SHINE_Set9_Day0_Day2_extension_validation_report.md"
    report_md.write_text("\n".join(report_lines), encoding="utf-8")
    figure_paths = [
        FIGURES_DIR / "set9_day0_vs_day2_spectral_pca.png",
        FIGURES_DIR / "set9_day0_vs_day2_bsv_pca.png",
        FIGURES_DIR / "set9_day0_vs_day2_family_pca.png",
        FIGURES_DIR / "set9_day0_vs_day2_extension_pca.png",
        FIGURES_DIR / "set9_day0_vs_day2_combined_pca.png",
        FIGURES_DIR / "set9_day_feature_shift_bars.png",
        FIGURES_DIR / "set9_day2_extension_concentration_pca.png",
        FIGURES_DIR / "set9_day2_extension_concentration_trends.png",
        FIGURES_DIR / "set9_day0_extension_concentration_pca.png",
        FIGURES_DIR / "set9_day0_extension_concentration_trends.png",
        FIGURES_DIR / "set9_temporary_stress_readout_day_comparison.png",
        FIGURES_DIR / "set9_temporary_stress_readout_day2_trend.png",
        FIGURES_DIR / "set9_day0_extended_radar_by_concentration.png",
        FIGURES_DIR / "set9_day2_extended_radar_by_concentration.png",
        FIGURES_DIR / "set9_day0_vs_day2_extended_radar_summary.png",
    ]
    build_pdf_report(report_md, [p for p in figure_paths if p.exists()], REPORT_DIR / "GAIRAv3_SHINE_Set9_Day0_Day2_extension_validation_report.pdf")


if __name__ == "__main__":
    main()
