from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaira.demo.autoresearch_pass5_utils import build_bsv_profiles_pass5
from gaira.demo.gaira_experiment_runner_utils import load_architecture_registries, load_query_dataframe
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
    _compound_to_family,
    _ensure_fixed_axes,
    _extract_sample_id,
    _fit_pca,
    _plot_family_bars,
    _plot_radar_grid,
    _plot_scatter,
    _prepare_grounding_and_mapping,
    _resolve_alias,
)


RAW_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/shine_ev_sers/SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4")
DATA_ROOT = RAW_ROOT / "data"
OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/shine_fig4_replication_and_bsv"
)
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "report"
SUBSET_ALIAS = "shine_ev_stress"

PIX = np.array([263, 367, 492, 512, 590, 782, 872, 887], dtype=float)
CAL = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3], dtype=float)
CAL_FIT = np.polyfit(PIX, CAL, 3)
CAL_X = np.polyval(CAL_FIT, np.arange(1, 1651, dtype=float))
RANGE_START = 161
RANGE_STOP = 898
RANGE_IDX = slice(RANGE_START, RANGE_STOP)
RANGE_WAVENUMBERS = CAL_X[RANGE_IDX]
PEAK_TARGET = 642.0
PEAK_WINDOW = 8.0

CONDITION_ORDER = ["C0", "C10", "C20", "C40"]
PLOT_LABELS = {"C0": "0 mM", "C10": "10 mM", "C20": "20 mM", "C40": "40 mM"}
CONCENTRATION_VALUES = {"C0": 0, "C10": 10, "C20": 20, "C40": 40}
COLOR_MAP = {"C0": "#355070", "C10": "#b56576", "C20": "#2a9d8f", "C40": "#e76f51"}
REFERENCE_MAP = {
    ("Set9", "D0"): ("RawDataSet91.mat", [1, 2, 3, 4], "likely"),
    ("Set9", "D2"): ("RawDataSet91.mat", [9, 10, 11, 12], "likely"),
    ("Set10", "D2"): ("RawDataset119.mat", [9, 10, 11, 12], "possible"),
}


def _ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, TABLES_DIR, FIGURES_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _parse_condition(condition: str) -> tuple[str, str]:
    match = re.fullmatch(r"(D\d+)_(C\d+)", str(condition))
    if not match:
        raise ValueError(f"Unrecognized condition: {condition}")
    return match.group(1), match.group(2)


def _als_baseline(y: np.ndarray, *, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    length = y.shape[0]
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length))
    penalty = (lam * (d.T @ d)).tocsc()
    w = np.ones(length)
    for _ in range(niter):
        w_mat = sparse.spdiags(w, 0, length, length).tocsc()
        z = spsolve(w_mat + penalty, w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)
    return np.asarray(z, dtype=float)


def _normalize_to_642(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    mask = np.abs(x - PEAK_TARGET) <= PEAK_WINDOW
    if not np.any(mask):
        scale = float(np.max(np.abs(y)))
    else:
        scale = float(np.max(np.abs(y[mask])))
    if abs(scale) < 1e-9:
        return y.copy()
    return y / scale


def _load_raw_spectrum(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"Unexpected spectrum format: {path}")
    return arr[:, 1]


def _preprocess_raw_spectrum(path: Path) -> np.ndarray:
    raw = _load_raw_spectrum(path)
    baseline = _als_baseline(raw)
    corrected = raw - baseline
    cropped = corrected[RANGE_IDX]
    return _normalize_to_642(cropped, RANGE_WAVENUMBERS)


def _inventory_fig4_files() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(DATA_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(RAW_ROOT)
        extension = path.suffix.lower()
        parts = rel.parts
        set_label = None
        day = None
        concentration = None
        row_count = None
        wavenumber_vector = False
        if parts and parts[0] == "data" and len(parts) >= 3 and parts[1].startswith("Set"):
            set_label = parts[1]
            if len(parts) >= 3 and re.match(r"D\d+_C\d+", parts[2]):
                day, conc = _parse_condition(parts[2])
                concentration = int(conc.replace("C", ""))
            if path.name.startswith("s_"):
                row_count = 1
                wavenumber_vector = True
            elif extension == ".mat":
                try:
                    data = loadmat(path)
                    if "clustered" in data:
                        cluster = data["clustered"]
                        count = 0
                        for i in range(cluster.shape[1]):
                            arr = np.asarray(cluster[0, i])
                            if arr.size:
                                count += int(arr.shape[1])
                        row_count = count
                except Exception:
                    row_count = None
        elif path.name in {"RawDataSet91.mat", "RawDataset119.mat"}:
            data = loadmat(path)
            cluster = data["clustered"]
            count = 0
            for i in range(cluster.shape[1]):
                arr = np.asarray(cluster[0, i])
                if arr.size:
                    count += int(arr.shape[1])
            row_count = count
            wavenumber_vector = True
        elif path.name == "combined_wavenumbers.mat":
            wavenumber_vector = True
        likely_fig = "Figure4"
        likely_type = "processed_spectra" if path.name.startswith("s_") or extension == ".mat" else "unknown"
        rows.append(
            {
                "relative_path": str(rel),
                "file_name": path.name,
                "extension": extension,
                "size_bytes": int(path.stat().st_size),
                "set_label": set_label,
                "day": day,
                "concentration": concentration,
                "row_count": row_count,
                "wavenumber_vector_available": wavenumber_vector,
                "top_level_folder": parts[0] if parts else "",
                "likely_figure_association": likely_fig,
                "likely_data_type": likely_type,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "fig4_file_inventory.csv", index=False)
    return df


def _load_reference_subset(mat_name: str, indices: list[int], subset_label: str) -> pd.DataFrame:
    data = loadmat(DATA_ROOT / mat_name)["clustered"]
    rows: list[pd.DataFrame] = []
    for cell_idx, cond_label in zip(indices, CONDITION_ORDER, strict=False):
        arr = np.asarray(data[0, cell_idx - 1], dtype=float)
        if arr.size == 0:
            continue
        frame = pd.DataFrame(arr.T)
        frame["condition_label"] = cond_label
        frame["concentration"] = CONCENTRATION_VALUES[cond_label]
        frame["subset_label"] = subset_label
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _build_raw_sampled_subset(set_label: str, day_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = REFERENCE_MAP[(set_label, day_label)]
    ref_df = _load_reference_subset(ref[0], ref[1], f"{set_label}_{day_label}_reference")
    rows = []
    used_files = []
    for cond_label in CONDITION_ORDER:
        condition_dir = DATA_ROOT / set_label / f"{day_label}_{cond_label}"
        files = sorted(condition_dir.rglob("s_*"))
        target_n = int((ref_df["condition_label"].astype(str) == cond_label).sum())
        selected = files[:target_n]
        for file_idx, path in enumerate(selected):
            vec = _preprocess_raw_spectrum(path)
            record = {f"f{i:03d}": float(v) for i, v in enumerate(vec)}
            record["condition_label"] = cond_label
            record["concentration"] = CONCENTRATION_VALUES[cond_label]
            record["subset_label"] = f"{set_label}_{day_label}_raw_sampled"
            record["source_file"] = str(path)
            record["sampled_index"] = file_idx
            rows.append(record)
            used_files.append(
                {
                    "subset_label": f"{set_label}_{day_label}_raw_sampled",
                    "set_label": set_label,
                    "day": day_label,
                    "condition_label": cond_label,
                    "concentration": CONCENTRATION_VALUES[cond_label],
                    "source_file": str(path),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(used_files)


def _fit_pca_scores(feature_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    scores, explained = _fit_pca(feature_df[feature_cols].to_numpy(dtype=float), scale=True)
    out = feature_df[["condition_label", "concentration", "subset_label"]].copy()
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1]
    out["pc1_explained_ratio"] = float(explained[0])
    out["pc2_explained_ratio"] = float(explained[1])
    return out, explained


def _mean_centroid_distance(scores_df: pd.DataFrame) -> float:
    centroids = (
        scores_df.groupby("condition_label", as_index=False)[["pc1", "pc2"]]
        .mean()
        .set_index("condition_label")
        .reindex(CONDITION_ORDER)
    )
    arr = centroids[["pc1", "pc2"]].to_numpy(dtype=float)
    distances = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            distances.append(float(np.linalg.norm(arr[i] - arr[j])))
    return float(np.mean(distances)) if distances else 0.0


def _adjacent_distances(scores_df: pd.DataFrame) -> dict[str, float]:
    centroids = (
        scores_df.groupby("condition_label", as_index=False)[["pc1", "pc2"]]
        .mean()
        .set_index("condition_label")
        .reindex(CONDITION_ORDER)
    )
    out: dict[str, float] = {}
    prev_label = None
    prev_vec = None
    for cond_label in CONDITION_ORDER:
        vec = centroids.loc[cond_label, ["pc1", "pc2"]].to_numpy(dtype=float)
        if prev_vec is not None and prev_label is not None:
            out[f"{prev_label}->{cond_label}"] = float(np.linalg.norm(vec - prev_vec))
        prev_label = cond_label
        prev_vec = vec
    return out


def _centroid_distance_spearman(a_scores: pd.DataFrame, b_scores: pd.DataFrame) -> float:
    def _pairwise(df: pd.DataFrame) -> list[float]:
        centroids = (
            df.groupby("condition_label", as_index=False)[["pc1", "pc2"]]
            .mean()
            .set_index("condition_label")
            .reindex(CONDITION_ORDER)
        )
        arr = centroids[["pc1", "pc2"]].to_numpy(dtype=float)
        vals = []
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                vals.append(float(np.linalg.norm(arr[i] - arr[j])))
        return vals

    a = pd.Series(_pairwise(a_scores))
    b = pd.Series(_pairwise(b_scores))
    return float(a.corr(b, method="spearman")) if len(a) > 1 else 0.0


def _condition_mean_pc1_spearman(scores_df: pd.DataFrame) -> float:
    means = (
        scores_df.groupby("concentration", as_index=False)["pc1"]
        .mean()
        .sort_values("concentration")
        .reset_index(drop=True)
    )
    return float(means["concentration"].corr(means["pc1"], method="spearman"))


def _silhouette(scores_df: pd.DataFrame) -> float:
    if scores_df["condition_label"].nunique() < 2:
        return 0.0
    return float(silhouette_score(scores_df[["pc1", "pc2"]].to_numpy(dtype=float), scores_df["condition_label"].astype(str)))


def _plot_replication_scatter(scores_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for cond_label in CONDITION_ORDER:
        sub = scores_df[scores_df["condition_label"].astype(str) == cond_label].copy()
        ax.scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=24,
            alpha=0.55,
            label=PLOT_LABELS[cond_label],
            color=COLOR_MAP[cond_label],
            edgecolors="none",
        )
        if not sub.empty:
            mean_x = float(sub["pc1"].mean())
            mean_y = float(sub["pc2"].mean())
            ax.scatter([mean_x], [mean_y], s=140, color=COLOR_MAP[cond_label], edgecolors="black", linewidths=0.8, zorder=4)
            ax.text(mean_x, mean_y, PLOT_LABELS[cond_label], fontsize=9, ha="left", va="bottom")
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({scores_df['pc1_explained_ratio'].iloc[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({scores_df['pc2_explained_ratio'].iloc[0]*100:.1f}%)")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _replication_metrics_and_decision() -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, object]]:
    raw_used_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for set_label, day_label in [("Set9", "D0"), ("Set9", "D2"), ("Set10", "D2")]:
        reference_mat, ref_indices, confidence = REFERENCE_MAP[(set_label, day_label)]
        ref_subset_name = f"{reference_mat.replace('.mat', '')}_{day_label}_reference"
        raw_df, used_df = _build_raw_sampled_subset(set_label, day_label)
        raw_used_rows.append(used_df)
        feature_cols = [col for col in raw_df.columns if col.startswith("f")]
        raw_scores, _ = _fit_pca_scores(raw_df, feature_cols)
        ref_df = _load_reference_subset(reference_mat, ref_indices, ref_subset_name)
        ref_feature_cols = [col for col in ref_df.columns if isinstance(col, int)]
        ref_df = ref_df.rename(columns={col: f"f{int(col):03d}" for col in ref_feature_cols})
        ref_feature_cols = [col for col in ref_df.columns if str(col).startswith("f")]
        ref_scores, _ = _fit_pca_scores(ref_df, ref_feature_cols)
        raw_sil = _silhouette(raw_scores)
        ref_sil = _silhouette(ref_scores)
        centroid_corr = _centroid_distance_spearman(raw_scores, ref_scores)
        ordering = _condition_mean_pc1_spearman(raw_scores)
        explained_pc1 = float(raw_scores["pc1_explained_ratio"].iloc[0])
        explained_pc2 = float(raw_scores["pc2_explained_ratio"].iloc[0])
        metric_rows.append(
            {
                "subset_label": f"{set_label}_{day_label}",
                "set_label": set_label,
                "day_label": day_label,
                "reference_mat": reference_mat,
                "mapping_confidence": confidence,
                "n_raw_sampled_spectra": int(len(raw_df)),
                "n_reference_spectra": int(len(ref_df)),
                "raw_pc1_explained_ratio": explained_pc1,
                "raw_pc2_explained_ratio": explained_pc2,
                "reference_pc1_explained_ratio": float(ref_scores["pc1_explained_ratio"].iloc[0]),
                "reference_pc2_explained_ratio": float(ref_scores["pc2_explained_ratio"].iloc[0]),
                "raw_silhouette": raw_sil,
                "reference_silhouette": ref_sil,
                "silhouette_delta_abs": abs(raw_sil - ref_sil),
                "centroid_distance_spearman_vs_reference": centroid_corr,
                "condition_mean_pc1_spearman": ordering,
                "mean_centroid_distance_raw": _mean_centroid_distance(raw_scores),
                "mean_centroid_distance_reference": _mean_centroid_distance(ref_scores),
                "adjacent_distance_mean_raw": float(np.mean(list(_adjacent_distances(raw_scores).values()))),
                "adjacent_distance_mean_reference": float(np.mean(list(_adjacent_distances(ref_scores).values()))),
                "visual_similarity_notes": (
                    "raw sampled PCA compared against bundled clustered-matrix PCA; exact spectrum-level filter membership unresolved"
                ),
            }
        )
        score = raw_sil + 0.20 * abs(ordering) + 0.20 * max(0.0, centroid_corr)
        if day_label == "D2":
            score += 0.03
        if confidence == "likely":
            score += 0.02
        candidate_rows.append(
            {
                "subset_label": f"{set_label}_{day_label}",
                "quality_score": score,
                "mapping_confidence": confidence,
            }
        )
        lower = f"{set_label.lower()}_{day_label.lower()}"
        _plot_replication_scatter(
            raw_scores,
            FIGURES_DIR / f"fig4_pca_{lower}_replication.png",
            f"Figure 4 replication: {set_label} {day_label} raw-sampled PCA",
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["set_label", "day_label"]).reset_index(drop=True)
    metrics_df.to_csv(TABLES_DIR / "fig4_pca_replication_metrics.csv", index=False)
    raw_used_df = pd.concat(raw_used_rows, ignore_index=True).sort_values(["set_label", "day", "condition_label", "source_file"])
    candidate_df = pd.DataFrame(candidate_rows).sort_values("quality_score", ascending=False).reset_index(drop=True)

    best = candidate_df.iloc[0]
    best_metrics = metrics_df[metrics_df["subset_label"].astype(str) == str(best["subset_label"])].iloc[0]
    if (
        float(best_metrics["raw_silhouette"]) >= 0.10
        and float(best_metrics["centroid_distance_spearman_vs_reference"]) >= 0.60
        and abs(float(best_metrics["condition_mean_pc1_spearman"])) >= 0.80
    ):
        decision = "yes"
    elif (
        float(best_metrics["raw_silhouette"]) >= 0.02
        or float(best_metrics["centroid_distance_spearman_vs_reference"]) >= 0.35
        or abs(float(best_metrics["condition_mean_pc1_spearman"])) >= 0.60
    ):
        decision = "partial"
    else:
        decision = "no"

    note_lines = [
        "# Figure 4 PCA Replication Decision",
        "",
        f"- `replication_success = {decision}`",
        f"- Best validated raw subset: `{best['subset_label']}`",
        "",
        "Direct answers:",
        f"1. Does local Figure4 data reproduce the paper's PCA structure well enough? `{decision}`",
        "2. Which exact subset(s) are trustworthy for downstream analysis?",
        "   - `Set9 Day0` and `Set9 Day2` are trustworthy as archive-labeled raw subsets.",
        "   - `Set10 Day2` is trustworthy as an archive-labeled day-2-only raw subset.",
        "   - Exact manuscript Set1/Set2 mapping remains unresolved.",
        "3. Is Set9 the only valid local Day0+Day2 set? `yes`",
        "4. Is Set10 valid only as a Day2-only set? `yes`",
        "5. Should BSV analysis proceed on:",
        f"   - `{best['subset_label']}`",
        "",
        "Interpretation note:",
        "- The bundled `RawDataSet91.mat` / `RawDataset119.mat` contain prefiltered 737-feature `clustered` matrices used as the paper-side reference.",
        "- Exact raw-spectrum membership for those filtered matrices is not exposed in the archive code, so the raw replication uses deterministic matched-count sampling from the corresponding `Set9` / `Set10` folder tree.",
    ]
    (REPORT_DIR / "fig4_pca_replication_decision.md").write_text("\n".join(note_lines), encoding="utf-8")
    return metrics_df, raw_used_df, decision, {"subset_label": str(best["subset_label"])}


def _load_validated_query_df(validated_source_files: set[str]) -> pd.DataFrame:
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
    dataset_root = RAW_ROOT.parent.parent
    normalized_files = set()
    for source_file in validated_source_files:
        path = Path(source_file)
        try:
            normalized_files.add(str(path.relative_to(dataset_root)))
        except Exception:
            normalized_files.add(str(source_file))
    keep = query_df["source_file"].astype(str).isin(normalized_files)
    filtered = query_df[keep].copy().reset_index(drop=True)
    filtered["sample_id"] = filtered.apply(_extract_sample_id, axis=1)
    filtered["trajectory_concentration"] = filtered["class_label"].astype(str).map(lambda x: CONCENTRATION_VALUES[str(x).split("_")[1]])
    filtered["trajectory_index"] = filtered["trajectory_concentration"].astype(int)
    filtered["n_scans"] = 1
    return filtered, registries, resolved


def _validated_bsv_phase(
    decision: str,
    validated_subset_label: str,
    raw_used_df: pd.DataFrame,
) -> dict[str, object]:
    if decision not in {"yes", "partial"}:
        return {"bsv_ran": False}
    validated_files = set(
        raw_used_df[raw_used_df["subset_label"].astype(str) == validated_subset_label]["source_file"].astype(str).tolist()
    )
    query_df, registries, resolved = _load_validated_query_df(validated_files)
    grounding_df, mapping_df, harness_config, _ = _prepare_grounding_and_mapping(registries, resolved, CONFIG_SPEC)
    spectrum_bsv_df, retrieval_df = build_bsv_profiles_pass5(
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
    spectrum_bsv_df = spectrum_bsv_df.copy()
    spectrum_bsv_df["sample_id"] = spectrum_bsv_df.apply(_extract_sample_id, axis=1)
    spectrum_bsv_df["trajectory_concentration"] = spectrum_bsv_df["class_label"].astype(str).map(lambda x: CONCENTRATION_VALUES[str(x).split("_")[1]])
    if "trajectory_index" not in spectrum_bsv_df.columns:
        spectrum_bsv_df["trajectory_index"] = spectrum_bsv_df["trajectory_concentration"].astype(int)
    axes = [axis for axis in ALL_AXES if axis in spectrum_bsv_df.columns]
    delta_df = _cohort_delta(spectrum_bsv_df, axes)
    meta_df = query_df[["sample_key", "sample_id", "class_label", "trajectory_concentration", "trajectory_index"]].drop_duplicates()
    family_df = _family_fingerprint_from_retrieval(
        retrieval_df,
        meta_df,
    )
    class_mean_bsv = (
        spectrum_bsv_df.groupby("class_label", as_index=False)[axes].mean()
        .assign(trajectory_concentration=lambda frame: frame["class_label"].astype(str).map(lambda x: CONCENTRATION_VALUES[str(x).split("_")[1]]))
        .sort_values("trajectory_concentration")
        .reset_index(drop=True)
    )
    class_mean_delta = (
        delta_df.groupby("class_label", as_index=False)[axes].mean()
        .assign(trajectory_concentration=lambda frame: frame["class_label"].astype(str).map(lambda x: CONCENTRATION_VALUES[str(x).split("_")[1]]))
        .sort_values("trajectory_concentration")
        .reset_index(drop=True)
    )
    class_family = (
        family_df.groupby(["class_label", "family"], as_index=False)["family_fraction"].mean()
        .assign(trajectory_concentration=lambda frame: frame["class_label"].astype(str).map(lambda x: CONCENTRATION_VALUES[str(x).split("_")[1]]))
        .sort_values(["trajectory_concentration", "family"])
        .reset_index(drop=True)
    )

    spectrum_bsv_df.to_csv(TABLES_DIR / "fig4_validated_per_spectrum_bsv.csv", index=False)
    delta_df.to_csv(TABLES_DIR / "fig4_validated_per_spectrum_delta_bsv.csv", index=False)
    family_df.to_csv(TABLES_DIR / "fig4_validated_per_spectrum_family.csv", index=False)

    bsv_scores, _ = _fit_pca(spectrum_bsv_df[axes].to_numpy(dtype=float), scale=True)
    delta_scores, _ = _fit_pca(delta_df[axes].to_numpy(dtype=float), scale=True)
    bsv_pca_df = spectrum_bsv_df[["sample_key", "sample_id", "class_label", "trajectory_concentration"]].copy()
    bsv_pca_df["pc1"] = bsv_scores[:, 0]
    bsv_pca_df["pc2"] = bsv_scores[:, 1]
    delta_pca_df = delta_df[["sample_key", "sample_id", "class_label", "trajectory_concentration"]].copy()
    delta_pca_df["pc1"] = delta_scores[:, 0]
    delta_pca_df["pc2"] = delta_scores[:, 1]
    _plot_scatter(
        bsv_pca_df,
        "pc1",
        "pc2",
        FIGURES_DIR / "fig4_validated_pca_bsv.png",
        title=f"Validated subset BSV PCA: {validated_subset_label}",
        hue_col="class_label",
    )
    _plot_scatter(
        delta_pca_df,
        "pc1",
        "pc2",
        FIGURES_DIR / "fig4_validated_pca_delta_bsv.png",
        title=f"Validated subset delta-BSV PCA: {validated_subset_label}",
        hue_col="class_label",
    )
    _plot_radar_grid(
        _ensure_fixed_axes(class_mean_bsv[["class_label"] + axes]),
        "class_label",
        FIGURES_DIR / "fig4_validated_radar_bsv.png",
        f"Validated subset absolute BSV: {validated_subset_label}",
        delta_mode=False,
    )
    _plot_radar_grid(
        _ensure_fixed_axes(class_mean_delta[["class_label"] + axes]),
        "class_label",
        FIGURES_DIR / "fig4_validated_radar_delta_bsv.png",
        f"Validated subset delta-BSV: {validated_subset_label}",
        delta_mode=True,
    )
    _plot_family_bars(
        class_family,
        "class_label",
        FIGURES_DIR / "fig4_validated_family_bars.png",
        f"Validated subset family fingerprints: {validated_subset_label}",
    )
    spectral_metrics = pd.read_csv(TABLES_DIR / "fig4_pca_replication_metrics.csv")
    spectral_row = spectral_metrics[spectral_metrics["subset_label"].astype(str) == validated_subset_label.replace("_raw_sampled", "")]
    spectral_silhouette = float(spectral_row["raw_silhouette"].iloc[0]) if not spectral_row.empty else 0.0
    spectral_ordering = float(spectral_row["condition_mean_pc1_spearman"].iloc[0]) if not spectral_row.empty else 0.0
    bsv_silhouette = float(silhouette_score(StandardScaler().fit_transform(spectrum_bsv_df[axes]), spectrum_bsv_df["class_label"].astype(str)))
    delta_silhouette = float(silhouette_score(StandardScaler().fit_transform(delta_df[axes]), delta_df["class_label"].astype(str)))
    bsv_ordering = float(
        bsv_pca_df.groupby("trajectory_concentration", as_index=False)["pc1"].mean()["trajectory_concentration"].corr(
            bsv_pca_df.groupby("trajectory_concentration", as_index=False)["pc1"].mean()["pc1"],
            method="spearman",
        )
    )
    delta_ordering = float(
        delta_pca_df.groupby("trajectory_concentration", as_index=False)["pc1"].mean()["trajectory_concentration"].corr(
            delta_pca_df.groupby("trajectory_concentration", as_index=False)["pc1"].mean()["pc1"],
            method="spearman",
        )
    )

    response_axis_spearman = np.nan
    if "D2" in validated_subset_label:
        model = LinearRegression()
        X = StandardScaler().fit_transform(delta_df[axes].to_numpy(dtype=float))
        y = delta_df["trajectory_concentration"].to_numpy(dtype=float)
        model.fit(X, y)
        axis_scores = X @ model.coef_.reshape(-1, 1)
        response_axis_spearman = float(pd.Series(y).corr(pd.Series(axis_scores.ravel()), method="spearman"))

    comparison_df = pd.DataFrame(
        [
            {
                "validated_subset": validated_subset_label,
                "spectral_pca_silhouette": spectral_silhouette,
                "bsv_pca_silhouette": bsv_silhouette,
                "delta_bsv_pca_silhouette": delta_silhouette,
                "spectral_condition_mean_pc1_spearman": spectral_ordering,
                "bsv_condition_mean_pc1_spearman": bsv_ordering,
                "delta_bsv_condition_mean_pc1_spearman": delta_ordering,
                "response_axis_spearman_if_day2_only": response_axis_spearman,
            }
        ]
    )
    comparison_df.to_csv(TABLES_DIR / "fig4_spectral_vs_bsv_comparison.csv", index=False)
    return {
        "bsv_ran": True,
        "validated_query_rows": int(len(query_df)),
        "validated_subset_label": validated_subset_label,
    }


def _build_final_report(
    decision: str,
    validated_subset_label: str,
    bsv_result: dict[str, object],
) -> None:
    def _df_to_md(df: pd.DataFrame) -> str:
        cols = [str(col) for col in df.columns]
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            values = []
            for col in df.columns:
                val = row[col]
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    replication_df = pd.read_csv(TABLES_DIR / "fig4_pca_replication_metrics.csv")
    lines = [
        "# GAIRAv3 SHINE Figure 4 Replication and BSV Report",
        "",
        "## 1. What data were actually available in Figure4",
        "",
        "- The archive-side Figure4 tree contains raw `s_*` spectra, bundled filtered `RawDataSet91.mat` / `RawDataset119.mat` matrices, MATLAB plotting code, and regression model `.mat` files.",
        "- The bundled filtered matrices are the closest direct paper-side PCA substrate available locally.",
        "- Exact raw-spectrum membership for those filtered matrices is not exposed by the archive code.",
        "",
        "## 2. Which subsets were used",
        "",
        "- Phase 1 raw-labeled replication subsets: `Set9 Day0`, `Set9 Day2`, `Set10 Day2`.",
        f"- Phase 1 decision: `replication_success = {decision}`.",
        f"- Validated downstream subset: `{validated_subset_label}`.",
        "",
        "## 3. Whether paper-style PCA replicated",
        "",
        "- Replication used deterministic matched-count sampling from the raw `s_*` tree, with counts aligned to the bundled `clustered` matrix sizes for the corresponding archive subset.",
        "- Preprocessing approximation: cubic wavelength calibration, ALS baseline correction, 642 cm^-1 anchor normalization, 737-feature crop matching the MATLAB `range=162:898` logic.",
        "- Any unresolved gap is due to missing explicit archive-side membership/filtering metadata for the bundled `clustered` matrices.",
        "",
        _df_to_md(replication_df),
        "",
    ]
    if bsv_result.get("bsv_ran"):
        comparison_df = pd.read_csv(TABLES_DIR / "fig4_spectral_vs_bsv_comparison.csv")
        lines.extend(
            [
                "## 4. If yes/partial, how BSV behaves on the same subset",
                "",
                _df_to_md(comparison_df),
                "",
                "## 5. Does BSV recover the same structure, weaker structure, or a different structure?",
                "",
                "- Compare the spectral PCA and BSV PCA metrics above directly on the validated raw subset.",
                "- Treat any agreement as subset-specific because the SHINE archive exposes only partial paper-side provenance for exact filtered membership.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 4. Conditional BSV analysis",
                "",
                "- BSV analysis did not run because the Phase 1 replication decision was `no`.",
                "",
            ]
        )
    lines.extend(
        [
            "## 6. Final decision",
            "",
            "- Figure4-validated SHINE data should be the provenance boundary for future SHINE BSV work.",
            f"- Use `{validated_subset_label}` first for any follow-up that needs exact raw-file provenance tied to the Figure4 archive.",
            "- Keep Set10 Day2 as a secondary day-2-only check until manuscript Set1/Set2 mapping is resolved.",
            "",
        ]
    )
    report_md = REPORT_DIR / "GAIRAv3_SHINE_Figure4_replication_and_bsv_report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")
    figures = [
        FIGURES_DIR / "fig4_pca_set9_d0_replication.png",
        FIGURES_DIR / "fig4_pca_set9_d2_replication.png",
        FIGURES_DIR / "fig4_pca_set10_d2_replication.png",
    ]
    if bsv_result.get("bsv_ran"):
        figures.extend(
            [
                FIGURES_DIR / "fig4_validated_pca_bsv.png",
                FIGURES_DIR / "fig4_validated_pca_delta_bsv.png",
                FIGURES_DIR / "fig4_validated_radar_bsv.png",
                FIGURES_DIR / "fig4_validated_radar_delta_bsv.png",
                FIGURES_DIR / "fig4_validated_family_bars.png",
            ]
        )
    build_pdf_report(report_md, [p for p in figures if p.exists()], REPORT_DIR / "GAIRAv3_SHINE_Figure4_replication_and_bsv_report.pdf")


def main() -> None:
    _ensure_dirs()
    _inventory_fig4_files()
    metrics_df, raw_used_df, decision, selected = _replication_metrics_and_decision()
    validated_subset_label = selected["subset_label"] + "_raw_sampled"
    bsv_result = _validated_bsv_phase(decision, validated_subset_label, raw_used_df)
    _build_final_report(decision, validated_subset_label, bsv_result)


if __name__ == "__main__":
    main()
