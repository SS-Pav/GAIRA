from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from gaira.autoresearch_storage import (
    DEFAULT_STORAGE_CONFIG_PATH,
    initialize_autoresearch_sprint,
    load_autoresearch_storage_config,
)
from gaira.demo.gaira_pilot_utils import ALL_AXES, build_pdf_report, infer_mixture_order


ROOT = Path(__file__).resolve().parents[1]
PILOT1A_V5_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot1a_celltype_probe1_v5"
)
PILOT1B_PROBE1_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot1b_mixture_probe1_v1"
)
COMPARATOR_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot1_comparator_cfg05_cfg08"
)
SPRINT_SUBDIR = "pilot1c_probe_consistency_v1"

CONFIG_SPECS = [
    {
        "config_id": "candidate_v2_cfg05_max_desaturation",
        "short_label": "cfg05",
        "display_name": "Candidate v2 cfg05 max desaturation",
    },
    {
        "config_id": "candidate_v2_cfg08_balanced_update",
        "short_label": "cfg08",
        "display_name": "Candidate v2 cfg08 balanced update",
    },
]

PROBE1_ALIAS = "small2023_mixture_probe1"
PROBE2_ALIAS = "small2023_mixture_probe2"

BASE_REUSED_FILES = [
    "per_spectrum_bsv.csv",
    "class_mean_bsv.csv",
    "pairwise_delta_bsv.csv",
    "class_topk_neighborhood_composition.csv",
    "class_neighborhood_entropy.csv",
    "class_top1_dominance.csv",
    "class_axis_entropy.csv",
    "retrieval_hit_summary_by_class.csv",
    "per_spectrum_retrieval_hits.csv",
    "pca_coordinates_spectral.csv",
]

PROBE1_EXTRA_FILES = [
    "class_mean_bsv_delta_vs_cohort.csv",
    "class_neighborhood_family_composition.csv",
    "class_fingerprint_summary.csv",
    "endpoint_alignment_summary.csv",
    "progression_metrics.csv",
    "noncollapse_metrics.csv",
    "adjacent_progression_steps.csv",
    "pca_coordinates_bsv.csv",
    "pca_coordinates_bsv_class_mean.csv",
]

FAMILY_ORDER = [
    "purine_core_like",
    "methylated_purine_like",
    "guanidine_like",
    "sulfur_small_molecule_like",
    "aromatic_small_molecule_like",
    "generic_other_metabolite",
]

CLASS_COLORS = {
    "c00": "#355070",
    "c01": "#6d597a",
    "c10": "#b56576",
    "c25": "#e56b6f",
    "c50": "#eaac8b",
    "c100": "#f4a261",
}
PROBE_MARKERS = {"probe1": "o", "probe2": "s"}
PROBE_LINESTYLES = {"probe1": "-", "probe2": "--"}
CONFIG_COLORS = {"cfg05": "#f3722c", "cfg08": "#43aa8b"}


def _mixture_sort_key(label: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    return (int(digits) if digits else 10**9, str(label))


def _require_inputs() -> None:
    for root in [PILOT1A_V5_ROOT, PILOT1B_PROBE1_ROOT, COMPARATOR_ROOT]:
        if not root.exists():
            raise RuntimeError(f"Missing required root: {root}")
    for spec in CONFIG_SPECS:
        probe1_root = PILOT1B_PROBE1_ROOT / "runs" / str(spec["config_id"])
        for name in BASE_REUSED_FILES + PROBE1_EXTRA_FILES:
            path = probe1_root / name
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(f"Missing Probe 1 artifact: {path}")
        probe2_root = COMPARATOR_ROOT / "runs" / str(spec["config_id"]) / PROBE2_ALIAS / "tables"
        for name in [
            "per_spectrum_bsv.csv",
            "class_mean_bsv.csv",
            "pairwise_delta_bsv.csv",
            "class_topk_neighborhood_composition.csv",
            "class_neighborhood_entropy.csv",
            "class_top1_dominance.csv",
            "class_axis_entropy.csv",
            "retrieval_hit_summary_by_class.csv",
            "per_spectrum_retrieval_hits.csv",
            "pca_coordinates.csv",
        ]:
            path = probe2_root / name
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(f"Missing Probe 2 artifact: {path}")
        endpoint_root = PILOT1A_V5_ROOT / "runs" / str(spec["config_id"])
        for name in ["class_mean_bsv.csv", "delta_class_mean_bsv.csv", "class_family_fingerprint.csv"]:
            path = endpoint_root / name
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(f"Missing endpoint artifact: {path}")


def _compound_to_fine_family(name: str) -> str:
    lower = str(name).strip().lower()
    if any(token in lower for token in ["3-methyladenine", "methyladenine"]):
        return "methylated_purine_like"
    if any(token in lower for token in ["guanidine", "guanidino"]):
        return "guanidine_like"
    if any(token in lower for token in ["cyste", "glutath", "methion", "seleno", "sulfoximine", "sulfur"]):
        return "sulfur_small_molecule_like"
    if any(token in lower for token in ["tyr", "trypt", "phenyl", "indole", "dopamine", "3-methoxytyramine"]):
        return "aromatic_small_molecule_like"
    if any(token in lower for token in ["adenine", "xanth", "hypox", "uric", "urate", "inos", "purine"]):
        return "purine_core_like"
    return "generic_other_metabolite"


def _axes_present(df: pd.DataFrame) -> list[str]:
    return [axis for axis in ALL_AXES if axis in df.columns]


def _copy_probe1_outputs(run_dir: Path, config_id: str) -> None:
    source_root = PILOT1B_PROBE1_ROOT / "runs" / config_id
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name in BASE_REUSED_FILES + PROBE1_EXTRA_FILES:
        src = source_root / name
        shutil.copy2(src, run_dir / name)
        shutil.copy2(src, tables_dir / name)


def _build_family_fingerprint(class_neighborhood_df: pd.DataFrame) -> pd.DataFrame:
    df = class_neighborhood_df.copy()
    df["neighborhood_family"] = df["compound_label"].astype(str).map(_compound_to_fine_family)
    grouped = (
        df.groupby(["class_label", "neighborhood_family"], as_index=False)["support_fraction"]
        .sum()
        .rename(columns={"support_fraction": "family_support_fraction"})
    )
    rows = []
    for class_label in sorted(df["class_label"].astype(str).unique().tolist(), key=_mixture_sort_key):
        sub = grouped[grouped["class_label"].astype(str) == class_label].copy()
        existing = {str(x) for x in sub["neighborhood_family"].tolist()}
        for family in FAMILY_ORDER:
            if family not in existing:
                rows.append(
                    {
                        "class_label": class_label,
                        "neighborhood_family": family,
                        "family_support_fraction": 0.0,
                    }
                )
    if rows:
        grouped = pd.concat([grouped, pd.DataFrame(rows)], ignore_index=True)
    return grouped.sort_values(["class_label", "neighborhood_family"]).reset_index(drop=True)


def _fit_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, keepdims=True)
    std = np.where(std < 1e-9, 1.0, std)
    centered = centered / std
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * s[:2]
    denom = max(float((s**2).sum()), 1e-12)
    explained = (s[:2] ** 2) / denom if len(s) >= 2 else np.array([1.0, 0.0])
    return scores, explained


def _pca_dataframe(df: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    scores, explained = _fit_pca(df[axes].to_numpy(dtype=float))
    out = df[["class_label"]].copy()
    if "sample_key" in df.columns:
        out["sample_key"] = df["sample_key"].astype(str)
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1] if scores.shape[1] > 1 else 0.0
    out["pc1_explained_ratio"] = float(explained[0])
    out["pc2_explained_ratio"] = float(explained[1]) if len(explained) > 1 else 0.0
    return out


def _build_probe2_outputs(run_dir: Path, config_id: str) -> dict[str, pd.DataFrame]:
    src_root = COMPARATOR_ROOT / "runs" / config_id / PROBE2_ALIAS / "tables"
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_map = {
        "per_spectrum_bsv.csv": "per_spectrum_bsv.csv",
        "class_mean_bsv.csv": "class_mean_bsv.csv",
        "pairwise_delta_bsv.csv": "pairwise_delta_bsv.csv",
        "class_topk_neighborhood_composition.csv": "class_topk_neighborhood_composition.csv",
        "class_neighborhood_entropy.csv": "class_neighborhood_entropy.csv",
        "class_top1_dominance.csv": "class_top1_dominance.csv",
        "class_axis_entropy.csv": "class_axis_entropy.csv",
        "retrieval_hit_summary_by_class.csv": "retrieval_hit_summary_by_class.csv",
        "per_spectrum_retrieval_hits.csv": "per_spectrum_retrieval_hits.csv",
        "pca_coordinates.csv": "pca_coordinates_spectral.csv",
    }
    outputs: dict[str, pd.DataFrame] = {}
    for src_name, dst_name in raw_map.items():
        shutil.copy2(src_root / src_name, run_dir / dst_name)
        shutil.copy2(src_root / src_name, tables_dir / dst_name)
        outputs[dst_name] = pd.read_csv(src_root / src_name)

    per_spectrum_bsv_df = outputs["per_spectrum_bsv.csv"]
    class_mean_bsv_df = outputs["class_mean_bsv.csv"]
    class_neighborhood_df = outputs["class_topk_neighborhood_composition.csv"]
    axes = _axes_present(class_mean_bsv_df)
    cohort_mean = per_spectrum_bsv_df[axes].mean(axis=0)
    delta_class_df = class_mean_bsv_df[["sample_key", "dataset_id", "subset_id", "class_label"]].copy()
    for axis in axes:
        delta_class_df[axis] = class_mean_bsv_df[axis].to_numpy(dtype=float) - float(cohort_mean[axis])
    family_df = _build_family_fingerprint(class_neighborhood_df)
    pca_bsv_df = _pca_dataframe(per_spectrum_bsv_df, axes)

    for name, df in [
        ("class_mean_bsv_delta_vs_cohort.csv", delta_class_df),
        ("class_neighborhood_family_composition.csv", family_df),
        ("pca_coordinates_bsv.csv", pca_bsv_df),
    ]:
        df.to_csv(run_dir / name, index=False)
        df.to_csv(tables_dir / name, index=False)
        outputs[name] = df
    return outputs


def _pairwise_distance_vector(df: pd.DataFrame, value_cols: list[str], *, label_col: str = "class_label") -> tuple[list[str], np.ndarray]:
    labels = sorted(df[label_col].astype(str).tolist(), key=_mixture_sort_key)
    work = df.set_index(label_col).loc[labels]
    pairs = []
    dists = []
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            diff = work.loc[left, value_cols].to_numpy(dtype=float) - work.loc[right, value_cols].to_numpy(dtype=float)
            pairs.append(f"{left}__{right}")
            dists.append(float(np.linalg.norm(diff)))
    return pairs, np.asarray(dists, dtype=float)


def _build_combined_fingerprint_df(delta_df: pd.DataFrame, family_df: pd.DataFrame) -> pd.DataFrame:
    fam_wide = (
        family_df.pivot(index="class_label", columns="neighborhood_family", values="family_support_fraction")
        .reindex(columns=FAMILY_ORDER)
        .fillna(0.0)
        .reset_index()
    )
    axes = _axes_present(delta_df)
    merged = delta_df[["class_label"] + axes].merge(fam_wide, on="class_label", how="left")
    return merged


def _geometry_metrics(probe1_combined: pd.DataFrame, probe2_combined: pd.DataFrame) -> dict[str, float]:
    value_cols = [c for c in probe1_combined.columns if c != "class_label"]
    pairs1, d1 = _pairwise_distance_vector(probe1_combined, value_cols)
    pairs2, d2 = _pairwise_distance_vector(probe2_combined, value_cols)
    if pairs1 != pairs2:
        raise RuntimeError("Pairwise class ordering mismatch between probes")
    sp = float(spearmanr(d1, d2).statistic)
    pr = float(pearsonr(d1, d2).statistic)
    rk = float(kendalltau(d1, d2).statistic)
    pair_df = pd.DataFrame({"pair_label": pairs1, "distance_probe1": d1, "distance_probe2": d2})
    return {
        "pairwise_distance_spearman": sp,
        "pairwise_distance_pearson": pr,
        "rank_consistency": rk,
        "pair_df": pair_df,
    }


def _build_endpoint_alignment_summary(
    *,
    config_id: str,
    short_label: str,
    mixture_class_mean_df: pd.DataFrame,
    mixture_delta_df: pd.DataFrame,
    mixture_family_df: pd.DataFrame,
    endpoint_class_mean_df: pd.DataFrame,
    endpoint_delta_df: pd.DataFrame,
    endpoint_family_df: pd.DataFrame,
    low_endpoint_class: str,
    high_endpoint_class: str,
) -> pd.DataFrame:
    ordered = infer_mixture_order(mixture_class_mean_df["class_label"].astype(str).tolist())
    axes = _axes_present(mixture_class_mean_df)
    abs_mix = mixture_class_mean_df.set_index("class_label")
    delta_mix = mixture_delta_df.set_index("class_label")
    fam_mix = (
        mixture_family_df.pivot(index="class_label", columns="neighborhood_family", values="family_support_fraction")
        .reindex(columns=FAMILY_ORDER)
        .fillna(0.0)
    )
    abs_end = endpoint_class_mean_df.set_index("class_label")
    delta_end = endpoint_delta_df.set_index("class_label")
    fam_end = (
        endpoint_family_df.pivot(index="class_label", columns="family", values="family_fraction")
        .reindex(columns=FAMILY_ORDER)
        .fillna(0.0)
    )

    rows = []
    for label in ordered:
        abs_vec = abs_mix.loc[label, axes].to_numpy(dtype=float)
        delta_vec = delta_mix.loc[label, axes].to_numpy(dtype=float)
        fam_vec = fam_mix.loc[label, FAMILY_ORDER].to_numpy(dtype=float)

        abs_low = abs_end.loc[low_endpoint_class, axes].to_numpy(dtype=float)
        abs_high = abs_end.loc[high_endpoint_class, axes].to_numpy(dtype=float)
        delta_low = delta_end.loc[low_endpoint_class, axes].to_numpy(dtype=float)
        delta_high = delta_end.loc[high_endpoint_class, axes].to_numpy(dtype=float)
        fam_low = fam_end.loc[low_endpoint_class, FAMILY_ORDER].to_numpy(dtype=float)
        fam_high = fam_end.loc[high_endpoint_class, FAMILY_ORDER].to_numpy(dtype=float)

        dist_abs_low = float(np.linalg.norm(abs_vec - abs_low))
        dist_abs_high = float(np.linalg.norm(abs_vec - abs_high))
        dist_delta_low = float(np.linalg.norm(delta_vec - delta_low))
        dist_delta_high = float(np.linalg.norm(delta_vec - delta_high))
        dist_fam_low = float(np.linalg.norm(fam_vec - fam_low))
        dist_fam_high = float(np.linalg.norm(fam_vec - fam_high))

        def toward_high(d_low: float, d_high: float) -> float:
            return d_low / max(d_low + d_high, 1e-12)

        rows.append(
            {
                "config_id": config_id,
                "config_short_label": short_label,
                "class_label": label,
                "mixture_code_numeric": int("".join(ch for ch in label if ch.isdigit()) or 0),
                "low_endpoint_class": low_endpoint_class,
                "high_endpoint_class": high_endpoint_class,
                "absolute_distance_to_low_endpoint": dist_abs_low,
                "absolute_distance_to_high_endpoint": dist_abs_high,
                "absolute_toward_high_score": toward_high(dist_abs_low, dist_abs_high),
                "delta_distance_to_low_endpoint": dist_delta_low,
                "delta_distance_to_high_endpoint": dist_delta_high,
                "delta_toward_high_score": toward_high(dist_delta_low, dist_delta_high),
                "family_distance_to_low_endpoint": dist_fam_low,
                "family_distance_to_high_endpoint": dist_fam_high,
                "family_toward_high_score": toward_high(dist_fam_low, dist_fam_high),
            }
        )
    out = pd.DataFrame(rows)
    out["combined_toward_high_score"] = out[
        ["absolute_toward_high_score", "delta_toward_high_score", "family_toward_high_score"]
    ].mean(axis=1)
    return out.sort_values("mixture_code_numeric").reset_index(drop=True)


def _compute_progression_metrics(alignment_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = alignment_df.sort_values("mixture_code_numeric").copy()
    numeric = pd.Series(work["mixture_code_numeric"].to_numpy(dtype=float))
    steps = []
    for left, right in zip(work.itertuples(index=False), work.iloc[1:].itertuples(index=False), strict=False):
        steps.append(
            {
                "config_id": left.config_id,
                "config_short_label": left.config_short_label,
                "left_class": left.class_label,
                "right_class": right.class_label,
                "left_code_numeric": left.mixture_code_numeric,
                "absolute_adjacent_distance": abs(float(right.absolute_toward_high_score) - float(left.absolute_toward_high_score)),
                "delta_adjacent_distance": abs(float(right.delta_toward_high_score) - float(left.delta_toward_high_score)),
                "family_adjacent_distance": abs(float(right.family_toward_high_score) - float(left.family_toward_high_score)),
                "combined_adjacent_distance": abs(float(right.combined_toward_high_score) - float(left.combined_toward_high_score)),
            }
        )
    step_df = pd.DataFrame(steps)
    summary = {
        "config_id": str(work["config_id"].iloc[0]),
        "config_short_label": str(work["config_short_label"].iloc[0]),
        "progression_absolute_spearman": float(spearmanr(numeric, work["absolute_toward_high_score"]).statistic),
        "progression_delta_spearman": float(spearmanr(numeric, work["delta_toward_high_score"]).statistic),
        "progression_family_spearman": float(spearmanr(numeric, work["family_toward_high_score"]).statistic),
        "progression_combined_spearman": float(spearmanr(numeric, work["combined_toward_high_score"]).statistic),
        "max_combined_jump": float(step_df["combined_adjacent_distance"].max()) if not step_df.empty else 0.0,
        "mean_combined_adjacent_distance": float(step_df["combined_adjacent_distance"].mean()) if not step_df.empty else 0.0,
        "endpoint_alignment_score": float(work["combined_toward_high_score"].iloc[-1] - work["combined_toward_high_score"].iloc[0]),
        "collapse_region_count": int((step_df["combined_adjacent_distance"] < 0.05).sum()) if not step_df.empty else 0,
    }
    return pd.DataFrame([summary]), step_df


def _compute_noncollapse_metrics(
    *,
    config_id: str,
    short_label: str,
    class_mean_bsv_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    family_df: pd.DataFrame,
    top1_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
) -> pd.DataFrame:
    ordered = infer_mixture_order(class_mean_bsv_df["class_label"].astype(str).tolist())
    abs_work = class_mean_bsv_df.set_index("class_label").loc[ordered, _axes_present(class_mean_bsv_df)]
    delta_work = delta_df.set_index("class_label").loc[ordered, _axes_present(delta_df)]
    fam_work = (
        family_df.pivot(index="class_label", columns="neighborhood_family", values="family_support_fraction")
        .reindex(ordered)
        .reindex(FAMILY_ORDER, axis=1)
        .fillna(0.0)
    )
    unique_abs = np.unique(np.round(abs_work.to_numpy(dtype=float), 8), axis=0).shape[0]
    unique_delta = np.unique(np.round(delta_work.to_numpy(dtype=float), 8), axis=0).shape[0]
    adj_delta = []
    for left, right in zip(ordered[:-1], ordered[1:], strict=False):
        diff = delta_work.loc[left].to_numpy(dtype=float) - delta_work.loc[right].to_numpy(dtype=float)
        adj_delta.append(float(np.linalg.norm(diff)))
    intermediate = [label for label in ordered if label not in {ordered[0], ordered[-1]}]
    intermediate_distinct = 0
    for label in intermediate:
        vec = delta_work.loc[label].to_numpy(dtype=float)
        others = [o for o in ordered if o != label]
        min_dist = min(float(np.linalg.norm(vec - delta_work.loc[o].to_numpy(dtype=float))) for o in others)
        if min_dist > 1e-3:
            intermediate_distinct += 1
    endpoint_sep = float(
        np.linalg.norm(delta_work.loc[ordered[0]].to_numpy(dtype=float) - delta_work.loc[ordered[-1]].to_numpy(dtype=float))
    )
    fam_pairwise = [
        np.linalg.norm(fam_work.loc[a].to_numpy(dtype=float) - fam_work.loc[b].to_numpy(dtype=float))
        for i, a in enumerate(ordered)
        for b in ordered[i + 1 :]
    ]
    return pd.DataFrame(
        [
            {
                "config_id": config_id,
                "config_short_label": short_label,
                "noncollapse_ratio": unique_delta / max(len(ordered), 1),
                "unique_absolute_profile_ratio": unique_abs / max(len(ordered), 1),
                "adjacent_nonzero_ratio": float(np.mean([x > 1e-6 for x in adj_delta])) if adj_delta else 0.0,
                "min_adjacent_delta_distance": float(min(adj_delta)) if adj_delta else 0.0,
                "mean_adjacent_delta_distance": float(np.mean(adj_delta)) if adj_delta else 0.0,
                "intermediate_distinct_count": int(intermediate_distinct),
                "endpoint_delta_separation": endpoint_sep,
                "mean_top1_dominance": float(top1_df["top1_fraction"].mean()),
                "mean_neighborhood_entropy": float(entropy_df["neighborhood_entropy"].mean()),
                "mean_family_distance": float(np.mean(fam_pairwise)) if fam_pairwise else 0.0,
            }
        ]
    )


def _build_probe1_probe2_progression_table(
    config_id: str,
    short_label: str,
    probe1_alignment: pd.DataFrame,
    probe2_alignment: pd.DataFrame,
) -> pd.DataFrame:
    p1 = probe1_alignment.copy()
    p2 = probe2_alignment.copy()
    p1["probe"] = "probe1"
    p2["probe"] = "probe2"
    p1["config_id"] = config_id
    p1["config_short_label"] = short_label
    p2["config_id"] = config_id
    p2["config_short_label"] = short_label
    return pd.concat([p1, p2], ignore_index=True)


def _progression_summary(alignment_df: pd.DataFrame) -> dict[str, float]:
    work = alignment_df.sort_values("mixture_code_numeric").copy()
    numeric = pd.Series(work["mixture_code_numeric"].to_numpy(dtype=float))
    combined = pd.Series(work["combined_toward_high_score"].to_numpy(dtype=float))
    step_diffs = np.abs(np.diff(combined.to_numpy(dtype=float)))
    return {
        "progression_combined_spearman": float(spearmanr(numeric, combined).statistic),
        "max_combined_jump": float(step_diffs.max()) if len(step_diffs) else 0.0,
        "mean_combined_adjacent_distance": float(step_diffs.mean()) if len(step_diffs) else 0.0,
        "endpoint_alignment_score": float(combined.iloc[-1] - combined.iloc[0]),
    }


def _class_level_drift(
    *,
    config_id: str,
    probe1_abs: pd.DataFrame,
    probe2_abs: pd.DataFrame,
    probe1_delta: pd.DataFrame,
    probe2_delta: pd.DataFrame,
    probe1_family: pd.DataFrame,
    probe2_family: pd.DataFrame,
    probe1_top1: pd.DataFrame,
    probe2_top1: pd.DataFrame,
    probe1_entropy: pd.DataFrame,
    probe2_entropy: pd.DataFrame,
    probe1_neighborhood: pd.DataFrame,
    probe2_neighborhood: pd.DataFrame,
) -> pd.DataFrame:
    labels = sorted(probe1_abs["class_label"].astype(str).tolist(), key=_mixture_sort_key)
    abs_axes = _axes_present(probe1_abs)
    delta_axes = _axes_present(probe1_delta)
    fam1 = (
        probe1_family.pivot(index="class_label", columns="neighborhood_family", values="family_support_fraction")
        .reindex(labels)
        .reindex(FAMILY_ORDER, axis=1)
        .fillna(0.0)
    )
    fam2 = (
        probe2_family.pivot(index="class_label", columns="neighborhood_family", values="family_support_fraction")
        .reindex(labels)
        .reindex(FAMILY_ORDER, axis=1)
        .fillna(0.0)
    )
    out = []
    p1_abs = probe1_abs.set_index("class_label").loc[labels]
    p2_abs = probe2_abs.set_index("class_label").loc[labels]
    p1_delta = probe1_delta.set_index("class_label").loc[labels]
    p2_delta = probe2_delta.set_index("class_label").loc[labels]
    top1_1 = probe1_top1.set_index("class_label")
    top1_2 = probe2_top1.set_index("class_label")
    ent1 = probe1_entropy.set_index("class_label")
    ent2 = probe2_entropy.set_index("class_label")
    n1 = probe1_neighborhood.copy()
    n2 = probe2_neighborhood.copy()
    for label in labels:
        abs_drift = float(np.linalg.norm(p2_abs.loc[label, abs_axes].to_numpy(dtype=float) - p1_abs.loc[label, abs_axes].to_numpy(dtype=float)))
        delta_drift = float(np.linalg.norm(p2_delta.loc[label, delta_axes].to_numpy(dtype=float) - p1_delta.loc[label, delta_axes].to_numpy(dtype=float)))
        fam_drift = float(np.linalg.norm(fam2.loc[label].to_numpy(dtype=float) - fam1.loc[label].to_numpy(dtype=float)))
        p1_top_axis = str(p1_delta.loc[label, delta_axes].astype(float).sort_values(ascending=False).index[0])
        p2_top_axis = str(p2_delta.loc[label, delta_axes].astype(float).sort_values(ascending=False).index[0])
        p1_compounds = set(n1[n1["class_label"].astype(str) == label].sort_values("support_fraction", ascending=False).head(5)["compound_label"].astype(str))
        p2_compounds = set(n2[n2["class_label"].astype(str) == label].sort_values("support_fraction", ascending=False).head(5)["compound_label"].astype(str))
        overlap = len(p1_compounds & p2_compounds) / max(len(p1_compounds | p2_compounds), 1)
        out.append(
            {
                "config_id": config_id,
                "class_label": label,
                "absolute_bsv_drift": abs_drift,
                "delta_bsv_drift": delta_drift,
                "family_fingerprint_drift": fam_drift,
                "top1_dominance_probe1": float(top1_1.loc[label, "top1_fraction"]),
                "top1_dominance_probe2": float(top1_2.loc[label, "top1_fraction"]),
                "entropy_probe1": float(ent1.loc[label, "neighborhood_entropy"]),
                "entropy_probe2": float(ent2.loc[label, "neighborhood_entropy"]),
                "probe1_top_axis": p1_top_axis,
                "probe2_top_axis": p2_top_axis,
                "dominant_axis_shift": bool(p1_top_axis != p2_top_axis),
                "top_compound_overlap_fraction": float(overlap),
            }
        )
    return pd.DataFrame(out)


def _plot_probe_overlay(
    probe1_df: pd.DataFrame,
    probe2_df: pd.DataFrame,
    output_path: Path,
    title: str,
    *,
    standardize_probe_local: bool = False,
) -> None:
    p1 = probe1_df.copy()
    p2 = probe2_df.copy()
    for df, probe in [(p1, "probe1"), (p2, "probe2")]:
        df["probe"] = probe
        if standardize_probe_local:
            for col in ["pc1", "pc2"]:
                vals = df[col].to_numpy(dtype=float)
                std = max(float(vals.std()), 1e-9)
                df[col] = (vals - float(vals.mean())) / std
    merged = pd.concat([p1, p2], ignore_index=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    for probe in ["probe1", "probe2"]:
        sub_probe = merged[merged["probe"] == probe]
        for label in sorted(sub_probe["class_label"].astype(str).unique().tolist(), key=_mixture_sort_key):
            sub = sub_probe[sub_probe["class_label"].astype(str) == label]
            ax.scatter(
                sub["pc1"],
                sub["pc2"],
                s=38,
                alpha=0.78,
                label=f"{probe}:{label}",
                color=CLASS_COLORS.get(label, "#4c78a8"),
                marker=PROBE_MARKERS[probe],
                edgecolors="white",
                linewidths=0.4,
            )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.22, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_progression_probe_compare(probe1_alignment: pd.DataFrame, probe2_alignment: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for df, probe in [(probe1_alignment, "probe1"), (probe2_alignment, "probe2")]:
        work = df.sort_values("mixture_code_numeric")
        ax.plot(
            work["class_label"],
            work["combined_toward_high_score"],
            marker="o",
            linestyle=PROBE_LINESTYLES[probe],
            linewidth=2.0,
            label=f"{probe} combined",
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Toward high-endpoint score")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_adjacent_probe_compare(probe1_steps: pd.DataFrame, probe2_steps: pd.DataFrame, output_path: Path, title: str) -> None:
    p1 = probe1_steps.sort_values("left_code_numeric").copy()
    p2 = probe2_steps.sort_values("left_code_numeric").copy()
    labels = [f"{a}->{b}" for a, b in zip(p1["left_class"], p1["right_class"], strict=False)]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.bar(x - width / 2, p1["combined_adjacent_distance"], width=width, label="probe1", color="#577590")
    ax.bar(x + width / 2, p2["combined_adjacent_distance"], width=width, label="probe2", color="#f3722c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Combined adjacent distance")
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_family_probe_compare(probe1_family: pd.DataFrame, probe2_family: pd.DataFrame, output_path: Path, title: str) -> None:
    labels = sorted(probe1_family["class_label"].astype(str).unique().tolist(), key=_mixture_sort_key)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    for ax, df, probe in [(axes[0], probe1_family, "Probe 1"), (axes[1], probe2_family, "Probe 2")]:
        heat = (
            df.pivot(index="class_label", columns="neighborhood_family", values="family_support_fraction")
            .reindex(labels)
            .reindex(FAMILY_ORDER, axis=1)
            .fillna(0.0)
        )
        im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0.0, vmax=max(float(heat.to_numpy(dtype=float).max()), 1e-9))
        ax.set_xticks(np.arange(len(FAMILY_ORDER)))
        ax.set_xticklabels(FAMILY_ORDER, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(probe)
    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_class_drift(class_drift_df: pd.DataFrame, output_path: Path, title: str) -> None:
    work = class_drift_df.sort_values("class_label", key=lambda s: s.map(_mixture_sort_key))
    x = np.arange(len(work))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.bar(x - width, work["absolute_bsv_drift"], width=width, label="absolute")
    ax.bar(x, work["delta_bsv_drift"], width=width, label="delta")
    ax.bar(x + width, work["family_fingerprint_drift"], width=width, label="family")
    ax.set_xticks(x)
    ax.set_xticklabels(work["class_label"].tolist())
    ax.set_title(title)
    ax.set_ylabel("Probe drift")
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_pairwise_corr(pair_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.scatter(pair_df["distance_probe1"], pair_df["distance_probe2"], s=44, alpha=0.82, color="#355070")
    for row in pair_df.itertuples(index=False):
        ax.annotate(str(row.pair_label), (row.distance_probe1, row.distance_probe2), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Probe 1 pairwise distance")
    ax.set_ylabel("Probe 2 pairwise distance")
    ax.set_title(title)
    ax.grid(True, alpha=0.24, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_tradeoff(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))
    panels = [
        ("pairwise_distance_spearman", "Geometry Spearman"),
        ("probe2_noncollapse_ratio", "Probe 2 noncollapse"),
        ("mean_family_drift", "Mean family drift"),
        ("mean_top1_dominance_shift", "Mean top1 shift"),
    ]
    for ax, (col, title) in zip(axes.ravel(), panels, strict=False):
        x = np.arange(len(summary_df))
        ax.bar(
            x,
            summary_df[col].to_numpy(dtype=float),
            color=[CONFIG_COLORS.get(x_, "#4c78a8") for x_ in summary_df["config_short_label"]],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["config_short_label"].tolist())
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.24, linewidth=0.5)
    fig.suptitle("cfg05 vs cfg08 probe-consistency tradeoff")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _build_report(report_path: Path, summary_df: pd.DataFrame) -> None:
    cfg05 = summary_df[summary_df["config_short_label"] == "cfg05"].iloc[0]
    cfg08 = summary_df[summary_df["config_short_label"] == "cfg08"].iloc[0]
    lines = [
        "# GAIRAv3 Pilot 1c Probe Consistency Report",
        "",
        "## 1. What changed from the originally intended 1c",
        "- A true `small2023_celltype_probe2` subset is not present locally.",
        "- This 1c therefore evaluates mixture probe consistency only, using the available `small2023_mixture_probe1` and `small2023_mixture_probe2` subsets.",
        "",
        "## 2. Direct answers per config",
        f"- cfg05: Probe1 progression `{cfg05['probe1_progression_spearman']:.4f}`, Probe2 progression `{cfg05['probe2_progression_spearman']:.4f}`, pairwise distance Spearman `{cfg05['pairwise_distance_spearman']:.4f}`, mean family drift `{cfg05['mean_family_drift']:.4f}`",
        f"- cfg08: Probe1 progression `{cfg08['probe1_progression_spearman']:.4f}`, Probe2 progression `{cfg08['probe2_progression_spearman']:.4f}`, pairwise distance Spearman `{cfg08['pairwise_distance_spearman']:.4f}`, mean family drift `{cfg08['mean_family_drift']:.4f}`",
        "",
        "## 3. cfg05 vs cfg08 comparison",
        "- Higher progression preservation and lower class drift are good.",
        "- Lower family drift and lower dominance shift indicate more stable chemistry support.",
        f"- cfg05 intermediate-class preservation delta: Probe1 `{cfg05['probe1_intermediate_distinct_count']:.0f}` vs Probe2 `{cfg05['probe2_intermediate_distinct_count']:.0f}`",
        f"- cfg08 intermediate-class preservation delta: Probe1 `{cfg08['probe1_intermediate_distinct_count']:.0f}` vs Probe2 `{cfg08['probe2_intermediate_distinct_count']:.0f}`",
        "",
        "## 4. Final decision",
    ]
    if (
        float(cfg05["pairwise_distance_spearman"]) >= float(cfg08["pairwise_distance_spearman"])
        and float(cfg05["mean_family_drift"]) <= float(cfg08["mean_family_drift"])
        and float(cfg05["progression_spearman_delta"]) <= float(cfg08["progression_spearman_delta"])
    ):
        lines.append("- Recommendation: `cfg05` is the more probe-stable choice.")
    elif (
        float(cfg08["pairwise_distance_spearman"]) >= float(cfg05["pairwise_distance_spearman"])
        and float(cfg08["mean_family_drift"]) <= float(cfg05["mean_family_drift"])
        and float(cfg08["progression_spearman_delta"]) <= float(cfg05["progression_spearman_delta"])
    ):
        lines.append("- Recommendation: `cfg08` is the more probe-stable choice.")
    else:
        lines.append("- Recommendation: neither config is sufficiently probe-stable without reservation.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _require_inputs()
    storage_cfg = load_autoresearch_storage_config(DEFAULT_STORAGE_CONFIG_PATH)
    sprint_paths = initialize_autoresearch_sprint(
        DEFAULT_STORAGE_CONFIG_PATH,
        sprint_id=f"{storage_cfg.sprint_id}/{SPRINT_SUBDIR}",
    )
    sprint_paths.tables_dir.mkdir(parents=True, exist_ok=True)
    sprint_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    sprint_paths.report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    class_drift_rows = []
    progression_rows = []
    report_figures: list[Path] = []

    for spec in CONFIG_SPECS:
        config_id = str(spec["config_id"])
        short_label = str(spec["short_label"])
        run_root = sprint_paths.runs_dir / config_id
        probe1_run = run_root / "probe1"
        probe2_run = run_root / "probe2"
        probe1_run.mkdir(parents=True, exist_ok=True)
        probe2_run.mkdir(parents=True, exist_ok=True)

        _copy_probe1_outputs(probe1_run, config_id)
        probe2_outputs = _build_probe2_outputs(probe2_run, config_id)

        probe1_abs = pd.read_csv(probe1_run / "class_mean_bsv.csv")
        probe1_delta = pd.read_csv(probe1_run / "class_mean_bsv_delta_vs_cohort.csv")
        probe1_family = pd.read_csv(probe1_run / "class_neighborhood_family_composition.csv")
        probe1_top1 = pd.read_csv(probe1_run / "class_top1_dominance.csv")
        probe1_entropy = pd.read_csv(probe1_run / "class_neighborhood_entropy.csv")
        probe1_neighborhood = pd.read_csv(probe1_run / "class_topk_neighborhood_composition.csv")
        probe1_per_spectrum_bsv = pd.read_csv(probe1_run / "per_spectrum_bsv.csv")
        probe1_bsv_pca = pd.read_csv(probe1_run / "pca_coordinates_bsv.csv")
        probe1_spectral_pca = pd.read_csv(probe1_run / "pca_coordinates_spectral.csv")
        probe1_alignment = pd.read_csv(probe1_run / "endpoint_alignment_summary.csv")
        probe1_steps = pd.read_csv(probe1_run / "adjacent_progression_steps.csv")
        probe1_noncollapse = pd.read_csv(probe1_run / "noncollapse_metrics.csv").iloc[0]
        probe1_prog = _progression_summary(probe1_alignment)

        probe2_abs = probe2_outputs["class_mean_bsv.csv"]
        probe2_delta = probe2_outputs["class_mean_bsv_delta_vs_cohort.csv"]
        probe2_family = probe2_outputs["class_neighborhood_family_composition.csv"]
        probe2_top1 = probe2_outputs["class_top1_dominance.csv"]
        probe2_entropy = probe2_outputs["class_neighborhood_entropy.csv"]
        probe2_neighborhood = probe2_outputs["class_topk_neighborhood_composition.csv"]
        probe2_per_spectrum_bsv = probe2_outputs["per_spectrum_bsv.csv"]
        probe2_bsv_pca = probe2_outputs["pca_coordinates_bsv.csv"]
        probe2_spectral_pca = probe2_outputs["pca_coordinates_spectral.csv"]

        low_endpoint = str(probe1_alignment["low_endpoint_class"].iloc[0])
        high_endpoint = str(probe1_alignment["high_endpoint_class"].iloc[0])
        endpoint_abs = pd.read_csv(PILOT1A_V5_ROOT / "runs" / config_id / "class_mean_bsv.csv")
        endpoint_delta = pd.read_csv(PILOT1A_V5_ROOT / "runs" / config_id / "delta_class_mean_bsv.csv")
        endpoint_family = pd.read_csv(PILOT1A_V5_ROOT / "runs" / config_id / "class_family_fingerprint.csv")

        probe2_alignment = _build_endpoint_alignment_summary(
            config_id=config_id,
            short_label=short_label,
            mixture_class_mean_df=probe2_abs,
            mixture_delta_df=probe2_delta,
            mixture_family_df=probe2_family,
            endpoint_class_mean_df=endpoint_abs,
            endpoint_delta_df=endpoint_delta,
            endpoint_family_df=endpoint_family,
            low_endpoint_class=low_endpoint,
            high_endpoint_class=high_endpoint,
        )
        probe2_progression_df, probe2_steps = _compute_progression_metrics(probe2_alignment)
        probe2_noncollapse = _compute_noncollapse_metrics(
            config_id=config_id,
            short_label=short_label,
            class_mean_bsv_df=probe2_abs,
            delta_df=probe2_delta,
            family_df=probe2_family,
            top1_df=probe2_top1,
            entropy_df=probe2_entropy,
        ).iloc[0]
        probe2_alignment.to_csv(probe2_run / "endpoint_alignment_summary.csv", index=False)
        probe2_alignment.to_csv(probe2_run / "tables" / "endpoint_alignment_summary.csv", index=False)
        probe2_progression_df.to_csv(probe2_run / "progression_metrics.csv", index=False)
        probe2_progression_df.to_csv(probe2_run / "tables" / "progression_metrics.csv", index=False)
        probe2_steps.to_csv(probe2_run / "adjacent_progression_steps.csv", index=False)
        probe2_steps.to_csv(probe2_run / "tables" / "adjacent_progression_steps.csv", index=False)
        pd.DataFrame([probe2_noncollapse]).to_csv(probe2_run / "noncollapse_metrics.csv", index=False)
        pd.DataFrame([probe2_noncollapse]).to_csv(probe2_run / "tables" / "noncollapse_metrics.csv", index=False)

        probe1_combined = _build_combined_fingerprint_df(probe1_delta, probe1_family)
        probe2_combined = _build_combined_fingerprint_df(probe2_delta, probe2_family)
        geom = _geometry_metrics(probe1_combined, probe2_combined)
        geom["pair_df"].to_csv(probe2_run / "tables" / "pairwise_distance_probe1_vs_probe2.csv", index=False)

        class_drift_df = _class_level_drift(
            config_id=config_id,
            probe1_abs=probe1_abs,
            probe2_abs=probe2_abs,
            probe1_delta=probe1_delta,
            probe2_delta=probe2_delta,
            probe1_family=probe1_family,
            probe2_family=probe2_family,
            probe1_top1=probe1_top1,
            probe2_top1=probe2_top1,
            probe1_entropy=probe1_entropy,
            probe2_entropy=probe2_entropy,
            probe1_neighborhood=probe1_neighborhood,
            probe2_neighborhood=probe2_neighborhood,
        )
        class_drift_rows.append(class_drift_df)

        progression_compare_df = _build_probe1_probe2_progression_table(
            config_id,
            short_label,
            probe1_alignment,
            probe2_alignment,
        )
        progression_rows.append(progression_compare_df)

        summary_rows.append(
            {
                "config_id": config_id,
                "config_short_label": short_label,
                "probe1_progression_spearman": float(probe1_prog["progression_combined_spearman"]),
                "probe2_progression_spearman": float(probe2_progression_df["progression_combined_spearman"].iloc[0]),
                "progression_spearman_delta": float(
                    abs(float(probe2_progression_df["progression_combined_spearman"].iloc[0]) - float(probe1_prog["progression_combined_spearman"]))
                ),
                "probe1_noncollapse_ratio": float(probe1_noncollapse["noncollapse_ratio"]),
                "probe2_noncollapse_ratio": float(probe2_noncollapse["noncollapse_ratio"]),
                "noncollapse_delta": float(abs(float(probe2_noncollapse["noncollapse_ratio"]) - float(probe1_noncollapse["noncollapse_ratio"]))),
                "pairwise_distance_spearman": float(geom["pairwise_distance_spearman"]),
                "pairwise_distance_pearson": float(geom["pairwise_distance_pearson"]),
                "rank_consistency": float(geom["rank_consistency"]),
                "mean_class_drift_absolute": float(class_drift_df["absolute_bsv_drift"].mean()),
                "mean_class_drift_delta": float(class_drift_df["delta_bsv_drift"].mean()),
                "mean_family_drift": float(class_drift_df["family_fingerprint_drift"].mean()),
                "mean_top1_dominance_shift": float(np.mean(np.abs(class_drift_df["top1_dominance_probe2"] - class_drift_df["top1_dominance_probe1"]))),
                "mean_entropy_shift": float(np.mean(np.abs(class_drift_df["entropy_probe2"] - class_drift_df["entropy_probe1"]))),
                "probe1_intermediate_distinct_count": float(probe1_noncollapse["intermediate_distinct_count"]),
                "probe2_intermediate_distinct_count": float(probe2_noncollapse["intermediate_distinct_count"]),
                "probe1_endpoint_alignment_score": float(probe1_prog["endpoint_alignment_score"]),
                "probe2_endpoint_alignment_score": float(probe2_progression_df["endpoint_alignment_score"].iloc[0]),
                "neighborhood_overlap_score": float(class_drift_df["top_compound_overlap_fraction"].mean()),
            }
        )

        _plot_probe_overlay(
            probe1_spectral_pca,
            probe2_spectral_pca,
            sprint_paths.figures_dir / f"pca_spectral_probe1_vs_probe2_overlay_{short_label}.png",
            f"Probe-local spectral PCA overlay: {short_label}",
            standardize_probe_local=True,
        )
        _plot_probe_overlay(
            probe1_bsv_pca,
            probe2_bsv_pca,
            sprint_paths.figures_dir / f"pca_bsv_probe1_vs_probe2_overlay_{short_label}.png",
            f"BSV PCA overlay: {short_label}",
            standardize_probe_local=False,
        )
        _plot_progression_probe_compare(
            probe1_alignment,
            probe2_alignment,
            sprint_paths.figures_dir / f"progression_probe1_vs_probe2_{short_label}.png",
            f"Progression preservation: {short_label}",
        )
        _plot_adjacent_probe_compare(
            probe1_steps,
            probe2_steps,
            sprint_paths.figures_dir / f"adjacent_distance_probe1_vs_probe2_{short_label}.png",
            f"Adjacent distance preservation: {short_label}",
        )
        _plot_family_probe_compare(
            probe1_family,
            probe2_family,
            sprint_paths.figures_dir / f"family_fingerprint_probe1_vs_probe2_{short_label}.png",
            f"Family fingerprint comparison: {short_label}",
        )
        _plot_class_drift(
            class_drift_df,
            sprint_paths.figures_dir / f"class_drift_barplot_{short_label}.png",
            f"Class-level probe drift: {short_label}",
        )
        _plot_pairwise_corr(
            geom["pair_df"],
            sprint_paths.figures_dir / "pairwise_distance_correlation.png" if short_label == "cfg05" else sprint_paths.figures_dir / f"pairwise_distance_correlation_{short_label}.png",
            f"Pairwise distance preservation: {short_label}",
        )
        report_figures.extend(
            [
                sprint_paths.figures_dir / f"pca_spectral_probe1_vs_probe2_overlay_{short_label}.png",
                sprint_paths.figures_dir / f"pca_bsv_probe1_vs_probe2_overlay_{short_label}.png",
                sprint_paths.figures_dir / f"progression_probe1_vs_probe2_{short_label}.png",
                sprint_paths.figures_dir / f"adjacent_distance_probe1_vs_probe2_{short_label}.png",
                sprint_paths.figures_dir / f"family_fingerprint_probe1_vs_probe2_{short_label}.png",
                sprint_paths.figures_dir / f"class_drift_barplot_{short_label}.png",
            ]
        )

    summary_df = pd.DataFrame(summary_rows)
    class_drift_all_df = pd.concat(class_drift_rows, ignore_index=True)
    progression_all_df = pd.concat(progression_rows, ignore_index=True)
    summary_df.to_csv(sprint_paths.tables_dir / "probe_consistency_metrics.csv", index=False)
    class_drift_all_df.to_csv(sprint_paths.tables_dir / "class_level_probe_drift.csv", index=False)
    progression_all_df.to_csv(sprint_paths.tables_dir / "progression_probe1_vs_probe2.csv", index=False)
    summary_df.to_csv(sprint_paths.tables_dir / "cfg05_vs_cfg08_probe_consistency_comparison.csv", index=False)

    _plot_tradeoff(summary_df, sprint_paths.figures_dir / "cfg05_vs_cfg08_probe_consistency_tradeoff.png")
    report_figures.append(sprint_paths.figures_dir / "cfg05_vs_cfg08_probe_consistency_tradeoff.png")

    report_md = sprint_paths.report_dir / "GAIRAv3_Pilot1c_probe_consistency_report.md"
    _build_report(report_md, summary_df)
    report_pdf = sprint_paths.report_dir / "GAIRAv3_Pilot1c_probe_consistency_report.pdf"
    build_pdf_report(report_md, report_figures, report_pdf)


if __name__ == "__main__":
    main()
