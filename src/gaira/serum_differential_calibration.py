from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


POSITIVE_THEMES = [
    "lipid_membrane_associated",
    "protein_peptide_associated",
    "nucleic_acid_purine_associated",
    "carbohydrate_glycan_associated",
    "oxidative_metabolic_stress_associated",
]
CAUTION_THEMES = [
    "matrix_dominance_caution",
    "probe_substrate_caution",
    "modality_mismatch_caution",
    "weak_label_or_cohort_caution",
    "low_specificity_caution",
]


@dataclass
class SerumCalibrationBundle:
    calibrated_theme_df: pd.DataFrame
    sample_summary_df: pd.DataFrame
    shared_background_df: pd.DataFrame
    shifted_theme_df: pd.DataFrame
    differential_evidence_df: pd.DataFrame
    before_after_metrics_df: pd.DataFrame
    failure_mode_note: str
    theme_space_df: pd.DataFrame


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _effect_size(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0))
    if pooled <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def build_failure_mode_note(
    query_df: pd.DataFrame,
    theme_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> str:
    positive_df = theme_df[theme_df["category"] == "positive"].copy()
    caution_df = theme_df[theme_df["category"] == "caution"].copy()
    grouped = (
        positive_df.groupby("theme_name", as_index=False)
        .agg(mean_score=("score", "mean"), mean_confidence=("confidence", "mean"))
        .sort_values("mean_score", ascending=False)
    )
    top_positive = grouped.head(3)["theme_name"].tolist()
    caution_summary = (
        caution_df.groupby("theme_name", as_index=False)["score"]
        .mean()
        .sort_values("score", ascending=False)
    )
    top_cautions = caution_summary.head(2)["theme_name"].tolist()
    representative_tier1 = representative_df["top_tier1"].dropna().astype(str).tolist()
    representative_tier2 = representative_df["top_tier2"].dropna().astype(str).tolist()
    metric_lookup = dict(zip(metrics_df["metric_name"], metrics_df["metric_value"]))
    return "\n".join(
        [
            "Targeted HCC serum calibration note",
            f"- Positive themes are compressed: top mean positive themes are {', '.join(top_positive)}.",
            f"- Confidence is suppressed: mean_positive_confidence={metric_lookup.get('mean_positive_confidence', 0.0):.3f}.",
            f"- Shared serum anchors dominate representative cases: tier1={', '.join(sorted(set(representative_tier1)))}.",
            f"- Tier2 support is broad rather than differential: tier2={', '.join(sorted(set(representative_tier2)))}.",
            f"- Caution remains heavy: top caution themes are {', '.join(top_cautions)}.",
            "- The main failure mode is absolute-score dominance by shared serum background rather than differential contrast between HCC and CTR.",
        ]
    )


def build_differential_evidence_summary(results: list[dict], metadata_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metadata_map = metadata_df.set_index("biosample_id")["class_label"].to_dict()
    for result in results:
        class_label = metadata_map.get(result["query_id"], "")
        for evidence_name, hits in [
            ("tier1", result.get("tier1_grounding_hits", [])[:3]),
            ("tier2", result.get("tier2_support_hits", [])[:3]),
            ("knowledge", result.get("knowledge_support_hits", [])[:3]),
            ("semantic", result.get("semantic_region_support_hits", [])[:3]),
            ("context", result.get("domain_context_hits", [])[:3]),
        ]:
            for hit in hits:
                label = hit.get("source_label") or hit.get("document_id") or ""
                rows.append(
                    {
                        "class_label": class_label,
                        "evidence_family": evidence_name,
                        "label": str(label),
                    }
                )
    freq_df = (
        pd.DataFrame(rows)
        .groupby(["class_label", "evidence_family", "label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    if freq_df.empty:
        return freq_df
    pivot = freq_df.pivot_table(index=["evidence_family", "label"], columns="class_label", values="count", fill_value=0.0)
    if len(pivot.columns) == 2:
        cols = list(pivot.columns)
        pivot["count_diff"] = pivot[cols[1]] - pivot[cols[0]]
        pivot["abs_count_diff"] = pivot["count_diff"].abs()
    else:
        pivot["count_diff"] = 0.0
        pivot["abs_count_diff"] = 0.0
    return pivot.reset_index().sort_values(["abs_count_diff", "evidence_family", "label"], ascending=[False, True, True])


def calibrate_serum_holdout(
    query_df: pd.DataFrame,
    theme_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    results: list[dict],
) -> SerumCalibrationBundle:
    failure_mode_note = build_failure_mode_note(query_df, theme_df, representative_df, metrics_df)

    theme_df = theme_df.copy()
    caution_means = (
        theme_df[theme_df["theme_name"].isin(CAUTION_THEMES)]
        .groupby("query_id", as_index=False)["score"]
        .mean()
        .rename(columns={"score": "base_caution_load"})
    )
    positive_df = theme_df[theme_df["theme_name"].isin(POSITIVE_THEMES)].copy()
    group_means = positive_df.groupby(["class_label", "theme_name"], as_index=False)["score"].mean()

    mean_lookup = {(row["class_label"], row["theme_name"]): float(row["score"]) for row in group_means.to_dict(orient="records")}
    classes = sorted(positive_df["class_label"].dropna().astype(str).unique().tolist())

    calibrated_rows = []
    for row in theme_df.to_dict(orient="records"):
        if row["theme_name"] not in POSITIVE_THEMES:
            calibrated_rows.append(
                {
                    **row,
                    "shared_background_burden": 0.0,
                    "differential_signal_index": 0.0,
                    "serum_specificity_index": float(row.get("specificity_index", 0.0)),
                    "comparison_score": float(row["score"]),
                    "comparison_confidence": float(row["confidence"]),
                    "shared_background_score": 0.0,
                    "group_shift_score": 0.0,
                    "sample_excess_score": 0.0,
                }
            )
            continue

        theme_name = str(row["theme_name"])
        class_label = str(row["class_label"])
        score = float(row["score"])
        base_conf = float(row["confidence"])
        specificity = float(row.get("specificity_index", 0.0))
        other_classes = [value for value in classes if value != class_label]
        own_mean = float(mean_lookup.get((class_label, theme_name), score))
        other_mean = float(np.mean([mean_lookup.get((value, theme_name), own_mean) for value in other_classes])) if other_classes else own_mean
        shared_background = float(min(own_mean, other_mean))
        group_shift = float(own_mean - other_mean)
        sample_excess = float(max(0.0, score - shared_background))
        shared_background_burden = _clip01(shared_background / max(score, 1e-6))
        differential_signal_index = _clip01(4.5 * sample_excess + 7.0 * max(0.0, group_shift))
        serum_specificity_index = _clip01(
            0.55 * specificity
            + 0.45 * differential_signal_index
            - 0.30 * shared_background_burden
        )

        comparison_score = _clip01(
            score * (1.0 - 0.60 * shared_background_burden)
            + 0.75 * max(0.0, group_shift)
            + 0.40 * sample_excess
        )

        evidence_diversity = min(
            1.0,
            (
                int(row.get("n_tier1_hits", 0) > 0)
                + int(row.get("n_tier2_hits", 0) > 0)
                + int(row.get("n_knowledge_hits", 0) > 0)
                + int(row.get("n_semantic_hits", 0) > 0)
                + int(row.get("context_contrib", 0.0) > 0.02)
            )
            / 4.0,
        )
        calibrated_rows.append(
            {
                **row,
                "shared_background_burden": round(shared_background_burden, 4),
                "differential_signal_index": round(differential_signal_index, 4),
                "serum_specificity_index": round(serum_specificity_index, 4),
                "comparison_score": round(comparison_score, 4),
                "comparison_confidence_pre_caution": round(
                    _clip01(
                        0.35 * base_conf
                        + 0.25 * serum_specificity_index
                        + 0.20 * differential_signal_index
                        + 0.20 * evidence_diversity
                    ),
                    4,
                ),
                "shared_background_score": round(shared_background, 4),
                "group_shift_score": round(max(0.0, group_shift), 4),
                "sample_excess_score": round(sample_excess, 4),
            }
        )

    calibrated_theme_df = pd.DataFrame(calibrated_rows)
    calibrated_theme_df = calibrated_theme_df.merge(caution_means, on="query_id", how="left")
    calibrated_theme_df["base_caution_load"] = calibrated_theme_df["base_caution_load"].fillna(0.0)
    positive_mask = calibrated_theme_df["theme_name"].isin(POSITIVE_THEMES)
    calibrated_theme_df.loc[positive_mask, "comparison_confidence"] = calibrated_theme_df.loc[positive_mask].apply(
        lambda row: round(
            _clip01(
                float(row["comparison_confidence_pre_caution"])
                * (1.0 - 0.25 * float(row["base_caution_load"]))
                * (1.0 - 0.15 * float(row["shared_background_burden"]))
            ),
            4,
        ),
        axis=1,
    )
    calibrated_theme_df.loc[~positive_mask, "comparison_confidence"] = calibrated_theme_df.loc[~positive_mask, "confidence"]

    sample_positive = calibrated_theme_df[calibrated_theme_df["theme_name"].isin(POSITIVE_THEMES)].copy()
    differential_mean = (
        sample_positive.groupby("query_id", as_index=False)["differential_signal_index"]
        .mean()
        .rename(columns={"differential_signal_index": "mean_differential_signal"})
    )
    background_mean = (
        sample_positive.groupby("query_id", as_index=False)["shared_background_burden"]
        .mean()
        .rename(columns={"shared_background_burden": "mean_shared_background_burden"})
    )
    comparison_conf_mean = (
        sample_positive.groupby("query_id", as_index=False)["comparison_confidence"]
        .mean()
        .rename(columns={"comparison_confidence": "mean_comparison_positive_confidence"})
    )
    comparison_score_wide = (
        sample_positive.pivot_table(index="query_id", columns="theme_name", values="comparison_score", aggfunc="mean")
        .reset_index()
        .fillna(0.0)
    )
    sample_summary_df = (
        query_df.merge(differential_mean, on="query_id", how="left")
        .merge(background_mean, on="query_id", how="left")
        .merge(comparison_conf_mean, on="query_id", how="left")
        .merge(comparison_score_wide, on="query_id", how="left", suffixes=("", "_comparison"))
    )
    if "class_label_x" in sample_summary_df.columns and "class_label_y" in sample_summary_df.columns:
        sample_summary_df["class_label"] = sample_summary_df["class_label_y"].fillna(sample_summary_df["class_label_x"])
        sample_summary_df = sample_summary_df.drop(columns=["class_label_x", "class_label_y"])
    elif "class_label_x" in sample_summary_df.columns:
        sample_summary_df = sample_summary_df.rename(columns={"class_label_x": "class_label"})
    elif "class_label_y" in sample_summary_df.columns:
        sample_summary_df = sample_summary_df.rename(columns={"class_label_y": "class_label"})
    sample_summary_df["comparison_caution_load"] = (
        sample_summary_df["mean_caution_score"] * (1.0 - 0.18 * sample_summary_df["mean_differential_signal"].fillna(0.0))
    ).clip(lower=0.0, upper=1.0)

    shared_background_df = (
        calibrated_theme_df[calibrated_theme_df["theme_name"].isin(POSITIVE_THEMES)]
        .groupby("theme_name", as_index=False)
        .agg(
            shared_background_score=("shared_background_score", "mean"),
            mean_group_shift=("group_shift_score", "mean"),
            mean_comparison_score=("comparison_score", "mean"),
        )
        .sort_values(["shared_background_score", "mean_group_shift"], ascending=[False, False])
    )

    shifted_rows = []
    for theme_name in POSITIVE_THEMES:
        theme_subset = calibrated_theme_df[calibrated_theme_df["theme_name"] == theme_name]
        if theme_subset.empty:
            continue
        for score_column, label in [("score", "before"), ("comparison_score", "after")]:
            class_means = theme_subset.groupby("class_label")[score_column].mean()
            if len(class_means) != 2:
                continue
            class_names = list(class_means.index)
            shifted_rows.append(
                {
                    "stage": label,
                    "theme_name": theme_name,
                    "class_a": class_names[0],
                    "class_b": class_names[1],
                    "mean_a": float(class_means.iloc[0]),
                    "mean_b": float(class_means.iloc[1]),
                    "difference_a_minus_b": float(class_means.iloc[0] - class_means.iloc[1]),
                }
            )
    shifted_theme_df = pd.DataFrame(shifted_rows)

    before_feature_df = (
        calibrated_theme_df[calibrated_theme_df["theme_name"].isin(POSITIVE_THEMES)]
        .pivot_table(index="query_id", columns="theme_name", values="score", aggfunc="mean")
        .reset_index()
        .fillna(0.0)
    )
    after_feature_df = (
        calibrated_theme_df[calibrated_theme_df["theme_name"].isin(POSITIVE_THEMES)]
        .pivot_table(index="query_id", columns="theme_name", values="comparison_score", aggfunc="mean")
        .reset_index()
        .fillna(0.0)
    )

    merged_before = before_feature_df.merge(metadata_df[["biosample_id", "class_label"]], left_on="query_id", right_on="biosample_id", how="left")
    merged_after = after_feature_df.merge(metadata_df[["biosample_id", "class_label"]], left_on="query_id", right_on="biosample_id", how="left")

    before_X = merged_before[POSITIVE_THEMES].to_numpy(dtype=float)
    after_X = merged_after[POSITIVE_THEMES].to_numpy(dtype=float)
    y = merged_before["class_label"].astype(str).to_numpy()
    before_silhouette = float(silhouette_score(before_X, y)) if len(np.unique(y)) > 1 else 0.0
    after_silhouette = float(silhouette_score(after_X, y)) if len(np.unique(y)) > 1 else 0.0

    before_effects = []
    after_effects = []
    for theme_name in POSITIVE_THEMES:
        a_before = merged_before[merged_before["class_label"] == classes[0]][theme_name].to_numpy(dtype=float)
        b_before = merged_before[merged_before["class_label"] == classes[1]][theme_name].to_numpy(dtype=float)
        a_after = merged_after[merged_after["class_label"] == classes[0]][theme_name].to_numpy(dtype=float)
        b_after = merged_after[merged_after["class_label"] == classes[1]][theme_name].to_numpy(dtype=float)
        before_effects.append(abs(_effect_size(a_before, b_before)))
        after_effects.append(abs(_effect_size(a_after, b_after)))

    before_after_metrics_df = pd.DataFrame(
        [
            {"metric_name": "theme_space_silhouette", "before": before_silhouette, "after": after_silhouette},
            {"metric_name": "mean_abs_theme_effect_size", "before": float(np.mean(before_effects)), "after": float(np.mean(after_effects))},
            {
                "metric_name": "mean_positive_confidence",
                "before": float(sample_summary_df["mean_positive_confidence"].mean()),
                "after": float(sample_summary_df["mean_comparison_positive_confidence"].mean()),
            },
            {
                "metric_name": "mean_caution_score",
                "before": float(sample_summary_df["mean_caution_score"].mean()),
                "after": float(sample_summary_df["comparison_caution_load"].mean()),
            },
            {
                "metric_name": "mean_shared_background_burden",
                "before": 0.0,
                "after": float(sample_summary_df["mean_shared_background_burden"].mean()),
            },
            {
                "metric_name": "mean_differential_signal_index",
                "before": 0.0,
                "after": float(sample_summary_df["mean_differential_signal"].mean()),
            },
        ]
    )

    pca_before = PCA(n_components=2, random_state=42).fit_transform(before_X)
    pca_after = PCA(n_components=2, random_state=42).fit_transform(after_X)
    theme_space_df = pd.DataFrame(
        {
            "query_id": merged_before["query_id"],
            "class_label": merged_before["class_label"],
            "pc1_before": pca_before[:, 0],
            "pc2_before": pca_before[:, 1],
            "pc1_after": pca_after[:, 0],
            "pc2_after": pca_after[:, 1],
        }
    )

    differential_evidence_df = build_differential_evidence_summary(results=results, metadata_df=metadata_df)

    return SerumCalibrationBundle(
        calibrated_theme_df=calibrated_theme_df,
        sample_summary_df=sample_summary_df,
        shared_background_df=shared_background_df,
        shifted_theme_df=shifted_theme_df,
        differential_evidence_df=differential_evidence_df,
        before_after_metrics_df=before_after_metrics_df,
        failure_mode_note=failure_mode_note,
        theme_space_df=theme_space_df,
    )
