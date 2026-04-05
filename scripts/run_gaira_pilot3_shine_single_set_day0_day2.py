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
from scripts.run_gaira_pilot3_shine_day2_controlanchored import (
    ARCH_DIR,
    CONFIG_SPEC,
    FAMILY_ORDER,
    FIXED_RADAR_AXES,
    PHASE1_DIR,
    ROOT,
    _adjacent_distance_metrics,
    _build_metric_frame,
    _class_family_means,
    _cohort_delta,
    _control_delta,
    _ensure_fixed_axes,
    _extract_sample_id,
    _family_fingerprint_from_retrieval,
    _family_response_contributions,
    _fit_pca,
    _fit_response_axis,
    _mean_by_class,
    _parse_concentration,
    _plot_adjacent_distance,
    _plot_condition_separation,
    _plot_family_bars,
    _plot_family_shift_vs_control,
    _plot_radar_grid,
    _plot_response_axis_boxplot,
    _plot_response_axis_scatter,
    _plot_response_axis_trend,
    _plot_scatter,
    _prepare_grounding_and_mapping,
    _representation_separation_metrics,
    _resolve_alias,
    _trajectory_index,
)


SPRINT_SUBDIR = "pilot3_shine_single_set_day0_day2"
SUBSET_ALIAS = "shine_ev_stress"
KEEP_CLASS_ORDER = ["D0_C0", "D0_C10", "D0_C20", "D0_C40", "D2_C0", "D2_C10", "D2_C20", "D2_C40"]
DAY_CLASS_ORDER = {
    "D0": ["D0_C0", "D0_C10", "D0_C20", "D0_C40"],
    "D2": ["D2_C0", "D2_C10", "D2_C20", "D2_C40"],
}
PREVIOUS_DAY2_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot3_shine_day2_controlanchored"
)
PREVIOUS_ALLDAY_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot3_shine_ev_sers_fullspectra"
)


def _prepare_query_df(query_df: pd.DataFrame) -> pd.DataFrame:
    work = query_df.reset_index(drop=True).copy()
    work["sample_id"] = work.apply(_extract_sample_id, axis=1)
    work["day_label"] = work["class_label"].astype(str).str.extract(r"^(D\d+)", expand=False)
    work["trajectory_concentration"] = work["class_label"].astype(str).map(_parse_concentration)
    work["trajectory_index"] = [
        _trajectory_index(day_label, concentration)
        for day_label, concentration in zip(
            work["day_label"].astype(str),
            work["trajectory_concentration"].astype(int),
            strict=False,
        )
    ]
    work["n_scans"] = 1
    return work


def _axes_present(df: pd.DataFrame) -> list[str]:
    return [axis for axis in ALL_AXES if axis in df.columns]


def _day_filter(df: pd.DataFrame, day_label: str) -> pd.DataFrame:
    return df[df["day_label"].astype(str) == str(day_label)].copy().reset_index(drop=True)


def _manual_markdown_table(df: pd.DataFrame) -> list[str]:
    header = "| " + " | ".join(df.columns.astype(str).tolist()) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    body = [
        "| "
        + " | ".join(
            f"{value:.4f}" if isinstance(value, float) and pd.notna(value) else str(value)
            for value in row
        )
        + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return [header, divider, *body]


def _inspect_sets(query_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subclass_label, sub in query_df.groupby("subclass_label", sort=True):
        row = {
            "subclass_label": str(subclass_label),
            "n_rows_total": int(len(sub)),
            "n_rows_day0": int((sub["class_label"].astype(str).str.startswith("D0_")).sum()),
            "n_rows_day2": int((sub["class_label"].astype(str).str.startswith("D2_")).sum()),
        }
        for day_label in ["D0", "D2"]:
            for concentration in [0, 10, 20, 40]:
                class_label = f"{day_label}_C{concentration}"
                row[f"n_rows_{day_label.lower()}_C{concentration}"] = int(
                    (sub["class_label"].astype(str) == class_label).sum()
                )
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("subclass_label").reset_index(drop=True)
    completeness_cols = [
        "n_rows_d0_C0",
        "n_rows_d0_C10",
        "n_rows_d0_C20",
        "n_rows_d0_C40",
        "n_rows_d2_C0",
        "n_rows_d2_C10",
        "n_rows_d2_C20",
        "n_rows_d2_C40",
    ]
    out["complete_day0_day2"] = out[completeness_cols].gt(0).all(axis=1)
    out["selection_priority"] = (
        out["complete_day0_day2"].astype(int) * 10_000_000
        + out["n_rows_day0"].astype(int) * 1_000
        + out["n_rows_day2"].astype(int)
    )
    return out


def _choose_set(inspect_df: pd.DataFrame) -> str:
    chosen = inspect_df.sort_values(
        ["complete_day0_day2", "n_rows_day0", "n_rows_day2", "n_rows_total", "subclass_label"],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    return str(chosen["subclass_label"])


def _day_verification(filtered_df: pd.DataFrame) -> pd.DataFrame:
    row = {
        "chosen_set": str(filtered_df["subclass_label"].astype(str).iloc[0]),
        "total_rows": int(len(filtered_df)),
        "unique_sample_ids": int(filtered_df["sample_id"].astype(str).nunique()),
    }
    for class_label in KEEP_CLASS_ORDER:
        row[f"rows_{class_label}"] = int((filtered_df["class_label"].astype(str) == class_label).sum())
    return pd.DataFrame([row])


def _run_day_metrics(
    label: str,
    spectrum_bsv_df: pd.DataFrame,
    spectrum_family_df: pd.DataFrame,
    axes: list[str],
) -> dict[str, object]:
    bsv_df = _day_filter(spectrum_bsv_df, label)
    family_df = _day_filter(spectrum_family_df, label)
    delta_cohort_df = _cohort_delta(bsv_df, axes)
    delta_control_df = _control_delta(bsv_df, axes, control_label=f"{label}_C0")
    separation_df = pd.concat(
        [
            _representation_separation_metrics(bsv_df, axes, representation="bsv"),
            _representation_separation_metrics(delta_cohort_df, axes, representation=f"delta_{label.lower()}_cohort"),
            _representation_separation_metrics(delta_control_df, axes, representation=f"delta_{label.lower()}_control"),
        ],
        ignore_index=True,
    )
    monotonicity_bsv_df, bsv_pca_df = _build_metric_frame("bsv", bsv_df, axes, family_df)
    monotonicity_control_df, control_pca_df = _build_metric_frame(
        f"delta_{label.lower()}_control",
        delta_control_df,
        axes,
        family_df,
    )
    monotonicity_df = pd.concat([monotonicity_bsv_df, monotonicity_control_df], ignore_index=True)
    adjacent_df = pd.concat(
        [
            _adjacent_distance_metrics("bsv", bsv_df, axes),
            _adjacent_distance_metrics(f"delta_{label.lower()}_control", delta_control_df, axes),
        ],
        ignore_index=True,
    )
    response_axis_scores_df, response_axis_metrics_df, response_axis_bsv_df = _fit_response_axis(
        delta_control_df, axes
    )
    response_axis_family_df = _family_response_contributions(family_df)
    class_mean_bsv_df = _mean_by_class(bsv_df, axes)
    class_mean_delta_control_df = _mean_by_class(delta_control_df, axes)
    class_mean_family_df = _class_family_means(family_df)
    return {
        "bsv_df": bsv_df,
        "family_df": family_df,
        "delta_cohort_df": delta_cohort_df,
        "delta_control_df": delta_control_df,
        "separation_df": separation_df,
        "monotonicity_df": monotonicity_df,
        "adjacent_df": adjacent_df,
        "response_axis_scores_df": response_axis_scores_df,
        "response_axis_metrics_df": response_axis_metrics_df,
        "response_axis_bsv_df": response_axis_bsv_df,
        "response_axis_family_df": response_axis_family_df,
        "class_mean_bsv_df": class_mean_bsv_df,
        "class_mean_delta_control_df": class_mean_delta_control_df,
        "class_mean_family_df": class_mean_family_df,
        "bsv_pca_df": bsv_pca_df,
        "control_pca_df": control_pca_df,
    }


def _plot_day0_day2_family_shift(
    day0_family_df: pd.DataFrame,
    day2_family_df: pd.DataFrame,
    output_path: Path,
) -> None:
    def build_shift(df: pd.DataFrame, class_order: list[str]) -> np.ndarray:
        control = df[df["class_label"].astype(str) == class_order[0]].set_index("family")["family_fraction"].reindex(FAMILY_ORDER, fill_value=0.0)
        target = df[df["class_label"].astype(str) == class_order[-1]].set_index("family")["family_fraction"].reindex(FAMILY_ORDER, fill_value=0.0)
        return (target - control).to_numpy(dtype=float)

    shift0 = build_shift(day0_family_df, DAY_CLASS_ORDER["D0"])
    shift2 = build_shift(day2_family_df, DAY_CLASS_ORDER["D2"])
    x = np.arange(len(FAMILY_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    ax.bar(x - width / 2, shift0, width=width, color="#355070", alpha=0.9, label="Day 0 C40 - C0")
    ax.bar(x + width / 2, shift2, width=width, color="#b56576", alpha=0.9, label="Day 2 C40 - C0")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILY_ORDER, rotation=25, ha="right")
    ax.set_ylabel("Family fraction shift")
    ax.set_title("Day 0 vs Day 2 Family Shift Comparison")
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_day0_day2_condition_comparison(
    day0_sep: pd.DataFrame,
    day2_sep: pd.DataFrame,
    output_path: Path,
) -> None:
    metrics = ["silhouette_by_concentration", "mean_centroid_distance", "nearest_neighbor_purity"]
    titles = ["Silhouette", "Mean Centroid Distance", "NN Purity"]
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.6))
    for ax, metric, title in zip(axs, metrics, titles, strict=False):
        day0_val = float(day0_sep.sort_values("silhouette_by_concentration", ascending=False).iloc[0][metric])
        day2_val = float(day2_sep.sort_values("silhouette_by_concentration", ascending=False).iloc[0][metric])
        ax.bar([0, 1], [day0_val, day2_val], color=["#355070", "#b56576"], alpha=0.9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Day 0", "Day 2"])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    fig.suptitle("Day 0 vs Day 2 Condition Separation")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_day0_day2_adjacent_comparison(
    day0_adj: pd.DataFrame,
    day2_adj: pd.DataFrame,
    output_path: Path,
) -> None:
    control0 = day0_adj[day0_adj["representation"].astype(str) == "delta_d0_control"].copy()
    control2 = day2_adj[day2_adj["representation"].astype(str) == "delta_d2_control"].copy()
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    x = np.arange(len(control0))
    width = 0.35
    ax.bar(x - width / 2, control0["adjacent_distance"].to_numpy(dtype=float), width=width, color="#355070", alpha=0.9, label="Day 0")
    ax.bar(x + width / 2, control2["adjacent_distance"].to_numpy(dtype=float), width=width, color="#b56576", alpha=0.9, label="Day 2")
    ax.set_xticks(x)
    ax.set_xticklabels(
        (control0["from_class"].astype(str) + "→" + control0["to_class"].astype(str)).tolist(),
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Adjacent centroid distance")
    ax.set_title("Day 0 vs Day 2 Adjacent Distance")
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _build_report(
    report_path: Path,
    set_inspection_df: pd.DataFrame,
    chosen_set: str,
    verification_df: pd.DataFrame,
    day0: dict[str, object],
    day2: dict[str, object],
    comparison_df: pd.DataFrame,
    set1_mapping_note: str,
) -> None:
    best_day0_sep = day0["separation_df"].sort_values("silhouette_by_concentration", ascending=False).iloc[0]
    best_day2_sep = day2["separation_df"].sort_values("silhouette_by_concentration", ascending=False).iloc[0]
    best_day0_mono = day0["monotonicity_df"].sort_values("monotonicity_score", ascending=False).iloc[0]
    best_day2_mono = day2["monotonicity_df"].sort_values("monotonicity_score", ascending=False).iloc[0]
    lines = [
        "# GAIRAv3 Pilot 3 SHINE Single-Set Day0 Day2 Report",
        "",
        "## 1. Why This Follow-Up Was Needed",
        "- Pooled-set SHINE analyses may be diluted by between-set variability.",
        "- The SHINE paper frames the task as both day-specific and set-sensitive.",
        "- This pass isolates a single local set and compares Day 0 vs Day 2 under the locked cfg05 representation.",
        "",
        "## 2. Set Inspection and Chosen Set",
        * _manual_markdown_table(set_inspection_df.drop(columns=["selection_priority"])),
        "",
        f"- Chosen set: `{chosen_set}`.",
        f"- Selection rule: prefer a set with complete Day 0 plus Day 2 coverage, then maximize usable rows.",
        f"- Set1 mapping note: {set1_mapping_note}",
        "",
        "## 3. Day 0 Results",
        f"- Best Day 0 concentration silhouette: `{best_day0_sep['representation']}` at `{float(best_day0_sep['silhouette_by_concentration']):.4f}`.",
        f"- Strongest Day 0 monotonicity: `{best_day0_mono['representation']}` / `{best_day0_mono['metric_name']}` / `{best_day0_mono['level']}` with Spearman `{float(best_day0_mono['spearman_r']):.4f}`.",
        f"- Day 0 response-axis Spearman: `{float(day0['response_axis_metrics_df']['spearman_r'].iloc[0]):.4f}`.",
        "- Day 0 is treated as the weaker-response comparator, so any concentration structure here should be limited.",
        "",
        "## 4. Day 2 Results",
        f"- Best Day 2 concentration silhouette: `{best_day2_sep['representation']}` at `{float(best_day2_sep['silhouette_by_concentration']):.4f}`.",
        f"- Strongest Day 2 monotonicity: `{best_day2_mono['representation']}` / `{best_day2_mono['metric_name']}` / `{best_day2_mono['level']}` with Spearman `{float(best_day2_mono['spearman_r']):.4f}`.",
        f"- Day 2 response-axis Spearman: `{float(day2['response_axis_metrics_df']['spearman_r'].iloc[0]):.4f}`.",
        "- Day 2 is the main test for whether cfg05 can expose a clearer APAP-response direction once one set is isolated.",
        "",
        "## 5. Day 2 vs Day 0 Comparison",
        * _manual_markdown_table(comparison_df),
        "",
        "## 6. Biochemical Interpretation",
        "- Interpret only broad biochemical themes, not molecules.",
        "- Day 0 top BSV contributions: "
        + ", ".join(
            f"`{row.axis_name}` ({row.direction}, {row.abs_fraction:.3f})"
            for row in day0["response_axis_bsv_df"].head(4).itertuples(index=False)
        )
        + ".",
        "- Day 2 top BSV contributions: "
        + ", ".join(
            f"`{row.axis_name}` ({row.direction}, {row.abs_fraction:.3f})"
            for row in day2["response_axis_bsv_df"].head(4).itertuples(index=False)
        )
        + ".",
        "- Day 0 top family contributions: "
        + ", ".join(
            f"`{row.family}` ({row.direction}, {row.abs_fraction:.3f})"
            for row in day0["response_axis_family_df"].head(4).itertuples(index=False)
        )
        + ".",
        "- Day 2 top family contributions: "
        + ", ".join(
            f"`{row.family}` ({row.direction}, {row.abs_fraction:.3f})"
            for row in day2["response_axis_family_df"].head(4).itertuples(index=False)
        )
        + ".",
        "",
        "## 7. Comparison to Previous SHINE Runs",
        "- This single-set analysis should only be considered stronger if Day 2 improves relative to both pooled all-day SHINE and pooled-set Day-2-only SHINE.",
        "",
        "## 8. Final Conclusion",
        "- Single-set isolation helps only if it increases Day 2 concentration ordering relative to Day 0 and relative to the pooled runs.",
        "- If Day 2 remains weak, SHINE under GAIRA is still primarily a latent-state dataset rather than a clean task-specific APAP-response readout.",
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
    query_df = _prepare_query_df(load_query_dataframe(resolved.dataset_row))

    inspect_df = _inspect_sets(query_df)
    inspect_df.to_csv(sprint_paths.tables_dir / "set_inspection_summary.csv", index=False)
    chosen_set = _choose_set(inspect_df)
    chosen_df = query_df[
        (query_df["subclass_label"].astype(str) == chosen_set)
        & (query_df["class_label"].astype(str).isin(KEEP_CLASS_ORDER))
    ].copy().reset_index(drop=True)

    verification_df = _day_verification(chosen_df)
    verification_df.to_csv(sprint_paths.tables_dir / "day0_day2_input_verification.csv", index=False)

    grounding_df, mapping_df, harness_config, _ = _prepare_grounding_and_mapping(
        registries, resolved, CONFIG_SPEC
    )
    spectrum_bsv_df, retrieval_df = build_bsv_profiles_pass5(
        chosen_df,
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
    spectrum_bsv_df = spectrum_bsv_df.copy()
    spectrum_bsv_df["sample_id"] = spectrum_bsv_df["sample_key"].astype(str).map(
        chosen_df.set_index("sample_key")["sample_id"].astype(str).to_dict()
    )
    spectrum_bsv_df["day_label"] = spectrum_bsv_df["class_label"].astype(str).str.extract(r"^(D\d+)", expand=False)
    spectrum_bsv_df["trajectory_concentration"] = spectrum_bsv_df["class_label"].astype(str).map(_parse_concentration)
    spectrum_bsv_df["trajectory_index"] = [
        _trajectory_index(day_label, concentration)
        for day_label, concentration in zip(
            spectrum_bsv_df["day_label"].astype(str),
            spectrum_bsv_df["trajectory_concentration"].astype(int),
            strict=False,
        )
    ]
    spectrum_bsv_df.to_csv(sprint_paths.tables_dir / "per_spectrum_bsv_day0_day2.csv", index=False)
    retrieval_df.to_csv(sprint_paths.tables_dir / "per_spectrum_retrieval_hits_day0_day2.csv", index=False)

    spectrum_family_df = _family_fingerprint_from_retrieval(
        retrieval_df,
        spectrum_bsv_df[["sample_key", "sample_id", "trajectory_concentration", "trajectory_index"]].assign(
            class_label=spectrum_bsv_df["class_label"].astype(str).values
        ),
    )
    spectrum_family_df["day_label"] = spectrum_family_df["class_label"].astype(str).str.extract(r"^(D\d+)", expand=False)
    spectrum_family_df.to_csv(sprint_paths.tables_dir / "per_spectrum_family_fingerprint_day0_day2.csv", index=False)
    _mean_by_class(spectrum_bsv_df, axes).to_csv(
        sprint_paths.tables_dir / "class_mean_bsv_day0_day2.csv", index=False
    )
    _class_family_means(spectrum_family_df).to_csv(
        sprint_paths.tables_dir / "class_mean_family_fingerprint_day0_day2.csv", index=False
    )

    day0 = _run_day_metrics("D0", spectrum_bsv_df, spectrum_family_df, axes)
    day2 = _run_day_metrics("D2", spectrum_bsv_df, spectrum_family_df, axes)

    day0["delta_cohort_df"].to_csv(sprint_paths.tables_dir / "per_spectrum_delta_bsv_day0_cohort.csv", index=False)
    day0["delta_control_df"].to_csv(sprint_paths.tables_dir / "per_spectrum_delta_bsv_day0_control.csv", index=False)
    day2["delta_cohort_df"].to_csv(sprint_paths.tables_dir / "per_spectrum_delta_bsv_day2_cohort.csv", index=False)
    day2["delta_control_df"].to_csv(sprint_paths.tables_dir / "per_spectrum_delta_bsv_day2_control.csv", index=False)
    _mean_by_class(day0["delta_cohort_df"], axes).to_csv(
        sprint_paths.tables_dir / "class_mean_delta_bsv_day0_cohort.csv", index=False
    )
    _mean_by_class(day0["delta_control_df"], axes).to_csv(
        sprint_paths.tables_dir / "class_mean_delta_bsv_day0_control.csv", index=False
    )
    _mean_by_class(day2["delta_cohort_df"], axes).to_csv(
        sprint_paths.tables_dir / "class_mean_delta_bsv_day2_cohort.csv", index=False
    )
    _mean_by_class(day2["delta_control_df"], axes).to_csv(
        sprint_paths.tables_dir / "class_mean_delta_bsv_day2_control.csv", index=False
    )

    day0["separation_df"].to_csv(sprint_paths.tables_dir / "day0_concentration_separation_metrics.csv", index=False)
    day2["separation_df"].to_csv(sprint_paths.tables_dir / "day2_concentration_separation_metrics.csv", index=False)

    day0["response_axis_scores_df"].to_csv(
        sprint_paths.tables_dir / "day0_response_axis_scores.csv", index=False
    )
    day2["response_axis_scores_df"].to_csv(
        sprint_paths.tables_dir / "day2_response_axis_scores.csv", index=False
    )
    day0["response_axis_metrics_df"].to_csv(
        sprint_paths.tables_dir / "day0_response_axis_metrics.csv", index=False
    )
    day2["response_axis_metrics_df"].to_csv(
        sprint_paths.tables_dir / "day2_response_axis_metrics.csv", index=False
    )
    day0["response_axis_bsv_df"].to_csv(
        sprint_paths.tables_dir / "day0_response_axis_bsv_contributions.csv", index=False
    )
    day2["response_axis_bsv_df"].to_csv(
        sprint_paths.tables_dir / "day2_response_axis_bsv_contributions.csv", index=False
    )
    day0["response_axis_family_df"].to_csv(
        sprint_paths.tables_dir / "day0_response_axis_family_contributions.csv", index=False
    )
    day2["response_axis_family_df"].to_csv(
        sprint_paths.tables_dir / "day2_response_axis_family_contributions.csv", index=False
    )

    day_compare_df = pd.DataFrame(
        [
            {
                "day_label": "D0",
                "strongest_monotonicity_correlation": float(day0["monotonicity_df"]["monotonicity_score"].fillna(0.0).max()),
                "strongest_concentration_silhouette": float(day0["separation_df"]["silhouette_by_concentration"].max()),
                "strongest_response_axis_correlation": float(day0["response_axis_metrics_df"]["spearman_r"].iloc[0]),
                "mean_adjacent_concentration_distance": float(
                    day0["adjacent_df"][day0["adjacent_df"]["representation"].astype(str) == "delta_d0_control"]["adjacent_distance"].mean()
                ),
            },
            {
                "day_label": "D2",
                "strongest_monotonicity_correlation": float(day2["monotonicity_df"]["monotonicity_score"].fillna(0.0).max()),
                "strongest_concentration_silhouette": float(day2["separation_df"]["silhouette_by_concentration"].max()),
                "strongest_response_axis_correlation": float(day2["response_axis_metrics_df"]["spearman_r"].iloc[0]),
                "mean_adjacent_concentration_distance": float(
                    day2["adjacent_df"][day2["adjacent_df"]["representation"].astype(str) == "delta_d2_control"]["adjacent_distance"].mean()
                ),
            },
        ]
    )
    day_compare_df.to_csv(sprint_paths.tables_dir / "day0_vs_day2_response_comparison.csv", index=False)

    prior_df = pd.read_csv(PREVIOUS_DAY2_ROOT / "tables" / "day2_vs_prior_shine_comparison.csv")
    pooled_all_day = prior_df[prior_df["run_name"].astype(str) == "all_day_fullspectra_pilot3"].iloc[0]
    pooled_day2 = prior_df[prior_df["run_name"].astype(str) == "day2_controlanchored_pass"].iloc[0]
    shine_compare_df = pd.DataFrame(
        [
            {
                "run_name": "pooled_all_day_shine",
                "best_concentration_silhouette": float(pooled_all_day["strongest_concentration_silhouette"]),
                "best_monotonicity_correlation": float(pooled_all_day["strongest_monotonicity_correlation"]) if pd.notna(pooled_all_day["strongest_monotonicity_correlation"]) else math.nan,
                "best_response_axis_correlation": float(pooled_all_day["best_response_axis_correlation"]),
            },
            {
                "run_name": "day2_only_pooled_set",
                "best_concentration_silhouette": float(pooled_day2["strongest_concentration_silhouette"]),
                "best_monotonicity_correlation": float(pooled_day2["strongest_monotonicity_correlation"]),
                "best_response_axis_correlation": float(pooled_day2["best_response_axis_correlation"]),
            },
            {
                "run_name": "single_set_day0",
                "best_concentration_silhouette": float(day_compare_df[day_compare_df["day_label"].astype(str) == "D0"]["strongest_concentration_silhouette"].iloc[0]),
                "best_monotonicity_correlation": float(day_compare_df[day_compare_df["day_label"].astype(str) == "D0"]["strongest_monotonicity_correlation"].iloc[0]),
                "best_response_axis_correlation": float(day_compare_df[day_compare_df["day_label"].astype(str) == "D0"]["strongest_response_axis_correlation"].iloc[0]),
            },
            {
                "run_name": "single_set_day2",
                "best_concentration_silhouette": float(day_compare_df[day_compare_df["day_label"].astype(str) == "D2"]["strongest_concentration_silhouette"].iloc[0]),
                "best_monotonicity_correlation": float(day_compare_df[day_compare_df["day_label"].astype(str) == "D2"]["strongest_monotonicity_correlation"].iloc[0]),
                "best_response_axis_correlation": float(day_compare_df[day_compare_df["day_label"].astype(str) == "D2"]["strongest_response_axis_correlation"].iloc[0]),
            },
        ]
    )
    shine_compare_df.to_csv(sprint_paths.tables_dir / "shine_single_set_vs_prior_comparison.csv", index=False)

    _, spectral_matrix = decode_and_align(chosen_df)
    spectral_scores, spectral_explained = _fit_pca(spectral_matrix, scale=False)
    spectral_pca_df = chosen_df[
        ["sample_key", "sample_id", "class_label", "day_label", "trajectory_concentration", "trajectory_index", "subclass_label"]
    ].copy()
    spectral_pca_df["pc1"] = spectral_scores[:, 0]
    spectral_pca_df["pc2"] = spectral_scores[:, 1]
    spectral_pca_df["pc1_explained_ratio"] = float(spectral_explained[0])
    spectral_pca_df["pc2_explained_ratio"] = float(spectral_explained[1])

    for day_label, day_metrics in [("D0", day0), ("D2", day2)]:
        spectral_sub = spectral_pca_df[spectral_pca_df["day_label"].astype(str) == day_label].copy()
        _plot_scatter(
            spectral_sub,
            "pc1",
            "pc2",
            sprint_paths.figures_dir / f"{day_label.lower()}_pca_spectral_by_concentration.png",
            title=f"{day_label} Spectral PCA by Concentration",
            hue_col="class_label",
        )
        _plot_scatter(
            day_metrics["bsv_pca_df"],
            "pc1",
            "pc2",
            sprint_paths.figures_dir / f"{day_label.lower()}_pca_bsv_by_concentration.png",
            title=f"{day_label} BSV PCA by Concentration",
            hue_col="class_label",
        )
        _plot_scatter(
            day_metrics["control_pca_df"],
            "pc1",
            "pc2",
            sprint_paths.figures_dir / f"{day_label.lower()}_pca_delta_control_by_concentration.png",
            title=f"{day_label} Control-Delta PCA by Concentration",
            hue_col="class_label",
        )
        _plot_radar_grid(
            _ensure_fixed_axes(day_metrics["class_mean_bsv_df"])[["class_label"] + FIXED_RADAR_AXES],
            "class_label",
            sprint_paths.figures_dir / f"{day_label.lower()}_radar_bsv_by_concentration.png",
            f"{day_label} Absolute BSV by Concentration",
            delta_mode=False,
        )
        _plot_radar_grid(
            _ensure_fixed_axes(day_metrics["class_mean_delta_control_df"])[["class_label"] + FIXED_RADAR_AXES],
            "class_label",
            sprint_paths.figures_dir / f"{day_label.lower()}_radar_delta_control_by_concentration.png",
            f"{day_label} Control-Anchored Delta by Concentration",
            delta_mode=True,
        )
        _plot_response_axis_boxplot(
            day_metrics["response_axis_scores_df"],
            sprint_paths.figures_dir / f"{day_label.lower()}_response_axis_boxplot.png",
        )
        _plot_response_axis_trend(
            day_metrics["response_axis_scores_df"],
            sprint_paths.figures_dir / f"{day_label.lower()}_response_axis_trend.png",
        )

    _plot_day0_day2_condition_comparison(
        day0["separation_df"],
        day2["separation_df"],
        sprint_paths.figures_dir / "day0_vs_day2_condition_separation_comparison.png",
    )
    _plot_day0_day2_adjacent_comparison(
        day0["adjacent_df"],
        day2["adjacent_df"],
        sprint_paths.figures_dir / "day0_vs_day2_adjacent_distance_comparison.png",
    )
    _plot_day0_day2_family_shift(
        day0["class_mean_family_df"],
        day2["class_mean_family_df"],
        sprint_paths.figures_dir / "day0_vs_day2_family_shift_comparison.png",
    )

    set1_mapping_note = (
        "local metadata exposes `Set9` and `Set10`, not manuscript-style `Set1`/`Set2`; "
        "no local mapping explicitly renames `Set9` to `Set1`, so the run uses `Set9` because it is the only set with complete Day 0 plus Day 2 coverage."
    )
    report_md = sprint_paths.report_dir / "GAIRAv3_Pilot3_SHINE_single_set_day0_day2_report.md"
    report_pdf = sprint_paths.report_dir / "GAIRAv3_Pilot3_SHINE_single_set_day0_day2_report.pdf"
    _build_report(
        report_md,
        inspect_df,
        chosen_set,
        verification_df,
        day0,
        day2,
        shine_compare_df,
        set1_mapping_note,
    )
    build_pdf_report(report_md, sorted(sprint_paths.figures_dir.glob("*.png")), report_pdf)


if __name__ == "__main__":
    main()
