#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from gaira.demo.shared_embedding_audit_utils import (
    SHARED_RUN_CANDIDATES,
    AuditUnit,
    align_run_values,
    build_dataset_inventory,
    ensure_dir,
    evaluate_representation,
    grounding_inventory,
    load_direct_baseline,
    ontology_rows,
    primary_composite,
    nuisance_composite,
    read_run_metadata,
    run_inventory,
    sample_unit_metadata,
)
from gaira.demo.v8_analysis_utils import save_barplot, save_heatmap, save_scatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_shared_embedding_audit_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit existing shared embedding runs for within-dataset usefulness.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples-per-unit", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def resolve_output_dir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        fallback = Path("reports/gaira_shared_embedding_audit_v1")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def table_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    return "```text\n" + df.to_string(index=False) + "\n```"


def useful_label(label: str) -> str:
    return label.replace("_", " ")


def run_maps_for_unit(
    sampled_meta: pd.DataFrame,
    direct_values: np.ndarray | None,
    best_name: str,
    best_values: np.ndarray,
    output_dir: Path,
    *,
    seed: int,
) -> None:
    from gaira.demo.v8_analysis_utils import reduce_for_plot

    plot_meta = sampled_meta.copy()
    if "ordered_value" in plot_meta.columns and plot_meta["ordered_value"].notna().any():
        plot_meta["plot_target"] = plot_meta["ordered_value"]
    else:
        plot_meta["plot_target"] = plot_meta["target_label"].replace("", "unlabeled")
    plot_meta["plot_nuisance"] = plot_meta["nuisance_label"].replace("", "none")

    if direct_values is not None:
        direct_coords = reduce_for_plot(direct_values, seed=seed)
        direct_df = plot_meta.copy()
        direct_df["dim1"] = direct_coords[:, 0]
        direct_df["dim2"] = direct_coords[:, 1]
        save_scatter(
            direct_df,
            x="dim1",
            y="dim2",
            hue="plot_target",
            style=None,
            size=None,
            title=f"{sampled_meta['audit_unit_id'].iloc[0]} direct baseline by target",
            output_path=output_dir / f"{sampled_meta['audit_unit_id'].iloc[0]}__direct_by_target.png",
        )
        if direct_df["plot_nuisance"].nunique() > 1:
            save_scatter(
                direct_df,
                x="dim1",
                y="dim2",
                hue="plot_nuisance",
                style=None,
                size=None,
                title=f"{sampled_meta['audit_unit_id'].iloc[0]} direct baseline by nuisance",
                output_path=output_dir / f"{sampled_meta['audit_unit_id'].iloc[0]}__direct_by_nuisance.png",
            )

    shared_coords = reduce_for_plot(best_values, seed=seed)
    shared_df = plot_meta.copy()
    shared_df["dim1"] = shared_coords[:, 0]
    shared_df["dim2"] = shared_coords[:, 1]
    save_scatter(
        shared_df,
        x="dim1",
        y="dim2",
        hue="plot_target",
        style=None,
        size=None,
        title=f"{sampled_meta['audit_unit_id'].iloc[0]} {best_name} by target",
        output_path=output_dir / f"{sampled_meta['audit_unit_id'].iloc[0]}__{best_name}_by_target.png",
    )
    if shared_df["plot_nuisance"].nunique() > 1:
        save_scatter(
            shared_df,
            x="dim1",
            y="dim2",
            hue="plot_nuisance",
            style=None,
            size=None,
            title=f"{sampled_meta['audit_unit_id'].iloc[0]} {best_name} by nuisance",
            output_path=output_dir / f"{sampled_meta['audit_unit_id'].iloc[0]}__{best_name}_by_nuisance.png",
        )


def decision_from_scores(primary: float, nuisance: float, direct_primary: float) -> tuple[str, bool]:
    direct_better = not math.isnan(direct_primary) and (math.isnan(primary) or direct_primary > primary + 0.03)
    if math.isnan(primary):
        return "insufficient_metadata_to_decide", direct_better
    if primary >= 0.68 and (math.isnan(nuisance) or primary >= nuisance - 0.08):
        return "shared_embedding_usable", direct_better
    if primary >= 0.5:
        return "shared_embedding_marginal", direct_better
    return "shared_embedding_not_useful", direct_better


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    inventory_df = run_inventory()
    inventory_df.to_csv(output_dir / "shared_embedding_run_inventory.csv", index=False)
    write_markdown(
        output_dir / "shared_embedding_run_inventory.md",
        [
            "# Shared Embedding Run Inventory",
            "",
            f"- Output directory: `{output_dir}`",
            f"- Shared/global runs locally usable: {int(inventory_df['usable_for_dataset_audit'].sum())}",
            f"- Missing shared candidate: {'embedding_v6_within_type_gpu_run1' if not bool(inventory_df.loc[inventory_df['run_id']=='embedding_v6_within_type_gpu_run1', 'usable_for_dataset_audit'].any()) else 'none'}",
            "",
            table_text(inventory_df),
        ],
    )

    v7_meta, _ = read_run_metadata(Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v7_anchor_gpu_run1"))
    if v7_meta is None:
        raise FileNotFoundError("v7 shared metadata not found; cannot build audit inventory.")
    dataset_inventory_df, audit_units = build_dataset_inventory(v7_meta)
    dataset_inventory_df.to_csv(output_dir / "dataset_inventory.csv", index=False)
    write_markdown(
        output_dir / "dataset_inventory_report.md",
        [
            "# Dataset Inventory",
            "",
            f"- EV datasets represented: {int((dataset_inventory_df['sample_type'] == 'ev').sum())}",
            f"- Serum datasets represented: {int((dataset_inventory_df['sample_type'] == 'serum').sum())}",
            f"- Distinct audit units defined: {len(audit_units)}",
            "",
            table_text(dataset_inventory_df),
        ],
    )

    shared_runs: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for row in inventory_df.itertuples(index=False):
        if not row.usable_for_dataset_audit:
            continue
        meta, emb = read_run_metadata(Path(row.local_path))
        if meta is None or emb is None:
            continue
        shared_runs[str(row.run_id)] = (meta, emb)

    per_metrics_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    example_map_specs: list[tuple[pd.DataFrame, np.ndarray | None, str, np.ndarray]] = []

    for unit in audit_units:
        sampled_meta = sample_unit_metadata(
            v7_meta,
            unit,
            max_samples=args.max_samples_per_unit,
            seed=args.seed,
        )
        sampled_meta["audit_unit_id"] = unit.audit_unit_id
        if sampled_meta.empty:
            continue

        direct_values = None
        direct_primary = float("nan")
        try:
            direct_values = load_direct_baseline(sampled_meta)
            direct_metrics, direct_clusters = evaluate_representation(direct_values, sampled_meta, seed=args.seed)
            direct_metrics["run_id"] = "direct_baseline"
            direct_metrics["audit_unit_id"] = unit.audit_unit_id
            direct_metrics["dataset_id"] = unit.dataset_id
            direct_metrics["target_kind"] = unit.target_kind
            direct_metrics["sample_count"] = int(len(sampled_meta))
            direct_metrics["primary_signal_score"] = primary_composite(direct_metrics, unit.target_kind)
            direct_metrics["nuisance_signal_score"] = nuisance_composite(direct_metrics)
            direct_primary = float(direct_metrics["primary_signal_score"])
            per_metrics_rows.append(direct_metrics)
            cluster_rows.append(
                {
                    "audit_unit_id": unit.audit_unit_id,
                    "dataset_id": unit.dataset_id,
                    "run_id": "direct_baseline",
                    **{k: direct_metrics[k] for k in ["cluster_count", "cluster_purity_target", "cluster_entropy_target", "cluster_order_spearman"]},
                }
            )
        except Exception as exc:
            per_metrics_rows.append(
                {
                    "audit_unit_id": unit.audit_unit_id,
                    "dataset_id": unit.dataset_id,
                    "run_id": "direct_baseline",
                    "target_kind": unit.target_kind,
                    "sample_count": int(len(sampled_meta)),
                    "direct_error": str(exc),
                }
            )

        best_name = ""
        best_score = -1.0
        best_values = None
        for run_id, (run_meta, run_values) in shared_runs.items():
            aligned_meta, aligned_values = align_run_values(run_meta, run_values, sampled_meta["sample_key"].astype(str).tolist())
            aligned_meta = sampled_meta.merge(aligned_meta[["sample_key"]], on="sample_key", how="inner")
            metrics, _ = evaluate_representation(aligned_values, aligned_meta, seed=args.seed)
            metrics["run_id"] = run_id
            metrics["audit_unit_id"] = unit.audit_unit_id
            metrics["dataset_id"] = unit.dataset_id
            metrics["target_kind"] = unit.target_kind
            metrics["sample_count"] = int(len(aligned_meta))
            metrics["primary_signal_score"] = primary_composite(metrics, unit.target_kind)
            metrics["nuisance_signal_score"] = nuisance_composite(metrics)
            per_metrics_rows.append(metrics)
            cluster_rows.append(
                {
                    "audit_unit_id": unit.audit_unit_id,
                    "dataset_id": unit.dataset_id,
                    "run_id": run_id,
                    **{k: metrics[k] for k in ["cluster_count", "cluster_purity_target", "cluster_entropy_target", "cluster_order_spearman"]},
                }
            )
            current_score = metrics["primary_signal_score"] if not math.isnan(metrics["primary_signal_score"]) else -1.0
            if current_score > best_score:
                best_name = run_id
                best_score = current_score
                best_values = aligned_values

        if best_values is not None and unit.audit_unit_id in {"small2023_ev__mixture", "shine_ev_sers", "cca_hcc_lm_serum_sers"}:
            example_map_specs.append((sampled_meta.copy(), direct_values, best_name, best_values.copy()))

    per_metrics_df = pd.DataFrame(per_metrics_rows)
    cluster_df = pd.DataFrame(cluster_rows)
    per_metrics_df.to_csv(output_dir / "per_dataset_per_run_metrics.csv", index=False)
    cluster_df.to_csv(output_dir / "per_dataset_per_run_cluster_metrics.csv", index=False)

    decision_rows = []
    decided_dataset_ids: set[str] = set()
    summary_lines = ["# Per-Dataset Per-Run Summary", ""]
    for unit_id, group in per_metrics_df.groupby("audit_unit_id", sort=True):
        shared = group[group["run_id"] != "direct_baseline"].copy()
        direct = group[group["run_id"] == "direct_baseline"].copy()
        direct_primary = float(direct["primary_signal_score"].iloc[0]) if not direct.empty and "primary_signal_score" in direct.columns and pd.notna(direct["primary_signal_score"].iloc[0]) else float("nan")
        if shared.empty:
            continue
        shared = shared.sort_values("primary_signal_score", ascending=False, na_position="last")
        best = shared.iloc[0]
        status, direct_better = decision_from_scores(float(best["primary_signal_score"]), float(best["nuisance_signal_score"]) if "nuisance_signal_score" in best else float("nan"), direct_primary)
        reason = []
        reason.append(f"best shared run {best['run_id']} primary score {best['primary_signal_score']:.3f}")
        if not math.isnan(direct_primary):
            reason.append(f"direct baseline primary score {direct_primary:.3f}")
        if pd.notna(best.get("nuisance_signal_score")):
            reason.append(f"nuisance score {best['nuisance_signal_score']:.3f}")
        if direct_better:
            reason.append("direct/raw outperformed shared runs")
        consider_specific = status != "shared_embedding_usable" or direct_better
        decision_rows.append(
            {
                "audit_unit_id": unit_id,
                "dataset_id": best["dataset_id"],
                "target_kind": best["target_kind"],
                "decision": status,
                "best_shared_run_id": best["run_id"],
                "best_shared_primary_score": float(best["primary_signal_score"]),
                "best_shared_nuisance_score": float(best["nuisance_signal_score"]) if pd.notna(best.get("nuisance_signal_score")) else float("nan"),
                "direct_raw_primary_score": direct_primary,
                "direct_raw_outperformed_all_shared": bool(direct_better),
                "consider_dataset_specific_embedding_next": bool(consider_specific),
                "reason": "; ".join(reason),
            }
        )
        decided_dataset_ids.add(str(best["dataset_id"]))
        summary_lines.extend(
            [
                f"## {unit_id}",
                "",
                f"- Best shared run: `{best['run_id']}`",
                f"- Decision: `{status}`",
                f"- Reason: {'; '.join(reason)}",
                "",
            ]
        )

    for row in dataset_inventory_df.itertuples(index=False):
        if row.dataset_id in decided_dataset_ids:
            continue
        decision_rows.append(
            {
                "audit_unit_id": row.dataset_id,
                "dataset_id": row.dataset_id,
                "target_kind": "not_audited",
                "decision": "insufficient_metadata_to_decide",
                "best_shared_run_id": "",
                "best_shared_primary_score": float("nan"),
                "best_shared_nuisance_score": float("nan"),
                "direct_raw_primary_score": float("nan"),
                "direct_raw_outperformed_all_shared": False,
                "consider_dataset_specific_embedding_next": True,
                "reason": str(row.evaluation_notes),
            }
        )

    decision_df = pd.DataFrame(decision_rows).sort_values(["dataset_id", "audit_unit_id"])
    decision_df.to_csv(output_dir / "dataset_embedding_decision_table.csv", index=False)
    write_markdown(output_dir / "per_dataset_per_run_summary.md", summary_lines)
    write_markdown(
        output_dir / "dataset_embedding_decision_report.md",
        [
            "# Dataset Embedding Decision Report",
            "",
            f"- Audit units decided: {len(decision_df)}",
            f"- Shared usable: {int((decision_df['decision'] == 'shared_embedding_usable').sum()) if not decision_df.empty else 0}",
            f"- Shared marginal: {int((decision_df['decision'] == 'shared_embedding_marginal').sum()) if not decision_df.empty else 0}",
            f"- Shared not useful: {int((decision_df['decision'] == 'shared_embedding_not_useful').sum()) if not decision_df.empty else 0}",
            "",
            table_text(decision_df) if not decision_df.empty else "No decisions computed.",
        ],
    )

    grounding_theme_table = pd.read_csv("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_grounding_analysis_v7/grounding_theme_table.csv")
    grounding_df = grounding_inventory(grounding_theme_table)
    grounding_df.to_csv(output_dir / "grounding_inventory.csv", index=False)
    write_markdown(
        output_dir / "grounding_inventory_report.md",
        [
            "# Grounding Inventory",
            "",
            "- Pure molecular references are proposed as cross-sample-type resources.",
            "- Serum-specific grounding should later be gated to serum queries only.",
            "",
            table_text(grounding_df),
        ],
    )

    ontology_df = ontology_rows()
    ontology_df.to_csv(output_dir / "proposed_biochemical_ontology_v1.csv", index=False)
    write_markdown(
        output_dir / "proposed_biochemical_ontology_v1.md",
        [
            "# Proposed Biochemical Ontology v1",
            "",
            "- `purine` is placed under `nucleic_acid` for now, but remains context-dependent and should retain cross-links to small-molecule metabolite interpretations.",
            "- `oxidative_redox` should stop acting as a headline junk-drawer label. In later attribution it should be represented as narrower metabolite/process channels, not as an overpowered umbrella theme.",
            "- The longer-term visualization should expose these as continuous state vectors rather than flat winner-take-all badges.",
            "",
            table_text(ontology_df),
        ],
    )

    write_markdown(
        output_dir / "bsv_normalization_plan.md",
        [
            "# BSV Normalization Test Plan",
            "",
            "## Candidate Schemes",
            "",
            "- `raw_support`: keep raw support weights; useful as a no-normalization baseline.",
            "- `per_spectrum_sum_normalization`: divide each spectrum BSV vector by its total support; tests composition proportions.",
            "- `temperature_softmax_normalization`: sharpen or smooth support weights before aggregation; tests whether noisy tails should be suppressed.",
            "- `theme_library_size_correction`: correct for theme library cardinality so larger libraries do not dominate by count alone.",
            "- `within_dataset_z_score_normalization`: standardize BSV channels within a cohort before comparing classes; tests relative enrichment instead of absolute support.",
            "- `delta_bsv_normalization`: compare class-level or condition-level deltas rather than raw supports; likely important for serum cohort-mode analysis.",
            "",
            "## Later Decision Metrics",
            "",
            "- class separation improvement after BSV projection",
            "- stability across bootstrap resamples",
            "- cross-dataset comparability of BSV deltas",
            "- reduction of library-size bias",
            "- interpretability under known controls and ordered-condition datasets",
        ],
    )

    if not per_metrics_df.empty:
        heat = (
            per_metrics_df[per_metrics_df["run_id"] != "direct_baseline"][["audit_unit_id", "run_id", "primary_signal_score"]]
            .pivot(index="audit_unit_id", columns="run_id", values="primary_signal_score")
            .sort_index()
        )
        if not heat.empty:
            save_heatmap(heat, title="Shared embedding primary signal score by audit unit", output_path=figures_dir / "shared_run_comparison_heatmap.png", cmap="crest")
        scorecard = decision_df.copy()
        if not scorecard.empty:
            mapping = {
                "shared_embedding_usable": 2,
                "shared_embedding_marginal": 1,
                "shared_embedding_not_useful": 0,
                "insufficient_metadata_to_decide": -1,
            }
            scorecard["decision_numeric"] = scorecard["decision"].map(mapping)
            plt.figure(figsize=(10.5, max(4.5, 0.45 * len(scorecard))))
            ax = sns.barplot(data=scorecard, y="audit_unit_id", x="decision_numeric", hue="best_shared_run_id", palette="deep")
            ax.set_title("Dataset-by-dataset shared embedding decision scorecard")
            ax.set_xlabel("Decision score (-1 insufficient, 0 no, 1 marginal, 2 usable)")
            ax.set_ylabel("Audit unit")
            ax.legend(frameon=False, title="Best shared run")
            plt.tight_layout()
            plt.savefig(figures_dir / "dataset_decision_scorecard.png", dpi=220)
            plt.close()

    for sampled_meta, direct_values, best_name, best_values in example_map_specs:
        run_maps_for_unit(sampled_meta, direct_values, best_name, best_values, figures_dir, seed=args.seed)

    if not grounding_df.empty:
        save_barplot(
            grounding_df,
            x="grounding_type",
            y="n_records",
            hue="proposed_allowed_sample_type",
            title="Grounding inventory by dataset type",
            output_path=figures_dir / "grounding_inventory_counts.png",
        )


if __name__ == "__main__":
    main()
