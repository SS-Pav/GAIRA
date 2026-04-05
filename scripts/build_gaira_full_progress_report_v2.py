from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "gaira_full_progress_report_v2"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
MARKDOWN_PATH = REPORT_DIR / "GAIRA_full_progress_report_v2.md"
PDF_PATH = REPORT_DIR / "GAIRA_full_progress_report_v2.pdf"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "shared_decisions": pd.read_csv(ROOT / "reports" / "gaira_shared_embedding_audit_v1" / "dataset_embedding_decision_table.csv"),
        "shared_metrics": pd.read_csv(ROOT / "reports" / "gaira_shared_embedding_audit_v1" / "per_dataset_per_run_metrics.csv"),
        "phase1_v1": pd.read_csv(ROOT / "reports" / "gaira_phase1_registry_audit_v1" / "phase1_dataset_registry.csv"),
        "phase1_v2": pd.read_csv(ROOT / "reports" / "gaira_phase1_registry_audit_v2" / "phase1_dataset_registry_v2.csv"),
        "phase1_map_v2": pd.read_csv(ROOT / "reports" / "gaira_phase1_registry_audit_v2" / "phase1_target_grounding_map_v2.csv"),
        "phase1_exclusions": pd.read_csv(ROOT / "reports" / "gaira_phase1_registry_audit_v2" / "phase1_grounding_exclusions.csv"),
        "grounding_family": pd.read_csv(ROOT / "config" / "gaira_grounding_family_registry_v1.csv"),
        "target_family": pd.read_csv(ROOT / "config" / "gaira_target_family_registry_v1.csv"),
        "inference_lane": pd.read_csv(ROOT / "config" / "gaira_inference_lane_registry_v1.csv"),
        "representation_mode": pd.read_csv(ROOT / "config" / "gaira_representation_mode_registry_v1.csv"),
        "dataset_registry": pd.read_csv(ROOT / "config" / "gaira_dataset_experiment_registry_v1.csv"),
        "experiment_plan": pd.read_csv(ROOT / "reports" / "gaira_architecture_scaffold_v1" / "first_pass_experiment_plan.csv"),
        "raw_delta": pd.read_csv(ROOT / "reports" / "gaira_bsv_raw_pilot_v1" / "delta_bsv.csv"),
        "calibration_retrieval": pd.read_csv(ROOT / "reports" / "gaira_bsv_calibration_debug_v1" / "retrieval_audit_by_class.csv"),
        "calibration_axis": pd.read_csv(ROOT / "reports" / "gaira_bsv_calibration_debug_v1" / "ontology_axis_coverage_audit.csv"),
        "strict_comparison": pd.read_csv(ROOT / "reports" / "gaira_bsv_rerun_strict_grounding_v1" / "strict_grounding_comparison.csv"),
        "subclass_comparison": pd.read_csv(ROOT / "reports" / "gaira_bsv_metabolite_subclass_rerun_v1" / "strict_universal_vs_refined_comparison.csv"),
        "subclass_coverage": pd.read_csv(ROOT / "reports" / "gaira_bsv_metabolite_subclass_rerun_v1" / "subclass_coverage_audit.csv"),
        "targeted_comparison": pd.read_csv(ROOT / "reports" / "gaira_bsv_targeted_grounding_rerun_v1" / "targeted_grounding_comparison.csv"),
        "grounding_coverage": pd.read_csv(ROOT / "reports" / "gaira_grounding_coverage_expansion_audit_v1" / "compound_coverage_audit.csv"),
        "grounding_priority": pd.read_csv(ROOT / "reports" / "gaira_grounding_coverage_expansion_audit_v1" / "missing_reference_priority_list.csv"),
        "runner_cspp_config": pd.read_json(ROOT / "reports" / "gaira_experiment_runner_v1" / "exp_diff_cspp_metabolite_spike" / "run_config.json", typ="series").to_frame().T,
        "runner_uricase_compare": pd.read_csv(ROOT / "reports" / "gaira_experiment_runner_v1" / "exp_localdiff_serum_uricase" / "representation_mode_comparison.csv"),
        "runner_uricase_pairs": pd.read_csv(ROOT / "reports" / "gaira_experiment_runner_v1" / "exp_localdiff_serum_uricase" / "matched_background_pairs.csv"),
        "ont_v1": pd.read_csv(ROOT / "config" / "phase2_bsv_ontology_rules_v1.csv"),
        "ont_v2": pd.read_csv(ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv"),
    }


def build_summary_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    shared = data["shared_metrics"]
    shared_runs = shared[shared["run_id"].isin(["embedding_v5_full_true_gpu_run1", "embedding_v7_anchor_gpu_run1"])].copy()
    mean_primary = (
        shared_runs.groupby("run_id", as_index=False)["primary_signal_score"]
        .mean()
        .rename(columns={"primary_signal_score": "mean_primary_signal_score"})
    )
    tables["shared_run_summary"] = mean_primary

    tables["shared_embedding_dataset_decisions"] = data["shared_decisions"][
        ["audit_unit_id", "decision", "best_shared_run_id", "direct_raw_outperformed_all_shared", "reason"]
    ].copy()

    registry_summary = pd.concat(
        [
            data["grounding_family"].assign(registry_type="grounding_family"),
            data["target_family"].assign(registry_type="target_family"),
            data["inference_lane"].assign(registry_type="inference_lane"),
            data["representation_mode"].assign(registry_type="representation_mode"),
        ],
        ignore_index=True,
        sort=False,
    )
    tables["architecture_registry_summary"] = registry_summary

    chronology = pd.DataFrame(
        [
            ("shared_embedding_audit", "Audit existing shared GPU embeddings for within-dataset usefulness", "reports/gaira_shared_embedding_audit_v1", "Shared geometry was conditional, not a universal biology engine."),
            ("phase1_registry_v1", "Separate targets, grounding, and validation assets for raw/direct work", "reports/gaira_phase1_registry_audit_v1", "Raw/direct local-first framing replaced shared-embedding default."),
            ("phase1_registry_v2", "Refine subset aliases and exclusion logic", "reports/gaira_phase1_registry_audit_v2", "Phase 2 code could consume registry and exclusion logic directly."),
            ("raw_bsv_pilot", "Build deterministic BSV scaffold on direct spectra", "reports/gaira_bsv_raw_pilot_v1", "Pipeline worked, but validation chemistry was ambiguous."),
            ("calibration_debug", "Audit retrieval composition and ontology coverage", "reports/gaira_bsv_calibration_debug_v1", "Grounding composition, not top-K weighting, dominated failure."),
            ("strict_grounding_rerun", "Separate primary biochemical from caveat-only serum grounding", "reports/gaira_bsv_rerun_strict_grounding_v1", "Serum leakage was isolated, but chemistry remained weak."),
            ("metabolite_subclass_rerun", "Add minimal metabolite subclasses", "reports/gaira_bsv_metabolite_subclass_rerun_v1", "Subclassing improved specificity slightly, exposing missing coverage."),
            ("grounding_coverage_audit", "Check what explicit compounds were locally available", "reports/gaira_grounding_coverage_expansion_audit_v1", "Critical serum-specific references were local but not in the active primary pool."),
            ("targeted_grounding_rerun", "Try validation-targeted serum biochemical references", "reports/gaira_bsv_targeted_grounding_rerun_v1", "Signal collapsed to the UA/HSA neighborhood; similarity regime remained limiting."),
            ("architecture_scaffold", "Formalize v3 registries and experiment dimensions", "reports/gaira_architecture_scaffold_v1", "Grounding families, target families, lanes, and modes became explicit architecture."),
            ("experiment_runner", "Execute planned experiments through registry-driven runner", "reports/gaira_experiment_runner_v1/exp_diff_cspp_metabolite_spike", "Engineering path became reusable and honest about scientific weakness."),
            ("local_structural_embedding", "Add PCA-based nearest-background baseline selection", "reports/gaira_experiment_runner_v1/exp_localdiff_serum_uricase", "Useful as infrastructure, but not yet a chemistry-improving mode."),
        ],
        columns=["phase_id", "objective", "report_path", "what_changed"],
    )
    tables["chronology_summary"] = chronology

    rerun_rows = []
    for comparison, group_label in [("Erg-vs-Bkg", "Erg"), ("Hyp-vs-Bkg", "Hyp")]:
        raw_row = data["raw_delta"][data["raw_delta"]["comparison"] == comparison].iloc[0]
        rerun_rows.append(
            {
                "variant": "raw_full_pool",
                "comparison": comparison,
                "group_label": group_label,
                "small_molecule_metabolite": raw_row["small_molecule_metabolite"],
                "matrix_background": raw_row["matrix_background"],
                "protein_peptide": raw_row["protein_peptide"],
            }
        )
    strict_delta = data["strict_comparison"][data["strict_comparison"]["row_type"] == "delta"].copy()
    for variant in ["strict_universal_only", "strict_universal_plus_serum_caveat_only"]:
        sub = strict_delta[strict_delta["variant_name"] == variant]
        for _, row in sub.iterrows():
            rerun_rows.append(
                {
                    "variant": variant,
                    "comparison": row["comparison"],
                    "group_label": row["class_label"],
                    "small_molecule_metabolite": row["small_molecule_metabolite"],
                    "matrix_background": row["matrix_background"],
                    "protein_peptide": row["protein_peptide"],
                }
            )
    for _, row in data["subclass_comparison"].iterrows():
        if row["axis"] == "small_molecule_metabolite":
            rerun_rows.append(
                {
                    "variant": "refined_subclass_ontology",
                    "comparison": row["comparison"],
                    "group_label": row["comparison"].split("-vs-")[0],
                    "small_molecule_metabolite": row["refined_subclass_value"],
                    "matrix_background": np.nan,
                    "protein_peptide": np.nan,
                }
            )
    for _, row in data["targeted_comparison"].iterrows():
        if row["axis"] == "small_molecule_metabolite":
            rerun_rows.append(
                {
                    "variant": "validation_targeted_biochemical_extension",
                    "comparison": row["comparison"],
                    "group_label": row["comparison"].split("-vs-")[0],
                    "small_molecule_metabolite": row["targeted_biochemical_extension"],
                    "matrix_background": np.nan,
                    "protein_peptide": np.nan,
                }
            )
    tables["bsv_rerun_summary"] = pd.DataFrame(rerun_rows)

    cov = data["grounding_coverage"]
    interest = ["hypoxanthine", "ergothioneine", "xanthine", "uric_acid", "adenine", "guanine", "inosine"]
    tables["grounding_coverage_status"] = (
        cov[cov["compound_name"].isin(interest)][
            ["compound_name", "record_type", "dataset_id", "subset_id", "source_status", "metadata_n", "processed_spectra_n", "recommended_role"]
        ]
        .copy()
        .sort_values(["compound_name", "record_type", "dataset_id", "subset_id"])
    )

    current_state = pd.DataFrame(
        [
            ("Architecture", "solid", "Registries for families, lanes, modes, and datasets exist and are consumable."),
            ("Runner", "solid", "Registry-driven runner executes raw/direct absolute and differential experiments."),
            ("Local structural mode", "experimental", "PCA nearest-background selection is implemented but not yet chemistry-improving."),
            ("Shared embedding use", "conditional", "Useful for some within-dataset organization, not as the main biology engine."),
            ("BSV engine", "working_but_calibrating", "Deterministic retrieval/attribution works end to end."),
            ("Chemical discrimination", "unsolved", "Current similarity/grounding regime does not cleanly separate targeted serum metabolites."),
        ],
        columns=["system_component", "status", "detail"],
    )
    tables["current_state_summary"] = current_state

    for name, df in tables.items():
        df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    return tables


def plot_timeline(timeline: pd.DataFrame) -> Path:
    plt.figure(figsize=(12, 6.5))
    y = np.arange(len(timeline))
    plt.hlines(y, 0, 1, color="#4c6a92", linewidth=2.0)
    plt.scatter(np.full_like(y, 0.5, dtype=float), y, s=90, color="#1f3b5c")
    for i, row in timeline.iterrows():
        plt.text(0.52, i, f"{row['phase_id']}: {row['objective']}", va="center", fontsize=9)
    plt.xlim(0, 1.6)
    plt.yticks([])
    plt.title("GAIRA Evolution Chronology")
    plt.xlabel("Ordered phase progression")
    plt.tight_layout()
    out = FIG_DIR / "timeline_chronology.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_shared_audit(decisions: pd.DataFrame) -> Path:
    score_map = {
        "shared_embedding_not_useful": 0,
        "shared_embedding_marginal": 1,
        "shared_embedding_usable": 2,
        "insufficient_metadata_to_decide": -1,
    }
    work = decisions.copy()
    work["score"] = work["decision"].map(score_map)
    work = work.sort_values("score", ascending=False)
    plt.figure(figsize=(11, max(4, 0.55 * len(work))))
    ax = sns.barplot(data=work, y="audit_unit_id", x="score", hue="best_shared_run_id", dodge=False)
    ax.set_xticks([-1, 0, 1, 2], ["insufficient", "not useful", "marginal", "usable"])
    ax.set_xlabel("Shared embedding usability decision")
    ax.set_ylabel("")
    ax.set_title("Shared Embedding Audit by Dataset")
    for i, row in enumerate(work.itertuples(index=False)):
        if bool(getattr(row, "direct_raw_outperformed_all_shared")):
            ax.text(row.score + 0.03, i, "direct > shared", va="center", fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "shared_embedding_audit_summary.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_architecture_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    boxes = {
        "Grounding Families": (0.05, 0.65, 0.24, 0.22, "#d9e6f2"),
        "Target Families": (0.37, 0.65, 0.24, 0.22, "#dff0e0"),
        "Inference Lanes": (0.69, 0.65, 0.24, 0.22, "#f4e4cf"),
        "Representation Modes": (0.37, 0.28, 0.24, 0.22, "#efe1f7"),
    }
    box_text = {
        "Grounding Families": "universal_biochemical_grounding\n\ndomain_specific_biochemical_grounding\n\ndomain_specific_caveat_support_grounding",
        "Target Families": "interpretation_target\n\nvalidation_target",
        "Inference Lanes": "absolute_bsv\n\ndifferential_bsv",
        "Representation Modes": "raw_direct\n\nlocal_structural_embedding\n\nlocal_query_grounding_alignment (planned)",
    }
    for title, (x, y, w, h, color) in boxes.items():
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="#2a2a2a", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.78, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.42, box_text[title], ha="center", va="center", fontsize=10)
    arrows = [
        ((0.29, 0.76), (0.37, 0.76)),
        ((0.61, 0.76), (0.69, 0.76)),
        ((0.49, 0.65), (0.49, 0.50)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=16, linewidth=1.4, color="#444"))
    ax.text(0.49, 0.12, "GAIRA execution binds these dimensions at experiment runtime,\nnot by forcing one universal latent space.", ha="center", va="center", fontsize=11)
    out = FIG_DIR / "architecture_scaffold_diagram.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_differential_lane_schematic() -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")
    nodes = [
        ("Query spectrum", 0.08, 0.55),
        ("Representation mode\n(raw_direct or local structural)", 0.30, 0.55),
        ("Baseline selection\n(mean or nearest background)", 0.52, 0.55),
        ("Residual / comparator\nconstruction", 0.72, 0.55),
        ("dBSV against grounding", 0.90, 0.55),
    ]
    for text, x, y in nodes:
        box = FancyBboxPatch((x - 0.08, y - 0.12), 0.16, 0.24, boxstyle="round,pad=0.02", fc="#f5f6f7", ec="#333")
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=10)
    for a, b in zip(nodes[:-1], nodes[1:]):
        ax.add_patch(FancyArrowPatch((a[1] + 0.08, a[2]), (b[1] - 0.08, b[2]), arrowstyle="->", mutation_scale=14))
    ax.text(0.5, 0.12, "Representation organizes the query. Grounding still performs biochemical attribution.", ha="center", va="center", fontsize=11)
    out = FIG_DIR / "differential_lane_schematic.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_cspp_delta_evolution(raw_delta: pd.DataFrame, strict: pd.DataFrame, subclass: pd.DataFrame, targeted: pd.DataFrame) -> Path:
    rows = []
    for _, row in raw_delta.iterrows():
        rows.append(("raw_full_pool", row["comparison"], "small_molecule_metabolite", row["small_molecule_metabolite"]))
        rows.append(("raw_full_pool", row["comparison"], "matrix_background", row["matrix_background"]))
    strict_delta = strict[strict["row_type"] == "delta"].copy()
    for _, row in strict_delta.iterrows():
        rows.append((row["variant_name"], row["comparison"], "small_molecule_metabolite", row["small_molecule_metabolite"]))
        rows.append((row["variant_name"], row["comparison"], "matrix_background", row["matrix_background"]))
    for _, row in subclass.iterrows():
        if row["axis"] in {"small_molecule_metabolite", "purine_like_metabolite", "sulfur_containing_metabolite"}:
            rows.append(("refined_subclass_ontology", row["comparison"], row["axis"], row["refined_subclass_value"]))
    for _, row in targeted.iterrows():
        if row["axis"] in {"small_molecule_metabolite", "purine_like_metabolite", "sulfur_containing_metabolite"}:
            rows.append(("validation_targeted_extension", row["comparison"], row["axis"], row["targeted_biochemical_extension"]))
    df = pd.DataFrame(rows, columns=["variant", "comparison", "axis_name", "value"])
    order = [
        "raw_full_pool",
        "strict_universal_only",
        "strict_universal_plus_serum_caveat_only",
        "refined_subclass_ontology",
        "validation_targeted_extension",
    ]
    df["variant"] = pd.Categorical(df["variant"], categories=order, ordered=True)
    plt.figure(figsize=(13, 6.5))
    sns.barplot(data=df, x="variant", y="value", hue="axis_name")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Delta value")
    plt.xlabel("")
    plt.title("CSPP Metabolite-Spike Delta Evolution Across Reruns")
    plt.tight_layout()
    out = FIG_DIR / "cspp_delta_evolution.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_retrieval_composition(retrieval_audit: pd.DataFrame) -> Path:
    grouped = (
        retrieval_audit.groupby(["query_class_label", "source_bucket"], as_index=False)["support_weight_fraction"]
        .sum()
    )
    pivot = grouped.pivot(index="query_class_label", columns="source_bucket", values="support_weight_fraction").fillna(0.0)
    pivot.plot(kind="bar", stacked=True, figsize=(10, 5.5), color=["#6c8ebf", "#d79b00", "#b85450"])
    plt.ylabel("Support-weight fraction")
    plt.xlabel("")
    plt.title("CSPP Retrieval Composition During Calibration Debug")
    plt.legend(title="Source bucket", loc="upper right")
    plt.tight_layout()
    out = FIG_DIR / "cspp_retrieval_composition.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_grounding_coverage(coverage: pd.DataFrame) -> Path:
    interest = ["hypoxanthine", "ergothioneine", "xanthine", "uric_acid", "adenine", "guanine", "inosine"]
    work = coverage[coverage["compound_name"].isin(interest)].copy()
    status_order = [
        "active_universal_grounding",
        "local_serum_specific_grounding",
        "local_biosample_candidate",
        "not_confirmed_locally",
    ]
    agg = (
        work.groupby(["compound_name", "source_status"], as_index=False)["processed_spectra_n"]
        .max()
    )
    agg["source_status"] = pd.Categorical(agg["source_status"], categories=status_order, ordered=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=agg, x="compound_name", y="processed_spectra_n", hue="source_status")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Max processed spectra count")
    plt.xlabel("")
    plt.title("Grounding Coverage Status for Validation-Critical Compounds")
    plt.tight_layout()
    out = FIG_DIR / "grounding_coverage_status.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_uricase_comparison(compare: pd.DataFrame) -> Path:
    long_df = compare.melt(
        id_vars=["mode", "comparison", "group_label"],
        value_vars=["small_molecule_metabolite", "substrate_adsorption_bias", "matrix_background"],
        var_name="axis_name",
        value_name="delta_value",
    )
    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=long_df, x="comparison", y="delta_value", hue="mode")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Delta BSV")
    plt.xlabel("")
    plt.title("Uricase Validation: Raw Mean Background vs Local PCA Nearest Background")
    plt.tight_layout()
    out = FIG_DIR / "uricase_representation_mode_comparison.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def write_markdown(data: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame], figures: dict[str, Path]) -> str:
    mean_primary = tables["shared_run_summary"].set_index("run_id")["mean_primary_signal_score"].to_dict()
    decisions = data["shared_decisions"]
    direct_better = int(decisions["direct_raw_outperformed_all_shared"].fillna(False).sum())
    decision_counts = decisions["decision"].value_counts().to_dict()
    current_state = tables["current_state_summary"]
    phase1_v2 = data["phase1_v2"]
    validation_panels = phase1_v2[phase1_v2["proposed_phase1_role"] == "validation_panel"]["subset_alias"].astype(str).tolist()
    support_only = phase1_v2[phase1_v2["proposed_phase1_role"].isin(["grounding_reference_serum_support", "support_grounding_only_subset"])]["subset_alias"].astype(str).tolist()
    runner_compare = data["runner_uricase_compare"]
    uricase_metabolite = float(runner_compare["small_molecule_metabolite"].iloc[0])

    md = f"""# GAIRA Full Progress Report v2

## Section 1. Executive Technical Summary

GAIRA has moved from an initial shared-embedding-centric exploration into a modular biochemical interpretation architecture built around explicit registries for grounding families, target families, inference lanes, and representation modes. The engineering state is now materially stronger than the scientific state: the system can execute registry-driven experiments end to end, but chemical discrimination on the key serum validation panels remains the main unresolved bottleneck.

The early shared/global embedding phase established that global latent structure was not useless, but it was too conditional to serve as the core biology engine. The shared embedding audit showed a near tie between the strongest available shared runs, with mean primary within-dataset scores of `{mean_primary['embedding_v5_full_true_gpu_run1']:.4f}` for `v5` and `{mean_primary['embedding_v7_anchor_gpu_run1']:.4f}` for `v7`. More importantly, dataset-level outcomes were split: usable for some panels, marginal for others, and explicitly not useful for several core biology datasets. Direct/raw spectra outperformed all shared runs for `{direct_better}` audit units. That changed the interpretation of shared geometry from "default biology extractor" to "conditional organizer that may help some datasets but cannot anchor the full inference architecture."

That pivot led to the Phase 1 registry work and then to the GAIRAv3 architecture scaffold. The architecture now treats four dimensions independently: what evidence is allowed to vote (`grounding families`), what kind of target is being interpreted (`target families`), what question is being asked (`inference lanes`), and what local geometry tool is being used (`representation modes`). This distinction matters because local structure can help organize baselines and neighborhoods, but it cannot justify biochemical attribution on its own.

The deterministic BSV/dBSV work has been the main scientific path since that pivot. The first raw/direct BSV pilot on `cspp_metabolite_spike_validation` was operationally successful but scientifically ambiguous: `Erg-vs-Bkg` and `Hyp-vs-Bkg` moved strongly into `matrix_background` while `small_molecule_metabolite` decreased instead of increasing. The calibration debug pass showed that serum-support references dominated retrieval, especially UA/HSA-like rows, and that the main issue was grounding composition plus ontology overlap rather than weighting alone. Strict grounding-role separation improved interpretability by isolating serum-support leakage, but the core metabolite signal remained weak. Minimal metabolite subclassing helped a little, especially by exposing a small purine-like component for `Hyp`, but it also clarified that the universal grounding pool lacked explicit hypoxanthine and ergothioneine references.

The subsequent grounding coverage audit showed that those critical references were already available locally in serum-specific grounding, but not in the active universal biochemical pool. A targeted biochemical rerun still failed: once those serum-specific rows were added for the validation family, retrieval collapsed onto the UA/HSA neighborhood instead of producing chemically cleaner Hyp/Erg matches. That means the main unresolved problem is now narrower and clearer than before: not overall architecture confusion, not shared embedding misuse, and not simple ontology naming. The main bottleneck is the chemical discrimination regime itself, including similarity behavior and the interaction between preprocessing and serum-controlled reference structure.

The current runner state is solid. The registry-driven runner now executes `absolute_bsv + raw_direct`, `differential_bsv + raw_direct`, and a first `local_structural_embedding` differential path. The first local structural experiment on `serum_ag_uricase_validation` used PCA only for matched-background selection and kept biochemical attribution in the existing dBSV lane. That is the correct architectural separation. However, on this panel the PCA-nearest-background mode did not improve the grouped dBSV result relative to the mean-background residual baseline, so local structure is worth keeping as an infrastructure mode rather than a demonstrated chemistry-improving mechanism.

The most scientifically justified next step is not broader rollout. It is a bounded calibration harness focused on the chemistry-discrimination regime: controlled preprocessing comparisons, chemistry-aware similarity scoring, and narrow validation reruns against the strongest panels. Only if those deterministic calibration steps plateau should GAIRA consider later local query-grounding alignment encoders. Shared global embeddings and RAG/context layers remain explicitly out of scope for the current phase.

## Section 2. System Evolution Chronology

GAIRA evolved through a sequence of phases, each changing both the code structure and the scientific interpretation of what the system should be.

### Shared embedding audit phase
- Objective: determine whether existing shared GPU embedding passes were useful enough to serve as geometry backbones across datasets.
- Important files: `reports/gaira_shared_embedding_audit_v1/*`, especially `dataset_embedding_decision_table.csv`, `per_dataset_per_run_metrics.csv`, `shared_embedding_run_inventory.csv`.
- Result: shared embeddings were useful for some within-dataset structures but failed as a universal biology engine.
- Interpretation: the audit moved the project away from assuming that one global latent space could extract biology across EV and serum simultaneously.

### Specialized branch experiments
- Objective: test whether branch-specific embedding objectives could rescue small2023 and EV stress settings.
- Important outputs lived under the earlier v8 branch reports and smoke runs; these informed the later raw/direct pivot by showing how easily probe or protocol structure could dominate.
- Result: engineering viability was demonstrated, but probe nuisance and dataset mismatch remained major issues.
- Interpretation: geometry and inference needed to be separated more cleanly.

### Phase 1 registry work
- Objective: define raw/direct Phase 1 datasets, subset aliases, support-grounding pools, validation panels, and exclusion rules.
- Important files: `reports/gaira_phase1_registry_audit_v1/*`, `reports/gaira_phase1_registry_audit_v2/*`.
- Result: targets, validation panels, and grounding/support subsets became explicit and machine-readable.
- Interpretation: Phase 2 code could now consume a registry instead of relying on one-off dataset assumptions.

### BSV scaffold
- Objective: build a deterministic raw/direct biochemical support vector scaffold with hard exclusion enforcement.
- Important files: `reports/gaira_bsv_raw_pilot_v1/*`, `config/phase2_bsv_ontology_rules_v1.csv`.
- Result: the pipeline ran end to end on the CSPP metabolite spike panel.
- Interpretation: the system was operational, but the chemistry was not clean.

### Calibration debug
- Objective: find out whether the ambiguous first BSV result was caused by weighting, ontology, or grounding composition.
- Important files: `reports/gaira_bsv_calibration_debug_v1/*`.
- Result: serum-support references dominated retrieval; caveat-heavy axes were overrepresented.
- Interpretation: weighting was not the main issue. Grounding composition and ontology design were.

### Strict grounding rerun
- Objective: separate primary biochemical grounding from caveat-only serum support.
- Important files: `reports/gaira_bsv_rerun_strict_grounding_v1/*`.
- Result: serum-support leakage was isolated and matrix pathology dropped sharply in the universal-only variant.
- Interpretation: role separation was necessary, but not sufficient.

### Metabolite subclass rerun
- Objective: replace a coarse metabolite axis with a minimal evidence-backed subclass layer.
- Important files: `reports/gaira_bsv_metabolite_subclass_rerun_v1/*`, `config/phase2_bsv_ontology_rules_v2.csv`.
- Result: some subclass signal emerged, especially weak purine-like movement for `Hyp`.
- Interpretation: ontology coarseness was part of the problem, but missing explicit reference coverage was still more important.

### Grounding coverage expansion audit
- Objective: check whether missing hypoxanthine and ergothioneine references were actually absent or merely inactive.
- Important files: `reports/gaira_grounding_coverage_expansion_audit_v1/*`.
- Result: the critical labels were already present locally in serum-specific grounding.
- Interpretation: the next rerun did not require external acquisition, but it still required controlled use of serum-specific references.

### Targeted grounding rerun
- Objective: add only validation-targeted biochemical serum references without reintroducing broad serum-support leakage.
- Important files: `reports/gaira_bsv_targeted_grounding_rerun_v1/*`.
- Result: leakage stayed controlled, but the signal collapsed onto the UA/HSA neighborhood rather than improving Hyp/Erg specificity.
- Interpretation: the bottleneck had narrowed to similarity/preprocessing behavior and the chemical discriminability of the current direct-spectrum regime.

### Architecture scaffold
- Objective: formalize GAIRAv3 around grounding families, target families, inference lanes, and representation modes.
- Important files: `config/gaira_*_registry_v1.csv`, `reports/gaira_architecture_scaffold_v1/*`.
- Result: the architecture became explicit and machine-readable.
- Interpretation: experiment configuration was separated from retrieval implementation.

### Experiment runner build
- Objective: execute planned experiments through the registry stack instead of ad hoc scripts.
- Important files: `scripts/run_gaira_experiment.py`, `src/gaira/demo/gaira_experiment_runner_utils.py`, `reports/gaira_experiment_runner_v1/exp_diff_cspp_metabolite_spike`.
- Result: the runner reproduced the weak CSPP chemistry cleanly.
- Interpretation: the scaffold was trustworthy enough to treat future failures as scientific, not routing, failures.

### Local structural embedding differential experiment
- Objective: add a first local structural representation mode for matched-background selection.
- Important files: `reports/gaira_experiment_runner_v1/exp_localdiff_serum_uricase/*`.
- Result: PCA-based nearest-background matching was operational, but it did not change the grouped dBSV outcome relative to mean-background residual subtraction.
- Interpretation: local structural embeddings are worth keeping as a baseline-selection mode, but not yet as a chemistry-improving representation.

## Section 3. Shared Embedding Phase in Detail

The shared embedding phase should not be summarized as a simple failure. It answered a narrower question: can one of the existing shared GPU embedding passes serve as a useful within-dataset geometry backbone? The audit showed that the answer was "sometimes," but not consistently enough to make shared geometry the core inference engine.

The best shared runs were effectively tied overall:
- `embedding_v5_full_true_gpu_run1`: mean primary score `{mean_primary['embedding_v5_full_true_gpu_run1']:.4f}`
- `embedding_v7_anchor_gpu_run1`: mean primary score `{mean_primary['embedding_v7_anchor_gpu_run1']:.4f}`

Dataset decisions were evenly split:
- usable: `{decision_counts.get('shared_embedding_usable', 0)}`
- marginal: `{decision_counts.get('shared_embedding_marginal', 0)}`
- not useful: `{decision_counts.get('shared_embedding_not_useful', 0)}`
- insufficient metadata: `{decision_counts.get('insufficient_metadata_to_decide', 0)}`

The most important conclusion was conditionality:
- `small2023_ev__cellline` was usable with a shared run, but direct spectra were still slightly stronger.
- `small2023_ev__mixture` was only marginal and still carried nuisance concerns.
- `shine_ev_sers` and `cca_hcc_lm_serum_sers` were not useful as shared-geometry backbones.
- `covid_serum_raman` and `diabetes_plasma_ev_sers` were only marginal, with direct/raw outperforming all shared runs.
- Protocol-like serum panels such as `serum_protocol_comparison` and calibration-like panels such as `ergothioneine_serum` were usable, but those are not the same scientific problem as extracting disease biology from heterogeneous cohorts.

The interpretation changed because the audit separated two questions that had been blurred earlier:
1. Can shared geometry organize some datasets?
2. Can one shared latent space serve as the main biology extraction engine?

The answer to the first question was sometimes yes. The answer to the second was no. That is why shared geometry became a conditional organizer rather than the core inference mechanism.

Key summary table: `tables/shared_embedding_dataset_decisions.csv`

![Shared Embedding Audit](figures/shared_embedding_audit_summary.png)

The central lesson from this phase was not that shared embeddings were worthless. It was that global latent biology extraction across EV and serum was too ambitious relative to the actual signal structure and confounders present in the corpus.

## Section 4. Architecture Pivot

The architecture pivot happened because three distinct roles had been conflated:
- organizing target geometry
- defining what evidence is allowed to vote
- deciding what scientific question is being asked

The Phase 1 registry work and the later v3 scaffold separated those roles into four explicit dimensions:
- grounding families
- target families
- inference lanes
- representation modes

This separation was necessary because a dataset can be:
- a target in one experiment family
- a held-out validation panel in another
- completely forbidden as grounding in that same path

Likewise, a serum support subset can be useful for caveat modeling without being allowed to vote on primary biochemical axes. The architecture therefore moved away from "which dataset should be in the model?" toward "which evidence policy, question type, and representation mode are valid for this experiment?"

The Phase 1 v2 registry also made the system implementation-ready by:
- adding stable subset aliases such as `small2023_cellline`, `small2023_mixture_probe1`, and `cspp_metabolite_spike_validation`
- formalizing exclusion rules in `phase1_grounding_exclusions.csv`
- resolving mixed-support serum datasets into explicit validation-only and support-grounding-only subsets

The v3 scaffold then turned those decisions into registries that code could consume directly.

Key files:
- `config/gaira_grounding_family_registry_v1.csv`
- `config/gaira_target_family_registry_v1.csv`
- `config/gaira_inference_lane_registry_v1.csv`
- `config/gaira_representation_mode_registry_v1.csv`
- `config/gaira_dataset_experiment_registry_v1.csv`

![GAIRAv3 Architecture](figures/architecture_scaffold_diagram.png)

The main conceptual advance here is that local structural embeddings are no longer treated as biochemical attribution engines. They are geometry tools. Attribution comes from grounding. This is also why `differential_bsv` became a first-class lane: perturbation settings are often only interpretable after subtraction against a baseline.

## Section 5. BSV / dBSV Development

BSV was introduced to make biochemical interpretation deterministic, inspectable, and grounded in explicit retrieval rather than in a black-box classifier or a shared latent space. The core question of BSV is: what biochemical support is present in a spectrum or a group based on the currently allowed grounding evidence?

That by itself was not enough for perturbation panels. Absolute support can be dominated by shared matrix and scaffold effects, especially in serum SERS. That is why dBSV became necessary. Its question is different: what changed relative to a baseline, a matched comparator, or a local neighborhood control?

Ontology development tracked that shift. Ontology v1 defined broad Tier 1 axes:
- protein_peptide
- lipid_membrane
- nucleic_acid
- carbohydrate_glycan
- small_molecule_metabolite
- plus caveat axes such as matrix_background and substrate_adsorption_bias

Ontology v2 preserved those Tier 1 axes but added a minimal, evidence-backed subclass layer:
- purine_like_metabolite
- organic_acid_like
- aromatic_metabolite_like
- sulfur_containing_metabolite
- amino_acid_like_small_molecule

This change improved interpretability slightly, but it also exposed an important limit: subclass names cannot rescue missing explicit reference coverage.

Key ontology files:
- `config/phase2_bsv_ontology_rules_v1.csv`
- `config/phase2_bsv_ontology_rules_v2.csv`

Key rerun sequence and what changed:
- raw pilot: proved the engine worked, but chemistry was ambiguous
- calibration debug: showed serum-support dominance and caveat overcoverage
- strict grounding rerun: separated primary biochemical from caveat support
- subclass rerun: tested whether coarse metabolite labeling was the main issue
- targeted grounding rerun: tested whether adding explicit serum-specific references solved the validation question

Key summary table: `tables/bsv_rerun_summary.csv`

![Differential Lane Schematic](figures/differential_lane_schematic.png)

![CSPP Delta Evolution](figures/cspp_delta_evolution.png)

The main lesson from the BSV/dBSV development is that deterministic attribution was the right design move, but the chemistry bottleneck migrated over time:
1. first it looked like an architecture problem
2. then it looked like a grounding composition problem
3. then an ontology granularity problem
4. now it looks most like a similarity/preprocessing discrimination problem under constrained serum conditions

## Section 6. Validation Experiments in Detail

### A. CSPP Metabolite Spike Validation

This panel was chosen because it is the strongest current differential validation asset: a controlled serum perturbation panel with `Bkg`, `Erg`, and `Hyp` classes. It asks a clean question: when known metabolites are added in a controlled serum background, do the inferred biochemical deltas move in chemically plausible directions?

#### Raw pilot
The first raw/direct BSV pilot was operationally successful, but the chemistry was wrong in recognizable ways:
- `Erg-vs-Bkg`: `small_molecule_metabolite = -0.0665`, `matrix_background = +0.2274`
- `Hyp-vs-Bkg`: `small_molecule_metabolite = -0.6278`, `matrix_background = +0.5091`, `protein_peptide = +0.2348`

That was the first sign that the engine was detecting something reproducible but not something chemically clean.

#### Calibration debug
The debug pass audited retrieval composition and ontology coverage. It showed that retrieval was overwhelmingly dominated by serum-support references:
- `Bkg` was almost entirely pulled toward `serum_ag_colloids_grounding / UA+HSA`
- `Erg` was also dominated by serum-support hits, even when `Ergo` appeared among the top compounds
- `Hyp` was dominated by glycerol, serum support, and only weakly by `Hypox`

The ontology audit showed why this mattered: caveat-like axes were simply too prevalent in the active reference pool.

![CSPP Retrieval Composition](figures/cspp_retrieval_composition.png)

The key scientific shift after this phase was that weighting alone was no longer a plausible rescue. The problem was evidence composition.

#### Strict grounding rerun
This rerun imposed a hard split:
- universal pure grounding for primary biochemical axes
- serum-support grounding only for caveat axes

That fixed the worst matrix leakage. The `strict_universal_only` variant removed the large matrix-background deltas, but `small_molecule_metabolite` still remained slightly negative for both `Erg` and `Hyp`.

This was the crucial architecture result: the grounding-family distinction was necessary and valid. But it did not solve the chemistry.

#### Metabolite subclass rerun
The subclass rerun tested whether broad metabolite labeling was too coarse. It did help a little:
- `Erg-vs-Bkg` broad metabolite moved from `-0.019` to `+0.014`
- `Hyp-vs-Bkg` broad metabolite moved from `-0.095` to `-0.047`
- `Hyp` gained a small `purine_like_metabolite = +0.031`

But the result was still not chemically clean:
- `Erg` did not move positively into the sulfur subclass
- retrieval was still landing on chemically adjacent but wrong compounds

This made the next limitation explicit: coverage, not just ontology naming.

#### Grounding coverage expansion audit
The coverage audit showed that the critical compounds were not absent from local assets:
- `Hypox`, `Xanth`, `UA`, and `Ergo` already existed in `serum_ag_colloids_grounding`
- `ergothioneine_serum` also existed as a biosample calibration archive

But those references were not in the active universal biochemical pool.

#### Targeted grounding rerun
The targeted rerun added explicit serum-specific controlled biochemical references only for this validation family. This was the most telling result in the sequence because it failed in a different way:
- broad matrix leakage remained controlled
- but `Erg` and `Hyp` deltas collapsed toward zero instead of becoming more correct
- the top hits collapsed toward the UA/HSA neighborhood, especially `UAbound` and `UA+HSA`

That changed the interpretation again. The main failure was no longer simple missing labels. It was the behavior of the direct similarity/preprocessing regime in the presence of serum-controlled references.

The key takeaway from the full CSPP sequence is that the failure mode evolved from "architecture confusion" to "grounding/similarity mismatch." That is real scientific progress even though the validation is not yet successful.

### B. Serum Uricase Validation

This panel matters because it tests a different architectural question: can local structure help choose a better background for differential inference in a controlled serum perturbation setting?

The `exp_localdiff_serum_uricase` experiment implemented the first `local_structural_embedding` mode in the runner:
- PCA was computed on the panel alone
- each `+Enzyme` sample was matched to the nearest untreated sample from the corresponding base class
- residual spectra were constructed before dBSV attribution

This is the correct use of local structure in GAIRAv3: not as attribution, but as baseline selection.

However, the first result was negative in the scientific sense:
- `raw_direct + mean_background` and `local_structural_embedding + nearest_background` produced the same grouped dBSV values
- both treated groups showed `small_molecule_metabolite = 0.7407` and `substrate_adsorption_bias = 0.2593`
- neither mode showed matrix collapse, but neither gained interpretability from PCA matching

![Uricase Representation Comparison](figures/uricase_representation_mode_comparison.png)

This result is still useful. It shows that:
- the representation-mode separation is architecturally correct
- the runner can now execute local structure as a first-class mode
- but local PCA neighborhoods are not automatically the missing ingredient for chemical discrimination

The right interpretation is therefore restrained: keep `local_structural_embedding` as a first-class GAIRAv3 mode, but treat it as infrastructure rather than a proven scientific improvement.

## Section 7. Grounding Coverage and Chemical Bottlenecks

The grounding coverage audit clarified a subtle but important distinction:
- some references were missing from the active universal biochemical pool
- but they were not actually absent from the local corpus

This matters because it narrows the problem. The system is not blocked by complete absence of hypoxanthine- or ergothioneine-related local references. It is blocked by how those references are being used and by how the direct similarity regime behaves around them.

The critical compounds for the CSPP validation question currently look like this:
- explicit `hypoxanthine`: locally present in `serum_ag_colloids_grounding`, not present in active universal grounding
- explicit `ergothioneine`: locally present in `serum_ag_colloids_grounding`, not present in active universal grounding
- `xanthine` and `uric_acid` neighborhood controls: locally present in serum-specific grounding
- `inosine`: not confirmed locally

Key coverage table: `tables/grounding_coverage_status.csv`

![Grounding Coverage Status](figures/grounding_coverage_status.png)

The most important conclusion from this section is that the current bottleneck is not only coverage. The targeted grounding rerun showed that even when the missing serum-specific labels were introduced, the engine still collapsed onto the UA/HSA neighborhood. That implies a deeper interaction between:
- preprocessing / spectral representation
- cosine-style direct similarity
- serum-controlled reference structure

So the report should not conclude "just add more references." The more defensible conclusion is:
- add better calibration structure
- test chemistry-aware scoring
- use coverage expansion only as one part of that harness

## Section 8. Current State of GAIRA

### Architecture status
The architecture is now explicit, modular, and registry-driven. This is solid engineering progress.

### Runner status
The registry-driven runner works for:
- `absolute_bsv + raw_direct`
- `differential_bsv + raw_direct`
- `differential_bsv + local_structural_embedding` with a first PCA-based nearest-background scaffold

### Validation status
The system is operational on the highest-value current validation panels:
- `cspp_metabolite_spike_validation`
- `serum_ag_uricase_validation`

But "operational" does not equal "validated chemistry."

### Chemistry discrimination bottleneck
This remains the core unresolved issue.
- The strongest serum perturbation panel still does not map cleanly onto the expected biochemical subclasses.
- Adding targeted serum-specific references did not solve that.
- Local baseline matching did not solve that.

### What is clearly not solved
- clean Hyp/Erg discrimination under the current direct similarity regime
- broad rollout-ready serum biochemical attribution
- any claim that local structure or targeted grounding has already fixed the chemistry problem

### What not to do next
- do not broaden to many target datasets yet
- do not default back to shared embeddings as the interpretation engine
- do not add RAG/context layers before the deterministic chemistry path is trustworthy
- do not treat local PCA structure as biochemical evidence

Key current-state table: `tables/current_state_summary.csv`

## Section 9. Next Steps

The next phase should be a bounded calibration harness, not a broader deployment pass. The recommended order is:

1. bounded calibration harness
   - compare preprocessing variants directly on the strongest serum validation panels
   - quantify how similarity scoring behaves around explicit serum-controlled references
   - test chemistry-aware scoring beyond the current plain direct-spectrum cosine regime

2. chemistry-aware scoring
   - retain the deterministic retrieval architecture
   - evaluate whether simple spectrum-region weighting, residual-aware scaling, or other constrained scoring improves compound discrimination before introducing any learned alignment

3. only later, if deterministic calibration plateaus, consider local query-grounding alignment encoders
   - these should remain local and experiment-scoped
   - they should not become a new giant shared embedding program

What GAIRA is explicitly not doing yet:
- no global shared embedding revival as the core inference engine
- no RAG/context injection layer
- no broad target-dataset rollout beyond validation-calibrated use

The scientific reason for this order is straightforward: the system already knows how to run. What it does not yet know is how to discriminate the chemistry cleanly enough on the strongest controlled serum panels.

## Section 10. Appendix

### Key scripts
- `scripts/build_gaira_shared_embedding_audit.py`
- `scripts/build_gaira_phase1_registry_audit.py`
- `scripts/build_gaira_phase1_registry_audit_v2.py`
- `scripts/build_gaira_bsv_raw_pilot.py`
- `scripts/build_gaira_bsv_calibration_debug.py`
- `scripts/build_gaira_bsv_strict_grounding_rerun.py`
- `scripts/build_gaira_bsv_metabolite_subclass_rerun.py`
- `scripts/build_gaira_grounding_coverage_expansion_audit.py`
- `scripts/build_gaira_bsv_targeted_grounding_rerun.py`
- `scripts/run_gaira_experiment.py`

### Key report directories
- `reports/gaira_shared_embedding_audit_v1`
- `reports/gaira_phase1_registry_audit_v1`
- `reports/gaira_phase1_registry_audit_v2`
- `reports/gaira_bsv_raw_pilot_v1`
- `reports/gaira_bsv_calibration_debug_v1`
- `reports/gaira_bsv_rerun_strict_grounding_v1`
- `reports/gaira_bsv_metabolite_subclass_rerun_v1`
- `reports/gaira_grounding_coverage_expansion_audit_v1`
- `reports/gaira_bsv_targeted_grounding_rerun_v1`
- `reports/gaira_architecture_scaffold_v1`
- `reports/gaira_experiment_runner_v1/exp_diff_cspp_metabolite_spike`
- `reports/gaira_experiment_runner_v1/exp_localdiff_serum_uricase`

### Registry summary
- grounding families: `{len(data['grounding_family'])}`
- target families: `{len(data['target_family'])}`
- inference lanes: `{len(data['inference_lane'])}`
- representation modes: `{len(data['representation_mode'])}`
- dataset experiment registry rows: `{len(data['dataset_registry'])}`
- validation panels in current registry: `{len(validation_panels)}`
- serum-specific support-only subsets in current registry: `{len(support_only)}`

### Generated figures
- `figures/timeline_chronology.png`
- `figures/shared_embedding_audit_summary.png`
- `figures/architecture_scaffold_diagram.png`
- `figures/differential_lane_schematic.png`
- `figures/cspp_delta_evolution.png`
- `figures/cspp_retrieval_composition.png`
- `figures/grounding_coverage_status.png`
- `figures/uricase_representation_mode_comparison.png`

### Generated tables
- `tables/chronology_summary.csv`
- `tables/shared_run_summary.csv`
- `tables/shared_embedding_dataset_decisions.csv`
- `tables/architecture_registry_summary.csv`
- `tables/bsv_rerun_summary.csv`
- `tables/grounding_coverage_status.csv`
- `tables/current_state_summary.csv`
"""
    MARKDOWN_PATH.write_text(md)
    return md


def _wrap_markdown_for_pdf(markdown_text: str, width: int = 100) -> list[str]:
    lines: list[str] = []
    for raw_line in markdown_text.splitlines():
        if not raw_line.strip():
            lines.append("")
            continue
        if raw_line.startswith("#"):
            lines.append(raw_line)
            continue
        wrapped = textwrap.wrap(raw_line, width=width, break_long_words=False, replace_whitespace=False)
        lines.extend(wrapped if wrapped else [""])
    return lines


def render_pdf(markdown_text: str, figure_paths: list[Path]) -> None:
    wrapped_lines = _wrap_markdown_for_pdf(markdown_text)
    lines_per_page = 34
    pages = [wrapped_lines[i : i + lines_per_page] for i in range(0, len(wrapped_lines), lines_per_page)]
    with PdfPages(PDF_PATH) as pdf:
        for page_lines in pages:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            y = 0.96
            for line in page_lines:
                size = 12 if line.startswith("# ") else 10 if line.startswith("## ") else 8.8
                weight = "bold" if line.startswith("#") else "normal"
                fig.text(0.06, y, line, ha="left", va="top", fontsize=size, fontweight=weight, family="DejaVu Sans Mono")
                y -= 0.027 if line.startswith("#") else 0.024
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        for path in figure_paths:
            img = plt.imread(path)
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0.03, 0.08, 0.94, 0.86])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(path.name, fontsize=13)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")
    data = load_inputs()
    tables = build_summary_tables(data)
    figures = {
        "timeline": plot_timeline(tables["chronology_summary"]),
        "shared_audit": plot_shared_audit(data["shared_decisions"]),
        "architecture": plot_architecture_diagram(),
        "lane_schematic": plot_differential_lane_schematic(),
        "cspp_evolution": plot_cspp_delta_evolution(
            data["raw_delta"],
            data["strict_comparison"],
            data["subclass_comparison"],
            data["targeted_comparison"],
        ),
        "retrieval_composition": plot_retrieval_composition(data["calibration_retrieval"]),
        "grounding_coverage": plot_grounding_coverage(data["grounding_coverage"]),
        "uricase_compare": plot_uricase_comparison(data["runner_uricase_compare"]),
    }
    markdown_text = write_markdown(data, tables, figures)
    render_pdf(markdown_text, list(figures.values()))
    print(f"Wrote {MARKDOWN_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
