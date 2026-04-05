from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from gaira.config import get_database_path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v1"

TARGET_DATASETS = {
    "small2023_ev",
    "shine_ev_sers",
    "diabetes_plasma_ev_sers",
    "cca_hcc_lm_serum_sers",
    "covid_serum_raman",
}

UNIVERSAL_GROUNDING = {
    "adenine_sers_control",
    "amino_acid_raman_grounding",
    "metabolite_sers63_support",
    "ramanbiolib",
}

SERUM_SUPPORT_GROUNDING = {
    "serum_ag_colloids_grounding",
}


@dataclass
class RegistryRow:
    dataset_id: str
    subset_id: str
    sample_type: str
    proposed_phase1_role: str
    allowed_future_roles: str
    keep_for_phase1: bool
    discard_reason: str
    likely_use_case: str
    expected_signal_type: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "subset_id": self.subset_id,
            "sample_type": self.sample_type,
            "proposed_phase1_role": self.proposed_phase1_role,
            "allowed_future_roles": self.allowed_future_roles,
            "keep_for_phase1": self.keep_for_phase1,
            "discard_reason": self.discard_reason,
            "likely_use_case": self.likely_use_case,
            "expected_signal_type": self.expected_signal_type,
            "notes": self.notes,
        }


def load_registry() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "registry" / "datasets.csv")


def load_subclass_context() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "context" / "subclass_domain_context_v1.csv"
    return pd.read_csv(path)


def load_dataset_stats() -> pd.DataFrame:
    con = duckdb.connect(str(get_database_path()), read_only=True)
    try:
        return con.execute(
            """
            select
              dataset_id,
              biosample_type as sample_type,
              count(*) as sample_count,
              count(distinct class_label) as n_class,
              count(distinct subclass_label) as n_subclass,
              string_agg(distinct subclass_label, ' | ' order by subclass_label) as subclass_labels,
              string_agg(distinct class_label, ' | ' order by class_label) as class_labels
            from biosample_metadata
            group by 1,2
            order by 1
            """
        ).fetchdf()
    finally:
        con.close()


def load_subset_stats() -> pd.DataFrame:
    con = duckdb.connect(str(get_database_path()), read_only=True)
    try:
        return con.execute(
            """
            select
              dataset_id,
              subclass_label,
              biosample_type as sample_type,
              count(*) as sample_count,
              count(distinct class_label) as n_class,
              string_agg(distinct class_label, ' | ' order by class_label) as class_labels
            from biosample_metadata
            where coalesce(subclass_label, '') <> ''
            group by 1,2,3
            order by 1,2
            """
        ).fetchdf()
    finally:
        con.close()


def stats_map(df: pd.DataFrame, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row.to_dict() for _, row in df.iterrows()}


def build_registry_rows(
    registry: pd.DataFrame,
    dataset_stats: dict[str, dict[str, Any]],
    subset_stats: pd.DataFrame,
    subclass_context: pd.DataFrame,
) -> list[RegistryRow]:
    rows: list[RegistryRow] = []
    subset_lookup = {
        (str(r["dataset_id"]), str(r["subclass_label"])): r.to_dict()
        for _, r in subset_stats.iterrows()
    }
    context_lookup = {
        (str(r["dataset_id"]), str(r["subclass_label"])): r.to_dict()
        for _, r in subclass_context.iterrows()
    }

    def dataset_note(dataset_id: str) -> str:
        info = dataset_stats.get(dataset_id, {})
        count = int(info.get("sample_count", 0))
        classes = int(info.get("n_class", 0))
        subclasses = int(info.get("n_subclass", 0))
        return f"{count} samples, {classes} class labels, {subclasses} subclass groups in canonical metadata."

    for _, record in registry.iterrows():
        dataset_id = str(record["dataset_id"])
        sample_type = str(record["sample_type"])
        notes = str(record.get("notes", ""))

        if dataset_id == "small2023_ev":
            rows.append(
                RegistryRow(
                    dataset_id=dataset_id,
                    subset_id="all",
                    sample_type=sample_type,
                    proposed_phase1_role="target_dataset_needs_subset_split",
                    allowed_future_roles="target_interpretation_dataset; validation_panel",
                    keep_for_phase1=True,
                    discard_reason="",
                    likely_use_case="Local raw/direct geometry for separate cell-line and mixture experiments, not one merged experiment path.",
                    expected_signal_type="Cell-line class structure and probe-family nuisance structure.",
                    notes=(
                        f"{dataset_note(dataset_id)} Phase 1 should treat Probe1 mixtures, Probe2 mixtures, and Fig3 cell-line archive as separate local-geometry targets."
                    ),
                )
            )
            for subset_id in ["normedprobe1", "normedprobe2", "fig3_norm_archive"]:
                info = subset_lookup.get((dataset_id, subset_id), {})
                ctx = context_lookup.get((dataset_id, subset_id), {})
                use_case = (
                    "Mixture-class target subset for direct/local geometry."
                    if subset_id.startswith("normedprobe")
                    else "Cell-line target subset for direct/local geometry."
                )
                rows.append(
                    RegistryRow(
                        dataset_id=dataset_id,
                        subset_id=subset_id,
                        sample_type=sample_type,
                        proposed_phase1_role="target_interpretation_subset",
                        allowed_future_roles="target_interpretation_dataset; validation_panel",
                        keep_for_phase1=True,
                        discard_reason="",
                        likely_use_case=use_case,
                        expected_signal_type="Probe-local class organization.",
                        notes=(
                            f"{int(info.get('sample_count', 0))} samples, {int(info.get('n_class', 0))} class labels. "
                            f"{ctx.get('notes', '')}"
                        ).strip(),
                    )
                )
            continue

        if dataset_id in {"shine_ev_sers", "diabetes_plasma_ev_sers", "cca_hcc_lm_serum_sers", "covid_serum_raman"}:
            rows.append(
                RegistryRow(
                    dataset_id=dataset_id,
                    subset_id="all",
                    sample_type=sample_type,
                    proposed_phase1_role="target_interpretation_dataset",
                    allowed_future_roles="target_interpretation_dataset; validation_panel",
                    keep_for_phase1=True,
                    discard_reason="",
                    likely_use_case="Primary Phase 1 target dataset using raw/direct local geometry.",
                    expected_signal_type=(
                        "Ordered dose/time or coarse cohort-state structure."
                        if dataset_id in {"shine_ev_sers", "diabetes_plasma_ev_sers"}
                        else "Cohort-level serum condition structure."
                    ),
                    notes=f"{dataset_note(dataset_id)} {notes}".strip(),
                )
            )
            continue

        if dataset_id == "serum_ag_colloids":
            rows.append(
                RegistryRow(
                    dataset_id=dataset_id,
                    subset_id="all",
                    sample_type=sample_type,
                    proposed_phase1_role="mixed_support_dataset_needs_subset_split",
                    allowed_future_roles="support_grounding_candidate; validation_panel",
                    keep_for_phase1=True,
                    discard_reason="",
                    likely_use_case="Support-only serum archive with healthy donors, controls, spiking, and enzyme-treatment experiments.",
                    expected_signal_type="Protocol/matrix support and controlled perturbation support.",
                    notes=f"{dataset_note(dataset_id)} Do not use the whole archive as one experiment path.",
                )
            )
            subset_specs = {
                "donors_serum_sers": ("validation_panel", "Healthy-donor serum reproducibility / local control panel."),
                "commercial_serum_merck": ("support_grounding_candidate", "Commercial serum background/support reference."),
                "spiked_commercial_serum_merck": ("support_grounding_candidate", "Serum-spiked controlled support library."),
                "uricase_serum_experiment": ("validation_panel", "Controlled serum perturbation panel for sanity checks."),
            }
            for subset_id, (role, use_case) in subset_specs.items():
                info = subset_lookup.get((dataset_id, subset_id), {})
                ctx = context_lookup.get((dataset_id, subset_id), {})
                rows.append(
                    RegistryRow(
                        dataset_id=dataset_id,
                        subset_id=subset_id,
                        sample_type=sample_type,
                        proposed_phase1_role=role,
                        allowed_future_roles="support_grounding_candidate; validation_panel",
                        keep_for_phase1=True,
                        discard_reason="",
                        likely_use_case=use_case,
                        expected_signal_type="Serum matrix/background or controlled perturbation support.",
                        notes=(
                            f"{int(info.get('sample_count', 0))} samples, {int(info.get('n_class', 0))} class labels. "
                            f"{ctx.get('notes', '')}"
                        ).strip(),
                    )
                )
            continue

        if dataset_id == "serum_protocol_comparison":
            info = dataset_stats.get(dataset_id, {})
            rows.append(
                RegistryRow(
                    dataset_id=dataset_id,
                    subset_id="protocol_comparison_archive",
                    sample_type=sample_type,
                    proposed_phase1_role="validation_panel",
                    allowed_future_roles="validation_panel",
                    keep_for_phase1=True,
                    discard_reason="",
                    likely_use_case="Protocol nuisance / acquisition-method validation panel only.",
                    expected_signal_type="Protocol-separable nuisance structure, not disease biology.",
                    notes=f"{dataset_note(dataset_id)} Single-serum protocol panel; reserve for invariance checks.",
                )
            )
            continue

        if dataset_id == "ergothioneine_serum":
            rows.append(
                RegistryRow(
                    dataset_id=dataset_id,
                    subset_id="ergothioneine_calibration_archive",
                    sample_type=sample_type,
                    proposed_phase1_role="validation_panel",
                    allowed_future_roles="validation_panel; support_grounding_candidate",
                    keep_for_phase1=True,
                    discard_reason="",
                    likely_use_case="Ordered concentration validation panel, not a disease target.",
                    expected_signal_type="Monotonic metabolite-spiking / concentration trend.",
                    notes=f"{dataset_note(dataset_id)} Useful for checking dose-order behavior in raw/direct geometry.",
                )
            )
            continue

        if dataset_id == "cspp_serum":
            rows.append(
                RegistryRow(
                    dataset_id=dataset_id,
                    subset_id="all",
                    sample_type=sample_type,
                    proposed_phase1_role="mixed_support_dataset_needs_manual_split",
                    allowed_future_roles="validation_panel; support_grounding_candidate",
                    keep_for_phase1=True,
                    discard_reason="",
                    likely_use_case="Method-development serum archive with mixed protocol, shelf-life, variability, and spiking families.",
                    expected_signal_type="Mostly method-factor structure plus one metabolite-spiking family.",
                    notes=f"{dataset_note(dataset_id)} Use only figure-level subsets, not the full archive as one panel.",
                )
            )
            subset_roles = {
                "figure2_processing_comparison": ("validation_panel", "Processing comparison panel."),
                "figure4_protocol_optimization": ("validation_panel", "Protocol optimization panel."),
                "figure5_strip_variability": ("validation_panel", "Substrate variability panel."),
                "figure6_shelf_life": ("validation_panel", "Shelf-life drift panel."),
                "figure7_metabolite_spiking": ("support_grounding_candidate_needs_manual_split", "Potential controlled spiking support after manual review."),
            }
            for subset_id, (role, use_case) in subset_roles.items():
                info = subset_lookup.get((dataset_id, subset_id), {})
                ctx = context_lookup.get((dataset_id, subset_id), {})
                rows.append(
                    RegistryRow(
                        dataset_id=dataset_id,
                        subset_id=subset_id,
                        sample_type=sample_type,
                        proposed_phase1_role=role,
                        allowed_future_roles="validation_panel; support_grounding_candidate",
                        keep_for_phase1=role != "support_grounding_candidate_needs_manual_split",
                        discard_reason="needs_manual_split" if "manual_split" in role else "",
                        likely_use_case=use_case,
                        expected_signal_type="Method-factor or controlled spiking structure.",
                        notes=(
                            f"{int(info.get('sample_count', 0))} samples, {int(info.get('n_class', 0))} class labels. "
                            f"{ctx.get('notes', '')}"
                        ).strip(),
                    )
                )
            continue

        if dataset_id in UNIVERSAL_GROUNDING:
            rows.append(
                RegistryRow(
                    dataset_id=dataset_id,
                    subset_id="all",
                    sample_type=sample_type,
                    proposed_phase1_role="grounding_reference_universal_pure",
                    allowed_future_roles="grounding_reference_universal_pure",
                    keep_for_phase1=True,
                    discard_reason="",
                    likely_use_case="Universal pure/reference grounding pool.",
                    expected_signal_type="Controlled analyte / pure biomolecule spectral motifs.",
                    notes=notes,
                )
            )
            continue

        if dataset_id in SERUM_SUPPORT_GROUNDING:
            rows.append(
                RegistryRow(
                    dataset_id=dataset_id,
                    subset_id="all",
                    sample_type=sample_type,
                    proposed_phase1_role="grounding_reference_serum_support",
                    allowed_future_roles="grounding_reference_serum_support",
                    keep_for_phase1=True,
                    discard_reason="",
                    likely_use_case="Serum-specific support grounding pool for serum targets only.",
                    expected_signal_type="Serum matrix-associated controlled support.",
                    notes=notes,
                )
            )
            continue

        if dataset_id in {
            "raman_knowledge_core",
            "serum_ag_colloids_literature_grounding",
            "sers_fingerprint_workingpaper_support",
            "sers24_metabolite_support",
        }:
            rows.append(
                RegistryRow(
                    dataset_id=dataset_id,
                    subset_id="all",
                    sample_type=sample_type,
                    proposed_phase1_role="defer_not_phase1_direct_grounding" if record["dataset_family"] == "grounding" else "defer_not_phase1_spectral_dataset",
                    allowed_future_roles="knowledge_support_only",
                    keep_for_phase1=False,
                    discard_reason="not_part_of_raw_direct_phase1",
                    likely_use_case="Future literature/context support, not direct Phase 1 geometry or grounding.",
                    expected_signal_type="Knowledge-only support.",
                    notes=notes,
                )
            )
            continue

        if dataset_id == "hcc_serum":
            rows.append(
                RegistryRow(
                    dataset_id=dataset_id,
                    subset_id="all",
                    sample_type=sample_type,
                    proposed_phase1_role="defer_or_review",
                    allowed_future_roles="holdout_validation_only",
                    keep_for_phase1=False,
                    discard_reason="holdout_or_manual_release_review",
                    likely_use_case="Holdout-only serum panel if explicitly approved later.",
                    expected_signal_type="Binary cohort classification.",
                    notes=notes,
                )
            )

    return rows


def build_grounding_map() -> pd.DataFrame:
    rows = [
        {
            "target_dataset_id": "small2023_ev",
            "sample_type": "extracellular vesicles",
            "use_universal_pure_grounding": True,
            "use_serum_support_grounding": False,
            "use_ev_support_grounding": False,
            "notes": "Use universal pure references only in Phase 1. EV-specific support grounding is currently sparse/absent.",
        },
        {
            "target_dataset_id": "shine_ev_sers",
            "sample_type": "extracellular vesicles",
            "use_universal_pure_grounding": True,
            "use_serum_support_grounding": False,
            "use_ev_support_grounding": False,
            "notes": "EV target should not borrow serum-support grounding in Phase 1.",
        },
        {
            "target_dataset_id": "diabetes_plasma_ev_sers",
            "sample_type": "extracellular vesicles",
            "use_universal_pure_grounding": True,
            "use_serum_support_grounding": False,
            "use_ev_support_grounding": False,
            "notes": "Universal pure references only until EV support grounding exists.",
        },
        {
            "target_dataset_id": "cca_hcc_lm_serum_sers",
            "sample_type": "serum",
            "use_universal_pure_grounding": True,
            "use_serum_support_grounding": True,
            "use_ev_support_grounding": False,
            "notes": "Serum target can use universal pure references plus serum-support grounding pools.",
        },
        {
            "target_dataset_id": "covid_serum_raman",
            "sample_type": "serum",
            "use_universal_pure_grounding": True,
            "use_serum_support_grounding": True,
            "use_ev_support_grounding": False,
            "notes": "Serum target can use serum-support grounding, but keep spontaneous Raman vs SERS caveat explicit.",
        },
    ]
    return pd.DataFrame(rows)


def write_summary(registry_df: pd.DataFrame, grounding_df: pd.DataFrame) -> None:
    targets = registry_df[registry_df["proposed_phase1_role"].isin(["target_interpretation_dataset", "target_interpretation_subset", "target_dataset_needs_subset_split"])]
    validation = registry_df[registry_df["proposed_phase1_role"].str.contains("validation_panel", na=False)]
    support = registry_df[registry_df["proposed_phase1_role"].str.contains("support_grounding_candidate|grounding_reference", na=False)]
    mixed = registry_df[registry_df["proposed_phase1_role"].str.contains("needs_subset_split|needs_manual_split", na=False)]

    lines = [
        "# GAIRAv2.0 Phase 1 Registry Audit",
        "",
        "This audit is raw/direct-only. It assigns Phase 1 dataset roles without using shared embeddings, branch embeddings, or RAG/context layers.",
        "",
        "## Targets",
    ]
    for _, row in targets.iterrows():
        lines.append(
            f"- `{row['dataset_id']}` / `{row['subset_id']}`: `{row['proposed_phase1_role']}`. {row['likely_use_case']}"
        )

    lines.extend(["", "## Validation-only For Now"])
    for _, row in validation.iterrows():
        lines.append(
            f"- `{row['dataset_id']}` / `{row['subset_id']}`: {row['likely_use_case']}"
        )

    lines.extend(["", "## Support-grounding Candidates For Now"])
    for _, row in support.iterrows():
        lines.append(
            f"- `{row['dataset_id']}` / `{row['subset_id']}`: {row['likely_use_case']}"
        )

    lines.extend(["", "## Mixed Datasets Requiring Split"])
    for _, row in mixed.iterrows():
        lines.append(
            f"- `{row['dataset_id']}` / `{row['subset_id']}`: {row['notes']}"
        )

    lines.extend(["", "## Allowed Grounding Pools For Targets"])
    for _, row in grounding_df.iterrows():
        lines.append(
            f"- `{row['target_dataset_id']}`: universal={row['use_universal_pure_grounding']}, serum_support={row['use_serum_support_grounding']}, ev_support={row['use_ev_support_grounding']}. {row['notes']}"
        )

    (OUTPUT_DIR / "phase1_registry_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    subclass_context = load_subclass_context()
    dataset_stats_df = load_dataset_stats()
    subset_stats_df = load_subset_stats()
    dataset_stats = stats_map(dataset_stats_df, "dataset_id")

    rows = build_registry_rows(registry, dataset_stats, subset_stats_df, subclass_context)
    registry_df = pd.DataFrame([row.as_dict() for row in rows]).sort_values(["dataset_id", "subset_id"]).reset_index(drop=True)
    grounding_df = build_grounding_map().sort_values("target_dataset_id").reset_index(drop=True)

    registry_df.to_csv(OUTPUT_DIR / "phase1_dataset_registry.csv", index=False)
    grounding_df.to_csv(OUTPUT_DIR / "phase1_target_grounding_map.csv", index=False)
    write_summary(registry_df, grounding_df)

    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
