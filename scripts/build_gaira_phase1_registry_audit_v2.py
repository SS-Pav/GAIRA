from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from gaira.config import get_database_path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"

UNIVERSAL_PURE_GROUNDING = [
    "ramanbiolib",
    "adenine_sers_control",
    "amino_acid_raman_grounding",
    "metabolite_sers63_support",
]

SERUM_SUPPORT_GROUNDING = [
    "serum_ag_colloids_grounding",
    "serum_ag_colloids::commercial_serum_merck",
    "serum_ag_colloids::spiked_commercial_serum_merck",
]


@dataclass
class RegistryRow:
    dataset_id: str
    subset_id: str
    subset_alias: str
    sample_type: str
    proposed_phase1_role: str
    keep_for_phase1: bool
    allowed_future_roles: str
    grounding_exclusion_group: str
    likely_use_case: str
    expected_signal_type: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "subset_id": self.subset_id,
            "subset_alias": self.subset_alias,
            "sample_type": self.sample_type,
            "proposed_phase1_role": self.proposed_phase1_role,
            "keep_for_phase1": self.keep_for_phase1,
            "allowed_future_roles": self.allowed_future_roles,
            "grounding_exclusion_group": self.grounding_exclusion_group,
            "likely_use_case": self.likely_use_case,
            "expected_signal_type": self.expected_signal_type,
            "notes": self.notes,
        }


def load_registry() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "registry" / "datasets.csv")


def load_subclass_context() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "raw" / "context" / "subclass_domain_context_v1.csv")


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
              count(distinct subclass_label) as n_subclass
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


def make_lookup(df: pd.DataFrame, keys: list[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    for _, row in df.iterrows():
        out[tuple(str(row[k]) for k in keys)] = row.to_dict()
    return out


def describe_count(info: dict[str, Any]) -> str:
    return f"{int(info.get('sample_count', 0))} samples, {int(info.get('n_class', 0))} class labels."


def build_registry_rows(
    registry: pd.DataFrame,
    dataset_stats: pd.DataFrame,
    subset_stats: pd.DataFrame,
    subclass_context: pd.DataFrame,
) -> list[RegistryRow]:
    rows: list[RegistryRow] = []
    dataset_lookup = make_lookup(dataset_stats, ["dataset_id"])
    subset_lookup = make_lookup(subset_stats, ["dataset_id", "subclass_label"])
    context_lookup = make_lookup(subclass_context, ["dataset_id", "subclass_label"])

    def dataset_note(dataset_id: str) -> str:
        info = dataset_lookup.get((dataset_id,), {})
        return (
            f"{int(info.get('sample_count', 0))} samples, "
            f"{int(info.get('n_class', 0))} class labels, "
            f"{int(info.get('n_subclass', 0))} subclass groups in canonical metadata."
        )

    def subset_note(dataset_id: str, subset_id: str) -> str:
        info = subset_lookup.get((dataset_id, subset_id), {})
        ctx = context_lookup.get((dataset_id, subset_id), {})
        return f"{describe_count(info)} {ctx.get('notes', '')}".strip()

    registry_notes = {str(r["dataset_id"]): str(r.get("notes", "")) for _, r in registry.iterrows()}

    rows.extend(
        [
            RegistryRow(
                dataset_id="small2023_ev",
                subset_id="all",
                subset_alias="small2023_split_required",
                sample_type="extracellular vesicles",
                proposed_phase1_role="target_dataset_needs_subset_split",
                keep_for_phase1=True,
                allowed_future_roles="target_interpretation_subset; validation_panel",
                grounding_exclusion_group="target_only::small2023",
                likely_use_case="Container dataset only; use the aliased scientific subsets below in Phase 1 and Phase 2 code.",
                expected_signal_type="Cell-line and mixture structure separated by probe/archive family.",
                notes=f"{dataset_note('small2023_ev')} Do not treat the full dataset as one experiment path.",
            ),
            RegistryRow(
                dataset_id="small2023_ev",
                subset_id="fig3_norm_archive",
                subset_alias="small2023_cellline",
                sample_type="extracellular vesicles",
                proposed_phase1_role="target_interpretation_subset",
                keep_for_phase1=True,
                allowed_future_roles="target_interpretation_subset; validation_panel",
                grounding_exclusion_group="target_only::small2023_cellline",
                likely_use_case="Primary small-EV cell-line target using direct geometry only.",
                expected_signal_type="Five-way cell-line class organization.",
                notes=subset_note("small2023_ev", "fig3_norm_archive"),
            ),
            RegistryRow(
                dataset_id="small2023_ev",
                subset_id="normedprobe1",
                subset_alias="small2023_mixture_probe1",
                sample_type="extracellular vesicles",
                proposed_phase1_role="target_interpretation_subset",
                keep_for_phase1=True,
                allowed_future_roles="target_interpretation_subset; validation_panel",
                grounding_exclusion_group="target_only::small2023_mixture_probe1",
                likely_use_case="Primary small-EV mixture target on Probe1 geometry only.",
                expected_signal_type="Six-way mixture class organization within Probe1 family.",
                notes=subset_note("small2023_ev", "normedprobe1"),
            ),
            RegistryRow(
                dataset_id="small2023_ev",
                subset_id="normedprobe2",
                subset_alias="small2023_mixture_probe2",
                sample_type="extracellular vesicles",
                proposed_phase1_role="target_interpretation_subset",
                keep_for_phase1=True,
                allowed_future_roles="target_interpretation_subset; validation_panel",
                grounding_exclusion_group="target_only::small2023_mixture_probe2",
                likely_use_case="Primary small-EV mixture target on Probe2 geometry only.",
                expected_signal_type="Six-way mixture class organization within Probe2 family.",
                notes=subset_note("small2023_ev", "normedprobe2"),
            ),
            RegistryRow(
                dataset_id="shine_ev_sers",
                subset_id="all",
                subset_alias="shine_ev_stress",
                sample_type="extracellular vesicles",
                proposed_phase1_role="target_interpretation_dataset",
                keep_for_phase1=True,
                allowed_future_roles="target_interpretation_dataset; validation_panel",
                grounding_exclusion_group="target_only::shine_ev_stress",
                likely_use_case="Primary EV stress/time-course target.",
                expected_signal_type="Dose/day-related EV state structure.",
                notes=f"{dataset_note('shine_ev_sers')} {registry_notes['shine_ev_sers']}".strip(),
            ),
            RegistryRow(
                dataset_id="diabetes_plasma_ev_sers",
                subset_id="all",
                subset_alias="diabetes_ev_state",
                sample_type="extracellular vesicles",
                proposed_phase1_role="target_interpretation_dataset",
                keep_for_phase1=True,
                allowed_future_roles="target_interpretation_dataset; validation_panel",
                grounding_exclusion_group="target_only::diabetes_ev_state",
                likely_use_case="Primary EV metabolic-state target.",
                expected_signal_type="Two-group EV cohort state structure.",
                notes=f"{dataset_note('diabetes_plasma_ev_sers')} {registry_notes['diabetes_plasma_ev_sers']}".strip(),
            ),
            RegistryRow(
                dataset_id="cca_hcc_lm_serum_sers",
                subset_id="all",
                subset_alias="cca_hcc_lm_serum",
                sample_type="serum",
                proposed_phase1_role="target_interpretation_dataset",
                keep_for_phase1=True,
                allowed_future_roles="target_interpretation_dataset; validation_panel",
                grounding_exclusion_group="target_only::cca_hcc_lm_serum",
                likely_use_case="Primary serum cohort target.",
                expected_signal_type="Four-way cohort structure across CCA/HCC/LM/healthy.",
                notes=f"{dataset_note('cca_hcc_lm_serum_sers')} {registry_notes['cca_hcc_lm_serum_sers']}".strip(),
            ),
            RegistryRow(
                dataset_id="covid_serum_raman",
                subset_id="all",
                subset_alias="covid_serum_cohort",
                sample_type="serum",
                proposed_phase1_role="target_interpretation_dataset",
                keep_for_phase1=True,
                allowed_future_roles="target_interpretation_dataset; validation_panel",
                grounding_exclusion_group="target_only::covid_serum_cohort",
                likely_use_case="Primary serum Raman cohort target.",
                expected_signal_type="COVID/healthy/suspected/tube-control cohort structure.",
                notes=f"{dataset_note('covid_serum_raman')} {registry_notes['covid_serum_raman']}".strip(),
            ),
            RegistryRow(
                dataset_id="serum_ag_colloids",
                subset_id="all",
                subset_alias="serum_ag_colloids_split_required",
                sample_type="serum",
                proposed_phase1_role="mixed_support_dataset_needs_subset_split",
                keep_for_phase1=True,
                allowed_future_roles="support_grounding_only_subset; validation_panel",
                grounding_exclusion_group="manual_split::serum_ag_colloids",
                likely_use_case="Container dataset only; use explicit support and validation subsets below.",
                expected_signal_type="Serum matrix support and controlled perturbation families.",
                notes=f"{dataset_note('serum_ag_colloids')} Never use the full dataset as one pool.",
            ),
            RegistryRow(
                dataset_id="serum_ag_colloids",
                subset_id="commercial_serum_merck",
                subset_alias="serum_ag_background_support",
                sample_type="serum",
                proposed_phase1_role="support_grounding_only_subset",
                keep_for_phase1=True,
                allowed_future_roles="support_grounding_only_subset",
                grounding_exclusion_group="support_only::serum_support_pool",
                likely_use_case="Commercial serum background reference for serum-support grounding.",
                expected_signal_type="Serum matrix background.",
                notes=subset_note("serum_ag_colloids", "commercial_serum_merck"),
            ),
            RegistryRow(
                dataset_id="serum_ag_colloids",
                subset_id="spiked_commercial_serum_merck",
                subset_alias="serum_ag_spiked_support",
                sample_type="serum",
                proposed_phase1_role="support_grounding_only_subset",
                keep_for_phase1=True,
                allowed_future_roles="support_grounding_only_subset",
                grounding_exclusion_group="support_only::serum_support_pool",
                likely_use_case="Controlled serum-spiking support library for serum grounding only.",
                expected_signal_type="Known metabolite additions on a serum background.",
                notes=subset_note("serum_ag_colloids", "spiked_commercial_serum_merck"),
            ),
            RegistryRow(
                dataset_id="serum_ag_colloids",
                subset_id="donors_serum_sers",
                subset_alias="serum_ag_donor_validation",
                sample_type="serum",
                proposed_phase1_role="validation_panel",
                keep_for_phase1=True,
                allowed_future_roles="validation_panel",
                grounding_exclusion_group="validation_only::serum_ag_donor_validation",
                likely_use_case="Healthy-donor serum validation panel only.",
                expected_signal_type="Healthy-serum reproducibility / local consistency checks.",
                notes=subset_note("serum_ag_colloids", "donors_serum_sers"),
            ),
            RegistryRow(
                dataset_id="serum_ag_colloids",
                subset_id="uricase_serum_experiment",
                subset_alias="serum_ag_uricase_validation",
                sample_type="serum",
                proposed_phase1_role="validation_panel",
                keep_for_phase1=True,
                allowed_future_roles="validation_panel",
                grounding_exclusion_group="validation_only::serum_ag_uricase_validation",
                likely_use_case="Held-out serum perturbation validation panel only.",
                expected_signal_type="Before/after enzyme perturbation structure.",
                notes=subset_note("serum_ag_colloids", "uricase_serum_experiment"),
            ),
            RegistryRow(
                dataset_id="ergothioneine_serum",
                subset_id="ergothioneine_calibration_archive",
                subset_alias="serum_erg_calibration_validation",
                sample_type="serum",
                proposed_phase1_role="validation_panel",
                keep_for_phase1=True,
                allowed_future_roles="validation_panel; support_grounding_candidate_after_manual_review",
                grounding_exclusion_group="validation_only::serum_erg_calibration_validation",
                likely_use_case="Ordered concentration validation panel for the first pilot.",
                expected_signal_type="Monotonic ergothioneine concentration trend.",
                notes=f"{subset_note('ergothioneine_serum', 'ergothioneine_calibration_archive')} Keep validation-only in Phase 1 unless a later dedicated split is justified.",
            ),
            RegistryRow(
                dataset_id="serum_protocol_comparison",
                subset_id="protocol_comparison_archive",
                subset_alias="serum_protocol_validation",
                sample_type="serum",
                proposed_phase1_role="validation_panel",
                keep_for_phase1=True,
                allowed_future_roles="validation_panel",
                grounding_exclusion_group="validation_only::serum_protocol_validation",
                likely_use_case="Protocol nuisance validation panel only.",
                expected_signal_type="Protocol/day effects, not disease biology.",
                notes=subset_note("serum_protocol_comparison", "protocol_comparison_archive"),
            ),
            RegistryRow(
                dataset_id="cspp_serum",
                subset_id="all",
                subset_alias="cspp_split_required",
                sample_type="serum",
                proposed_phase1_role="mixed_support_dataset_needs_manual_split",
                keep_for_phase1=True,
                allowed_future_roles="validation_panel; support_grounding_candidate",
                grounding_exclusion_group="manual_split::cspp_serum",
                likely_use_case="Container dataset only; use figure-level subsets below.",
                expected_signal_type="Protocol, stability, variability, and spiking families.",
                notes=f"{dataset_note('cspp_serum')} Do not use the full archive directly in Phase 1 code.",
            ),
            RegistryRow(
                dataset_id="cspp_serum",
                subset_id="figure2_processing_comparison",
                subset_alias="cspp_processing_validation",
                sample_type="serum",
                proposed_phase1_role="validation_panel",
                keep_for_phase1=True,
                allowed_future_roles="validation_panel",
                grounding_exclusion_group="validation_only::cspp_processing_validation",
                likely_use_case="Processing-variation validation panel.",
                expected_signal_type="Processing-state nuisance structure.",
                notes=subset_note("cspp_serum", "figure2_processing_comparison"),
            ),
            RegistryRow(
                dataset_id="cspp_serum",
                subset_id="figure4_protocol_optimization",
                subset_alias="cspp_protocol_validation",
                sample_type="serum",
                proposed_phase1_role="validation_panel",
                keep_for_phase1=True,
                allowed_future_roles="validation_panel",
                grounding_exclusion_group="validation_only::cspp_protocol_validation",
                likely_use_case="Protocol optimization validation panel.",
                expected_signal_type="Method-factor structure.",
                notes=subset_note("cspp_serum", "figure4_protocol_optimization"),
            ),
            RegistryRow(
                dataset_id="cspp_serum",
                subset_id="figure5_strip_variability",
                subset_alias="cspp_strip_variability_validation",
                sample_type="serum",
                proposed_phase1_role="validation_panel",
                keep_for_phase1=True,
                allowed_future_roles="validation_panel",
                grounding_exclusion_group="validation_only::cspp_strip_variability_validation",
                likely_use_case="Strip variability validation panel.",
                expected_signal_type="Substrate variability structure.",
                notes=subset_note("cspp_serum", "figure5_strip_variability"),
            ),
            RegistryRow(
                dataset_id="cspp_serum",
                subset_id="figure6_shelf_life",
                subset_alias="cspp_shelf_life_validation",
                sample_type="serum",
                proposed_phase1_role="validation_panel",
                keep_for_phase1=True,
                allowed_future_roles="validation_panel",
                grounding_exclusion_group="validation_only::cspp_shelf_life_validation",
                likely_use_case="Shelf-life validation panel.",
                expected_signal_type="Time-since-preparation drift structure.",
                notes=subset_note("cspp_serum", "figure6_shelf_life"),
            ),
            RegistryRow(
                dataset_id="cspp_serum",
                subset_id="figure7_metabolite_spiking",
                subset_alias="cspp_metabolite_spike_validation",
                sample_type="serum",
                proposed_phase1_role="validation_panel",
                keep_for_phase1=True,
                allowed_future_roles="validation_panel; support_grounding_candidate_after_manual_reingestion",
                grounding_exclusion_group="validation_only::cspp_metabolite_spike_validation",
                likely_use_case="High-value held-out metabolite-spiking validation asset.",
                expected_signal_type="Three-way controlled serum perturbation (Bkg/Erg/Hyp).",
                notes=(
                    f"{subset_note('cspp_serum', 'figure7_metabolite_spiking')} "
                    "Phase 1 decision: keep as validation-only, not support grounding, because it is a clean held-out serum perturbation panel and using it in grounding would blur validation independence."
                ),
            ),
            RegistryRow(
                dataset_id="ramanbiolib",
                subset_id="all",
                subset_alias="ramanbiolib_universal_grounding",
                sample_type="mixed biomolecule library",
                proposed_phase1_role="grounding_reference_universal_pure",
                keep_for_phase1=True,
                allowed_future_roles="grounding_reference_universal_pure",
                grounding_exclusion_group="support_only::universal_pure_grounding",
                likely_use_case="Universal pure grounding pool.",
                expected_signal_type="Controlled biomolecule spectra.",
                notes=registry_notes["ramanbiolib"],
            ),
            RegistryRow(
                dataset_id="adenine_sers_control",
                subset_id="all",
                subset_alias="adenine_control_grounding",
                sample_type="controlled grounding references",
                proposed_phase1_role="grounding_reference_universal_pure",
                keep_for_phase1=True,
                allowed_future_roles="grounding_reference_universal_pure",
                grounding_exclusion_group="support_only::universal_pure_grounding",
                likely_use_case="Universal controlled adenine reference.",
                expected_signal_type="Controlled analyte series.",
                notes=registry_notes["adenine_sers_control"],
            ),
            RegistryRow(
                dataset_id="amino_acid_raman_grounding",
                subset_id="all",
                subset_alias="amino_acid_grounding",
                sample_type="controlled grounding references",
                proposed_phase1_role="grounding_reference_universal_pure",
                keep_for_phase1=True,
                allowed_future_roles="grounding_reference_universal_pure",
                grounding_exclusion_group="support_only::universal_pure_grounding",
                likely_use_case="Universal amino-acid reference grounding pool.",
                expected_signal_type="Pure amino-acid and comparator Raman signatures.",
                notes=registry_notes["amino_acid_raman_grounding"],
            ),
            RegistryRow(
                dataset_id="metabolite_sers63_support",
                subset_id="all",
                subset_alias="metabolite_fingerprint_grounding",
                sample_type="controlled grounding references",
                proposed_phase1_role="grounding_reference_universal_pure",
                keep_for_phase1=True,
                allowed_future_roles="grounding_reference_universal_pure",
                grounding_exclusion_group="support_only::universal_pure_grounding",
                likely_use_case="Universal small-molecule support grounding pool.",
                expected_signal_type="Reconstructed metabolite fingerprints.",
                notes=registry_notes["metabolite_sers63_support"],
            ),
            RegistryRow(
                dataset_id="serum_ag_colloids_grounding",
                subset_id="all",
                subset_alias="serum_support_grounding_curated",
                sample_type="controlled grounding references",
                proposed_phase1_role="grounding_reference_serum_support",
                keep_for_phase1=True,
                allowed_future_roles="grounding_reference_serum_support",
                grounding_exclusion_group="support_only::serum_support_pool",
                likely_use_case="Curated serum-support grounding pool for serum targets only.",
                expected_signal_type="Serum matrix-associated controlled support.",
                notes=registry_notes["serum_ag_colloids_grounding"],
            ),
            RegistryRow(
                dataset_id="raman_knowledge_core",
                subset_id="all",
                subset_alias="knowledge_core_deferred",
                sample_type="literature-derived knowledge",
                proposed_phase1_role="defer_not_phase1",
                keep_for_phase1=False,
                allowed_future_roles="knowledge_support_only",
                grounding_exclusion_group="exclude::phase1",
                likely_use_case="Deferred until post-Phase 1 contextual work.",
                expected_signal_type="Knowledge-only support.",
                notes=registry_notes["raman_knowledge_core"],
            ),
            RegistryRow(
                dataset_id="serum_ag_colloids_literature_grounding",
                subset_id="all",
                subset_alias="serum_ag_literature_deferred",
                sample_type="literature-backed grounding support",
                proposed_phase1_role="defer_not_phase1",
                keep_for_phase1=False,
                allowed_future_roles="knowledge_support_only",
                grounding_exclusion_group="exclude::phase1",
                likely_use_case="Deferred literature-only support.",
                expected_signal_type="Knowledge-only support.",
                notes=registry_notes["serum_ag_colloids_literature_grounding"],
            ),
            RegistryRow(
                dataset_id="sers_fingerprint_workingpaper_support",
                subset_id="all",
                subset_alias="sers_workingpaper_deferred",
                sample_type="literature-backed grounding support",
                proposed_phase1_role="defer_not_phase1",
                keep_for_phase1=False,
                allowed_future_roles="knowledge_support_only",
                grounding_exclusion_group="exclude::phase1",
                likely_use_case="Deferred paper-only support.",
                expected_signal_type="Knowledge-only support.",
                notes=registry_notes["sers_fingerprint_workingpaper_support"],
            ),
            RegistryRow(
                dataset_id="sers24_metabolite_support",
                subset_id="all",
                subset_alias="sers24_deferred",
                sample_type="literature-backed grounding support",
                proposed_phase1_role="defer_not_phase1",
                keep_for_phase1=False,
                allowed_future_roles="knowledge_support_only",
                grounding_exclusion_group="exclude::phase1",
                likely_use_case="Deferred paper-only support.",
                expected_signal_type="Knowledge-only support.",
                notes=registry_notes["sers24_metabolite_support"],
            ),
            RegistryRow(
                dataset_id="hcc_serum",
                subset_id="all",
                subset_alias="hcc_serum_holdout",
                sample_type="serum",
                proposed_phase1_role="defer_holdout",
                keep_for_phase1=False,
                allowed_future_roles="holdout_validation_only",
                grounding_exclusion_group="exclude::holdout",
                likely_use_case="Held-out serum dataset only if explicitly approved later.",
                expected_signal_type="Binary cohort structure.",
                notes=registry_notes["hcc_serum"],
            ),
        ]
    )

    return rows


def build_target_grounding_map() -> pd.DataFrame:
    rows = [
        {
            "target_dataset_id": "small2023_ev",
            "target_subset_id": "fig3_norm_archive",
            "target_alias": "small2023_cellline",
            "sample_type": "extracellular vesicles",
            "use_universal_pure_grounding": True,
            "use_serum_support_grounding": False,
            "use_ev_support_grounding": False,
            "allowed_grounding_sources": "; ".join(UNIVERSAL_PURE_GROUNDING),
            "forbidden_exclusion_groups": "target_only::small2023_cellline; validation_only::*; support_only::serum_support_pool",
            "notes": "EV cell-line target uses universal pure grounding only.",
        },
        {
            "target_dataset_id": "small2023_ev",
            "target_subset_id": "normedprobe1",
            "target_alias": "small2023_mixture_probe1",
            "sample_type": "extracellular vesicles",
            "use_universal_pure_grounding": True,
            "use_serum_support_grounding": False,
            "use_ev_support_grounding": False,
            "allowed_grounding_sources": "; ".join(UNIVERSAL_PURE_GROUNDING),
            "forbidden_exclusion_groups": "target_only::small2023_mixture_probe1; validation_only::*; support_only::serum_support_pool",
            "notes": "Mixture Probe1 target uses universal pure grounding only.",
        },
        {
            "target_dataset_id": "small2023_ev",
            "target_subset_id": "normedprobe2",
            "target_alias": "small2023_mixture_probe2",
            "sample_type": "extracellular vesicles",
            "use_universal_pure_grounding": True,
            "use_serum_support_grounding": False,
            "use_ev_support_grounding": False,
            "allowed_grounding_sources": "; ".join(UNIVERSAL_PURE_GROUNDING),
            "forbidden_exclusion_groups": "target_only::small2023_mixture_probe2; validation_only::*; support_only::serum_support_pool",
            "notes": "Mixture Probe2 target uses universal pure grounding only.",
        },
        {
            "target_dataset_id": "shine_ev_sers",
            "target_subset_id": "all",
            "target_alias": "shine_ev_stress",
            "sample_type": "extracellular vesicles",
            "use_universal_pure_grounding": True,
            "use_serum_support_grounding": False,
            "use_ev_support_grounding": False,
            "allowed_grounding_sources": "; ".join(UNIVERSAL_PURE_GROUNDING),
            "forbidden_exclusion_groups": "target_only::shine_ev_stress; validation_only::*; support_only::serum_support_pool",
            "notes": "EV target does not use serum-support grounding in Phase 1.",
        },
        {
            "target_dataset_id": "diabetes_plasma_ev_sers",
            "target_subset_id": "all",
            "target_alias": "diabetes_ev_state",
            "sample_type": "extracellular vesicles",
            "use_universal_pure_grounding": True,
            "use_serum_support_grounding": False,
            "use_ev_support_grounding": False,
            "allowed_grounding_sources": "; ".join(UNIVERSAL_PURE_GROUNDING),
            "forbidden_exclusion_groups": "target_only::diabetes_ev_state; validation_only::*; support_only::serum_support_pool",
            "notes": "EV target does not use serum-support grounding in Phase 1.",
        },
        {
            "target_dataset_id": "cca_hcc_lm_serum_sers",
            "target_subset_id": "all",
            "target_alias": "cca_hcc_lm_serum",
            "sample_type": "serum",
            "use_universal_pure_grounding": True,
            "use_serum_support_grounding": True,
            "use_ev_support_grounding": False,
            "allowed_grounding_sources": "; ".join(UNIVERSAL_PURE_GROUNDING + SERUM_SUPPORT_GROUNDING),
            "forbidden_exclusion_groups": "target_only::cca_hcc_lm_serum; validation_only::*",
            "notes": "Serum target may use curated serum-support pools but must exclude held-out validation panels.",
        },
        {
            "target_dataset_id": "covid_serum_raman",
            "target_subset_id": "all",
            "target_alias": "covid_serum_cohort",
            "sample_type": "serum",
            "use_universal_pure_grounding": True,
            "use_serum_support_grounding": True,
            "use_ev_support_grounding": False,
            "allowed_grounding_sources": "; ".join(UNIVERSAL_PURE_GROUNDING + SERUM_SUPPORT_GROUNDING),
            "forbidden_exclusion_groups": "target_only::covid_serum_cohort; validation_only::*",
            "notes": "Serum Raman target may use serum-support pools with Raman-vs-SERS caution.",
        },
    ]
    return pd.DataFrame(rows)


def build_exclusion_logic() -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(experiment_family: str, dataset_id: str, subset_id: str, reason: str) -> None:
        rows.append(
            {
                "experiment_family": experiment_family,
                "forbidden_grounding_dataset_id": dataset_id,
                "forbidden_grounding_subset_id": subset_id,
                "reason": reason,
            }
        )

    serum_validation_subsets = [
        ("serum_ag_colloids", "donors_serum_sers"),
        ("serum_ag_colloids", "uricase_serum_experiment"),
        ("ergothioneine_serum", "ergothioneine_calibration_archive"),
        ("serum_protocol_comparison", "protocol_comparison_archive"),
        ("cspp_serum", "figure2_processing_comparison"),
        ("cspp_serum", "figure4_protocol_optimization"),
        ("cspp_serum", "figure5_strip_variability"),
        ("cspp_serum", "figure6_shelf_life"),
        ("cspp_serum", "figure7_metabolite_spiking"),
    ]
    for ds, subset in serum_validation_subsets:
        add(
            "serum_primary_targets",
            ds,
            subset,
            "Held-out serum validation panels must not appear in serum support grounding for Phase 1 or the Phase 2 pilot.",
        )
        add(
            f"{ds}::{subset}",
            ds,
            subset,
            "A validation subset cannot be used inside its own grounding pool.",
        )

    small_targets = [
        ("small2023_ev", "fig3_norm_archive", "small2023_cellline"),
        ("small2023_ev", "normedprobe1", "small2023_mixture_probe1"),
        ("small2023_ev", "normedprobe2", "small2023_mixture_probe2"),
    ]
    for ds, subset, alias in small_targets:
        add(
            alias,
            ds,
            subset,
            "Target subset cannot serve as its own grounding source in the same experiment path.",
        )
        for serum_ds, serum_subset in serum_validation_subsets:
            add(
                alias,
                serum_ds,
                serum_subset,
                "EV experiments do not borrow serum validation panels as grounding.",
            )

    for alias in ["shine_ev_stress", "diabetes_ev_state"]:
        for serum_ds, serum_subset in serum_validation_subsets:
            add(
                alias,
                serum_ds,
                serum_subset,
                "EV experiments do not borrow serum validation panels as grounding.",
            )

    for alias in ["cca_hcc_lm_serum", "covid_serum_cohort"]:
        add(
            alias,
            "serum_ag_colloids",
            "donors_serum_sers",
            "Healthy donor validation subset stays held out from serum support grounding.",
        )
        add(
            alias,
            "serum_ag_colloids",
            "uricase_serum_experiment",
            "Uricase perturbation panel stays held out from serum support grounding.",
        )
        add(
            alias,
            "ergothioneine_serum",
            "ergothioneine_calibration_archive",
            "Ergothioneine calibration panel remains validation-only in Phase 1.",
        )
        add(
            alias,
            "cspp_serum",
            "figure7_metabolite_spiking",
            "CSPP figure7 is held out as validation-only, not support grounding.",
        )

    return pd.DataFrame(rows)


def write_summary(registry_df: pd.DataFrame) -> None:
    targets = registry_df[registry_df["proposed_phase1_role"].isin(["target_interpretation_dataset", "target_interpretation_subset"])]
    validation = registry_df[registry_df["proposed_phase1_role"] == "validation_panel"]
    support = registry_df[
        registry_df["proposed_phase1_role"].isin(
            ["support_grounding_only_subset", "grounding_reference_universal_pure", "grounding_reference_serum_support"]
        )
    ]

    lines = [
        "# GAIRAv2.0 Phase 1 Registry Audit v2",
        "",
        "This cleanup pass keeps the raw/direct-only Phase 1 framing and refines the registry for direct Phase 2 BSV code consumption.",
        "",
        "## Final Recommended Phase 1 Targets",
    ]
    for _, row in targets.iterrows():
        lines.append(f"- `{row['subset_alias']}` from `{row['dataset_id']}::{row['subset_id']}`: {row['likely_use_case']}")

    lines.extend(["", "## Final Recommended Validation Panels"])
    for _, row in validation.iterrows():
        lines.append(f"- `{row['subset_alias']}`: {row['likely_use_case']}")

    lines.extend(["", "## Final Recommended Support-grounding-only Subsets"])
    for _, row in support.iterrows():
        lines.append(f"- `{row['subset_alias']}`: {row['likely_use_case']}")

    lines.extend(
        [
            "",
            "## Exact Decision On `cspp_serum::figure7_metabolite_spiking`",
            "- Final decision: `validation_panel`.",
            "- Reason: the subset is balanced (`Bkg`, `Erg`, `Hyp`, 50 spectra each) and is more valuable as a held-out serum perturbation check than as a support-grounding source.",
            "- It is therefore excluded from Phase 1 serum support grounding and from the initial Phase 2 serum BSV grounding pool.",
            "",
            "## Exact `serum_ag_colloids` Split",
            "- `commercial_serum_merck` -> support-grounding-only",
            "- `spiked_commercial_serum_merck` -> support-grounding-only",
            "- `donors_serum_sers` -> validation-only",
            "- `uricase_serum_experiment` -> validation-only",
            "- Whole-dataset pooling remains forbidden.",
            "",
            "## `small2023_ev` Alias Mapping",
            "- `small2023_ev::fig3_norm_archive` -> `small2023_cellline`",
            "- `small2023_ev::normedprobe1` -> `small2023_mixture_probe1`",
            "- `small2023_ev::normedprobe2` -> `small2023_mixture_probe2`",
        ]
    )

    (OUTPUT_DIR / "phase1_registry_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    dataset_stats = load_dataset_stats()
    subset_stats = load_subset_stats()
    subclass_context = load_subclass_context()

    registry_rows = build_registry_rows(registry, dataset_stats, subset_stats, subclass_context)
    registry_df = pd.DataFrame([row.as_dict() for row in registry_rows]).sort_values(
        ["dataset_id", "subset_id"]
    ).reset_index(drop=True)
    grounding_map_df = build_target_grounding_map().sort_values(
        ["target_dataset_id", "target_subset_id"]
    ).reset_index(drop=True)
    exclusion_df = build_exclusion_logic().sort_values(
        ["experiment_family", "forbidden_grounding_dataset_id", "forbidden_grounding_subset_id"]
    ).reset_index(drop=True)

    registry_df.to_csv(OUTPUT_DIR / "phase1_dataset_registry_v2.csv", index=False)
    grounding_map_df.to_csv(OUTPUT_DIR / "phase1_target_grounding_map_v2.csv", index=False)
    exclusion_df.to_csv(OUTPUT_DIR / "phase1_grounding_exclusions.csv", index=False)
    write_summary(registry_df)

    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
