from __future__ import annotations

from pathlib import Path

from gaira.config import get_database_path, require_data_root_exists


DATA_ROOT = require_data_root_exists()["data_root"]
DB_PATH = get_database_path()

OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_evidence_operationalization_v1"
)
QA_ROOT = OUTPUT_ROOT / "qa"
TABLES_ROOT = OUTPUT_ROOT / "tables"
REPORT_ROOT = OUTPUT_ROOT / "report"

REFINEMENT_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_evidence_phase1_refinement_v1"
)
REFINEMENT_QA_ROOT = REFINEMENT_OUTPUT_ROOT / "qa"
REFINEMENT_TABLES_ROOT = REFINEMENT_OUTPUT_ROOT / "tables"
REFINEMENT_REPORT_ROOT = REFINEMENT_OUTPUT_ROOT / "report"

CLEANUP_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_evidence_phase1_cleanup_audit_v1"
)
CLEANUP_QA_ROOT = CLEANUP_OUTPUT_ROOT / "qa"
CLEANUP_TABLES_ROOT = CLEANUP_OUTPUT_ROOT / "tables"
CLEANUP_REPORT_ROOT = CLEANUP_OUTPUT_ROOT / "report"

PATTERN_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_evidence_assignment_patterns_v1"
)
PATTERN_QA_ROOT = PATTERN_OUTPUT_ROOT / "qa"
PATTERN_TABLES_ROOT = PATTERN_OUTPUT_ROOT / "tables"
PATTERN_REPORT_ROOT = PATTERN_OUTPUT_ROOT / "report"

SOURCE_AUDIT_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_evidence_source_composition_audit_v1"
)
SOURCE_AUDIT_QA_ROOT = SOURCE_AUDIT_OUTPUT_ROOT / "qa"
SOURCE_AUDIT_TABLES_ROOT = SOURCE_AUDIT_OUTPUT_ROOT / "tables"
SOURCE_AUDIT_REPORT_ROOT = SOURCE_AUDIT_OUTPUT_ROOT / "report"

WAREHOUSE_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_evidence_warehouse_grounding_backbone_v1"
)
WAREHOUSE_QA_ROOT = WAREHOUSE_OUTPUT_ROOT / "qa"
WAREHOUSE_TABLES_ROOT = WAREHOUSE_OUTPUT_ROOT / "tables"
WAREHOUSE_REPORT_ROOT = WAREHOUSE_OUTPUT_ROOT / "report"

PATTERN_REFINEMENT_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_evidence_pattern_granularity_refinement_v1"
)
PATTERN_REFINEMENT_QA_ROOT = PATTERN_REFINEMENT_OUTPUT_ROOT / "qa"
PATTERN_REFINEMENT_TABLES_ROOT = PATTERN_REFINEMENT_OUTPUT_ROOT / "tables"
PATTERN_REFINEMENT_REPORT_ROOT = PATTERN_REFINEMENT_OUTPUT_ROOT / "report"

LITERATURE_PILOT_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_literature_three_paper_pilot_v1"
)
LITERATURE_PILOT_QA_ROOT = LITERATURE_PILOT_OUTPUT_ROOT / "qa"
LITERATURE_PILOT_TABLES_ROOT = LITERATURE_PILOT_OUTPUT_ROOT / "tables"
LITERATURE_PILOT_REPORT_ROOT = LITERATURE_PILOT_OUTPUT_ROOT / "report"

ONTOLOGY_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_evidence_ontology_expansion_v1"
)
ONTOLOGY_QA_ROOT = ONTOLOGY_OUTPUT_ROOT / "qa"
ONTOLOGY_TABLES_ROOT = ONTOLOGY_OUTPUT_ROOT / "tables"
ONTOLOGY_REPORT_ROOT = ONTOLOGY_OUTPUT_ROOT / "report"

ONTOLOGY_MOTIF_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_ontology_aligned_motif_summary_v1"
)
ONTOLOGY_MOTIF_QA_ROOT = ONTOLOGY_MOTIF_OUTPUT_ROOT / "qa"
ONTOLOGY_MOTIF_TABLES_ROOT = ONTOLOGY_MOTIF_OUTPUT_ROOT / "tables"
ONTOLOGY_MOTIF_REPORT_ROOT = ONTOLOGY_MOTIF_OUTPUT_ROOT / "report"

LOCAL_NEIGHBORHOOD_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_local_support_neighborhood_refinement_v1"
)
LOCAL_NEIGHBORHOOD_QA_ROOT = LOCAL_NEIGHBORHOOD_OUTPUT_ROOT / "qa"
LOCAL_NEIGHBORHOOD_TABLES_ROOT = LOCAL_NEIGHBORHOOD_OUTPUT_ROOT / "tables"
LOCAL_NEIGHBORHOOD_REPORT_ROOT = LOCAL_NEIGHBORHOOD_OUTPUT_ROOT / "report"

PAPER_QC_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_paper_evidence_qc_v1"
)
PAPER_QC_QA_ROOT = PAPER_QC_OUTPUT_ROOT / "qa"
PAPER_QC_TABLES_ROOT = PAPER_QC_OUTPUT_ROOT / "tables"
PAPER_QC_REPORT_ROOT = PAPER_QC_OUTPUT_ROOT / "report"

ENRICHMENT_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_paper_targeted_enrichment_v1"
)
ENRICHMENT_QA_ROOT = ENRICHMENT_OUTPUT_ROOT / "qa"
ENRICHMENT_TABLES_ROOT = ENRICHMENT_OUTPUT_ROOT / "tables"
ENRICHMENT_REPORT_ROOT = ENRICHMENT_OUTPUT_ROOT / "report"

REMAINING_PAPER_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_remaining_paper_controlled_ingest_v1"
)
REMAINING_PAPER_QA_ROOT = REMAINING_PAPER_OUTPUT_ROOT / "qa"
REMAINING_PAPER_TABLES_ROOT = REMAINING_PAPER_OUTPUT_ROOT / "tables"
REMAINING_PAPER_REPORT_ROOT = REMAINING_PAPER_OUTPUT_ROOT / "report"

CONDITION_LAYER_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_condition_ontology_layer_v1"
)
CONDITION_LAYER_QA_ROOT = CONDITION_LAYER_OUTPUT_ROOT / "qa"
CONDITION_LAYER_TABLES_ROOT = CONDITION_LAYER_OUTPUT_ROOT / "tables"
CONDITION_LAYER_REPORT_ROOT = CONDITION_LAYER_OUTPUT_ROOT / "report"

LITERATURE_PIPELINE_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_literature_acquisition_pipeline_v1"
)
LITERATURE_PIPELINE_QA_ROOT = LITERATURE_PIPELINE_OUTPUT_ROOT / "qa"
LITERATURE_PIPELINE_TABLES_ROOT = LITERATURE_PIPELINE_OUTPUT_ROOT / "tables"
LITERATURE_PIPELINE_REPORT_ROOT = LITERATURE_PIPELINE_OUTPUT_ROOT / "report"

LITERATURE_PIPELINE_ASSET_ROOT = (
    DATA_ROOT
    / "raw"
    / "gaira_literature_acquisition_pipeline_v1"
)

LITERATURE_ASSET_RESOLUTION_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_literature_asset_resolution_v1"
)
LITERATURE_ASSET_RESOLUTION_QA_ROOT = LITERATURE_ASSET_RESOLUTION_OUTPUT_ROOT / "qa"
LITERATURE_ASSET_RESOLUTION_TABLES_ROOT = LITERATURE_ASSET_RESOLUTION_OUTPUT_ROOT / "tables"
LITERATURE_ASSET_RESOLUTION_REPORT_ROOT = LITERATURE_ASSET_RESOLUTION_OUTPUT_ROOT / "report"
LITERATURE_ASSET_RESOLUTION_ASSET_ROOT = (
    DATA_ROOT
    / "raw"
    / "gaira_literature_asset_resolution_v1"
)

LITERATURE_BLOCKED_ASSET_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_literature_blocked_asset_registry_v1"
)
LITERATURE_BLOCKED_ASSET_QA_ROOT = LITERATURE_BLOCKED_ASSET_OUTPUT_ROOT / "qa"
LITERATURE_BLOCKED_ASSET_TABLES_ROOT = LITERATURE_BLOCKED_ASSET_OUTPUT_ROOT / "tables"
LITERATURE_BLOCKED_ASSET_REPORT_ROOT = LITERATURE_BLOCKED_ASSET_OUTPUT_ROOT / "report"

READY_PAPER_INGEST_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_ready_paper_controlled_ingest_v1"
)
READY_PAPER_INGEST_QA_ROOT = READY_PAPER_INGEST_OUTPUT_ROOT / "qa"
READY_PAPER_INGEST_TABLES_ROOT = READY_PAPER_INGEST_OUTPUT_ROOT / "tables"
READY_PAPER_INGEST_REPORT_ROOT = READY_PAPER_INGEST_OUTPUT_ROOT / "report"

ASSET_TRUTH_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_asset_truth_oa_validation_v1"
)
ASSET_TRUTH_QA_ROOT = ASSET_TRUTH_OUTPUT_ROOT / "qa"
ASSET_TRUTH_TABLES_ROOT = ASSET_TRUTH_OUTPUT_ROOT / "tables"
ASSET_TRUTH_REPORT_ROOT = ASSET_TRUTH_OUTPUT_ROOT / "report"
ASSET_TRUTH_ASSET_ROOT = (
    DATA_ROOT
    / "raw"
    / "gaira_literature_oa_first_v1"
)

OA_READY_INGEST_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_oa_ready_controlled_ingest_v1"
)
OA_READY_INGEST_QA_ROOT = OA_READY_INGEST_OUTPUT_ROOT / "qa"
OA_READY_INGEST_TABLES_ROOT = OA_READY_INGEST_OUTPUT_ROOT / "tables"
OA_READY_INGEST_REPORT_ROOT = OA_READY_INGEST_OUTPUT_ROOT / "report"

OA_TEXT_FIRST_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_oa_text_first_expansion_v1"
)
OA_TEXT_FIRST_QA_ROOT = OA_TEXT_FIRST_OUTPUT_ROOT / "qa"
OA_TEXT_FIRST_TABLES_ROOT = OA_TEXT_FIRST_OUTPUT_ROOT / "tables"
OA_TEXT_FIRST_REPORT_ROOT = OA_TEXT_FIRST_OUTPUT_ROOT / "report"
OA_TEXT_FIRST_ASSET_ROOT = (
    DATA_ROOT
    / "raw"
    / "gaira_literature_oa_text_first_v1"
)

OA_TEXT_FOLLOWUP_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_oa_text_followup_upgrade_v1"
)
OA_TEXT_FOLLOWUP_QA_ROOT = OA_TEXT_FOLLOWUP_OUTPUT_ROOT / "qa"
OA_TEXT_FOLLOWUP_TABLES_ROOT = OA_TEXT_FOLLOWUP_OUTPUT_ROOT / "tables"
OA_TEXT_FOLLOWUP_REPORT_ROOT = OA_TEXT_FOLLOWUP_OUTPUT_ROOT / "report"

OA_PHASE1_RERUN_OUTPUT_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_phase1_oa_rerun_v1"
)
OA_PHASE1_RERUN_QA_ROOT = OA_PHASE1_RERUN_OUTPUT_ROOT / "qa"
OA_PHASE1_RERUN_TABLES_ROOT = OA_PHASE1_RERUN_OUTPUT_ROOT / "tables"
OA_PHASE1_RERUN_REPORT_ROOT = OA_PHASE1_RERUN_OUTPUT_ROOT / "report"

SOURCE_BACKED_ROOT = (
    DATA_ROOT
    / "processed"
    / "gaira_autoresearch"
    / "gaira_autoresearch_v1"
    / "gaira_source_backed_evidence_v1_corrected"
)
SOURCE_BACKED_VALID_PATH = SOURCE_BACKED_ROOT / "tables" / "cleaned_peak_assignments.csv"
SOURCE_BACKED_MENTIONS_PATH = SOURCE_BACKED_ROOT / "tables" / "wavenumber_mentions.csv"
SOURCE_BACKED_NOISE_PATH = SOURCE_BACKED_ROOT / "tables" / "noise_mentions.csv"
DIGITIZATION_QUEUE_PATH = SOURCE_BACKED_ROOT / "tables" / "prioritized_figure_digitization_queue.csv"
SOURCE_BACKED_NOTE_PATH = SOURCE_BACKED_ROOT / "report" / "cleaned_assignment_layer_note.md"
DIGITIZATION_NOTE_PATH = SOURCE_BACKED_ROOT / "report" / "figure_digitization_priority_note.md"

KNOWLEDGE_ROOT = DATA_ROOT / "raw" / "raman_knowledge_core"
RAMANBIOLIB_ROOT = DATA_ROOT / "raw" / "ramanbiolib" / "ramanbiolib-main" / "ramanbiolib" / "db"

MINIMAL_CONTEXT_DOC_IDS = (
    "gaira_ev_context_cross_substrate_caveat",
    "gaira_ev_context_cargo_mixture_caveat",
    "gaira_ev_context_evidence_tiering_note",
    "gaira_serum_context_adsorption_protocol_caveat",
    "gaira_serum_context_metabolite_dominance_caveat",
    "gaira_serum_context_tiering_note",
)


def ensure_output_dirs() -> None:
    for path in (OUTPUT_ROOT, QA_ROOT, TABLES_ROOT, REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def ensure_refinement_output_dirs() -> None:
    for path in (
        REFINEMENT_OUTPUT_ROOT,
        REFINEMENT_QA_ROOT,
        REFINEMENT_TABLES_ROOT,
        REFINEMENT_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_cleanup_output_dirs() -> None:
    for path in (
        CLEANUP_OUTPUT_ROOT,
        CLEANUP_QA_ROOT,
        CLEANUP_TABLES_ROOT,
        CLEANUP_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_pattern_output_dirs() -> None:
    for path in (
        PATTERN_OUTPUT_ROOT,
        PATTERN_QA_ROOT,
        PATTERN_TABLES_ROOT,
        PATTERN_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_source_audit_output_dirs() -> None:
    for path in (
        SOURCE_AUDIT_OUTPUT_ROOT,
        SOURCE_AUDIT_QA_ROOT,
        SOURCE_AUDIT_TABLES_ROOT,
        SOURCE_AUDIT_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_warehouse_output_dirs() -> None:
    for path in (
        WAREHOUSE_OUTPUT_ROOT,
        WAREHOUSE_QA_ROOT,
        WAREHOUSE_TABLES_ROOT,
        WAREHOUSE_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_pattern_refinement_output_dirs() -> None:
    for path in (
        PATTERN_REFINEMENT_OUTPUT_ROOT,
        PATTERN_REFINEMENT_QA_ROOT,
        PATTERN_REFINEMENT_TABLES_ROOT,
        PATTERN_REFINEMENT_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_literature_pilot_output_dirs() -> None:
    for path in (
        LITERATURE_PILOT_OUTPUT_ROOT,
        LITERATURE_PILOT_QA_ROOT,
        LITERATURE_PILOT_TABLES_ROOT,
        LITERATURE_PILOT_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_ontology_output_dirs() -> None:
    for path in (
        ONTOLOGY_OUTPUT_ROOT,
        ONTOLOGY_QA_ROOT,
        ONTOLOGY_TABLES_ROOT,
        ONTOLOGY_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_ontology_motif_output_dirs() -> None:
    for path in (
        ONTOLOGY_MOTIF_OUTPUT_ROOT,
        ONTOLOGY_MOTIF_QA_ROOT,
        ONTOLOGY_MOTIF_TABLES_ROOT,
        ONTOLOGY_MOTIF_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_local_neighborhood_output_dirs() -> None:
    for path in (
        LOCAL_NEIGHBORHOOD_OUTPUT_ROOT,
        LOCAL_NEIGHBORHOOD_QA_ROOT,
        LOCAL_NEIGHBORHOOD_TABLES_ROOT,
        LOCAL_NEIGHBORHOOD_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_paper_qc_output_dirs() -> None:
    for path in (
        PAPER_QC_OUTPUT_ROOT,
        PAPER_QC_QA_ROOT,
        PAPER_QC_TABLES_ROOT,
        PAPER_QC_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_enrichment_output_dirs() -> None:
    for path in (
        ENRICHMENT_OUTPUT_ROOT,
        ENRICHMENT_QA_ROOT,
        ENRICHMENT_TABLES_ROOT,
        ENRICHMENT_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_remaining_paper_output_dirs() -> None:
    for path in (
        REMAINING_PAPER_OUTPUT_ROOT,
        REMAINING_PAPER_QA_ROOT,
        REMAINING_PAPER_TABLES_ROOT,
        REMAINING_PAPER_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_condition_layer_output_dirs() -> None:
    for path in (
        CONDITION_LAYER_OUTPUT_ROOT,
        CONDITION_LAYER_QA_ROOT,
        CONDITION_LAYER_TABLES_ROOT,
        CONDITION_LAYER_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_literature_pipeline_output_dirs() -> None:
    for path in (
        LITERATURE_PIPELINE_OUTPUT_ROOT,
        LITERATURE_PIPELINE_QA_ROOT,
        LITERATURE_PIPELINE_TABLES_ROOT,
        LITERATURE_PIPELINE_REPORT_ROOT,
        LITERATURE_PIPELINE_ASSET_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_literature_asset_resolution_output_dirs() -> None:
    for path in (
        LITERATURE_ASSET_RESOLUTION_OUTPUT_ROOT,
        LITERATURE_ASSET_RESOLUTION_QA_ROOT,
        LITERATURE_ASSET_RESOLUTION_TABLES_ROOT,
        LITERATURE_ASSET_RESOLUTION_REPORT_ROOT,
        LITERATURE_ASSET_RESOLUTION_ASSET_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_literature_blocked_asset_output_dirs() -> None:
    for path in (
        LITERATURE_BLOCKED_ASSET_OUTPUT_ROOT,
        LITERATURE_BLOCKED_ASSET_QA_ROOT,
        LITERATURE_BLOCKED_ASSET_TABLES_ROOT,
        LITERATURE_BLOCKED_ASSET_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_ready_paper_ingest_output_dirs() -> None:
    for path in (
        READY_PAPER_INGEST_OUTPUT_ROOT,
        READY_PAPER_INGEST_QA_ROOT,
        READY_PAPER_INGEST_TABLES_ROOT,
        READY_PAPER_INGEST_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_asset_truth_output_dirs() -> None:
    for path in (
        ASSET_TRUTH_OUTPUT_ROOT,
        ASSET_TRUTH_QA_ROOT,
        ASSET_TRUTH_TABLES_ROOT,
        ASSET_TRUTH_REPORT_ROOT,
        ASSET_TRUTH_ASSET_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_oa_ready_ingest_output_dirs() -> None:
    for path in (
        OA_READY_INGEST_OUTPUT_ROOT,
        OA_READY_INGEST_QA_ROOT,
        OA_READY_INGEST_TABLES_ROOT,
        OA_READY_INGEST_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_oa_text_first_output_dirs() -> None:
    for path in (
        OA_TEXT_FIRST_OUTPUT_ROOT,
        OA_TEXT_FIRST_QA_ROOT,
        OA_TEXT_FIRST_TABLES_ROOT,
        OA_TEXT_FIRST_REPORT_ROOT,
        OA_TEXT_FIRST_ASSET_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_oa_text_followup_output_dirs() -> None:
    for path in (
        OA_TEXT_FOLLOWUP_OUTPUT_ROOT,
        OA_TEXT_FOLLOWUP_QA_ROOT,
        OA_TEXT_FOLLOWUP_TABLES_ROOT,
        OA_TEXT_FOLLOWUP_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_oa_phase1_rerun_output_dirs() -> None:
    for path in (
        OA_PHASE1_RERUN_OUTPUT_ROOT,
        OA_PHASE1_RERUN_QA_ROOT,
        OA_PHASE1_RERUN_TABLES_ROOT,
        OA_PHASE1_RERUN_REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
