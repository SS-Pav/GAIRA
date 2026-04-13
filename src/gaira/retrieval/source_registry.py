"""
GAIRA retrieval source registry — full structured evidence corpus.

Source types:
  grounding_component  — BSV definitions, scoring rules, confidence rules
  evidence_rules       — peak matching, harmonization, contrast schema, trust rules
  context_source       — serum/EV/domain context, sample-type interpretation
  benchmark_summary    — landscape analysis, condition-level BSV, extraction yields
  analysis_summary     — spectral query reports, substrate sensitivity
  meta_summary         — process notes, operational summaries (downweighted)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SSD_REPORTS = Path("/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/reports")

TIER_WEIGHTS = {
    "grounding_component": 1.5,
    "context_source":      1.3,
    "evidence_rules":      1.3,
    "benchmark_summary":   1.0,
    "analysis_summary":    0.8,
    "meta_summary":        0.4,
}

TIER_DISPLAY_NAMES = {
    "grounding_component": "Grounding Component",
    "context_source":      "Context Source",
    "evidence_rules":      "Evidence Rules",
    "benchmark_summary":   "Benchmark Summary",
    "analysis_summary":    "Analysis Summary",
    "meta_summary":        "Meta Summary",
}

GROUNDING_TIERS = {"grounding_component", "evidence_rules"}
LITERATURE_TIERS = {"context_source", "benchmark_summary", "analysis_summary", "meta_summary"}


@dataclass
class SourceEntry:
    path: str
    tier: str
    reason: str
    base: str = "repo"
    display_name: str = ""

    @property
    def full_path(self) -> Path:
        if self.base == "ssd":
            return SSD_REPORTS / self.path
        return REPO_ROOT / self.path


# Section-level demotion patterns
META_TITLE_PATTERNS = [
    "what changed", "what was built", "what should come next",
    "verdict", "output files", "honest caveats", "validation checks",
    "outputs (", "overall viability", "next step",
    "status:", "deferred", "not yet",
]

# Section-level promotion patterns
GROUNDED_TITLE_PATTERNS = [
    "delta vs healthy", "hcc vs healthy", "cca vs healthy", "lm vs healthy",
    "biochemical shifts", "component", "spectral windows",
    "axes align", "axes that distinguish", "substrate sensitivity",
    "signal stability", "scoring", "key motifs", "biological meaning",
    "discriminative", "condition signatures", "hcc signal",
    "nucleic_acid_backbone", "aromatic_amino_acid", "membrane_lipid",
    "protein_backbone", "confidence", "support tier", "trust",
    "directionality", "harmoniz", "peak match",
    "purine", "pyrimidine", "glycan", "redox", "lipid",
    "evidence rows", "unique meanings", "hierarchy",
]


SOURCE_REGISTRY: list[SourceEntry] = [
    # ══════════════════════════════════════════════════════════════
    # GROUNDING COMPONENTS — BSV definitions and structure
    # ══════════════════════════════════════════════════════════════
    SourceEntry("docs/bsv_v1_component_definitions.md", "grounding_component",
                "8 BSV component definitions", display_name="BSV Component Definitions"),
    SourceEntry("docs/bsv_v2_component_revision_notes.md", "grounding_component",
                "BSV v2 scientific rationale", display_name="BSV v2 Revision Notes"),
    SourceEntry("docs/bsv_v1_scoring_logic.md", "grounding_component",
                "Motif-to-component scoring rules", display_name="BSV Scoring Logic"),
    SourceEntry("docs/bsv_v2_component_confidence_rules.md", "grounding_component",
                "Component confidence labels and criteria", display_name="BSV Confidence Rules"),
    SourceEntry("docs/bsv_v1_graph_explanation.md", "grounding_component",
                "BSV graph structure explanation", display_name="BSV Graph Structure"),
    SourceEntry("docs/bsv_v1_output_format.md", "grounding_component",
                "BSV output data format", display_name="BSV Output Format"),
    SourceEntry("docs/bsv_v2_table_format.md", "grounding_component",
                "BSV v2 comparison table format", display_name="BSV Table Format"),

    # ══════════════════════════════════════════════════════════════
    # EVIDENCE RULES — matching, harmonization, contrast, trust
    # ══════════════════════════════════════════════════════════════
    SourceEntry("docs/d1_contrast_schema.md", "evidence_rules",
                "Directional peak change schema", display_name="Contrast Schema"),
    SourceEntry("docs/d1_peak_matching_rules.md", "evidence_rules",
                "Peak tolerance and matching logic", display_name="Peak Matching Rules"),
    SourceEntry("docs/d1_harmonization_rules.md", "evidence_rules",
                "Magnitude/significance harmonization", display_name="Harmonization Rules"),
    SourceEntry("docs/gaira_landscape_v3_trust_rules.md", "evidence_rules",
                "Evidence support tier definitions", display_name="Trust Tier Rules"),
    SourceEntry("docs/gaira_landscape_v3_condition_classification_rules.md", "evidence_rules",
                "Condition classification categories", display_name="Condition Classification"),
    SourceEntry("docs/d0_contrast_schema_proposal.md", "evidence_rules",
                "Proposed contrast schema (deferred)", display_name="Contrast Schema Proposal"),

    # ══════════════════════════════════════════════════════════════
    # CONTEXT SOURCES — domain interpretive context
    # ══════════════════════════════════════════════════════════════
    SourceEntry("docs/gaira_serum_context.md", "context_source",
                "Serum SERS interpretive context", display_name="Serum SERS Context"),
    SourceEntry("docs/gaira_ev_context.md", "context_source",
                "EV SERS interpretive context", display_name="EV SERS Context"),
    SourceEntry("docs/d1_1_liver_contrast_summary.md", "context_source",
                "Liver contrast evidence findings", display_name="Liver Contrast Summary"),
    SourceEntry("docs/d1_contrast_pilot_summary.md", "context_source",
                "Contrast evidence pilot results", display_name="Contrast Pilot Summary"),
    SourceEntry("docs/d0_directionality_audit_summary.md", "context_source",
                "Directionality viability audit", display_name="Directionality Audit"),
    SourceEntry("docs/d1_1_liver_failure_modes.md", "context_source",
                "Liver evidence failure modes", display_name="Liver Evidence Limits"),
    SourceEntry("docs/d1_integration_notes.md", "context_source",
                "Contrast data integration hooks", display_name="Contrast Integration Notes"),
    SourceEntry("docs/domain_context_template.md", "context_source",
                "Domain context data format", display_name="Domain Context Template"),

    # ══════════════════════════════════════════════════════════════
    # BENCHMARK SUMMARIES — landscape analysis, ontology, extraction
    # ══════════════════════════════════════════════════════════════
    SourceEntry("docs/gaira_landscape_v2_summary.md", "benchmark_summary",
                "Landscape v2 BSV matrix construction", display_name="Landscape v2 Summary"),
    SourceEntry("docs/gaira_landscape_v3_summary.md", "benchmark_summary",
                "Landscape v3 condition-level evidence", display_name="Landscape v3 Summary"),
    SourceEntry("docs/gaira_landscape_v4_summary.md", "benchmark_summary",
                "Landscape v4 BSV deltas and variance", display_name="Landscape v4 BSV Deltas"),
    SourceEntry("docs/gaira_landscape_v5_summary.md", "benchmark_summary",
                "Landscape v5 subcomponent decomposition", display_name="Landscape v5 Decomposition"),
    SourceEntry("docs/gaira_landscape_v5_1_summary.md", "benchmark_summary",
                "Landscape v5.1 inference hardening", display_name="Landscape v5.1 Hardening"),
    SourceEntry("docs/gaira_spectral_query_v1_hcc_summary.md", "benchmark_summary",
                "HCC holdout spectral query results", display_name="HCC Spectral Query v1"),
    SourceEntry("docs/bsv_v2_delta_plot_notes.md", "benchmark_summary",
                "BSV delta visualization notes", display_name="BSV Delta Plot Notes"),
    SourceEntry("docs/gaira_landscape_v3_bsv_bugfix_notes.md", "benchmark_summary",
                "Landscape v3 BSV bugfix details", display_name="Landscape v3 BSV Bugfix"),

    # SSD benchmark reports
    SourceEntry("phaseC_summary.md", "benchmark_summary",
                "Evidence stabilization + ontology normalization", base="ssd",
                display_name="Evidence Ontology (Phase C)"),
    SourceEntry("phaseG_pilot_summary.md", "benchmark_summary",
                "OA extraction pilot: 20 papers, 164 assignment rows", base="ssd",
                display_name="Extraction Pilot (Phase G)"),

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS SUMMARIES — spectral query reports (SSD)
    # ══════════════════════════════════════════════════════════════
    SourceEntry("gaira_spectral_query_v3_cca_summary.md", "analysis_summary",
                "CCA multi-condition BSV analysis", base="ssd",
                display_name="CCA BSV Analysis (v3)"),
    SourceEntry("gaira_spectral_query_v3_2_cca_summary.md", "analysis_summary",
                "CCA refined query results", base="ssd",
                display_name="CCA Refined Query (v3.2)"),
    SourceEntry("gaira_spectral_query_v3_3_cca_summary.md", "analysis_summary",
                "CCA robustness test", base="ssd",
                display_name="CCA Robustness (v3.3)"),
    SourceEntry("gaira_spectral_query_v2_1b_substrate_sensitivity.md", "analysis_summary",
                "Au vs AgNP substrate sensitivity", base="ssd",
                display_name="Substrate Sensitivity"),
    SourceEntry("gaira_spectral_query_v2_2_cca_summary.md", "analysis_summary",
                "CCA substrate-aware refinement", base="ssd",
                display_name="CCA Substrate Refinement"),
]
