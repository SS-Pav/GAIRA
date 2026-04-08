from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from gaira.evidence_v1.assignment_patterns import build_assignment_patterns
from gaira.evidence_v1.constants import (
    DB_PATH,
    LITERATURE_PILOT_OUTPUT_ROOT,
    LITERATURE_PILOT_REPORT_ROOT,
    LITERATURE_PILOT_TABLES_ROOT,
    ensure_literature_pilot_output_dirs,
)
from gaira.evidence_v1.phase1_refinement import (
    CLUSTER_MERGE_TOLERANCE_CM,
    FAMILY_LABELS,
    build_phase1_refinement,
    normalize_meaning_family,
)
from gaira.evidence_v1.retrieval import PeakListRetrievalEngine
from gaira.evidence_v1.schema import initialize_schema, reset_phase1_refinement_tables


PILOT_CREATED_BY = "three_paper_literature_pilot_v1"
PILOT_SOURCE_KIND = "pilot_literature_digitization"
PILOT_ASSIGNMENT_PREFIX = "pilot3p"
PILOT_QUERY_TOLERANCE_CM = 10.0
NEW_MOTIF_DISTANCE_CM = 22.0
EXTENDED_MOTIF_DISTANCE_CM = 20.0
MISMATCH_DISTANCE_CM = 8.0


@dataclass(frozen=True)
class PaperSpec:
    paper_id: str
    source_id: str
    study_family: str
    title: str
    citation_short: str
    manuscript_path: str
    source_name: str
    sample_scope: str
    biosample_type: str
    modality: str
    disease_class: str
    stress_class: str
    matrix_context: str
    substrate: str
    selection_reason: str
    query_peaks: tuple[float, ...]


@dataclass(frozen=True)
class ExtractedAssignment:
    assignment_id: str
    paper_id: str
    source_id: str
    study_family: str
    peak_center_cm: float
    assigned_molecule: str
    assigned_group_or_theme: str
    evidence_text: str
    extraction_method: str
    figure_or_table_ref: str
    page_or_sheet: str
    confidence_label: str
    sample_type: str
    modality: str
    substrate: str
    matrix_context: str
    manuscript_or_si: str
    is_primary_retrieval_eligible: bool
    note_tag: str


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def pilot_papers() -> list[PaperSpec]:
    return [
        PaperSpec(
            paper_id="liu_2025_lung_ev_sers",
            source_id="src_liu_2025_lung_manuscript",
            study_family="liu_2025_lung",
            title="Lung cancer diagnosis through extracellular vesicle analysis using label-free surface-enhanced Raman spectroscopy coupled with machine learning",
            citation_short="Liu et al., Theranostics 2025",
            manuscript_path="/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_corpus/manuscripts/Liu_2025_lung_cancer_ev_sers_ml_theranostics.pdf",
            source_name="liu 2025 lung manuscript",
            sample_scope="structured_biological_paper",
            biosample_type="ev",
            modality="sers",
            disease_class="lung_disease_or_cancer",
            stress_class="",
            matrix_context="plasma_derived_ev_lung_cancer_label_free_sers",
            substrate="au_nanoparticle_substrate",
            selection_reason="EV-focused paper with a clear annotated SERS figure, explicit band assignments, and direct disease relevance.",
            query_peaks=(493.0, 741.0, 1078.0, 1221.0, 1437.0),
        ),
        PaperSpec(
            paper_id="cca_2024_serum_sers",
            source_id="src_cca_2024_manuscript",
            study_family="cca_2024",
            title="Highly Accurate and Robust Early Stage Detection of Cholangiocarcinoma Using Near-Lossless SERS Signal Processing with Machine Learning and 2D CNN for Point-of-care Mobile Application",
            citation_short="Danvirutai et al., ACS Omega 2025",
            manuscript_path="/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_corpus/manuscripts/CCA_2024_near_lossless_sers_detection_cca_hcc.pdf",
            source_name="cca 2024 manuscript",
            sample_scope="structured_biological_paper",
            biosample_type="serum",
            modality="sers",
            disease_class="cholangiocarcinoma_context",
            stress_class="",
            matrix_context="hamster_serum_cca_progression_sers",
            substrate="sers_nanostructured_surface",
            selection_reason="Serum disease-context paper with a labeled discriminative peak figure and explicit biological assignments.",
            query_peaks=(560.0, 1004.0, 1158.0, 1524.0, 1675.0),
        ),
        PaperSpec(
            paper_id="krafft_2018_reference_style",
            source_id="src_krafft_2018_manuscript",
            study_family="krafft_2018",
            title="Raman Spectroscopy of Proteins and Nucleic Acids: From Amino Acids and Nucleotides to Large Assemblies",
            citation_short="Krafft, Encyclopedia of Analytical Chemistry 2018",
            manuscript_path="/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_corpus/manuscripts/Krafft_2018_raman_proteins_nucleic_acids_encyclopedia.pdf",
            source_name="krafft 2018 manuscript",
            sample_scope="reference_style_literature_support",
            biosample_type="none",
            modality="raman",
            disease_class="",
            stress_class="",
            matrix_context="reference_style_protein_and_nucleic_acid_raman_review",
            substrate="none",
            selection_reason="Controlled reference-style literature with strong assignment fidelity across protein and nucleic-acid bands.",
            query_peaks=(508.0, 727.0, 832.0, 1092.0, 1240.0, 1656.0),
        ),
    ]


def pilot_assignments() -> list[ExtractedAssignment]:
    rows = [
        # Liu 2025 EV SERS
        ("liu_001", "liu_2025_lung_ev_sers", 493.0, "", "glycogen", "493 cm^-1 assigned to glycogen in the EV SERS discrimination figure.", "digitized_figure", "Figure 4", "11", "high", True, "figure_primary"),
        ("liu_002", "liu_2025_lung_ev_sers", 741.0, "", "amide IV O-CN bending", "741 cm^-1 assigned to O-CN bending of amide IV.", "digitized_figure", "Figure 4", "11", "high", True, "figure_primary"),
        ("liu_003", "liu_2025_lung_ev_sers", 1011.0, "", "aromatic ring / benzene breathing", "1011 cm^-1 assigned to breathing of the benzene ring.", "digitized_figure", "Figure 4", "11", "medium", True, "figure_primary"),
        ("liu_004", "liu_2025_lung_ev_sers", 1078.0, "lysine", "lysine side-chain stretch", "1078 cm^-1 assigned to C-C and Cε-Nζ stretching of lysine.", "digitized_figure", "Figure 4", "11", "high", True, "figure_primary"),
        ("liu_005", "liu_2025_lung_ev_sers", 1221.0, "", "amide III beta-sheet", "1221 cm^-1 assigned to amide III of beta-sheet structure.", "digitized_figure", "Figure 4", "11", "high", True, "figure_primary"),
        ("liu_006", "liu_2025_lung_ev_sers", 1349.0, "", "protein backbone C-alpha skeletal mode", "1349 cm^-1 assigned to Cα-H bending and Cα-C stretching.", "digitized_figure", "Figure 4", "11", "medium", True, "figure_primary"),
        ("liu_007", "liu_2025_lung_ev_sers", 1437.0, "", "lipid CH2 bending", "1437 cm^-1 assigned to CH2 bending of lipids.", "digitized_figure", "Figure 4", "11", "high", True, "figure_primary"),
        ("liu_008", "liu_2025_lung_ev_sers", 645.0, "", "", "645 cm^-1 described as a newly appeared band related to lipids, proteins, and DNA.", "text_assignment", "Figure 4 / Results text", "16", "low", False, "ambiguous_multifamily"),
        ("liu_009", "liu_2025_lung_ev_sers", 1163.0, "", "", "1163 cm^-1 described as a newly appeared band related to lipids, proteins, and DNA.", "text_assignment", "Figure 4 / Results text", "16", "low", False, "ambiguous_multifamily"),
        ("liu_010", "liu_2025_lung_ev_sers", 1598.0, "", "citrate on Au nanoparticle surface", "Around 1598 cm^-1 the broad band is attributed mainly to citrate adsorbed on the gold nanoparticle surface.", "text_assignment", "Figure 4 / Results text", "16", "low", False, "substrate_confounder"),
        ("liu_011", "liu_2025_lung_ev_sers", 2913.0, "", "lipid and protein C-H stretching", "2913 cm^-1 assigned to C-H stretching of lipids and proteins.", "digitized_figure", "Figure 4", "11", "low", False, "outside_phase1_band"),
        # CCA 2024 serum SERS
        ("cca_001", "cca_2024_serum_sers", 560.0, "", "protein disulfide bond", "560 cm^-1 assigned to disulfide bonds and S-S stretching.", "digitized_figure", "Figure 9", "12", "high", True, "figure_primary"),
        ("cca_002", "cca_2024_serum_sers", 785.0, "", "", "785 cm^-1 described as an aromatic-ring peak in a nucleic-acid/protein comparison context.", "text_assignment", "Figure 9 / Peak Associations", "12", "low", False, "ambiguous_overlap"),
        ("cca_003", "cca_2024_serum_sers", 833.0, "", "protein aromatic side chain", "833 cm^-1 assigned to protein aromatic side chains.", "digitized_figure", "Figure 9", "12", "medium", True, "figure_primary"),
        ("cca_004", "cca_2024_serum_sers", 1004.0, "phenylalanine", "phenylalanine ring breathing", "1004 cm^-1 assigned to phenylalanine and C-C ring breathing.", "digitized_figure", "Figure 9", "12", "high", True, "figure_primary"),
        ("cca_005", "cca_2024_serum_sers", 1126.0, "", "protein-associated C-N stretching", "1126 cm^-1 assigned to C-N stretching.", "digitized_figure", "Figure 9", "12", "medium", True, "figure_primary"),
        ("cca_006", "cca_2024_serum_sers", 1158.0, "", "beta-carotene C-C stretching", "1158 cm^-1 assigned to beta-carotene and C-C stretching.", "digitized_figure", "Figure 9", "12", "high", True, "figure_primary"),
        ("cca_007", "cca_2024_serum_sers", 1248.0, "", "amide III", "1248 cm^-1 assigned to amide III.", "digitized_figure", "Figure 9", "12", "high", True, "figure_primary"),
        ("cca_008", "cca_2024_serum_sers", 1265.0, "", "amide III", "1265 cm^-1 assigned to amide III.", "digitized_figure", "Figure 9", "12", "high", True, "figure_primary"),
        ("cca_009", "cca_2024_serum_sers", 1400.0, "", "lipid CH2 bending", "1400 cm^-1 assigned to CH2 bending.", "digitized_figure", "Figure 9", "12", "medium", True, "figure_primary"),
        ("cca_010", "cca_2024_serum_sers", 1524.0, "", "unsaturated lipid C=C stretching", "1524 cm^-1 assigned to unsaturated lipids and C=C stretching.", "digitized_figure", "Figure 9", "12", "high", True, "figure_primary"),
        ("cca_011", "cca_2024_serum_sers", 1590.0, "", "aromatic amino-acid C=C stretching", "1590 cm^-1 assigned to aromatic amino acids and C=C stretching.", "digitized_figure", "Figure 9", "12", "high", True, "figure_primary"),
        ("cca_012", "cca_2024_serum_sers", 1675.0, "", "amide I", "1675 cm^-1 assigned to amide I.", "digitized_figure", "Figure 9", "12", "high", True, "figure_primary"),
        ("cca_013", "cca_2024_serum_sers", 1860.0, "", "lipid oxidation carbonyl marker", "1860 cm^-1 assigned to C=O stretching of aldehydes or anhydrides as a lipid oxidation marker.", "text_assignment", "Figure 9 / Peak Associations", "12", "medium", True, "new_high_band_candidate"),
        ("cca_014", "cca_2024_serum_sers", 1930.0, "", "lipid oxidation carbonyl marker", "1930 cm^-1 assigned to C=O stretching of aldehydes or anhydrides as a lipid oxidation marker.", "text_assignment", "Figure 9 / Peak Associations", "12", "medium", True, "new_high_band_candidate"),
        # Krafft 2018 reference-style Raman
        ("krafft_001", "krafft_2018_reference_style", 508.0, "", "protein disulfide bond", "508 cm^-1 assigned to disulfide bonds in the protein spectra discussion.", "text_assignment", "Figure 3 / Protein text", "5", "high", True, "reference_text_explicit"),
        ("krafft_002", "krafft_2018_reference_style", 622.0, "phenylalanine", "phenylalanine aromatic band", "622 cm^-1 listed as a phenylalanine band.", "text_assignment", "Figure 3", "5", "high", True, "reference_text_explicit"),
        ("krafft_003", "krafft_2018_reference_style", 727.0, "adenine", "adenine marker band", "727 cm^-1 listed as an adenine marker band.", "text_assignment", "Figure 4 / Nucleic-acid text", "6", "high", True, "reference_text_explicit"),
        ("krafft_004", "krafft_2018_reference_style", 760.0, "tryptophan", "tryptophan aromatic band", "760 cm^-1 described as a tryptophan-associated band.", "text_assignment", "Figure 3 / Protein text", "5", "medium", True, "reference_text_explicit"),
        ("krafft_005", "krafft_2018_reference_style", 786.0, "cytosine", "cytosine marker band", "786 cm^-1 listed as a cytosine marker band.", "text_assignment", "Figure 4 / Nucleic-acid text", "6", "high", True, "reference_text_explicit"),
        ("krafft_006", "krafft_2018_reference_style", 813.0, "", "RNA A-form backbone", "813 cm^-1 assigned to the RNA or A-form backbone.", "text_assignment", "Figure 4 / Nucleic-acid text", "6", "high", True, "reference_text_explicit"),
        ("krafft_007", "krafft_2018_reference_style", 832.0, "", "B-DNA backbone", "832 cm^-1 assigned to the B-DNA backbone.", "text_assignment", "Figure 4 / Nucleic-acid text", "6", "high", True, "reference_text_explicit"),
        ("krafft_008", "krafft_2018_reference_style", 852.0, "tyrosine", "tyrosine aromatic band", "852 cm^-1 listed as a tyrosine band.", "text_assignment", "Figure 3", "5", "high", True, "reference_text_explicit"),
        ("krafft_009", "krafft_2018_reference_style", 940.0, "", "protein backbone C-Calpha", "940 cm^-1 assigned to the protein C-Cα skeletal mode.", "text_assignment", "Figure 3 / Protein text", "5", "high", True, "reference_text_explicit"),
        ("krafft_010", "krafft_2018_reference_style", 1003.0, "phenylalanine", "phenylalanine aromatic band", "1003 cm^-1 listed as a phenylalanine band.", "text_assignment", "Figure 3", "5", "high", True, "reference_text_explicit"),
        ("krafft_011", "krafft_2018_reference_style", 1092.0, "", "B-DNA backbone", "1092 cm^-1 assigned to the B-DNA backbone.", "text_assignment", "Figure 4 / Nucleic-acid text", "6", "high", True, "reference_text_explicit"),
        ("krafft_012", "krafft_2018_reference_style", 1098.0, "", "RNA A-form backbone", "1098 cm^-1 assigned to the RNA or A-form backbone.", "text_assignment", "Figure 4 / Nucleic-acid text", "6", "high", True, "reference_text_explicit"),
        ("krafft_013", "krafft_2018_reference_style", 1240.0, "", "amide III", "1240 cm^-1 assigned to amide III in concanavalin A.", "text_assignment", "Figure 3 / Protein text", "5", "high", True, "reference_text_explicit"),
        ("krafft_014", "krafft_2018_reference_style", 1275.0, "", "amide III", "1275 cm^-1 assigned to amide III in BSA.", "text_assignment", "Figure 3 / Protein text", "5", "high", True, "reference_text_explicit"),
        ("krafft_015", "krafft_2018_reference_style", 1342.0, "adenine", "adenine marker band", "1342 cm^-1 listed as an adenine marker band.", "text_assignment", "Figure 4 / Nucleic-acid text", "6", "high", True, "reference_text_explicit"),
        ("krafft_016", "krafft_2018_reference_style", 1375.0, "thymine", "thymine marker band", "1375 cm^-1 listed as a thymine marker band.", "text_assignment", "Figure 4 / Nucleic-acid text", "6", "high", True, "reference_text_explicit"),
        ("krafft_017", "krafft_2018_reference_style", 1449.0, "", "protein aliphatic CH deformation", "1449 cm^-1 assigned to aliphatic group deformation in proteins.", "text_assignment", "Figure 3 / Protein text", "5", "medium", True, "reference_text_explicit"),
        ("krafft_018", "krafft_2018_reference_style", 1489.0, "guanine", "guanine marker band", "1489 cm^-1 listed as a guanine marker band.", "text_assignment", "Figure 4 / Nucleic-acid text", "6", "high", True, "reference_text_explicit"),
        ("krafft_019", "krafft_2018_reference_style", 1577.0, "guanine", "guanine marker band", "1577 cm^-1 listed as a guanine marker band.", "text_assignment", "Figure 4 / Nucleic-acid text", "6", "high", True, "reference_text_explicit"),
        ("krafft_020", "krafft_2018_reference_style", 1656.0, "", "amide I", "1656 cm^-1 assigned to amide I in BSA.", "text_assignment", "Figure 3 / Protein text", "5", "high", True, "reference_text_explicit"),
        ("krafft_021", "krafft_2018_reference_style", 1660.0, "", "amide I collagen", "1660 cm^-1 assigned to amide I in collagen.", "text_assignment", "Figure 3 / Protein text", "5", "high", True, "reference_text_explicit"),
        ("krafft_022", "krafft_2018_reference_style", 1677.0, "", "amide I", "1677 cm^-1 assigned to amide I in concanavalin A.", "text_assignment", "Figure 3 / Protein text", "5", "high", True, "reference_text_explicit"),
    ]

    paper_lookup = {paper.paper_id: paper for paper in pilot_papers()}
    assignments: list[ExtractedAssignment] = []
    for row in rows:
        (
            assignment_id,
            paper_id,
            peak_center_cm,
            assigned_molecule,
            assigned_group_or_theme,
            evidence_text,
            extraction_method,
            figure_or_table_ref,
            page_or_sheet,
            confidence_label,
            is_primary_retrieval_eligible,
            note_tag,
        ) = row
        paper = paper_lookup[paper_id]
        assignments.append(
            ExtractedAssignment(
                assignment_id=assignment_id,
                paper_id=paper_id,
                source_id=paper.source_id,
                study_family=paper.study_family,
                peak_center_cm=peak_center_cm,
                assigned_molecule=assigned_molecule,
                assigned_group_or_theme=assigned_group_or_theme,
                evidence_text=evidence_text,
                extraction_method=extraction_method,
                figure_or_table_ref=figure_or_table_ref,
                page_or_sheet=page_or_sheet,
                confidence_label=confidence_label,
                sample_type=paper.biosample_type,
                modality=paper.modality,
                substrate=paper.substrate,
                matrix_context=paper.matrix_context,
                manuscript_or_si="manuscript",
                is_primary_retrieval_eligible=is_primary_retrieval_eligible,
                note_tag=note_tag,
            )
        )
    return assignments


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _delete_previous_pilot_rows(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"DELETE FROM evidence.peak_assignment_evidence WHERE assignment_record_id LIKE '{PILOT_ASSIGNMENT_PREFIX}_%'"
    )
    connection.execute(
        f"DELETE FROM evidence.evidence_items WHERE evidence_item_id LIKE '{PILOT_ASSIGNMENT_PREFIX}_%'"
    )
    connection.execute(
        "DELETE FROM registry.evidence_sources WHERE source_kind = ?",
        [PILOT_SOURCE_KIND],
    )
    connection.execute(
        "DELETE FROM registry.warehouse_sources WHERE source_kind = ?",
        [PILOT_SOURCE_KIND],
    )


def _insert_pilot_registry_rows(connection: duckdb.DuckDBPyConnection, papers: list[PaperSpec]) -> None:
    evidence_rows = []
    warehouse_rows = []
    for paper in papers:
        evidence_rows.append(
            (
                paper.source_id,
                paper.source_name,
                "disease_or_stress_paper",
                PILOT_SOURCE_KIND,
                paper.manuscript_path,
                "pilot_three_paper_literature_corpus",
                paper.citation_short,
                "explicit_literature_peak_assignments",
                "tier1_figure_or_explicit_text_assignment",
                False,
                "Three-paper pilot integration from existing local corpus with figure-first extraction.",
            )
        )
        warehouse_rows.append(
            (
                paper.source_id,
                paper.source_name,
                "disease_or_stress_paper",
                paper.sample_scope,
                paper.biosample_type,
                paper.modality,
                False,
                True,
                paper.disease_class,
                paper.stress_class,
                False,
                True,
                False,
                paper.manuscript_path,
                PILOT_SOURCE_KIND,
                "pilot_three_paper_literature_corpus",
                f"{paper.selection_reason} Added as a controlled pilot literature source.",
            )
        )
    connection.executemany(
        "INSERT INTO registry.evidence_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        evidence_rows,
    )
    connection.executemany(
        "INSERT INTO registry.warehouse_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        warehouse_rows,
    )


def _evidence_tier(assignment: ExtractedAssignment) -> str:
    if assignment.extraction_method == "digitized_figure":
        return "tier1_digitized_figure_assignment"
    if assignment.is_primary_retrieval_eligible:
        return "tier2_explicit_text_assignment"
    return "tier3_low_confidence_contextual_assignment"


def _insert_pilot_assignments(
    connection: duckdb.DuckDBPyConnection,
    papers: list[PaperSpec],
    assignments: list[ExtractedAssignment],
) -> list[dict]:
    paper_lookup = {paper.paper_id: paper for paper in papers}
    evidence_rows = []
    assignment_rows = []
    extraction_rows = []
    for assignment in assignments:
        paper = paper_lookup[assignment.paper_id]
        evidence_item_id = f"{PILOT_ASSIGNMENT_PREFIX}_{assignment.assignment_id}"
        assignment_record_id = f"{PILOT_ASSIGNMENT_PREFIX}_{assignment.assignment_id}"
        title = f"{paper.study_family} {assignment.peak_center_cm:.0f} cm^-1 literature assignment"
        provenance_detail = f"{assignment.figure_or_table_ref}; page {assignment.page_or_sheet}; {assignment.extraction_method}"
        evidence_rows.append(
            (
                evidence_item_id,
                assignment.source_id,
                assignment_record_id,
                "literature_peak_assignment",
                _evidence_tier(assignment),
                assignment.confidence_label,
                title,
                paper.manuscript_path,
                provenance_detail,
                assignment.is_primary_retrieval_eligible,
                PILOT_CREATED_BY,
                assignment.note_tag,
            )
        )
        assignment_rows.append(
            (
                evidence_item_id,
                assignment.source_id,
                assignment_record_id,
                f"pilot_literature_{assignment.extraction_method}",
                assignment.study_family,
                assignment.peak_center_cm,
                assignment.peak_center_cm,
                assignment.peak_center_cm,
                8.0,
                assignment.assigned_molecule,
                assignment.assigned_group_or_theme,
                assignment.sample_type,
                assignment.modality,
                assignment.substrate,
                assignment.matrix_context,
                assignment.manuscript_or_si,
                assignment.figure_or_table_ref,
                assignment.page_or_sheet,
                assignment.extraction_method,
                assignment.confidence_label,
                assignment.evidence_text,
                assignment.is_primary_retrieval_eligible,
                assignment.note_tag,
            )
        )
        family, normalized_label = normalize_meaning_family(
            assignment.assigned_molecule,
            assignment.assigned_group_or_theme,
            assignment.evidence_text,
        )
        extraction_rows.append(
            {
                "paper_id": paper.paper_id,
                "citation_short": paper.citation_short,
                "source_id": assignment.source_id,
                "study_family": assignment.study_family,
                "assignment_record_id": assignment_record_id,
                "peak_center_cm": assignment.peak_center_cm,
                "assigned_molecule": assignment.assigned_molecule,
                "assigned_group_or_theme": assignment.assigned_group_or_theme,
                "normalized_family": family,
                "normalized_meaning_label": normalized_label,
                "evidence_text": assignment.evidence_text,
                "extraction_method": assignment.extraction_method,
                "figure_or_table_ref": assignment.figure_or_table_ref,
                "page_or_sheet": assignment.page_or_sheet,
                "sample_type": assignment.sample_type,
                "modality": assignment.modality,
                "disease_class": paper.disease_class,
                "stress_class": paper.stress_class,
                "confidence_level": assignment.confidence_label,
                "is_primary_retrieval_eligible": assignment.is_primary_retrieval_eligible,
                "note_tag": assignment.note_tag,
            }
        )
    connection.executemany(
        "INSERT INTO evidence.evidence_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        evidence_rows,
    )
    connection.executemany(
        "INSERT INTO evidence.peak_assignment_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        assignment_rows,
    )
    return extraction_rows


def _fetch_clusters(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = connection.sql(
        """
        SELECT
            cluster_id,
            canonical_peak_cm,
            window_start_cm,
            window_end_cm,
            normalized_family,
            normalized_meaning_label,
            confidence_score,
            ambiguity_score,
            source_diversity_count,
            curated_assignment_count,
            explicit_assignment_count,
            reference_support_count,
            aligned_mention_support_count
        FROM evidence.peak_meaning_clusters
        ORDER BY normalized_family, canonical_peak_cm, cluster_id
        """
    ).fetchall()
    return [
        {
            "cluster_id": row[0],
            "canonical_peak_cm": float(row[1]),
            "window_start_cm": float(row[2]),
            "window_end_cm": float(row[3]),
            "normalized_family": row[4],
            "normalized_meaning_label": row[5],
            "confidence_score": float(row[6]),
            "ambiguity_score": float(row[7]),
            "source_diversity_count": int(row[8]),
            "curated_assignment_count": int(row[9]),
            "explicit_assignment_count": int(row[10]),
            "reference_support_count": int(row[11]),
            "aligned_mention_support_count": int(row[12]),
        }
        for row in rows
    ]


def _fetch_patterns(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = connection.sql(
        """
        SELECT
            ap.pattern_id,
            ap.normalized_family,
            ap.pattern_label,
            ap.core_member_count,
            ap.total_member_count,
            ap.confidence_score,
            ap.ambiguity_score,
            ap.coherence_score,
            ap.separability_score,
            ap.support_strength_score,
            apm.cluster_id,
            apm.canonical_peak_cm,
            apm.member_role
        FROM evidence.assignment_patterns ap
        JOIN evidence.assignment_pattern_members apm
          ON ap.pattern_id = apm.pattern_id
        ORDER BY ap.pattern_id, apm.canonical_peak_cm
        """
    ).fetchall()
    grouped: dict[str, dict] = {}
    for row in rows:
        pattern = grouped.setdefault(
            row[0],
            {
                "pattern_id": row[0],
                "normalized_family": row[1],
                "pattern_label": row[2],
                "core_member_count": int(row[3]),
                "total_member_count": int(row[4]),
                "confidence_score": float(row[5]),
                "ambiguity_score": float(row[6]),
                "coherence_score": float(row[7]),
                "separability_score": float(row[8]),
                "support_strength_score": float(row[9]),
                "members": [],
            },
        )
        pattern["members"].append(
            {
                "cluster_id": row[10],
                "canonical_peak_cm": float(row[11]),
                "member_role": row[12],
            }
        )
    return list(grouped.values())


def _nearest_family_cluster(clusters: list[dict], normalized_family: str, peak_center_cm: float) -> dict | None:
    candidates = [cluster for cluster in clusters if cluster["normalized_family"] == normalized_family]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda cluster: (abs(cluster["canonical_peak_cm"] - peak_center_cm), cluster["cluster_id"]),
    )


def _nearest_other_family_cluster(clusters: list[dict], normalized_family: str, peak_center_cm: float) -> dict | None:
    candidates = [cluster for cluster in clusters if cluster["normalized_family"] != normalized_family]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda cluster: (abs(cluster["canonical_peak_cm"] - peak_center_cm), cluster["cluster_id"]),
    )


def _pattern_match_for_peak(patterns: list[dict], normalized_family: str, peak_center_cm: float) -> dict | None:
    candidates = []
    for pattern in patterns:
        if pattern["normalized_family"] != normalized_family:
            continue
        member_distances = [abs(member["canonical_peak_cm"] - peak_center_cm) for member in pattern["members"]]
        if not member_distances:
            continue
        min_distance = min(member_distances)
        if min_distance <= NEW_MOTIF_DISTANCE_CM:
            candidates.append((min_distance, -pattern["support_strength_score"], pattern))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1], item[2]["pattern_id"]))[0][2]


def _snapshot_state(connection: duckdb.DuckDBPyConnection) -> dict:
    clusters = _fetch_clusters(connection)
    patterns = _fetch_patterns(connection)
    pattern_counts = Counter(pattern["normalized_family"] for pattern in patterns)
    return {
        "clusters": clusters,
        "patterns": patterns,
        "pattern_counts": dict(pattern_counts),
        "cluster_count": len(clusters),
        "pattern_count": len(patterns),
    }


def _paper_queries(papers: list[PaperSpec]) -> list[dict]:
    return [
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "query_peaks": list(paper.query_peaks),
            "domain_hint": paper.biosample_type if paper.biosample_type != "none" else None,
            "modality_hint": paper.modality,
        }
        for paper in papers
    ]


def _run_retrieval_queries(engine: PeakListRetrievalEngine, papers: list[PaperSpec]) -> list[dict]:
    results = []
    for query in _paper_queries(papers):
        payload = engine.search(
            query_peaks=query["query_peaks"],
            domain_hint=query["domain_hint"],
            modality_hint=query["modality_hint"],
            tolerance_cm=PILOT_QUERY_TOLERANCE_CM,
            top_k=5,
        )
        results.append(
            {
                "paper_id": query["paper_id"],
                "title": query["title"],
                "query_peaks": query["query_peaks"],
                "domain_hint": query["domain_hint"],
                "modality_hint": query["modality_hint"],
                "pattern_results": payload.get("pattern_results", []),
                "support_bundle_results": payload.get("support_bundle_results", []),
            }
        )
    return results


def _classify_interactions(
    assignments: list[ExtractedAssignment],
    papers: list[PaperSpec],
    before_state: dict,
    after_state: dict,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    paper_lookup = {paper.paper_id: paper for paper in papers}
    interaction_rows = []
    strengthened_rows = []
    new_motif_rows = []
    unresolved_rows = []

    for assignment in assignments:
        paper = paper_lookup[assignment.paper_id]
        family, normalized_label = normalize_meaning_family(
            assignment.assigned_molecule,
            assignment.assigned_group_or_theme,
            assignment.evidence_text,
        )
        before_cluster = _nearest_family_cluster(before_state["clusters"], family, assignment.peak_center_cm)
        after_cluster = _nearest_family_cluster(after_state["clusters"], family, assignment.peak_center_cm)
        before_pattern = _pattern_match_for_peak(before_state["patterns"], family, assignment.peak_center_cm)
        after_pattern = _pattern_match_for_peak(after_state["patterns"], family, assignment.peak_center_cm)
        before_distance = (
            abs(before_cluster["canonical_peak_cm"] - assignment.peak_center_cm)
            if before_cluster is not None
            else None
        )
        after_distance = (
            abs(after_cluster["canonical_peak_cm"] - assignment.peak_center_cm)
            if after_cluster is not None
            else None
        )
        other_family_cluster = _nearest_other_family_cluster(before_state["clusters"], family, assignment.peak_center_cm)
        other_family_distance = (
            abs(other_family_cluster["canonical_peak_cm"] - assignment.peak_center_cm)
            if other_family_cluster is not None
            else None
        )

        interaction_status = "strengthened_existing_motif"
        rationale = "Existing same-family motif window already present and now receives direct literature support."
        conflicting_flag = False

        if not assignment.is_primary_retrieval_eligible or family == "unresolved_assignment_support":
            interaction_status = "unresolved_or_ambiguous"
            rationale = "Assignment is explicit but too ambiguous, confounded, or outside the active Phase 1 motif band to promote."
        elif assignment.note_tag in {"substrate_confounder", "ambiguous_overlap"}:
            interaction_status = "conflicting_with_existing"
            rationale = "Assignment introduces a context-dependent or confounded interpretation that should not be merged as clean primary support."
            conflicting_flag = True
        elif before_cluster is not None and before_distance is not None and before_distance <= CLUSTER_MERGE_TOLERANCE_CM:
            interaction_status = "strengthened_existing_motif"
        elif before_cluster is not None and before_distance is not None and before_distance <= EXTENDED_MOTIF_DISTANCE_CM:
            interaction_status = "extended_existing_motif"
            rationale = "Assignment sits near an existing same-family window but broadens the observed constellation."
        elif after_cluster is not None and after_distance is not None and after_distance <= NEW_MOTIF_DISTANCE_CM:
            interaction_status = "new_motif_candidate"
            rationale = "Assignment required a newly expressed or newly isolated same-family motif window after pilot integration."
        elif other_family_distance is not None and other_family_distance <= MISMATCH_DISTANCE_CM:
            interaction_status = "conflicting_with_existing"
            rationale = "Nearest pre-pilot support was dominated by a different family, so the assignment should be treated as a mismatch pressure point."
            conflicting_flag = True

        row = {
            "paper_id": assignment.paper_id,
            "citation_short": paper.citation_short,
            "assignment_record_id": f"{PILOT_ASSIGNMENT_PREFIX}_{assignment.assignment_id}",
            "peak_center_cm": assignment.peak_center_cm,
            "normalized_family": family,
            "normalized_meaning_label": normalized_label,
            "interaction_status": interaction_status,
            "before_cluster_id": before_cluster["cluster_id"] if before_cluster else "",
            "before_cluster_distance_cm": round(before_distance, 3) if before_distance is not None else "",
            "after_cluster_id": after_cluster["cluster_id"] if after_cluster else "",
            "after_cluster_distance_cm": round(after_distance, 3) if after_distance is not None else "",
            "before_pattern_id": before_pattern["pattern_id"] if before_pattern else "",
            "before_pattern_label": before_pattern["pattern_label"] if before_pattern else "",
            "after_pattern_id": after_pattern["pattern_id"] if after_pattern else "",
            "after_pattern_label": after_pattern["pattern_label"] if after_pattern else "",
            "confidence_level": assignment.confidence_label,
            "extraction_method": assignment.extraction_method,
            "is_primary_retrieval_eligible": assignment.is_primary_retrieval_eligible,
            "conflicting_flag": conflicting_flag,
            "rationale": rationale,
            "evidence_text": assignment.evidence_text,
        }
        interaction_rows.append(row)
        if interaction_status in {"strengthened_existing_motif", "extended_existing_motif"}:
            strengthened_rows.append(row)
        if interaction_status == "new_motif_candidate":
            new_motif_rows.append(row)
        if interaction_status == "unresolved_or_ambiguous":
            unresolved_rows.append(row)

    return interaction_rows, strengthened_rows, new_motif_rows, unresolved_rows


def _summarize_interactions(interaction_rows: list[dict], papers: list[PaperSpec]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in interaction_rows:
        grouped[row["paper_id"]].append(row)
    rows = []
    for paper in papers:
        entries = grouped.get(paper.paper_id, [])
        counter = Counter(row["interaction_status"] for row in entries)
        rows.append(
            {
                "paper_id": paper.paper_id,
                "citation_short": paper.citation_short,
                "source_id": paper.source_id,
                "strengthened_existing_motifs": counter.get("strengthened_existing_motif", 0),
                "extended_existing_motifs": counter.get("extended_existing_motif", 0),
                "new_motif_candidates": counter.get("new_motif_candidate", 0),
                "unresolved_or_ambiguous": counter.get("unresolved_or_ambiguous", 0),
                "conflicting_with_existing": counter.get("conflicting_with_existing", 0),
                "primary_assignments_added": sum(1 for row in entries if row["is_primary_retrieval_eligible"]),
                "all_assignments_added": len(entries),
            }
        )
    return rows


def _pattern_family_summary(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    return connection.sql(
        """
        SELECT
            normalized_family,
            COUNT(*) AS pattern_count,
            AVG(core_member_count) AS avg_core_members,
            AVG(total_member_count) AS avg_total_members,
            AVG(coherence_score) AS avg_coherence,
            AVG(confidence_score) AS avg_confidence
        FROM evidence.assignment_patterns
        GROUP BY normalized_family
        ORDER BY normalized_family
        """
    ).df().to_dict(orient="records")


def _same_family_multi_pattern_rows(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    return connection.sql(
        """
        SELECT
            normalized_family,
            pattern_id,
            pattern_label,
            core_member_count,
            total_member_count,
            confidence_score,
            coherence_score
        FROM evidence.assignment_patterns
        WHERE normalized_family IN (
            SELECT normalized_family
            FROM evidence.assignment_patterns
            GROUP BY normalized_family
            HAVING COUNT(*) > 1
        )
        ORDER BY normalized_family, pattern_id
        """
    ).df().to_dict(orient="records")


def _write_reports(
    papers: list[PaperSpec],
    extracted_rows: list[dict],
    interaction_rows: list[dict],
    strengthened_rows: list[dict],
    new_motif_rows: list[dict],
    unresolved_rows: list[dict],
    before_state: dict,
    after_state: dict,
    before_retrieval: list[dict],
    after_retrieval: list[dict],
) -> dict:
    ensure_literature_pilot_output_dirs()

    selection_rows = [
        {
            "paper_id": paper.paper_id,
            "citation_short": paper.citation_short,
            "title": paper.title,
            "source_id": paper.source_id,
            "study_family": paper.study_family,
            "manuscript_path": paper.manuscript_path,
            "biosample_type": paper.biosample_type,
            "modality": paper.modality,
            "disease_class": paper.disease_class,
            "selection_reason": paper.selection_reason,
        }
        for paper in papers
    ]
    interaction_summary_rows = _summarize_interactions(interaction_rows, papers)

    _write_csv(
        LITERATURE_PILOT_TABLES_ROOT / "pilot_paper_selection.csv",
        selection_rows,
        list(selection_rows[0].keys()),
    )
    _write_csv(
        LITERATURE_PILOT_TABLES_ROOT / "extracted_peak_assignments.csv",
        extracted_rows,
        list(extracted_rows[0].keys()),
    )
    _write_csv(
        LITERATURE_PILOT_TABLES_ROOT / "motif_interaction_summary.csv",
        interaction_summary_rows,
        list(interaction_summary_rows[0].keys()),
    )
    _write_csv(
        LITERATURE_PILOT_TABLES_ROOT / "strengthened_motifs.csv",
        strengthened_rows,
        list(strengthened_rows[0].keys()) if strengthened_rows else list(interaction_rows[0].keys()),
    )
    _write_csv(
        LITERATURE_PILOT_TABLES_ROOT / "new_motif_candidates.csv",
        new_motif_rows,
        list(new_motif_rows[0].keys()) if new_motif_rows else list(interaction_rows[0].keys()),
    )
    _write_csv(
        LITERATURE_PILOT_TABLES_ROOT / "unresolved_assignments.csv",
        unresolved_rows,
        list(unresolved_rows[0].keys()) if unresolved_rows else list(interaction_rows[0].keys()),
    )

    before_pattern_counts = before_state["pattern_counts"]
    after_pattern_counts = after_state["pattern_counts"]
    before_after_lines = [
        "# Motif Before/After Comparison",
        "",
        f"- Pre-pilot clusters: `{before_state['cluster_count']}`",
        f"- Post-pilot clusters: `{after_state['cluster_count']}`",
        f"- Pre-pilot patterns: `{before_state['pattern_count']}`",
        f"- Post-pilot patterns: `{after_state['pattern_count']}`",
        "",
        "## Pattern Counts By Family",
        "",
    ]
    families = sorted(set(before_pattern_counts) | set(after_pattern_counts))
    for family in families:
        before_count = before_pattern_counts.get(family, 0)
        after_count = after_pattern_counts.get(family, 0)
        before_after_lines.append(f"- `{family}`: `{before_count} -> {after_count}`")
    before_after_lines.extend(["", "## Paper-Level Interaction Summary", ""])
    for row in interaction_summary_rows:
        before_after_lines.append(
            f"- `{row['citation_short']}`: strengthened `{row['strengthened_existing_motifs']}`, "
            f"extended `{row['extended_existing_motifs']}`, new motif candidates `{row['new_motif_candidates']}`, "
            f"unresolved `{row['unresolved_or_ambiguous']}`, conflicting `{row['conflicting_with_existing']}`."
        )
    (LITERATURE_PILOT_REPORT_ROOT / "motif_before_after_comparison.md").write_text(
        "\n".join(before_after_lines) + "\n",
        encoding="utf-8",
    )

    retrieval_lines = ["# Retrieval Changes After Pilot", ""]
    before_lookup = {row["paper_id"]: row for row in before_retrieval}
    after_lookup = {row["paper_id"]: row for row in after_retrieval}
    for paper in papers:
        before_row = before_lookup[paper.paper_id]
        after_row = after_lookup[paper.paper_id]
        retrieval_lines.append(f"## {paper.citation_short}")
        retrieval_lines.append("")
        retrieval_lines.append(f"- Query peaks: `{before_row['query_peaks']}`")
        retrieval_lines.append(
            "- Before top patterns: "
            + ", ".join(result["pattern_label"] for result in before_row["pattern_results"][:3])
            if before_row["pattern_results"]
            else "- Before top patterns: none"
        )
        retrieval_lines.append(
            "- After top patterns: "
            + ", ".join(result["pattern_label"] for result in after_row["pattern_results"][:3])
            if after_row["pattern_results"]
            else "- After top patterns: none"
        )
        retrieval_lines.append(
            "- After top cluster bundles: "
            + ", ".join(result["title"] for result in after_row["support_bundle_results"][:3])
            if after_row["support_bundle_results"]
            else "- After top cluster bundles: none"
        )
        retrieval_lines.append("")
    (LITERATURE_PILOT_REPORT_ROOT / "retrieval_changes_after_pilot.md").write_text(
        "\n".join(retrieval_lines) + "\n",
        encoding="utf-8",
    )

    implementation_lines = [
        "# Implementation Note",
        "",
        "This pilot reused the existing warehouse and motif stack, then added only three local manuscript sources with explicit peak assignments extracted from figures, captions, and explicit assignment text.",
        "",
        "## What Changed",
        "",
        "- Reused the existing manuscript-level `source_id`s so source diversity remains paper-level rather than extraction-run-level.",
        "- Added one `pilot_literature_digitization` registry row per selected paper in both source registries.",
        "- Inserted a small set of explicit literature assignments into `evidence.peak_assignment_evidence` with provenance to figure, page, extraction method, and confidence.",
        "- Rebuilt the Phase 1 cluster and motif layers from the expanded warehouse.",
        "",
        "## Controlled Handling Rules",
        "",
        "- Explicit figure/caption assignments were promoted to primary support.",
        "- Ambiguous, confounded, or outside-band rows were preserved as non-primary support rows and tracked separately.",
        "- No broad PDF parsing or bulk digitization was run.",
        "",
        "## What Remains Deferred",
        "",
        "- Manual spectral trace digitization from figures.",
        "- Larger literature scaling and paper-family normalization.",
        "- Any molecule-level inference layer.",
        "",
    ]
    (LITERATURE_PILOT_REPORT_ROOT / "implementation_note.md").write_text(
        "\n".join(implementation_lines),
        encoding="utf-8",
    )

    strengthened_count = len({row["after_pattern_id"] for row in strengthened_rows if row["after_pattern_id"]})
    new_motif_count = len(new_motif_rows)
    unresolved_count = len(unresolved_rows)
    conflict_count = sum(1 for row in interaction_rows if row["interaction_status"] == "conflicting_with_existing")
    readiness = "needs refinement"
    if new_motif_count <= 2 and conflict_count <= 2 and strengthened_count >= 6:
        readiness = "working well"

    assessment_lines = [
        "# Pilot Assessment",
        "",
        f"Total extracted assignments: `{len(extracted_rows)}`",
        f"Primary retrieval-eligible assignments added: `{sum(1 for row in extracted_rows if row['is_primary_retrieval_eligible'])}`",
        f"Strengthened/extended motif-linked assignments: `{len(strengthened_rows)}`",
        f"New motif candidate assignments: `{new_motif_count}`",
        f"Unresolved assignments: `{unresolved_count}`",
        f"Conflicting assignments: `{conflict_count}`",
        "",
        "## Critical Readout",
        "",
        f"1. Existing motifs were strengthened meaningfully: `{'yes' if strengthened_count > 0 else 'no'}`.",
        f"2. Missing motifs were revealed: `{'yes' if new_motif_count > 0 else 'no'}`.",
        f"3. Contradictions or context-sensitive mismatches were observed: `{'yes' if conflict_count > 0 else 'no'}`.",
        f"4. Current motif system verdict: `{readiness}`.",
        "",
        "## Top Weaknesses",
        "",
        "- Literature support still enters a reference-heavy scaffold, so family labels remain broad in some windows.",
        "- High-band oxidative or substrate-confounded peaks do not yet have dedicated context-aware motifs.",
        "- Context-sensitive EV vs serum shifts are still attached through metadata rather than explicit motif branching.",
        "",
        "## Scaling Readiness",
        "",
        "The system is not yet ready for uncontrolled literature scaling. It is suitable for another small pilot or a tightly reviewed digitization batch, but motif branching and context-specific handling should be tightened first.",
    ]
    (LITERATURE_PILOT_REPORT_ROOT / "pilot_assessment.md").write_text(
        "\n".join(assessment_lines) + "\n",
        encoding="utf-8",
    )

    build_summary = {
        "papers_selected": [paper.paper_id for paper in papers],
        "total_extracted_assignments": len(extracted_rows),
        "primary_assignments_added": sum(1 for row in extracted_rows if row["is_primary_retrieval_eligible"]),
        "strengthened_assignments": len(strengthened_rows),
        "new_motif_candidates": len(new_motif_rows),
        "unresolved_assignments": len(unresolved_rows),
        "pattern_count_before": before_state["pattern_count"],
        "pattern_count_after": after_state["pattern_count"],
    }
    (LITERATURE_PILOT_REPORT_ROOT / "build_summary.json").write_text(
        json.dumps(build_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return build_summary


def run_three_paper_literature_pilot(db_path: str = str(DB_PATH)) -> dict:
    ensure_literature_pilot_output_dirs()
    papers = pilot_papers()
    assignments = pilot_assignments()
    before_engine = PeakListRetrievalEngine(db_path)
    before_retrieval = _run_retrieval_queries(before_engine, papers)

    with duckdb.connect(db_path) as connection:
        initialize_schema(connection)
        before_state = _snapshot_state(connection)
        _delete_previous_pilot_rows(connection)
        _insert_pilot_registry_rows(connection, papers)
        extracted_rows = _insert_pilot_assignments(connection, papers, assignments)
        reset_phase1_refinement_tables(connection)
        initialize_schema(connection)
        refinement_counts = build_phase1_refinement(connection)
        pattern_counts = build_assignment_patterns(connection)
        after_state = _snapshot_state(connection)
        pattern_family_summary = _pattern_family_summary(connection)
        same_family_multi_pattern_rows = _same_family_multi_pattern_rows(connection)

    after_engine = PeakListRetrievalEngine(db_path)
    after_retrieval = _run_retrieval_queries(after_engine, papers)
    interaction_rows, strengthened_rows, new_motif_rows, unresolved_rows = _classify_interactions(
        assignments=assignments,
        papers=papers,
        before_state=before_state,
        after_state=after_state,
    )
    build_summary = _write_reports(
        papers=papers,
        extracted_rows=extracted_rows,
        interaction_rows=interaction_rows,
        strengthened_rows=strengthened_rows,
        new_motif_rows=new_motif_rows,
        unresolved_rows=unresolved_rows,
        before_state=before_state,
        after_state=after_state,
        before_retrieval=before_retrieval,
        after_retrieval=after_retrieval,
    )
    _write_csv(
        LITERATURE_PILOT_TABLES_ROOT / "pattern_family_summary.csv",
        pattern_family_summary,
        list(pattern_family_summary[0].keys()) if pattern_family_summary else ["normalized_family"],
    )
    _write_csv(
        LITERATURE_PILOT_TABLES_ROOT / "same_family_multi_pattern_examples.csv",
        same_family_multi_pattern_rows,
        list(same_family_multi_pattern_rows[0].keys()) if same_family_multi_pattern_rows else ["normalized_family"],
    )

    return {
        "output_root": str(LITERATURE_PILOT_OUTPUT_ROOT),
        "report_root": str(LITERATURE_PILOT_REPORT_ROOT),
        "tables_root": str(LITERATURE_PILOT_TABLES_ROOT),
        "papers": [asdict(paper) for paper in papers],
        "build_summary": build_summary,
        "refinement_counts": refinement_counts,
        "pattern_counts": pattern_counts,
        "before_state": {
            "cluster_count": before_state["cluster_count"],
            "pattern_count": before_state["pattern_count"],
            "pattern_counts": before_state["pattern_counts"],
        },
        "after_state": {
            "cluster_count": after_state["cluster_count"],
            "pattern_count": after_state["pattern_count"],
            "pattern_counts": after_state["pattern_counts"],
        },
        "before_retrieval": before_retrieval,
        "after_retrieval": after_retrieval,
        "interaction_rows": interaction_rows,
        "strengthened_rows": strengthened_rows,
        "new_motif_rows": new_motif_rows,
        "unresolved_rows": unresolved_rows,
    }
