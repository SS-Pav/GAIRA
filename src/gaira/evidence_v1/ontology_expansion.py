from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    DB_PATH,
    ONTOLOGY_OUTPUT_ROOT,
    ONTOLOGY_REPORT_ROOT,
    ONTOLOGY_TABLES_ROOT,
    ensure_ontology_output_dirs,
)
from gaira.evidence_v1.schema import initialize_schema, reset_ontology_tables


BROADER_FAMILY_REGISTRY = [
    ("protein_support", "protein support", "Protein backbone, amide, and related protein-origin bands.", ""),
    ("nucleic_acid_support", "nucleic-acid support", "Backbone and nucleobase-associated bands.", ""),
    ("amino_acid_support", "amino-acid support", "Amino-acid side-chain and aromatic amino-acid bands.", ""),
    ("lipid_membrane_support", "lipid / membrane support", "Lipid, membrane, CH deformation, and lipid-oxidation bands.", ""),
    ("carbohydrate_support", "carbohydrate support", "Carbohydrate, glycan, and glycogen-related bands.", ""),
    ("metabolite_support", "small-molecule metabolite support", "Low-molecular-weight metabolite signals.", ""),
    ("pigment_cofactor_support", "chromophore / cofactor support", "Pigment and cofactor signals.", ""),
    ("confounder_support", "confounder support", "Substrate, reagent, buffer, and other nonbiological confounders.", ""),
    ("unresolved_support", "unresolved support", "Assignments too mixed or underspecified for stable normalization.", ""),
]

SUBFAMILY_REGISTRY = [
    ("sf_protein_disulfide", "protein_disulfide_support", "protein disulfide support", "protein_support", "biological_signal", "", "existing"),
    ("sf_amide_i", "amide_i_support", "amide I support", "protein_support", "biological_signal", "", "existing"),
    ("sf_amide_iii", "amide_iii_support", "amide III support", "protein_support", "biological_signal", "", "existing"),
    ("sf_protein_backbone", "protein_backbone_skeletal_support", "protein backbone skeletal support", "protein_support", "biological_signal", "", "existing"),
    ("sf_protein_aromatic_sidechain", "protein_aromatic_sidechain_support", "protein aromatic side-chain support", "protein_support", "biological_signal", "", "expanded"),
    ("sf_nucleic_backbone_b", "b_dna_backbone_support", "B-DNA backbone support", "nucleic_acid_support", "biological_signal", "", "expanded"),
    ("sf_nucleic_backbone_a", "rna_a_form_backbone_support", "RNA / A-form backbone support", "nucleic_acid_support", "biological_signal", "", "expanded"),
    ("sf_adenine", "adenine_nucleobase_support", "adenine nucleobase support", "nucleic_acid_support", "biological_signal", "", "expanded"),
    ("sf_guanine", "guanine_nucleobase_support", "guanine nucleobase support", "nucleic_acid_support", "biological_signal", "", "expanded"),
    ("sf_cytosine", "cytosine_nucleobase_support", "cytosine nucleobase support", "nucleic_acid_support", "biological_signal", "", "expanded"),
    ("sf_thymine", "thymine_nucleobase_support", "thymine nucleobase support", "nucleic_acid_support", "biological_signal", "", "expanded"),
    ("sf_nucleic_generic", "nucleic_acid_backbone_generic_support", "generic nucleic-acid backbone support", "nucleic_acid_support", "biological_signal", "", "existing"),
    ("sf_aromatic_generic", "aromatic_ring_support", "aromatic ring support", "amino_acid_support", "biological_signal", "", "existing"),
    ("sf_phenylalanine", "phenylalanine_ring_support", "phenylalanine ring support", "amino_acid_support", "biological_signal", "", "expanded"),
    ("sf_tyrosine", "tyrosine_ring_support", "tyrosine ring support", "amino_acid_support", "biological_signal", "", "expanded"),
    ("sf_tryptophan", "tryptophan_ring_support", "tryptophan ring support", "amino_acid_support", "biological_signal", "", "expanded"),
    ("sf_nonaromatic_aa", "nonaromatic_amino_acid_sidechain_support", "nonaromatic amino-acid side-chain support", "amino_acid_support", "biological_signal", "", "new_promoted"),
    ("sf_lipid_ch", "lipid_ch2_ch3_deformation_support", "lipid CH2/CH3 deformation support", "lipid_membrane_support", "biological_signal", "", "existing"),
    ("sf_lipid_unsat", "lipid_unsaturated_cc_support", "lipid unsaturated C=C support", "lipid_membrane_support", "biological_signal", "", "expanded"),
    ("sf_lipid_oxidation", "lipid_oxidation_carbonyl_support", "lipid oxidation carbonyl support", "lipid_membrane_support", "biological_signal", "", "new_promoted"),
    ("sf_glycogen", "glycogen_support", "glycogen support", "carbohydrate_support", "biological_signal", "", "expanded"),
    ("sf_carbohydrate_generic", "glycan_carbohydrate_support", "glycan / carbohydrate support", "carbohydrate_support", "biological_signal", "", "existing"),
    ("sf_metabolite_generic", "small_molecule_metabolite_support", "small-molecule metabolite support", "metabolite_support", "biological_signal", "", "existing"),
    ("sf_carotene", "beta_carotene_chromophore_support", "beta-carotene chromophore support", "pigment_cofactor_support", "biological_signal", "", "expanded"),
    ("sf_pigment_generic", "chromophore_cofactor_support", "chromophore / cofactor support", "pigment_cofactor_support", "biological_signal", "", "existing"),
    ("sf_citrate_cap", "citrate_capping_agent_signal", "citrate capping-agent signal", "confounder_support", "confounder_signal", "capping_agent", "new_promoted"),
    ("sf_substrate_generic", "substrate_surface_signal", "substrate / surface signal", "confounder_support", "confounder_signal", "substrate_related", "existing"),
    ("sf_highw_ch", "high_wavenumber_ch_stretching_support", "high-wavenumber CH stretching support", "unresolved_support", "unresolved_signal", "", "new_promoted"),
    ("sf_ambiguous_overlap", "ambiguous_multifamily_overlap_support", "ambiguous multifamily overlap support", "unresolved_support", "unresolved_signal", "", "existing"),
    ("sf_unresolved", "unresolved_assignment_support", "unresolved assignment support", "unresolved_support", "unresolved_signal", "", "existing"),
]

ALIAS_RULES = [
    ("citrate on au nanoparticle surface", "citrate_capping_agent_signal", "confounder_support", "confounder_signal", "capping_agent", "exact"),
    ("citrate adsorbed on the gold nanoparticle surface", "citrate_capping_agent_signal", "confounder_support", "confounder_signal", "capping_agent", "exact"),
    ("lysine side-chain stretch", "nonaromatic_amino_acid_sidechain_support", "amino_acid_support", "biological_signal", "", "exact"),
    ("protein disulfide bond", "protein_disulfide_support", "protein_support", "biological_signal", "", "exact"),
    ("disulfide", "protein_disulfide_support", "protein_support", "biological_signal", "", "contains"),
    ("amide iii", "amide_iii_support", "protein_support", "biological_signal", "", "exact"),
    ("amide i", "amide_i_support", "protein_support", "biological_signal", "", "exact"),
    ("amide iii", "amide_iii_support", "protein_support", "biological_signal", "", "contains"),
    ("amide i", "amide_i_support", "protein_support", "biological_signal", "", "contains"),
    ("protein backbone c-calpha", "protein_backbone_skeletal_support", "protein_support", "biological_signal", "", "exact"),
    ("protein backbone c-alpha skeletal mode", "protein_backbone_skeletal_support", "protein_support", "biological_signal", "", "exact"),
    ("protein aromatic side chain", "protein_aromatic_sidechain_support", "protein_support", "biological_signal", "", "exact"),
    ("b-dna backbone", "b_dna_backbone_support", "nucleic_acid_support", "biological_signal", "", "exact"),
    ("rna a-form backbone", "rna_a_form_backbone_support", "nucleic_acid_support", "biological_signal", "", "exact"),
    ("rna / a-form backbone", "rna_a_form_backbone_support", "nucleic_acid_support", "biological_signal", "", "exact"),
    ("adenine marker band", "adenine_nucleobase_support", "nucleic_acid_support", "biological_signal", "", "exact"),
    ("guanine marker band", "guanine_nucleobase_support", "nucleic_acid_support", "biological_signal", "", "exact"),
    ("cytosine marker band", "cytosine_nucleobase_support", "nucleic_acid_support", "biological_signal", "", "exact"),
    ("thymine marker band", "thymine_nucleobase_support", "nucleic_acid_support", "biological_signal", "", "exact"),
    ("phenylalanine ring breathing", "phenylalanine_ring_support", "amino_acid_support", "biological_signal", "", "exact"),
    ("phenylalanine aromatic band", "phenylalanine_ring_support", "amino_acid_support", "biological_signal", "", "exact"),
    ("tyrosine aromatic band", "tyrosine_ring_support", "amino_acid_support", "biological_signal", "", "exact"),
    ("tryptophan aromatic band", "tryptophan_ring_support", "amino_acid_support", "biological_signal", "", "exact"),
    ("aromatic amino-acid c=c stretching", "aromatic_ring_support", "amino_acid_support", "biological_signal", "", "exact"),
    ("aromatic ring / benzene breathing", "aromatic_ring_support", "amino_acid_support", "biological_signal", "", "exact"),
    ("glycogen", "glycogen_support", "carbohydrate_support", "biological_signal", "", "exact"),
    ("beta-carotene c-c stretching", "beta_carotene_chromophore_support", "pigment_cofactor_support", "biological_signal", "", "exact"),
    ("lipid ch2 bending", "lipid_ch2_ch3_deformation_support", "lipid_membrane_support", "biological_signal", "", "exact"),
    ("lipid ch2/ch3 deformation", "lipid_ch2_ch3_deformation_support", "lipid_membrane_support", "biological_signal", "", "exact"),
    ("unsaturated lipid c=c stretching", "lipid_unsaturated_cc_support", "lipid_membrane_support", "biological_signal", "", "exact"),
    ("lipid oxidation carbonyl marker", "lipid_oxidation_carbonyl_support", "lipid_membrane_support", "biological_signal", "", "exact"),
    ("high_wavenumber_ch", "high_wavenumber_ch_stretching_support", "unresolved_support", "unresolved_signal", "", "synthetic"),
    ("ambiguous_multifamily", "ambiguous_multifamily_overlap_support", "unresolved_support", "unresolved_signal", "", "synthetic"),
]


@dataclass
class OntologyMapping:
    evidence_item_id: str
    source_id: str
    assignment_record_id: str
    exact_source_phrase: str
    normalized_subfamily: str
    broader_family: str
    meaning_class: str
    confounder_subclass: str
    spectral_region: str
    mapping_status: str
    alias_text_used: str
    notes: str


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _spectral_region(peak_center_cm: float | None) -> str:
    if peak_center_cm is None:
        return "other_region"
    if 400.0 <= peak_center_cm <= 1800.0:
        return "fingerprint_400_1800"
    if 1700.0 < peak_center_cm <= 1900.0:
        return "carbonyl_1700_1900"
    if 2800.0 <= peak_center_cm <= 3200.0:
        return "high_wavenumber_2800_3200"
    return "other_region"


def _subfamily_lookup() -> dict[str, dict]:
    return {
        row[1]: {
            "subfamily_id": row[0],
            "normalized_subfamily_label": row[2],
            "broader_family": row[3],
            "meaning_class": row[4],
            "default_confounder_subclass": row[5],
            "status": row[6],
        }
        for row in SUBFAMILY_REGISTRY
    }


def _alias_rows() -> list[tuple]:
    rows = []
    for index, (alias_text, normalized_subfamily, broader_family, meaning_class, confounder_subclass, mapping_type) in enumerate(ALIAS_RULES, start=1):
        rows.append(
            (
                f"alias_{index:03d}",
                alias_text,
                normalized_subfamily,
                broader_family,
                meaning_class,
                confounder_subclass,
                mapping_type,
                "phase1_ontology_expansion_v1",
            )
        )
    return rows


def _registry_rows() -> tuple[list[tuple], list[tuple]]:
    broader_rows = [(row[0], row[1], row[2], row[3]) for row in BROADER_FAMILY_REGISTRY]
    subfamily_rows = [(row[0], row[1], row[2], row[3], row[4], row[5], row[6], "") for row in SUBFAMILY_REGISTRY]
    return broader_rows, subfamily_rows


def _extract_exact_phrase(assigned_molecule: str, assigned_group: str, evidence_text: str) -> str:
    if _normalize_text(assigned_group):
        return _normalize_text(assigned_group)
    if _normalize_text(assigned_molecule):
        return _normalize_text(assigned_molecule)
    lowered = _normalize_text(evidence_text)
    for token in ("assigned to ", "listed as ", "attributed to ", "described as "):
        if token in lowered.lower():
            start = lowered.lower().index(token) + len(token)
            candidate = lowered[start:].split(".")[0].strip(" ;,")
            if candidate:
                return candidate
    return lowered[:160]


def _match_alias(exact_phrase: str, evidence_text: str) -> tuple[str, str, str, str, str] | None:
    phrase = _normalize_text(exact_phrase).lower()
    text = _normalize_text(evidence_text).lower()
    for alias_text, normalized_subfamily, broader_family, meaning_class, confounder_subclass, mapping_type in ALIAS_RULES:
        alias = alias_text.lower()
        if mapping_type == "exact" and phrase == alias:
            return normalized_subfamily, broader_family, meaning_class, confounder_subclass, alias_text
        if mapping_type == "contains" and (alias in phrase or alias in text):
            return normalized_subfamily, broader_family, meaning_class, confounder_subclass, alias_text
    return None


def _heuristic_mapping(exact_phrase: str, assigned_molecule: str, assigned_group: str, evidence_text: str, peak_center_cm: float) -> tuple[str, str, str, str, str]:
    phrase = " ".join(filter(None, [_normalize_text(exact_phrase), _normalize_text(assigned_molecule), _normalize_text(assigned_group), _normalize_text(evidence_text)])).lower()
    region = _spectral_region(peak_center_cm)
    if any(term in phrase for term in ("lipids, proteins, and dna", "nucleic-acid/protein", "comparison context", "multifamily", "mixed band")):
        return "ambiguous_multifamily_overlap_support", "unresolved_support", "unresolved_signal", "", "heuristic_ambiguous"
    if any(term in phrase for term in ("gold nanoparticle", "aunp", "substrate", "surface", "capping")):
        return "substrate_surface_signal", "confounder_support", "confounder_signal", "substrate_related", "heuristic_confounder"
    if "citrate" in phrase and any(term in phrase for term in ("surface", "au", "gold", "nanoparticle")):
        return "citrate_capping_agent_signal", "confounder_support", "confounder_signal", "capping_agent", "heuristic_citrate"
    if "lysine" in phrase or any(term in phrase for term in ("valine", "alanine", "glycine", "serine", "arginine", "glutamate", "aspartic")):
        return "nonaromatic_amino_acid_sidechain_support", "amino_acid_support", "biological_signal", "", "heuristic_nonaromatic_aa"
    if any(term in phrase for term in ("phenylalanine", "phenyl")):
        return "phenylalanine_ring_support", "amino_acid_support", "biological_signal", "", "heuristic_phe"
    if "tyrosine" in phrase:
        return "tyrosine_ring_support", "amino_acid_support", "biological_signal", "", "heuristic_tyr"
    if "tryptophan" in phrase or "trp" in phrase:
        return "tryptophan_ring_support", "amino_acid_support", "biological_signal", "", "heuristic_trp"
    if any(term in phrase for term in ("adenine", "guanine", "cytosine", "thymine")):
        mapping = {
            "adenine": "adenine_nucleobase_support",
            "guanine": "guanine_nucleobase_support",
            "cytosine": "cytosine_nucleobase_support",
            "thymine": "thymine_nucleobase_support",
        }
        for term, subfamily in mapping.items():
            if term in phrase:
                return subfamily, "nucleic_acid_support", "biological_signal", "", f"heuristic_{term}"
    if "b-dna" in phrase:
        return "b_dna_backbone_support", "nucleic_acid_support", "biological_signal", "", "heuristic_bdna"
    if "rna" in phrase or "a-form" in phrase:
        return "rna_a_form_backbone_support", "nucleic_acid_support", "biological_signal", "", "heuristic_rna"
    if any(term in phrase for term in ("amide i", "collagen", "albumin")):
        return "amide_i_support", "protein_support", "biological_signal", "", "heuristic_amide_i"
    if "amide iii" in phrase:
        return "amide_iii_support", "protein_support", "biological_signal", "", "heuristic_amide_iii"
    if any(term in phrase for term in ("disulfide", "s-s")):
        return "protein_disulfide_support", "protein_support", "biological_signal", "", "heuristic_disulfide"
    if any(term in phrase for term in ("protein backbone", "cα", "c-alpha", "skeletal")):
        return "protein_backbone_skeletal_support", "protein_support", "biological_signal", "", "heuristic_backbone"
    if "glycogen" in phrase:
        return "glycogen_support", "carbohydrate_support", "biological_signal", "", "heuristic_glycogen"
    if any(term in phrase for term in ("carbohydrate", "glycan", "saccharide", "glucose", "fructose", "mannose", "ribose")):
        return "glycan_carbohydrate_support", "carbohydrate_support", "biological_signal", "", "heuristic_carb"
    if "beta-carotene" in phrase or "carotene" in phrase:
        return "beta_carotene_chromophore_support", "pigment_cofactor_support", "biological_signal", "", "heuristic_carotene"
    if any(term in phrase for term in ("riboflavin", "biliverdin", "heme", "porphyrin", "chromophore", "cofactor")):
        return "chromophore_cofactor_support", "pigment_cofactor_support", "biological_signal", "", "heuristic_pigment"
    if region == "carbonyl_1700_1900" and any(term in phrase for term in ("oxidation", "aldehyde", "anhydride", "carbonyl")):
        return "lipid_oxidation_carbonyl_support", "lipid_membrane_support", "biological_signal", "", "heuristic_oxidation"
    if region == "high_wavenumber_2800_3200":
        return "high_wavenumber_ch_stretching_support", "unresolved_support", "unresolved_signal", "", "heuristic_highw"
    if any(term in phrase for term in ("lipid", "membrane", "ch2", "ch3")):
        return "lipid_ch2_ch3_deformation_support", "lipid_membrane_support", "biological_signal", "", "heuristic_lipid_ch"
    if any(term in phrase for term in ("unsaturated", "c=c")):
        return "lipid_unsaturated_cc_support", "lipid_membrane_support", "biological_signal", "", "heuristic_lipid_unsat"
    if any(term in phrase for term in ("metabolite", "citrate", "pyruvate", "fumarate", "succinate", "acetoacetate", "phosphoenolpyruvate", "pep")):
        return "small_molecule_metabolite_support", "metabolite_support", "biological_signal", "", "heuristic_metabolite"
    if any(term in phrase for term in ("aromatic", "benzene", "ring breathing")):
        return "aromatic_ring_support", "amino_acid_support", "biological_signal", "", "heuristic_aromatic"
    return "unresolved_assignment_support", "unresolved_support", "unresolved_signal", "", "fallback_unresolved"


def _candidate_from_phrase(exact_phrase: str, broader_family: str, meaning_class: str, confounder_subclass: str) -> tuple[str, str]:
    base = _slug(exact_phrase) or "unnamed_phrase"
    if meaning_class == "confounder_signal":
        return f"{base}_signal", "candidate exact phrase suggests confounder-specific normalization."
    if broader_family == "lipid_membrane_support" and "carbonyl" in base:
        return "lipid_oxidation_carbonyl_support", "Pilot carbonyl band phrase justifies a lipid-oxidation subfamily."
    if broader_family == "amino_acid_support" and "lysine" in base:
        return "nonaromatic_amino_acid_sidechain_support", "Pilot lysine case justifies separating nonaromatic amino-acid side-chain support."
    if "ch_stretch" in base or "c_h_stretch" in base:
        return "high_wavenumber_ch_stretching_support", "High-wavenumber CH stretching needs a region-specific unresolved subfamily."
    return f"{base}_support", "Exact phrase did not map cleanly to an existing reusable subfamily."


def _fetch_assignment_rows(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = connection.sql(
        """
        SELECT
            p.evidence_item_id,
            p.source_id,
            p.assignment_record_id,
            p.peak_center_cm,
            p.assigned_molecule,
            p.assigned_group_or_theme,
            p.evidence_text,
            p.extraction_method,
            p.notes
        FROM evidence.peak_assignment_evidence p
        ORDER BY p.evidence_item_id
        """
    ).fetchall()
    structured_rows = [
        {
            "evidence_item_id": row[0],
            "source_id": row[1],
            "assignment_record_id": row[2],
            "peak_center_cm": float(row[3]) if row[3] is not None else None,
            "assigned_molecule": _normalize_text(row[4]),
            "assigned_group_or_theme": _normalize_text(row[5]),
            "evidence_text": _normalize_text(row[6]),
            "extraction_method": _normalize_text(row[7]),
            "notes": _normalize_text(row[8]),
        }
        for row in rows
    ]
    reference_rows = connection.sql(
        """
        SELECT
            f.evidence_item_id,
            f.source_id,
            f.feature_id AS assignment_record_id,
            f.peak_center_cm,
            r.component AS assigned_molecule,
            r.biochemical_class AS assigned_group_or_theme,
            COALESCE(r.component, '') || ' reference peak from RamanBioLib' AS evidence_text,
            'reference_peak' AS extraction_method,
            COALESCE(r.source_origin, '') AS notes
        FROM features.spectral_features f
        JOIN evidence.reference_spectrum_evidence r
          ON r.evidence_item_id = f.evidence_item_id
        WHERE f.feature_origin = 'ramanbiolib_peak'
        ORDER BY f.feature_id
        """
    ).fetchall()
    structured_rows.extend(
        {
            "evidence_item_id": row[0],
            "source_id": row[1],
            "assignment_record_id": row[2],
            "peak_center_cm": float(row[3]) if row[3] is not None else None,
            "assigned_molecule": _normalize_text(row[4]),
            "assigned_group_or_theme": _normalize_text(row[5]),
            "evidence_text": _normalize_text(row[6]),
            "extraction_method": _normalize_text(row[7]),
            "notes": _normalize_text(row[8]),
        }
        for row in reference_rows
    )
    return structured_rows


def build_ontology_mappings(connection: duckdb.DuckDBPyConnection) -> dict:
    initialize_schema(connection)
    reset_ontology_tables(connection)
    broader_rows, subfamily_rows = _registry_rows()
    connection.executemany("INSERT INTO ontology.broader_family_registry VALUES (?, ?, ?, ?)", broader_rows)
    connection.executemany("INSERT INTO ontology.normalized_subfamily_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?)", subfamily_rows)
    alias_rows = _alias_rows()
    connection.executemany("INSERT INTO ontology.alias_mapping VALUES (?, ?, ?, ?, ?, ?, ?, ?)", alias_rows)

    subfamily_lookup = _subfamily_lookup()
    assignment_rows = _fetch_assignment_rows(connection)
    evidence_mappings: list[OntologyMapping] = []
    candidate_counter: Counter[tuple[str, str, str, str, str]] = Counter()
    candidate_sources: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)

    for row in assignment_rows:
        exact_phrase = _extract_exact_phrase(row["assigned_molecule"], row["assigned_group_or_theme"], row["evidence_text"])
        alias_match = _match_alias(exact_phrase, row["evidence_text"])
        if alias_match is not None:
            normalized_subfamily, broader_family, meaning_class, confounder_subclass, alias_text = alias_match
            mapping_status = "reused_alias"
        else:
            normalized_subfamily, broader_family, meaning_class, confounder_subclass, alias_text = _heuristic_mapping(
                exact_phrase,
                row["assigned_molecule"],
                row["assigned_group_or_theme"],
                row["evidence_text"],
                row["peak_center_cm"],
            )
            mapping_status = "heuristic_reuse" if normalized_subfamily in subfamily_lookup else "new_subfamily_candidate"

        if normalized_subfamily not in subfamily_lookup:
            proposed_subfamily, rationale = _candidate_from_phrase(
                exact_phrase=exact_phrase,
                broader_family=broader_family,
                meaning_class=meaning_class,
                confounder_subclass=confounder_subclass,
            )
            normalized_subfamily = proposed_subfamily
            mapping_status = "new_subfamily_candidate"
            alias_text = alias_text or rationale
            key = (exact_phrase, normalized_subfamily, broader_family, meaning_class, confounder_subclass)
            candidate_counter[key] += 1
            candidate_sources[key].add(row["source_id"])

        evidence_mappings.append(
            OntologyMapping(
                evidence_item_id=row["evidence_item_id"],
                source_id=row["source_id"],
                assignment_record_id=row["assignment_record_id"],
                exact_source_phrase=exact_phrase,
                normalized_subfamily=normalized_subfamily,
                broader_family=broader_family,
                meaning_class=meaning_class,
                confounder_subclass=confounder_subclass,
                spectral_region=_spectral_region(row["peak_center_cm"]),
                mapping_status=mapping_status,
                alias_text_used=alias_text,
                notes=row["notes"],
            )
        )

    connection.executemany(
        "INSERT INTO ontology.evidence_ontology_mappings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                item.evidence_item_id,
                item.source_id,
                item.assignment_record_id,
                item.exact_source_phrase,
                item.normalized_subfamily,
                item.broader_family,
                item.meaning_class,
                item.confounder_subclass,
                item.spectral_region,
                item.mapping_status,
                item.alias_text_used,
                item.notes,
            )
            for item in evidence_mappings
        ],
    )

    candidate_rows = []
    for index, ((exact_phrase, proposed_normalized_subfamily, proposed_broader_family, meaning_class, confounder_subclass), count) in enumerate(sorted(candidate_counter.items()), start=1):
        candidate_rows.append(
            (
                f"cand_{index:03d}",
                exact_phrase,
                proposed_normalized_subfamily,
                proposed_broader_family,
                meaning_class,
                confounder_subclass,
                next(item.spectral_region for item in evidence_mappings if item.exact_source_phrase == exact_phrase and item.normalized_subfamily == proposed_normalized_subfamily),
                count,
                json.dumps(sorted(candidate_sources[(exact_phrase, proposed_normalized_subfamily, proposed_broader_family, meaning_class, confounder_subclass)])),
                "candidate",
                "Phrase did not hit the curated alias map and should be reviewed before registry promotion.",
            )
        )
    if candidate_rows:
        connection.executemany("INSERT INTO ontology.new_subfamily_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", candidate_rows)

    cluster_rows = connection.sql(
        """
        SELECT cluster_id, linked_evidence_ids_json, canonical_peak_cm
        FROM evidence.peak_meaning_clusters
        ORDER BY cluster_id
        """
    ).fetchall()
    evidence_lookup = {item.evidence_item_id: item for item in evidence_mappings}
    cluster_mapping_rows = []
    for cluster_id, linked_evidence_ids_json, canonical_peak_cm in cluster_rows:
        linked_ids = json.loads(linked_evidence_ids_json)
        mapped = [evidence_lookup[eid] for eid in linked_ids if eid in evidence_lookup]
        if not mapped:
            continue
        subfamily_counts = Counter(item.normalized_subfamily for item in mapped)
        broader_counts = Counter(item.broader_family for item in mapped)
        meaning_counts = Counter(item.meaning_class for item in mapped)
        confounder_counts = Counter(item.confounder_subclass for item in mapped if item.confounder_subclass)
        region = _spectral_region(float(canonical_peak_cm))
        dominant_subfamily = sorted(subfamily_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        dominant_broader = sorted(broader_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        dominant_meaning = sorted(meaning_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        dominant_confounder = sorted(confounder_counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if confounder_counts else ""
        cluster_mapping_rows.append(
            (
                cluster_id,
                dominant_subfamily,
                dominant_broader,
                dominant_meaning,
                dominant_confounder,
                region,
                len(mapped),
                len({item.evidence_item_id for item in mapped}),
                json.dumps(dict(sorted(subfamily_counts.items()))),
                "Dominant ontology mapping aggregated from linked evidence items.",
            )
        )
    if cluster_mapping_rows:
        connection.executemany("INSERT INTO ontology.cluster_ontology_mappings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", cluster_mapping_rows)

    pattern_rows = connection.sql(
        """
        SELECT ap.pattern_id, apm.cluster_id, apm.canonical_peak_cm
        FROM evidence.assignment_patterns ap
        JOIN evidence.assignment_pattern_members apm ON ap.pattern_id = apm.pattern_id
        ORDER BY ap.pattern_id, apm.cluster_id
        """
    ).fetchall()
    cluster_lookup = {row[0]: row for row in cluster_mapping_rows}
    pattern_grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for pattern_id, cluster_id, canonical_peak_cm in pattern_rows:
        pattern_grouped[pattern_id].append((cluster_id, float(canonical_peak_cm)))
    pattern_mapping_rows = []
    for pattern_id, members in pattern_grouped.items():
        mapped_clusters = [cluster_lookup[cluster_id] for cluster_id, _ in members if cluster_id in cluster_lookup]
        if not mapped_clusters:
            continue
        subfamily_counts = Counter(item[1] for item in mapped_clusters)
        broader_counts = Counter(item[2] for item in mapped_clusters)
        meaning_counts = Counter(item[3] for item in mapped_clusters)
        confounder_counts = Counter(item[4] for item in mapped_clusters if item[4])
        region_counts = Counter(_spectral_region(peak) for _, peak in members)
        dominant_subfamily = sorted(subfamily_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        dominant_broader = sorted(broader_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        dominant_meaning = sorted(meaning_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        dominant_confounder = sorted(confounder_counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if confounder_counts else ""
        dominant_region = sorted(region_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        pattern_mapping_rows.append(
            (
                pattern_id,
                dominant_subfamily,
                dominant_broader,
                dominant_meaning,
                dominant_confounder,
                dominant_region,
                len(members),
                len(mapped_clusters),
                json.dumps(dict(sorted(subfamily_counts.items()))),
                "Dominant ontology mapping aggregated from mapped member clusters.",
            )
        )
    if pattern_mapping_rows:
        connection.executemany("INSERT INTO ontology.pattern_ontology_mappings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", pattern_mapping_rows)

    return {
        "evidence_mappings": len(evidence_mappings),
        "new_subfamily_candidates": len(candidate_rows),
        "cluster_mappings": len(cluster_mapping_rows),
        "pattern_mappings": len(pattern_mapping_rows),
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_ontology_reports(connection: duckdb.DuckDBPyConnection) -> dict:
    ensure_ontology_output_dirs()
    field_summary = connection.sql(
        """
        SELECT 'evidence' AS layer, COUNT(*) AS row_count,
               COUNT(DISTINCT exact_source_phrase) AS distinct_exact_source_phrases,
               COUNT(DISTINCT normalized_subfamily) AS distinct_normalized_subfamilies,
               COUNT(DISTINCT broader_family) AS distinct_broader_families,
               COUNT(DISTINCT meaning_class) AS distinct_meaning_classes,
               COUNT(DISTINCT spectral_region) AS distinct_spectral_regions
        FROM ontology.evidence_ontology_mappings
        UNION ALL
        SELECT 'cluster' AS layer, COUNT(*) AS row_count,
               0 AS distinct_exact_source_phrases,
               COUNT(DISTINCT normalized_subfamily) AS distinct_normalized_subfamilies,
               COUNT(DISTINCT broader_family) AS distinct_broader_families,
               COUNT(DISTINCT meaning_class) AS distinct_meaning_classes,
               COUNT(DISTINCT spectral_region) AS distinct_spectral_regions
        FROM ontology.cluster_ontology_mappings
        UNION ALL
        SELECT 'pattern' AS layer, COUNT(*) AS row_count,
               0 AS distinct_exact_source_phrases,
               COUNT(DISTINCT normalized_subfamily) AS distinct_normalized_subfamilies,
               COUNT(DISTINCT broader_family) AS distinct_broader_families,
               COUNT(DISTINCT meaning_class) AS distinct_meaning_classes,
               COUNT(DISTINCT spectral_region) AS distinct_spectral_regions
        FROM ontology.pattern_ontology_mappings
        """
    ).df().to_dict(orient="records")
    _write_csv(ONTOLOGY_TABLES_ROOT / "ontology_field_summary.csv", field_summary, list(field_summary[0].keys()))

    subfamily_registry = connection.sql(
        "SELECT * FROM ontology.normalized_subfamily_registry ORDER BY normalized_subfamily"
    ).df().to_dict(orient="records")
    _write_csv(ONTOLOGY_TABLES_ROOT / "normalized_subfamily_registry.csv", subfamily_registry, list(subfamily_registry[0].keys()))

    broader_registry = connection.sql(
        "SELECT * FROM ontology.broader_family_registry ORDER BY broader_family"
    ).df().to_dict(orient="records")
    _write_csv(ONTOLOGY_TABLES_ROOT / "broader_family_registry.csv", broader_registry, list(broader_registry[0].keys()))

    alias_mapping = connection.sql(
        "SELECT * FROM ontology.alias_mapping ORDER BY alias_id"
    ).df().to_dict(orient="records")
    _write_csv(ONTOLOGY_TABLES_ROOT / "ontology_alias_mapping.csv", alias_mapping, list(alias_mapping[0].keys()))

    candidate_rows = connection.sql(
        "SELECT * FROM ontology.new_subfamily_candidates ORDER BY candidate_id"
    ).df().to_dict(orient="records")
    _write_csv(
        ONTOLOGY_TABLES_ROOT / "new_subfamily_candidates.csv",
        candidate_rows,
        list(candidate_rows[0].keys()) if candidate_rows else ["candidate_id"],
    )

    confounder_summary = connection.sql(
        """
        SELECT confounder_subclass, COUNT(*) AS evidence_rows
        FROM ontology.evidence_ontology_mappings
        WHERE confounder_subclass <> ''
        GROUP BY confounder_subclass
        ORDER BY evidence_rows DESC, confounder_subclass
        """
    ).df().to_dict(orient="records")
    _write_csv(
        ONTOLOGY_TABLES_ROOT / "confounder_class_summary.csv",
        confounder_summary,
        list(confounder_summary[0].keys()) if confounder_summary else ["confounder_subclass", "evidence_rows"],
    )

    spectral_region_summary = connection.sql(
        """
        SELECT spectral_region, meaning_class, COUNT(*) AS evidence_rows
        FROM ontology.evidence_ontology_mappings
        GROUP BY spectral_region, meaning_class
        ORDER BY spectral_region, meaning_class
        """
    ).df().to_dict(orient="records")
    _write_csv(ONTOLOGY_TABLES_ROOT / "spectral_region_summary.csv", spectral_region_summary, list(spectral_region_summary[0].keys()))

    pilot_reclass = connection.sql(
        """
        SELECT
            e.assignment_record_id,
            e.exact_source_phrase,
            e.normalized_subfamily,
            e.broader_family,
            e.meaning_class,
            e.confounder_subclass,
            e.spectral_region,
            e.mapping_status
        FROM ontology.evidence_ontology_mappings e
        WHERE e.assignment_record_id LIKE 'pilot3p_%'
        ORDER BY e.assignment_record_id
        """
    ).df().to_dict(orient="records")
    _write_csv(ONTOLOGY_TABLES_ROOT / "pilot_case_reclassification_summary.csv", pilot_reclass, list(pilot_reclass[0].keys()))

    before_after_unresolved = connection.sql(
        """
        SELECT
            e.assignment_record_id,
            e.exact_source_phrase,
            CASE
                WHEN e.assignment_record_id IN (
                    'pilot3p_liu_004','pilot3p_liu_008','pilot3p_liu_009',
                    'pilot3p_liu_010','pilot3p_liu_011','pilot3p_cca_002'
                ) THEN 'previously_unresolved_in_pilot'
                ELSE 'not_previously_flagged'
            END AS previous_status,
            e.normalized_subfamily,
            e.broader_family,
            e.meaning_class,
            e.confounder_subclass,
            e.spectral_region,
            e.mapping_status
        FROM ontology.evidence_ontology_mappings e
        WHERE e.assignment_record_id IN (
            'pilot3p_liu_004','pilot3p_liu_008','pilot3p_liu_009',
            'pilot3p_liu_010','pilot3p_liu_011','pilot3p_cca_002'
        )
        ORDER BY e.assignment_record_id
        """
    ).df().to_dict(orient="records")
    _write_csv(ONTOLOGY_TABLES_ROOT / "before_after_unresolved_cases.csv", before_after_unresolved, list(before_after_unresolved[0].keys()))

    implementation_note = "\n".join(
        [
            "# Implementation Note",
            "",
            "This pass adds an ontology layer on top of the existing evidence warehouse rather than replacing the current family fields.",
            "",
            "## Ontology Levels",
            "",
            "- `exact_source_phrase`: preserved from the assignment phrase or the closest explicit source wording.",
            "- `normalized_subfamily`: reusable ontology unit for stable interpretation growth.",
            "- `broader_family`: larger semantic bucket kept separate from the subfamily.",
            "",
            "## Added Semantics",
            "",
            "- `meaning_class`: `biological_signal`, `confounder_signal`, or `unresolved_signal`.",
            "- `confounder_subclass`: explicit nonbiological categories such as `capping_agent` and `substrate_related`.",
            "- `spectral_region`: fingerprint, carbonyl, high-wavenumber, or other.",
            "",
            "## Pilot Edge Cases",
            "",
            "- Lysine was promoted from unresolved handling into `nonaromatic_amino_acid_sidechain_support`.",
            "- Citrate on AuNPs was moved into explicit confounder handling as `citrate_capping_agent_signal`.",
            "- 2913 cm^-1 CH stretching remains unresolved but is now region-aware via `high_wavenumber_ch_stretching_support`.",
            "- 1860/1930 cm^-1 lipid-oxidation bands are now represented as `lipid_oxidation_carbonyl_support` in the carbonyl region.",
            "",
            "## Remaining Constraint",
            "",
            "The motif layer still uses the pre-existing family-level clustering logic; this pass makes ontology growth auditable first, before tightening motif rebuilding around the new ontology fields.",
            "",
        ]
    )
    (ONTOLOGY_REPORT_ROOT / "implementation_note.md").write_text(implementation_note, encoding="utf-8")

    assessment = "\n".join(
        [
            "# Current State Assessment",
            "",
            "The warehouse is now materially cleaner for future literature scaling because exact source wording, reusable subfamilies, broader families, confounders, and spectral regions are separated explicitly.",
            "",
            "## Ready Now",
            "",
            "- Exact phrases are preserved without forcing them to become ontology labels.",
            "- Vocabulary growth is auditable through alias mappings and candidate tables.",
            "- Confounders and non-fingerprint regions are represented explicitly rather than being buried inside broad biological families.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Some current patterns still aggregate at the older family layer rather than the new subfamily layer.",
            "- A small number of low-information grounding labels remain broad by design to avoid ontology junk.",
            "- Candidate review is still manual; no approval workflow exists yet.",
            "",
            "## Scaling Readiness",
            "",
            "The ontology backbone is ready for larger but still supervised literature scaling. The next safe step is to route future paper assignments through this ontology layer before letting them influence motif branching.",
            "",
        ]
    )
    (ONTOLOGY_REPORT_ROOT / "current_state_assessment.md").write_text(assessment, encoding="utf-8")

    return {
        "field_summary_rows": len(field_summary),
        "normalized_subfamilies": len(subfamily_registry),
        "broader_families": len(broader_registry),
        "new_subfamily_candidates": len(candidate_rows),
    }


def run_ontology_expansion(db_path: str = str(DB_PATH)) -> dict:
    ensure_ontology_output_dirs()
    with duckdb.connect(db_path) as connection:
        counts = build_ontology_mappings(connection)
        report_counts = generate_ontology_reports(connection)
    return {
        "output_root": str(ONTOLOGY_OUTPUT_ROOT),
        "tables_root": str(ONTOLOGY_TABLES_ROOT),
        "report_root": str(ONTOLOGY_REPORT_ROOT),
        "counts": counts,
        "report_counts": report_counts,
    }
