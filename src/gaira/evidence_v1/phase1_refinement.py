from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass

import duckdb


CLUSTER_MERGE_TOLERANCE_CM = 12.0
FAMILY_OVERLAP_TOLERANCE_CM = 6.0
MENTION_ALIGN_TOLERANCE_CM = 8.0

SUPPORT_KIND_WEIGHTS = {
    "curated_assignment": 1.0,
    "source_backed_assignment": 0.8,
    "reference_grounding_assignment": 0.65,
    "serum_grounding_assignment": 0.58,
    "reference_peak": 0.35,
    "aligned_mention": 0.08,
}

DOMAIN_PRIORITY = ["ev", "serum", "plasma", "pathogen", "biofluids", "reference", "generic"]
MODALITY_PRIORITY = ["sers", "raman", "ec-sers", "unknown"]
FAMILY_LABEL_VERSION = "phase1_cleanup_audit_v1"

ASSIGNMENT_CUE_PHRASES = (
    "assigned to",
    "attributed to",
    "associated with",
    "corresponding to",
    "characteristic of",
    "indicative of",
    "linked to",
    "related to",
    "due to",
    "arises from",
)

FAMILY_LABELS = {
    "protein_amide_support": "protein backbone / amide support",
    "nucleic_acid_base_support": "nucleic-acid base / backbone support",
    "amino_acid_aromatic_support": "aromatic amino-acid support",
    "lipid_membrane_support": "lipid / membrane support",
    "carbohydrate_glycan_support": "carbohydrate / glycan support",
    "small_molecule_metabolite_support": "small-molecule metabolite support",
    "pigment_cofactor_support": "chromophore / cofactor support",
    "unresolved_assignment_support": "unresolved assignment support",
}

PREVIOUS_FAMILY_LABELS = {
    "protein_amide_support": "protein / amide support",
    "nucleic_acid_base_support": "nucleic acid / base support",
    "amino_acid_aromatic_support": "amino acid / aromatic support",
    "lipid_membrane_support": "lipid / membrane support",
    "carbohydrate_glycan_support": "carbohydrate / glycan support",
    "small_molecule_metabolite_support": "small-molecule metabolite support",
    "pigment_cofactor_support": "pigment / cofactor support",
    "unresolved_assignment_support": "unresolved assignment support",
}


@dataclass
class SupportRecord:
    support_id: str
    evidence_item_id: str
    source_id: str
    source_type: str
    support_kind: str
    support_role: str
    peak_center_cm: float
    normalized_family: str
    normalized_meaning_label: str
    raw_label: str
    domain_hint: str
    modality_hint: str
    study_family: str
    weight: float
    provenance: dict
    note: str


@dataclass
class ClusterRecord:
    cluster_id: str
    normalized_family: str
    normalized_meaning_label: str
    supports: list[SupportRecord]


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _safe_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _domain_from_text(*values: str | None) -> str:
    text = " ".join(_normalize_text(value).lower() for value in values)
    if "extracellular vesicle" in text or " exosome" in text or "ev" in text:
        return "ev"
    if "serum" in text:
        return "serum"
    if "plasma" in text:
        return "plasma"
    if "pathogen" in text or "bacteria" in text or "virus" in text:
        return "pathogen"
    if "biofluid" in text:
        return "biofluids"
    if "pure_biomolecule" in text or "reference" in text:
        return "reference"
    return "generic"


def _modality_from_text(*values: str | None) -> str:
    text = " ".join(_normalize_text(value).lower() for value in values)
    if "ec-sers" in text or "ec_sers" in text:
        return "ec-sers"
    if "sers" in text:
        return "sers"
    if "raman" in text:
        return "raman"
    return "unknown"


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def normalize_meaning_family(*values: str | None) -> tuple[str, str]:
    text = " ".join(_normalize_text(value).lower() for value in values if _normalize_text(value))
    if _contains_any(
        text,
        (
            "protein",
            "amide",
            "keratin",
            "insulin",
            "collagen",
            "elastin",
            "disulfide",
            "peptide",
            "globin",
            "albumin",
            "trypsin",
            "triosephosphate isomerase",
            "enzyme",
        ),
    ):
        return "protein_amide_support", FAMILY_LABELS["protein_amide_support"]
    if _contains_any(
        text,
        (
            "nucleic",
            "adenine",
            "guanine",
            "cytosine",
            "thymine",
            "uracil",
            "dna",
            "rna",
            "b-dna",
            "a-dna",
            "base",
            "backbone",
            "deoxythymi",
        ),
    ):
        return "nucleic_acid_base_support", FAMILY_LABELS["nucleic_acid_base_support"]
    if _contains_any(
        text,
        (
            "amino acid",
            "aminoacid",
            "tryptophan",
            "tyrosine",
            "phenyl",
            "aromatic",
            "valine",
            "histidine",
            "proline",
            "serine",
            "arginine",
            "glutamate",
            "aspartic",
            "glycine",
            "alanine",
        ),
    ):
        return "amino_acid_aromatic_support", FAMILY_LABELS["amino_acid_aromatic_support"]
    if _contains_any(
        text,
        (
            "lipid",
            "fatty",
            "phospholipid",
            "sterol",
            "hormone",
            "membrane",
            "triglyceride",
            "sphingomyelin",
            "cholesterol",
            "ch2",
            "ch3",
        ),
    ):
        return "lipid_membrane_support", FAMILY_LABELS["lipid_membrane_support"]
    if _contains_any(
        text,
        (
            "carbohydrate",
            "saccharide",
            "glycan",
            "glucose",
            "fructose",
            "mannose",
            "xylose",
            "ribose",
            "cellulose",
            "glycogen",
            "chitin",
            "monosaccharide",
            "polysaccharide",
            "disaccharide",
            "trisaccharide",
        ),
    ):
        return "carbohydrate_glycan_support", FAMILY_LABELS["carbohydrate_glycan_support"]
    if _contains_any(
        text,
        (
            "metabolite",
            "primary metabolite",
            "primarymetabolites",
            "acetoacetate",
            "phosphoenolpyruvate",
            "coenzyme",
            "citric acid",
            "malic acid",
            "ascorbic acid",
            "fumarate",
            "pyruvate",
            "succinic acid",
            "small molecule",
            "small-molecule",
            "citrate",
        ),
    ):
        return "small_molecule_metabolite_support", FAMILY_LABELS["small_molecule_metabolite_support"]
    if _contains_any(text, ("pigment", "cofactor", "heme", "porphyrin", "carotene", "riboflavin", "riboﬂavin", "chromophore")):
        return "pigment_cofactor_support", FAMILY_LABELS["pigment_cofactor_support"]
    return "unresolved_assignment_support", FAMILY_LABELS["unresolved_assignment_support"]


def _domain_priority(value: str) -> int:
    return DOMAIN_PRIORITY.index(value) if value in DOMAIN_PRIORITY else len(DOMAIN_PRIORITY)


def _modality_priority(value: str) -> int:
    return MODALITY_PRIORITY.index(value) if value in MODALITY_PRIORITY else len(MODALITY_PRIORITY)


def _dominant_domain(values: list[str]) -> str:
    counts = Counter(value for value in values if value and value != "generic")
    if not counts:
        return "generic"
    return sorted(counts.items(), key=lambda item: (-item[1], _domain_priority(item[0]), item[0]))[0][0]


def _dominant_modality(values: list[str]) -> str:
    counts = Counter(value for value in values if value and value != "unknown")
    if not counts:
        return "unknown"
    return sorted(counts.items(), key=lambda item: (-item[1], _modality_priority(item[0]), item[0]))[0][0]


def _text_has_assignment_cue(text: str | None) -> bool:
    lowered = _normalize_text(text).lower()
    return any(phrase in lowered for phrase in ASSIGNMENT_CUE_PHRASES)


def _fetch_primary_supports(connection: duckdb.DuckDBPyConnection) -> list[SupportRecord]:
    supports: list[SupportRecord] = []
    peak_rows = connection.sql(
        """
        SELECT
            p.evidence_item_id,
            p.source_id,
            p.assignment_origin,
            p.study_family,
            p.peak_center_cm,
            p.assigned_molecule,
            p.assigned_group_or_theme,
            p.evidence_text,
            p.sample_type,
            p.modality,
            p.matrix_context,
            p.confidence_label
        FROM evidence.peak_assignment_evidence p
        WHERE p.is_primary_retrieval_eligible = TRUE
        ORDER BY p.evidence_item_id
        """
    ).fetchall()
    for (
        evidence_item_id,
        source_id,
        assignment_origin,
        study_family,
        peak_center_cm,
        assigned_molecule,
        assigned_group_or_theme,
        evidence_text,
        sample_type,
        modality,
        matrix_context,
        confidence_label,
    ) in peak_rows:
        family, label = normalize_meaning_family(assigned_molecule, assigned_group_or_theme, evidence_text)
        if assignment_origin == "curated_assignment":
            support_kind = "curated_assignment"
        elif assignment_origin == "reference_grounding_peak":
            support_kind = "reference_grounding_assignment"
        elif assignment_origin == "serum_grounding_peak":
            support_kind = "serum_grounding_assignment"
        else:
            support_kind = "source_backed_assignment"
        supports.append(
            SupportRecord(
                support_id=f"support_{evidence_item_id}",
                evidence_item_id=evidence_item_id,
                source_id=source_id,
                source_type=assignment_origin,
                support_kind=support_kind,
                support_role="primary",
                peak_center_cm=float(peak_center_cm),
                normalized_family=family,
                normalized_meaning_label=label,
                raw_label=_normalize_text(assigned_molecule) or _normalize_text(assigned_group_or_theme),
                domain_hint=_domain_from_text(study_family, sample_type, matrix_context),
                modality_hint=_modality_from_text(modality),
                study_family=study_family,
                weight=SUPPORT_KIND_WEIGHTS[support_kind],
                provenance={
                    "assignment_origin": assignment_origin,
                    "confidence_label": confidence_label,
                    "matrix_context": matrix_context,
                },
                note=evidence_text,
            )
        )

    reference_rows = connection.sql(
        """
        SELECT
            f.feature_id,
            f.evidence_item_id,
            f.source_id,
            f.peak_center_cm,
            f.feature_weight,
            r.component,
            r.biochemical_class,
            r.modality
        FROM features.spectral_features f
        JOIN evidence.reference_spectrum_evidence r
          ON r.evidence_item_id = f.evidence_item_id
        WHERE f.feature_origin = 'ramanbiolib_peak'
        ORDER BY f.feature_id
        """
    ).fetchall()
    for (
        feature_id,
        evidence_item_id,
        source_id,
        peak_center_cm,
        feature_weight,
        component,
        biochemical_class,
        modality,
    ) in reference_rows:
        family, label = normalize_meaning_family(component, biochemical_class)
        supports.append(
            SupportRecord(
                support_id=f"support_{feature_id}",
                evidence_item_id=evidence_item_id,
                source_id=source_id,
                source_type="reference_peak",
                support_kind="reference_peak",
                support_role="primary",
                peak_center_cm=float(peak_center_cm),
                normalized_family=family,
                normalized_meaning_label=label,
                raw_label=_normalize_text(component) or _normalize_text(biochemical_class),
                domain_hint="reference",
                modality_hint=_modality_from_text(modality),
                study_family="ramanbiolib",
                weight=float(feature_weight or SUPPORT_KIND_WEIGHTS["reference_peak"]),
                provenance={"component": component, "biochemical_class": biochemical_class},
                note="reference_peak",
            )
        )
    return supports


def _build_clusters(primary_supports: list[SupportRecord]) -> list[ClusterRecord]:
    grouped: dict[str, list[SupportRecord]] = defaultdict(list)
    for support in primary_supports:
        grouped[support.normalized_family].append(support)

    clusters: list[ClusterRecord] = []
    for normalized_family, supports in grouped.items():
        supports_sorted = sorted(supports, key=lambda item: item.peak_center_cm)
        active: list[SupportRecord] = []
        for support in supports_sorted:
            if not active:
                active = [support]
                continue
            current_center = sum(item.peak_center_cm * item.weight for item in active) / sum(item.weight for item in active)
            if abs(support.peak_center_cm - current_center) <= CLUSTER_MERGE_TOLERANCE_CM:
                active.append(support)
                continue
            cluster_center = sum(item.peak_center_cm * item.weight for item in active) / sum(item.weight for item in active)
            clusters.append(
                ClusterRecord(
                    cluster_id=f"cluster_{normalized_family}_{int(round(cluster_center))}_{len(clusters)+1:04d}",
                    normalized_family=normalized_family,
                    normalized_meaning_label=FAMILY_LABELS[normalized_family],
                    supports=list(active),
                )
            )
            active = [support]
        if active:
            cluster_center = sum(item.peak_center_cm * item.weight for item in active) / sum(item.weight for item in active)
            clusters.append(
                ClusterRecord(
                    cluster_id=f"cluster_{normalized_family}_{int(round(cluster_center))}_{len(clusters)+1:04d}",
                    normalized_family=normalized_family,
                    normalized_meaning_label=FAMILY_LABELS[normalized_family],
                    supports=list(active),
                )
            )
    return sorted(
        clusters,
        key=lambda cluster: (
            min(item.peak_center_cm for item in cluster.supports),
            cluster.normalized_family,
            cluster.cluster_id,
        ),
    )


def _cluster_centers(clusters: list[ClusterRecord]) -> dict[str, float]:
    return {
        cluster.cluster_id: sum(item.peak_center_cm * item.weight for item in cluster.supports) / sum(item.weight for item in cluster.supports)
        for cluster in clusters
    }


def _align_mentions(
    connection: duckdb.DuckDBPyConnection,
    clusters: list[ClusterRecord],
) -> tuple[list[tuple], list[SupportRecord], dict[str, int], dict[str, int]]:
    mention_rows = connection.sql(
        """
        SELECT evidence_item_id, source_id, wavenumber_cm, assigned_molecule_hint,
               biochemical_theme_hint, mention_text, sample_type, modality, study_family
        FROM evidence.wavenumber_mentions
        ORDER BY evidence_item_id
        """
    ).fetchall()

    cluster_centers = _cluster_centers(clusters)
    cluster_lookup = {cluster.cluster_id: cluster for cluster in clusters}
    mention_links = []
    aligned_supports: list[SupportRecord] = []
    counts = {
        "aligned_secondary_mentions": 0,
        "bare_mentions_excluded": 0,
    }
    excluded_by_family: dict[str, int] = defaultdict(int)

    for (
        evidence_item_id,
        source_id,
        wavenumber_cm,
        assigned_molecule_hint,
        biochemical_theme_hint,
        mention_text,
        sample_type,
        modality,
        study_family,
    ) in mention_rows:
        peak = float(wavenumber_cm)
        mention_family, mention_label = normalize_meaning_family(assigned_molecule_hint, biochemical_theme_hint, mention_text)
        if mention_family == "unresolved_assignment_support":
            mention_links.append(
                (
                    f"mention_link_{evidence_item_id}",
                    evidence_item_id,
                    None,
                    "excluded_bare_mention",
                    "no_family_hint_or_assignment_semantics",
                    None,
                    mention_label,
                    False,
                    0.0,
                    mention_text,
                )
            )
            counts["bare_mentions_excluded"] += 1
            excluded_by_family[mention_family] += 1
            continue

        candidate_clusters = []
        for cluster_id, center in cluster_centers.items():
            cluster = cluster_lookup[cluster_id]
            if cluster.normalized_family != mention_family:
                continue
            distance = abs(center - peak)
            if distance <= MENTION_ALIGN_TOLERANCE_CM:
                candidate_clusters.append((distance, cluster))

        if candidate_clusters and (_text_has_assignment_cue(mention_text) or mention_family != "unresolved_assignment_support"):
            candidate_clusters.sort(key=lambda item: (item[0], item[1].cluster_id))
            distance, cluster = candidate_clusters[0]
            mention_links.append(
                (
                    f"mention_link_{evidence_item_id}",
                    evidence_item_id,
                    cluster.cluster_id,
                    "aligned_secondary_support",
                    "family_hint_plus_window_match",
                    distance,
                    cluster.normalized_meaning_label,
                    True,
                    SUPPORT_KIND_WEIGHTS["aligned_mention"],
                    mention_text,
                )
            )
            aligned_supports.append(
                SupportRecord(
                    support_id=f"support_aligned_{evidence_item_id}",
                    evidence_item_id=evidence_item_id,
                    source_id=source_id,
                    source_type="aligned_mention",
                    support_kind="aligned_mention",
                    support_role="secondary",
                    peak_center_cm=peak,
                    normalized_family=cluster.normalized_family,
                    normalized_meaning_label=cluster.normalized_meaning_label,
                    raw_label=_normalize_text(assigned_molecule_hint) or _normalize_text(biochemical_theme_hint) or "mention_only",
                    domain_hint=_domain_from_text(study_family, sample_type),
                    modality_hint=_modality_from_text(modality),
                    study_family=study_family,
                    weight=SUPPORT_KIND_WEIGHTS["aligned_mention"],
                    provenance={
                        "mention_text": mention_text,
                        "alignment_reason": "family_hint_plus_window_match",
                        "assigned_molecule_hint": assigned_molecule_hint,
                        "biochemical_theme_hint": biochemical_theme_hint,
                    },
                    note=mention_text,
                )
            )
            counts["aligned_secondary_mentions"] += 1
            continue

        mention_links.append(
            (
                f"mention_link_{evidence_item_id}",
                evidence_item_id,
                None,
                "excluded_bare_mention",
                "family_hint_without_cluster_alignment",
                None,
                mention_label,
                False,
                0.0,
                mention_text,
            )
        )
        counts["bare_mentions_excluded"] += 1
        excluded_by_family[mention_family] += 1

    return mention_links, aligned_supports, counts, dict(excluded_by_family)


def _context_graph_rows() -> tuple[list[tuple], list[tuple]]:
    node_rows = [
        ("sample_ev", "sample_type", "EV", "ev", None, None, None, "Extracellular-vesicle query domain."),
        ("sample_serum", "sample_type", "Serum", "serum", None, None, None, "Serum query domain."),
        ("sample_plasma", "sample_type", "Plasma", "plasma", None, None, None, "Plasma query domain."),
        ("sample_pathogen", "sample_type", "Pathogen", "pathogen", None, None, None, "Pathogen query domain."),
        ("modality_raman", "modality", "Raman", None, "raman", None, None, "Spontaneous Raman modality."),
        ("modality_sers", "modality", "SERS", None, "sers", None, None, "Surface-enhanced Raman modality."),
        ("modality_ec_sers", "modality", "EC-SERS", None, "ec-sers", None, None, "Electrochemical SERS modality placeholder."),
        ("caveat_metabolite_dominance", "caveat", "metabolite dominance caveat", "serum", "sers", None, None, "Serum SERS can over-emphasize small molecules."),
        ("caveat_multicomponent_mixture", "caveat", "multicomponent mixture caveat", "ev", "sers", None, None, "EV SERS is a mixed cargo readout."),
        ("caveat_adsorption_bias", "caveat", "adsorption bias caveat", None, "sers", None, None, "Adsorption reshapes apparent visibility under SERS."),
        ("caveat_aromatic_assignment_ambiguity", "caveat", "aromatic assignment ambiguity", None, None, 950.0, 1050.0, "Around 1000 cm^-1, aromatic assignments overlap strongly."),
        ("caveat_ch_overlap", "caveat", "CH deformation overlap", None, None, 1300.0, 1500.0, "CH-rich windows overlap lipid and protein deformation signals."),
        ("caveat_amide_overlap", "caveat", "protein/aromatic high-band overlap", None, None, 1500.0, 1700.0, "High-wavenumber protein and aromatic bands overlap."),
        ("family_protein_amide_support", "affected_evidence_family", FAMILY_LABELS["protein_amide_support"], None, None, None, None, ""),
        ("family_nucleic_acid_base_support", "affected_evidence_family", FAMILY_LABELS["nucleic_acid_base_support"], None, None, None, None, ""),
        ("family_amino_acid_aromatic_support", "affected_evidence_family", FAMILY_LABELS["amino_acid_aromatic_support"], None, None, None, None, ""),
        ("family_lipid_membrane_support", "affected_evidence_family", FAMILY_LABELS["lipid_membrane_support"], None, None, None, None, ""),
        ("family_small_molecule_metabolite_support", "affected_evidence_family", FAMILY_LABELS["small_molecule_metabolite_support"], None, None, None, None, ""),
        ("region_950_1050", "peak_region", "region around 1000 cm^-1", None, None, 950.0, 1050.0, "Peak region node for aromatic-rich overlap."),
        ("region_1300_1500", "peak_region", "region 1300-1500 cm^-1", None, None, 1300.0, 1500.0, "Peak region node for CH deformation overlap."),
        ("region_1500_1700", "peak_region", "region 1500-1700 cm^-1", None, None, 1500.0, 1700.0, "Peak region node for high-band overlap."),
    ]
    edge_rows = [
        ("edge_sample_serum_caveat_metabolite", "sample_serum", "caveat_metabolite_dominance", "has_caveat", 1.0, "phase1_curated_context", ""),
        ("edge_sample_ev_caveat_multicomponent", "sample_ev", "caveat_multicomponent_mixture", "has_caveat", 1.0, "phase1_curated_context", ""),
        ("edge_modality_sers_caveat_adsorption", "modality_sers", "caveat_adsorption_bias", "has_caveat", 1.0, "phase1_curated_context", ""),
        ("edge_region_1000_aromatic", "region_950_1050", "caveat_aromatic_assignment_ambiguity", "affected_by", 0.9, "phase1_curated_context", ""),
        ("edge_region_1300_ch_overlap", "region_1300_1500", "caveat_ch_overlap", "affected_by", 0.9, "phase1_curated_context", ""),
        ("edge_region_1500_amide_overlap", "region_1500_1700", "caveat_amide_overlap", "affected_by", 0.9, "phase1_curated_context", ""),
        ("edge_adsorption_small_molecule", "caveat_adsorption_bias", "family_small_molecule_metabolite_support", "affects_visibility_of", 0.8, "phase1_curated_context", ""),
        ("edge_adsorption_lipid", "caveat_adsorption_bias", "family_lipid_membrane_support", "affects_visibility_of", 0.7, "phase1_curated_context", ""),
        ("edge_serum_metabolite_family", "caveat_metabolite_dominance", "family_small_molecule_metabolite_support", "affects_interpretation_of", 0.8, "phase1_curated_context", ""),
        ("edge_ev_multicomponent_protein", "caveat_multicomponent_mixture", "family_protein_amide_support", "affects_interpretation_of", 0.7, "phase1_curated_context", ""),
        ("edge_ev_multicomponent_lipid", "caveat_multicomponent_mixture", "family_lipid_membrane_support", "affects_interpretation_of", 0.7, "phase1_curated_context", ""),
        ("edge_aromatic_family", "caveat_aromatic_assignment_ambiguity", "family_amino_acid_aromatic_support", "affects_interpretation_of", 0.8, "phase1_curated_context", ""),
        ("edge_ch_family", "caveat_ch_overlap", "family_lipid_membrane_support", "affects_interpretation_of", 0.8, "phase1_curated_context", ""),
        ("edge_amide_family", "caveat_amide_overlap", "family_protein_amide_support", "affects_interpretation_of", 0.8, "phase1_curated_context", ""),
    ]
    return node_rows, edge_rows


def _context_for_cluster(
    connection: duckdb.DuckDBPyConnection,
    normalized_family: str,
    window_start_cm: float,
    window_end_cm: float,
    domain_hint: str,
    modality_hint: str,
) -> tuple[list[str], list[str], list[dict]]:
    node_rows = connection.sql("SELECT * FROM context.context_nodes").fetchall()
    edge_rows = connection.sql("SELECT * FROM context.context_edges").fetchall()

    active_nodes = []
    for node_id, node_type, node_label, node_domain, node_modality, region_start_cm, region_end_cm, notes in node_rows:
        domain_ok = node_domain in (None, "", domain_hint)
        modality_ok = node_modality in (None, "", modality_hint)
        region_ok = True
        if region_start_cm is not None and region_end_cm is not None:
            region_ok = not (window_end_cm < float(region_start_cm) or window_start_cm > float(region_end_cm))
        family_ok = True
        if node_type == "affected_evidence_family":
            family_ok = node_id == f"family_{normalized_family}"
        if domain_ok and modality_ok and region_ok and family_ok:
            active_nodes.append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "node_label": node_label,
                    "notes": notes,
                }
            )
    active_node_ids = {node["node_id"] for node in active_nodes}
    active_edges = []
    for edge_id, source_node_id, target_node_id, edge_type, weight, evidence_basis, notes in edge_rows:
        if source_node_id in active_node_ids or target_node_id in active_node_ids:
            active_edges.append(
                {
                    "edge_id": edge_id,
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                    "edge_type": edge_type,
                    "weight": weight,
                    "evidence_basis": evidence_basis,
                    "notes": notes,
                }
            )
            active_node_ids.add(source_node_id)
            active_node_ids.add(target_node_id)
    return sorted(active_node_ids), [edge["edge_id"] for edge in active_edges], active_edges


def _compute_overlap_index(clusters: list[ClusterRecord]) -> dict[str, list[str]]:
    centers = _cluster_centers(clusters)
    overlaps: dict[str, list[str]] = defaultdict(list)
    for cluster in clusters:
        for other in clusters:
            if cluster.cluster_id == other.cluster_id:
                continue
            if cluster.normalized_family == other.normalized_family:
                continue
            if abs(centers[cluster.cluster_id] - centers[other.cluster_id]) <= FAMILY_OVERLAP_TOLERANCE_CM:
                overlaps[cluster.cluster_id].append(other.normalized_family)
    return {cluster_id: sorted(set(families)) for cluster_id, families in overlaps.items()}


def build_phase1_refinement(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    node_rows, edge_rows = _context_graph_rows()
    connection.executemany("INSERT INTO context.context_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)", node_rows)
    connection.executemany("INSERT INTO context.context_edges VALUES (?, ?, ?, ?, ?, ?, ?)", edge_rows)

    primary_supports = _fetch_primary_supports(connection)
    clusters = _build_clusters(primary_supports)
    overlap_index = _compute_overlap_index(clusters)
    mention_links, aligned_mentions, mention_counts, excluded_by_family = _align_mentions(connection, clusters)
    connection.executemany("INSERT INTO evidence.operational_mention_links VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", mention_links)

    aligned_mentions_by_cluster: dict[str, list[SupportRecord]] = defaultdict(list)
    mention_link_lookup = {row[1]: row for row in mention_links if row[2] and row[7] is True}
    for support in aligned_mentions:
        mention_link = mention_link_lookup.get(support.evidence_item_id)
        if mention_link is not None:
            aligned_mentions_by_cluster[mention_link[2]].append(support)

    digitization_rows = connection.sql(
        """
        SELECT queue_id, study_family, priority
        FROM evidence.digitized_spectrum_registry
        ORDER BY queue_id
        """
    ).fetchall()
    digitization_by_family: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for queue_id, study_family, priority in digitization_rows:
        digitization_by_family[_normalize_text(study_family)].append((queue_id, priority))

    cluster_rows = []
    support_rows = []
    retrieval_rows = []

    for cluster in clusters:
        cluster_supports = list(cluster.supports) + aligned_mentions_by_cluster.get(cluster.cluster_id, [])
        weighted_primary = [support for support in cluster.supports if support.support_role == "primary"]
        canonical_peak_cm = sum(item.peak_center_cm * item.weight for item in weighted_primary) / sum(item.weight for item in weighted_primary)
        window_start_cm = min(item.peak_center_cm for item in cluster_supports)
        window_end_cm = max(item.peak_center_cm for item in cluster_supports)

        curated_assignment_count = sum(1 for item in weighted_primary if item.support_kind == "curated_assignment")
        explicit_assignment_count = sum(1 for item in weighted_primary if item.support_kind == "source_backed_assignment")
        reference_grounding_count = sum(1 for item in weighted_primary if item.support_kind == "reference_grounding_assignment")
        serum_grounding_count = sum(1 for item in weighted_primary if item.support_kind == "serum_grounding_assignment")
        reference_support_count = sum(1 for item in weighted_primary if item.support_kind == "reference_peak")
        aligned_mention_support_count = len(aligned_mentions_by_cluster.get(cluster.cluster_id, []))
        source_diversity_count = len({item.source_id for item in cluster_supports})
        raw_label_diversity = len({item.raw_label for item in weighted_primary if item.raw_label})
        unresolved_primary_count = sum(
            1 for item in weighted_primary if item.normalized_family == "unresolved_assignment_support" or item.raw_label in {"", "unknown"}
        )
        source_type_counts = Counter(item.source_type for item in cluster_supports)
        evidence_counts = Counter(item.support_kind for item in cluster_supports)
        unresolved_ratio = unresolved_primary_count / max(1, len(weighted_primary))
        reference_dominance = reference_support_count / max(1, len(weighted_primary))
        overlapping_families = overlap_index.get(cluster.cluster_id, [])
        overlapping_family_count = len(overlapping_families)
        mixed_family_flag = overlapping_family_count > 0

        curated_component = min(0.45, 0.18 * curated_assignment_count)
        explicit_component = min(0.25, 0.09 * explicit_assignment_count)
        reference_grounding_component = min(0.16, 0.04 * reference_grounding_count)
        serum_grounding_component = min(0.14, 0.035 * serum_grounding_count)
        reference_component = min(0.10, 0.003 * reference_support_count)
        aligned_mention_component = min(0.02, 0.004 * aligned_mention_support_count)
        source_diversity_component = min(0.12, 0.025 * source_diversity_count)
        base_confidence = (
            curated_component
            + explicit_component
            + reference_grounding_component
            + serum_grounding_component
            + reference_component
            + aligned_mention_component
            + source_diversity_component
        )

        raw_label_diversity_penalty = min(0.30, 0.06 * max(0, raw_label_diversity - 1))
        unresolved_penalty = min(0.35, 0.35 * unresolved_ratio)
        reference_dominance_penalty = min(0.18, 0.18 * reference_dominance)
        family_overlap_penalty = min(0.28, 0.08 * overlapping_family_count)
        ambiguity_score = min(
            1.0,
            raw_label_diversity_penalty + unresolved_penalty + reference_dominance_penalty + family_overlap_penalty,
        )
        confidence_score = max(0.0, min(1.0, base_confidence * (1.0 - 0.35 * ambiguity_score)))
        score_components = {
            "confidence_components": {
                "curated_component": round(curated_component, 6),
                "explicit_component": round(explicit_component, 6),
                "reference_grounding_component": round(reference_grounding_component, 6),
                "serum_grounding_component": round(serum_grounding_component, 6),
                "reference_component": round(reference_component, 6),
                "aligned_mention_component": round(aligned_mention_component, 6),
                "source_diversity_component": round(source_diversity_component, 6),
                "base_confidence": round(base_confidence, 6),
                "final_confidence": round(confidence_score, 6),
            },
            "ambiguity_components": {
                "raw_label_diversity_penalty": round(raw_label_diversity_penalty, 6),
                "unresolved_penalty": round(unresolved_penalty, 6),
                "reference_dominance_penalty": round(reference_dominance_penalty, 6),
                "family_overlap_penalty": round(family_overlap_penalty, 6),
                "final_ambiguity": round(ambiguity_score, 6),
            },
            "support_counts": {
                "curated_assignment_count": curated_assignment_count,
                "explicit_assignment_count": explicit_assignment_count,
                "reference_grounding_count": reference_grounding_count,
                "serum_grounding_count": serum_grounding_count,
                "reference_support_count": reference_support_count,
                "aligned_mention_support_count": aligned_mention_support_count,
                "source_diversity_count": source_diversity_count,
                "raw_label_diversity_count": raw_label_diversity,
                "overlapping_family_count": overlapping_family_count,
            },
            "overlapping_families": overlapping_families,
        }
        associated_families = {item.study_family for item in cluster_supports if item.study_family}
        linked_digitization_ids = []
        priority_counter = Counter()
        for family in associated_families:
            for queue_id, priority in digitization_by_family.get(family, []):
                linked_digitization_ids.append(queue_id)
                priority_counter[priority] += 1

        for support in cluster_supports:
            support_rows.append(
                (
                    support.support_id,
                    cluster.cluster_id,
                    support.evidence_item_id,
                    support.source_id,
                    support.support_kind,
                    support.support_role,
                    support.source_type,
                    support.peak_center_cm,
                    abs(support.peak_center_cm - canonical_peak_cm),
                    cluster.normalized_meaning_label,
                    support.raw_label,
                    support.weight,
                    support.support_role == "primary",
                    support.support_role == "secondary",
                    json.dumps(support.provenance, sort_keys=True),
                    support.note,
                )
            )

        dominant_domain = _dominant_domain([item.domain_hint for item in weighted_primary])
        dominant_modality = _dominant_modality([item.modality_hint for item in weighted_primary])
        context_node_ids, context_edge_ids, context_edge_records = _context_for_cluster(
            connection,
            normalized_family=cluster.normalized_family,
            window_start_cm=window_start_cm,
            window_end_cm=window_end_cm,
            domain_hint=dominant_domain,
            modality_hint=dominant_modality,
        )
        support_summary = {
            "family_label_version": FAMILY_LABEL_VERSION,
            "previous_normalized_meaning_label": PREVIOUS_FAMILY_LABELS[cluster.normalized_family],
            "curated_assignment_count": curated_assignment_count,
            "explicit_assignment_count": explicit_assignment_count,
            "reference_grounding_count": reference_grounding_count,
            "serum_grounding_count": serum_grounding_count,
            "reference_support_count": reference_support_count,
            "aligned_mention_support_count": aligned_mention_support_count,
            "source_type_counts": source_type_counts,
            "evidence_counts": evidence_counts,
            "mixed_family_flag": mixed_family_flag,
            "overlapping_families": overlapping_families,
            "top_raw_labels": Counter(item.raw_label for item in cluster_supports if item.raw_label).most_common(5),
        }
        summary_text = (
            f"{cluster.normalized_meaning_label} bundle around {canonical_peak_cm:.1f} cm^-1 "
            f"with {curated_assignment_count} curated, {explicit_assignment_count} explicit, "
            f"{reference_grounding_count} reference-grounding, {serum_grounding_count} serum-grounding, "
            f"{reference_support_count} reference, and {aligned_mention_support_count} aligned secondary supports."
        )
        cluster_rows.append(
            (
                cluster.cluster_id,
                canonical_peak_cm,
                window_start_cm,
                window_end_cm,
                cluster.normalized_meaning_label,
                cluster.normalized_family,
                FAMILY_LABEL_VERSION,
                mixed_family_flag,
                overlapping_family_count,
                raw_label_diversity,
                json.dumps(source_type_counts, sort_keys=True),
                source_diversity_count,
                explicit_assignment_count,
                curated_assignment_count,
                reference_support_count,
                aligned_mention_support_count,
                excluded_by_family.get(cluster.normalized_family, 0),
                priority_counter["high_priority_digitize"],
                priority_counter["medium_priority_digitize"],
                priority_counter["low_priority_or_redundant"],
                ambiguity_score,
                confidence_score,
                json.dumps(score_components, sort_keys=True),
                json.dumps(sorted({item.evidence_item_id for item in cluster_supports}), sort_keys=True),
                json.dumps(sorted({item.source_id for item in cluster_supports}), sort_keys=True),
                json.dumps(sorted(linked_digitization_ids), sort_keys=True),
                "Phase 1 operational peak-meaning bundle. Cleanup/audit refined label semantics and score decomposition.",
            )
        )
        retrieval_rows.append(
            (
                f"pmdoc_{cluster.cluster_id}",
                cluster.cluster_id,
                cluster.normalized_meaning_label,
                cluster.normalized_family,
                f"{canonical_peak_cm:.0f} cm^-1 {cluster.normalized_meaning_label} bundle",
                summary_text,
                canonical_peak_cm,
                window_start_cm,
                window_end_cm,
                dominant_domain,
                dominant_modality,
                cluster.normalized_family != "unresolved_assignment_support",
                confidence_score,
                ambiguity_score,
                mixed_family_flag,
                overlapping_family_count,
                source_diversity_count,
                curated_assignment_count,
                explicit_assignment_count,
                reference_support_count,
                aligned_mention_support_count,
                json.dumps(score_components, sort_keys=True),
                json.dumps(context_node_ids, sort_keys=True),
                json.dumps(context_edge_ids, sort_keys=True),
                json.dumps(support_summary, sort_keys=True),
                json.dumps(
                    {
                        "cluster_id": cluster.cluster_id,
                        "family_label_version": FAMILY_LABEL_VERSION,
                        "linked_evidence_ids": sorted({item.evidence_item_id for item in cluster_supports}),
                        "context_edge_records": context_edge_records,
                    },
                    sort_keys=True,
                ),
            )
        )

    connection.executemany(
        "INSERT INTO evidence.peak_meaning_clusters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        cluster_rows,
    )
    connection.executemany("INSERT INTO evidence.peak_meaning_support VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", support_rows)
    connection.executemany(
        "INSERT INTO retrieval.peak_meaning_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        retrieval_rows,
    )

    return {
        "peak_meaning_clusters_loaded": len(cluster_rows),
        "peak_meaning_support_rows_loaded": len(support_rows),
        "aligned_secondary_mentions": mention_counts["aligned_secondary_mentions"],
        "bare_mentions_excluded": mention_counts["bare_mentions_excluded"],
        "context_nodes_loaded": len(node_rows),
        "context_edges_loaded": len(edge_rows),
    }
