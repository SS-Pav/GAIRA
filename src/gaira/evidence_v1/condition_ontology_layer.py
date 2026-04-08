from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    CONDITION_LAYER_REPORT_ROOT,
    CONDITION_LAYER_TABLES_ROOT,
    DB_PATH,
    ensure_condition_layer_output_dirs,
)
from gaira.evidence_v1.schema import initialize_schema, reset_condition_layer_tables


@dataclass(frozen=True)
class ConditionContext:
    source_id: str
    raw_condition_text: str
    normalized_condition_label: str
    condition_family: str
    aliases: tuple[str, ...]
    sample_type: str
    experimental_context: str
    control_group_present: bool
    control_label: str
    comparison_type: str
    trajectory_type: str
    context_role: str
    notes: str


CONDITION_CONTEXTS = [
    ConditionContext(
        "src_cca_2024_manuscript",
        "cholangiocarcinoma",
        "cholangiocarcinoma",
        "cancer",
        ("CCA",),
        "serum",
        "serum SERS discriminative peak-assignment context",
        True,
        "healthy control",
        "case_vs_control",
        "classification",
        "condition",
        "Pilot-integrated serum disease paper with explicit labeled assignments.",
    ),
    ConditionContext(
        "src_cca_2024_manuscript",
        "healthy control",
        "healthy_control",
        "healthy_control",
        ("HC", "healthy"),
        "serum",
        "serum SERS comparator cohort",
        True,
        "healthy control",
        "case_vs_control",
        "classification",
        "control",
        "Comparator state only; not linked as positive spectral support unless explicitly present.",
    ),
    ConditionContext(
        "src_exosome_sers_2023_manuscript",
        "multi-cancer exosome diagnosis",
        "multicancer_exosome_diagnosis",
        "cancer",
        ("pan-cancer exosome diagnosis",),
        "EV",
        "exosome SERS cancer-diagnosis workflow",
        True,
        "healthy control",
        "multi_class",
        "classification",
        "condition",
        "Existing structured evidence is sparse and mostly contextual.",
    ),
    ConditionContext(
        "src_exosome_sers_2023_manuscript",
        "healthy control",
        "healthy_control",
        "healthy_control",
        ("HC", "healthy"),
        "EV",
        "exosome SERS healthy comparator",
        True,
        "healthy control",
        "multi_class",
        "classification",
        "control",
        "Comparator state only.",
    ),
    ConditionContext(
        "src_liu_2024_exo_manuscript",
        "cancer diagnosis exosome context",
        "cancer_exosome_diagnosis_context",
        "cancer",
        ("label-free exosome cancer diagnosis",),
        "EV",
        "label-free exosome SERS diagnosis context",
        True,
        "healthy control",
        "classification",
        "classification",
        "condition",
        "Very sparse current paper evidence; retained as weak condition context only.",
    ),
    ConditionContext(
        "src_liu_2024_exo_manuscript",
        "healthy control",
        "healthy_control",
        "healthy_control",
        ("HC", "healthy"),
        "EV",
        "exosome SERS healthy comparator",
        True,
        "healthy control",
        "classification",
        "classification",
        "control",
        "Comparator state only.",
    ),
    ConditionContext(
        "src_liu_2025_lung_manuscript",
        "lung cancer EVs",
        "lung_cancer",
        "cancer",
        ("lung cancer", "lung cancer EV"),
        "EV",
        "EV SERS on nanoparticle substrate",
        True,
        "healthy control",
        "case_vs_control",
        "classification",
        "condition",
        "EV-focused cancer paper with explicit figure/text assignments.",
    ),
    ConditionContext(
        "src_liu_2025_lung_manuscript",
        "healthy control",
        "healthy_control",
        "healthy_control",
        ("HC", "healthy"),
        "EV",
        "EV SERS healthy comparator",
        True,
        "healthy control",
        "case_vs_control",
        "classification",
        "control",
        "Comparator state only.",
    ),
    ConditionContext(
        "src_miao_2024_manuscript",
        "lung tumor diagnosis",
        "lung_tumor",
        "cancer",
        ("lung tumor", "lung cancer tissue context"),
        "tissue",
        "tissue Raman diagnostic context",
        True,
        "healthy control",
        "case_vs_control",
        "classification",
        "condition",
        "Current paper evidence is sparse and regex-derived.",
    ),
    ConditionContext(
        "src_miao_2024_manuscript",
        "healthy control",
        "healthy_control",
        "healthy_control",
        ("HC", "healthy"),
        "tissue",
        "tissue Raman healthy comparator",
        True,
        "healthy control",
        "case_vs_control",
        "classification",
        "control",
        "Comparator state only.",
    ),
    ConditionContext(
        "src_parlatan_2023_manuscript",
        "multiclass EV Raman cancer context",
        "multiclass_ev_ml_cancer_context",
        "cancer",
        ("EV Raman machine-learning cancer context",),
        "EV",
        "label-free EV Raman ML context",
        False,
        "",
        "multi_class",
        "classification",
        "condition",
        "Sparse existing paper evidence; context retained conservatively.",
    ),
    ConditionContext(
        "src_plasma_ev_2026_manuscript",
        "type 2 diabetes subgroup context",
        "type_2_diabetes_subtype_context",
        "metabolic_disease",
        ("T2DM subgroup context", "overweight versus normal-weight diabetes"),
        "ev_enriched_plasma",
        "plasma EV-enriched SERS with lipoprotein overlap caveat",
        False,
        "",
        "multi_class",
        "classification",
        "condition",
        "Region-level only; treated as tentative condition-aware support.",
    ),
    ConditionContext(
        "src_sers_2023_scirep_manuscript",
        "primary Sjögren's syndrome",
        "primary_sjogrens_syndrome",
        "autoimmune_disease",
        ("pSS",),
        "serum",
        "serum SERS on AgNP substrate",
        True,
        "healthy control",
        "multi_class",
        "classification",
        "condition",
        "Shared assignment table used for HC/pSS and HC/DN comparisons.",
    ),
    ConditionContext(
        "src_sers_2023_scirep_manuscript",
        "diabetic nephropathy",
        "diabetic_nephropathy",
        "metabolic_disease",
        ("DN",),
        "serum",
        "serum SERS on AgNP substrate",
        True,
        "healthy control",
        "multi_class",
        "classification",
        "condition",
        "Shared assignment table used for HC/pSS and HC/DN comparisons.",
    ),
    ConditionContext(
        "src_sers_2023_scirep_manuscript",
        "healthy control",
        "healthy_control",
        "healthy_control",
        ("HC", "healthy"),
        "serum",
        "serum SERS healthy comparator",
        True,
        "healthy control",
        "multi_class",
        "classification",
        "control",
        "Comparator state only.",
    ),
    ConditionContext(
        "src_shine_2026_manuscript",
        "acetaminophen-induced hepatotoxicity",
        "acetaminophen_induced_hepatotoxicity",
        "drug_induced_toxicity",
        ("APAP hepatotoxicity", "drug-induced liver injury"),
        "ev_from_hepatic_cell_culture",
        "EV SERS on Au nanopillar substrate under APAP dose-response",
        True,
        "untreated control",
        "dose_response",
        "perturbation",
        "condition",
        "Dose-response perturbation context, not discrete case-control disease assignment.",
    ),
    ConditionContext(
        "src_shine_2026_manuscript",
        "untreated control",
        "untreated_control",
        "healthy_control",
        ("baseline", "untreated"),
        "ev_from_hepatic_cell_culture",
        "baseline EV comparator under APAP perturbation",
        True,
        "untreated control",
        "dose_response",
        "perturbation",
        "control",
        "Comparator state only.",
    ),
]


CONDITION_BEARING_SOURCE_IDS = {item.source_id for item in CONDITION_CONTEXTS if item.context_role == "condition"}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _weight_from_row(extraction_method: str, is_primary: bool, qc_classification: str) -> float:
    if extraction_method == "digitized_figure":
        return 1.0 if is_primary else 0.65
    if extraction_method == "text_assignment":
        return 0.9 if is_primary else 0.55
    if extraction_method == "text_regex":
        if qc_classification == "validated_primary":
            return 0.7
        if qc_classification == "validated_secondary":
            return 0.35
        return 0.0
    return 0.25 if is_primary else 0.1


def _signal_characterization(unique_motif_count: int, shared_motif_count: int, motif_count: int, neighborhood_count: int) -> str:
    if motif_count == 0 and neighborhood_count > 0:
        return "sparse_local_only"
    if unique_motif_count >= 2 and unique_motif_count > shared_motif_count:
        return "discriminative_tendency"
    if shared_motif_count > 0 and shared_motif_count >= unique_motif_count:
        return "overlapping"
    if neighborhood_count == 0:
        return "sparse"
    return "ambiguous"


def _build_condition_context_rows(processed_sources: set[str]) -> list[ConditionContext]:
    return [item for item in CONDITION_CONTEXTS if item.source_id in processed_sources]


def run_condition_ontology_layer(db_path: Path = DB_PATH) -> dict[str, object]:
    ensure_condition_layer_output_dirs()
    connection = duckdb.connect(str(db_path))
    try:
        initialize_schema(connection)
        reset_condition_layer_tables(connection)

        processed_sources = {
            row[0]
            for row in connection.sql(
                """
                SELECT DISTINCT source_id
                FROM evidence.peak_assignment_evidence
                WHERE source_id LIKE 'src_%_manuscript'
                """
            ).fetchall()
        }
        context_rows = _build_condition_context_rows(processed_sources)

        ontology_rows = []
        seen_conditions = set()
        for index, row in enumerate(context_rows, start=1):
            if not row.normalized_condition_label or row.normalized_condition_label in seen_conditions:
                continue
            seen_conditions.add(row.normalized_condition_label)
            ontology_rows.append(
                (
                    f"cond_{index:03d}",
                    row.raw_condition_text,
                    row.normalized_condition_label,
                    row.condition_family,
                    json.dumps(list(row.aliases)),
                    row.notes,
                )
            )
        if ontology_rows:
            connection.executemany(
                "INSERT INTO evidence.condition_ontology VALUES (?, ?, ?, ?, ?, ?)",
                ontology_rows,
            )

        paper_context_rows = [
            (
                row.source_id,
                row.raw_condition_text,
                row.normalized_condition_label,
                row.condition_family,
                row.sample_type,
                row.experimental_context,
                row.control_group_present,
                row.control_label,
                row.comparison_type,
                row.trajectory_type,
                row.context_role,
                row.notes,
            )
            for row in context_rows
        ]
        if paper_context_rows:
            connection.executemany(
                "INSERT INTO evidence.paper_condition_context VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                paper_context_rows,
            )

        evidence_rows = connection.sql(
            """
            WITH eligible AS (
                SELECT
                    pae.source_id,
                    pae.evidence_item_id,
                    pae.assignment_record_id,
                    pae.extraction_method,
                    pae.is_primary_retrieval_eligible,
                    COALESCE(qc.qc_classification, '') AS qc_classification,
                    CASE
                        WHEN pae.extraction_method = 'text_regex' THEN COALESCE(qc.include_in_local_layer, FALSE)
                        ELSE TRUE
                    END AS include_row
                FROM evidence.peak_assignment_evidence pae
                LEFT JOIN evidence.paper_assignment_qc qc
                  ON qc.assignment_record_id = pae.assignment_record_id
                 AND qc.source_id = pae.source_id
                WHERE pae.source_id LIKE 'src_%_manuscript'
            )
            SELECT
                e.source_id,
                e.evidence_item_id,
                e.assignment_record_id,
                e.extraction_method,
                e.is_primary_retrieval_eligible,
                e.qc_classification,
                m.neighborhood_id,
                n.broader_family,
                n.spectral_region,
                n.local_ambiguity_score,
                COALESCE(l.pattern_id, '') AS pattern_id,
                COALESCE(ap.ambiguity_score, 0.0) AS pattern_ambiguity_score
            FROM eligible e
            JOIN evidence.local_support_neighborhood_members m
              ON m.assignment_record_id = e.assignment_record_id
             AND m.source_id = e.source_id
            JOIN evidence.local_support_neighborhoods n
              ON n.neighborhood_id = m.neighborhood_id
            LEFT JOIN evidence.neighborhood_motif_links l
              ON l.neighborhood_id = m.neighborhood_id
            LEFT JOIN evidence.assignment_patterns ap
              ON ap.pattern_id = l.pattern_id
            WHERE e.include_row = TRUE
            ORDER BY e.source_id, e.assignment_record_id
            """
        ).df().to_dict("records")

        condition_rows_by_source = defaultdict(list)
        for row in context_rows:
            condition_rows_by_source[row.source_id].append(row)

        neighborhood_acc = {}
        motif_acc = {}
        for row in evidence_rows:
            contexts = [item for item in condition_rows_by_source[row["source_id"]] if item.context_role == "condition"]
            if not contexts:
                continue
            weight = _weight_from_row(
                row["extraction_method"],
                bool(row["is_primary_retrieval_eligible"]),
                row["qc_classification"],
            )
            if weight <= 0:
                continue
            for ctx in contexts:
                n_key = (ctx.normalized_condition_label, ctx.condition_family, ctx.source_id, row["neighborhood_id"])
                n_entry = neighborhood_acc.setdefault(
                    n_key,
                    {
                        "normalized_condition_label": ctx.normalized_condition_label,
                        "condition_family": ctx.condition_family,
                        "source_id": ctx.source_id,
                        "neighborhood_id": row["neighborhood_id"],
                        "sample_type": ctx.sample_type,
                        "experimental_context": ctx.experimental_context,
                        "comparison_type": ctx.comparison_type,
                        "trajectory_type": ctx.trajectory_type,
                        "evidence_row_ids": set(),
                        "primary_evidence_count": 0,
                        "secondary_evidence_count": 0,
                        "digitized_figure_count": 0,
                        "text_assignment_count": 0,
                        "regex_secondary_count": 0,
                        "support_strength": 0.0,
                        "ambiguities": [],
                    },
                )
                if row["assignment_record_id"] not in n_entry["evidence_row_ids"]:
                    n_entry["evidence_row_ids"].add(row["assignment_record_id"])
                    if row["is_primary_retrieval_eligible"]:
                        n_entry["primary_evidence_count"] += 1
                    else:
                        n_entry["secondary_evidence_count"] += 1
                    if row["extraction_method"] == "digitized_figure":
                        n_entry["digitized_figure_count"] += 1
                    elif row["extraction_method"] == "text_assignment":
                        n_entry["text_assignment_count"] += 1
                    elif row["extraction_method"] == "text_regex":
                        n_entry["regex_secondary_count"] += 1
                    n_entry["support_strength"] += weight
                    n_entry["ambiguities"].append(float(row["local_ambiguity_score"]))

                if row["pattern_id"]:
                    p_key = (ctx.normalized_condition_label, ctx.condition_family, ctx.source_id, row["pattern_id"])
                    p_entry = motif_acc.setdefault(
                        p_key,
                        {
                            "normalized_condition_label": ctx.normalized_condition_label,
                            "condition_family": ctx.condition_family,
                            "source_id": ctx.source_id,
                            "pattern_id": row["pattern_id"],
                            "sample_type": ctx.sample_type,
                            "experimental_context": ctx.experimental_context,
                            "comparison_type": ctx.comparison_type,
                            "trajectory_type": ctx.trajectory_type,
                            "evidence_row_ids": set(),
                            "primary_evidence_count": 0,
                            "secondary_evidence_count": 0,
                            "digitized_figure_count": 0,
                            "text_assignment_count": 0,
                            "regex_secondary_count": 0,
                            "support_strength": 0.0,
                            "ambiguities": [],
                        },
                    )
                    if row["assignment_record_id"] not in p_entry["evidence_row_ids"]:
                        p_entry["evidence_row_ids"].add(row["assignment_record_id"])
                        if row["is_primary_retrieval_eligible"]:
                            p_entry["primary_evidence_count"] += 1
                        else:
                            p_entry["secondary_evidence_count"] += 1
                        if row["extraction_method"] == "digitized_figure":
                            p_entry["digitized_figure_count"] += 1
                        elif row["extraction_method"] == "text_assignment":
                            p_entry["text_assignment_count"] += 1
                        elif row["extraction_method"] == "text_regex":
                            p_entry["regex_secondary_count"] += 1
                        p_entry["support_strength"] += weight
                        p_entry["ambiguities"].append(float(row["pattern_ambiguity_score"]))

        neighborhood_link_rows = []
        for entry in neighborhood_acc.values():
            composition = {
                "primary": entry["primary_evidence_count"],
                "secondary": entry["secondary_evidence_count"],
                "digitized_figure": entry["digitized_figure_count"],
                "text_assignment": entry["text_assignment_count"],
                "regex_secondary": entry["regex_secondary_count"],
            }
            neighborhood_link_rows.append(
                (
                    entry["normalized_condition_label"],
                    entry["condition_family"],
                    entry["source_id"],
                    entry["neighborhood_id"],
                    entry["sample_type"],
                    entry["experimental_context"],
                    entry["comparison_type"],
                    entry["trajectory_type"],
                    len(entry["evidence_row_ids"]),
                    entry["primary_evidence_count"],
                    entry["secondary_evidence_count"],
                    entry["digitized_figure_count"],
                    entry["text_assignment_count"],
                    entry["regex_secondary_count"],
                    json.dumps(composition, sort_keys=True),
                    round(entry["support_strength"], 6),
                    round(sum(entry["ambiguities"]) / max(1, len(entry["ambiguities"])), 6),
                    "Condition link derived from structured paper evidence rows landing in the same local neighborhood.",
                )
            )
        if neighborhood_link_rows:
            connection.executemany(
                "INSERT INTO evidence.condition_to_neighborhood_links VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                neighborhood_link_rows,
            )

        motif_link_rows = []
        for entry in motif_acc.values():
            composition = {
                "primary": entry["primary_evidence_count"],
                "secondary": entry["secondary_evidence_count"],
                "digitized_figure": entry["digitized_figure_count"],
                "text_assignment": entry["text_assignment_count"],
                "regex_secondary": entry["regex_secondary_count"],
            }
            motif_link_rows.append(
                (
                    entry["normalized_condition_label"],
                    entry["condition_family"],
                    entry["source_id"],
                    entry["pattern_id"],
                    entry["sample_type"],
                    entry["experimental_context"],
                    entry["comparison_type"],
                    entry["trajectory_type"],
                    len(entry["evidence_row_ids"]),
                    entry["primary_evidence_count"],
                    entry["secondary_evidence_count"],
                    entry["digitized_figure_count"],
                    entry["text_assignment_count"],
                    entry["regex_secondary_count"],
                    json.dumps(composition, sort_keys=True),
                    round(entry["support_strength"], 6),
                    round(sum(entry["ambiguities"]) / max(1, len(entry["ambiguities"])), 6),
                    "Condition link derived from condition-linked neighborhoods that map into existing motifs.",
                )
            )
        if motif_link_rows:
            connection.executemany(
                "INSERT INTO evidence.condition_to_motif_links VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                motif_link_rows,
            )

        motif_shared = connection.sql(
            """
            SELECT pattern_id, COUNT(DISTINCT normalized_condition_label) AS n_conditions
            FROM evidence.condition_to_motif_links
            GROUP BY 1
            """
        ).df().to_dict("records")
        shared_lookup = {row["pattern_id"]: int(row["n_conditions"]) for row in motif_shared}

        summary_rows = []
        family_counter = defaultdict(Counter)
        region_counter = defaultdict(Counter)
        confounder_counter = defaultdict(int)
        condition_to_patterns = defaultdict(set)
        condition_to_neighborhoods = defaultdict(set)
        for row in connection.sql(
            """
            SELECT
                l.normalized_condition_label,
                l.pattern_id,
                p.broader_family,
                p.spectral_region,
                CASE WHEN p.meaning_class = 'confounder_signal' THEN 1 ELSE 0 END AS is_confounder
            FROM evidence.condition_to_motif_links l
            JOIN evidence.assignment_patterns p
              ON p.pattern_id = l.pattern_id
            """
        ).df().to_dict("records"):
            condition_to_patterns[row["normalized_condition_label"]].add(row["pattern_id"])
            family_counter[row["normalized_condition_label"]][row["broader_family"]] += 1
            region_counter[row["normalized_condition_label"]][row["spectral_region"]] += 1
            confounder_counter[row["normalized_condition_label"]] += int(row["is_confounder"])

        for row in connection.sql(
            """
            SELECT
                l.normalized_condition_label,
                l.neighborhood_id,
                n.broader_family,
                n.spectral_region,
                CASE WHEN n.meaning_class = 'confounder_signal' THEN 1 ELSE 0 END AS is_confounder
            FROM evidence.condition_to_neighborhood_links l
            JOIN evidence.local_support_neighborhoods n
              ON n.neighborhood_id = l.neighborhood_id
            """
        ).df().to_dict("records"):
            condition_to_neighborhoods[row["normalized_condition_label"]].add(row["neighborhood_id"])
            family_counter[row["normalized_condition_label"]][row["broader_family"]] += 1
            region_counter[row["normalized_condition_label"]][row["spectral_region"]] += 1
            confounder_counter[row["normalized_condition_label"]] += int(row["is_confounder"])

        paper_counts = connection.sql(
            """
            SELECT normalized_condition_label, COUNT(DISTINCT source_id) AS paper_count
            FROM evidence.condition_to_neighborhood_links
            GROUP BY 1
            """
        ).df().to_dict("records")
        paper_count_lookup = {row["normalized_condition_label"]: int(row["paper_count"]) for row in paper_counts}

        for row in connection.sql(
            "SELECT normalized_condition_label, condition_family FROM evidence.condition_ontology ORDER BY normalized_condition_label"
        ).df().to_dict("records"):
            cond = row["normalized_condition_label"]
            patterns = condition_to_patterns[cond]
            neighborhoods = condition_to_neighborhoods[cond]
            shared_motif_count = sum(1 for pattern_id in patterns if shared_lookup.get(pattern_id, 0) > 1)
            unique_motif_count = sum(1 for pattern_id in patterns if shared_lookup.get(pattern_id, 0) == 1)
            summary_rows.append(
                (
                    cond,
                    row["condition_family"],
                    paper_count_lookup.get(cond, 0),
                    len(patterns),
                    len(neighborhoods),
                    json.dumps(dict(family_counter[cond].most_common()), sort_keys=True),
                    json.dumps(dict(region_counter[cond].most_common()), sort_keys=True),
                    confounder_counter[cond],
                    shared_motif_count,
                    unique_motif_count,
                    _signal_characterization(unique_motif_count, shared_motif_count, len(patterns), len(neighborhoods)),
                    "Descriptive condition summary only; no disease-specific biomarker claim implied.",
                )
            )
        if summary_rows:
            connection.executemany(
                "INSERT INTO evidence.condition_support_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                summary_rows,
            )

        connection.commit()

        condition_registry = connection.sql(
            "SELECT * FROM evidence.condition_ontology ORDER BY normalized_condition_label"
        ).df().to_dict("records")
        condition_family_summary = connection.sql(
            """
            SELECT condition_family, COUNT(*) AS condition_count
            FROM evidence.condition_ontology
            GROUP BY 1
            ORDER BY condition_count DESC, condition_family
            """
        ).df().to_dict("records")
        condition_to_motif = connection.sql(
            """
            SELECT *
            FROM evidence.condition_to_motif_links
            ORDER BY normalized_condition_label, source_id, pattern_id
            """
        ).df().to_dict("records")
        condition_to_neighborhood = connection.sql(
            """
            SELECT *
            FROM evidence.condition_to_neighborhood_links
            ORDER BY normalized_condition_label, source_id, neighborhood_id
            """
        ).df().to_dict("records")
        condition_support_summary = connection.sql(
            "SELECT * FROM evidence.condition_support_summary ORDER BY normalized_condition_label"
        ).df().to_dict("records")
        comparison_summary = connection.sql(
            """
            SELECT
                source_id,
                normalized_condition_label,
                condition_family,
                sample_type,
                control_group_present,
                control_label,
                comparison_type,
                experimental_context
            FROM evidence.paper_condition_context
            WHERE context_role = 'condition'
            ORDER BY source_id, normalized_condition_label
            """
        ).df().to_dict("records")
        trajectory_summary = connection.sql(
            """
            SELECT
                source_id,
                normalized_condition_label,
                trajectory_type,
                comparison_type,
                context_role,
                notes
            FROM evidence.paper_condition_context
            ORDER BY source_id, normalized_condition_label, context_role
            """
        ).df().to_dict("records")

        overlap_rows = connection.sql(
            """
            SELECT normalized_condition_label, pattern_id, support_strength
            FROM evidence.condition_to_motif_links
            ORDER BY normalized_condition_label, pattern_id
            """
        ).df().to_dict("records")
        pattern_labels = {
            row["pattern_id"]: row["pattern_label"]
            for row in connection.sql("SELECT pattern_id, pattern_label FROM evidence.assignment_patterns").df().to_dict("records")
        }
        overlap_matrix = defaultdict(dict)
        for row in overlap_rows:
            overlap_matrix[row["normalized_condition_label"]][pattern_labels.get(row["pattern_id"], row["pattern_id"])] = row["support_strength"]
        all_pattern_columns = sorted({col for data in overlap_matrix.values() for col in data})
        overlap_matrix_rows = []
        for condition_label in sorted(overlap_matrix):
            base = {"normalized_condition_label": condition_label}
            for column in all_pattern_columns:
                base[column] = overlap_matrix[condition_label].get(column, 0.0)
            overlap_matrix_rows.append(base)

        ambiguous_condition_support = connection.sql(
            """
            SELECT
                c.normalized_condition_label,
                c.condition_family,
                SUM(CASE WHEN n.local_ambiguity_score >= 0.35 THEN 1 ELSE 0 END) AS ambiguous_neighborhood_links,
                SUM(CASE WHEN p.ambiguity_score >= 0.50 THEN 1 ELSE 0 END) AS ambiguous_motif_links
            FROM evidence.condition_ontology c
            LEFT JOIN evidence.condition_to_neighborhood_links l
              ON l.normalized_condition_label = c.normalized_condition_label
            LEFT JOIN evidence.local_support_neighborhoods n
              ON n.neighborhood_id = l.neighborhood_id
            LEFT JOIN evidence.condition_to_motif_links ml
              ON ml.normalized_condition_label = c.normalized_condition_label
            LEFT JOIN evidence.assignment_patterns p
              ON p.pattern_id = ml.pattern_id
            GROUP BY 1, 2
            HAVING ambiguous_neighborhood_links > 0 OR ambiguous_motif_links > 0
            ORDER BY c.normalized_condition_label
            """
        ).df().to_dict("records")

        _write_csv(
            CONDITION_LAYER_TABLES_ROOT / "condition_ontology_registry.csv",
            list(condition_registry[0].keys()) if condition_registry else ["condition_id"],
            condition_registry,
        )
        _write_csv(
            CONDITION_LAYER_TABLES_ROOT / "condition_family_summary.csv",
            list(condition_family_summary[0].keys()) if condition_family_summary else ["condition_family"],
            condition_family_summary,
        )
        _write_csv(
            CONDITION_LAYER_TABLES_ROOT / "condition_to_motif_links.csv",
            list(condition_to_motif[0].keys()) if condition_to_motif else ["normalized_condition_label"],
            condition_to_motif,
        )
        _write_csv(
            CONDITION_LAYER_TABLES_ROOT / "condition_to_neighborhood_links.csv",
            list(condition_to_neighborhood[0].keys()) if condition_to_neighborhood else ["normalized_condition_label"],
            condition_to_neighborhood,
        )
        _write_csv(
            CONDITION_LAYER_TABLES_ROOT / "condition_support_summary.csv",
            list(condition_support_summary[0].keys()) if condition_support_summary else ["normalized_condition_label"],
            condition_support_summary,
        )
        _write_csv(
            CONDITION_LAYER_TABLES_ROOT / "condition_comparison_summary.csv",
            list(comparison_summary[0].keys()) if comparison_summary else ["source_id"],
            comparison_summary,
        )
        _write_csv(
            CONDITION_LAYER_TABLES_ROOT / "condition_trajectory_summary.csv",
            list(trajectory_summary[0].keys()) if trajectory_summary else ["source_id"],
            trajectory_summary,
        )
        _write_csv(
            CONDITION_LAYER_TABLES_ROOT / "condition_vs_motif_overlap_matrix.csv",
            list(overlap_matrix_rows[0].keys()) if overlap_matrix_rows else ["normalized_condition_label"],
            overlap_matrix_rows,
        )
        _write_csv(
            CONDITION_LAYER_TABLES_ROOT / "ambiguous_condition_support.csv",
            list(ambiguous_condition_support[0].keys()) if ambiguous_condition_support else ["normalized_condition_label"],
            ambiguous_condition_support,
        )

        contributing_papers = len({row.source_id for row in context_rows if row.context_role == "condition"})
        condition_linked_motifs = len({row["pattern_id"] for row in condition_to_motif})
        condition_linked_neighborhoods = len({row["neighborhood_id"] for row in condition_to_neighborhood})
        signal_counter = Counter(row["signal_characterization"] for row in condition_support_summary)

        implementation_note = f"""# Implementation Note

This pass adds a conservative condition layer on top of the existing structured paper evidence without reprocessing PDFs or introducing inference.

Design:
- source-level condition metadata are normalized into `evidence.condition_ontology`
- comparison structure is stored in `evidence.paper_condition_context`
- condition-to-neighborhood and condition-to-motif links are generated only from already-ingested structured evidence rows
- control labels are preserved as comparison metadata rather than treated as positive biomarker evidence

Important constraints preserved:
- no new literature was ingested
- no motif was reinterpreted as disease-specific by itself
- multiple conditions can share the same motif
- ambiguous and sparse condition support remain explicit
"""
        (CONDITION_LAYER_REPORT_ROOT / "implementation_note.md").write_text(implementation_note)

        current_state = f"""# Current State Assessment

- Unique conditions extracted: `{len(condition_registry)}`.
- Condition families: `{len(condition_family_summary)}`.
- Papers contributing condition-aware evidence: `{contributing_papers}`.
- Condition-linked motifs: `{condition_linked_motifs}`.
- Condition-linked neighborhoods: `{condition_linked_neighborhoods}`.
- Signal characterization counts: `{dict(signal_counter)}`.

Condition signals in the current warehouse are mixed:
- some conditions show discriminative tendency only at the descriptive metadata level
- many motifs are overlapping across conditions
- several conditions remain sparse or local-only

Readiness:
- condition-aware retrieval: `yes`
- early-stage inference: `no`
- dataset-level validation: `no`

This layer is suitable as a conservative bridge between evidence and future inference, but it is not itself an inference engine and should not be treated as disease-biomarker truth.
"""
        (CONDITION_LAYER_REPORT_ROOT / "current_state_assessment.md").write_text(current_state)

        return {
            "condition_count": len(condition_registry),
            "condition_family_count": len(condition_family_summary),
            "papers_contributing": contributing_papers,
            "condition_to_motif_link_count": len(condition_to_motif),
            "condition_to_neighborhood_link_count": len(condition_to_neighborhood),
            "condition_linked_motifs": condition_linked_motifs,
            "condition_linked_neighborhoods": condition_linked_neighborhoods,
            "signal_counter": dict(signal_counter),
        }
    finally:
        connection.close()


if __name__ == "__main__":
    print(json.dumps(run_condition_ontology_layer(), indent=2, sort_keys=True))
