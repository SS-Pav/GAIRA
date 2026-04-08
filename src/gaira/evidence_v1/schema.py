from __future__ import annotations

import duckdb


DDL_STATEMENTS = [
    "CREATE SCHEMA IF NOT EXISTS registry",
    "CREATE SCHEMA IF NOT EXISTS evidence",
    "CREATE SCHEMA IF NOT EXISTS features",
    "CREATE SCHEMA IF NOT EXISTS interpretation",
    "CREATE SCHEMA IF NOT EXISTS retrieval",
    "CREATE SCHEMA IF NOT EXISTS context",
    "CREATE SCHEMA IF NOT EXISTS ontology",
    "CREATE SCHEMA IF NOT EXISTS literature",
    """
    CREATE TABLE IF NOT EXISTS registry.evidence_sources (
        source_id TEXT,
        source_name TEXT,
        source_family TEXT,
        source_kind TEXT,
        source_path TEXT,
        parent_dataset_id TEXT,
        source_provenance TEXT,
        evidence_role TEXT,
        default_evidence_tier TEXT,
        direct_retrieval_allowed BOOLEAN,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS registry.warehouse_sources (
        source_id TEXT,
        source_name TEXT,
        source_family TEXT,
        sample_scope TEXT,
        biosample_type TEXT,
        modality TEXT,
        is_reference_grounding BOOLEAN,
        is_disease_or_stress_evidence BOOLEAN,
        disease_class TEXT,
        stress_class TEXT,
        structured_spectral_data_available BOOLEAN,
        structured_peak_assignments_available BOOLEAN,
        digitization_required BOOLEAN,
        repo_origin TEXT,
        source_kind TEXT,
        parent_dataset_id TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.evidence_items (
        evidence_item_id TEXT,
        source_id TEXT,
        source_record_id TEXT,
        evidence_kind TEXT,
        evidence_tier TEXT,
        confidence_label TEXT,
        title TEXT,
        provenance_path TEXT,
        provenance_detail TEXT,
        retrieval_eligible BOOLEAN,
        created_by TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.peak_assignment_evidence (
        evidence_item_id TEXT,
        source_id TEXT,
        assignment_record_id TEXT,
        assignment_origin TEXT,
        study_family TEXT,
        peak_center_cm DOUBLE,
        peak_min_cm DOUBLE,
        peak_max_cm DOUBLE,
        tolerance_cm DOUBLE,
        assigned_molecule TEXT,
        assigned_group_or_theme TEXT,
        sample_type TEXT,
        modality TEXT,
        substrate TEXT,
        matrix_context TEXT,
        manuscript_or_si TEXT,
        figure_or_table_ref TEXT,
        page_or_sheet TEXT,
        extraction_method TEXT,
        confidence_label TEXT,
        evidence_text TEXT,
        is_primary_retrieval_eligible BOOLEAN,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.reference_spectrum_evidence (
        evidence_item_id TEXT,
        source_id TEXT,
        ref_id TEXT,
        source_row_id INTEGER,
        component TEXT,
        biochemical_class TEXT,
        source_origin TEXT,
        citation_or_reference TEXT,
        sample_substrate TEXT,
        modality TEXT,
        preprocessing_summary TEXT,
        x_min DOUBLE,
        x_max DOUBLE,
        n_points INTEGER,
        peak_count INTEGER,
        is_reference_only BOOLEAN,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.grounding_spectrum_evidence (
        evidence_item_id TEXT,
        source_id TEXT,
        grounding_record_id TEXT,
        dataset_id TEXT,
        source_dataset_id TEXT,
        source_family TEXT,
        sample_scope TEXT,
        biosample_type TEXT,
        compound_label TEXT,
        class_label TEXT,
        experiment_family TEXT,
        grounding_role TEXT,
        modality TEXT,
        matrix_context TEXT,
        processing_version TEXT,
        x_min DOUBLE,
        x_max DOUBLE,
        n_points INTEGER,
        peak_count INTEGER,
        structured_peak_assignments_available BOOLEAN,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.wavenumber_mentions (
        evidence_item_id TEXT,
        source_id TEXT,
        mention_record_id TEXT,
        study_family TEXT,
        wavenumber_cm DOUBLE,
        wavenumber_min_cm DOUBLE,
        wavenumber_max_cm DOUBLE,
        mention_text TEXT,
        assigned_molecule_hint TEXT,
        biochemical_theme_hint TEXT,
        sample_type TEXT,
        modality TEXT,
        manuscript_or_si TEXT,
        page_or_sheet TEXT,
        extraction_method TEXT,
        confidence_label TEXT,
        mention_classification TEXT,
        exclusion_reason TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.digitized_spectrum_registry (
        evidence_item_id TEXT,
        source_id TEXT,
        queue_id TEXT,
        study_family TEXT,
        figure_ref TEXT,
        is_spectral_figure TEXT,
        has_source_data_in_dataset_layer TEXT,
        is_methods_paper TEXT,
        priority TEXT,
        priority_reason TEXT,
        digitization_status TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.context_rules (
        rule_id TEXT,
        evidence_item_id TEXT,
        source_id TEXT,
        source_document_id TEXT,
        intended_domain TEXT,
        modality TEXT,
        rule_type TEXT,
        rule_priority INTEGER,
        region_start_cm DOUBLE,
        region_end_cm DOUBLE,
        rule_text TEXT,
        evidence_basis TEXT,
        retrieval_append_allowed BOOLEAN,
        direct_grounding_allowed BOOLEAN,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS features.spectral_features (
        feature_id TEXT,
        evidence_item_id TEXT,
        source_id TEXT,
        feature_type TEXT,
        peak_center_cm DOUBLE,
        peak_min_cm DOUBLE,
        peak_max_cm DOUBLE,
        tolerance_cm DOUBLE,
        feature_weight DOUBLE,
        assigned_label TEXT,
        feature_origin TEXT,
        ambiguity_flag BOOLEAN,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS interpretation.bsv_definitions (
        bsv_id TEXT,
        bsv_label TEXT,
        bsv_family TEXT,
        description TEXT,
        caution_note TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS interpretation.evidence_bsv_links (
        link_id TEXT,
        evidence_item_id TEXT,
        bsv_id TEXT,
        link_origin TEXT,
        link_confidence TEXT,
        link_weight DOUBLE,
        rationale TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS retrieval.retrieval_documents (
        document_id TEXT,
        evidence_item_id TEXT,
        source_id TEXT,
        document_kind TEXT,
        evidence_tier TEXT,
        title TEXT,
        summary_text TEXT,
        domain_hint TEXT,
        modality_hint TEXT,
        direct_retrieval_eligible BOOLEAN,
        provenance_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS retrieval.retrieval_runs (
        run_id TEXT,
        query_peaks_json TEXT,
        domain_hint TEXT,
        modality_hint TEXT,
        query_tolerance_cm DOUBLE,
        top_k INTEGER,
        result_count INTEGER,
        direct_results_json TEXT,
        context_results_json TEXT,
        created_at TIMESTAMP,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.peak_meaning_clusters (
        cluster_id TEXT,
        canonical_peak_cm DOUBLE,
        window_start_cm DOUBLE,
        window_end_cm DOUBLE,
        normalized_meaning_label TEXT,
        normalized_family TEXT,
        family_label_version TEXT,
        mixed_family_flag BOOLEAN,
        overlapping_family_count INTEGER,
        raw_label_diversity_count INTEGER,
        evidence_counts_by_source_type_json TEXT,
        source_diversity_count INTEGER,
        explicit_assignment_count INTEGER,
        curated_assignment_count INTEGER,
        reference_support_count INTEGER,
        aligned_mention_support_count INTEGER,
        bare_mention_excluded_count INTEGER,
        high_priority_digitization_count INTEGER,
        medium_priority_digitization_count INTEGER,
        low_priority_digitization_count INTEGER,
        ambiguity_score DOUBLE,
        confidence_score DOUBLE,
        score_components_json TEXT,
        linked_evidence_ids_json TEXT,
        linked_source_ids_json TEXT,
        linked_digitization_queue_ids_json TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.peak_meaning_support (
        support_id TEXT,
        cluster_id TEXT,
        evidence_item_id TEXT,
        source_id TEXT,
        support_kind TEXT,
        support_role TEXT,
        source_type TEXT,
        peak_center_cm DOUBLE,
        peak_distance_to_cluster_cm DOUBLE,
        normalized_meaning_label TEXT,
        raw_label TEXT,
        support_weight DOUBLE,
        is_primary_support BOOLEAN,
        is_secondary_support BOOLEAN,
        provenance_json TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.operational_mention_links (
        mention_link_id TEXT,
        evidence_item_id TEXT,
        cluster_id TEXT,
        alignment_status TEXT,
        alignment_reason TEXT,
        peak_distance_cm DOUBLE,
        normalized_meaning_label TEXT,
        included_as_secondary_support BOOLEAN,
        support_weight DOUBLE,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.assignment_patterns (
        pattern_id TEXT,
        normalized_subfamily TEXT,
        normalized_family TEXT,
        broader_family TEXT,
        meaning_class TEXT,
        confounder_subclass TEXT,
        spectral_region TEXT,
        pattern_label TEXT,
        pattern_type TEXT,
        canonical_multi_peak BOOLEAN,
        loose_constellation BOOLEAN,
        provisional BOOLEAN,
        core_member_count INTEGER,
        total_member_count INTEGER,
        source_diversity INTEGER,
        evidence_count INTEGER,
        ambiguity_score DOUBLE,
        confidence_score DOUBLE,
        coherence_score DOUBLE,
        separability_score DOUBLE,
        support_strength_score DOUBLE,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.assignment_pattern_members (
        pattern_id TEXT,
        cluster_id TEXT,
        normalized_subfamily TEXT,
        broader_family TEXT,
        meaning_class TEXT,
        confounder_subclass TEXT,
        spectral_region TEXT,
        canonical_peak_cm DOUBLE,
        window_start_cm DOUBLE,
        window_end_cm DOUBLE,
        member_role TEXT,
        member_weight DOUBLE,
        evidence_support_count INTEGER,
        curated_support_count INTEGER,
        explicit_support_count INTEGER,
        reference_support_count INTEGER,
        ambiguity_contribution DOUBLE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS retrieval.peak_meaning_documents (
        document_id TEXT,
        cluster_id TEXT,
        normalized_meaning_label TEXT,
        normalized_family TEXT,
        title TEXT,
        summary_text TEXT,
        canonical_peak_cm DOUBLE,
        window_start_cm DOUBLE,
        window_end_cm DOUBLE,
        domain_hint TEXT,
        modality_hint TEXT,
        direct_retrieval_eligible BOOLEAN,
        confidence_score DOUBLE,
        ambiguity_score DOUBLE,
        mixed_family_flag BOOLEAN,
        overlapping_family_count INTEGER,
        source_diversity_count INTEGER,
        curated_assignment_count INTEGER,
        explicit_assignment_count INTEGER,
        reference_support_count INTEGER,
        aligned_mention_support_count INTEGER,
        score_components_json TEXT,
        applicable_context_node_ids_json TEXT,
        applicable_context_edge_ids_json TEXT,
        support_summary_json TEXT,
        provenance_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS context.context_nodes (
        node_id TEXT,
        node_type TEXT,
        node_label TEXT,
        domain_hint TEXT,
        modality_hint TEXT,
        region_start_cm DOUBLE,
        region_end_cm DOUBLE,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS context.context_edges (
        edge_id TEXT,
        source_node_id TEXT,
        target_node_id TEXT,
        edge_type TEXT,
        weight DOUBLE,
        evidence_basis TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ontology.broader_family_registry (
        broader_family TEXT,
        broader_family_label TEXT,
        description TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ontology.normalized_subfamily_registry (
        subfamily_id TEXT,
        normalized_subfamily TEXT,
        normalized_subfamily_label TEXT,
        broader_family TEXT,
        meaning_class TEXT,
        default_confounder_subclass TEXT,
        status TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ontology.alias_mapping (
        alias_id TEXT,
        alias_text TEXT,
        normalized_subfamily TEXT,
        broader_family TEXT,
        meaning_class TEXT,
        confounder_subclass TEXT,
        mapping_type TEXT,
        source_note TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ontology.evidence_ontology_mappings (
        evidence_item_id TEXT,
        source_id TEXT,
        assignment_record_id TEXT,
        exact_source_phrase TEXT,
        normalized_subfamily TEXT,
        broader_family TEXT,
        meaning_class TEXT,
        confounder_subclass TEXT,
        spectral_region TEXT,
        mapping_status TEXT,
        alias_text_used TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ontology.cluster_ontology_mappings (
        cluster_id TEXT,
        normalized_subfamily TEXT,
        broader_family TEXT,
        meaning_class TEXT,
        confounder_subclass TEXT,
        spectral_region TEXT,
        support_count INTEGER,
        evidence_item_count INTEGER,
        subfamily_support_json TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ontology.pattern_ontology_mappings (
        pattern_id TEXT,
        normalized_subfamily TEXT,
        broader_family TEXT,
        meaning_class TEXT,
        confounder_subclass TEXT,
        spectral_region TEXT,
        member_count INTEGER,
        mapped_cluster_count INTEGER,
        subfamily_member_json TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ontology.new_subfamily_candidates (
        candidate_id TEXT,
        exact_source_phrase TEXT,
        proposed_normalized_subfamily TEXT,
        proposed_broader_family TEXT,
        meaning_class TEXT,
        confounder_subclass TEXT,
        spectral_region TEXT,
        evidence_count INTEGER,
        source_ids_json TEXT,
        status TEXT,
        rationale TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.local_support_neighborhoods (
        neighborhood_id TEXT,
        canonical_peak_cm DOUBLE,
        window_start_cm DOUBLE,
        window_end_cm DOUBLE,
        spectral_region TEXT,
        dominant_normalized_subfamily TEXT,
        candidate_normalized_subfamilies_json TEXT,
        broader_family TEXT,
        meaning_class TEXT,
        confounder_subclass TEXT,
        local_confidence_score DOUBLE,
        local_ambiguity_score DOUBLE,
        source_diversity_count INTEGER,
        evidence_row_count INTEGER,
        motif_link_count INTEGER,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.local_support_neighborhood_members (
        neighborhood_id TEXT,
        evidence_item_id TEXT,
        assignment_record_id TEXT,
        source_id TEXT,
        peak_center_cm DOUBLE,
        exact_source_phrase TEXT,
        normalized_subfamily TEXT,
        broader_family TEXT,
        meaning_class TEXT,
        confounder_subclass TEXT,
        spectral_region TEXT,
        evidence_origin TEXT,
        support_role TEXT,
        is_dominant_subfamily BOOLEAN,
        provenance_json TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.neighborhood_motif_links (
        neighborhood_id TEXT,
        pattern_id TEXT,
        link_role TEXT,
        strongest_member_role TEXT,
        shared_normalized_subfamily TEXT,
        matched_member_count INTEGER,
        link_confidence DOUBLE,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.paper_assignment_qc (
        evidence_item_id TEXT,
        source_id TEXT,
        assignment_record_id TEXT,
        extraction_method TEXT,
        qc_classification TEXT,
        include_in_local_layer BOOLEAN,
        include_in_motif_layer BOOLEAN,
        is_primary_after_qc BOOLEAN,
        aligned_neighborhood_id TEXT,
        aligned_pattern_id TEXT,
        rationale TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.paper_enrichment_events (
        enrichment_event_id TEXT,
        source_id TEXT,
        artifact_id TEXT,
        artifact_type TEXT,
        figure_or_si_ref TEXT,
        extraction_method TEXT,
        caption_or_context_text TEXT,
        peak_center_cm DOUBLE,
        peak_min_cm DOUBLE,
        peak_max_cm DOUBLE,
        spectral_region TEXT,
        normalized_subfamily TEXT,
        broader_family TEXT,
        meaning_class TEXT,
        confounder_subclass TEXT,
        linked_evidence_item_id TEXT,
        linked_neighborhood_id TEXT,
        linked_pattern_id TEXT,
        impact_type TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.condition_ontology (
        condition_id TEXT,
        raw_condition_text TEXT,
        normalized_condition_label TEXT,
        condition_family TEXT,
        aliases_json TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.paper_condition_context (
        source_id TEXT,
        raw_condition_text TEXT,
        normalized_condition_label TEXT,
        condition_family TEXT,
        sample_type TEXT,
        experimental_context TEXT,
        control_group_present BOOLEAN,
        control_label TEXT,
        comparison_type TEXT,
        trajectory_type TEXT,
        context_role TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.condition_to_motif_links (
        normalized_condition_label TEXT,
        condition_family TEXT,
        source_id TEXT,
        pattern_id TEXT,
        sample_type TEXT,
        experimental_context TEXT,
        comparison_type TEXT,
        trajectory_type TEXT,
        evidence_row_count INTEGER,
        primary_evidence_count INTEGER,
        secondary_evidence_count INTEGER,
        digitized_figure_count INTEGER,
        text_assignment_count INTEGER,
        regex_secondary_count INTEGER,
        evidence_type_composition_json TEXT,
        support_strength DOUBLE,
        ambiguity_score DOUBLE,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.condition_to_neighborhood_links (
        normalized_condition_label TEXT,
        condition_family TEXT,
        source_id TEXT,
        neighborhood_id TEXT,
        sample_type TEXT,
        experimental_context TEXT,
        comparison_type TEXT,
        trajectory_type TEXT,
        evidence_row_count INTEGER,
        primary_evidence_count INTEGER,
        secondary_evidence_count INTEGER,
        digitized_figure_count INTEGER,
        text_assignment_count INTEGER,
        regex_secondary_count INTEGER,
        evidence_type_composition_json TEXT,
        support_strength DOUBLE,
        ambiguity_score DOUBLE,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence.condition_support_summary (
        normalized_condition_label TEXT,
        condition_family TEXT,
        contributing_paper_count INTEGER,
        motif_count INTEGER,
        neighborhood_count INTEGER,
        dominant_broader_families_json TEXT,
        spectral_regions_json TEXT,
        confounder_involvement_count INTEGER,
        shared_motif_count INTEGER,
        unique_motif_count INTEGER,
        signal_characterization TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS literature.candidate_papers (
        paper_id TEXT,
        title TEXT,
        authors TEXT,
        year INTEGER,
        journal TEXT,
        doi TEXT,
        source TEXT,
        source_list_json TEXT,
        abstract TEXT,
        query_source TEXT,
        disease_keywords_detected TEXT,
        sample_keywords_detected TEXT,
        spectral_keywords_detected TEXT,
        title_key TEXT,
        already_in_local_corpus BOOLEAN,
        already_processed_in_evidence BOOLEAN,
        open_access_candidate BOOLEAN,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS literature.paper_assets (
        asset_id TEXT,
        paper_id TEXT,
        source_provider TEXT,
        asset_kind TEXT,
        manuscript_pdf_path TEXT,
        supplementary_files_json TEXT,
        source_data_links_json TEXT,
        figures_detected BOOLEAN,
        tables_detected BOOLEAN,
        remote_url TEXT,
        local_path TEXT,
        file_type TEXT,
        download_status TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS literature.paper_triage (
        paper_id TEXT,
        condition_relevance_score DOUBLE,
        sample_relevance_score DOUBLE,
        spectral_density_score DOUBLE,
        comparison_structure_score DOUBLE,
        figure_value_score DOUBLE,
        si_value_score DOUBLE,
        final_score DOUBLE,
        triage_decision TEXT,
        decision_rationale TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS literature.processing_queue (
        queue_id TEXT,
        paper_id TEXT,
        queue_status TEXT,
        rank_order INTEGER,
        selected_for_ingestion BOOLEAN,
        asset_ready BOOLEAN,
        ingestion_attempted BOOLEAN,
        extraction_row_count INTEGER,
        selection_reason TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS literature.paper_asset_resolution (
        paper_id TEXT,
        canonical_article_url TEXT,
        doi_resolved BOOLEAN,
        manuscript_asset_status TEXT,
        supplementary_asset_status TEXT,
        source_data_asset_status TEXT,
        manuscript_local_path TEXT,
        supplementary_local_paths_json TEXT,
        source_data_local_paths_json TEXT,
        readiness_status TEXT,
        asset_resolution_notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS literature.resolved_assets (
        resolved_asset_id TEXT,
        paper_id TEXT,
        asset_type TEXT,
        source_url TEXT,
        local_path TEXT,
        file_type TEXT,
        resolution_method TEXT,
        resolution_status TEXT,
        checksum_sha256 TEXT,
        manuscript_or_si TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS literature.blocked_assets (
        blocked_asset_id TEXT,
        paper_id TEXT,
        title TEXT,
        doi TEXT,
        publisher TEXT,
        canonical_article_url TEXT,
        blocker_type TEXT,
        blocker_detail TEXT,
        manuscript_status TEXT,
        supplementary_status TEXT,
        source_data_status TEXT,
        manual_download_needed BOOLEAN,
        manual_upload_pending BOOLEAN,
        retry_recommended BOOLEAN,
        suggested_manual_action TEXT,
        preferred_asset_type_to_rescue_first TEXT,
        local_upload_target_path TEXT,
        resolved_manually BOOLEAN,
        resolved_manually_at TIMESTAMP,
        resolved_manually_notes TEXT,
        priority_score DOUBLE,
        last_attempted_at TIMESTAMP,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS literature.asset_truth_validation (
        asset_validation_id TEXT,
        paper_id TEXT,
        asset_type TEXT,
        local_path TEXT,
        source_url TEXT,
        expected_file_type TEXT,
        detected_file_type TEXT,
        file_exists BOOLEAN,
        header_valid BOOLEAN,
        parseable_text BOOLEAN,
        page_count INTEGER,
        text_char_count INTEGER,
        validation_status TEXT,
        readiness_impact TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS literature.queue_partition (
        paper_id TEXT,
        queue_partition TEXT,
        readiness_basis TEXT,
        oa_candidate BOOLEAN,
        priority_score DOUBLE,
        notes TEXT
    )
    """,
]

V1_TABLES = [
    "registry.evidence_sources",
    "registry.warehouse_sources",
    "evidence.evidence_items",
    "evidence.peak_assignment_evidence",
    "evidence.reference_spectrum_evidence",
    "evidence.grounding_spectrum_evidence",
    "evidence.wavenumber_mentions",
    "evidence.digitized_spectrum_registry",
    "evidence.context_rules",
    "features.spectral_features",
    "interpretation.bsv_definitions",
    "interpretation.evidence_bsv_links",
    "retrieval.retrieval_documents",
    "retrieval.retrieval_runs",
]

REFINEMENT_TABLES = [
    "evidence.peak_meaning_clusters",
    "evidence.peak_meaning_support",
    "evidence.operational_mention_links",
    "evidence.assignment_pattern_members",
    "evidence.assignment_patterns",
    "retrieval.peak_meaning_documents",
    "context.context_nodes",
    "context.context_edges",
    "retrieval.retrieval_runs",
]

ONTOLOGY_TABLES = [
    "ontology.new_subfamily_candidates",
    "ontology.pattern_ontology_mappings",
    "ontology.cluster_ontology_mappings",
    "ontology.evidence_ontology_mappings",
    "ontology.alias_mapping",
    "ontology.normalized_subfamily_registry",
    "ontology.broader_family_registry",
]

LOCAL_NEIGHBORHOOD_TABLES = [
    "evidence.neighborhood_motif_links",
    "evidence.local_support_neighborhood_members",
    "evidence.local_support_neighborhoods",
]

PAPER_QC_TABLES = [
    "evidence.paper_assignment_qc",
]

ENRICHMENT_TABLES = [
    "evidence.paper_enrichment_events",
]

CONDITION_LAYER_TABLES = [
    "evidence.condition_support_summary",
    "evidence.condition_to_neighborhood_links",
    "evidence.condition_to_motif_links",
    "evidence.paper_condition_context",
    "evidence.condition_ontology",
]

LITERATURE_TABLES = [
    "literature.processing_queue",
    "literature.paper_triage",
    "literature.paper_assets",
    "literature.candidate_papers",
]

LITERATURE_ASSET_RESOLUTION_TABLES = [
    "literature.resolved_assets",
    "literature.paper_asset_resolution",
]

LITERATURE_BLOCKED_ASSET_TABLES = [
    "literature.blocked_assets",
]

LITERATURE_TRUTH_TABLES = [
    "literature.queue_partition",
    "literature.asset_truth_validation",
]


def initialize_schema(connection: duckdb.DuckDBPyConnection) -> None:
    for statement in DDL_STATEMENTS:
        connection.execute(statement)


def reset_v1_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for table_name in reversed(V1_TABLES):
        connection.execute(f"DELETE FROM {table_name}")


def reset_phase1_refinement_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for table_name in reversed(REFINEMENT_TABLES):
        connection.execute(f"DROP TABLE IF EXISTS {table_name}")
    for statement in DDL_STATEMENTS:
        connection.execute(statement)


def reset_pattern_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("DROP TABLE IF EXISTS evidence.assignment_pattern_members")
    connection.execute("DROP TABLE IF EXISTS evidence.assignment_patterns")
    connection.execute("DELETE FROM retrieval.retrieval_runs")
    for statement in DDL_STATEMENTS:
        connection.execute(statement)


def reset_ontology_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for table_name in ONTOLOGY_TABLES:
        connection.execute(f"DELETE FROM {table_name}")


def reset_local_neighborhood_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for table_name in LOCAL_NEIGHBORHOOD_TABLES:
        connection.execute(f"DELETE FROM {table_name}")


def reset_paper_qc_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for table_name in PAPER_QC_TABLES:
        connection.execute(f"DELETE FROM {table_name}")


def reset_enrichment_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for table_name in ENRICHMENT_TABLES:
        connection.execute(f"DELETE FROM {table_name}")


def reset_condition_layer_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for table_name in CONDITION_LAYER_TABLES:
        connection.execute(f"DELETE FROM {table_name}")


def reset_literature_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for table_name in LITERATURE_TABLES:
        connection.execute(f"DELETE FROM {table_name}")


def reset_literature_asset_resolution_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for table_name in LITERATURE_ASSET_RESOLUTION_TABLES:
        connection.execute(f"DELETE FROM {table_name}")


def reset_literature_blocked_asset_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for table_name in LITERATURE_BLOCKED_ASSET_TABLES:
        connection.execute(f"DELETE FROM {table_name}")


def reset_literature_truth_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for table_name in LITERATURE_TRUTH_TABLES:
        connection.execute(f"DELETE FROM {table_name}")
