from gaira.parsers.knowledge.base import KnowledgeParserBase


class ExampleKnowledgeParser(KnowledgeParserBase):
    """Template parser showing the expected knowledge ingestion methods."""

    def audit(self) -> None:
        print("ExampleKnowledgeParser.audit() is a scaffold only.")
        print("Add source inspection logic for a real knowledge dataset here.")

    def extract_metadata(self):
        raise NotImplementedError(
            "Knowledge parsers should use extract_sources() instead of extract_metadata()."
        )

    def extract_spectra(self):
        raise NotImplementedError(
            "Knowledge datasets do not use extract_spectra() in the reference parser sense."
        )

    def extract_sources(self):
        raise NotImplementedError(
            "Implement extract_sources() for a concrete literature or RAG dataset."
        )

    def extract_chunks(self):
        raise NotImplementedError(
            "Implement extract_chunks() to create retrieval-ready text chunks."
        )

    def extract_peak_assignments(self):
        raise NotImplementedError(
            "Implement extract_peak_assignments() for literature-derived peak tables."
        )

    def extract_biomarker_claims(self):
        raise NotImplementedError(
            "Implement extract_biomarker_claims() for paper-derived claims."
        )

    def extract_confounder_notes(self):
        raise NotImplementedError(
            "Implement extract_confounder_notes() for confounder and mitigation notes."
        )

    def ingest(self):
        raise NotImplementedError(
            "Implement ingest() after the source and chunk extraction methods are defined."
        )
