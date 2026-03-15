from abc import abstractmethod

from gaira.parsers.base import DatasetParser


class KnowledgeParserBase(DatasetParser):
    """Base interface for future literature and RAG ingestion parsers."""

    @abstractmethod
    def extract_sources(self):
        """Read source-level literature metadata into a dataframe."""

    @abstractmethod
    def extract_chunks(self):
        """Chunk text into retrieval-friendly records."""

    @abstractmethod
    def extract_peak_assignments(self):
        """Extract literature peak assignment statements."""

    @abstractmethod
    def extract_biomarker_claims(self):
        """Extract biomarker claim statements from the source material."""

    @abstractmethod
    def extract_confounder_notes(self):
        """Extract notes about confounders and mitigation strategies."""

    @abstractmethod
    def ingest(self):
        """Run the full knowledge ingestion workflow."""
