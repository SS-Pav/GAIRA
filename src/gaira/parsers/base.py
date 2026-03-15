from abc import ABC, abstractmethod
from pathlib import Path


class DatasetParser(ABC):
    """Base interface for dataset-specific ingestion parsers."""

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        self.dataset_id = dataset_id
        self.dataset_root = Path(dataset_root)
        self.db_path = Path(db_path)

    @abstractmethod
    def audit(self) -> None:
        """Inspect dataset files and report what is available."""

    @abstractmethod
    def extract_metadata(self) -> None:
        """Read sample-level metadata from the dataset files."""

    @abstractmethod
    def extract_spectra(self) -> None:
        """Find and prepare spectrum file references for ingestion."""

    @abstractmethod
    def ingest(self) -> None:
        """Run the full ingestion workflow for one dataset."""
