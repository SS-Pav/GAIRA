from pathlib import Path

from gaira.parsers.base import DatasetParser


class RamanBioLibParser(DatasetParser):
    """Scaffold parser for the RamanBioLib dataset."""

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)

    def audit(self) -> None:
        """Inspect the dataset folder and report what is present."""
        print(f"Auditing dataset folder: {self.dataset_root}")
        print("TODO: List expected RamanBioLib files and check which ones are available.")

    def extract_metadata(self) -> None:
        """Extract sample metadata from RamanBioLib source files."""
        print("Preparing to extract RamanBioLib sample metadata.")
        print("TODO: Read metadata tables and map fields into the samples table schema.")

    def extract_spectra(self) -> None:
        """Locate spectrum files and prepare spectrum-level records."""
        print("Preparing to locate RamanBioLib spectra files.")
        print("TODO: Identify x/y arrays or source files and record their paths.")

    def ingest(self) -> None:
        """Run the future end-to-end RamanBioLib ingestion flow."""
        print(f"Starting ingestion scaffold for dataset: {self.dataset_id}")
        self.audit()
        print("TODO: Call extract_metadata() and extract_spectra() once parser logic is added.")
        print(f"No database rows were written. Database path is: {self.db_path}")
