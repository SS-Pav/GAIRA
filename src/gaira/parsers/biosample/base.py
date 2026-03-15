from abc import abstractmethod

from gaira.parsers.base import DatasetParser


class BiosampleParserBase(DatasetParser):
    """Base interface for future biosample dataset parsers."""

    @abstractmethod
    def extract_metadata(self):
        """Read biosample-level metadata into a dataframe."""

    @abstractmethod
    def extract_spectra(self):
        """Read full biosample spectra into a dataframe."""

    @abstractmethod
    def extract_spectrum_points(self):
        """Explode full spectra into one row per spectral point when needed."""

    @abstractmethod
    def extract_peaks(self):
        """Read or detect peak lists if the dataset provides them."""

    @abstractmethod
    def ingest(self):
        """Run the full biosample ingestion workflow."""
