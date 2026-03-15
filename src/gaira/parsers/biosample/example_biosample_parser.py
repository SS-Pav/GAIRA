from gaira.parsers.biosample.base import BiosampleParserBase


class ExampleBiosampleParser(BiosampleParserBase):
    """Template parser showing the expected biosample ingestion methods."""

    def audit(self) -> None:
        print("ExampleBiosampleParser.audit() is a scaffold only.")
        print("Add file inspection logic for a real biosample dataset here.")

    def extract_metadata(self):
        raise NotImplementedError(
            "Implement extract_metadata() for a concrete biosample dataset parser."
        )

    def extract_spectra(self):
        raise NotImplementedError(
            "Implement extract_spectra() for a concrete biosample dataset parser."
        )

    def extract_spectrum_points(self):
        raise NotImplementedError(
            "Implement extract_spectrum_points() when full spectra need to be exploded."
        )

    def extract_peaks(self):
        raise NotImplementedError(
            "Implement extract_peaks() if the biosample dataset includes peak lists."
        )

    def ingest(self):
        raise NotImplementedError(
            "Implement ingest() after metadata and spectra extraction are defined."
        )
