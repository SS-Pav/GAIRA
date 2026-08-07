"""GAIRA V7 input adapters.

Registered adapters are tried in order by `load()`. Adding a format — Renishaw, B&W Tek, SPC,
JCAMP-DX — means writing a class satisfying `SpectrumAdapter` and appending it to `ADAPTERS`.
No other GAIRA module changes.
"""
from __future__ import annotations

from pathlib import Path

from .arrays import ArrayAdapter
from .base import ParsedSpectrum, SpectrumAdapter, err, info, warn
from .text import DelimitedTextAdapter

ADAPTERS: list = [ArrayAdapter(), DelimitedTextAdapter()]

# Formats with a defined interface but no implementation. Named so a caller receives an honest
# "not implemented" rather than a confusing parse failure.
PLANNED_FORMATS: dict[str, str] = {
    ".spc": "Thermo Galactic SPC — binary; needs a dedicated reader",
    ".jdx": "JCAMP-DX — needs compressed-ASCII (DIFDUP) decoding",
    ".dx": "JCAMP-DX",
    ".wdf": "Renishaw WiRE — binary",
    ".sp": "PerkinElmer — binary",
}


def load(payload, filename: str | None = None) -> ParsedSpectrum:
    """Parse a spectrum from bytes, text, or a (wavenumber, intensity) pair."""
    suffix = Path(filename).suffix.lower() if filename else ""
    if suffix in PLANNED_FORMATS:
        return ParsedSpectrum(
            __import__("numpy").array([]), __import__("numpy").array([]),
            [err("input.format_not_supported",
                 f"{suffix} is a planned format, not an implemented one: "
                 f"{PLANNED_FORMATS[suffix]}. Export to CSV or two-column text.",
                 format=suffix)], "unsupported")
    for a in ADAPTERS:
        # Guard only `sniff`, which is allowed to fail cheaply on input it does not recognise.
        # A `parse` that raises is a defect, and swallowing it here would report "unrecognised
        # format" for a file the adapter genuinely accepted — which is how a column-alignment
        # bug hid during Phase 10 development.
        try:
            recognised = a.sniff(payload, filename)
        except Exception:
            continue
        if recognised:
            try:
                return a.parse(payload, filename)
            except Exception as exc:
                return ParsedSpectrum(
                    __import__("numpy").array([]), __import__("numpy").array([]),
                    [err("input.parse_failed",
                         f"the {a.name} adapter recognised this input but failed to parse it: "
                         f"{type(exc).__name__}: {exc}", adapter=a.name)], a.name)
    return ParsedSpectrum(
        __import__("numpy").array([]), __import__("numpy").array([]),
        [err("input.unrecognised",
             "no registered adapter recognised this input; supported formats are CSV, TSV, "
             "two-column text, and (wavenumber, intensity) arrays")], "unknown")


__all__ = ["ADAPTERS", "PLANNED_FORMATS", "ParsedSpectrum", "SpectrumAdapter", "load",
           "DelimitedTextAdapter", "ArrayAdapter", "err", "warn", "info"]
