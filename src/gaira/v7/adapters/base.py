"""GAIRA V7 — Phase 10: the input-adapter interface.

An adapter turns some external representation of a spectrum into `(wavenumber, intensity)` plus
structured diagnostics. It never repairs a serious problem silently: an adapter that quietly
drops rows produces a spectrum whose provenance is a lie.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from gaira.v7.contracts import Diagnostic, Severity


@dataclass
class ParsedSpectrum:
    wavenumber: np.ndarray
    intensity: np.ndarray
    diagnostics: list[Diagnostic] = field(default_factory=list)
    source_format: str = "unknown"
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(d.severity is Severity.ERROR for d in self.diagnostics)


@runtime_checkable
class SpectrumAdapter(Protocol):
    """Implement this to add a format. Nothing else in GAIRA needs to change."""

    name: str
    extensions: tuple[str, ...]

    def sniff(self, payload: bytes | str, filename: str | None = None) -> bool:
        """Cheap test: could this adapter handle the payload? No exceptions."""

    def parse(self, payload: bytes | str, filename: str | None = None) -> ParsedSpectrum:
        """Full parse. Raises only on programmer error; data problems become diagnostics."""


def err(code: str, message: str, **detail) -> Diagnostic:
    return Diagnostic(severity=Severity.ERROR, code=code, message=message, detail=detail)


def warn(code: str, message: str, **detail) -> Diagnostic:
    return Diagnostic(severity=Severity.WARNING, code=code, message=message, detail=detail)


def info(code: str, message: str, **detail) -> Diagnostic:
    return Diagnostic(severity=Severity.INFO, code=code, message=message, detail=detail)
