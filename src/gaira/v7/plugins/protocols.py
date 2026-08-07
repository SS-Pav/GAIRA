"""GAIRA V7 — Phase 10: extension contracts.

**Specifications only.** Nothing in this package performs inference, and nothing may. A stub
that returns plausible numbers is worse than no stub at all, so every unimplemented adapter here
raises rather than guesses, and a test asserts it.

The architectural line this package draws:

    GAIRA Core  — scientific representation.
                  Preprocessing, CSM projection, grounded retrieval, Chemistry Evidence,
                  confidence, provenance. Frozen after Phase 09. Raman only.

    Modality adapters — the physics between the sample and the spectrum.
                  Substrate observation models, modality-specific detection gates, transfer
                  functions, wavelength corrections. A modality adapter may transform a spectrum
                  BEFORE the core sees it, and may veto; it may never alter what the core computes.

    Context adapters — the biology around the measurement.
                  Domain caveats, interpretation framing, domain-specific evidence weighting,
                  dataset context. A context adapter operates on the core's OUTPUT and may only
                  add framing and caveats.

    Interpretation adapters — how an answer is narrated.
                  Deterministic templates today. A future language model would sit here, ABOVE
                  the validated science, never inside it.

    Trajectory adapters — time.
                  DART and other time-resolved work consumes a SEQUENCE of core results. It
                  belongs downstream of the spectral representation because a trajectory of
                  CSM activations is only meaningful if each activation was computed identically.

Why the ordering matters: `scientific representation ≠ domain interpretation`. Phase 04 measured
what happens when the two are confused — a Raman motif dictionary reconstructs Ag-SERS of the
same metabolites comfortably (AUROC 0.548), so a SERS spectrum run through the Raman core
produces confident numbers with no validated meaning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from gaira.v7.contracts import (Diagnostic, InferenceResult, Modality, SampleType, Severity)


class NotImplementedAdapter(NotImplementedError):
    """Raised by a declared-but-unimplemented adapter. Never returns a fabricated result."""


@dataclass(frozen=True)
class ModalityDecision:
    """A modality adapter's verdict on whether the core may run, and on what."""
    admissible: bool
    wavenumber: np.ndarray | None
    intensity: np.ndarray | None
    diagnostics: tuple[Diagnostic, ...] = ()
    transfer_applied: str = "none"


@runtime_checkable
class ModalityAdapter(Protocol):
    """The physics between sample and spectrum.

    Runs BEFORE the core. It may correct, veto, or pass through. It may not touch the core's
    dictionaries, its retrieval, or its chemistry model.
    """
    modality: Modality
    name: str
    implemented: bool

    def admit(self, wavenumber: np.ndarray, intensity: np.ndarray,
              metadata: dict) -> ModalityDecision:
        """Decide whether this spectrum may enter the Raman core, and in what form."""


@dataclass(frozen=True)
class ContextFraming:
    """A context adapter's contribution: caveats and framing, never numbers."""
    caveats: tuple[str, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    framing: str = ""
    evidence_weighting: dict[str, float] | None = None   # advisory; applied by no V7 code path


@runtime_checkable
class SampleContextAdapter(Protocol):
    """The biology around the measurement. Runs AFTER the core, on its output."""
    sample_type: SampleType
    name: str
    implemented: bool

    def frame(self, result: InferenceResult) -> ContextFraming:
        """Add domain caveats and framing to a completed result."""


@runtime_checkable
class InterpretationAdapter(Protocol):
    """How a result is narrated. Deterministic templates today.

    A language model would live here, above the validated science. It must be given the
    engine's result and may only rephrase it — it may never compute a scientific quantity, and
    every number it repeats must be traceable to an `InferenceResult` field.
    """
    name: str
    deterministic: bool

    def narrate(self, result: InferenceResult) -> str:
        ...


@runtime_checkable
class TrajectoryAdapter(Protocol):
    """Time-resolved analysis over a sequence of completed core results.

    Downstream by construction: a trajectory of CSM activations means something only if every
    activation was produced by the same frozen path.
    """
    name: str
    implemented: bool

    def analyse(self, results: Sequence[InferenceResult], timestamps: Sequence[float]) -> dict:
        ...


def scope_diagnostic(code: str, message: str, **detail) -> Diagnostic:
    return Diagnostic(severity=Severity.WARNING, code=code, message=message, detail=detail)
