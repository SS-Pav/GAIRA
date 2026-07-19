"""GAIRA engine — DART integration INTERFACES (Part 14).

DESIGN ONLY. Nothing here is implemented. These abstract interfaces mark exactly
where future electrochemical perturbation trajectories enter the frozen-atlas
architecture and where trajectory fingerprints will be compared against the
reference perturbation-trajectory library built in the Perturbation Response Audit.

A DART experiment produces a time/potential SERIES of spectra. Projected into the
frozen atlas each series becomes a TRAJECTORY of BSVs — the same object type this
codebase already builds for chemical dose series. The interfaces below are the seam
between that existing machinery and a future DART front-end.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class TrajectoryPoint:
    """One point of a perturbation trajectory (potential/time-stamped BSV)."""
    control_variable: float          # e.g. applied potential (V) or time (s)
    bsv: object                      # a BSV
    ood_score: float


class DARTAcquisition(ABC):
    """FUTURE: yields modality-aware, atlas-native spectra from a DART cell."""

    @abstractmethod
    def stream(self) -> Sequence:      # -> Iterable[(control_variable, spectrum)]
        raise NotImplementedError("DART acquisition is a future phase (interface only).")


class TrajectoryProjector(ABC):
    """FUTURE: projects a DART spectral series into the frozen atlas → BSV trajectory.
    Reuses the SAME frozen-atlas projection and BSVBuilder as chemical dose series;
    nothing about the atlas changes for DART."""

    @abstractmethod
    def project_series(self, series) -> "list[TrajectoryPoint]":
        raise NotImplementedError


class TrajectoryComparator(ABC):
    """FUTURE: compares a DART trajectory fingerprint against the reference
    perturbation-trajectory library (results/.../perturbation_response/artifacts/
    trajectory_library.json). Returns similarity + the nearest reference perturbation."""

    REFERENCE_LIBRARY = ("results/v5_rebuild/perturbation_response/artifacts/"
                         "trajectory_library.json")

    @abstractmethod
    def fingerprint(self, trajectory: "list[TrajectoryPoint]") -> dict:
        """Compute path length / straightness / curvature / component turnover /
        response entropy — the schema already produced for chemical trajectories."""
        raise NotImplementedError

    @abstractmethod
    def compare_to_reference(self, fingerprint: dict) -> dict:
        raise NotImplementedError


# Falsifiable prediction handed forward from the Perturbation Response Audit, to be
# tested by the first DART experiment (documented, not executed):
FIRST_DART_PREDICTION = (
    "Redox-cycling a strong Ag adsorber with a known atlas component (e.g. a purine) "
    "should move the projected trajectory predominantly along that component and "
    "return toward baseline on the reverse sweep. Falsifiable against the reference "
    "trajectory library."
)
