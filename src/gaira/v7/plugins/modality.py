"""GAIRA V7 — Phase 10: modality adapters.

One implementation (Raman, which is the identity) and four declared-but-unimplemented stubs.
Every stub raises. None fabricates a result.
"""
from __future__ import annotations

import numpy as np

from gaira.v7.contracts import Diagnostic, Modality, Severity

from .protocols import ModalityDecision, NotImplementedAdapter


class PureRamanAdapter:
    """The current V7 implementation: pass-through.

    Raman is what the core was built and validated on, so the adapter's whole job is to say so.
    It exists as a class rather than as an implicit default so that the modality layer has a
    concrete reference implementation to check the protocol against.
    """
    modality = Modality.RAMAN
    name = "pure_raman"
    implemented = True

    def admit(self, wavenumber, intensity, metadata) -> ModalityDecision:
        return ModalityDecision(
            admissible=True, wavenumber=np.asarray(wavenumber, float),
            intensity=np.asarray(intensity, float), transfer_applied="none",
            diagnostics=(Diagnostic(
                severity=Severity.INFO, code="modality.raman",
                message="Pure Raman: the core's validated domain. No transfer applied."),))


class _Unimplemented:
    """Shared refusal. What a future implementation must supply is stated, not stubbed."""
    modality: Modality
    name: str
    implemented = False
    requires: tuple[str, ...] = ()

    def admit(self, wavenumber, intensity, metadata) -> ModalityDecision:
        raise NotImplementedAdapter(
            f"{self.name} is a declared extension point, not an implementation. Running "
            f"{self.modality.value} through the Raman core would produce confident numbers with "
            f"no validated meaning — Phase 04 measured a Raman motif dictionary reconstructing "
            f"real Ag-SERS at AUROC 0.548. A working adapter must first supply: "
            f"{'; '.join(self.requires)}.")


class AgSERSAdapter(_Unimplemented):
    modality = Modality.AG_SERS
    name = "ag_sers"
    requires = ("a silver-substrate observation model (enhancement is analyte- and "
                "orientation-dependent, not a constant factor)",
                "a detection gate — Phase 04 showed Ag homogenises many analytes onto a purine "
                "attractor, so which bands survive is a measured question",
                "a validated transfer function from SERS to the Raman motif basis",
                "its own held-out validation corpus; no Raman number transfers")


class AuSERSAdapter(_Unimplemented):
    modality = Modality.AU_SERS
    name = "au_sers"
    requires = ("a gold-substrate observation model",
                "a detection gate distinct from silver's — the chemisorption chemistry differs",
                "a validated transfer function",
                "its own held-out validation corpus")


class SERSGenericAdapter(_Unimplemented):
    modality = Modality.SERS
    name = "sers_generic"
    requires = ("substrate identification — 'SERS' without a named substrate is not a "
                "well-defined measurement channel",
                "everything the substrate-specific adapters require")


class DARTAdapter(_Unimplemented):
    modality = Modality.DART
    name = "dart"
    requires = ("a mass-spectrometric to vibrational correspondence, which does not exist as a "
                "spectral transform",
                "a decision about whether DART belongs at the modality layer at all — DART is "
                "better modelled as a TRAJECTORY over an orthogonal measurement, downstream of "
                "the spectral representation rather than upstream of it")


REGISTRY: dict[Modality, object] = {
    Modality.RAMAN: PureRamanAdapter(),
    Modality.AG_SERS: AgSERSAdapter(),
    Modality.AU_SERS: AuSERSAdapter(),
    Modality.SERS: SERSGenericAdapter(),
    Modality.DART: DARTAdapter(),
}


def get(modality: Modality):
    if modality not in REGISTRY:
        raise NotImplementedAdapter(f"no adapter declared for modality {modality.value!r}")
    return REGISTRY[modality]
