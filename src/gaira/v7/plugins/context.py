"""GAIRA V7 — Phase 10: sample-context adapters.

One implementation (pure analyte) and six declared stubs. A context adapter frames a completed
result; it never changes a number, and the protocol gives it no way to.
"""
from __future__ import annotations

from gaira.v7.contracts import Diagnostic, SampleType, Severity

from .protocols import ContextFraming, NotImplementedAdapter


class PureAnalyteContext:
    """The validated context. Every V7 number was measured here."""
    sample_type = SampleType.PURE
    name = "pure_analyte"
    implemented = True

    def frame(self, result) -> ContextFraming:
        return ContextFraming(
            framing="Pure reference compound — the context in which every V7 performance figure "
                    "was measured.",
            caveats=("Chemistry Evidence remains relative even here; a pure compound does not "
                     "make the radar a concentration.",),
            diagnostics=(Diagnostic(severity=Severity.INFO, code="context.pure",
                                    message="Validated interpretation context."),))


class _Unimplemented:
    sample_type: SampleType
    name: str
    implemented = False
    open_questions: tuple[str, ...] = ()

    def frame(self, result) -> ContextFraming:
        raise NotImplementedAdapter(
            f"{self.name} is a declared extension point, not an implementation. V7 has no "
            f"validated interpretation capability for {self.sample_type.value} samples. Open "
            f"questions a working adapter must answer: {'; '.join(self.open_questions)}.")


class MixtureContext(_Unimplemented):
    sample_type = SampleType.MIXTURE
    name = "mixture"
    open_questions = ("whether CSM activation shares track component proportions at all — "
                      "unmeasured, and the L2 normalisation removes absolute scale before "
                      "anything else happens",
                      "how a minor component's evidence should be distinguished from noise")


class SerumContext(_Unimplemented):
    sample_type = SampleType.SERUM
    name = "serum"
    open_questions = ("which analytes are visible at physiological concentration — prior GAIRA "
                      "work found concentration and visibility are not the same thing",
                      "how albumin dominance should be handled",
                      "a serum-specific held-out validation corpus")


class PlasmaContext(_Unimplemented):
    sample_type = SampleType.PLASMA
    name = "plasma"
    open_questions = ("everything serum requires, plus the anticoagulant's own signature",)


class EVContext(_Unimplemented):
    sample_type = SampleType.EV
    name = "extracellular_vesicle"
    open_questions = ("membrane versus cargo attribution",
                      "whether the lipid signal saturates the representation",
                      "isolation-method confounding, which is large in published EV Raman work")


class BacteriaContext(_Unimplemented):
    sample_type = SampleType.BACTERIA
    name = "bacteria"
    open_questions = ("whether envelope-level abstraction survives transfer — prior GAIRA work "
                      "on a 78.5k-spectrum benchmark kept organism identity but lost the "
                      "Gram/envelope abstraction",
                      "growth-phase and medium confounding")


class TissueContext(_Unimplemented):
    sample_type = SampleType.TISSUE
    name = "tissue"
    open_questions = ("spatial heterogeneity — a tissue spectrum is a mixture whose composition "
                      "varies within the illuminated volume",
                      "fixation and preparation artefacts",
                      "whether a per-pixel result is even the right unit")


REGISTRY: dict[SampleType, object] = {
    SampleType.PURE: PureAnalyteContext(),
    SampleType.MIXTURE: MixtureContext(),
    SampleType.SERUM: SerumContext(),
    SampleType.PLASMA: PlasmaContext(),
    SampleType.EV: EVContext(),
    SampleType.BACTERIA: BacteriaContext(),
    SampleType.TISSUE: TissueContext(),
}


def get(sample_type: SampleType):
    if sample_type not in REGISTRY:
        raise NotImplementedAdapter(f"no adapter declared for sample type {sample_type.value!r}")
    return REGISTRY[sample_type]
