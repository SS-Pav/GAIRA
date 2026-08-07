"""GAIRA V7 — Phase 10: the public data contract.

Typed, versioned schemas for everything that crosses the boundary between the frozen scientific
engine and the outside world. Pydantic v2 models, so one definition serves the Python SDK, the
FastAPI request/response layer, the MCP tool schemas and the report generator.

Design rules, in order of importance:

1. **No NumPy in the public contract.** Arrays cross as lists of floats. A caller should never
   need to know the engine is implemented with NumPy, and a JSON round-trip must be lossless.
2. **Nothing here computes.** These are shapes. Every value is produced by the frozen engine and
   copied across unchanged.
3. **Scope is carried, never silently applied.** `sample_type` and `modality` are recorded and
   may raise warnings; they must not alter a single number the engine computes.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "gaira_v7_contracts_v1"
MAX_POINTS = 200_000          # a spectrum larger than this is a file, not a measurement


class Frozen(BaseModel):
    """Immutable by default. A result object that can be edited is not a record."""
    model_config = ConfigDict(frozen=True, extra="forbid")


# ── enumerations ─────────────────────────────────────────────────────────────
class Modality(str, Enum):
    """What produced the spectrum.

    V7's scientific core is Raman-only (A-09). Every other member exists so an unsupported
    modality can be *named and rejected* rather than silently treated as Raman.
    """
    RAMAN = "raman"
    AG_SERS = "ag_sers"
    AU_SERS = "au_sers"
    SERS = "sers"
    DART = "dart"
    OTHER = "other"


SUPPORTED_MODALITIES: frozenset[Modality] = frozenset({Modality.RAMAN})


class SampleType(str, Enum):
    """The measurement context. Recorded as metadata; never applied to the calculation.

    V7 was built and validated on pure reference compounds. Any other value is accepted so the
    provenance is honest, and produces an explicit scope warning stating that V7 has no validated
    interpretation capability for that context.
    """
    PURE = "pure"
    MIXTURE = "mixture"
    SERUM = "serum"
    PLASMA = "plasma"
    EV = "EV"
    BACTERIA = "bacteria"
    TISSUE = "tissue"
    OTHER = "other"


VALIDATED_SAMPLE_TYPES: frozenset[SampleType] = frozenset({SampleType.PURE})


class Severity(str, Enum):
    ERROR = "error"          # cannot run
    WARNING = "warning"      # runs, interpretation limited
    INFO = "info"            # metadata / scope message


class Diagnostic(Frozen):
    """One validation finding. Structured so a client can group and filter without parsing text."""
    severity: Severity
    code: str = Field(description="stable machine-readable identifier, e.g. 'coverage.partial'")
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


# ── input ────────────────────────────────────────────────────────────────────
class SpectrumInput(Frozen):
    wavenumber: list[float] = Field(min_length=2)
    intensity: list[float] = Field(min_length=2)

    @field_validator("wavenumber", "intensity")
    @classmethod
    def _bounded(cls, v: list[float]) -> list[float]:
        if len(v) > MAX_POINTS:
            raise ValueError(f"spectrum has {len(v)} points; the limit is {MAX_POINTS}")
        return v

    def model_post_init(self, _ctx) -> None:
        if len(self.wavenumber) != len(self.intensity):
            raise ValueError(f"wavenumber has {len(self.wavenumber)} points but intensity has "
                             f"{len(self.intensity)}")


class SpectrumMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modality: Modality = Modality.RAMAN
    sample_type: SampleType = SampleType.PURE
    excitation_nm: Optional[float] = Field(default=None, gt=0, lt=100_000)
    source_name: Optional[str] = Field(default=None, max_length=256)
    sample_id: Optional[str] = Field(default=None, max_length=256)
    notes: Optional[str] = Field(default=None, max_length=4096)


class InferenceOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k_molecules: int = Field(default=10, ge=1, le=154)
    include_lsm: bool = True
    include_csm: bool = True
    include_provenance: bool = True
    include_audit: bool = True
    include_reconstruction: bool = Field(
        default=False, description="return the 676-point CSM reconstruction and residual; large")
    already_preprocessed: bool = Field(
        default=False, description="the spectrum is already on the canonical 676-bin grid")


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spectrum: SpectrumInput
    metadata: SpectrumMetadata = Field(default_factory=SpectrumMetadata)
    options: InferenceOptions = Field(default_factory=InferenceOptions)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: InferenceRequest
    b: InferenceRequest
    label_a: str = Field(default="A", max_length=64)
    label_b: str = Field(default="B", max_length=64)


# ── output ───────────────────────────────────────────────────────────────────
class PreprocessingResult(Frozen):
    n_input_points: int
    input_range: tuple[float, float]
    resampled_to: str
    baseline_method: str
    smoothing: str
    normalisation: str
    n_peaks: int
    signal_quality: float
    snr_estimate: float
    grid_coverage: float = Field(description="fraction of 450-1800 cm-1 the input actually spans")
    warnings: list[str] = Field(default_factory=list)
    processed_intensity: Optional[list[float]] = Field(
        default=None, description="the 676-point canonical spectrum, when requested")
    grid: Optional[list[float]] = None


class MotifActivation(Frozen):
    """One motif's contribution. Used for both LSM and CSM tops."""
    motif_id: str
    weight: float
    share: float
    dominant_bands: list[float] = Field(default_factory=list)
    band_assignment: str = ""
    contributing_lsms: list[str] = Field(default_factory=list)


class LSMResult(Frozen):
    activation: list[float]
    explained_variance: float
    reconstruction_error: float
    n_active: int
    top: list[MotifActivation]


class CSMResult(Frozen):
    activation: list[float]
    explained_variance: float
    residual_fraction: float
    sparsity: float
    entropy: float
    n_active: int
    top: list[MotifActivation]
    reconstruction: Optional[list[float]] = None


class CSMContribution(Frozen):
    csm_id: str
    contribution: float
    share_of_similarity: float
    diagnostic_bands: list[float] = Field(default_factory=list)
    contributing_lsms: list[str] = Field(default_factory=list)


class MolecularHit(Frozen):
    """A retrieved reference analogue.

    Deliberately NOT called an identification. Phase 09 measured top-1 at 0.6053, and 68 of 375
    corpus queries are unretrievable because their molecule has a single spectrum.
    """
    rank: int
    molecule: str
    chemistry_class: str
    similarity: float
    supporting_csms: list[CSMContribution]
    contribution_sum: float
    reconciles: bool = Field(
        description="|sum(contributions) - similarity| < 1e-9; false is a defect, not a warning")


class RetrievalResult(Frozen):
    top: list[MolecularHit]
    margin: float
    n_candidates: int
    interpretation_note: str = (
        "Candidates are retrieved reference analogues, not definitive molecular "
        "identifications.")


class ChemistryAxis(Frozen):
    axis: str
    evidence: float
    share: float
    calibrated_probability: float
    rank: int


class ChemistryEvidenceResult(Frozen):
    axis_names: list[str]
    evidence: list[float]
    evidence_l1: list[float] = Field(
        description="the radar radii — RELATIVE BIOCHEMICAL EVIDENCE, not concentration")
    calibrated_probability: list[float]
    top: list[ChemistryAxis]
    predicted_class: str
    margin: float
    entropy: float
    units_note: str = (
        "Relative biochemical evidence. NOT a concentration, NOT an abundance, NOT a mixture "
        "fraction.")


class ConfidenceResult(Frozen):
    overall: float
    evidence_coverage: float
    top1_confidence: float
    top3_confidence: float
    retrieval_margin: float
    chemistry_confidence: float
    reconstruction_explained_variance: float
    unknown_warning: bool
    outlier_warning: bool
    notes: list[str] = Field(default_factory=list)


class AuditResult(Frozen):
    """Everything a reviewer needs to decide how much weight the answer can bear."""
    csm_explained_variance: float
    csm_residual_fraction: float
    n_active_csms: int
    spectral_coverage: float
    top_hit_margin: float
    chemistry_margin: float
    chemistry_entropy: float
    all_scores_reconcile: bool
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    open_set_limitation: str = (
        "The V7 engine does NOT provide validated open-set molecule detection. Phase 09 measured "
        "white noise reconstructing at CSM explained variance around 0.61, above the 0.50 "
        "warning floor. Low confidence and poor evidence quality are caution signals, not proof "
        "of novelty.")


class ProvenanceNode(Frozen):
    kind: Literal["lsm", "csm", "chemistry", "molecule"]
    identifier: str
    weight: float
    detail: dict[str, Any] = Field(default_factory=dict)


class ProvenanceResult(Frozen):
    root: str = "spectrum"
    lsm_layer: list[ProvenanceNode] = Field(default_factory=list)
    csm_layer: list[ProvenanceNode] = Field(default_factory=list)
    chemistry_layer: list[ProvenanceNode] = Field(default_factory=list)
    molecule_layer: list[ProvenanceNode] = Field(default_factory=list)
    atlas_fingerprint: str


class EngineInfo(Frozen):
    schema_version: str = SCHEMA_VERSION
    gaira_version: str
    engine_version: str
    atlas_fingerprint: str
    fingerprints: dict[str, str]
    frozen_assets_verified: bool
    n_lsms: int
    n_csms: int
    n_molecules: int
    n_chemistry_axes: int
    chemistry_axes: list[str]
    grid: dict[str, float]
    corpus: dict[str, Any]
    validated_performance: dict[str, float]
    supported_modalities: list[str]
    validated_sample_types: list[str]
    known_limitations: list[str]


class ReportMetadata(Frozen):
    schema_version: str = SCHEMA_VERSION
    generated_utc: str
    sample_id: Optional[str] = None
    sample_name: Optional[str] = None
    engine_version: str
    atlas_fingerprint: str
    generator: str = "gaira.v7.reporting"


class InferenceResult(Frozen):
    """The single object every Phase 10 surface returns. Clients render subsets of it; none may
    compute a value it does not contain (P-20)."""
    schema_version: str = SCHEMA_VERSION
    request_metadata: SpectrumMetadata
    preprocessing: PreprocessingResult
    lsm: Optional[LSMResult] = None
    csm: Optional[CSMResult] = None
    retrieval: RetrievalResult
    chemistry: ChemistryEvidenceResult
    confidence: ConfidenceResult
    audit: Optional[AuditResult] = None
    provenance: Optional[ProvenanceResult] = None
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    interpretation: str = Field(
        default="", description="deterministic template text; no language model is involved")
    engine: EngineInfo
    result_digest: str = Field(default="", description="MD5 over the canonical scientific fields")


class AxisDelta(Frozen):
    axis: str
    a: float
    b: float
    delta: float


class ComparisonResult(Frozen):
    schema_version: str = SCHEMA_VERSION
    label_a: str
    label_b: str
    a: InferenceResult
    b: InferenceResult
    csm_cosine: float
    chemistry_cosine: float
    chemistry_delta: list[AxisDelta]
    shared_top_molecules: list[str]
    rank_agreement: float = Field(
        description="Jaccard overlap of the two top-k retrieval sets")
    interpretation: str = ""
    scope_note: str = (
        "Differences are described in spectral motif evidence, chemistry evidence and reference "
        "neighbourhoods. V7 does not license a claim about biological state change.")


class ValidationResult(Frozen):
    ok: bool = Field(description="false when any diagnostic has severity 'error'")
    can_run: bool
    diagnostics: list[Diagnostic]
    n_points: int
    range_cm: Optional[tuple[float, float]] = None
    grid_coverage: Optional[float] = None


class HealthResult(Frozen):
    status: Literal["ok", "degraded", "unavailable"]
    engine_loaded: bool
    frozen_assets_verified: bool
    n_frozen_assets: int
    detail: dict[str, Any] = Field(default_factory=dict)
