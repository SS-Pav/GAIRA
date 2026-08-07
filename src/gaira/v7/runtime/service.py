"""GAIRA V7 — Phase 10: the runtime service layer.

One object between the frozen scientific engine and every consumer: the Python SDK, the CLI, the
FastAPI service, the MCP tool server and the Streamlit client all call this and nothing else.

**This layer orchestrates. It does not compute.** Preprocessing, NNLS projection, LSM and CSM
activation, molecular retrieval, chemistry evidence, calibration, confidence and the audit
metrics all live in `gaira.v7.canonical.engine` and are reached only through its public methods.
What happens here is: validate the input, call the engine once, translate the engine's dicts into
the typed public contract, attach scope diagnostics, and render deterministic template text.

If a number appears in this file that the engine did not return, that is a defect (P-19).
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from gaira.v7 import __version__ as _pkg_version
from gaira.v7.canonical import GAIRAEngine
from gaira.v7.canonical.engine import GRID_HI, GRID_LO, N_BINS
from gaira.v7.contracts import (AuditResult, AxisDelta, ChemistryAxis, ChemistryEvidenceResult,
                                CompareRequest, ComparisonResult, ConfidenceResult,
                                CSMContribution, CSMResult, Diagnostic, EngineInfo, HealthResult,
                                InferenceRequest, InferenceResult, LSMResult, Modality,
                                MolecularHit, MotifActivation, PreprocessingResult,
                                ProvenanceNode, ProvenanceResult, RetrievalResult, SampleType,
                                Severity, SpectrumMetadata, SUPPORTED_MODALITIES,
                                VALIDATED_SAMPLE_TYPES, ValidationResult)
from gaira.v7.runtime import freeze as FREEZE
from gaira.v7.runtime.interpret import comparison_text, interpretation
from gaira.v7.validation import coverage as _coverage, validate as _validate

# Quoted from committed Phase 09 artifacts; displayed, never recomputed here.
VALIDATED_PERFORMANCE = {
    "molecule_top1": 0.6053, "molecule_top3": 0.7627, "molecule_top5": 0.7947,
    "molecule_top10": 0.8107, "molecule_mrr": 0.6870, "molecule_ndcg5": 0.7112,
    "chemistry_top1_heldout": 0.8507, "chemistry_top3_heldout": 0.9760,
    "chemistry_macro_f1_heldout": 0.8110,
    "csm_mean_explained_variance": 0.8232, "csm_replicate_consistency": 0.8927,
    "radar_reproducibility": 0.9596,
    "robustness_radar_cosine": 0.9648, "robustness_chemistry_top1": 0.8890,
    "robustness_molecule_top1": 0.8106,
}

KNOWN_LIMITATIONS = [
    "No validated open-set molecule detection. The engine cannot determine that the true "
    "molecule is absent from its 154-molecule bank; white noise reconstructs at CSM explained "
    "variance around 0.61, above the 0.50 warning floor (Phase 09 audit C5b).",
    "Molecule retrieval top-1 is 0.6053. 68 of 375 corpus queries are unretrievable by "
    "construction because 66 of 154 molecules have a single spectrum.",
    "Chemistry Evidence is RELATIVE. It is not a concentration, an abundance, or a mixture "
    "fraction.",
    "Pure Raman reference spectra only. SERS, serum, plasma, EV, bacteria and tissue behaviour "
    "is unmeasured in V7.",
    "The 16 chemistry classes are a curated cut through a continuum, not a discovered structure "
    "(Phase 06.5: no internal index has an interior optimum across K = 2 to 30).",
    "Class-prior bias (R-01) remains open; class sizes range from 3 to 80 spectra.",
    "In-sample chemistry figures describe the shipped fit, not expected performance on new "
    "molecules. Quote the held-out 0.8507.",
]

ENGINE_VERSION = "gaira-v7-canonical-phase09"


class GAIRAService:
    """Load once, serve read-only. Safe to share across threads.

    The engine holds no mutable state and draws no random numbers, so concurrency needs no lock
    on the science. The lock here guards only lazy construction.
    """

    _lock = threading.Lock()
    _singleton: Optional["GAIRAService"] = None

    def __init__(self, engine: GAIRAEngine, frozen_report: dict):
        self._engine = engine
        self._frozen = frozen_report
        self._loaded_utc = datetime.now(timezone.utc).isoformat()

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls, verify_assets: bool = True) -> "GAIRAService":
        """Verify the frozen assets, then load the engine. Both checks, in that order."""
        report = FREEZE.verify(strict=verify_assets)
        return cls(GAIRAEngine.load(), report)

    @classmethod
    def instance(cls) -> "GAIRAService":
        """The process-wide service. Used by the API, MCP and CLI so the atlas loads once."""
        if cls._singleton is None:
            with cls._lock:
                if cls._singleton is None:
                    cls._singleton = cls.load()
        return cls._singleton

    # ── operations ───────────────────────────────────────────────────────────
    def health(self) -> HealthResult:
        ok = all(v["present"] and v["match"] for v in self._frozen.values())
        return HealthResult(
            status="ok" if ok else "degraded", engine_loaded=True,
            frozen_assets_verified=ok, n_frozen_assets=len(self._frozen),
            detail={"loaded_utc": self._loaded_utc,
                    "atlas_fingerprint": self._engine.atlas_fingerprint})

    def engine_info(self) -> EngineInfo:
        e = self._engine
        return EngineInfo(
            gaira_version=_pkg_version, engine_version=ENGINE_VERSION,
            atlas_fingerprint=e.atlas_fingerprint, fingerprints=dict(e.fingerprints),
            frozen_assets_verified=all(v["present"] and v["match"]
                                       for v in self._frozen.values()),
            n_lsms=len(e._lsm_ids), n_csms=len(e._csm_ids),
            n_molecules=len(e.reference_molecules), n_chemistry_axes=len(e.chemistry_axes),
            chemistry_axes=list(e.chemistry_axes),
            grid={"low_cm": GRID_LO, "high_cm": GRID_HI, "step_cm": 2.0, "n_bins": float(N_BINS)},
            corpus={"n_spectra": 375, "n_canonical_molecules": 154, "n_chemistry_classes": 16,
                    "n_singleton_molecules": 66, "n_source_libraries": 3,
                    "scope": "pure Raman reference spectra"},
            validated_performance=VALIDATED_PERFORMANCE,
            supported_modalities=[m.value for m in SUPPORTED_MODALITIES],
            validated_sample_types=[s.value for s in VALIDATED_SAMPLE_TYPES],
            known_limitations=KNOWN_LIMITATIONS)

    def validate_input(self, wavenumber, intensity, *, modality: Modality = Modality.RAMAN,
                       sample_type: SampleType = SampleType.PURE,
                       extra: list[Diagnostic] | None = None) -> ValidationResult:
        return _validate(wavenumber, intensity, modality=modality, sample_type=sample_type,
                         extra=extra)

    def infer(self, request: InferenceRequest,
              extra_diagnostics: list[Diagnostic] | None = None) -> InferenceResult:
        """The whole path, once. Raises `ValueError` when validation says the engine cannot run."""
        x = np.asarray(request.spectrum.wavenumber, float)
        y = np.asarray(request.spectrum.intensity, float)
        md = request.metadata
        opts = request.options

        v = self.validate_input(x, y, modality=md.modality, sample_type=md.sample_type,
                                extra=extra_diagnostics)
        if not v.can_run:
            raise SpectrumRejected(v)

        r = self._engine.infer(y, x, top_k=opts.top_k_molecules,
                               already_preprocessed=opts.already_preprocessed)
        return self._translate(r, md, opts, v.diagnostics, x, y)

    def compare(self, request: CompareRequest) -> ComparisonResult:
        a = self.infer(request.a)
        b = self.infer(request.b)
        av = np.asarray(a.csm.activation if a.csm else [], float)
        bv = np.asarray(b.csm.activation if b.csm else [], float)
        if av.size == 0 or bv.size == 0:                 # csm suppressed by options
            ra = self._engine.infer(np.asarray(request.a.spectrum.intensity, float),
                                    np.asarray(request.a.spectrum.wavenumber, float),
                                    already_preprocessed=request.a.options.already_preprocessed)
            rb = self._engine.infer(np.asarray(request.b.spectrum.intensity, float),
                                    np.asarray(request.b.spectrum.wavenumber, float),
                                    already_preprocessed=request.b.options.already_preprocessed)
            av, bv = np.asarray(ra.csm["activation"]), np.asarray(rb.csm["activation"])
        ae = np.asarray(a.chemistry.evidence, float)
        be = np.asarray(b.chemistry.evidence, float)
        deltas = [AxisDelta(axis=ax, a=float(ae[i]), b=float(be[i]),
                            delta=float(be[i] - ae[i]))
                  for i, ax in enumerate(a.chemistry.axis_names)]
        sa = {h.molecule for h in a.retrieval.top}
        sb = {h.molecule for h in b.retrieval.top}
        shared = [h.molecule for h in a.retrieval.top if h.molecule in sb]
        jac = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
        ccos, chcos = _cos(av, bv), _cos(ae, be)
        return ComparisonResult(
            label_a=request.label_a, label_b=request.label_b, a=a, b=b,
            csm_cosine=ccos, chemistry_cosine=chcos, chemistry_delta=deltas,
            shared_top_molecules=shared, rank_agreement=float(jac),
            interpretation=comparison_text(request.label_a, request.label_b, ccos, chcos,
                                           deltas, shared, jac))

    def generate_report(self, result: InferenceResult, fmt: str = "pdf", **kw):
        """Delegated to `gaira.v7.reporting`, which is the single report implementation."""
        from gaira.v7.reporting import render
        return render(result, fmt=fmt, **kw)

    # ── translation: engine dicts → typed contract ───────────────────────────
    def _translate(self, r, md: SpectrumMetadata, opts, diags: list[Diagnostic],
                   x_in: np.ndarray, y_in: np.ndarray) -> InferenceResult:
        pre, csm, ret, chem, conf, prov = (r.preprocessing, r.csm, r.retrieval, r.chemistry,
                                           r.confidence, r.provenance)
        cov = _coverage(x_in)

        pre_out = PreprocessingResult(
            n_input_points=pre.n_input_points, input_range=tuple(pre.input_range),
            resampled_to=pre.resampled_to, baseline_method=pre.baseline_method,
            smoothing=pre.smoothing, normalisation=pre.normalisation, n_peaks=pre.n_peaks,
            signal_quality=pre.signal_quality, snr_estimate=pre.snr_estimate,
            grid_coverage=cov, warnings=list(pre.warnings),
            processed_intensity=None, grid=None)

        lsm_out = None
        if opts.include_lsm:
            lsm_out = LSMResult(
                activation=[float(v) for v in r.lsm["activation"]],
                explained_variance=r.lsm["explained_variance"],
                reconstruction_error=r.lsm["reconstruction_error"],
                n_active=r.lsm["n_active"],
                top=[MotifActivation(motif_id=t["lsm_id"], weight=t["weight"], share=t["share"])
                     for t in r.lsm["top"]])

        csm_out = None
        if opts.include_csm:
            csm_out = CSMResult(
                activation=[float(v) for v in csm["activation"]],
                explained_variance=csm["explained_variance"],
                residual_fraction=csm["residual_fraction"], sparsity=csm["sparsity"],
                entropy=csm["entropy"], n_active=csm["n_active"],
                top=[MotifActivation(motif_id=t["csm_id"], weight=t["weight"], share=t["share"],
                                     dominant_bands=list(t["dominant_bands"]),
                                     band_assignment=t["band_assignment"],
                                     contributing_lsms=list(t["lsms"]))
                     for t in csm["top"]],
                reconstruction=None)

        if opts.include_reconstruction:
            # Both vectors come from the engine. `prepare` is the exact routine `infer` used, so
            # what a viewer draws is what the projection consumed — not a second implementation.
            processed, _ = self._engine.prepare(y_in, x_in, opts.already_preprocessed)
            recon = np.asarray(csm["activation"], float) @ self._engine._CSM
            pre_out = pre_out.model_copy(update={
                "grid": [float(v) for v in self._engine.grid],
                "processed_intensity": [float(v) for v in processed]})
            if csm_out is not None:
                csm_out = csm_out.model_copy(
                    update={"reconstruction": [float(v) for v in recon]})

        hits = [MolecularHit(
            rank=t["rank"], molecule=t["molecule"], chemistry_class=t["chemistry_class"],
            similarity=t["similarity"], contribution_sum=t["contribution_sum"],
            reconciles=t["reconciles"],
            supporting_csms=[CSMContribution(
                csm_id=c["csm_id"], contribution=c["contribution"],
                share_of_similarity=c["share_of_similarity"],
                diagnostic_bands=list(c["diagnostic_bands"]),
                contributing_lsms=list(c["lsms"])) for c in t["supporting_csms"]])
            for t in ret["top"]]
        ret_out = RetrievalResult(top=hits, margin=ret["margin"],
                                  n_candidates=ret["n_candidates"])

        chem_out = ChemistryEvidenceResult(
            axis_names=list(chem["axis_names"]),
            evidence=[float(v) for v in chem["evidence"]],
            evidence_l1=[float(v) for v in chem["evidence_l1"]],
            calibrated_probability=[float(v) for v in chem["calibrated_probability"]],
            top=[ChemistryAxis(axis=t["axis"], evidence=t["evidence"], share=t["share"],
                               calibrated_probability=t["calibrated_probability"], rank=t["rank"])
                 for t in chem["top"]],
            predicted_class=chem["predicted_class"], margin=chem["margin"],
            entropy=chem["entropy"])

        conf_out = ConfidenceResult(
            overall=conf["overall"], evidence_coverage=conf["evidence_coverage"],
            top1_confidence=conf["top1_confidence"], top3_confidence=conf["top3_confidence"],
            retrieval_margin=conf["retrieval_margin"],
            chemistry_confidence=conf["chemistry_confidence"],
            reconstruction_explained_variance=conf["reconstruction_explained_variance"],
            unknown_warning=conf["unknown_warning"], outlier_warning=conf["outlier_warning"],
            notes=list(conf["notes"]))

        audit_out = None
        if opts.include_audit:
            audit_out = AuditResult(
                csm_explained_variance=csm["explained_variance"],
                csm_residual_fraction=csm["residual_fraction"],
                n_active_csms=csm["n_active"], spectral_coverage=cov,
                top_hit_margin=ret["margin"], chemistry_margin=chem["margin"],
                chemistry_entropy=chem["entropy"],
                all_scores_reconcile=all(h.reconciles for h in hits),
                diagnostics=list(diags))

        prov_out = None
        if opts.include_provenance:
            prov_out = ProvenanceResult(
                lsm_layer=[ProvenanceNode(kind="lsm", identifier=n["id"], weight=n["weight"])
                           for n in prov["lsm_layer"]],
                csm_layer=[ProvenanceNode(kind="csm", identifier=n["id"], weight=n["weight"],
                                          detail={"lsms": n["lsms"], "bands": n["bands"],
                                                  "assignment": n["assignment"]})
                           for n in prov["csm_layer"]],
                chemistry_layer=[ProvenanceNode(kind="chemistry", identifier=n["axis"],
                                                weight=n["evidence"])
                                 for n in prov["chemistry_layer"]],
                molecule_layer=[ProvenanceNode(kind="molecule", identifier=n["molecule"],
                                               weight=n["similarity"],
                                               detail={"class": n["class"],
                                                       "via_csms": n["via_csms"]})
                                for n in prov["molecule_layer"]],
                atlas_fingerprint=prov["atlas_fingerprint"])

        text = interpretation(csm["explained_variance"], chem_out.top, hits, conf_out,
                              list(pre.warnings))
        result = InferenceResult(
            request_metadata=md, preprocessing=pre_out, lsm=lsm_out, csm=csm_out,
            retrieval=ret_out, chemistry=chem_out, confidence=conf_out, audit=audit_out,
            provenance=prov_out, diagnostics=list(diags), interpretation=text,
            engine=self.engine_info())
        return result.model_copy(update={"result_digest": result_digest(result)})


class SpectrumRejected(ValueError):
    """Validation found an ERROR. Carries the full ValidationResult for the caller to render."""

    def __init__(self, validation: ValidationResult):
        self.validation = validation
        msgs = [d.message for d in validation.diagnostics if d.severity is Severity.ERROR]
        super().__init__("; ".join(msgs) or "spectrum rejected")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def result_digest(result: InferenceResult) -> str:
    """MD5 over the scientific fields only — excludes timestamps and free text.

    Two surfaces returning the same digest is the parity check the whole phase turns on.
    """
    core = {
        "csm": [round(v, 12) for v in (result.csm.activation if result.csm else [])],
        "csm_ev": round(result.csm.explained_variance, 12) if result.csm else None,
        "chem": [round(v, 12) for v in result.chemistry.evidence],
        "chem_cal": [round(v, 12) for v in result.chemistry.calibrated_probability],
        "pred": result.chemistry.predicted_class,
        "hits": [[h.rank, h.molecule, round(h.similarity, 12)] for h in result.retrieval.top],
        "conf": round(result.confidence.overall, 12),
        "atlas": result.engine.atlas_fingerprint,
    }
    return hashlib.md5(json.dumps(core, sort_keys=True).encode()).hexdigest()
