"""GAIRA V7 — Phase 10: FastAPI routes.

Every route delegates to `GAIRAService`. There is no scientific expression anywhere in this
file, and a static test enforces that.
"""
from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, HTTPException, Response

from gaira.v7.contracts import (CompareRequest, ComparisonResult, EngineInfo, HealthResult,
                                InferenceRequest, InferenceResult, Modality, SampleType,
                                SpectrumInput, ValidationResult)
from gaira.v7.runtime.service import GAIRAService, SpectrumRejected

from .dependencies import get_service
from .schemas import ReportRequest, ValidateRequest

router = APIRouter(prefix="/v1")


@router.get("/health", response_model=HealthResult, tags=["system"])
def health(svc: GAIRAService = Depends(get_service)) -> HealthResult:
    return svc.health()


@router.get("/engine", response_model=EngineInfo, tags=["system"])
def engine(svc: GAIRAService = Depends(get_service)) -> EngineInfo:
    return svc.engine_info()


@router.post("/validate-spectrum", response_model=ValidationResult, tags=["inference"])
def validate_spectrum(body: ValidateRequest,
                      svc: GAIRAService = Depends(get_service)) -> ValidationResult:
    spec = SpectrumInput(**body.spectrum)
    md = body.metadata or {}
    return svc.validate_input(
        spec.wavenumber, spec.intensity,
        modality=Modality(md.get("modality", "raman")),
        sample_type=SampleType(md.get("sample_type", "pure")))


@router.post("/infer", response_model=InferenceResult, tags=["inference"])
def infer(request: InferenceRequest,
          svc: GAIRAService = Depends(get_service)) -> InferenceResult:
    try:
        return svc.infer(request)
    except SpectrumRejected as rejected:
        raise HTTPException(status_code=422, detail={
            "error": str(rejected), "code": "spectrum_rejected",
            "validation": rejected.validation.model_dump(mode="json")}) from None


@router.post("/compare", response_model=ComparisonResult, tags=["inference"])
def compare(request: CompareRequest,
            svc: GAIRAService = Depends(get_service)) -> ComparisonResult:
    try:
        return svc.compare(request)
    except SpectrumRejected as rejected:
        raise HTTPException(status_code=422, detail={
            "error": str(rejected), "code": "spectrum_rejected",
            "validation": rejected.validation.model_dump(mode="json")}) from None


@router.post("/report", tags=["reporting"])
def report(body: ReportRequest, svc: GAIRAService = Depends(get_service)):
    """Render a report. Never writes to a caller-supplied path — the artefact is returned inline.

    JSON and HTML come back as text; a PDF comes back base64-encoded in a JSON envelope so the
    endpoint has one predictable response shape.
    """
    if body.inference is None and body.request is None:
        raise HTTPException(status_code=400, detail={
            "error": "supply either `inference` (a prior result) or `request` (a spectrum)",
            "code": "report.no_input"})
    if body.inference is not None:
        result = InferenceResult.model_validate(body.inference)
    else:
        try:
            result = svc.infer(InferenceRequest.model_validate(body.request))
        except SpectrumRejected as rejected:
            raise HTTPException(status_code=422, detail={
                "error": str(rejected), "code": "spectrum_rejected",
                "validation": rejected.validation.model_dump(mode="json")}) from None

    payload = svc.generate_report(result, fmt=body.format, title=body.title)
    if body.format == "json":
        return Response(content=payload, media_type="application/json")
    if body.format == "html":
        return Response(content=payload, media_type="text/html; charset=utf-8")
    return {"format": "pdf", "encoding": "base64",
            "filename": f"gaira_v7_report_{result.result_digest[:12]}.pdf",
            "content": base64.b64encode(payload).decode("ascii"),
            "result_digest": result.result_digest}
