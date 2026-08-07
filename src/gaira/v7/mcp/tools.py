"""GAIRA V7 — Phase 10: MCP tool definitions.

Eight read-only tools. Every one routes through `GAIRAService`; none contains a scientific
expression, and a static test enforces that.

The tools are deliberately **coarse**. An agent should be able to ask "interpret this spectrum"
and receive a scientifically coherent answer; it should not be able to assemble its own inference
path out of primitives like "run NNLS against this matrix". Fine-grained numerical operations are
where a caller could construct a result the engine never sanctioned, so they are not exposed.

No language model is involved in this server. It is a tool provider, nothing more.
"""
from __future__ import annotations

import json
from typing import Any

from gaira.v7.adapters import load as load_spectrum
from gaira.v7.contracts import (CompareRequest, InferenceOptions, InferenceRequest, Modality,
                                SampleType, SpectrumInput, SpectrumMetadata)
from gaira.v7.runtime.service import GAIRAService, SpectrumRejected

MAX_INLINE_POINTS = 100_000

_SPECTRUM_SCHEMA = {
    "type": "object",
    "properties": {
        "wavenumber": {"type": "array", "items": {"type": "number"},
                       "description": "wavenumbers in cm-1"},
        "intensity": {"type": "array", "items": {"type": "number"}},
        "text": {"type": "string",
                 "description": "alternatively, the raw contents of a CSV/TSV/two-column text "
                                "spectrum; parsed by the same adapters the CLI uses"},
    },
}
_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "modality": {"type": "string", "enum": [m.value for m in Modality],
                     "default": "raman",
                     "description": "only 'raman' is supported; anything else is REJECTED "
                                    "rather than run silently"},
        "sample_type": {"type": "string", "enum": [s.value for s in SampleType],
                        "default": "pure",
                        "description": "recorded as metadata; never applied to the calculation"},
        "excitation_nm": {"type": "number"},
        "sample_id": {"type": "string"},
        "source_name": {"type": "string"},
        "notes": {"type": "string"},
    },
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "gaira_engine_info",
        "description": (
            "Metadata about the frozen GAIRA V7 engine: version, artefact fingerprints, atlas "
            "shape, corpus, validated performance figures, supported modalities and the known "
            "limitations. Call this first to learn what the engine can and cannot support."),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "gaira_validate_spectrum",
        "description": (
            "Check whether a spectrum can be run, without running it. Returns structured "
            "diagnostics at three severities: error (cannot run), warning (runs, interpretation "
            "limited) and info. Use this before gaira_infer_spectrum on unfamiliar input."),
        "inputSchema": {"type": "object",
                        "properties": {"spectrum": _SPECTRUM_SCHEMA,
                                       "metadata": _METADATA_SCHEMA},
                        "required": ["spectrum"]},
    },
    {
        "name": "gaira_infer_spectrum",
        "description": (
            "Run the complete frozen GAIRA V7 inference path on one Raman spectrum: canonical "
            "preprocessing, LSM and CSM projection, grounded molecular retrieval, 16-axis "
            "Chemistry Evidence, calibrated confidence, audit and provenance. Returns the full "
            "InferenceResult. Chemistry Evidence is RELATIVE, never a concentration; retrieved "
            "molecules are reference analogues, never identifications."),
        "inputSchema": {
            "type": "object",
            "properties": {"spectrum": _SPECTRUM_SCHEMA, "metadata": _METADATA_SCHEMA,
                           "top_k": {"type": "integer", "default": 10, "minimum": 1,
                                     "maximum": 154},
                           "include_provenance": {"type": "boolean", "default": True}},
            "required": ["spectrum"]},
    },
    {
        "name": "gaira_compare_spectra",
        "description": (
            "Run two spectra independently through the full engine and compare their CSM "
            "activations, Chemistry Evidence and retrieval neighbourhoods. Differences are "
            "described in spectral and chemistry terms only; V7 does not license a claim about "
            "biological state change."),
        "inputSchema": {
            "type": "object",
            "properties": {"spectrum_a": _SPECTRUM_SCHEMA, "spectrum_b": _SPECTRUM_SCHEMA,
                           "metadata_a": _METADATA_SCHEMA, "metadata_b": _METADATA_SCHEMA,
                           "label_a": {"type": "string", "default": "A"},
                           "label_b": {"type": "string", "default": "B"}},
            "required": ["spectrum_a", "spectrum_b"]},
    },
    {
        "name": "gaira_get_molecular_evidence",
        "description": (
            "The grounded retrieval view of a spectrum: ranked reference analogues with their "
            "CSM cosine similarity and the per-motif decomposition of each score. Every score "
            "reconciles exactly to its listed contributions. A narrower view of "
            "gaira_infer_spectrum for agents that only need candidates."),
        "inputSchema": {
            "type": "object",
            "properties": {"spectrum": _SPECTRUM_SCHEMA, "metadata": _METADATA_SCHEMA,
                           "top_k": {"type": "integer", "default": 10}},
            "required": ["spectrum"]},
    },
    {
        "name": "gaira_get_chemistry_evidence",
        "description": (
            "The 16-axis Chemistry Evidence profile of a spectrum with calibrated confidences "
            "and the ordered ranking. RELATIVE BIOCHEMICAL EVIDENCE — not a concentration, not "
            "an abundance, not a mixture fraction."),
        "inputSchema": {
            "type": "object",
            "properties": {"spectrum": _SPECTRUM_SCHEMA, "metadata": _METADATA_SCHEMA},
            "required": ["spectrum"]},
    },
    {
        "name": "gaira_explain_result",
        "description": (
            "The audit and provenance view of a spectrum: reconstruction quality, spectral "
            "coverage, margins, warnings, the full spectrum-to-wavenumber provenance chain, and "
            "the deterministic interpretation paragraph. Use this to justify or challenge an "
            "answer rather than to obtain one."),
        "inputSchema": {
            "type": "object",
            "properties": {"spectrum": _SPECTRUM_SCHEMA, "metadata": _METADATA_SCHEMA},
            "required": ["spectrum"]},
    },
    {
        "name": "gaira_generate_report",
        "description": (
            "Render a deterministic report for a spectrum as JSON or HTML. Template-driven; no "
            "language model is involved. PDF is available through the HTTP API and the CLI, "
            "which can return binary."),
        "inputSchema": {
            "type": "object",
            "properties": {"spectrum": _SPECTRUM_SCHEMA, "metadata": _METADATA_SCHEMA,
                           "format": {"type": "string", "enum": ["json", "html"],
                                      "default": "json"},
                           "title": {"type": "string"}},
            "required": ["spectrum"]},
    },
]

TOOL_NAMES = tuple(t["name"] for t in TOOLS)


# ── argument handling ────────────────────────────────────────────────────────
def _spectrum(payload: dict) -> tuple[list[float], list[float], list]:
    """Accept explicit arrays or raw delimited text, through the same adapters as everything."""
    if not isinstance(payload, dict):
        raise ValueError("spectrum must be an object with wavenumber/intensity or text")
    if payload.get("text"):
        parsed = load_spectrum(str(payload["text"]), "inline.csv")
        if not parsed.ok:
            raise ValueError("; ".join(d.message for d in parsed.diagnostics
                                       if d.severity.value == "error"))
        return ([float(v) for v in parsed.wavenumber], [float(v) for v in parsed.intensity],
                list(parsed.diagnostics))
    w, i = payload.get("wavenumber"), payload.get("intensity")
    if not isinstance(w, list) or not isinstance(i, list):
        raise ValueError("spectrum needs 'wavenumber' and 'intensity' arrays, or 'text'")
    if len(w) > MAX_INLINE_POINTS:
        raise ValueError(f"{len(w)} points exceeds the inline limit of {MAX_INLINE_POINTS}")
    return [float(v) for v in w], [float(v) for v in i], []


def _request(payload: dict, top_k: int = 10, include_provenance: bool = True,
             include_reconstruction: bool = False) -> tuple[InferenceRequest, list]:
    x, y, diags = _spectrum(payload.get("spectrum") or payload.get("spectrum_a") or {})
    md = payload.get("metadata") or {}
    return InferenceRequest(
        spectrum=SpectrumInput(wavenumber=x, intensity=y),
        metadata=SpectrumMetadata(**md),
        options=InferenceOptions(top_k_molecules=int(top_k),
                                 include_provenance=bool(include_provenance),
                                 include_reconstruction=include_reconstruction)), diags


def call(name: str, arguments: dict, svc: GAIRAService | None = None) -> dict:
    """Dispatch one tool call. Returns a JSON-serialisable dict; raises ValueError on bad input."""
    svc = svc or GAIRAService.instance()
    args = arguments or {}

    if name == "gaira_engine_info":
        return svc.engine_info().model_dump(mode="json")

    if name == "gaira_validate_spectrum":
        x, y, diags = _spectrum(args.get("spectrum") or {})
        md = SpectrumMetadata(**(args.get("metadata") or {}))
        v = svc.validate_input(x, y, modality=md.modality, sample_type=md.sample_type,
                               extra=diags)
        return v.model_dump(mode="json")

    if name == "gaira_compare_spectra":
        xa, ya, _ = _spectrum(args.get("spectrum_a") or {})
        xb, yb, _ = _spectrum(args.get("spectrum_b") or {})
        req = CompareRequest(
            a=InferenceRequest(spectrum=SpectrumInput(wavenumber=xa, intensity=ya),
                               metadata=SpectrumMetadata(**(args.get("metadata_a") or {}))),
            b=InferenceRequest(spectrum=SpectrumInput(wavenumber=xb, intensity=yb),
                               metadata=SpectrumMetadata(**(args.get("metadata_b") or {}))),
            label_a=str(args.get("label_a", "A"))[:64],
            label_b=str(args.get("label_b", "B"))[:64])
        return svc.compare(req).model_dump(mode="json")

    req, diags = _request(args, top_k=args.get("top_k", 10),
                          include_provenance=args.get("include_provenance", True))
    result = svc.infer(req, extra_diagnostics=diags)

    if name == "gaira_infer_spectrum":
        return result.model_dump(mode="json")

    if name == "gaira_get_molecular_evidence":
        return {"result_digest": result.result_digest,
                "atlas_fingerprint": result.engine.atlas_fingerprint,
                "retrieval": result.retrieval.model_dump(mode="json"),
                "confidence": result.confidence.model_dump(mode="json"),
                "note": "Retrieved reference analogues, not identifications. Validated molecule "
                        "top-1 is 0.6053."}

    if name == "gaira_get_chemistry_evidence":
        return {"result_digest": result.result_digest,
                "atlas_fingerprint": result.engine.atlas_fingerprint,
                "chemistry": result.chemistry.model_dump(mode="json"),
                "chemistry_confidence": result.confidence.chemistry_confidence,
                "note": "RELATIVE BIOCHEMICAL EVIDENCE — not a concentration, not an abundance, "
                        "not a mixture fraction."}

    if name == "gaira_explain_result":
        return {"result_digest": result.result_digest,
                "interpretation": result.interpretation,
                "audit": result.audit.model_dump(mode="json") if result.audit else None,
                "provenance": (result.provenance.model_dump(mode="json")
                               if result.provenance else None),
                "confidence": result.confidence.model_dump(mode="json"),
                "preprocessing": result.preprocessing.model_dump(mode="json"),
                "diagnostics": [d.model_dump(mode="json") for d in result.diagnostics]}

    if name == "gaira_generate_report":
        fmt = str(args.get("format", "json")).lower()
        if fmt not in ("json", "html"):
            raise ValueError("MCP report format must be 'json' or 'html'; use the HTTP API or "
                             "CLI for PDF")
        payload = svc.generate_report(result, fmt=fmt, title=args.get("title"))
        return {"format": fmt, "result_digest": result.result_digest,
                "content": payload if isinstance(payload, str) else payload.decode()}

    raise ValueError(f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}")
