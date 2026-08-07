"""GAIRA V7 — Phase 10: the Python SDK.

The easiest developer-facing surface, and the thinnest. It builds typed requests, calls
`GAIRAService`, and returns typed results. It contains no scientific logic and no second path::

    from gaira.v7 import GAIRA

    gaira  = GAIRA.load()
    result = gaira.infer(wavenumber=x, intensity=y, metadata={"sample_id": "S1"})

    print(result.chemistry.predicted_class)
    print(result.retrieval.top[0].molecule, result.retrieval.top[0].similarity)
    print(result.interpretation)

    pdf = gaira.report(result, fmt="pdf")

A remote mode is available for talking to a deployed API instead of a local engine::

    gaira = GAIRA.remote("http://localhost:8000")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from gaira.v7.adapters import load as load_spectrum
from gaira.v7.contracts import (CompareRequest, ComparisonResult, Diagnostic, EngineInfo,
                                HealthResult, InferenceOptions, InferenceRequest,
                                InferenceResult, Modality, SampleType, SpectrumInput,
                                SpectrumMetadata, ValidationResult)
from gaira.v7.runtime.service import GAIRAService, SpectrumRejected

PathLike = Union[str, Path]


class GAIRA:
    """The public Python client. `GAIRA.load()` for in-process, `GAIRA.remote(url)` for HTTP."""

    def __init__(self, service: GAIRAService):
        self._svc = service

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls, verify_assets: bool = True) -> "GAIRA":
        return cls(GAIRAService.load(verify_assets=verify_assets))

    @classmethod
    def shared(cls) -> "GAIRA":
        """The process-wide service, loaded once. Preferred inside long-running programs."""
        return cls(GAIRAService.instance())

    @classmethod
    def remote(cls, base_url: str, timeout: float = 120.0) -> "RemoteGAIRA":
        return RemoteGAIRA(base_url, timeout)

    # ── input ────────────────────────────────────────────────────────────────
    @staticmethod
    def read(path: PathLike) -> tuple[list[float], list[float], list[Diagnostic]]:
        """Parse a spectrum file into `(wavenumber, intensity, diagnostics)`."""
        p = Path(path)
        parsed = load_spectrum(p.read_bytes(), p.name)
        return ([float(v) for v in parsed.wavenumber], [float(v) for v in parsed.intensity],
                list(parsed.diagnostics))

    # ── operations ───────────────────────────────────────────────────────────
    def engine_info(self) -> EngineInfo:
        return self._svc.engine_info()

    def health(self) -> HealthResult:
        return self._svc.health()

    def validate(self, wavenumber, intensity, metadata: Optional[dict] = None
                 ) -> ValidationResult:
        md = SpectrumMetadata(**(metadata or {}))
        return self._svc.validate_input(wavenumber, intensity, modality=md.modality,
                                        sample_type=md.sample_type)

    def infer(self, wavenumber, intensity, metadata: Optional[dict] = None,
              options: Optional[dict] = None,
              diagnostics: Optional[list[Diagnostic]] = None) -> InferenceResult:
        return self._svc.infer(_request(wavenumber, intensity, metadata, options),
                               extra_diagnostics=diagnostics)

    def infer_file(self, path: PathLike, metadata: Optional[dict] = None,
                   options: Optional[dict] = None) -> InferenceResult:
        x, y, diags = self.read(path)
        md = dict(metadata or {})
        md.setdefault("source_name", Path(path).name)
        return self.infer(x, y, md, options, diagnostics=diags)

    def compare(self, a: tuple, b: tuple, label_a: str = "A", label_b: str = "B",
                metadata_a: Optional[dict] = None, metadata_b: Optional[dict] = None,
                options: Optional[dict] = None) -> ComparisonResult:
        return self._svc.compare(CompareRequest(
            a=_request(a[0], a[1], metadata_a, options),
            b=_request(b[0], b[1], metadata_b, options),
            label_a=label_a, label_b=label_b))

    def report(self, result: InferenceResult, fmt: str = "pdf", title: Optional[str] = None):
        return self._svc.generate_report(result, fmt=fmt, title=title)

    def __repr__(self) -> str:
        i = self.engine_info()
        return (f"GAIRA(local, atlas={i.atlas_fingerprint[:12]}…, {i.n_csms} CSMs, "
                f"{i.n_molecules} molecules)")


class RemoteGAIRA:
    """The same surface over HTTP. Useful when the engine runs elsewhere; identical results."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _client(self):
        import httpx
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def health(self) -> HealthResult:
        with self._client() as c:
            return HealthResult.model_validate(c.get("/v1/health").json())

    def engine_info(self) -> EngineInfo:
        with self._client() as c:
            return EngineInfo.model_validate(c.get("/v1/engine").json())

    def validate(self, wavenumber, intensity, metadata: Optional[dict] = None
                 ) -> ValidationResult:
        with self._client() as c:
            r = c.post("/v1/validate-spectrum", json={
                "spectrum": {"wavenumber": list(map(float, wavenumber)),
                             "intensity": list(map(float, intensity))},
                "metadata": metadata or {}})
            r.raise_for_status()
            return ValidationResult.model_validate(r.json())

    def infer(self, wavenumber, intensity, metadata: Optional[dict] = None,
              options: Optional[dict] = None) -> InferenceResult:
        req = _request(wavenumber, intensity, metadata, options)
        with self._client() as c:
            r = c.post("/v1/infer", json=req.model_dump(mode="json"))
            if r.status_code == 422:
                raise RemoteRejected(r.json())
            r.raise_for_status()
            return InferenceResult.model_validate(r.json())

    def compare(self, a: tuple, b: tuple, label_a: str = "A", label_b: str = "B",
                metadata_a: Optional[dict] = None, metadata_b: Optional[dict] = None,
                options: Optional[dict] = None) -> ComparisonResult:
        body = CompareRequest(a=_request(a[0], a[1], metadata_a, options),
                              b=_request(b[0], b[1], metadata_b, options),
                              label_a=label_a, label_b=label_b)
        with self._client() as c:
            r = c.post("/v1/compare", json=body.model_dump(mode="json"))
            r.raise_for_status()
            return ComparisonResult.model_validate(r.json())

    def report(self, result: InferenceResult, fmt: str = "pdf", title: Optional[str] = None):
        import base64
        with self._client() as c:
            r = c.post("/v1/report", json={"format": fmt,
                                           "inference": result.model_dump(mode="json"),
                                           "title": title})
            r.raise_for_status()
            if fmt == "pdf":
                return base64.b64decode(r.json()["content"])
            return r.text

    def __repr__(self) -> str:
        return f"GAIRA(remote, {self.base_url})"


class RemoteRejected(ValueError):
    def __init__(self, payload: dict):
        self.payload = payload
        detail = payload.get("detail", payload)
        super().__init__(detail.get("error", "spectrum rejected") if isinstance(detail, dict)
                         else str(detail))


def _request(wavenumber, intensity, metadata: Optional[dict], options: Optional[dict]
             ) -> InferenceRequest:
    return InferenceRequest(
        spectrum=SpectrumInput(wavenumber=[float(v) for v in wavenumber],
                               intensity=[float(v) for v in intensity]),
        metadata=SpectrumMetadata(**(metadata or {})),
        options=InferenceOptions(**(options or {})))


__all__ = ["GAIRA", "RemoteGAIRA", "RemoteRejected", "SpectrumRejected", "Modality", "SampleType"]
