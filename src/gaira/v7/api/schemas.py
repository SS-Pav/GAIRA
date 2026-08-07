"""GAIRA V7 — Phase 10: API-layer schemas.

Deliberately thin. The request and response bodies ARE the public contract from
`gaira.v7.contracts`; only transport-specific envelopes live here.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from gaira.v7.contracts import Diagnostic, ValidationResult


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: str
    code: str
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    validation: Optional[ValidationResult] = None


class ReportRequest(BaseModel):
    """Report generation from an existing result, or from a spectrum in one call."""
    model_config = ConfigDict(extra="forbid")
    format: str = Field(default="json", pattern="^(json|html|pdf)$")
    inference: Optional[dict] = Field(
        default=None, description="a previously returned InferenceResult, verbatim")
    request: Optional[dict] = Field(
        default=None, description="an InferenceRequest, if the spectrum should be re-run")
    title: Optional[str] = Field(default=None, max_length=200)


class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spectrum: dict
    metadata: Optional[dict] = None
