"""GAIRA V7 — Phase 10: the FastAPI application.

Local research software, but not carelessly so: the body-size limit is enforced, the frozen
assets are verified at startup, and no filesystem path is reachable through any route.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gaira.v7 import __version__
from gaira.v7.runtime.service import GAIRAService

from .dependencies import MAX_UPLOAD_BYTES
from .routes import router

log = logging.getLogger("gaira.v7.api")

DESCRIPTION = """
**GAIRA V7 — Grounded Raman Biochemical Inference.**

Project a Raman spectrum into a frozen biochemical motif atlas, retrieve grounded reference
evidence, and obtain an interpretable 16-axis Chemistry Evidence profile with calibrated
confidence and complete provenance.

The scientific architecture is frozen after Phase 09. This service is a transport around it and
computes nothing itself.

**Scope.** Pure Raman reference spectra. Chemistry Evidence is *relative* — never a
concentration, an abundance, or a mixture fraction. Retrieved molecules are reference analogues,
not identifications. The engine provides no validated open-set detection.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = GAIRAService.instance()          # verifies frozen assets, then loads the atlas
    info = svc.engine_info()
    log.info("GAIRA V7 engine loaded: atlas=%s csms=%d molecules=%d",
             info.atlas_fingerprint[:12], info.n_csms, info.n_molecules)
    app.state.service = svc
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="GAIRA V7", version=__version__, description=DESCRIPTION, lifespan=lifespan,
        contact={"name": "GAIRA"},
        openapi_tags=[{"name": "system", "description": "health and engine metadata"},
                      {"name": "inference", "description": "validation, inference, comparison"},
                      {"name": "reporting", "description": "deterministic report rendering"}])

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        n = request.headers.get("content-length")
        if n is not None and int(n) > MAX_UPLOAD_BYTES:
            return JSONResponse(status_code=413, content={
                "error": f"request body exceeds {MAX_UPLOAD_BYTES} bytes",
                "code": "request.too_large"})
        return await call_next(request)

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "GAIRA V7", "version": __version__, "docs": "/docs",
                "routes": ["/v1/health", "/v1/engine", "/v1/validate-spectrum", "/v1/infer",
                           "/v1/compare", "/v1/report"]}

    return app


app = create_app()
