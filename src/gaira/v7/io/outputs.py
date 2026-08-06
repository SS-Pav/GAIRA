"""GAIRA V7 — phase output resolution.

Every phase writes the same seven buckets under one root. The root is resolved rather than
hard-coded so a run can be redirected — to a scratch volume, a CI workspace, a side-by-side
comparison — without editing any phase code:

    GAIRA_V7_OUTPUT_ROOT  >  <repo>/results/v7_rebuild

Frozen *inputs* are never resolved this way. A phase reads its upstream artefacts from the
committed tree and only its own outputs move, so redirecting the output root can never cause a
phase to silently consume a different upstream dictionary.
"""
from __future__ import annotations

import os
from pathlib import Path

BUCKETS = ("code", "artifacts", "tables", "validation", "figures", "reports", "logs")
_ENV = "GAIRA_V7_OUTPUT_ROOT"


def repo_root() -> Path:
    """The repository root, located from this file rather than from the working directory."""
    return Path(__file__).resolve().parents[4]


def output_root() -> Path:
    """Where phase outputs go. Overridable by `GAIRA_V7_OUTPUT_ROOT`."""
    env = os.environ.get(_ENV)
    return Path(env).expanduser().resolve() if env else repo_root() / "results" / "v7_rebuild"


def frozen_root() -> Path:
    """Where frozen upstream artefacts are read from — always the committed tree."""
    return repo_root() / "results" / "v7_rebuild"


class PhaseOutputs:
    """The seven buckets for one phase, created on demand.

    `rel()` returns a repository-relative path for manifests where that is possible, and an
    absolute path when the output root has been redirected outside the repository — a manifest
    that silently records a path which does not resolve is worse than a long one.
    """

    def __init__(self, phase: str, extra: tuple[str, ...] = ()):
        self.phase = phase
        self.root = output_root() / f"phase{phase.replace('.', '_')}"
        self.buckets = BUCKETS + tuple(extra)
        for b in self.buckets:
            setattr(self, b, self.root / b)

    def ensure(self) -> "PhaseOutputs":
        for b in self.buckets:
            getattr(self, b).mkdir(parents=True, exist_ok=True)
        return self

    def rel(self, p: Path) -> str:
        p = Path(p).resolve()
        try:
            return str(p.relative_to(repo_root()))
        except ValueError:
            return str(p)

    def __repr__(self) -> str:
        return f"PhaseOutputs(phase={self.phase!r}, root={self.root})"
