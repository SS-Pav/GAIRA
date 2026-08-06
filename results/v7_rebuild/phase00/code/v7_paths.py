"""GAIRA V7 Phase 00 — path resolution, environment capture and hashing.

No lab-specific absolute path is committed. The raw data root resolves by the policy
inherited from tools/reproduce_gaira_foundation.py:

    --data-root  >  $GAIRA_DATA_ROOT  >  $GAIRA_DEFAULT_DATA_ROOT  >  unavailable

When the raw root is unavailable the pipeline runs in DEGRADED mode from committed
artefacts only; the mode is recorded in every manifest so no result can silently
depend on a mounted volume.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── repository anchors (relative only) ────────────────────────────────────────
CODE = Path(__file__).resolve().parent
PHASE00 = CODE.parent
V7_RESULTS = PHASE00.parent
REPO = V7_RESULTS.parent.parent
SRC = REPO / "src"

TABLES = PHASE00 / "tables"
FIGURES = PHASE00 / "figures"
REPORTS = PHASE00 / "reports"
LOGS = PHASE00 / "logs"
ARTIFACTS = PHASE00 / "artifacts"
VALIDATION = PHASE00 / "validation"
MANIFESTS = PHASE00 / "manifests"

# ── frozen inputs (READ ONLY — never written by V7) ───────────────────────────
FOUNDATION = REPO / "assets" / "foundation"
V5_FOUNDATION = REPO / "results/v5_rebuild/foundation"
V5_REPRODUCTION = REPO / "results/v5_rebuild/reproduction"
V6_REBUILD = REPO / "results/v6_rebuild"
V63 = V6_REBUILD / "v63_ontology_revalidation"
SV_REPS = V6_REBUILD / "semantic_validation/artifacts/sv_reps.npz"

CANONICAL_ATLAS_FINGERPRINT = "09ed804a40836f4a05a91ba10900cded"

# canonical preprocessing, frozen (assets/foundation/MANIFEST.json → preprocessing)
WINDOW_CM = (450.0, 1800.0)
GRID_STEP_CM = 2.0
N_BINS = 676
PREPROC = {"baseline": "asls", "smooth": "savgol", "norm": "l2"}


def ensure_dirs() -> None:
    for d in (TABLES, FIGURES, REPORTS, LOGS, ARTIFACTS, VALIDATION, MANIFESTS):
        d.mkdir(parents=True, exist_ok=True)


def data_root(explicit: str | None = None) -> Path | None:
    """Resolve the raw data root. Returns None when unavailable (degraded mode)."""
    for cand in (explicit, os.environ.get("GAIRA_DATA_ROOT"),
                 os.environ.get("GAIRA_DEFAULT_DATA_ROOT")):
        if cand:
            p = Path(cand).expanduser()
            if p.exists():
                return p
    return None


def add_src_to_path() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


# ── hashing (canonicalised, per architecture/ARTIFACT_AND_MANIFEST_SPEC.md §2) ──
def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_array(a) -> str:
    """32-hex digest of a float64 C-contiguous array — the V5 fingerprint convention."""
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:32]


def sha256_json(obj) -> str:
    """Canonical JSON digest: UTF-8, sorted keys, no insignificant whitespace."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_frame(df) -> str:
    """Canonical CSV digest: fixed column order, UTF-8, LF endings."""
    return hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


# ── environment / code provenance ─────────────────────────────────────────────
def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    sha = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    porcelain = run("status", "--porcelain")
    tracked_dirty = [l for l in porcelain.splitlines() if not l.startswith("??")]
    return {
        "git_sha": sha,
        "branch": branch,
        "dirty": bool(tracked_dirty),
        "tracked_modifications": len(tracked_dirty),
        "untracked_entries": len([l for l in porcelain.splitlines() if l.startswith("??")]),
    }


def environment() -> dict:
    import numpy
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
    }
    for mod in ("scipy", "sklearn", "pandas", "matplotlib"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = "unavailable"
    try:
        cfg = numpy.__config__.show(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        env["blas"] = f"{blas.get('name', '?')} {blas.get('version', '?')}"
    except Exception:
        env["blas"] = "unknown"
    return env
