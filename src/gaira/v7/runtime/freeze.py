"""GAIRA V7 — Phase 10: the frozen-asset ledger.

Phase 09's engine verifies four *declared* fingerprints on load: values recorded inside
`PHASE_STATE.json` and `csm_registry_v1.json` by the phases that produced them. That check
answers "did the producing phase say this was the artefact?" — it does not answer "is the
file on disk still the file that phase wrote". A dictionary could be edited in place and the
declared fingerprint would not move.

Phase 10 closes that gap by pinning the **content digest of every file the engine reads**,
recomputed from the committed tree. Ten files, ten digests, verified before the runtime will
serve anything. Nothing here modifies an upstream artefact; the ledger is a read-only check.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from gaira.v7.io import frozen_root

# Every file GAIRAEngine.load() opens, recorded by instrumenting the loader. Repository-relative
# under `results/v7_rebuild/`. If the engine ever reads an eleventh file, `verify()` fails loudly
# rather than checking nine of ten.
FROZEN_ASSETS: tuple[str, ...] = (
    "phase00/tables/canonical_analytes_v1.csv",
    "phase00/tables/chemical_partition_v1.csv",
    "phase01/PHASE_STATE.json",
    "phase01/artifacts/balanced_references_v1.npz",
    "phase01/artifacts/lsm_dictionary_v1.npz",
    "phase02/artifacts/csm_dictionary_v1.npz",
    "phase02/artifacts/csm_registry_v1.json",
    "phase05/PHASE_STATE.json",
    "phase06/artifacts/chemistry_evidence_calibrator_v1.json",
    "phase06/artifacts/chemistry_evidence_model_v1.json",
)

# Content digests recomputed from the committed tree during the Phase 10 freeze audit.
EXPECTED_DIGESTS: dict[str, str] = {
    "phase00/tables/canonical_analytes_v1.csv": "dabd2834db31804fa948f5d30ff0fd44",
    "phase00/tables/chemical_partition_v1.csv": "0285392b5a70f55f4938344462486d45",
    "phase01/PHASE_STATE.json": "c66f7304b08aa6dce8415ca09c8a600b",
    "phase01/artifacts/balanced_references_v1.npz": "06fb6b7f2f58746023c77473c54f04d0",
    "phase01/artifacts/lsm_dictionary_v1.npz": "9d4bafe596e390d1ed0cd4eeecb50b6b",
    "phase02/artifacts/csm_dictionary_v1.npz": "3692ad772d661273c183fb23cf587c72",
    "phase02/artifacts/csm_registry_v1.json": "f75bce02c75747507034cd235ef2e9eb",
    "phase05/PHASE_STATE.json": "395e9abb425eab6118bdc8c89031827b",
    "phase06/artifacts/chemistry_evidence_calibrator_v1.json":
        "c9c6e8068d6116cbd22306addea24ac2",
    "phase06/artifacts/chemistry_evidence_model_v1.json": "0b387f2b26a16710e2436cb9e4d7865b",
}


class FrozenAssetError(RuntimeError):
    """A frozen asset is missing or its content has changed. The runtime must not serve."""


def digest(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def pin() -> dict[str, str]:
    """Recompute every digest from the committed tree. Used to author EXPECTED_DIGESTS.

    The two `PHASE_STATE.json` files are pinned alongside the dictionaries deliberately. Re-running
    an upstream phase rewrites them, and that *should* invalidate the Phase 10 freeze — a runtime
    validated against one atlas build must not silently continue against another.
    """
    root = frozen_root()
    return {rel: digest(root / rel) for rel in FROZEN_ASSETS}


def verify(strict: bool = True) -> dict[str, dict]:
    """Check presence and content of every frozen asset.

    `strict=False` still reports every mismatch; it only declines to raise. The API and MCP
    servers verify strictly at startup — a runtime that serves inference from an unverified
    atlas is worse than one that refuses to start.
    """
    root = frozen_root()
    report: dict[str, dict] = {}
    problems: list[str] = []
    for rel in FROZEN_ASSETS:
        p = root / rel
        if not p.exists():
            report[rel] = {"present": False, "digest": None, "expected": EXPECTED_DIGESTS.get(rel),
                           "match": False}
            problems.append(f"{rel}: missing")
            continue
        got = digest(p)
        want = EXPECTED_DIGESTS.get(rel) or ""
        ok = (got == want) if want else True
        report[rel] = {"present": True, "digest": got, "expected": want or None,
                       "match": ok, "bytes": p.stat().st_size,
                       "unpinned": not want}
        if not ok:
            problems.append(f"{rel}: {got} != {want}")
    if problems and strict:
        raise FrozenAssetError(
            "frozen asset verification failed; the GAIRA runtime will not serve inference from a "
            f"changed atlas: {problems}")
    return report


def summary() -> dict:
    rep = verify(strict=False)
    return {"n_assets": len(rep),
            "n_pinned": sum(1 for v in rep.values() if not v.get("unpinned", False)),
            "all_present": all(v["present"] for v in rep.values()),
            "all_match": all(v["match"] for v in rep.values()),
            "assets": rep}
