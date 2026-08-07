#!/usr/bin/env python3
"""GAIRA V7 — Phase 10.1: documentation consistency verifier.

Reads the canonical values from the committed Phase 10 artifacts and checks that every Phase 10
document agrees with them and with itself. Reads only; writes one report.

It answers three questions:
  1. does any document contradict a committed artifact?
  2. does the corrected DART model appear anywhere as the old mass-spectrometric one?
  3. is the atlas-identity terminology used consistently?
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "src"))
from gaira.v7.io import PhaseOutputs                                    # noqa: E402

OUT = PhaseOutputs("10")
DOCS = sorted((OUT.root / "reports").glob("PHASE_10_*.md"))

SAF = "09ed804a40836f4a05a91ba10900cded"     # Scientific Atlas Fingerprint
FRCH = "2e43ddcca7d3be41c5f9da016fb8277f"    # Frozen Runtime Content Hash

# Phrases that encode the SUPERSEDED DART model. A document may quote them only when it is
# explicitly recording the divergence, which these markers identify.
FORBIDDEN_DART = (
    "mass-spectrometric channel",
    "DART modality",
    "mass spectrometric correspondence",
    "no vibrational correspondence",
    "orthogonal measurement",
)
DIVERGENCE_MARKERS = ("Known divergence", "known divergence", "supersedes", "superseded")


def log(m: str) -> None:
    print(f"[phase10.1/docs] {m}", flush=True)


def main() -> int:
    frz = json.loads((OUT.artifacts / "engine_freeze_audit_v1.json").read_text())
    par = json.loads((OUT.artifacts / "parity_and_performance_v1.json").read_text())
    e, perf = frz["engine"], par["performance"]

    # ── canonical values, read from artifacts ────────────────────────────────
    canonical = {
        "Scientific Atlas Fingerprint": frz["declared_fingerprints"]["recomputed"]["atlas"],
        "Frozen Runtime Content Hash": e["atlas_fingerprint"],
        "LSM registry": frz["declared_fingerprints"]["recomputed"]["lsm"],
        "CSM registry": frz["declared_fingerprints"]["recomputed"]["csm"],
        "Phase 05 engine": frz["declared_fingerprints"]["recomputed"]["engine"],
        "n_lsms": e["n_lsms"], "n_csms": e["n_csms"],
        "n_molecules": e["n_molecules"], "n_chemistry_axes": e["n_chemistry_axes"],
        "single_inference_ms_median": perf["single_inference_ms_median"],
        "api_overhead_ms": perf["api_overhead_ms"],
        "mcp_overhead_ms": perf["mcp_overhead_ms"],
        "parity_comparisons": par["parity"]["n_comparisons"],
        "parity_divergent": par["parity"]["n_divergent"],
        "parity_max_abs_diff": par["parity"]["max_abs_diff"],
        "gates_total": frz["gates"]["n"] + par["gates"]["n"],
        "gates_failed": frz["gates"]["failed"] + par["gates"]["failed"],
    }
    assert canonical["Scientific Atlas Fingerprint"] == SAF
    assert canonical["Frozen Runtime Content Hash"] == FRCH
    log("canonical values read from artifacts:")
    for k, v in canonical.items():
        log(f"  {k:<32s} {v}")

    problems: list[str] = []
    findings: list[dict] = []

    def check(doc: Path, ok: bool, code: str, detail: str) -> None:
        findings.append({"document": doc.name, "check": code, "status": "PASS" if ok else "FAIL",
                         "detail": detail})
        if not ok:
            problems.append(f"{doc.name}: {code} — {detail}")

    for doc in DOCS:
        text = doc.read_text()

        # 1. no document may state a hash that is not canonical
        for h in set(re.findall(r"\b[0-9a-f]{32}\b", text)):
            known = h in set(str(v) for v in canonical.values()) or h in {
                "dabd2834db31804fa948f5d30ff0fd44", "0285392b5a70f55f4938344462486d45",
                "c66f7304b08aa6dce8415ca09c8a600b", "06fb6b7f2f58746023c77473c54f04d0",
                "9d4bafe596e390d1ed0cd4eeecb50b6b", "3692ad772d661273c183fb23cf587c72",
                "f75bce02c75747507034cd235ef2e9eb", "395e9abb425eab6118bdc8c89031827b",
                "c9c6e8068d6116cbd22306addea24ac2", "0b387f2b26a16710e2436cb9e4d7865b"}
            check(doc, known, "hash.known", f"{h[:12]}… is not a committed value")

        # 2. engine dimensions, wherever stated
        for pat, want, name in ((r"(\d+)\s+LSMs", e["n_lsms"], "n_lsms"),
                                (r"(\d+)\s+CSMs", e["n_csms"], "n_csms"),
                                (r"(\d+)[- ]molecule bank", e["n_molecules"], "n_molecules"),
                                (r"(\d+) chemistry axes", e["n_chemistry_axes"], "n_axes")):
            got = {int(m) for m in re.findall(pat, text)}
            check(doc, not got or got == {want}, f"dims.{name}",
                  f"found {sorted(got)}, canonical {want}")

        # 3. the superseded DART model must not appear except where the divergence is recorded
        for phrase in FORBIDDEN_DART:
            if phrase in text:
                idx = text.index(phrase)
                window = text[max(0, idx - 600):idx + 400]
                excused = any(m in window for m in DIVERGENCE_MARKERS)
                check(doc, excused, "dart.superseded_model",
                      f"{phrase!r} appears without a divergence marker")

        # 4. atlas identity terminology
        if SAF in text or FRCH in text:
            named = ("Scientific Atlas Fingerprint" in text
                     or "Frozen Runtime Content Hash" in text
                     or "PHASE_10_ARCHITECTURE.md" in text)
            check(doc, named, "identity.named",
                  "quotes an atlas hash without naming which identity it is")

        # 5. stale test counts
        check(doc, "1436 passed" not in text, "tests.count",
              "quotes the superseded 1436-test count")

    # ── DART model must be described correctly where it is described at all ──
    dart_docs = [d for d in DOCS if "DART" in d.read_text()]
    for doc in dart_docs:
        t = doc.read_text()
        correct = ("dynamic perturbation" in t or "TrajectoryAdapter" in t
                   or "trajectory layer" in t)
        check(doc, correct, "dart.corrected_model",
              "mentions DART without the dynamic-perturbation / trajectory model")

    tab_path = OUT.tables / "documentation_consistency_v1.csv"
    import pandas as pd
    pd.DataFrame(findings).to_csv(tab_path, index=False)

    n_fail = sum(1 for f in findings if f["status"] == "FAIL")
    log(f"{len(DOCS)} documents · {len(findings)} checks · {n_fail} failed")
    for p in problems:
        log(f"  FAIL {p}")
    (OUT.artifacts / "documentation_consistency_v1.json").write_text(json.dumps({
        "canonical": canonical, "n_documents": len(DOCS), "n_checks": len(findings),
        "n_failed": n_fail, "documents": [d.name for d in DOCS],
        "findings": findings}, indent=1))
    log("PASS — documentation is internally consistent" if not n_fail
        else "FAIL — documentation is inconsistent")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
