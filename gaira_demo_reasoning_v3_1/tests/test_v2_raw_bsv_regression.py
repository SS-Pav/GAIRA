"""V3 raw BSV must equal V2 raw BSV (atol <= 1e-9) and the engine must be
byte-identical to V2's frozen engine."""
import hashlib
import json
from pathlib import Path

import numpy as np

DEMO_ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = DEMO_ROOT.parent / "gaira_demo_reasoning_v2"
BASELINE = DEMO_ROOT / "tests" / "baselines" / "v2_raw_bsv_baseline.json"

ENGINE_FILES = [
    "preprocessing.py", "primitive_extraction.py", "motif_scoring.py",
    "mss_scoring.py", "substrate_physics.py", "bsv_projection.py",
    "evidence_synthesis.py", "report_builder.py",
]


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_engine_byte_identical_to_v2():
    for f in ENGINE_FILES:
        v3 = DEMO_ROOT / "gaira_core" / f
        v2 = V2_ROOT / "gaira_core" / f
        assert _sha(v3) == _sha(v2), f"engine file diverged from V2: {f}"


def test_raw_bsv_matches_v2_baseline():
    import _raw_bsv_cases as c
    baseline = json.loads(BASELINE.read_text())
    current = c.standard_raw_bsvs()
    assert set(current) == set(baseline), "case set changed vs baseline"
    max_abs = 0.0
    for name in baseline:
        for axis, v in baseline[name].items():
            max_abs = max(max_abs, abs(float(v) - float(current[name][axis])))
    assert max_abs <= 1e-9, f"raw BSV drift vs V2: max|diff|={max_abs:.2e}"
