"""V3.1 raw BSV == V3 (byte-identical engine); frozen coordinates unchanged."""
import hashlib, json
from pathlib import Path
import numpy as np
DEMO = Path(__file__).resolve().parent.parent
V3 = DEMO.parent / "gaira_demo_reasoning_v3"
ENGINE = ["preprocessing.py","primitive_extraction.py","motif_scoring.py","mss_scoring.py",
          "substrate_physics.py","bsv_projection.py","evidence_synthesis.py","report_builder.py"]


def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_engine_byte_identical_to_v3():
    for f in ENGINE:
        assert _sha(DEMO/"gaira_core"/f) == _sha(V3/"gaira_core"/f), f"engine diverged: {f}"


def test_raw_bsv_matches_v3_baseline():
    import _raw_bsv_cases as c
    base = json.loads((DEMO/"tests"/"baselines"/"v2_raw_bsv_baseline.json").read_text())
    cur = c.standard_raw_bsvs()
    mx = max(abs(base[n][a]-cur[n][a]) for n in base for a in base[n])
    assert mx <= 1e-9, f"raw BSV drift {mx:.2e}"


def test_frozen_calibration_unchanged_vs_v3():
    a = json.loads((DEMO/"data"/"generated"/"global_coordinate_calibration_v1.json").read_text())
    b = json.loads((V3/"data"/"generated"/"global_coordinate_calibration_v1.json").read_text())
    assert a["content_sha256"] == b["content_sha256"], "frozen calibration content changed vs V3"
