"""GAIRA V5 — serum spike-in projection validation tests.

Central guarantees: the frozen atlas is never modified, despiking cannot delete
real bands on coarse axes, and trajectory statistics behave correctly against nulls.
"""
import json
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v5_rebuild/spike_validation/code"))

from gaira.foundation import serialization as SER

FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"
SV = REPO / "results/v5_rebuild/spike_validation"
VOL = Path("/Volumes/SSD_Rad/GAIRA_DATA")
needs_data = pytest.mark.skipif(not VOL.exists(), reason="data volume not mounted")
needs_frozen = pytest.mark.skipif(not (FROZEN / "manifold.json").exists(),
                                  reason="frozen atlas absent")


def _band_spectrum(wn, centers=(600, 1000, 1450), width=12.0, amp=1.0):
    y = np.zeros_like(wn, dtype=float)
    for c in centers:
        y += amp * np.exp(-0.5 * ((wn - c) / (width / 2.355)) ** 2)
    return y


# ── despiking must not destroy real bands ──
def test_despike_declines_on_coarse_axis():
    import spike_lib as SL
    wn = np.arange(400, 2000, 3.0)                     # 3 cm-1 sampling
    y = _band_spectrum(wn)
    out, n = SL.despike(y, wn=wn)
    assert n == -1, "coarse axis should be declined, not despiked"
    assert np.array_equal(out, y), "declined despiking must not alter the spectrum"


def test_despike_preserves_narrow_bands_on_fine_axis():
    import spike_lib as SL
    wn = np.arange(400, 2000, 1.0)
    y = _band_spectrum(wn)
    out, n = SL.despike(y, wn=wn)
    assert n == 0
    assert np.allclose(out, y), "a clean band spectrum must survive despiking untouched"


def test_despike_removes_a_true_cosmic_ray():
    import spike_lib as SL
    wn = np.arange(400, 2000, 1.0)
    y = _band_spectrum(wn) + np.random.default_rng(0).normal(0, 0.005, len(wn))
    i = 700
    y[i] += 50.0                                        # single-point spike
    out, n = SL.despike(y, wn=wn)
    assert n >= 1
    assert out[i] < 1.0, "the cosmic ray should be replaced by the local median"
    # the real bands must be untouched
    for c in (600, 1000, 1450):
        j = int(np.argmin(np.abs(wn - c)))
        assert out[j] == pytest.approx(y[j], abs=0.05)


def test_replicate_outlier_detection():
    import spike_lib as SL
    rng = np.random.default_rng(0)
    base = np.abs(rng.normal(size=200))
    X = np.vstack([base + rng.normal(0, 0.01, 200) for _ in range(5)] + [rng.normal(size=200)])
    bad, cosv = SL.replicate_outliers(X)
    assert bad[-1] and not bad[:-1].any()


# ── frozen atlas contract ──
@needs_frozen
def test_projection_never_mutates_atlas():
    atlas = SER.load_frozen_manifold(FROZEN)
    fp0 = hashlib.sha256(np.ascontiguousarray(atlas.components).tobytes()).hexdigest()[:32]
    X = np.abs(np.random.default_rng(0).normal(1, 0.3, (30, len(atlas.grid))))
    atlas.coordinates(X); atlas.project(X); atlas.reconstruct(X)
    fp1 = hashlib.sha256(np.ascontiguousarray(atlas.components).tobytes()).hexdigest()[:32]
    assert fp0 == fp1 == atlas.meta["fingerprint"]


# ── trajectory statistics ──
def test_trajectory_metrics_on_a_known_straight_line():
    import spike_lib as SL
    d = np.zeros(24); d[3] = 1.0
    concs = np.array([0, 1, 2, 3, 4], float)
    M = np.vstack([c * d for c in concs])
    tm = SL.trajectory_metrics(concs, M)
    assert tm["monotonicity_rho"] == pytest.approx(1.0)
    assert tm["straightness"] == pytest.approx(1.0, abs=1e-6)
    assert tm["mean_step_cosine"] == pytest.approx(1.0, abs=1e-6)


def test_trajectory_detects_zigzag():
    import spike_lib as SL
    rng = np.random.default_rng(0)
    concs = np.arange(6, dtype=float)
    M = np.cumsum(rng.normal(size=(6, 24)), axis=0)
    tm = SL.trajectory_metrics(concs, M)
    assert tm["straightness"] < 1.0
    assert -1.0 <= tm["mean_step_cosine"] <= 1.0


def test_monotonicity_null_rejects_random_labels():
    import spike_lib as SL
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 24))
    labels = rng.choice([0.0, 1.0, 2.0, 3.0], size=60)      # labels unrelated to Z
    res = SL.monotonicity_null(None, Z, labels, n_perm=200, seed=0)
    assert res["p_value"] > 0.05


def test_monotonicity_null_accepts_real_dose_effect():
    import spike_lib as SL
    rng = np.random.default_rng(0)
    d = np.zeros(24); d[5] = 1.0
    labels, rows = [], []
    for c in [0.0, 1.0, 2.0, 3.0, 4.0]:
        for _ in range(6):
            rows.append(c * d + rng.normal(0, 0.05, 24)); labels.append(c)
    res = SL.monotonicity_null(None, np.vstack(rows), np.array(labels), n_perm=200, seed=0)
    assert res["p_value"] < 0.05 and res["observed_rho"] > 0.8


def test_dose_response_model_selection():
    import spike_lib as SL
    c = np.array([0, 1, 2, 4, 8, 16, 32], float)
    lin = SL.dose_response_fits(c, 0.5 * c)
    assert lin["best_model"] in ("linear", "saturating")
    assert lin["linear_r2"] > 0.99
    sat = SL.dose_response_fits(c, 5 * c / (2 + c))
    assert sat.get("saturating_r2", 0) > sat["linear_r2"]


def test_displacement_and_angle():
    import spike_lib as SL
    ctrl = np.zeros((4, 24))
    treat = np.tile(np.eye(24)[7], (4, 1))
    d = SL.displacement(treat, ctrl)
    assert d["norm"] == pytest.approx(1.0)
    assert d["replicate_direction_cos"] == pytest.approx(1.0)
    assert SL.angle_deg(np.eye(24)[7], np.eye(24)[7]) == pytest.approx(0.0, abs=1e-3)
    assert SL.angle_deg(np.eye(24)[7], np.eye(24)[8]) == pytest.approx(90.0, abs=1e-3)


# ── produced artifacts ──
@needs_data
@needs_frozen
def test_study_manifest_confirms_atlas_unchanged():
    p = SV / "artifacts/study_manifest.json"
    if not p.exists():
        pytest.skip("study not run")
    m = json.loads(p.read_text())
    assert m["atlas"]["verified_unchanged"] is True
    assert m["atlas"]["k"] == 24


@needs_data
def test_all_perturbation_datasets_declared_out_of_domain():
    p = SV / "tables/phase1_dataset_audit.csv"
    if not p.exists():
        pytest.skip("audit not run")
    d = pd.read_csv(p)
    assert d.modality.str.contains("OUT OF DOMAIN").all()
    assert len(d) >= 5


@needs_data
def test_phase7_reports_null_against_mismatched_control():
    p = SV / "tables/phase7_summary.json"
    if not p.exists():
        pytest.skip("phase 7 not run")
    s = json.loads(p.read_text())
    for k in ("matched_cos_vs_pureSERS_median", "null_cos_vs_pureSERS_median",
              "n_analytes_cos_above_null_p05"):
        assert k in s
    assert 0 <= s["n_analytes_cos_above_null_p05"] <= s["n_analytes"]


@needs_data
def test_uricase_depletion_direction_is_negative():
    """Enzymatic urate removal must move AWAY from the urate direction."""
    p = SV / "tables/phase11_controls.json"
    if not p.exists():
        pytest.skip("controls not run")
    c = json.loads(p.read_text())
    if "uricase_depletion" not in c:
        pytest.skip("uricase unavailable")
    assert c["uricase_depletion"]["cos_vs_urate_raman_direction"] < 0
