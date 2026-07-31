"""Tests for tools/reproduce_gaira_foundation.py.

CI-safe: the normal suite needs NO raw dataset and NO SSD — it exercises the
interpretation-only mode (committed assets) + synthetic fixtures for orchestration
mechanics. The full raw rebuild is an OPTIONAL integration test, skipped unless a
data-root is available.
"""
import os
import sys
import json
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "reproduce_gaira_foundation.py"
ASSETS = REPO / "assets" / "foundation"
CANON_FP = "09ed804a40836f4a05a91ba10900cded"


def _load_tool():
    spec = importlib.util.spec_from_file_location("reproduce_gaira_foundation", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_tool()


def _canon_fp():
    return hashlib.sha256(
        np.ascontiguousarray(np.load(ASSETS / "manifold_components.npz")["components"]).tobytes()
    ).hexdigest()[:32]


# ── orchestration mechanics (synthetic; no data) ──
def test_nmf_parameters_match_source():
    sys.path.insert(0, str(REPO / "src"))
    from gaira.foundation.representation import fit_nmf
    rep = fit_nmf(np.abs(np.random.RandomState(0).rand(30, 40)), 24, seed=0)
    p = rep.model.get_params()
    assert p["n_components"] == 24 and p["init"] == "nndsvda"
    assert p["solver"] == "cd" and p["beta_loss"] == "frobenius"
    assert p["max_iter"] == 1500 and p["random_state"] == 0 and p["shuffle"] is False
    assert p["l1_ratio"] == 0.0 and p["alpha_W"] == 0.0


def test_preprocessing_parameters_match_source():
    sys.path.insert(0, str(REPO / "src"))
    from gaira.foundation import dataset as DS
    assert DS.PREPROC == {"baseline": "asls", "smooth": "savgol", "norm": "l2"}
    assert tuple(DS.WINDOW) == (450.0, 1800.0)
    assert len(DS.GRID) == 676


def test_hungarian_alignment_recovers_permutation():
    rng = np.random.RandomState(0)
    H = rng.rand(6, 20)
    perm = [3, 0, 5, 1, 4, 2]
    H_shuffled = H[perm]
    a = R.hungarian_align(H_shuffled, H)
    # rebuilt row i is canonical row perm[i]
    assert a["permutation_new_to_canon"] == perm
    assert a["min_cosine"] > 0.999


def test_norm_from_Z_matches_canonical_formula(tmp_path):
    Z = np.abs(np.random.RandomState(1).rand(50, 24))
    out = R._norm_from_Z(Z, [f"a{i}" for i in range(50)], tmp_path / "n.json",
                         tmp_path / "s.npz", CANON_FP)
    Zn = np.clip(Z, 0, None)
    center = np.median(Zn, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(Zn - center), axis=0), 1e-3)
    assert np.allclose(out["component_center"], np.round(center, 6))
    assert np.allclose(out["component_spread"], np.round(spread, 6))


def test_full_mode_refuses_without_data_root(monkeypatch):
    monkeypatch.delenv("GAIRA_DATA_ROOT", raising=False)
    monkeypatch.setattr(R, "DEFAULT_DATA_ROOT", Path("/nonexistent/data/root"))
    with pytest.raises(SystemExit):
        R.resolve_data_root(None)


def test_refuses_to_write_into_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--mode", "interpretation-only",
                                      "--output-dir", str(ASSETS / "sub")])
    with pytest.raises(SystemExit):
        R.main()


# ── interpretation-only end-to-end (committed assets; NO raw, NO SSD) ──
@pytest.fixture(scope="module")
def interp_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("interp")
    sys.argv = ["x", "--mode", "interpretation-only", "--foundation-root", str(ASSETS),
                "--output-dir", str(out)]
    fp_before = _canon_fp()
    R.main()
    return out / "reproduction_run", fp_before


def test_interpretation_only_runs_without_raw(interp_run):
    run, _ = interp_run
    for f in ["component_registry_v1.json", "component_theme_weights_v1.json",
              "mss_registry_v1.json", "bsv_regression_results.json", "run_manifest.json",
              "environment.json", "downstream_comparison.json"]:
        assert (run / f).exists(), f


def test_theme_weights_and_mss_reproduce_canonical(interp_run):
    run, _ = interp_run
    eng = REPO / "results/v5_rebuild/engine_v1/artifacts"
    assert json.loads((run / "component_theme_weights_v1.json").read_text()) == \
        json.loads((eng / "component_theme_weights_v1.json").read_text())
    assert json.loads((run / "mss_registry_v1.json").read_text()) == \
        json.loads((eng / "mss_registry_v1.json").read_text())


def test_registry_numerically_reproduces_canonical(interp_run):
    run, _ = interp_run
    eng = REPO / "results/v5_rebuild/engine_v1/artifacts"
    a = json.loads((run / "component_registry_v1.json").read_text())
    b = json.loads((eng / "component_registry_v1.json").read_text())
    # numeric content identical modulo the known PYTHONHASHSEED text-order cosmetic
    assert R._normalize_registry(a) == R._normalize_registry(b)


def test_canonical_assets_unmodified(interp_run):
    _, fp_before = interp_run
    assert fp_before == _canon_fp() == CANON_FP


def test_run_manifest_records_env_and_verdicts(interp_run):
    run, _ = interp_run
    man = json.loads((run / "run_manifest.json").read_text())
    assert man["canonical_asset_unmodified"] is True
    assert "scikit_learn" in man["environment"]
    assert "downstream_comparison" in man


def test_running_twice_gives_identical_interpretation(tmp_path):
    outs = []
    for i in range(2):
        d = tmp_path / f"run{i}"
        sys.argv = ["x", "--mode", "interpretation-only", "--foundation-root", str(ASSETS),
                    "--output-dir", str(d)]
        R.main()
        outs.append(d / "reproduction_run")
    for f in ["component_theme_weights_v1.json", "mss_registry_v1.json",
              "bsv_regression_results.json"]:
        assert (outs[0] / f).read_text() == (outs[1] / f).read_text(), f


# ── optional full raw integration (skipped unless data available) ──
def _data_root():
    c = os.environ.get("GAIRA_DATA_ROOT")     # integration test runs only if this is set
    return c if c and Path(c).exists() else None


@pytest.mark.skipif(_data_root() is None, reason="no raw data-root available")
def test_full_mode_reproduces_basis_fingerprint(tmp_path):
    sys.argv = ["x", "--mode", "full", "--data-root", _data_root(),
                "--output-dir", str(tmp_path / "full")]
    R.main()
    cmp = json.loads((tmp_path / "full/reproduction_run/basis_comparison.json").read_text())
    assert cmp["fingerprint_match"] is True
    assert cmp["exact_array_equality"] is True
    assert cmp["rebuilt_fingerprint"] == CANON_FP
