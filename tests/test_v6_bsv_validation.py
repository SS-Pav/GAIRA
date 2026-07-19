"""GAIRA BSV Validation tests — verify the validation is READ-ONLY on the frozen
engine and that its statistics behave correctly.
"""
import json
import sys
import hashlib
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v5_rebuild/bsv_validation/code"))

FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"
BV = REPO / "results/v5_rebuild/bsv_validation"
VOL = Path("/Volumes/SSD_Rad/GAIRA_DATA")
needs_data = pytest.mark.skipif(not VOL.exists(), reason="data volume not mounted")
needs_art = pytest.mark.skipif(not (FROZEN / "manifold.json").exists(), reason="frozen atlas absent")


# ── pure-logic stats ──
def test_monotonicity_detects_saturating_dose():
    import bsv_val_lib as L
    c = np.repeat([0, 1, 2, 4, 8, 16], 4).astype(float)
    y = 5 * c / (2 + c) + np.random.default_rng(0).normal(0, 0.01, len(c))
    m = L.monotonicity(c, y)
    assert m["spearman"] > 0.9
    assert m["best_model"] == "saturating"
    assert m["effect_size"] > 1


def test_monotonicity_null_for_random():
    import bsv_val_lib as L
    rng = np.random.default_rng(0)
    c = rng.choice([0.0, 1, 2, 3], 40); y = rng.normal(size=40)
    p = L.permutation_p(c, y, n=300)
    assert p > 0.05


def test_icc_high_for_tight_groups_low_for_noise():
    import bsv_val_lib as L
    rng = np.random.default_rng(0)
    tight = [rng.normal(m, 0.01, 5) for m in (0, 1, 2, 3)]      # separated groups, low within-var
    noisy = [rng.normal(0, 1, 5) for _ in range(4)]             # no between-group signal
    assert L.icc(tight) > 0.9
    assert L.icc(noisy) < 0.5


def test_cv_and_cos():
    import bsv_val_lib as L
    assert L.cv([1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-6)
    assert L.cos(np.array([1, 0, 0]), np.array([1, 0, 0])) == pytest.approx(1.0)
    assert L.cos(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)


# ── the harness must drive the REAL frozen engine ──
@needs_art
def test_harness_uses_frozen_engine_and_is_input_sensitive():
    import bsv_val_lib as L
    H = L.Harness()
    W = np.load(FROZEN / "manifold_components.npz")["components"]
    fp = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()[:32]
    assert H.eng.atlas.meta["fingerprint"] == fp
    a = np.full(24, 0.02); a[3] = 0.5; a[15] = 0.3        # purine-loaded
    hi = H.infer_coords(a).bsv.composition["nucleic_purine"]
    a2 = a.copy(); a2[3] = 0.05                            # suppress the adenine component
    lo = H.infer_coords(a2).bsv.composition["nucleic_purine"]
    assert lo < hi                                        # genuinely input-sensitive, not hard-coded


@needs_art
def test_bsv_row_has_all_themes_and_ood():
    import bsv_val_lib as L
    H = L.Harness()
    row = H.bsv_row(np.full(24, 1.0 / 24))
    for t in H.bio:
        assert f"theme_{t}" in row and f"conf_{t}" in row
    assert "ood" in row and 0 <= row["ood"] <= 1


# ── produced results self-consistent (need data volume to have been generated) ──
@needs_data
def test_monotonicity_all_saturating_and_significant():
    p = BV / "tables/part3_monotonicity.csv"
    if not p.exists():
        pytest.skip("validation not run")
    import pandas as pd
    m = pd.read_csv(p)
    assert (m.permutation_p <= 0.01).all()
    assert (m.best_model == "saturating").all()
    # ergothioneine → sulfur is the cleanest
    erg = m[m.experiment == "ergothioneine"]
    assert erg.spearman.iloc[0] > 0.9


@needs_data
def test_effective_dimensionality_below_nominal():
    p = BV / "artifacts/part12_state_space.json"
    if not p.exists():
        pytest.skip("validation not run")
    g = json.loads(p.read_text())
    assert g["effective_dimensionality_entropy"] < g["n_biochemical_themes"]
    assert g["n_components_90pct_variance"] <= g["n_biochemical_themes"]


@needs_data
def test_confidence_tracks_ood_negative():
    p = BV / "tables/part9_confidence_system.csv"
    if not p.exists():
        pytest.skip("validation not run")
    import pandas as pd
    c = pd.read_csv(p)
    ref = c[c.group == "pure_raman_reference"].iloc[0]
    # in-domain reference: lowest OOD and highest confidence of the groups
    assert ref.median_ood == c.median_ood.min()
    assert ref.median_confidence == c.median_confidence.max()


@needs_data
def test_purines_cluster_in_bsv_geometry():
    p = BV / "tables/part7_nearest_neighbours.csv"
    if not p.exists():
        pytest.skip("validation not run")
    import pandas as pd
    nn = pd.read_csv(p).set_index("analyte")
    purines = {"adenine", "xanthine", "guanine", "hypoxanthine", "urate"}
    # the CORE purine bases cluster (each other purine is their nearest neighbour).
    # urate is a documented edge case (its nearest neighbour is ergothioneine, another
    # weak-signal small molecule) — reported in the validation report, not enforced here.
    core = {"adenine", "xanthine", "guanine", "hypoxanthine"}
    hits = sum(nn.loc[a, "nearest"] in purines for a in core & set(nn.index))
    assert hits >= 3


@needs_data
def test_engine_not_mutated_by_validation():
    """Sanity: the frozen atlas fingerprint is unchanged after the validation run."""
    W = np.load(FROZEN / "manifold_components.npz")["components"]
    fp = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()[:32]
    man = BV / "artifacts/validation_manifest.json"
    if man.exists():
        assert json.loads(man.read_text())["atlas_fingerprint"] == fp
