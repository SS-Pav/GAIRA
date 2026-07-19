"""GAIRA V5 Foundation — Raman-only foundation model tests."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gaira.foundation import (representation as RP, benchmark as BM, latent_space as LS,
                              axes as AX, bsv as BSV, mss as MSS, validation as VAL,
                              projection as PRJ, serialization as SER)
from gaira.foundation.families_raman import family_of, _norm

VOL = Path("/Volumes/SSD_Rad/GAIRA_DATA")
needs_data = pytest.mark.skipif(not VOL.exists(), reason="data volume not mounted")
FDIR = REPO / "results/v5_rebuild/foundation"


def _synth(n_analytes=18, reps=3, d=200, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(450, 1800, d)
    xx = np.arange(d)
    X, rows = [], []
    for a in range(n_analytes):
        cen = rng.uniform(10, d - 10, 4); wid = rng.uniform(3, 6, 4); amp = rng.uniform(.5, 1.5, 4)
        base = np.zeros(d)
        for c, w, m in zip(cen, wid, amp):
            base += m * np.exp(-0.5 * ((xx - c) / w) ** 2)
        for r in range(reps):
            X.append(np.clip(base + rng.normal(0, 0.01, d), 0, None))
            rows.append(dict(spectrum_id=f"{a}:{r}", analyte=f"an{a:02d}",
                             source="s1" if r < 2 else "s2", excitation_nm=785.0 if r < 2 else 532.0,
                             modality="raman", replicate=str(r)))
    return np.vstack(X), pd.DataFrame(rows), grid


class _C:
    def __init__(self, X, meta, grid):
        self.X, self.meta, self.grid = X, meta, grid


# ── family assignment (chemistry, not spectra) ──
def test_family_prefix_bug_fixed():
    """Regression: the stereochemistry stripper must not eat leading d/l letters."""
    assert _norm("lectin") == "lectin"
    assert _norm("dextrose") == "dextrose"
    assert _norm("(+)-dextrose") == "dextrose"
    assert _norm("l-alanine") == "alanine"


def test_family_rules():
    assert family_of("tripalmitin") == "triglyceride"
    assert family_of("(+)-galactose") == "saccharide"
    assert family_of("lectin") == "protein"
    assert family_of("horseradish peroxidase") == "protein"
    assert family_of("arachidonic acid") == "fatty_acid"
    assert family_of("amylopectin") == "polysaccharide"
    assert family_of("adenine") == "purine"
    assert family_of("uracil") == "pyrimidine"
    assert family_of("a-dna") == "nucleic_acid"


def test_base_family_map_takes_precedence():
    """The Stage-B curated map wins over the rule engine, so already-curated
    analytes keep their existing coarse label (cholesterol -> 'lipid', not 'sterol')."""
    assert family_of("cholesterol") == "lipid"
    # rules only fire where the curated map has no entry
    assert family_of("estrone") == "sterol"


# ── representations ──
@pytest.mark.parametrize("name", ["PCA", "NMF", "ICA", "SparseDict"])
def test_representation_roundtrip(name):
    X, meta, grid = _synth()
    rep = RP.FITTERS[name](X, 6, seed=0)
    Z = rep.transform(X); R = rep.reconstruct(X)
    assert Z.shape == (len(X), 6)
    assert R.shape == X.shape
    assert rep.components_.shape[0] == 6


def test_nmf_components_and_activations_nonnegative():
    X, meta, grid = _synth()
    rep = RP.FITTERS["NMF"](X, 6, seed=0)
    assert (rep.components_ >= -1e-9).all()
    assert (rep.transform(X) >= -1e-9).all()


def test_representation_deterministic():
    X, meta, grid = _synth()
    a = RP.FITTERS["NMF"](X, 5, seed=0).transform(X)
    b = RP.FITTERS["NMF"](X, 5, seed=0).transform(X)
    assert np.allclose(a, b)


# ── benchmark + selection ──
def test_benchmark_runs_and_scores():
    X, meta, grid = _synth()
    c = _C(X, meta, grid)
    df = BM.run_benchmark(c, ks=(4, 6), names=("PCA", "NMF"), n_splits=3, verbose=False)
    assert len(df) == 4
    s = BM.score(df)
    assert "total_score" in s and s.total_score.notna().all()


def test_reconstruction_is_not_dominant_criterion():
    assert BM.SELECTION_WEIGHTS["reconstruction"] <= 0.15
    assert BM.SELECTION_WEIGHTS["neighbourhood_preservation"] >= BM.SELECTION_WEIGHTS["reconstruction"]
    assert abs(sum(BM.SELECTION_WEIGHTS.values()) - 1.0) < 1e-9


def test_tiebreak_prefers_nonnegative_then_score():
    df = pd.DataFrame([
        dict(representation="ICA", k=32, total_score=0.787, nonneg=False,
             loading_sparsity=0.50, component_stability=0.77),
        dict(representation="NMF", k=24, total_score=0.786, nonneg=True,
             loading_sparsity=0.82, component_stability=0.81),
        dict(representation="NMF", k=32, total_score=0.767, nonneg=True,
             loading_sparsity=0.85, component_stability=0.80),
    ])
    pick, tied, why = BM.select_with_tiebreak(df)
    assert pick.representation == "NMF" and pick.k == 24     # constraint, then score (not sparsity)
    assert "non-negative" in why


def test_tiebreak_unique_winner_untouched():
    df = pd.DataFrame([
        dict(representation="PCA", k=8, total_score=0.90, nonneg=False,
             loading_sparsity=0.4, component_stability=0.5),
        dict(representation="NMF", k=8, total_score=0.50, nonneg=True,
             loading_sparsity=0.9, component_stability=0.9),
    ])
    pick, tied, why = BM.select_with_tiebreak(df)
    assert pick.representation == "PCA" and "no tie" in why


# ── manifold ──
def test_manifold_freeze_and_coordinates():
    X, meta, grid = _synth()
    man = LS.build_manifold(_C(X, meta, grid), "NMF", 6, seed=0)
    Z = man.coordinates(X, normalise=True)
    assert (Z >= -1e-9).all()
    assert np.allclose(Z.sum(axis=1), 1.0, atol=1e-6)        # BSV = shares summing to 1
    assert 0.0 <= man.stats["explained_variance"] <= 1.0
    assert man.stats["intrinsic_dimensionality"]["participation_ratio"] > 0


def test_manifold_projection_is_frozen(tmp_path):
    X, meta, grid = _synth()
    man = LS.build_manifold(_C(X, meta, grid), "NMF", 6, seed=0)
    z1 = man.project(X[:5])
    fp = SER.freeze_manifold(man, tmp_path)
    z2 = man.project(X[:5])                                   # freezing must not alter it
    assert np.allclose(z1, z2)
    assert (tmp_path / "manifold_components.npz").exists()
    assert len(fp) == 32


# ── axes / BSV / MSS ──
def test_axes_are_not_hardcoded_and_carry_uncertainty():
    X, meta, grid = _synth()
    man = LS.build_manifold(_C(X, meta, grid), "NMF", 6, seed=0)
    Z = man.project(X)
    axes, comp_df, A, cl = AX.build_axes(Z, meta, man.rep.components_, grid, pd.DataFrame())
    assert len(axes) >= 1
    for a in axes:
        assert "tentative_theme" in a and "theme_confidence" in a
        assert "TENTATIVE" in a["uncertainty_note"]
        assert set(a["components"]).issubset(set(range(6)))


def test_bsv_has_uncertainty_and_sums_to_one():
    X, meta, grid = _synth()
    man = LS.build_manifold(_C(X, meta, grid), "NMF", 6, seed=0)
    axes, comp_df, A, cl = AX.build_axes(man.project(X), meta, man.rep.components_, grid, pd.DataFrame())
    bsv, axis_df, Z = BSV.build_bsv(man, X, meta, axes)
    assert len(bsv) == meta.analyte.nunique()
    cols = [c for c in bsv.columns if c.startswith("bsv_c")]
    assert np.allclose(bsv[cols].sum(axis=1), 1.0, atol=1e-6)
    assert (bsv.mean_uncertainty >= 0).all()
    assert any(c.startswith("sd_c") for c in bsv.columns)


def test_mss_is_sparse_and_maps_to_bands():
    X, meta, grid = _synth()
    man = LS.build_manifold(_C(X, meta, grid), "NMF", 8, seed=0)
    axes, comp_df, A, cl = AX.build_axes(man.project(X), meta, man.rep.components_, grid, pd.DataFrame())
    m = MSS.build_mss(man, X, meta, axes, comp_df)
    assert len(m) == meta.analyte.nunique()
    assert (m.n_components_used <= 8).all()
    assert (m.latent_energy_captured >= 0.5).all()
    assert m.signature_bands_cm.apply(len).sum() > 0


# ── validation / projection ──
def test_heldout_projection_no_analyte_leak():
    X, meta, grid = _synth()
    v = VAL.heldout_analyte_projection(_C(X, meta, grid), "NMF", 5, n_splits=3, seed=0)
    assert len(v) == 3
    assert (v.n_test_analytes > 0).all()


def test_excitation_transfer_beats_null_on_structured_data():
    X, meta, grid = _synth()
    man = LS.build_manifold(_C(X, meta, grid), "NMF", 8, seed=0)
    df, null = VAL.excitation_transfer(man, _C(X, meta, grid))
    assert len(df) > 0
    assert df.cross_level_cos.mean() > null      # same analyte across lasers > different analytes


def test_projection_does_not_refit():
    X, meta, grid = _synth()
    man = LS.build_manifold(_C(X, meta, grid), "NMF", 6, seed=0)
    before = man.rep.components_.copy()
    ext = np.clip(X[:10] + 0.05, 0, None)
    PRJ.project(man, ext, meta.iloc[:10].reset_index(drop=True), None)
    PRJ.ood_score(man, ext, _C(X, meta, grid))
    assert np.allclose(before, man.rep.components_)   # frozen manifold untouched


def test_out_of_domain_sets_declared_excluded():
    assert "adenine_sers_control" in VAL.EXCLUDED_OUT_OF_DOMAIN
    assert any("Au-SERS" in v or "SERS" in v for v in VAL.EXCLUDED_OUT_OF_DOMAIN.values())


# ── real corpus integrity ──
@needs_data
def test_reference_corpus_is_raman_only():
    from gaira.foundation import dataset as DS
    c = DS.load_reference_corpus()
    assert (c.meta.modality == "raman").all()
    assert not c.meta.source.str.contains("sers", case=False).any()
    assert c.X.shape[0] == len(c.meta) == 375
    assert c.meta.analyte.nunique() == 167


@needs_data
def test_dataset_card_declares_exclusions():
    from gaira.foundation import dataset as DS
    card = DS.dataset_card(DS.load_reference_corpus())
    for d in ("Ag-SERS", "Au-SERS", "DART"):
        assert any(d in e for e in card["excluded_domains"])


@needs_data
def test_frozen_artifacts_present_and_consistent():
    j = FDIR / "artifacts/manifold.json"
    if not j.exists():
        pytest.skip("manifold not yet frozen")
    meta = json.loads(j.read_text())
    npz = np.load(FDIR / "artifacts/manifold_components.npz")
    assert npz["components"].shape[0] == meta["k"]
    assert meta["representation"] in RP.FITTERS
    assert len(meta["fingerprint"]) == 32
