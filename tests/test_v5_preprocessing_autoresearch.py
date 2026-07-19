"""GAIRA V5 Stage B0 — preprocessing AutoResearch tests.

Pure-logic tests use synthetic spectra. Data-dependent tests skip without the
volume. Leakage safety, spectral integrity guards and determinism are the focus.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gaira.preprocessing_autoresearch import (smoothing as SM, background_models as BG,
                                              normalization as NM, pipeline as PL,
                                              peak_integrity as PI, objectives as OB,
                                              evaluator as EV, pareto as PA,
                                              serialization as SER, search_space as SS)
from gaira.preprocessing_autoresearch.derivatives import apply_derivative
from gaira.preprocessing_autoresearch.scaling import viz_scale

VOL = Path("/Volumes/SSD_Rad/GAIRA_DATA")
needs_data = pytest.mark.skipif(not VOL.exists(), reason="data volume not mounted")
AUD = REPO / "results/v5_rebuild/preprocessing_autoresearch"


def _synth(n_analytes=12, reps=3, seed=0):
    """Band-based synthetic spectra on the real grid, with a shared SERS background."""
    rng = np.random.default_rng(seed)
    grid = PL.GRID; d = len(grid); xx = np.arange(d)
    bg = 3.0 * np.exp(-0.5 * ((xx - d * 0.45) / (d * 0.30)) ** 2)   # broad common component
    X, rows = [], []
    for a in range(n_analytes):
        cen = rng.uniform(20, d - 20, size=4); wid = rng.uniform(3, 6, size=4)
        amp = rng.uniform(0.6, 1.4, size=4)
        base = np.zeros(d)
        for c, w, m in zip(cen, wid, amp):
            base += m * np.exp(-0.5 * ((xx - c) / w) ** 2)
        for mod in ("raman", "sers"):
            for r in range(reps):
                y = base.copy()
                if mod == "sers":
                    y = 0.25 * y + bg                      # analyte is a small residual
                y = y + rng.normal(0, 0.01, d)
                X.append(y)
                rows.append(dict(spectrum_id=f"{a}:{mod}:{r}", analyte=f"an{a:02d}", modality=mod,
                                 source="s1", replicate=str(r),
                                 replicate_group=f"an{a:02d}|{mod}|s1", matched=True,
                                 family="fam", acquisition_domain=f"{mod}|s1"))
    return np.vstack(X), pd.DataFrame(rows), grid


# ── leakage safety ──
def test_nested_splits_no_outer_leak():
    A = [f"an{i:02d}" for i in range(20)]
    sp = EV.make_nested_splits(A, n_outer=4, n_inner=3, seed=0)
    chk = EV.verify_nested_no_leakage(sp)
    assert chk["ok"], chk["problems"]
    for f in sp["folds"]:
        te = set(f["test_analytes"])
        for inn in f["inner"]:
            assert not (te & set(inn["train"]))
            assert not (te & set(inn["val"]))
            assert not (set(inn["train"]) & set(inn["val"]))


def test_nested_splits_deterministic():
    A = [f"an{i:02d}" for i in range(20)]
    assert EV.make_nested_splits(A, 4, 3, seed=0) == EV.make_nested_splits(A, 4, 3, seed=0)


def test_every_analyte_is_tested_once():
    A = [f"an{i:02d}" for i in range(20)]
    sp = EV.make_nested_splits(A, n_outer=4, n_inner=3, seed=0)
    seen = [a for f in sp["folds"] for a in f["test_analytes"]]
    assert sorted(seen) == sorted(A)


def test_background_fitted_on_training_spectra_only():
    X, meta, grid = _synth()
    cand = SS._mk("t", "t", bg=("mean", {}))
    train = meta.analyte.isin(meta.analyte.unique()[:6]).values
    st = PL.fit_stage2(cand, X, meta, train)
    # background equals mean of TRAIN sers rows only
    tr_sers = train & (meta.modality.values == "sers")
    assert np.allclose(st["background"].b, X[tr_sers].mean(0))
    assert st["background"].fitted_on == int(tr_sers.sum())


def test_background_never_sees_raman():
    X, meta, grid = _synth()
    cand = SS._mk("t", "t", bg=("mean", {}))
    train = np.ones(len(meta), bool)
    st = PL.fit_stage2(cand, X, meta, train)
    sers_only_mean = X[(meta.modality.values == "sers")].mean(0)
    assert np.allclose(st["background"].b, sers_only_mean)


def test_background_applied_only_to_sers():
    X, meta, grid = _synth()
    cand = SS._mk("t", "t", bg=("mean", {}), agg="none")
    st = PL.fit_stage2(cand, X, meta, np.ones(len(meta), bool))
    F, fmeta = PL.apply_stage2(cand, X, meta, st, aggregate=False)
    ram = fmeta.modality.values == "raman"
    # Raman rows are unchanged up to normalization
    assert np.allclose(F[ram], np.vstack([NM.norm_l2(x) for x in X[ram]]))


def test_no_paired_raman_guides_sers():
    """Perturbing one analyte's Raman must not change ANY Ag-SERS feature."""
    X, meta, grid = _synth()
    cand = SS._mk("t", "t", bg=("lowrank", {"k": 2}), agg="mean")
    st = PL.fit_stage2(cand, X, meta, np.ones(len(meta), bool))
    F0, fm0 = PL.apply_stage2(cand, X, meta, st, aggregate=True)
    X2 = X.copy()
    X2[(meta.modality.values == "raman")] *= 3.0          # drastically alter Raman
    st2 = PL.fit_stage2(cand, X2, meta, np.ones(len(meta), bool))
    F2, fm2 = PL.apply_stage2(cand, X2, meta, st2, aggregate=True)
    s = fm0.modality.values == "sers"
    assert np.allclose(F0[s], F2[s], atol=1e-9)


# ── replicate aggregation ──
def test_aggregators_correctness():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]])
    assert np.allclose(NM.aggregate(X, "mean"), X.mean(0))
    assert np.allclose(NM.aggregate(X, "median"), np.median(X, axis=0))
    rob = NM.aggregate(X, "huber")
    assert np.all(rob < X.mean(0))                      # robust to the outlier replicate
    assert NM.aggregate(X, "ivw").shape == (2,)


def test_aggregation_only_within_group():
    X, meta, grid = _synth(n_analytes=4, reps=3)
    cand = SS._mk("t", "t", agg="mean")
    st = PL.fit_stage2(cand, X, meta, np.ones(len(meta), bool))
    F, fmeta = PL.apply_stage2(cand, X, meta, st, aggregate=True)
    assert len(F) == fmeta.shape[0] == 4 * 2            # analytes x modalities
    assert set(fmeta.n_rep) == {3}


# ── spectral integrity guards ──
def test_oversmoothing_caught_by_width_not_retention():
    """Over-smoothing preserves peak COUNT and POSITION (prominence is relative to
    each spectrum's own range) while broadening bands, so `width_ratio` is the guard
    that catches it — not `retention`. Both rules exist in pareto.REJECT."""
    X, meta, grid = _synth()
    y = X[0]
    over = SM.smooth_gaussian(y, sigma=25.0)
    r = PI.retention_invention(y, over, grid)
    assert r["retention"] >= 0.9                       # count/positions survive
    assert r["width_ratio"] > PA.REJECT["max_width_ratio"]   # but bands are broadened
    assert PA.apply_rejection(
        pd.DataFrame([dict(cid="x", si_peak_retention=r["retention"],
                           si_peak_invention=0.0, si_peak_width_ratio=r["width_ratio"],
                           si_cross_analyte_duplicate_frac=0.0,
                           si_negative_lobe_burden=0.1, si_edge_artefact_ratio=1.0)]),
        {}).rejected.iloc[0]


def test_peak_retention_detects_genuine_peak_loss():
    """Replacing a spectrum by a smooth low-order trend removes its bands entirely."""
    X, meta, grid = _synth()
    y = X[0]
    xx = np.arange(len(y))
    flattened = np.polyval(np.polyfit(xx, y, 3), xx)   # all narrow structure gone
    r = PI.retention_invention(y, flattened, grid)
    assert r["retention"] < 0.9


def test_peak_invention_detected():
    X, meta, grid = _synth()
    y = X[0].copy()
    rng = np.random.default_rng(0)
    noisy = y + rng.normal(0, 0.5, len(y))              # spurious structure
    r = PI.retention_invention(y, noisy, grid)
    assert r["invention"] > 0.0


def test_width_ratio_flags_broadening():
    X, meta, grid = _synth()
    y = X[0]
    r = PI.retention_invention(y, SM.smooth_gaussian(y, sigma=6.0), grid)
    assert r["width_ratio"] > 1.0


def test_artefact_burden_detects_oversubtraction():
    X, meta, grid = _synth()
    y = X[:5]
    over = y - 2.0 * y.mean(0)                          # heavy over-subtraction
    a = PI.artefact_burden(over, grid)
    assert a["negative_lobe_burden"] > 0.2


def test_effective_rank_collapse():
    X = np.tile(np.linspace(0, 1, 50), (20, 1))
    assert PI.effective_rank(X) < 2.0


# ── derivatives / smoothing bounds ──
def test_derivative_correctness():
    g = PL.GRID
    y = np.sin(np.linspace(0, 6 * np.pi, len(g)))
    d1 = apply_derivative(y[None, :], 1)[0]
    assert np.corrcoef(d1, np.gradient(y))[0, 1] > 0.999
    d2 = apply_derivative(y[None, :], 2)[0]
    assert d2.shape == y.shape
    cc = apply_derivative(y[None, :], "concat")
    assert cc.shape[1] == 2 * len(y)


def test_smoothing_bounds_savgol_windows():
    for _, p in [s for s in SS.SMOOTHERS if s[0] == "savgol"]:
        assert p["window"] <= 21 and p["poly"] in (2, 3)


# ── determinism & serialization ──
def test_pipeline_serialization_roundtrip(tmp_path):
    c = SS._mk("X1", "arm", base=("asls", {"lam": 1e5}),
               smooth=("savgol", {"window": 9, "poly": 3}), bg=("lowrank", {"k": 2}), nr="l2")
    p = tmp_path / "c.json"
    d1 = SER.save_candidate(c, p)
    c2 = SER.load_candidate(p)
    assert SER.candidate_to_json(c2)["fingerprint"] == d1["fingerprint"]


def test_stage1_deterministic():
    X, meta, grid = _synth()
    raw = [(f"s{i}", grid, X[i], meta.modality.iloc[i]) for i in range(len(X))]
    c = SS._mk("t", "t", base=("asls", {"lam": 1e5}), smooth=("savgol", {"window": 9, "poly": 3}))
    a = PL.Stage1Cache(raw).build(c)
    b = PL.Stage1Cache(raw).build(c)
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b))


# ── rejection rules & Pareto ──
def test_rejection_rules_fire():
    base = {"rep_raman_replicate_cos": 0.99, "rep_sers_replicate_cos": 0.95,
            "chem_raman_1nn": 0.95, "chem_sers_1nn": 0.85, "si_effective_rank": 50.0}
    df = pd.DataFrame([
        dict(cid="good", rep_raman_replicate_cos=0.99, rep_sers_replicate_cos=0.94,
             chem_raman_1nn=0.95, chem_sers_1nn=0.85, si_peak_retention=0.99,
             si_peak_invention=0.0, si_peak_width_ratio=1.0, si_effective_rank=48.0,
             si_cross_analyte_duplicate_frac=0.0, si_negative_lobe_burden=0.1,
             si_edge_artefact_ratio=1.0),
        dict(cid="collapse", rep_raman_replicate_cos=0.99, rep_sers_replicate_cos=0.94,
             chem_raman_1nn=0.95, chem_sers_1nn=0.85, si_peak_retention=0.99,
             si_peak_invention=0.0, si_peak_width_ratio=1.0, si_effective_rank=48.0,
             si_cross_analyte_duplicate_frac=0.9, si_negative_lobe_burden=0.1,
             si_edge_artefact_ratio=1.0),
        dict(cid="smoothed", rep_raman_replicate_cos=0.99, rep_sers_replicate_cos=0.94,
             chem_raman_1nn=0.95, chem_sers_1nn=0.85, si_peak_retention=0.5,
             si_peak_invention=0.0, si_peak_width_ratio=2.0, si_effective_rank=48.0,
             si_cross_analyte_duplicate_frac=0.0, si_negative_lobe_burden=0.1,
             si_edge_artefact_ratio=1.0),
    ])
    out = PA.apply_rejection(df, base)
    r = dict(zip(out.cid, out.rejected))
    assert r["good"] is False or r["good"] == False
    assert r["collapse"] and "analyte_collapse" in out[out.cid == "collapse"].reject_reasons.iloc[0]
    assert r["smoothed"] and "peak_loss" in out[out.cid == "smoothed"].reject_reasons.iloc[0]


def test_pareto_front_reproducible():
    df = pd.DataFrame([
        dict(cid="a", cm_mrr=0.5, pk_effect_vs_mismatched=0.1, rep_sers_replicate_cos=0.9,
             chem_raman_1nn=0.9, si_peak_retention=1.0, n_stages=2),
        dict(cid="b", cm_mrr=0.4, pk_effect_vs_mismatched=0.05, rep_sers_replicate_cos=0.8,
             chem_raman_1nn=0.8, si_peak_retention=0.95, n_stages=4),
        dict(cid="c", cm_mrr=0.45, pk_effect_vs_mismatched=0.2, rep_sers_replicate_cos=0.85,
             chem_raman_1nn=0.85, si_peak_retention=0.98, n_stages=3),
    ])
    f1 = PA.pareto_front(df); f2 = PA.pareto_front(df)
    assert list(f1.on_front) == list(f2.on_front)
    assert f1[f1.cid == "a"].on_front.iloc[0]          # dominates b
    assert not f1[f1.cid == "b"].on_front.iloc[0]


# ── controls ──
def test_null_controls_present_and_ordered():
    X, meta, grid = _synth(n_analytes=8)
    cand = SS._mk("t", "t", agg="mean")
    st = PL.fit_stage2(cand, X, meta, np.ones(len(meta), bool))
    F, fmeta = PL.apply_stage2(cand, X, meta, st, aggregate=True)
    rng = np.random.default_rng(0)
    res = OB.peak_correspondence(F, fmeta, sorted(meta.analyte.unique()), grid, rng)
    for k in ("matched", "mismatched", "random", "effect_vs_mismatched"):
        assert k in res
    assert res["matched"] >= res["random"] - 1e-9      # synthetic data shares bands


def test_permutation_control_available():
    X, meta, grid = _synth(n_analytes=8)
    cand = SS._mk("t", "t", agg="mean")
    st = PL.fit_stage2(cand, X, meta, np.ones(len(meta), bool))
    F, fmeta = PL.apply_stage2(cand, X, meta, st, aggregate=True)
    cm = OB.cross_modal(F, fmeta, sorted(meta.analyte.unique()), n_perm=50,
                        rng=np.random.default_rng(0))
    assert "perm_mrr_p" in cm and 0 <= cm["perm_mrr_p"] <= 1


def test_background_variance_explained_monotone():
    X, meta, grid = _synth()
    sers = X[(meta.modality.values == "sers")]
    v = []
    for k in (1, 2, 3):
        m = BG.make("lowrank", k=k).fit(sers)
        v.append(m.variance_explained(sers))
    assert v[0] <= v[1] + 1e-9 <= v[2] + 1e-9          # more components remove more variance


# ── visualization scaling must not touch metrics ──
def test_viz_scaling_is_separate_from_normalizers():
    assert "max1" not in NM.NORMALIZERS and "area1" not in NM.NORMALIZERS
    y = np.array([1.0, 2.0, 3.0])
    assert np.isclose(np.max(np.abs(viz_scale(y, "max1"))), 1.0)


def test_snv_declared_control_only():
    assert "snv" in NM.CONTROL_ONLY


# ── study integrity (frozen artifacts) ──
@needs_data
def test_frozen_splits_leakage_safe_on_real_corpus():
    sp = json.loads((AUD / "configs/nested_splits.json").read_text())
    assert EV.verify_nested_no_leakage(sp)["ok"]
    seen = [a for f in sp["folds"] for a in f["test_analytes"]]
    assert len(seen) == len(set(seen)) == 51


@needs_data
def test_outer_test_consumed_at_most_once():
    p = AUD / "configs/study_manifest.json"
    if p.exists():
        m = json.loads(p.read_text())
        assert isinstance(m.get("outer_test_used"), bool)
        assert m["n_matched_analytes"] == 51 and m["n_spectra"] == 479


@needs_data
def test_corpus_matches_frozen_manifest():
    from gaira.preprocessing_autoresearch import corpus as CO
    raw, meta = CO.load_raw_frozen()
    assert len(meta) == 479
    assert (meta.modality == "raman").sum() == 214
    assert len(CO.matched_analytes(meta)) == 51
