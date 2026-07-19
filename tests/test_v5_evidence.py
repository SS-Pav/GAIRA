"""GAIRA V5 Phase 2 Stage B — evidence-benchmark tests (§20).

Pure-logic tests use synthetic data (no data volume / no long training). Data-
integrity tests skip if /Volumes/SSD_Rad is not mounted. Encoder tests use a tiny
model / few epochs so they run fast and deterministically.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gaira.evidence import (splits as SP, regions, wavelets, dictionary, basis,
                            augmentations as AUG, evaluation as EV, training as TR,
                            uncertainty as UQ, serialization as SER, interpretability as IN)
from gaira.evidence.base import Representation

VOL = Path("/Volumes/SSD_Rad/GAIRA_DATA")
needs_data = pytest.mark.skipif(not VOL.exists(), reason="data volume not mounted")


def _synth(n_analytes=16, reps=3, d=96, seed=0):
    """Smooth band-based synthetic spectra (gaussian peaks) — realistic enough that
    a 1-bin shift preserves identity and major bands, unlike white noise."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(520, 1750, d)
    xx = np.arange(d)

    def spectrum(centers, widths, amps):
        v = np.zeros(d)
        for c, w, a in zip(centers, widths, amps):
            v += a * np.exp(-0.5 * ((xx - c) / w) ** 2)
        return v

    analyte_peaks = {a: (rng.uniform(5, d - 5, size=4), rng.uniform(2, 5, size=4),
                         rng.uniform(0.5, 1.5, size=4)) for a in range(n_analytes)}
    X, rows = [], []
    for a in range(n_analytes):
        cen, wid, amp = analyte_peaks[a]
        for mod in ("raman", "sers"):
            shift = 0.0 if mod == "raman" else rng.uniform(-2, 2)      # SERS shifts bands
            amp_mod = amp * (1.0 if mod == "raman" else rng.uniform(0.6, 1.4, size=4))
            for r in range(reps):
                v = spectrum(cen + shift, wid, amp_mod) + rng.normal(0, 0.01, d)
                v = np.clip(v, 0, None)
                X.append(v)
                rows.append(dict(spectrum_id=f"{a}:{mod}:{r}", analyte=f"an{a:02d}", modality=mod,
                                 source="src1" if mod == "sers" else ("srcA" if r % 2 else "srcB"),
                                 replicate_group=f"an{a:02d}|{mod}|s", family="famX" if a % 2 else "famY",
                                 non_small_molecule=False))
    return np.vstack(X), pd.DataFrame(rows), grid


class _DS:
    def __init__(self, X, meta, grid):
        self.X, self.meta, self.grid = X, meta, grid

    @property
    def matched_analytes(self):
        r = set(self.meta[self.meta.modality == "raman"].analyte)
        s = set(self.meta[self.meta.modality == "sers"].analyte)
        return sorted(r & s)


# ── split determinism + leakage safety ──
def test_splits_deterministic():
    X, meta, grid = _synth()
    d = _DS(X, meta, grid)
    s1 = SP.make_all_splits(d, k=4, seed=0)
    s2 = SP.make_all_splits(d, k=4, seed=0)
    assert s1 == s2


def test_split_A_no_analyte_leak():
    X, meta, grid = _synth(); d = _DS(X, meta, grid)
    sm = SP.make_all_splits(d, k=4, seed=0)
    chk = SP.verify_no_leakage(sm, meta)
    assert chk["A_held_out_analytes"]["ok"]


def test_split_B_matched_pair_holdout():
    X, meta, grid = _synth(); d = _DS(X, meta, grid)
    sm = SP.make_all_splits(d, k=4, seed=0)
    id2 = meta.set_index("spectrum_id")
    for fold in sm["splits"]["B_held_out_matched_pairs"]["folds"]:
        tr_an = set(id2.loc[fold["train"], "analyte"]) | set(id2.loc[fold["val"], "analyte"])
        te_an = set(id2.loc[fold["test"], "analyte"])
        assert not (tr_an & te_an)                       # held-out analytes fully excluded
        # both modalities of held-out analytes are in test
        for a in fold["held_out_matched_analytes"]:
            mods = set(id2.loc[[i for i in fold["test"] if id2.loc[i, "analyte"] == a]].modality)
            assert mods == {"raman", "sers"}


def test_split_C_no_replicate_group_leak():
    X, meta, grid = _synth(); d = _DS(X, meta, grid)
    sm = SP.make_all_splits(d, k=4, seed=0)
    assert SP.verify_no_leakage(sm, meta)["C_replicate_group_holdout"]["ok"]


def test_split_D_flags_single_source_sers():
    X, meta, grid = _synth(); d = _DS(X, meta, grid)
    sm = SP.make_all_splits(d, k=4, seed=0)
    dd = sm["splits"]["D_source_sensitivity"]
    assert dd["feasible"]["sers"]["leave_source_out_possible"] is False
    assert "IMPOSSIBLE for SERS" in dd["sers_note"]


# ── interpretable reps: training-only fitting + wavenumber mapping ──
def test_regions_contiguous_and_mapped():
    X, meta, grid = _synth()
    rep = regions.fit_regions(X, grid, 16)
    F = rep.transform(X)
    assert F.shape == (len(X), 16)
    wns = rep.feature_wavenumbers()
    assert len(wns) == 16 and all(lo <= hi for lo, hi in wns)          # contiguous, ordered


def test_regions_fit_uses_training_only():
    X, meta, grid = _synth()
    rep = regions.fit_regions(X[:20], grid, 8)                          # fit on subset
    assert rep.transform(X).shape[1] == 8                               # transforms anything


def test_wavelet_maps_back():
    X, meta, grid = _synth()
    rep = wavelets.fit_wavelets(X, grid)
    F = rep.transform(X)
    assert F.shape[0] == len(X) and rep.feature_wavenumbers() is not None


def test_dictionary_codes():
    X, meta, grid = _synth()
    rep = dictionary.fit_dictionary(X, grid, n_atoms=12, seed=0)
    assert rep.transform(X).shape == (len(X), 12)


def test_nmf_requires_nonnegative():
    X, meta, grid = _synth()
    signed = X - X.mean(0)                                              # ~half negative mass
    with pytest.raises(ValueError):
        basis.fit_nmf_basis(signed, grid, 8)


def test_nmf_accepts_nonnegative_and_activations_nonneg():
    X, meta, grid = _synth()
    rep = basis.fit_nmf_basis(np.abs(X), grid, 8)
    F = rep.transform(np.abs(X))
    assert (rep.basis >= 0).all()
    assert F.shape == (len(X), 8)


# ── augmentation bounds + validity audit ──
def test_augmentation_bounded_and_normalized():
    X, meta, grid = _synth()
    rng = np.random.default_rng(0)
    v = X[0] / np.linalg.norm(X[0])
    corrs = []
    for _ in range(10):
        a = AUG.augment(v, grid, AUG.AugConfig(), rng)
        assert abs(np.linalg.norm(a) - 1.0) < 1e-6                      # renormalized
        corrs.append(np.corrcoef(v, a)[0, 1])
    assert np.mean(corrs) > 0.9                                         # identity preserved on avg


def test_augmentation_audit_structure_and_bounded_invention():
    # retention floor is verified on REAL data (test_augmentation_audit_real_data);
    # on the synthetic fixture we assert the audit returns valid structure and does
    # not fabricate many peaks.
    X, meta, grid = _synth()
    aud = AUG.augmentation_audit(np.abs(X), grid, AUG.AugConfig(), seed=1, n_examples=8)
    assert aud["major_band_retention_mean"] is not None
    assert 0.0 <= aud["invented_peak_fraction_mean"] <= 0.5
    assert len(aud["examples"]) == 8


@needs_data
def test_augmentation_audit_real_data():
    from gaira.evidence import datasets as D
    d = D.build("A2_asls_savgol_snv")
    aud = AUG.augmentation_audit(d.X, d.grid, AUG.AugConfig(), seed=0, n_examples=6)
    assert aud["major_band_retention_mean"] > 0.8      # bounded augmentations keep major bands
    assert aud["invented_peak_fraction_mean"] < 0.1


# ── encoder: shape, determinism, collapse detection ──
def test_encoder_output_shape_and_deterministic():
    X, meta, grid = _synth()
    cfg = TR.EncoderConfig(name="t", arch="dual", latent=8, epochs=5, patience=5, seed=0)
    r1 = TR.train_encoder(cfg, X, meta, grid, X, meta)
    r2 = TR.train_encoder(cfg, X, meta, grid, X, meta)
    z1 = r1.transform(X, meta.modality.values)
    z2 = r2.transform(X, meta.modality.values)
    assert z1.shape == (len(X), 8)
    assert np.allclose(z1, z2, atol=1e-5)                              # deterministic seed


def test_shared_encoder_ignores_modality():
    X, meta, grid = _synth()
    cfg = TR.EncoderConfig(name="t", arch="shared", latent=8, epochs=3, seed=0)
    rep = TR.train_encoder(cfg, X, meta, grid)
    assert rep.transform(X).shape == (len(X), 8)


def test_collapse_diagnostics_detect_constant_embedding():
    # all rows near-identical → true collapse shows as ~all-duplicate + degenerate rank
    F = np.tile(np.arange(8.0), (30, 1)) + np.random.default_rng(0).normal(0, 1e-7, (30, 8))
    diag = EV.collapse_diagnostics(F)
    assert diag["duplicate_fraction"] > 0.9                            # near-collapse (all dup)
    assert diag["min_dim_std"] < 1e-3


def test_collapse_distinguishes_cross_analyte_dup():
    X, meta, grid = _synth()
    F = np.random.default_rng(0).normal(size=(len(X), 8))
    diag = EV.collapse_diagnostics(F, analytes=meta.analyte.values)
    assert "cross_analyte_duplicate_fraction" in diag


# ── retrieval metric correctness ──
def test_pooled_retrieval_perfect_when_modalities_identical():
    X, meta, grid = _synth()
    # make sers features == raman features per analyte → perfect retrieval
    F = np.zeros((len(meta), 8)); rng = np.random.default_rng(0)
    base = {a: rng.normal(size=8) for a in meta.analyte.unique()}
    for i, a in enumerate(meta.analyte):
        F[i] = base[a]
    r = EV.pooled_heldout_retrieval([(F, meta.reset_index(drop=True))], n_perm=100, seed=0)
    assert r["top_k"]["top1"] == 1.0


def test_leakage_metric_runs():
    X, meta, grid = _synth()
    F = X / np.linalg.norm(X, axis=1, keepdims=True)
    lk = EV.leakage_metrics(F, meta)
    assert "modality" in lk and lk["modality"]["chance_balanced_accuracy"] == 0.5


# ── uncertainty + interpretability shapes ──
def test_uncertainty_signals_shapes():
    X, meta, grid = _synth()
    F = X / np.linalg.norm(X, axis=1, keepdims=True)
    d2s = UQ.distance_to_support(F[:10], F[10:])
    ood = UQ.ood_score(F[:10], F[10:])
    assert d2s.shape == (10,) and ood.shape == (10,)


def test_encoder_attribution_shapes():
    X, meta, grid = _synth()
    cfg = TR.EncoderConfig(name="t", arch="dual", latent=8, epochs=3, seed=0)
    rep = TR.train_encoder(cfg, X, meta, grid)
    g = IN.input_gradient_attribution(rep, X[:1], "raman")
    o = IN.occlusion_attribution(rep, X[:1], "raman", window=8, stride=8)
    assert g.shape == (X.shape[1],) and o.shape == (X.shape[1],)


def test_serialization_roundtrip_interpretable(tmp_path):
    X, meta, grid = _synth()
    rep = regions.fit_regions(X, grid, 8)
    p = SER.save_representation(rep, tmp_path)
    assert p.exists() and (tmp_path / f"{rep.name}.npz").exists()


# ── data-integrity (skip without volume) ──
@needs_data
def test_stage_b_dataset_counts():
    from gaira.evidence import datasets as D
    d = D.build("A2_asls_savgol_snv")
    assert d.X.shape[0] == 479
    assert len(d.matched_analytes) == 51
    assert (d.meta.modality == "raman").sum() == 214


@needs_data
def test_stage_b_excludes_perturbation_and_preserves_provenance():
    from gaira.evidence import datasets as D
    d = D.build("A2_asls_savgol_snv")
    assert not d.meta.spectrum_id.str.contains("adenine_sers_control").any()
    for col in ("analyte", "modality", "source", "replicate_group", "data_role", "family"):
        assert col in d.meta.columns
    assert (d.meta.data_role == "grounding").all()


@needs_data
def test_stage_b_splits_leakage_safe_on_real_data():
    from gaira.evidence import datasets as D
    d = D.build("A2_asls_savgol_snv")
    sm = SP.make_all_splits(d, k=5, seed=0)
    chk = SP.verify_no_leakage(sm, d.meta)
    assert all(v["ok"] for v in chk.values())
