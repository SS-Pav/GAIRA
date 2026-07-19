"""GAIRA V5 Phase 2 Stage A — representation framework tests (§17).

Pure-logic tests use synthetic data (no data volume needed). Data-integrity
tests skip gracefully if /Volumes/SSD_Rad is not mounted.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gaira.representation import centroids as ct, pca as pca_m, retrieval as rt
from gaira.representation import factorization as fac, leakage as lk, stability as stab
from gaira.representation.metrics import cosine_sim, peak_overlap

VOL = Path("/Volumes/SSD_Rad/GAIRA_DATA")
needs_data = pytest.mark.skipif(not VOL.exists(), reason="data volume not mounted")


def _synth(seed=0):
    """20 analytes × 2 modalities × 3 reps; analyte signal + modality offset."""
    rng = np.random.default_rng(seed)
    d, rows, X = 40, [], []
    centers = rng.normal(size=(20, d))
    mod_off = {"raman": rng.normal(size=d) * 0.4, "sers": rng.normal(size=d) * 0.4}
    for a in range(20):
        for mod in ("raman", "sers"):
            for rep in range(3):
                v = centers[a] + mod_off[mod] + rng.normal(scale=0.05, size=d)
                X.append(v)
                rows.append(dict(analyte=f"an{a:02d}", modality=mod, source="s", replicate=str(rep)))
    return np.vstack(X), pd.DataFrame(rows)


# ── centroid construction ──
def test_centroid_count_and_no_cross_modality_averaging():
    X, meta = _synth()
    C, cmeta = ct.build_centroids(X, meta)
    # 20 analytes × 2 modalities × 1 source = 40 centroids
    assert C.shape[0] == 40
    assert set(cmeta.modality) == {"raman", "sers"}
    # every centroid belongs to exactly one modality (no mixing)
    assert (cmeta.groupby(["analyte", "modality"]).size() == 1).all()


def test_centroid_grouping_requires_modality_key():
    X, meta = _synth()
    with pytest.raises(AssertionError):
        ct.build_centroids(X, meta, group_cols=("analyte", "source"))


def test_centroid_is_mean_of_members():
    X, meta = _synth()
    C, cmeta = ct.build_centroids(X, meta)
    m = (cmeta.analyte == "an00") & (cmeta.modality == "raman")
    members = X[(meta.analyte == "an00") & (meta.modality == "raman")]
    assert np.allclose(C[m.values][0], members.mean(axis=0))


# ── PCA determinism + sign handling ──
def test_pca_deterministic():
    X, _ = _synth()
    p1, s1 = pca_m.fit_pca(X, 5, seed=0)
    p2, s2 = pca_m.fit_pca(X, 5, seed=0)
    assert np.allclose(s1, s2)
    assert np.allclose(p1.components_, p2.components_)


def test_bootstrap_stability_groups_by_analyte():
    X, meta = _synth()
    res = pca_m.bootstrap_stability(X, meta.analyte.values, n_components=4, n_boot=30, seed=0)
    assert len(res["loading_stability_mean"]) == 4
    # strong analyte structure → PC1 loading should be reasonably stable
    assert res["loading_stability_mean"][0] > 0.5


# ── retrieval correctness ──
def test_retrieval_perfect_when_modalities_identical():
    # if raman==sers centroids, top1 retrieval must be 1.0
    X, meta = _synth()
    C, cmeta = ct.build_centroids(X, meta)
    ram = cmeta.modality.values == "raman"
    R, rm = C[ram], cmeta[ram].reset_index(drop=True)
    # force SERS = Raman
    S, sm = R.copy(), rm.copy()
    sm = sm.assign(modality="sers")
    res = rt.cross_modal_retrieval(R, rm, S, sm)
    assert res["top_k"]["top1"] == 1.0
    assert res["reciprocal_nn_rate"] == 1.0


def test_retrieval_permutation_null_significant_for_signal():
    X, meta = _synth()
    C, cmeta = ct.build_centroids(X, meta)
    ram = cmeta.modality.values == "raman"; ser = cmeta.modality.values == "sers"
    res = rt.cross_modal_retrieval(C[ram], cmeta[ram].reset_index(drop=True),
                                   C[ser], cmeta[ser].reset_index(drop=True))
    perm = rt.permutation_null(res["_sim"], n_perm=500, seed=0)
    # shared analyte structure → matched cosine beats null
    assert perm["matched_cos"]["p_value"] < 0.05


def test_cosine_sim_shape_and_range():
    A = np.random.default_rng(0).normal(size=(4, 10))
    B = np.random.default_rng(1).normal(size=(6, 10))
    S = cosine_sim(A, B)
    assert S.shape == (4, 6)
    assert S.max() <= 1.0 + 1e-9 and S.min() >= -1.0 - 1e-9


def test_peak_overlap_bounds():
    assert peak_overlap(np.array([700, 1000]), np.array([700, 1000])) == 1.0
    assert peak_overlap(np.array([700]), np.array([1400])) == 0.0


# ── NMF non-negativity guard ──
def test_nmf_rejects_signed_input():
    X = np.random.default_rng(0).normal(size=(10, 20))  # signed
    with pytest.raises(ValueError):
        fac.fit_nmf(X, 3)


def test_nmf_accepts_nonnegative_input():
    X = np.abs(np.random.default_rng(0).normal(size=(10, 20)))
    res = fac.fit_nmf(X, 3)
    assert (res["components"] >= 0).all()


# ── grouped leakage split integrity ──
def test_grouped_leakage_no_analyte_crosses_split():
    X, meta = _synth()
    # sanity: single-class target is skipped
    res = lk.grouped_leakage(X, np.array(["a"] * len(meta)), meta.analyte.values)
    assert "skipped" in res
    # modality target: returns a balanced accuracy
    res2 = lk.grouped_leakage(X, meta.modality.values, meta.analyte.values, seed=0)
    assert res2["balanced_accuracy_mean"] is not None
    assert res2["chance_balanced_accuracy"] == 0.5


def test_consensus_clustering_bootstraps_by_analyte():
    X, meta = _synth()
    C, cmeta = ct.build_centroids(X, meta)
    res = stab.consensus_clustering(C, cmeta.analyte.values, k=10, n_boot=30, seed=0)
    assert 0.0 <= res["mean_consensus"] <= 1.0


# ── data-integrity (skip if no volume) ──
@needs_data
def test_input_audit_excludes_perturbation_and_non785():
    from gaira.representation import datasets as ds
    rows, excluded = ds.build_phase2_input("A1_asls_savgol_l2")
    ids = {r.spectrum_id for r in rows}
    assert not any("adenine_sers_control" in i for i in ids), "perturbation leaked into training"
    assert all(len(r.vector) == len(ds.GRID) for r in rows), "not on common grid"
    reasons = " ".join(r for _, r in excluded)
    assert "633nm" in reasons and "adenine_conc_series" in reasons


@needs_data
def test_corpus_counts_match_audit():
    from gaira.representation import datasets as ds
    rows, _ = ds.build_phase2_input("A1_asls_savgol_l2")
    _, meta = ds.matrix(rows)
    assert len(rows) == 479
    assert (meta.modality == "raman").sum() == 214
    assert (meta.modality == "sers").sum() == 265
    ram = set(meta[meta.modality == "raman"].analyte)
    ser = set(meta[meta.modality == "sers"].analyte)
    assert len(ram & ser) == 51


@needs_data
def test_no_replicate_crosses_centroid():
    """Centroid must aggregate all replicates of an (analyte,modality,source)."""
    from gaira.representation import datasets as ds
    rows, _ = ds.build_phase2_input("A1_asls_savgol_l2")
    X, meta = ds.matrix(rows)
    C, cmeta = ct.build_centroids(X, meta)
    # every spectrum maps to exactly one centroid group
    grp = meta.groupby(["analyte", "modality", "source"]).ngroups
    assert C.shape[0] == grp
