"""GAIRA V7 — regression tests for the CONTROL EXPERIMENT: frozen-atlas decomposition.

NOT the canonical V7 Phase 01. This layer decomposes the frozen V5 atlas components; the
approved architecture derives LSMs from an independent class-local NMF over balanced
references. See GAIRA_v7_rebuild/context/ARCHITECTURE_COMPLIANCE_AUDIT.md.

Its objects are Atlas Component Substructures (ACS); the term ACS is reserved for the
specification's object.

Two kinds of test:

  * UNIT tests of `src/gaira/v7/lsm/` on synthetic inputs — these run anywhere, need no
    data, and pin the behaviour of the motif object, the clustering, the matching and the
    serialisation.
  * CONTRACT tests against the committed Phase-01 artefacts — these pin the frozen result:
    the atlas is unchanged, the registry is intact, discovery is deterministic.

The single most important test in this file is `test_atlas_fingerprint_unchanged`. Phase 01
is an interpretation layer; the moment it alters the atlas it has failed its premise.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

P01 = REPO / "results/v7_rebuild/control_experiments/frozen_atlas_decomposition"
T, A, F = P01 / "tables", P01 / "artifacts", P01 / "figures"
FOUNDATION = REPO / "assets" / "foundation"
CANONICAL_ATLAS_FINGERPRINT = "09ed804a40836f4a05a91ba10900cded"

from gaira.v7.atlas_decomposition import clustering as CL          # noqa: E402
from gaira.v7.atlas_decomposition import discovery as DIS          # noqa: E402
from gaira.v7.atlas_decomposition import matching as MATCH         # noqa: E402
from gaira.v7.atlas_decomposition import serialization as SER      # noqa: E402
from gaira.v7.atlas_decomposition import validation as VAL         # noqa: E402
from gaira.v7.atlas_decomposition.motif import ACS, Band, build_motif_spectrum   # noqa: E402
from gaira.v7.atlas_decomposition.registry import ACSRegistry      # noqa: E402

ran = pytest.mark.skipif(not (A / "acs_manifest_v1.json").is_file(),
                         reason="the control experiment has not been run in this checkout")


# ── synthetic fixtures ────────────────────────────────────────────────────────
def _toy_component(n_bins: int = 676) -> tuple[np.ndarray, np.ndarray]:
    """A component with four clean, well-separated bands."""
    grid = np.linspace(450.0, 1800.0, n_bins)
    h = np.zeros(n_bins)
    for centre, amp in ((120, 1.0), (260, 0.8), (410, 0.9), (560, 0.7)):
        h += amp * np.exp(-0.5 * ((np.arange(n_bins) - centre) / 3.0) ** 2)
    return h, grid


def _toy_analytes(h: np.ndarray, bands: list[Band]) -> np.ndarray:
    """Two groups: one carrying bands 0+1, one carrying bands 2+3. Noise-free."""
    rows = []
    for grp in (0, 1):
        for _ in range(6):
            x = np.zeros_like(h)
            for b in (bands[0], bands[1]) if grp == 0 else (bands[2], bands[3]):
                x[b.slice()] += h[b.slice()]
            rows.append(x)
    return np.vstack(rows)


@pytest.fixture(scope="module")
def toy():
    h, grid = _toy_component()
    bands = DIS.component_bands(h, grid)
    Xa = _toy_analytes(h, bands)
    return h, grid, bands, Xa


# ── A. motif object invariants ────────────────────────────────────────────────
def test_component_bands_are_deterministic_and_ordered(toy):
    h, grid, bands, _ = toy
    again = DIS.component_bands(h, grid)
    assert [b.index for b in bands] == [b.index for b in again]
    assert [b.index for b in bands] == sorted(b.index for b in bands)
    assert len(bands) == 4
    for b in bands:
        assert b.lo_bin <= b.index <= b.hi_bin
        assert b.component_weight > 0


def test_motif_spectrum_is_a_masked_restriction_of_its_parent(toy):
    h, _, bands, _ = toy
    m = build_motif_spectrum(h, bands, [0, 1], np.array([0.5, 0.5]))
    assert m.shape == h.shape
    assert np.all(m >= 0), "non-negativity is an architectural invariant"
    assert np.all(m <= h + 1e-12), "a motif can never exceed its parent"
    outside = np.ones_like(h, bool)
    for b in (bands[0], bands[1]):
        outside[b.slice()] = False
    assert np.allclose(m[outside], 0.0), "a motif must be exactly zero outside its bands"


def test_motif_spectrum_with_no_bands_is_zero(toy):
    h, _, bands, _ = toy
    assert np.allclose(build_motif_spectrum(h, bands, [], np.array([])), 0.0)


def test_lsm_validate_catches_violations(toy):
    h, _, bands, _ = toy
    good = ACS(motif_id="c00.m00", parent_component=0, index_in_component=0,
               spectrum=build_motif_spectrum(h, bands, [0, 1], np.array([.5, .5])),
               band_indices=[bands[0].index, bands[1].index],
               band_centers_cm=[bands[0].center_cm, bands[1].center_cm],
               band_weights=[.5, .5], analytes=["a", "b", "c"], n_analytes=3, n_spectra=3,
               fine_classes={"x": 3}, broad_classes={"X": 3}, sources={"s": 3},
               stability=1.0, purity=1.0, coverage_analytes=.5, coverage_spectra=.5,
               dominant_class="x", band_fidelity=1.0, redundancy_max=0.0)
    assert good.validate() == []

    bad = ACS(**{**good.__dict__, "spectrum": -good.spectrum})
    assert any("negative" in v for v in bad.validate())

    bad = ACS(**{**good.__dict__, "n_analytes": 99})
    assert any("n_analytes" in v for v in bad.validate())

    bad = ACS(**{**good.__dict__, "retained": False})
    assert any("rejection reason" in v for v in bad.validate())


# ── B. clustering determinism ─────────────────────────────────────────────────
def test_select_cut_is_deterministic(toy):
    h, grid, bands, Xa = toy
    Q = DIS.band_profiles(Xa, np.arange(len(Xa)), bands)
    a = CL.select_cut(Q)
    b = CL.select_cut(Q)
    assert a["n_motifs"] == b["n_motifs"]
    assert list(a["labels"]) == list(b["labels"])
    assert a["silhouette"] == b["silhouette"]


def test_select_cut_recovers_a_planted_two_group_structure(toy):
    h, grid, bands, Xa = toy
    Q = DIS.band_profiles(Xa, np.arange(len(Xa)), bands)
    sel = CL.select_cut(Q)
    assert sel["n_motifs"] == 2, "two clean, disjoint band groups must give two motifs"
    lab = np.asarray(sel["labels"])
    assert len(set(lab[:6])) == 1 and len(set(lab[6:])) == 1
    assert lab[0] != lab[6]


def test_select_cut_allows_one_motif_when_there_is_no_structure():
    """A component whose analytes all show the same profile must not be forced to split."""
    h, grid = _toy_component()
    bands = DIS.component_bands(h, grid)
    Xa = np.vstack([h for _ in range(8)])
    Q = DIS.band_profiles(Xa, np.arange(8), bands)
    sel = CL.select_cut(Q)
    lab = np.asarray(sel["labels"])
    biggest = max(np.bincount(lab)[1:]) if lab.max() > 0 else len(lab)
    assert biggest >= 7, "identical profiles must not be split into balanced motifs"


def test_jackknife_stability_is_deterministic_and_bounded(toy):
    h, grid, bands, Xa = toy
    Q = DIS.band_profiles(Xa, np.arange(len(Xa)), bands)
    sel = CL.select_cut(Q)
    s1 = CL.jackknife_stability(Q, sel["labels"])
    s2 = CL.jackknife_stability(Q, sel["labels"])
    assert np.array_equal(s1, s2)
    assert np.all((s1 >= 0) & (s1 <= 1))
    assert np.all(s1 > 0.9), "cleanly planted groups must be highly stable"


def test_size_gini_bounds():
    assert CL.size_gini([5, 5, 5]) == pytest.approx(0.0, abs=1e-9)
    assert CL.size_gini([1, 1, 100]) > 0.5
    assert CL.size_gini([]) == 0.0


def test_linkage_rule_prefers_balance_within_tolerance():
    rows = [{"linkage": "average", "mean_silhouette": 0.455, "mean_size_gini": 0.427},
            {"linkage": "ward", "mean_silhouette": 0.426, "mean_size_gini": 0.293},
            {"linkage": "complete", "mean_silhouette": 0.447, "mean_size_gini": 0.358}]
    assert CL.apply_linkage_rule(rows)["selected_linkage"] == "ward"
    # outside the tolerance, the balanced option is no longer admissible
    rows[1]["mean_silhouette"] = 0.20
    assert CL.apply_linkage_rule(rows)["selected_linkage"] == "complete"


def test_no_rng_on_the_discovery_path():
    """Discovery must not seed or draw from any RNG — determinism by construction."""
    import inspect
    for mod in (DIS, CL):
        src = inspect.getsource(mod)
        for banned in ("np.random", "random.", "default_rng", "RandomState"):
            assert banned not in src, f"{mod.__name__} touches an RNG ({banned})"


# ── C. rejection ──────────────────────────────────────────────────────────────
def _mk(mid, n_analytes, stability=1.0, n_bands=3, spectrum=None):
    sp = np.zeros(676) if spectrum is None else spectrum
    return ACS(motif_id=mid, parent_component=0, index_in_component=int(mid.split("m")[1]),
               spectrum=sp, band_indices=list(range(n_bands)),
               band_centers_cm=[500.0 + 10 * i for i in range(n_bands)],
               band_weights=[1.0 / n_bands] * n_bands,
               analytes=[f"a{i}" for i in range(n_analytes)], n_analytes=n_analytes,
               n_spectra=n_analytes, fine_classes={"x": n_analytes},
               broad_classes={"X": n_analytes}, sources={"s": n_analytes},
               stability=stability, purity=1.0, coverage_analytes=.3, coverage_spectra=.3,
               dominant_class="x", band_fidelity=1.0, redundancy_max=0.0)


def test_rejection_reasons_are_deterministic_and_specific():
    small, unstable, thin = _mk("c00.m00", 2), _mk("c00.m01", 5, stability=0.1), \
        _mk("c00.m02", 5, n_bands=1)
    ms = [small, unstable, thin, _mk("c00.m03", 6)]
    DIS._reject(ms)
    assert not small.retained and "too_few_analytes" in small.rejection_reason
    assert not unstable.retained and "low_stability" in unstable.rejection_reason
    assert not thin.retained and "noise_single_band" in thin.rejection_reason
    assert ms[3].retained and ms[3].rejection_reason == ""


def test_redundant_motifs_are_rejected_keeping_the_better_supported_one():
    sp = np.zeros(676)
    sp[100:110] = 1.0
    a, b = _mk("c00.m00", 4, spectrum=sp.copy()), _mk("c00.m01", 9, spectrum=sp.copy())
    ms = [a, b]
    DIS._score_redundancy(ms)
    DIS._reject(ms)
    assert b.retained, "the motif with more participating molecules must survive"
    assert not a.retained and "redundant" in a.rejection_reason


# ── D. matching conserves atlas evidence ──────────────────────────────────────
def test_attribution_conserves_the_component_activation(toy):
    h, grid, bands, Xa = toy
    ms = [_mk("c00.m00", 4), _mk("c00.m01", 4)]
    for m, bs in zip(ms, ([0, 1], [2, 3])):
        m.band_indices = [bands[i].index for i in bs]
        m.band_weights = [0.5, 0.5]
    got = MATCH.attribute_component(Xa[0], 3.0, ms)
    assert sum(got.values()) == pytest.approx(3.0, abs=1e-12)


def test_attribution_never_loses_evidence_when_no_motif_matches():
    m = _mk("c00.m00", 4)
    m.band_indices = [10]
    m.band_weights = [1.0]
    got = MATCH.attribute_component(np.zeros(676), 2.5, [m])
    assert got == {MATCH.UNATTRIBUTED: 2.5}
    got = MATCH.attribute_component(np.zeros(676), 2.5, [])
    assert got == {MATCH.UNATTRIBUTED: 2.5}


def test_attribution_is_batch_independent(toy):
    h, grid, bands, Xa = toy
    ms = [_mk("c00.m00", 4), _mk("c00.m01", 4)]
    for m, bs in zip(ms, ([0, 1], [2, 3])):
        m.band_indices = [bands[i].index for i in bs]
        m.band_weights = [0.5, 0.5]
    reg = ACSRegistry([{"component": 0, "status": "DECOMPOSED", "n_bands": 4,
                        "n_participants": 12, "motifs": ms}], "fp", "v", {})
    W = np.zeros((len(Xa), 24))
    W[:, 0] = 1.0
    A_all, ids = MATCH.attribution_matrix(Xa, W, reg)
    A_one, _ = MATCH.attribution_matrix(Xa[:1], W[:1], reg)
    assert np.allclose(A_all[0], A_one[0]), "a spectrum's output must not depend on its batch"
    assert MATCH.conservation_error(A_all, W) < 1e-12


# ── E. registry and serialisation ─────────────────────────────────────────────
def test_registry_integrity_catches_a_broken_parent_link():
    ms = [_mk("c00.m00", 5), _mk("c00.m01", 5)]
    reg = ACSRegistry([{"component": 0, "status": "DECOMPOSED", "n_bands": 4,
                        "n_participants": 10, "motifs": ms}], "fp", "v", {})
    assert reg.check_integrity() == []
    ms[0].parent_component = 7
    assert any("parent" in v or "encode" in v for v in reg.check_integrity())


def test_registry_status_must_match_retained_count():
    ms = [_mk("c00.m00", 5)]
    reg = ACSRegistry([{"component": 0, "status": "DECOMPOSED", "n_bands": 4,
                        "n_participants": 10, "motifs": ms}], "fp", "v", {})
    assert any("DECOMPOSED with 1" in v for v in reg.check_integrity())


def test_registry_keeps_rejected_motifs_queryable():
    keep, drop = _mk("c00.m00", 5), _mk("c00.m01", 1)
    DIS._reject([keep, drop])
    reg = ACSRegistry([{"component": 0, "status": "IRREDUCIBLE", "n_bands": 4,
                        "n_participants": 6, "motifs": [keep, drop]}], "fp", "v", {})
    assert len(reg.retained) == 1 and len(reg.rejected) == 1
    assert len(reg.motif_table()) == 2, "rejected motifs stay in the registry"
    assert reg.rejection_table().iloc[0].rejection_reason


def test_serialization_round_trips(tmp_path):
    sp = np.zeros(676)
    sp[100:110] = np.linspace(0.1, 1.0, 10)
    a = _mk("c00.m00", 5, spectrum=sp)
    a.band_indices, a.band_centers_cm, a.band_weights = [104], [650.0], [1.0]
    reg = ACSRegistry([{"component": 0, "status": "DECOMPOSED", "n_bands": 2,
                        "n_participants": 5, "motifs": [a, _mk("c00.m01", 5)]}],
                      CANONICAL_ATLAS_FINGERPRINT, "v7_lsm_v1", {"linkage": "ward"})
    man = SER.save_registry(reg, tmp_path)
    df, spectra, ids, man2 = SER.load_registry(tmp_path)
    assert man2["registry_fingerprint"] == man["registry_fingerprint"]
    back = SER.motifs_from_table(df, spectra, ids)
    assert len(back) == 2
    got = next(m for m in back if m.motif_id == "c00.m00")
    assert np.allclose(got.spectrum, a.spectrum)
    assert got.analytes == a.analytes
    assert got.band_centers_cm == a.band_centers_cm


def test_registry_fingerprint_changes_with_content(tmp_path):
    a = _mk("c00.m00", 5)
    reg = ACSRegistry([{"component": 0, "status": "DECOMPOSED", "n_bands": 2,
                        "n_participants": 5, "motifs": [a, _mk("c00.m01", 5)]}],
                      "fp", "v", {})
    f1 = SER.registry_fingerprint(reg)
    a.spectrum = a.spectrum + 1.0
    assert SER.registry_fingerprint(reg) != f1


# ── F. frozen-artefact contract ───────────────────────────────────────────────
def test_atlas_fingerprint_unchanged():
    """Phase 01 is an interpretation layer. Altering the atlas would void its premise."""
    H = np.asarray(np.load(FOUNDATION / "manifold_components.npz")["components"], float)
    fp = hashlib.sha256(np.ascontiguousarray(H).tobytes()).hexdigest()[:32]
    assert fp == CANONICAL_ATLAS_FINGERPRINT, f"FROZEN ATLAS CHANGED: {fp}"
    assert H.shape == (24, 676)


def test_frozen_foundation_files_unchanged():
    man = json.loads((FOUNDATION / "MANIFEST.json").read_text())
    bad = [n for n, rec in man["files"].items()
           if hashlib.sha256((FOUNDATION / n).read_bytes()).hexdigest() != rec["sha256"]]
    assert not bad, f"frozen assets modified: {bad}"


@ran
def test_phase00_artifacts_untouched():
    """Phase 00 is frozen; Phase 01 must read it and write nothing into it."""
    p00 = json.loads((REPO / "results/v7_rebuild/phase00/manifests/"
                      "phase_00_manifest_v1.json").read_text())
    stale = [o["artifact_id"] for o in p00["outputs"]
             if (REPO / o["path"]).is_file()
             and hashlib.sha256((REPO / o["path"]).read_bytes()).hexdigest() != o["sha256"]]
    assert not stale, f"Phase 00 artefacts modified by the control experiment: {stale}"


@ran
def test_control_writes_only_inside_its_own_tree():
    man = json.loads((A / "control_manifest_v1.json").read_text())
    stray = [o["path"] for o in man["outputs"]
             if not o["path"].startswith("results/v7_rebuild/control_experiments/")]
    assert not stray, f"the control experiment wrote outside its tree: {stray}"


@ran
def test_manifest_records_the_atlas_unchanged_before_and_after():
    man = json.loads((A / "control_manifest_v1.json").read_text())
    assert man["atlas_fingerprint_before"] == CANONICAL_ATLAS_FINGERPRINT
    assert man["atlas_fingerprint_after"] == CANONICAL_ATLAS_FINGERPRINT


@ran
def test_discovery_was_deterministic():
    det = json.loads((A / "determinism_v1.json").read_text())
    assert det["identical"] is True
    assert len(set(det["signatures"])) == 1
    assert det["n_runs"] >= 3


@ran
def test_committed_registry_is_intact():
    df = pd.read_csv(T / "acs_registry_v1.csv")
    assert df.motif_id.is_unique
    assert len(df) > 0
    for _, r in df.iterrows():
        assert r.motif_id.startswith(f"c{int(r.parent_component):02d}.")
    kept = df[df.retained]
    assert kept.rejection_reason.isna().all() or (kept.rejection_reason == "").all()
    dropped = df[~df.retained]
    assert dropped.rejection_reason.astype(str).str.len().gt(3).all()


@ran
def test_committed_motif_spectra_are_non_negative_and_gridded():
    z = np.load(A / "acs_spectra_v1.npz", allow_pickle=True)
    S = np.asarray(z["spectra"], float)
    assert S.shape[1] == 676
    assert np.all(S >= 0), "non-negativity is an architectural invariant"
    assert np.all(np.isfinite(S))
    assert S.shape[0] == len(z["motif_ids"])


@ran
def test_every_motif_is_bounded_by_its_parent_component():
    """A motif is a restriction of its parent: it can never exceed it anywhere."""
    H = np.asarray(np.load(FOUNDATION / "manifold_components.npz")["components"], float)
    df, S, ids, _ = SER.load_registry(A)
    kept = df[df.retained].set_index("motif_id")
    for j, mid in enumerate(ids):
        k = int(kept.loc[mid, "parent_component"])
        assert np.all(S[j] <= H[k] + 1e-9), f"{mid} exceeds parent component c{k:02d}"


@ran
def test_component_linkage_is_complete():
    comp = pd.read_csv(T / "acs_components_v1.csv")
    reg = pd.read_csv(T / "acs_registry_v1.csv")
    assert len(comp) == 24, "every atlas component must be accounted for"
    assert set(reg.parent_component) <= set(comp.component)
    for _, r in comp.iterrows():
        n = int((reg[reg.parent_component == r.component].retained).sum())
        assert n == r.n_retained_motifs
        if r.status == "DECOMPOSED":
            assert n >= 2
        if r.status == "IRREDUCIBLE":
            assert n < 2


@ran
def test_attribution_conserves_evidence_on_the_real_corpus():
    cons = json.loads((A / "attribution_conservation_v1.json").read_text())
    assert cons["max_conservation_error"] < 1e-9, (
        "the motif layer must redistribute atlas activation, never create or destroy it")


@ran
def test_scientific_benefit_is_measured_against_a_size_matched_null():
    """A raw purity gain is not evidence — more clusters raise purity mechanically."""
    pn = pd.read_csv(T / "purity_null_v1.csv")
    assert len(pn) > 0
    assert {"gain_beyond_mechanical", "null_purity_mean", "p_permutation"} <= set(pn.columns)
    assert int(pn.significant.sum()) >= 1
    assert pn.gain_beyond_mechanical.median() > 0


@ran
def test_chemical_alignment_has_a_permutation_null():
    al = pd.read_csv(T / "chemical_alignment_v1.csv")
    sig = al[al.significant]
    assert len(sig) >= 1
    assert (sig.ami_fine > sig.null_ami_p95).all(), (
        "a significant component must beat its own null 95th percentile")


@ran
def test_reproducibility_was_measured_across_sources_and_replicates():
    rp = pd.read_csv(T / "reproducibility_v1.csv")
    assert {"RamanBioLib_only", "replicate_half_0"} <= set(rp.subset)
    assert rp.ari.notna().sum() > 0


@ran
def test_method_comparisons_are_published():
    """Selection rules were pre-registered; the comparisons must be committed either way."""
    for name in ("profile_mode_comparison_v1.csv", "linkage_comparison_v1.csv",
                 "motif_construction_comparison_v1.csv"):
        df = pd.read_csv(T / name)
        assert len(df) >= 2, f"{name} must compare at least two candidates"


@ran
def test_all_control_gates_pass():
    state = json.loads((P01 / "CONTROL_STATE.json").read_text())
    failed = [g["gate"] for g in state["gates"] if not g["passed"]]
    assert not failed, f"control-experiment gates failed: {failed}"
    assert state["status"] == "COMPLETE"
    assert state["atlas_unchanged"] is True
    assert state["classification"].startswith("CONTROL EXPERIMENT")


@ran
@pytest.mark.parametrize("stem", [
    "fig01_component_motif_tree", "fig02_motif_spectra", "fig03_motif_overlap_graph",
    "fig04_motif_quality", "fig05_ambiguity_resolution", "fig06_coverage",
    "fig07_participation_heatmap", "fig08_representative_motifs", "fig09_motif_hierarchy",
])
def test_figures_exist_in_vector_and_raster(stem):
    assert (F / f"{stem}.svg").is_file()
    assert (F / f"{stem}.png").is_file()


@ran
def test_no_later_phase_work_leaked_in():
    """Phase 01 is Strategy A only: no consensus motifs, no themes, no BSV, no balancing."""
    forbidden = ("csm_", "consensus_", "theme_", "bsv_", "balanced_reference")
    hits = [p.name for p in list(T.glob("*")) + list(A.glob("*"))
            if any(f in p.name.lower() for f in forbidden)]
    assert not hits, f"later-phase artefacts present in Phase 01: {hits}"
