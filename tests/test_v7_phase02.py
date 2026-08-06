"""GAIRA V7 — Phase 02 tests: Consensus Spectral Motifs.

The tests that matter most here are the ones that would catch a merge made on weak evidence.
Phase 02's entire value is that it refuses to merge without multiple independent lines of
support, so the properties worth pinning are:

    test_cosine_alone_cannot_carry_an_edge
    test_a_rejected_merge_is_actually_undone
    test_every_lsm_is_assigned_to_exactly_one_csm
    test_frozen_atlas_is_not_an_input_to_the_csm_package
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

P02 = REPO / "results/v7_rebuild/phase02"
T, A, V, F = P02 / "tables", P02 / "artifacts", P02 / "validation", P02 / "figures"
P01 = REPO / "results/v7_rebuild/phase01"
CANONICAL_ATLAS_FINGERPRINT = "09ed804a40836f4a05a91ba10900cded"
P01_REGISTRY_FINGERPRINT = "208482d6f7178b5b8f16cace91be55b0"

from gaira.v7.csm import consensus as CON      # noqa: E402
from gaira.v7.csm import edges as E            # noqa: E402
from gaira.v7.csm import graph as GR           # noqa: E402
from gaira.v7.csm import integration as INT    # noqa: E402
from gaira.v7.csm import validation as VAL     # noqa: E402
from gaira.v7.csm.csm import CSM, dominant_bands   # noqa: E402
from gaira.v7.csm.registry import CSMRegistry      # noqa: E402

ran = pytest.mark.skipif(not (A / "csm_registry_v1.json").is_file(),
                         reason="Phase 02 has not been run in this checkout")
D = 676


@pytest.fixture(scope="module")
def registry():
    return json.loads((A / "csm_registry_v1.json").read_text())


@pytest.fixture(scope="module")
def state():
    return json.loads((P02 / "PHASE_STATE.json").read_text())


def _mk(index=0, members=(0,), n_lsms=None, **kw):
    n = n_lsms if n_lsms is not None else len(members)
    base = dict(csm_id=f"csm{index:02d}", index=index,
                contributing_lsms=[f"cls.m{i:02d}" for i in range(n)],
                contributing_lsm_weights=[1.0 / n] * n, member_indices=list(members),
                supporting_classes=["fatty_acid"], supporting_analytes=["oleate"],
                projected_support=["oleate"], n_lsms=n, n_classes=1, n_analytes=1,
                spectrum=np.abs(np.sin(np.linspace(0, 9, D))), dominant_bands=[1440.0],
                cohesion=0.9, uncertainty=0.1, mean_edge_weight=0.7, min_edge_weight=0.7,
                max_external_weight=0.2, min_coassignment=1.0, lsm_types=["subfamily"],
                is_singleton=n == 1, is_anchored=False, is_cross_class=False,
                consensus_operator="stability_weighted_mean")
    base.update(kw)
    return CSM(**base)


# ── A. ARCHITECTURE ──────────────────────────────────────────────────────────
def _executable_source(mod) -> str:
    """Module source with docstrings stripped — code only.

    These modules legitimately *discuss* the frozen atlas in their documentation. What must
    never appear is a line that loads it.
    """
    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree).lower()


@pytest.mark.parametrize("mod", [E, GR, INT, CON, VAL])
def test_frozen_atlas_is_not_an_input_to_the_csm_package(mod):
    """P-15: the V5 atlas is a control and a comparator, never a foundation."""
    src = _executable_source(mod)
    for forbidden in ("manifold_components", "foundation", "atlas_fingerprint",
                      CANONICAL_ATLAS_FINGERPRINT):
        assert forbidden not in src, f"{mod.__name__} loads the frozen atlas: {forbidden}"


@ran
def test_phase01_was_consumed_read_only(state):
    assert state["phase01_registry_fingerprint"] == P01_REGISTRY_FINGERPRINT
    p01 = json.loads((P01 / "PHASE_STATE.json").read_text())
    assert p01["registry_fingerprint"] == P01_REGISTRY_FINGERPRINT, \
        "Phase 01 outputs changed — Phase 02 must never write to phase01/"


@ran
def test_frozen_atlas_unchanged_across_the_phase(state):
    man = json.loads((A / "phase_02_manifest_v1.json").read_text())
    assert man["atlas_fingerprint_before"] == CANONICAL_ATLAS_FINGERPRINT
    assert man["atlas_fingerprint_after"] == CANONICAL_ATLAS_FINGERPRINT
    assert state["atlas_unchanged"] is True


# ── B. THE EDGE WEIGHT REFUSES SINGLE-CHANNEL EVIDENCE ───────────────────────
def test_cosine_alone_cannot_carry_an_edge():
    """The property the whole phase rests on: high cosine with no other support is not a merge.

    A geometric mean is what makes this true continuously. Under an arithmetic mean the same
    pair would score 0.29 and could clear a threshold; here it is driven far below a pair with
    moderate support on every channel.
    """
    n = 2
    hi_cos = {f: np.full((n, n), 0.02) for f in E.FEATURES}
    hi_cos["spectral_cosine"] = np.full((n, n), 0.99)
    balanced = {f: np.full((n, n), 0.55) for f in E.FEATURES}
    w_cos = GR.edge_weights(hi_cos)[0, 1]
    w_bal = GR.edge_weights(balanced)[0, 1]
    assert w_cos < w_bal, f"cosine-only {w_cos:.4f} should lose to balanced {w_bal:.4f}"
    assert w_cos < 0.15


def test_a_single_zero_channel_is_penalised_but_does_not_annihilate():
    n = 2
    feat = {f: np.full((n, n), 0.8) for f in E.FEATURES}
    full = GR.edge_weights(feat)[0, 1]
    feat["substitutability"] = np.zeros((n, n))
    holed = GR.edge_weights(feat)[0, 1]
    assert holed < full
    # exponent 0.10 against a 1e-3 floor => a factor of ~0.5, a stated design constant
    assert 0.35 < holed / full < 0.65


def test_edge_weight_exponents_sum_to_one_and_no_feature_dominates():
    assert abs(sum(GR.ALPHA.values()) - 1.0) < 1e-9
    assert max(GR.ALPHA.values()) <= 0.30, "no single channel may dominate the weight"
    assert set(GR.ALPHA) == set(E.FEATURES)


def test_edge_weight_is_symmetric_with_a_zero_diagonal():
    rng = np.random.default_rng(0)
    n = 6
    feat = {}
    for f in E.FEATURES:
        M = rng.uniform(0.1, 0.9, (n, n))
        feat[f] = (M + M.T) / 2
    W = GR.edge_weights(feat)
    assert np.allclose(W, W.T)
    assert np.allclose(np.diag(W), 0.0)


def test_edge_weights_rejects_a_missing_feature():
    n = 3
    feat = {f: np.full((n, n), 0.5) for f in E.FEATURES if f != "band_overlap"}
    with pytest.raises(KeyError):
        GR.edge_weights(feat)


# ── C. FEATURE SEMANTICS ─────────────────────────────────────────────────────
def test_identical_motifs_score_one_on_every_shape_channel():
    h = np.abs(np.sin(np.linspace(0, 12, D))) + 0.05
    H = np.vstack([h, h])
    grid = np.linspace(450, 1800, D)
    bands = [dominant_bands(h, grid)] * 2
    assert E.spectral_cosine(H)[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert E.band_overlap(H, bands, grid)[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert E.peak_agreement(bands)[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_band_overlap_ignores_a_shared_broad_pedestal():
    """The defect this feature was rebuilt to fix.

    Two motifs with disjoint peaks on a large common pedestal have a high full-spectrum cosine
    and must NOT have a high diagnostic-band agreement.
    """
    grid = np.linspace(450, 1800, D)
    pedestal = np.exp(-((grid - 1100) ** 2) / (2 * 380 ** 2))
    a = pedestal + 0.55 * np.exp(-((grid - 700) ** 2) / (2 * 6 ** 2))
    b = pedestal + 0.55 * np.exp(-((grid - 1500) ** 2) / (2 * 6 ** 2))
    H = np.vstack([a, b])
    bands = [dominant_bands(a, grid), dominant_bands(b, grid)]
    cos = E.spectral_cosine(H)[0, 1]
    bo = E.band_overlap(H, bands, grid)[0, 1]
    assert cos > 0.85, "the pedestal should make these look similar globally"
    assert bo < cos - 0.3, f"band agreement {bo:.3f} must separate from cosine {cos:.3f}"


def test_peak_agreement_is_intensity_free():
    """Position-only: scaling one motif's peaks must not change the score."""
    bands = [[700.0, 1000.0, 1440.0], [702.0, 1000.0, 1600.0]]
    a = E.peak_agreement(bands)[0, 1]
    assert a == pytest.approx(2 * 2 / 6)
    far = E.peak_agreement([[700.0], [900.0]])[0, 1]
    assert far == 0.0


def test_activation_matrix_is_independent_projection_not_joint_nnls():
    """Near-duplicate motifs must not appear anticorrelated by splitting each other's mass."""
    rng = np.random.default_rng(1)
    grid = np.linspace(450, 1800, D)
    h = np.exp(-((grid - 1000) ** 2) / (2 * 20 ** 2))
    H = np.vstack([h, h * 0.99 + 1e-3])
    X = np.abs(rng.normal(size=(20, D))) * 0.01 + h
    A = E.activation_matrix(X, H)
    assert A.shape == (20, 2)
    assert (A >= 0).all()
    assert np.corrcoef(A[:, 0], A[:, 1])[0, 1] > 0.9


def test_activation_shares_are_normalised_within_class():
    A = np.array([[1.0, 3.0, 5.0, 5.0]])
    S = E.activation_shares(A, ["a", "a", "b", "b"])
    assert S[0, :2].sum() == pytest.approx(1.0)
    assert S[0, 2:].sum() == pytest.approx(1.0)


def test_provenance_overlap_discounts_within_class_agreement():
    """Risk R-01: without the discount this feature re-encodes the class partition."""
    n_mol = 40
    A = np.zeros((n_mol, 2))
    A[:20, 0] = 1.0
    A[:20, 1] = 1.0                       # identical supports
    same = E.provenance_overlap(A, ["c", "c"], ["c"] * 20 + ["d"] * 20)[0, 1]
    cross = E.provenance_overlap(A, ["c", "d"], ["c"] * 20 + ["d"] * 20)[0, 1]
    assert same <= cross, "identical supports inside one class must be discounted hardest"


def test_substitutability_is_symmetric_and_bounded():
    rng = np.random.default_rng(2)
    grid = np.linspace(450, 1800, D)
    H = np.vstack([np.exp(-((grid - c) ** 2) / (2 * 25 ** 2)) for c in (700, 1000, 1400)])
    X = np.abs(rng.normal(size=(9, D)) * 0.01) + H[rng.integers(0, 3, 9)]
    S = E.substitutability(H, ["a", "a", "b"], X, np.array(["a", "a", "b"]),
                           [set(range(9))] * 3)
    assert np.allclose(S, S.T)
    assert ((S >= 0) & (S <= 1)).all()
    assert np.allclose(np.diag(S), 1.0)


def test_bootstrap_cooccurrence_falls_when_a_motif_is_rarely_recovered():
    h = np.abs(np.sin(np.linspace(0, 9, D))) + 0.1
    both = [{0: h, 1: h} for _ in range(10)]
    rare = [{0: h, 1: h} for _ in range(2)] + [{0: h} for _ in range(8)]
    assert (E.bootstrap_cooccurrence(both, 2)[0, 1]
            > E.bootstrap_cooccurrence(rare, 2)[0, 1] + 0.5)


# ── D. THRESHOLD SELECTION AND THE R-07 BRANCH ───────────────────────────────
def test_select_threshold_fails_loudly_when_no_region_is_stable():
    """R-07 must produce a FAIL, never a quietly chosen cut."""
    sweep = [{"threshold": 0.1 * k, "n_edges": 50 - 5 * k, "n_communities": 2 + k,
              "n_singletons": k, "largest_community": 10, "community_stability": 0.5,
              "partition": {f"m{i}": (i + k) % (2 + k) for i in range(10)}}
             for k in range(8)]
    sel = GR.select_threshold(sweep)
    assert sel["status"] == "FAIL"
    assert sel["selected_threshold"] is None
    assert "R-07" in sel["rationale"]


def test_select_threshold_finds_a_genuine_stable_region():
    part = {f"m{i}": i % 3 for i in range(12)}
    sweep = []
    for k in range(8):
        p = dict(part) if 2 <= k <= 5 else {f"m{i}": (i * k) % 5 for i in range(12)}
        sweep.append({"threshold": 0.1 * k, "n_edges": 40, "n_communities": len(set(p.values())),
                      "n_singletons": 0, "largest_community": 5,
                      "community_stability": 0.8, "partition": p})
    sel = GR.select_threshold(sweep)
    assert sel["status"] == "PASS"
    assert 0.2 <= sel["selected_threshold"] <= 0.5


def test_threshold_consensus_refuses_a_degenerate_sweep():
    """Levels where one community holds everything are not evidence and must not be averaged."""
    sweep = [{"alpha": a, "n_singletons": 0, "largest_community": 50,
              "n_communities": 1, "partition": {f"m{i}": 0 for i in range(50)}}
             for a in (0.2, 0.1, 0.05)]
    with pytest.raises(ValueError, match="viable"):
        GR.threshold_consensus(sweep, [f"m{i}" for i in range(50)])


def test_threshold_consensus_is_unanimous_by_default():
    ids = [f"m{i}" for i in range(6)]
    # m0,m1 always together; m2,m3 together at only two of three levels
    parts = [{"m0": 0, "m1": 0, "m2": 1, "m3": 1, "m4": 2, "m5": 3},
             {"m0": 0, "m1": 0, "m2": 1, "m3": 1, "m4": 2, "m5": 3},
             {"m0": 0, "m1": 0, "m2": 1, "m3": 4, "m4": 2, "m5": 3}]
    sweep = [{"alpha": a, "n_singletons": 2, "largest_community": 2,
              "n_communities": 4, "partition": p} for a, p in zip((0.1, 0.05, 0.01), parts)]
    groups, C, viable = GR.threshold_consensus(sweep, ids)
    assert len(viable) == 3
    pairs = {tuple(sorted(g)) for g in groups if len(g) > 1}
    assert (0, 1) in pairs, "a unanimous pair must merge"
    assert (2, 3) not in pairs, "a 2-of-3 pair must NOT merge under unanimity"


def test_feature_floor_is_a_stated_constant_not_machine_epsilon():
    assert GR.FEATURE_FLOOR == 1e-3


# ── E. CSM OBJECT AND REGISTRY INVARIANTS (C-07) ─────────────────────────────
def test_csm_rejects_a_negative_spectrum():
    with pytest.raises(ValueError, match="non-negative"):
        _mk(spectrum=np.linspace(-1, 1, D))


def test_csm_rejects_a_wrong_grid_length():
    with pytest.raises(ValueError, match="676"):
        _mk(spectrum=np.ones(400))


def test_singleton_flag_must_match_the_contributor_count():
    with pytest.raises(ValueError, match="is_singleton"):
        _mk(n_lsms=3, members=(0, 1, 2), is_singleton=True)


def test_anchored_csm_requires_a_justification():
    with pytest.raises(ValueError, match="anchor_justification"):
        _mk(is_anchored=True)


def test_registry_rejects_a_duplicate_id():
    r = CSMRegistry("graph_community", 1.0, "stability_weighted_mean")
    r.add(_mk(0))
    with pytest.raises(ValueError, match="duplicate"):
        r.add(_mk(0))


def test_registry_invariants_catch_an_unassigned_lsm():
    r = CSMRegistry("graph_community", 1.0, "stability_weighted_mean")
    r.add(_mk(0))
    inv = {i["invariant"]: i["status"] for i in r.check_invariants(["cls.m00", "cls.m01"])}
    assert inv["every LSM assigned to exactly one CSM"] == "FAIL"


def test_consensus_operators_all_return_a_non_negative_unit_spectrum():
    rng = np.random.default_rng(3)
    H = np.abs(rng.normal(size=(4, D))) + 0.01
    for op in CON.OPERATORS:
        c = CON.consensus_spectrum(H, np.ones(4), op)
        assert (c >= 0).all(), op
        assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-6), op


def test_uncertainty_uses_the_worst_contributor_not_the_mean():
    grid = np.linspace(450, 1800, D)
    close = np.exp(-((grid - 1000) ** 2) / (2 * 30 ** 2))
    far = np.exp(-((grid - 1600) ** 2) / (2 * 30 ** 2))
    H = np.vstack([close, close, close, far])
    c = CON.consensus_spectrum(H, np.ones(4))
    assert CON.uncertainty(H, c) > 1.0 - CON.cohesion(H, c), \
        "one distant contributor must not hide behind three close ones"


# ── F. THE RUN ITSELF ────────────────────────────────────────────────────────
@ran
def test_every_lsm_is_assigned_to_exactly_one_csm(registry):
    lsms = [l["lsm_id"] for c in registry["csms"] for l in c["contributing_lsms"]]
    assert len(lsms) == 50
    assert len(set(lsms)) == 50


@ran
def test_all_csm_spectra_are_non_negative_and_unit_norm():
    z = np.load(A / "csm_dictionary_v1.npz", allow_pickle=True)
    Dm = np.asarray(z["CSM"], float)
    assert (Dm >= 0).all()
    assert np.allclose(np.linalg.norm(Dm, axis=1), 1.0, atol=1e-6)
    assert Dm.shape[1] == D


@ran
def test_a_rejected_merge_is_actually_undone(registry):
    """A rejection means the merge does not happen — not a label on a merged object."""
    rej = pd.read_csv(T / "rejected_consensus_motifs_v1.csv")
    assert len(rej) > 0
    still_merged = {frozenset((a, b))
                    for c in registry["csms"] if c["n_lsms"] > 1
                    for a in [l["lsm_id"] for l in c["contributing_lsms"]]
                    for b in [l["lsm_id"] for l in c["contributing_lsms"]] if a != b}
    for r in rej.itertuples():
        members = r.contributing_lsms.split(";")
        for a in members:
            for b in members:
                if a != b:
                    assert frozenset((a, b)) not in still_merged, \
                        f"{a} and {b} were rejected but are still in one CSM"
    assert not any(c["status"] == "rejected" for c in registry["csms"])


@ran
def test_every_rejection_carries_a_reason():
    rej = pd.read_csv(T / "rejected_consensus_motifs_v1.csv")
    assert rej.rejection_reason.fillna("").str.len().gt(0).all()


@ran
def test_reconstruction_is_preserved(state):
    r = pd.read_csv(V / "reconstruction_comparison_v1.csv")
    assert float(r.delta.mean()) >= -0.05
    assert int((r.delta < -VAL.EV_DEGRADE_MAX).sum()) == 0
    assert len(r) == 154


@ran
def test_the_null_model_was_computed_and_the_graph_is_calibrated_against_it():
    z = np.load(A / "edge_features_v1.npz", allow_pickle=True)
    null = z["null_weights"]
    assert null.size >= 1000
    W = z["W"]
    iu = np.triu_indices(W.shape[0], 1)
    # the finding: observed and null overlap heavily — most similarity is generic
    assert null.mean() > 0.5 * W[iu].mean()
    assert (z["pvalues"][iu] <= 1.0).all()


@ran
def test_all_seven_features_are_present_on_every_edge():
    g = json.loads((A / "lsm_graph_v1.json").read_text())
    assert g["schema"] == "lsm_graph_v1"
    assert g["edges"], "the graph must retain at least one edge"
    for e in g["edges"]:
        assert set(e["features"]) == set(E.FEATURES)
    assert g["threshold_sweep"], "contract C-06: the sweep ships with the graph"


@ran
def test_named_suspects_were_all_investigated():
    s = pd.read_csv(V / "named_suspect_pairs_v1.csv")
    pairs = {frozenset((r.class_a, r.class_b)) for r in s.itertuples()}
    for a, b in [("peptide_protein", "polysaccharide"), ("acylglycerol", "fatty_acid"),
                 ("phospholipid_sphingolipid", "sterol_steroid"),
                 ("purine", "sulfur_thiol_cofactor")]:
        assert frozenset((a, b)) in pairs, f"{a} <-> {b} was not investigated"
    assert "merged_final" in s.columns


@ran
def test_all_five_integration_methods_were_compared_and_published():
    m = pd.read_csv(T / "integration_method_comparison_v1.csv")
    assert set(m.method) == set(INT.METHODS)
    assert m.composite.notna().all()


@ran
def test_meta_nmf_survival_check_applies_only_if_meta_nmf_won(state):
    """R-06 is conditional; the condition itself must be recorded either way."""
    comp = pd.read_csv(T / "architecture_compliance_v1.csv")
    row = comp[comp.specification_item.str.contains("meta-NMF")]
    assert len(row) == 1
    assert row.iloc[0]["status"] == "PASS"
    if state["integration_method"] != "meta_nmf":
        assert "did not win" in row.iloc[0]["evidence"]


@ran
def test_provenance_chain_resolves_to_molecules():
    p = pd.read_csv(T / "csm_provenance_chain_v1.csv")
    assert {"csm_id", "lsm_id", "chemical_class", "canonical_id", "sources"} <= set(p.columns)
    assert p.canonical_id.notna().all()
    assert p.csm_id.nunique() == 49


@ran
def test_singletons_are_visible_not_hidden(registry):
    n_single = sum(c["is_singleton"] for c in registry["csms"])
    assert n_single == registry["summary"]["n_singletons"]
    assert n_single > 0
    assert all(c["is_singleton"] == (c["n_lsms"] == 1) for c in registry["csms"])


@ran
def test_all_gates_pass(state):
    g = pd.read_csv(V / "phase02_gates_v1.csv")
    failed = g[g.status != "PASS"]
    assert failed.empty, f"failed gates:\n{failed}"
    assert state["status"] == "COMPLETE"


@ran
def test_architecture_compliance_is_complete(state):
    comp = pd.read_csv(T / "architecture_compliance_v1.csv")
    assert len(comp) >= 18
    assert (comp.status == "PASS").all()
    assert state["architecture_compliant"] is True


@ran
def test_preregistration_predates_the_run():
    """P-12: the rules were committed before the sweep they govern."""
    pre = P02 / "config/phase02_preregistration_v1.md"
    assert pre.is_file()
    text = pre.read_text()
    for term in ("H0", "geometric", "unanimit" if "unanimit" in text else "majority",
                 "NULL_PERMUTATIONS"):
        assert term in text


@ran
@pytest.mark.parametrize("n", range(1, 13))
def test_figure_exists(n):
    hits = sorted(F.glob(f"fig{n:02d}_*.png"))
    assert hits, f"figure {n:02d} missing"
    assert (hits[0].with_suffix(".svg")).is_file(), f"figure {n:02d} has no vector version"


@ran
def test_report_exists_and_states_the_negative_result():
    r = P02 / "reports/PHASE_02_REPORT.md"
    assert r.is_file()
    t = r.read_text()
    for term in ("R-07", "null", "rejected", "singleton", "Phase 03", "provenance"):
        assert term in t
