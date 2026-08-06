"""GAIRA V7 — Phase 05 regression tests: the canonical CSM inference engine.

Three kinds of test here, and the distinction matters:

* **contract tests** on the inference modules — properties that must hold for any corpus, checked
  on small synthetic inputs so a failure points at the code rather than at the data;
* **artifact tests** that the committed run produced what the report claims;
* **adversarial tests** that encode the specific defects found during the phase, so that a
  regression reintroducing one of them fails loudly. Those are marked in their docstrings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gaira.v7.io import PhaseOutputs, frozen_root
from gaira.v7.inference import (calibration as CAL, evidence as EV, openset as OS,
                                projection as PRJ, provenance as PROV, retrieval as RET)
from gaira.v7.inference.engine import CanonicalEngine

OUT = PhaseOutputs("05")
T, A, F, R = OUT.tables, OUT.artifacts, OUT.figures, OUT.reports
FROZEN = frozen_root()
HAS_RUN = (OUT.root / "PHASE_STATE.json").exists()
needs_run = pytest.mark.skipif(not HAS_RUN, reason="Phase 05 has not been run")


@pytest.fixture(scope="module")
def summary():
    return json.loads((A / "phase05_summary_v1.json").read_text())


@pytest.fixture(scope="module")
def toy():
    """A tiny dictionary with three well-separated Gaussian bands."""
    grid = np.arange(450.0, 1800.1, 2.0)
    centres = [700.0, 1100.0, 1650.0]
    H = np.vstack([np.exp(-0.5 * ((grid - c) / 12.0) ** 2) for c in centres])
    H /= np.linalg.norm(H, axis=1, keepdims=True)
    return grid, H


# ── Step 1: projection ───────────────────────────────────────────────────────
def test_projection_is_non_negative(toy):
    grid, H = toy
    X = np.abs(np.random.default_rng(0).normal(0, 1, (10, len(grid))))
    A_ = PRJ.project(X, H)
    assert (A_ >= 0).all()


def test_projection_recovers_a_known_mixture(toy):
    grid, H = toy
    w = np.array([0.5, 0.0, 0.3])
    a = PRJ.project((w @ H)[None, :], H)[0]
    assert np.allclose(a, w, atol=1e-6)


def test_projection_is_deterministic(toy):
    grid, H = toy
    X = np.abs(np.random.default_rng(1).normal(0, 1, (5, len(grid))))
    assert np.array_equal(PRJ.project(X, H), PRJ.project(X, H))


def test_explained_variance_is_one_for_an_exact_mixture(toy):
    grid, H = toy
    x = (np.array([1.0, 0.4, 0.0]) @ H)[None, :]
    d = PRJ.diagnostics(x, PRJ.project(x, H), H)
    assert d["explained_variance"][0] > 0.999


def test_sparsity_ranks_a_single_component_above_a_uniform_one(toy):
    grid, H = toy
    x1 = H[0:1]
    x2 = (np.ones(3) @ H)[None, :]
    d1 = PRJ.diagnostics(x1, PRJ.project(x1, H), H)
    d2 = PRJ.diagnostics(x2, PRJ.project(x2, H), H)
    assert d1["component_sparsity"][0] > d2["component_sparsity"][0]
    assert d1["activation_entropy"][0] < d2["activation_entropy"][0]


# ── Step 2: retrieval ────────────────────────────────────────────────────────
def test_reference_bank_has_one_row_per_molecule():
    A_ = np.abs(np.random.default_rng(2).normal(0, 1, (12, 5)))
    y = np.array(["a"] * 5 + ["b"] * 4 + ["c"] * 3)
    R_, labs = RET.build_reference_bank(A_, y)
    assert R_.shape == (3, 5) and labs == ["a", "b", "c"]
    assert np.allclose(R_[0], A_[:5].mean(axis=0))


@pytest.mark.parametrize("metric", ["cosine", "pearson", "spearman", "centered_cosine",
                                    "correlation_distance", "angular"])
def test_a_vector_is_most_similar_to_itself(metric):
    rng = np.random.default_rng(3)
    A_ = np.abs(rng.normal(0, 1, (8, 12)))
    S = RET.similarity(A_, A_, metric)
    assert (np.argmax(S, axis=1) == np.arange(8)).all()


def test_mahalanobis_is_dropped_when_the_covariance_is_singular():
    A_ = np.tile(np.arange(6, dtype=float), (4, 1))       # rank 1
    assert RET.stable_covariance(A_, shrinkage=0.0) is None


def test_retrieve_returns_a_non_negative_margin():
    rng = np.random.default_rng(4)
    Q, Rb = np.abs(rng.normal(0, 1, (6, 9))), np.abs(rng.normal(0, 1, (10, 9)))
    out = RET.retrieve(Q, Rb, [f"m{i}" for i in range(10)], "cosine")
    assert (out["margin"] >= -1e-12).all()
    assert all(len(t) == 5 for t in out["topk"])


# ── Step 3: calibration ──────────────────────────────────────────────────────
def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    rng = np.random.default_rng(5)
    conf = rng.uniform(0, 1, 20000)
    correct = (rng.uniform(0, 1, 20000) < conf).astype(float)
    assert CAL.expected_calibration_error(conf, correct) < 0.02


def test_ece_cannot_distinguish_a_constant_predictor_from_a_good_one():
    """ADVERSARIAL — the defect that made Platt scaling win the first benchmark.

    A predictor that reports the base rate for every input is perfectly calibrated and carries
    no information. ECE says it is excellent; Brier, sharpness and discrimination all say it is
    not. The selection rule must use the latter, and this test pins the reason.
    """
    rng = np.random.default_rng(6)
    correct = (rng.uniform(0, 1, 4000) < 0.6).astype(float)
    constant = np.full(4000, 0.6)
    informative = np.where(correct > 0, rng.uniform(0.6, 1.0, 4000),
                           rng.uniform(0.2, 0.6, 4000))
    assert CAL.expected_calibration_error(constant, correct) < \
        CAL.expected_calibration_error(informative, correct)      # ECE prefers the useless one
    assert CAL.brier(constant, correct) > CAL.brier(informative, correct)
    assert CAL.sharpness(constant) == pytest.approx(0.0, abs=1e-9)
    assert CAL.discrimination(constant, correct.astype(bool)) == pytest.approx(0.5, abs=0.02)
    assert CAL.discrimination(informative, correct.astype(bool)) > 0.85


def test_benchmark_ranks_by_brier_not_ece():
    """ADVERSARIAL — guards the corrected selection rule.

    The regime is chosen so the two orderings genuinely disagree, and the test asserts that
    up front. Without that guard the assertion below passes vacuously whenever ECE and Brier
    happen to agree, which is how an earlier version of this test let a mutation through.
    """
    rng = np.random.default_rng(7)
    S = rng.normal(0, 1, (600, 20))
    S[np.arange(600), 0] += 2.0
    correct = (np.argmax(S, axis=1) == 0).astype(float)
    tab = CAL.benchmark(S[:300], correct[:300], S[300:], correct[300:])
    by_ece = tab.sort_values("ece").method.tolist()
    by_brier = tab.sort_values("brier").method.tolist()
    assert by_ece != by_brier, "regime does not discriminate the two rules; test is vacuous"
    assert tab.method.tolist() == by_brier
    for col in ("sharpness", "discrimination", "ece", "brier"):
        assert col in tab.columns


def test_temperature_scaling_optimises_brier_not_ece():
    """ADVERSARIAL — the fitted temperature must be the Brier minimiser, not the ECE one.

    Asserted as a direct property of the fitted parameter rather than through a downstream
    proxy: on synthetic data the two objectives often land close together, so a
    sharpness-based check would not reliably detect the objective being swapped back.
    """
    rng = np.random.default_rng(8)
    S = rng.normal(0, 1, (600, 20))
    S[np.arange(600), 0] += 2.0
    correct = (np.argmax(S, axis=1) == 0).astype(float)
    T = CAL.Calibrator("temperature").fit(S, correct).params_["T"]
    grid = np.exp(np.linspace(np.log(1e-3), np.log(1e2), 200))
    t_brier = min(grid, key=lambda t: CAL.brier(CAL._softmax(S, t).max(axis=1), correct))
    t_ece = min(grid, key=lambda t: CAL.expected_calibration_error(
        CAL._softmax(S, t).max(axis=1), correct))
    assert t_brier != t_ece, "regime does not discriminate the two objectives; test is vacuous"
    assert T == pytest.approx(t_brier)


def test_calibrator_output_stays_in_the_unit_interval():
    rng = np.random.default_rng(9)
    S = rng.normal(0, 1, (200, 12))
    correct = (rng.uniform(0, 1, 200) < 0.5).astype(float)
    for m in CAL.METHODS:
        p = CAL.Calibrator(m).fit(S, correct).transform(S)
        assert p.min() >= -1e-9 and p.max() <= 1 + 1e-9, m


# ── Step 4: open-set ─────────────────────────────────────────────────────────
def test_auroc_matches_a_known_case():
    assert OS.auroc([1, 2, 3, 4], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert OS.auroc([4, 3, 2, 1], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert OS.auroc([1, 1, 1, 1], [0, 0, 1, 1]) == pytest.approx(0.5)


def test_every_channel_has_a_declared_sign():
    assert set(OS.CHANNELS) == set(OS.CHANNEL_SIGN)


def test_mahalanobis_channel_uses_the_supplied_reference_mean():
    """ADVERSARIAL — centring negatives on their own mean asks the wrong question.

    The first version scored an inverted AUROC of 0.176 because each batch was centred on
    itself. Passing a reference mean must change the answer; if it does not, the argument is
    being ignored again.
    """
    rng = np.random.default_rng(10)
    A_in = np.abs(rng.normal(1.0, 0.2, (40, 6)))
    A_out = np.abs(rng.normal(5.0, 0.2, (40, 6)))
    ci = RET.stable_covariance(A_in)
    diag = {"residual_fraction": np.zeros(40), "explained_variance": np.ones(40),
            "component_sparsity": np.zeros(40), "activation_entropy": np.zeros(40)}
    Rb = A_in[:5]
    own = OS.channel_scores(A_out, diag, Rb, ci)["ood_mahalanobis"]
    ref = OS.channel_scores(A_out, diag, Rb, ci, A_in.mean(axis=0))["ood_mahalanobis"]
    assert ref.mean() > own.mean() * 2


def test_joint_score_standardises_against_the_in_domain_channels():
    rng = np.random.default_rng(11)
    ch_in = {"residual_fraction": rng.normal(0.1, 0.02, 50),
             "explained_variance": rng.normal(0.9, 0.02, 50)}
    ch_out = {"residual_fraction": rng.normal(0.5, 0.02, 50),
              "explained_variance": rng.normal(0.4, 0.02, 50)}
    assert OS.joint_score(ch_out, ch_in).mean() > OS.joint_score(ch_in, ch_in).mean() + 3


def test_operating_point_respects_its_target():
    rng = np.random.default_rng(12)
    s_in, s_out = rng.normal(0, 1, 500), rng.normal(3, 1, 500)
    op = OS.operating_point(s_in, s_out, 0.95)
    assert op["in_domain_accept"] == pytest.approx(0.95, abs=0.02)


# ── Step 6: the evidence profile ─────────────────────────────────────────────
def test_eleven_axes_are_declared_and_every_window_is_inside_the_corpus_range():
    assert len(EV.AXIS_NAMES) == 11
    for name, wins in EV.AXES.items():
        assert wins, name
        for lo, hi, w, desc in wins:
            assert 450 <= lo < hi <= 1800, name
            assert 0 < w <= 1.0 and desc


def test_a_csm_loads_only_on_axes_its_diagnostic_bands_reach():
    """ADVERSARIAL — without the diagnostic-band mask every axis loaded on all 49 CSMs."""
    grid = np.arange(450.0, 1800.1, 2.0)
    H = np.exp(-0.5 * ((grid - 725.0) / 8.0) ** 2)[None, :]
    recs = [{"csm_id": "csm00", "dominant_bands": [725.0]}]
    M, _ = EV.build_axis_map(H, grid, recs)
    i_purine = list(EV.AXIS_NAMES).index("purine")
    i_amide = list(EV.AXIS_NAMES).index("amide_protein")
    assert M[0, i_purine] > 0.5
    assert M[0, i_amide] == 0.0


def test_unassigned_mass_is_reported_not_redistributed():
    """A band in no axis window must land in `unassigned`, never on a spoke."""
    grid = np.arange(450.0, 1800.1, 2.0)
    H = np.exp(-0.5 * ((grid - 1450.0) / 8.0) ** 2)[None, :] + \
        np.exp(-0.5 * ((grid - 468.0) / 8.0) ** 2)[None, :]
    recs = [{"csm_id": "csm00", "dominant_bands": [1450.0, 468.0]}]
    M, un = EV.build_axis_map(H, grid, recs)
    assert un[0] > 0.2
    assert M[0].sum() < 1.0


def test_specificity_demotes_a_ubiquitous_axis():
    M = np.zeros((20, len(EV.AXIS_NAMES)))
    M[:, 0] = 0.5                      # on every CSM
    M[:2, 1] = 0.5                     # on two
    s = EV.axis_specificity(M)
    assert s[1] > s[0]


def test_profile_gives_zero_to_an_axis_with_no_evidence():
    """No spectrum is forced onto every axis — the Phase 03 softmax failure, in this layer."""
    M = np.zeros((3, len(EV.AXIS_NAMES)))
    M[0, 0], M[1, 3], M[2, 6] = 1.0, 1.0, 1.0
    a = np.array([[1.0, 0.0, 0.0]])
    p = EV.profile(a, M, np.ones(len(EV.AXIS_NAMES)))
    assert p["magnitude"][0, 0] == pytest.approx(1.0)
    assert p["magnitude"][0, 3] == 0.0 and p["magnitude"][0, 6] == 0.0


def test_confidence_is_bounded_and_falls_with_reconstruction_quality():
    M = np.zeros((2, len(EV.AXIS_NAMES)))
    M[0, 0], M[1, 1] = 1.0, 1.0
    a = np.array([[1.0, 1.0]])
    good = EV.profile(a, M, np.ones(len(EV.AXIS_NAMES)), np.array([1.0]))["confidence"]
    poor = EV.profile(a, M, np.ones(len(EV.AXIS_NAMES)), np.array([0.2]))["confidence"]
    assert (good <= 1.0 + 1e-9).all() and (good >= 0).all()
    assert (poor <= good + 1e-12).all() and poor.max() < good.max()


def test_window_overlap_reports_the_known_ambiguities():
    ov = EV.window_overlap()
    pair = {frozenset((r.axis_a, r.axis_b)) for _, r in ov.iterrows()}
    assert frozenset(("heterocyclic_ring", "purine")) in pair
    assert frozenset(("unsaturation", "amide_protein")) in pair


def test_validate_axes_marks_a_random_axis_as_not_discriminative():
    rng = np.random.default_rng(13)
    E = rng.uniform(0, 1, (200, len(EV.AXIS_NAMES)))
    cls = np.array(["purine"] * 100 + ["other"] * 100)
    v = EV.validate_axes(E, cls, {"purine": ["purine"]})
    row = v[v.axis == "purine"].iloc[0]
    assert row.verdict in ("weak", "not discriminative")


# ── Step 8: provenance ───────────────────────────────────────────────────────
def test_axis_chain_contributions_sum_to_the_axis_value():
    M = np.zeros((4, len(EV.AXIS_NAMES)))
    M[:, 0] = [0.4, 0.3, 0.2, 0.1]
    recs = [{"csm_id": f"csm{i:02d}", "contributing_lsms": [{"lsm_id": f"l{i}"}],
             "supporting_analytes": [f"mol{i}"], "supporting_classes": ["c"],
             "dominant_bands": [1440.0]} for i in range(4)]
    a = np.array([1.0, 2.0, 3.0, 4.0])
    ch = PROV.axis_chain(EV.AXIS_NAMES[0], a, M, recs, {EV.AXIS_NAMES[0]: 0})
    assert ch["total_contribution"] == pytest.approx(float(a @ M[:, 0]))
    assert sum(l["contribution"] for l in ch["csm_chain"]) == \
        pytest.approx(ch["total_contribution"])


def test_verify_chains_flags_an_unknown_molecule():
    ch = [{"axis": "purine", "csm_chain": [{"csm_id": "csm00"}], "lsms": ["l0"],
           "molecules": ["not_a_real_molecule"], "classes": ["c"]}]
    v = PROV.verify_chains(ch, {"l0"}, {"adenine"})
    assert not bool(v.iloc[0].intact) and int(v.iloc[0].unknown_molecules) == 1


# ── the engine ───────────────────────────────────────────────────────────────
@needs_run
def test_engine_reconstructs_and_is_deterministic():
    z = np.load(A / "csm_activations_v1.npz", allow_pickle=True)
    cfg = json.loads((A / "canonical_engine_config_v1.json").read_text())
    CSM = np.load(FROZEN / "phase02/artifacts/csm_dictionary_v1.npz")["CSM"]
    recs = json.loads((FROZEN / "phase02/artifacts/csm_registry_v1.json").read_text())["csms"]
    br = np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
    X, grid = np.asarray(br["X"], float), np.asarray(br["grid"], float)
    bank = np.load(A / "reference_bank_v1.npz", allow_pickle=True)
    m = np.load(A / "evidence_axis_map_v1.npz", allow_pickle=True)
    cal = CAL.Calibrator(cfg["calibration"])
    S = np.random.default_rng(0).normal(0, 1, (100, 20))
    cal.fit(S, (np.random.default_rng(1).uniform(0, 1, 100) < 0.6).astype(float))
    eng = CanonicalEngine(CSM, recs, grid, bank["R"], [str(s) for s in bank["labels"]],
                          [str(s) for s in bank["classes"]], m["M"], m["unassigned"],
                          m["specificity"], cal, cfg["metric"])
    r1, r2 = eng.infer(X[:6]), eng.infer(X[:6])
    for i, (a, b) in enumerate(zip(r1, r2)):
        assert np.array_equal(a.activation, b.activation)
        assert a.confidence == b.confidence
        # and the engine reproduces the activations the committed run stored
        assert np.allclose(a.activation, z["A"][i], atol=1e-8)


@needs_run
def test_engine_report_serialises_completely():
    reps = json.loads((A / "representative_reports_v1.json").read_text())
    assert reps
    for r in reps:
        for k in ("activation", "diagnostics", "top_molecules", "confidence",
                  "chemistry_class", "evidence_profile", "provenance", "rejected"):
            assert k in r, k
        assert len(r["activation"]) == 49
        assert len(r["evidence_profile"]["magnitude"]) == 11


@needs_run
def test_rejected_spectra_report_no_molecule_identity():
    """A rejected spectrum must not carry a molecule claim — rejection is not a hedge."""
    rej = json.loads((A / "rejected_examples_v1.json").read_text())
    for r in rej:
        if r["rejected"]:
            assert r["top_molecules"] == []
            assert any("REJECTED" in n for n in r["notes"])


# ── artifacts of the committed run ───────────────────────────────────────────
@needs_run
def test_phase_state_records_scope_and_replacement():
    st = json.loads((OUT.root / "PHASE_STATE.json").read_text())
    assert "Raman only" in st["scope"] and "SERS" in st["scope"]
    assert "Theme/BSV" in st["replaces"]
    assert st["seed"] == 0


@needs_run
def test_frozen_fingerprints_are_the_expected_ones():
    cfg = json.loads((A / "canonical_engine_config_v1.json").read_text())
    assert cfg["csm_fingerprint"] == "0b4aa550ccefed3edabdbde5bae11c8d"
    assert cfg["lsm_fingerprint"] == "208482d6f7178b5b8f16cace91be55b0"


@needs_run
def test_geometry_is_not_used_in_inference():
    cfg = json.loads((A / "canonical_engine_config_v1.json").read_text())
    assert cfg["geometry_used_in_inference"] is False


@needs_run
def test_no_upstream_artifact_was_modified():
    """Phase 05 must not write anywhere but its own tree."""
    st = json.loads((OUT.root / "PHASE_STATE.json").read_text())
    for o in st["outputs"]:
        assert o["path"].startswith("phase05/") or "/phase05/" in o["path"], o["path"]


@needs_run
def test_split_b_molecule_top1_is_undefined_not_zero(summary):
    """ADVERSARIAL — Phase 04 reported 0.000, which described the design, not the engine."""
    assert summary["split_b"]["molecule_top1"] is None
    assert "undefined" in summary["split_b"]["molecule_top1_note"]


@needs_run
def test_headline_numbers_match_the_report(summary):
    assert summary["split_a"]["molecule_top1"] == pytest.approx(0.605, abs=0.002)
    assert summary["split_a"]["molecule_top5"] == pytest.approx(0.795, abs=0.002)
    assert summary["split_b"]["class_top1"] == pytest.approx(0.845, abs=0.002)
    assert summary["split_b"]["macro_f1"] == pytest.approx(0.807, abs=0.002)
    assert summary["openset"]["joint_auroc"] == pytest.approx(0.921, abs=0.005)
    assert summary["projection"]["mean_ev"] == pytest.approx(0.821, abs=0.002)


@needs_run
def test_the_selected_calibrator_is_informative(summary):
    """ADVERSARIAL — the constant-confidence calibrator must never be selected again."""
    assert summary["split_a"]["sharpness"] > 0.05
    assert summary["split_a"]["discrimination"] > 0.75
    assert summary["split_a"]["calibration"] != "platt"


@needs_run
def test_platt_is_still_recorded_as_the_ece_winner():
    """The finding itself is an artifact: ECE alone would have chosen the useless calibrator."""
    s = pd.read_csv(T / "calibration_summary_v1.csv")
    a = s[s.split == "splitA_molecule"]
    assert a.sort_values("ece").iloc[0]["method"] == "platt"
    assert float(a[a.method == "platt"].sharpness.iloc[0]) < 0.01


@needs_run
def test_csm_beats_raw_on_both_halves_of_the_hypothesis():
    d = pd.read_csv(T / "robustness_summary_v1.csv").set_index("representation")
    assert d.loc["csm", "class_retention_grouped"] > d.loc["raw", "class_retention_grouped"]
    assert d.loc["csm", "clean_class_top1_grouped"] > d.loc["raw", "clean_class_top1_grouped"]


@needs_run
def test_the_in_sample_comparison_is_still_reported():
    """ADVERSARIAL — R-10. In-sample, raw wins by self-matching; both numbers must be visible."""
    d = pd.read_csv(T / "robustness_summary_v1.csv").set_index("representation")
    assert d.loc["raw", "clean_class_top1"] > d.loc["csm", "clean_class_top1"]
    assert "clean_class_top1_grouped" in d.columns


@needs_run
def test_intensity_scaling_is_invariant_for_every_representation():
    r = pd.read_csv(T / "noise_robustness_v1.csv")
    s = r[r.perturbation == "intensity_scaling"]
    for rep in s.representation.unique():
        v = s[s.representation == rep].class_top1_grouped
        assert v.max() - v.min() < 1e-9, rep


@needs_run
def test_no_provenance_chain_is_broken(summary):
    assert summary["provenance"]["broken"] == 0
    assert summary["provenance"]["n_chains"] > 1000


@needs_run
def test_axis_grounding_is_stable_under_the_support_floor():
    s = pd.read_csv(T / "evidence_axis_sensitivity_v1.csv")
    f = s[s.parameter == "support_floor"]
    assert len(f) == 4
    assert f.n_grounded.nunique() == 1, "the floor must not change the grounding verdicts"
    assert f.mean_axes_per_csm.is_monotonic_decreasing


@needs_run
def test_the_unsaturation_secondary_test_is_recorded_alongside_the_failure(summary):
    """The primary failure must survive in the artifacts, not be replaced by the rescue."""
    prim = [v for v in summary["evidence"]["validation"] if v["axis"] == "unsaturation"][0]
    sec = [v for v in summary["evidence"]["secondary_tests"] if v["axis"] == "unsaturation"][0]
    assert prim["verdict"] == "not discriminative" and prim["auroc"] < 0.6
    assert sec["auroc"] > 0.95


@needs_run
def test_two_openset_channels_are_reported_below_chance():
    """The inverted channels are a finding; silently flipping their sign would hide it."""
    ch = pd.read_csv(T / "openset_channel_auroc_v1.csv").set_index("channel")
    assert ch.loc["ood_mahalanobis", "auroc"] < 0.5
    assert ch.loc["centroid_distance", "auroc"] < 0.5
    assert ch.loc["JOINT", "auroc"] > 0.85


@needs_run
def test_the_calibration_gate_fails_and_is_not_hidden():
    g = pd.read_csv(T / "phase05_gates_v1.csv")
    g6 = g[g.gate.str.startswith("G6 ")].iloc[0]
    assert g6.status == "FAIL"
    assert (g[g.gate.str.startswith("G6b")].iloc[0].status) == "PASS"
    assert int((g.status == "FAIL").sum()) == 1


@needs_run
def test_all_fifteen_figures_and_the_pdf_exist():
    pngs = sorted(F.glob("F*.png"))
    assert len(pngs) == 15
    assert not list(F.glob("*.svg")), "PNG only from Phase 02.5 onward"
    assert (R / "PHASE_05_RESULTS.pdf").exists()
    assert (R / "PHASE_05_CANONICAL_INFERENCE_ENGINE.md").exists()
    assert (R / "PHASE_05_SCIENTIFIC_AUDIT.md").exists()


@needs_run
def test_no_sers_or_cross_modality_artifact_was_produced():
    """Raman only — scope enforced on the artifacts, not just asserted in prose."""
    banned = ("sers", "ag_sers", "cross_modal", "modality")
    for p in list(T.glob("*")) + list(A.glob("*")) + list(F.glob("*")):
        assert not any(b in p.name.lower() for b in banned), p.name


@needs_run
def test_the_run_script_hardcodes_no_output_path():
    src = (OUT.root / "code" / "run_phase05.py").read_text()
    assert "/Users/" not in src and "/Volumes/" not in src
    assert "PhaseOutputs" in src
