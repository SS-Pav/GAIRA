"""GAIRA V7 — Phase 07 regression tests: the BSV2 biochemical programme layer.

Contract tests on the programme modules, artifact tests on the committed run, mutation-style
adversarial tests encoding the three defects found during the phase, and a hard test that the
factorisation never saw anything but the Chemistry Evidence matrix.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gaira.v7.io import PhaseOutputs, frozen_root
from gaira.v7.programs import (explain as EXP, factorization as FAC, selection as SEL,
                               validation as VAL)

OUT = PhaseOutputs("07", extra=("interactive", "manifests"))
T, A_, F, R = OUT.tables, OUT.artifacts, OUT.figures, OUT.reports
FROZEN = frozen_root()
HAS_RUN = (OUT.root / "PHASE_STATE.json").exists()
needs_run = pytest.mark.skipif(not HAS_RUN, reason="Phase 07 has not been run")


@pytest.fixture(scope="module")
def summary():
    return json.loads((A_ / "phase07_summary_v1.json").read_text())


@pytest.fixture(scope="module")
def toy():
    """A synthetic evidence matrix built from 3 known programmes over 16 axes."""
    rng = np.random.default_rng(0)
    P = np.zeros((3, 16))
    P[0, [0, 3, 9]] = [0.5, 0.4, 0.3]          # a lipid-like programme
    P[1, [5, 10]] = [0.6, 0.4]                 # a carbohydrate-like programme
    P[2, [1, 8]] = [0.5, 0.5]                  # an energy-like programme
    W = np.abs(rng.normal(0, 1, (150, 3)))
    Ev = np.clip(W @ P + 0.01 * rng.random((150, 16)), 0, None)
    y = np.array([f"m{i // 2}" for i in range(150)])
    cls = np.array([["a", "b", "c"][int(np.argmax(w))] for w in W])
    folds = np.array([int(m[1:]) % 5 for m in y])
    return Ev, W, P, y, cls, folds


# ── factorisation contracts ──────────────────────────────────────────────────
def test_every_family_fits_and_returns_the_requested_K(toy):
    Ev = toy[0]
    for fam in FAC.FAMILIES:
        m = FAC.fit(fam, Ev, 4, seed=0)
        assert m["P"].shape == (4, 16), fam
        assert m["W"].shape == (len(Ev), 4), fam


def test_non_negative_families_have_non_negative_activations_and_loadings(toy):
    Ev = toy[0]
    for fam in FAC.NON_NEGATIVE:
        m = FAC.fit(fam, Ev, 4, seed=0)
        assert m["W"].min() >= -1e-9, fam
        assert m["P"].min() >= -1e-9, fam


def test_semi_nmf_keeps_activations_non_negative_but_allows_signed_loadings(toy):
    m = FAC.fit("semi_nmf", toy[0], 4, seed=0)
    assert m["W"].min() >= -1e-9
    assert m.get("signed_loadings") is True


def test_controls_are_marked_signed(toy):
    for fam in FAC.CONTROLS:
        assert FAC.fit(fam, toy[0], 4, seed=0).get("signed") is True


def test_factorisation_is_deterministic(toy):
    for fam in FAC.FAMILIES:
        a = FAC.fit(fam, toy[0], 4, seed=0)["P"]
        b = FAC.fit(fam, toy[0], 4, seed=0)["P"]
        assert np.allclose(a, b), fam


def test_nmf_recovers_planted_programmes(toy):
    """Sanity: with three planted programmes, NMF at K=3 should find them."""
    from scipy.optimize import linear_sum_assignment
    Ev, _, P, *_ = toy
    m = FAC.fit("nmf", Ev, 3, seed=0)
    N1 = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    N2 = m["P"] / (np.linalg.norm(m["P"], axis=1, keepdims=True) + 1e-12)
    C = N1 @ N2.T
    r, c = linear_sum_assignment(-C)
    assert C[r, c].mean() > 0.90


def test_projection_of_new_evidence_needs_no_fitting(toy):
    Ev = toy[0]
    m = FAC.fit("nmf", Ev[:100], 4, seed=0)
    W = FAC.project(m, Ev[100:])
    assert W.shape == (50, 4)
    assert W.min() >= -1e-9
    assert np.allclose(W, FAC.project(m, Ev[100:])), "projection must be deterministic"


def test_max_single_axis_share_detects_a_permutation_of_the_identity():
    """ADVERSARIAL — at K=16 over 16 axes, NMF learns the identity and every programme IS a
    chemistry class. The first version of the rule selected exactly that."""
    assert FAC.max_single_axis_share(np.eye(16)) == pytest.approx(1.0)
    assert FAC.max_single_axis_share(np.ones((4, 16))) == pytest.approx(1 / 16, abs=1e-6)


def test_dominance_detects_a_background_programme():
    W = np.zeros((100, 4)); W[:, 0] = 1.0; W[:5, 1] = 2.0
    assert FAC.dominance(W) > 0.9
    assert FAC.dominance(np.eye(4)[np.arange(100) % 4]) == pytest.approx(0.25, abs=0.01)


def test_redundancy_detects_a_duplicated_programme():
    P = np.eye(4, 16)
    P[3] = P[2]
    assert FAC.redundancy(P) == pytest.approx(1.0)


# ── the pre-registered rule ──────────────────────────────────────────────────
def test_the_rule_rejects_signed_activations():
    """PCA and ICA are controls; P-02 makes them permanently ineligible."""
    ok, why = SEL.eligible({"non_negative_activations": False, "K": 6,
                            "information_retained_vs_chemistry_evidence": 0.99,
                            "heldout_chemistry_retention": 0.99, "max_pairwise_overlap": 0.1,
                            "dominance": 0.2, "max_single_axis_share": 0.3})
    assert not ok and "control only" in why


def test_the_rule_rejects_K_above_the_input_effective_rank():
    """ADVERSARIAL — the defect that selected K=16 over a 16-dimensional input."""
    base = {"non_negative_activations": True,
            "information_retained_vs_chemistry_evidence": 0.99,
            "heldout_chemistry_retention": 0.99, "max_pairwise_overlap": 0.1,
            "dominance": 0.2, "max_single_axis_share": 0.3}
    ok, why = SEL.eligible({**base, "K": 16})
    assert not ok and "rotation, not a compression" in why
    assert SEL.eligible({**base, "K": 12})[0]


def test_the_rule_rejects_a_programme_that_is_one_chemistry_class():
    """ADVERSARIAL — the brief's constraint, now encoded in the floors."""
    base = {"non_negative_activations": True, "K": 6,
            "information_retained_vs_chemistry_evidence": 0.99,
            "heldout_chemistry_retention": 0.99, "max_pairwise_overlap": 0.1,
            "dominance": 0.2}
    ok, why = SEL.eligible({**base, "max_single_axis_share": 0.98})
    assert not ok and "IS a chemistry class" in why


def test_the_rule_enforces_both_informativeness_floors():
    base = {"non_negative_activations": True, "K": 6, "max_pairwise_overlap": 0.1,
            "dominance": 0.2, "max_single_axis_share": 0.3,
            "information_retained_vs_chemistry_evidence": 0.99,
            "heldout_chemistry_retention": 0.99}
    assert not SEL.eligible({**base, "information_retained_vs_chemistry_evidence": 0.3})[0]
    assert not SEL.eligible({**base, "heldout_chemistry_retention": 0.3})[0]


def test_the_objective_is_a_product_so_neither_axis_can_compensate():
    """ADVERSARIAL — a sum would let perfect stability buy a useless representation (P-18)."""
    assert "*" in SEL.OBJECTIVE
    t = pd.DataFrame([
        {"family": "a", "K": 6, "non_negative_activations": True,
         "information_retained_vs_chemistry_evidence": 0.9, "heldout_chemistry_retention": 0.55,
         "bootstrap_stability": 1.00, "max_pairwise_overlap": 0.1, "dominance": 0.2,
         "max_single_axis_share": 0.3},
        {"family": "b", "K": 6, "non_negative_activations": True,
         "information_retained_vs_chemistry_evidence": 0.9, "heldout_chemistry_retention": 0.90,
         "bootstrap_stability": 0.90, "max_pairwise_overlap": 0.1, "dominance": 0.2,
         "max_single_axis_share": 0.3}])
    assert SEL.select(t)["family"] == "b"


def test_ties_break_toward_the_smaller_K():
    t = pd.DataFrame([
        {"family": "a", "K": 10, "non_negative_activations": True,
         "information_retained_vs_chemistry_evidence": 0.9, "heldout_chemistry_retention": 0.90,
         "bootstrap_stability": 0.90, "max_pairwise_overlap": 0.1, "dominance": 0.2,
         "max_single_axis_share": 0.3},
        {"family": "a", "K": 5, "non_negative_activations": True,
         "information_retained_vs_chemistry_evidence": 0.9, "heldout_chemistry_retention": 0.895,
         "bootstrap_stability": 0.90, "max_pairwise_overlap": 0.1, "dominance": 0.2,
         "max_single_axis_share": 0.3}])
    assert SEL.select(t)["K"] == 5


def test_no_eligible_candidate_is_reported_not_papered_over():
    t = pd.DataFrame([{"family": "a", "K": 6, "non_negative_activations": False,
                       "information_retained_vs_chemistry_evidence": 0.1,
                       "heldout_chemistry_retention": 0.1, "max_pairwise_overlap": 0.99,
                       "dominance": 0.99, "max_single_axis_share": 0.99,
                       "bootstrap_stability": 0.1}])
    assert SEL.select(t)["decision"] == "NO ELIGIBLE CANDIDATE"


# ── validation contracts ─────────────────────────────────────────────────────
def test_bootstrap_recovery_is_high_for_a_clean_planted_structure(toy):
    b = VAL.bootstrap_recovery(toy[0], "nmf", 3, n_boot=8, seed=0)
    assert b["mean"] > 0.85 and b["min"] <= b["mean"]


def test_heldout_chemistry_never_lets_a_molecule_cross_the_split(toy):
    Ev, W, P, y, cls, folds = toy
    h = VAL.heldout_chemistry(W, cls, folds, y)
    assert 0.0 <= h["top1"] <= 1.0 and h["top1"] <= h["top3"]
    for f in set(folds.tolist()):
        assert not (set(y[folds == f]) & set(y[folds != f]))


def test_information_retained_is_bounded(toy):
    m = FAC.fit("nmf", toy[0], 3, seed=0)
    assert 0.0 <= VAL.information_retained(toy[0], m) <= 1.0


def test_programme_coherence_reports_every_programme(toy):
    Ev, W, P, y, cls, folds = toy
    m = FAC.fit("nmf", Ev, 4, seed=0)
    c = VAL.programme_coherence(m["W"], m["P"], cls)
    assert len(c) == 4 and "usage_share" in c.columns


# ── explanation contracts ────────────────────────────────────────────────────
def test_descriptions_are_composed_from_evidence_not_hand_written(toy):
    Ev, W, P, y, cls, folds = toy
    from gaira.v7.chemistry.registry import CLASS_ORDER
    m = FAC.fit("nmf", Ev, 3, seed=0)
    recs = EXP.programme_evidence(m["P"], m["W"], list(CLASS_ORDER), y, cls)
    broad = {c: "lipid" for c in CLASS_ORDER}
    for r in recs:
        desc, basis = EXP.describe(r, broad)
        assert isinstance(desc, str) and len(basis) > 10
        assert "%" in basis or "loading" in basis


def test_a_diffuse_programme_is_not_given_a_name_it_has_not_earned():
    """ADVERSARIAL — a template that always produces a chemistry name would be naming noise."""
    rec = {"top_chemistry_axes": [{"chemistry_axis": "purine", "loading": 0.1,
                                   "share_of_programme": 0.09}],
           "usage_share": 0.1}
    desc, _ = EXP.describe(rec, {"purine": "nucleic"})
    assert desc.startswith("diffuse")


# ── artifacts of the committed run ───────────────────────────────────────────
@needs_run
def test_phase_state_and_fingerprints():
    st = json.loads((OUT.root / "PHASE_STATE.json").read_text())
    fp = st["input_fingerprints"]
    assert fp["csm"] == "0b4aa550ccefed3edabdbde5bae11c8d"
    assert fp["lsm"] == "208482d6f7178b5b8f16cace91be55b0"
    assert fp["engine"] == "20d8bd99ce71f45a125c6a2b1d719e51"
    assert st["phase08_begun"] is False
    assert st["input"] == "chemistry evidence only"
    assert "Raman only" in st["scope"]


@needs_run
def test_the_factorisation_saw_only_chemistry_evidence():
    """ADVERSARIAL — the phase's central methodological claim.

    Spectra, CSM activations and perturbations appear in the script, but only AFTER the model is
    fitted: for noise robustness and for explanation. Nothing above the sweep may touch them.
    """
    src = (OUT.root / "code" / "run_phase07.py").read_text()
    build = src[src.index("# ── 1. the sweep"):src.index("# ── 3. reconstruction")]
    for banned in ("balanced_references", "PRJ.project", "csm_dictionary", "PERT.apply",
                   "embeddings", "continuous_coordinates", "theme"):
        assert banned not in build, f"{banned} appears in the model-fitting section"
    assert "Ev" in build


@needs_run
def test_no_upstream_artifact_was_written():
    st = json.loads((OUT.root / "PHASE_STATE.json").read_text())
    for o in st["outputs"]:
        assert "phase07" in o["path"], o["path"]


@needs_run
def test_all_families_and_K_were_fitted():
    """ADVERSARIAL — sparse NMF scored EV -0.401 at every K on a single untuned alpha."""
    t = pd.read_csv(T / "programme_sweep_v1.csv")
    assert bool(t.usable.all()), "no candidate may fail silently"
    assert set(t.K) == set(range(2, 17))
    fams = set(t.family)
    assert any("sparse_nmf" in f for f in fams)
    assert len([f for f in fams if "sparse_nmf" in f]) >= 3, "the penalty must be swept"
    assert {"nmf", "orthogonal_nmf", "semi_nmf", "pca_control", "ica_control"} <= fams


@needs_run
def test_controls_never_became_candidates():
    t = pd.read_csv(T / "programme_selection_v1.csv")
    for c in ("pca_control", "ica_control"):
        assert not bool(t[t.family == c].eligible.any()), f"{c} must never be eligible"


@needs_run
def test_K_is_below_the_input_effective_rank(summary):
    assert summary["model"]["K"] <= SEL.FLOORS["max_K"]
    assert summary["model"]["K"] < summary["input"]["shape"][1]
    assert 16 / summary["model"]["K"] > 1.0, "BSV2 must compress"


@needs_run
def test_no_programme_is_a_single_chemistry_class(summary):
    """ADVERSARIAL — the brief's hard constraint."""
    comp = summary["compositeness"]
    assert max(comp["programme_top_axis_share"]) <= SEL.FLOORS["max_single_axis_share"]
    assert comp["n_genuinely_composite"] >= 1


@needs_run
def test_compositeness_is_reported_honestly(summary):
    """Only 3 of 9 programmes are multi-chemistry; that number must survive in the artifacts."""
    comp = summary["compositeness"]
    assert comp["n_genuinely_composite"] + comp["n_near_single_class"] == summary["model"]["K"]
    assert "composite_definition" in comp


@needs_run
def test_floors_were_cleared_by_the_adopted_model(summary):
    assert summary["information_retained"] >= \
        SEL.FLOORS["information_retained_vs_chemistry_evidence"]
    O = np.array(summary["overlap_matrix"])
    iu = np.triu_indices(len(O), 1)
    assert O[iu].max() <= SEL.FLOORS["max_pairwise_overlap"]


@needs_run
def test_generalisation_gap_is_reported(summary):
    gen = [g["explained_variance"] for g in summary["generalisation"]]
    assert len(gen) >= 3
    assert np.mean(gen) <= summary["reconstruction"]["explained_variance"] + 1e-9, \
        "held-out reconstruction cannot exceed in-sample"


@needs_run
def test_activations_are_non_negative():
    z = np.load(A_ / "bsv2_programmes_v1.npz", allow_pickle=True)
    assert z["W"].min() >= -1e-9
    assert z["W"].shape[1] == json.loads(
        (A_ / "phase07_summary_v1.json").read_text())["model"]["K"]


@needs_run
def test_the_p02_loading_sign_question_is_surfaced(summary):
    """ADVERSARIAL — the rule's winner has signed loadings; that must not be hidden."""
    p02 = summary.get("p02_compliant_alternative")
    assert p02 is not None
    assert p02["family"] in FAC.NON_NEGATIVE
    assert "objective_cost_vs_rule_winner" in p02


@needs_run
def test_bsv2_beats_the_pca_control(summary):
    d = {r["representation"]: r for r in summary["compression"]}
    assert d["BSV2_programmes"]["dim"] == d["PCA_control"]["dim"]
    assert d["BSV2_programmes"]["non_negative"] is True
    assert d["PCA_control"]["non_negative"] is False


@needs_run
def test_png_only_and_all_twelve_figures():
    assert len(list(F.glob("F*.png"))) == 12
    assert not list(F.glob("*.svg"))


@needs_run
def test_all_four_documents_exist():
    for n in ("PHASE_07_REPORT.md", "PHASE_07_SCIENTIFIC_AUDIT.md",
              "PHASE_07_DECISION_GATE.md", "PHASE_07_FIGURES.pdf"):
        assert (R / n).exists(), n


@needs_run
def test_gates_all_pass():
    g = pd.read_csv(T / "phase07_gates_v1.csv")
    assert int((g.status == "FAIL").sum()) == 0
    ids = " ".join(g.gate)
    for k in ("G3 input is Chemistry Evidence only", "G8 K not chosen by hand",
              "G10 soft membership", "G16 Phase 08 not begun"):
        assert k in ids


@needs_run
def test_manifest_complete():
    m = json.loads((OUT.manifests / "bsv2_manifest_v1.json").read_text())
    assert len(m["artifacts"]) > 12
    for a in m["artifacts"]:
        assert "sha256" in a and "path" in a
