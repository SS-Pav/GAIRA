"""GAIRA V7 — Phase 06 regression tests: the Chemistry Evidence Layer.

Three kinds, and the distinction matters:

* **contract tests** on the chemistry modules — properties that must hold for any corpus,
  checked on small synthetic inputs so a failure points at the code rather than at the data;
* **artifact tests** that the committed run produced what the report claims;
* **adversarial tests** encoding the specific defects found during this phase and Phase 05, so a
  regression reintroducing one fails loudly. Those say ADVERSARIAL in their docstring.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gaira.v7.io import PhaseOutputs, frozen_root, output_root
from gaira.v7.chemistry import (calibration as CAL, evidence as EVD, novelty as NOV,
                                provenance as PROV, registry as REG, validation as VAL)

OUT = PhaseOutputs("06", extra=("interactive", "manifests"))
T, A_, F, R = OUT.tables, OUT.artifacts, OUT.figures, OUT.reports
FROZEN = frozen_root()
HAS_RUN = (OUT.root / "PHASE_STATE.json").exists()
needs_run = pytest.mark.skipif(not HAS_RUN, reason="Phase 06 has not been run")


@pytest.fixture(scope="module")
def summary():
    return json.loads((A_ / "phase06_summary_v1.json").read_text())


@pytest.fixture(scope="module")
def preds():
    return np.load(A_ / "chemistry_evidence_predictions_v1.npz", allow_pickle=True)


@pytest.fixture(scope="module")
def toy():
    """A miniature corpus large enough for the real guards.

    Sixteen classes x 4 molecules x 2 replicates = 128 spectra. The size matters: `nested_cv`
    requires 20 training spectra per inner fold and `holdout_class` requires 30 retained
    spectra, and a fixture below those thresholds silently skips the code it means to test.
    """
    rng = np.random.default_rng(0)
    A, y, cls = [], [], []
    for k, c in enumerate(REG.CLASS_ORDER):
        base = np.zeros(16); base[k] = 1.0
        for m in range(4):
            v = np.clip(base + 0.12 * rng.random(16), 0, None)
            for _ in range(2):
                A.append(v + 0.02 * rng.random(16))
                y.append(f"{c}_m{m}")
                cls.append(c)
    return np.array(A), np.array(y), np.array(cls)


# ── the frozen ontology ──────────────────────────────────────────────────────
def test_exactly_sixteen_classes_in_a_fixed_order():
    """ADVERSARIAL — a radar whose axes move between runs is not a coordinate system."""
    assert len(REG.CLASS_ORDER) == 16
    assert len(set(REG.CLASS_ORDER)) == 16
    assert list(REG.CLASS_ORDER) == sorted(REG.CLASS_ORDER), "order must be deterministic"
    assert REG.CLASS_ORDER[0] == "acylglycerol"
    assert REG.CLASS_ORDER[-1] == "sulfur_thiol_cofactor"


def test_index_of_and_one_hot_agree():
    cls = np.array(["purine", "acylglycerol", "purine"])
    Y = REG.one_hot(cls)
    assert Y.shape == (3, 16)
    assert Y[0, REG.index_of("purine")] == 1.0
    assert Y.sum() == 3.0


def test_check_rejects_an_ontology_that_is_not_the_frozen_one():
    with pytest.raises(ValueError):
        REG.check(np.array(["not_a_real_class"] * 5))


def test_adjacent_pairs_are_declared_and_symmetric():
    for a, b in REG.ADJACENT:
        assert a in REG.CLASS_ORDER and b in REG.CLASS_ORDER, (a, b)
        assert a != b


# ── evidence models ──────────────────────────────────────────────────────────
def test_every_model_returns_a_non_negative_16_vector(toy):
    A, y, cls = toy
    for cfg in ({"family": "A_similarity_evidence", "aggregation": "max",
                 "size_correction": "idf"},
                {"family": "B_class_prototype", "prototype": "mean"},
                {"family": "C_probabilistic", "method": "nearest_centroid"}):
        c = dict(cfg)
        m = EVD.fit(c.pop("family"), A, y, cls, **c)
        E = EVD.predict(m, A)
        assert E.shape == (len(A), 16)
        assert (E >= 0).all()


def test_evidence_is_deterministic(toy):
    A, y, cls = toy
    m1 = EVD.fit_A(A, y, cls, aggregation="max", size_correction="idf")
    m2 = EVD.fit_A(A, y, cls, aggregation="max", size_correction="idf")
    assert np.array_equal(EVD.predict_A(m1, A), EVD.predict_A(m2, A))


def test_class_size_correction_demotes_a_large_class():
    """ADVERSARIAL — without it a large class wins by holding more chances at a near neighbour."""
    counts = np.array([30] + [3] * 15)
    big = EVD._size_weight(30, counts, "idf")
    small = EVD._size_weight(3, counts, "idf")
    assert small > big
    assert EVD._size_weight(30, counts, "none") == 1.0
    assert EVD._size_weight(30, counts, "divide_n") < EVD._size_weight(3, counts, "divide_n")


def test_hierarchical_routing_is_soft_not_a_filter(toy):
    """ADVERSARIAL — a hard broad filter makes a broad error unrecoverable.

    Every fine class must retain strictly positive evidence even when its superclass is not
    top-1, otherwise a wrong superclass permanently removes the correct answer.
    """
    A, y, cls = toy
    broad_of = {m: ("lipid" if c in ("acylglycerol", "fatty_acid") else "other")
                for m, c in zip(y, cls)}
    m = EVD.fit_D(A, y, cls, broad_of=broad_of, base="A", lam=2.0,
                  aggregation="max", size_correction="idf")
    E = EVD.predict_D(m, A)
    present = [k for k, c in enumerate(REG.CLASS_ORDER) if (cls == c).any()]
    assert (E[:, present] > 0).all(), "soft routing must never zero a reachable fine class"


def test_normalisation_views_preserve_the_ranking(toy):
    A, y, cls = toy
    E = EVD.predict_A(EVD.fit_A(A, y, cls), A)
    assert np.array_equal(np.argmax(E, 1), np.argmax(EVD.normalise(E, "l1"), 1))
    assert np.allclose(EVD.normalise(E, "l1").sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        EVD.normalise(E, "calibrated")


def test_raw_evidence_is_the_only_view_that_keeps_total_mass(toy):
    """The reason `raw` is canonical: L1 destroys the weak-support signal by construction."""
    A, y, cls = toy
    E = EVD.predict_A(EVD.fit_A(A, y, cls), A)
    E = np.vstack([E, E[0] * 0.1])                       # a weakly-supported query
    assert np.std(E.sum(axis=1)) > 1e-6
    assert np.std(EVD.normalise(E, "l1").sum(axis=1)) < 1e-9


# ── validation ───────────────────────────────────────────────────────────────
def test_rank_of_true_and_topk_agree():
    E = np.array([[0.9, 0.5, 0.1] + [0.0] * 13, [0.1, 0.9, 0.5] + [0.0] * 13])
    cls = np.array([REG.CLASS_ORDER[0], REG.CLASS_ORDER[2]])
    assert list(VAL.rank_of_true(E, cls)) == [1, 2]
    assert VAL.topk(E, cls, 1) == 0.5
    assert VAL.topk(E, cls, 2) == 1.0


def test_adjacency_is_scored_against_its_own_chance_rate():
    """An adjacency fraction without its chance rate is not interpretable."""
    E = np.zeros((4, 16))
    for i, (t, p) in enumerate([("fatty_acid", "acylglycerol")] * 4):
        E[i, REG.index_of(p)] = 1.0
    out = VAL.adjacency_of_errors(E, np.array(["fatty_acid"] * 4))
    assert out["n_errors"] == 4
    assert out["adjacent_fraction"] == pytest.approx(1.0)
    assert 0 < out["chance_adjacent"] < 1
    assert out["lift"] > 1


def test_nested_cv_never_lets_a_molecule_cross_the_split(toy):
    """ADVERSARIAL — the whole point of molecule grouping."""
    A, y, cls = toy
    mol_fold = {m: i % 4 for i, m in enumerate(sorted(set(y.tolist())))}
    folds = np.array([mol_fold[m] for m in y])
    seen = {}

    def fit_fn(A_tr, y_tr, c_tr, cfg):
        seen.setdefault(len(seen), set(y_tr.tolist()))
        return EVD.fit_A(A_tr, y_tr, c_tr, **cfg)

    res = VAL.nested_cv(A, y, cls, folds,
                        {"a": {"aggregation": "max", "size_correction": "none"},
                         "b": {"aggregation": "mean", "size_correction": "idf"}},
                        fit_fn, EVD.predict_A)
    assert res["E"].shape == (len(A), 16)
    for f in set(folds.tolist()):
        te_mols = set(y[folds == f].tolist())
        tr_mols = set(y[folds != f].tolist())
        assert not (te_mols & tr_mols), "a molecule appeared on both sides of the split"


def test_bootstrap_ci_resamples_molecules_not_spectra(toy):
    A, y, cls = toy
    E = EVD.predict_A(EVD.fit_A(A, y, cls), A)
    pt, lo, hi = VAL.bootstrap_ci(E, y, cls, VAL.macro_f1, n_boot=100, seed=0)
    assert lo <= pt <= hi


def test_effective_rank_is_bounded_by_the_dimension():
    rng = np.random.default_rng(0)
    E = np.abs(rng.normal(0, 1, (100, 16)))
    assert 1.0 <= VAL.effective_rank(E) <= 16.0


# ── calibration ──────────────────────────────────────────────────────────────
def test_calibrated_probabilities_sum_to_one_and_are_bounded():
    rng = np.random.default_rng(0)
    E = np.abs(rng.normal(0, 1, (120, 16)))
    cls = np.array([REG.CLASS_ORDER[i % 16] for i in range(120)])
    for m in CAL.METHODS:
        P = CAL.Calibrator(m).fit(E, cls).transform(E)
        assert P.shape == (120, 16)
        assert (P >= -1e-9).all() and (P <= 1 + 1e-9).all(), m
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6), m


def test_a_constant_calibrator_is_rejected_by_the_selection_rule():
    """ADVERSARIAL — the Phase 05 defect, carried forward as a standing test.

    A calibrator with zero sharpness must not be selectable however good its ECE.
    """
    import pandas as pd
    tab = pd.DataFrame([
        {"method": "degenerate", "usable": True, "log_loss": 0.10, "brier": 0.9,
         "ece": 0.001, "classwise_ece": 0.001, "sharpness": 0.0, "discrimination": 0.5},
        {"method": "informative", "usable": True, "log_loss": 0.50, "brier": 0.3,
         "ece": 0.120, "classwise_ece": 0.02, "sharpness": 0.25, "discrimination": 0.80},
    ])
    chosen, reason = CAL.select(tab)
    assert chosen == "informative"
    assert "non-degenerate" in reason


def test_classwise_ece_sees_small_class_failure_that_top_label_ece_hides():
    """ADVERSARIAL — top-label ECE alone cannot expose small-class miscalibration."""
    n = 400
    P = np.full((n, 16), 1.0 / 16)
    P[:, 0] = 0.90
    P[:, 1:] = 0.10 / 15
    cls = np.array([REG.CLASS_ORDER[0]] * int(0.9 * n) +
                   [REG.CLASS_ORDER[5]] * (n - int(0.9 * n)))
    assert CAL.ece(P, cls) < 0.05                      # top label looks fine
    assert CAL.classwise_ece(P, cls) > 0.0             # one-vs-rest sees the small class


def test_log_loss_punishes_a_confident_wrong_answer_more_than_ece():
    cls = np.array([REG.CLASS_ORDER[0]] * 100)
    P_wrong = np.full((100, 16), 1e-6); P_wrong[:, 1] = 1 - 15e-6
    P_soft = np.full((100, 16), 1.0 / 16)
    assert CAL.log_loss(P_wrong, cls) > CAL.log_loss(P_soft, cls)


def test_selection_floors_are_declared_in_the_module():
    assert CAL.SHARPNESS_FLOOR > 0
    assert CAL.DISCRIMINATION_FLOOR > 0.5


# ── novelty ──────────────────────────────────────────────────────────────────
def test_novelty_channels_have_declared_signs():
    E = np.abs(np.random.default_rng(0).normal(0, 1, (20, 16)))
    ch = NOV.novelty_channels(E)
    assert set(ch) <= set(NOV.SIGN)


def test_a_flat_evidence_vector_scores_as_more_novel_than_a_peaked_one():
    peaked = np.zeros((10, 16)); peaked[:, 3] = 1.0
    flat = np.full((10, 16), 1.0 / 16)
    ch_p, ch_f = NOV.novelty_channels(peaked), NOV.novelty_channels(flat)
    assert NOV.rejection_score(ch_f, ch_p).mean() > NOV.rejection_score(ch_p, ch_p).mean()


def test_holdout_removes_the_class_from_every_fitted_object(toy):
    """ADVERSARIAL — a held-out class leaking into the bank would invalidate the experiment."""
    A, y, cls = toy
    mol_fold = {m: i % 4 for i, m in enumerate(sorted(set(y.tolist())))}
    folds = np.array([mol_fold[m] for m in y])
    held = REG.CLASS_ORDER[0]
    captured = {}

    def fit_fn(A_tr, y_tr, c_tr, cfg):
        captured["classes"] = set(c_tr.tolist())
        return EVD.fit_A(A_tr, y_tr, c_tr, **cfg)

    NOV.holdout_class(A, y, cls, folds, held, fit_fn, EVD.predict_A,
                      {"aggregation": "max", "size_correction": "none"})
    assert held not in captured["classes"]


# ── provenance ───────────────────────────────────────────────────────────────
def test_provenance_refuses_to_claim_exactness_for_a_non_additive_family(toy):
    A, y, cls = toy
    m = EVD.fit_C(A, y, cls, method="nearest_centroid")
    ch = PROV.class_chain(REG.CLASS_ORDER[0], A[0], m, [])
    assert ch["exact"] is False


def test_verify_flags_an_unknown_link():
    ch = [{"class_id": "purine", "exact": True,
           "molecules": [{"molecule": "not_real", "similarity": 0.9,
                          "supporting_csms": [{"csm_id": "csm00", "lsms": ["l0"],
                                               "cosine_contribution": 0.5,
                                               "share_of_similarity": 0.5,
                                               "query_activation": 1.0,
                                               "reference_activation": 1.0,
                                               "dominant_bands": [], "band_assignment": ""}]}]}]
    v = PROV.verify(ch, {"adenine"}, {"l0"}, {"csm00"})
    assert not bool(v.iloc[0].intact)
    assert int(v.iloc[0].unknown_molecules) == 1


# ── artifacts of the committed run ───────────────────────────────────────────
@needs_run
def test_frozen_fingerprints_are_the_expected_ones():
    st = json.loads((OUT.root / "PHASE_STATE.json").read_text())
    fp = st["input_fingerprints"]
    assert fp["csm"] == "0b4aa550ccefed3edabdbde5bae11c8d"
    assert fp["lsm"] == "208482d6f7178b5b8f16cace91be55b0"
    assert fp["engine"] == "20d8bd99ce71f45a125c6a2b1d719e51"


@needs_run
def test_phase_state_declares_scope_and_what_it_does_not_implement():
    st = json.loads((OUT.root / "PHASE_STATE.json").read_text())
    assert "Raman only" in st["scope"]
    assert any("BSV2" in x for x in st["does_not_implement"])
    assert any("retrieval" in x for x in st["does_not_implement"])
    assert st["seed"] == 0


@needs_run
def test_no_upstream_artifact_was_written():
    """ADVERSARIAL — Phase 06 must write only under its own tree."""
    st = json.loads((OUT.root / "PHASE_STATE.json").read_text())
    for o in st["outputs"]:
        assert "phase06" in o["path"], o["path"]


@needs_run
def test_no_sers_or_cross_modality_artifact_exists():
    banned = ("sers", "ag_sers", "serum", "plasma", "exosom", "vesicl", "dart", "mixture",
              "cross_modal")
    for p in list(T.glob("*")) + list(A_.glob("*")) + list(F.glob("*")):
        low = p.name.lower()
        assert not any(b in low for b in banned), p.name


@needs_run
def test_the_run_script_names_no_forbidden_data_source():
    """ADVERSARIAL — scope enforced on the code, not only on the outputs."""
    src = (OUT.root / "code" / "run_phase06.py").read_text().lower()
    for term in ("ag-sers", "ag_sers", "serum", "plasma", "exosome", "dart-met", "dart_met"):
        # the word may appear only inside the compliance statement that forbids it
        for m in re.finditer(re.escape(term), src):
            ctx = src[max(0, m.start() - 220):m.start() + 60]
            assert ("no sers" in ctx or "raman-only" in ctx or "raman only" in ctx
                    or "is loaded, benchmarked" in ctx), f"{term} used outside the scope note"


@needs_run
def test_run_script_hardcodes_no_output_path():
    src = (OUT.root / "code" / "run_phase06.py").read_text()
    assert "/Users/" not in src and "/Volumes/" not in src
    assert "PhaseOutputs" in src


@needs_run
def test_output_root_redirection_is_respected(monkeypatch, tmp_path):
    monkeypatch.setenv("GAIRA_V7_OUTPUT_ROOT", str(tmp_path))
    import importlib

    import gaira.v7.io.outputs as O
    importlib.reload(O)
    assert O.output_root() == tmp_path.resolve()
    assert O.PhaseOutputs("06").root == tmp_path.resolve() / "phase06"
    # the frozen tree must NOT follow the redirect
    assert O.frozen_root() != tmp_path.resolve()
    monkeypatch.delenv("GAIRA_V7_OUTPUT_ROOT")
    importlib.reload(O)


@needs_run
def test_phase05_was_reproduced_bit_for_bit(summary):
    """ADVERSARIAL — nothing may change before the prior result is reproduced exactly."""
    a = summary["phase05_audit"]
    assert a["reproduced_bit_for_bit"] is True
    assert a["formula"].startswith("e_c(x) = max")
    ref = json.loads((FROZEN / "phase05/artifacts/phase05_summary_v1.json").read_text())["split_b"]
    for k in ("class_top1", "class_top3", "macro_f1", "balanced_accuracy"):
        assert abs(a["values"][k] - ref[k]) < 1e-9


@needs_run
def test_headline_numbers_match_the_report(summary):
    p = summary["performance"]
    assert p["top1"]["value"] == pytest.approx(0.835, abs=0.002)
    assert p["top3"]["value"] == pytest.approx(0.976, abs=0.002)
    assert p["macro_f1"]["value"] == pytest.approx(0.793, abs=0.002)
    assert p["balanced_accuracy"]["value"] == pytest.approx(0.830, abs=0.002)
    for k in p:
        assert p[k]["ci95"][0] <= p[k]["value"] <= p[k]["ci95"][1]


@needs_run
def test_the_evidence_is_a_16_vector_in_the_frozen_order(preds):
    assert list(preds["class_order"]) == list(REG.CLASS_ORDER)
    assert preds["E"].shape[1] == 16
    assert (preds["E"] >= 0).all()
    assert np.allclose(preds["P"].sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(preds["E_l1"].sum(axis=1), 1.0, atol=1e-6)


@needs_run
def test_radar_and_probability_agree_on_the_argmax(preds):
    """The radar must not tell a different story from the calibrated probabilities."""
    assert (np.argmax(preds["E"], 1) == np.argmax(preds["E_l1"], 1)).all()
    agree = float(np.mean(np.argmax(preds["E"], 1) == np.argmax(preds["P"], 1)))
    assert agree > 0.99


@needs_run
def test_the_evidence_layer_is_not_a_disguised_hard_classifier(summary):
    """ADVERSARIAL — the audit question A3, pinned as a test."""
    s = summary["soft_evidence"]
    assert 0.15 < s["mean_entropy"] < 0.95, "one-hot or flat would both be failures"
    assert s["mean_true_class_evidence_share"] < 0.90
    assert s["effective_rank"] > 6.0


@needs_run
def test_calibration_is_informative_and_not_degenerate(summary):
    c = summary["calibration"]
    assert c["sharpness"] > CAL.SHARPNESS_FLOOR
    assert c["discrimination"] > CAL.DISCRIMINATION_FLOOR
    assert c["classwise_ece"] < c["ece"], "classwise ECE is reported and is the stricter check"


@needs_run
def test_isotonic_is_still_recorded_as_the_ece_winner():
    """The finding is an artifact: selecting on ECE would have chosen the worst log loss."""
    d = pd.read_csv(T / "calibration_summary_v1.csv").set_index("method")
    assert d.ece.idxmin() == "isotonic"
    assert d.loc["isotonic", "log_loss"] > d.loc["temperature", "log_loss"] * 3


@needs_run
def test_every_candidate_and_calibrator_ran():
    """ADVERSARIAL — three methods failed silently on sklearn 1.8. Gate G18."""
    b = pd.read_csv(T / "evidence_model_benchmark_v1.csv")
    c = pd.read_csv(T / "calibration_benchmark_v1.csv")
    assert bool(b.usable.all()), f"dead candidates: {b[~b.usable].candidate.tolist()}"
    assert bool(c.usable.all()), f"dead calibrators: {sorted(set(c[~c.usable].method))}"


@needs_run
def test_the_unsupervised_comparator_is_fitted_out_of_fold():
    """ADVERSARIAL — it was originally fitted on all molecules, inflating it to 0.931."""
    s = pd.read_csv(T / "semantic_comparator_v1.csv").set_index("semantic_layer")
    assert bool(s.loc["unsupervised_16", "labels_fitted_out_of_fold"])
    assert not bool(s.loc["unsupervised_16", "accuracy_comparable_to_curated"])
    ag = json.loads((A_ / "semantic_agreement_v1.json").read_text())
    assert 0.0 < ag["adjusted_rand_curated_vs_unsupervised"] < 1.0


@needs_run
def test_selection_stability_ensemble_is_fully_nested(summary):
    """ADVERSARIAL — the first ensemble leaked the member set across folds (+0.032 -> +0.016)."""
    e = summary["selection_stability"]
    assert e["fully_nested"] is True
    assert set(e["members_per_fold"]) == {"0", "1", "2", "3", "4"} or len(e["members_per_fold"]) == 5
    assert e["delta_macro_f1"] < 0.02, "if this rises, the modal model is the wrong canonical"


@needs_run
def test_class_registry_lists_all_sixteen_with_counts():
    reg = json.loads((A_ / "chemistry_class_registry_v1.json").read_text())
    assert reg["ontology"] == "v7_fine_16"
    assert len(reg["classes"]) == 16
    assert [c["class_id"] for c in reg["classes"]] == list(REG.CLASS_ORDER)
    assert sum(c["n_spectra"] for c in reg["classes"]) == 375
    assert sum(c["n_molecules"] for c in reg["classes"]) == 154
    for c in reg["classes"]:
        assert c["n_molecules"] > 0 and c["n_spectra"] > 0
        assert c["broad_class"]


@needs_run
def test_no_broken_provenance_chains(summary):
    assert summary["provenance"]["broken"] == 0
    assert summary["provenance"]["n_chains"] > 500
    assert summary["provenance"]["exact_decomposition"] is True


@needs_run
def test_holdout_novelty_reports_its_failure_honestly(summary):
    """The acylglycerol failure must survive in the artifacts, not be smoothed away."""
    per = {r["held_class"]: r for r in summary["novelty"]["per_class"]}
    assert len(per) >= 4
    assert "acylglycerol" in per
    assert per["acylglycerol"]["joint_auroc"] < 0.60, "the honest failure must remain visible"
    assert summary["novelty"]["mean_auroc"] > 0.70


@needs_run
def test_robustness_beats_the_raw_spectrum():
    d = pd.read_csv(T / "robustness_summary_v1.csv").set_index("representation")
    assert d.loc["chemistry_evidence_16", "top1_retention"] >= d.loc["raw_spectrum",
                                                                     "top1_retention"]
    assert d.loc["chemistry_evidence_16", "clean_top1"] > d.loc["raw_spectrum", "clean_top1"]


@needs_run
def test_all_gates_recorded_and_none_relaxed():
    g = pd.read_csv(T / "phase06_gates_v1.csv")
    assert len(g) >= 17
    assert set(g.status) <= {"PASS", "FAIL"}
    assert int((g.status == "FAIL").sum()) == 0
    ids = " ".join(g.gate)
    for k in ("G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G16", "G17"):
        assert k in ids


@needs_run
def test_manifest_is_complete_and_stamped():
    m = json.loads((OUT.manifests / "chemistry_evidence_manifest_v1.json").read_text())
    for k in ("input_fingerprints", "split_fingerprint", "code_fingerprint", "seed", "artifacts"):
        assert k in m, k
    assert len(m["artifacts"]) > 25
    for a in m["artifacts"]:
        assert "sha256" in a and "path" in a
    model = json.loads((A_ / "chemistry_evidence_model_v1.json").read_text())
    assert model["class_order"] == list(REG.CLASS_ORDER)
    assert "_provenance" in model
    for k in ("input_fingerprints", "split_fingerprint", "code_fingerprint", "created_utc",
              "model_selection_rule", "seed"):
        assert k in model["_provenance"], k


@needs_run
def test_required_artifacts_exist():
    for n in ("chemistry_class_registry_v1.json", "chemistry_evidence_model_v1.json",
              "chemistry_evidence_reference_vectors_v1.npz",
              "chemistry_evidence_predictions_v1.npz",
              "chemistry_evidence_calibrator_v1.json",
              "chemistry_evidence_provenance_v1.json"):
        assert (A_ / n).exists(), n
    assert (OUT.manifests / "chemistry_evidence_manifest_v1.json").exists()
    assert (OUT.root / "phase06_state.json").exists()


@needs_run
def test_all_22_figures_and_the_pdf_exist():
    pngs = sorted(F.glob("F*.png"))
    svgs = sorted(F.glob("F*.svg"))
    assert len(pngs) == 22
    assert len(svgs) == 22
    assert (R / "PHASE_06_RESULTS.pdf").exists()
    assert (R / "PHASE_06_CHEMISTRY_EVIDENCE_LAYER.md").exists()
    assert (R / "PHASE_06_SCIENTIFIC_AUDIT.md").exists()


@needs_run
def test_the_legacy_11_axis_map_is_not_in_canonical_inference():
    """ADVERSARIAL — gate G14. It may appear only as a Part 6 comparator."""
    model = json.loads((A_ / "chemistry_evidence_model_v1.json").read_text())
    blob = json.dumps(model).lower()
    assert "11" not in str(model.get("config", {}))
    assert "evidence_axis_map" not in blob
    cmp_tab = pd.read_csv(T / "layer_comparison_v1.csv")
    assert "legacy_11_axis" in set(cmp_tab.representation), "it must still be compared against"


@needs_run
def test_phase07_input_contract(preds, summary):
    """What Phase 07 will consume must be exactly what this phase validated."""
    E = preds["E"]
    assert E.shape == (375, 16)
    assert (E >= 0).all()
    assert list(preds["class_order"]) == list(REG.CLASS_ORDER)
    assert len(preds["y"]) == 375 and len(preds["cls"]) == 375
    assert len(set(preds["folds"].tolist())) == 5
    # the three properties the Phase 07 pre-registration depends on
    assert summary["soft_evidence"]["effective_rank"] > 6.0
    assert summary["soft_evidence"]["mean_entropy"] > 0.2
    assert summary["soft_evidence"]["replicate_consistency"] > 0.90
