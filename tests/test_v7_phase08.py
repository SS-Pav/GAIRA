"""GAIRA V7 — Phase 08 regression tests: hierarchical molecular retrieval.

Contract tests on the retrieval modules, artifact tests on the committed run, and adversarial
tests encoding the two defects that would have produced the opposite decision.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gaira.v7.io import PhaseOutputs, frozen_root
from gaira.v7.retrieval import evaluation as EVAL, explain as EXP, models as MOD

OUT = PhaseOutputs("08", extra=("interactive", "manifests"))
T, A_, F, R = OUT.tables, OUT.artifacts, OUT.figures, OUT.reports
FROZEN = frozen_root()
HAS_RUN = (OUT.root / "PHASE_STATE.json").exists()
needs_run = pytest.mark.skipif(not HAS_RUN, reason="Phase 08 has not been run")


@pytest.fixture(scope="module")
def summary():
    return json.loads((A_ / "phase08_summary_v1.json").read_text())


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    grid = np.arange(450.0, 1800.1, 2.0)
    mols = [f"m{i}" for i in range(20)]
    base = np.abs(rng.normal(0, 1, (20, 49)))
    A = np.vstack([np.clip(base[i] + 0.03 * rng.random(49), 0, None)
                   for i in range(20) for _ in range(2)])
    y = np.array([m for m in mols for _ in range(2)])
    E = np.abs(rng.normal(0, 1, (40, 16)))
    X = np.abs(rng.normal(0, 1, (40, len(grid))))
    return A, E, X, y, grid


# ── model contracts ──────────────────────────────────────────────────────────
def test_bank_has_one_row_per_molecule(toy):
    A, _, _, y, _ = toy
    R_, mols = MOD.build_bank(A, y)
    assert R_.shape == (20, 49) and len(mols) == 20
    assert np.allclose(R_[0], A[y == mols[0]].mean(axis=0))


def test_all_similarity_channels_are_bounded_in_unit_interval(toy):
    A, E, X, y, grid = toy
    Ab, mols = MOD.build_bank(A, y)
    Eb, _ = MOD.build_bank(E, y)
    Pq = MOD.prominence(X, grid)
    bands = MOD.molecule_diagnostic_bands(Ab, [{"csm_id": f"csm{i:02d}",
                                                "dominant_bands": [700.0 + 10 * i]}
                                               for i in range(49)])
    for S in (MOD.score_A(X, MOD.build_bank(X, y)[0]), MOD.score_B(A, Ab),
              MOD.band_support(Pq, grid, bands)):
        assert S.min() >= -1e-9 and S.max() <= 1 + 1e-9


def test_incompatibility_is_non_negative_and_never_zeroes_a_candidate():
    """ADVERSARIAL — a hard filter makes a chemistry error unrecoverable (Phase 06 lesson)."""
    Eq = np.array([[0.9, 0.1, 0.0]])
    pen = MOD.incompatibility(Eq, np.array([0, 1, 2]), np.array([0.5, 0.5, 0.5]))
    assert (pen >= 0).all()
    assert pen[0, 0] == 0.0            # ample evidence → no penalty
    assert pen[0, 2] == pytest.approx(0.5)


def test_model_c_never_removes_a_candidate(toy):
    """ADVERSARIAL — no hard filtering. Every molecule must remain rankable."""
    A, E, X, y, grid = toy
    Ab, mols = MOD.build_bank(A, y)
    Eb, _ = MOD.build_bank(E, y)
    Pq = MOD.prominence(X, grid)
    recs = [{"csm_id": f"csm{i:02d}", "dominant_bands": [700.0 + 10 * i]} for i in range(49)]
    bands = MOD.molecule_diagnostic_bands(Ab, recs)
    mc = np.arange(len(mols)) % 16
    rce = np.full(len(mols), 0.4)
    sc = MOD.score_C(A, Ab, E, Eb, Pq, grid, bands, mc, rce,
                     {"alpha": 0.6, "beta": 0.3, "gamma": 0.05, "delta": 0.2}, top_n=5)
    assert sc["total"].shape == (len(A), len(mols))
    assert np.isfinite(sc["total"]).all()
    assert (sc["total"] > -np.inf).all(), "no candidate may be excluded"


def test_zero_chemistry_weights_make_model_c_identical_to_model_b(toy):
    """ADVERSARIAL — the phase's central finding, pinned as a property of the code."""
    A, E, X, y, grid = toy
    Ab, mols = MOD.build_bank(A, y)
    Eb, _ = MOD.build_bank(E, y)
    Pq = MOD.prominence(X, grid)
    recs = [{"csm_id": f"csm{i:02d}", "dominant_bands": [700.0 + 10 * i]} for i in range(49)]
    bands = MOD.molecule_diagnostic_bands(Ab, recs)
    sc = MOD.score_C(A, Ab, E, Eb, Pq, grid, bands, np.arange(len(mols)) % 16,
                     np.full(len(mols), 0.4),
                     {"alpha": 0.4, "beta": 0.0, "gamma": 0.0, "delta": 0.0}, top_n=25)
    rb = EVAL.ranks(MOD.score_B(A, Ab), mols, y)
    rc = EVAL.ranks(sc["total"], mols, y)
    assert np.array_equal(rb, rc)


def test_scores_are_deterministic(toy):
    A, E, X, y, grid = toy
    Ab, _ = MOD.build_bank(A, y)
    assert np.array_equal(MOD.score_B(A, Ab), MOD.score_B(A, Ab))


# ── evaluation contracts ─────────────────────────────────────────────────────
def test_ranks_marks_an_absent_molecule_as_worse_than_the_bank():
    S = np.array([[0.9, 0.5, 0.1]])
    assert EVAL.ranks(S, ["a", "b", "c"], np.array(["b"]))[0] == 2
    assert EVAL.ranks(S, ["a", "b", "c"], np.array(["z"]))[0] == 4


def test_split_a_metrics_are_consistent():
    m = EVAL.split_a_metrics(np.array([1, 1, 2, 5, 40]), 100)
    assert m["top1"] == 0.4 and m["top3"] == 0.6 and m["top5"] == 0.8
    assert m["top1"] <= m["top3"] <= m["top5"] <= m["top10"]
    assert m["median_rank"] == 2.0


def test_paired_test_finds_a_real_difference_and_not_a_fake_one():
    rng = np.random.default_rng(0)
    y = np.array([f"m{i//3}" for i in range(150)])
    a = rng.random(150) < 0.5
    b_same = a.copy()
    b_better = a | (rng.random(150) < 0.3)
    assert not EVAL.paired_test(a, b_same, y, n_boot=200)["significant"]
    assert EVAL.paired_test(a, b_better, y, n_boot=200)["significant"]


def test_identical_models_give_exactly_zero_delta():
    """ADVERSARIAL — the phase reports Δ = +0.0000; that must be exact, not rounded."""
    y = np.array([f"m{i//2}" for i in range(40)])
    a = np.random.default_rng(0).random(40) < 0.6
    r = EVAL.paired_test(a, a.copy(), y, n_boot=100)
    assert r["delta"] == 0.0 and r["ci95"] == [0.0, 0.0] and r["p_value"] == 1.0
    assert not r["significant"]


def test_confidence_from_rank_would_be_circular_and_is_not_used():
    """ADVERSARIAL — conf = 1/rank against correct = (rank<=1) gives discrimination 1.000."""
    rk = np.array([1, 1, 2, 3, 7, 1, 4])
    circular = 1.0 / rk
    correct = (rk <= 1).astype(float)
    assert EVAL.discrimination(circular, correct) == pytest.approx(1.0)
    src = (OUT.root / "code" / "run_phase08.py").read_text() if HAS_RUN else ""
    if src:
        cal = src[src.index("PART 6"):src.index("PART 7")]
        assert "1.0 / rk" not in cal, "confidence must not be derived from the rank"
        assert "margin" in cal


def test_risk_coverage_is_monotone_in_threshold():
    rng = np.random.default_rng(0)
    conf = rng.random(300)
    correct = (rng.random(300) < conf).astype(float)
    rc = EVAL.risk_coverage(conf, correct)
    assert rc.coverage.is_monotonic_decreasing
    assert (rc.risk + rc.accuracy - 1.0).abs().max() < 1e-9


# ── explanation contracts ────────────────────────────────────────────────────
def test_decomposition_reconciles_exactly(toy):
    A, E, X, y, grid = toy
    Ab, mols = MOD.build_bank(A, y)
    Eb, _ = MOD.build_bank(E, y)
    Pq = MOD.prominence(X, grid)
    recs = [{"csm_id": f"csm{i:02d}", "dominant_bands": [700.0 + 10 * i],
             "contributing_lsms": [{"lsm_id": f"l{i}"}]} for i in range(49)]
    bands = MOD.molecule_diagnostic_bands(Ab, recs)
    w = {"alpha": 0.6, "beta": 0.3, "gamma": 0.05, "delta": 0.2}
    sc = MOD.score_C(A, Ab, E, Eb, Pq, grid, bands, np.arange(len(mols)) % 16,
                     np.full(len(mols), 0.4), w, top_n=5)
    j = int(np.argmax(sc["total"][0]))
    d = EXP.decompose(0, j, sc, mols, [f"c{k%16}" for k in range(len(mols))], recs,
                      A[0:1], Ab, [f"ax{k}" for k in range(16)], E[0:1], Eb)
    recon = (1.0 if d["reranked"] else 0.0) + d["terms_subtotal"]
    assert abs(recon - d["score_total"]) < 1e-9
    assert len(d["terms"]) == 4
    assert d["terms"][3]["weight"] < 0, "the penalty must enter negatively"


def test_rank_change_reports_both_directions(toy):
    A, E, X, y, grid = toy
    Ab, mols = MOD.build_bank(A, y)
    Eb, _ = MOD.build_bank(E, y)
    Pq = MOD.prominence(X, grid)
    recs = [{"csm_id": f"csm{i:02d}", "dominant_bands": [700.0 + 10 * i]} for i in range(49)]
    sc = MOD.score_C(A, Ab, E, Eb, Pq, grid, MOD.molecule_diagnostic_bands(Ab, recs),
                     np.arange(len(mols)) % 16, np.full(len(mols), 0.4),
                     {"alpha": 0.5, "beta": 0.4, "gamma": 0.05, "delta": 0.1}, top_n=8)
    t = EXP.rank_change(sc, mols, 0)
    assert {"csm_rank", "final_rank", "moved"} <= set(t.columns)


# ── artifacts of the committed run ───────────────────────────────────────────
@needs_run
def test_fingerprints_and_scope():
    st = json.loads((OUT.root / "PHASE_STATE.json").read_text())
    fp = st["input_fingerprints"]
    assert fp["csm"] == "0b4aa550ccefed3edabdbde5bae11c8d"
    assert fp["lsm"] == "208482d6f7178b5b8f16cace91be55b0"
    assert st["bsv2_used"] is False
    assert "Raman only" in st["scope"]


@needs_run
def test_bsv2_is_absent_from_the_retrieval_package():
    """ADVERSARIAL — the brief's hard constraint: BSV2 is not on the inference path."""
    import gaira.v7.retrieval as pkg
    root = __import__("pathlib").Path(pkg.__file__).parent
    for f in root.glob("*.py"):
        src = f.read_text().lower()
        assert "bsv2" not in src.replace("bsv2 is not", "").replace("bsv2 (phase 07", ""), f.name
        assert "programs" not in src or "programme" in src
    run = (OUT.root / "code" / "run_phase08.py").read_text()
    assert "from gaira.v7.programs" not in run
    assert "import programs" not in run


@needs_run
def test_no_forbidden_modality():
    banned = ("sers", "ag_sers", "serum", "plasma", "exosom", "dart", "umap", "cluster")
    for p in list(T.glob("*")) + list(A_.glob("*")) + list(F.glob("*")):
        assert not any(b in p.name.lower() for b in banned), p.name


@needs_run
def test_baselines_reproduced_exactly(summary):
    """ADVERSARIAL — nothing new may be measured until the prior result reproduces."""
    assert summary["baselines_reproduced"] is True
    b = json.loads((A_ / "baseline_reproduction_v1.json").read_text())
    for k in ("top1", "top3", "top5"):
        assert abs(b["csm"][k] - b["phase05_reference"][f"molecule_{k}"]) < 1e-9


@needs_run
def test_all_five_models_were_benchmarked(summary):
    assert set(summary["split_a"]) == set(MOD.MODELS)
    t = pd.read_csv(T / "split_a_metrics_v1.csv")
    assert len(t) == 5


@needs_run
def test_chemistry_rerank_is_exactly_zero_improvement(summary):
    """ADVERSARIAL — the central finding. If this changes, the decision changes."""
    s = summary["significance_vs_csm"]["C_chemistry_rerank"]
    assert s["top1"]["delta"] == 0.0
    assert s["top1"]["significant"] is False
    assert summary["decision"]["outcome"] == "A"


@needs_run
def test_the_weight_search_chose_zero_chemistry_in_every_fold(summary):
    w = summary["weights_per_fold"]
    assert len(w) == 5
    for f, ww in w.items():
        assert ww["beta"] == 0.0 and ww["gamma"] == 0.0 and ww["delta"] == 0.0, f


@needs_run
def test_chemistry_axis_importance_is_zero():
    imp = pd.read_csv(T / "chemistry_axis_importance_v1.csv")
    assert len(imp) == 16
    assert imp.delta_mrr.abs().max() < 1e-9


@needs_run
def test_failure_analysis_shows_no_movement_in_either_direction():
    """ADVERSARIAL — '94 helped, 0 hurt' was the tell for the bank bug."""
    f = pd.read_csv(T / "rank_changes_v1.csv")
    assert int(f.helped.sum()) == 0 and int(f.hurt.sum()) == 0
    assert int((f.moved == 0).sum()) == len(f)


@needs_run
def test_every_model_was_scored_against_the_same_bank():
    """ADVERSARIAL — gate G7b. Models C/D/E once used a ~123-molecule bank vs B's 154."""
    src = (OUT.root / "code" / "run_phase08.py").read_text()
    assert "kk = np.ones(len(X), bool); kk[i] = False" in src
    assert "keep_all & (tr | (y == y[i]))" not in src, "the restricted bank must be gone"
    z = np.load(A_ / "retrieval_ranks_v1.npz", allow_pickle=True)
    for m in ("B_csm", "C_chemistry_rerank", "D_probabilistic", "E_bayesian_fusion"):
        assert int(z[m].max()) <= 155, f"{m} ranks imply a bank larger than the corpus"


@needs_run
def test_calibration_is_not_circular(summary):
    c = {r["model"]: r for r in summary["calibration"]}
    for m, r in c.items():
        assert r["discrimination"] < 0.99, f"{m}: discrimination 1.000 means circularity"
        assert 0.0 < r["sharpness"] < 1.0


@needs_run
def test_split_b_reports_no_molecule_top1(summary):
    for r in summary["split_b"]:
        assert "chem_top1" in r
        assert "molecule_top1" not in r and "top1" not in r


@needs_run
def test_every_decomposition_reconciles(summary):
    assert summary["explainability"]["non_reconciling"] == 0
    assert summary["explainability"]["decompositions_checked"] >= 60
    d = json.loads((A_ / "evidence_decomposition_v1.json").read_text())
    for e in d["examples"]:
        assert e["reconciles"] is True
        assert len(e["terms"]) == 4


@needs_run
def test_png_only_and_documents_exist():
    assert len(list(F.glob("F*.png"))) == 9
    assert not list(F.glob("*.svg"))
    for n in ("PHASE_08_REPORT.md", "PHASE_08_SCIENTIFIC_AUDIT.md",
              "PHASE_08_DECISION_GATE.md", "PHASE_08_FIGURES.pdf"):
        assert (R / n).exists(), n


@needs_run
def test_all_gates_pass():
    g = pd.read_csv(T / "phase08_gates_v1.csv")
    assert int((g.status == "FAIL").sum()) == 0
    ids = " ".join(g.gate)
    for k in ("G2 baselines reproduced", "G3 BSV2 absent", "G7b", "G7c", "G8 every score"):
        assert k in ids


@needs_run
def test_manifest_complete():
    m = json.loads((OUT.manifests / "retrieval_manifest_v1.json").read_text())
    assert len(m["artifacts"]) > 15
    for a in m["artifacts"]:
        assert "sha256" in a and "path" in a
