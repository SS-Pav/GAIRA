"""GAIRA V7 — Phase 06.5 regression tests: the latent spectral geometry audit.

Contract tests on the latent modules, artifact tests on the committed run, and adversarial tests
encoding the four defects found during the phase. Those say ADVERSARIAL in their docstring.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gaira.v7.io import PhaseOutputs, frozen_root
from gaira.v7.latent import (clustering as CLU, composition as COMP, confounding as CONF,
                             coordinates as COORD, hierarchy as HIER)

OUT = PhaseOutputs("06_5", extra=("interactive", "manifests"))
T, A_, F, R = OUT.tables, OUT.artifacts, OUT.figures, OUT.reports
FROZEN = frozen_root()
HAS_RUN = (OUT.root / "PHASE_STATE.json").exists()
needs_run = pytest.mark.skipif(not HAS_RUN, reason="Phase 06.5 has not been run")


@pytest.fixture(scope="module")
def summary():
    return json.loads((A_ / "phase06_5_summary_v1.json").read_text())


@pytest.fixture(scope="module")
def toy():
    """Four well-separated blobs in a 12-dimensional non-negative space, 10 points each."""
    rng = np.random.default_rng(0)
    M, lab = [], []
    for k in range(4):
        base = np.zeros(12); base[k * 3:(k + 1) * 3] = 1.0
        for _ in range(10):
            M.append(np.clip(base + 0.08 * rng.random(12), 0, None))
            lab.append(k)
    return np.array(M), np.array(lab)


# ── clustering contracts ─────────────────────────────────────────────────────
def test_cosine_distance_has_an_exactly_zero_diagonal(toy):
    """ADVERSARIAL — 1e-11 of residue made sklearn reject the matrix and silhouette went NaN
    across an entire 56-row sweep, hidden by a bare except."""
    M, _ = toy
    D = CLU.cosine_distance(M)
    assert np.array_equal(np.diag(D), np.zeros(len(M)))
    assert np.allclose(D, D.T)
    assert (D >= 0).all()
    from sklearn.metrics import silhouette_score
    silhouette_score(D, np.array([i % 2 for i in range(len(M))]), metric="precomputed")


def test_internal_indices_never_return_a_silent_nan(toy):
    """ADVERSARIAL — an index that cannot be computed must raise, not become NaN."""
    M, lab = toy
    iv = CLU.internal_indices(M, lab)
    assert np.isfinite(iv["silhouette"]), "silhouette must be computed, not swallowed"
    assert np.isfinite(iv["calinski_harabasz"])
    assert iv["n_clusters"] == 4


def test_every_fixed_k_algorithm_returns_the_requested_k(toy):
    M, _ = toy
    for algo in CLU.FIXED_K_ALGORITHMS:
        lab = CLU.fit(algo, M, 4, seed=0)
        assert len({int(v) for v in lab if v >= 0}) == 4, algo


def test_clustering_is_deterministic(toy):
    M, _ = toy
    for algo in CLU.FIXED_K_ALGORITHMS:
        assert np.array_equal(CLU.fit(algo, M, 3, 0), CLU.fit(algo, M, 3, 0)), algo


def test_hdbscan_unassigned_members_are_preserved_not_forced(toy):
    """-1 means 'not in any dense region' and must survive as information."""
    M, _ = toy
    lab = CLU.fit("hdbscan", np.vstack([M, np.full((3, 12), 0.5)]), None, 0, 3)
    assert set(lab.tolist()) - {-1}, "some points must be clustered"
    iv = CLU.internal_indices(np.vstack([M, np.full((3, 12), 0.5)]), lab)
    assert "n_unassigned" in iv


def test_bootstrap_stability_recovers_a_clean_partition(toy):
    M, _ = toy
    bs = CLU.bootstrap_stability(M, "ward", 4, n_boot=10, seed=0)
    assert bs["bootstrap_ari_mean"] > 0.85
    assert bs["consensus"].shape == (len(M), len(M))


def test_cluster_survival_reports_every_cluster(toy):
    M, _ = toy
    sv = CLU.cluster_survival(M, "ward", 4, n_boot=8, seed=0)
    assert len(sv["per_cluster_jaccard"]) == 4
    assert sv["min_survival"] <= sv["mean_survival"]


def test_neighbour_preservation_is_perfect_for_a_perfect_partition(toy):
    M, lab = toy
    assert CLU.neighbour_preservation(M, lab, k=3) > 0.95


def test_membership_entropy_detects_a_degenerate_partition(toy):
    M, _ = toy
    lab = np.zeros(len(M), int); lab[0] = 1
    assert CLU.membership_entropy(lab) < 0.5
    assert CLU.membership_entropy(np.arange(len(M)) % 4) > 0.95


# ── composition contracts ────────────────────────────────────────────────────
def test_classification_flags_acquisition_confounding_over_chemistry():
    """ADVERSARIAL — a cluster purer in source than in chemistry must never be called chemical."""
    rec = {"fine_purity": 0.40, "broad_purity": 0.45, "source_purity": 0.95,
           "excitation_purity": 0.50, "bridge_members": [], "n_molecules": 10,
           "dominant_fine_class": "x", "dominant_broad_class": "y", "fine_entropy": 0.8,
           "within_cluster_mean_distance": 0.2}
    kind, why = COMP.classify(rec, 154, 0.50, 0.50)
    assert kind == "acquisition_confounded"
    assert "source purity" in why


def test_classification_names_a_pure_cluster_chemical():
    rec = {"fine_purity": 0.95, "broad_purity": 1.0, "source_purity": 0.5,
           "excitation_purity": 0.5, "bridge_members": [], "n_molecules": 10,
           "dominant_fine_class": "purine", "dominant_broad_class": "nucleic",
           "fine_entropy": 0.1, "within_cluster_mean_distance": 0.2}
    kind, why = COMP.classify(rec, 154, 0.50, 0.50)
    assert kind == "chemically_coherent" and "purine" in why


def test_every_classification_carries_a_justification():
    for fp in (0.1, 0.4, 0.75, 0.95):
        rec = {"fine_purity": fp, "broad_purity": fp, "source_purity": 0.3,
               "excitation_purity": 0.3, "bridge_members": [], "n_molecules": 8,
               "dominant_fine_class": "x", "dominant_broad_class": "y", "fine_entropy": 0.5,
               "within_cluster_mean_distance": 0.5}
        kind, why = COMP.classify(rec, 154, 0.5, 0.5)
        assert kind in COMP.KINDS and len(why) > 20


# ── confounding contracts ────────────────────────────────────────────────────
def test_permanova_finds_a_real_grouping_and_not_a_random_one(toy):
    M, lab = toy
    D = CLU.cosine_distance(M)
    real = CONF.permanova(D, lab.astype(str), n_perm=199, seed=0)
    rng = np.random.default_rng(0)
    fake = CONF.permanova(D, rng.permutation(lab).astype(str), n_perm=199, seed=0)
    assert real["R2"] > fake["R2"] and real["p_value"] < 0.05
    assert 0.0 <= real["R2"] <= 1.0


def test_ami_is_chance_corrected(toy):
    M, lab = toy
    rng = np.random.default_rng(0)
    tab = CONF.cluster_vs_factor(lab, {"real": lab.astype(str),
                                       "random": rng.integers(0, 4, len(lab)).astype(str)})
    d = tab.set_index("factor")
    assert d.loc["real", "AMI"] > 0.9
    assert abs(d.loc["random", "AMI"]) < 0.3


# ── coordinate contracts ─────────────────────────────────────────────────────
def test_coordinates_lie_on_the_simplex(toy):
    M, lab = toy
    P, _ = COORD.prototypes(M, lab)
    for kern in COORD.KERNELS:
        U = COORD.coordinates(M, P, kern, 0.1, np.arange(M.shape[1]))
        assert U.shape == (len(M), 4)
        assert (U >= 0).all()
        assert np.allclose(U.sum(axis=1), 1.0, atol=1e-8), kern


def test_temperature_controls_sharpness(toy):
    M, lab = toy
    P, _ = COORD.prototypes(M, lab)
    hot = COORD.entropy(COORD.coordinates(M, P, "softmax_cosine", 1.0)).mean()
    cold = COORD.entropy(COORD.coordinates(M, P, "softmax_cosine", 0.02)).mean()
    assert cold < hot


def test_coordinate_selection_rejects_degenerate_settings(toy):
    """ADVERSARIAL — a near-uniform coordinate is reproducible and useless (P-18)."""
    M, lab = toy
    P, _ = COORD.prototypes(M, lab)
    tab = COORD.sweep(M, P, np.arange(len(M)).astype(str),
                      csm_order=np.arange(M.shape[1]))
    kern, temp, why = COORD.select(tab)
    row = tab[(tab.kernel == kern) & (tab.temperature == temp)].iloc[0]
    assert 0.10 < row.mean_entropy < 0.90, "a degenerate setting must not be selectable"
    assert row.fraction_degenerate < 0.05


def test_prototypes_ignore_unassigned_members(toy):
    M, lab = toy
    lab2 = lab.copy(); lab2[:5] = -1
    P, ids = COORD.prototypes(M, lab2)
    assert len(ids) == len(set(int(v) for v in lab2 if v >= 0))


def test_neighbour_preservation_is_one_for_an_identity_map(toy):
    M, _ = toy
    assert COORD.neighbour_preservation(M, M, k=5) == pytest.approx(1.0)


# ── hierarchy contracts ──────────────────────────────────────────────────────
def test_modularity_beats_a_degree_preserving_null(toy):
    M, _ = toy
    m = HIER.modularity_vs_null(M, k=3, n_null=30, seed=0)
    assert m["modularity"] > m["null_mean"]
    assert m["z_score"] > 2


def test_gap_statistic_detects_separated_blobs(toy):
    M, _ = toy
    assert HIER.gap_statistic(M)["valley_depth"] > 0.2


def test_intrinsic_dimension_reports_both_estimators(toy):
    M, _ = toy
    d = HIER.intrinsic_dimension(M)
    assert "levina_bickel_mle" in d and "correlation_dimension" in d
    assert "estimators_agree" in d, "disagreement must be surfaced, not hidden"
    assert d["ambient_dimension"] == 12


# ── artifacts of the committed run ───────────────────────────────────────────
@needs_run
def test_phase_state_declares_audit_only():
    st = json.loads((OUT.root / "PHASE_STATE.json").read_text())
    assert st["audit_only"] is True
    assert st["architecture_changed"] is False
    assert st["phase07_begun"] is False
    assert "Raman only" in st["scope"]


@needs_run
def test_frozen_fingerprints_verified():
    st = json.loads((OUT.root / "PHASE_STATE.json").read_text())
    fp = st["input_fingerprints"]
    assert fp["csm"] == "0b4aa550ccefed3edabdbde5bae11c8d"
    assert fp["lsm"] == "208482d6f7178b5b8f16cace91be55b0"
    assert fp["engine"] == "20d8bd99ce71f45a125c6a2b1d719e51"


@needs_run
def test_no_upstream_artifact_was_written():
    st = json.loads((OUT.root / "PHASE_STATE.json").read_text())
    for o in st["outputs"]:
        assert "phase06_5" in o["path"], o["path"]


@needs_run
def test_no_chemistry_label_entered_the_construction():
    """ADVERSARIAL — the phase's central methodological claim, pinned as a test."""
    src = (OUT.root / "code" / "run_phase06_5.py").read_text()
    build = src[src.index("SECTION 1"):src.index("SECTION 2")]
    for banned in ("cls_m", "broad_m", "cls_of", "fine_class"):
        assert banned not in build, f"{banned} appears in the geometry construction section"
    z = np.load(A_ / "continuous_coordinates_v1.npz", allow_pickle=True)
    assert "prototypes" in z.files


@needs_run
def test_no_forbidden_modality_appears():
    banned = ("sers", "ag_sers", "serum", "plasma", "exosom", "vesicl", "dart", "mixture")
    for p in list(T.glob("*")) + list(A_.glob("*")) + list(F.glob("*")):
        assert not any(b in p.name.lower() for b in banned), p.name


@needs_run
def test_png_only_no_svg():
    """This phase reverts to the PNG-only figure policy."""
    assert len(list(F.glob("*.png"))) == 14
    assert not list(F.glob("*.svg"))


@needs_run
def test_no_index_has_an_interior_optimum(summary):
    """ADVERSARIAL — the phase's headline finding. If this changes, the conclusion changes."""
    ks = summary["k_selection"]
    assert ks["no_preferred_k"] is True
    assert ks["n_indices_with_interior_optimum"] == 0
    mono = pd.read_csv(T / "k_selection_monotonicity_v1.csv")
    assert not mono.has_interior_optimum.any()
    sil = mono[mono["index"] == "silhouette"].iloc[0]
    assert sil.spearman_rho_vs_K > 0.95, "silhouette must be monotone increasing in K"


@needs_run
def test_k16_is_recorded_as_a_convention_not_a_discovery(summary):
    """ADVERSARIAL — an earlier rule chose K=4 by maximising bootstrap ARI (P-18 trap)."""
    assert summary["k_selection"]["canonical_K_is_a_convention"] is True
    assert summary["canonical_partition"]["K"] == 16
    assert summary["canonical_partition"]["selected_without_labels"] is True


@needs_run
def test_chemistry_dominates_acquisition(summary):
    vp = {r["factor"]: r["marginal_R2"] for r in summary["confounding"]["variance_partition"]}
    assert summary["confounding"]["chemistry_dominates"] is True
    assert vp["fine_chemistry"] > 2 * vp["excitation"]
    assert vp["fine_chemistry"] > 5 * vp["source"]


@needs_run
def test_continuous_coordinates_beat_hard_cluster_ids(summary):
    c = summary["coordinates"]
    assert c["neighbour_preservation_k10"] > c["neighbour_preservation_hard_ids"]
    assert c["effective_rank"] > c["effective_rank_hard_ids"]
    assert 0.05 < c["mean_entropy"] < 0.95


@needs_run
def test_the_retrieval_gain_is_reported_as_not_significant(summary):
    """ADVERSARIAL — the phase would have recommended Option C without this test."""
    sg = summary["retrieval_significance"]
    for task in ("molecule", "chemistry"):
        assert "ci95" in sg[task] and "p_value" in sg[task]
        assert sg[task]["significant"] is False, "if this becomes True, revisit the recommendation"
        assert sg[task]["ci95"][0] <= sg[task]["delta"] <= sg[task]["ci95"][1]


@needs_run
def test_recommendation_is_option_a_and_evidence_based(summary):
    r = summary["recommendation"]
    assert r["option"] == "Option A"
    assert "significantly" in r["rationale"]
    crit = summary["section9_criteria"]
    assert crit["retrieval_improvement"]["pass"] is False
    assert crit["reproducibility"]["pass"] is True
    assert crit["stability"]["pass"] is True


@needs_run
def test_agreement_is_recorded_as_conditional_on_k():
    """ADVERSARIAL — AMI 0.703 is one point on a monotone curve peaking at K=24."""
    ag = json.loads((A_ / "geometry_chemistry_agreement_v1.json").read_text())
    assert ag["agreement_is_conditional_on_K"] is True
    assert ag["ami_peaks_at_K"] != 16, "if AMI peaked at 16 the framing would change"
    curve = pd.read_csv(T / "agreement_vs_k_v1.csv")
    m = curve.groupby("K").AMI.mean()
    assert m.loc[2] < m.loc[16] < m.max()


@needs_run
def test_split_a_singletons_are_counted_not_dropped():
    """ADVERSARIAL — 66 of 154 molecules vanish from the bank; dropping them inflates MRR."""
    t = pd.read_csv(T / "retrieval_benchmark_v1.csv")
    assert "n_unretrievable_singletons" in t.columns
    assert int(t.n_unretrievable_singletons.max()) > 50


@needs_run
def test_clustering_was_refitted_inside_every_training_fold():
    src = (OUT.root / "code" / "run_phase06_5.py").read_text()
    sec6 = src[src.index("SECTION 6 —"):src.index("SECTION 7 —")]
    assert "CLU.fit(ALGO, Mtr" in sec6, "the clustering must be refitted per fold"
    assert "tr_mols = sorted(set(y[tr]" in sec6, "prototypes must use training molecules only"


@needs_run
def test_all_gates_pass_and_none_relaxed():
    g = pd.read_csv(T / "phase06_5_gates_v1.csv")
    assert int((g.status == "FAIL").sum()) == 0
    ids = " ".join(g.gate)
    for k in ("G3 no chemistry label", "G6c", "G9 chemistry dominates", "G14 Phase 07 not begun"):
        assert k in ids


@needs_run
def test_reports_and_pdf_exist():
    assert (R / "PHASE_06_5_LATENT_GEOMETRY_AUDIT.md").exists()
    assert (R / "PHASE_06_5_SCIENTIFIC_AUDIT.md").exists()
    assert (R / "PHASE_06_5_LATENT_GEOMETRY_AUDIT.pdf").exists()
    audit = (R / "PHASE_06_5_SCIENTIFIC_AUDIT.md").read_text()
    for section in ("Strongly supported", "Weakly supported", "Unsupported"):
        assert section in audit, f"the audit must classify conclusions: {section}"


@needs_run
def test_manifest_complete():
    m = json.loads((OUT.manifests / "latent_geometry_manifest_v1.json").read_text())
    assert len(m["artifacts"]) > 20
    for a in m["artifacts"]:
        assert "sha256" in a and "path" in a
