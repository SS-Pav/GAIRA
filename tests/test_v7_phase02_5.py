"""GAIRA V7 — Phase 02.5 tests: latent geometry.

The properties that matter here are the two firewalls and determinism. This phase's whole claim
is that the geometry was found without looking at chemistry or source, so the tests that earn
their place are the ones that would catch a label leaking into a representation:

    test_no_representation_uses_a_chemistry_label
    test_no_representation_uses_a_source_label
    test_provenance_view_excludes_class_identity
    test_pca_is_deterministic_and_sign_stable
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

PH = REPO / "GAIRA_v7_rebuild/results/phase_02_5_latent_geometry"
T, A, V, F = PH / "tables", PH / "artifacts", PH / "validation", PH / "figures"
P01 = REPO / "results/v7_rebuild/phase01"
P02 = REPO / "results/v7_rebuild/phase02"
ATLAS_FP = "09ed804a40836f4a05a91ba10900cded"
LSM_FP = "208482d6f7178b5b8f16cace91be55b0"
CSM_FP = "0b4aa550ccefed3edabdbde5bae11c8d"

from gaira.v7.geometry import embedding as EMB      # noqa: E402
from gaira.v7.geometry import fusion as FUS         # noqa: E402
from gaira.v7.geometry import metrics as MET        # noqa: E402
from gaira.v7.geometry import neighbourhoods as NBH  # noqa: E402
from gaira.v7.geometry import nulls as NUL          # noqa: E402
from gaira.v7.geometry import representations as REP  # noqa: E402
from gaira.v7.geometry import structure as STR      # noqa: E402

ran = pytest.mark.skipif(not (A / "phase_02_5_manifest_v1.json").is_file(),
                         reason="Phase 02.5 has not been run in this checkout")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((A / "phase_02_5_manifest_v1.json").read_text())


@pytest.fixture(scope="module")
def state():
    return json.loads((PH / "PHASE_STATE.json").read_text())


@pytest.fixture(scope="module")
def priors():
    return json.loads((A / "phase03_geometry_priors.json").read_text())


@pytest.fixture(scope="module")
def geom():
    return np.load(A / "geometry_v1.npz", allow_pickle=True)


# ── A. FROZEN INPUTS ─────────────────────────────────────────────────────────
@ran
def test_frozen_fingerprints_verified(manifest):
    f = manifest["frozen_inputs"]
    assert f["atlas_fingerprint"] == ATLAS_FP
    assert f["lsm_registry_fingerprint"] == LSM_FP
    assert f["csm_dictionary_fingerprint"] == CSM_FP


@ran
def test_phase01_and_phase02_assets_unmodified():
    """Phase 02.5 must not write upstream. Verified against the fingerprints they publish."""
    assert json.loads((P01 / "PHASE_STATE.json").read_text())["registry_fingerprint"] == LSM_FP
    assert json.loads((P02 / "PHASE_STATE.json").read_text())["csm_fingerprint"] == CSM_FP


@ran
def test_phase_declares_itself_analysis_only(manifest, state):
    assert manifest["nature"].startswith("ANALYSIS ONLY")
    assert state["analysis_only"] is True
    assert state["themes_created"] is False


# ── B. THE TWO FIREWALLS ─────────────────────────────────────────────────────
def _code(mod) -> str:
    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree).lower()


@pytest.mark.parametrize("fn_name", ["spectral_profile", "band_family", "activation_view",
                                     "peak_vector", "peak_summary", "edge_feature_view"])
def test_no_representation_uses_a_chemistry_label(fn_name):
    """None of the primary views may take a class label at all — not even to ignore it."""
    fn = getattr(REP, fn_name)
    params = set(inspect.signature(fn).parameters)
    forbidden = {"chemical_class", "classes", "class_of", "labels", "fine_class"}
    assert not (params & forbidden), f"{fn_name} accepts a chemistry label: {params & forbidden}"


def test_no_representation_uses_a_source_label():
    for name in ("spectral_profile", "band_family", "activation_view", "peak_vector"):
        params = set(inspect.signature(getattr(REP, name)).parameters)
        assert not (params & {"source", "sources", "sources_of", "excitation", "excit_of"})


def test_provenance_view_excludes_class_identity():
    """The provenance view carries breadth, never which class a motif came from.

    A provenance feature encoding the class would re-encode the Phase 00 partition into the
    geometry, and every "discovered" community would be the partition looking back (R-01).
    """
    src = _code(REP)
    fn = inspect.getsource(REP.provenance_view)
    assert "chemical_class" not in fn
    assert "class_of" not in fn
    # the columns it emits must be counts/entropies and source/excitation shares only
    import numpy as _np
    meta = [{"analytes": ["a", "b"]}, {"analytes": ["b"]}]
    V, cols = REP.provenance_view(meta, {"a": ["s1"], "b": ["s2"]},
                                  {"a": ["532"], "b": ["785"]}, {"a": 2, "b": 1})
    assert not any("class" in c for c in cols)
    assert V.shape[0] == 2


def test_reconstruction_view_uses_class_only_to_select_a_dictionary():
    """`class_of` appears here, and the docstring must say why it is not a feature."""
    doc = REP.reconstruction_contribution.__doc__
    assert "not" in doc.lower() and "distance" in doc.lower()


@ran
def test_manifest_records_both_firewalls(manifest):
    fw = manifest["firewalls"]
    assert fw["chemistry_labels_used_in_fitting"] is False
    assert fw["source_labels_used_in_fitting"] is False
    assert fw["revealed_at_step"] == 8


# ── C. DETERMINISM ───────────────────────────────────────────────────────────
def test_pca_is_deterministic_and_sign_stable():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(30, 60))
    a, b = EMB.fit_pca(V, 5), EMB.fit_pca(V, 5)
    assert np.allclose(a["scores"], b["scores"])
    assert np.allclose(a["loadings"], b["loadings"])
    # the sign convention makes the largest-magnitude loading positive in every component
    for row in a["loadings"]:
        assert row[np.argmax(np.abs(row))] > 0


def test_pca_uses_a_full_svd_not_a_randomised_solver():
    assert "full" in inspect.getsource(EMB.fit_pca)


def test_distance_matrices_are_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(1)
    H = np.abs(rng.normal(size=(8, 120))) + 0.05
    grid = np.linspace(450, 1800, 120)
    Ds = MET.all_distances(H, grid, H, H, np.abs(rng.normal(size=(20, 8))),
                           np.abs(rng.normal(size=(8, 8))))
    for m, Dm in Ds.items():
        assert np.allclose(Dm, Dm.T), m
        assert np.allclose(np.diag(Dm), 0.0), m
        assert (Dm >= 0).all(), m


def test_nulls_are_reproducible_from_a_seed():
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    H = np.abs(np.sin(np.linspace(0, 9, 200)))[None, :].repeat(5, 0) + 0.1
    assert np.allclose(NUL.band_position_null(H, rng_a), NUL.band_position_null(H, rng_b))
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    W = np.abs(np.random.default_rng(0).normal(size=(10, 10)))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    assert np.allclose(NUL.degree_preserving_graph_null(W, rng_a),
                       NUL.degree_preserving_graph_null(W, rng_b))


def test_degree_preserving_null_actually_preserves_degree():
    """The earlier version permuted all weights, which is a different null entirely."""
    import networkx as nx
    rng = np.random.default_rng(0)
    W = np.zeros((12, 12))
    for i in range(12):
        for j in ((i + 1) % 12, (i + 3) % 12):
            W[i, j] = W[j, i] = 1.0
    Wn = NUL.degree_preserving_graph_null(W, rng)
    assert ((W > 0).sum(axis=1) == (Wn > 0).sum(axis=1)).all()


def test_jensen_shannon_is_finite_for_overlapping_gaussians():
    """scipy returns inf here; the direct computation must not."""
    grid = np.linspace(450, 1800, 676)
    g = lambda c: np.exp(-((grid - c) ** 2) / (2 * 12.0 ** 2))
    Dm = MET.d_jensen_shannon(np.vstack([g(1000), g(1006)]))
    assert np.isfinite(Dm).all()
    assert 0 < Dm[0, 1] < 0.9


def test_wasserstein_is_normalised_by_the_grid_span_not_the_matrix_max():
    """Normalising by the matrix max makes any two-motif probe return 1.0 by construction."""
    grid = np.linspace(450, 1800, 676)
    g = lambda c: np.exp(-((grid - c) ** 2) / (2 * 12.0 ** 2))
    d = MET.d_wasserstein(np.vstack([g(1000), g(1006)]), grid=grid)[0, 1]
    assert d < 0.05, f"a 6 cm-1 shift should be a small EMD, got {d}"


def test_probe_scores_are_scale_free():
    grid = np.linspace(450, 1800, 300)
    H = np.abs(np.random.default_rng(0).normal(size=(10, 300))) + 0.05
    D1 = MET.d_spectral_cosine(H)
    p = MET.scale_free_probes(MET.d_spectral_cosine, grid, D1)
    assert set(p) >= {"median_observed_distance", "background_separation", "peak_shift_cost"}
    assert p["median_observed_distance"] > 0


# ── D. ARTIFACT SHAPES AND COMPLETENESS ──────────────────────────────────────
@ran
def test_the_two_primary_geometries_are_named_distinctly(manifest, geom):
    """The metric geometry and the fused geometry are different objects and must not share a
    key — the neighbourhood cards are built from one of them, not the other."""
    assert "D_primary" not in geom.files
    assert manifest["neighbourhoods_computed_on"] == "D_primary_metric"
    assert 0.0 <= manifest["metric_vs_fused_knn_agreement"] <= 1.0


@ran
def test_all_50_lsms_are_represented(geom):
    ids = [str(s) for s in geom["motif_ids"]]
    assert len(ids) == 50 and len(set(ids)) == 50
    assert geom["D_primary_metric"].shape == (50, 50)
    assert geom["D_primary_geometry"].shape == (50, 50)


@ran
def test_all_49_csms_are_represented():
    s = json.loads((A / "csm_sensitivity_v1.json").read_text())
    assert s["n_csms"] == 49
    assert 0.0 <= s["neighbour_agreement_lsm_vs_csm"] <= 1.0


@ran
def test_embedding_artifacts_have_the_right_shapes():
    e = np.load(A / "embeddings_v1.npz", allow_pickle=True)
    assert e["umap"].shape == (50, 2)
    assert e["diffusion"].shape[0] == 50 and e["diffusion"].shape[1] >= 2
    assert e["spectral"].shape[0] == 50
    p = np.load(A / "pca_v1.npz", allow_pickle=True)
    assert p["spectral_profile_scores"].shape[0] == 50
    assert p["spectral_profile_loadings"].shape[1] == 676


@ran
def test_neighbour_cards_cover_every_motif_with_k_neighbours():
    c = pd.read_csv(T / "nearest_neighbour_cards_v1.csv")
    assert c.motif.nunique() == 50
    assert set(c.groupby("motif").size()) == {5}
    assert c.relationship_tier.isin(NBH.RELATIONSHIP_TIERS).all()


@ran
def test_neighbours_are_reproducible_from_the_stored_geometry(geom):
    """The published cards must match what the published distance matrix says."""
    ids = [str(s) for s in geom["motif_ids"]]
    D = geom["D_primary_metric"]        # the matrix the cards were actually built from
    c = pd.read_csv(T / "nearest_neighbour_cards_v1.csv")
    for i, m in enumerate(ids):
        expected = [ids[j] for j in np.argsort(D[i])[1:6]]
        got = c[c.motif == m].sort_values("rank").neighbour.tolist()
        assert got == expected, m


@ran
@pytest.mark.parametrize("name", [
    "metric_comparison_v1.csv", "pca_components_v1.csv", "umap_stability_sweep_v1.csv",
    "embedding_quality_v1.csv", "cluster_sweep_v1.csv", "graph_roles_v1.csv",
    "geometry_regions_v1.csv", "nearest_neighbour_cards_v1.csv",
    "rejected_proposal_geometry_v1.csv", "proposal_gradients_v1.csv",
    "multiview_comparison_v1.csv"])
def test_table_exists(name):
    assert (T / name).is_file()


@ran
def test_manifest_lists_every_output(manifest):
    assert len(manifest["outputs"]) >= 15
    for o in manifest["outputs"]:
        assert (REPO / o["path"]).is_file(), o["path"]
        assert o["sha256"]


# ── E. CONFOUNDING WAS ASSESSED, NOT ASSUMED ─────────────────────────────────
@ran
def test_source_and_excitation_confounding_were_tested():
    c = pd.read_csv(V / "confounding_v1.csv")
    assert set(c.label) == {"chemistry_class", "source", "excitation"}
    assert c.permanova_p.notna().all()
    assert c.knn_accuracy.notna().all()


@ran
def test_chemistry_explains_more_of_the_geometry_than_source():
    c = pd.read_csv(V / "confounding_v1.csv").set_index("label")
    assert c.loc["chemistry_class", "permanova_R2"] > c.loc["source", "permanova_R2"]


@ran
def test_leave_one_out_geometries_were_computed():
    for f in ("leave_one_source_out_v1.csv", "leave_one_excitation_out_v1.csv"):
        d = pd.read_csv(V / f)
        assert len(d) > 0
        assert "testable" in d.columns


@ran
def test_untestable_motifs_are_counted(state):
    assert state["n_single_source_motifs"] >= 0


# ── F. PRIORS ARE PRIORS, NOT THEMES ─────────────────────────────────────────
@ran
def test_no_final_theme_labels_are_created(priors, state):
    assert "PROVISIONAL" in priors["status"]
    assert state["themes_created"] is False
    for p in priors["priors"]:
        assert p["status"].startswith("PROVISIONAL")
        assert "theme" not in p["prior_id"].lower() or "prior" in p["prior_id"].lower()


@ran
def test_every_prior_has_the_required_fields(priors):
    required = {"prior_id", "provisional_name", "supporting_lsms", "supporting_csms",
                "geometry_type", "shared_bands_cm1", "evidence_strength",
                "source_confounding", "confidence", "must_not_hard_merge", "notes"}
    for p in priors["priors"]:
        assert required <= set(p), f"{p['prior_id']} missing {required - set(p)}"
        assert p["geometry_type"] in NBH.GEOMETRY_TYPES
        assert 0.0 <= p["confidence"] <= 1.0


@ran
def test_priors_resolve_to_real_motifs(priors, geom):
    ids = set(str(s) for s in geom["motif_ids"])
    for p in priors["priors"]:
        assert p["supporting_lsms"], p["prior_id"]
        assert set(p["supporting_lsms"]) <= ids, p["prior_id"]


@ran
def test_the_phase02_equivalence_is_carried_forward_and_protected(priors):
    p = next(x for x in priors["priors"] if x["prior_id"] == "prior_cis_unsaturation")
    assert p["geometry_type"] == "discrete"
    assert len(p["supporting_lsms"]) == 2
    assert p["confidence"] >= 0.8


@ran
def test_rejected_phase02_groups_became_priors_with_do_not_merge_lists(priors):
    for pid in ("prior_lipid_superfamily", "prior_polar_skeletal_backbone",
                "prior_heterocyclic_ring_system"):
        p = next(x for x in priors["priors"] if x["prior_id"] == pid)
        assert p["must_not_hard_merge"], f"{pid} must carry the Phase 02 distinctions forward"
        assert "rejected" in p["notes"].lower() or "rejected" in str(p["evidence"]).lower()


# ── G. STRUCTURE HELPERS ─────────────────────────────────────────────────────
def test_conductance_is_low_for_an_island_and_high_for_a_smear():
    n = 20
    D = np.full((n, n), 1.0)
    D[:10, :10] = 0.05
    D[10:, 10:] = 0.05
    np.fill_diagonal(D, 0.0)
    island = STR.graph_conductance(D, [list(range(10))])[0]
    mixed = STR.graph_conductance(D, [list(range(0, 20, 2))])[0]
    assert island < mixed


def test_local_intrinsic_dimension_is_low_on_a_line():
    x = np.linspace(0, 1, 40)[:, None]
    D = np.abs(x - x.T)
    lid = STR.local_intrinsic_dimension(D, k=8)
    assert np.nanmean(lid) < 3.0


def test_classify_region_thresholds_are_stated_not_free():
    src = inspect.getsource(STR.classify_region)
    for token in ("0.35", "1.6", "0.55", "0.4"):
        assert token in src


def test_permanova_returns_a_p_value_and_detects_real_separation():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.2, (15, 4))
    b = rng.normal(4, 0.2, (15, 4))
    X = np.vstack([a, b])
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    r = NBH.permanova(D, ["a"] * 15 + ["b"] * 15, n_perm=199, seed=0)
    assert r["p"] < 0.05 and r["R2"] > 0.5
    r2 = NBH.permanova(D, ["a"] * 30, n_perm=99, seed=0)
    assert "not testable" in r2["note"]


def test_pareto_weights_sum_to_one():
    assert abs(sum(w for w, _ in FUS.CRITERIA.values()) - 1.0) < 1e-9


# ── H. FIGURES AND REPORT ────────────────────────────────────────────────────
@ran
@pytest.mark.parametrize("n", range(1, 26))
def test_figure_exists(n):
    hits = sorted(F.glob(f"fig{n:02d}_*.png"))
    assert hits, f"figure {n:02d} missing"
    assert hits[0].with_suffix(".svg").is_file()


@ran
def test_interactive_view_is_self_contained():
    h = (PH / "interactive/motif_geometry.html").read_text()
    assert "<script>" in h and "http://" not in h.replace("http://www.w3.org/2000/svg", "")


@ran
def test_report_exists_and_states_the_geometry_verdict():
    r = PH / "reports/PHASE_02_5_LATENT_GEOMETRY_REPORT.md"
    assert r.is_file()
    t = r.read_text()
    for term in ("firewall", "PERMANOVA", "wasserstein", "bridge", "isolated",
                 "Phase 03", "continuum"):
        assert term.lower() in t.lower(), term
