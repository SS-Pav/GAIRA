"""GAIRA V7 — Phase 01 tests: balanced references → class-local NMF → Local Spectral Motifs.

The most important tests here are the ARCHITECTURE tests. Phase 01 was once implemented
against a different architecture (decomposing the frozen atlas) and had to be reclassified as
a control experiment. These tests exist so that cannot recur silently:

    test_frozen_atlas_is_not_an_input_to_the_lsm_package
    test_lsm_is_a_row_of_a_class_local_fit_not_a_restriction
    test_registry_is_indexed_by_chemistry_class
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

P01 = REPO / "results/v7_rebuild/phase01"
T, A, F = P01 / "tables", P01 / "artifacts", P01 / "figures"
FOUNDATION = REPO / "assets" / "foundation"
CANONICAL_ATLAS_FINGERPRINT = "09ed804a40836f4a05a91ba10900cded"

from gaira.v7.lsm import classlocal as CLS      # noqa: E402
from gaira.v7.lsm import discovery as DIS       # noqa: E402
from gaira.v7.lsm import references as REF      # noqa: E402
from gaira.v7.lsm import serialization as SER   # noqa: E402
from gaira.v7.lsm.lsm import LSM, classify_type, dominant_bands   # noqa: E402
from gaira.v7.lsm.registry import LSMRegistry   # noqa: E402

ran = pytest.mark.skipif(not (A / "lsm_manifest_v1.json").is_file(),
                         reason="Phase 01 has not been run in this checkout")


# ── A. ARCHITECTURE — the tests that would have caught the original drift ─────
def _executable_source(mod) -> str:
    """Module source with docstrings and comments stripped — code only.

    Checked against CODE rather than prose: these modules legitimately *discuss* the frozen
    atlas in their documentation (explaining what V5 did and why V7 differs). What must never
    appear is a line that actually loads it.
    """
    import ast
    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree).lower()


def test_frozen_atlas_is_not_an_input_to_the_lsm_package():
    """P-15: the frozen atlas is a control, never a foundation.

    No module in the canonical LSM package may LOAD the frozen basis. Docstrings may discuss
    it; code may not touch it. If a future change reintroduces it, this fails.
    """
    banned = ("manifold_components", "assets/foundation", "assets\\foundation",
              "09ed804a", "foundation_dir", "canonical_atlas_fingerprint")
    for mod in (REF, CLS, DIS, SER):
        src = _executable_source(mod)
        for b in banned:
            assert b not in src, (
                f"{mod.__name__} LOADS the frozen atlas ({b!r}) — the LSM layer must be built "
                f"from balanced references alone (P-15)")


def test_lsm_is_a_row_of_a_class_local_fit_not_a_restriction():
    """An LSM must be free to occupy directions no existing component contains.

    A decomposition of the frozen atlas is bounded by it; a class-local fit is not. Fitting a
    class whose spectra have a band the atlas never modelled must place mass on that band.
    """
    D = 676
    grid = np.linspace(450.0, 1800.0, D)
    X = np.zeros((6, D))
    for i in range(6):
        X[i, 600 + i] = 1.0                       # a region nothing else occupies
        X[i, 100] = 0.5
    W, H, _ = CLS.fit_nmf(X, 2, seed=0)
    assert H.shape == (2, D)
    assert np.all(H >= 0)
    assert H[:, 595:610].sum() > 0, "the fit must be free to model bands from its own data"


def test_registry_is_indexed_by_chemistry_class():
    """Contract C-05: one LSM dictionary PER CLASS, not per atlas component."""
    m = _mk("protein.m00", "protein", 3)
    reg = LSMRegistry([_res("protein", [m])], "v", {}, "arm")
    tbl = reg.motif_table()
    assert "chemical_class" in tbl.columns
    assert "parent_component" not in tbl.columns, "class-indexed, never component-indexed"
    assert tbl.motif_id.iat[0].startswith("protein.")


def test_no_cross_class_clustering_in_phase01():
    """Cross-class integration is Phase 02. Phase 01 must not contain a similarity graph."""
    banned = ("consensus", "similarity_graph", "leiden", "louvain", "community")
    for mod in (DIS, CLS, REF):
        src = inspect.getsource(mod).lower()
        for b in banned:
            assert f"def {b}" not in src and f"{b}(" not in src, (
                f"{mod.__name__} appears to do cross-class work ({b}) — that is Phase 02")


# ── B. balanced references ────────────────────────────────────────────────────
def _toy_corpus():
    D = 676
    rng = np.random.default_rng(0)
    rows, recs, qs = [], [], []
    for cid, n in (("m1", 3), ("m2", 1), ("m3", 2)):
        for r in range(n):
            x = np.zeros(D)
            x[100] = 1.0 + 0.05 * r
            x[300] = 0.5
            rows.append(x)
            sid = f"{cid}::{r}"
            recs.append({"spectrum_id": sid, "canonical_id": cid, "analyte": cid,
                         "excitation_nm": 785.0, "source": "S"})
            qs.append({"spectrum_id": sid, "quality_score": 0.5 + 0.1 * r})
    return np.vstack(rows), pd.DataFrame(recs), pd.DataFrame(qs)


@pytest.mark.parametrize("arm", REF.ARMS)
def test_every_arm_builds_and_is_wellformed(arm):
    X, meta, q = _toy_corpus()
    rows, rmeta = REF.build_arm(arm, X, meta, q)
    assert rows.shape[1] == X.shape[1]
    assert len(rows) == len(rmeta)
    assert {"canonical_id", "weight", "n_source_spectra", "provenance"} <= set(rmeta.columns)
    assert np.all(np.isfinite(rows))


def test_control_arm_is_v5_behaviour():
    """Arm A must be one row per spectrum with equal weight — exactly what V5 did."""
    X, meta, q = _toy_corpus()
    rows, rmeta = REF.build_arm("A_all_spectra", X, meta, q)
    assert len(rows) == len(X)
    assert np.allclose(rmeta.weight.values, 1.0)


def test_balancing_equalises_molecule_weight():
    """The core V7 change: one molecule, one unit — regardless of replicate count."""
    X, meta, q = _toy_corpus()
    for arm in ("B_analyte_weighted", "B_uniform", "C_mean", "C_medoid"):
        _, rmeta = REF.build_arm(arm, X, meta, q)
        w = rmeta.groupby("canonical_id").weight.sum()
        assert np.allclose(w.values, 1.0), f"{arm} does not give each molecule unit weight"
    _, ctrl = REF.build_arm("A_all_spectra", X, meta, q)
    wc = ctrl.groupby("canonical_id").weight.sum()
    assert wc.max() > wc.min(), "the control must remain unbalanced — that is the point"


def test_prototype_arms_emit_one_row_per_molecule():
    X, meta, q = _toy_corpus()
    for arm in ("C_mean", "C_median", "C_trimmed", "C_medoid", "C_quality"):
        rows, rmeta = REF.build_arm(arm, X, meta, q)
        assert len(rows) == meta.canonical_id.nunique()


def test_medoid_is_always_a_real_measured_spectrum():
    X, meta, q = _toy_corpus()
    rows, rmeta = REF.build_arm("C_medoid", X, meta, q)
    for i, cid in enumerate(rmeta.canonical_id):
        members = X[(meta.canonical_id == cid).values]
        assert any(np.allclose(rows[i], m) for m in members), (
            "a medoid must be one of the measured spectra, never a synthesised one")


def test_class_balance_reports_gini_and_ratio():
    X, meta, q = _toy_corpus()
    _, rmeta = REF.build_arm("B_analyte_weighted", X, meta, q)
    b = REF.class_balance(rmeta, {"m1": "c1", "m2": "c1", "m3": "c2"})
    assert 0.0 <= b["effective_class_gini"] <= 1.0
    assert b["molecule_weight_equal"] is True


def test_discarded_variance_is_retained():
    """Collapsing replicates discards the only measurement-uncertainty estimate available."""
    X, meta, q = _toy_corpus()
    dv = REF.discarded_variance(X, meta)
    assert len(dv) == meta.canonical_id.nunique()
    assert (dv[dv.n_spectra > 1].mean_bin_std > 0).all()


# ── C. adaptive k_c ───────────────────────────────────────────────────────────
def test_activation_sparsity_is_zero_at_k1():
    """The bug that made 'do not decompose' win everywhere: k=1 has NO selectivity."""
    assert CLS._activation_sparsity(np.ones((5, 1))) == 0.0
    assert CLS._activation_sparsity(np.eye(4)) > 0.9


def test_residual_structure_falls_as_the_fit_improves():
    """It measures unexplained band energy, not the peakiness of shrinking noise."""
    D = 676
    X = np.zeros((6, D))
    for i in range(6):
        X[i, 100 + 10 * (i % 3)] = 1.0
    prev = None
    for k in (1, 2, 3):
        W, H, _ = CLS.fit_nmf(X, k, seed=0)
        r = CLS._residual_structure(X, W, H)
        if prev is not None:
            assert r <= prev + 1e-9, "residual structure must not rise with k"
        prev = r


def test_select_k_uses_the_contiguous_plateau():
    """An isolated low-k point near the maximum is not a plateau."""
    sweep = [{"k": 1, "composite": 0.500}, {"k": 2, "composite": 0.300},
             {"k": 3, "composite": 0.310}, {"k": 4, "composite": 0.495},
             {"k": 5, "composite": 0.505}]
    sel = CLS.select_k(sweep, tolerance=0.02)
    assert sel["k"] == 4, "must take the plateau containing the maximum, not the stray k=1"
    assert sel["best_k"] == 5
    assert sel["plateau_is_contiguous"] is False
    assert 1 in sel["within_tolerance_anywhere"]


def test_select_k_prefers_the_smallest_on_the_plateau_not_argmax():
    sweep = [{"k": 1, "composite": 0.20}, {"k": 2, "composite": 0.49},
             {"k": 3, "composite": 0.50}, {"k": 4, "composite": 0.495}]
    assert CLS.select_k(sweep, tolerance=0.02)["k"] == 2


def test_every_k_scored_on_the_same_criteria():
    """Composites computed over different criterion sets are not comparable."""
    src = inspect.getsource(CLS.sweep_k)
    assert "n_criteria_applicable" not in src
    assert "COMPOSITE_WEIGHTS[c] * v for c, v in crit.items()" in src


def test_hungarian_alignment_matches_permuted_components():
    H = np.abs(np.random.default_rng(0).normal(size=(4, 50)))
    perm = [2, 0, 3, 1]
    c, s = CLS.align(H, H[perm])
    assert list(np.argsort(c)) == list(np.argsort(np.argsort(perm)))
    assert np.all(s > 0.99)


# ── D. the LSM object ─────────────────────────────────────────────────────────
def _mk(mid, cls, n, stability=1.0, spectrum=None, anchor=False, index=0):
    sp = np.zeros(676) if spectrum is None else spectrum
    if spectrum is None:
        sp[100:110] = 1.0
    return LSM(motif_id=mid, chemical_class=cls, index_in_class=index,
               spectrum=sp, dominant_bands=[{"center_cm": 650.0, "prominence": 1.0,
                                             "weight": 1.0}],
               analytes=[f"a{i}" for i in range(n)], n_analytes=n, n_spectra=n,
               activation_share=0.5, activation_sparsity=0.4, stability=stability,
               matched_similarity=0.9, purity=1.0, reconstruction_share=0.3,
               redundancy_max=0.0, lsm_type="class_shared", k_c=2, n_class_analytes=10,
               dominant_broad_class="B", is_anchor=anchor,
               anchor_justification="test justification" if anchor else "")


def _res(cls, lsms, status="DECOMPOSED"):
    return {"chemical_class": cls, "status": status, "n_analytes": 10, "n_spectra": 12,
            "k_ceiling": 5, "k_c": 2, "lsms": lsms, "sweep": [], "k_selection": None}


def test_lsm_validate_catches_violations():
    m = _mk("protein.m00", "protein", 3)
    assert m.validate() == []
    bad = LSM(**{**m.__dict__, "spectrum": -m.spectrum})
    assert any("negative" in v for v in bad.validate())
    bad = LSM(**{**m.__dict__, "motif_id": "wrongclass.m00"})
    assert any("chemical class" in v for v in bad.validate())
    bad = LSM(**{**m.__dict__, "lsm_type": "nonsense"})
    assert any("lsm_type" in v for v in bad.validate())


def test_anchor_must_declare_one_analyte_and_a_justification():
    a = _mk("polyol.anchor00", "polyol", 1, anchor=True)
    assert a.validate() == []
    bad = LSM(**{**a.__dict__, "anchor_justification": ""})
    assert any("justification" in v for v in bad.validate())
    bad = LSM(**{**a.__dict__, "n_analytes": 4,
                 "analytes": ["a", "b", "c", "d"]})
    assert any("exactly one supporting analyte" in v for v in bad.validate())


def test_classify_type_covers_the_three_specified_types():
    assert classify_type(np.ones(10)) == "class_shared"
    assert classify_type(np.array([1.0] + [0.0] * 9)) == "molecule_discriminating"
    assert classify_type(np.array([1.0] * 4 + [0.0] * 6)) == "subfamily"


def test_dominant_bands_are_ordered_and_weighted():
    grid = np.linspace(450, 1800, 676)
    h = np.zeros(676)
    for c in (100, 300, 500):
        h += np.exp(-0.5 * ((np.arange(676) - c) / 3.0) ** 2)
    b = dominant_bands(h, grid)
    assert len(b) == 3
    assert [x["center_cm"] for x in b] == sorted(x["center_cm"] for x in b)
    assert abs(sum(x["weight"] for x in b) - 1.0) < 1e-6


# ── E. rejection and registry ─────────────────────────────────────────────────
def test_rejection_reasons_are_deterministic():
    unstable = _mk("protein.m01", "protein", 5, stability=0.1, index=1)
    good = _mk("protein.m02", "protein", 5, index=2)
    DIS._reject([unstable, good])
    assert not unstable.retained and "low_stability" in unstable.rejection_reason
    assert good.retained and good.rejection_reason == ""


def test_redundant_lsm_rejected_keeping_the_better_supported():
    sp = np.zeros(676)
    sp[200:210] = 1.0
    a = _mk("protein.m00", "protein", 3, spectrum=sp.copy(), index=0)
    b = _mk("protein.m01", "protein", 9, spectrum=sp.copy(), index=1)
    DIS._score_redundancy([a, b])
    DIS._reject([a, b])
    assert b.retained and not a.retained
    assert "redundant" in a.rejection_reason


def test_registry_integrity_and_kc_ceiling():
    reg = LSMRegistry([_res("protein", [_mk("protein.m00", "protein", 3)])], "v", {}, "arm")
    assert reg.check_integrity() == []
    bad = _res("protein", [_mk("protein.m00", "protein", 3)])
    bad["k_c"] = 99
    assert any("exceeds ceiling" in v
               for v in LSMRegistry([bad], "v", {}, "arm").check_integrity())


def test_registry_keeps_rejected_lsms():
    keep = _mk("protein.m00", "protein", 5, index=0)
    drop = _mk("protein.m01", "protein", 5, stability=0.1, index=1)
    DIS._reject([keep, drop])
    reg = LSMRegistry([_res("protein", [keep, drop])], "v", {}, "arm")
    assert len(reg.retained) == 1 and len(reg.rejected) == 1
    assert len(reg.motif_table()) == 2
    assert reg.rejection_table().iloc[0].rejection_reason


def test_serialization_round_trips(tmp_path):
    sp = np.zeros(676)
    sp[100:110] = np.linspace(.1, 1., 10)
    m = _mk("protein.m00", "protein", 5, spectrum=sp)
    reg = LSMRegistry([_res("protein", [m])], "v7_lsm_classlocal_v1", {"arm": "B"}, "B")
    man = SER.save_registry(reg, tmp_path)
    df, H, ids, man2 = SER.load_registry(tmp_path)
    assert man2["registry_fingerprint"] == man["registry_fingerprint"]
    back = SER.lsms_from_table(df, H, ids)
    assert len(back) == 1
    assert np.allclose(back[0].spectrum, m.spectrum)
    assert back[0].chemical_class == "protein"


# ── F. frozen-artefact contract ───────────────────────────────────────────────
def test_atlas_fingerprint_unchanged():
    H = np.asarray(np.load(FOUNDATION / "manifold_components.npz")["components"], float)
    fp = hashlib.sha256(np.ascontiguousarray(H).tobytes()).hexdigest()[:32]
    assert fp == CANONICAL_ATLAS_FINGERPRINT


def test_frozen_foundation_files_unchanged():
    man = json.loads((FOUNDATION / "MANIFEST.json").read_text())
    bad = [n for n, r in man["files"].items()
           if hashlib.sha256((FOUNDATION / n).read_bytes()).hexdigest() != r["sha256"]]
    assert not bad


@ran
def test_phase00_artifacts_untouched():
    p00 = json.loads((REPO / "results/v7_rebuild/phase00/manifests/"
                      "phase_00_manifest_v1.json").read_text())
    stale = [o["artifact_id"] for o in p00["outputs"]
             if (REPO / o["path"]).is_file()
             and hashlib.sha256((REPO / o["path"]).read_bytes()).hexdigest() != o["sha256"]]
    assert not stale


@ran
def test_all_architecture_compliance_items_pass():
    """The gate opens only if every specification item passes."""
    comp = pd.read_csv(T / "architecture_compliance_v1.csv")
    failed = comp[comp.status != "PASS"]
    assert failed.empty, f"architecture non-compliance: {failed.specification_item.tolist()}"
    assert len(comp) >= 15


@ran
def test_kc_adapts_and_respects_its_ceiling():
    ct = pd.read_csv(T / "lsm_classes_v1.csv")
    dec = ct[ct.status == "DECOMPOSED"]
    assert dec.k_c.nunique() > 1, "k_c must adapt per class — no global k"
    assert (dec.k_c <= dec.k_ceiling).all(), "k_c must respect floor(n_analytes/2)"
    assert (dec.k_c >= 1).all()


@ran
def test_every_class_fitted_independently():
    ct = pd.read_csv(T / "lsm_classes_v1.csv")
    reg = pd.read_csv(T / "lsm_registry_v1.csv")
    assert len(ct) >= 10
    for cls in ct.chemical_class:
        assert reg.motif_id[reg.chemical_class == cls].str.startswith(f"{cls}.").all()


@ran
def test_lsm_dictionary_is_nonnegative_and_gridded():
    z = np.load(A / "lsm_dictionary_v1.npz", allow_pickle=True)
    H = np.asarray(z["H"], float)
    assert H.shape[1] == 676
    assert np.all(H >= 0) and np.all(np.isfinite(H))
    assert H.shape[0] == len(z["motif_ids"])


@ran
def test_lsms_are_not_bounded_by_the_frozen_atlas():
    """The decisive difference from the control experiment.

    An atlas decomposition satisfies 0 <= m <= h_k pointwise. A class-local LSM has no such
    bound and must, somewhere, exceed every frozen component.
    """
    Hatlas = np.asarray(np.load(FOUNDATION / "manifold_components.npz")["components"], float)
    z = np.load(A / "lsm_dictionary_v1.npz", allow_pickle=True)
    H = np.asarray(z["H"], float)
    exceeds = [j for j in range(H.shape[0])
               if np.all((H[j][None, :] > Hatlas + 1e-9).any(axis=1))]
    assert exceeds, ("every LSM is pointwise bounded by some frozen component — that is the "
                     "signature of an atlas decomposition, not a class-local fit")


@ran
def test_balanced_references_carry_unit_weight_per_molecule():
    z = np.load(A / "balanced_references_v1.npz", allow_pickle=True)
    cid = np.array([str(c) for c in z["canonical_id"]])
    w = pd.Series(np.asarray(z["weight"], float)).groupby(cid).sum()
    sel = json.loads((A / "reference_arm_selection_v1.json").read_text())
    if sel["selected_arm"] != "A_all_spectra":
        assert np.allclose(w.values, 1.0, atol=1e-9)


@ran
def test_reference_arm_comparison_is_complete_and_published():
    arms = pd.read_csv(T / "reference_arm_comparison_v1.csv")
    assert len(arms) == 8
    assert (arms.arm == "A_all_spectra").any() and (arms.arm == "B_uniform").any()
    for c in ("band_fidelity_replicated_only", "band_fidelity_multi_excitation"):
        assert c in arms.columns, "both mandatory stratifications must be reported"


@ran
def test_risk_checks_reported():
    bias = pd.read_csv(T / "class_prior_bias_v1.csv")
    ct = pd.read_csv(T / "lsm_classes_v1.csv")
    assert "prior_dominated" in bias.columns, "R-01 class-prior bias must be tested"
    assert "source_confounded" in ct.columns, "R-16 source composition must be reported"


@ran
def test_determinism():
    det = json.loads((A / "determinism_v1.json").read_text())
    assert det["identical"] is True
    assert len(set(det["signatures"])) == 1


@ran
def test_all_gates_pass_and_phase02_not_started():
    st = json.loads((P01 / "PHASE_STATE.json").read_text())
    failed = [g["gate"] for g in st["gates"] if not g["passed"]]
    assert not failed, f"gates failed: {failed}"
    assert st["status"] == "COMPLETE"
    assert st["architecture_compliant"] is True
    assert "NOT STARTED" in st["next_phase"]


@ran
def test_phase00_corrections_emitted():
    """Audit corrections C-9 and C-10."""
    assert (T / "dataset_role_map_v7.csv").is_file()
    assert (T / "evaluation_ontology_v7.csv").is_file()


@ran
@pytest.mark.parametrize("stem", [
    "fig01_pipeline", "fig02_reference_arms", "fig03_capacity_allocation",
    "fig04_kc_optimisation", "fig05_basis_spectra", "fig06_activation_heatmap",
    "fig07_stability_quality", "fig08_lsm_typing", "fig09_reconstruction",
    "fig10_architecture_compliance",
])
def test_figures_exist(stem):
    assert (F / f"{stem}.svg").is_file() and (F / f"{stem}.png").is_file()
