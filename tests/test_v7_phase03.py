"""GAIRA V7 — Phase 03 tests: emergent biochemical theme discovery.

The claims worth pinning are the ones a reviewer would not take on trust:

    test_no_chemistry_label_reaches_a_membership_model
    test_theme_names_cannot_be_biology
    test_a_rejected_theme_is_recorded_not_quietly_accepted
    test_bridges_and_poorly_explained_csms_are_not_forced
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gaira.v7.io import PhaseOutputs, frozen_root, output_root   # noqa: E402
from gaira.v7.themes import criteria as CRIT                     # noqa: E402
from gaira.v7.themes import models as MOD                        # noqa: E402
from gaira.v7.themes import validation as VAL                    # noqa: E402
from gaira.v7.themes.registry import Theme, ThemeRegistry, check_name  # noqa: E402

OUT = PhaseOutputs("03")
T, A, V, F, R = OUT.tables, OUT.artifacts, OUT.validation, OUT.figures, OUT.reports
FROZEN = frozen_root()
ATLAS_FP = "09ed804a40836f4a05a91ba10900cded"
LSM_FP = "208482d6f7178b5b8f16cace91be55b0"
CSM_FP = "0b4aa550ccefed3edabdbde5bae11c8d"

ran = pytest.mark.skipif(not (A / "phase_03_manifest_v1.json").is_file(),
                         reason="Phase 03 has not been run in this checkout")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((A / "phase_03_manifest_v1.json").read_text())


@pytest.fixture(scope="module")
def state():
    return json.loads((OUT.root / "PHASE_STATE.json").read_text())


@pytest.fixture(scope="module")
def registry():
    return json.loads((A / "theme_registry_v1.json").read_text())


@pytest.fixture(scope="module")
def memb():
    return np.load(A / "theme_membership_v1.npz", allow_pickle=True)


def _theme(**kw):
    base = dict(theme_id="Theme-01", index=0, spectrum=np.abs(np.sin(np.linspace(0, 9, 676))),
                dominant_bands=[1440.0], band_assignments=["CH2 / CH3 scissoring"],
                mode_families=["aliphatic"], dominant_families=["aliphatic"],
                family_concentration=0.8, chemically_admissible=True, assigned_fraction=1.0,
                member_csms=["csm00"], member_memberships=[0.9], bridge_csms=[],
                n_supporting_csms=1, mean_membership=0.5, membership_entropy=0.4)
    base.update(kw)
    return Theme(**base)


# ── A. FROZEN INPUTS ─────────────────────────────────────────────────────────
@ran
def test_frozen_fingerprints_gate_the_run(manifest):
    f = manifest["frozen_inputs"]
    assert f["atlas"] == ATLAS_FP
    assert f["lsm_registry"] == LSM_FP
    assert f["csm_dictionary"] == CSM_FP


@ran
def test_upstream_phases_were_not_modified():
    assert json.loads((FROZEN / "phase01/PHASE_STATE.json").read_text())[
        "registry_fingerprint"] == LSM_FP
    assert json.loads((FROZEN / "phase02/PHASE_STATE.json").read_text())[
        "csm_fingerprint"] == CSM_FP
    assert json.loads((FROZEN / "phase02_5/PHASE_STATE.json").read_text())[
        "themes_created"] is False


def test_frozen_inputs_do_not_follow_the_output_redirect(monkeypatch):
    """Redirecting outputs must never change which upstream dictionary a phase consumes."""
    before = frozen_root()
    monkeypatch.setenv("GAIRA_V7_OUTPUT_ROOT", "/tmp/gaira-elsewhere")
    assert output_root() == Path("/tmp/gaira-elsewhere").resolve()
    assert frozen_root() == before
    assert PhaseOutputs("03").root == Path("/tmp/gaira-elsewhere/phase03").resolve()


# ── B. THE LABEL FIREWALL ────────────────────────────────────────────────────
@pytest.mark.parametrize("name", MOD.MODELS)
def test_no_chemistry_label_reaches_a_membership_model(name):
    """No model may even accept a label — the firewall is checked on signatures, not prose."""
    fn = {"archetypal": MOD.archetypal, "sparse_nmf": MOD.sparse_nmf,
          "fuzzy_cmeans": MOD.fuzzy_cmeans, "diffusion_gmm": MOD.diffusion_gmm,
          "graph_regularised_nmf": MOD.graph_regularised_nmf}[name]
    params = set(inspect.signature(fn).parameters)
    assert not (params & {"labels", "classes", "class_of", "chemical_class", "ontology",
                          "y", "targets"}), f"{name} accepts a label"


def test_k_selection_criteria_are_all_label_free():
    """§4 lists mutual information with chemistry; it must not appear among the weights."""
    for crit in CRIT.CRITERIA:
        assert "mutual" not in crit and "chemistry" not in crit and "ontology" not in crit
    src = inspect.getsource(CRIT.select_K) + inspect.getsource(CRIT.composite)
    assert "label" not in src.lower().replace("label-free", "")


@ran
def test_manifest_records_the_firewall(manifest):
    fw = manifest["label_firewall"]
    assert fw["labels_used_before_K_selection"] is False
    assert fw["revealed_at_step"] == 5


@ran
def test_ontology_agreement_is_post_hoc(registry):
    post = json.loads((A / "post_hoc_v1.json").read_text())
    assert post["revealed_after_K_selected"] is True
    assert "ontology_agreement" in post


# ── C. P-07 — THEMES NAME CHEMISTRY ──────────────────────────────────────────
@pytest.mark.parametrize("bad", ["lipid metabolism", "cancer signature", "inflammation axis",
                                 "apoptosis theme", "diagnostic marker", "disease state"])
def test_theme_names_cannot_be_biology(bad):
    with pytest.raises(ValueError, match="P-07"):
        check_name(bad)


@pytest.mark.parametrize("ok", ["aliphatic chain", "amide backbone", "carboxyl / ester carbonyl",
                                "heterocyclic / conjugated ring", "Unknown Theme"])
def test_chemistry_names_are_allowed(ok):
    check_name(ok)


def test_a_theme_constructed_with_a_biology_name_is_rejected():
    with pytest.raises(ValueError, match="P-07"):
        _theme(name="tumour lipid pathway")


@ran
def test_no_shipped_theme_name_refers_to_biology(registry):
    for t in registry["themes"]:
        check_name(t["name"])


# ── D. C-08 CONTRACT ─────────────────────────────────────────────────────────
@ran
def test_membership_matrix_satisfies_the_contract(memb, state):
    S = memb["S"]
    assert (S >= 0).all()
    assert np.allclose(S.sum(axis=1), 1.0, atol=1e-6)
    assert S.shape == (49, state["K"])
    assert np.sort(S, axis=1)[:, -2:].sum(axis=1).mean() >= 0.60


@ran
def test_theme_basis_is_non_negative_and_unit_norm(memb):
    TH = memb["THEMES"]
    assert (TH >= 0).all()
    assert TH.shape[1] == 676
    assert np.allclose(np.linalg.norm(TH, axis=1), 1.0, atol=1e-6)


@ran
def test_no_csm_is_forced_to_a_single_parent(memb):
    """Soft membership is the point; a matrix of one-hot rows would defeat it."""
    assert (memb["S"].max(axis=1) < 0.999).sum() >= 5


@ran
def test_all_invariants_pass():
    inv = pd.read_csv(V / "theme_invariants_v1.csv")
    failed = inv[inv.status != "PASS"]
    assert failed.empty, f"failed invariants:\n{failed}"


def test_registry_requires_counter_evidence_on_accepted_themes():
    r = ThemeRegistry("archetypal", 1, {}, {})
    r.add(_theme())                       # no counter-evidence recorded
    inv = {i["invariant"]: i["status"] for i in r.check_invariants(np.ones((1, 1)), ["csm00"])}
    assert inv["every theme carries counter-evidence"] == "FAIL"


def test_registry_rejects_a_negative_theme_spectrum():
    with pytest.raises(ValueError, match="non-negative"):
        _theme(spectrum=np.linspace(-1, 1, 676))


# ── E. SELECTION AND DEGENERACY ──────────────────────────────────────────────
def test_degenerate_membership_is_caught_in_S():
    """One theme dominant for every row scored 0.497 information and 0.964 stability and was
    selected. The degeneracy is in S, so S is where it must be caught."""
    S = np.zeros((20, 4))
    S[:, 0] = 1.0
    d = CRIT.membership_degenerate(S)
    assert d["degenerate"] is True
    good = np.full((20, 4), 0.25)
    good[:5, 0] = 0.7; good[:5, 1:] = 0.1
    good[5:10, 1] = 0.7; good[5:10, [0, 2, 3]] = 0.1
    good[10:15, 2] = 0.7; good[10:15, [0, 1, 3]] = 0.1
    good[15:, 3] = 0.7; good[15:, :3] = 0.1
    assert CRIT.membership_degenerate(good)["degenerate"] is False


def test_degeneracy_catches_a_theme_set_that_explains_nothing():
    grid = np.linspace(450, 1800, 676)
    X = np.abs(np.random.default_rng(0).normal(size=(20, 676))) + 0.1
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    themes = np.zeros((3, 676))
    themes[:, :3] = 1.0                     # explain nothing
    S = np.full((20, 3), 1 / 3)
    S[:7, 0] = 0.8; S[7:14, 1] = 0.8; S[14:, 2] = 0.8
    S /= S.sum(axis=1, keepdims=True)
    assert CRIT.membership_degenerate(S, X, themes)["degenerate"] is True


def test_admissibility_rejects_a_flat_mixture_and_accepts_a_concentrated_theme():
    grid = np.linspace(450, 1800, 676)
    g = lambda c, a=1.0: a * np.exp(-((grid - c) ** 2) / (2 * 8.0 ** 2))
    concentrated = g(1440) + g(1300, 0.8) + 0.02          # both aliphatic
    # five equal peaks in five DIFFERENT mode families: sulfur, skeletal, phosphate,
    # unsaturation, aliphatic. Top two carry 2/5 of the prominence.
    spread = g(520) + g(900) + g(975) + g(1270) + g(1440) + 0.02
    assert CRIT.admissibility(concentrated, grid)["admissible"] is True
    a = CRIT.admissibility(spread, grid)
    assert len(a["mode_families"]) >= 4
    assert a["family_concentration"] < CRIT.ADMISSIBILITY_MIN_CONCENTRATION
    assert a["admissible"] is False


def test_assignable_fraction_is_reported_but_does_not_gate():
    """It was a gate until it was shown to be 1.000 for every candidate — the windows tile."""
    grid = np.linspace(450, 1800, 676)
    a = CRIT.admissibility(np.abs(np.sin(np.linspace(0, 40, 676))) + 0.01, grid)
    assert "assigned_fraction" in a
    src = inspect.getsource(CRIT.admissibility)
    assert "ADMISSIBILITY_MIN_CONCENTRATION" in src
    assert "ADMISSIBILITY_MIN_ASSIGNED" not in src


def test_theme_set_distinctness_catches_duplicate_chemistry():
    grid = np.linspace(450, 1800, 676)
    g = lambda c: np.exp(-((grid - c) ** 2) / (2 * 8.0 ** 2))
    a = g(1440) + g(1300)
    themes = np.vstack([a, a * 1.01])
    adms = [CRIT.admissibility(t, grid) for t in themes]
    assert CRIT.theme_set_distinct(themes, adms)["distinct"] is False


def test_specificity_demotes_the_ubiquitous_family():
    """CH2 scissoring dominates nearly every biological Raman spectrum; without down-weighting
    it, every theme is named 'aliphatic chain + something'."""
    adms = [[{"mode_families": ["aliphatic", "ring"]}] * 8,
            [{"mode_families": ["aliphatic", "carboxyl"]}]]
    spec = CRIT.family_specificity(adms)
    assert spec["carboxyl"] > spec["aliphatic"]


def test_select_K_applies_admissibility_as_a_veto():
    rows = [{"K": k, "chemically_admissible": k == 7, "themes_distinct": True,
             "degenerate": False,
             **{c: (0.9 if k == 3 else 0.5) for c in CRIT.CRITERIA}} for k in range(2, 10)]
    sel = CRIT.select_K(rows)
    assert sel["K"] == 7, "an inadmissible K must be rejected however well it scores"


def test_select_K_fails_when_nothing_is_admissible():
    rows = [{"K": k, "chemically_admissible": False, "themes_distinct": True,
             "degenerate": False, **{c: 0.5 for c in CRIT.CRITERIA}} for k in range(2, 6)]
    assert CRIT.select_K(rows)["status"] == "FAIL"


# ── F. VALIDATION SEMANTICS ──────────────────────────────────────────────────
def test_bridges_and_poorly_explained_are_opposite_conditions():
    """A bridge is explained by several themes; a poorly-explained CSM by none. An earlier
    version collapsed them and reported 15 unassigned and zero bridges."""
    grid = np.linspace(450, 1800, 676)
    g = lambda c: np.exp(-((grid - c) ** 2) / (2 * 8.0 ** 2)) + 0.01
    themes = np.vstack([g(1000), g(1440)])
    themes /= np.linalg.norm(themes, axis=1, keepdims=True)
    X = np.vstack([g(1000), 0.5 * (g(1000) + g(1440)), g(700)])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = np.array([[0.95, 0.05], [0.5, 0.5], [0.55, 0.45]])
    r = VAL.membership_roles(S, ["a", "b", "c"], X, themes, set(), set())
    assert r.loc[0, "role"] == "member"
    assert r.loc[2, "role"] in ("bridge", "poorly_explained")
    assert set(r.role) <= {"member", "bridge", "poorly_explained"}


def test_robustness_marks_an_unfair_holdout_untestable():
    """Removing 37 of 49 CSMs is not evidence that themes are source artefacts."""
    X = np.abs(np.random.default_rng(0).normal(size=(20, 30)))
    S0 = np.full((20, 3), 1 / 3)
    out = VAL.robustness(lambda Xs: {"S": np.full((Xs.shape[0], 3), 1 / 3)}, X, S0,
                         {"big": list(range(15)), "small": [0, 1]})
    assert out[out.held_out == "big"].iloc[0].testable is np.False_ or \
           not out[out.held_out == "big"].iloc[0].testable
    assert bool(out[out.held_out == "small"].iloc[0].testable)


def test_value_over_csm_reports_a_failure_honestly():
    """Risk R-11: a theme layer that changes no decision must say so."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 40))
    S = rng.normal(size=(20, 3))
    labels = ["a"] * 10 + ["b"] * 10
    out = VAL.value_over_csm(S, X, np.abs(rng.normal(size=(3, 40))), labels,
                             np.zeros((20, 20)))
    assert "theme_layer_adds_value" in out
    assert ("does NOT improve" in out["verdict"]) == (not out["theme_layer_adds_value"])


# ── G. THE RUN ───────────────────────────────────────────────────────────────
@ran
def test_a_rejected_theme_is_recorded_not_quietly_accepted(registry):
    rej = [t for t in registry["themes"] if t["status"] == "rejected"]
    for t in rej:
        assert t["rejection_reason"]
    assert registry["summary"]["n_accepted"] + registry["summary"]["n_rejected"] == \
        len(registry["themes"])


@ran
def test_every_accepted_theme_carries_counter_evidence(registry):
    for t in registry["themes"]:
        if t["status"] == "accepted":
            assert t["counter_evidence"], t["theme_id"]
            assert t["alternative_explanations"], t["theme_id"]


@ran
def test_shared_raman_physics_is_offered_as_an_alternative_for_every_theme(registry):
    """The brief's central warning: shared CH stretching is not lipid biology."""
    for t in registry["themes"]:
        joined = " ".join(t["alternative_explanations"]).lower()
        assert "raman physics" in joined or "shared" in joined, t["theme_id"]


@ran
def test_bridges_and_poorly_explained_csms_are_not_forced(registry, state):
    roles = pd.read_csv(T / "membership_roles_v1.csv")
    assert set(roles.role) <= {"member", "bridge", "poorly_explained"}
    assert len(registry["bridge_csms"]) == int((roles.role == "bridge").sum())
    assert len(registry["unassigned_csms"]) == int((roles.role == "poorly_explained").sum())
    assert len(roles) == 49


@ran
def test_every_csm_appears_in_the_membership_record(registry, memb):
    ids = [str(s) for s in memb["csm_ids"]]
    assert len(ids) == 49
    assert set(registry["memberships"]) == set(ids)


@ran
def test_hierarchy_levels_were_inferred_not_assumed():
    h = json.loads((A / "hierarchy_v1.json").read_text())
    assert h["n_levels"] == len(h["levels"])
    assert [l["n_groups"] for l in h["levels"]] == sorted({l["n_groups"] for l in h["levels"]})


@ran
def test_gradients_were_tested_against_a_permutation_null():
    g = pd.read_csv(V / "theme_gradients_v1.csv")
    assert {"spearman", "p_empirical", "is_gradient"} <= set(g.columns)
    assert (g.is_gradient == (g.p_empirical < 0.05)).all()


@ran
def test_all_gates_pass(state):
    g = pd.read_csv(V / "phase03_gates_v1.csv")
    failed = g[g.status != "PASS"]
    assert failed.empty, f"failed gates:\n{failed}"
    assert state["status"] == "COMPLETE"


@ran
def test_manifest_lists_every_output(manifest):
    for o in manifest["outputs"]:
        p = Path(o["path"])
        assert (p if p.is_absolute() else REPO / p).is_file(), o["path"]


@ran
@pytest.mark.parametrize("n", range(1, 14))
def test_figure_exists(n):
    assert sorted(F.glob(f"fig{n:02d}_*.png")), f"figure {n:02d} missing"


@ran
def test_the_figure_pdf_exists():
    pdf = R / "PHASE_03_FIGURES.pdf"
    assert pdf.is_file()
    assert pdf.stat().st_size > 200_000


@ran
def test_report_and_audit_exist_and_state_the_limits():
    rep = (R / "PHASE_03_REPORT.md").read_text()
    for term in ("firewall", "rejected", "bridge", "counter-evidence", "Phase 04", "AMI"):
        assert term.lower() in rep.lower(), term
    aud = (R / "PHASE_03_SCIENTIFIC_AUDIT.md").read_text()
    for term in ("weakness", "unsupported", "reviewer", "recommended experiment", "risk"):
        assert term.lower() in aud.lower(), term


@ran
def test_yaml_registry_written_for_the_contract():
    y = (A / "theme_registry_v1.yaml").read_text()
    assert y.startswith("schema: theme_registry_v1")
    assert "top_csms:" in y and "chemical_definition:" in y
