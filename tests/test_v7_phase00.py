"""GAIRA V7 — Phase 00 regression tests.

Tests the FROZEN Phase-00 contract: the artefacts exist, carry the promised invariants,
and cannot drift without a test failing. Runs without the raw data volume — every test
reads committed Phase-00 outputs or committed frozen assets.

Scientific-model tests belong to Phase 02 and later; there is no V7 model yet.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
P00 = REPO / "results/v7_rebuild/phase00"
T = P00 / "tables"
M = P00 / "manifests"
V = P00 / "validation"
F = P00 / "figures"
FOUNDATION = REPO / "assets" / "foundation"

CANONICAL_ATLAS_FINGERPRINT = "09ed804a40836f4a05a91ba10900cded"

pytestmark = pytest.mark.skipif(not (M / "phase_00_manifest_v1.json").is_file(),
                                reason="Phase 00 has not been run in this checkout")


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def canon():
    return pd.read_csv(T / "canonical_analytes_v1.csv")


@pytest.fixture(scope="module")
def alias():
    return pd.read_csv(T / "alias_table_v1.csv")


@pytest.fixture(scope="module")
def part():
    return pd.read_csv(T / "chemical_partition_v1.csv")


@pytest.fixture(scope="module")
def folds():
    return pd.read_csv(T / "cv_folds_v1.csv")


@pytest.fixture(scope="module")
def card():
    return json.loads((M / "dataset_card_v7.json").read_text())


@pytest.fixture(scope="module")
def splits():
    return json.loads((M / "cv_splits_v1.json").read_text())


@pytest.fixture(scope="module")
def manifest():
    return json.loads((M / "phase_00_manifest_v1.json").read_text())


@pytest.fixture(scope="module")
def state():
    return json.loads((P00 / "PHASE_STATE.json").read_text())


# ── A. frozen atlas is untouched ──────────────────────────────────────────────
def test_atlas_fingerprint_unchanged():
    H = np.asarray(np.load(FOUNDATION / "manifold_components.npz")["components"], float)
    fp = hashlib.sha256(np.ascontiguousarray(H).tobytes()).hexdigest()[:32]
    assert fp == CANONICAL_ATLAS_FINGERPRINT, f"FROZEN ATLAS CHANGED: {fp}"
    assert H.shape == (24, 676)


def test_frozen_file_hashes_unchanged():
    man = json.loads((FOUNDATION / "MANIFEST.json").read_text())
    bad = [n for n, rec in man["files"].items()
           if hashlib.sha256((FOUNDATION / n).read_bytes()).hexdigest() != rec["sha256"]]
    assert not bad, f"frozen assets modified: {bad}"


def test_phase00_writes_only_inside_its_own_tree(manifest):
    stray = [o["path"] for o in manifest["outputs"]
             if not o["path"].startswith("results/v7_rebuild/phase00/")]
    assert not stray, f"Phase 00 wrote outside its tree: {stray}"


def test_phase00_reads_frozen_assets_read_only():
    dep = pd.read_csv(T / "frozen_dependency_graph_v1.csv")
    assert (dep.access == "READ").all(), "a frozen asset is not marked READ-only"


# ── B. benchmark lock ─────────────────────────────────────────────────────────
def test_benchmark_lock_all_checks_pass():
    b = pd.read_csv(T / "benchmark_lock_v1.csv")
    failed = b[b.status != "PASS"]
    assert failed.empty, f"benchmark lock failures: {failed.check.tolist()}"


def test_benchmark_lock_reached_level_3(state):
    assert state["benchmark_lock_level"] == 3, (
        "Phase 00 must reach lock level 3 (basis refitted from raw); "
        f"got level {state['benchmark_lock_level']}")
    assert state["atlas_rebuilt_bit_exact"] is True


def test_rebuilt_basis_is_bit_exact():
    b = pd.read_csv(T / "benchmark_lock_v1.csv")
    row = b[b.check == "rebuild.max_abs_difference"]
    assert len(row) == 1
    assert float(row.iloc[0].got) == 0.0, "refitted basis is not element-wise identical"


# ── C. corpus ─────────────────────────────────────────────────────────────────
def test_corpus_card_matches_frozen_v5(card):
    assert card["n_spectra"] == 375
    assert card["n_analytes"] == 167
    assert card["n_bins"] == 676
    assert card["sources"] == {"RamanBioLib": 202, "gobbato_raman_metabolites": 153,
                               "amino_acid_raman_grounding": 20}


def test_preprocessing_unchanged(card):
    assert card["preprocessing"] == {"baseline": "asls", "smooth": "savgol", "norm": "l2"}
    assert card["window_cm"] == [450.0, 1800.0]
    assert card["grid_step_cm"] == 2.0


def test_corpus_is_raman_only(card):
    for d in ("Ag-SERS", "Au-SERS", "DART"):
        assert d in card["excluded_domains"], f"{d} must remain excluded"


# ── D. canonical identity ─────────────────────────────────────────────────────
def test_canonical_ids_unique(canon):
    assert canon.canonical_id.is_unique


def test_every_surface_form_maps_to_exactly_one_id(alias, card):
    assert alias.surface_form.nunique() == card["n_analytes"]
    assert alias.groupby("surface_form").canonical_id.nunique().max() == 1


def test_merging_conserves_spectra(canon, card):
    """Canonicalisation is a relabelling: it must not create or destroy measurements."""
    assert int(canon.n_spectra.sum()) == card["n_spectra"]


def test_expected_canonical_count(canon):
    assert len(canon) == 154, "167 surface forms collapse onto 154 canonical molecules"


def test_riboflavin_ligature_is_merged(alias):
    """The U+FB02 ligature is the mechanical leakage path NFKC exists to close."""
    lig = [s for s in alias.surface_form if "ﬂ" in s]
    assert lig, "the ligature form is absent from the corpus — check the loader"
    for s in lig:
        assert alias[alias.surface_form == s].canonical_id.iat[0] == "riboflavin"


@pytest.mark.parametrize("a,b", [
    ("(+)-arabinose", "(-)-arabinose"),      # enantiomers
    ("(+)-glucose", "β-d-glucose"),          # anomers
    ("(-)-ribose", "2-deoxy-d-ribose"),      # different molecules
])
def test_protected_pairs_are_not_merged(alias, a, b):
    ca = alias[alias.surface_form == a].canonical_id
    cb = alias[alias.surface_form == b].canonical_id
    assert len(ca) and len(cb), f"{a} or {b} missing from the alias table"
    assert ca.iat[0] != cb.iat[0], f"{a} and {b} must remain distinct molecules"


def test_every_alias_has_a_written_justification(alias):
    al = alias[alias.is_alias]
    assert len(al) == 13
    assert al.justification.astype(str).str.len().gt(10).all()


def test_near_miss_decisions_are_explicit():
    near = pd.read_csv(T / "alias_near_miss_audit_v1.csv")
    assert len(near) > 0
    assert near.decision.isin(["MERGED", "NOT_MERGED", "NOT_MERGED_PROTECTED",
                               "NOT_MERGED_UNRESOLVED"]).all()
    assert near.reason.astype(str).str.len().gt(5).all()


# ── E. partition ──────────────────────────────────────────────────────────────
def test_partition_covers_every_canonical_id(part, canon):
    assert len(part) == len(canon)
    assert part.fine_class.astype(str).str.len().gt(0).all()
    assert part.broad_class.astype(str).str.len().gt(0).all()


def test_unknown_class_is_dissolved(part):
    """`unknown` is not a chemistry — Phase 00 had to resolve it before any fitting."""
    assert not (part.fine_class == "unknown").any()
    assert not (part.broad_class == "unknown").any()


def test_lipid_overlap_is_resolved(part):
    fine = set(part.fine_class)
    for c in ("fatty_acid", "acylglycerol", "phospholipid_sphingolipid"):
        assert c in fine, f"{c} missing — the lipid three-way split is not in force"
    assert "lipid" not in fine, "the ambiguous `lipid` bucket must not survive"


def test_polysaccharide_kept_separate(part):
    fine = set(part.fine_class)
    assert "polysaccharide" in fine and "mono_oligosaccharide" in fine


def test_every_class_has_a_written_rationale():
    census = pd.read_csv(T / "class_census_v1.csv")
    assert census.rationale.astype(str).str.len().gt(20).all()


def test_class_conflicts_are_recorded_and_resolved(part):
    conf = pd.read_csv(T / "class_conflicts_v1.csv")
    assert len(conf) > 0, "the acetyl-CoA protein/cofactor conflict must be recorded"
    assert conf.resolution.astype(str).str.len().gt(5).all()
    assert part.groupby("canonical_id").fine_class.nunique().max() == 1


def test_k_c_ceiling_respected():
    census = pd.read_csv(T / "class_census_v1.csv")
    assert (census.k_c_ceiling == census.n_canonical_analytes // 2).all(), (
        "k_c ceiling must be floor(n_analytes / 2) — no class may memorise its molecules")


# ── F. cross-validation splits ────────────────────────────────────────────────
def test_all_three_leakage_checks_are_false(splits):
    lc = splits["leakage_checks"]
    assert lc["canonical_id_across_folds"] is False
    assert lc["alias_collision"] is False
    assert lc["replicate_across_folds"] is False
    assert splits["all_checks_false"] is True


def test_splits_group_by_canonical_id(splits):
    assert splits["grouping"] == "canonical_id"


def test_every_canonical_id_is_in_exactly_one_fold(folds, canon):
    assert len(folds) == len(canon)
    assert folds.canonical_id.is_unique
    assert folds.fold.notna().all()


def test_fold_membership_matches_the_manifest(folds, splits):
    for f in splits["folds"]:
        got = sorted(folds[folds.fold == f["fold"]].canonical_id.tolist())
        assert got == sorted(f["test"]), f"fold {f['fold']} disagrees with the manifest"


def test_splits_are_deterministic(canon, part, folds):
    """Re-cutting from the written tables must reproduce the fold assignment exactly."""
    import sys
    sys.path.insert(0, str(P00 / "code"))
    import v7_splits as SP                                            # noqa: E402
    re = SP.make_folds(canon, part).sort_values("canonical_id").reset_index(drop=True)
    ref = folds.sort_values("canonical_id").reset_index(drop=True)
    assert (re.fold.values == ref.fold.values).all()


# ── G. quality score ──────────────────────────────────────────────────────────
def test_quality_covers_every_spectrum(card):
    qual = pd.read_csv(T / "spectrum_quality_v1.csv")
    assert len(qual) == card["n_spectra"]
    assert qual.spectrum_id.is_unique, "duplicate spectrum_id would collapse an id-keyed join"


def test_quality_score_in_unit_interval():
    qual = pd.read_csv(T / "spectrum_quality_v1.csv")
    assert ((qual.quality_score >= 0) & (qual.quality_score <= 1)).all()


def test_weights_sum_to_one_per_canonical_molecule():
    qual = pd.read_csv(T / "spectrum_quality_v1.csv")
    for col in ("weight_quality", "weight_uniform"):
        s = qual.groupby("canonical_id")[col].sum()
        assert np.allclose(s.values, 1.0, atol=1e-9), f"{col} does not sum to 1 per molecule"


def test_quality_score_is_not_degenerate():
    """A constant q would silently make Strategy B identical to the control."""
    qsum = json.loads((M / "quality_summary_v1.json").read_text())
    assert qsum["q_max_over_min"] > 1.5
    assert qsum["q_version"] == "v7_q_v2"


def test_quality_uniform_arm_exists():
    qual = pd.read_csv(T / "spectrum_quality_v1.csv")
    assert "weight_uniform" in qual.columns, "the mandatory B-uniform arm must be frozen too"


# ── H. baseline ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("level,expected", [
    ("coord", 0.6467), ("mss", 0.6707), ("theme_raw", 0.6228), ("system_raw", 0.5689),
])
def test_baseline_reproduces_v63_fine_numbers(level, expected):
    """The V7 harness must reproduce the published V6.3 fine-ontology numbers exactly."""
    b = pd.read_csv(T / "phase00_baseline_metrics.csv")
    got = float(b[(b.level == level) & (b.labels == "v7_fine_16")].retrieval_p1.iat[0])
    assert abs(got - expected) < 5e-4, f"{level}: expected {expected}, got {got}"


@pytest.mark.parametrize("level,expected", [("coord", 0.8204), ("mss", 0.8084)])
def test_baseline_reproduces_v63_broad_numbers(level, expected):
    b = pd.read_csv(T / "phase00_baseline_metrics.csv")
    got = float(b[(b.level == level) & (b.labels == "v7_broad_6")].retrieval_p1.iat[0])
    assert abs(got - expected) < 5e-4


def test_random_control_is_near_chance():
    b = pd.read_csv(T / "phase00_baseline_metrics.csv")
    rnd = b[b.labels == "size_matched_random"]
    assert len(rnd) == 4
    assert (rnd.retrieval_p1 < 0.15).all(), "size-matched random must score near chance"


def test_baseline_carries_confidence_intervals():
    b = pd.read_csv(T / "phase00_baseline_metrics.csv")
    real = b[b.labels != "size_matched_random"]
    assert (real.ci95_low < real.retrieval_p1).all()
    assert (real.retrieval_p1 < real.ci95_high).all()


# ── I. provenance ─────────────────────────────────────────────────────────────
def test_manifest_hashes_every_artifact(manifest):
    assert all(o.get("sha256") for o in manifest["outputs"])
    assert all(i.get("sha256") for i in manifest["inputs"])


def test_manifest_output_hashes_still_match_disk(manifest):
    stale = []
    for o in manifest["outputs"]:
        p = REPO / o["path"]
        if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() != o["sha256"]:
            stale.append(o["artifact_id"])
    assert not stale, f"artefacts changed after the manifest was written: {stale}"


def test_manifest_records_seeds_and_environment(manifest):
    assert manifest["seeds"]["nmf"] == 0
    assert manifest["environment"]["python"]
    assert manifest["code"]["git_sha"]


def test_every_decision_names_its_preregistered_rule(manifest):
    assert len(manifest["decisions"]) >= 5
    for d in manifest["decisions"]:
        assert d["rule_preregistered_in"], f"post-hoc decision: {d['decision']}"
        assert d["rationale"]


def test_all_phase00_gates_pass(state):
    failed = [g["gate"] for g in state["gates"] if not g["passed"]]
    assert not failed, f"Phase 00 gates failed: {failed}"


def test_phase_state_is_complete_and_does_not_start_phase_01(state):
    assert state["status"] == "COMPLETE"
    assert state["atlas_fingerprint"] == CANONICAL_ATLAS_FINGERPRINT
    assert "NOT STARTED" in state["next_phase"]


# ── J. validation suite and figures ───────────────────────────────────────────
def test_validation_suite_has_no_failures():
    v = pd.read_csv(V / "phase00_validation_v1.csv")
    fails = v[v.status == "FAIL"]
    assert fails.empty, f"validation failures: {fails.item.tolist()}"
    assert len(v) >= 80


def test_validation_warnings_are_documented():
    """Every WARN must carry an explanatory note — a bare warning is not a finding."""
    v = pd.read_csv(V / "phase00_validation_v1.csv")
    for _, r in v[v.status == "WARN"].iterrows():
        assert str(r.note).strip() not in ("", "nan"), f"undocumented WARN: {r['item']}"


@pytest.mark.parametrize("stem", [
    "fig01_canonical_resolution_workflow", "fig02_alias_graph", "fig03_replicate_grouping",
    "fig04_dataset_composition", "fig05_provenance_flow", "fig06_benchmark_lock",
    "fig07_frozen_dependency_graph", "fig08_v5_control_baseline",
])
def test_figures_exist_in_vector_and_raster(stem):
    assert (F / f"{stem}.svg").is_file()
    assert (F / f"{stem}.png").is_file()


# ── K. no phase-01 work leaked in ─────────────────────────────────────────────
def test_no_phase01_artifacts_exist():
    """Phase 00 must not have produced balanced references, LSMs, CSMs, themes or a BSV."""
    forbidden = ["balanced_references", "lsm_", "csm_", "theme_membership", "bsv_"]
    hits = [p.name for p in (T.glob("*") if T.is_dir() else [])
            if any(f in p.name for f in forbidden)]
    assert not hits, f"later-phase artefacts present in Phase 00: {hits}"


def test_no_model_files_written():
    suffixes = {".npz", ".npy", ".pkl", ".joblib", ".pt", ".h5"}
    hits = [p.relative_to(P00).as_posix() for p in P00.rglob("*")
            if p.is_file() and p.suffix.lower() in suffixes]
    assert not hits, f"Phase 00 must not write model files: {hits}"
