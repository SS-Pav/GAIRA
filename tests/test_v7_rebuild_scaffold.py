"""Structural tests for the GAIRA V7 rebuild scaffold.

This suite checks STRUCTURE and NON-MODIFICATION only. V7 is specified, not implemented,
so there is deliberately nothing scientific to test here yet:

  * the V7 directory tree and its required documents exist
  * the frozen V5 atlas fingerprint is unchanged
  * no V7 document hard-codes a local absolute path
  * V7 documents define CSM consistently and describe the BSV as absolute
  * the V7 pass created no model files
  * no existing V5/V6 artefact was modified

Scientific-model tests belong in later phases.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
V7 = REPO / "GAIRA_v7_rebuild"
FOUNDATION = REPO / "assets" / "foundation"

CANONICAL_ATLAS_FINGERPRINT = "09ed804a40836f4a05a91ba10900cded"


def _v7_markdown() -> list[Path]:
    return sorted(p for p in V7.rglob("*.md") if p.is_file())


# ─────────────────────────── directory structure ───────────────────────────
def test_v7_root_exists():
    assert V7.is_dir(), "GAIRA_v7_rebuild/ must exist at the repository root"
    assert (V7 / "README.md").is_file()


@pytest.mark.parametrize("d", [
    "context", "plan", "architecture", "phases", "code", "data_contracts",
    "results", "reports", "tests", "archive",
    "results/tables", "results/figures", "results/manifests",
    "results/checkpoints", "results/phase_outputs",
])
def test_v7_directories_exist(d):
    assert (V7 / d).is_dir(), f"missing V7 directory: {d}"


@pytest.mark.parametrize("name", [
    "GAIRA_V7_CONTEXT.md",
    "PRIOR_ARCHITECTURE_LIMITATIONS.md",
    "SCIENTIFIC_DESIGN_PRINCIPLES.md",
    "TERMINOLOGY_AND_DEFINITIONS.md",
    "DATASET_AND_PROVENANCE_CONTEXT.md",
    "REPOSITORY_BASELINE.md",
    "CONSISTENCY_AUDIT.md",
])
def test_context_documents_exist(name):
    p = V7 / "context" / name
    assert p.is_file(), f"missing context document: {name}"
    assert len(p.read_text().strip()) > 500, f"{name} is a stub"


@pytest.mark.parametrize("name", [
    "GAIRA_V7_REBUILD_PLAN.md",
    "PHASE_DEPENDENCY_MAP.md",
    "VALIDATION_AND_DECISION_RULES.md",
    "SUCCESS_CRITERIA.md",
    "RISK_REGISTER.md",
    "GIT_AND_VERSIONING_PLAN.md",
])
def test_planning_documents_exist(name):
    p = V7 / "plan" / name
    assert p.is_file(), f"missing planning document: {name}"
    assert len(p.read_text().strip()) > 500, f"{name} is a stub"


@pytest.mark.parametrize("name", [
    "GAIRA_V7_TARGET_ARCHITECTURE.md",
    "LEARNING_MODE_ARCHITECTURE.md",
    "INFERENCE_MODE_ARCHITECTURE.md",
    "DATA_CONTRACTS.md",
    "ARTIFACT_AND_MANIFEST_SPEC.md",
    "LIVE_DART_COMPATIBILITY.md",
])
def test_architecture_documents_exist(name):
    p = V7 / "architecture" / name
    assert p.is_file(), f"missing architecture document: {name}"
    assert len(p.read_text().strip()) > 500, f"{name} is a stub"


@pytest.mark.parametrize("phase", [
    "phase_00_benchmark_lock",
    # Numbering adopted 2026-08-06: the original plan's Phase 01 (balanced references) and
    # Phase 02 (LSMs) are one pipeline and were merged; everything below shifts down by one.
    "phase_01_balanced_references_and_lsms",
    "phase_02_consensus_spectral_motifs",
    "phase_03_biochemical_themes",
    "phase_04_biochemical_state_vector",
    "phase_05_end_to_end_integration",
    "phase_06_in_domain_raman_validation",
    "phase_07_chemistry_aware_learning",
    "phase_08_targeted_corpus_expansion",
])
def test_phase_directories_exist(phase):
    d = V7 / "phases" / phase
    assert d.is_dir(), f"missing phase directory: {phase}"
    assert (d / "README.md").is_file(), f"{phase} has no README"


def test_every_directory_has_a_readme():
    """Directories that hold work products must say what belongs in them.

    Exempt: leaf result buckets, and the three document collections (context/, plan/,
    architecture/) whose named documents are indexed in the root README.
    """
    exempt = {"results/tables", "results/figures", "results/manifests",
              "results/checkpoints", "results/phase_outputs",
              "context", "plan", "architecture"}
    for d in sorted(p for p in V7.rglob("*") if p.is_dir()):
        rel = d.relative_to(V7).as_posix()
        if rel in exempt or rel.startswith("results/figures/"):
            continue
        # Phase result buckets are self-describing through their phase README and manifest;
        # requiring a README in each of code/tables/figures/logs would be noise.
        if rel.startswith("results/phase_") and rel.count("/") >= 2:
            continue
        assert (d / "README.md").is_file(), f"{rel}/ has no README.md"


# ─────────────────────────── frozen atlas integrity ───────────────────────────
def test_frozen_atlas_fingerprint_unchanged():
    """The V5 atlas fingerprint must be byte-identical before and after the V7 pass."""
    npz = FOUNDATION / "manifold_components.npz"
    assert npz.is_file(), "frozen atlas basis is missing"
    H = np.load(npz)["components"]
    assert H.shape == (24, 676), f"unexpected basis shape {H.shape}"
    recomputed = hashlib.sha256(np.ascontiguousarray(H).tobytes()).hexdigest()[:32]
    assert recomputed == CANONICAL_ATLAS_FINGERPRINT, (
        f"FROZEN ATLAS CHANGED: {recomputed} != {CANONICAL_ATLAS_FINGERPRINT}")


def test_frozen_atlas_manifest_agrees():
    manifest = json.loads((FOUNDATION / "MANIFEST.json").read_text())
    assert manifest["atlas_fingerprint"] == CANONICAL_ATLAS_FINGERPRINT
    assert manifest["versions"]["atlas_fingerprint"] == CANONICAL_ATLAS_FINGERPRINT
    meta = json.loads((FOUNDATION / "manifold.json").read_text())
    assert meta["fingerprint"] == CANONICAL_ATLAS_FINGERPRINT
    assert meta["k"] == 24


def test_frozen_foundation_file_hashes_unchanged():
    """Every file listed in the frozen MANIFEST still matches its recorded SHA-256."""
    manifest = json.loads((FOUNDATION / "MANIFEST.json").read_text())
    for name, rec in manifest["files"].items():
        p = FOUNDATION / name
        assert p.is_file(), f"frozen asset missing: {name}"
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest == rec["sha256"], f"frozen asset modified: {name}"


def test_v7_created_no_model_files():
    """The V7 DOCUMENTATION tree holds no basis, weights or fitted artefacts.

    `results/` is excluded from this invariant. Phase 02.5 was commissioned to write under
    `GAIRA_v7_rebuild/results/phase_02_5_latent_geometry/`, so analysis artefacts now live
    there; the invariant that matters is that the specification tree (context, plan,
    architecture, phases) stays free of fitted objects, and that is what is asserted.
    """
    model_suffixes = {".npz", ".npy", ".pkl", ".joblib", ".pt", ".pth", ".h5", ".onnx"}
    offenders = [p.relative_to(V7).as_posix()
                 for p in V7.rglob("*") if p.is_file() and p.suffix.lower() in model_suffixes
                 and not p.relative_to(V7).as_posix().startswith("results/")]
    assert not offenders, f"V7 documentation pass must create no model files: {offenders}"


def test_v7_contains_no_code_directory_implementation():
    """code/ holds only its README until Phase 00 begins."""
    entries = [p.name for p in (V7 / "code").iterdir() if p.name != "__pycache__"]
    assert entries == ["README.md"], f"unexpected entries in V7 code/: {entries}"


# ─────────────────────────── document hygiene ───────────────────────────
# Always forbidden: a path containing a machine-specific user directory.
HARD_BANNED = [
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"[A-Za-z]:\\\\"),
]
# Forbidden as a real path, but legitimate when the line is stating the prohibition
# (e.g. "`/Volumes/` is gitignored", "no `SSD_Rad` defaults").
POLICY_ONLY = [
    re.compile(r"/Volumes/"),
    re.compile(r"SSD_Rad"),
]
POLICY_CONTEXT = re.compile(r"ignor|never|\bno\b|\bnot\b|prohibit|default|polic|exclud",
                            re.IGNORECASE)


def _scan_for_absolute_paths(paths):
    offenders = []
    for p in paths:
        for lineno, line in enumerate(p.read_text().splitlines(), 1):
            rel = p.relative_to(V7).as_posix()
            for pat in HARD_BANNED:
                m = pat.search(line)
                if m:
                    offenders.append(f"{rel}:{lineno}: {m.group(0)!r}")
            for pat in POLICY_ONLY:
                m = pat.search(line)
                if m and not POLICY_CONTEXT.search(line):
                    offenders.append(f"{rel}:{lineno}: {m.group(0)!r} (not a policy statement)")
    return offenders


def test_no_hardcoded_absolute_paths_in_v7_documents():
    offenders = _scan_for_absolute_paths(_v7_markdown())
    assert not offenders, "hard-coded local absolute paths found:\n" + "\n".join(offenders)


def test_no_hardcoded_absolute_paths_in_v7_python():
    offenders = _scan_for_absolute_paths(sorted(V7.rglob("*.py")))
    assert not offenders, "hard-coded local absolute paths found:\n" + "\n".join(offenders)


# Documents that introduce the vocabulary must spell the acronym out; downstream documents
# may then use it, since the terminology document is the single binding definition.
PRIMARY_DOCS = [
    "README.md",
    "context/GAIRA_V7_CONTEXT.md",
    "context/TERMINOLOGY_AND_DEFINITIONS.md",
    "architecture/GAIRA_V7_TARGET_ARCHITECTURE.md",
    "architecture/LEARNING_MODE_ARCHITECTURE.md",
    "architecture/INFERENCE_MODE_ARCHITECTURE.md",
    "plan/GAIRA_V7_REBUILD_PLAN.md",
]


def test_csm_is_defined_consistently():
    """CSM is defined once, canonically, and expanded in every primary document."""
    terminology = (V7 / "context" / "TERMINOLOGY_AND_DEFINITIONS.md").read_text()
    assert "Consensus Spectral Motif — CSM" in terminology
    assert "canonical spectroscopic evidence unit" in terminology

    for name in PRIMARY_DOCS:
        text = (V7 / name).read_text()
        assert "CSM" in text, f"{name} never mentions the CSM"
        assert "Consensus Spectral Motif" in text, (
            f"{name} uses 'CSM' without expanding it")


def test_no_document_redefines_csm():
    """No document may give CSM a different expansion."""
    wrong = re.compile(r"CSM\s*(?:=|stands for|means)\s*(?!Consensus Spectral Motif)"
                       r"[A-Z][a-z]", re.IGNORECASE)
    offenders = [p.relative_to(V7).as_posix() for p in _v7_markdown()
                 if wrong.search(p.read_text())]
    assert not offenders, f"CSM redefined in: {offenders}"


def test_mss_is_marked_legacy():
    terminology = (V7 / "context" / "TERMINOLOGY_AND_DEFINITIONS.md").read_text()
    assert "legacy MSS → V7 Consensus Spectral Motif (CSM)" in terminology.lower() or \
           "Legacy MSS → V7 Consensus Spectral Motif (CSM)" in terminology
    assert "legacy terminology" in terminology.lower()

    readme = (V7 / "README.md").read_text()
    assert "legacy terminology" in readme.lower()


def test_the_canonical_coordinate_is_described_as_absolute():
    """The BSV was archived after Phase 05 (A-14); the *invariant* it carried was not.

    This test originally required the phrase "absolute BSV" in three documents. The BSV is now
    legacy and BSV2 is planned, so the phrase moved — but the property must never be optional:
    a canonical coordinate is a position, differences are derived, and neither is a label.
    """
    terminology = (V7 / "context" / "TERMINOLOGY_AND_DEFINITIONS.md").read_text()
    assert "**absolute**" in terminology
    assert "a delta or difference" in terminology
    assert "a hard label" in terminology
    assert "**The BSV is not:**" in terminology, "the legacy definition must be preserved"
    assert "BSV2" in terminology, "the successor must be defined"
    assert "**absolute**" in terminology.split("## BSV2")[1], "BSV2 must inherit absoluteness"

    arch = (V7 / "architecture" / "GAIRA_V7_TARGET_ARCHITECTURE.md").read_text().lower()
    assert "absolute coordinates" in arch or "absolute bsv" in arch
    assert "computed as a difference" in arch


def test_delta_bsv_is_described_as_derived():
    terminology = (V7 / "context" / "TERMINOLOGY_AND_DEFINITIONS.md").read_text()
    assert "ΔBSV" in terminology
    assert "derived" in terminology.lower()


def test_no_document_calls_pca_or_umap_inference():
    """PCA/UMAP are offline or visualisation-only; no document may present them as inference."""
    inference_arch = (V7 / "architecture" / "INFERENCE_MODE_ARCHITECTURE.md").read_text()
    assert "UMAP" in inference_arch and "visualisation" in inference_arch.lower()
    # the prohibition itself must be stated
    for name in ("architecture/GAIRA_V7_TARGET_ARCHITECTURE.md",
                 "architecture/INFERENCE_MODE_ARCHITECTURE.md"):
        text = (V7 / name).read_text()
        assert any(p in text for p in ("never the canonical BSV", "not the canonical BSV",
                                       "never a canonical coordinate")), \
            f"{name} must state that a visualisation projection is not a canonical coordinate"
        assert "describe PCA or UMAP as inference" in text, \
            f"{name} must state the PCA/UMAP prohibition"


def test_second_nmf_is_not_presupposed():
    learning = (V7 / "architecture" / "LEARNING_MODE_ARCHITECTURE.md").read_text()
    assert '**not** "NMF on NMF"' in learning, \
        "learning-mode architecture must state that V7 is not NMF-on-NMF"
    assert "candidate method, not a step" in learning
    rules = (V7 / "plan" / "VALIDATION_AND_DECISION_RULES.md").read_text()
    assert "does not presuppose" in rules


def test_success_criteria_freeze_state_matches_phase00():
    """Before Phase 00 the criteria are provisional; after it they are frozen and pinned."""
    text = (V7 / "plan" / "SUCCESS_CRITERIA.md").read_text()
    state = REPO / "results/v7_rebuild/phase00/PHASE_STATE.json"
    if state.is_file() and json.loads(state.read_text())["status"] == "COMPLETE":
        assert "STATUS: FROZEN in Phase 00" in text
        assert "Frozen at source commit" in text
        assert "0.7507" in text, "the frozen S-01 threshold must be pinned to a number"
    else:
        assert "PROVISIONAL" in text


def test_no_document_claims_v7_performance():
    """V7 is unimplemented; no document may assert a V7 result."""
    banned = [
        re.compile(r"V7 (?:achieves|achieved|outperforms|outperformed|improves by)", re.I),
        re.compile(r"V7 (?:top-1|retrieval) (?:of|is|was) 0\.\d", re.I),
        re.compile(r"V7 results? show", re.I),
    ]
    offenders = []
    for p in _v7_markdown():
        text = p.read_text()
        for pat in banned:
            for m in pat.finditer(text):
                offenders.append(f"{p.relative_to(V7).as_posix()}: {m.group(0)!r}")
    assert not offenders, "V7 performance claimed before implementation:\n" + "\n".join(offenders)


def test_readme_status_is_truthful_about_what_has_been_built():
    """The README must state the current build status, not a frozen snapshot of an old one.

    This test originally asserted "no V7 model has been fitted". That was true through Phase 01
    and is now false: eight phases are complete. The invariant worth testing is not a particular
    sentence but that the README names the completed phases, names the archived ones, and still
    carries the V5 atlas fingerprint it must never modify.
    """
    readme = (V7 / "README.md").read_text()
    low = readme.lower()
    assert CANONICAL_ATLAS_FINGERPRINT in readme
    assert "gaira-v7-rebuild" in readme
    assert "complete" in low, "the README must state which phases are complete"
    assert "archived on evidence" in low, "retired layers must be labelled, not deleted"
    assert "unmodified" in low and "production" in low, \
        "the README must keep stating that the V5 atlas is untouched and in production"
    assert "no v7 model has been fitted" not in low, \
        "this claim is false as of Phase 01 and must not be reintroduced"


def test_readme_phase_status_table_is_current():
    """The status table must not claim a phase is done before its gates pass."""
    readme = (V7 / "README.md").read_text()
    state = REPO / "results/v7_rebuild/phase00/PHASE_STATE.json"
    if state.is_file():
        complete = json.loads(state.read_text())["status"] == "COMPLETE"
        assert ("| 00 | Benchmark lock | ✅ **COMPLETE**" in readme) == complete
    else:
        assert "| 00 | Benchmark lock | Not started" in readme


# ─────────────────────────── existing artefacts untouched ───────────────────────────
@pytest.mark.parametrize("protected", [
    "assets/foundation",
    "results/v5_rebuild",
    "results/v6_rebuild",
    "src/gaira/engine",
    "src/gaira/preprocessing",
    "tools/reproduce_gaira_foundation.py",
])
def test_protected_paths_still_present(protected):
    """V7 must not have removed or relocated any existing scientific asset."""
    assert (REPO / protected).exists(), f"protected path missing: {protected}"


def test_v7_writes_nothing_outside_its_own_tree():
    """No V7 document or script may reference writing into a protected tree."""
    write_into_protected = re.compile(
        r"(?:write|save|output|dump)\w*\s*\(?\s*[\"']?(?:assets/foundation|results/v5_rebuild|"
        r"results/v6_rebuild|src/gaira)", re.I)
    offenders = []
    for p in sorted(V7.rglob("*.py")):
        for m in write_into_protected.finditer(p.read_text()):
            offenders.append(f"{p.relative_to(V7).as_posix()}: {m.group(0)!r}")
    assert not offenders, "V7 code writes into a protected tree:\n" + "\n".join(offenders)


# ─────────────────────────── planning figures ───────────────────────────
@pytest.mark.parametrize("stem", [
    "fig01_flat_vs_hierarchical", "fig02_learning_pipeline", "fig03_inference_pipeline",
    "fig04_coverage_imbalance", "fig05_representation_hierarchy", "fig06_offline_vs_live",
    "fig07_phase_roadmap", "fig08_failure_taxonomy", "fig09_atlas_structure",
    "fig10_dart_trajectory",
])
def test_planning_figures_exist_in_vector_and_raster(stem):
    d = V7 / "results" / "figures" / "planning"
    assert (d / f"{stem}.svg").is_file(), f"missing vector figure: {stem}.svg"
    assert (d / f"{stem}.png").is_file(), f"missing preview figure: {stem}.png"
