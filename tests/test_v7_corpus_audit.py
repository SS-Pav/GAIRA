"""GAIRA V7 — corpus identity, modality purity and canonical-count enforcement.

These tests exist to make the pre-Phase-02 audit *binding* rather than a one-time report.
Every failure mode below is one that would silently corrupt the V7 representation foundation
and produce no error anywhere in Phase 00 or Phase 01:

    a leaked Ag-SERS spectrum          → plasmonically distorted bands treated as pure Raman
    a missing Gobbato pure-Raman file  → up to 21 canonical molecules vanish without a warning
    a collapsed anomer/enantiomer      → a distinct reference permanently destroyed
    a one-to-many canonical mapping    → the same spectrum counted as two molecules

The audited counts are asserted as literals on purpose. If the corpus legitimately changes,
these tests are supposed to fail and force a re-audit rather than absorb the drift.
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

AUD = REPO / "results/v7_rebuild/corpus_audit"
T, A = AUD / "tables", AUD / "artifacts"
P00 = REPO / "results/v7_rebuild/phase00"
P01 = REPO / "results/v7_rebuild/phase01"

ran = pytest.mark.skipif(not (A / "corpus_audit_summary_v1.json").is_file(),
                         reason="the corpus audit has not been run in this checkout")

# ── The audited manifest. Drift from any of these must fail loudly. ───────────
MANIFEST = {
    "raw_raman_spectra": 375,
    "dataset_specific_source_labels": 212,
    "distinct_raw_label_strings": 194,
    "normalized_analyte_names": 167,
    "canonical_molecule_ids": 154,
    "chemistry_classes": 16,
}
PER_SOURCE = {"RamanBioLib": 202, "gobbato_raman_metabolites": 153,
              "amino_acid_raman_grounding": 20}
GOBBATO_RAMAN_FILES = 153
GOBBATO_RAMAN_LABELS = 51
GOBBATO_SERS_FILES = 265
PHASE01_REGISTRY_FINGERPRINT = "208482d6f7178b5b8f16cace91be55b0"

# Substrates/media that mark a spectrum as enhanced or as a biological mixture. Matched
# against the substrate and source fields — never against file names, which are not evidence
# of modality.
PLASMONIC_PAT = re.compile(
    r"ag[\s_-]*(colloid|nanop|np|flake|film|island)|silver|au[\s_-]*(colloid|nanop|np)"
    r"|gold[\s_-]*(colloid|nanop|np)|sers|nanostructur|roughen|klarite",
    re.I)
# NB "ev" is delimited by non-letters rather than \b, so that "small2023_ev" (underscore is a
# word character) is caught while "evidence" is not.
MIXTURE_PAT = re.compile(
    r"serum|plasma|urine|saliva|f(a)?ec|exosom|vesicl|blood|lysate|(?:^|[^a-z])ev(?:[^a-z]|$)",
    re.I)


@pytest.fixture(scope="module")
def summary():
    return json.loads((A / "corpus_audit_summary_v1.json").read_text())


@pytest.fixture(scope="module")
def inv():
    return pd.read_csv(T / "spectrum_level_audit_v1.csv")


@pytest.fixture(scope="module")
def included(inv):
    return inv[inv["included_in_v7_raman"].astype(bool)].copy()


# ── A. MODALITY PURITY ───────────────────────────────────────────────────────
@ran
def test_no_ag_sers_spectrum_enters_the_v7_raman_corpus(included):
    """Not one enhanced spectrum in the representation foundation.

    Checked on the substrate field rather than the modality flag, so that a record whose
    modality was mislabelled upstream is still caught by what it was measured on.
    """
    bad = included[included["substrate"].fillna("").astype(str).apply(
        lambda s: bool(PLASMONIC_PAT.search(s)))]
    # 'Gold-coated glass' (insulin, id 197) is a smooth reflective film, not a plasmonic
    # nanostructure: it is retained and flagged in the audit, and excluded from this pattern
    # by construction (no colloid/nanoparticle/roughening term). Anything else is a leak.
    assert bad.empty, f"plasmonic substrate(s) in the Raman corpus:\n{bad[['spectrum_id', 'substrate']]}"


@ran
def test_declared_modality_is_raman_for_every_included_spectrum(included):
    assert set(included["modality"].str.lower()) == {"raman"}


@ran
def test_no_serum_or_mixture_spectrum_enters_the_v7_raman_corpus(included):
    """P-04: the foundation is fitted on pure references only; mixtures are projected, never fitted."""
    fields = (included["source_dataset"].astype(str) + " "
              + included["substrate"].fillna("").astype(str))
    bad = included[fields.apply(lambda s: bool(MIXTURE_PAT.search(s)))]
    assert bad.empty, f"mixture/biofluid spectra in the Raman corpus:\n{bad[['spectrum_id', 'source_dataset']]}"
    assert included["is_pure_analyte"].astype(bool).all()


@ran
def test_only_the_three_audited_pure_raman_sources_contribute(included):
    assert set(included["source_dataset"]) == set(PER_SOURCE)


@ran
def test_every_excluded_spectrum_carries_a_reason(inv):
    excluded = inv[~inv["included_in_v7_raman"].astype(bool)]
    assert not excluded.empty
    assert excluded["exclusion_reason"].fillna("").str.len().gt(0).all()


# ── B. GOBBATO COMPLETENESS AND RAMAN/SERS SEPARATION ────────────────────────
@ran
def test_every_expected_gobbato_pure_raman_file_is_present(summary):
    g = summary["gobbato"]
    assert g["archive_raman_files"] == GOBBATO_RAMAN_FILES
    assert g["archive_raman_parseable"] == GOBBATO_RAMAN_FILES
    assert g["loaded_raman_spectra"] == GOBBATO_RAMAN_FILES
    assert g["raman_files_missing_from_corpus"] == 0
    assert g["archive_raman_unmatched_filenames"] == []


@ran
def test_gobbato_replicate_structure_is_exactly_three_per_label(summary):
    g = summary["gobbato"]
    assert g["replicates_per_label"] == {"3": GOBBATO_RAMAN_LABELS}
    assert g["archive_raman_source_labels"] == GOBBATO_RAMAN_LABELS
    assert GOBBATO_RAMAN_LABELS * 3 == GOBBATO_RAMAN_FILES


@ran
def test_no_gobbato_ag_sers_file_leaks_into_the_corpus(summary):
    g = summary["gobbato"]
    assert g["archive_sers_files"] == GOBBATO_SERS_FILES
    assert g["sers_files_leaked_into_corpus"] == 0
    assert g["loaded_sers_spectra_EXCLUDED"] == GOBBATO_SERS_FILES


@ran
def test_gobbato_raman_and_sers_records_are_never_conflated():
    """The two modalities share one archive; they must be separable by their record fields."""
    ram = pd.read_csv(T / "gobbato_pure_raman_inventory.csv")
    ser = pd.read_csv(T / "gobbato_ag_sers_inventory.csv")
    assert not set(ram["archive_file"]) & set(ser["archive_file"])
    assert set(ram["modality"].str.lower()) == {"raman"}
    assert set(ser["modality"].str.lower()) == {"ag_sers"}
    # every Raman file is loaded; no SERS file is
    assert ram["loaded"].astype(bool).all()
    assert not ser["included_in_v7_raman"].astype(bool).any()
    # the two filename conventions are disjoint, so a mixed listing cannot silently merge
    assert ram["archive_file"].str.startswith("Raman_pwd_").all()
    assert ser["archive_file"].str.startswith("SERS_met_").all()


@ran
def test_gobbato_loader_returns_both_modalities_and_caller_must_filter():
    """A latent hazard, pinned.

    ``load_gobbato_785`` returns Raman *and* Ag-SERS in one list; only the caller filters.
    If it ever starts returning a pre-filtered list, this test fails and the caller-side
    filter in the V7 corpus builder must be re-checked before it is removed as redundant.
    """
    from gaira.data import gobbato
    src = inspect.getsource(gobbato.load_gobbato_785).lower()
    assert "modality.sers" in src and "modality.raman" in src, \
        "the Gobbato loader no longer emits both modalities — re-check the caller-side filter"
    builder = (AUD / "code" / "audit_corpus.py").read_text().lower()
    assert "modality" in builder and "raman" in builder


# ── C. CANONICAL IDENTITY ────────────────────────────────────────────────────
@ran
def test_no_protected_stereoisomer_or_anomer_is_collapsed():
    prot = pd.read_csv(T / "canonicalization_protected_distinctions.csv")
    assert not prot.empty
    assert not prot["collapsed"].astype(bool).any(), \
        f"protected distinction collapsed:\n{prot[prot['collapsed'].astype(bool)]}"
    assert (prot["canonical_a"] != prot["canonical_b"]).all()
    assert set(prot["status"]) == {"protected"}


@ran
def test_canonical_ids_are_unique():
    reg = pd.read_csv(T / "canonical_molecule_registry_v2.csv")
    assert reg["canonical_id"].is_unique
    assert len(reg) == MANIFEST["canonical_molecule_ids"]


@ran
def test_one_raw_spectrum_never_maps_to_more_than_one_canonical_molecule(included):
    per_spectrum = included.groupby("spectrum_id")["canonical_id"].nunique()
    offenders = per_spectrum[per_spectrum > 1]
    assert offenders.empty, f"spectra with multiple canonical ids: {list(offenders.index)}"
    assert included["spectrum_id"].is_unique
    conflicts = pd.read_csv(T / "canonicalization_one_to_many_conflicts.csv")
    assert conflicts.empty


@ran
def test_no_canonical_molecule_carries_conflicting_chemistry_classes(included):
    per_mol = included.groupby("canonical_id")["chemistry_class"].nunique()
    offenders = per_mol[per_mol > 1]
    assert offenders.empty, f"conflicting class assignment: {list(offenders.index)}"


@ran
def test_every_recorded_merge_is_classified_and_none_is_accidental():
    merges = pd.read_csv(T / "canonicalization_many_to_one_audit.csv")
    assert not merges.empty
    assert merges["merge_classification"].fillna("").str.len().gt(0).all()
    bad = merges[merges["merge_classification"].str.contains(
        "incorrect|accidental|unresolved", case=False, na=False)]
    assert bad.empty, f"merge classified as incorrect:\n{bad}"


@ran
def test_acid_base_merges_were_empirically_verified_as_same_protonation_state():
    """Merging an acid with its conjugate base is forbidden *silently*; here it is evidenced.

    The discriminator is the C=O stretch near 1710 cm-1: present in the free acid, absent in
    the carboxylate. Comparable share in both members means one material labelled two ways.
    """
    ab = pd.read_csv(T / "acid_base_merge_verification.csv")
    assert not ab.empty
    verified = ab[ab["verdict"].str.contains("same protonation", case=False, na=False)]
    assert len(verified) == len(ab.dropna(subset=["cosine"])), \
        f"unverified acid/base merge:\n{ab[~ab.index.isin(verified.index)]}"


# ── D. COUNT MANIFEST ────────────────────────────────────────────────────────
@ran
@pytest.mark.parametrize("key,expected", sorted(MANIFEST.items()))
def test_audited_counts_do_not_drift(summary, key, expected):
    assert summary[key] == expected, (
        f"{key}: {summary[key]} != audited {expected}. If the corpus legitimately changed, "
        "re-run the corpus audit and update the manifest deliberately.")


@ran
def test_per_source_spectrum_counts_do_not_drift(included):
    got = included["source_dataset"].value_counts().to_dict()
    assert got == PER_SOURCE
    assert len(included) == MANIFEST["raw_raman_spectra"]


@ran
def test_the_count_reconciliation_chain_is_monotone_and_ends_at_154():
    rec = pd.read_csv(T / "count_reconciliation_v1.csv")
    counts = list(rec["count"])
    assert counts == sorted(counts, reverse=True), "reconciliation must never increase"
    assert counts[0] == MANIFEST["raw_raman_spectra"]
    assert counts[-1] == MANIFEST["canonical_molecule_ids"]


@ran
def test_unique_chemical_structures_are_reported_as_undetermined(summary):
    """No source carries InChIKey/SMILES/CID. Uncertainty is preserved, not guessed."""
    assert summary["unique_chemical_structures"] is None
    assert summary["structure_identifier_note"]


@ran
def test_phase00_and_the_audit_agree_on_the_canonical_count():
    state = json.loads((P00 / "PHASE_STATE.json").read_text())
    assert state["corpus"]["n_spectra"] == MANIFEST["raw_raman_spectra"]
    assert state["canonical"]["n_surface_forms"] == MANIFEST["normalized_analyte_names"]
    assert state["canonical"]["n_canonical_ids"] == MANIFEST["canonical_molecule_ids"]
    assert (state["canonical"]["n_surface_forms"] - state["canonical"]["n_merges"]
            == state["canonical"]["n_canonical_ids"])
    assert state["partition"]["n_fine_classes"] == MANIFEST["chemistry_classes"]


# ── E. PHASE 01 REPRODUCIBILITY FROM THE AUDITED CORPUS ──────────────────────
@pytest.mark.skipif(not (P01 / "artifacts" / "lsm_manifest_v1.json").is_file(),
                    reason="Phase 01 has not been run in this checkout")
def test_phase01_dictionary_reproduces_from_the_audited_corpus():
    man = json.loads((P01 / "artifacts" / "lsm_manifest_v1.json").read_text())
    blob = json.dumps(man)
    assert PHASE01_REGISTRY_FINGERPRINT in blob, (
        "the Phase 01 LSM registry fingerprint changed. The corpus audit found no corpus "
        "change, so a fingerprint change means something else moved and must be explained.")


@pytest.mark.skipif(not (P01 / "artifacts" / "lsm_manifest_v1.json").is_file(),
                    reason="Phase 01 has not been run in this checkout")
def test_phase01_was_fitted_on_the_audited_corpus_size():
    """Phase 01's own artefacts must be sized by the audited corpus, not merely mention it."""
    man = json.loads((P01 / "artifacts" / "phase_01_manifest_v1.json").read_text())
    rows = {a["artifact_id"]: a.get("rows") for a in man["outputs"] if "rows" in a}
    per_spectrum = {k: v for k, v in rows.items() if v == MANIFEST["raw_raman_spectra"]}
    per_molecule = {k: v for k, v in rows.items() if v == MANIFEST["canonical_molecule_ids"]}
    assert per_spectrum, f"no Phase 01 artefact has one row per audited spectrum: {rows}"
    assert per_molecule, f"no Phase 01 artefact has one row per canonical molecule: {rows}"


# ── F. THE AUDIT IS PRESENT AND SELF-CONSISTENT ──────────────────────────────
@ran
@pytest.mark.parametrize("name", [
    "spectrum_level_audit_v1.csv", "gobbato_pure_raman_inventory.csv",
    "gobbato_ag_sers_inventory.csv", "gobbato_raman_sers_pair_map.csv",
    "gobbato_unique_raman_molecules.csv", "canonicalization_many_to_one_audit.csv",
    "canonicalization_one_to_many_conflicts.csv",
    "canonicalization_protected_distinctions.csv", "canonicalization_unresolved.csv",
    "acid_base_merge_verification.csv", "count_reconciliation_v1.csv",
    "canonical_count_by_source_v1.csv", "canonical_count_by_class_v1.csv",
    "spectra_per_canonical_molecule_v1.csv", "canonical_molecule_registry_v2.csv",
])
def test_audit_table_exists(name):
    assert (T / name).is_file()


def test_audit_report_exists_and_distinguishes_the_four_units():
    rep = (REPO / "GAIRA_v7_rebuild/reports"
           / "PHASE_01_CORPUS_IDENTITY_AND_COMPLETENESS_AUDIT.md")
    assert rep.is_file()
    text = rep.read_text()
    for term in ("SPECTRUM", "SOURCE LABEL", "NORMALIZED NAME", "CANONICAL MOLECULE",
                 "UNIQUE CHEMICAL STRUCTURE"):
        assert term in text, f"the report must define {term} explicitly"
    assert "375" in text and "167" in text and "154" in text


@ran
def test_class_and_source_tables_sum_correctly():
    by_class = pd.read_csv(T / "canonical_count_by_class_v1.csv")
    assert by_class["canonical"].sum() == MANIFEST["canonical_molecule_ids"]
    assert by_class["spectra"].sum() == MANIFEST["raw_raman_spectra"]
    assert len(by_class) == MANIFEST["chemistry_classes"]
    by_src = pd.read_csv(T / "canonical_count_by_source_v1.csv")
    assert by_src["spectra"].sum() == MANIFEST["raw_raman_spectra"]
    # per-source canonical counts exceed the total: 55 molecules appear in >1 library
    assert by_src["canonical"].sum() > MANIFEST["canonical_molecule_ids"]
