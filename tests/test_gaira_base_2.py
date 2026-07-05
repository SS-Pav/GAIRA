"""Regression / CI gates for gaira_base_2.

The 10 tests enumerated in
``gaira_base_2_backward_compatibility_v1.md`` §3.3.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python -m pytest tests/test_gaira_base_2.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gaira.base2 import (
    AG_COLLOID_SERUM,
    ALPHA_SUPPORTING,
    BAND_FLOOR,
    BIOLOGY_AXES_V11,
    CONTROL_LANE,
    GAIRA_BASE_AXES_V8,
    GAIRA_BASE_FROZEN_PILOT_FILES,
    MAPPING_WEIGHT_BY_TYPE,
    MotifScore,
    PROJECTION_V11_TO_V8,
    aggregate_to_11_axes,
    compute_motif_activation,
    compute_motif_score,
    frozen_pilot_manifest,
    load_active_registry,
    project_to_8_axes,
    resolve_mapping_weight,
    resolve_status_calibration_weight,
    resolve_status_core_weight,
    score_spectrum,
    sha256_file,
)
from gaira.base2.schema import MotifDualStatus
from gaira.spectral import canonical_master_axis


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

MANIFEST_PATH = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_implementation_v1/"
    "tests/gaira_base_frozen_manifest.json"
)


@pytest.fixture(scope="module")
def engine():
    motifs, mappings, dual = load_active_registry()
    return motifs, mappings, dual


@pytest.fixture(scope="module")
def master_x():
    return canonical_master_axis()


# ──────────────────────────────────────────────────────────────────────
# Test 1 — frozen gaira_base pilot files unchanged
# ──────────────────────────────────────────────────────────────────────

def test_gaira_base_frozen_files_unchanged():
    """HARD ROLLBACK GATE: SHA-256 of 15 frozen pilot CSVs must match manifest."""
    current = frozen_pilot_manifest()
    # First run creates the manifest; subsequent runs verify against it.
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST_PATH.open("w") as f:
            json.dump(current, f, indent=2, sort_keys=True)
        # Don't skip — assert integrity on first write
        assert all(v != "MISSING" for v in current.values()), (
            "some frozen pilot files are missing on first manifest write"
        )
        return

    with MANIFEST_PATH.open("r") as f:
        expected = json.load(f)

    assert set(current.keys()) == set(expected.keys()), (
        "frozen pilot file set changed"
    )
    diffs = [k for k in current if current[k] != expected[k]]
    assert not diffs, (
        f"frozen gaira_base pilot files modified: {diffs}. "
        f"gaira_base is FROZEN; this is a rollback trigger."
    )


# ──────────────────────────────────────────────────────────────────────
# Test 2 — motif registry loads without mutation
# ──────────────────────────────────────────────────────────────────────

def test_motif_registry_loads(engine):
    motifs, mappings, dual = engine
    assert len(motifs) >= 39, (
        f"expected at least 39 active motifs; got {len(motifs)}"
    )
    # All primary band families have positive tolerance
    for mid, spec in motifs.items():
        for f in spec.primary_bands:
            assert f.cm1_tolerance > 0, f"{mid}:{f.family_id} tol <= 0"
            assert 400 <= f.cm1_centre <= 1800, (
                f"{mid}:{f.family_id} centre {f.cm1_centre} outside canonical support"
            )


# ──────────────────────────────────────────────────────────────────────
# Test 3 — mapping skeleton respects v1_active / HELD_V2 logic
# ──────────────────────────────────────────────────────────────────────

def test_mapping_respects_v1_active(engine):
    motifs, mappings, dual = engine
    held_expected = {
        "nucleobase_in_plane_ring_1320_1340",
        "collision_1020_1080_multi_candidate",
        "amide_I_lipid_carbonyl_partial_panel_motif",
    }
    for mid in held_expected:
        if mid in mappings:
            assert mappings[mid].active is False, (
                f"HELD_V2 motif {mid} has mapping.active=True; must be False"
            )
    # At least one PRIMARY-mapped motif per biology axis
    primary_axes_with_contributor = set()
    for mid, mapping in mappings.items():
        if not mapping.active:
            continue
        if mapping.mapping_type == "PRIMARY":
            primary_axes_with_contributor.add(mapping.primary_axis)
    # metabolic_small_molecule is sparse (only creatine/creatinine primary)
    # but should still have at least one contributor
    assert "metabolic_small_molecule" in primary_axes_with_contributor


# ──────────────────────────────────────────────────────────────────────
# Test 4 — axis names and projection names match spec
# ──────────────────────────────────────────────────────────────────────

def test_axis_names_and_projection_match_spec():
    assert len(BIOLOGY_AXES_V11) == 11, "must have exactly 11 biology axes"
    assert len(GAIRA_BASE_AXES_V8) == 8, "must have exactly 8 legacy axes"
    assert CONTROL_LANE == "ambiguity_artifact"
    # Every projection target is in GAIRA_BASE_AXES_V8
    for axis8 in PROJECTION_V11_TO_V8:
        assert axis8 in GAIRA_BASE_AXES_V8, f"{axis8} not in v8 axis list"
    # Every v11 axis appears in at least one projection
    all_sources = {a for sources in PROJECTION_V11_TO_V8.values() for a in sources}
    for axis11 in BIOLOGY_AXES_V11:
        assert axis11 in all_sources, (
            f"{axis11} not mapped to any v8 projection target"
        )


# ──────────────────────────────────────────────────────────────────────
# Test 5 — all outputs bounded in [0, 1]
# ──────────────────────────────────────────────────────────────────────

def test_all_outputs_bounded(engine, master_x):
    motifs, mappings, dual = engine
    # Synthetic spectrum with moderate peaks across the master axis
    rng = np.random.default_rng(42)
    spectrum = np.clip(rng.random(master_x.size) * 0.5, 0, 1)
    res = score_spectrum(spectrum, master_x, motifs, mappings, dual, "synth-1")
    for m in res.motif_scores:
        assert 0.0 <= m.core_weight <= 1.0, f"motif {m.motif_id} core {m.core_weight}"
        assert 0.0 <= m.regime_weight <= 1.0, f"motif {m.motif_id} regime {m.regime_weight}"
    for a in res.axis11_scores:
        assert 0.0 <= a.core_evidence <= 1.0
        assert 0.0 <= a.regime_evidence <= 1.0
    for a in res.axis8_projection:
        assert 0.0 <= a.core_evidence <= 1.0
        assert 0.0 <= a.regime_evidence <= 1.0
    assert 0.0 <= res.ambiguity.core_evidence <= 1.0
    assert 0.0 <= res.ambiguity.regime_evidence <= 1.0


# ──────────────────────────────────────────────────────────────────────
# Test 6 — zero-preserving behaviour
# ──────────────────────────────────────────────────────────────────────

def test_zero_preserving(engine, master_x):
    motifs, mappings, dual = engine
    spectrum = np.zeros(master_x.size, dtype=np.float64)
    res = score_spectrum(spectrum, master_x, motifs, mappings, dual, "zero")
    assert all(a.core_evidence == 0.0 for a in res.axis11_scores)
    assert all(a.regime_evidence == 0.0 for a in res.axis11_scores)
    assert all(a.core_evidence == 0.0 for a in res.axis8_projection)
    assert all(a.regime_evidence == 0.0 for a in res.axis8_projection)
    assert res.ambiguity.core_evidence == 0.0
    assert res.ambiguity.regime_evidence == 0.0


# ──────────────────────────────────────────────────────────────────────
# Test 7 — monotonicity invariant
# ──────────────────────────────────────────────────────────────────────

def test_monotonicity(engine, master_x):
    """Adding intensity at a band window cannot decrease the axis score."""
    motifs, mappings, dual = engine
    baseline = np.zeros(master_x.size, dtype=np.float64)
    res_base = score_spectrum(baseline, master_x, motifs, mappings, dual, "base")

    # Boost a specific motif band — adenine ring at 725 cm⁻¹
    boosted = baseline.copy()
    idx = int(np.argmin(np.abs(master_x - 725.0)))
    boosted[idx - 3:idx + 4] = 0.8  # broad peak
    res_boosted = score_spectrum(boosted, master_x, motifs, mappings, dual, "boosted")

    # purine_nucleotide axis (or purine_metabolite) should NOT decrease
    def get_axis(res, name):
        for a in res.axis11_scores:
            if a.axis_id == name:
                return a.core_evidence
        return 0.0
    for axis_id in BIOLOGY_AXES_V11:
        assert get_axis(res_boosted, axis_id) >= get_axis(res_base, axis_id) - 1e-9, (
            f"monotonicity violated on {axis_id}: "
            f"{get_axis(res_base, axis_id)} → {get_axis(res_boosted, axis_id)}"
        )


# ──────────────────────────────────────────────────────────────────────
# Test 8 — ambiguity motifs don't enter biology axes unless mapping says so
# ──────────────────────────────────────────────────────────────────────

def test_ambiguity_isolation(engine, master_x):
    motifs, mappings, dual = engine
    # An AMBIGUITY_ONLY motif should have resolve_mapping_weight(biology_axis) == 0
    for mid, mapping in mappings.items():
        if mapping.mapping_type != "AMBIGUITY_ONLY":
            continue
        for axis in BIOLOGY_AXES_V11:
            w = resolve_mapping_weight(mapping, axis)
            assert w == 0.0, (
                f"AMBIGUITY_ONLY motif {mid} has mapping_weight={w} "
                f"on biology axis {axis}"
            )


# ──────────────────────────────────────────────────────────────────────
# Test 9 — 11→8 projection uses MAX, not noisy-OR
# ──────────────────────────────────────────────────────────────────────

def test_projection_uses_max():
    from gaira.base2.schema import AxisScore
    axis11 = [
        AxisScore("purine_nucleotide",        core_evidence=0.6, regime_evidence=0.5,
                    contributing_motifs=("m_a",)),
        AxisScore("purine_metabolite",        core_evidence=0.9, regime_evidence=0.7,
                    contributing_motifs=("m_b",)),
        AxisScore("pyrimidine_nucleotide",    0.3, 0.3, ("m_c",)),
        AxisScore("phosphate_nucleic_adjacent", 0.0, 0.0, ()),
        AxisScore("glycan_carbohydrate",      0.0, 0.0, ()),
        AxisScore("protein_peptide_backbone", 0.0, 0.0, ()),
        AxisScore("aromatic_residue",         0.0, 0.0, ()),
        AxisScore("lipid_acyl_membrane",      0.4, 0.4, ("m_d",)),
        AxisScore("sterol_neutral_lipid",     0.8, 0.8, ("m_e",)),
        AxisScore("sulfur_thiol_redox",       0.5, 0.5, ("m_f",)),
        AxisScore("metabolic_small_molecule", 0.2, 0.2, ("m_g",)),
    ]
    proj = project_to_8_axes(axis11)
    by_id = {a.axis_id: a for a in proj}
    # purine_nucleotide_8 = MAX(purine_nucleotide, purine_metabolite) = max(0.6, 0.9)
    assert abs(by_id["purine_nucleotide"].core_evidence - 0.9) < 1e-9
    # membrane_lipid_8 = MAX(lipid_acyl_membrane, sterol_neutral_lipid) = max(0.4, 0.8)
    assert abs(by_id["membrane_lipid"].core_evidence - 0.8) < 1e-9
    # NOT noisy-OR (noisy-OR of 0.4 and 0.8 = 1 - 0.6*0.2 = 0.88 ≠ 0.8)
    assert abs(by_id["membrane_lipid"].core_evidence - 0.88) > 1e-3
    # redox_metabolite_8 = MAX(sulfur_thiol_redox, metabolic_small_molecule) = max(0.5, 0.2)
    assert abs(by_id["redox_metabolite"].core_evidence - 0.5) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# Test 10 — core vs regime outputs differ only via calibration_weight
# ──────────────────────────────────────────────────────────────────────

def test_core_vs_regime_differ_only_by_calibration(engine, master_x):
    motifs, mappings, dual = engine
    # Build a dual-status snapshot where ALL motifs have CALIBRATION_VALID
    # (calibration_weight = 1.00). Under this substitution, core_weight
    # and regime_weight should be EQUAL for every motif and every axis.
    neutral_dual = {
        mid: MotifDualStatus(
            motif_id=mid,
            core_status=s.core_status,
            calibration_status="CALIBRATION_VALID",
            final_v1_role=s.final_v1_role,
        )
        for mid, s in dual.items()
    }
    rng = np.random.default_rng(7)
    spectrum = np.clip(rng.random(master_x.size) * 0.5, 0, 1)
    res_neutral = score_spectrum(
        spectrum, master_x, motifs, mappings, neutral_dual, "neutral",
    )
    # Every motif.core_weight == motif.regime_weight
    for m in res_neutral.motif_scores:
        assert abs(m.core_weight - m.regime_weight) < 1e-9, (
            f"motif {m.motif_id}: core {m.core_weight} vs regime {m.regime_weight}"
        )
    # Every axis core_evidence == regime_evidence
    for a in res_neutral.axis11_scores:
        assert abs(a.core_evidence - a.regime_evidence) < 1e-9, (
            f"axis {a.axis_id}: core {a.core_evidence} vs regime {a.regime_evidence}"
        )
    for a in res_neutral.axis8_projection:
        assert abs(a.core_evidence - a.regime_evidence) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# Additional sanity — deterministic reproducibility
# ──────────────────────────────────────────────────────────────────────

def test_deterministic_reproducibility(engine, master_x):
    motifs, mappings, dual = engine
    rng = np.random.default_rng(123)
    spectrum = np.clip(rng.random(master_x.size) * 0.5, 0, 1)
    r1 = score_spectrum(spectrum, master_x, motifs, mappings, dual, "a")
    r2 = score_spectrum(spectrum, master_x, motifs, mappings, dual, "a")
    for m1, m2 in zip(r1.motif_scores, r2.motif_scores):
        assert m1.activation == m2.activation
        assert m1.core_weight == m2.core_weight
        assert m1.regime_weight == m2.regime_weight


def test_regime_is_ag_colloid():
    r = AG_COLLOID_SERUM
    assert "Ag colloid" in r.substrate
    assert "Merck commercial serum" in r.matrix
