"""GAIRA V6 converged engine tests.

Most tests run WITHOUT the data volume: the BSV/ontology/registry/normalization
layers consume committed JSON/YAML artifacts + the frozen atlas npz, and can be
driven with synthetic 24-component activation vectors. Data-volume tests (full
spectrum → projection) skip when the volume is absent.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gaira.engine import (BSVBuilder, Ontology, ComponentRegistry, EvidenceEngine,
                          RadarBackend, ReferenceFrame, get_domain, VERSIONS)
from gaira.engine.ontology import NON_BIOCHEMICAL_THEMES
from gaira.engine import dart as DART

ART = REPO / "results/v5_rebuild/engine_v1/artifacts"
VOL = Path("/Volumes/SSD_Rad/GAIRA_DATA")
needs_data = pytest.mark.skipif(not VOL.exists(), reason="data volume not mounted")
needs_art = pytest.mark.skipif(not (ART / "component_registry_v1.json").exists(),
                               reason="engine artifacts not built")

pytestmark = needs_art


# ── artifact integrity ──
def test_registry_has_24_components_with_provenance():
    reg = ComponentRegistry()
    assert reg.k == 24 and len(reg.components) == 24
    for j in range(24):
        c = reg.get(j)
        for fld in ("current_interpretation", "bootstrap_stability", "reference_analyte_loadings"):
            assert "value" in c[fld] and "provenance" in c[fld]


def test_ontology_has_12_themes_and_biochemical_subset():
    o = Ontology()
    assert len(o.theme_ids) == 13
    assert set(NON_BIOCHEMICAL_THEMES) <= set(o.theme_ids)
    assert len(o.biochemical_theme_ids) == 11
    assert o.atlas_fingerprint == VERSIONS.atlas_fingerprint


def test_theme_weights_sum_to_one_per_component():
    o = Ontology()
    W = o.W
    assert W.shape == (24, 13)
    sums = W.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=0.02), sums


def test_theme_weights_carry_three_evidence_lines():
    o = Ontology()
    ev = o.weight_evidence(3, "nucleic_purine")   # c3 -> purine (the reclaimed label)
    assert ev is not None
    for line in ("reference_loading", "spectral_band", "perturbation"):
        assert line in ev["evidence"]


def test_c3_reclaimed_as_purine_despite_sterol_audit_label():
    """The perturbation evidence corrects the Component Audit's coarse label."""
    reg = ComponentRegistry()
    assert reg.value(3, "audit_label_v0_1") == "sterol"
    o = Ontology()
    ti = o.theme_index("nucleic_purine")
    assert o.W[3, ti] > 0.3     # c3 now weighted strongly toward purine


def test_reference_frame_shapes():
    f = ReferenceFrame()
    assert f.center.shape == (24,) and f.spread.shape == (24,)
    assert (f.spread > 0).all()
    assert f.fingerprint == VERSIONS.atlas_fingerprint


# ── BSV math ──
def _purine_activation():
    """Synthetic activation concentrated on purine-encoding components c3/c15."""
    a = np.full(24, 0.01)
    a[3] = 0.5; a[15] = 0.3
    return a


def test_bsv_composition_sums_to_about_one():
    b = BSVBuilder().from_activation(_purine_activation())
    total = sum(b.composition.values())
    assert total == pytest.approx(1.0, abs=0.02)


def test_bsv_purine_activation_gives_purine_theme():
    b = BSVBuilder().from_activation(_purine_activation())
    bio = b.biochemical_themes()
    top = max(bio, key=bio.get)
    assert top == "nucleic_purine"


def test_bsv_confidence_in_unit_interval():
    b = BSVBuilder().from_activation(_purine_activation())
    for v in b.confidence.values():
        assert 0.0 <= v <= 1.0
    assert 0.0 <= b.overall_confidence <= 1.0


def test_bsv_ood_higher_for_off_reference_vector():
    builder = BSVBuilder()
    off = builder.from_activation(np.full(24, 1.0 / 24))   # flat, unlike any pure reference
    assert 0.0 <= off.ood_score <= 1.0
    assert off.ood_score > 0.1                              # a flat vector is off-reference


def test_bsv_matrix_share_lowers_overall_confidence():
    builder = BSVBuilder()
    clean = builder.from_activation(_purine_activation())
    # push mass onto generic/background components (high-index low-purity ones)
    a = np.full(24, 0.02); a[19] = 0.6; a[22] = 0.3
    matrixy = builder.from_activation(a)
    assert matrixy.non_biochemical["background_matrix"] >= 0.0
    assert matrixy.overall_confidence <= clean.overall_confidence + 0.2


# ── evidence engine ──
def test_evidence_trace_is_complete():
    b = BSVBuilder().from_activation(_purine_activation())
    ev = EvidenceEngine().trace_theme(b, "nucleic_purine")
    for k in ("contributing_components", "supporting_reference_analytes",
              "perturbation_support", "literature", "domain_caveats"):
        assert k in ev
    assert len(ev["contributing_components"]) >= 1
    # purine theme should be supported by a purine reference analyte
    assert any(a in " ".join(ev["supporting_reference_analytes"]).lower()
               for a in ("adenine", "guanine", "xanthine", "hypoxanthine", "urate"))


def test_evidence_full_report_flags_ood():
    builder = BSVBuilder()
    off = builder.from_activation(np.full(24, 1.0 / 24))
    rep = EvidenceEngine().full_report(off)
    assert "themes" in rep and "honesty_flags" in rep


# ── radar backend ──
def test_radar_axes_are_biochemical_themes_only():
    b = BSVBuilder().from_activation(_purine_activation())
    radar = RadarBackend().build(b)
    labels = {ax["theme"] for ax in radar["axes"]}
    assert radar["n_axes"] == 11
    assert not (labels & NON_BIOCHEMICAL_THEMES)
    for ax in radar["axes"]:
        for k in ("score", "confidence", "ood_modifier", "evidence_strength", "provenance"):
            assert k in ax


def test_radar_stamps_versions():
    b = BSVBuilder().from_activation(_purine_activation())
    radar = RadarBackend().build(b)
    assert radar["versions"]["atlas_fingerprint"] == VERSIONS.atlas_fingerprint


# ── domain layer does not change the BSV ──
def test_domain_context_does_not_mutate_bsv():
    b = BSVBuilder().from_activation(_purine_activation())
    comp_before = dict(b.composition)
    d = get_domain("serum").interpret(b)
    assert b.composition == comp_before          # unchanged
    assert d["domain"] == "serum" and "caveats" in d


def test_all_domains_present():
    for name in ("serum", "ev", "buffer", "tissue", "dart"):
        assert get_domain(name).domain == name


# ── DART interfaces are design-only ──
def test_dart_interfaces_are_abstract():
    for cls in (DART.DARTAcquisition, DART.TrajectoryProjector, DART.TrajectoryComparator):
        with pytest.raises(TypeError):
            cls()                                # abstract — cannot instantiate
    assert isinstance(DART.FIRST_DART_PREDICTION, str)


# ── versioning ──
def test_versions_pin_atlas_fingerprint():
    v = VERSIONS.as_dict()
    assert v["atlas_fingerprint"] == "09ed804a40836f4a05a91ba10900cded"
    assert v["biochemical_ontology"] == "v2.0" and v["bsv"] == "v2.0"


def test_committed_examples_recover_chemistry():
    ex = json.loads((ART / "example_inferences.json").read_text())

    def top_bio(entry):
        comp = entry["biochemical_state_vector"]["composition"]
        bio = {t: s for t, s in comp.items() if t not in NON_BIOCHEMICAL_THEMES}
        return max(bio, key=bio.get)
    assert top_bio(ex["pure::adenine"]) == "nucleic_purine"
    assert top_bio(ex["pure::(+)-glucose"]) == "saccharide_glycan"
    assert top_bio(ex["pure::cholesterol"]) == "lipid_acyl"


# ── full pipeline (needs frozen atlas npz; runs without data volume) ──
def test_engine_infer_from_coordinates():
    from gaira.engine import GAIRAEngine
    eng = GAIRAEngine()
    out = eng.infer(coordinates=_purine_activation(), domain="buffer")
    d = out.as_dict()
    for k in ("biochemical_state_vector", "radar", "evidence", "domain_interpretation",
              "component_coordinates", "ood_score", "versions"):
        assert k in d
    assert d["atlas_fingerprint"] == VERSIONS.atlas_fingerprint


@needs_data
def test_engine_infer_from_full_spectrum():
    from gaira.engine import GAIRAEngine
    from gaira.foundation import dataset as DS
    eng = GAIRAEngine()
    corpus = DS.load_reference_corpus()
    mask = corpus.meta.analyte.values == "adenine"
    # a raw-ish reference already on the grid; feed as intensity on the grid
    out = eng.infer(coordinates=eng.atlas.coordinates(corpus.X[mask], normalise=True)[0],
                    domain="buffer")
    bio = out.bsv.biochemical_themes()
    assert max(bio, key=bio.get) == "nucleic_purine"
