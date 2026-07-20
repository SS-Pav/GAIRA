"""GAIRA V6 — Molecular Spectral Signatures (MSS) layer tests.

The MSS layer is ADDITIVE and DERIVED from the frozen atlas. These tests assert:
  1. it is a pure function of the frozen artifacts (fingerprint-locked, reproducible),
  2. the derivation recovers the correct chemistry (purine->adenine components, etc.),
  3. it is genuinely input-sensitive (a purine-loaded query elevates the purine motif),
  4. it never mutates the frozen engine/BSV.
"""
import json
import sys
import hashlib
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"
ART = REPO / "results/v5_rebuild/engine_v1/artifacts"
needs_art = pytest.mark.skipif(not (FROZEN / "manifold.json").exists(), reason="frozen atlas absent")


@needs_art
def test_mss_layer_fingerprint_locked_and_deterministic():
    from gaira.engine.mss import MSSLayer
    from gaira.engine.versioning import VERSIONS
    a, b = MSSLayer(), MSSLayer()
    assert a.registry()["atlas_fingerprint"] == VERSIONS.atlas_fingerprint
    # pure function of the frozen artifacts -> byte-identical across constructions
    assert json.dumps(a.registry(), sort_keys=True) == json.dumps(b.registry(), sort_keys=True)


@needs_art
def test_mss_covers_every_biochemical_theme():
    from gaira.engine.mss import MSSLayer
    from gaira.engine.ontology import Ontology
    L = MSSLayer()
    onto = Ontology()
    covered = {m.parent_theme for m in L.motifs}
    for t in onto.biochemical_theme_ids:
        assert t in covered, f"no MSS motif maps to biochemical theme {t}"
    assert len(L.motifs) >= len(onto.biochemical_theme_ids)


@needs_art
def test_derivation_recovers_expected_chemistry():
    """Curated motifs must derive the correct latent components from frozen data."""
    from gaira.engine.mss import MSSLayer
    L = MSSLayer()
    by_id = {m.id: m for m in L.motifs}
    top = lambda mid: {c["component"] for c in by_id[mid].contributors}
    assert {3, 15} & top("purine_ring_breathing")          # adenine / guanine components
    assert 17 in top("pyrimidine_ring")                    # uracil/cytosine component
    assert 2 in top("protein_amide_backbone")              # protein component
    assert 19 in top("sulfur_heterocycle_thione")          # ergothioneine component
    assert 0 in top("flavin_redox_cofactor")               # riboflavin component
    assert 8 in top("porphyrin_macrocycle")                # heme-protein component


@needs_art
def test_contributor_weights_normalised():
    from gaira.engine.mss import MSSLayer
    L = MSSLayer()
    for m in L.motifs:
        w = sum(c["weight"] for c in m.contributors)
        assert m.contributors, f"{m.id} has no contributors"
        # weights are stored 4-decimal rounded (<=6 contributors -> <=3e-4 drift)
        assert abs(w - 1.0) < 2e-3, f"{m.id} weights sum to {w}"
        assert len(m.contributors) <= L.max_contrib


@needs_art
def test_uricase_depletion_only_on_purine_motifs():
    """Uricase depletion is purine-specific; it must not decorate other motifs."""
    from gaira.engine.mss import MSSLayer
    L = MSSLayer()
    for m in L.motifs:
        dep = m.perturbation["depletion_matches"]
        if m.parent_theme != "nucleic_purine":
            assert dep == [], f"{m.id} wrongly carries uricase depletion evidence"


@needs_art
def test_activation_is_input_sensitive_for_purine():
    """A purine-loaded activation must elevate the purine motif above a flat one."""
    from gaira.engine import GAIRAEngine
    from gaira.engine.mss import MSSLayer
    eng = GAIRAEngine()
    L = MSSLayer.from_engine(eng)

    def purine_elevation(a):
        acts = L.activate(eng.infer(coordinates=a).bsv)
        return next(x.elevation for x in acts if x.id == "purine_ring_breathing")

    flat = np.full(24, 1.0 / 24)
    loaded = np.full(24, 0.01); loaded[3] = 0.5; loaded[15] = 0.3   # adenine+guanine components
    assert purine_elevation(loaded) > purine_elevation(flat)


@needs_art
def test_activate_does_not_mutate_bsv_or_atlas():
    from gaira.engine import GAIRAEngine
    from gaira.engine.mss import MSSLayer
    eng = GAIRAEngine()
    fp0 = eng.atlas.meta["fingerprint"]
    W = np.load(FROZEN / "manifold_components.npz")["components"]
    disk_fp = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()[:32]
    bsv = eng.infer(coordinates=np.full(24, 1.0 / 24)).bsv
    coord_before = bsv.component_coord.copy()
    MSSLayer.from_engine(eng).activate(bsv)
    assert np.allclose(bsv.component_coord, coord_before)     # read-only on the BSV
    assert eng.atlas.meta["fingerprint"] == fp0 == disk_fp    # atlas untouched


@needs_art
def test_registry_artifact_matches_live_derivation():
    """The committed artifact must equal a fresh derivation (no drift)."""
    from gaira.engine.mss import MSSLayer
    p = ART / "mss_registry_v1.json"
    if not p.exists():
        pytest.skip("registry not built")
    on_disk = json.loads(p.read_text())
    live = MSSLayer().registry()
    assert json.dumps(on_disk, sort_keys=True) == json.dumps(live, sort_keys=True)
