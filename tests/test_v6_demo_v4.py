"""GAIRA V6 demo (gaira_demo_reasoning_v4) — guard tests.

The demo is presentation only: it must load the FROZEN engine, render its figures
from real engine outputs, and never mutate the atlas. Streamlit render() functions
need a script context and are exercised by selfcheck.py, not here; these tests cover
the engine bridge, the figures, and that all page modules import cleanly.
"""
import sys
import hashlib
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "gaira_demo_reasoning_v4"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(DEMO))

FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"
needs_art = pytest.mark.skipif(not (FROZEN / "manifold.json").exists(), reason="frozen atlas absent")
needs_st = pytest.mark.skipif(__import__("importlib").util.find_spec("streamlit") is None,
                              reason="streamlit not installed")


@needs_art
def test_bridge_loads_frozen_and_reports_stats():
    from demo_core.engine_bridge import Bridge
    from gaira.engine.versioning import VERSIONS
    b = Bridge()
    s = b.platform_stats()
    assert b.eng.atlas.meta["fingerprint"] == VERSIONS.atlas_fingerprint
    assert s["n_components"] == 24 and s["n_biochemical_themes"] == 11
    assert s["n_mss_motifs"] == 13 and s["n_reference_spectra"] == 375


@needs_art
def test_figures_render_from_real_engine_output():
    import matplotlib
    matplotlib.use("Agg")
    from demo_core.engine_bridge import Bridge
    from demo_core import figures as F, data as D
    b = Bridge()
    Z, meta = D.load_projection("ils_adenine")
    hi = int(np.asarray(meta["conc_uM"], float).argmax())
    out, acts = b.bsv_and_mss(Z[hi], domain="buffer")
    # centerpiece: adenine max-dose must put the purine motif on top by elevation
    bio = [a for a in acts if not a.non_biochemical]
    assert max(bio, key=lambda a: a.elevation).id == "purine_ring_breathing"
    for fig in (F.architecture_diagram(), F.radar(out.radar["axes"]), F.mss_hierarchy(acts),
                F.component_fingerprint(out.bsv.component_coord),
                F.band_collision_map(b.mss.motifs)):
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


@needs_art
def test_demo_does_not_mutate_atlas():
    from demo_core.engine_bridge import Bridge
    b = Bridge()
    W = np.load(FROZEN / "manifold_components.npz")["components"]
    disk_fp = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()[:32]
    b.bsv_and_mss(np.full(24, 1.0 / 24))
    assert b.eng.atlas.meta["fingerprint"] == disk_fp


@needs_art
@needs_st
def test_all_pages_import_and_expose_render():
    from demo_core.pages import (p1_overview, p2_reference_atlas, p3_reasoning, p4_calibration,
                                 p5_serum, p6_biological, p7_dart, p8_methods)
    for m in (p1_overview, p2_reference_atlas, p3_reasoning, p4_calibration,
              p5_serum, p6_biological, p7_dart, p8_methods):
        assert callable(getattr(m, "render", None)), m.__name__
