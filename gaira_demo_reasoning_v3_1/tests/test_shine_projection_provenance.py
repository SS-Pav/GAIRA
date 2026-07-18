"""SHINE must not be presented as an independent recomputed 11-axis projection."""
import inspect
from pathlib import Path
DEMO = Path(__file__).resolve().parent.parent
from gaira_core import v3_1_views as v
from gaira_core import data_loader as dl
from gaira_core import config as cfg


def test_shine_uses_reduced_dimensional_renderer():
    src = inspect.getsource(v.render_global_projection)
    assert "_render_shine_reduced" in src
    # the reduced renderer must NOT draw an 11-axis radar
    rsrc = inspect.getsource(v._render_shine_reduced)
    assert "radar_figure" not in rsrc
    assert "reduced-dimensional" in rsrc.lower() or "Legacy reduced" in rsrc


def test_shine_is_actually_collapsed():
    sh, ph = dl.load_pilot_cohorts("shine_liver_injury")
    if ph or sh is None or sh.empty:
        return
    nz = [(sh.iloc[i][[a for a in cfg.BSV_AXES]].abs() > 1e-4).sum() for i in range(len(sh))]
    assert max(nz) <= 3, "SHINE 11-axis BSV is collapsed upstream (<=3 active axes)"
