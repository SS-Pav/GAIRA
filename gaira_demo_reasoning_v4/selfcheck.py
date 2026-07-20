"""Headless self-check for the V6 demo: import every page, run the engine, and
render every publication figure to PNG so visualisations can be audited without a
browser. Also verifies the frozen atlas fingerprint is untouched.

    python selfcheck.py            # renders to ./_selfcheck/
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "_selfcheck"
OUT.mkdir(exist_ok=True)


def main():
    from demo_core.engine_bridge import Bridge
    from demo_core import figures as F, data as D
    import matplotlib.pyplot as plt

    b = Bridge()
    s = b.platform_stats()
    fp = b.eng.atlas.meta["fingerprint"]
    from gaira.engine.versioning import VERSIONS
    assert fp == VERSIONS.atlas_fingerprint, "atlas fingerprint drift!"
    print(f"engine OK · atlas {fp[:12]}… · {s['n_reference_spectra']} spectra · "
          f"{s['n_mss_motifs']} MSS motifs")

    # a real reasoning trace
    Z, meta = D.load_projection("ils_adenine")
    hi = int(np.asarray(meta["conc_uM"], float).argmax())
    out, acts = b.bsv_and_mss(Z[hi], domain="buffer")
    print(f"adenine max-dose top motif: {[a.name for a in acts if not a.non_biochemical][0]}")

    figs = {
        "architecture": F.architecture_diagram(),
        "radar": F.radar(out.radar["axes"]),
        "mss_hierarchy": F.mss_hierarchy(acts),
        "fingerprint": F.component_fingerprint(out.bsv.component_coord, highlight=[3, 15]),
        "basis_c3": F.basis_spectrum(*b.basis_spectrum(3), bands=b.motif_by_id("purine_ring_breathing").bands_cm),
        "collisions": F.band_collision_map(b.mss.motifs),
        "dose": F.dose_response(np.asarray(meta["conc_uM"], float),
                                [b.infer(Z[i]).bsv.composition["nucleic_purine"] for i in range(len(Z))],
                                xlabel="adenine (uM)", ylabel="purine share"),
    }
    for name, fig in figs.items():
        p = OUT / f"{name}.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        print(f"  rendered {p.relative_to(HERE)}")

    # every page module imports and exposes render()
    from demo_core.pages import (p1_overview, p2_reference_atlas, p3_reasoning, p4_calibration,
                                  p5_serum, p6_biological, p7_dart, p8_methods)
    for m in (p1_overview, p2_reference_atlas, p3_reasoning, p4_calibration,
              p5_serum, p6_biological, p7_dart, p8_methods):
        assert hasattr(m, "render"), m.__name__
    print(f"all 8 pages import · figures in {OUT.relative_to(HERE)} · atlas untouched")


if __name__ == "__main__":
    main()
