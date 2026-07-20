"""Bridge between the demo and the FROZEN V6 engine + MSS layer.

The demo never re-implements any science. It loads the real
``gaira.engine.GAIRAEngine`` and ``gaira.engine.mss.MSSLayer`` once and drives them.
Atlas / ontology / registry / theme-weights / BSV / MSS are all frozen; the demo is
presentation only. Every number shown by the demo comes from this bridge.
"""
from __future__ import annotations
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gaira.engine import GAIRAEngine                 # noqa: E402
from gaira.engine.mss import MSSLayer                # noqa: E402
from gaira.engine.versioning import VERSIONS         # noqa: E402

K = 24


class Bridge:
    """Loaded-once frozen engine + MSS layer, with demo-friendly accessors."""

    def __init__(self):
        self.eng = GAIRAEngine()
        self.mss = MSSLayer.from_engine(self.eng)
        self.onto = self.eng.builder.onto
        self.reg = self.eng.builder.reg
        self.bio_themes = self.onto.biochemical_theme_ids
        self.all_themes = self.onto.theme_ids
        assert self.eng.atlas.meta["fingerprint"] == VERSIONS.atlas_fingerprint

    # ── inference ──
    def infer(self, coordinates, domain="buffer"):
        return self.eng.infer(coordinates=np.asarray(coordinates, float), domain=domain)

    def infer_spectrum(self, wavenumber, intensity, domain="buffer"):
        return self.eng.infer(wavenumber=wavenumber, intensity=intensity, domain=domain)

    def bsv_and_mss(self, coordinates, domain="buffer"):
        """Full reasoning trace for one query: (inference, [MSSActivation])."""
        out = self.infer(coordinates, domain)
        return out, self.mss.activate(out.bsv)

    # ── static reference data for explorers ──
    def component_row(self, j):
        c = self.reg.get(j)
        v = lambda k: c[k]["value"]
        return {
            "component": j, "bands": v("dominant_raman_peaks_cm"),
            "stability": v("bootstrap_stability"), "purity": v("purity"),
            "interpretation": v("current_interpretation"),
            "loadings": v("reference_analyte_loadings"),
            "families": v("top_biochemical_families"),
            "audit_label": v("audit_label_v0_1"),
            "n_dose_responsive": v("n_dose_experiments_responsive"),
        }

    def basis_spectrum(self, j):
        W, grid = self.reg.basis()
        return grid, W[j]

    def theme_meta(self, theme_id):
        t = self.onto.theme(theme_id)
        return {"id": theme_id, "name": t.get("name", theme_id),
                "definition": t.get("description", ""),
                "bands": t.get("characteristic_bands_cm", []),
                "caveats": t.get("domain_caveats", ""), "ambiguities": t.get("ambiguities", ""),
                "literature": t.get("literature", ""),
                "reference_families": t.get("reference_families", [])}

    def theme_contributors(self, theme_id, top=6):
        return self.onto.contributors(theme_id, top=top)

    def motif_by_id(self, mid):
        return next(m for m in self.mss.motifs if m.id == mid)

    # ── platform stats (Overview) ──
    def platform_stats(self):
        cc = self.eng.atlas.meta.get("corpus_card", {})
        return {
            "n_reference_spectra": cc.get("n_spectra"),
            "n_reference_analytes": cc.get("n_analytes"),
            "n_components": K,
            "n_themes": len(self.all_themes),
            "n_biochemical_themes": len(self.bio_themes),
            "n_mss_motifs": len(self.mss.motifs),
            "n_biochemical_mss": len(self.mss.biochemical()),
            "grid": (self.eng.atlas.meta["grid_min"], self.eng.atlas.meta["grid_max"],
                     self.eng.atlas.meta["n_bins"]),
            "sources": cc.get("sources", {}),
            "excitations": cc.get("excitations", {}),
            "explained_variance": self.eng.atlas.meta.get("stats", {}).get("explained_variance"),
            "excitation_transfer": self.eng.atlas.meta.get("validation", {})
                .get("excitation_transfer", {}),
            "fingerprint": VERSIONS.atlas_fingerprint,
            "versions": VERSIONS.as_dict(),
        }


@lru_cache(maxsize=1)
def get_bridge():
    """Cached singleton (also memoised by Streamlit's cache in app.py)."""
    return Bridge()
