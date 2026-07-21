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

    def reconstruct(self, coord):
        """The frozen atlas's reconstruction of a spectrum from its 24 coordinates
        (coord @ basis). This is exactly what the engine 'sees' — derived from the
        frozen NMF basis, always available, no raw-file dependency."""
        W, grid = self.reg.basis()
        return grid, np.asarray(coord, float) @ W

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

    def component_linked_motifs(self, j):
        """MSS motifs that component j contributes to, with the motif weight."""
        out = []
        for m in self.mss.motifs:
            for c in m.contributors:
                if c["component"] == j:
                    out.append({"id": m.id, "name": m.name, "weight": c["weight"],
                                "parent_theme": m.parent_theme})
        return sorted(out, key=lambda x: -x["weight"])

    def component_theme_weights(self, j, top=6):
        """component→theme weights (many-to-many) with the three evidence lines."""
        W = self.onto.W
        rows = []
        for ti, t in enumerate(self.all_themes):
            w = float(W[j, ti])
            if w > 0.02:
                ev = self.onto.weight_evidence(j, t) or {}
                rows.append({"theme": t, "weight": w, "evidence": ev.get("evidence", {})})
        return sorted(rows, key=lambda r: -r["weight"])[:top]

    def component_distance(self):
        """Deterministic 24×24 cosine DISTANCE between the frozen NMF basis spectra —
        the learned representation itself. Symmetric, zero diagonal."""
        W, _ = self.reg.basis()
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        D = 1.0 - Wn @ Wn.T
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        return np.clip(D, 0, None)

    def component_dominant_theme(self, j):
        """The biochemical theme with the largest component→theme weight for c_j."""
        ws = [(t, float(self.onto.W[j, self.onto.theme_index(t)])) for t in self.bio_themes]
        return max(ws, key=lambda x: x[1])[0]

    def component_top_motif(self, j):
        lm = self.component_linked_motifs(j)
        return lm[0]["name"] if lm else "—"

    def reference_map(self):
        """167 reference analytes as frozen L1 coordinates + biochemical family.
        Used by the Page-2 reference PCA (explanatory only, not the inference model)."""
        import pandas as pd
        df = pd.read_csv(REPO / "results/v5_rebuild/foundation/tables/c3_analyte_activation_matrix.csv")
        cols = [str(j) for j in range(K)]
        A = df[cols].values.astype(float)
        A = A / (A.sum(1, keepdims=True) + 1e-12)          # L1 shares (engine convention)
        # family per analyte from the frozen registry loadings (deterministic)
        fam = {}
        for j in range(K):
            for l in self.reg.value(j, "reference_analyte_loadings"):
                fam.setdefault(l["analyte"].strip().lower(), l["family"])
        families = [fam.get(a.strip().lower(), "unassigned") for a in df["analyte"]]
        return {"analytes": list(df["analyte"]), "coords": A, "families": families}

    def sankey_links(self):
        """(components → MSS motifs → biochemical themes) flow for the atlas Sankey.
        Many-to-many by construction (never one-to-one)."""
        motifs = self.mss.biochemical()
        comp_nodes = [f"c{j}" for j in range(K)]
        motif_nodes = [m.name for m in motifs]
        theme_nodes = [t for t in self.bio_themes]
        idx = {**{n: i for i, n in enumerate(comp_nodes)},
               **{n: K + i for i, n in enumerate(motif_nodes)},
               **{n: K + len(motif_nodes) + i for i, n in enumerate(theme_nodes)}}
        links = []
        for mi, m in enumerate(motifs):
            for c in m.contributors:
                links.append((idx[f"c{c['component']}"], idx[m.name], c["weight"]))
            links.append((idx[m.name], idx[m.parent_theme], max(m.confidence, 0.05)))
        return {"comp_nodes": comp_nodes, "motif_nodes": motif_nodes,
                "theme_nodes": theme_nodes, "links": links}

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
