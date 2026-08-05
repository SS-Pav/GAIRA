"""GAIRA V6 — Part 0 (hierarchy audit) + Part 1 (leakage-free MSS rebuild).

Quantifies exactly how much of the V1 MSS layer is copied from the component->theme
ontology matrix, rebuilds MSS without it, and compares the two.

Read-only w.r.t. every frozen asset.
"""
from __future__ import annotations
import sys, json, hashlib, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v6_rebuild/code"))

from gaira.engine import GAIRAEngine
from gaira.engine.mss import MSSLayer as MSSLayerV1
from gaira.foundation import dataset as DS
from v6_semantic.mss_v6 import MSSLayerV6, V6_WEIGHTS, V6_WEIGHTS_NOPERT

OUT = REPO / "results/v6_rebuild"
for d in ("artifacts", "tables", "figures", "reports"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
CANON = "09ed804a40836f4a05a91ba10900cded"
FROZEN_MOTIFS = REPO / "assets/foundation/mss_motifs_v1.yaml"


def main():
    eng = GAIRAEngine()
    assert eng.atlas.meta["fingerprint"] == CANON, "FROZEN ATLAS CHANGED"
    H, grid = eng.atlas.components, eng.atlas.grid
    reg, onto = eng.builder.reg, eng.builder.onto

    v1 = MSSLayerV1.from_engine(eng)
    v6 = MSSLayerV6(FROZEN_MOTIFS, reg, H, grid, weights=V6_WEIGHTS)
    v6np = MSSLayerV6(FROZEN_MOTIFS, reg, H, grid, weights=V6_WEIGHTS_NOPERT)

    # ═════ PART 0 — quantify the leakage ═════
    leak_rows = []
    for mi, mot in enumerate(v1.motifs):
        ti = onto.theme_index(mot.parent_theme)
        for c in mot.contributors:
            j = c["component"]
            theme_term = v1.wt * float(onto.W[j, ti])
            raw = v1.wb * c["band"] + v1.we * c["exemplar"] + theme_term
            leak_rows.append({
                "motif": mot.id, "parent_theme": mot.parent_theme, "component": j,
                "weight_v1": c["weight"], "band": c["band"], "exemplar": c["exemplar"],
                "theme_W": c["theme"], "raw": round(raw, 5),
                "theme_share_of_raw": round(theme_term / (raw + 1e-12), 4),
                "band_only_evidence": bool(c["band"] > 0 and c["exemplar"] == 0),
            })
    leak = pd.DataFrame(leak_rows)
    leak.to_csv(OUT / "tables/p0_mss_theme_leakage.csv", index=False)

    # contributors that exist ONLY because of the theme term
    solely_theme = leak[(leak.band == 0) & (leak.exemplar == 0)]
    # contributors whose raw score would fall below keep_threshold without the theme term
    leak["raw_wo_theme"] = v1.wb * leak.band + v1.we * leak.exemplar
    would_drop = leak[leak.raw_wo_theme < v1.keep]

    audit = {
        "atlas_fingerprint": CANON,
        "v1_mixing_weights": {"band": v1.wb, "exemplar": v1.we, "theme": v1.wt},
        "v1_theme_source": "gaira.engine.ontology.Ontology.W[component, parent_theme]",
        "leakage_code_sites": {
            "src/gaira/engine/mss.py:190": "ti = self.onto.theme_index(parent)",
            "src/gaira/engine/mss.py:195": "theme = float(self.onto.W[j, ti])",
            "src/gaira/engine/mss.py:196": "raw = wb*band + we*exemplar + wt*theme   <-- 25% from themes",
            "src/gaira/engine/mss.py:209": "evidence_breadth includes the theme indicator",
            "src/gaira/engine/mss.py:169": "purine_motif = parent_theme == 'nucleic_purine' (perturbation gate)",
        },
        "n_contributor_edges": int(len(leak)),
        "mean_theme_share_of_raw_score": round(float(leak.theme_share_of_raw.mean()), 4),
        "median_theme_share_of_raw_score": round(float(leak.theme_share_of_raw.median()), 4),
        "max_theme_share_of_raw_score": round(float(leak.theme_share_of_raw.max()), 4),
        "n_edges_with_no_spectral_or_chemical_evidence": int(len(solely_theme)),
        "n_edges_that_would_drop_below_keep_threshold": int(len(would_drop)),
        "edges_existing_only_because_of_theme": solely_theme[["motif", "component", "weight_v1"]]
            .to_dict("records"),
    }

    # ═════ PART 1 — old vs new comparison ═════
    corpus = DS.load_reference_corpus()
    Z = eng.atlas.coordinates(corpus.X)
    ra = corpus.meta.analyte.values
    analytes = sorted(set(ra))
    zA = np.array([Z[ra == a].mean(0) for a in analytes])

    A1 = np.array([[x.composition for x in v1.activate(eng.builder.from_activation(z))]
                   for z in zA])
    # v1.activate returns motifs sorted by elevation — realign to motif id order
    ids1 = [m.id for m in v1.motifs]
    A1 = np.zeros((len(zA), len(ids1)))
    for i, z in enumerate(zA):
        acts = {x.id: x.composition for x in v1.activate(eng.builder.from_activation(z))}
        A1[i] = [acts[m] for m in ids1]
    A6 = np.array([v6.activate(z) for z in zA])
    A6np = np.array([v6np.activate(z) for z in zA])

    def norm_rows(X):
        s = X.sum(1, keepdims=True)
        return np.divide(X, s, out=np.zeros_like(X), where=s > 1e-12)

    cmp_rows = []
    for mi, mid in enumerate(ids1):
        w1, w6 = v1.M[:, mi], v6.M[:, mi]
        cos = float(np.dot(w1, w6) / (np.linalg.norm(w1) * np.linalg.norm(w6) + 1e-12))
        c1 = set(np.nonzero(w1)[0].tolist()); c6 = set(np.nonzero(w6)[0].tolist())
        m1, m6 = v1.motifs[mi], v6.motifs[mi]
        # activation rank correlation across the corpus
        from scipy.stats import spearmanr
        rho = float(spearmanr(A1[:, mi], A6[:, mi]).correlation)
        cmp_rows.append({
            "motif": mid, "component_weight_cosine": round(cos, 4),
            "components_v1": sorted(c1), "components_v6": sorted(c6),
            "components_shared": len(c1 & c6), "components_added": len(c6 - c1),
            "components_dropped": len(c1 - c6),
            "activation_spearman": round(rho, 4),
            "stability_v1": round(m1.stability, 4), "stability_v6": round(m6.stability, 4),
            "breadth_v1": round(m1.evidence_breadth, 4), "breadth_v6": round(m6.evidence_breadth, 4),
            "confidence_v1": round(m1.confidence, 4), "confidence_v6": round(m6.confidence, 4),
            "spectral_purity_v6": round(m6.spectral_purity, 4),
        })
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(OUT / "tables/p1_mss_v1_vs_v6.csv", index=False)

    # motif-level spectral fidelity: does the motif's implied spectrum match its band profile?
    from v6_semantic.mss_v6 import motif_profile
    fid = []
    for mi, mid in enumerate(ids1):
        prof = motif_profile(v6.motifs[mi].bands_cm, grid)
        s1 = v1.M[:, mi] @ H
        s6 = v6.M[:, mi] @ H
        f1 = float(np.dot(s1, prof) / (np.linalg.norm(s1) * np.linalg.norm(prof) + 1e-12))
        f6 = float(np.dot(s6, prof) / (np.linalg.norm(s6) * np.linalg.norm(prof) + 1e-12))
        fid.append({"motif": mid, "band_fidelity_v1": round(f1, 4), "band_fidelity_v6": round(f6, 4),
                    "delta": round(f6 - f1, 4)})
    fid = pd.DataFrame(fid)
    fid.to_csv(OUT / "tables/p1_motif_band_fidelity.csv", index=False)

    summary = {
        **audit,
        "part1": {
            "mean_component_weight_cosine_v1_v6": round(float(cmp.component_weight_cosine.mean()), 4),
            "mean_activation_spearman": round(float(cmp.activation_spearman.mean()), 4),
            "mean_band_fidelity_v1": round(float(fid.band_fidelity_v1.mean()), 4),
            "mean_band_fidelity_v6": round(float(fid.band_fidelity_v6.mean()), 4),
            "band_fidelity_improved_for": int((fid.delta > 0).sum()),
            "band_fidelity_n_motifs": int(len(fid)),
            "mean_stability_v1": round(float(cmp.stability_v1.mean()), 4),
            "mean_stability_v6": round(float(cmp.stability_v6.mean()), 4),
            "breadth_v1_is_constant": bool(cmp.breadth_v1.nunique() == 1),
            "breadth_v1_value": float(cmp.breadth_v1.iloc[0]),
            "breadth_v6_range": [round(float(cmp.breadth_v6.min()), 3), round(float(cmp.breadth_v6.max()), 3)],
            "perturbation_ablation_mean_cosine_v6_vs_v6nopert": round(float(np.mean([
                float(np.dot(v6.M[:, i], v6np.M[:, i]) /
                      (np.linalg.norm(v6.M[:, i]) * np.linalg.norm(v6np.M[:, i]) + 1e-12))
                for i in range(len(ids1))])), 4),
        },
    }
    (OUT / "artifacts/p0_p1_audit.json").write_text(json.dumps(summary, indent=2, default=str))

    np.savez(OUT / "artifacts/p1_mss_matrices.npz",
             M_v1=v1.M, M_v6=v6.M, M_v6_nopert=v6np.M,
             motif_ids=np.array(ids1), analytes=np.array(analytes),
             A_v1=A1, A_v6=A6, zA=zA, grid=grid)
    (OUT / "artifacts/mss_registry_v6.json").write_text(
        json.dumps(v6.registry(CANON), indent=2, default=str))

    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("edges_existing_only_because_of_theme", "leakage_code_sites")},
                     indent=2, default=str))
    print("\nper-motif comparison:")
    print(cmp[["motif", "component_weight_cosine", "activation_spearman", "components_shared",
               "components_added", "components_dropped", "confidence_v1", "confidence_v6"]]
          .to_string(index=False))
    print("\nband fidelity (does the motif's implied spectrum match its declared bands?):")
    print(fid.to_string(index=False))


if __name__ == "__main__":
    main()
