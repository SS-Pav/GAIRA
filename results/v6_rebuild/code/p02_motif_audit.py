"""GAIRA V6 — Part 2. Audit every MSS motif; decide the V6 motif set on evidence.

For each motif: spectral justification, supporting components, supporting analytes,
spectral purity, corpus coverage, discriminative power and redundancy against every
other motif. Then identify chemistry the corpus contains but no motif describes.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v6_rebuild/code"))

from gaira.engine import GAIRAEngine
from gaira.foundation import dataset as DS
from gaira.foundation.families_raman import family_of
from v6_semantic.mss_v6 import MSSLayerV6, name_matches, motif_profile

OUT = REPO / "results/v6_rebuild"
CANON = "09ed804a40836f4a05a91ba10900cded"
FROZEN_MOTIFS = REPO / "assets/foundation/mss_motifs_v1.yaml"


def auc(pos, neg):
    """Mann-Whitney AUC: P(a random positive scores above a random negative)."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    eng = GAIRAEngine()
    assert eng.atlas.meta["fingerprint"] == CANON
    H, grid = eng.atlas.components, eng.atlas.grid
    v6 = MSSLayerV6(FROZEN_MOTIFS, eng.builder.reg, H, grid)

    corpus = DS.load_reference_corpus()
    Z = eng.atlas.coordinates(corpus.X)
    ra = corpus.meta.analyte.values
    analytes = sorted(set(ra))
    zA = np.array([Z[ra == a].mean(0) for a in analytes])
    A = np.array([v6.activate(z) for z in zA])          # (n_analytes, n_motifs)
    fams = np.array([family_of(a) for a in analytes])

    # ── per-motif audit ──
    rows = []
    for mi, m in enumerate(v6.motifs):
        # corpus analytes matching this motif's exemplars
        hit = np.array([any(name_matches(e, a) for e in m.exemplars) for a in analytes])
        # discriminative power: does the motif activate more on its own chemistry?
        a_auc = auc(A[hit, mi], A[~hit, mi])
        # family-level coverage: which families does it claim?
        fam_hit = pd.Series(fams[hit]).value_counts().to_dict() if hit.any() else {}
        # top corpus analytes by activation (regardless of exemplar list)
        top = [analytes[i] for i in np.argsort(-A[:, mi])[:6]]
        top_fams = pd.Series([family_of(t) for t in top]).value_counts()
        prof = motif_profile(m.bands_cm, grid)
        s6 = v6.M[:, mi] @ H
        band_fid = float(np.dot(s6, prof) / (np.linalg.norm(s6) * np.linalg.norm(prof) + 1e-12))
        rows.append({
            "motif": m.id, "chemical_class": m.chemical_class or "-",
            "n_bands": len(m.bands_cm), "bands_cm": m.bands_cm,
            "n_exemplars": len(m.exemplars), "exemplars": ", ".join(m.exemplars),
            "n_components": len(m.contributors),
            "components": [c["component"] for c in m.contributors],
            "top_component": m.contributors[0]["component"] if m.contributors else None,
            "top_component_weight": m.contributors[0]["weight"] if m.contributors else 0.0,
            "spectral_purity": round(m.spectral_purity, 4),
            "band_fidelity": round(band_fid, 4),
            "stability": round(m.stability, 4),
            "evidence_breadth": round(m.evidence_breadth, 4),
            "confidence": round(m.confidence, 4),
            "corpus_coverage_n": int(hit.sum()),
            "corpus_coverage_pct": round(100 * hit.mean(), 1),
            "discriminative_auc": round(a_auc, 4) if a_auc == a_auc else None,
            "claimed_families": ", ".join(f"{k}({v})" for k, v in sorted(fam_hit.items(), key=lambda x: -x[1])[:4]),
            "top_activating_analytes": ", ".join(top[:4]),
            "top_activating_family": top_fams.index[0] if len(top_fams) else "-",
            "mean_activation": round(float(A[:, mi].mean()), 5),
            "non_biochemical": m.non_biochemical,
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "tables/p2_motif_audit.csv", index=False)

    # ── redundancy: motif-motif activation correlation + component-support overlap ──
    C = np.corrcoef(A.T)
    Mn = v6.M / (np.linalg.norm(v6.M, axis=0, keepdims=True) + 1e-12)
    S = Mn.T @ Mn
    ids = v6.motif_ids
    red = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            red.append({"motif_a": ids[i], "motif_b": ids[j],
                        "activation_corr": round(float(C[i, j]), 4),
                        "component_support_cosine": round(float(S[i, j]), 4)})
    red = pd.DataFrame(red).sort_values("component_support_cosine", ascending=False)
    red.to_csv(OUT / "tables/p2_motif_redundancy.csv", index=False)

    # ── coverage gap: corpus chemistry no motif claims ──
    claimed = np.zeros(len(analytes), bool)
    for m in v6.motifs:
        claimed |= np.array([any(name_matches(e, a) for e in m.exemplars) for a in analytes])
    gap = pd.DataFrame({"analyte": np.array(analytes)[~claimed],
                        "family": fams[~claimed]})
    gap_fam = gap.family.value_counts()
    gap.to_csv(OUT / "tables/p2_uncovered_analytes.csv", index=False)

    # family-level census of the whole corpus
    census = pd.DataFrame({"family": fams}).family.value_counts().rename("n_analytes").to_frame()
    census["n_uncovered"] = [int(gap_fam.get(f, 0)) for f in census.index]
    census["pct_uncovered"] = (100 * census.n_uncovered / census.n_analytes).round(1)
    census.to_csv(OUT / "tables/p2_family_census.csv")

    summary = {
        "atlas_fingerprint": CANON,
        "n_motifs_audited": len(v6.motifs),
        "n_corpus_analytes": len(analytes),
        "exemplar_coverage_pct": round(100 * claimed.mean(), 1),
        "n_uncovered_analytes": int((~claimed).sum()),
        "largest_uncovered_families": gap_fam.head(8).to_dict(),
        "mean_discriminative_auc": round(float(audit.discriminative_auc.dropna().mean()), 4),
        "motifs_with_auc_below_0.6": audit[audit.discriminative_auc < 0.6].motif.tolist(),
        "most_redundant_pairs": red.head(5).to_dict("records"),
        "mean_spectral_purity": round(float(audit.spectral_purity.mean()), 4),
        "mean_band_fidelity": round(float(audit.band_fidelity.mean()), 4),
        "family_census": census.to_dict("index"),
    }
    (OUT / "artifacts/p2_motif_audit.json").write_text(json.dumps(summary, indent=2, default=str))

    pd.set_option("display.width", 250)
    print(audit[["motif", "n_components", "spectral_purity", "band_fidelity", "stability",
                 "confidence", "corpus_coverage_n", "discriminative_auc",
                 "top_activating_family"]].to_string(index=False))
    print("\ncorpus exemplar coverage: %.1f%% (%d of %d analytes unclaimed)"
          % (100 * claimed.mean(), (~claimed).sum(), len(analytes)))
    print("\nlargest uncovered families:"); print(gap_fam.head(10).to_string())
    print("\nmost redundant motif pairs (component support):")
    print(red.head(6).to_string(index=False))
    print("\nfamily census (corpus chemistry vs motif coverage):")
    print(census.head(18).to_string())


if __name__ == "__main__":
    main()
