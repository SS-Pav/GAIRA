"""Pure Ag-SERS validation — project the Gobbato pure-analyte Ag-SERS spectra through the
FROZEN Raman atlas (no retraining, no modality correction) and quantify how well the
Raman-trained biochemical representation recognises them, per analyte.

The scientific bridge between "what the atlas learns" (pure Raman) and matrix-perturbed
serum: can the atlas interpret pure Ag-SERS analytes BEFORE serum competition?

Deterministic. Writes:
  tables/pure_ag_sers_validation.json      (summary + per-analyte)
  tables/pure_ag_sers_per_analyte.csv      (flat per-analyte table for the demo)
Nothing frozen is modified.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v5_rebuild/spike_validation/code"))
from gaira.engine import GAIRAEngine
from gaira.engine.mss import MSSLayer
from gaira.foundation import dataset as DS
from gaira.foundation.families_raman import family_of
from gaira.data.synonyms import canonical
import spike_lib as SL

OUT = REPO / "results/v5_rebuild/foundation_audit/tables"

# recoverability tiers by Raman↔SERS coordinate cosine (does the SERS representation land
# where the Raman one does, in the frozen 24-component space?)
TIERS = [("Excellent", 0.80), ("Good", 0.65), ("Moderate", 0.45), ("Weak", 0.25), ("Poor", 0.0)]


def tier(cos):
    for name, thr in TIERS:
        if cos >= thr:
            return name
    return "Poor"


def main():
    eng = GAIRAEngine()
    mss = MSSLayer.from_engine(eng)
    atlas = eng.atlas
    THEMES = eng.builder.onto.biochemical_theme_ids

    def coords(V):
        return atlas.coordinates(np.atleast_2d(np.nan_to_num(V)))

    def dominant_theme(coord, domain="buffer"):
        b = eng.infer(coordinates=np.asarray(coord, float), domain=domain).bsv
        tv = {t: b.composition[t] for t in THEMES}
        return max(tv, key=tv.get), b

    # ── Raman reference: per-analyte mean coords over the FULL frozen corpus (167 analytes) ──
    corpus = DS.load_reference_corpus()
    Zr = coords(corpus.X)
    raman_by = {}
    for a in pd.unique(corpus.meta.analyte):
        raman_by[a] = Zr[corpus.meta.analyte.values == a].mean(0)
    raman_names = list(raman_by)
    R = np.array([raman_by[a] for a in raman_names])
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)

    # ── Pure Ag-SERS: per-analyte mean coords ──
    Xs, rs = SL.load_pure_sers()
    Zs = coords(Xs)
    ood_s = [float(eng.infer(coordinates=z, domain="buffer").bsv.ood_score) for z in Zs]
    rs = rs.assign(_ood=ood_s)
    sers_by, ood_by = {}, {}
    for a in pd.unique(rs.analyte):
        m = rs.analyte.values == a
        sers_by[canonical(a)] = Zs[m].mean(0)
        ood_by[canonical(a)] = float(np.mean(np.array(ood_s)[m]))

    matched = sorted(set(raman_by) & set(sers_by))
    rows = []
    for a in matched:
        r_, s_ = raman_by[a], sers_by[a]
        cc = float(np.dot(r_, s_) / (np.linalg.norm(r_) * np.linalg.norm(s_) + 1e-12))
        b_s = eng.infer(coordinates=s_, domain="buffer").bsv
        rt, _ = dominant_theme(r_); st_, _ = dominant_theme(s_)
        # nearest Raman references to the SERS projection (full atlas)
        sn = s_ / (np.linalg.norm(s_) + 1e-12)
        sims = Rn @ sn
        order = np.argsort(-sims)
        nearest = [(raman_names[i], round(float(sims[i]), 3)) for i in order[:4]]
        self_rank = int(np.where(np.array(raman_names)[order] == a)[0][0]) + 1
        # matched vs mismatched: cosine to own Raman vs mean to other matched Ramans
        others = [np.dot(sn, raman_by[o] / (np.linalg.norm(raman_by[o]) + 1e-12))
                  for o in matched if o != a]
        mm = {a2.id: a2.composition for a2 in mss.activate(b_s)}
        top_mss = sorted([(k, round(float(v), 4)) for k, v in mm.items()
                          if k != "colloid_matrix_background"], key=lambda x: -x[1])[:3]
        rows.append({
            "analyte": a, "family": family_of(a), "coord_cosine": round(cc, 4),
            "recoverability_tier": tier(cc), "theme_preserved": rt == st_,
            "raman_theme": rt, "sers_theme": st_,
            "sers_ood": round(ood_by[a], 4),
            "sers_confidence": round(float(b_s.overall_confidence), 4),
            "self_is_nearest": nearest[0][0] == a, "self_rank": self_rank,
            "nearest_raman": nearest,
            "matched_cosine": round(cc, 3),
            "mean_mismatched_cosine": round(float(np.mean(others)), 3),
            "separation": round(cc - float(np.mean(others)), 3),
            "top_mss": top_mss,
            "sers_coord": [round(float(x), 5) for x in s_],
            "raman_coord": [round(float(x), 5) for x in r_],
        })
    df = pd.DataFrame(rows).sort_values("coord_cosine", ascending=False).reset_index(drop=True)

    # ── aggregate scientific summary ──
    def fam_class(f):
        if f in ("purine", "pyrimidine", "nucleic_acid", "nucleoside"): return "nucleobase/purine"
        if f in ("cofactor",) or f in ("amino_acid",) and False: return f
        return f
    df["family_class"] = df.family.map(lambda f: f)
    tier_counts = df.recoverability_tier.value_counts().to_dict()
    by_family = (df.groupby("family").coord_cosine.agg(["mean", "count"])
                 .sort_values("mean", ascending=False).round(3))
    cos = df.coord_cosine.values
    summary = {
        "dataset": "Gobbato pure Ag-SERS metabolites", "instrument": "B&WTek i-Raman Plus",
        "substrate": "Ag colloid", "laser_nm": 785,
        "n_sers_spectra": int(len(Xs)), "n_sers_analytes": int(rs.analyte.nunique()),
        "n_matched_to_raman": len(matched),
        "median_coord_cosine": round(float(np.median(cos)), 4),
        "mean_coord_cosine": round(float(np.mean(cos)), 4),
        "n_theme_preserved": int(df.theme_preserved.sum()),
        "n_self_nearest": int(df.self_is_nearest.sum()),
        "mean_sers_ood": round(float(df.sers_ood.mean()), 4),
        "mean_matched_cosine": round(float(df.matched_cosine.mean()), 3),
        "mean_mismatched_cosine": round(float(df.mean_mismatched_cosine.mean()), 3),
        "mean_separation": round(float(df.separation.mean()), 3),
        "tier_counts": tier_counts,
        "family_coord_cosine": {k: {"mean": float(v["mean"]), "n": int(v["count"])}
                                for k, v in by_family.to_dict("index").items()},
        "tiers_definition": {name: f">= {thr}" for name, thr in TIERS},
    }
    (OUT / "pure_ag_sers_validation.json").write_text(json.dumps(
        {"summary": summary, "per_analyte": rows}, indent=2))
    df.drop(columns=["sers_coord", "raman_coord", "nearest_raman", "top_mss", "family_class"]).to_csv(
        OUT / "pure_ag_sers_per_analyte.csv", index=False)

    print(json.dumps(summary, indent=2))
    print("\nTop 6 (best transfer):")
    print(df.head(6)[["analyte", "family", "coord_cosine", "recoverability_tier",
                      "theme_preserved"]].to_string(index=False))
    print("\nBottom 6 (worst transfer):")
    print(df.tail(6)[["analyte", "family", "coord_cosine", "recoverability_tier",
                      "theme_preserved"]].to_string(index=False))


if __name__ == "__main__":
    main()
