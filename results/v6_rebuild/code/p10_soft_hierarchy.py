"""GAIRA V6.2 — Parts 1, 2, 3, 6, 10.

Soft theme membership, theme uncertainty, shared biochemical motifs, learned vs
manual weights, and the three-resolution multi-scale hierarchy.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "results/v6_rebuild/code"))
from v62 import core as C

OUT = REPO / "results/v6_rebuild"

# Level names are assigned from the chemistry of each derived group, never hard-coded
# as the grouping itself.
NAME_HINTS = {
    "nucleobase_purine": "Purine", "nucleobase_pyrimidine": "Pyrimidine",
    "phosphate_ester": "Phosphate ester", "aromatic_sidechain": "Aromatic residue",
    "polypeptide": "Protein backbone", "free_amino_acid": "Free amino acid",
    "fatty_acid": "Fatty acyl", "acylglycerol": "Acylglycerol", "sterol": "Sterol",
    "monosaccharide": "Monosaccharide", "polysaccharide": "Polysaccharide",
    "organic_acid": "Organic acid", "sulfur_metabolite": "Sulfur metabolite",
    "tetrapyrrole": "Porphyrin", "redox_cofactor": "Flavin", "polyene": "Carotenoid",
}
SUPER_NAME = {"nucleic": "Nucleic chemistry", "protein": "Protein chemistry",
              "lipid": "Lipid chemistry", "carbohydrate": "Carbohydrate chemistry",
              "metabolite": "Energy metabolism", "cofactor": "Redox / pigments",
              "BRIDGING": "Phosphate ester"}


def name_group(members, class_of):
    if len(members) == 1:
        return NAME_HINTS.get(class_of[members[0]], members[0])
    sc = {C.SUPERCLASS.get(class_of[m], "?") for m in members}
    sc.discard("BRIDGING")
    if len(sc) == 1:
        return SUPER_NAME.get(next(iter(sc)), next(iter(sc)).title())
    return " + ".join(sorted(SUPER_NAME.get(s, s).split()[0] for s in sc))


def hybrid_distance(ctx):
    """Same three-way distance as V6: co-activation, spectral shape, chemistry."""
    A, M, H = ctx.A, ctx.M, ctx.H
    Ca = np.nan_to_num(np.corrcoef(A.T)); Da = 1 - Ca
    Sp = M.T @ H; Sp = Sp / (np.linalg.norm(Sp, axis=1, keepdims=True) + C.EPS)
    Ds = 1 - Sp @ Sp.T
    n = len(ctx.motif_ids)
    Do = np.zeros((n, n))
    hits = [{i for i, a in enumerate(ctx.analytes)
             if any(_nm(e, a) for e in m.exemplars)} for m in ctx.motifs]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            same = ctx.class_of[ctx.motif_ids[i]] == ctx.class_of[ctx.motif_ids[j]]
            jac = len(hits[i] & hits[j]) / (len(hits[i] | hits[j]) + C.EPS)
            Do[i, j] = 1 - (0.5 * same + 0.5 * jac)
    for D in (Da, Ds, Do):
        np.fill_diagonal(D, 0.0)
    Dh = (Da / (Da.max() + C.EPS) + Ds / (Ds.max() + C.EPS) + Do / (Do.max() + C.EPS)) / 3
    return np.clip((Dh + Dh.T) / 2, 0, 2), Da, Ds, Do


def _nm(e, a):
    from v6_semantic.mss_v6 import name_matches
    return name_matches(e, a)


def partition(D, K, ids):
    if K >= len(ids):
        return [[i] for i in ids]
    Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, K, criterion="maxclust")
    g = {}
    for i, l in enumerate(lab):
        g.setdefault(int(l), []).append(ids[i])
    return [sorted(v) for v in g.values()]


def main():
    ctx = C.load_context()
    ids, class_of = ctx.motif_ids, ctx.class_of
    print(f"V6.2 · frozen atlas {C.CANON} · {len(ids)} biochemical MSS motifs "
          f"· {len(ctx.analytes)} analytes")

    Dh, Da, Ds, Do = hybrid_distance(ctx)

    # ── PART 10 · derive three semantic resolutions ──
    # L1 = the motifs themselves. L2 = derived from the hybrid distance, constrained to be
    # chemically admissible. L3 = derived by agglomerating the L2 THEME CENTROIDS, so the
    # coarse level is a genuine abstraction of the medium one rather than a re-clustering.
    levels = {}
    levels["L1_fine"] = {"K": len(ids), "groups": [[m] for m in ids],
                         "names": [name_group([m], class_of) for m in ids],
                         "admissible": True, "source": "the MSS motifs themselves"}
    g8 = partition(Dh, 8, ids)
    if C.admissible(g8, class_of):
        src2 = "derived by average-linkage on the hybrid distance at K=8"
        g2 = g8
    else:
        byc = {}
        for m in ids:
            sc = C.SUPERCLASS.get(class_of[m], "?")
            byc.setdefault(sc if sc != "BRIDGING" else "carbohydrate", []).append(m)
        g2 = [sorted(v) for v in byc.values()]
        src2 = ("K=8 hybrid clustering was chemically inadmissible; fell back to the "
                "chemical-superclass grouping")
    levels["L2_medium"] = {"K": len(g2), "groups": g2,
                           "names": [name_group(x, class_of) for x in g2],
                           "admissible": C.admissible(g2, class_of), "source": src2}
    # L3: agglomerate the L2 centroids in motif-profile space down to 4 systems
    Pm = C.motif_profiles(ctx.A)
    idxm = {m: i for i, m in enumerate(ids)}
    Cc = np.array([Pm[[idxm[m] for m in g]].mean(0) for g in g2])
    Cc = Cc / (np.linalg.norm(Cc, axis=1, keepdims=True) + C.EPS)
    D2 = np.clip(1 - Cc @ Cc.T, 0, 2); np.fill_diagonal(D2, 0.0)
    lab3 = fcluster(linkage(squareform((D2 + D2.T) / 2, checks=False), method="average"),
                    4, criterion="maxclust")
    merged = {}
    for i, l in enumerate(lab3):
        merged.setdefault(int(l), []).extend(g2[i])
    g3 = [sorted(v) for v in merged.values()]
    levels["L3_coarse"] = {"K": len(g3), "groups": g3,
                           "names": [name_group(x, class_of) for x in g3],
                           "admissible": C.admissible(g3, class_of),
                           "source": "agglomerated from the L2 theme centroids to K=4"}
    for name in ("L1_fine", "L2_medium", "L3_coarse"):
        g, src = levels[name]["groups"], levels[name]["source"]
        print(f"  {name}: K={len(g)}  admissible={levels[name]['admissible']}  ({src})")
        for nm, mem in zip(levels[name]["names"], g):
            print(f"      {nm:<26} <- {', '.join(mem)}")

    # ── PART 1 · soft membership at every level ──
    # The membership must be SPARSE. Criterion fixed in advance: pick the largest
    # (softest) temperature on a fixed grid for which the mean number of themes carrying
    # a motif is <= 2.5 — "a motif belongs to one, sometimes two chemistries". The full
    # sweep is reported so the choice is auditable.
    tsweep = []
    for tau in (0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20):
        St, _ = C.soft_membership(ctx.A, levels["L2_medium"]["groups"], ids, temperature=tau)
        tsweep.append({"temperature": tau,
                       "mean_support": round(float((St > 0).sum(1).mean()), 3),
                       "mean_entropy": round(float(C.norm_entropy(St, axis=1).mean()), 4),
                       "mean_dominant_weight": round(float(St.max(1).mean()), 4)})
    pd.DataFrame(tsweep).to_csv(C.tab("v62_temperature_sweep.csv"), index=False)
    ok = [r for r in tsweep if r["mean_support"] <= 2.5]
    TAU = max(r["temperature"] for r in ok) if ok else min(r["temperature"] for r in tsweep)
    print(f"\n  soft-membership temperature selected: {TAU} "
          f"(mean support {[r for r in tsweep if r['temperature']==TAU][0]['mean_support']} themes/motif)")

    soft, tables = {}, []
    for lv, d in levels.items():
        S, sim = C.soft_membership(ctx.A, d["groups"], ids, temperature=TAU)
        soft[lv] = S
        H = C.norm_entropy(S, axis=1)
        for i, m in enumerate(ids):
            order = np.argsort(-S[i])
            tables.append({
                "level": lv, "motif": m, "chemical_class": class_of[m],
                "dominant_theme": d["names"][order[0]],
                "dominant_weight": round(float(S[i, order[0]]), 4),
                "runner_up_theme": d["names"][order[1]] if d["K"] > 1 else "-",
                "runner_up_weight": round(float(S[i, order[1]]), 4) if d["K"] > 1 else 0.0,
                "entropy": round(float(H[i]), 4),
                "n_themes_above_floor": int((S[i] > 0).sum()),
                "weights": json.dumps({d["names"][t]: round(float(S[i, t]), 4)
                                       for t in order if S[i, t] > 0}),
            })
    memb = pd.DataFrame(tables)
    memb.to_csv(C.tab("v62_theme_membership.csv"), index=False)

    # theme_membership.yaml (the requested artifact)
    ymap = {}
    for lv, d in levels.items():
        ymap[lv] = {
            "K": d["K"], "source": d["source"], "chemically_admissible": bool(d["admissible"]),
            "themes": [{"name": n, "seed_motifs": g} for n, g in zip(d["names"], d["groups"])],
            "motifs": {},
        }
        S = soft[lv]
        for i, m in enumerate(ids):
            order = [t for t in np.argsort(-S[i]) if S[i, t] > 0]
            ymap[lv]["motifs"][m] = {
                "chemical_class": class_of[m],
                "theme_weights": {d["names"][t]: round(float(S[i, t]), 4) for t in order},
                "entropy": round(float(C.norm_entropy(S[i])), 4),
                "dominant_theme": d["names"][order[0]],
                "runner_up_theme": d["names"][order[1]] if len(order) > 1 else None,
            }
    C.dump_yaml({"schema": "v62_theme_membership", "atlas_fingerprint": C.CANON,
                 "temperature": TAU, "floor": C.SOFT_FLOOR,
                 "note": "Soft, non-negative, row-stochastic motif->theme membership. A motif "
                         "may belong to several themes; shared biochemical motifs are "
                         "represented, not suppressed.",
                 "levels": ymap}, "theme_membership.yaml")

    # ── PART 3 · shared biochemical motifs ──
    L2 = levels["L2_medium"]; S2 = soft["L2_medium"]
    shared = []
    for i, m in enumerate(ids):
        order = np.argsort(-S2[i])
        if S2[i, order[0]] < 0.85 and (S2[i].sum() > 0):
            # which components drive the second theme?
            comps = sorted(ctx.motifs[i].contributors, key=lambda c: -c["weight"])[:3]
            # spectral reason: bands shared with the runner-up theme's seed motifs
            other = L2["groups"][order[1]]
            mine = set(np.round(ctx.motifs[i].bands_cm, 0).astype(int))
            theirs = set()
            for om in other:
                theirs |= set(np.round(ctx.motifs[ids.index(om)].bands_cm, 0).astype(int))
            close = sorted({b for b in mine for t in theirs if abs(b - t) <= 25})
            shared.append({
                "motif": m, "dominant_theme": L2["names"][order[0]],
                "dominant_weight": round(float(S2[i, order[0]]), 4),
                "shared_with": L2["names"][order[1]],
                "shared_weight": round(float(S2[i, order[1]]), 4),
                "entropy": round(float(C.norm_entropy(S2[i])), 4),
                "n_themes": int((S2[i] > 0).sum()),
                "components_responsible": ", ".join(f"c{c['component']}({c['weight']:.2f})"
                                                    for c in comps),
                "overlapping_bands_cm": ", ".join(str(b) for b in close[:8]) or "—",
                "confidence": round(float(np.mean([c["weight"] for c in comps])), 4),
            })
    sh = pd.DataFrame(shared).sort_values("shared_weight", ascending=False)
    sh.to_csv(C.tab("v62_shared_motifs.csv"), index=False)

    # ── PART 6 · learn the membership, compare to the derived one ──
    # target: the analyte's dominant L2 theme, one-hot from its own motif profile
    hard = np.zeros((len(ctx.analytes), L2["K"]))
    dom = (ctx.A @ S2).argmax(1)
    hard[np.arange(len(ctx.analytes)), dom] = 1.0
    S_learn = C.learn_membership(ctx.A, hard)
    cmp_rows = []
    for i, m in enumerate(ids):
        a, b = S2[i], S_learn[i]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + C.EPS))
        cmp_rows.append({
            "motif": m,
            "derived_dominant": L2["names"][int(np.argmax(a))],
            "learned_dominant": L2["names"][int(np.argmax(b))],
            "agree": bool(np.argmax(a) == np.argmax(b)),
            "cosine": round(cos, 4),
            "derived_entropy": round(float(C.norm_entropy(a)), 4),
            "learned_entropy": round(float(C.norm_entropy(b)), 4),
            "max_abs_diff": round(float(np.abs(a - b).max()), 4),
        })
    cmpdf = pd.DataFrame(cmp_rows)
    cmpdf.to_csv(C.tab("v62_learned_vs_derived.csv"), index=False)

    # ── PART 2 · per-analyte theme uncertainty ──
    post = C.theme_posterior(ctx.A, S2)
    urows = []
    for i, a in enumerate(ctx.analytes):
        p = post["posterior"][i]
        o = np.argsort(-p)
        urows.append({
            "analyte": a, "family": ctx.families[i],
            "theme_1": L2["names"][o[0]], "p_1": round(float(p[o[0]]), 4),
            "theme_2": L2["names"][o[1]], "p_2": round(float(p[o[1]]), 4),
            "theme_3": L2["names"][o[2]], "p_3": round(float(p[o[2]]), 4),
            "entropy": round(float(post["entropy"][i]), 4),
            "margin": round(float(post["margin"][i]), 4),
            "confidence": round(float(post["confidence"][i]), 4),
        })
    unc = pd.DataFrame(urows)
    unc.to_csv(C.tab("v62_theme_uncertainty.csv"), index=False)

    summary = {
        "atlas_fingerprint": C.CANON,
        "philosophy": "maximum biochemical abstraction subject to minimum information loss; "
                      "analyte accuracy is a secondary metric",
        "soft_parameters": {"temperature": TAU, "floor": C.SOFT_FLOOR,
                            "selection_rule": "largest temperature with mean support <= 2.5 themes/motif",
                            "sweep": tsweep},
        "levels": {k: {"K": v["K"], "names": v["names"], "admissible": bool(v["admissible"]),
                       "source": v["source"], "groups": v["groups"]} for k, v in levels.items()},
        "membership": {
            "mean_entropy_L2": round(float(C.norm_entropy(S2, axis=1).mean()), 4),
            "n_motifs_multi_theme_L2": int((( S2 > 0).sum(1) > 1).sum()),
            "n_motifs_single_theme_L2": int(((S2 > 0).sum(1) == 1).sum()),
            "mean_dominant_weight_L2": round(float(S2.max(1).mean()), 4),
        },
        "shared_motifs": {
            "n_detected": int(len(sh)),
            "criterion": "dominant theme weight < 0.85 at L2",
            "top": sh.head(6).to_dict("records"),
        },
        "learned_vs_derived": {
            "mean_cosine": round(float(cmpdf.cosine.mean()), 4),
            "dominant_theme_agreement": round(float(cmpdf.agree.mean()), 4),
            "mean_entropy_derived": round(float(cmpdf.derived_entropy.mean()), 4),
            "mean_entropy_learned": round(float(cmpdf.learned_entropy.mean()), 4),
        },
        "uncertainty": {
            "mean_entropy": round(float(unc.entropy.mean()), 4),
            "mean_margin": round(float(unc.margin.mean()), 4),
            "mean_confidence": round(float(unc.confidence.mean()), 4),
            "n_low_confidence": int((unc.confidence < 0.2).sum()),
        },
    }
    C.dump_json(summary, "v62_soft_hierarchy.json")
    np.savez(C.art("v62_membership.npz"),
             S_L1=soft["L1_fine"], S_L2=S2, S_L3=soft["L3_coarse"], S_learned=S_learn,
             A=ctx.A, zA=ctx.zA, motif_ids=np.array(ids),
             L2_names=np.array(L2["names"]), L3_names=np.array(levels["L3_coarse"]["names"]),
             analytes=np.array(ctx.analytes), families=ctx.families,
             D_hybrid=Dh, D_act=Da, D_spec=Ds, D_onto=Do,
             grid=ctx.grid, corpusX=ctx.corpusX, M=ctx.M, H=ctx.H)

    pd.set_option("display.width", 250)
    print("\nPART 1/2 — soft membership at L2 (K=%d)" % L2["K"])
    print(memb[memb.level == "L2_medium"][
        ["motif", "dominant_theme", "dominant_weight", "runner_up_theme",
         "runner_up_weight", "entropy", "n_themes_above_floor"]].to_string(index=False))
    print(f"\nPART 3 — shared biochemical motifs: {len(sh)} of {len(ids)}")
    print(sh[["motif", "dominant_theme", "dominant_weight", "shared_with", "shared_weight",
              "overlapping_bands_cm"]].head(10).to_string(index=False))
    print(f"\nPART 6 — learned vs derived: mean cosine "
          f"{cmpdf.cosine.mean():.3f}, dominant agreement {cmpdf.agree.mean():.1%}")
    print(f"\nPART 2 — analyte uncertainty: mean entropy {unc.entropy.mean():.3f}, "
          f"mean margin {unc.margin.mean():.3f}, mean confidence {unc.confidence.mean():.3f}")


if __name__ == "__main__":
    main()
