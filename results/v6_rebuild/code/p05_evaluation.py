"""GAIRA V6 — Parts 6, 7 and 8.

Runs the full V6 stack (components -> MSS -> chemical themes) over EVERY Raman
grounding analyte, produces per-analyte results, confusion, calibration and
reliability, builds the per-theme reference tables, and selects representative
analytes at five performance tiers.
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
from v6_semantic.mss_v6 import MSSLayerV6, name_matches
from v6_semantic import themes_v6 as TV

OUT = REPO / "results/v6_rebuild"
CANON = "09ed804a40836f4a05a91ba10900cded"
MOTIFS_V6 = OUT / "artifacts/mss_motifs_v6.yaml"
RNG = np.random.default_rng(0)


def main():
    eng = GAIRAEngine()
    assert eng.atlas.meta["fingerprint"] == CANON, "FROZEN ATLAS CHANGED"
    H, grid = eng.atlas.components, eng.atlas.grid
    v6 = MSSLayerV6(MOTIFS_V6, eng.builder.reg, H, grid)
    bio_idx = [i for i, m in enumerate(v6.motifs) if not m.non_biochemical]
    bio_ids = [v6.motifs[i].id for i in bio_idx]
    bio_motifs = [v6.motifs[i] for i in bio_idx]

    sel = json.loads((OUT / "artifacts/p4_theme_optimisation.json").read_text())["selected_partition"]
    groups = [t["motifs"] for t in sel["themes"]]
    L = TV.ThemeLayer(groups, bio_ids)
    K = L.K
    print(f"V6 hierarchy: 24 components -> {len(bio_ids)} MSS motifs -> {K} chemical themes")

    corpus = DS.load_reference_corpus()
    Zs = eng.atlas.coordinates(corpus.X)
    ra = corpus.meta.analyte.values
    analytes = sorted(set(ra))
    zA = np.array([Zs[ra == a].mean(0) for a in analytes])
    A = np.array([v6.activate(z) for z in zA])[:, bio_idx]
    Th = L.compose(A)
    fams = np.array([family_of(a) for a in analytes])

    hits = [{i for i, a in enumerate(analytes) if any(name_matches(e, a) for e in m.exemplars)}
            for m in bio_motifs]
    rows = []
    for i, a in enumerate(analytes):
        ms = [bio_ids[k] for k in range(len(bio_ids)) if i in hits[k]]
        prim = min(ms, key=lambda mid: len(bio_motifs[bio_ids.index(mid)].exemplars)) if ms else None
        exp_t = sorted({L.of_motif[m] for m in ms}) if ms else []
        th = Th[i]; mo = A[i]; z = zA[i]
        torder = list(np.argsort(-th)); morder = list(np.argsort(-mo))
        trank = min([torder.index(t) + 1 for t in exp_t], default=99)
        mrank = min([morder.index(bio_ids.index(m)) + 1 for m in ms], default=99)
        conf = float(th.max() / (th.sum() + 1e-12))
        # nearest reference analytes in theme space (evidence, never identification)
        Tn = Th / (np.linalg.norm(Th, axis=1, keepdims=True) + 1e-12)
        sims = Tn @ Tn[i]
        sims[i] = -np.inf
        near = [analytes[j] for j in np.argsort(-sims)[:3]]
        rows.append({
            "analyte": a, "family": fams[i], "n_spectra": int((ra == a).sum()),
            "expected_motifs": "|".join(ms), "primary_motif": prim or "",
            "expected_themes": "|".join(L.names[t] for t in exp_t),
            "top_component": int(np.argmax(z)), "top_component_share": round(float(z.max()), 4),
            "n_active_components": int((z > 1e-6).sum()),
            "predicted_motif": bio_ids[morder[0]], "motif_rank": mrank,
            "motif_top1": mrank == 1, "motif_top3": mrank <= 3,
            "predicted_theme": L.names[torder[0]], "theme_rank": trank,
            "theme_top1": trank == 1, "theme_top3": trank <= 3,
            "theme_confidence": round(conf, 4),
            "expected_theme_share": round(float(max([th[t] for t in exp_t], default=0.0)), 4),
            "top_theme_share": round(float(th.max()), 4),
            "nearest_analytes": ", ".join(near),
            "labelled": prim is not None,
        })
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "tables/p7_per_analyte.csv", index=False)

    lab = per[per.labelled].copy()
    n = len(lab)

    # ── confusion (primary label) ──
    y_true = np.array([L.of_motif[m] for m in lab.primary_motif])
    y_pred = np.array([L.names.index(t) for t in lab.predicted_theme])
    C = np.zeros((K, K), int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    pd.DataFrame(C, index=L.names, columns=L.names).to_csv(OUT / "tables/p7_confusion.csv")

    # ── calibration / reliability ──
    conf = lab.theme_confidence.values
    correct = lab.theme_top1.values.astype(float)
    bins = np.linspace(0, 1, 11)
    rel = []
    for i in range(10):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            rel.append({"bin_low": round(bins[i], 2), "bin_high": round(bins[i + 1], 2),
                        "n": int(m.sum()), "mean_confidence": round(float(conf[m].mean()), 4),
                        "accuracy": round(float(correct[m].mean()), 4)})
    rel = pd.DataFrame(rel)
    rel.to_csv(OUT / "tables/p7_reliability.csv", index=False)
    ece = float(sum(r["n"] / n * abs(r["accuracy"] - r["mean_confidence"]) for _, r in rel.iterrows()))

    # ── per-theme reference table (Part 6) ──
    prows = []
    for t, (name, gmem) in enumerate(zip(L.names, L.groups)):
        mem = lab[y_true == t]
        gi = [bio_ids.index(m) for m in gmem]
        bands = sorted({b for m in gmem for b in bio_motifs[bio_ids.index(m)].bands_cm})
        comps = sorted({c["component"] for m in gmem
                        for c in bio_motifs[bio_ids.index(m)].contributors},
                       key=lambda j: -float(np.max(v6.M[j, [bio_idx[i] for i in gi]])))
        confs = [bio_motifs[i].confidence for i in gi]
        prows.append({
            "theme": name, "K_index": t, "n_motifs": len(gmem), "motifs": ", ".join(gmem),
            "n_analytes": int(len(mem)), "coverage_pct": round(100 * len(mem) / n, 1),
            "top1": round(float(mem.theme_top1.mean()), 3) if len(mem) else None,
            "top3": round(float(mem.theme_top3.mean()), 3) if len(mem) else None,
            "median_rank": int(mem.theme_rank.median()) if len(mem) else None,
            "mean_confidence": round(float(mem.theme_confidence.mean()), 3) if len(mem) else None,
            "motif_confidence": round(float(np.mean(confs)), 3),
            "key_bands_cm": ", ".join(str(int(b)) for b in bands[:10]),
            "key_components": ", ".join(f"c{j}" for j in comps[:6]),
            "example_analytes": ", ".join(sorted(mem.analyte)[:6]),
            "failure_cases": ", ".join(sorted(mem[~mem.theme_top1].analyte)[:6]),
            "most_confused_with": (L.names[int(np.argmax(np.delete(C[t], t)) +
                                                (1 if int(np.argmax(np.delete(C[t], t))) >= t else 0))]
                                   if len(mem) and C[t].sum() > C[t, t] else "-"),
        })
    themes = pd.DataFrame(prows)
    themes.to_csv(OUT / "tables/p6_theme_reference.csv", index=False)

    # ── representative analytes at 5 tiers (Part 8) ──
    lab = lab.copy()
    lab["margin"] = lab.top_theme_share - lab.expected_theme_share
    tiers = {}
    excellent = lab[(lab.theme_rank == 1) & (lab.motif_rank == 1)].sort_values(
        "theme_confidence", ascending=False)
    good = lab[(lab.theme_rank == 1) & (lab.motif_rank > 1)].sort_values(
        "theme_confidence", ascending=False)
    moderate = lab[(lab.theme_rank == 2) | ((lab.theme_rank <= 3) & (lab.theme_rank > 1))]
    poor = lab[(lab.theme_rank > 3) & (lab.theme_rank <= 6)]
    failure = lab[lab.theme_rank > 6].sort_values("margin", ascending=False)
    for nm, d in [("excellent", excellent), ("good", good), ("moderate", moderate),
                  ("poor", poor), ("failure", failure)]:
        tiers[nm] = d.analyte.tolist()[:8]
    reps = []
    for nm, d in [("excellent", excellent), ("good", good), ("moderate", moderate),
                  ("poor", poor), ("failure", failure)]:
        if len(d):
            r = d.iloc[0]
            reps.append({"tier": nm, "analyte": r.analyte, "family": r.family,
                         "expected_themes": r.expected_themes, "predicted_theme": r.predicted_theme,
                         "theme_rank": int(r.theme_rank), "motif_rank": int(r.motif_rank),
                         "predicted_motif": r.predicted_motif,
                         "theme_confidence": r.theme_confidence,
                         "top_component": int(r.top_component),
                         "nearest_analytes": r.nearest_analytes})
    pd.DataFrame(reps).to_csv(OUT / "tables/p8_representatives.csv", index=False)

    summary = {
        "atlas_fingerprint": CANON,
        "hierarchy": {"components": 24, "mss_motifs": len(bio_ids), "chemical_themes": K,
                      "themes": L.as_dict()["themes"]},
        "n_analytes": len(analytes), "n_labelled": int(n),
        "theme_top1": round(float(lab.theme_top1.mean()), 4),
        "theme_top3": round(float(lab.theme_top3.mean()), 4),
        "motif_top1": round(float(lab.motif_top1.mean()), 4),
        "motif_top3": round(float(lab.motif_top3.mean()), 4),
        "median_theme_rank": int(lab.theme_rank.median()),
        "ece": round(ece, 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "tiers": {k: len(v) for k, v in tiers.items()},
        "tier_members": tiers,
        "representatives": reps,
        "per_theme": themes.to_dict("records"),
    }
    (OUT / "artifacts/p7_evaluation.json").write_text(json.dumps(summary, indent=2, default=str))
    np.savez(OUT / "artifacts/p7_vectors.npz",
             analytes=np.array(analytes), families=fams, zA=zA, A_bio=A, Th=Th,
             theme_names=np.array(L.names), motif_ids=np.array(bio_ids),
             T=L.T, M_bio=v6.M[:, bio_idx], grid=grid, confusion=C,
             corpusX=np.array([np.nan_to_num(corpus.X[ra == a]).mean(0) for a in analytes]))

    pd.set_option("display.width", 250)
    print(f"\nscored {n} labelled analytes of {len(analytes)}")
    print(f"theme top-1 {lab.theme_top1.mean():.3f} | top-3 {lab.theme_top3.mean():.3f} | ECE {ece:.3f}")
    print(f"motif top-1 {lab.motif_top1.mean():.3f} | top-3 {lab.motif_top3.mean():.3f}")
    print("\nper-theme:")
    print(themes[["theme", "n_motifs", "n_analytes", "top1", "top3", "median_rank",
                  "mean_confidence", "most_confused_with"]].to_string(index=False))
    print("\ntiers:", {k: len(v) for k, v in tiers.items()})
    print("\nrepresentatives:")
    print(pd.DataFrame(reps)[["tier", "analyte", "family", "expected_themes", "predicted_theme",
                              "theme_rank", "motif_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
