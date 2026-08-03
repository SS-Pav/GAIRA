"""GAIRA V5 — pure Ag-SERS abstraction-recovery analysis.

Asks: when exact analyte identity is lost, does the correct broader chemistry survive? Evaluates
component-evidence, MSS-motif, molecular-subclass (leave-one-analyte-out classification), and
broad-theme recovery in the FROZEN representations. Reproduces V4's per-analyte latent/MSS/theme
vectors bit-for-bit; adds only abstraction-level evaluation. Deterministic. Atlas 09ed804a… unchanged.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "results/v5_rebuild/spike_validation/code"))
from gaira.engine import GAIRAEngine
from gaira.engine.mss import MSSLayer
from gaira.foundation import dataset as DS
from gaira.foundation.families_raman import family_of
from gaira.data.synonyms import canonical
import spike_lib as SL

OUT = REPO / "results/v5_rebuild/abstraction_recovery_v5"
(OUT / "tables").mkdir(parents=True, exist_ok=True); (OUT / "artifacts").mkdir(parents=True, exist_ok=True)
V4 = REPO / "results/v5_rebuild/hierarchical_recoverability_v4/tables/per_analyte_evidence_profile.csv"
PHASE7 = REPO / "results/v5_rebuild/spike_validation/tables/phase7_serum_vs_pure.csv"
CANON = "09ed804a40836f4a05a91ba10900cded"
OVERLAY = pd.read_csv(OUT / "tables/analyte_classification_overlay.csv").set_index("canonical_analyte")


def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def cos_rows(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12); Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T
def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); q = np.empty(n); prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        i = n - rank; prev = min(prev, p[idx] * n / i); q[idx] = prev
    return q


def main():
    eng = GAIRAEngine(); mss = MSSLayer.from_engine(eng); atlas = eng.atlas
    assert atlas.meta["fingerprint"] == CANON, "FROZEN ATLAS CHANGED"
    THEMES = eng.builder.onto.biochemical_theme_ids
    MOTIFS = [m.id for m in mss.motifs if not m.non_biochemical]
    PUR = THEMES.index("nucleic_purine")

    def coords(V): return atlas.coordinates(np.atleast_2d(np.nan_to_num(V)))
    def vecs(c):
        b = eng.infer(coordinates=np.asarray(c, float), domain="buffer").bsv
        tv = np.array([b.composition[t] for t in THEMES]); acts = {a.id: a.composition for a in mss.activate(b)}
        mv = np.array([acts.get(x, 0.0) for x in MOTIFS]); return tv, mv, float(b.ood_score), float(b.overall_confidence)

    corpus = DS.load_reference_corpus(); Zr = coords(corpus.X)
    Xs, rs = SL.load_pure_sers(); Zs = coords(Xs)
    ra = corpus.meta.analyte.values; sa = np.array([canonical(a) for a in rs.analyte.values])
    matched = sorted(set(ra) & set(sa))
    zR = {a: Zr[ra == a].mean(0) for a in matched}; zS = {a: Zs[sa == a].mean(0) for a in matched}
    prof = {a: {"R": vecs(zR[a]), "S": vecs(zS[a])} for a in matched}
    N = len(matched)
    ZR = np.array([zR[a] for a in matched]); ZS = np.array([zS[a] for a in matched])
    TR = np.array([prof[a]["R"][0] for a in matched]); TS = np.array([prof[a]["S"][0] for a in matched])
    MR = np.array([prof[a]["R"][1] for a in matched]); MS = np.array([prof[a]["S"][1] for a in matched])
    oodS = {a: prof[a]["S"][2] for a in matched}; confS = {a: prof[a]["S"][3] for a in matched}
    Xb, _ = SL.load_serum_baseline(); tBlank, mBlank, _, _ = vecs(coords(Xb).mean(0))

    v4 = pd.read_csv(V4).set_index("analyte")
    p7 = pd.read_csv(PHASE7).set_index("analyte")

    SPACE = {"latent": (ZR, ZS), "MSS": (MR, MS), "theme": (TR, TS)}

    # ── LEVEL 1 — NMF component evidence ──
    comp_rows = []
    for i, a in enumerate(matched):
        topR = set(np.argsort(-ZR[i])[:3]); topS = set(np.argsort(-ZS[i])[:3])
        overlap = len(topR & topS) / 3
        mass_ret = float(ZS[i][list(topR)].sum() / (ZS[i].sum() + 1e-12))
        null = [len(topR & set(np.argsort(-ZS[j])[:3])) / 3 for j in range(N) if j != i]
        n95 = float(np.percentile(null, 95))
        comp_rows.append({"analyte": a, "comp_top3_overlap": round(overlap, 3),
                          "comp_mass_retained": round(mass_ret, 3), "comp_null95": round(n95, 3),
                          "component_recovered": bool(overlap > n95 and overlap >= 2 / 3)})
    comp = pd.DataFrame(comp_rows).set_index("analyte")

    # ── LEVEL 2 — MSS motif recovery (expected motif) ──
    mss_rows = []
    for i, a in enumerate(matched):
        exp = [OVERLAY.loc[a, "expected_mss_primary"], OVERLAY.loc[a, "expected_mss_secondary"]]
        exp = [e for e in exp if isinstance(e, str) and e in MOTIFS]
        if not exp:
            mss_rows.append({"analyte": a, "expected_mss": "unassigned", "mss_rank_R": None,
                             "mss_rank_S": None, "mss_top3": None, "mss_enrich_null": None,
                             "mss_motif_recovered": False, "mss_status": "assignment unavailable"}); continue
        # rank of best expected motif in Ag-SERS
        oS = np.argsort(-MS[i]); rankS = {MOTIFS[k]: r for r, k in enumerate(oS, 1)}
        oR = np.argsort(-MR[i]); rankR = {MOTIFS[k]: r for r, k in enumerate(oR, 1)}
        best = min(exp, key=lambda e: rankS[e]); r_s = rankS[best]; r_r = rankR[best]
        top3 = r_s <= 3
        ei = MOTIFS.index(best)
        # enrichment: expected-motif score vs out-of-group analytes' score for that motif
        fam = OVERLAY.loc[a, "broad_family"]
        outgrp = [j for j, b in enumerate(matched) if OVERLAY.loc[b, "broad_family"] != fam]
        null = MS[outgrp, ei]; n95 = float(np.percentile(null, 95))
        enriched = MS[i, ei] > n95
        recovered = bool(top3 and enriched and MS[i, ei] > mBlank[ei])
        status = ("expected motif recovered" if recovered else
                  "motif present but nonspecific" if top3 else "motif not recovered")
        mss_rows.append({"analyte": a, "expected_mss": best, "mss_rank_R": r_r, "mss_rank_S": r_s,
                         "mss_top3": bool(top3), "mss_enrich_null": round(float(MS[i, ei] - n95), 4),
                         "mss_motif_recovered": recovered, "mss_status": status})
    mssdf = pd.DataFrame(mss_rows).set_index("analyte")

    # ── LEVEL 3 — LOAO nearest-centroid classification (subclass / family / theme granularity) ──
    labels = {"subclass": OVERLAY["subclass_primary"].reindex(matched).values,
              "family": np.array([family_of(a) for a in matched]),
              "theme": OVERLAY["expected_theme_primary"].reindex(matched).values}
    rng = np.random.default_rng(0)

    def loao(Rvec, Svec, y):
        """leave-one-analyte-out nearest Raman-centroid; classify Ag-SERS. Returns pred, classifiable mask."""
        y = np.asarray(y); pred = np.array([None] * N, dtype=object); classifiable = np.zeros(N, bool)
        for i in range(N):
            cls = {}
            for c in set(y):
                idx = [j for j in range(N) if j != i and y[j] == c]
                if idx: cls[c] = Rvec[idx].mean(0)
            if not cls: continue
            classifiable[i] = (y[i] in cls)         # true class must have another member
            best = max(cls, key=lambda c: cos(Svec[i], cls[c])); pred[i] = best
        return pred, classifiable

    def metrics(y, pred, mask):
        y = np.asarray(y); m = mask & np.array([p is not None for p in pred])
        yt = y[m]; yp = np.array([pred[i] for i in range(N)])[m]
        acc = float((yt == yp).mean()) if len(yt) else 0.0
        classes = sorted(set(yt)); recalls = []; f1s = []
        for c in classes:
            tp = int(((yt == c) & (yp == c)).sum()); fn = int(((yt == c) & (yp != c)).sum())
            fp = int(((yt != c) & (yp == c)).sum())
            rec = tp / (tp + fn) if tp + fn else 0.0; prec = tp / (tp + fp) if tp + fp else 0.0
            recalls.append(rec); f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        return {"n": int(m.sum()), "accuracy": round(acc, 3),
                "balanced_accuracy": round(float(np.mean(recalls)) if recalls else 0.0, 3),
                "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 3)}

    class_results = {}; loao_pred = {}
    for gran, y in labels.items():
        for sp, (Rv, Sv) in SPACE.items():
            pred, mask = loao(Rv, Sv, y)
            loao_pred[(gran, sp)] = (pred, mask)
            mt = metrics(y, pred, mask)
            # permutation null (shuffle labels, analyte-level)
            perm = []
            for _ in range(200):
                ys = rng.permutation(y); pp, mm = loao(Rv, Sv, ys); perm.append(metrics(ys, pp, mm)["accuracy"])
            pval = (1 + int(np.sum(np.array(perm) >= mt["accuracy"]))) / (1 + len(perm))
            # bootstrap CI over analytes
            idx = np.where(mask & np.array([p is not None for p in pred]))[0]
            boots = []
            yv = np.asarray(y)
            for _ in range(800):
                s = idx[rng.integers(0, len(idx), len(idx))]
                boots.append(float((yv[s] == np.array([pred[k] for k in s])).mean()))
            class_results[(gran, sp)] = {**mt, "perm_null_mean": round(float(np.mean(perm)), 3),
                "perm_p": round(pval, 4), "acc_ci95": [round(float(np.percentile(boots, 2.5)), 3),
                round(float(np.percentile(boots, 97.5)), 3)]}
    # CONTROL: Raman→Raman classification (same modality) — proves the classes ARE separable and
    # any Ag-SERS failure is the MODALITY GAP, not an ill-defined taxonomy.
    control = {}
    for gran, y in labels.items():
        Rv = SPACE["latent"][0]
        pred, mask = loao(Rv, Rv, y)               # classify held-out RAMAN by Raman centroids
        control[gran] = metrics(y, pred, mask)
    cr = pd.DataFrame([{"granularity": g, "space": s, **v} for (g, s), v in class_results.items()]
                      + [{"granularity": g, "space": "latent(Raman→Raman control)", **v} for g, v in control.items()])
    cr.to_csv(OUT / "tables/subclass_classification_results.csv", index=False)

    # ── LEVEL 4 — broad-theme recovery (expected theme, family-mismatched null) ──
    theme_rows = []
    for i, a in enumerate(matched):
        exp = [OVERLAY.loc[a, "expected_theme_primary"], OVERLAY.loc[a, "expected_theme_secondary"]]
        exp = [e for e in exp if isinstance(e, str) and e in THEMES]
        if not exp:
            theme_rows.append({"analyte": a, "expected_theme": "unassigned", "theme_rank_S": None,
                               "theme_top3": None, "theme_enrich_null": None, "theme_recovered": False}); continue
        oS = np.argsort(-TS[i]); rankS = {THEMES[k]: r for r, k in enumerate(oS, 1)}
        best = min(exp, key=lambda e: rankS[e]); r_s = rankS[best]; ti = THEMES.index(best)
        fam = OVERLAY.loc[a, "broad_family"]
        outgrp = [j for j, b in enumerate(matched) if OVERLAY.loc[b, "broad_family"] != fam]
        n95 = float(np.percentile(TS[outgrp, ti], 95))
        enriched = TS[i, ti] > n95; top3 = r_s <= 3
        # background-corrected: expected theme above the serum blank's share of that theme
        above_bg = TS[i, ti] > tBlank[ti]
        recovered = bool(top3 and enriched and above_bg)
        theme_rows.append({"analyte": a, "expected_theme": best, "theme_rank_S": r_s,
                           "theme_top3": bool(top3), "theme_enrich_null": round(float(TS[i, ti] - n95), 4),
                           "theme_above_background": bool(above_bg), "theme_recovered": recovered})
    themedf = pd.DataFrame(theme_rows).set_index("analyte")

    # ── nearest-neighbour SAME-CLASS retrieval (graceful degradation; robust to global shift) ──
    # For each analyte's Ag-SERS, its nearest OTHER Raman analyte (self excluded). Does it share
    # subclass / family / theme? This is the intuitive "does the error stay in the right chemistry".
    sim_SR = cos_rows(ZS, ZR)                       # sim[i,j] = cos(SERS_i, Raman_j)
    np.fill_diagonal(sim_SR, -np.inf)               # exclude self
    nn = sim_SR.argmax(1)
    fam_arr = np.array([family_of(a) for a in matched]); sub_arr = OVERLAY["subclass_primary"].reindex(matched).values
    th_arr = OVERLAY["expected_theme_primary"].reindex(matched).values
    nn_sub = np.array([sub_arr[nn[i]] == sub_arr[i] for i in range(N)])
    nn_fam = np.array([fam_arr[nn[i]] == fam_arr[i] for i in range(N)])
    nn_theme = np.array([th_arr[nn[i]] == th_arr[i] for i in range(N)])

    def chance(y):
        y = np.asarray(y); n = len(y)
        return float(sum(c * (c - 1) for c in pd.Series(y).value_counts()) / (n * (n - 1)))
    nn_summary = {}
    for nm, hit, y in [("subclass", nn_sub, sub_arr), ("family", nn_fam, fam_arr), ("theme", nn_theme, th_arr)]:
        obs = float(hit.mean()); ch = chance(y)
        # permutation p: shuffle labels, recompute same-class rate at fixed nn
        perm = []
        for _ in range(2000):
            ys = rng.permutation(y); perm.append(float(np.mean([ys[nn[i]] == ys[i] for i in range(N)])))
        p = (1 + int(np.sum(np.array(perm) >= obs))) / (1 + len(perm))
        boots = [float(hit[rng.integers(0, N, N)].mean()) for _ in range(2000)]
        nn_summary[nm] = {"observed": round(obs, 3), "chance": round(ch, 3), "perm_p": round(p, 4),
                          "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]}

    # ── graded MSS / theme tiers (present / enriched / specific) ──
    mss_present = mssdf["mss_top3"].fillna(False).astype(bool).values
    mss_specific = mssdf["mss_motif_recovered"].values
    mss_enriched = (mssdf["mss_enrich_null"].fillna(-1) > 0).values
    theme_present = themedf["theme_top3"].fillna(False).astype(bool).values
    theme_specific = themedf["theme_recovered"].values
    theme_enriched = (themedf["theme_enrich_null"].fillna(-1) > 0).values

    # ── best-space LOAO correctness per analyte (for the ladder) ──
    def best_correct(gran):
        y = labels[gran]; ok = np.zeros(N, bool); classif = np.zeros(N, bool)
        for sp in SPACE:
            pred, mask = loao_pred[(gran, sp)]
            classif |= mask
            ok |= np.array([mask[i] and pred[i] == y[i] for i in range(N)])
        return ok, classif
    sub_ok, sub_classif = best_correct("subclass")
    fam_ok, fam_classif = best_correct("family")

    # ── master per-analyte abstraction table ──
    rows = []
    for i, a in enumerate(matched):
        latent_id = bool(v4.loc[a, "latent_recovered"]); mss_id = bool(v4.loc[a, "MSS_recovered"])
        theme_id = bool(v4.loc[a, "theme_recovered"])
        c = comp.loc[a]; ms = mssdf.loc[a]; th = themedf.loc[a]
        pert = a in ("adenine", "ergothioneine", "urate")
        serum = a in p7.index; serum_tier = v4.loc[a, "serum_tier"] if serum else "not tested"
        matrix_rec = bool(v4.loc[a, "matrix_recovered"]) if serum else False
        # highest recovered level — uses only STATISTICALLY DEFENSIBLE specific flags.
        # Cross-modal subclass/family classification is at chance (see nearest_neighbor_retrieval /
        # subclass_classification_results), so it is NOT counted as per-analyte recovery here — only
        # reported in aggregate. "broad presence only" captures top-3 presence that is not specific.
        if latent_id: highest = "exact analyte"
        elif ms["mss_motif_recovered"] or c["component_recovered"]: highest = "component/motif (specific)"
        elif th["theme_recovered"]: highest = "theme (specific)"
        elif pert: highest = "perturbation-only"
        elif mss_present[i] or theme_present[i]: highest = "broad presence only (non-specific)"
        else: highest = "none"
        # transparent profile
        prof_bits = []
        if latent_id: prof_bits.append("exact analyte recovered")
        elif mss_id or ms["mss_motif_recovered"]: prof_bits.append("latent fingerprint lost, expected MSS retained")
        if th["theme_recovered"] and not latent_id: prof_bits.append("expected theme specifically recovered")
        elif (mss_present[i] or theme_present[i]) and not (latent_id or ms["mss_motif_recovered"]):
            prof_bits.append("expected motif/theme PRESENT (top-3) but not specific — common-background dominated")
        if pert: prof_bits.append("perturbation validated")
        if matrix_rec: prof_bits.append("matrix recovered")
        if not prof_bits: prof_bits = ["no specific evidence"]
        rows.append({
            "analyte": a, "broad_family": family_of(a), "subclass": OVERLAY.loc[a, "subclass_primary"],
            "subclass_exploratory": bool(OVERLAY.loc[a, "subclass_exploratory"]),
            "latent_identity_recovered": latent_id, "mss_identity_recovered": mss_id, "theme_identity_recovered": theme_id,
            "comp_top3_overlap": c["comp_top3_overlap"], "comp_mass_retained": c["comp_mass_retained"],
            "component_recovered": bool(c["component_recovered"]),
            "expected_mss": ms["expected_mss"], "mss_rank_S": ms["mss_rank_S"],
            "mss_enrich_null": ms["mss_enrich_null"], "mss_motif_recovered": bool(ms["mss_motif_recovered"]),
            "mss_status": ms["mss_status"],
            "subclass_loao_recovered": bool(sub_ok[i]), "subclass_classifiable": bool(sub_classif[i]),
            "family_loao_recovered": bool(fam_ok[i]),
            "expected_theme": th["expected_theme"], "expected_theme_rank_S": th["theme_rank_S"],
            "theme_present_top3": bool(theme_present[i]), "theme_enrich_null": th["theme_enrich_null"],
            "theme_recovered": bool(th["theme_recovered"]),
            "mss_present_top3": bool(mss_present[i]),
            "nn_same_subclass": bool(nn_sub[i]), "nn_same_family": bool(nn_fam[i]), "nn_same_theme": bool(nn_theme[i]),
            "nn_nearest_raman": matched[nn[i]],
            "perturbation_status": ("dose-response" if a in ("adenine", "ergothioneine")
                                    else "directional depletion" if a == "urate" else "not tested"),
            "serum_tier": serum_tier, "matrix_recovered": matrix_rec,
            "delta_purine": round(float(TS[i][PUR] - TR[i][PUR]), 4),
            "ood_sers": round(oodS[a], 4), "confidence_sers": round(confS[a], 4),
            "highest_recovered_level": highest, "evidence_profile": "; ".join(prof_bits),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables/per_analyte_abstraction_recovery.csv", index=False)

    # ── recovery-by-abstraction-level counts ──
    n_mss_assigned = int((df.expected_mss != "unassigned").sum())
    n_theme_assigned = int((df.expected_theme != "unassigned").sum())
    ladder = [
        # (level, tier, n, denominator)
        ("exact analyte", "specific (V4 latent rank-1)", int(df.latent_identity_recovered.sum()), N),
        ("NMF component", "top-3 overlap > null", int(df.component_recovered.sum()), N),
        ("expected MSS motif", "present (top-3)", int(mss_present.sum()), n_mss_assigned),
        ("expected MSS motif", "specific (>null & >bg)", int(mss_specific.sum()), n_mss_assigned),
        ("molecular subclass", "NN same-subclass retrieval", int(nn_sub.sum()), N),
        ("molecular subclass", "LOAO centroid (best space)", int(df.subclass_loao_recovered.sum()), int(df.subclass_classifiable.sum())),
        ("broad family", "NN same-family retrieval", int(nn_fam.sum()), N),
        ("broad family", "LOAO centroid (best space)", int(df.family_loao_recovered.sum()), int(fam_classif.sum())),
        ("broad theme", "present (expected top-3)", int(theme_present.sum()), n_theme_assigned),
        ("broad theme", "specific (>null & >bg)", int(theme_specific.sum()), n_theme_assigned),
        ("perturbation", "functional (3 analytes)", 3, N),
        ("matrix (serum)", "strong tier", int(df.matrix_recovered.sum()), int((df.serum_tier != "not tested").sum())),
    ]
    lad = pd.DataFrame([{"level": l, "tier": t, "n_recovered": n, "denominator": d,
                         "fraction": round(n / d, 3) if d else 0.0} for l, t, n, d in ladder])
    lad.to_csv(OUT / "tables/recovery_by_abstraction_level.csv", index=False)
    pd.DataFrame([{"granularity": k, **v} for k, v in nn_summary.items()]).to_csv(
        OUT / "tables/nearest_neighbor_retrieval.csv", index=False)

    # family breakdown
    fam_rows = []
    for f, g in df.groupby("broad_family"):
        fam_rows.append({"broad_family": f, "n": len(g),
                         "exact": int(g.latent_identity_recovered.sum()),
                         "mss": int(g.mss_motif_recovered.sum()),
                         "subclass": int(g.subclass_loao_recovered.sum()),
                         "family_loao": int(g.family_loao_recovered.sum()),
                         "theme": int(g.theme_recovered.sum()),
                         "highest_common": g.highest_recovered_level.mode().iloc[0]})
    pd.DataFrame(fam_rows).to_csv(OUT / "tables/family_abstraction_breakdown.csv", index=False)

    # ── summary json ──
    v4id = {"latent": int(df.latent_identity_recovered.sum()), "MSS": int(df.mss_identity_recovered.sum()),
            "theme": int(df.theme_identity_recovered.sum())}
    summary = {
        "atlas_fingerprint": CANON, "n_matched": N,
        "reproducibility_vs_v4_identity": {"latent": v4id["latent"] == 7, "MSS": v4id["MSS"] == 3, "theme": v4id["theme"] == 4},
        "exact_identity": v4id,
        "recovery_ladder": [{"level": l, "tier": t, "n": n, "denom": d} for l, t, n, d in ladder],
        "recovery_ladder_fractions": {f"{l} — {t}": round(n / d, 3) if d else 0.0 for l, t, n, d in ladder},
        "classification": {f"{g}|{s}": class_results[(g, s)] for (g, s) in class_results},
        "recovered_lists": {
            "exact_latent": df[df.latent_identity_recovered].analyte.tolist(),
            "mss_motif": df[df.mss_motif_recovered].analyte.tolist(),
            "subclass_loao": df[df.subclass_loao_recovered].analyte.tolist(),
            "family_loao": df[df.family_loao_recovered].analyte.tolist(),
            "theme": df[df.theme_recovered].analyte.tolist(),
        },
        "highest_level_counts": df.highest_recovered_level.value_counts().to_dict(),
        "nearest_neighbor_retrieval": nn_summary,
        "graded_tiers": {"mss_present": int(mss_present.sum()), "mss_enriched": int(mss_enriched.sum()),
                         "mss_specific": int(mss_specific.sum()),
                         "theme_present": int(theme_present.sum()), "theme_enriched": int(theme_enriched.sum()),
                         "theme_specific": int(theme_specific.sum())},
        "serum_blank_purine_theme": round(float(tBlank[PUR]), 4),
        "mss_vs_latent_subclass": {"latent": class_results[("subclass", "latent")]["balanced_accuracy"],
                                   "MSS": class_results[("subclass", "MSS")]["balanced_accuracy"],
                                   "theme": class_results[("subclass", "theme")]["balanced_accuracy"]},
        "raman_raman_control": control,
    }
    (OUT / "artifacts/abstraction_summary.json").write_text(json.dumps(summary, indent=2))
    # save vectors for figures
    np.savez(OUT / "artifacts/vectors_v5.npz", analytes=np.array(matched),
             families=np.array([family_of(a) for a in matched]),
             subclass=OVERLAY["subclass_primary"].reindex(matched).values.astype(str),
             ZR=ZR, ZS=ZS, MR=MR, MS=MS, TR=TR, TS=TS, themes=np.array(THEMES), motifs=np.array(MOTIFS),
             ram_spec=np.array([np.nan_to_num(corpus.X[ra == a]).mean(0) for a in matched]),
             sers_spec=np.array([np.nan_to_num(Xs[sa == a]).mean(0) for a in matched]),
             grid=np.asarray(getattr(atlas, "grid", np.arange(TR.shape[0])), float), t_blank=tBlank)

    print(json.dumps({"fingerprint": CANON, "N": N, "exact_identity": v4id,
                      "nearest_neighbor_retrieval": nn_summary, "graded_tiers": summary["graded_tiers"],
                      "subclass_balanced_acc": summary["mss_vs_latent_subclass"],
                      "highest_level_counts": summary["highest_level_counts"]}, indent=2))
    print("\nrecovery ladder (graded):"); print(lad.to_string(index=False))
    print("\nclassification (granularity | space : balanced_acc, macro_f1, perm_p):")
    for (g, s), v in class_results.items():
        print(f"  {g:9s} | {s:6s}: bacc {v['balanced_accuracy']}, f1 {v['macro_f1']}, p {v['perm_p']}, acc {v['accuracy']} (n={v['n']})")


if __name__ == "__main__":
    main()
