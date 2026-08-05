"""GAIRA V6 — Parts 3, 4 and 5.

Builds the components -> MSS -> chemical-theme hierarchy, sweeps K = 2..17 across
five partition-generation methods, and evaluates every level against a permutation
null. Selects the Pareto optimum on recoverability x interpretability.

Recoverability is CHANCE-CORRECTED (kappa), because raw top-1 accuracy rises
mechanically as K falls: a 2-theme hierarchy is right half the time by guessing.
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
KS = list(range(2, 18))
N_PERM = 2000
RNG = np.random.default_rng(0)


# ── PRE-STATED chemical-admissibility constraint (Part 4 tie-break) ──────────
# Fixed on chemistry BEFORE any partition was scored, and applied uniformly. A theme
# must correspond to a nameable biochemical class: its member motifs may not span more
# than one chemical SUPERCLASS. `phosphate_ester` and nothing else is marked BRIDGING,
# because a phosphate ester genuinely occurs inside nucleic acids, sugar phosphates and
# phospholipids alike, so it cannot discriminate a superclass.
SUPERCLASS = {
    "nucleobase_purine": "nucleic", "nucleobase_pyrimidine": "nucleic",
    "phosphate_ester": "BRIDGING",
    "aromatic_sidechain": "protein", "polypeptide": "protein", "free_amino_acid": "protein",
    "fatty_acid": "lipid", "acylglycerol": "lipid", "sterol": "lipid",
    "monosaccharide": "carbohydrate", "polysaccharide": "carbohydrate",
    "organic_acid": "metabolite", "sulfur_metabolite": "metabolite",
    "tetrapyrrole": "cofactor", "redox_cofactor": "cofactor", "polyene": "cofactor",
}


def admissible(groups, class_of):
    """True if every theme spans at most one non-bridging chemical superclass."""
    for g in groups:
        sc = {SUPERCLASS.get(class_of[m], "?") for m in g}
        sc.discard("BRIDGING")
        if len(sc) > 1:
            return False
    return True


def norm_entropy(p):
    p = np.asarray(p, float); p = p[p > 0]
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def ece(conf, correct, bins=10):
    """Expected calibration error."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(conf)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main():
    eng = GAIRAEngine()
    assert eng.atlas.meta["fingerprint"] == CANON, "FROZEN ATLAS CHANGED"
    H, grid = eng.atlas.components, eng.atlas.grid
    v6 = MSSLayerV6(MOTIFS_V6, eng.builder.reg, H, grid)
    bio_idx = [i for i, m in enumerate(v6.motifs) if not m.non_biochemical]
    bio_ids = [v6.motifs[i].id for i in bio_idx]
    bio_motifs = [v6.motifs[i] for i in bio_idx]

    corpus = DS.load_reference_corpus()
    Z = eng.atlas.coordinates(corpus.X)
    ra = corpus.meta.analyte.values
    analytes = sorted(set(ra))
    zA = np.array([Z[ra == a].mean(0) for a in analytes])
    A_all = np.array([v6.activate(z) for z in zA])
    A = A_all[:, bio_idx]                                  # (n_analytes, 17)
    fams = np.array([family_of(a) for a in analytes])

    # ── expected motif(s) per analyte, from exemplar membership only ──
    hits = []                       # per motif: set of analyte indices
    for m in bio_motifs:
        hits.append({i for i, a in enumerate(analytes) if any(name_matches(e, a) for e in m.exemplars)})
    exp_motifs, primary = [], []
    for i, a in enumerate(analytes):
        ms = [bio_ids[k] for k in range(len(bio_ids)) if i in hits[k]]
        exp_motifs.append(ms)
        # primary = the most SPECIFIC matching motif (shortest exemplar list)
        primary.append(min(ms, key=lambda mid: len(bio_motifs[bio_ids.index(mid)].exemplars))
                       if ms else None)
    labelled = np.array([p is not None for p in primary])
    print(f"corpus {len(analytes)} analytes | {labelled.sum()} carry an expected motif "
          f"({100*labelled.mean():.1f}% coverage) | {len(bio_ids)} biochemical motifs")

    # ── distance matrices ──
    D_act = TV.dist_activation(A)
    D_spec = TV.dist_spectral(v6.M[:, bio_idx], H)
    D_onto = TV.dist_ontology(bio_motifs, hits)
    D_hyb = (D_act / (D_act.max() + 1e-12) + D_spec / (D_spec.max() + 1e-12)
             + D_onto / (D_onto.max() + 1e-12)) / 3

    # Method A is only DEFINED at the levels the expert hierarchy specifies; evaluating it
    # at interpolated K (by merging the two smallest groups) produces partitions no
    # chemist proposed and unfairly penalises the method. Restrict it to its own levels.
    MANUAL_KS = set(TV.MANUAL_LEVELS)

    METHODS = {
        "A_manual": lambda K: TV.manual_partition(K, bio_ids) if K in MANUAL_KS else None,
        "B_activation": lambda K: TV.partition_from_distance(D_act, K, bio_ids),
        "C_spectral": lambda K: TV.partition_from_distance(D_spec, K, bio_ids),
        "D_ontology": lambda K: TV.partition_from_distance(D_onto, K, bio_ids),
        "E_hybrid": lambda K: TV.partition_from_distance(D_hyb, K, bio_ids),
    }

    # motif implied spectra (for spectral coherence)
    MS = (v6.M[:, bio_idx].T @ H)
    MS = MS / (np.linalg.norm(MS, axis=1, keepdims=True) + 1e-12)
    MScos = MS @ MS.T

    rows, partitions, confusions = [], {}, {}
    for meth, fn in METHODS.items():
        for K in KS:
            if K > len(bio_ids):
                continue
            groups = fn(K)
            if groups is None or len(groups) != K:
                continue
            L = TV.ThemeLayer(groups, bio_ids)
            adm = admissible(groups, {m.id: m.chemical_class for m in bio_motifs})
            Th = L.compose(A)                                  # (n_analytes, K)
            partitions[f"{meth}|{K}"] = L.as_dict()

            # expected theme set and primary label
            exp_sets = [sorted({L.of_motif[m] for m in ms}) for ms in exp_motifs]
            y_true = np.array([L.of_motif[p] if p else -1 for p in primary])
            mask = labelled & np.array([len(s) > 0 for s in exp_sets])
            Tm, ym = Th[mask], y_true[mask]
            es = [exp_sets[i] for i in np.where(mask)[0]]
            n = mask.sum()

            order = np.argsort(-Tm, axis=1)
            rank = np.array([min([list(order[i]).index(t) + 1 for t in es[i]]) for i in range(n)])
            top1 = float((rank == 1).mean()); top3 = float((rank <= 3).mean())
            y_pred = order[:, 0]

            # macro F1 + balanced accuracy on the primary label
            f1s, recs = [], []
            for t in range(K):
                tp = int(((y_pred == t) & (ym == t)).sum())
                fp = int(((y_pred == t) & (ym != t)).sum())
                fn_ = int(((y_pred != t) & (ym == t)).sum())
                if tp + fn_ == 0:
                    continue
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn_)
                f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
                recs.append(rec)
            macro_f1 = float(np.mean(f1s)) if f1s else 0.0
            bal_acc = float(np.mean(recs)) if recs else 0.0

            # calibration: predicted theme's share of total theme mass
            tot = Tm.sum(1, keepdims=True)
            conf = np.divide(Tm.max(1), tot[:, 0], out=np.zeros(n), where=tot[:, 0] > 1e-12)
            cal = ece(conf, (rank == 1).astype(float))

            # permutation null: shuffle the analyte -> expected-theme assignment
            nulls = np.empty(N_PERM)
            for b in range(N_PERM):
                perm = RNG.permutation(n)
                nulls[b] = np.mean([min([list(order[i]).index(t) + 1
                                         for t in es[perm[i]]]) == 1 for i in range(n)])
            null_mean, null_p95 = float(nulls.mean()), float(np.percentile(nulls, 95))
            kappa = (top1 - null_mean) / (1 - null_mean + 1e-12)

            # ── interpretability ──
            chem, spec, sizes = [], [], []
            for t, gmem in enumerate(L.groups):
                members = [i for i in range(n) if ym[i] == t]
                fam_counts = pd.Series(fams[mask][members]).value_counts().values if members else np.array([1])
                chem.append(1 - norm_entropy(fam_counts))
                gi = [bio_ids.index(m) for m in gmem]
                if len(gi) > 1:
                    iu = np.triu_indices(len(gi), 1)
                    spec.append(float(MScos[np.ix_(gi, gi)][iu].mean()))
                else:
                    spec.append(1.0)
                sizes.append(len(gmem))
            chem_coh, spec_coh = float(np.mean(chem)), float(np.mean(spec))
            resolution = float(np.log(K) / np.log(len(bio_ids)))
            interp = 0.4 * chem_coh + 0.3 * spec_coh + 0.3 * resolution

            # theme overlap + usage entropy
            Tn = Tm / (np.linalg.norm(Tm, axis=0, keepdims=True) + 1e-12)
            O = Tn.T @ Tn
            iu = np.triu_indices(K, 1)
            overlap = float(O[iu].mean()) if K > 1 else 0.0
            usage = norm_entropy(np.bincount(y_pred, minlength=K))

            rows.append({
                "method": meth, "K": K, "n_scored": int(n), "chemically_admissible": bool(adm),
                "top1": round(top1, 4), "top3": round(top3, 4),
                "null_top1": round(null_mean, 4), "null_p95": round(null_p95, 4),
                "kappa": round(float(kappa), 4),
                "above_null_p95": bool(top1 > null_p95),
                "macro_f1": round(macro_f1, 4), "balanced_acc": round(bal_acc, 4),
                "mean_expected_rank": round(float(rank.mean()), 3),
                "median_expected_rank": int(np.median(rank)),
                "ece": round(cal, 4), "mean_confidence": round(float(conf.mean()), 4),
                "chem_coherence": round(chem_coh, 4), "spec_coherence": round(spec_coh, 4),
                "resolution": round(resolution, 4), "interpretability": round(interp, 4),
                "score_kappa_x_interp": round(float(kappa * interp), 4),
                "theme_overlap": round(overlap, 4), "theme_usage_entropy": round(usage, 4),
                "max_theme_size": int(max(sizes)), "min_theme_size": int(min(sizes)),
                "theme_names": " | ".join(L.names),
            })
            if meth == "A_manual" or K in (6, 8, 10):
                C = np.zeros((K, K), int)
                for i in range(n):
                    C[ym[i], y_pred[i]] += 1
                confusions[f"{meth}|{K}"] = {"names": L.names, "matrix": C.tolist()}

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables/p4_theme_sweep.csv", index=False)

    # ── Pareto front on (kappa, interpretability) ──
    pts = df[["kappa", "interpretability"]].values
    pareto = np.ones(len(df), bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j and pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and \
               (pts[j] > pts[i]).any():
                pareto[i] = False
                break
    df["pareto"] = pareto
    df.to_csv(OUT / "tables/p4_theme_sweep.csv", index=False)

    raw_best = df.loc[df.score_kappa_x_interp.idxmax()]
    best_pareto = df[df.pareto].sort_values("score_kappa_x_interp", ascending=False)
    # tie-break: the composite score is flat across the front, so apply the pre-stated
    # chemical-admissibility constraint, then prefer genuine abstraction (smaller K).
    adm_df = df[df.chemically_admissible]
    TOL = 0.02
    top_adm = adm_df.score_kappa_x_interp.max()
    band = adm_df[adm_df.score_kappa_x_interp >= top_adm - TOL]
    best = band.sort_values(["K", "score_kappa_x_interp"], ascending=[True, False]).iloc[0]

    summary = {
        "atlas_fingerprint": CANON,
        "n_biochemical_motifs": len(bio_ids),
        "motif_ids": bio_ids,
        "K_range": [min(KS), max(KS)],
        "methods": list(METHODS),
        "n_permutations": N_PERM,
        "label_coverage": round(float(labelled.mean()), 4),
        "interpretability_definition":
            "0.4*chemical_coherence + 0.3*spectral_coherence + 0.3*resolution, where "
            "chemical_coherence = 1 - normalised entropy of member analytes' chemical families, "
            "spectral_coherence = mean pairwise cosine of member motifs' implied Raman spectra, "
            "resolution = log(K)/log(K_max). Resolution is included so that a 2-theme hierarchy "
            "is not rewarded for being trivially coherent.",
        "recoverability_definition":
            "kappa = (top1 - permutation_null_top1) / (1 - permutation_null_top1). Raw top-1 "
            "rises mechanically as K falls; kappa removes that.",
        "admissibility_rule": "A theme must span at most one non-bridging chemical superclass. Superclasses fixed on chemistry before scoring; phosphate_ester is the only BRIDGING class.",
        "superclass_map": SUPERCLASS,
        "n_admissible": int(df.chemically_admissible.sum()),
        "n_total": int(len(df)),
        "raw_score_optimum": {k: (raw_best[k].item() if hasattr(raw_best[k], "item") else raw_best[k])
                              for k in ("method", "K", "kappa", "interpretability",
                                        "score_kappa_x_interp", "chemically_admissible", "theme_names")},
        "tie_break_tolerance": TOL,
        "selected": {k: (best[k].item() if hasattr(best[k], "item") else best[k])
                     for k in ("method", "K", "top1", "null_top1", "kappa", "macro_f1",
                               "balanced_acc", "ece", "interpretability",
                               "score_kappa_x_interp", "chemically_admissible", "theme_names")},
        "pareto_front": best_pareto[["method", "K", "kappa", "interpretability",
                                     "score_kappa_x_interp", "top1", "null_top1",
                                     "chemically_admissible", "theme_names"]]
            .to_dict("records"),
        "selected_partition": partitions[f"{best.method}|{int(best.K)}"],
        "best_per_method": df.loc[df.groupby("method").score_kappa_x_interp.idxmax()][
            ["method", "K", "kappa", "interpretability", "score_kappa_x_interp", "top1"]]
            .to_dict("records"),
    }
    (OUT / "artifacts/p4_theme_optimisation.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "artifacts/p4_partitions.json").write_text(json.dumps(partitions, indent=2))
    (OUT / "artifacts/p4_confusions.json").write_text(json.dumps(confusions, indent=2))
    np.savez(OUT / "artifacts/p3_mss_v6.npz",
             M_v6=v6.M, motif_ids=np.array([m.id for m in v6.motifs]),
             bio_ids=np.array(bio_ids), analytes=np.array(analytes), families=fams,
             A_bio=A, zA=zA, grid=grid, D_act=D_act, D_spec=D_spec, D_onto=D_onto, D_hyb=D_hyb,
             primary=np.array([p or "" for p in primary]))

    pd.set_option("display.width", 250)
    print("\nbest per method:")
    print(pd.DataFrame(summary["best_per_method"]).to_string(index=False))
    print("\nPareto front (kappa x interpretability):")
    print(best_pareto[["method", "K", "top1", "null_top1", "kappa", "interpretability",
                       "score_kappa_x_interp", "chemically_admissible"]].head(12).to_string(index=False))
    print(f"\nraw score optimum: {raw_best.method} K={int(raw_best.K)} "
          f"score={raw_best.score_kappa_x_interp:.4f} admissible={bool(raw_best.chemically_admissible)}")
    print(f"admissible partitions: {int(df.chemically_admissible.sum())} of {len(df)}")
    print("admissible band (within %.2f of the admissible optimum):" % TOL)
    print(band[["method", "K", "kappa", "interpretability", "score_kappa_x_interp"]]
          .sort_values("K").to_string(index=False))
    print(f"\nSELECTED: {best.method} at K={best.K}  "
          f"(top1 {best.top1:.3f} vs null {best.null_top1:.3f}, kappa {best.kappa:.3f}, "
          f"interp {best.interpretability:.3f})")
    print("themes:", best.theme_names)
    print("\nA_manual sweep:")
    print(df[df.method == "A_manual"][["K", "top1", "null_top1", "kappa", "macro_f1",
                                       "balanced_acc", "ece", "interpretability",
                                       "score_kappa_x_interp"]].to_string(index=False))


if __name__ == "__main__":
    main()
