"""GAIRA V5 Phase 2 Stage B — Biochemical Representation Strategy Benchmark (driver).

Compares direct baselines (Stage A reproduction), interpretable evidence
representations (I1-I4), and small encoder embeddings (E1-E3, +E4 hybrid) under one
leakage-safe framework. Primary metric: HELD-OUT matched-analyte cross-modal
retrieval (Split B). Emits tables/figures + a pre-declared scorecard and B1-B5 decision.

Deterministic. Encoder feasibility study — NOT foundation-model training.
Run: python results/v5_rebuild/phase2_stage_b/code/run_stage_b.py
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.evidence import (datasets as D, splits as SP, regions, wavelets, dictionary, basis,
                            evaluation as EV, training as TR, hybrid as HY, projection as PJ,
                            interpretability as IN, uncertainty as UQ, serialization as SER)
from gaira.evidence.augmentations import AugConfig, augmentation_audit
from gaira.evidence.base import Representation

SEED = 0
PH = REPO / "results/v5_rebuild/phase2_stage_b"
FIG, TAB, CFG, MOD, LOG = PH/"figures", PH/"tables", PH/"configs", PH/"models", PH/"logs"
for p in (FIG, TAB, CFG, MOD, LOG): p.mkdir(parents=True, exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class IdentityRep(Representation):
    """Direct-spectrum baseline: features = preprocessed spectra (L2-normalized)."""
    def __init__(self, grid, name):
        super().__init__(name=name, branch="direct", grid=grid); self.n_features = len(grid)
    def transform(self, X, modality=None):
        X = np.nan_to_num(np.atleast_2d(X)); n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (n + 1e-12)


# ── candidate registry: fit_fn(Xtr, meta_tr, grid, seed) -> Representation; preproc key ──
def _enc_cfg(**kw):
    def fit(Xtr, meta_tr, grid, seed, Xval=None, meta_val=None):
        cfg = TR.EncoderConfig(seed=seed, **kw)
        return TR.train_encoder(cfg, Xtr, meta_tr, grid, Xval, meta_val)
    return fit

CANDIDATES = [
    dict(name="direct_SNV", branch="direct", preproc="SNV", fit=lambda X,m,g,s,**k: IdentityRep(g,"direct_SNV")),
    dict(name="direct_L2", branch="direct", preproc="L2", fit=lambda X,m,g,s,**k: IdentityRep(g,"direct_L2")),
    dict(name="direct_deriv", branch="direct", preproc="DERIV", fit=lambda X,m,g,s,**k: IdentityRep(g,"direct_deriv")),
    dict(name="I1_regions", branch="interpretable", preproc="SNV",
         fit=lambda X,m,g,s,**k: regions.fit_regions(X, g, 32)),
    dict(name="I2_wavelets", branch="interpretable", preproc="SNV",
         fit=lambda X,m,g,s,**k: wavelets.fit_wavelets(X, g)),
    dict(name="I3_dictionary", branch="interpretable", preproc="SNV",
         fit=lambda X,m,g,s,**k: dictionary.fit_dictionary(X, g, 24, 1.0, s)),
    dict(name="I4_nmf", branch="interpretable", preproc="L2",
         fit=lambda X,m,g,s,**k: basis.fit_nmf_basis(X, g, 16, s)),
    dict(name="E1_shared_supcon", branch="encoder", preproc="SNV",
         fit=_enc_cfg(name="E1_shared_supcon", arch="shared", w_supcon=1, w_infonce=0, cross_modal=False)),
    dict(name="E2_dual_supcon_infonce", branch="encoder", preproc="SNV",
         fit=_enc_cfg(name="E2_dual_supcon_infonce", arch="dual", w_supcon=1, w_infonce=1, cross_modal=True)),
    dict(name="E2_dual_supcon_infonce_vicreg", branch="encoder", preproc="SNV",
         fit=_enc_cfg(name="E2_dual_supcon_infonce_vicreg", arch="dual", w_supcon=1, w_infonce=1, w_vicreg=1.0, cross_modal=True)),
    dict(name="E2_dual_triplet_infonce", branch="encoder", preproc="SNV",
         fit=_enc_cfg(name="E2_dual_triplet_infonce", arch="dual", w_supcon=0, w_triplet=1, w_infonce=1, cross_modal=True)),
    dict(name="E3_dual_within_only", branch="encoder", preproc="SNV",
         fit=_enc_cfg(name="E3_dual_within_only", arch="dual", w_supcon=1, w_infonce=0, cross_modal=False)),
]
PRIMARY_ENCODER = "E2_dual_supcon_infonce"


def get_src(datasets, preproc):
    return datasets[preproc]


def fit_and_embed(cand, datasets, train_ids, eval_ids, seed):
    src = get_src(datasets, cand["preproc"])
    tr = src.meta.spectrum_id.isin(train_ids).values
    rep = cand["fit"](src.X[tr], src.meta[tr].reset_index(drop=True), src.grid, seed)
    ev = src.meta.spectrum_id.isin(eval_ids).values
    Xe, me = src.X[ev], src.meta[ev].reset_index(drop=True)
    F = rep.transform(Xe, me.modality.values)
    return rep, F, me


def eval_heldout_B(cand, datasets, bfolds, seed=SEED):
    """Pooled held-out matched-analyte cross-modal retrieval (Split B)."""
    fold_feats = []
    for fold in bfolds:
        _, F, me = fit_and_embed(cand, datasets, fold["train"], fold["test"], seed)
        fold_feats.append((F, me))
    r = EV.pooled_heldout_retrieval(fold_feats, n_perm=1500, seed=seed)
    if not r.get("insufficient"):
        r["mrr_bootstrap_ci95"] = EV.bootstrap_ci_mrr(fold_feats, n_boot=800, seed=seed)
    return r


def eval_within_C(cand, datasets, cfolds, seed=SEED):
    """Held-out replicate-group within-modality chemistry retention (Split C)."""
    rates = {"raman": [], "sers": []}; aris = {"raman": [], "sers": []}
    for fold in cfolds:
        _, F, me = fit_and_embed(cand, datasets, fold["train"], fold["test"], seed)
        wm = EV.within_modality_chem(F, me, seed=seed)
        for mod in ("raman", "sers"):
            if mod in wm:
                rates[mod].append(wm[mod]["nn_same_analyte_rate"])
                if wm[mod]["ari_analyte"] is not None: aris[mod].append(wm[mod]["ari_analyte"])
    return {mod: {"nn_same_analyte_rate": float(np.mean(rates[mod])) if rates[mod] else None,
                  "ari_analyte": float(np.mean(aris[mod])) if aris[mod] else None,
                  "n_folds": len(rates[mod])} for mod in ("raman", "sers")}


def eval_family_A(cand, datasets, afolds, seed=SEED):
    """Held-out-analyte family-neighborhood purity (Split A)."""
    purities = []
    for fold in afolds:
        _, F, me = fit_and_embed(cand, datasets, fold["train"], fold["test"], seed)
        fn = EV.family_neighborhood(F, me, k=5)
        if "family_knn_purity" in fn:
            purities.append((fn["family_knn_purity"], fn["chance_purity"]))
    if not purities:
        return {"skipped": True}
    p = np.array(purities)
    return {"family_knn_purity": float(p[:, 0].mean()), "chance_purity": float(p[:, 1].mean()),
            "n_folds": len(purities)}


def insample_diag(cand, datasets, seed=SEED):
    """In-sample (full-data) descriptive diagnostics: leakage, collapse, shortcuts,
    family neighborhood, uncertainty. Labeled in-sample (NOT a generalization claim)."""
    src = get_src(datasets, cand["preproc"])
    ids = src.meta.spectrum_id.tolist()
    rep, F, me = fit_and_embed(cand, datasets, ids, ids, seed)
    leak = EV.leakage_metrics(F, me, seed=seed)
    coll = EV.collapse_diagnostics(F, analytes=me.analyte.values)
    short = EV.signal_stat_correlation(F, src.X)
    fam = EV.family_neighborhood(F, me, k=5)
    xm = EV.cross_modal_metrics(F, me, n_perm=1500, seed=seed)
    return rep, {"leakage": leak, "collapse": coll, "signal_shortcut": short,
                 "family_neighborhood": fam,
                 "insample_cross_modal": {k: xm.get(k) for k in
                                          ("top_k", "mrr", "reciprocal_nn_rate", "matched_minus_unmatched")}
                 if not xm.get("insufficient") else xm}


def main():
    t0 = time.time()
    log("building datasets (SNV, L2, DERIV) …")
    datasets = {"SNV": D.build("A2_asls_savgol_snv"), "L2": D.build("A1_asls_savgol_l2"),
                "DERIV": D.build("A3_deriv_l2")}
    d = datasets["SNV"]
    (TAB/"stage_b_dataset_card.json").write_text(json.dumps(D.dataset_card(d), indent=2, default=float))

    log("building + verifying splits …")
    sm = SP.make_all_splits(d, k=5, seed=SEED)
    leak_chk = SP.verify_no_leakage(sm, d.meta)
    assert all(v["ok"] for v in leak_chk.values()), f"SPLIT LEAKAGE: {leak_chk}"
    (CFG/"stage_b_splits.json").write_text(json.dumps(sm, indent=2, default=float))
    (TAB/"stage_b_split_leakage_checks.json").write_text(json.dumps(leak_chk, indent=2))
    bfolds = sm["splits"]["B_held_out_matched_pairs"]["folds"]
    cfolds = sm["splits"]["C_replicate_group_holdout"]["folds"]
    afolds = sm["splits"]["A_held_out_analytes"]["folds"]

    log("augmentation validity audit …")
    aud = augmentation_audit(d.X, d.grid, AugConfig(), seed=SEED, n_examples=6)
    (TAB/"augmentation_audit.json").write_text(json.dumps(aud, indent=2, default=float))

    # ── Split B (primary): held-out matched-analyte cross-modal retrieval ──
    heldB, within, family, insample = {}, {}, {}, {}
    for cand in CANDIDATES:
        log(f"Split B  {cand['name']} …")
        heldB[cand["name"]] = eval_heldout_B(cand, datasets, bfolds)
    for cand in CANDIDATES:
        log(f"Split C  {cand['name']} …")
        within[cand["name"]] = eval_within_C(cand, datasets, cfolds)
    for cand in CANDIDATES:  # family retrieval: interpretable+direct cheap; encoders too (5 folds)
        if cand["branch"] in ("direct", "interpretable") or cand["name"] in (PRIMARY_ENCODER, "E1_shared_supcon"):
            log(f"Split A  {cand['name']} …")
            family[cand["name"]] = eval_family_A(cand, datasets, afolds)

    # ── in-sample diagnostics + serialize reps ──
    reps = {}
    for cand in CANDIDATES:
        log(f"in-sample diag  {cand['name']} …")
        rep, diag = insample_diag(cand, datasets)
        insample[cand["name"]] = diag; reps[cand["name"]] = rep
        if cand["branch"] in ("interpretable", "encoder"):
            SER.save_representation(rep, MOD)

    # ── seed stability for the two key encoders on Split B ──
    log("seed stability (E1, E2) …")
    seed_stab = {}
    for name in ("E1_shared_supcon", PRIMARY_ENCODER):
        cand = next(c for c in CANDIDATES if c["name"] == name)
        mrrs, top1s = [], []
        for s in (0, 1, 2):
            r = eval_heldout_B(cand, datasets, bfolds, seed=s)
            mrrs.append(r["mrr"]); top1s.append(r["top_k"]["top1"])
        seed_stab[name] = {"mrr_mean": float(np.mean(mrrs)), "mrr_std": float(np.std(mrrs)),
                           "top1_mean": float(np.mean(top1s)), "top1_std": float(np.std(top1s)),
                           "seeds": [0, 1, 2], "mrr_per_seed": mrrs}
    # (seed stability kept at 3 seeds for the two key encoders — the decisive stability signal)

    # ── E4 hybrid (best encoder + best interpretable by Split-B MRR) ──
    def mrr_of(n): return heldB[n].get("mrr", 0.0)
    best_enc = max([c["name"] for c in CANDIDATES if c["branch"] == "encoder"], key=mrr_of)
    best_int = max([c["name"] for c in CANDIDATES if c["branch"] == "interpretable"], key=mrr_of)
    log(f"hybrid E4 from {best_enc} + {best_int} …")
    hybrid_cand = dict(name="E4_hybrid", branch="hybrid", preproc="SNV",
                       fit=lambda X, m, g, s, _be=best_enc, _bi=best_int: HY.HybridRepresentation(
                           next(c for c in CANDIDATES if c["name"] == _be)["fit"](X, m, g, s),
                           next(c for c in CANDIDATES if c["name"] == _bi)["fit"](X, m, g, s)))
    heldB["E4_hybrid"] = eval_heldout_B(hybrid_cand, datasets, bfolds)
    within["E4_hybrid"] = eval_within_C(hybrid_cand, datasets, cfolds)

    # ── encoder interpretability (attribution stability) + sparse probe for primary ──
    log("encoder interpretability …")
    interp = {}
    prim = reps[PRIMARY_ENCODER]
    Xby = {a: d.X[(d.meta.analyte == a) & (d.meta.modality == "raman")].__array__()
           for a in d.matched_analytes[:12]}
    Xby = {a: v for a, v in Xby.items() if len(v) >= 2}
    interp["attribution_stability_raman"] = IN.attribution_stability(prim, Xby, "raman", method="occlusion")
    # sparse linear probe: primary embedding -> I1 region activations (interpretable evidence)
    reg = reps["I1_regions"]; Fz = prim.transform(d.X, d.meta.modality.values); Treg = reg.transform(d.X)
    ntr = int(0.7 * len(d.X)); rng = np.random.default_rng(0); perm = rng.permutation(len(d.X))
    tr_i, te_i = perm[:ntr], perm[ntr:]
    interp["sparse_probe_embed_to_regions"] = PJ.sparse_linear_probe(Fz[tr_i], Treg[tr_i], Fz[te_i], Treg[te_i])

    # ── uncertainty (primary encoder, in-sample support) ──
    log("uncertainty signals …")
    Fz_all = prim.transform(d.X, d.meta.modality.values)
    C, cm = EV.feature_centroids(Fz_all, d.meta)
    Fr, ar = C[cm.modality == "raman"], cm[cm.modality == "raman"].analyte.values
    Fs, as_ = C[cm.modality == "sers"], cm[cm.modality == "sers"].analyte.values
    xagr = UQ.cross_modal_agreement(Fr, ar, Fs, as_)
    unc = {"cross_modal_agreement_mean": float(np.mean(list(xagr.values()))) if xagr else None,
           "cross_modal_agreement_min": float(np.min(list(xagr.values()))) if xagr else None,
           "n_matched_refs": len(xagr)}

    results = {"held_out_B": heldB, "within_modality_C": within, "family_A": family,
               "insample": insample, "seed_stability": seed_stab,
               "interpretability": interp, "uncertainty": unc,
               "hybrid_from": {"encoder": best_enc, "interpretable": best_int},
               "runtime_s": round(time.time() - t0, 1)}
    (TAB/"stage_b_results.json").write_text(json.dumps(results, indent=2, default=float))
    log(f"results written. runtime {results['runtime_s']}s")

    # scorecard + decision + figures in a companion module (imported here to keep one entry)
    from stage_b_decide import scorecard_and_decision, make_figures
    sc, dec = scorecard_and_decision(results, CANDIDATES, PRIMARY_ENCODER)
    (TAB/"stage_b_scorecard.json").write_text(json.dumps(sc, indent=2, default=float))
    (TAB/"stage_b_decision.json").write_text(json.dumps(dec, indent=2, default=float))
    make_figures(results, reps, d, datasets, sc, FIG)
    log(f"DECISION: {dec['outcome']} — {dec['headline']}")
    return results, sc, dec


if __name__ == "__main__":
    main()
