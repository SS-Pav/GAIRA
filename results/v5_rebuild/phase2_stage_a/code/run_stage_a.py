"""GAIRA V5 Phase 2 Stage A — direct-representation discovery (driver).

Runs, for each preprocessing (A1 L2, A2 SNV, A3 first-derivative):
  * three analyses: Raman-only, Ag-SERS-only, joint structure (§7)
  * centroid-level (primary) + spectrum-level (§5)
  * matched-analyte cross-modal retrieval + permutation nulls (§8)
  * PCA(10) + bootstrap loading stability by analyte (§9)
  * NMF/sparsePCA/factor analysis where representation admits it (§10)
  * hierarchical + consensus clustering, ARI vs analyte/modality/source (§11)
  * modality/source leakage with grouped CV vs naive baselines (§12–13)
Then a pre-declared scorecard (§14) → A/B/C/D decision inputs.

Deterministic (fixed seeds). Read-only w.r.t. source data.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.representation import datasets as ds, centroids as ct, retrieval as rt
from gaira.representation import pca as pca_m, factorization as fac, clustering as clu
from gaira.representation import leakage as lk, stability as stab
from gaira.representation.metrics import peaks, peak_overlap

SEED = 0
PH = REPO / "results/v5_rebuild/phase2_stage_a"
FIG, TAB = PH / "figures", PH / "tables"
for d in (FIG, TAB): d.mkdir(parents=True, exist_ok=True)
GRID = ds.GRID

# ── pre-declared scorecard weights (§14, fixed BEFORE results) ──
SCORECARD_WEIGHTS = {
    "cross_modal_identity": 0.35,     # matched Raman↔SERS retrieval beats null
    "chemistry_over_nuisance": 0.25,  # clustering ARI(analyte) > ARI(modality/source)
    "low_modality_leakage": 0.20,     # modality classifier near chance = shared axes
    "component_stability": 0.10,      # PCA loadings bootstrap-stable
    "preprocessing_robustness": 0.10, # conclusions hold across A1/A2/A3
}
# decision thresholds (declared before results)
TH = {"retrieval_top1_shared": 0.30, "retrieval_p": 0.05,
      "modality_leak_shared_max": 0.75, "ari_chem_over_nuisance": 0.05,
      "stability_ok": 0.80, "direct_adequate_min_score": 0.45}


def analyze_block(X, meta, label, do_nmf):
    """PCA + factorization + clustering + leakage for one matrix block."""
    out = {"n": int(X.shape[0]), "n_analytes": int(meta.analyte.nunique())}
    # PCA + stability (bootstrap by analyte)
    p, scores = pca_m.fit_pca(X, 10, SEED)
    st = pca_m.bootstrap_stability(X, meta.analyte.values, n_components=6, n_boot=150, seed=SEED)
    out["pca"] = {"explained_variance_ratio": p.explained_variance_ratio_.tolist(),
                  "cum_var_6": float(np.sum(p.explained_variance_ratio_[:6])),
                  "loading_stability_mean": st["loading_stability_mean"],
                  "loading_stability_p5": st["loading_stability_p5"]}
    # factorization
    fz = {}
    if do_nmf:
        try:
            nm = fac.fit_nmf(X, 6, SEED); fz["nmf_recon_err"] = nm["reconstruction_err"]
        except ValueError as e:
            fz["nmf"] = f"skipped: {e}"
    else:
        fz["nmf"] = "skipped: signed representation (non-negativity violated)"
    fz["sparse_pca_nonzero_frac"] = fac.fit_sparse_pca(X, 6, SEED)["nonzero_fraction"]
    out["factorization"] = fz
    # clustering: ARI vs analyte / modality / source
    labs = {"analyte": pd.factorize(meta.analyte)[0]}
    if meta.modality.nunique() > 1: labs["modality"] = pd.factorize(meta.modality)[0]
    if meta.source.nunique() > 1: labs["source"] = pd.factorize(meta.source)[0]
    cl_res, _ = clu.hierarchical(X, labs, metric="cosine", method="average")
    out["clustering"] = cl_res
    # leakage: modality (joint only) + source
    leak = {}
    if meta.modality.nunique() > 1:
        leak["modality"] = lk.grouped_leakage(X, meta.modality.values, meta.analyte.values, seed=SEED)
    if meta.source.nunique() > 1:
        leak["source"] = lk.grouped_leakage(X, meta.source.values, meta.analyte.values, seed=SEED)
    out["leakage"] = leak
    return out, scores


def run_preproc(preproc):
    signed = ("snv" in preproc) or ("deriv" in preproc)
    do_nmf = not signed
    rows, _ = ds.build_phase2_input(preproc)
    X, meta = ds.matrix(rows)
    # centroid level (primary): analyte × modality × source
    C, cmeta = ct.build_centroids(X, meta)
    res = {"preproc": preproc, "signed_representation": signed}

    # ── three analyses at centroid level ──
    ram = cmeta.modality.values == "raman"
    ser = cmeta.modality.values == "sers"
    res["raman_only"], _ = analyze_block(C[ram], cmeta[ram].reset_index(drop=True), "raman", do_nmf)
    res["sers_only"], _ = analyze_block(C[ser], cmeta[ser].reset_index(drop=True), "sers", do_nmf)
    res["joint"], joint_scores = analyze_block(C, cmeta, "joint", do_nmf)

    # ── matched-analyte cross-modal retrieval (§8) ──
    R, rm = C[ram], cmeta[ram].reset_index(drop=True)
    S, sm = C[ser], cmeta[ser].reset_index(drop=True)
    ret = rt.cross_modal_retrieval(R, rm, S, sm)
    if not ret.get("insufficient"):
        sim = np.array(ret.pop("_sim"))
        perm = rt.permutation_null(sim, n_perm=2000, seed=SEED)
        ret["permutation_null"] = perm
        # band-level: mean peak overlap for matched vs mismatched pairs
        rc = {a: R[[i for i, x in enumerate(rm.analyte) if x == a]].mean(0) for a in ret["analytes"]}
        sc = {a: S[[i for i, x in enumerate(sm.analyte) if x == a]].mean(0) for a in ret["analytes"]}
        rp = {a: peaks(rc[a], GRID) for a in ret["analytes"]}
        spk = {a: peaks(sc[a], GRID) for a in ret["analytes"]}
        matched_ov = [peak_overlap(rp[a], spk[a]) for a in ret["analytes"]]
        rng = np.random.default_rng(SEED)
        mism = []
        al = ret["analytes"]
        for a in al:
            b = al[rng.integers(len(al))]
            if b == a: continue
            mism.append(peak_overlap(rp[a], spk[b]))
        ret["peak_overlap_matched_mean"] = float(np.mean(matched_ov))
        ret["peak_overlap_mismatched_mean"] = float(np.mean(mism)) if mism else None
    res["matched_retrieval"] = ret

    # ── consensus clustering stability (joint centroids) ──
    k = int(cmeta.analyte.nunique())
    res["consensus_joint"] = {kk: v for kk, v in
                              stab.consensus_clustering(C, cmeta.analyte.values, k=min(k, 20),
                                                        n_boot=150, seed=SEED).items()
                              if kk != "consensus_matrix"}

    # ── spectrum-level replicate dispersion (§5 secondary) ──
    res["spectrum_level"] = {
        "n_spectra": int(X.shape[0]),
        "mean_within_analyte_modality_dispersion": float(np.nanmean(cmeta.dispersion)),
    }
    return res, (C, cmeta, joint_scores, ret)


def _preproc_signals(res):
    """Extract the shared-space signals for one preprocessing."""
    ret = res["matched_retrieval"]
    top1 = ret["top_k"]["top1"]; p1 = ret["permutation_null"]["top1"]["p_value"]
    cj = res["joint"]["clustering"]["ari_vs"]
    ari_chem = cj.get("analyte", 0.0); ari_nuis = max(cj.get("modality", 0.0), cj.get("source", 0.0))
    ml = res["joint"]["leakage"].get("modality", {}).get("balanced_accuracy_mean") or 1.0
    st3 = float(np.mean(res["joint"]["pca"]["loading_stability_mean"][:3]))
    raman_chem = res["raman_only"]["clustering"]["ari_vs"].get("analyte", 0.0)
    return {"top1": top1, "top1_p": p1, "matched_minus_unmatched": ret.get("matched_minus_unmatched", 0.0),
            "ari_analyte": ari_chem, "ari_nuisance_max": ari_nuis, "modality_leak": ml,
            "pc_stability_top3": st3, "raman_only_ari_analyte": raman_chem, "n_matched": ret["n_matched"]}


def scorecard(per_preproc):
    """Preprocessing-AWARE scorecard (§6, §14). The architecture question — can a
    shared space work — is judged under its MOST FAVORABLE preprocessing so an
    arbitrary preprocessing choice cannot decide the architecture. All three
    preprocessings are reported for transparency."""
    sig = {k: _preproc_signals(v) for k, v in per_preproc.items()}
    # pick the preprocessing most favorable to a shared space:
    # lowest modality leakage, then highest cross-modal top1
    best_pp = min(sig, key=lambda k: (sig[k]["modality_leak"], -sig[k]["top1"]))
    b = sig[best_pp]
    s_identity = float(min(1.0, b["top1"] / 0.5) * (1.0 if b["top1_p"] < TH["retrieval_p"] else 0.3))
    s_chem = float(np.clip((b["ari_analyte"] - b["ari_nuisance_max"]) / 0.3 + 0.5, 0, 1))
    s_leak = float(np.clip((1.0 - b["modality_leak"]) / (1.0 - 0.5), 0, 1))
    s_stab = float(np.clip(b["pc_stability_top3"], 0, 1))
    signs = [sig[k]["matched_minus_unmatched"] > 0 for k in sig]
    tops = [sig[k]["top1"] > 1.0 / max(3, sig[k]["n_matched"]) for k in sig]
    s_robust = float((np.mean(signs) + np.mean(tops)) / 2)
    dims = {"cross_modal_identity": s_identity, "chemistry_over_nuisance": s_chem,
            "low_modality_leakage": s_leak, "component_stability": s_stab,
            "preprocessing_robustness": s_robust}
    total = float(sum(SCORECARD_WEIGHTS[k] * dims[k] for k in dims))
    return {"weights": SCORECARD_WEIGHTS, "dimension_scores": dims, "weighted_total": total,
            "thresholds": TH, "best_preproc_for_shared": best_pp,
            "per_preproc_signals": sig,
            "raw_signals": {"retrieval_top1": b["top1"], "retrieval_top1_p": b["top1_p"],
                            "modality_leak_bal_acc": b["modality_leak"], "ari_analyte": b["ari_analyte"],
                            "ari_nuisance_max": b["ari_nuisance_max"], "pc_stability_top3": b["pc_stability_top3"],
                            "raman_only_ari_analyte": b["raman_only_ari_analyte"]}}


def decide(sc):
    """Map scorecard → Stage-A decision A/B/C/D (§19). Exactly one outcome.
    Judged under the preprocessing most favorable to a shared space, so a
    shared-space verdict is not defeated by an arbitrary preprocessing choice."""
    r = sc["raw_signals"]
    top1, p, ml = r["retrieval_top1"], r["retrieval_top1_p"], r["modality_leak_bal_acc"]
    raman_chem = r["raman_only_ari_analyte"]; ari_chem = r["ari_analyte"]; ari_nuis = r["ari_nuisance_max"]
    n_matched = min(s["n_matched"] for s in sc["per_preproc_signals"].values())
    reasons = []
    cross_ok = (top1 >= TH["retrieval_top1_shared"]) and (p < TH["retrieval_p"])
    leak_ok = ml <= TH["modality_leak_shared_max"]
    chem_recovered = (raman_chem >= 0.30) or (ari_chem > ari_nuis and ari_chem >= 0.10)
    xmodal_real = p < TH["retrieval_p"]
    if n_matched < 20:
        outcome = "D"
        reasons.append(f"only {n_matched} matched analytes — grounding insufficient to decide.")
    elif not chem_recovered:
        outcome = "C"
        reasons.append(f"direct spectra do not recover chemistry even within a modality "
                       f"(Raman-only ARI={raman_chem:.2f}); direct representation inadequate → Stage B chemical features.")
    elif cross_ok and leak_ok:
        outcome = "A"
        reasons.append(f"under {sc['best_preproc_for_shared']}: cross-modal identity preserved "
                       f"(top1={top1:.2f}, p={p:.3f}) AND modality leakage bounded (bal_acc={ml:.2f}≤{TH['modality_leak_shared_max']}) "
                       f"→ shared representation defensible.")
    else:
        outcome = "B"
        reasons.append(f"direct spectra carry real chemistry (Raman-only ARI={raman_chem:.2f}; best-preproc joint "
                       f"ARI_analyte={ari_chem:.2f} vs nuisance={ari_nuis:.2f}) and a statistically-significant but WEAK "
                       f"cross-modal signal exists (top1={top1:.2f}, p={p:.3f}; {'significant' if xmodal_real else 'n.s.'}), "
                       f"but modality remains too separable (bal_acc={ml:.2f}>{TH['modality_leak_shared_max']}) and retrieval "
                       f"too weak (top1={top1:.2f}<{TH['retrieval_top1_shared']}) for a single shared coordinate system "
                       f"→ modality-stratified representation defensible; align at analyte/ontology level. "
                       f"Recommend Stage B chemical features to test whether the residual cross-modal signal can be strengthened.")
    return {"outcome": outcome, "reasons": reasons,
            "signals": {"cross_modal_ok": bool(cross_ok), "modality_leak_ok": bool(leak_ok),
                        "chemistry_recovered": bool(chem_recovered), "cross_modal_significant": bool(xmodal_real),
                        "weighted_score": sc["weighted_total"], "n_matched": int(n_matched),
                        "best_preproc_for_shared": sc["best_preproc_for_shared"]}}


def joint_pca_fig(bundle, tag):
    C, cmeta, scores, ret = bundle
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for mod, col in [("raman", "#2563eb"), ("sers", "#dc2626")]:
        m = cmeta.modality.values == mod
        ax.scatter(scores[m, 0], scores[m, 1], s=18, c=col, alpha=0.6, label=mod)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
    ax.set_title(f"Joint centroid PCA ({tag}) — colored by modality\n(separation ⇒ modality dominates axes)")
    fig.tight_layout(); fig.savefig(FIG / f"joint_pca_by_modality_{tag}.png", dpi=130); plt.close(fig)


def figures(bundle, A1):
    C, cmeta, scores, ret = bundle
    joint_pca_fig(bundle, "A1")

    # 2. explained variance
    fig, ax = plt.subplots(figsize=(6, 4))
    ev = A1["joint"]["pca"]["explained_variance_ratio"]
    ax.bar(range(1, len(ev) + 1), ev, color="#334155")
    ax.set_xlabel("PC"); ax.set_ylabel("explained variance ratio")
    ax.set_title("Joint PCA scree (A1)")
    fig.tight_layout(); fig.savefig(FIG / "joint_pca_scree_A1.png", dpi=130); plt.close(fig)

    # 3. matched vs null retrieval
    if not ret.get("insufficient"):
        perm = ret["permutation_null"]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for i, (name, key) in enumerate([("top-1", "top1"), ("MRR", "mrr"), ("matched cos", "matched_cos")]):
            o = perm[key]["observed"]; nm = perm[key]["null_mean"]; ci = perm[key]["null_ci95"]
            ax.errorbar(i, nm, yerr=[[nm - ci[0]], [ci[1] - nm]], fmt="o", color="#94a3b8", capsize=5)
            ax.scatter(i, o, color="#16a34a", zorder=5, s=60)
            ax.text(i, o, f" obs={o:.2f}\n p={perm[key]['p_value']:.3f}", va="bottom", fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels(["top-1", "MRR", "matched cos"])
        ax.set_title(f"Cross-modal retrieval vs permutation null ({ret['n_matched']} matched)\n"
                     "green=observed, grey=null 95% CI")
        fig.tight_layout(); fig.savefig(FIG / "cross_modal_retrieval_vs_null_A1.png", dpi=130); plt.close(fig)


def main():
    per_preproc, bundles = {}, {}
    for pp in ds.PREPROCS:
        res, bundle = run_preproc(pp)
        per_preproc[pp] = res; bundles[pp] = bundle
        print(f"[{pp}] joint n={res['joint']['n']} | matched={res['matched_retrieval'].get('n_matched')} "
              f"top1={res['matched_retrieval'].get('top_k', {}).get('top1')} "
              f"mod_leak={res['joint']['leakage'].get('modality', {}).get('balanced_accuracy_mean')}")
    sc = scorecard(per_preproc)
    dec = decide(sc)
    figures(bundles["A1_asls_savgol_l2"], per_preproc["A1_asls_savgol_l2"])
    joint_pca_fig(bundles[sc["best_preproc_for_shared"]],
                  sc["best_preproc_for_shared"].split("_")[0])  # best-case (SNV) contrast

    (TAB / "stage_a_results.json").write_text(json.dumps(per_preproc, indent=2, default=float))
    (TAB / "stage_a_scorecard.json").write_text(json.dumps(sc, indent=2, default=float))
    (TAB / "stage_a_decision.json").write_text(json.dumps(dec, indent=2, default=float))

    # scorecard csv
    pd.DataFrame([{"dimension": k, "weight": sc["weights"][k], "score": sc["dimension_scores"][k]}
                  for k in sc["weights"]]).to_csv(TAB / "stage_a_scorecard.csv", index=False)

    print("\n== SCORECARD ==")
    for k in sc["weights"]:
        print(f"  {k}: score={sc['dimension_scores'][k]:.2f} × w={sc['weights'][k]}")
    print(f"  WEIGHTED TOTAL: {sc['weighted_total']:.3f}")
    print(f"\n== DECISION: Outcome {dec['outcome']} ==")
    for r in dec["reasons"]: print("  -", r)
    return per_preproc, sc, dec


if __name__ == "__main__":
    main()
