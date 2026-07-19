"""Stage B model selection (§15-16) + figures (§17). Imported by run_stage_b.py.

Model selection is a Pareto decision with pre-declared gates — NOT a single weighted
metric. The primary axis is HELD-OUT (Split B) matched-analyte cross-modal retrieval,
compared apples-to-apples against the direct_SNV held-out baseline.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt

# pre-declared gate thresholds (fixed before inspecting results)
GATES = {
    "material_top1_delta": 0.03,     # candidate top1 must exceed baseline by >= this
    "seed_mrr_std_max": 0.03,        # encoder MRR std across seeds for "stable"
    "collapse_dup_max": 0.10,        # cross-analyte duplicate fraction ceiling
    "collapse_effrank_min": 4.0,     # effective rank floor (latent 16)
    "modality_leak_ref": 0.83,       # Stage A SNV in-sample modality bal-acc
    "shortcut_corr_max": 0.6,        # |corr| of embedding with a trivial signal stat
    "within_retention_frac": 0.8,    # keep >= 80% of direct's within-modality NN rate
}
SCORE_WEIGHTS = {  # transparency scorecard (reporting only; decision uses gates)
    "held_out_cross_modal": 0.35, "within_modality_retention": 0.20,
    "nuisance_control": 0.15, "stability": 0.15, "interpretability": 0.10,
    "simplicity": 0.05,
}


def _mrr(h): return h.get("mrr", 0.0) if not h.get("insufficient") else 0.0
def _top1(h): return h.get("top_k", {}).get("top1", 0.0) if not h.get("insufficient") else 0.0
def _ci(h): return h.get("mrr_bootstrap_ci95") or [0.0, 0.0]


def _within_nn(within, name):
    w = within.get(name, {})
    vals = [w.get(m, {}).get("nn_same_analyte_rate") for m in ("raman", "sers")]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else 0.0


def scorecard_and_decision(results, candidates, primary_encoder):
    heldB, within, insample, seed_stab = (results["held_out_B"], results["within_modality_C"],
                                          results["insample"], results["seed_stability"])
    base = heldB["direct_SNV"]
    base_mrr, base_top1 = _mrr(base), _top1(base)
    branch = {c["name"]: c["branch"] for c in candidates}
    branch["E4_hybrid"] = "hybrid"

    rows = {}
    for name, h in heldB.items():
        ins = insample.get(name, {})
        coll = ins.get("collapse", {})
        leak = ins.get("leakage", {}).get("modality", {}).get("balanced_accuracy_mean")
        short = ins.get("signal_shortcut", {})
        rows[name] = {
            "branch": branch.get(name, "?"),
            "heldout_top1": _top1(h), "heldout_mrr": _mrr(h), "heldout_mrr_ci": _ci(h),
            "heldout_perm_top1_p": h.get("perm_top1_p"), "chance_top1": h.get("chance_top1"),
            "within_nn": _within_nn(within, name),
            "modality_leak": leak,
            "cross_analyte_dup": coll.get("cross_analyte_duplicate_fraction"),
            "effective_rank": coll.get("effective_rank"),
            "max_shortcut_corr": max([abs(v) for v in short.values()], default=None),
            "n_params": (ins.get("collapse", {}) and None),
        }
    # gates per candidate (vs direct_SNV baseline)
    for name, r in rows.items():
        beats = (r["heldout_top1"] - base_top1) >= GATES["material_top1_delta"]
        ci_above = r["heldout_mrr_ci"][0] > base_mrr
        no_collapse = ((r["cross_analyte_dup"] is None or r["cross_analyte_dup"] <= GATES["collapse_dup_max"])
                       and (r["effective_rank"] is None or r["effective_rank"] >= GATES["collapse_effrank_min"]))
        within_ok = r["within_nn"] >= GATES["within_retention_frac"] * _within_nn(within, "direct_SNV")
        no_shortcut = (r["max_shortcut_corr"] is None) or (r["max_shortcut_corr"] <= GATES["shortcut_corr_max"])
        stable = True
        if name in seed_stab:
            stable = seed_stab[name]["mrr_std"] <= GATES["seed_mrr_std_max"]
        r["gates"] = {"material_cross_modal": bool(beats), "mrr_ci_above_baseline": bool(ci_above),
                      "no_collapse": bool(no_collapse), "within_retention": bool(within_ok),
                      "no_source_shortcut": bool(no_shortcut), "seed_stable": bool(stable)}
        r["passes_cross_modal_gate"] = bool(beats and ci_above)

    # rank by held-out MRR
    order = sorted(rows, key=lambda n: rows[n]["heldout_mrr"], reverse=True)
    best = order[0]
    best_enc = max((n for n in rows if rows[n]["branch"] == "encoder"), key=lambda n: rows[n]["heldout_mrr"])
    best_int = max((n for n in rows if rows[n]["branch"] == "interpretable"), key=lambda n: rows[n]["heldout_mrr"])
    hybrid_mrr = rows.get("E4_hybrid", {}).get("heldout_mrr", 0.0)

    # ── decision tree (B1-B5) ──
    def improves(name):
        r = rows[name]; return r["gates"]["material_cross_modal"] and r["gates"]["mrr_ci_above_baseline"]
    enc_stable = seed_stab.get(primary_encoder, {}).get("mrr_std", 1.0) <= GATES["seed_mrr_std_max"]

    reasons = []
    any_beats = any(improves(n) for n in rows if n != "direct_SNV")
    if not any_beats:
        # nothing materially beats direct held-out cross-modal
        # is within-modality chemistry fine? then retain modality-stratified (B4)
        outcome = "B4"
        reasons.append(f"No representation materially beats the direct_SNV held-out cross-modal baseline "
                       f"(top1={base_top1:.3f}, MRR={base_mrr:.3f}); best held-out MRR = {rows[best]['heldout_mrr']:.3f} ({best}). "
                       f"Retain modality-stratified representations; align at ontology level.")
        # but if encoders were the only hope and are unstable/insufficient, flag B5 alternative
        if not enc_stable:
            reasons.append(f"Encoders are seed-unstable (primary MRR std={seed_stab.get(primary_encoder,{}).get('mrr_std')}); "
                           "encoder conclusions are corpus-limited (see B5 caveat).")
    else:
        # something beats direct. Which branch and is it stable + auditable?
        if improves(best_enc) and rows[best_enc]["gates"]["no_collapse"] and enc_stable and improves(best) and rows[best]["branch"] == "encoder":
            outcome = "B2"
            reasons.append(f"Encoder {best_enc} materially and reproducibly beats direct held-out "
                           f"(top1 {rows[best_enc]['heldout_top1']:.3f} vs {base_top1:.3f}) without collapse and stable across seeds.")
        elif rows.get("E4_hybrid", {}).get("passes_cross_modal_gate") and hybrid_mrr > max(rows[best_enc]["heldout_mrr"], rows[best_int]["heldout_mrr"]):
            outcome = "B3"
            reasons.append(f"Hybrid E4 gives a reproducible Pareto improvement over both branches (MRR {hybrid_mrr:.3f}).")
        elif improves(best_int) and rows[best_int]["branch"] == "interpretable":
            outcome = "B1"
            reasons.append(f"Interpretable {best_int} offers the best balance (held-out MRR {rows[best_int]['heldout_mrr']:.3f}, "
                           f"auditable, no training instability).")
        else:
            outcome = "B5"
            reasons.append("A candidate beats direct but only via an unstable/uninterpretable path; "
                           "corpus insufficient for a firm encoder conclusion.")

    # transparency scorecard (reporting)
    def norm(x, lo, hi): return float(np.clip((x - lo) / (hi - lo + 1e-9), 0, 1))
    sc_dims = {}
    for name, r in rows.items():
        leak = r["modality_leak"] if r["modality_leak"] is not None else 1.0
        sc_dims[name] = {
            "held_out_cross_modal": norm(r["heldout_mrr"], base_mrr, base_mrr + 0.25),
            "within_modality_retention": norm(r["within_nn"], 0.0, 1.0),
            "nuisance_control": norm(1 - leak, 0.0, 0.5),
            "stability": 1.0 if r["gates"]["seed_stable"] else 0.0,
            "interpretability": {"interpretable": 1.0, "hybrid": 0.7, "direct": 0.9, "encoder": 0.4}[r["branch"]],
            "simplicity": {"direct": 1.0, "interpretable": 0.8, "hybrid": 0.4, "encoder": 0.5}[r["branch"]],
        }
        r["weighted_score"] = float(sum(SCORE_WEIGHTS[k] * sc_dims[name][k] for k in SCORE_WEIGHTS))

    sc = {"gates": GATES, "weights": SCORE_WEIGHTS, "baseline": {"name": "direct_SNV",
          "heldout_top1": base_top1, "heldout_mrr": base_mrr},
          "candidates": rows, "dimension_scores": sc_dims, "ranking_by_heldout_mrr": order}
    dec = {"outcome": outcome, "headline": _headline(outcome), "reasons": reasons,
           "best_overall": best, "best_encoder": best_enc, "best_interpretable": best_int,
           "hybrid_mrr": hybrid_mrr, "any_candidate_beats_direct": bool(any_beats),
           "primary_encoder_seed_std": seed_stab.get(primary_encoder, {}).get("mrr_std"),
           "frozen_representation": _frozen(outcome, best, best_int),
           "source_generalization_caveat": (
               "Ag-SERS is single-source (Gobbato); leave-source-out is impossible for SERS. "
               "Cross-modal results are within the present 785 nm matched corpus, NOT observation-domain-invariant.")}
    return sc, dec


def _headline(o):
    return {"B1": "Interpretable evidence representation selected",
            "B2": "Encoder representation selected",
            "B3": "Hybrid representation selected",
            "B4": "Modality-stratified representations retained (no shared representation supported)",
            "B5": "Corpus insufficient for encoder conclusions"}[o]


def _frozen(outcome, best, best_int):
    if outcome == "B1": return best_int
    if outcome == "B2": return best
    if outcome == "B3": return "E4_hybrid"
    return None   # B4/B5: nothing frozen as a shared representation


# ─────────────────────────── figures ───────────────────────────
def _save(fig, path): fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def make_figures(results, reps, d, datasets, sc, FIG):
    heldB, within, insample = results["held_out_B"], results["within_modality_C"], results["insample"]
    names = sc["ranking_by_heldout_mrr"]
    branch = {n: sc["candidates"][n]["branch"] for n in names}
    col = {"direct": "#64748b", "interpretable": "#2563eb", "encoder": "#dc2626", "hybrid": "#16a34a"}

    # 1. held-out cross-modal retrieval (MRR with CI) per candidate
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(names))
    mrr = [sc["candidates"][n]["heldout_mrr"] for n in names]
    ci = np.array([sc["candidates"][n]["heldout_mrr_ci"] for n in names])
    err = np.abs(ci.T - np.array(mrr))
    ax.bar(xs, mrr, color=[col[branch[n]] for n in names], yerr=err, capsize=3)
    ax.axhline(sc["baseline"]["heldout_mrr"], ls="--", c="k", label="direct_SNV baseline")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("held-out MRR (Split B)"); ax.legend()
    ax.set_title("Held-out matched-analyte cross-modal retrieval (higher = better)")
    _save(fig, FIG/"heldout_cross_modal_mrr.png")

    # 2. Pareto: modality leakage (in-sample) vs within-modality retention
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for n in names:
        r = sc["candidates"][n]; leak = r["modality_leak"]
        if leak is None: continue
        ax.scatter(leak, r["within_nn"], c=col[branch[n]], s=60)
        ax.annotate(n, (leak, r["within_nn"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axvline(0.83, ls="--", c="grey"); ax.set_xlabel("modality leakage bal-acc (in-sample; lower=better)")
    ax.set_ylabel("within-modality NN same-analyte (Split C; higher=better)")
    ax.set_title("Pareto: nuisance control vs chemistry retention")
    _save(fig, FIG/"pareto_leakage_vs_chemistry.png")

    # 3. PCA of primary encoder embedding colored by modality / analyte / source
    from sklearn.decomposition import PCA
    prim = reps.get("E2_dual_supcon_infonce")
    if prim is not None:
        Z = prim.transform(d.X, d.meta.modality.values)
        P = PCA(2, random_state=0).fit_transform(Z - Z.mean(0))
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
        # modality
        for mod, c in [("raman", "#2563eb"), ("sers", "#dc2626")]:
            m = d.meta.modality.values == mod
            axs[0].scatter(P[m, 0], P[m, 1], s=12, c=c, alpha=0.6, label=mod)
        axs[0].legend(); axs[0].set_title("by modality")
        # analyte (color = family for readability)
        fams = d.meta.family.values
        for f in np.unique(fams):
            m = fams == f
            axs[1].scatter(P[m, 0], P[m, 1], s=12, alpha=0.6, label=f)
        axs[1].legend(fontsize=6, ncol=2); axs[1].set_title("by chemical family")
        # source
        for srcv in d.meta.source.unique():
            m = d.meta.source.values == srcv
            axs[2].scatter(P[m, 0], P[m, 1], s=12, alpha=0.6, label=srcv)
        axs[2].legend(fontsize=6); axs[2].set_title("by source")
        fig.suptitle("Primary encoder (E2) embedding — PCA")
        _save(fig, FIG/"primary_encoder_embedding_pca.png")

        # 4. training/val curves
        h = prim.history
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(h["train_loss"], label="train")
        vl = [x for x in h["val_loss"] if x is not None]
        if vl: ax.plot(vl, label="val")
        ax.axvline(h["stopped_epoch"], ls=":", c="grey", label="stop")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.set_title("E2 training")
        _save(fig, FIG/"encoder_training_curves.png")

    # 5. collapse diagnostics (effective rank + cross-analyte dup) per encoder/hybrid
    encs = [n for n in names if branch[n] in ("encoder", "hybrid")]
    if encs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        er = [insample.get(n, {}).get("collapse", {}).get("effective_rank", 0) for n in encs]
        du = [insample.get(n, {}).get("collapse", {}).get("cross_analyte_duplicate_fraction", 0) or 0 for n in encs]
        x = np.arange(len(encs))
        ax.bar(x - 0.2, er, 0.4, label="effective rank", color="#0ea5e9")
        ax2 = ax.twinx(); ax2.bar(x + 0.2, du, 0.4, label="cross-analyte dup frac", color="#f59e0b")
        ax.set_xticks(x); ax.set_xticklabels(encs, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("effective rank"); ax2.set_ylabel("cross-analyte dup fraction")
        ax.set_title("Encoder collapse diagnostics")
        _save(fig, FIG/"encoder_collapse_diagnostics.png")

    # 6. interpretable atoms/regions/basis
    if "I3_dictionary" in reps:
        atoms = reps["I3_dictionary"].atoms[:8]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, a in enumerate(atoms):
            ax.plot(d.grid, a + i * 0.3, lw=0.8)
        ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_yticks([]); ax.set_title("I3 dictionary atoms (first 8)")
        _save(fig, FIG/"dictionary_atoms.png")
    if "I4_nmf" in reps:
        B = reps["I4_nmf"].basis[:8]; gL = datasets["L2"].grid
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, b in enumerate(B):
            ax.plot(gL, b + i * 0.3, lw=0.8)
        ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_yticks([]); ax.set_title("I4 NMF basis spectra (first 8)")
        _save(fig, FIG/"nmf_basis.png")
    if "I1_regions" in reps:
        reg = reps["I1_regions"]
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(d.grid, np.nan_to_num(d.X).mean(0), c="k", lw=0.8)
        for (lo, hi) in reg.feature_wavenumbers():
            ax.axvspan(lo, hi, alpha=0.08, color="#2563eb")
        ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_title("I1 adaptive region boundaries over mean spectrum")
        _save(fig, FIG/"adaptive_regions.png")

    # 7. augmentation examples
    aud = json.loads((FIG.parent/"tables"/"augmentation_audit.json").read_text())
    ex = aud["examples"][:3]
    fig, axs = plt.subplots(len(ex), 1, figsize=(8, 2.2 * len(ex)))
    if len(ex) == 1: axs = [axs]
    for a, e in zip(axs, ex):
        a.plot(d.grid, e["orig"], label="orig", lw=0.8)
        a.plot(d.grid, e["aug"], label="aug", lw=0.8, alpha=0.7)
        a.set_yticks([]); a.legend(fontsize=7)
    fig.suptitle(f"Augmentation validity (band retention={aud['major_band_retention_mean']:.2f}, "
                 f"invented={aud['invented_peak_fraction_mean']:.2f})")
    _save(fig, FIG/"augmentation_examples.png")
