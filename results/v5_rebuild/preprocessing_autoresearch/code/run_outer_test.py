"""Stage B0 step 3 — the ONE-TIME outer test.

Evaluates the pipeline selected on inner folds, plus the frozen reference
baselines, on the held-out outer test analytes. Acceptance thresholds were frozen
in configs/acceptance_thresholds.json BEFORE this script was ever run; this script
records that the outer test has now been consumed.

For each outer fold: fit stage-2 on the DEVEL analytes only, evaluate on the TEST
analytes (both modalities test-only). Adds analyte-bootstrap CIs, permutation
nulls, and fold-stability, then applies the frozen acceptance rule -> P1-P5.
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from gaira.preprocessing_autoresearch import corpus as CO, search_space as SS
from gaira.preprocessing_autoresearch import pipeline as PL, evaluator as EV
from gaira.preprocessing_autoresearch import objectives as OB, serialization as SER
from gaira.preprocessing_autoresearch import diagnostics as DG

OUT = REPO / "results/v5_rebuild/preprocessing_autoresearch"
TAB, CFG, ART, LOG = OUT / "tables", OUT / "configs", OUT / "artifacts", OUT / "logs"
for p in (TAB, ART, LOG): p.mkdir(parents=True, exist_ok=True)
SEED = 0
N_BOOT, N_PERM = 2000, 2000


def _as_obj(v):
    """Catalog cells hold Python reprs of dicts, not JSON."""
    if isinstance(v, dict):
        return v
    import ast
    return ast.literal_eval(v)


def rebuild(cid, catalog):
    d = catalog[catalog.cid == cid].iloc[0].to_dict()
    bgd = _as_obj(d["background"])
    deriv = d["derivative"]
    if isinstance(deriv, str) and deriv.isdigit():
        deriv = int(deriv)
    return PL.Candidate(
        cid=d["cid"], arm=d["arm"],
        raman=_as_obj(d["raman"]), sers=_as_obj(d["sers"]),
        background=(bgd["method"], bgd["params"]),
        aggregate=d["aggregate"], derivative=deriv,
        norm_raman=d["norm_raman"], norm_sers=d["norm_sers"],
        peak_transform=d["peak_transform"])


def eval_outer(cand, cache, meta, grid, splits, ref_features):
    """Per-fold outer-test metrics + pooled ranks for bootstrap/permutation."""
    rows, all_ranks, per_analyte = [], [], []
    for fold in splits["folds"]:
        rng = np.random.default_rng(SEED + fold["outer_fold"])
        m, (F, fmeta, state) = EV.evaluate_fold(
            cand, cache, meta, grid, fold["devel_analytes"], fold["test_analytes"],
            ref_features=ref_features, rng=rng, n_perm=0)
        m["outer_fold"] = fold["outer_fold"]; m["n_test"] = len(fold["test_analytes"])
        rows.append(m)
        cm = OB.cross_modal(F, fmeta, fold["test_analytes"])
        if not cm.get("insufficient"):
            all_ranks.extend(cm["ranks"])
            R, S, keep = OB._split_modalities(F, fmeta, fold["test_analytes"])
            if R is not None:
                Sim = OB._unit(R) @ OB._unit(S).T
                n = len(keep)
                for i, a in enumerate(keep):
                    per_analyte.append({"analyte": a, "outer_fold": fold["outer_fold"],
                                        "rank": 1 + int((Sim[i] > Sim[i, i]).sum()),
                                        "matched_cos": float(Sim[i, i]),
                                        "mismatched_cos": float((Sim[i].sum() - Sim[i, i]) / (n - 1))})
    df = pd.DataFrame(rows)
    ranks = np.array(all_ranks)
    return df, ranks, pd.DataFrame(per_analyte)


def boot_ci(vals, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    t0 = time.time()
    manifest = json.loads((CFG / "study_manifest.json").read_text())
    if manifest.get("outer_test_used"):
        print("!! outer test already consumed — refusing to rerun (see study_manifest.json)")
        sys.exit(1)

    acc = json.loads((CFG / "acceptance_thresholds.json").read_text())
    splits = json.loads((CFG / "nested_splits.json").read_text())
    sel = json.loads((TAB / "inner_selection.json").read_text())
    catalog = pd.read_csv(TAB / "pipeline_catalog.csv")
    raw, meta = CO.load_raw_frozen()
    cache = PL.Stage1Cache(raw); grid = PL.GRID

    ref = SS._mk("REF", "reference", base=("none", {}), smooth=("none", {}), nr="l2")
    X1r = cache.build(ref)
    st = PL.fit_stage2(ref, X1r, meta, np.ones(len(meta), bool))
    ref_features, _ = PL.apply_stage2(ref, X1r, meta, st, aggregate=True)

    base = acc["reference_baseline"]
    to_eval = ["BASE_raw_l2", "BASE_asls_sg_l2", "BASE_asls_sg_snv", "BASE_asls_sg_l2_bgmean"]
    selected = sel["selection"].get("selected")
    if selected and selected not in to_eval:
        to_eval.append(selected)

    # If inner selection produced NO eligible candidate, still characterise, on the
    # held-out data, (a) the highest-MRR candidate that FAILED eligibility and (b) the
    # best explicit background-correction candidate. These are labelled ineligible and
    # are never promoted to "selected"; this remains a single outer-test consumption.
    judged = pd.read_csv(TAB / "search_results_judged.csv")
    characterisation = []
    if selected is None:
        imp = judged[judged.cm_mrr > judged[judged.cid == base].cm_mrr.iloc[0]] \
            if base in set(judged.cid) else judged
        if len(imp):
            top = imp.sort_values("cm_mrr", ascending=False).iloc[0].cid
            if top not in to_eval:
                to_eval.append(top); characterisation.append(top)
        bgc = judged[judged.arm == "D_background"]
        bgc = bgc[bgc.cfg_background != "none"] if "cfg_background" in bgc else bgc
        if len(bgc):
            topbg = bgc.sort_values("cm_mrr", ascending=False).iloc[0].cid
            if topbg not in to_eval:
                to_eval.append(topbg); characterisation.append(topbg)

    results, ranks_by, per_an_by = {}, {}, {}
    for cid in to_eval:
        cand = rebuild(cid, catalog)
        df, ranks, pa = eval_outer(cand, cache, meta, grid, splits, ref_features)
        results[cid] = df; ranks_by[cid] = ranks; per_an_by[cid] = pa
        print(f"  outer[{cid}] MRR {np.mean(1.0/ranks):.3f} top1 {np.mean(ranks==1):.3f}", flush=True)

    rb = ranks_by[base]
    summary = {}
    for cid, ranks in ranks_by.items():
        mrr = float(np.mean(1.0 / ranks)); top1 = float(np.mean(ranks == 1))
        d = results[cid]
        summary[cid] = {
            "mrr": mrr, "top1": top1, "top3": float(np.mean(ranks <= 3)),
            "top5": float(np.mean(ranks <= 5)), "median_rank": float(np.median(ranks)),
            "mrr_ci": boot_ci(1.0 / ranks),
            "delta_mrr_vs_base": mrr - float(np.mean(1.0 / rb)),
            "delta_top1_vs_base": top1 - float(np.mean(rb == 1)),
            "delta_mrr_ci": ([0.0, 0.0] if cid == base
                             else _paired_delta_ci(1.0 / ranks, 1.0 / rb)),
            "peak_effect": float(d.pk_effect_vs_mismatched.mean()),
            "peak_matched": float(d.pk_matched.mean()), "peak_mismatched": float(d.pk_mismatched.mean()),
            "peak_random": float(d.pk_random.mean()),
            "sers_replicate_cos": float(d.rep_sers_replicate_cos.mean()),
            "raman_replicate_cos": float(d.rep_raman_replicate_cos.mean()),
            "sers_replicate_margin": float(d.get("rep_sers_replicate_margin", pd.Series([np.nan])).mean()),
            "raman_1nn": float(d.chem_raman_1nn.mean()), "sers_1nn": float(d.chem_sers_1nn.mean()),
            "peak_retention": float(d.si_peak_retention.mean()),
            "peak_invention": float(d.si_peak_invention.mean()),
            "duplicate_frac": float(d.si_cross_analyte_duplicate_frac.mean()),
            "effective_rank": float(d.si_effective_rank.mean()),
            "bg_variance_explained": float(d.bg_variance_explained.mean()),
            "fold_mrr": d.cm_mrr.tolist(),
        }
        # permutation null on pooled MRR
        rng = np.random.default_rng(SEED)
        null = []
        for _ in range(N_PERM):
            null.append(np.mean(1.0 / rng.permutation(ranks)))
        summary[cid]["perm_mrr_p"] = float((np.sum(np.array(null) >= mrr) + 1) / (N_PERM + 1))

    pd.DataFrame(summary).T.to_csv(TAB / "outer_test_results.csv")
    for cid, pa in per_an_by.items():
        pa.to_csv(TAB / f"outer_per_analyte_{cid}.csv", index=False)

    decision = decide(summary, base, selected, acc, characterisation)
    (TAB / "final_decision.json").write_text(json.dumps(decision, indent=2, default=float))

    manifest["outer_test_used"] = True
    manifest["outer_test_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["outer_test_candidates"] = to_eval
    (CFG / "study_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== OUTER TEST (used once) ===")
    for cid, s in summary.items():
        print(f"  {cid:26s} MRR {s['mrr']:.3f} CI{np.round(s['mrr_ci'],3)} top1 {s['top1']:.3f} "
              f"pkEff {s['peak_effect']:+.3f} repS {s['sers_replicate_cos']:.3f} "
              f"chemR {s['raman_1nn']:.3f} chemS {s['sers_1nn']:.3f}")
    print(f"\n=== DECISION: {decision['outcome']} — {decision['headline']} ===")
    for r in decision["reasons"]:
        print("  -", r)
    print(f"runtime {time.time()-t0:.0f}s")
    return summary, decision


def _paired_delta_ci(a, b, n_boot=N_BOOT, seed=0):
    """Bootstrap CI for the difference of two independent rank-derived means."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b)
    d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
         for _ in range(n_boot)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def decide(summary, base, selected, acc, characterisation=None):
    b = summary[base]
    reasons = []
    if selected is None or selected not in summary:
        # No eligible pipeline. Distinguish "nothing helps at all" (P3) from
        # "things appear to help but only by damaging the spectra" (P4) using the
        # ineligible characterisation candidates evaluated on the held-out data.
        reasons.append("Inner selection produced NO eligible candidate: of 120 pipelines, none "
                       "improved cross-modal retrieval AND matched-peak specificity while "
                       "satisfying the spectral-integrity rejection rules.")
        chars = [c for c in (characterisation or []) if c in summary]
        gain_via_damage = False
        for c in chars:
            s = summary[c]
            dm = s["mrr"] - b["mrr"]
            rep_ok = s["sers_replicate_cos"] >= acc["sers_replicate_min_frac_of_L2"] * b["sers_replicate_cos"]
            pk_gain = s["peak_effect"] - b["peak_effect"]
            reasons.append(f"characterisation {c}: ΔMRR {dm:+.3f}, Ag-SERS replicate cosine "
                           f"{s['sers_replicate_cos']:.3f} vs baseline {b['sers_replicate_cos']:.3f} "
                           f"(integrity_ok={rep_ok}), Δpeak-specificity {pk_gain:+.4f}.")
            if dm > 0 and (not rep_ok) and pk_gain <= 0:
                gain_via_damage = True
        if gain_via_damage:
            return {"outcome": "P4",
                    "headline": "apparent improvement is caused by overprocessing",
                    "reasons": reasons + [
                        "Cross-modal retrieval rises only for pipelines that strip the broad shared "
                        "component, which collapses Ag-SERS replicate agreement without any gain in "
                        "matched-vs-mismatched peak specificity — appearance, not shared chemistry."],
                    "baseline": b, "selected": None, "characterisation": chars,
                    "summary": {c: summary[c] for c in chars}}
        return {"outcome": "P3", "headline": "preprocessing does not rescue comparability",
                "reasons": reasons, "baseline": b, "selected": None,
                "characterisation": chars, "summary": {c: summary[c] for c in chars}}
    s = summary[selected]
    d_mrr = s["mrr"] - b["mrr"]; d_top1 = s["top1"] - b["top1"]
    ci = s["delta_mrr_ci"]; ci_excl = (ci[0] > 0) or (ci[1] < 0)
    pk_rel = ((s["peak_effect"] - b["peak_effect"]) / abs(b["peak_effect"])) if b["peak_effect"] else np.inf
    rep_ok = s["sers_replicate_cos"] >= acc["sers_replicate_min_frac_of_L2"] * b["sers_replicate_cos"]
    chem_ok = (s["raman_1nn"] >= acc["within_modality_retrieval_min_frac"] * b["raman_1nn"] and
               s["sers_1nn"] >= acc["within_modality_retrieval_min_frac"] * b["sers_1nn"])
    peak_ok = (s["peak_retention"] >= acc["peak_retention_min"] and
               s["peak_invention"] <= acc["peak_invention_max"])
    dup_ok = s["duplicate_frac"] <= b["duplicate_frac"] + 1e-9
    folds = np.array(s["fold_mrr"]) - np.array(b["fold_mrr"])
    stable = float(np.mean(folds > 0))

    hit_mrr = d_mrr >= acc["outer_mrr_improvement_abs"]
    hit_top1 = d_top1 >= acc["outer_top1_improvement_abs"]
    hit_pk = pk_rel >= acc["peak_effect_relative_improvement"]
    hit_stab = stable >= acc["min_fraction_outer_folds_stable"]

    reasons.append(f"selected={selected}: ΔMRR {d_mrr:+.3f} (need ≥{acc['outer_mrr_improvement_abs']}), "
                   f"Δtop1 {d_top1:+.3f} (need ≥{acc['outer_top1_improvement_abs']}), "
                   f"ΔMRR 95% CI {np.round(ci,3).tolist()} (excludes 0: {ci_excl}), "
                   f"peak-effect relative change {pk_rel:+.2f} (need ≥{acc['peak_effect_relative_improvement']}), "
                   f"folds improved {stable:.0%} (need ≥{acc['min_fraction_outer_folds_stable']:.0%}).")
    integrity_ok = rep_ok and chem_ok and peak_ok and dup_ok
    if not integrity_ok:
        reasons.append(f"integrity: replicate_ok={rep_ok}, chemistry_ok={chem_ok}, "
                       f"peak_ok={peak_ok}, no_collapse={dup_ok}.")

    if hit_mrr and hit_top1 and ci_excl and hit_pk and hit_stab and integrity_ok:
        out = "P1"; head = "preprocessing materially rescues comparability"
    elif integrity_ok and (d_mrr > 0) and (s["peak_effect"] > b["peak_effect"]) and not hit_mrr:
        out = "P2"; head = "background correction helps but remains insufficient"
    elif (d_mrr > 0) and not integrity_ok:
        out = "P4"; head = "apparent improvement is caused by overprocessing"
    else:
        out = "P3"; head = "preprocessing does not rescue comparability"
    return {"outcome": out, "headline": head, "reasons": reasons, "selected": selected,
            "baseline_name": base, "baseline": b, "selected_metrics": s,
            "gates": {"mrr": bool(hit_mrr), "top1": bool(hit_top1), "ci_excludes_zero": bool(ci_excl),
                      "peak_effect": bool(hit_pk), "fold_stability": bool(hit_stab),
                      "replicate": bool(rep_ok), "chemistry": bool(chem_ok),
                      "peak_integrity": bool(peak_ok), "no_collapse": bool(dup_ok)},
            "fold_improvement_fraction": stable}


if __name__ == "__main__":
    main()
