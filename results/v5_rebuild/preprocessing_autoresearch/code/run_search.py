"""Stage B0 step 2 — the constrained AutoResearch search (INNER folds only).

Runs the controlled arms A-G sequentially, each informed by the previous arm's
winner, plus the frozen prior baselines. The OUTER test folds are never touched.
Deterministic under the frozen seed. Every candidate is fully logged.
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.preprocessing_autoresearch import corpus as CO, search_space as SS
from gaira.preprocessing_autoresearch import pipeline as PL, evaluator as EV, pareto as PA
from gaira.preprocessing_autoresearch import serialization as SER

OUT = REPO / "results/v5_rebuild/preprocessing_autoresearch"
TAB, CFG, LOG = OUT / "tables", OUT / "configs", OUT / "logs"
for p in (TAB, CFG, LOG): p.mkdir(parents=True, exist_ok=True)
SEED = 0
REF_BASELINE = "BASE_asls_sg_l2"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_reference(cache, meta):
    """Reference features for spectral-integrity comparison: raw + L2, no smoothing."""
    ref = SS._mk("REF_raw_l2", "reference", base=("none", {}), smooth=("none", {}), nr="l2")
    X1 = ref_state = cache.build(ref)
    state = PL.fit_stage2(ref, X1, meta, np.ones(len(meta), bool))
    F, fmeta = PL.apply_stage2(ref, X1, meta, state, aggregate=True)
    return F


def run_arm(name, cands, cache, meta, grid, splits, ref_features, results, per_fold_log):
    log(f"ARM {name}: {len(cands)} candidates")
    for i, c in enumerate(cands):
        rows = []
        for fold in splits["folds"]:
            agg, df = EV.evaluate_candidate_inner(c, cache, meta, grid, fold,
                                                  ref_features=ref_features, seed=SEED)
            agg["outer_fold"] = fold["outer_fold"]
            rows.append(agg)
            df = df.assign(cid=c.cid, outer_fold=fold["outer_fold"])
            per_fold_log.append(df)
        R = pd.DataFrame(rows)
        num = R.select_dtypes(include=[np.number]).mean(numeric_only=True).to_dict()
        num["cm_mrr_fold_std"] = float(R.cm_mrr.std())
        num["cid"] = c.cid; num["arm"] = c.arm
        num.update({f"cfg_{k}": v for k, v in
                    {"baseline_r": c.raman["baseline"], "smooth_r": c.smooth_name("raman"),
                     "baseline_s": c.sers["baseline"], "smooth_s": c.smooth_name("sers"),
                     "background": c.background[0], "bg_params": json.dumps(c.background[1]),
                     "aggregate": c.aggregate, "derivative": str(c.derivative),
                     "norm_r": c.norm_raman, "norm_s": c.norm_sers,
                     "peak_transform": c.peak_transform}.items()})
        results[c.cid] = {"metrics": num, "candidate": c}
        if (i + 1) % 10 == 0 or i == len(cands) - 1:
            log(f"  {name} {i+1}/{len(cands)}  best MRR so far "
                f"{max(v['metrics']['cm_mrr'] for v in results.values()):.3f}")


def best_of(results, arm, key="cm_mrr", require_pass=True, base_row=None):
    d = pd.DataFrame([v["metrics"] for v in results.values()])
    d = d[d.arm == arm]
    if d.empty:
        return None
    if require_pass and base_row is not None:
        d2 = PA.apply_rejection(d, base_row)
        d2 = d2[~d2.rejected]
        if not d2.empty:
            d = d2
    return d.sort_values(key, ascending=False).iloc[0].cid


def cfg_of(results, cid):
    return results[cid]["candidate"]


def main():
    t0 = time.time()
    raw, meta = CO.load_raw_frozen()
    splits = json.loads((CFG / "nested_splits.json").read_text())
    grid = PL.GRID
    cache = PL.Stage1Cache(raw)
    log(f"corpus {len(meta)} spectra | matched {meta.matched.sum()} | grid {len(grid)}")

    ref_features = build_reference(cache, meta)
    results, per_fold_log = {}, []

    # frozen prior baselines (reproduced unchanged)
    run_arm("BASELINES", SS.baseline_arm(), cache, meta, grid, splits, ref_features,
            results, per_fold_log)
    base_row = results[REF_BASELINE]["metrics"]
    log(f"reference baseline {REF_BASELINE}: MRR {base_row['cm_mrr']:.3f} "
        f"top1 {base_row['cm_top1']:.3f} peak_effect {base_row['pk_effect_vs_mismatched']:.3f}")

    # ARM A — baseline x normalization
    run_arm("A", SS.arm_A(), cache, meta, grid, splits, ref_features, results, per_fold_log)
    bA = best_of(results, "A_baseline_norm", base_row=base_row)
    cA = cfg_of(results, bA)
    best_base = (cA.raman["baseline"], cA.raman["baseline_params"])
    log(f"  ARM A winner: {bA}  baseline={best_base[0]} norm={cA.norm_raman}")

    # ARM B — smoothing on the two best baselines
    dA = pd.DataFrame([v["metrics"] for v in results.values()])
    dA = dA[dA.arm == "A_baseline_norm"].sort_values("cm_mrr", ascending=False)
    top_bases = []
    for cid in dA.cid.head(6):
        c = cfg_of(results, cid); b = (c.raman["baseline"], c.raman["baseline_params"])
        if b not in top_bases:
            top_bases.append(b)
        if len(top_bases) == 2:
            break
    run_arm("B", SS.arm_B(top_bases), cache, meta, grid, splits, ref_features, results, per_fold_log)
    bB = best_of(results, "B_smoothing", base_row=base_row)
    cB = cfg_of(results, bB)
    best_smooth = (cB.raman["smooth"], cB.raman["smooth_params"])
    best_base = (cB.raman["baseline"], cB.raman["baseline_params"])
    log(f"  ARM B winner: {bB}  smooth={best_smooth}")

    # ARM C — replicate aggregation
    run_arm("C", SS.arm_C(best_base, best_smooth), cache, meta, grid, splits, ref_features,
            results, per_fold_log)
    bC = best_of(results, "C_aggregation", base_row=base_row)
    best_agg = cfg_of(results, bC).aggregate
    log(f"  ARM C winner: {bC}  aggregate={best_agg}")

    # ARM D — Ag-SERS common-background correction (primary)
    run_arm("D", SS.arm_D(best_base, best_smooth, agg=best_agg), cache, meta, grid, splits,
            ref_features, results, per_fold_log)
    bD = best_of(results, "D_background", base_row=base_row)
    best_bg = cfg_of(results, bD).background
    log(f"  ARM D winner: {bD}  background={best_bg}")

    # ARM E — derivatives
    run_arm("E", SS.arm_E(best_base, best_smooth, best_bg, agg=best_agg), cache, meta, grid,
            splits, ref_features, results, per_fold_log)
    bE = best_of(results, "E_derivative", base_row=base_row)
    best_deriv = cfg_of(results, bE).derivative
    log(f"  ARM E winner: {bE}  derivative={best_deriv}")

    # ARM F — modality-specific vs global
    run_arm("F", SS.arm_F(best_base, best_base, best_smooth, best_smooth, best_bg, agg=best_agg),
            cache, meta, grid, splits, ref_features, results, per_fold_log)

    # ARM G — a few rational combinations
    combos = [
        dict(base=best_base, smooth=best_smooth, bg=best_bg, agg=best_agg, deriv=best_deriv, nr="l2"),
        dict(base=best_base, smooth=best_smooth, bg=best_bg, agg=best_agg, deriv=0, nr="l2", ns="area"),
        dict(base=best_base, smooth=best_smooth, bg=best_bg, agg="huber", deriv=0, nr="l2"),
        dict(base=best_base, smooth=("none", {}), bg=best_bg, agg=best_agg, deriv=0, nr="l2"),
        dict(base=best_base, smooth=best_smooth, bg=("lowrank", {"k": 2}), agg=best_agg, deriv=0, nr="l2"),
        dict(base=best_base, smooth=best_smooth, bg=("lowrank", {"k": 3}), agg=best_agg, deriv=1, nr="l2"),
        dict(base=best_base, smooth=best_smooth, bg=("scaled_mean", {"alpha": "robust"}),
             agg=best_agg, deriv=0, nr="l2"),
        dict(base=best_base, smooth=best_smooth, bg=best_bg, agg=best_agg, deriv=0, nr="area"),
    ]
    run_arm("G", SS.arm_G(combos), cache, meta, grid, splits, ref_features, results, per_fold_log)

    # ── persist ──
    D = pd.DataFrame([v["metrics"] for v in results.values()])
    D.to_csv(TAB / "search_results.csv", index=False)
    pd.concat(per_fold_log, ignore_index=True).to_csv(LOG / "per_fold_metrics.csv", index=False)
    cat = pd.DataFrame([SER.candidate_to_json(v["candidate"]) for v in results.values()])
    cat.to_csv(TAB / "pipeline_catalog.csv", index=False)

    sel, judged = PA.select(D, base_row)
    judged.to_csv(TAB / "search_results_judged.csv", index=False)
    pf = PA.pareto_front(judged[~judged.rejected]) if (~judged.rejected).any() else judged.head(0)
    if len(pf):
        pf[pf.on_front].to_csv(TAB / "pareto_front.csv", index=False)
    (TAB / "inner_selection.json").write_text(json.dumps(
        {"reference_baseline": REF_BASELINE,
         "baseline_metrics": {k: base_row[k] for k in
                              ("cm_mrr", "cm_top1", "pk_effect_vs_mismatched",
                               "rep_sers_replicate_cos", "chem_raman_1nn")},
         "arm_winners": {"A": bA, "B": bB, "C": bC, "D": bD, "E": bE},
         "selection": sel, "n_candidates": int(len(D))}, indent=2, default=float))

    log(f"candidates evaluated: {len(D)}")
    log(f"rejected by hard rules: {int(judged.rejected.sum())}")
    log(f"selection: {sel}")
    log(f"runtime {time.time()-t0:.0f}s")
    return D, sel


if __name__ == "__main__":
    main()
