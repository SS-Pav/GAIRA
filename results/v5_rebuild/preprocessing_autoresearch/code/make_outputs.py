"""Stage B0 step 4 — derived tables, controls and artifacts (READ-ONLY analysis).

Produces the remaining required tables (peak integrity, background models,
replicate preservation, cross-modal retrieval, null controls, per-analyte
before/after, family results) plus the artifacts. Uses the already-computed search
and outer-test results; the outer test is NOT re-run.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from gaira.preprocessing_autoresearch import corpus as CO, search_space as SS
from gaira.preprocessing_autoresearch import pipeline as PL, evaluator as EV
from gaira.preprocessing_autoresearch import objectives as OB, background_models as BG
from gaira.preprocessing_autoresearch import serialization as SER
from run_outer_test import rebuild

OUT = REPO / "results/v5_rebuild/preprocessing_autoresearch"
TAB, CFG, ART = OUT / "tables", OUT / "configs", OUT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
KEY = ["BASE_raw_l2", "BASE_asls_sg_l2", "BASE_asls_sg_snv", "BASE_asls_sg_l2_bgmean",
       "B_0_savgol6", "D_scaled_mean2"]


def main():
    judged = pd.read_csv(TAB / "search_results_judged.csv")
    catalog = pd.read_csv(TAB / "pipeline_catalog.csv")
    outer = pd.read_csv(TAB / "outer_test_results.csv", index_col=0)
    splits = json.loads((CFG / "nested_splits.json").read_text())
    decision = json.loads((TAB / "final_decision.json").read_text())
    raw, meta = CO.load_raw_frozen()
    cache = PL.Stage1Cache(raw); grid = PL.GRID

    # ── derived tables from the search ──
    j = judged
    j[["cid", "arm", "si_peak_retention", "si_peak_invention", "si_peak_width_ratio",
       "si_effective_rank", "si_cross_analyte_duplicate_frac", "si_negative_lobe_burden",
       "si_edge_artefact_ratio", "rejected", "reject_reasons"]] \
        .to_csv(TAB / "peak_integrity_results.csv", index=False)
    j[["cid", "arm", "rep_raman_replicate_cos", "rep_sers_replicate_cos",
       "rep_raman_replicate_margin", "rep_sers_replicate_margin",
       "rep_raman_between_analyte_cos", "rep_sers_between_analyte_cos",
       "rep_raman_replicate_var", "rep_sers_replicate_var"]] \
        .to_csv(TAB / "replicate_preservation.csv", index=False)
    j[["cid", "arm", "cm_mrr", "cm_top1", "cm_top3", "cm_top5", "cm_median_rank",
       "cm_top1_r2s", "cm_top1_s2r", "cm_matched_cos", "cm_mismatched_cos",
       "cm_matched_minus_mismatched", "cm_mrr_fold_std"]] \
        .to_csv(TAB / "cross_modal_retrieval.csv", index=False)
    j[["cid", "arm", "pk_matched", "pk_mismatched", "pk_random",
       "pk_effect_vs_mismatched", "pk_effect_size"]] \
        .to_csv(TAB / "null_control_results.csv", index=False)
    bgt = j[j.arm == "D_background"][["cid", "cfg_background", "cfg_bg_params", "bg_variance_explained",
                                      "cm_mrr", "cm_top1", "pk_effect_vs_mismatched",
                                      "rep_sers_replicate_cos", "chem_sers_1nn", "rejected",
                                      "reject_reasons"]]
    bgt.to_csv(TAB / "background_model_results.csv", index=False)

    # ── Control 4/5 on held-out folds: background variance removed vs analyte retention ──
    rows = []
    for m, p in BG.CANDIDATES:
        cand = SS._mk(f"ctl_{m}", "control", base=("asls", {"lam": 1e5}),
                      smooth=("savgol", {"window": 9, "poly": 3}), bg=(m, p), nr="l2")
        ve, s1nn, r1nn = [], [], []
        for fold in splits["folds"]:
            X1 = cache.build(cand)
            tr = meta.analyte.isin(fold["devel_analytes"]).values | (~meta.matched.values)
            st = PL.fit_stage2(cand, X1, meta, tr)
            ve.append(st["background"].variance_explained(X1[meta.modality.values == "sers"]))
            Xs, smeta = PL.apply_stage2(cand, X1, meta, st, aggregate=False)
            ch = OB.within_modality_chemistry(Xs, smeta, fold["test_analytes"])
            s1nn.append(ch.get("sers_1nn")); r1nn.append(ch.get("raman_1nn"))
        rows.append({"background": m, "params": json.dumps(p),
                     "variance_explained": float(np.mean(ve)),
                     "sers_analyte_1nn_heldout": float(np.nanmean(s1nn)),
                     "raman_analyte_1nn_heldout": float(np.nanmean(r1nn))})
    pd.DataFrame(rows).to_csv(TAB / "background_variance_vs_retention.csv", index=False)

    # ── per-analyte before/after on the outer test ──
    per = {}
    for cid in KEY:
        f = TAB / f"outer_per_analyte_{cid}.csv"
        if f.exists():
            d = pd.read_csv(f)[["analyte", "outer_fold", "rank", "matched_cos", "mismatched_cos"]]
            per[cid] = d.set_index("analyte")
    base = per["BASE_asls_sg_l2"]
    ba = base.rename(columns={"rank": "rank_baseline", "matched_cos": "matched_cos_baseline",
                              "mismatched_cos": "mismatched_cos_baseline"})
    for cid, d in per.items():
        if cid == "BASE_asls_sg_l2":
            continue
        ba[f"rank_{cid}"] = d["rank"]
        ba[f"matched_cos_{cid}"] = d["matched_cos"]
    ba = ba.reset_index().merge(
        meta.drop_duplicates("analyte")[["analyte", "family", "raman_multi_source"]],
        on="analyte", how="left")
    ba["rank_delta_best"] = ba["rank_baseline"] - ba.get("rank_B_0_savgol6", ba["rank_baseline"])
    ba.to_csv(TAB / "per_analyte_before_after.csv", index=False)

    # ── family results ──
    fam = ba.groupby("family").agg(
        n=("analyte", "count"),
        rank_baseline=("rank_baseline", "mean"),
        rank_best_candidate=("rank_B_0_savgol6", "mean"),
        matched_cos_baseline=("matched_cos_baseline", "mean"),
        matched_cos_best=("matched_cos_B_0_savgol6", "mean")).reset_index()
    fam["rank_improvement"] = fam.rank_baseline - fam.rank_best_candidate
    fam.sort_values("rank_improvement", ascending=False).to_csv(TAB / "family_results.csv", index=False)

    # ── artifacts ──
    sel = decision.get("selected")
    (ART / "selected_pipeline.json").write_text(json.dumps({
        "selected": sel,
        "outcome": decision["outcome"],
        "frozen": False,
        "reason": ("No preprocessing pipeline was frozen. Outcome "
                   f"{decision['outcome']}: {decision['headline']}. Cross-modal gains were "
                   "obtained only by pipelines that damage Ag-SERS replicate structure without "
                   "improving matched-vs-mismatched peak specificity."),
    }, indent=2))
    man = {
        "study": "GAIRA V5 Stage B0 — Preprocessing AutoResearch",
        "corpus": {"n_spectra": int(len(meta)), "n_raman": int((meta.modality == "raman").sum()),
                   "n_sers": int((meta.modality == "sers").sum()),
                   "n_analytes": int(meta.analyte.nunique()),
                   "n_matched": int(meta.matched.sum() and len(CO.matched_analytes(meta)))},
        "grid": "520-1750 cm-1 @ 2 cm-1 (fixed)",
        "n_candidates_evaluated": int(len(judged)),
        "n_rejected_by_integrity_rules": int(judged.rejected.sum()),
        "outer_design": {"n_outer": splits["n_outer"], "n_inner": splits["n_inner"],
                         "seed": splits["seed"], "outer_test_used_once": True},
        "outcome": decision["outcome"], "frozen_pipeline": None,
        "reference_baseline": "BASE_asls_sg_l2",
        "outer_test_results": {k: {kk: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                   for kk, v in outer.loc[k].to_dict().items()
                                   if kk in ("mrr", "top1", "peak_effect",
                                             "sers_replicate_cos", "raman_1nn", "sers_1nn")}
                               for k in outer.index},
    }
    (ART / "preprocessing_manifest.json").write_text(json.dumps(man, indent=2, default=str))

    for cid in KEY:
        SER.save_candidate(rebuild(cid, catalog), ART / f"candidate_{cid}.json")

    print("derived tables + artifacts written")
    for f in sorted(TAB.glob("*.csv")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
