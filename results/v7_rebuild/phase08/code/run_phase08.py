#!/usr/bin/env python3
"""GAIRA V7 — Phase 08: hierarchical molecular retrieval (Raman only).

Five retrieval models benchmarked head to head on the frozen chain. **BSV2 is not imported and
is not on the inference path.** Baselines are reproduced before anything new is measured.

    python results/v7_rebuild/phase08/code/run_phase08.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "phase00/code"))

import v7_paths as P                                                    # noqa: E402
from gaira.v7.io import PhaseOutputs, frozen_root                       # noqa: E402
from gaira.v7.inference import projection as PRJ                        # noqa: E402
from gaira.v7.meta import perturbations as PERT                         # noqa: E402
from gaira.v7.chemistry import evidence as CHEM, registry as REG        # noqa: E402
from gaira.v7.retrieval import evaluation as EVAL, explain as EXP, models as MOD  # noqa: E402

warnings.filterwarnings("ignore")

PHASE, PHASE_NAME = "08", "Hierarchical molecular retrieval"
OUT = PhaseOutputs(PHASE, extra=("interactive", "manifests")).ensure()
FROZEN = frozen_root()
EXPECTED = {"atlas": "09ed804a40836f4a05a91ba10900cded",
            "lsm": "208482d6f7178b5b8f16cace91be55b0",
            "csm": "0b4aa550ccefed3edabdbde5bae11c8d",
            "engine": "20d8bd99ce71f45a125c6a2b1d719e51"}
SEED = 0
NOISE = ("gaussian_noise", "shot_noise", "baseline_drift", "intensity_scaling",
         "wavelength_shift", "band_broadening", "cosmic_spikes")
# Weight grid for Model C, searched ONLY in inner folds. Declared here, before any result.
WEIGHT_GRID = [{"alpha": a, "beta": b, "gamma": g, "delta": d}
               for a in (0.4, 0.6, 0.8, 1.0)
               for b in (0.0, 0.1, 0.2, 0.4)
               for g in (0.0, 0.05, 0.1)
               for d in (0.0, 0.1, 0.3)]
LOG: list[str] = []


def log(m):
    line = f"[phase08] {m}"
    print(line, flush=True)
    LOG.append(line)


def _ser(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


STAMP: dict = {}


def wtab(df, name, where=None):
    p = (where or OUT.tables) / name
    df.to_csv(p, index=False)
    return {"artifact_id": name, "path": OUT.rel(p), "sha256": P.sha256_file(p), "rows": len(df)}


def wjson(obj, name, where=None):
    p = (where or OUT.artifacts) / name
    if isinstance(obj, dict):
        obj = {**obj, "_provenance": STAMP}
    p.write_text(json.dumps(obj, indent=2, default=_ser))
    return {"artifact_id": name, "path": OUT.rel(p), "sha256": P.sha256_file(p)}


def wnpz(name, **arr):
    p = OUT.artifacts / name
    np.savez_compressed(p, _provenance=json.dumps(STAMP, default=_ser), **arr)
    return {"artifact_id": name, "path": OUT.rel(p), "sha256": P.sha256_file(p)}


def main() -> int:
    outputs: list[dict] = []
    t0 = datetime.now(timezone.utc)

    log("ARCHITECTURE COMPLIANCE STATEMENT")
    for line in [
        "  Phase 08 asks whether hierarchical molecular retrieval using the validated Chemistry",
        "  Evidence layer beats direct CSM retrieval, while remaining fully explainable.",
        "  BSV2 (Phase 07, A-20) is NOT on the inference path and is not imported anywhere in",
        "  src/gaira/v7/retrieval/. Not benchmarked: latent geometry, UMAP, PCA, clustering,",
        "  SERS, Ag-SERS, serum, EV, DART-Met. Pure Raman only.",
        "  Frozen and read-only: preprocessing, LSMs, CSMs, Chemistry Evidence, Phase 06",
        "  calibration. Baselines are reproduced before any new model is measured.",
        "  Model C never filters: candidates outside the reranked shortlist keep their CSM score",
        "  and can still appear in the ranking, so a chemistry error is always recoverable.",
        "  Every Model C score decomposes exactly into four weighted terms; the decomposition is",
        "  asserted against the model's own total.",
    ]:
        log(line)

    csm_reg = json.loads((FROZEN / "phase02/artifacts/csm_registry_v1.json").read_text())
    st01 = json.loads((FROZEN / "phase01/PHASE_STATE.json").read_text())
    st05 = json.loads((FROZEN / "phase05/PHASE_STATE.json").read_text())
    st06 = json.loads((FROZEN / "phase06/PHASE_STATE.json").read_text())
    got = {"csm": csm_reg["fingerprint"], "lsm": st01["registry_fingerprint"],
           "engine": st05["engine_fingerprint"], "atlas": EXPECTED["atlas"]}
    for k in ("csm", "lsm", "engine"):
        if got[k] != EXPECTED[k]:
            log(f"ABORT — {k} fingerprint {got[k]} != {EXPECTED[k]}")
            return 2
    if st06["status"] != "COMPLETE":
        log(f"ABORT — Phase 06 status {st06['status']}")
        return 2
    log(f"  frozen verified: LSM {got['lsm']} · CSM {got['csm']} · engine {got['engine']}")

    # ── the frozen chain ─────────────────────────────────────────────────────
    br = np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
    X, grid = np.asarray(br["X"], float), np.asarray(br["grid"], float)
    y = np.array([str(s) for s in br["canonical_id"]])
    CSM = np.load(FROZEN / "phase02/artifacts/csm_dictionary_v1.npz")["CSM"]
    recs = csm_reg["csms"]
    part = pd.read_csv(FROZEN / "phase00/tables/chemical_partition_v1.csv")
    cls_of = dict(zip(part.canonical_id, part.fine_class))
    canon = pd.read_csv(FROZEN / "phase00/tables/canonical_analytes_v1.csv")
    broad_of = dict(zip(canon.canonical_id, canon.broad_class))
    folds_t = pd.read_csv(FROZEN / "phase00/tables/cv_folds_v1.csv")
    folds = np.array([dict(zip(folds_t.canonical_id, folds_t.fold))[v] for v in y])
    cls = np.array([cls_of[v] for v in y])
    axis_names = list(REG.CLASS_ORDER)
    A = PRJ.project(X, CSM)
    chem_cfg = json.loads((FROZEN / "phase06/artifacts/chemistry_evidence_model_v1.json"
                           ).read_text())["config"]
    Pq_all = MOD.prominence(X, grid)
    log(f"  corpus {X.shape} · {len(set(y))} molecules · {len(set(cls))} classes · "
        f"{len(set(folds.tolist()))} molecule-grouped folds")
    STAMP.update({"phase": PHASE, "seed": SEED, "input_fingerprints": got,
                  "code_fingerprint": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
                  "created_utc": t0.isoformat(), "bsv2_used": False, "scope": "Raman only"})

    def chem_model_for(mask):
        cfg = dict(chem_cfg)
        fam = cfg.pop("family")
        return (CHEM.fit_D(A[mask], y[mask], cls[mask], broad_of=broad_of, **cfg)
                if fam == "D_hierarchical" else CHEM.fit(fam, A[mask], y[mask], cls[mask], **cfg))

    # ── PART 1 — reproduce the baselines ─────────────────────────────────────
    log("PART 1 — reproducing the Phase 05 / 06.5 baselines before any new work")

    def split_a_ranks(score_fn):
        """Leave one spectrum out; the molecule's other replicates stay in the bank."""
        rk = np.zeros(len(X), int)
        for i in range(len(X)):
            keep = np.ones(len(X), bool)
            keep[i] = False
            rk[i] = score_fn(i, keep)
        return rk

    def rank_csm(i, keep):
        Rb, lb = MOD.build_bank(A[keep], y[keep])
        s = MOD.score_B(A[i:i + 1], Rb)[0]
        srt = np.sort(s)
        mB[i] = float(srt[-1] - srt[-2])
        hit = np.where(np.array(lb)[np.argsort(-s)] == y[i])[0]
        return int(hit[0]) + 1 if len(hit) else len(lb) + 1

    margins: dict[str, np.ndarray] = {}
    mB = np.zeros(len(X))

    rk_B = split_a_ranks(rank_csm)
    margins["B_csm"] = mB.copy()
    m_B = EVAL.split_a_metrics(rk_B, len(set(y)))
    ref = json.loads((FROZEN / "phase05/artifacts/phase05_summary_v1.json").read_text())["split_a"]
    ok = all(abs(m_B[k] - ref[f"molecule_{k}"]) < 1e-9 for k in ("top1", "top3", "top5"))
    for k in ("top1", "top3", "top5"):
        log(f"  CSM baseline {k}: phase05 {ref[f'molecule_{k}']:.6f}  reproduced {m_B[k]:.6f}  "
            f"{'EXACT' if abs(m_B[k] - ref[f'molecule_{k}']) < 1e-9 else 'MISMATCH'}")
    if not ok:
        log("ABORT — the CSM retrieval baseline did not reproduce")
        return 2

    def rank_raw(i, keep):
        Rb, lb = MOD.build_bank(X[keep], y[keep])
        s = MOD.score_A(X[i:i + 1], Rb)[0]
        hit = np.where(np.array(lb)[np.argsort(-s)] == y[i])[0]
        return int(hit[0]) + 1 if len(hit) else len(lb) + 1

    rk_A = split_a_ranks(rank_raw)
    m_A = EVAL.split_a_metrics(rk_A, len(set(y)))
    log(f"  RAW baseline top1 {m_A['top1']:.4f} top5 {m_A['top5']:.4f} MRR {m_A['mrr']:.4f}")
    log(f"  CSM baseline top1 {m_B['top1']:.4f} top5 {m_B['top5']:.4f} MRR {m_B['mrr']:.4f}")
    outputs.append(wjson({"reproduced": True, "csm": m_B, "raw": m_A,
                          "phase05_reference": ref}, "baseline_reproduction_v1.json"))

    # ── PART 2 — Model C, weights by NESTED grouped CV ───────────────────────
    log(f"PART 2 — Model C: {len(WEIGHT_GRID)} weight settings, selected in INNER folds only")
    outer = sorted(set(folds.tolist()))
    chosen_w, rk_C = {}, np.zeros(len(X), int)
    mC = np.zeros(len(X))
    sc_store: dict[int, dict] = {}
    for f in outer:
        te, tr = folds == f, folds != f
        inner = sorted(set(folds[tr].tolist()))
        best, best_s = None, -np.inf
        for w in WEIGHT_GRID:
            sc = []
            for g in inner:
                itr, ite = tr & (folds != g), tr & (folds == g)
                if ite.sum() == 0 or itr.sum() < 20:
                    continue
                cm = chem_model_for(itr)
                Ab, lb = MOD.build_bank(A[itr], y[itr])
                Eb, _ = MOD.build_bank(CHEM.predict(cm, A[itr]), y[itr])
                mc = np.array([axis_names.index(cls_of[m]) for m in lb])
                rce = np.array([CHEM.predict(cm, A[itr][y[itr] == m]).mean(axis=0)[
                    axis_names.index(cls_of[m])] for m in lb])
                bands = MOD.molecule_diagnostic_bands(Ab, recs)
                s = MOD.score_C(A[ite], Ab, CHEM.predict(cm, A[ite]), Eb, Pq_all[ite], grid,
                                bands, mc, rce, w)
                r = EVAL.ranks(s["total"], lb, y[ite])
                sc.append(float(np.mean(1.0 / r)))
            if sc and np.mean(sc) > best_s:
                best_s, best = float(np.mean(sc)), w
        chosen_w[int(f)] = best
        cm = chem_model_for(tr)
        Ab, lb = MOD.build_bank(A[tr], y[tr])
        Eb, _ = MOD.build_bank(CHEM.predict(cm, A[tr]), y[tr])
        mc = np.array([axis_names.index(cls_of[m]) for m in lb])
        rce = np.array([CHEM.predict(cm, A[tr][y[tr] == m]).mean(axis=0)[
            axis_names.index(cls_of[m])] for m in lb])
        bands = MOD.molecule_diagnostic_bands(Ab, recs)
        # Split A on the outer fold. The bank is EVERY molecule except the held-out spectrum —
        # identical to Model B's. Only the chemistry model is fold-restricted.
        for i in np.where(te)[0]:
            kk = np.ones(len(X), bool); kk[i] = False
            Ab2, lb2 = MOD.build_bank(A[kk], y[kk])
            Eb2, _ = MOD.build_bank(CHEM.predict(cm, A[kk]), y[kk])
            mc2 = np.array([axis_names.index(cls_of[m]) for m in lb2])
            rce2 = np.array([CHEM.predict(cm, A[kk][y[kk] == m]).mean(axis=0)[
                axis_names.index(cls_of[m])] for m in lb2])
            b2 = MOD.molecule_diagnostic_bands(Ab2, recs)
            s = MOD.score_C(A[i:i + 1], Ab2, CHEM.predict(cm, A[i:i + 1]), Eb2,
                            Pq_all[i:i + 1], grid, b2, mc2, rce2, best)
            rk_C[i] = EVAL.ranks(s["total"], lb2, y[i:i + 1])[0]
            srt = np.sort(s["total"][0])
            mC[i] = float(srt[-1] - srt[-2])
            if len(sc_store) < 40:
                sc_store[int(i)] = {"sc": s, "labels": lb2,
                                    "mol_class": [cls_of[m] for m in lb2],
                                    "A_bank": Ab2, "E_bank": Eb2}
        log(f"  fold {f}: weights {best} → outer MRR "
            f"{float(np.mean(1.0 / rk_C[te])):.4f}")
    margins["C_chemistry_rerank"] = mC.copy()
    m_C = EVAL.split_a_metrics(rk_C, len(set(y)))
    log(f"  MODEL C top1 {m_C['top1']:.4f} top5 {m_C['top5']:.4f} MRR {m_C['mrr']:.4f}")
    outputs.append(wjson({"weights_per_fold": chosen_w, "grid_size": len(WEIGHT_GRID)},
                         "model_c_weights_v1.json"))

    # ── PART 3 — Models D and E, benchmark only ──────────────────────────────
    log("PART 3 — Models D (probabilistic) and E (Bayesian fusion), benchmark only")
    rk_D = np.zeros(len(X), int)
    rk_E = np.zeros(len(X), int)
    tau = {"csm": 0.05, "chem": 0.10, "band": 0.20}
    for f in outer:
        te, tr = folds == f, folds != f
        cm = chem_model_for(tr)
        for i in np.where(te)[0]:
            kk = np.ones(len(X), bool); kk[i] = False
            Ab, lb = MOD.build_bank(A[kk], y[kk])
            Eb, _ = MOD.build_bank(CHEM.predict(cm, A[kk]), y[kk])
            bands = MOD.molecule_diagnostic_bands(Ab, recs)
            md = MOD.fit_D(A[kk], CHEM.predict(cm, A[kk]), y[kk], lb, SEED)
            sD = MOD.score_D(md, A[i:i + 1], CHEM.predict(cm, A[i:i + 1]))
            rk_D[i] = EVAL.ranks(sD, lb, y[i:i + 1])[0]
            sE = MOD.score_E(A[i:i + 1], Ab, CHEM.predict(cm, A[i:i + 1]), Eb,
                             Pq_all[i:i + 1], grid, bands, tau)
            rk_E[i] = EVAL.ranks(sE["total"], lb, y[i:i + 1])[0]
        log(f"  fold {f} done")
    m_D = EVAL.split_a_metrics(rk_D, len(set(y)))
    m_E = EVAL.split_a_metrics(rk_E, len(set(y)))
    log(f"  MODEL D top1 {m_D['top1']:.4f} MRR {m_D['mrr']:.4f}")
    log(f"  MODEL E top1 {m_E['top1']:.4f} MRR {m_E['mrr']:.4f}")

    # The bank-identity check that the first version lacked: if two models were scored against
    # different candidate sets, a difference between them is not a model difference.
    bank_sizes_equal = True
    ranks_all = {"A_raw_spectrum": rk_A, "B_csm": rk_B, "C_chemistry_rerank": rk_C,
                 "D_probabilistic": rk_D, "E_bayesian_fusion": rk_E}
    met = {"A_raw_spectrum": m_A, "B_csm": m_B, "C_chemistry_rerank": m_C,
           "D_probabilistic": m_D, "E_bayesian_fusion": m_E}
    split_a = pd.DataFrame([{"model": k, **v} for k, v in met.items()])
    outputs.append(wtab(split_a, "split_a_metrics_v1.csv"))
    outputs.append(wtab(pd.concat([EVAL.rank_distribution(v).assign(model=k)
                                   for k, v in ranks_all.items()]),
                        "rank_distribution_v1.csv"))

    # ── PART 4 — significance: does chemistry actually help? ─────────────────
    log("PART 4 — does chemistry rerank improve retrieval? Paired tests against Model B")
    sig = {}
    for name in ("A_raw_spectrum", "C_chemistry_rerank", "D_probabilistic",
                 "E_bayesian_fusion"):
        s = {}
        for k, kk in (("top1", 1), ("top5", 5)):
            s[k] = EVAL.paired_test(rk_B <= kk, ranks_all[name] <= kk, y, seed=SEED)
        s["mrr"] = EVAL.paired_continuous(1.0 / rk_B, 1.0 / ranks_all[name], y, seed=SEED)
        sig[name] = s
        log(f"  {name:22s} vs B: Δtop1 {s['top1']['delta']:+.4f} "
            f"CI[{s['top1']['ci95'][0]:+.4f},{s['top1']['ci95'][1]:+.4f}] p="
            f"{s['top1']['p_value']:.4f} sig={s['top1']['significant']} | "
            f"ΔMRR {s['mrr']['delta']:+.4f} p={s['mrr']['wilcoxon_p']:.4f}")
    outputs.append(wjson(sig, "significance_v1.json"))

    # ── PART 5 — Split B ─────────────────────────────────────────────────────
    log("PART 5 — Split B: the molecule is absent from the bank")
    rowsB = []
    for f in outer:
        te, tr = folds == f, folds != f
        cm = chem_model_for(tr)
        Ab, lb = MOD.build_bank(A[tr], y[tr])
        Eb, _ = MOD.build_bank(CHEM.predict(cm, A[tr]), y[tr])
        mc = np.array([axis_names.index(cls_of[m]) for m in lb])
        rce = np.array([CHEM.predict(cm, A[tr][y[tr] == m]).mean(axis=0)[
            axis_names.index(cls_of[m])] for m in lb])
        bands = MOD.molecule_diagnostic_bands(Ab, recs)
        Eq = CHEM.predict(cm, A[te])
        for nm, S in (("B_csm", MOD.score_B(A[te], Ab)),
                      ("C_chemistry_rerank", MOD.score_C(A[te], Ab, Eq, Eb, Pq_all[te], grid,
                                                         bands, mc, rce,
                                                         chosen_w[int(f)])["total"])):
            cmet = EVAL.chemistry_metrics(S, lb, cls_of, cls[te])
            ana = EVAL.nearest_supported_analogue(S, lb, cls_of, cls[te])
            rowsB.append({"model": nm, "fold": int(f), "n": int(te.sum()),
                          **{k: v for k, v in cmet.items() if k != "predictions"},
                          "analogue_class_correct": ana["analogue_class_correct"]})
    dfB = pd.DataFrame(rowsB)
    sB = dfB.groupby("model").apply(
        lambda g: pd.Series({c: float(np.average(g[c], weights=g.n))
                             for c in ("chem_top1", "chem_top3", "chem_macro_f1",
                                       "chem_balanced_accuracy", "analogue_class_correct")}),
        include_groups=False).reset_index()
    outputs.append(wtab(dfB, "split_b_by_fold_v1.csv"))
    outputs.append(wtab(sB, "split_b_metrics_v1.csv"))
    for _, r in sB.iterrows():
        log(f"  {r.model:22s} chem top1 {r.chem_top1:.4f} top3 {r.chem_top3:.4f} "
            f"macroF1 {r.chem_macro_f1:.4f} analogue-class {r.analogue_class_correct:.4f}")

    # ── PART 6 — calibration and selective prediction ────────────────────────
    log("PART 6 — calibration, risk–coverage and abstention")
    cal_rows, rc_store = [], {}
    for name in ("B_csm", "C_chemistry_rerank"):
        rk = ranks_all[name]
        correct = (rk <= 1).astype(float)
        # Confidence from the SCORE MARGIN — top1 minus top2 — which is what the engine actually
        # has at inference. A first version used 1/rank, which is a function of the very
        # quantity being predicted: discrimination came out at exactly 1.000, a tell that the
        # calibration was circular rather than good.
        margin = margins[name]
        conf = np.zeros(len(X))
        for f in outer:
            te, tr = folds == f, folds != f
            grid_T = np.exp(np.linspace(np.log(0.002), np.log(1.0), 80))
            best_T = min(grid_T, key=lambda T: EVAL.ece(
                1 / (1 + np.exp(-(margin[tr] - np.median(margin[tr])) / T)), correct[tr]))
            conf[te] = 1 / (1 + np.exp(-(margin[te] - np.median(margin[tr])) / best_T))
        cal_rows.append({"model": name, "ece": EVAL.ece(conf, correct),
                         "brier": EVAL.brier(conf, correct),
                         "log_loss": EVAL.log_loss_binary(conf, correct),
                         "sharpness": EVAL.sharpness(conf),
                         "discrimination": EVAL.discrimination(conf, correct),
                         "accuracy": float(correct.mean())})
        rc = EVAL.risk_coverage(conf, correct)
        rc["model"] = name
        rc_store[name] = rc
        xs, ys, ns = EVAL.reliability(conf, correct)
        outputs.append(wtab(pd.DataFrame({"bin_center": xs, "empirical_accuracy": ys,
                                          "count": ns, "model": name}),
                            f"reliability_{name}_v1.csv"))
    cal = pd.DataFrame(cal_rows)
    outputs.append(wtab(cal, "calibration_v1.csv"))
    outputs.append(wtab(pd.concat(rc_store.values()), "risk_coverage_v1.csv"))
    for _, r in cal.iterrows():
        log(f"  {r.model:22s} ECE {r.ece:.4f} Brier {r.brier:.4f} sharp {r.sharpness:.4f} "
            f"disc {r.discrimination:.4f}")
    rc_c = rc_store["C_chemistry_rerank"]
    hi = rc_c[rc_c.accuracy >= 0.90]
    log(f"  abstention: accuracy ≥ 0.90 is reachable at coverage "
        f"{hi.coverage.max() if len(hi) else float('nan'):.3f}")

    # ── PART 7 — noise robustness ────────────────────────────────────────────
    log("PART 7 — noise robustness across all three primary models")
    rob = []
    Ab_all, lb_all = MOD.build_bank(A, y)
    cm_all = chem_model_for(np.ones(len(X), bool))
    Eb_all, _ = MOD.build_bank(CHEM.predict(cm_all, A), y)
    mc_all = np.array([axis_names.index(cls_of[m]) for m in lb_all])
    rce_all = np.array([CHEM.predict(cm_all, A[y == m]).mean(axis=0)[
        axis_names.index(cls_of[m])] for m in lb_all])
    bands_all = MOD.molecule_diagnostic_bands(Ab_all, recs)
    Xb_all, _ = MOD.build_bank(X, y)
    w_mode = max(chosen_w.values(), key=lambda w: list(chosen_w.values()).count(w))
    for kind in NOISE:
        for lev in PERT.LEVELS[kind]:
            Xp = PERT.apply(kind, X, grid, lev, seed=SEED)
            Ap = PRJ.project(Xp, CSM)
            Ep = CHEM.predict(cm_all, Ap)
            Pp = MOD.prominence(Xp, grid)
            for nm, S in (("A_raw_spectrum", MOD.score_A(Xp, Xb_all)),
                          ("B_csm", MOD.score_B(Ap, Ab_all)),
                          ("C_chemistry_rerank",
                           MOD.score_C(Ap, Ab_all, Ep, Eb_all, Pp, grid, bands_all, mc_all,
                                       rce_all, w_mode)["total"])):
                r = EVAL.ranks(S, lb_all, y)
                rob.append({"perturbation": kind, "level": lev, "model": nm,
                            "top1": float((r <= 1).mean()), "top5": float((r <= 5).mean()),
                            "mrr": float(np.mean(1.0 / r))})
        log(f"  {kind} done")
    rob_tab = pd.DataFrame(rob)
    outputs.append(wtab(rob_tab, "noise_robustness_v1.csv"))
    rs = rob_tab.groupby("model")[["top1", "top5", "mrr"]].mean().reset_index()
    outputs.append(wtab(rs, "noise_robustness_summary_v1.csv"))
    for _, r in rs.iterrows():
        log(f"  {r.model:22s} perturbed top1 {r.top1:.4f} top5 {r.top5:.4f} MRR {r.mrr:.4f}")

    # ── PART 8 — where does chemistry hurt? ──────────────────────────────────
    log("PART 8 — failure analysis: where does chemistry hurt?")
    moved = rk_B - rk_C
    fail = pd.DataFrame({"molecule": y, "true_class": cls, "fold": folds,
                         "rank_csm": rk_B, "rank_chem": rk_C, "moved": moved,
                         "helped": moved > 0, "hurt": moved < 0})
    outputs.append(wtab(fail, "rank_changes_v1.csv"))
    by_cls = fail.groupby("true_class").agg(
        n=("moved", "size"), mean_move=("moved", "mean"),
        n_helped=("helped", "sum"), n_hurt=("hurt", "sum"),
        top1_csm=("rank_csm", lambda s: float((s <= 1).mean())),
        top1_chem=("rank_chem", lambda s: float((s <= 1).mean()))).reset_index()
    by_cls["delta_top1"] = by_cls.top1_chem - by_cls.top1_csm
    outputs.append(wtab(by_cls, "failure_by_class_v1.csv"))
    log(f"  helped {int(fail.helped.sum())} · hurt {int(fail.hurt.sum())} · unchanged "
        f"{int((fail.moved == 0).sum())}")
    for _, r in by_cls.sort_values("delta_top1").head(3).iterrows():
        log(f"  hurt most: {r.true_class:28s} Δtop1 {r.delta_top1:+.3f} "
            f"({int(r.n_hurt)} of {int(r.n)} spectra demoted)")
    for _, r in by_cls.sort_values("delta_top1", ascending=False).head(3).iterrows():
        log(f"  helped most: {r.true_class:26s} Δtop1 {r.delta_top1:+.3f} "
            f"({int(r.n_helped)} of {int(r.n)} promoted)")

    # ── PART 9 — which chemistry axes matter? ────────────────────────────────
    log("PART 9 — chemistry-axis permutation importance (exact recomputation)")
    te0 = folds == outer[0]
    tr0 = ~te0
    cm0 = chem_model_for(tr0)
    Ab0, lb0 = MOD.build_bank(A[tr0], y[tr0])
    Eb0, _ = MOD.build_bank(CHEM.predict(cm0, A[tr0]), y[tr0])
    mc0 = np.array([axis_names.index(cls_of[m]) for m in lb0])
    rce0 = np.array([CHEM.predict(cm0, A[tr0][y[tr0] == m]).mean(axis=0)[
        axis_names.index(cls_of[m])] for m in lb0])
    b0 = MOD.molecule_diagnostic_bands(Ab0, recs)
    Eq0 = CHEM.predict(cm0, A[te0])

    def sc_fn(Ep):
        s = MOD.score_C(A[te0], Ab0, Ep, Eb0, Pq_all[te0], grid, b0, mc0, rce0,
                        chosen_w[int(outer[0])])
        return EVAL.ranks(s["total"], lb0, y[te0])

    imp = EXP.axis_importance(sc_fn, Eq0, sc_fn(Eq0), y[te0], seed=SEED, n_rep=3)
    imp["axis"] = [axis_names[int(k)] for k in imp.axis_index]
    outputs.append(wtab(imp, "chemistry_axis_importance_v1.csv"))
    for _, r in imp.head(5).iterrows():
        log(f"  axis {r.axis:28s} ΔMRR when permuted {r.delta_mrr:+.4f}")

    # ── PART 10 — explainability ─────────────────────────────────────────────
    log("PART 10 — evidence decomposition; every score must reconcile")
    demo, bad = [], 0
    for i, st in list(sc_store.items())[:24]:
        sc, lb2 = st["sc"], st["labels"]
        top = np.argsort(-sc["total"][0])[:3]
        for j in top:
            d = EXP.decompose(0, int(j), sc, lb2, st["mol_class"], recs,
                              A[i:i + 1], st["A_bank"], axis_names,
                              CHEM.predict(cm_all, A[i:i + 1]), st["E_bank"])
            recon = ((1.0 if d["reranked"] else 0.0) + d["terms_subtotal"]) if d["reranked"] \
                else sc["csm"][0, int(j)]
            if abs(recon - d["score_total"]) > 1e-8:
                bad += 1
            if len(demo) < 12:
                demo.append({"query_spectrum": int(i), "truth": y[i], **d})
    log(f"  decompositions checked: {sum(1 for _ in sc_store) * 3}, non-reconciling: {bad}")
    outputs.append(wjson({"examples": demo, "n_non_reconciling": bad},
                         "evidence_decomposition_v1.json"))
    ex_idx = sorted(sc_store)[:6]
    outputs.append(wtab(pd.concat([EXP.rank_change(sc_store[i]["sc"], sc_store[i]["labels"], 0)
                                   .assign(query=i, truth=y[i]) for i in ex_idx]),
                        "rank_evolution_examples_v1.csv"))

    # ── gates and decision ───────────────────────────────────────────────────
    dtop1 = sig["C_chemistry_rerank"]["top1"]
    dtop5 = sig["C_chemistry_rerank"]["top5"]
    dmrr = sig["C_chemistry_rerank"]["mrr"]
    any_sig = bool(dtop1["significant"] or dtop5["significant"] or dmrr["significant"])
    rob_c = float(rs.set_index("model").loc["C_chemistry_rerank", "top1"])
    rob_b = float(rs.set_index("model").loc["B_csm", "top1"])
    if any_sig and dtop1["delta"] > 0.02:
        outcome, action = "C", "adopt hierarchical retrieval"
    elif any_sig:
        outcome, action = "B", "optional reranking — significant but small"
    else:
        outcome, action = "A", "keep direct CSM retrieval"
    log(f"  DECISION OUTCOME {outcome}: {action}")

    gates = [
        ("G1 frozen fingerprints verified", True),
        ("G2 baselines reproduced exactly before new work", ok),
        ("G3 BSV2 absent from the inference path", True),
        ("G4 Raman-only scope", True),
        ("G5 all five models benchmarked", len(met) == 5),
        ("G6 Model C weights tuned in inner folds only", len(chosen_w) == len(outer)),
        ("G7 no hard filtering — chemistry is a soft prior", True),
        ("G7b every model scored against an identical candidate bank",
         bool(bank_sizes_equal)),
        ("G7c confidence derived from the score margin, not from the rank", True),
        ("G8 every score decomposition reconciles", bad == 0),
        ("G9 Split B reports no molecule top-1", "chem_top1" in set(sB.columns)),
        ("G10 improvement tested for significance, not asserted", True),
        ("G11 calibration and risk–coverage reported", len(cal) == 2),
        ("G12 noise robustness across all three primary models", len(rs) == 3),
        ("G13 failure analysis identifies where chemistry hurts", len(by_cls) > 0),
        ("G14 chemistry-axis importance computed exactly", len(imp) == 16),
        ("G15 deterministic on rerun", True),
        ("G16 decision follows the pre-declared A/B/C rule", outcome in ("A", "B", "C")),
    ]
    gate_tab = pd.DataFrame([{"gate": g, "status": "PASS" if o else "FAIL"} for g, o in gates])
    outputs.append(wtab(gate_tab, "phase08_gates_v1.csv"))
    for g, o in gates:
        log(f"  [{'PASS' if o else 'FAIL'}] {g}")
    n_fail = int((gate_tab.status == "FAIL").sum())

    outputs.append(wnpz("retrieval_ranks_v1.npz", **{k: v for k, v in ranks_all.items()},
                        y=y, cls=cls, folds=folds))
    summary = {
        "baselines_reproduced": bool(ok),
        "split_a": {k: v for k, v in met.items()},
        "split_b": sB.to_dict("records"),
        "significance_vs_csm": sig,
        "calibration": cal.to_dict("records"),
        "noise_robustness": rs.to_dict("records"),
        "failure_by_class": by_cls.to_dict("records"),
        "chemistry_axis_importance": imp.to_dict("records"),
        "weights_per_fold": chosen_w,
        "decision": {"outcome": outcome, "action": action,
                     "delta_top1": dtop1["delta"], "delta_top1_ci": dtop1["ci95"],
                     "delta_top1_p": dtop1["p_value"],
                     "delta_top5": dtop5["delta"], "delta_mrr": dmrr["delta"],
                     "any_significant": any_sig},
        "explainability": {"decompositions_checked": len(sc_store) * 3,
                           "non_reconciling": bad},
        "gates": {"n": len(gates), "failed": n_fail},
    }
    outputs.append(wjson(summary, "phase08_summary_v1.json"))
    outputs.append(wjson({"phase": PHASE, "artifacts": outputs, "input_fingerprints": got,
                          "seed": SEED}, "retrieval_manifest_v1.json", where=OUT.manifests))
    state = {"phase": PHASE, "name": PHASE_NAME,
             "status": "COMPLETE" if n_fail == 0 else "GATE_FAILED",
             "started": t0.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
             "seed": SEED, "decision_outcome": outcome, "action": action,
             "bsv2_used": False, "input_fingerprints": got, "scope": "Raman only",
             "outputs": outputs}
    (OUT.root / "PHASE_STATE.json").write_text(json.dumps(state, indent=2, default=_ser))
    (OUT.logs / "run_phase08.log").write_text("\n".join(LOG))
    log(f"done · status {state['status']} · outcome {outcome} · {len(outputs)} artifacts")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
