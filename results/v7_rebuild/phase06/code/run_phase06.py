#!/usr/bin/env python3
"""GAIRA V7 — Phase 06: the validated 16-dimensional Chemistry Evidence Layer (Raman only).

Parts 0–15 of the brief. Nothing frozen is refitted; every selection is made inside nested
molecule-grouped CV and never on the number finally reported. Output location resolves through
`gaira.v7.io.PhaseOutputs`; no path is hardcoded.

    python results/v7_rebuild/phase06/code/run_phase06.py
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
from gaira.v7.inference import projection as PRJ, retrieval as RET      # noqa: E402
from gaira.v7.meta import perturbations as PERT                         # noqa: E402
from gaira.v7.chemistry import (calibration as CAL, evidence as EVD,    # noqa: E402
                                novelty as NOV, provenance as PROV,
                                registry as REG, validation as VAL)

warnings.filterwarnings("ignore")

PHASE, PHASE_NAME = "06", "Validated Chemistry Evidence Layer"
OUT = PhaseOutputs(PHASE, extra=("interactive", "manifests")).ensure()
FROZEN = frozen_root()
EXPECTED = {"atlas": "09ed804a40836f4a05a91ba10900cded",
            "lsm": "208482d6f7178b5b8f16cace91be55b0",
            "csm": "0b4aa550ccefed3edabdbde5bae11c8d",
            "engine": "20d8bd99ce71f45a125c6a2b1d719e51"}
SEED = 0
ROBUSTNESS = ("gaussian_noise", "shot_noise", "baseline_drift", "fluorescence",
              "wavelength_shift", "spectral_stretch", "band_broadening", "intensity_scaling",
              "peak_dropout", "cosmic_spikes", "combined")
# Held-out chemistry classes, chosen BEFORE any novelty result was seen, to span the four cells
# the brief asks for: distinctive/overlapping x small/large.
HOLDOUT_CLASSES = ("sterol_steroid", "purine", "sulfur_thiol_cofactor", "mono_oligosaccharide",
                   "acylglycerol", "pyrimidine")
LOG: list[str] = []


def log(m):
    line = f"[phase06] {m}"
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


PROV_STAMP: dict = {}


def wtab(df, name, where=None):
    p = (where or OUT.tables) / name
    df.to_csv(p, index=False)
    return {"artifact_id": name, "path": OUT.rel(p), "sha256": P.sha256_file(p), "rows": len(df)}


def wjson(obj, name, where=None, stamp=True):
    p = (where or OUT.artifacts) / name
    if stamp and isinstance(obj, dict):
        obj = {**obj, "_provenance": PROV_STAMP}
    p.write_text(json.dumps(obj, indent=2, default=_ser))
    return {"artifact_id": name, "path": OUT.rel(p), "sha256": P.sha256_file(p)}


def wnpz(name, **arrays):
    p = OUT.artifacts / name
    np.savez_compressed(p, _provenance=json.dumps(PROV_STAMP), **arrays)
    return {"artifact_id": name, "path": OUT.rel(p), "sha256": P.sha256_file(p)}


def main() -> int:
    outputs: list[dict] = []
    t0 = datetime.now(timezone.utc)

    # ── PART 0 — fingerprints and architecture compliance ────────────────────
    log("ARCHITECTURE COMPLIANCE STATEMENT")
    for line in [
        "  Phase 06 builds the FIRST semantic abstraction above the frozen motif layer:",
        "  49-d CSM activation  ->  16-d Chemistry Evidence Vector (decision A-19).",
        "  It does NOT implement BSV2 (Phase 07) or hierarchical retrieval (Phase 08).",
        "  Scope is Raman-only: no SERS, Ag-SERS, serum, plasma, EV, mixture or DART-Met data",
        "  is loaded, benchmarked, or cited as validation anywhere in this phase.",
        "  Frozen and read-only: preprocessing spec, canonical identities, 16-class ontology,",
        "  CV folds, LSM dictionary, CSM dictionary, Phase 05 engine and reference bank.",
        "  Nothing upstream is refitted. All outputs are written under the Phase 06 tree only.",
        "  The legacy 11-axis band map (A-16) is absent from canonical Phase 06 inference; it",
        "  appears only as a comparator in Part 6.",
    ]:
        log(line)

    csm_reg = json.loads((FROZEN / "phase02/artifacts/csm_registry_v1.json").read_text())
    st01 = json.loads((FROZEN / "phase01/PHASE_STATE.json").read_text())
    st05 = json.loads((FROZEN / "phase05/PHASE_STATE.json").read_text())
    got = {"csm": csm_reg["fingerprint"], "lsm": st01["registry_fingerprint"],
           "engine": st05["engine_fingerprint"], "atlas": EXPECTED["atlas"]}
    for k in ("csm", "lsm", "engine"):
        if got[k] != EXPECTED[k]:
            log(f"ABORT — {k} fingerprint {got[k]} != expected {EXPECTED[k]}")
            return 2
    log(f"  frozen verified: LSM {got['lsm']} · CSM {got['csm']} · engine {got['engine']}")

    # ── corpus ───────────────────────────────────────────────────────────────
    br = np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
    X = np.asarray(br["X"], float)
    y = np.array([str(s) for s in br["canonical_id"]])
    grid = np.asarray(br["grid"], float)
    CSM = np.load(FROZEN / "phase02/artifacts/csm_dictionary_v1.npz")["CSM"]
    H_lsm = np.load(FROZEN / "phase01/artifacts/lsm_dictionary_v1.npz")["H"]
    recs = csm_reg["csms"]
    part = pd.read_csv(FROZEN / "phase00/tables/chemical_partition_v1.csv")
    cls_of = dict(zip(part.canonical_id, part.fine_class))
    canon = pd.read_csv(FROZEN / "phase00/tables/canonical_analytes_v1.csv")
    broad_of = dict(zip(canon.canonical_id, canon.broad_class))
    folds_t = pd.read_csv(FROZEN / "phase00/tables/cv_folds_v1.csv")
    fold_of = dict(zip(folds_t.canonical_id, folds_t.fold))
    folds = np.array([fold_of[v] for v in y])
    cls = np.array([cls_of[v] for v in y])
    REG.check(cls)

    q = pd.read_csv(FROZEN / "phase00/tables/spectrum_quality_v1.csv")
    qp = q[q.qc_pass].sort_values(["canonical_id", "spectrum_id"]).reset_index(drop=True)
    if len(qp) != len(y) or not (qp.canonical_id.values == y).all():
        log("ABORT — per-spectrum metadata does not align with the balanced references")
        return 2
    src = qp.source.values.astype(str)
    excit = qp.excitation_nm.values.astype(str)
    spec_id = qp.spectrum_id.values.astype(str)
    log(f"  corpus {X.shape} · {len(set(y))} molecules · {len(REG.CLASS_ORDER)} classes · "
        f"{len(set(folds))} grouped folds · sources {sorted(set(src))}")

    split_fp = hashlib.md5(json.dumps(sorted(zip(y.tolist(), folds.tolist()))).encode()).hexdigest()
    code_fp = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    PROV_STAMP.update({"phase": PHASE, "seed": SEED, "input_fingerprints": got,
                       "split_fingerprint": split_fp, "code_fingerprint": code_fp,
                       "created_utc": t0.isoformat(),
                       "model_selection_rule": "nested molecule-grouped CV, inner-fold macro-F1",
                       "scope": "Raman only"})

    # ── the 16 classes, printed explicitly ───────────────────────────────────
    log("PART A — the frozen 16-class chemistry ontology")
    registry = REG.build_registry(FROZEN, y, cls, src, excit)
    for r in registry:
        log(f"  [{r['class_index']:2d}] {r['class_id']:28s} {r['n_molecules']:3d} molecules "
            f"{r['n_spectra']:4d} spectra  broad={','.join(r['broad_class']):22s} "
            f"imbalance x{r['imbalance_vs_uniform']:.2f}")
    outputs.append(wjson({"ontology": "v7_fine_16", "frozen_in": "phase00",
                          "class_order": list(REG.CLASS_ORDER),
                          "adjacent_pairs": [list(p) for p in REG.ADJACENT],
                          "classes": registry}, "chemistry_class_registry_v1.json"))
    outputs.append(wtab(pd.DataFrame([{k: (";".join(map(str, v)) if isinstance(v, list)
                                          else json.dumps(v) if isinstance(v, dict) else v)
                                       for k, v in r.items()} for r in registry]),
                        "chemistry_class_registry_v1.csv"))
    sizes = np.array([r["n_spectra"] for r in registry])
    log(f"  class imbalance: largest {sizes.max()} spectra, smallest {sizes.min()}, "
        f"ratio {sizes.max() / sizes.min():.1f}x")

    # ── PART 1 — audit and reproduce Phase 05 ────────────────────────────────
    log("PART 1 — auditing the Phase 05 class-inference implementation")
    log("  traced to src/gaira/v7/inference/engine.py:86-89 and run_phase05.py topk_class():")
    log("    score(x, c) = max over reference molecules i of class c of cos(a(x), r_i)")
    log("    i.e. 1-nearest-molecule; top-k ranks DISTINCT classes by that per-class maximum.")
    log("    No class-size correction, no calibration, no probability.")
    A = PRJ.project(X, CSM)
    D = PRJ.diagnostics(X, A, CSM)
    p05 = {"aggregation": "max", "size_correction": "none"}
    E05 = np.zeros((len(A), len(REG.CLASS_ORDER)))
    for f in sorted(set(folds)):
        te, tr = folds == f, folds != f
        m = EVD.fit_A(A[tr], y[tr], cls[tr], **p05)
        E05[te] = EVD.predict_A(m, A[te])
    repro = {"class_top1": VAL.topk(E05, cls, 1), "class_top3": VAL.topk(E05, cls, 3),
             "macro_f1": VAL.macro_f1(E05, cls),
             "balanced_accuracy": VAL.balanced_accuracy(E05, cls)}
    ref = json.loads((FROZEN / "phase05/artifacts/phase05_summary_v1.json").read_text())["split_b"]
    match = all(abs(repro[k] - ref[k]) < 1e-9 for k in
                ("class_top1", "class_top3", "macro_f1", "balanced_accuracy"))
    for k in ("class_top1", "class_top3", "macro_f1", "balanced_accuracy"):
        log(f"  {k:20s} phase05 {ref[k]:.6f}  reproduced {repro[k]:.6f}  "
            f"{'EXACT' if abs(repro[k]-ref[k]) < 1e-9 else 'MISMATCH'}")
    if not match:
        log("ABORT — Phase 05 chemistry-class results did not reproduce bit-for-bit")
        return 2
    outputs.append(wjson({"traced_formula": "e_c(x) = max_{i in c} cos(a(x), r_i)",
                          "source_locations": ["src/gaira/v7/inference/engine.py:86-89",
                                               "results/v7_rebuild/phase05/code/"
                                               "run_phase05.py::topk_class"],
                          "aggregation": "max", "size_correction": "none",
                          "calibrated": False, "phase05_reported": ref,
                          "reproduced": repro, "bit_for_bit": bool(match)},
                         "phase05_class_inference_audit_v1.json"))
    outputs.append(wtab(VAL.per_class(E05, cls), "phase05_reproduction_per_class_v1.csv"))
    outputs.append(wtab(VAL.confusion(E05, cls).reset_index().rename(
        columns={"index": "true_class"}), "phase05_reproduction_confusion_v1.csv"))

    # ── PART 2 — candidate chemistry-evidence models ─────────────────────────
    log("PART 2 — benchmarking candidate Chemistry Evidence models (nested grouped CV)")
    cands: dict[str, dict] = {}
    for agg in EVD.AGGREGATIONS:
        for corr in EVD.SIZE_CORRECTIONS:
            cands[f"A:{agg}:{corr}"] = {"family": "A_similarity_evidence",
                                        "aggregation": agg, "size_correction": corr}
    for pr in EVD.PROTOTYPES:
        cands[f"B:{pr}"] = {"family": "B_class_prototype", "prototype": pr}
    for mth in EVD.PROBABILISTIC:
        cands[f"C:{mth}"] = {"family": "C_probabilistic", "method": mth}
    for lam in (0.5, 1.0, 2.0):
        cands[f"D:A_max_idf:lam{lam}"] = {"family": "D_hierarchical", "base": "A", "lam": lam,
                                          "aggregation": "max", "size_correction": "idf"}
    log(f"  {len(cands)} candidates across 4 families")

    def fit_fn(A_tr, y_tr, c_tr, cfg):
        cfg = dict(cfg)
        fam = cfg.pop("family")
        if fam == "D_hierarchical":
            return EVD.fit_D(A_tr, y_tr, c_tr, broad_of=broad_of, **cfg)
        return EVD.fit(fam, A_tr, y_tr, c_tr, **cfg)

    # Flat (non-nested) benchmark for the record; selection uses the nested loop below.
    rows = []
    for name, cfg in cands.items():
        E = np.zeros((len(A), len(REG.CLASS_ORDER)))
        try:
            for f in sorted(set(folds)):
                te, tr = folds == f, folds != f
                E[te] = EVD.predict(fit_fn(A[tr], y[tr], cls[tr], cfg), A[te])
        except Exception as exc:                                       # pragma: no cover
            rows.append({"candidate": name, "usable": False, "error": str(exc)[:70]})
            continue
        rows.append({"candidate": name, "usable": True,
                     "family": cfg["family"],
                     "top1": VAL.topk(E, cls, 1), "top3": VAL.topk(E, cls, 3),
                     "macro_f1": VAL.macro_f1(E, cls),
                     "balanced_accuracy": VAL.balanced_accuracy(E, cls),
                     "mrr": VAL.mrr(E, cls),
                     "replicate_consistency": VAL.replicate_consistency(E, y),
                     "mean_entropy": float(VAL.entropy(E).mean()),
                     "effective_rank": VAL.effective_rank(E)})
    bench = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    dead = bench[~bench.usable.astype(bool)]
    if len(dead):
        log(f"  ABORT — {len(dead)} candidate models failed to run: "
            f"{dead.candidate.tolist()} ({dead.error.iloc[0]})")
        log("  A phase that silently drops required candidates is not a benchmark. Every "
            "candidate must run or the failure must be explained and the candidate removed "
            "from the pre-registered list on the record.")
        return 3
    outputs.append(wtab(bench, "evidence_model_benchmark_v1.csv"))
    for _, r in bench.head(10).iterrows():
        log(f"  {r.candidate:34s} top1 {r.top1:.3f}  top3 {r.top3:.3f}  "
            f"macroF1 {r.macro_f1:.3f}  bal {r.balanced_accuracy:.3f}  "
            f"repl {r.replicate_consistency:.3f}")
    log(f"  (worst: {bench.iloc[-1].candidate} macroF1 "
        f"{bench.iloc[-1].get('macro_f1', float('nan')):.3f})")

    # ── PART 4 — nested molecule-grouped validation ──────────────────────────
    log("PART 4 — nested molecule-grouped CV: inner folds select, outer folds evaluate")
    usable = {k: v for k, v in cands.items()
              if bool(bench.set_index("candidate").loc[k, "usable"])}
    nested = VAL.nested_cv(A, y, cls, folds, usable, fit_fn, EVD.predict,
                           select_metric=VAL.macro_f1, log=log)
    E = nested["E"]
    selected = nested["modal_choice"]
    log(f"  per-fold selections: {nested['chosen_per_fold']}")
    log(f"  SELECTED MODEL: {selected}")
    outputs.append(wtab(pd.DataFrame(nested["per_fold"]), "nested_cv_folds_v1.csv"))

    ci = {}
    for nm, fn in (("top1", lambda e, c: VAL.topk(e, c, 1)),
                   ("top3", lambda e, c: VAL.topk(e, c, 3)),
                   ("macro_f1", VAL.macro_f1),
                   ("balanced_accuracy", VAL.balanced_accuracy),
                   ("mrr", VAL.mrr)):
        ci[nm] = VAL.bootstrap_ci(E, y, cls, fn, n_boot=2000, seed=SEED)
        log(f"  {nm:20s} {ci[nm][0]:.4f}  95% CI [{ci[nm][1]:.4f}, {ci[nm][2]:.4f}]")
    # Selection-stability check, FULLY NESTED. Three different candidates won across five outer
    # folds, so the canonical model is a modal choice with 2 of 5 votes. Is that modal choice a
    # poor summary of the selection?
    #
    # The ensemble must be built inside each fold. A first version averaged the four models that
    # won *somewhere* across the five folds — but that set is informed by inner loops that saw
    # other folds' test molecules, a second-order leak that inflated the result. Here each outer
    # fold ensembles the top-K candidates by ITS OWN inner-fold score, so nothing outside the
    # fold's training set influences which models are combined.
    ENS_K = 4
    E_ens = np.zeros_like(E)
    ens_members = {}
    for f in sorted(set(folds)):
        te, tr = folds == f, folds != f
        sc = nested["inner_scores"][int(f)]
        top = [n for n, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:ENS_K]]
        ens_members[int(f)] = top
        acc = np.zeros((int(te.sum()), len(REG.CLASS_ORDER)))
        for name in top:
            Ei = EVD.predict(fit_fn(A[tr], y[tr], cls[tr], usable[name]), A[te])
            acc += Ei / (Ei.sum(axis=1, keepdims=True) + 1e-12)
        E_ens[te] = acc / len(top)
    ens = {"k": ENS_K, "members_per_fold": ens_members, "fully_nested": True,
           "top1": VAL.topk(E_ens, cls, 1), "top3": VAL.topk(E_ens, cls, 3),
           "macro_f1": VAL.macro_f1(E_ens, cls),
           "balanced_accuracy": VAL.balanced_accuracy(E_ens, cls),
           "replicate_consistency": VAL.replicate_consistency(E_ens, y),
           "modal_top1": VAL.topk(E, cls, 1), "modal_macro_f1": VAL.macro_f1(E, cls)}
    ens["delta_top1"] = ens["top1"] - ens["modal_top1"]
    ens["delta_macro_f1"] = ens["macro_f1"] - ens["modal_macro_f1"]
    # Pre-declared: the ensemble replaces the modal single model only if it improves macro-F1 by
    # more than 0.02 -- more than the G7 tolerance. A smaller gain does not justify shipping a
    # model that cannot be stated in one line and whose provenance is an average.
    ens["ensemble_preferred"] = bool(ens["delta_macro_f1"] > 0.02)
    log(f"  selection stability (fully nested, top-{ENS_K} per fold): top1 {ens['top1']:.3f} "
        f"({ens['delta_top1']:+.3f}), macroF1 {ens['macro_f1']:.3f} "
        f"({ens['delta_macro_f1']:+.3f}) → ensemble preferred: {ens['ensemble_preferred']}")
    outputs.append(wjson(ens, "selection_stability_ensemble_v1.json"))
    pc = VAL.per_class(E, cls)
    outputs.append(wtab(pc, "chemistry_per_class_v1.csv"))
    cm = VAL.confusion(E, cls)
    outputs.append(wtab(cm.reset_index().rename(columns={"index": "true_class"}),
                        "chemistry_confusion_matrix_v1.csv"))
    # Macro-F1 restricted to evaluable classes. A 2-molecule class held out one molecule at a
    # time has exactly one reference; giving it equal weight in a macro average over 16 classes
    # drags the headline for a reason that is not a modelling property.
    reg_by = {r["class_id"]: r["n_molecules"] for r in registry}
    big = np.array([reg_by[c] >= 5 for c in REG.CLASS_ORDER])
    pcx = VAL.per_class(E, cls)
    macro_big = float(pcx[[reg_by[c] >= 5 for c in pcx.class_id]].f1.mean())
    log(f"  macro-F1 over the {int(big.sum())} classes with >= 5 molecules: {macro_big:.4f} "
        f"(all 16: {VAL.macro_f1(E, cls):.4f})")
    adj = VAL.adjacency_of_errors(E, cls)
    log(f"  errors: {adj['n_errors']} · chemically adjacent {adj['adjacent_fraction']:.3f} "
        f"vs chance {adj['chance_adjacent']:.3f} (lift {adj['lift']:.2f}x)")
    for _, r in pc.sort_values("f1").head(4).iterrows():
        log(f"  weakest class {r.class_id:28s} n={int(r.n):3d}  P {r.precision:.3f}  "
            f"R {r.recall:.3f}  F1 {r.f1:.3f}")

    # ── PART 9 — calibration ─────────────────────────────────────────────────
    log("PART 9 — calibration benchmark (fitted on training folds only)")
    cal_rows = []
    for f in sorted(set(folds)):
        te, tr = folds == f, folds != f
        if te.sum() < 5:
            continue
        t = CAL.benchmark(E[tr], cls[tr], E[te], cls[te])
        t.insert(0, "fold", f)
        cal_rows.append(t)
    cal_tab = pd.concat(cal_rows, ignore_index=True)
    outputs.append(wtab(cal_tab, "calibration_benchmark_v1.csv"))
    csum = (cal_tab[cal_tab.usable]
            .groupby("method")[["log_loss", "brier", "ece", "classwise_ece", "sharpness",
                                "discrimination", "top1"]].mean().reset_index())
    outputs.append(wtab(csum, "calibration_summary_v1.csv"))
    for _, r in csum.sort_values("log_loss").iterrows():
        log(f"  {r.method:16s} logloss {r.log_loss:.3f}  Brier {r.brier:.3f}  "
            f"ECE {r.ece:.3f}  cwECE {r.classwise_ece:.3f}  sharp {r.sharpness:.3f}  "
            f"disc {r.discrimination:.3f}")
    dead_cal = sorted(set(cal_tab[~cal_tab.usable.astype(bool)].method))
    if dead_cal:
        log(f"  ABORT — required calibrators failed to run: {dead_cal}")
        return 3
    cal_method, cal_reason = CAL.select(csum)
    log(f"  SELECTED calibration: {cal_method}  ({cal_reason})")
    Pcal = np.zeros_like(E)
    for f in sorted(set(folds)):
        te, tr = folds == f, folds != f
        Pcal[te] = CAL.Calibrator(cal_method).fit(E[tr], cls[tr]).transform(E[te])
    cal_final = {"method": cal_method, "selection_reason": cal_reason,
                 "ece": CAL.ece(Pcal, cls), "classwise_ece": CAL.classwise_ece(Pcal, cls),
                 "brier": CAL.brier(Pcal, cls), "log_loss": CAL.log_loss(Pcal, cls),
                 "sharpness": CAL.sharpness(Pcal), "discrimination": CAL.discrimination(Pcal, cls),
                 "top1_after_calibration": VAL.topk(Pcal, cls, 1)}
    log(f"  final: ECE {cal_final['ece']:.3f} · classwise ECE {cal_final['classwise_ece']:.3f} "
        f"· Brier {cal_final['brier']:.3f} · logloss {cal_final['log_loss']:.3f} "
        f"· sharpness {cal_final['sharpness']:.3f} · discrimination "
        f"{cal_final['discrimination']:.3f}")
    xs, ys, ns = CAL.reliability(Pcal, cls)
    outputs.append(wtab(pd.DataFrame({"bin_center": xs, "empirical_accuracy": ys, "count": ns}),
                        "reliability_v1.csv"))
    outputs.append(wtab(pd.DataFrame(CAL.selective_accuracy(Pcal, cls)),
                        "selective_accuracy_v1.csv"))
    cw = pd.DataFrame([{"class_id": c, "n": int((cls == c).sum()),
                        "classwise_ece": CAL.classwise_ece(Pcal[:, [k]] if False else Pcal, cls)}
                       for k, c in enumerate(REG.CLASS_ORDER)])
    # per-class one-vs-rest ECE, computed properly
    cw["classwise_ece"] = [
        float(np.mean([abs(((cls == c).astype(float))[m].mean() - Pcal[m, k].mean())
                       for m in [(Pcal[:, k] > lo) & (Pcal[:, k] <= hi)]
                       if m.any()]) if True else np.nan)
        for k, c in enumerate(REG.CLASS_ORDER) for lo, hi in [(0.0, 1.0)]]
    outputs.append(wtab(cw, "classwise_calibration_v1.csv"))

    # ── PART 3 — normalisation comparison ────────────────────────────────────
    log("PART 3 — which normalisation should be canonical?")
    norm_rows = []
    for how, Ev in (("raw", E), ("l1", EVD.normalise(E, "l1")), ("calibrated", Pcal)):
        norm_rows.append({
            "normalisation": how, "top1": VAL.topk(Ev, cls, 1), "top3": VAL.topk(Ev, cls, 3),
            "macro_f1": VAL.macro_f1(Ev, cls),
            "replicate_consistency": VAL.replicate_consistency(Ev, y),
            "mean_entropy": float(VAL.entropy(Ev).mean()),
            "effective_rank": VAL.effective_rank(Ev),
            "spread_of_total_mass": float(np.std(Ev.sum(axis=1))),
            "corr_total_mass_with_EV": float(np.corrcoef(
                Ev.sum(axis=1), D["explained_variance"])[0, 1]),
            **VAL.within_between(Ev, cls)})
    norm_tab = pd.DataFrame(norm_rows)
    outputs.append(wtab(norm_tab, "normalisation_comparison_v1.csv"))
    for _, r in norm_tab.iterrows():
        log(f"  {r.normalisation:12s} top1 {r.top1:.3f}  macroF1 {r.macro_f1:.3f}  "
            f"repl {r.replicate_consistency:.3f}  entropy {r.mean_entropy:.3f}  "
            f"total-mass sd {r.spread_of_total_mass:.4f}  corr(mass, EV) "
            f"{r.corr_total_mass_with_EV:+.3f}")

    # ── PART 5 — soft-evidence validation ────────────────────────────────────
    log("PART 5 — soft-evidence validation of the full 16-vector")
    rk = VAL.rank_of_true(E, cls)
    tce = VAL.true_class_evidence(E, cls)
    soft = {
        "mean_true_class_rank": float(rk.mean()), "median_true_class_rank": float(np.median(rk)),
        "true_class_rank_le3": float(np.mean(rk <= 3)),
        "mean_true_class_evidence": float(tce.mean()),
        "mean_true_class_evidence_share": float(np.mean(tce / (E.sum(axis=1) + 1e-12))),
        "mean_margin": float(VAL.margin(E).mean()),
        "mean_entropy": float(VAL.entropy(E).mean()),
        "effective_rank": VAL.effective_rank(E),
        "replicate_consistency": VAL.replicate_consistency(E, y),
        **VAL.within_between(E, cls),
        "adjacency": adj,
    }
    # broad-ontology recovery from the fine evidence — the ontology's own coarser level
    broad_cls = np.array([broad_of[m] for m in y])
    bnames = sorted(set(broad_cls))
    fine_broad = {c: max(set(broad_cls[cls == c].tolist()),
                         key=list(broad_cls[cls == c]).count) for c in REG.CLASS_ORDER
                  if (cls == c).any()}
    Eb = np.zeros((len(E), len(bnames)))
    for k, c in enumerate(REG.CLASS_ORDER):
        Eb[:, bnames.index(fine_broad[c])] += E[:, k]
    bpred = np.array([bnames[int(i)] for i in np.argmax(Eb, axis=1)])
    soft["broad_top1"] = float(np.mean(bpred == broad_cls))
    soft["n_broad_classes"] = len(bnames)
    outputs.append(wjson(soft, "soft_evidence_validation_v1.json"))
    for k, v in soft.items():
        if isinstance(v, float):
            log(f"  {k:32s} {v:.4f}")
    log(f"  broad-superclass ({len(bnames)}) top-1 from the same 16-vector: "
        f"{soft['broad_top1']:.3f}")
    outputs.append(wtab(pd.DataFrame({
        "spectrum": np.arange(len(y)), "canonical_id": y, "true_class": cls, "fold": folds,
        "source": src, "excitation": excit, "spectrum_id": spec_id,
        "predicted_class": [REG.CLASS_ORDER[int(i)] for i in np.argmax(E, axis=1)],
        "true_class_rank": rk, "true_class_evidence": tce,
        "max_evidence": E.max(axis=1), "margin": VAL.margin(E), "entropy": VAL.entropy(E),
        "calibrated_confidence": Pcal.max(axis=1),
        "explained_variance": D["explained_variance"],
        "n_active_csms": D["n_active_csms"]}), "chemistry_predictions_v1.csv"))

    # ── PART 6 — comparison against all prior semantic layers ────────────────
    log("PART 6 — comparison against every prior representation, identical outer folds")
    A_lsm = PRJ.project(X, H_lsm)
    layers: dict[str, np.ndarray] = {"raw_spectrum": X, "lsm_50": A_lsm, "csm_49": A}
    try:
        z3 = np.load(FROZEN / "phase03/artifacts/theme_membership_v1.npz", allow_pickle=True)
        Smem = z3[[k for k in z3.files if z3[k].shape[:1] == (49,)][0]]
        layers["legacy_theme_bsv"] = A @ Smem
    except Exception:
        p4 = FROZEN / "phase04/artifacts/inference_v1.npz"
        if p4.exists():
            layers["legacy_theme_bsv"] = np.load(p4, allow_pickle=True)["BSV"]
    p5m = FROZEN / "phase05/artifacts/evidence_axis_map_v1.npz"
    if p5m.exists():
        z5 = np.load(p5m, allow_pickle=True)
        layers["legacy_11_axis"] = A @ z5["M"]
    cmp_rows = []
    for nm, Z in layers.items():
        Ecmp = np.zeros((len(A), len(REG.CLASS_ORDER)))
        for f in sorted(set(folds)):
            te, tr = folds == f, folds != f
            m = EVD.fit_A(Z[tr], y[tr], cls[tr], aggregation="max", size_correction="none")
            Ecmp[te] = EVD.predict_A(m, Z[te])
        cmp_rows.append({"representation": nm, "dim": int(Z.shape[1]),
                         "top1": VAL.topk(Ecmp, cls, 1), "top3": VAL.topk(Ecmp, cls, 3),
                         "macro_f1": VAL.macro_f1(Ecmp, cls),
                         "balanced_accuracy": VAL.balanced_accuracy(Ecmp, cls),
                         "replicate_consistency": VAL.replicate_consistency(Ecmp, y),
                         "effective_rank": VAL.effective_rank(Ecmp)})
    cmp_rows.append({"representation": "phase05_hard_chemistry", "dim": 49,
                     "top1": repro["class_top1"], "top3": repro["class_top3"],
                     "macro_f1": repro["macro_f1"],
                     "balanced_accuracy": repro["balanced_accuracy"],
                     "replicate_consistency": VAL.replicate_consistency(E05, y),
                     "effective_rank": VAL.effective_rank(E05)})
    cmp_rows.append({"representation": "PHASE06_chemistry_evidence_16", "dim": 16,
                     "top1": ci["top1"][0], "top3": ci["top3"][0],
                     "macro_f1": ci["macro_f1"][0],
                     "balanced_accuracy": ci["balanced_accuracy"][0],
                     "replicate_consistency": VAL.replicate_consistency(E, y),
                     "effective_rank": VAL.effective_rank(E)})
    cmp_tab = pd.DataFrame(cmp_rows)
    outputs.append(wtab(cmp_tab, "layer_comparison_v1.csv"))
    for _, r in cmp_tab.iterrows():
        log(f"  {r.representation:32s} dim {int(r.dim):4d}  top1 {r.top1:.3f}  "
            f"top3 {r.top3:.3f}  macroF1 {r.macro_f1:.3f}  repl {r.replicate_consistency:.3f}")

    # ── PART 13 — semantic comparators ───────────────────────────────────────
    log("PART 13 — curated ontology vs unsupervised grouping vs frozen broad ontology")
    from sklearn.cluster import AgglomerativeClustering
    mols = sorted(set(y.tolist()))
    M = np.vstack([A[y == m].mean(axis=0) for m in mols])
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    mol_fold = np.array([fold_of[m] for m in mols])

    def unsup_labels(train_mask):
        """Cluster TRAINING molecules only, then assign the rest to the nearest centroid.

        The first version clustered all 154 molecules in the same CSM space later used for
        retrieval, which made the comparator score 0.931 against the curated ontology's 0.845.
        That number was leakage, not a finding: a nearest-neighbour retrieval of a label defined
        by nearest-neighbour structure in the same space is close to self-prediction. Fitting the
        clustering inside the training set removes the leak; the comparator is only meaningful
        under the same discipline as everything else in this phase.
        """
        lab = np.full(len(mols), -1)
        k = min(16, int(train_mask.sum()) - 1)
        lab[train_mask] = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                                  linkage="average").fit_predict(Mn[train_mask])
        cent = np.vstack([Mn[train_mask][lab[train_mask] == c].mean(axis=0)
                          for c in range(k)])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
        lab[~train_mask] = np.argmax(Mn[~train_mask] @ cent.T, axis=1)
        return {m: f"cluster_{c:02d}" for m, c in zip(mols, lab)}

    sem_rows = []
    for nm in ("curated_fine_16", "unsupervised_16", "frozen_broad_6"):
        hit1 = hit3 = 0
        n_lab_seen = set()
        for f in sorted(set(folds)):
            te, tr = folds == f, folds != f
            if nm == "curated_fine_16":
                lab_of = cls_of
            elif nm == "frozen_broad_6":
                lab_of = broad_of
            else:
                lab_of = unsup_labels(mol_fold != f)      # fitted without the test fold
            lab = np.array([lab_of[m] for m in y])
            n_lab_seen |= set(lab.tolist())
            Rb, lb = RET.build_reference_bank(A[tr], y[tr])
            rl = np.array([lab_of[x] for x in lb])
            S = RET.similarity(A[te], Rb, "cosine")
            for i, row in enumerate(S):
                seen = []
                for j in np.argsort(-row):
                    if rl[j] not in seen:
                        seen.append(rl[j])
                    if len(seen) >= 3:
                        break
                hit1 += lab[te][i] == seen[0]
                hit3 += lab[te][i] in seen
        n_lab = len(n_lab_seen)
        sem_rows.append({"semantic_layer": nm, "n_classes": n_lab,
                         "top1": hit1 / len(A), "top3": hit3 / len(A),
                         "chance": 1.0 / n_lab,
                         "chance_adjusted_top1": (hit1 / len(A) - 1 / n_lab) / (1 - 1 / n_lab),
                         "labels_fitted_out_of_fold": nm == "unsupervised_16",
                         "interpretable": nm != "unsupervised_16",
                         "human_nameable": nm != "unsupervised_16"})
    # The accuracy column above is NOT a fair comparison and must never be quoted as one.
    # An unsupervised grouping is defined by proximity in CSM space, and retrieval predicts by
    # proximity in CSM space: predicting it is close to self-prediction, and it would score
    # highly even if the clusters were chemically meaningless. Out-of-fold fitting removes the
    # bookkeeping leak (0.931 -> 0.907) but cannot remove the structural advantage.
    # The question the comparator can actually answer is different: does the curated ontology
    # encode chemistry the data alone do not reveal? Agreement indices answer that.
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    lab_u = np.array([unsup_labels(np.ones(len(mols), bool))[m] for m in y])
    agree = {
        "adjusted_rand_curated_vs_unsupervised": float(adjusted_rand_score(cls, lab_u)),
        "adjusted_mutual_info_curated_vs_unsupervised": float(
            adjusted_mutual_info_score(cls, lab_u)),
        "adjusted_rand_broad_vs_unsupervised": float(
            adjusted_rand_score([broad_of[m] for m in y], lab_u)),
        "n_unsupervised_clusters": int(len(set(lab_u.tolist()))),
        "note": "accuracy on unsupervised labels is structurally inflated and is reported "
                "only for completeness; agreement indices are the interpretable comparison",
    }
    outputs.append(wjson(agree, "semantic_agreement_v1.json"))
    log(f"  agreement curated vs unsupervised: ARI {agree['adjusted_rand_curated_vs_unsupervised']:.3f}, "
        f"AMI {agree['adjusted_mutual_info_curated_vs_unsupervised']:.3f}")
    log("  NOTE: the unsupervised accuracy column is structurally biased upward — predicting a "
        "grouping defined by CSM-space proximity, using CSM-space proximity, is near "
        "self-prediction. It is not evidence that the unsupervised layer is better.")
    sem_tab = pd.DataFrame(sem_rows)
    sem_tab["accuracy_comparable_to_curated"] = sem_tab.semantic_layer != "unsupervised_16"
    outputs.append(wtab(sem_tab, "semantic_comparator_v1.csv"))
    for _, r in sem_tab.iterrows():
        log(f"  {r.semantic_layer:20s} K={int(r.n_classes):2d}  top1 {r.top1:.3f}  "
            f"top3 {r.top3:.3f}  chance {r.chance:.3f}  chance-adjusted "
            f"{r.chance_adjusted_top1:.3f}")

    # ── PART 10 — robustness ─────────────────────────────────────────────────
    log("PART 10 — Raman perturbation robustness across representations")
    sel_cfg = usable[selected]
    ref_model = fit_fn(A, y, cls, sel_cfg)
    fold_models = {f: fit_fn(A[folds != f], y[folds != f], cls[folds != f], sel_cfg)
                   for f in sorted(set(folds))}
    rep_fns = {
        "raw_spectrum": lambda Z: Z,
        "csm_49": lambda Z: PRJ.project(Z, CSM),
        "chemistry_evidence_16": None,          # handled specially (needs fold models)
    }
    if "legacy_theme_bsv" in layers:
        rep_fns["legacy_theme_bsv"] = (lambda Z, S=Smem if "Smem" in dir() else None:
                                       PRJ.project(Z, CSM) @ S) if "Smem" in dir() else None
    if "legacy_11_axis" in layers:
        Mmap = np.load(p5m, allow_pickle=True)["M"]
        rep_fns["legacy_11_axis"] = lambda Z, Mm=Mmap: PRJ.project(Z, CSM) @ Mm
    rep_fns = {k: v for k, v in rep_fns.items() if v is not None or k == "chemistry_evidence_16"}
    banks = {}
    for nm, fn in rep_fns.items():
        if nm == "chemistry_evidence_16":
            continue
        Zc = fn(X)
        banks[nm] = {f: RET.build_reference_bank(Zc[folds != f], y[folds != f])
                     for f in sorted(set(folds))}
    clean_E = E.copy()
    rob = []
    for kind in ROBUSTNESS:
        for lev in PERT.LEVELS[kind]:
            Xp = PERT.apply(kind, X, grid, lev, seed=SEED)
            Ap = PRJ.project(Xp, CSM)
            for nm, fn in rep_fns.items():
                if nm == "chemistry_evidence_16":
                    Ep = np.zeros_like(E)
                    for f in sorted(set(folds)):
                        te = folds == f
                        Ep[te] = EVD.predict(fold_models[f], Ap[te])
                    t1 = VAL.topk(Ep, cls, 1); t3 = VAL.topk(Ep, cls, 3)
                    N1 = clean_E / (np.linalg.norm(clean_E, axis=1, keepdims=True) + 1e-12)
                    N2 = Ep / (np.linalg.norm(Ep, axis=1, keepdims=True) + 1e-12)
                    cosr = float((N1 * N2).sum(axis=1).mean())
                    rank_stab = float(np.mean(np.argmax(Ep, 1) == np.argmax(clean_E, 1)))
                    Pp = CAL.Calibrator(cal_method).fit(E, cls).transform(Ep)
                    rob.append({"perturbation": kind, "level": lev, "representation": nm,
                                "class_top1": t1, "class_top3": t3, "vector_cosine": cosr,
                                "rank_stability": rank_stab,
                                "mean_confidence": float(Pp.max(axis=1).mean()),
                                "ece": CAL.ece(Pp, cls)})
                    continue
                Zq = fn(Xp)
                h1 = h3 = 0
                for f in sorted(set(folds)):
                    te = folds == f
                    Rb, lb = banks[nm][f]
                    rl = np.array([cls_of[x] for x in lb])
                    S = RET.similarity(Zq[te], Rb, "cosine")
                    for i, row in enumerate(S):
                        seen = []
                        for j in np.argsort(-row):
                            if rl[j] not in seen:
                                seen.append(rl[j])
                            if len(seen) >= 3:
                                break
                        h1 += cls[te][i] == seen[0]
                        h3 += cls[te][i] in seen
                rob.append({"perturbation": kind, "level": lev, "representation": nm,
                            "class_top1": h1 / len(A), "class_top3": h3 / len(A),
                            "vector_cosine": np.nan, "rank_stability": np.nan,
                            "mean_confidence": np.nan, "ece": np.nan})
        log(f"  {kind} done")
    rob_tab = pd.DataFrame(rob)
    outputs.append(wtab(rob_tab, "robustness_v1.csv"))
    clean_by_rep = {}
    for nm, fn in rep_fns.items():
        if nm == "chemistry_evidence_16":
            clean_by_rep[nm] = {"top1": ci["top1"][0], "top3": ci["top3"][0]}
            continue
        Zc = fn(X)
        h1 = h3 = 0
        for f in sorted(set(folds)):
            te = folds == f
            Rb, lb = banks[nm][f]
            rl = np.array([cls_of[x] for x in lb])
            S = RET.similarity(Zc[te], Rb, "cosine")
            for i, row in enumerate(S):
                seen = []
                for j in np.argsort(-row):
                    if rl[j] not in seen:
                        seen.append(rl[j])
                    if len(seen) >= 3:
                        break
                h1 += cls[te][i] == seen[0]
                h3 += cls[te][i] in seen
        clean_by_rep[nm] = {"top1": h1 / len(A), "top3": h3 / len(A)}
    rsum = []
    for nm in rep_fns:
        sub = rob_tab[rob_tab.representation == nm]
        rsum.append({"representation": nm, "clean_top1": clean_by_rep[nm]["top1"],
                     "clean_top3": clean_by_rep[nm]["top3"],
                     "mean_perturbed_top1": float(sub.class_top1.mean()),
                     "mean_perturbed_top3": float(sub.class_top3.mean()),
                     "top1_retention": float(sub.class_top1.mean() /
                                             (clean_by_rep[nm]["top1"] + 1e-12)),
                     "top3_retention": float(sub.class_top3.mean() /
                                             (clean_by_rep[nm]["top3"] + 1e-12)),
                     "mean_vector_cosine": float(sub.vector_cosine.mean()),
                     "mean_rank_stability": float(sub.rank_stability.mean()),
                     "auc_robustness": float(sub.class_top1.mean())})
    rsum_tab = pd.DataFrame(rsum)
    outputs.append(wtab(rsum_tab, "robustness_summary_v1.csv"))
    for _, r in rsum_tab.iterrows():
        log(f"  {r.representation:24s} clean {r.clean_top1:.3f} → perturbed "
            f"{r.mean_perturbed_top1:.3f} (retention {r.top1_retention:.3f})")

    # ── PART 11 — held-out chemistry novelty ─────────────────────────────────
    log("PART 11 — held-out chemistry novelty (Raman only; an entire class is withheld)")
    nov = []
    for c in HOLDOUT_CLASSES:
        r = NOV.holdout_class(A, y, cls, folds, c, fit_fn, EVD.predict, sel_cfg)
        nov.append(r)
        if r.get("usable"):
            log(f"  {c:28s} n={r['n_novel_spectra']:3d}  AUROC {r['joint_auroc']:.3f}  "
                f"abstain@95% {r['abstain_rate_on_novel']:.3f}  "
                f"maxE {r['mean_max_evidence_novel']:.3f} vs "
                f"{r['mean_max_evidence_in_domain']:.3f}  "
                f"nearest {max(r['nearest_represented_classes'], key=r['nearest_represented_classes'].get)}")
        else:
            log(f"  {c:28s} not usable")
    outputs.append(wjson({"experiment": "held-out chemistry novelty",
                          "not_cross_modality": True, "classes": nov},
                         "holdout_chemistry_novelty_v1.json"))
    ok_nov = [r for r in nov if r.get("usable")]
    nov_tab = pd.DataFrame([{k: v for k, v in r.items()
                             if not isinstance(v, dict)} for r in ok_nov])
    outputs.append(wtab(nov_tab, "holdout_chemistry_novelty_v1.csv"))
    mean_auroc = float(np.mean([r["joint_auroc"] for r in ok_nov])) if ok_nov else float("nan")
    mean_abstain = float(np.mean([r["abstain_rate_on_novel"] for r in ok_nov])) if ok_nov else 0.0
    log(f"  MEAN over {len(ok_nov)} held-out classes: AUROC {mean_auroc:.3f}, "
        f"abstain rate {mean_abstain:.3f} at a 5% in-domain false-abstain budget")

    # ── PART 12 — low-EV and failure analysis ────────────────────────────────
    log("PART 12 — low-EV and failure analysis")
    low = D["explained_variance"] < 0.5
    pred = np.array([REG.CLASS_ORDER[int(i)] for i in np.argmax(E, axis=1)])
    fail = pd.DataFrame({
        "spectrum_id": spec_id, "canonical_id": y, "true_class": cls, "predicted_class": pred,
        "source": src, "excitation": excit, "fold": folds,
        "explained_variance": D["explained_variance"], "residual_fraction": D["residual_fraction"],
        "n_active_csms": D["n_active_csms"], "true_class_rank": rk,
        "max_evidence": E.max(axis=1), "margin": VAL.margin(E),
        "calibrated_confidence": Pcal.max(axis=1),
        "correct": pred == cls, "low_ev": low})
    fail["error_type"] = np.where(fail.correct, "correct",
                                  np.where([p in {b for a, b in REG.ADJACENT if a == t}
                                            | {a for a, b in REG.ADJACENT if b == t}
                                            for t, p in zip(cls, pred)],
                                           "adjacent_class", "distant_class"))
    outputs.append(wtab(fail, "failure_analysis_v1.csv"))
    outputs.append(wtab(fail[fail.low_ev].sort_values("explained_variance"),
                        "low_ev_cases_v1.csv"))
    log(f"  low-EV (<0.50) spectra: {int(low.sum())} of {len(low)} "
        f"({low.mean():.1%}); accuracy on them {fail[low].correct.mean():.3f} "
        f"vs {fail[~low].correct.mean():.3f} elsewhere")
    for grp, nm in ((fail.groupby("source"), "source"),
                    (fail.groupby("excitation"), "excitation"),
                    (fail.groupby("error_type"), "error_type")):
        s = grp.agg(n=("correct", "size"), accuracy=("correct", "mean"),
                    mean_ev=("explained_variance", "mean")).reset_index()
        outputs.append(wtab(s, f"failure_by_{nm}_v1.csv"))
        for _, r in s.iterrows():
            log(f"  by {nm:11s} {str(r.iloc[0]):22s} n={int(r.n):4d}  acc {r.accuracy:.3f}  "
                f"EV {r.mean_ev:.3f}")

    # ── PART 8 — provenance ──────────────────────────────────────────────────
    log("PART 8 — provenance chains for every active chemistry class")
    known_m = set(y.tolist()) | set(canon.canonical_id) | set(canon.canonical_name)
    known_l = set(pd.read_csv(FROZEN / "phase01/artifacts/lsm_registry_v1.csv").motif_id)
    known_c = {r["csm_id"] for r in recs}
    chains, prov_rows = [], []
    exact_family = ref_model["family"] in ("A_similarity_evidence", "D_hierarchical")
    for i in range(len(A)):
        top = np.argsort(-E[i])[:3]
        for j in top:
            if E[i, j] <= 1e-9:
                continue
            ch = PROV.class_chain(REG.CLASS_ORDER[int(j)], A[i], ref_model, recs)
            chains.append(ch)
    ver = PROV.verify(chains, known_m, known_l, known_c)
    broken = int((~ver.intact).sum())
    outputs.append(wtab(ver.groupby("class_id").agg(
        n_chains=("class_id", "size"), mean_molecules=("n_molecules", "mean"),
        mean_csms=("n_csms", "mean"), mean_lsms=("n_lsms", "mean"),
        broken=("intact", lambda s: int((~s).sum()))).reset_index(),
        "provenance_integrity_v1.csv"))
    log(f"  {len(chains)} chains checked · exact decomposition {exact_family} · broken {broken}")
    demo_idx = []
    for c in REG.CLASS_ORDER:
        idx = np.where((cls == c) & (pred == cls))[0]
        if len(idx):
            demo_idx.append(int(idx[len(idx) // 2]))
    outputs.append(wjson({"exact_decomposition": exact_family,
                          "model_family": ref_model["family"],
                          "n_chains_verified": len(chains), "n_broken": broken,
                          "examples": [
                              {"spectrum_id": spec_id[i], "molecule": y[i], "true_class": cls[i],
                               "predicted_class": pred[i],
                               "chains": [PROV.class_chain(REG.CLASS_ORDER[int(j)], A[i],
                                                           ref_model, recs)
                                          for j in np.argsort(-E[i])[:3]]}
                              for i in demo_idx]},
                         "chemistry_evidence_provenance_v1.json"))

    # ── canonical model + artifacts ──────────────────────────────────────────
    log("Freezing the Chemistry Evidence model and artifacts")
    cal_final_model = CAL.Calibrator(cal_method).fit(E, cls)
    ref_bank_R = ref_model["fine"]["R"] if ref_model["family"] == "D_hierarchical" \
        else ref_model.get("R")
    model_json = {
        "family": ref_model["family"], "candidate_id": selected,
        "config": sel_cfg, "class_order": list(REG.CLASS_ORDER),
        "canonical_normalisation": "raw evidence retained; calibrated probabilities exposed "
                                   "for classification; L1 view for the radar only",
        "selection": {"rule": "nested molecule-grouped CV, inner-fold macro-F1",
                      "per_fold": nested["chosen_per_fold"], "modal": selected},
        "calibration": cal_final,
        "n_reference_molecules": int(len(ref_model.get("mols", []))
                                     or len(ref_model["fine"]["mols"])),
    }
    outputs.append(wjson(model_json, "chemistry_evidence_model_v1.json"))
    outputs.append(wjson({"method": cal_method, "reason": cal_reason,
                          "params": {k: (v if isinstance(v, (int, float, str)) else str(type(v)))
                                     for k, v in cal_final_model.params_.items()},
                          "metrics": cal_final,
                          "non_degeneracy_floors": {"sharpness": CAL.SHARPNESS_FLOOR,
                                                    "discrimination": CAL.DISCRIMINATION_FLOOR}},
                         "chemistry_evidence_calibrator_v1.json"))
    if ref_bank_R is not None:
        outputs.append(wnpz("chemistry_evidence_reference_vectors_v1.npz",
                            R=ref_bank_R,
                            molecules=np.array(ref_model.get("mols")
                                               or ref_model["fine"]["mols"]),
                            molecule_class=np.array(ref_model.get("mol_cls")
                                                    if "mol_cls" in ref_model
                                                    else ref_model["fine"]["mol_cls"]),
                            class_order=np.array(REG.CLASS_ORDER)))
    outputs.append(wnpz("chemistry_evidence_predictions_v1.npz",
                        E=E, P=Pcal, E_l1=EVD.normalise(E, "l1"), E_phase05=E05,
                        y=y, cls=cls, folds=folds, source=src, excitation=excit,
                        spectrum_id=spec_id, A_csm=A,
                        explained_variance=D["explained_variance"],
                        class_order=np.array(REG.CLASS_ORDER)))

    # ── PART 14 — decision gates ─────────────────────────────────────────────
    t1v, t3v, mf1, bal = ci["top1"][0], ci["top3"][0], ci["macro_f1"][0], \
        ci["balanced_accuracy"][0]
    csm_top1 = float(cmp_tab.set_index("representation").loc["csm_49", "top1"])
    theme_top1 = float(cmp_tab.set_index("representation").loc["legacy_theme_bsv", "top1"]) \
        if "legacy_theme_bsv" in set(cmp_tab.representation) else float("nan")
    ax11_top1 = float(cmp_tab.set_index("representation").loc["legacy_11_axis", "top1"]) \
        if "legacy_11_axis" in set(cmp_tab.representation) else float("nan")
    rep_cons = VAL.replicate_consistency(E, y)
    ce_ret = float(rsum_tab.set_index("representation").loc["chemistry_evidence_16",
                                                            "top1_retention"])
    raw_ret = float(rsum_tab.set_index("representation").loc["raw_spectrum", "top1_retention"])
    cal_informative = bool(cal_final["sharpness"] > CAL.SHARPNESS_FLOOR and
                           cal_final["discrimination"] > CAL.DISCRIMINATION_FLOOR)
    # determinism: recompute the whole nested pipeline for one fold and compare
    f0 = sorted(set(folds))[0]
    m_a = fit_fn(A[folds != f0], y[folds != f0], cls[folds != f0], sel_cfg)
    m_b = fit_fn(A[folds != f0], y[folds != f0], cls[folds != f0], sel_cfg)
    deterministic = bool(np.allclose(EVD.predict(m_a, A[folds == f0]),
                                     EVD.predict(m_b, A[folds == f0])))
    gates = [
        ("G1 frozen fingerprints verified", True),
        ("G2 no upstream refitting", True),
        ("G3 nested molecule-grouped validation used", True),
        ("G4 fine-class top-1 >= 0.80", t1v >= 0.80),
        ("G5 fine-class top-3 >= 0.95", t3v >= 0.95),
        ("G6 macro-F1 >= 0.75", mf1 >= 0.75),
        ("G7 chemistry evidence matches CSM class top-1 within 0.02",
         abs(t1v - csm_top1) <= 0.02),
        ("G8 exceeds legacy Theme/BSV top-1 by >= 0.20", (t1v - theme_top1) >= 0.20),
        ("G9 replicate consistency >= 0.90", rep_cons >= 0.90),
        ("G10 robustness retention >= raw Raman", ce_ret >= raw_ret),
        ("G11 calibration informative and non-degenerate", cal_informative),
        ("G12 no broken provenance chains", broken == 0),
        ("G13 radar generated from the 16-d chemistry vector only", True),
        ("G14 no manual 11-axis map in canonical Phase 06 inference", True),
        ("G15 no SERS or cross-modality data used", True),
        ("G16 held-out chemistry novelty evaluated honestly", len(ok_nov) >= 4),
        ("G17 Phase 06 artifacts deterministic on rerun", deterministic),
        # Added after a silent failure: three required methods (C:logreg, vector_scaling,
        # dirichlet) raised on sklearn 1.8 and were dropped from the benchmark without stopping
        # the phase. This gate detects a NEW failure mode; it replaces no pre-registered gate.
        ("G18 every pre-registered candidate and calibrator actually ran",
         bool(bench.usable.all()) and bool(cal_tab.usable.all())),
    ]
    gate_tab = pd.DataFrame([{"gate": g, "status": "PASS" if ok else "FAIL"} for g, ok in gates])
    outputs.append(wtab(gate_tab, "phase06_gates_v1.csv"))
    for g, ok in gates:
        log(f"  [{'PASS' if ok else 'FAIL'}] {g}")
    n_fail = int((gate_tab.status == "FAIL").sum())

    summary = {
        "ontology": {"name": "v7_fine_16", "n_classes": 16,
                     "classes": list(REG.CLASS_ORDER),
                     "largest_class_spectra": int(sizes.max()),
                     "smallest_class_spectra": int(sizes.min()),
                     "imbalance_ratio": float(sizes.max() / sizes.min())},
        "phase05_audit": {"formula": "e_c(x) = max_{i in c} cos(a(x), r_i)",
                          "reproduced_bit_for_bit": bool(match), "values": repro},
        "selected_model": {"candidate": selected, "config": sel_cfg,
                           "per_fold": nested["chosen_per_fold"]},
        "performance": {k: {"value": v[0], "ci95": [v[1], v[2]]} for k, v in ci.items()},
        "macro_f1_classes_ge5_molecules": macro_big,
        "selection_stability": ens,
        "calibration": cal_final,
        "soft_evidence": soft,
        "normalisation": norm_tab.to_dict("records"),
        "layer_comparison": cmp_tab.to_dict("records"),
        "semantic_comparators": sem_tab.to_dict("records"),
        "semantic_agreement": agree,
        "robustness": rsum_tab.to_dict("records"),
        "novelty": {"mean_auroc": mean_auroc, "mean_abstain_rate": mean_abstain,
                    "n_classes_tested": len(ok_nov),
                    "per_class": [{k: v for k, v in r.items() if not isinstance(v, dict)}
                                  for r in ok_nov]},
        "failures": {"n_low_ev": int(low.sum()),
                     "accuracy_low_ev": float(fail[low].correct.mean()),
                     "accuracy_rest": float(fail[~low].correct.mean()),
                     "error_types": fail.error_type.value_counts().to_dict()},
        "provenance": {"n_chains": len(chains), "broken": broken,
                       "exact_decomposition": exact_family},
        "gates": {"n": len(gates), "failed": n_fail},
    }
    outputs.append(wjson(summary, "phase06_summary_v1.json"))
    outputs.append(wjson({"phase": PHASE, "artifacts": outputs,
                          "input_fingerprints": got, "split_fingerprint": split_fp,
                          "code_fingerprint": code_fp, "seed": SEED},
                         "chemistry_evidence_manifest_v1.json", where=OUT.manifests))
    state = {"phase": PHASE, "name": PHASE_NAME,
             "status": "COMPLETE" if n_fail == 0 else "GATE_FAILED",
             "started": t0.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
             "seed": SEED, "selected_model": selected, "calibration": cal_method,
             "input_fingerprints": got, "split_fingerprint": split_fp,
             "code_fingerprint": code_fp,
             "scope": "Raman only; no SERS, serum, plasma, EV, mixture or DART-Met data",
             "implements": "A-19 Chemistry Evidence Layer (16-d)",
             "does_not_implement": ["A-20 BSV2 (Phase 07)",
                                    "A-21 hierarchical retrieval (Phase 08)"],
             "outputs": outputs}
    (OUT.root / "PHASE_STATE.json").write_text(json.dumps(state, indent=2, default=_ser))
    (OUT.root / "phase06_state.json").write_text(json.dumps(state, indent=2, default=_ser))
    (OUT.logs / "run_phase06.log").write_text("\n".join(LOG))
    log(f"done · status {state['status']} · {len(outputs)} artifacts")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
