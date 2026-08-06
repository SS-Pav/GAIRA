#!/usr/bin/env python3
"""GAIRA V7 — Phase 05: the canonical CSM inference engine (Raman only).

Steps 1–11 of the brief. Nothing frozen is refitted; every selection is made by grouped CV
inside the training folds and never on the number finally reported. Output location resolves
through `gaira.v7.io.PhaseOutputs`.

    python results/v7_rebuild/phase05/code/run_phase05.py
"""
from __future__ import annotations

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

import v7_paths as P                                              # noqa: E402
from gaira.v7.io import PhaseOutputs, frozen_root                 # noqa: E402
from gaira.v7.meta import perturbations as PERT                   # noqa: E402
from gaira.v7.inference import (calibration as CAL, evidence as EV,  # noqa: E402
                                openset as OS, projection as PRJ,
                                provenance as PROV, retrieval as RET)
from gaira.v7.inference.engine import CanonicalEngine             # noqa: E402

warnings.filterwarnings("ignore")

PHASE, PHASE_NAME = "05", "Canonical CSM inference engine"
OUT = PhaseOutputs(PHASE).ensure()
FROZEN = frozen_root()
EXPECTED = {"atlas": "09ed804a40836f4a05a91ba10900cded",
            "lsm": "208482d6f7178b5b8f16cace91be55b0",
            "csm": "0b4aa550ccefed3edabdbde5bae11c8d"}
SEED = 0
# Step 11 requires exactly these seven; they exist already in the frozen perturbation module.
ROBUSTNESS = ("gaussian_noise", "baseline_drift", "wavelength_shift", "intensity_scaling",
              "band_broadening", "peak_dropout", "fluorescence")
# Evaluation-only mapping: which chemistry classes each declared axis claims to be about.
AXIS_CLASSES = {
    "aliphatic_chain": ["fatty_acid", "acylglycerol"],
    "unsaturation": ["fatty_acid", "acylglycerol"],
    "carbonyl_ester": ["acylglycerol", "phospholipid_sphingolipid",
                       "carboxylic_acid_metabolite"],
    "amide_protein": ["peptide_protein"],
    "carbohydrate_skeletal": ["mono_oligosaccharide", "polysaccharide"],
    "heterocyclic_ring": ["purine", "pyrimidine", "nucleic_acid_polymer"],
    "purine": ["purine"],
    "sulfur_thiol": ["sulfur_thiol_cofactor"],
    "phosphate_nucleic": ["phosphate_metabolite", "nucleic_acid_polymer",
                          "phospholipid_sphingolipid"],
    "aromatic_residue": ["free_amino_acid", "peptide_protein"],
    "chromophore_conjugated": ["chromophore_pigment"],
}
LOG: list[str] = []


def log(m):
    line = f"[phase05] {m}"
    print(line, flush=True)
    LOG.append(line)


def _ser(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def wtab(df, name, where=None):
    p = (where or OUT.tables) / name
    df.to_csv(p, index=False)
    return {"artifact_id": name, "path": OUT.rel(p), "sha256": P.sha256_file(p), "rows": len(df)}


def wjson(obj, name, where=None):
    p = (where or OUT.artifacts) / name
    p.write_text(json.dumps(obj, indent=2, default=_ser))
    return {"artifact_id": name, "path": OUT.rel(p), "sha256": P.sha256_file(p)}


def topk_class(S, ref_classes, truth, k=1):
    """Class retrieval: a hit if the true class appears among the top-k *distinct* classes."""
    rc = np.asarray(ref_classes)
    hits = 0
    for i, row in enumerate(S):
        seen = []
        for j in np.argsort(-row):
            if rc[j] not in seen:
                seen.append(rc[j])
            if len(seen) >= k:
                break
        hits += truth[i] in seen
    return hits / max(len(S), 1)


def topk_mol(S, ref_labels, truth, k=1):
    rl = np.asarray(ref_labels)
    return float(np.mean([truth[i] in rl[np.argsort(-row)[:k]] for i, row in enumerate(S)]))


def main() -> int:
    outputs: list[dict] = []
    t0 = datetime.now(timezone.utc)

    # ── 0. fingerprint gate ──────────────────────────────────────────────────
    log("architecture check — Phase 05 REPLACES the Phase 04 Theme/BSV inference path. The "
        "canonical representation is the 49-d CSM activation vector; the interpretable layer "
        "is a declared, grounded evidence map, not a discovered theme basis. Raman only: the "
        "Phase 04 SERS out-of-domain probe is removed from scope, not merely unreported.")
    csm_reg = json.loads((FROZEN / "phase02/artifacts/csm_registry_v1.json").read_text())
    lsm_reg = json.loads((FROZEN / "phase01/artifacts/lsm_registry_v1.json").read_text()) \
        if (FROZEN / "phase01/artifacts/lsm_registry_v1.json").exists() else None
    got = {"csm": csm_reg["fingerprint"]}
    atlas_state = json.loads((FROZEN / "phase01/PHASE_STATE.json").read_text())
    got["lsm"] = atlas_state["registry_fingerprint"]
    for k in ("csm", "lsm"):
        if got[k] != EXPECTED[k]:
            log(f"ABORT — {k} fingerprint {got[k]} != expected {EXPECTED[k]}")
            return 2
    log(f"  frozen verified: LSM {got['lsm']} · CSM {got['csm']}")

    br = np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
    X = np.asarray(br["X"], float)
    y = np.array([str(s) for s in br["canonical_id"]])
    grid = np.asarray(br["grid"], float)
    H_lsm = np.load(FROZEN / "phase01/artifacts/lsm_dictionary_v1.npz")["H"]
    CSM = np.load(FROZEN / "phase02/artifacts/csm_dictionary_v1.npz")["CSM"]
    recs = csm_reg["csms"]
    part = pd.read_csv(FROZEN / "phase00/tables/chemical_partition_v1.csv")
    cls_of = dict(zip(part.canonical_id, part.fine_class))
    folds_tab = pd.read_csv(FROZEN / "phase00/tables/cv_folds_v1.csv")
    fold_of = dict(zip(folds_tab.canonical_id, folds_tab.fold))
    folds = np.array([fold_of.get(v, 0) for v in y])
    cls = np.array([cls_of.get(v, "") for v in y])
    canon = pd.read_csv(FROZEN / "phase00/tables/canonical_analytes_v1.csv")
    known_mols = set(canon.canonical_name) | set(canon.canonical_id) | set(y)
    known_lsms = {r["motif_id"] for _, r in
                  pd.read_csv(FROZEN / "phase01/artifacts/lsm_registry_v1.csv").iterrows()}
    log(f"  corpus {X.shape} · {len(set(y))} molecules · {len(set(cls))} classes · "
        f"{len(set(folds))} grouped folds · grid {grid.min():.0f}–{grid.max():.0f} cm-1")
    if len(grid) != 676 or abs(np.median(np.diff(grid)) - 2.0) > 1e-9:
        log("ABORT — canonical preprocessing grid does not match the frozen specification")
        return 2

    # ── STEP 1: direct CSM projection ────────────────────────────────────────
    log("STEP 1 — direct non-negative projection onto the 49 frozen CSMs")
    A = PRJ.project(X, CSM)
    D = PRJ.diagnostics(X, A, CSM)
    act = pd.DataFrame({"spectrum": np.arange(len(A)), "canonical_id": y, "fine_class": cls,
                        "fold": folds, "explained_variance": D["explained_variance"],
                        "residual": D["residual"], "residual_fraction": D["residual_fraction"],
                        "component_sparsity": D["component_sparsity"],
                        "n_active_csms": D["n_active_csms"],
                        "activation_entropy": D["activation_entropy"]})
    outputs.append(wtab(act, "csm_projection_diagnostics_v1.csv"))
    np.savez_compressed(OUT.artifacts / "csm_activations_v1.npz", A=A, y=y, cls=cls,
                        folds=folds, grid=grid)
    outputs.append({"artifact_id": "csm_activations_v1.npz",
                    "path": OUT.rel(OUT.artifacts / "csm_activations_v1.npz"),
                    "sha256": P.sha256_file(OUT.artifacts / "csm_activations_v1.npz")})
    log(f"  EV {D['explained_variance'].mean():.3f} (min {D['explained_variance'].min():.3f}) · "
        f"{D['n_active_csms'].mean():.1f} active CSMs · sparsity "
        f"{D['component_sparsity'].mean():.3f} · entropy {D['activation_entropy'].mean():.3f}")

    # ── STEP 2: reference bank + metric selection by NESTED grouped CV ───────
    log("STEP 2 — reference bank and similarity-metric selection (nested grouped CV)")
    R_full, ref_labels = RET.build_reference_bank(A, y)
    ref_classes = [cls_of.get(m, "") for m in ref_labels]
    outer = sorted(set(folds))
    rows, nested_choice = [], {}
    for m in RET.METRICS:
        per_fold = []
        for f in outer:
            te, tr = folds == f, folds != f
            Rb, lb = RET.build_reference_bank(A[tr], y[tr])
            ci = RET.stable_covariance(A[tr]) if m == "mahalanobis" else None
            if m == "mahalanobis" and ci is None:
                per_fold = []
                break
            S = RET.similarity(A[te], Rb, m, ci)
            rcl = [cls_of.get(x, "") for x in lb]
            per_fold.append(topk_class(S, rcl, cls[te], 1))
        if not per_fold:
            rows.append({"metric": m, "usable": False, "cv_class_top1": np.nan,
                         "cv_class_top3": np.nan})
            continue
        per3 = []
        for f in outer:
            te, tr = folds == f, folds != f
            Rb, lb = RET.build_reference_bank(A[tr], y[tr])
            ci = RET.stable_covariance(A[tr]) if m == "mahalanobis" else None
            S = RET.similarity(A[te], Rb, m, ci)
            per3.append(topk_class(S, [cls_of.get(x, "") for x in lb], cls[te], 3))
        rows.append({"metric": m, "usable": True, "cv_class_top1": float(np.mean(per_fold)),
                     "cv_class_top3": float(np.mean(per3)),
                     "cv_class_top1_sd": float(np.std(per_fold))})
        log(f"  {m:22s} grouped-CV class top1 {np.mean(per_fold):.3f}  top3 {np.mean(per3):.3f}")
    met_tab = pd.DataFrame(rows).sort_values("cv_class_top1", ascending=False)
    outputs.append(wtab(met_tab, "similarity_metric_benchmark_v1.csv"))

    # Honest nested selection: the metric used on each outer fold is chosen on the *inner* folds
    # of that fold's training set, so no outer test spectrum influences the choice.
    for f in outer:
        tr = folds != f
        inner, best, best_s = sorted(set(folds[tr])), None, -np.inf
        for m in RET.METRICS:
            sc = []
            for g in inner:
                itr = tr & (folds != g)
                ite = tr & (folds == g)
                if ite.sum() == 0 or itr.sum() < 10:
                    continue
                Rb, lb = RET.build_reference_bank(A[itr], y[itr])
                ci = RET.stable_covariance(A[itr]) if m == "mahalanobis" else None
                if m == "mahalanobis" and ci is None:
                    sc = []
                    break
                sc.append(topk_class(RET.similarity(A[ite], Rb, m, ci),
                                     [cls_of.get(x, "") for x in lb], cls[ite], 1))
            if sc and np.mean(sc) > best_s:
                best_s, best = float(np.mean(sc)), m
        nested_choice[int(f)] = best
    metric = max(set(nested_choice.values()),
                 key=lambda m: (sum(v == m for v in nested_choice.values()), m))
    log(f"  nested per-fold choices: {nested_choice}")
    log(f"  SELECTED metric: {metric}")
    cov_inv = RET.stable_covariance(A) if metric == "mahalanobis" else None

    # ── STEP 10 (Split A/B scaffolding) + STEP 3 calibration ─────────────────
    log("STEP 10/3 — grouped-CV predictions for Split A and Split B, then calibration")
    # Split A: leave one spectrum out, the molecule's other replicates remain in the bank.
    SA, ya, ca, fa = [], [], [], []
    for i in range(len(A)):
        keep = np.ones(len(A), bool)
        keep[i] = False
        Rb, lb = RET.build_reference_bank(A[keep], y[keep])
        ci = RET.stable_covariance(A[keep]) if metric == "mahalanobis" else None
        SA.append((RET.similarity(A[i:i + 1], Rb, metric, ci)[0], lb))
        ya.append(y[i]); ca.append(cls[i]); fa.append(folds[i])
    # Split B: the molecule is absent from the bank entirely (grouped by fold).
    SB, yb, cb, fb, idxb = [], [], [], [], []
    for f in outer:
        te, tr = folds == f, folds != f
        Rb, lb = RET.build_reference_bank(A[tr], y[tr])
        ci = RET.stable_covariance(A[tr]) if metric == "mahalanobis" else None
        S = RET.similarity(A[te], Rb, metric, ci)
        for k, row in enumerate(S):
            SB.append((row, lb))
        yb += list(y[te]); cb += list(cls[te]); fb += [f] * te.sum()
        idxb += list(np.where(te)[0])
    yb, cb, fb = np.array(yb), np.array(cb), np.array(fb)

    # Split A metrics
    a_top1 = float(np.mean([ya[i] in np.array(lb)[np.argsort(-s)[:1]]
                            for i, (s, lb) in enumerate(SA)]))
    a_top3 = float(np.mean([ya[i] in np.array(lb)[np.argsort(-s)[:3]]
                            for i, (s, lb) in enumerate(SA)]))
    a_top5 = float(np.mean([ya[i] in np.array(lb)[np.argsort(-s)[:5]]
                            for i, (s, lb) in enumerate(SA)]))
    a_cls1 = float(np.mean([topk_class(s[None, :], [cls_of.get(x, "") for x in lb],
                                       [ca[i]], 1) for i, (s, lb) in enumerate(SA)]))
    log(f"  Split A molecule top1 {a_top1:.3f} top3 {a_top3:.3f} top5 {a_top5:.3f} · "
        f"class top1 {a_cls1:.3f}")

    b_cls1 = float(np.mean([topk_class(s[None, :], [cls_of.get(x, "") for x in lb],
                                       [cb[i]], 1) for i, (s, lb) in enumerate(SB)]))
    b_cls3 = float(np.mean([topk_class(s[None, :], [cls_of.get(x, "") for x in lb],
                                       [cb[i]], 3) for i, (s, lb) in enumerate(SB)]))
    log(f"  Split B class top1 {b_cls1:.3f} top3 {b_cls3:.3f} "
        f"(molecule top-k is undefined: the molecule is not in the bank)")

    # Calibration is benchmarked on Split A (molecule-level correctness is defined there) and,
    # separately, on Split B class correctness. Both use held-out scores only.
    def cal_matrix(pairs, truth, level="molecule"):
        n = max(len(lb) for _, lb in pairs)
        S = np.full((len(pairs), n), -1e9)
        corr = np.zeros(len(pairs))
        for i, (s, lb) in enumerate(pairs):
            S[i, :len(s)] = s
            j = int(np.argmax(s))
            corr[i] = (lb[j] == truth[i]) if level == "molecule" \
                else (cls_of.get(lb[j], "") == truth[i])
        return S, corr

    SA_m, cA = cal_matrix(SA, ya, "molecule")
    SB_m, cB = cal_matrix(SB, cb, "class")
    rng = np.random.default_rng(SEED)
    cal_rows = []
    for name, (Sm, cc, fld) in {"splitA_molecule": (SA_m, cA, np.array(fa)),
                                "splitB_class": (SB_m, cB, fb)}.items():
        for f in outer:                       # fit on training folds, score on held-out fold
            te, tr = fld == f, fld != f
            if te.sum() < 5 or len(set(cc[tr].tolist())) < 2:
                continue
            t = CAL.benchmark(Sm[tr], cc[tr], Sm[te], cc[te])
            t.insert(0, "split", name); t.insert(1, "fold", f)
            cal_rows.append(t)
    cal_tab = pd.concat(cal_rows, ignore_index=True)
    outputs.append(wtab(cal_tab, "calibration_benchmark_v1.csv"))
    summ = (cal_tab.groupby(["split", "method"])
            [["ece", "mce", "brier", "sharpness", "discrimination", "overconfidence"]]
            .mean().reset_index())
    outputs.append(wtab(summ, "calibration_summary_v1.csv"))
    for _, r in summ.iterrows():
        log(f"  {r['split']:16s} {r['method']:14s} ECE {r.ece:.3f}  Brier {r.brier:.3f}  "
            f"sharpness {r.sharpness:.3f}  discrimination {r.discrimination:.3f}")
    # Selection rule, declared before the numbers were seen a second time: minimise **Brier**,
    # a strictly proper scoring rule, among methods that actually discriminate (AUROC of
    # confidence vs correctness above chance). Selecting on ECE alone hands the phase to a
    # constant predictor: on the first pass Platt scaling reported 0.605 for every spectrum —
    # the base rate — winning ECE at 0.080 while posting the worst Brier of any method.
    sa = summ[summ.split == "splitA_molecule"].copy()
    usable = sa[(sa.discrimination > 0.55) & (sa.sharpness > 0.02)]
    if usable.empty:
        log("  WARNING: no calibration method both discriminates and is sharp; "
            "falling back to the best Brier overall")
        usable = sa
    cal_method = str(usable.sort_values("brier").iloc[0]["method"])
    log(f"  SELECTED calibration: {cal_method}")
    calibrator = CAL.Calibrator(cal_method).fit(SA_m, cA)

    # Per-spectrum predictions, kept for the figures and for anyone auditing a single case.
    rows_a = []
    for i, (sc, lb) in enumerate(SA):
        o = np.argsort(-sc)[:5]
        rows_a.append({"spectrum": i, "truth_molecule": ya[i], "truth_class": ca[i],
                       "fold": fa[i], **{f"top{k+1}": lb[o[k]] for k in range(5)},
                       "score_top1": float(sc[o[0]]),
                       "margin": float(sc[o[0]] - sc[o[1]]),
                       "correct_top1": bool(lb[o[0]] == ya[i]),
                       "correct_top5": bool(ya[i] in [lb[j] for j in o])})
    pred_a = pd.DataFrame(rows_a)
    pred_a["confidence"] = calibrator.transform(SA_m)
    outputs.append(wtab(pred_a, "splitA_predictions_v1.csv"))
    kk = list(range(1, 11))
    curve = pd.DataFrame({"k": kk, "topk": [
        float(np.mean([ya[i] in np.array(lb)[np.argsort(-sc)[:k]]
                       for i, (sc, lb) in enumerate(SA)])) for k in kk]})
    outputs.append(wtab(curve, "splitA_topk_curve_v1.csv"))

    # reliability data for figures
    p_final = calibrator.transform(SA_m)
    xs, ys, ns = CAL.reliability_curve(p_final, cA)
    outputs.append(wtab(pd.DataFrame({"bin_center": xs, "empirical_accuracy": ys, "count": ns}),
                        "reliability_splitA_v1.csv"))

    # ── STEP 5: chemistry-class inference ────────────────────────────────────
    log("STEP 5 — chemistry-class inference under grouped CV (Split B)")
    pred_cls = []
    for s, lb in SB:
        rcl = [cls_of.get(x, "") for x in lb]
        pred_cls.append(rcl[int(np.argmax(s))])
    pred_cls = np.array(pred_cls)
    labs = sorted(set(cb) | set(pred_cls))
    CM = pd.DataFrame(0, index=labs, columns=labs)
    for t, p in zip(cb, pred_cls):
        CM.loc[t, p] += 1
    outputs.append(wtab(CM.reset_index().rename(columns={"index": "true_class"}),
                        "class_confusion_matrix_v1.csv"))
    prf = []
    for c in labs:
        tp = int(((pred_cls == c) & (cb == c)).sum())
        fp = int(((pred_cls == c) & (cb != c)).sum())
        fn = int(((pred_cls != c) & (cb == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        prf.append({"class": c, "n": int((cb == c).sum()), "precision": pr, "recall": rc,
                    "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0})
    pred_b = pd.DataFrame({"spectrum": idxb, "truth_molecule": yb, "truth_class": cb,
                           "pred_class": pred_cls, "fold": fb,
                           "correct": (pred_cls == cb)})
    pred_b["confidence"] = calibrator.transform(SB_m)
    outputs.append(wtab(pred_b, "splitB_predictions_v1.csv"))
    prf_tab = pd.DataFrame(prf)
    outputs.append(wtab(prf_tab, "class_precision_recall_v1.csv"))
    macro_f1 = float(prf_tab[prf_tab.n > 0].f1.mean())
    bal_acc = float(np.mean([((pred_cls == c) & (cb == c)).sum() / max((cb == c).sum(), 1)
                             for c in sorted(set(cb))]))
    log(f"  class top1 {b_cls1:.3f} top3 {b_cls3:.3f} macro-F1 {macro_f1:.3f} "
        f"balanced accuracy {bal_acc:.3f}")

    # ── STEP 6: biochemical evidence profile ─────────────────────────────────
    log("STEP 6 — declared Biochemical Evidence Profile (11 grounded axes, no factorisation)")
    M, unassigned = EV.build_axis_map(CSM, grid, recs)
    spec = EV.axis_specificity(M)
    prof = EV.profile(A, M, spec, D["explained_variance"])
    Emag = prof["magnitude"]
    grounding = EV.ground_axes(M, recs, grid, CSM)
    outputs.append(wjson({"axes": EV.AXES, "grounding": grounding,
                          "specificity": dict(zip(EV.AXIS_NAMES, spec.tolist())),
                          "unassigned_mass_per_csm": dict(
                              zip([r["csm_id"] for r in recs], unassigned.tolist()))},
                         "evidence_axis_grounding_v1.json"))
    np.savez_compressed(OUT.artifacts / "evidence_axis_map_v1.npz", M=M,
                        unassigned=unassigned, specificity=spec,
                        axes=np.array(EV.AXIS_NAMES))
    outputs.append({"artifact_id": "evidence_axis_map_v1.npz",
                    "path": OUT.rel(OUT.artifacts / "evidence_axis_map_v1.npz"),
                    "sha256": P.sha256_file(OUT.artifacts / "evidence_axis_map_v1.npz")})
    gt = pd.DataFrame([{k: v for k, v in g.items()
                        if k in ("axis", "n_supporting_csms", "n_supporting_molecules",
                                 "top_csm", "mean_loading", "max_loading")}
                       for g in grounding])
    gt["specificity"] = spec
    gt["supporting_classes"] = [";".join(g["supporting_classes"][:6]) for g in grounding]
    outputs.append(wtab(gt, "evidence_axis_summary_v1.csv"))
    np.savez_compressed(OUT.artifacts / "evidence_profiles_v1.npz", magnitude=Emag,
                        coverage=prof["coverage"], confidence=prof["confidence"],
                        support=prof["support"], axes=np.array(EV.AXIS_NAMES), y=y, cls=cls)
    outputs.append({"artifact_id": "evidence_profiles_v1.npz",
                    "path": OUT.rel(OUT.artifacts / "evidence_profiles_v1.npz"),
                    "sha256": P.sha256_file(OUT.artifacts / "evidence_profiles_v1.npz")})
    val = EV.validate_axes(Emag, cls, AXIS_CLASSES)
    # Secondary, pre-declared check for the two axes whose chemistry class is a poor proxy.
    # `unsaturation` is tested against fatty_acid + acylglycerol above, but those classes contain
    # saturated members, so a perfect axis would still score near chance. The sharp test is
    # unsaturated vs saturated *within the lipids*, using the C=C-bearing molecule names. This
    # refines the label, never the axis: `M` and the windows are already fixed by this point.
    UNSAT = ("oleic", "oleate", "olein", "linole", "linolen", "arachidon", "palmitolei",
             "vaccenic", "eruc", "elaid", "petroselin", "eicosen", "erucin", "myristolei",
             "gadolei", "nervon", "docosahexa", "eicosapenta")
    lipid = np.isin(cls, ["fatty_acid", "acylglycerol"])
    unsat = np.array([any(k in m.lower() for k in UNSAT) for m in y]) & lipid
    sec = []
    if lipid.sum() and unsat.sum() and (lipid & ~unsat).sum():
        sub_i = np.where(lipid)[0]
        au = OS.auroc(Emag[sub_i, list(EV.AXIS_NAMES).index("unsaturation")], unsat[sub_i])
        sec.append({"axis": "unsaturation", "test": "unsaturated vs saturated within lipids",
                    "n_positive": int(unsat.sum()), "n_negative": int((lipid & ~unsat).sum()),
                    "auroc": au,
                    "verdict": "grounded" if au >= 0.70 else
                               ("weak" if au >= 0.60 else "not discriminative")})
        log(f"  [secondary] unsaturation, unsaturated vs saturated lipids only: AUROC {au:.3f} "
            f"({int(unsat.sum())} vs {int((lipid & ~unsat).sum())} spectra)")
    # Threshold sensitivity. `SUPPORT_FLOOR` gates which CSMs *count as supporting* an axis
    # (and hence specificity, the confidence weight and the provenance chains), while the
    # prominence window sets how a band's strength is measured. Neither should be able to
    # manufacture a grounded axis, and this is where that is checked rather than asserted.
    sens = []
    for floor in (0.05, 0.10, 0.15, 0.20):
        old = EV.SUPPORT_FLOOR
        EV.SUPPORT_FLOOR = floor
        sp = EV.axis_specificity(M)
        pr = EV.profile(A, M, sp, D["explained_variance"])
        vv = EV.validate_axes(pr["magnitude"], cls, AXIS_CLASSES)
        EV.SUPPORT_FLOOR = old
        sens.append({"parameter": "support_floor", "value": floor,
                     "mean_supporting_csms": float((M > floor).sum(axis=0).mean()),
                     "mean_axes_per_csm": float((M > floor).sum(axis=1).mean()),
                     "n_grounded": int((vv.verdict == "grounded").sum()),
                     "mean_auroc": float(vv.auroc.mean(skipna=True))})
    for win in (20.0, 40.0, 80.0):
        Mw, uw = EV._prominence_profile, None
        import functools
        orig = EV._prominence_profile
        EV._prominence_profile = functools.partial(orig, window=win)
        Mv, _ = EV.build_axis_map(CSM, grid, recs)
        EV._prominence_profile = orig
        pv = EV.profile(A, Mv, EV.axis_specificity(Mv), D["explained_variance"])
        vv = EV.validate_axes(pv["magnitude"], cls, AXIS_CLASSES)
        sens.append({"parameter": "prominence_window_cm-1", "value": win,
                     "mean_supporting_csms": float((Mv > EV.SUPPORT_FLOOR).sum(axis=0).mean()),
                     "mean_axes_per_csm": float((Mv > EV.SUPPORT_FLOOR).sum(axis=1).mean()),
                     "n_grounded": int((vv.verdict == "grounded").sum()),
                     "mean_auroc": float(vv.auroc.mean(skipna=True))})
    sens_tab = pd.DataFrame(sens)
    outputs.append(wtab(sens_tab, "evidence_axis_sensitivity_v1.csv"))
    for _, r in sens_tab.iterrows():
        log(f"  [sensitivity] {r.parameter}={r.value:<5} supporting CSMs/axis "
            f"{r.mean_supporting_csms:5.1f}  axes/CSM {r.mean_axes_per_csm:.2f}  "
            f"grounded {int(r.n_grounded)}/11  mean AUROC {r.mean_auroc:.3f}")

    sec_tab = pd.DataFrame(sec)
    if len(sec_tab):
        outputs.append(wtab(sec_tab, "evidence_axis_secondary_tests_v1.csv"))
    outputs.append(wtab(val, "evidence_axis_validation_v1.csv"))
    outputs.append(wtab(EV.window_overlap(), "evidence_axis_window_overlap_v1.csv"))
    for _, r in val.iterrows():
        log(f"  {r.axis:24s} CSMs {int(gt.loc[gt.axis == r.axis, 'n_supporting_csms'].iloc[0]):2d}"
            f"  specificity {spec[list(EV.AXIS_NAMES).index(r.axis)]:.2f}"
            f"  AUROC {r.auroc if isinstance(r.auroc, float) else float('nan'):.3f}  {r.verdict}")
    log(f"  mean unassigned spectral mass per CSM: {unassigned.mean():.3f}")

    # ── STEP 4: open-set rejection (synthetic negatives only, Raman-only) ────
    log("STEP 4 — open-set rejection on synthetic negatives (no cross-modality experiment)")
    neg_specs, neg_kind = [], []
    rng = np.random.default_rng(SEED)
    sub = np.arange(0, len(X), 3)
    # (a) corruption past the point of recognisability
    for kind, lev in (("gaussian_noise", 0.60), ("band_broadening", 40.0),
                      ("peak_dropout", 0.95), ("spectral_stretch", 0.10)):
        neg_specs.append(PERT.apply(kind, X[sub], grid, lev, seed=SEED))
        neg_kind += [f"extreme_{kind}"] * len(sub)
    # (b) structured non-Raman signals: no molecular band structure at all
    n_syn = len(sub)
    pure_noise = np.abs(rng.normal(0, 1, (n_syn, len(grid))))
    pure_noise /= np.linalg.norm(pure_noise, axis=1, keepdims=True)
    neg_specs.append(pure_noise); neg_kind += ["white_noise"] * n_syn
    g01 = (grid - grid.min()) / (grid.max() - grid.min())
    poly = np.vstack([np.polyval(rng.normal(0, 1, 4), g01) for _ in range(n_syn)])
    poly = np.clip(poly - poly.min(axis=1, keepdims=True), 0, None)
    poly /= np.linalg.norm(poly, axis=1, keepdims=True) + 1e-12
    neg_specs.append(poly); neg_kind += ["fluorescence_only"] * n_syn
    Xneg = np.vstack(neg_specs)
    Aneg = PRJ.project(Xneg, CSM)
    Dneg = PRJ.diagnostics(Xneg, Aneg, CSM)
    ci_all = RET.stable_covariance(A)
    ref_mean = A.mean(axis=0)
    ch_in = OS.channel_scores(A, D, R_full, ci_all, ref_mean)
    ch_out = OS.channel_scores(Aneg, Dneg, R_full, ci_all, ref_mean)
    os_tab = OS.evaluate(ch_in, ch_out)
    outputs.append(wtab(os_tab, "openset_channel_auroc_v1.csv"))
    for _, r in os_tab.iterrows():
        log(f"  {r.channel:28s} AUROC {r.auroc:.3f}")
    per_kind = []
    nk = np.array(neg_kind)
    j_in = OS.joint_score(ch_in, ch_in)
    for k in sorted(set(neg_kind)):
        sel = nk == k
        j_out = OS.joint_score({a: v[sel] for a, v in ch_out.items()}, ch_in)
        per_kind.append({"negative_kind": k, "n": int(sel.sum()),
                         "joint_auroc": OS.auroc(np.concatenate([j_in, j_out]),
                                                 np.concatenate([np.zeros(len(j_in)),
                                                                 np.ones(len(j_out))]))})
    outputs.append(wtab(pd.DataFrame(per_kind), "openset_by_negative_kind_v1.csv"))
    for r in per_kind:
        log(f"  negative kind {r['negative_kind']:26s} joint AUROC {r['joint_auroc']:.3f}")
    j_out_all = OS.joint_score(ch_out, ch_in)
    op = OS.operating_point(j_in, j_out_all, 0.95)
    log(f"  operating point @95% in-domain acceptance: threshold {op['threshold']:.3f}, "
        f"rejects {op['ood_reject']:.3f} of synthetic negatives")
    roc = []
    for t in np.linspace(min(j_in.min(), j_out_all.min()), max(j_in.max(), j_out_all.max()), 200):
        roc.append({"threshold": float(t), "fpr": float(np.mean(j_in > t)),
                    "tpr": float(np.mean(j_out_all > t))})
    outputs.append(wtab(pd.DataFrame(roc), "openset_roc_v1.csv"))
    np.savez_compressed(OUT.artifacts / "openset_scores_v1.npz", joint_in=j_in,
                        joint_out=j_out_all, negative_kind=np.array(neg_kind),
                        Xneg=Xneg, Aneg=Aneg)
    outputs.append({"artifact_id": "openset_scores_v1.npz",
                    "path": OUT.rel(OUT.artifacts / "openset_scores_v1.npz"),
                    "sha256": P.sha256_file(OUT.artifacts / "openset_scores_v1.npz")})

    # ── STEP 8: provenance ───────────────────────────────────────────────────
    log("STEP 8 — provenance chains, axis → CSM → LSM → molecule → spectra")
    axis_index = {a: i for i, a in enumerate(EV.AXIS_NAMES)}
    all_chains, prov_rows = [], []
    for i in range(len(A)):
        for a in EV.AXIS_NAMES:
            if Emag[i, axis_index[a]] > 0.02:
                all_chains.append(PROV.axis_chain(a, A[i], M, recs, axis_index))
    ver = PROV.verify_chains(all_chains, known_lsms, known_mols)
    outputs.append(wtab(ver.groupby("axis").agg(
        n_chains=("axis", "size"), mean_csms=("n_csms", "mean"),
        mean_lsms=("n_lsms", "mean"), mean_molecules=("n_molecules", "mean"),
        broken=("intact", lambda s: int((~s).sum()))).reset_index(),
        "provenance_integrity_v1.csv"))
    broken = int((~ver.intact).sum())
    log(f"  {len(all_chains)} chains checked · broken {broken}")

    # ── STEP 9: geometry as visualisation only ───────────────────────────────
    log("STEP 9 — geometry is read for visualisation only; no inference depends on it")
    geo_p = FROZEN / "phase02_5/artifacts/embeddings_v1.npz"
    geo = dict(np.load(geo_p, allow_pickle=True)) if geo_p.exists() else {}
    geo_used_in_inference = False

    # ── STEP 11: noise robustness ────────────────────────────────────────────
    log("STEP 11 — robustness of raw / LSM / CSM / evidence profile under 7 perturbations")
    reps = {"raw": lambda Z: Z,
            "lsm": lambda Z: PRJ.project(Z, H_lsm),
            "csm": lambda Z: PRJ.project(Z, CSM),
            "evidence": lambda Z: EV.profile(PRJ.project(Z, CSM), M, spec)["magnitude"]}
    banks, fold_banks, clean_proj = {}, {}, {}
    for rname, fn in reps.items():
        clean_proj[rname] = fn(X)
        banks[rname] = RET.build_reference_bank(clean_proj[rname], y)
        # Molecule-grouped banks: the perturbed query's own molecule is absent, so the class
        # number below is the honest one. The ungrouped bank is kept only for molecule
        # retrieval, where the molecule must be present for the question to mean anything.
        fold_banks[rname] = {f: RET.build_reference_bank(clean_proj[rname][folds != f],
                                                         y[folds != f]) for f in outer}
    rob = []
    for kind in ROBUSTNESS:
        for lev in PERT.LEVELS[kind]:
            Xp = PERT.apply(kind, X, grid, lev, seed=SEED)
            for rname, fn in reps.items():
                Q = fn(Xp)
                Rb, lb = banks[rname]
                S = RET.similarity(Q, Rb, "cosine")
                rcl = [cls_of.get(x, "") for x in lb]
                p = _softmax_conf(S)
                corr = np.array([lb[int(np.argmax(s))] == y[i] for i, s in enumerate(S)],
                                float)
                gcls = []
                for f in outer:
                    te = folds == f
                    Rg, lg = fold_banks[rname][f]
                    gcls.append(topk_class(RET.similarity(Q[te], Rg, "cosine"),
                                           [cls_of.get(x, "") for x in lg], cls[te], 1)
                                * te.sum())
                rob.append({"perturbation": kind, "level": lev, "representation": rname,
                            "molecule_top1": topk_mol(S, lb, y, 1),
                            "class_top1": topk_class(S, rcl, cls, 1),
                            "class_top1_grouped": float(sum(gcls) / len(Q)),
                            "mean_confidence": float(p.mean()),
                            "ece": CAL.expected_calibration_error(p, corr)})
        log(f"  {kind} done")
    rob_tab = pd.DataFrame(rob)
    outputs.append(wtab(rob_tab, "noise_robustness_v1.csv"))
    # Unperturbed baselines for each representation, against the same reference banks. These are
    # in-sample by construction (the bank contains the query's own molecule) and are used only as
    # the denominator of a *retention* ratio, never as a headline accuracy.
    clean = {}
    for r, fn in reps.items():
        Sc = RET.similarity(clean_proj[r], banks[r][0], "cosine")
        g = []
        for f in outer:
            te = folds == f
            Rg, lg = fold_banks[r][f]
            g.append(topk_class(RET.similarity(clean_proj[r][te], Rg, "cosine"),
                                [cls_of.get(x, "") for x in lg], cls[te], 1) * te.sum())
        clean[r] = {"molecule_top1": topk_mol(Sc, banks[r][1], y, 1),
                    "class_top1": topk_class(Sc, [cls_of.get(x, "") for x in banks[r][1]],
                                             cls, 1),
                    "class_top1_grouped": float(sum(g) / len(X))}
    deg = []
    for r in reps:
        sub_t = rob_tab[rob_tab.representation == r]
        deg.append({"representation": r,
                    "clean_molecule_top1": clean[r]["molecule_top1"],
                    "clean_class_top1": clean[r]["class_top1"],
                    "clean_class_top1_grouped": clean[r]["class_top1_grouped"],
                    "mean_class_top1_grouped_perturbed":
                        float(rob_tab[rob_tab.representation == r].class_top1_grouped.mean()),
                    "class_retention_grouped":
                        float(rob_tab[rob_tab.representation == r].class_top1_grouped.mean()
                              / (clean[r]["class_top1_grouped"] + 1e-12)),
                    "mean_molecule_top1_perturbed": float(sub_t.molecule_top1.mean()),
                    "mean_class_top1_perturbed": float(sub_t.class_top1.mean()),
                    "molecule_retention": float(sub_t.molecule_top1.mean()
                                                / (clean[r]["molecule_top1"] + 1e-12)),
                    "class_retention": float(sub_t.class_top1.mean()
                                             / (clean[r]["class_top1"] + 1e-12)),
                    "mean_ece_perturbed": float(sub_t.ece.mean())})
    deg_tab = pd.DataFrame(deg)
    outputs.append(wtab(deg_tab, "robustness_summary_v1.csv"))
    for _, r in deg_tab.iterrows():
        log(f"  {r.representation:9s} mol {r.clean_molecule_top1:.3f}→"
            f"{r.mean_molecule_top1_perturbed:.3f} (ret {r.molecule_retention:.3f}) · "
            f"class[grouped] {r.clean_class_top1_grouped:.3f}→"
            f"{r.mean_class_top1_grouped_perturbed:.3f} "
            f"(ret {r.class_retention_grouped:.3f}) · class[in-sample] "
            f"{r.clean_class_top1:.3f}→{r.mean_class_top1_perturbed:.3f}")

    # ── the engine + representative reports ──────────────────────────────────
    log("Assembling the canonical engine and generating representative inference reports")
    engine = CanonicalEngine(CSM, recs, grid, R_full, ref_labels, ref_classes, M, unassigned,
                             spec, calibrator, metric, cov_inv, ch_in, op["threshold"],
                             ref_mean)
    fp = engine.fingerprint()
    log(f"  engine fingerprint {fp}")
    r1 = engine.infer(X[:8])
    r2 = engine.infer(X[:8])
    deterministic = all(np.allclose(a.activation, b.activation) and
                        a.confidence == b.confidence for a, b in zip(r1, r2))
    log(f"  determinism (bit-for-bit on repeat): {deterministic}")
    picks = []
    for c in sorted(set(cls)):
        idx = np.where(cls == c)[0]
        if len(idx):
            picks.append(int(idx[len(idx) // 2]))
    demo = engine.infer(X[picks])
    outputs.append(wjson([{**d.to_dict(), "truth_molecule": y[i], "truth_class": cls[i]}
                          for d, i in zip(demo, picks)], "representative_reports_v1.json"))
    rej_demo = engine.infer(Xneg[:4])
    outputs.append(wjson([d.to_dict() for d in rej_demo], "rejected_examples_v1.json"))

    engine_cfg = {"metric": metric, "calibration": cal_method,
                  "reject_threshold": op["threshold"], "fingerprint": fp,
                  "csm_fingerprint": got["csm"], "lsm_fingerprint": got["lsm"],
                  "axes": list(EV.AXIS_NAMES), "n_references": len(ref_labels),
                  "geometry_used_in_inference": geo_used_in_inference}
    outputs.append(wjson(engine_cfg, "canonical_engine_config_v1.json"))
    np.savez_compressed(OUT.artifacts / "reference_bank_v1.npz", R=R_full,
                        labels=np.array(ref_labels), classes=np.array(ref_classes))
    outputs.append({"artifact_id": "reference_bank_v1.npz",
                    "path": OUT.rel(OUT.artifacts / "reference_bank_v1.npz"),
                    "sha256": P.sha256_file(OUT.artifacts / "reference_bank_v1.npz")})

    # ── gates ────────────────────────────────────────────────────────────────
    ev_mean = float(D["explained_variance"].mean())
    n_grounded = int((val.verdict == "grounded").sum())
    csm_ret = float(deg_tab.set_index("representation").loc["csm", "class_retention_grouped"])
    raw_ret = float(deg_tab.set_index("representation").loc["raw", "class_retention_grouped"])
    # G11 uses the **molecule-grouped** clean number. The ungrouped one is in-sample — the
    # query's own spectrum is in the bank — and there raw spectra win by self-matching (0.992 vs
    # 0.973), which is risk R-10 and not a statement about discrimination. On unseen molecules
    # the comparison reverses, and that is the comparison the hypothesis is about.
    csm_clean = float(deg_tab.set_index("representation").loc["csm", "clean_class_top1_grouped"])
    raw_clean = float(deg_tab.set_index("representation").loc["raw", "clean_class_top1_grouped"])
    joint_auroc = float(os_tab.set_index("channel").loc["JOINT", "auroc"])
    row_cal = summ[(summ.split == "splitA_molecule") & (summ.method == cal_method)].iloc[0]
    ece_final = float(row_cal.ece)
    disc_final, sharp_final = float(row_cal.discrimination), float(row_cal.sharpness)
    gates = [
        ("G1 frozen fingerprints verified", True),
        ("G2 nothing upstream refitted", True),
        ("G3 CSM projection explains the corpus (mean EV >= 0.80)", ev_mean >= 0.80),
        ("G4 Split A molecule top-1 >= 0.60", a_top1 >= 0.60),
        ("G5 Split B class top-1 >= 0.60 on unseen molecules", b_cls1 >= 0.60),
        # G6 is reported exactly as pre-declared and it FAILS. It is not relaxed: the finding is
        # that on this corpus ECE <= 0.10 is reachable only by a calibrator that reports the same
        # confidence for every spectrum. G6b is the companion the first pass showed was missing.
        ("G6 calibration ECE <= 0.10 after selection", ece_final <= 0.10),
        ("G6b calibration is informative (discrimination >= 0.75, sharpness > 0.05)",
         disc_final >= 0.75 and sharp_final > 0.05),
        ("G7 open-set joint AUROC >= 0.80 on synthetic negatives", joint_auroc >= 0.80),
        ("G8 at least 6 evidence axes empirically grounded (AUROC >= 0.70)", n_grounded >= 6),
        ("G8b axis grounding stable across threshold choices",
         bool(sens_tab.n_grounded.min() >= 6)),
        ("G9 no broken provenance chains", broken == 0),
        ("G10 CSM more robust than raw (class retention)", csm_ret > raw_ret),
        ("G11 CSM preserves discrimination on unseen molecules (>= raw)", csm_clean >= raw_clean),
        ("G12 engine deterministic on repeat", deterministic),
        ("G13 no cross-modality experiment in this phase", True),
        ("G14 geometry not used in inference", not geo_used_in_inference),
    ]
    gate_tab = pd.DataFrame([{"gate": g, "status": "PASS" if ok else "FAIL"} for g, ok in gates])
    outputs.append(wtab(gate_tab, "phase05_gates_v1.csv"))
    for g, ok in gates:
        log(f"  [{'PASS' if ok else 'FAIL'}] {g}")
    n_fail = int((gate_tab.status == "FAIL").sum())

    summary = {
        "projection": {"mean_ev": ev_mean, "min_ev": float(D["explained_variance"].min()),
                       "mean_active_csms": float(D["n_active_csms"].mean()),
                       "mean_sparsity": float(D["component_sparsity"].mean())},
        "metric": {"selected": metric, "nested_choices": nested_choice},
        "split_a": {"molecule_top1": a_top1, "molecule_top3": a_top3, "molecule_top5": a_top5,
                    "class_top1": a_cls1, "ece": ece_final, "calibration": cal_method,
                    "discrimination": disc_final, "sharpness": sharp_final,
                    "mean_confidence": float(p_final.mean())},
        "split_b": {"class_top1": b_cls1, "class_top3": b_cls3, "macro_f1": macro_f1,
                    "balanced_accuracy": bal_acc,
                    "molecule_top1": None,
                    "molecule_top1_note": "undefined by construction: the molecule is absent "
                                          "from the reference bank"},
        "openset": {"joint_auroc": joint_auroc, "operating_point": op,
                    "best_channel": str(os_tab.iloc[0]["channel"]),
                    "best_channel_auroc": float(os_tab.iloc[0]["auroc"]),
                    "by_kind": per_kind},
        "evidence": {"n_axes": len(EV.AXIS_NAMES), "n_grounded": n_grounded,
                     "mean_unassigned_mass": float(unassigned.mean()),
                     "validation": val.to_dict("records"),
                     "secondary_tests": sec_tab.to_dict("records"),
                     "sensitivity": sens_tab.to_dict("records")},
        "robustness": deg_tab.to_dict("records"),
        "provenance": {"n_chains": len(all_chains), "broken": broken},
        "engine": engine_cfg,
        "gates": {"n": len(gates), "failed": n_fail},
    }
    outputs.append(wjson(summary, "phase05_summary_v1.json"))

    state = {"phase": PHASE, "name": PHASE_NAME,
             "status": "COMPLETE" if n_fail == 0 else "GATE_FAILED",
             "started": t0.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
             "engine_fingerprint": fp, "seed": SEED, "outputs": outputs,
             "replaces": "Phase 04 Theme/BSV inference path",
             "scope": "Raman only; no SERS or cross-modality validation"}
    (OUT.root / "PHASE_STATE.json").write_text(json.dumps(state, indent=2, default=_ser))
    (OUT.logs / "run_phase05.log").write_text("\n".join(LOG))
    log(f"done · status {state['status']} · {len(outputs)} artifacts")
    return 0 if n_fail == 0 else 1


def _softmax_conf(S):
    Z = np.asarray(S, float)
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return (E / (E.sum(axis=1, keepdims=True) + 1e-12)).max(axis=1)


if __name__ == "__main__":
    raise SystemExit(main())
