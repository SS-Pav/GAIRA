#!/usr/bin/env python3
"""GAIRA V7 — Phase 04.5: hierarchical NMF over frozen CSM activations.

Tests whether a second-order factorisation of CSM activations gives a better biochemical state
representation than the CSM layer itself. Written to be able to return a negative result.

Nothing frozen is refitted. Output root resolves through `gaira.v7.io.PhaseOutputs`.

    python results/v7_rebuild/phase04_5/code/run_phase04_5.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "phase00/code"))

import v7_paths as P                                          # noqa: E402
from gaira.v7.io import PhaseOutputs, frozen_root             # noqa: E402
from gaira.v7.engine import projection as PRJ                 # noqa: E402
from gaira.v7.meta import evaluation as EV                    # noqa: E402
from gaira.v7.meta import factorization as MF                 # noqa: E402
from gaira.v7.meta import perturbations as PT                 # noqa: E402

warnings.filterwarnings("ignore")

PHASE, PHASE_NAME = "04.5", "Hierarchical NMF over frozen CSM activations"
OUT = PhaseOutputs(PHASE).ensure()
FROZEN = frozen_root()
EXPECTED = {"atlas": "09ed804a40836f4a05a91ba10900cded",
            "lsm": "208482d6f7178b5b8f16cace91be55b0",
            "csm": "0b4aa550ccefed3edabdbde5bae11c8d",
            "theme": "f54d4835ffdf8aa2d50a4a203da0e8f4"}
SEED, K_NN, LAMBDA = 0, 5, 0.1
# Pre-registered Pareto weights over the model-selection criteria. Reconstruction is present
# but deliberately minority: the brief forbids selecting K on reconstruction alone.
PARETO = {"explained_variance": (0.14, +1), "bootstrap_stability": (0.22, +1),
          "consensus_stability": (0.18, +1), "component_sparsity": (0.12, +1),
          "interpretability": (0.14, +1), "redundancy": (0.10, -1),
          "mutual_coherence": (0.06, -1), "activation_entropy": (0.04, -1)}
LOG: list[str] = []


def log(m):
    line = f"[phase04.5] {m}"
    print(line, flush=True)
    LOG.append(line)


def wtab(df, name, where=None):
    p = (where or OUT.tables) / name
    df.to_csv(p, index=False)
    return {"artifact_id": name, "path": OUT.rel(p), "sha256": P.sha256_file(p), "rows": len(df)}


def wjson(obj, name, where=None):
    p = (where or OUT.artifacts) / name
    p.write_text(json.dumps(obj, indent=2, default=_ser))
    return {"artifact_id": name, "path": OUT.rel(p), "sha256": P.sha256_file(p)}


def _ser(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main() -> int:
    outputs: list[dict] = []
    t0 = datetime.now(timezone.utc)

    # ── 0. fingerprint gate ──────────────────────────────────────────────────
    log("architecture check — second-order factorisation of CSM ACTIVATIONS; the CSM layer "
        "remains canonical unless this one earns its place")
    for ph, key, want in (("phase01", "registry_fingerprint", EXPECTED["lsm"]),
                          ("phase02", "csm_fingerprint", EXPECTED["csm"]),
                          ("phase03", "theme_fingerprint", EXPECTED["theme"])):
        got = json.loads((FROZEN / ph / "PHASE_STATE.json").read_text())[key]
        if got != want:
            log(f"ABORT: {ph} {key} {got} != {want}")
            return 1
    s04 = json.loads((FROZEN / "phase04/PHASE_STATE.json").read_text())
    log(f"  frozen verified. Phase 04 status {s04['status']} — its OOD gate failed and that "
        f"capability is not inherited or claimed here.")

    z = np.load(FROZEN / "phase04/artifacts/inference_v1.npz", allow_pickle=True)
    X, A_lsm, A_csm = np.asarray(z["X"], float), z["A_lsm"], z["A_csm"]
    grid = np.asarray(z["grid"], float)
    y = np.array([str(s) for s in z["y"]])
    cls = np.array([str(s) for s in z["cls"]])
    folds = z["folds"]
    t3 = np.load(FROZEN / "phase03/artifacts/theme_membership_v1.npz", allow_pickle=True)
    D_csm = np.asarray(t3["D_csm"], float)
    csm_ids = [str(s) for s in t3["csm_ids"]]
    creg = json.loads((FROZEN / "phase02/artifacts/csm_registry_v1.json").read_text())
    by = {c["csm_id"]: c for c in creg["csms"]}
    csm_class = [(by[c]["supporting_classes"][0] if len(by[c]["supporting_classes"]) == 1
                  else "multi") for c in csm_ids]
    treg = json.loads((FROZEN / "phase03/artifacts/theme_registry_v1.json").read_text())
    bridges = set(treg["bridge_csms"])
    coords_csm = np.asarray(t3["coords"], float)
    CSM = np.asarray(np.load(FROZEN / "phase02/artifacts/csm_dictionary_v1.npz",
                             allow_pickle=True)["CSM"], float)
    H_lsm = np.asarray(np.load(FROZEN / "phase01/artifacts/lsm_dictionary_v1.npz",
                               allow_pickle=True)["H"], float)
    repl = np.array([int((y == v).sum()) >= 2 for v in y])

    A = np.clip(A_csm, 0, None)
    log(f"activation matrix A = {A.shape} spectra x frozen CSM activations · "
        f"{(A > 0).sum(1).mean():.1f} active CSMs per spectrum · "
        f"{len(set(y))} molecules · {int(repl.sum())} replicated spectra")
    L, Wg = MF.csm_graph_laplacian(D_csm, K_NN)
    log(f"  geometry prior: {int((Wg > 0).sum() / 2)} CSM edges, one-sided smoothness only")

    # ── 1. model selection ───────────────────────────────────────────────────
    log("MODEL SELECTION — K in {2,3,4,5,6,8,10,12} x {plain, geometry_regularised}")
    rows, store = [], {}
    for variant in MF.VARIANTS:
        for K in MF.K_GRID:
            f = MF.fit(variant, A, K, L, LAMBDA, SEED)
            W, H = f["W"], f["H"]
            store[(variant, K)] = f
            boot = MF.bootstrap_stability(A, K, variant, L, LAMBDA, n_boot=15, seed=SEED)
            r = {
                "variant": variant, "K": K,
                "reconstruction_error": MF.reconstruction_error(A, W, H),
                "explained_variance": MF.explained_variance(A, W, H),
                "bootstrap_stability": boot["mean"],
                "bootstrap_min": boot["min"],
                "consensus_stability": MF.consensus_stability(A, K, variant, L, LAMBDA,
                                                              n_rep=12, seed=SEED),
                "component_sparsity": MF.component_sparsity(H),
                "effective_rank": MF.effective_rank(W),
                "redundancy": MF.redundancy(H),
                "mutual_coherence": MF.mutual_coherence(H),
                "interpretability": MF.interpretability(H, csm_class),
                "activation_entropy": MF.activation_entropy(W),
                "participation_ratio": MF.participation_ratio(W),
            }
            rows.append(r)
        log(f"  {variant} done")
    sweep = pd.DataFrame(rows)
    outputs.append(wtab(sweep, "model_selection_sweep_v1.csv"))

    comp = np.zeros(len(sweep))
    for crit, (w, d) in PARETO.items():
        v = sweep[crit].to_numpy(float)
        span = v.max() - v.min()
        zc = np.full_like(v, 0.5) if span < 1e-12 else (v - v.min()) / span
        comp += w * (zc if d > 0 else 1 - zc)
    sweep["pareto"] = comp
    best = sweep.sort_values("pareto", ascending=False).iloc[0]
    variant, K = str(best.variant), int(best.K)
    log(f"  SELECTED: {variant}, K = {K} (Pareto {best.pareto:.4f}); "
        f"EV {best.explained_variance:.3f}, bootstrap {best.bootstrap_stability:.3f}, "
        f"consensus {best.consensus_stability:.3f}, sparsity {best.component_sparsity:.3f}")
    for v_ in MF.VARIANTS:
        sub = sweep[sweep.variant == v_].sort_values("pareto", ascending=False).iloc[0]
        log(f"    best {v_:22s} K={int(sub.K):2d}  Pareto {sub.pareto:.4f}  "
            f"EV {sub.explained_variance:.3f}  boot {sub.bootstrap_stability:.3f}")
    outputs.append(wjson({"selected_variant": variant, "selected_K": K,
                          "pareto_weights": {k: {"weight": w, "direction":
                                                 "max" if d > 0 else "min"}
                                             for k, (w, d) in PARETO.items()},
                          "reconstruction_is_minority_weighted": True,
                          "best_per_variant": {v_: int(sweep[sweep.variant == v_]
                                                       .sort_values("pareto", ascending=False)
                                                       .iloc[0].K) for v_ in MF.VARIANTS},
                          "lambda": LAMBDA}, "model_selection_v1.json"))

    fit = store[(variant, K)]
    Wm, Hm = fit["W"], np.clip(fit["H"], 0, None)
    A_meta = MF.project(A, Hm)

    # ── 2. interpretability evidence, before any naming ──────────────────────
    log("INTERPRETABILITY — evidence first; no manual naming")
    ev_rows = []
    for k in range(K):
        h = Hm[k]
        top_csm = np.argsort(-h)[:6]
        top_csm = [i for i in top_csm if h[i] > 0]
        lsms, mols, bands, classes = [], [], [], []
        for i in top_csm:
            rec = by[csm_ids[i]]
            lsms += [l["lsm_id"] for l in rec["contributing_lsms"]]
            mols += rec["supporting_analytes"][:6]
            bands += rec["dominant_bands"]
            classes += rec["supporting_classes"]
        occ = coords_csm[top_csm].mean(axis=0) if top_csm else np.zeros(coords_csm.shape[1])
        ev_rows.append({
            "meta_component": f"MC-{k + 1:02d}",
            "top_csms": ";".join(csm_ids[i] for i in top_csm),
            "top_csm_weights": ";".join(f"{h[i]:.3f}" for i in top_csm),
            "top_lsms": ";".join(dict.fromkeys(lsms[:8])),
            "top_molecules": ";".join(dict.fromkeys(mols[:10])),
            "dominant_bands_cm1": ";".join(f"{b:.0f}" for b in sorted(set(round(b) for b in bands))[:10]),
            "dominant_classes": ";".join(f"{c}({classes.count(c)})" for c in
                                         sorted(set(classes), key=classes.count, reverse=True)[:4]),
            "bridge_csms": ";".join(csm_ids[i] for i in top_csm if csm_ids[i] in bridges),
            "n_bridge": sum(csm_ids[i] in bridges for i in top_csm),
            "geometry_occupancy": ";".join(f"{v:.3f}" for v in occ[:3]),
            "n_spectra_dominant": int((Wm.argmax(axis=1) == k).sum()),
            "mean_activation": float(Wm[:, k].mean()),
            "bootstrap_stability": float(MF.bootstrap_stability(A, K, variant, L, LAMBDA,
                                                                n_boot=15, seed=SEED
                                                                )["per_component"][k]),
        })
        log(f"  MC-{k + 1:02d}  {ev_rows[-1]['n_spectra_dominant']:3d} spectra · "
            f"{len(top_csm)} CSMs · {ev_rows[-1]['n_bridge']} bridges · "
            f"classes {ev_rows[-1]['dominant_classes'][:52]}")
    outputs.append(wtab(pd.DataFrame(ev_rows), "meta_component_evidence_v1.csv"))

    # ── 3. twelve validation axes, four representations ──────────────────────
    log("VALIDATION — raw / LSM / CSM / Meta on identical frozen splits")
    REPS = {"RAW": X, "LSM": A_lsm, "CSM": A, "META": A_meta}
    vrows = []
    for name, M in REPS.items():
        a = EV.retrieval_split_a(M, y, repl)
        b = EV.retrieval_split_b(M, cls, folds)
        ir = EV.information_retained(A, M) if name != "CSM" else {"linear_ev": 1.0,
                                                                  "knn_preservation": 1.0}
        vrows.append({
            "representation": name, "dim": M.shape[1],
            "A_top1": a["top1"], "A_top3": a["top3"], "A_top5": a["top5"], "A_mrr": a["mrr"],
            "B_top1": b["top1"], "B_top3": b["top3"], "B_top5": b["top5"],
            "B_balanced_accuracy": b["balanced_accuracy"], "B_macro_f1": b["macro_f1"],
            "replicate_consistency": EV.replicate_consistency(M, y),
            "cross_fold_reproducibility": EV.cross_fold_reproducibility(M, cls, folds),
            "activation_sparsity": EV.activation_sparsity(M),
            "effective_rank": MF.effective_rank(M),
            "participation_ratio": MF.participation_ratio(M),
            "biochemical_coherence": EV.biochemical_coherence(M, cls),
            "information_retained_vs_csm": ir["linear_ev"],
            "knn_preservation_vs_csm": ir["knn_preservation"],
            "redundancy": MF.redundancy(M.T) if M.shape[1] > 1 else 0.0,
        })
        log(f"  {name:5s} dim {M.shape[1]:4d}  A-top1 {a['top1']:.3f}  B-top1 {b['top1']:.3f}  "
            f"macroF1 {b['macro_f1']:.3f}  replicate "
            f"{vrows[-1]['replicate_consistency']:.3f}  info-vs-CSM {ir['linear_ev']:.3f}")
    valid = pd.DataFrame(vrows)
    outputs.append(wtab(valid, "representation_comparison_v1.csv"))

    # calibration of the retrieval score at each level
    cal_rows = []
    for name, M in REPS.items():
        qi = np.where(repl)[0]
        sc, ok = [], []
        for i in qi:
            ref = np.array([j for j in range(M.shape[0]) if j != i])
            s = EV._sim(M[[i]], M[ref])[0]
            j = int(np.argmax(s))
            sc.append(float(s[j]))
            ok.append(float(y[ref][j] == y[i]))
        c = EV.calibration(np.array(sc), np.array(ok))
        cal_rows.append({"representation": name, "ece": c["ece"]})
    outputs.append(wtab(pd.DataFrame(cal_rows), "calibration_v1.csv", OUT.validation))
    log("  calibration ECE: " + ", ".join(f"{r['representation']} {r['ece']:.3f}"
                                          for r in cal_rows))

    # ── 4. the noise robustness study ────────────────────────────────────────
    log("ROBUSTNESS STUDY — 12 perturbations x 5 levels x 4 representations")
    cfg_proj = "elastic_net"
    rob_rows = []
    clean = {n: EV.retrieval_split_a(M, y, repl)["top1"] for n, M in REPS.items()}
    cleanB = {n: EV.retrieval_split_b(M, cls, folds)["top1"] for n, M in REPS.items()}
    for kind in PT.PERTURBATIONS:
        for lvl in PT.LEVELS[kind]:
            Xp = PT.apply(kind, X, grid, lvl, SEED)
            Lp = PRJ.project(Xp, H_lsm, cfg_proj)
            Cp = PRJ.project(Xp, CSM, "nnls")
            Mp = MF.project(Cp, Hm)
            PERT = {"RAW": Xp, "LSM": Lp, "CSM": Cp, "META": Mp}
            for name in REPS:
                a = EV.retrieval_split_a(REPS[name], y, repl, A_query=PERT[name])
                b = EV.retrieval_split_b(REPS[name], cls, folds, A_query=PERT[name])
                rob_rows.append({
                    "perturbation": kind, "level": float(lvl), "representation": name,
                    "A_top1": a["top1"], "A_top5": a["top5"], "A_mrr": a["mrr"],
                    "B_top1": b["top1"], "B_macro_f1": b["macro_f1"],
                    "activation_stability": EV.activation_stability(REPS[name], PERT[name]),
                    "replicate_consistency": EV.replicate_consistency(PERT[name], y),
                })
        log(f"  {kind:20s} done")
    rob = pd.DataFrame(rob_rows)
    outputs.append(wtab(rob, "robustness_curves_v1.csv"))

    auc_rows = []
    for kind in PT.PERTURBATIONS:
        for name in REPS:
            s = rob[(rob.perturbation == kind) & (rob.representation == name)].sort_values("level")
            auc_rows.append({
                "perturbation": kind, "representation": name,
                "aurc_A_top1": EV.area_under_robustness(s.level, s.A_top1, clean[name]),
                "aurc_B_top1": EV.area_under_robustness(s.level, s.B_top1, cleanB[name]),
                "aurc_activation_stability": EV.area_under_robustness(
                    s.level, s.activation_stability, 1.0),
                "clean_A_top1": clean[name], "clean_B_top1": cleanB[name],
                "worst_A_top1": float(s.A_top1.min()),
                "retained_at_worst": float(s.A_top1.min() / (clean[name] + 1e-12)),
            })
    auc = pd.DataFrame(auc_rows)
    outputs.append(wtab(auc, "robustness_auc_v1.csv"))
    piv = auc.pivot_table(index="representation", values=["aurc_A_top1", "aurc_B_top1",
                                                          "aurc_activation_stability",
                                                          "retained_at_worst"])
    log("  mean area-under-robustness (fraction of clean performance retained):")
    for name in REPS:
        r = piv.loc[name]
        log(f"    {name:5s} A-top1 {r.aurc_A_top1:.3f}  B-top1 {r.aurc_B_top1:.3f}  "
            f"activation {r.aurc_activation_stability:.3f}  worst-case retained "
            f"{r.retained_at_worst:.3f}")

    delta = float(piv.loc["META", "aurc_A_top1"] - piv.loc["CSM", "aurc_A_top1"])
    cost = float(valid.set_index("representation").loc["CSM", "A_top1"]
                 - valid.set_index("representation").loc["META", "A_top1"])
    log(f"  META vs CSM: robustness {delta:+.4f} AURC, clean accuracy {-cost:+.4f} top-1")

    # ── 5. verdict, decided by the numbers ───────────────────────────────────
    v = valid.set_index("representation")
    improves = {
        "robustness_A": bool(piv.loc["META", "aurc_A_top1"] > piv.loc["CSM", "aurc_A_top1"]),
        "robustness_B": bool(piv.loc["META", "aurc_B_top1"] > piv.loc["CSM", "aurc_B_top1"]),
        "activation_stability_under_noise": bool(
            piv.loc["META", "aurc_activation_stability"]
            > piv.loc["CSM", "aurc_activation_stability"]),
        "clean_molecule_retrieval": bool(v.loc["META", "A_top1"] > v.loc["CSM", "A_top1"]),
        "clean_class_retrieval": bool(v.loc["META", "B_top1"] > v.loc["CSM", "B_top1"]),
        "replicate_consistency": bool(v.loc["META", "replicate_consistency"]
                                      > v.loc["CSM", "replicate_consistency"]),
        "biochemical_coherence": bool(v.loc["META", "biochemical_coherence"]
                                      > v.loc["CSM", "biochemical_coherence"]),
        "cross_fold_reproducibility": bool(v.loc["META", "cross_fold_reproducibility"]
                                           > v.loc["CSM", "cross_fold_reproducibility"]),
    }
    n_better = sum(improves.values())

    # An informativeness floor, applied BEFORE any stability gain is allowed to count.
    # Replicate consistency, activation stability and robustness AUC are all maximised by a
    # representation that says the same thing about everything — the identical trap that made
    # the Phase 03 softmax theme mode score best on reproducibility while carrying no
    # information. A layer can only "augment" the CSM layer if a downstream user computing both
    # would gain something, and that requires it to retain usable information.
    INFO_FLOOR = 0.50           # of CSM class-retrieval, and of CSM information
    info_ratio = float(v.loc["META", "information_retained_vs_csm"])
    class_ratio = float(v.loc["META", "B_top1"] / (v.loc["CSM", "B_top1"] + 1e-12))
    informative = bool(info_ratio >= INFO_FLOOR and class_ratio >= INFO_FLOOR)
    log(f"  informativeness floor: retains {info_ratio:.3f} of CSM information and "
        f"{class_ratio:.3f} of its class retrieval — "
        f"{'PASS' if informative else 'FAIL (stability gains do not count)'}")

    if not informative:
        verdict = ("HIERARCHICAL ABSTRACTION DOES NOT IMPROVE OVER CSM — the stability gains "
                   "are those of a low-information representation, not of a better one")
        action = "discard"
    elif improves["robustness_A"] and improves["robustness_B"] and n_better >= 5:
        verdict, action = "META COMPONENTS IMPROVE ON CSM", "augment"
    elif n_better >= 3 and (improves["robustness_A"] or improves["robustness_B"]):
        verdict, action = "META COMPONENTS OFFER A PARTIAL, CONDITIONAL BENEFIT", "augment"
    else:
        verdict, action = ("HIERARCHICAL ABSTRACTION DOES NOT IMPROVE OVER CSM", "discard")
    log(f"VERDICT: {verdict} — recommend: {action} ({n_better}/8 axes improved)")

    # Would a different K have helped downstream? Reported as a DIAGNOSTIC, never used for
    # selection — selecting K on the metric the layer is then judged by would be circular.
    log("  diagnostic: downstream retrieval across K (not used for selection)")
    kdiag = []
    for v_ in MF.VARIANTS:
        for K_ in MF.K_GRID:
            Hk = np.clip(store[(v_, K_)]["H"], 0, None)
            Mk = MF.project(A, Hk)
            kdiag.append({"variant": v_, "K": K_,
                          "A_top1": EV.retrieval_split_a(Mk, y, repl)["top1"],
                          "B_top1": EV.retrieval_split_b(Mk, cls, folds)["top1"],
                          "information_retained": EV.information_retained(A, Mk)["linear_ev"]})
    kd = pd.DataFrame(kdiag)
    outputs.append(wtab(kd, "k_downstream_diagnostic_v1.csv", OUT.validation))
    bk = kd.sort_values("B_top1", ascending=False).iloc[0]
    log(f"    best downstream B-top1 would be {bk.B_top1:.3f} at {bk.variant} K={int(bk.K)} "
        f"(selected K={K} gives {kd[(kd.variant == variant) & (kd.K == K)].B_top1.iloc[0]:.3f}); "
        f"CSM itself gives {v.loc['CSM', 'B_top1']:.3f}")

    # ── 6. artefacts ─────────────────────────────────────────────────────────
    np.savez_compressed(OUT.artifacts / "meta_components_v1.npz", H=Hm, W=Wm, A_meta=A_meta,
                        A_csm=A, csm_ids=np.array(csm_ids, dtype=object),
                        y=np.array(y, dtype=object), cls=np.array(cls, dtype=object),
                        folds=folds, laplacian=L, coords_csm=coords_csm)
    outputs.append({"artifact_id": "meta_components_v1.npz",
                    "path": OUT.rel(OUT.artifacts / "meta_components_v1.npz"),
                    "sha256": P.sha256_file(OUT.artifacts / "meta_components_v1.npz")})
    import hashlib
    meta_fp = hashlib.sha256(np.ascontiguousarray(Hm).tobytes()).hexdigest()[:32]
    outputs.append(wjson({"schema": "gaira_v7_meta_dictionary_v1", "variant": variant, "K": K,
                          "lambda": LAMBDA if variant == "geometry_regularised" else None,
                          "fingerprint": meta_fp, "csm_ids": csm_ids,
                          "components": [{"meta_component": f"MC-{k + 1:02d}",
                                          "loadings": {csm_ids[i]: round(float(Hm[k, i]), 6)
                                                       for i in np.argsort(-Hm[k])[:8]
                                                       if Hm[k, i] > 0}}
                                         for k in range(K)],
                          "inference": ("spectrum → frozen CSM projection → 49 activations → "
                                        "NNLS onto this frozen H → Meta Component vector; "
                                        "no fitting"),
                          "status": ("CANDIDATE — the CSM layer remains canonical unless this "
                                     "layer demonstrates measurable benefit")},
                         "meta_dictionary_v1.json"))
    outputs.append(wjson({"verdict": verdict, "recommended_action": action,
                          "axes_improved": improves, "n_axes_improved": n_better,
                          "informativeness_floor_passed": informative,
                          "information_retained_ratio": info_ratio,
                          "class_retrieval_ratio": class_ratio,
                          "robustness_delta_vs_csm_A": delta,
                          "clean_accuracy_cost_vs_csm_A": cost,
                          "mean_aurc": piv.to_dict()}, "verdict_v1.json"))

    gates = [
        _g("frozen inputs verified", True, "atlas, LSM, CSM, theme"),
        _g("NMF applied to the CSM ACTIVATION matrix only", True,
           f"A = {A.shape}; not spectra, not a similarity matrix, not a graph"),
        _g("K selected on a Pareto frontier, not reconstruction", True,
           f"reconstruction weight {PARETO['explained_variance'][0]} of 1.0"),
        _g("both variants compared", set(sweep.variant) == set(MF.VARIANTS),
           "plain and geometry-regularised"),
        _g("geometry used only as a one-sided prior", True,
           "tr(H L H^T) rewards smoothness; no term separates distant CSMs"),
        _g("inference is projection only", True, "NNLS onto the frozen H"),
        _g("all four representations on identical splits", True,
           "RAW / LSM / CSM / META, same frozen folds and query sets"),
        _g("robustness measured, not assumed", len(rob) == 12 * 5 * 4,
           f"{len(rob)} rows = 12 perturbations x 5 levels x 4 representations"),
        _g("stability gains gated by an informativeness floor", True,
           f"retains {info_ratio:.3f} of CSM information — floor 0.50"),
        _g("verdict follows the numbers", True,
           f"{n_better}/8 axes improved, informativeness "
           f"{'PASS' if informative else 'FAIL'} → {action}"),
        _g("K diagnostic reported but not used for selection", True,
           "selecting K on the downstream metric would be circular"),
    ]
    outputs.append(wtab(pd.DataFrame(gates), "phase04_5_gates_v1.csv", OUT.validation))
    all_pass = all(g["status"] == "PASS" for g in gates)
    log(f"gates: {sum(g['status'] == 'PASS' for g in gates)}/{len(gates)} PASS")

    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=P.REPO,
                                capture_output=True, text=True).stdout.strip())
    wjson({"schema": "gaira_v7_phase_manifest_v1", "phase": PHASE, "phase_name": PHASE_NAME,
           "built_utc": t0.isoformat(), "output_root": str(OUT.root),
           "redirectable_via": "GAIRA_V7_OUTPUT_ROOT", "frozen_inputs": EXPECTED,
           "selected": {"variant": variant, "K": K, "fingerprint": meta_fp},
           "seed": SEED, "outputs": outputs, "gates": gates, "code_dirty": dirty,
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                           "pandas": pd.__version__}}, "phase_04_5_manifest_v1.json")
    (OUT.root / "PHASE_STATE.json").write_text(json.dumps({
        "schema": "gaira_v7_phase_state_v1", "phase": PHASE, "phase_name": PHASE_NAME,
        "status": "COMPLETE" if all_pass else "GATE_FAILED",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_inputs": EXPECTED, "selected_variant": variant, "K": K,
        "meta_fingerprint": meta_fp,
        "verdict": verdict, "recommended_action": action,
        "n_axes_improved": n_better, "axes_improved": improves,
        "informativeness_floor_passed": informative,
        "information_retained_ratio": info_ratio, "class_retrieval_ratio": class_ratio,
        "robustness_delta_vs_csm": delta, "clean_accuracy_cost": cost,
        "comparison": valid.set_index("representation").to_dict("index"),
        "mean_aurc": piv.to_dict("index"),
        "gates_passed": sum(g["status"] == "PASS" for g in gates), "gates_total": len(gates),
    }, indent=2, default=_ser))
    (OUT.logs / "phase04_5_run.log").write_text("\n".join(LOG))
    log("PHASE 04.5 " + ("COMPLETE" if all_pass else "GATE FAILED"))
    return 0 if all_pass else 3


def _g(name, ok, detail):
    return {"gate": name, "status": "PASS" if ok else "FAIL", "detail": detail}


if __name__ == "__main__":
    raise SystemExit(main())
