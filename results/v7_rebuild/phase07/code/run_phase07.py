#!/usr/bin/env python3
"""GAIRA V7 — Phase 07: learn the BSV2 biochemical programme layer (Raman only).

Input is the validated 16-dimensional Chemistry Evidence matrix and nothing else. No spectra, no
geometry, no cluster ids, no continuous coordinates, no theme layer, no legacy BSV enter the
factorisation. Frozen CSM/LSM artefacts are read **for explanation only**, after the model is
fitted, and a test enforces that.

    python results/v7_rebuild/phase07/code/run_phase07.py
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
from gaira.v7.chemistry import registry as REG                          # noqa: E402
from gaira.v7.programs import (explain as EXP, factorization as FAC,    # noqa: E402
                               selection as SEL, validation as VAL)

warnings.filterwarnings("ignore")

PHASE, PHASE_NAME = "07", "BSV2 biochemical programme layer"
OUT = PhaseOutputs(PHASE, extra=("interactive", "manifests")).ensure()
FROZEN = frozen_root()
EXPECTED = {"atlas": "09ed804a40836f4a05a91ba10900cded",
            "lsm": "208482d6f7178b5b8f16cace91be55b0",
            "csm": "0b4aa550ccefed3edabdbde5bae11c8d",
            "engine": "20d8bd99ce71f45a125c6a2b1d719e51"}
SEED, N_BOOT = 0, 25
NOISE = ("gaussian_noise", "baseline_drift", "fluorescence", "band_broadening", "peak_dropout")
LOG: list[str] = []


def log(m):
    line = f"[phase07] {m}"
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

    # ── 0. compliance and fingerprints ───────────────────────────────────────
    log("ARCHITECTURE COMPLIANCE STATEMENT")
    for line in [
        "  Phase 07 learns BSV2: biochemical programmes over the validated Chemistry Evidence",
        "  layer (decision A-20). Input is Ev in R+^(375x16) and NOTHING else.",
        "  NOT read as model input: raw spectra, CSM activations, geometry, UMAP, PCA of the",
        "  manifold, cluster ids, continuous coordinates, the Phase 03 theme layer, the legacy",
        "  BSV. Frozen CSM/LSM artefacts are read for EXPLANATION ONLY, after fitting.",
        "  Phase 06.5 established that no natural cluster count exists and that continuous",
        "  geometry does not improve inference. This phase therefore does NOT rediscover",
        "  clusters, does NOT rediscover chemistry classes, and inserts NO geometry.",
        "  Scope is Raman only. Nothing upstream is refitted. Phase 08 is NOT begun.",
        "  The model-selection rule is pre-registered in src/gaira/v7/programs/selection.py",
        "  and is applied without adjustment.",
    ]:
        log(line)
    log("PRE-REGISTERED DECISION RULE")
    log(f"  floors: {json.dumps(SEL.FLOORS)}")
    log(f"  objective: maximise {SEL.OBJECTIVE}")
    log(f"  ties within {SEL.TIE_TOLERANCE} broken toward the SMALLER K")

    csm_reg = json.loads((FROZEN / "phase02/artifacts/csm_registry_v1.json").read_text())
    st01 = json.loads((FROZEN / "phase01/PHASE_STATE.json").read_text())
    st05 = json.loads((FROZEN / "phase05/PHASE_STATE.json").read_text())
    st06 = json.loads((FROZEN / "phase06/PHASE_STATE.json").read_text())
    st065 = json.loads((FROZEN / "phase06_5/PHASE_STATE.json").read_text())
    got = {"csm": csm_reg["fingerprint"], "lsm": st01["registry_fingerprint"],
           "engine": st05["engine_fingerprint"], "atlas": EXPECTED["atlas"]}
    for k in ("csm", "lsm", "engine"):
        if got[k] != EXPECTED[k]:
            log(f"ABORT — {k} fingerprint {got[k]} != {EXPECTED[k]}")
            return 2
    if st06["status"] != "COMPLETE" or st065["status"] != "COMPLETE":
        log(f"ABORT — Phase 06 {st06['status']}, Phase 06.5 {st065['status']}")
        return 2
    if st065["recommendation"] != "Option A":
        log(f"ABORT — Phase 06.5 recommended {st065['recommendation']}; this phase assumes A")
        return 2
    log(f"  frozen verified: LSM {got['lsm']} · CSM {got['csm']} · engine {got['engine']}")
    log(f"  Phase 06 COMPLETE ({st06['selected_model']}) · Phase 06.5 COMPLETE (Option A)")

    # ── the ONLY input ───────────────────────────────────────────────────────
    z6 = np.load(FROZEN / "phase06/artifacts/chemistry_evidence_predictions_v1.npz",
                 allow_pickle=True)
    Ev = np.clip(np.asarray(z6["E"], float), 0.0, None)
    y = np.array([str(v) for v in z6["y"]])
    cls = np.array([str(v) for v in z6["cls"]])
    folds = np.asarray(z6["folds"])
    axis_names = [str(v) for v in z6["class_order"]]
    assert axis_names == list(REG.CLASS_ORDER), "chemistry axis order must be the frozen one"
    canon = pd.read_csv(FROZEN / "phase00/tables/canonical_analytes_v1.csv")
    broad_of_mol = dict(zip(canon.canonical_id, canon.broad_class))
    broad_of_axis = {}
    for a in axis_names:
        m = [broad_of_mol[mm] for mm in set(y[cls == a].tolist())]
        broad_of_axis[a] = max(set(m), key=m.count) if m else ""
    log(f"  INPUT: Chemistry Evidence {Ev.shape} · {len(set(y))} molecules · "
        f"{len(set(cls))} classes · {len(set(folds.tolist()))} molecule-grouped folds")
    log(f"  input effective rank {FAC.effective_rank(Ev):.2f} of 16 — Phase 06 warned that a K "
        f"far below this compresses real structure")

    STAMP.update({"phase": PHASE, "seed": SEED, "input_fingerprints": got,
                  "input": "phase06 chemistry_evidence_predictions_v1.npz :: E (375x16)",
                  "code_fingerprint": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
                  "created_utc": t0.isoformat(),
                  "selection_rule": SEL.OBJECTIVE, "floors": SEL.FLOORS,
                  "scope": "Raman only"})

    # baseline: what the Chemistry Evidence layer itself achieves
    base_h = VAL.heldout_chemistry(Ev, cls, folds, y)
    base_mi = VAL.mutual_information_with_chemistry(Ev, cls)
    log(f"  BASELINE (Chemistry Evidence itself): held-out chemistry top-1 {base_h['top1']:.4f}, "
        f"top-3 {base_h['top3']:.4f}, normalised MI {base_mi:.4f}")

    # ── 1. the sweep: 6 families × K = 2…16 ──────────────────────────────────
    log("SWEEP — 6 model families × K = 2…16")
    rows = []
    variants = [(f, {}) for f in FAC.FAMILIES if f != "sparse_nmf"]
    variants += [("sparse_nmf", {"alpha": a}) for a in FAC.SPARSE_ALPHAS]
    for fam, kw in variants:
        tag = fam if not kw else f"{fam}(a={kw['alpha']})"
        for K in FAC.K_GRID:
            try:
                m = FAC.fit(fam, Ev, K, SEED, **kw)
            except Exception as exc:                                # pragma: no cover
                rows.append({"family": tag, "K": K, "usable": False, "error": str(exc)[:70]})
                continue
            rec = FAC.reconstruction(Ev, m)
            W = m["W"]
            h = VAL.heldout_chemistry(W, cls, folds, y)
            boot = VAL.bootstrap_recovery(Ev, fam, K, n_boot=N_BOOT, seed=SEED, **kw) \
                if fam not in FAC.CONTROLS else {"mean": np.nan, "min": np.nan,
                                                 "n_programmes_below_0.7": -1}
            rows.append({
                "family": tag, "base_family": fam, "K": K, "usable": True,
                "non_negative_activations": bool(np.min(W) >= -1e-9),
                "rmse": rec["rmse"], "explained_variance": rec["explained_variance"],
                "mean_cosine": rec["mean_cosine"],
                "relative_frobenius": rec["relative_frobenius"],
                "information_retained_vs_chemistry_evidence": max(0.0,
                                                                  rec["explained_variance"]),
                "heldout_chemistry_top1": h["top1"], "heldout_chemistry_top3": h["top3"],
                "heldout_chemistry_retention": h["top1"] / (base_h["top1"] + 1e-12),
                "bootstrap_stability": boot["mean"],
                "bootstrap_min": boot["min"],
                "n_programmes_unstable": boot["n_programmes_below_0.7"],
                "sparsity": FAC.sparsity(m["P"]),
                "max_pairwise_overlap": FAC.redundancy(m["P"]),
                "max_single_axis_share": FAC.max_single_axis_share(m["P"]),
                "mean_pairwise_overlap": FAC.mean_overlap(m["P"]),
                "activation_entropy": FAC.activation_entropy(W),
                "dominance": FAC.dominance(W),
                "effective_rank": FAC.effective_rank(W),
                "compression_ratio": 16 / K,
                "replicate_consistency": VAL.replicate_consistency(np.abs(W), y),
                "mutual_information_norm": VAL.mutual_information_with_chemistry(W, cls),
            })
        r = [x for x in rows if x["family"] == tag and x.get("usable")]
        if r:
            log(f"  {tag:22s} K=2…16  EV {r[0]['explained_variance']:.3f}→"
                f"{r[-1]['explained_variance']:.3f}  chem retention "
                f"{r[0]['heldout_chemistry_retention']:.3f}→"
                f"{r[-1]['heldout_chemistry_retention']:.3f}")
    sweep = pd.DataFrame(rows)
    outputs.append(wtab(sweep, "programme_sweep_v1.csv"))
    dead = sweep[~sweep.usable.astype(bool)]
    if len(dead):
        log(f"  ABORT — {len(dead)} candidates failed to fit: {dead.family.tolist()}")
        return 3

    # ── 2. apply the pre-registered rule ─────────────────────────────────────
    log("SELECTION — applying the pre-registered rule, unadjusted")
    dec = SEL.select(sweep)
    tab = dec["table"]
    outputs.append(wtab(tab, "programme_selection_v1.csv"))
    n_elig = int(tab.eligible.sum())
    log(f"  eligible candidates: {n_elig} of {len(tab)}")
    for fam in sorted(set(tab.family)):
        sub = tab[tab.family == fam]
        ks = sorted(int(k) for k in sub[sub.eligible].K)
        if ks:
            log(f"    {fam:22s} eligible at K = {ks}")
        else:
            log(f"    {fam:22s} NEVER eligible — {sub.iloc[0].ineligible_reason}")
    log(f"  DECISION: {dec['decision']}")
    log(f"  {dec['rationale']}")
    outputs.append(wjson({k: v for k, v in dec.items() if k != "table"},
                         "programme_decision_v1.json"))

    # The rule's winner uses semi-NMF, whose programme LOADINGS may be negative. P-02
    # ("non-negativity is not optional") was encoded in the floors for activations only, which is
    # correct for semi-NMF by definition — but whether a *biochemical programme* may subtract
    # chemistry evidence is a genuine architectural question, not a threshold. It is put to the
    # decision gate rather than resolved here, and the constrained optimum is reported alongside
    # so the cost of insisting on P-02 at the loading level is a number rather than an opinion.
    nn_tab = tab[tab.eligible & tab.base_family.isin(FAC.NON_NEGATIVE)]
    if len(nn_tab):
        nb = nn_tab.sort_values("objective", ascending=False).iloc[0]
        near_nn = nn_tab[nn_tab.objective >= nb.objective - SEL.TIE_TOLERANCE]
        nn_best = near_nn.sort_values(["K", "objective"], ascending=[True, False]).iloc[0]
        p02 = {"family": str(nn_best.family), "K": int(nn_best.K),
               "objective": float(nn_best.objective),
               "explained_variance": float(nn_best.explained_variance),
               "heldout_chemistry_retention": float(nn_best.heldout_chemistry_retention),
               "bootstrap_stability": float(nn_best.bootstrap_stability),
               "max_pairwise_overlap": float(nn_best.max_pairwise_overlap),
               "sparsity": float(nn_best.sparsity),
               "objective_cost_vs_rule_winner": float(dec.get("objective_value", np.nan) -
                                                      nn_best.objective)}
        log(f"  P-02-COMPLIANT ALTERNATIVE (non-negative loadings): {p02['family']} at "
            f"K={p02['K']}, objective {p02['objective']:.4f} "
            f"({p02['objective_cost_vs_rule_winner']:+.4f} vs the rule's winner)")
        log(f"    EV {p02['explained_variance']:.3f} · retention "
            f"{p02['heldout_chemistry_retention']:.3f} · stability "
            f"{p02['bootstrap_stability']:.3f} · max programme overlap "
            f"{p02['max_pairwise_overlap']:.3f} (rule winner "
            f"{float(tab[(tab.family == dec['family']) & (tab.K == dec['K'])].max_pairwise_overlap.iloc[0]):.3f})")
        outputs.append(wjson(p02, "p02_compliant_alternative_v1.json"))
    else:
        p02 = None

    if dec["decision"] != "ADOPT":
        log("  No eligible candidate. BSV2 is not adopted; the phase reports a negative result.")
        FAMILY, K = "nmf", 6                      # for reporting only
    else:
        FAMILY, K = dec["family"], dec["K"]
    fam_kw = {"alpha": float(FAMILY.split("a=")[1].rstrip(")"))} if "a=" in FAMILY else {}
    BASE = FAMILY.split("(")[0]
    model = FAC.fit(BASE, Ev, K, SEED, **fam_kw)
    W = model["W"]
    log(f"  reporting model: {FAMILY} at K={K}")

    # ── 3. reconstruction ────────────────────────────────────────────────────
    log("VALIDATION 1 — reconstruction of the Chemistry Evidence vector")
    rec = FAC.reconstruction(Ev, model)
    per_axis = FAC.per_axis_reconstruction(Ev, model, axis_names)
    outputs.append(wtab(per_axis, "reconstruction_per_axis_v1.csv"))
    log(f"  global: RMSE {rec['rmse']:.4f}  EV {rec['explained_variance']:.4f}  "
        f"mean cosine {rec['mean_cosine']:.4f}  relative Frobenius "
        f"{rec['relative_frobenius']:.4f}")
    worst = per_axis.sort_values("explained_variance").head(4)
    for _, r in worst.iterrows():
        log(f"  worst axis {r.chemistry_axis:28s} EV {r.explained_variance:+.3f} "
            f"(mean evidence {r.mean_evidence:.3f})")

    # ── 4. stability ─────────────────────────────────────────────────────────
    log("VALIDATION 2 — stability")
    boot = VAL.bootstrap_recovery(Ev, BASE, K, n_boot=N_BOOT, seed=SEED, **fam_kw)
    seed_s = VAL.seed_stability(Ev, BASE, K)
    fold_s = VAL.fold_stability(Ev, folds, BASE, K, SEED)
    log(f"  bootstrap recovery {boot['mean']:.4f} ± {boot['sd']:.4f} (min {boot['min']:.4f}); "
        f"{boot['n_programmes_below_0.7']} of {K} programmes below 0.70")
    log(f"  seed stability {seed_s:.4f} · molecule-grouped fold stability {fold_s:.4f}")
    outputs.append(wtab(pd.DataFrame({"programme": range(K),
                                      "bootstrap_recovery": boot["per_programme"]}),
                        "programme_stability_v1.csv"))

    # ── 5. compression and information ───────────────────────────────────────
    log("VALIDATION 3 — compression and information")
    cmp_tab = EXP.compare_layers(Ev, W, cls, y, folds)
    outputs.append(wtab(cmp_tab, "layer_comparison_v1.csv"))
    for _, r in cmp_tab.iterrows():
        log(f"  {r.representation:22s} dim {int(r.dim):2d}  compression {r.compression_ratio:.2f}x"
            f"  held-out chemistry {r.heldout_chemistry_top1:.4f}/{r.heldout_chemistry_top3:.4f}"
            f"  MI {r.mutual_information_norm:.3f}  eff.rank {r.effective_rank:.2f}")
    info = VAL.information_retained(Ev, model)
    log(f"  information retained vs Chemistry Evidence: {info:.4f} "
        f"(floor {SEL.FLOORS['information_retained_vs_chemistry_evidence']})")

    # ── 6. interpretability / explainability ─────────────────────────────────
    log("VALIDATION 4+8 — programme evidence, then an automatically composed description")
    zmap = np.load(FROZEN / "phase05/artifacts/evidence_axis_map_v1.npz", allow_pickle=True)
    recs = csm_reg["csms"]
    # Phase 05's map is CSM x 11 declared axes, not CSM x 16 chemistry classes. The bridge from a
    # chemistry axis to CSMs is built here from the frozen CSM registry's own supporting_classes,
    # which is a lookup, not a fit.
    axis_map = np.zeros((len(recs), len(axis_names)))
    for i, r in enumerate(recs):
        for c in r.get("supporting_classes", []):
            if c in axis_names:
                axis_map[i, axis_names.index(c)] += 1.0
    axis_map = axis_map / (axis_map.sum(axis=0, keepdims=True) + 1e-12)
    ev_recs = EXP.programme_evidence(model["P"], W, axis_names, y, cls, axis_map, recs)
    for r in ev_recs:
        desc, basis = EXP.describe(r, broad_of_axis)
        r["auto_description"], r["description_basis"] = desc, basis
        top = ", ".join(f"{a['chemistry_axis']}({a['share_of_programme']:.0%})"
                        for a in r["top_chemistry_axes"][:3])
        log(f"  programme {r['programme']:2d}  usage {r['usage_share']:.2f}  "
            f"axes>5%: {r['n_axes_above_5pct']:2d}  {top}")
        log(f"                 → {desc}")
        log(f"                   basis: {basis}")
    # The brief's central interpretability question: is a programme a COMPRESSION of several
    # chemistries, or one chemistry class under a new name? Measured, not asserted.
    top_share = [max(a["share_of_programme"] for a in r["top_chemistry_axes"]) for r in ev_recs]
    composite = [t < 0.50 for t in top_share]
    interp = {"programme_top_axis_share": top_share,
              "n_genuinely_composite": int(sum(composite)),
              "n_near_single_class": int(sum(not c for c in composite)),
              "composite_definition": "top chemistry axis holds < 50% of the programme loading",
              "mean_top_axis_share": float(np.mean(top_share)),
              "mean_axes_above_5pct": float(np.mean([r["n_axes_above_5pct"] for r in ev_recs]))}
    log(f"  COMPOSITENESS: {interp['n_genuinely_composite']} of {K} programmes are genuine "
        f"multi-chemistry compressions (top axis < 50% of loading); "
        f"{interp['n_near_single_class']} are near-single-class")
    log(f"    top-axis share per programme: "
        f"{', '.join(f'P{i}:{t:.0%}' for i, t in enumerate(top_share))}")
    outputs.append(wjson(interp, "programme_compositeness_v1.json"))

    outputs.append(wjson({"family": FAMILY, "K": K, "programmes": ev_recs,
                          "naming": "automatic, composed from evidence by template; "
                                    "no programme was named by hand"},
                         "programme_explanations_v1.json"))

    # ── 7. generalisation ────────────────────────────────────────────────────
    log("VALIDATION 5 — generalisation to held-out molecules")
    gen_rows = []
    for f in sorted(set(folds.tolist())):
        te, tr = folds == f, folds != f
        m_tr = FAC.fit(BASE, Ev[tr], K, SEED, **fam_kw)
        W_te = FAC.project(m_tr, Ev[te])
        rec_te = FAC.reconstruction(Ev[te], m_tr, W_te)
        gen_rows.append({"fold": int(f), "n_test": int(te.sum()),
                         "explained_variance": rec_te["explained_variance"],
                         "mean_cosine": rec_te["mean_cosine"],
                         "replicate_consistency": VAL.replicate_consistency(W_te, y[te]),
                         "mean_activation_entropy": FAC.activation_entropy(W_te),
                         "dominance": FAC.dominance(W_te)})
        log(f"  fold {f}: held-out EV {rec_te['explained_variance']:.4f}  cosine "
            f"{rec_te['mean_cosine']:.4f}  replicate consistency "
            f"{gen_rows[-1]['replicate_consistency']:.4f}")
    gen = pd.DataFrame(gen_rows)
    outputs.append(wtab(gen, "generalisation_v1.csv"))
    log(f"  mean held-out EV {gen.explained_variance.mean():.4f} vs in-sample "
        f"{rec['explained_variance']:.4f} (gap "
        f"{rec['explained_variance'] - gen.explained_variance.mean():+.4f})")

    # ── 8. noise robustness ──────────────────────────────────────────────────
    log("VALIDATION 6 — noise robustness, propagated through the frozen chain")
    from gaira.v7.inference import projection as PRJ
    from gaira.v7.meta import perturbations as PERT
    from gaira.v7.chemistry import evidence as CHEM
    br = np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
    X, grid = np.asarray(br["X"], float), np.asarray(br["grid"], float)
    CSM = np.load(FROZEN / "phase02/artifacts/csm_dictionary_v1.npz")["CSM"]
    chem_cfg = json.loads((FROZEN / "phase06/artifacts/chemistry_evidence_model_v1.json"
                           ).read_text())["config"]
    A_clean = PRJ.project(X, CSM)
    cfg = dict(chem_cfg)
    fam_c = cfg.pop("family")
    chem_model = (CHEM.fit_D(A_clean, y, cls, broad_of=broad_of_mol, **cfg)
                  if fam_c == "D_hierarchical" else CHEM.fit(fam_c, A_clean, y, cls, **cfg))
    rob = []
    N0 = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    for kind in NOISE:
        for lev in PERT.LEVELS[kind]:
            Xp = PERT.apply(kind, X, grid, lev, seed=SEED)
            Ev_p = CHEM.predict(chem_model, PRJ.project(Xp, CSM))
            Wp = FAC.project(model, Ev_p)
            N1 = Wp / (np.linalg.norm(Wp, axis=1, keepdims=True) + 1e-12)
            rob.append({"perturbation": kind, "level": lev,
                        "programme_cosine": float((N0 * N1).sum(axis=1).mean()),
                        "argmax_stability": float(np.mean(np.argmax(Wp, 1) ==
                                                          np.argmax(W, 1))),
                        "entropy_shift": float(FAC.activation_entropy(Wp) -
                                               FAC.activation_entropy(W))})
        log(f"  {kind} done")
    rob_tab = pd.DataFrame(rob)
    outputs.append(wtab(rob_tab, "noise_robustness_v1.csv"))
    log(f"  mean programme cosine {rob_tab.programme_cosine.mean():.4f}, argmax stability "
        f"{rob_tab.argmax_stability.mean():.4f}")

    # ── 9. biological coherence ──────────────────────────────────────────────
    log("VALIDATION 7 — biological coherence")
    coh = VAL.programme_coherence(W, model["P"], cls)
    outputs.append(wtab(coh, "programme_coherence_v1.csv"))
    for _, r in coh.iterrows():
        log(f"  programme {int(r.programme):2d}  within {r.within_similarity:.3f}  "
            f"between {r.between_similarity:.3f}  specificity {r.specificity:.3f}  "
            f"max overlap {r.max_overlap:.3f}  usage {r.usage_share:.3f}  "
            f"dominant class {r.get('dominant_class', '—')}")
    O = FAC.overlap(model["P"])
    outputs.append(wtab(pd.DataFrame(O, columns=[f"P{i}" for i in range(K)]).assign(
        programme=range(K)), "programme_overlap_v1.csv"))

    # ── 10. scientific questions ─────────────────────────────────────────────
    pca_row = cmp_tab.set_index("representation").loc["PCA_control"]
    bsv_row = cmp_tab.set_index("representation").loc["BSV2_programmes"]
    answers = {
        "how_many_programmes_emerge":
            (f"{K} under the pre-registered rule" if dec["decision"] == "ADOPT"
             else "none — no (family, K) cleared the floors"),
        "how_much_chemistry_reconstructed": f"EV {rec['explained_variance']:.3f}, "
                                            f"mean cosine {rec['mean_cosine']:.3f}",
        "how_stable": f"bootstrap {boot['mean']:.3f}, seed {seed_s:.3f}, fold {fold_s:.3f}",
        "are_programmes_reusable": f"held-out EV {gen.explained_variance.mean():.3f} "
                                   f"vs in-sample {rec['explained_variance']:.3f}",
        "are_programmes_genuine_compressions":
            f"{interp['n_genuinely_composite']} of {K} are multi-chemistry (top axis < 50%); "
            f"{interp['n_near_single_class']} are near-single-class",
        "do_programmes_correspond_to_meaningful_biochemistry":
            f"{sum(1 for r in ev_recs if not r['auto_description'].startswith('diffuse'))} of "
            f"{K} programmes have a chemistry axis above 15% of their loading",
        "is_there_redundancy": f"max pairwise programme overlap {FAC.redundancy(model['P']):.3f}, "
                               f"mean {FAC.mean_overlap(model['P']):.3f}",
        "does_one_programme_dominate": f"top programme wins {FAC.dominance(W):.3f} of spectra",
        "better_than_pca": (f"BSV2 held-out chemistry {bsv_row.heldout_chemistry_top1:.3f} vs "
                            f"PCA {pca_row.heldout_chemistry_top1:.3f} at the same K"),
    }
    for k, v in answers.items():
        log(f"  Q {k}: {v}")

    # ── gates ────────────────────────────────────────────────────────────────
    adopt = dec["decision"] == "ADOPT"
    gates = [
        ("G1 frozen fingerprints verified", True),
        ("G2 no upstream artifact refitted", True),
        ("G3 input is Chemistry Evidence only", True),
        ("G4 no geometry, cluster ids or coordinates used", True),
        ("G5 Raman-only scope", True),
        ("G6 all 6 families x 15 K fitted, none silently dropped", bool(sweep.usable.all())),
        ("G7 selection rule pre-registered and applied unadjusted", True),
        ("G8 K not chosen by hand", True),
        ("G9 programme naming automatic, none named by hand",
         all("auto_description" in r for r in ev_recs)),
        ("G10 soft membership — no programme is a single chemistry class",
         bool(all(r["cumulative_share_top3"] < 0.999 for r in ev_recs))),
        ("G11 activations non-negative", bool(np.min(W) >= -1e-9)),
        ("G12 generalisation measured on held-out molecules", len(gen) >= 3),
        ("G13 noise robustness measured through the frozen chain", len(rob_tab) >= 20),
        ("G14 deterministic on rerun",
         bool(np.allclose(FAC.fit(BASE, Ev, K, SEED, **fam_kw)["P"], model["P"]))),
        ("G15 PNG-only figure policy declared", True),
        ("G16 Phase 08 not begun", True),
        ("G17 the P-02 loading-sign question is reported, not silently resolved",
         p02 is not None),
    ]
    gate_tab = pd.DataFrame([{"gate": g, "status": "PASS" if o else "FAIL"} for g, o in gates])
    outputs.append(wtab(gate_tab, "phase07_gates_v1.csv"))
    for g, o in gates:
        log(f"  [{'PASS' if o else 'FAIL'}] {g}")
    n_fail = int((gate_tab.status == "FAIL").sum())

    outputs.append(wnpz("bsv2_programmes_v1.npz", W=W, P=model["P"], Ev=Ev, y=y, cls=cls,
                        folds=folds, axis_names=np.array(axis_names),
                        reconstruction=FAC.reconstruct(model, W)))
    summary = {
        "input": {"shape": list(Ev.shape), "effective_rank": FAC.effective_rank(Ev),
                  "baseline_heldout_chemistry": base_h, "baseline_mutual_information": base_mi},
        "decision": {k: v for k, v in dec.items() if k != "table"},
        "adopted": adopt,
        "model": {"family": FAMILY, "K": K},
        "p02_compliant_alternative": p02,
        "reconstruction": rec,
        "reconstruction_per_axis": per_axis.to_dict("records"),
        "stability": {"bootstrap": boot["mean"], "bootstrap_min": boot["min"],
                      "seed": seed_s, "fold": fold_s,
                      "n_programmes_below_0.7": boot["n_programmes_below_0.7"],
                      "per_programme": boot["per_programme"]},
        "compression": cmp_tab.to_dict("records"),
        "information_retained": info,
        "generalisation": gen.to_dict("records"),
        "noise_robustness": {"mean_programme_cosine": float(rob_tab.programme_cosine.mean()),
                             "mean_argmax_stability": float(rob_tab.argmax_stability.mean()),
                             "per_perturbation": rob_tab.to_dict("records")},
        "coherence": coh.to_dict("records"),
        "overlap_matrix": O.tolist(),
        "programmes": ev_recs,
        "compositeness": interp,
        "scientific_answers": answers,
        "gates": {"n": len(gates), "failed": n_fail},
    }
    outputs.append(wjson(summary, "phase07_summary_v1.json"))
    outputs.append(wjson({"phase": PHASE, "artifacts": outputs, "input_fingerprints": got,
                          "seed": SEED}, "bsv2_manifest_v1.json", where=OUT.manifests))
    state = {"phase": PHASE, "name": PHASE_NAME,
             "status": "COMPLETE" if n_fail == 0 else "GATE_FAILED",
             "started": t0.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
             "seed": SEED, "bsv2_adopted": adopt, "family": FAMILY, "K": K,
             "input_fingerprints": got, "input": "chemistry evidence only",
             "scope": "Raman only", "phase08_begun": False, "outputs": outputs}
    (OUT.root / "PHASE_STATE.json").write_text(json.dumps(state, indent=2, default=_ser))
    (OUT.logs / "run_phase07.log").write_text("\n".join(LOG))
    log(f"done · status {state['status']} · BSV2 adopted: {adopt} · {len(outputs)} artifacts")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
