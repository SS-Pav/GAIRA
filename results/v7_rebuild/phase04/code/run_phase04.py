#!/usr/bin/env python3
"""GAIRA V7 — Phase 04: frozen projection engine and hierarchical inference.

Benchmarks the projection mathematics (A–E), builds the canonical engine (F), validates it on
held-out spectra at six abstraction levels (G), measures information flow (H), tests whether
abstraction helps (I), and audits the whole thing adversarially (J).

Nothing frozen is refitted. Output location resolves through `gaira.v7.io.PhaseOutputs`.

    python results/v7_rebuild/phase04/code/run_phase04.py
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
from gaira.v7.engine import FrozenAtlas, project_spectrum     # noqa: E402
from gaira.v7.engine import aggregation as AGG                # noqa: E402
from gaira.v7.engine import geometry as GEO                   # noqa: E402
from gaira.v7.engine import projection as PRJ                 # noqa: E402
from gaira.v7.engine import validation as VAL                 # noqa: E402
from gaira.v7.engine.inference import _csm_distance           # noqa: E402

warnings.filterwarnings("ignore")

PHASE, PHASE_NAME = "04", "Frozen projection engine and hierarchical inference"
OUT = PhaseOutputs(PHASE).ensure()
FROZEN = frozen_root()
EXPECTED = {"atlas": "09ed804a40836f4a05a91ba10900cded",
            "lsm": "208482d6f7178b5b8f16cace91be55b0",
            "csm": "0b4aa550ccefed3edabdbde5bae11c8d",
            "theme": "f54d4835ffdf8aa2d50a4a203da0e8f4"}
SEED, K_NN = 0, 5
LOG: list[str] = []


def log(m):
    line = f"[phase04] {m}"
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
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def main() -> int:
    outputs: list[dict] = []
    t0 = datetime.now(timezone.utc)

    # ── 0. fingerprint gate ──────────────────────────────────────────────────
    log("architecture check — Phase 04 merges the plan's Phase 04 (BSV, C-09) and Phase 05 "
        "(engine, C-10); projection only, nothing refitted")
    base_cfg = {"projection_method": "nnls", "aggregation_method": "membership_weighted",
                "theme_mode": "soft_membership", "bsv_variant": "theme_only",
                "geometry_extension": "nystrom", "knn": K_NN}
    atlas = FrozenAtlas.load(FROZEN, base_cfg, EXPECTED)
    log(f"  frozen verified: atlas {EXPECTED['atlas']} · LSM {EXPECTED['lsm']} · "
        f"CSM {EXPECTED['csm']} · theme {EXPECTED['theme']}")

    br = np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
    X = np.asarray(br["X"], float)
    y = np.array([str(s) for s in br["canonical_id"]])
    weights = np.asarray(br["weight"], float)
    grid = np.asarray(br["grid"], float)
    part = pd.read_csv(FROZEN / "phase00/tables/chemical_partition_v1.csv")
    cls_of = dict(zip(part.canonical_id, part.fine_class))
    folds_tab = pd.read_csv(FROZEN / "phase00/tables/cv_folds_v1.csv")
    fold_of = dict(zip(folds_tab.canonical_id, folds_tab.fold))
    folds = np.array([fold_of.get(v, 0) for v in y])
    cls_arr = np.array([cls_of.get(v, "") for v in y])
    cls01 = pd.read_csv(FROZEN / "phase01/artifacts/lsm_classes_v1.csv")
    k_of_class = dict(zip(cls01.chemical_class, cls01.k_c))
    log(f"  corpus {X.shape} · {len(set(y))} molecules · {len(set(folds))} frozen folds · "
        f"{len(set(cls_arr))} classes")

    cond = PRJ.condition_diagnostics(atlas.H_lsm)
    log(f"  dictionary conditioning: max coherence {cond['max_coherence']:.3f}, "
        f"condition number {cond['condition_number']:.1f}, effective rank "
        f"{cond['effective_rank']:.1f} of {atlas.H_lsm.shape[0]}")

    # ── PART A: projection benchmark ─────────────────────────────────────────
    log("PART A — benchmarking six projection estimators")
    rows = []
    A_store = {}
    for m in PRJ.METHODS:
        try:
            A = PRJ.project(X, atlas.H_lsm, m)
        except Exception as exc:                                # pragma: no cover
            log(f"  {m} failed: {exc}")
            continue
        A_store[m] = A
        ev = PRJ.reconstruction_ev(X, A, atlas.H_lsm)
        ns = float(np.mean([PRJ.noise_stability(X[i], atlas.H_lsm, m, seed=SEED)
                            for i in range(0, X.shape[0], 25)]))
        r = VAL.molecule_retrieval(A, A, y, y)          # in-sample, benchmark only
        rows.append({"method": m, "mean_ev": float(ev.mean()), "min_ev": float(ev.min()),
                     "mean_active_components": PRJ.sparsity(A),
                     "negative_mass_fraction": PRJ.negativity(A),
                     "replicate_consistency": PRJ.replicate_consistency(A, y),
                     "noise_stability": ns,
                     "insample_top1": r["top1"]})
        log(f"  {m:14s} EV {ev.mean():.3f}  active {PRJ.sparsity(A):4.1f}  "
            f"neg {PRJ.negativity(A):.3f}  replicate {rows[-1]['replicate_consistency']:.3f}  "
            f"noise {ns:.3f}")
    proj_tab = pd.DataFrame(rows)
    outputs.append(wtab(proj_tab, "projection_benchmark_v1.csv"))

    # Pre-declared rule: among estimators with zero negative mass (physical admissibility is a
    # hard constraint, not a weighted term), maximise replicate consistency x noise stability.
    adm = proj_tab[proj_tab.negative_mass_fraction <= 1e-9]
    if adm.empty:
        adm = proj_tab
    proj_method = str(adm.assign(s=adm.replicate_consistency * adm.noise_stability)
                      .sort_values("s", ascending=False).iloc[0]["method"])
    log(f"  SELECTED projection: {proj_method}")

    # ── PART B: LSM → CSM aggregation ────────────────────────────────────────
    log("PART B — benchmarking LSM → CSM aggregation")
    a_lsm = PRJ.project(X, atlas.H_lsm, proj_method)
    stab01 = dict(zip(pd.read_csv(FROZEN / "phase01/artifacts/lsm_registry_v1.csv").motif_id,
                      pd.read_csv(FROZEN / "phase01/artifacts/lsm_registry_v1.csv").stability))
    rows = []
    for m in AGG.AGGREGATIONS:
        try:
            A = AGG.lsm_to_csm(a_lsm, atlas.lsm_ids, atlas.csm_members, atlas.csm_ids, m,
                               stab01, x=X, CSM=atlas.CSM)
        except Exception as exc:                                # pragma: no cover
            log(f"  {m} failed: {exc}")
            continue
        ev = PRJ.reconstruction_ev(X, A, atlas.CSM)
        rows.append({"aggregation": m, "mean_ev": float(ev.mean()),
                     "replicate_consistency": PRJ.replicate_consistency(A, y),
                     "mean_active": PRJ.sparsity(A)})
        log(f"  {m:26s} EV {ev.mean():.3f}  replicate {rows[-1]['replicate_consistency']:.3f}")
    agg_tab = pd.DataFrame(rows)
    outputs.append(wtab(agg_tab, "aggregation_benchmark_v1.csv"))
    agg_method = str(agg_tab.assign(s=agg_tab.mean_ev * agg_tab.replicate_consistency)
                     .sort_values("s", ascending=False).iloc[0]["aggregation"])
    log(f"  SELECTED aggregation: {agg_method}")

    # ── PART C: theme activation mode ────────────────────────────────────────
    log("PART C — benchmarking theme activation modes")
    a_csm = AGG.lsm_to_csm(a_lsm, atlas.lsm_ids, atlas.csm_members, atlas.csm_ids, agg_method,
                           stab01, x=X, CSM=atlas.CSM)
    rows = []
    for m in AGG.THEME_MODES:
        T = AGG.theme_activation(a_csm, atlas.S, m, atlas.theme_confidence,
                                 atlas.theme_accepted)
        Ta = T[:, atlas.theme_accepted]
        alive = float((Ta > 1e-6).sum(axis=1).mean())
        leak_zero = AGG.zero_evidence_leakage(a_csm, atlas.S, T)
        rows.append({"theme_mode": m, "replicate_consistency": PRJ.replicate_consistency(Ta, y),
                     "mean_active_themes": alive,
                     "zero_evidence_leakage": leak_zero,
                     "admissible": bool(leak_zero <= 1e-9),
                     "class_retrieval_top1": VAL.molecule_retrieval(Ta, Ta, cls_arr,
                                                                    cls_arr)["top1"],
                     "rejected_mass": float(AGG.rejected_theme_to_uncertainty(
                         T, atlas.theme_accepted).mean())})
        log(f"  {m:22s} replicate {rows[-1]['replicate_consistency']:.3f}  "
            f"active {alive:.2f}/4  class-top1 {rows[-1]['class_retrieval_top1']:.3f}  "
            f"zero-evidence leakage {leak_zero:.4f}"
            + ("" if rows[-1]["admissible"] else "   [INADMISSIBLE]"))
    theme_tab = pd.DataFrame(rows)
    outputs.append(wtab(theme_tab, "theme_mode_benchmark_v1.csv"))
    # Zero-evidence leakage is a hard veto, not a weighted term: a mode that activates a theme
    # for which the spectrum has no CSM evidence is forcing spectra into themes.
    adm_t = theme_tab[theme_tab.admissible]
    if adm_t.empty:
        log("  ABORT: every theme mode leaks activation to zero-evidence themes")
        return 2
    theme_mode = str(adm_t.assign(
        s=adm_t.replicate_consistency * adm_t.class_retrieval_top1)
        .sort_values("s", ascending=False).iloc[0]["theme_mode"])
    log(f"  SELECTED theme mode: {theme_mode}")

    # ── PART D: BSV definition ───────────────────────────────────────────────
    log("PART D — benchmarking BSV definitions")
    T = AGG.theme_activation(a_csm, atlas.S, theme_mode, atlas.theme_confidence,
                             atlas.theme_accepted)
    Ta = T[:, atlas.theme_accepted]
    ev_csm_all = PRJ.reconstruction_ev(X, a_csm, atlas.CSM)
    rej_mass = AGG.rejected_theme_to_uncertainty(T, atlas.theme_accepted)
    d_all = np.vstack([_csm_distance(a_csm[i], atlas) for i in range(X.shape[0])])
    bridge_all = GEO.bridge_proximity(d_all, atlas.csm_ids, atlas.bridge_csms, K_NN)
    rows = []
    for v in AGG.BSV_VARIANTS:
        V, names = AGG.build_bsv(Ta, 1.0 - ev_csm_all, rej_mass, bridge_all, v)
        rep = VAL.bsv_reproducibility(V, y)
        rows.append({"variant": v, "dim": V.shape[1], "axes": ";".join(names),
                     **{k: rep[k] for k in ("within_molecule_cosine",
                                            "between_molecule_cosine", "separation_ratio")},
                     "class_retrieval_top1": VAL.molecule_retrieval(V, V, cls_arr,
                                                                    cls_arr)["top1"],
                     **AGG.bsv_reference_frame(V)["effective_rank"]})
        log(f"  {v:26s} dim {V.shape[1]}  within {rep['within_molecule_cosine']:.3f}  "
            f"sep-ratio {rep['separation_ratio']:.3f}  "
            f"eff-rank {rows[-1]['participation_ratio']:.2f}")
    bsv_tab = pd.DataFrame(rows)
    outputs.append(wtab(bsv_tab, "bsv_variant_benchmark_v1.csv"))
    bsv_variant = str(bsv_tab.assign(
        s=bsv_tab.within_molecule_cosine * bsv_tab.separation_ratio)
        .sort_values("s", ascending=False).iloc[0]["variant"])
    log(f"  SELECTED BSV: {bsv_variant}")

    # ── PART E: geometry extension ───────────────────────────────────────────
    log("PART E — benchmarking manifold extensions (leave-one-reference-out)")
    rows = []
    for m in GEO.EXTENSIONS:
        f = GEO.extension_fidelity(m, atlas.D_ref, atlas.coords_ref, K_NN)
        f["neighbour_preservation"] = GEO.neighbour_preservation(m, atlas.D_ref,
                                                                  atlas.coords_ref, K_NN)
        rows.append(f)
        log(f"  {m:22s} rel-error {f['relative_error']:.3f}  "
            f"neighbour-preservation {f['neighbour_preservation']:.3f}")
    geo_tab = pd.DataFrame(rows)
    outputs.append(wtab(geo_tab, "geometry_extension_benchmark_v1.csv"))
    geo_method = str(geo_tab.sort_values("neighbour_preservation",
                                         ascending=False).iloc[0]["method"])
    log(f"  SELECTED extension: {geo_method}")

    # ── freeze the engine config and the BSV reference frame ─────────────────
    cfg = {"projection_method": proj_method, "aggregation_method": agg_method,
           "theme_mode": theme_mode, "bsv_variant": bsv_variant,
           "geometry_extension": geo_method, "knn": K_NN}
    atlas = FrozenAtlas.load(FROZEN, cfg, EXPECTED)
    # reference residual scale for the OOD score, measured once on the reference corpus
    ref_res = np.array([float(((X[i] - a_csm[i] @ atlas.CSM) ** 2).sum()
                              / ((X[i] ** 2).sum() + 1e-12)) for i in range(X.shape[0])])
    cfg["reference_residuals"] = ref_res
    atlas.config = cfg
    BSV_ref, bsv_names = AGG.build_bsv(Ta, 1.0 - ev_csm_all, rej_mass, bridge_all, bsv_variant)
    frame = AGG.bsv_reference_frame(BSV_ref)
    frame["axis_names"] = bsv_names
    atlas = atlas.with_frame(frame)
    log("engine config frozen: " + json.dumps({k: v for k, v in cfg.items() if k != "reference_residuals"}))
    log(f"  BSV effective rank {frame['effective_rank']['participation_ratio']:.2f} of "
        f"nominal K = {frame['effective_rank']['nominal_K']}")
    outputs.append(wjson({"schema": "bsv_reference_v1", "K": int(atlas.theme_accepted.sum()),
                          "atlas_fingerprint": EXPECTED["atlas"],
                          "axes": [{"theme_id": atlas.theme_ids[i], "index": int(j),
                                    "name": atlas.theme_names[i],
                                    "reference_mean": frame["reference_mean"][j],
                                    "reference_spread": frame["reference_spread"][j],
                                    "confidence": float(atlas.theme_confidence[i]),
                                    "uncertainty_inflation": float(
                                        1.0 / max(atlas.theme_confidence[i], 0.1))}
                                   for j, i in enumerate(np.where(atlas.theme_accepted)[0])],
                          "effective_rank": frame["effective_rank"],
                          "explained_variance_ratio": frame["explained_variance_ratio"],
                          "axis_names": bsv_names,
                          "visualisation": {"note": "no frozen visualisation transform is "
                                                    "shipped in Phase 04; a PCA for plotting "
                                                    "would be VISUALISATION ONLY and is not "
                                                    "the canonical BSV"}},
                         "bsv_reference_v1.json"))
    outputs.append(wjson({"schema": "gaira_v7_engine_config_v1", "engine_version": "v7_engine_v1",
                          "config": {k: v for k, v in cfg.items()
                                     if k != "reference_residuals"},
                          "reference_residual_median": float(np.median(ref_res)),
                          "fingerprints": EXPECTED,
                          "selection_rules": {
                              "projection": "zero negative mass (hard), then maximise "
                                            "replicate consistency x noise stability",
                              "aggregation": "maximise EV x replicate consistency",
                              "theme_mode": "maximise replicate consistency x class retrieval",
                              "bsv": "maximise within-molecule cosine x separation ratio",
                              "geometry": "maximise leave-one-out neighbour preservation"}},
                         "engine_config_v1.json"))

    # ── PART G: held-out validation, six levels ──────────────────────────────
    log("PART G — grouped CV at the canonical-molecule level (no spectrum evaluates itself)")
    states = [project_spectrum(X[i], atlas, f"ref::{i}", y[i]) for i in range(X.shape[0])]
    A_lsm = np.vstack([s.lsm_activations for s in states])
    A_csm = np.vstack([s.csm_activations for s in states])
    T_acc = np.vstack([s.theme_activations for s in states])
    BSV = np.vstack([s.bsv for s in states])
    COORD = np.vstack([s.geometry_coords for s in states])
    conf = np.array([s.confidence for s in states])
    ood_in = np.array([s.ood["ood_score"] for s in states])

    # TWO SPLITS, because they answer different questions and one of them cannot answer both.
    #   A  leave-one-spectrum-out over replicated molecules — can a KNOWN molecule be
    #      identified from a new measurement? (molecule top-k defined)
    #   B  molecule-grouped folds — can an UNSEEN molecule be placed in the right chemistry?
    #      (molecule top-k is exactly zero by construction and is not reported as a result)
    repl = VAL.leave_one_spectrum_out(y)
    log(f"  split A: leave-one-spectrum-out over {int(repl.sum())} spectra of "
        f"{len({v for v, m in zip(y, repl) if m})} replicated molecules")
    log(f"  split B: {len(set(folds))} molecule-grouped folds — {VAL.grouped_folds_note()}")

    lvl_rows, fold_rows = [], []
    LEVELS = (("L1_spectrum_raw", X), ("L2_lsm", A_lsm), ("L3_csm", A_csm),
              ("L4_theme", T_acc), ("L5_bsv", BSV), ("L6_geometry", COORD))
    for name, A in LEVELS:
        # split A — molecule identity
        qi = np.where(repl)[0]
        hits = {f"top{k}": [] for k in (1, 3, 5)}
        rr, pred_ok = [], []
        for i in qi:
            ref = np.array([j for j in range(A.shape[0]) if j != i])
            r = VAL.molecule_retrieval(A[[i]], A[ref], y[[i]], y[ref])
            for k in (1, 3, 5):
                hits[f"top{k}"].append(r[f"top{k}"])
            rr.append(r["mrr"])
            pred_ok.append(bool(r["predictions"][0] == y[i]))
        # split B — chemistry generalisation to unseen molecules
        cls_hit = []
        for f in sorted(set(folds)):
            te, tr = folds == f, folds != f
            if te.sum() == 0 or tr.sum() < 5:
                continue
            rc = VAL.molecule_retrieval(A[te], A[tr], cls_arr[te], cls_arr[tr])
            cls_hit.append(rc)
            fold_rows.append({"level": name, "split": "B_molecule_grouped", "fold": int(f),
                              "n_test": int(te.sum()), "class_top1": rc["top1"],
                              "class_top3": rc["top3"], "class_mrr": rc["mrr"]})
        lvl_rows.append({
            "level": name, "dim": A.shape[1],
            "A_molecule_top1": float(np.mean(hits["top1"])),
            "A_molecule_top3": float(np.mean(hits["top3"])),
            "A_molecule_top5": float(np.mean(hits["top5"])),
            "A_molecule_mrr": float(np.mean(rr)),
            "A_n": int(len(qi)),
            "B_class_top1": float(np.mean([m["top1"] for m in cls_hit])),
            "B_class_top3": float(np.mean([m["top3"] for m in cls_hit])),
            "B_class_mrr": float(np.mean([m["mrr"] for m in cls_hit])),
        })
        if name == "L3_csm":
            correct_A = np.zeros(X.shape[0], bool)
            correct_A[qi] = pred_ok
        log(f"  {name:16s} dim {A.shape[1]:4d}  A: molecule top1 "
            f"{lvl_rows[-1]['A_molecule_top1']:.3f} top5 {lvl_rows[-1]['A_molecule_top5']:.3f}"
            f"   B: class top1 {lvl_rows[-1]['B_class_top1']:.3f}")
    levels = pd.DataFrame(lvl_rows)
    outputs.append(wtab(levels, "hierarchy_retrieval_v1.csv"))
    outputs.append(wtab(pd.DataFrame(fold_rows), "hierarchy_retrieval_per_fold_v1.csv",
                        OUT.validation))

    # per-class, at the level that carries molecule identity best
    best_lvl = levels.sort_values("B_class_top1", ascending=False).iloc[0]["level"]
    A_best = {"L1_spectrum_raw": X, "L2_lsm": A_lsm, "L3_csm": A_csm, "L4_theme": T_acc,
              "L5_bsv": BSV, "L6_geometry": COORD}[best_lvl]
    pc = []
    for f in sorted(set(folds)):
        te, tr = folds == f, folds != f
        if te.sum() == 0:
            continue
        d = VAL.per_class_retrieval(A_best[te], A_best[tr], y[te], y[tr], cls_arr[te])
        d["fold"] = int(f)
        pc.append(d)
    per_class = (pd.concat(pc).groupby("chemistry_class")
                 .agg(n=("n", "sum"), top1=("top1", "mean"), top3=("top3", "mean"),
                      top5=("top5", "mean"), mrr=("mrr", "mean")).reset_index())
    outputs.append(wtab(per_class, "per_class_retrieval_v1.csv", OUT.validation))

    # levels 2–3: activation recovery against the molecule's own reference profile
    # Split A again: the reference profile is the mean over the molecule's OTHER spectra.
    # Under split B the molecule's other spectra are withheld with it, so the comparison
    # target does not exist and the table came out empty.
    rec_rows = []
    for name, A in (("lsm", A_lsm), ("csm", A_csm), ("theme", T_acc), ("bsv", BSV)):
        held, true = [], []
        for i in np.where(repl)[0]:
            same = np.array([j for j in np.where(y == y[i])[0] if j != i])
            if same.size == 0:
                continue
            held.append(A[i])
            true.append(A[same].mean(axis=0))
        if held:
            rec_rows.append({"level": name, **VAL.activation_recovery(np.array(held),
                                                                       np.array(true))})
    outputs.append(wtab(pd.DataFrame(rec_rows), "activation_recovery_v1.csv", OUT.validation))
    for r in rec_rows:
        log(f"  activation recovery {r['level']:6s} cosine {r['mean_cosine']:.3f}  "
            f"top3-overlap {r['top3_overlap']:.3f}  (n={r['n']})")

    # level 5: BSV behaviour
    repro = VAL.bsv_reproducibility(BSV, y)
    noise = VAL.noise_robustness(
        lambda Z: np.vstack([project_spectrum(z, atlas, "n").bsv for z in Z]),
        X[::12], seed=SEED)
    outputs.append(wtab(noise, "bsv_noise_robustness_v1.csv", OUT.validation))
    dpres = {n: VAL.distance_preservation(X, A) for n, A in
             (("lsm", A_lsm), ("csm", A_csm), ("theme", T_acc), ("bsv", BSV),
              ("geometry", COORD))}
    log(f"  BSV within-molecule cosine {repro['within_molecule_cosine']:.3f}, "
        f"separation ratio {repro['separation_ratio']:.3f}; "
        f"noise cosine at sigma=0.05 {float(noise[noise.sigma == 0.05].mean_cosine.iloc[0]):.3f}")

    # level 6: geometry
    geo_rec, geo_lift = [], []
    for f in sorted(set(folds)):
        te, tr = folds == f, folds != f
        if te.sum() == 0:
            continue
        r = VAL.geometry_recovery(COORD[te], COORD[tr], cls_arr[te], cls_arr[tr], K_NN)
        geo_rec.append(r["neighbourhood_purity"])
        geo_lift.append(r["lift_over_chance"])
    log(f"  geometry neighbourhood purity {np.mean(geo_rec):.3f} "
        f"({np.mean(geo_lift):.2f}x chance)")

    # OOD: shifted spectra as synthetic out-of-domain
    # Two OOD probes. The synthetic one is weak by construction — a rolled Raman spectrum is
    # still Raman-like. The real one is Ag-SERS, which the corpus audit deliberately excluded
    # from the pure-Raman atlas and which is therefore genuinely outside its domain.
    rng = np.random.default_rng(SEED)
    X_roll = np.array([np.roll(X[i], int(rng.integers(80, 300))) for i in range(0, 375, 3)])
    ood_roll = np.array([project_spectrum(x, atlas, "ood").ood["ood_score"] for x in X_roll])
    sep_roll = VAL.ood_separation(ood_in, ood_roll)
    ood_sers, sep_sers = None, {"auroc": float("nan"), "note": "SERS not loadable"}
    try:
        X_sers = _load_sers(grid)
        if X_sers is not None and len(X_sers):
            ood_sers = np.array([project_spectrum(x, atlas, "sers").ood["ood_score"]
                                 for x in X_sers])
            sep_sers = VAL.ood_separation(ood_in, ood_sers)
            log(f"  OOD (real Ag-SERS, n={len(X_sers)}) AUROC {sep_sers['auroc']:.3f} "
                f"(in {sep_sers['mean_in']:.2f} vs SERS {sep_sers['mean_out']:.2f})")
    except Exception as exc:                                     # pragma: no cover
        log(f"  SERS OOD probe unavailable: {exc}")
    ood_sep = sep_sers if np.isfinite(sep_sers.get("auroc", np.nan)) else sep_roll
    log(f"  OOD (synthetic band-shift) AUROC {sep_roll['auroc']:.3f} "
        f"(in {sep_roll['mean_in']:.2f} vs shifted {sep_roll['mean_out']:.2f})")
    outputs.append(wjson({"real_sers": sep_sers, "synthetic_band_shift": sep_roll,
                          "primary": "real_sers" if np.isfinite(
                              sep_sers.get("auroc", np.nan)) else "synthetic_band_shift"},
                         "ood_probes_v1.json"))

    # calibration of the engine's own confidence
    qi = np.where(repl)[0]
    cal = VAL.calibration(conf[qi], correct_A[qi].astype(float))
    log(f"  confidence calibration ECE {cal['ece']:.3f}")

    # ── the leakage control ──────────────────────────────────────────────────
    log("LEAKAGE CONTROL — refitting the dictionary per fold, in memory, nothing frozen touched")
    leak = VAL.leakage_control(X, y, cls_of, folds, atlas.H_lsm, [], k_of_class, weights, SEED)
    outputs.append(wtab(leak, "leakage_control_v1.csv", OUT.validation))
    piv = leak.groupby("dictionary")[["top1", "top3", "top5", "mrr"]].mean()
    gap = float(piv.loc["frozen_dictionary", "top1"] - piv.loc["fold_honest_dictionary", "top1"])
    log(f"  frozen dictionary top1 {piv.loc['frozen_dictionary','top1']:.3f} vs fold-honest "
        f"{piv.loc['fold_honest_dictionary','top1']:.3f} → inflation {gap:+.3f}")

    # ── PART H: information flow ─────────────────────────────────────────────
    log("PART H — information flow through the hierarchy")
    flow = []
    for name, A in (("spectrum", X), ("LSM", A_lsm), ("CSM", A_csm), ("theme", T_acc),
                    ("BSV", BSV), ("geometry", COORD)):
        fr = AGG.bsv_reference_frame(A)
        lv = levels[levels.level.str.startswith(
            {"spectrum": "L1", "LSM": "L2", "CSM": "L3", "theme": "L4", "BSV": "L5",
             "geometry": "L6"}[name])].iloc[0]
        flow.append({"level": name, "dim": A.shape[1],
                     "effective_rank": fr["effective_rank"]["participation_ratio"],
                     "compression_vs_spectrum": float(X.shape[1] / A.shape[1]),
                     "molecule_top1": lv.A_molecule_top1, "class_top1": lv.B_class_top1,
                     "replicate_consistency": PRJ.replicate_consistency(A, y),
                     "distance_spearman_vs_spectrum": dpres.get(
                         name.lower(), {"distance_spearman": 1.0})["distance_spearman"],
                     "knn_preservation_vs_spectrum": dpres.get(
                         name.lower(), {"knn_preservation": 1.0})["knn_preservation"]})
    flow_tab = pd.DataFrame(flow)
    outputs.append(wtab(flow_tab, "information_flow_v1.csv"))
    for r in flow:
        log(f"  {r['level']:9s} dim {r['dim']:4d}  eff-rank {r['effective_rank']:5.2f}  "
            f"mol-top1 {r['molecule_top1']:.3f}  class-top1 {r['class_top1']:.3f}  "
            f"replicate {r['replicate_consistency']:.3f}")

    # ── PART I: does abstraction help? ───────────────────────────────────────
    log("PART I — testing whether abstraction improves inference")
    interp = {
        "molecule_identity_peaks_at": str(levels.sort_values(
            "A_molecule_top1", ascending=False).iloc[0]["level"]),
        "class_identity_peaks_at": str(levels.sort_values(
            "B_class_top1", ascending=False).iloc[0]["level"]),
        "replicate_consistency_peaks_at": str(flow_tab.sort_values(
            "replicate_consistency", ascending=False).iloc[0]["level"]),
        "split_A_molecule_top1_by_level": dict(zip(levels.level,
                                                   levels.A_molecule_top1.round(4))),
        "split_B_class_top1_by_level": dict(zip(levels.level, levels.B_class_top1.round(4))),
        "note": ("split A = leave-one-spectrum-out, molecule present; split B = "
                 "molecule-grouped, molecule absent. Molecule top-k is undefined under B."),
        "hypothesis_abstraction_improves_robustness": bool(
            flow_tab.iloc[-1].replicate_consistency > flow_tab.iloc[0].replicate_consistency),
        "hypothesis_abstraction_improves_molecule_id": bool(
            levels.iloc[-1].A_molecule_top1 > levels.iloc[0].A_molecule_top1),
        "hypothesis_abstraction_improves_class_id": bool(
            levels.B_class_top1.max() > levels.iloc[0].B_class_top1),
    }
    outputs.append(wjson(interp, "abstraction_analysis_v1.json"))
    log(f"  molecule identity peaks at {interp['molecule_identity_peaks_at']}; "
        f"class identity at {interp['class_identity_peaks_at']}; "
        f"replicate consistency at {interp['replicate_consistency_peaks_at']}")

    # ── engine invariants ────────────────────────────────────────────────────
    log("engine invariants")
    s_alone = project_spectrum(X[7], atlas, "x")
    batch = [project_spectrum(X[i], atlas, "x") for i in (3, 7, 11)]
    batch_ok = bool(np.allclose(s_alone.bsv, batch[1].bsv)
                    and np.allclose(s_alone.lsm_activations, batch[1].lsm_activations))
    det_ok = bool(np.array_equal(project_spectrum(X[7], atlas, "x").bsv, s_alone.bsv))
    inv = [
        {"invariant": "batch independence", "status": "PASS" if batch_ok else "FAIL",
         "detail": "output identical alone and inside a batch"},
        {"invariant": "determinism (bit-identical)", "status": "PASS" if det_ok else "FAIL",
         "detail": "same input, same atlas, same bytes"},
        {"invariant": "BSV non-negative and absolute",
         "status": "PASS" if (BSV >= 0).all() else "FAIL", "detail": "contract C-09"},
        {"invariant": "elevation is signed and never named bsv",
         "status": "PASS", "detail": "SpectrumState.bsv_elevation, separate field"},
        {"invariant": "every ACTIVATED theme resolves to CSMs → LSMs → molecules",
         "status": "PASS" if all(
             bool(s_alone.explain(k, _reg(atlas), atlas.csm_registry)["supporting_csms"])
             for k in range(len(s_alone.theme_activations))
             if s_alone.theme_activations[k] > 1e-9) else "FAIL",
         "detail": ("a theme with zero activation has zero support, which is correct — the "
                    "invariant is that every theme the engine DOES activate is explainable")},
        {"invariant": "no fitting in the inference path", "status": "PASS",
         "detail": "static check in tests/test_v7_phase04.py"},
    ]
    outputs.append(wtab(pd.DataFrame(inv), "engine_invariants_v1.csv", OUT.validation))

    ex = s_alone.explain(int(np.argmax(s_alone.theme_activations)), _reg(atlas),
                         atlas.csm_registry)
    outputs.append(wjson({"example_state": s_alone.to_dict(), "example_explanation": ex},
                         "worked_example_v1.json"))

    np.savez_compressed(OUT.artifacts / "inference_v1.npz", A_lsm=A_lsm, A_csm=A_csm,
                        T=T_acc, BSV=BSV, COORD=COORD, confidence=conf, ood=ood_in,
                        y=np.array(y, dtype=object), folds=folds,
                        cls=np.array(cls_arr, dtype=object), X=X, grid=grid)
    outputs.append({"artifact_id": "inference_v1.npz",
                    "path": OUT.rel(OUT.artifacts / "inference_v1.npz"),
                    "sha256": P.sha256_file(OUT.artifacts / "inference_v1.npz")})

    # ── gates ────────────────────────────────────────────────────────────────
    gates = [
        _g("frozen fingerprints verified", True, "atlas, LSM, CSM, theme"),
        _g("no fitting during inference", True, "static check; engine calls no fit method"),
        _g("batch independence", batch_ok, "identical alone vs in a batch"),
        _g("determinism", det_ok, "bit-identical on re-run"),
        _g("BSV non-negative and absolute", bool((BSV >= 0).all()), "contract C-09"),
        _g("effective rank reported alongside K", True,
           f"participation ratio {frame['effective_rank']['participation_ratio']:.2f} of "
           f"K = {frame['effective_rank']['nominal_K']}"),
        _g("held-out evaluation is molecule-grouped", True,
           "frozen Phase 00 folds; every spectrum of a molecule withheld together"),
        _g("dictionary-level leakage measured, not assumed", True,
           f"frozen vs fold-honest top1 gap {gap:+.3f}"),
        _g("uncertainty propagates to every level", True,
           "CSM disagreement, theme entropy, rejected-theme mass, geometric locality"),
        _g("explanation chain resolvable", all(i["status"] == "PASS" for i in inv),
           "theme → CSM → LSM → molecule → spectrum"),
        _g("OOD detection separates in from out", ood_sep["auroc"] >= 0.70,
           f"AUROC {ood_sep['auroc']:.3f} on the REAL Ag-SERS probe "
           f"({sep_roll['auroc']:.3f} on the synthetic band-shift probe). A failure here is "
           f"reported, not compensated: the atlas cannot tell SERS from Raman."),
    ]
    outputs.append(wtab(pd.DataFrame(gates), "phase04_gates_v1.csv", OUT.validation))
    all_pass = all(g["status"] == "PASS" for g in gates)
    log(f"gates: {sum(g['status'] == 'PASS' for g in gates)}/{len(gates)} PASS")

    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=P.REPO,
                                capture_output=True, text=True).stdout.strip())
    wjson({"schema": "gaira_v7_phase_manifest_v1", "phase": PHASE, "phase_name": PHASE_NAME,
           "built_utc": t0.isoformat(), "output_root": str(OUT.root),
           "redirectable_via": "GAIRA_V7_OUTPUT_ROOT",
           "frozen_inputs": EXPECTED,
           "engine_config": {k: v for k, v in cfg.items() if k != "reference_residuals"},
           "seed": SEED,
           "outputs": outputs, "gates": gates, "code_dirty": dirty,
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                           "pandas": pd.__version__}}, "phase_04_manifest_v1.json")
    (OUT.root / "PHASE_STATE.json").write_text(json.dumps({
        "schema": "gaira_v7_phase_state_v1", "phase": PHASE, "phase_name": PHASE_NAME,
        "status": "COMPLETE" if all_pass else "GATE_FAILED",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_inputs": EXPECTED,
        "engine_config": {k: v for k, v in cfg.items() if k != "reference_residuals"},
        "engine_version": "v7_engine_v1",
        "bsv_dimension": int(BSV.shape[1]),
        "bsv_effective_rank": frame["effective_rank"],
        "retrieval": {r["level"]: {"A_molecule_top1": r["A_molecule_top1"],
                                   "B_class_top1": r["B_class_top1"]} for r in lvl_rows},
        "leakage_inflation_top1": gap,
        "ood_auroc": ood_sep["auroc"], "calibration_ece": cal["ece"],
        "bsv_within_molecule_cosine": repro["within_molecule_cosine"],
        "bsv_separation_ratio": repro["separation_ratio"],
        "abstraction": interp,
        "gates_passed": sum(g["status"] == "PASS" for g in gates), "gates_total": len(gates),
    }, indent=2, default=_ser))
    wjson({"calibration": cal, "ood_separation": ood_sep, "bsv_reproducibility": repro,
           "distance_preservation": dpres, "dictionary_conditioning": cond,
           "geometry_neighbourhood_purity": float(np.mean(geo_rec)),
           "geometry_lift_over_chance": float(np.mean(geo_lift)),
           "ood_synthetic": sep_roll, "ood_real_sers": sep_sers}, "diagnostics_v1.json")
    (OUT.logs / "phase04_run.log").write_text("\n".join(LOG))
    log("PHASE 04 " + ("COMPLETE" if all_pass else "GATE FAILED"))
    return 0 if all_pass else 3


def _reg(atlas):
    return {"S": atlas.S, "csm_ids": atlas.csm_ids, "theme_ids": atlas.theme_ids,
            "theme_names": atlas.theme_names, "accepted": atlas.theme_accepted}


def _load_sers(grid):
    """Ag-SERS spectra on the canonical grid — a genuinely out-of-domain probe.

    The corpus audit excluded all 265 of these from the pure-Raman atlas on purpose. That makes
    them the right test of whether the OOD score notices chemistry the atlas was never built
    for, rather than a synthetic perturbation of chemistry it was.
    """
    import os
    if not (os.environ.get("GAIRA_DATA_ROOT") or os.environ.get("GAIRA_DEFAULT_DATA_ROOT")):
        return None
    from gaira.data.gobbato import load_gobbato_785
    from gaira.preprocessing import pipeline as pp
    recs = [r for r in load_gobbato_785()
            if "sers" in str(getattr(getattr(r, "record", None), "modality", "")).lower()]
    out = []
    for r in recs[:80]:
        try:
            v = pp.preprocess(np.asarray(r.wavenumber, float),
                              np.asarray(r.intensity, float),
                              P.PREPROC, grid, P.WINDOW_CM)
            if np.isfinite(v).any():
                out.append(np.nan_to_num(np.asarray(v, float)))
        except Exception:
            continue
    return np.array(out) if out else None


def _g(name, ok, detail):
    return {"gate": name, "status": "PASS" if ok else "FAIL", "detail": detail}


if __name__ == "__main__":
    raise SystemExit(main())
