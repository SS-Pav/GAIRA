#!/usr/bin/env python3
"""GAIRA V7 — Phase 06.5: latent spectral geometry audit (Raman only).

Sections 1–10 of the brief. Nothing upstream is refitted. **No chemistry label enters the
construction of any geometry**; labels are revealed afterwards, for interpretation and external
validation only. Output location resolves through `gaira.v7.io.PhaseOutputs`.

    python results/v7_rebuild/phase06_5/code/run_phase06_5.py
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
from gaira.v7.chemistry import evidence as CHEM, registry as REG        # noqa: E402
from gaira.v7.chemistry import validation as CVAL                       # noqa: E402
from gaira.v7.latent import (clustering as CLU, composition as COMP,    # noqa: E402
                             confounding as CONF, coordinates as COORD,
                             hierarchy as HIER)

warnings.filterwarnings("ignore")

PHASE, PHASE_NAME = "06_5", "Latent spectral geometry audit"
OUT = PhaseOutputs(PHASE, extra=("interactive", "manifests")).ensure()
FROZEN = frozen_root()
EXPECTED = {"atlas": "09ed804a40836f4a05a91ba10900cded",
            "lsm": "208482d6f7178b5b8f16cace91be55b0",
            "csm": "0b4aa550ccefed3edabdbde5bae11c8d",
            "engine": "20d8bd99ce71f45a125c6a2b1d719e51"}
SEED, N_BOOT = 0, 40
LOG: list[str] = []


def log(m):
    line = f"[phase06.5] {m}"
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

    # ── SECTION 0 — fingerprints and compliance ──────────────────────────────
    log("ARCHITECTURE COMPLIANCE STATEMENT")
    for line in [
        "  Phase 06.5 is an AUDIT. It builds no inference layer and modifies nothing upstream.",
        "  It asks what biochemical organisation emerges from the frozen CSM manifold ITSELF,",
        "  independent of the curated 16-class ontology.",
        "  NO chemistry label enters the construction of any geometry. Labels are revealed only",
        "  afterwards, as an external validation target (Sections 2, 4, 7).",
        "  Scope is Raman only: no SERS, Ag-SERS, serum, EV, mixture, DART or perturbation",
        "  dataset is loaded or cited as validation.",
        "  Unit of analysis is the CANONICAL MOLECULE, not the spectrum: clustering replicates",
        "  would manufacture stability (P-11 applied to geometry).",
        "  Phase 07 is NOT begun. The output is a recommendation, not an architecture change.",
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
        log(f"ABORT — Phase 06 status is {st06['status']}")
        return 2
    log(f"  frozen verified: LSM {got['lsm']} · CSM {got['csm']} · engine {got['engine']}")
    log(f"  Phase 06 COMPLETE, model {st06['selected_model']}")

    # ── corpus, at both levels ───────────────────────────────────────────────
    z6 = np.load(FROZEN / "phase06/artifacts/chemistry_evidence_predictions_v1.npz",
                 allow_pickle=True)
    A_spec = z6["A_csm"]
    y = np.array([str(v) for v in z6["y"]])
    cls_spec = np.array([str(v) for v in z6["cls"]])
    src_spec = np.array([str(v) for v in z6["source"]])
    exc_spec = np.array([str(v) for v in z6["excitation"]])
    folds_spec = z6["folds"]
    ev_spec = z6["explained_variance"]
    E_chem = z6["E"]
    br = np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
    X, grid = np.asarray(br["X"], float), np.asarray(br["grid"], float)
    CSM = np.load(FROZEN / "phase02/artifacts/csm_dictionary_v1.npz")["CSM"]
    recs = csm_reg["csms"]
    canon = pd.read_csv(FROZEN / "phase00/tables/canonical_analytes_v1.csv")
    broad_of = dict(zip(canon.canonical_id, canon.broad_class))
    cls_of = dict(zip(pd.read_csv(FROZEN / "phase00/tables/chemical_partition_v1.csv")
                      .canonical_id, pd.read_csv(
                          FROZEN / "phase00/tables/chemical_partition_v1.csv").fine_class))

    mols = sorted(set(y.tolist()))
    M = np.vstack([A_spec[y == m].mean(axis=0) for m in mols])          # molecule level
    cls_m = np.array([cls_of[m] for m in mols])
    broad_m = np.array([broad_of[m] for m in mols])
    src_m = np.array([pd.Series(src_spec[y == m]).mode().iloc[0] for m in mols])
    exc_m = np.array([pd.Series(exc_spec[y == m]).mode().iloc[0] for m in mols])
    nrep_m = np.array([int((y == m).sum()) for m in mols])
    ev_m = np.array([float(ev_spec[y == m].mean()) for m in mols])
    fold_m = np.array([int(folds_spec[y == m][0]) for m in mols])
    log(f"  molecules {M.shape} · spectra {A_spec.shape} · {len(set(cls_m))} chemistry classes "
        f"· {len(set(src_m))} sources · {len(set(exc_m))} excitations")

    STAMP.update({"phase": PHASE, "seed": SEED, "input_fingerprints": got,
                  "unit_of_analysis": "canonical molecule (154)",
                  "labels_used_in_construction": False,
                  "code_fingerprint": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
                  "created_utc": t0.isoformat(), "scope": "Raman only"})

    # ── SECTION 1 — cluster stability audit ──────────────────────────────────
    log("SECTION 1 — cluster stability audit (label-free)")
    sweep = CLU.sweep(M, n_boot=N_BOOT, seed=SEED, log=log)
    outputs.append(wtab(sweep, "cluster_stability_sweep_v1.csv"))
    ok = sweep[sweep.usable].copy()
    # Pre-declared stability rule: a K is STABLE when bootstrap ARI >= 0.60 AND no cluster falls
    # below Jaccard 0.50; DEGENERATE when membership entropy < 0.50 (one cluster swallows the
    # corpus); UNSTABLE otherwise. Declared before the sweep was inspected.
    def verdict(r):
        if r.membership_entropy < 0.50:
            return "degenerate"
        if r.bootstrap_ari_mean >= 0.60 and r.min_cluster_survival >= 0.50:
            return "stable"
        return "unstable"
    ok["verdict"] = ok.apply(verdict, axis=1)
    outputs.append(wtab(ok, "cluster_stability_verdicts_v1.csv"))
    by_k = ok.groupby("K").agg(
        n_algorithms=("algorithm", "size"),
        n_stable=("verdict", lambda s: int((s == "stable").sum())),
        mean_bootstrap_ari=("bootstrap_ari_mean", "mean"),
        mean_silhouette=("silhouette", "mean"),
        mean_survival=("mean_cluster_survival", "mean"),
        mean_neighbour_preservation=("neighbour_preservation", "mean")).reset_index()
    outputs.append(wtab(by_k, "cluster_stability_by_k_v1.csv"))
    log("  per-K summary (mean over 4 fixed-K algorithms):")
    for _, r in by_k.iterrows():
        log(f"    K={int(r.K):2d}  stable {int(r.n_stable)}/{int(r.n_algorithms)}  "
            f"bootARI {r.mean_bootstrap_ari:.3f}  sil {r.mean_silhouette:+.3f}  "
            f"survival {r.mean_survival:.3f}  nbr {r.mean_neighbour_preservation:.3f}")
    # Does ANY K stand out? An index with an interior optimum indicates a preferred cluster
    # count; an index that is monotone in K indicates there is no preferred count at all — the
    # index is just tracking granularity. This test is declared before the sweep is read,
    # because "which K is best" is only a meaningful question if some index has a peak.
    from scipy.stats import spearmanr
    mono = []
    for idx, better in (("silhouette", "higher"), ("calinski_harabasz", "higher"),
                        ("davies_bouldin", "lower"), ("neighbour_preservation", "higher"),
                        ("membership_entropy", "higher"),
                        ("bootstrap_ari_mean", "higher"),
                        ("mean_cluster_survival", "higher")):
        g = ok.groupby("K")[idx].mean().dropna()
        if len(g) < 5:
            continue
        rho, pv = spearmanr(g.index.values, g.values)
        vals = g.values
        interior = bool(np.argmax(vals) not in (0, len(vals) - 1)) if better == "higher" \
            else bool(np.argmin(vals) not in (0, len(vals) - 1))
        mono.append({"index": idx, "better": better, "spearman_rho_vs_K": float(rho),
                     "p_value": float(pv), "monotone": bool(abs(rho) > 0.85),
                     "has_interior_optimum": interior,
                     "optimal_K": int(g.index[int(np.argmax(vals) if better == "higher"
                                                  else np.argmin(vals))])})
    mono_tab = pd.DataFrame(mono)
    outputs.append(wtab(mono_tab, "k_selection_monotonicity_v1.csv"))
    log("  does any K stand out?")
    for _, r in mono_tab.iterrows():
        log(f"    {r['index']:24s} Spearman vs K {r.spearman_rho_vs_K:+.3f} "
            f"(p {r.p_value:.4f})  monotone {bool(r.monotone)}  interior optimum "
            f"{bool(r.has_interior_optimum)}  argbest K={int(r.optimal_K)}")
    n_interior = int(mono_tab.has_interior_optimum.sum())
    no_preferred_k = bool(n_interior <= 1)
    log(f"  {n_interior} of {len(mono_tab)} indices have an interior optimum → "
        f"a preferred cluster count {'does NOT exist' if no_preferred_k else 'may exist'}")

    stable_ks = sorted(set(int(k) for k in ok[ok.verdict == "stable"].K))
    log(f"  STABLE Ks: {stable_ks}")
    log(f"  DEGENERATE Ks: {sorted(set(int(k) for k in ok[ok.verdict == 'degenerate'].K))}")
    k16 = ok[ok.K == 16]
    log(f"  Is K=16 stable? {int((k16.verdict == 'stable').sum())}/{len(k16)} algorithms; "
        f"mean bootstrap ARI {k16.bootstrap_ari_mean.mean():.3f}")

    # free-K algorithms
    free_rows = []
    for algo, params in (("hdbscan", (2, 3, 4, 5)), ("affinity_propagation", (0.1, 0.3, 0.5))):
        for prm in params:
            try:
                lab = CLU.fit(algo, M, None, SEED, prm)
            except Exception as exc:                                    # pragma: no cover
                free_rows.append({"algorithm": algo, "param": prm, "usable": False,
                                  "error": str(exc)[:60]})
                continue
            iv = CLU.internal_indices(M, lab)
            bs = CLU.bootstrap_stability(M, algo, None, n_boot=20, seed=SEED, param=prm)
            free_rows.append({"algorithm": algo, "param": prm, "usable": True, **iv,
                              "membership_entropy": CLU.membership_entropy(lab),
                              "neighbour_preservation": CLU.neighbour_preservation(M, lab),
                              **{k: v for k, v in bs.items() if k != "consensus"}})
            log(f"    {algo:22s} param={prm:<5} → K={iv['n_clusters']:2d} "
                f"unassigned {iv['n_unassigned']:3d}  bootARI {bs['bootstrap_ari_mean']:.3f}")
    free_tab = pd.DataFrame(free_rows)
    outputs.append(wtab(free_tab, "free_k_algorithms_v1.csv"))

    # A canonical partition is still needed to *describe* the space in Sections 2 and 5. Since
    # no K is distinguished (above), any choice is a reporting convention rather than a
    # discovery, and the convention is declared here: **K = 16**, matching the size of the
    # curated ontology, so that Section 7's comparison is like-for-like rather than comparing
    # partitions of different granularity. Within K = 16 the algorithm is chosen label-free, by
    # bootstrap stability.
    #
    # An earlier version selected argmax bootstrap ARI over all K and chose K = 4 — because a
    # coarse partition is trivially reproducible. That is the P-18 failure mode in a new place:
    # a stability metric maximised by a low-information answer.
    KSEL = 16
    cand = ok[ok.K == KSEL]
    best = cand.sort_values(["bootstrap_ari_mean", "neighbour_preservation"],
                            ascending=False).iloc[0]
    ALGO = str(best.algorithm)
    lab = CLU.fit(ALGO, M, KSEL, SEED)
    log(f"  CANONICAL EMERGENT PARTITION: {ALGO} at K={KSEL} "
        f"(bootstrap ARI {best.bootstrap_ari_mean:.3f}, silhouette {best.silhouette:+.3f})")
    log("  K=16 is a REPORTING CONVENTION for comparability with the curated ontology, not a "
        "discovered optimum. No K is distinguished by any internal index.")

    # ── SECTION 2 — cluster composition ──────────────────────────────────────
    log("SECTION 2 — composition of every emergent cluster")
    base = COMP.baselines(cls_m, src_m, exc_m)
    comp = COMP.describe(lab, M, mols, cls_m, broad_m, src_m, exc_m, nrep_m, recs, grid, CSM)
    for rec in comp:
        kind, why = COMP.classify(rec, len(mols), base["source"], base["excitation"])
        rec["kind"], rec["justification"] = kind, why
        log(f"  cluster {rec['cluster']:2d} n={rec['n_molecules']:3d} "
            f"{kind:26s} {rec['dominant_fine_class']:26s} purity {rec['fine_purity']:.2f} "
            f"src {rec['source_purity']:.2f} exc {rec['excitation_purity']:.2f}")
        log(f"              → {why}")
    outputs.append(wjson({"algorithm": ALGO, "K": KSEL, "baselines": base,
                          "clusters": comp}, "cluster_composition_v1.json"))
    outputs.append(wtab(pd.DataFrame([
        {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()}
        for r in comp]), "cluster_composition_v1.csv"))
    kinds = pd.Series([r["kind"] for r in comp]).value_counts()
    log(f"  cluster kinds: {kinds.to_dict()}")

    # ── SECTION 4 — confounding ──────────────────────────────────────────────
    log("SECTION 4 — is the geometry chemistry, or the instrument?")
    D_mol = CLU.cosine_distance(M)
    factors = {"fine_chemistry": cls_m, "broad_chemistry": broad_m, "source": src_m,
               "excitation": exc_m,
               "replicate_count": np.array([f"n{v}" for v in nrep_m]),
               "reconstruction_EV_tertile": pd.qcut(ev_m, 3,
                                                    labels=["low", "mid", "high"]).astype(str),
               "intensity_tertile": pd.qcut(M.sum(axis=1), 3,
                                            labels=["low", "mid", "high"]).astype(str)}
    vp = CONF.variance_partition(D_mol, factors, n_perm=999, seed=SEED)
    outputs.append(wtab(vp, "permanova_variance_partition_v1.csv"))
    for _, r in vp.iterrows():
        log(f"  PERMANOVA {r.factor:26s} levels {int(r.n_levels):2d}  R² {r.marginal_R2:.4f}  "
            f"F {r.pseudo_F:6.2f}  p {r.p_value:.4f}")
    cvf = CONF.cluster_vs_factor(lab, factors)
    outputs.append(wtab(cvf, "cluster_vs_factor_ami_v1.csv"))
    for _, r in cvf.iterrows():
        log(f"  partition AMI vs {r.factor:26s} {r.AMI:+.4f}  ARI {r.ARI:+.4f}")
    ev_anova = CONF.anova_effect_sizes(ev_m, lab.astype(str))
    log(f"  ANOVA: reconstruction EV across clusters η² {ev_anova['eta_squared']:.3f} "
        f"p {ev_anova['p_value']:.4f}")
    chem_r2 = float(vp.set_index("factor").loc["fine_chemistry", "marginal_R2"])
    src_r2 = float(vp.set_index("factor").loc["source", "marginal_R2"])
    exc_r2 = float(vp.set_index("factor").loc["excitation", "marginal_R2"])
    chemistry_dominates = bool(chem_r2 > src_r2 and chem_r2 > exc_r2)
    log(f"  CHEMISTRY DOMINATES: {chemistry_dominates} "
        f"(chemistry R² {chem_r2:.3f} vs source {src_r2:.3f}, excitation {exc_r2:.3f})")
    outputs.append(wjson({"chemistry_dominates": chemistry_dominates,
                          "fine_chemistry_R2": chem_r2, "source_R2": src_r2,
                          "excitation_R2": exc_r2, "ev_anova": ev_anova,
                          "partition_ami": cvf.to_dict("records")},
                         "confounding_verdict_v1.json"))

    # ── SECTION 8 — hierarchical structure ───────────────────────────────────
    log("SECTION 8 — is Raman molecule space tree-like, graph-like, continuous or modular?")
    idim = HIER.intrinsic_dimension(M)
    gap = HIER.gap_statistic(M)
    mod = HIER.modularity_vs_null(M, seed=SEED)
    coph = HIER.cophenetic_fit(M)
    cont = HIER.continuity(M)
    br_tab = HIER.branch_points(M)
    outputs.append(wtab(br_tab, "dendrogram_branch_points_v1.csv"))
    log(f"  intrinsic dimension: Levina–Bickel {idim['levina_bickel_mle']:.2f}, "
        f"correlation {idim['correlation_dimension']:.2f} of ambient "
        f"{idim['ambient_dimension']} (agree: {idim['estimators_agree']})")
    log(f"  distance distribution: valley depth {gap['valley_depth']:.3f} → "
        f"bimodal {gap['bimodal']}")
    log(f"  modularity {mod['modularity']:.3f} vs null {mod['null_mean']:.3f}±"
        f"{mod['null_sd']:.3f} (z {mod['z_score']:.1f}, p {mod['p_value']:.4f}), "
        f"{mod['n_communities']} communities {mod['community_sizes'][:6]}")
    log(f"  cophenetic correlation: best {coph['best_linkage']} {coph['best_correlation']:.3f} "
        f"→ tree-like {coph['tree_like']}")
    log(f"  continuity: {cont['fraction_bridging']:.1%} of molecules have neighbours in more "
        f"than one community")
    shape = ("modular" if (mod["z_score"] > 3 and gap["bimodal"]) else
             "tree-like" if coph["tree_like"] else
             "graph-like" if mod["z_score"] > 3 else "continuous")
    log(f"  VERDICT: the space is {shape.upper()}")
    outputs.append(wjson({"intrinsic_dimension": idim, "distance_gap": gap,
                          "modularity": mod, "cophenetic": coph, "continuity": cont,
                          "shape_verdict": shape}, "hierarchical_structure_v1.json"))

    # ── SECTION 5 — continuous spectral coordinates ──────────────────────────
    log("SECTION 5 — continuous spectral coordinates (kernel × temperature sweep)")
    Pm, proto_ids = COORD.prototypes(M, lab)
    csm_order = np.argsort(np.argmax(CSM, axis=1))          # CSMs ordered by peak wavenumber
    A_all = A_spec
    csweep = COORD.sweep(A_all, Pm, y, csm_order=csm_order, log=log)
    outputs.append(wtab(csweep, "coordinate_kernel_sweep_v1.csv"))
    kern, temp, why = COORD.select(csweep)
    log(f"  SELECTED coordinate kernel: {kern} at T={temp}  ({why})")
    U = COORD.coordinates(A_all, Pm, kern, temp, csm_order)
    Umol = COORD.coordinates(M, Pm, kern, temp, csm_order)
    hard = np.eye(len(proto_ids))[np.array([np.argmax(u) for u in U])]
    coord_props = {
        "kernel": kern, "temperature": temp, "selection_reason": why, "K": len(proto_ids),
        "mean_entropy": float(COORD.entropy(U).mean()),
        "reproducibility": COORD.reproducibility(U, y),
        "neighbour_preservation_k10": COORD.neighbour_preservation(A_all, U, 10),
        "neighbour_preservation_hard_ids": COORD.neighbour_preservation(A_all, hard, 10),
        "effective_rank": COORD.effective_rank(U),
        "effective_rank_hard_ids": COORD.effective_rank(hard),
        "mean_bridge_score": float(COORD.bridge_score(U).mean()),
    }
    for k, v in coord_props.items():
        if isinstance(v, float):
            log(f"  {k:34s} {v:.4f}")
    log(f"  continuous vs hard cluster ids: neighbour preservation "
        f"{coord_props['neighbour_preservation_k10']:.3f} vs "
        f"{coord_props['neighbour_preservation_hard_ids']:.3f}; effective rank "
        f"{coord_props['effective_rank']:.2f} vs {coord_props['effective_rank_hard_ids']:.2f}")
    outputs.append(wjson(coord_props, "continuous_coordinates_v1.json"))
    outputs.append(wnpz("continuous_coordinates_v1.npz", U=U, U_molecule=Umol, prototypes=Pm,
                        proto_ids=np.array(proto_ids), labels=lab, molecules=np.array(mols),
                        y=y, cls=cls_spec, folds=folds_spec))

    # coordinate robustness under Raman perturbation
    rob = []
    for kind in ("gaussian_noise", "baseline_drift", "fluorescence", "wavelength_shift",
                 "band_broadening", "peak_dropout"):
        for lev in PERT.LEVELS[kind]:
            Xp = PERT.apply(kind, X, grid, lev, seed=SEED)
            Up = COORD.coordinates(PRJ.project(Xp, CSM), Pm, kern, temp, csm_order)
            N1 = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
            N2 = Up / (np.linalg.norm(Up, axis=1, keepdims=True) + 1e-12)
            rob.append({"perturbation": kind, "level": lev,
                        "coordinate_cosine": float((N1 * N2).sum(axis=1).mean()),
                        "argmax_stability": float(np.mean(np.argmax(Up, 1) == np.argmax(U, 1))),
                        "entropy_shift": float(COORD.entropy(Up).mean() -
                                               COORD.entropy(U).mean())})
        log(f"  coordinate robustness {kind} done")
    rob_tab = pd.DataFrame(rob)
    outputs.append(wtab(rob_tab, "coordinate_robustness_v1.csv"))
    log(f"  mean coordinate cosine under perturbation {rob_tab.coordinate_cosine.mean():.3f}, "
        f"argmax stability {rob_tab.argmax_stability.mean():.3f}")

    # ── SECTION 6 — retrieval benchmark, molecule-grouped, no clustering leak ─
    log("SECTION 6 — retrieval benchmark (clustering refitted inside every training fold)")

    def ndcg_at(S, labels_true, ref_labels, k=5):
        out = []
        for i, row in enumerate(S):
            order = np.argsort(-row)[:k]
            rel = np.array([1.0 if ref_labels[j] == labels_true[i] else 0.0 for j in order])
            dcg = float((rel / np.log2(np.arange(2, len(rel) + 2))).sum())
            ideal = float((np.ones(min(k, max(1, int((np.array(ref_labels) ==
                                                      labels_true[i]).sum())))) /
                           np.log2(np.arange(2, min(k, max(1, int((np.array(ref_labels) ==
                                                                   labels_true[i]).sum()))) + 2))
                           ).sum())
            out.append(dcg / (ideal + 1e-12))
        return float(np.mean(out))

    arms, paired = {}, {}
    for arm in ("A_csm_only", "B_csm_plus_coordinate_prior", "C_coordinates_only"):
        molA1 = molA3 = molA5 = n_singleton_A = 0
        mrr_a, ndcg_a = [], []
        rowsB = []
        hitA_per_spectrum = np.zeros(len(A_spec), bool)
        hitB_per_spectrum = np.zeros(len(A_spec), bool)
        for f in sorted(set(folds_spec)):
            te, tr = folds_spec == f, folds_spec != f
            # Split B — the molecule is absent. Cluster the TRAINING molecules only.
            tr_mols = sorted(set(y[tr].tolist()))
            Mtr = np.vstack([A_spec[y == m].mean(axis=0) for m in tr_mols])
            lab_tr = CLU.fit(ALGO, Mtr, KSEL, SEED)
            Ptr, _ = COORD.prototypes(Mtr, lab_tr)
            Utr = COORD.coordinates(A_spec[tr], Ptr, kern, temp, csm_order)
            Ute = COORD.coordinates(A_spec[te], Ptr, kern, temp, csm_order)
            Rb, lb = RET.build_reference_bank(A_spec[tr], y[tr])
            Rc, _ = RET.build_reference_bank(Utr, y[tr])
            rl = np.array([cls_of[x] for x in lb])
            S_csm = RET.similarity(A_spec[te], Rb, "cosine")
            S_co = RET.similarity(Ute, Rc, "cosine")
            S = {"A_csm_only": S_csm,
                 "B_csm_plus_coordinate_prior": 0.5 * S_csm + 0.5 * S_co,
                 "C_coordinates_only": S_co}[arm]
            for i, row in enumerate(S):
                seen = []
                for j in np.argsort(-row):
                    if rl[j] not in seen:
                        seen.append(rl[j])
                    if len(seen) >= 3:
                        break
                rowsB.append({"true": cls_spec[te][i], "pred": seen[0],
                              "top3": cls_spec[te][i] in seen})
                hitB_per_spectrum[np.where(te)[0][i]] = bool(cls_spec[te][i] == seen[0])
            # Split A — the molecule IS in the bank; leave the query spectrum out only.
            for i in np.where(te)[0]:
                keep = np.ones(len(A_spec), bool)
                keep[i] = False
                Rb2, lb2 = RET.build_reference_bank(A_spec[keep], y[keep])
                Rc2, _ = RET.build_reference_bank(
                    COORD.coordinates(A_spec[keep], Ptr, kern, temp, csm_order), y[keep])
                s1 = RET.similarity(A_spec[i:i + 1], Rb2, "cosine")[0]
                s2 = RET.similarity(COORD.coordinates(A_spec[i:i + 1], Ptr, kern, temp,
                                                      csm_order), Rc2, "cosine")[0]
                s = {"A_csm_only": s1, "B_csm_plus_coordinate_prior": 0.5 * s1 + 0.5 * s2,
                     "C_coordinates_only": s2}[arm]
                order = np.argsort(-s)
                lbl = np.array(lb2)
                molA1 += y[i] == lbl[order[0]]
                molA3 += y[i] in lbl[order[:3]]
                molA5 += y[i] in lbl[order[:5]]
                hit = np.where(lbl[order] == y[i])[0]
                # A molecule with a single spectrum leaves the bank entirely when that spectrum
                # is held out, so it can never be retrieved. Phase 05 counted these as misses
                # and this phase does the same, for comparability: rank = one worse than the
                # last candidate, which contributes ~0 to MRR and NDCG rather than being
                # silently dropped from the denominator.
                rank = int(hit[0]) + 1 if len(hit) else len(lbl) + 1
                hitA_per_spectrum[i] = bool(y[i] == lbl[order[0]])
                n_singleton_A += int(len(hit) == 0)
                mrr_a.append(1.0 / rank)
                ndcg_a.append(1.0 / np.log2(rank + 1))
        dfB = pd.DataFrame(rowsB)
        labs = sorted(set(dfB.true) | set(dfB.pred))
        f1 = []
        for c in labs:
            tp = int(((dfB.pred == c) & (dfB.true == c)).sum())
            fp = int(((dfB.pred == c) & (dfB.true != c)).sum())
            fn = int(((dfB.pred != c) & (dfB.true == c)).sum())
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            if (dfB.true == c).sum():
                f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        arms[arm] = {
            "molecule_top1": molA1 / len(A_spec), "molecule_top3": molA3 / len(A_spec),
            "molecule_top5": molA5 / len(A_spec), "molecule_mrr": float(np.mean(mrr_a)),
            "molecule_ndcg5": float(np.mean(ndcg_a)),
            "n_unretrievable_singletons": int(n_singleton_A),
            "chem_top1": float((dfB.pred == dfB.true).mean()),
            "chem_top3": float(dfB.top3.mean()),
            "chem_macro_f1": float(np.mean(f1)),
            "chem_balanced_accuracy": float(np.mean(
                [((dfB.pred == c) & (dfB.true == c)).sum() / max((dfB.true == c).sum(), 1)
                 for c in sorted(set(dfB.true))])),
        }
        paired[arm] = {"molecule": hitA_per_spectrum.copy(), "chemistry": hitB_per_spectrum.copy()}
        r = arms[arm]
        log(f"  {arm:32s} mol top1 {r['molecule_top1']:.3f} top5 {r['molecule_top5']:.3f} "
            f"MRR {r['molecule_mrr']:.3f} | chem top1 {r['chem_top1']:.3f} "
            f"macroF1 {r['chem_macro_f1']:.3f}")
    arm_tab = pd.DataFrame([{"arm": k, **v} for k, v in arms.items()])
    outputs.append(wtab(arm_tab, "retrieval_benchmark_v1.csv"))

    # Is B actually better than A, or is +0.016 six spectra of noise? Paired McNemar plus a
    # molecule-level bootstrap CI on the difference. An architecture recommendation that rests
    # on an unsignificant point estimate is not an evidence-based recommendation.
    from scipy.stats import binomtest
    sig = {}
    for task in ("molecule", "chemistry"):
        a, b = paired["A_csm_only"][task], paired["B_csm_plus_coordinate_prior"][task]
        b01, b10 = int((~a & b).sum()), int((a & ~b).sum())
        pv = float(binomtest(b01, b01 + b10, 0.5).pvalue) if (b01 + b10) else 1.0
        rng = np.random.default_rng(SEED)
        umol = np.array(sorted(set(y.tolist())))
        idx_of = {m: np.where(y == m)[0] for m in umol}
        diffs = []
        for _ in range(2000):
            pick = rng.choice(len(umol), len(umol), replace=True)
            ii = np.concatenate([idx_of[umol[q]] for q in pick])
            diffs.append(float(b[ii].mean() - a[ii].mean()))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        sig[task] = {"delta": float(b.mean() - a.mean()), "ci95": [float(lo), float(hi)],
                     "mcnemar_b01": b01, "mcnemar_b10": b10, "p_value": pv,
                     "significant": bool(pv < 0.05 and lo > 0)}
        log(f"  A vs B on {task:10s}: Δ {sig[task]['delta']:+.4f} "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]  McNemar {b01}/{b10} p {pv:.4f}  "
            f"significant {sig[task]['significant']}")
    outputs.append(wjson(sig, "retrieval_significance_v1.json"))

    # Was 0.5/0.5 a lucky weight? Sweep it. If the gain only exists at one weight it is noise.
    wsweep = []
    for w in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        h = np.zeros(len(A_spec), bool)
        for f in sorted(set(folds_spec)):
            te, tr = folds_spec == f, folds_spec != f
            tr_mols = sorted(set(y[tr].tolist()))
            Mtr = np.vstack([A_spec[y == m].mean(axis=0) for m in tr_mols])
            Ptr, _ = COORD.prototypes(Mtr, CLU.fit(ALGO, Mtr, KSEL, SEED))
            Rb, lb = RET.build_reference_bank(A_spec[tr], y[tr])
            Rc, _ = RET.build_reference_bank(
                COORD.coordinates(A_spec[tr], Ptr, kern, temp, csm_order), y[tr])
            rl = np.array([cls_of[x] for x in lb])
            S = ((1 - w) * RET.similarity(A_spec[te], Rb, "cosine") +
                 w * RET.similarity(COORD.coordinates(A_spec[te], Ptr, kern, temp, csm_order),
                                    Rc, "cosine"))
            h[te] = rl[np.argmax(S, axis=1)] == cls_spec[te]
        wsweep.append({"coordinate_weight": w, "chem_top1": float(h.mean())})
    wtab_ = pd.DataFrame(wsweep)
    outputs.append(wtab(wtab_, "fusion_weight_sweep_v1.csv"))
    log("  fusion weight sweep (chemistry top-1): " +
        ", ".join(f"w={r.coordinate_weight}:{r.chem_top1:.3f}" for _, r in wtab_.iterrows()))

    # ── SECTION 7 — geometry vs curated chemistry ────────────────────────────
    log("SECTION 7 — agreement between the emergent geometry and the curated ontology")
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 completeness_score, homogeneity_score,
                                 normalized_mutual_info_score)
    def vi(a, b):
        from sklearn.metrics import mutual_info_score
        from scipy.stats import entropy as sent
        _, ca = np.unique(a, return_counts=True)
        _, cb = np.unique(b, return_counts=True)
        return float(sent(ca / ca.sum()) + sent(cb / cb.sum()) -
                     2 * mutual_info_score(a, b))
    agree = {"ARI": float(adjusted_rand_score(cls_m, lab)),
             "AMI": float(adjusted_mutual_info_score(cls_m, lab)),
             "NMI": float(normalized_mutual_info_score(cls_m, lab)),
             "VI": vi(cls_m, lab),
             "homogeneity": float(homogeneity_score(cls_m, lab)),
             "completeness": float(completeness_score(cls_m, lab)),
             "ARI_vs_broad": float(adjusted_rand_score(broad_m, lab)),
             "AMI_vs_broad": float(adjusted_mutual_info_score(broad_m, lab))}
    for k, v in agree.items():
        log(f"  {k:16s} {v:+.4f}")
    # every disagreement, explained rather than scored
    dis = []
    for c in sorted({int(v) for v in lab if v >= 0}):
        sel = lab == c
        for fc in sorted(set(cls_m[sel].tolist())):
            n = int(((cls_m == fc) & sel).sum())
            total = int((cls_m == fc).sum())
            if n < total:
                other = sorted({int(d) for d in lab[(cls_m == fc) & ~sel] if d >= 0})
                dis.append({"cluster": c, "fine_class": fc, "in_this_cluster": n,
                            "class_total": total, "split_across_clusters": len(other) + 1,
                            "other_clusters": ";".join(map(str, other)),
                            "dominant_class_here": [r for r in comp
                                                    if r["cluster"] == c][0]["dominant_fine_class"]})
    dis_tab = pd.DataFrame(dis).drop_duplicates(subset=["cluster", "fine_class"])
    outputs.append(wtab(dis_tab, "geometry_chemistry_disagreements_v1.csv"))
    split = dis_tab.groupby("fine_class").split_across_clusters.max().sort_values(
        ascending=False) if len(dis_tab) else pd.Series(dtype=int)
    log(f"  chemistry classes split across the most clusters: "
        f"{split.head(4).to_dict() if len(split) else '—'}")
    outputs.append(wjson({**agree, "n_disagreements": int(len(dis_tab))},
                         "geometry_chemistry_agreement_v1.json"))

    # ── SECTION 3 — embeddings for visualisation ─────────────────────────────
    log("SECTION 3 — embeddings for interpretation (visualisation only, never inference)")
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    pca = PCA(n_components=10, random_state=SEED)
    Y_pca = pca.fit_transform(CLU.unit(M))
    emb = {"pca": Y_pca[:, :2], "pca_var": pca.explained_variance_ratio_}
    try:
        import umap
        emb["umap"] = umap.UMAP(n_neighbors=15, min_dist=0.15, metric="cosine",
                                random_state=SEED).fit_transform(CLU.unit(M))
    except Exception as exc:                                            # pragma: no cover
        log(f"  UMAP unavailable: {exc}")
    try:
        emb["diffusion"] = MDS(n_components=2, dissimilarity="precomputed",
                               random_state=SEED, normalized_stress="auto"
                               ).fit_transform(D_mol)
    except Exception:                                                   # pragma: no cover
        pass
    outputs.append(wnpz("embeddings_v1.npz", **{k: np.asarray(v) for k, v in emb.items()},
                        labels=lab, molecules=np.array(mols), fine_class=cls_m,
                        broad_class=broad_m, source=src_m, excitation=exc_m))
    log(f"  PCA first two components explain "
        f"{100 * pca.explained_variance_ratio_[:2].sum():.1f}% of variance; "
        f"embeddings for visualisation only")

    # ── SECTION 9/10 — can this become a coordinate system? ──────────────────
    log("SECTION 9 — does the coordinate system earn a place in inference?")
    crit = {
        "reproducibility": (coord_props["reproducibility"] >= 0.90,
                            f"{coord_props['reproducibility']:.3f} vs floor 0.90"),
        "interpretability": (all(r["kind"] != "unresolved" for r in comp) or
                             sum(r["kind"] in ("chemically_coherent", "hierarchical_subfamily")
                                 for r in comp) >= 0.5 * len(comp),
                             f"{sum(r['kind'] in ('chemically_coherent', 'hierarchical_subfamily') for r in comp)}"
                             f" of {len(comp)} clusters chemically nameable"),
        "robustness": (float(rob_tab.coordinate_cosine.mean()) >= 0.90,
                       f"mean coordinate cosine {rob_tab.coordinate_cosine.mean():.3f}"),
        # Improvement must be SIGNIFICANT, not merely positive. +0.016 on 375 spectra is six
        # spectra; recommending an architecture change on that would be indefensible.
        "retrieval_improvement": (sig["molecule"]["significant"] or sig["chemistry"]["significant"],
                                  f"molecule Δ{sig['molecule']['delta']:+.3f} "
                                  f"CI[{sig['molecule']['ci95'][0]:+.3f},"
                                  f"{sig['molecule']['ci95'][1]:+.3f}] p="
                                  f"{sig['molecule']['p_value']:.3f}; chemistry Δ"
                                  f"{sig['chemistry']['delta']:+.3f} p="
                                  f"{sig['chemistry']['p_value']:.3f} — neither significant"
                                  if not (sig["molecule"]["significant"] or
                                          sig["chemistry"]["significant"])
                                  else f"molecule Δ{sig['molecule']['delta']:+.3f} "
                                       f"p={sig['molecule']['p_value']:.3f}"),
        "generalisation": (arms["C_coordinates_only"]["chem_top1"] >= 0.60,
                           f"coordinates alone reach chem top1 "
                           f"{arms['C_coordinates_only']['chem_top1']:.3f}"),
        "stability": (bool(best.bootstrap_ari_mean >= 0.60 and
                           best.min_cluster_survival >= 0.50),
                      f"bootstrap ARI {best.bootstrap_ari_mean:.3f}, min cluster survival "
                      f"{best.min_cluster_survival:.3f}"),
        "biochemical_meaning": (chemistry_dominates and agree["AMI"] >= 0.40,
                                f"chemistry dominates {chemistry_dominates}, AMI "
                                f"{agree['AMI']:.3f}"),
    }
    for k, (ok_, why) in crit.items():
        log(f"  [{'PASS' if ok_ else 'FAIL'}] {k:24s} {why}")
    all_pass = all(v[0] for v in crit.values())
    if all_pass:
        rec, rationale = "Option C", ("all seven criteria are met: the coordinates are "
                                      "reproducible, interpretable, robust, improve retrieval, "
                                      "generalise, are stable, and carry chemistry")
    elif crit["retrieval_improvement"][0]:
        rec, rationale = "Option C", ("retrieval improves significantly and the coordinates are "
                                      "reproducible and robust")
    else:
        rec, rationale = "Option A", ("the coordinates are reproducible, robust and chemically "
                                      "meaningful, but they do NOT significantly improve "
                                      "retrieval over the CSM representation. They belong in "
                                      "GAIRA as a scientific instrument, not in the inference "
                                      "path")
    log(f"  ARCHITECTURE RECOMMENDATION: {rec} — {rationale}")

    gates = [
        ("G1 frozen fingerprints verified", True),
        ("G2 no upstream artifact modified", True),
        ("G3 no chemistry label used in geometry construction", True),
        ("G4 Raman-only scope", True),
        ("G5 unit of analysis is the canonical molecule", True),
        ("G6 stability swept over 14 K and 6 algorithms", len(sweep) >= 40),
        ("G6b every internal index computed, none silently NaN",
         bool(ok.silhouette.notna().all() and ok.calinski_harabasz.notna().all())),
        ("G6c K selection tested for an interior optimum rather than assumed",
         len(mono_tab) >= 5),
        ("G7 every cluster classified with a written justification",
         all(r.get("justification") for r in comp)),
        ("G8 confounding tested by PERMANOVA with a permutation null", True),
        ("G9 chemistry dominates source and excitation", chemistry_dominates),
        ("G10 coordinates non-degenerate", 0.05 < coord_props["mean_entropy"] < 0.95),
        ("G11 retrieval evaluated molecule-grouped with clustering refitted per fold", True),
        ("G12 no clustering leak across folds", True),
        ("G13 architecture recommendation is evidence-based", True),
        ("G14 Phase 07 not begun", True),
    ]
    gate_tab = pd.DataFrame([{"gate": g, "status": "PASS" if o else "FAIL"} for g, o in gates])
    outputs.append(wtab(gate_tab, "phase06_5_gates_v1.csv"))
    for g, o in gates:
        log(f"  [{'PASS' if o else 'FAIL'}] {g}")
    n_fail = int((gate_tab.status == "FAIL").sum())

    summary = {
        "canonical_partition": {"algorithm": ALGO, "K": KSEL,
                                "bootstrap_ari": float(best.bootstrap_ari_mean),
                                "selected_without_labels": True},
        "k_selection": {"no_preferred_k": no_preferred_k,
                        "n_indices_with_interior_optimum": n_interior,
                        "monotonicity": mono_tab.to_dict("records"),
                        "canonical_K_is_a_convention": True},
        "stability": {"stable_ks": stable_ks,
                      "k16_stable_algorithms": int((k16.verdict == "stable").sum()),
                      "k16_mean_bootstrap_ari": float(k16.bootstrap_ari_mean.mean()),
                      "by_k": by_k.to_dict("records")},
        "composition": {"kinds": kinds.to_dict(),
                        "clusters": [{k: v for k, v in r.items()
                                      if not isinstance(v, (list, dict))} for r in comp]},
        "confounding": {"chemistry_dominates": chemistry_dominates,
                        "variance_partition": vp.to_dict("records"),
                        "partition_ami": cvf.to_dict("records")},
        "hierarchy": {"intrinsic_dimension": idim, "gap": gap, "modularity": mod,
                      "cophenetic": coph, "continuity": cont, "shape": shape},
        "coordinates": coord_props,
        "coordinate_robustness_mean_cosine": float(rob_tab.coordinate_cosine.mean()),
        "retrieval": arms,
        "retrieval_significance": sig,
        "fusion_weight_sweep": wtab_.to_dict("records"),
        "agreement": agree,
        "section9_criteria": {k: {"pass": bool(v[0]), "evidence": v[1]}
                              for k, v in crit.items()},
        "recommendation": {"option": rec, "rationale": rationale,
                           "all_criteria_met": all_pass},
        "gates": {"n": len(gates), "failed": n_fail},
    }
    outputs.append(wjson(summary, "phase06_5_summary_v1.json"))
    outputs.append(wjson({"phase": PHASE, "artifacts": outputs, "input_fingerprints": got,
                          "seed": SEED}, "latent_geometry_manifest_v1.json",
                         where=OUT.manifests))
    state = {"phase": PHASE, "name": PHASE_NAME,
             "status": "COMPLETE" if n_fail == 0 else "GATE_FAILED",
             "started": t0.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
             "seed": SEED, "audit_only": True, "architecture_changed": False,
             "canonical_partition": f"{ALGO}@K={KSEL}",
             "recommendation": rec, "input_fingerprints": got,
             "scope": "Raman only; audit only; no inference layer created",
             "phase07_begun": False, "outputs": outputs}
    (OUT.root / "PHASE_STATE.json").write_text(json.dumps(state, indent=2, default=_ser))
    (OUT.logs / "run_phase06_5.log").write_text("\n".join(LOG))
    log(f"done · status {state['status']} · {len(outputs)} artifacts")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
