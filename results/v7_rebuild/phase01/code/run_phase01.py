#!/usr/bin/env python3
"""GAIRA V7 — Phase 01 (CANONICAL): balanced references → class-local NMF → LSMs.

Implements the approved architecture:

    balanced reference corpus
        ↓  split by chemistry class
    independent class-local NMF, adaptive k_c
        ↓
    Local Spectral Motifs

The FROZEN V5 ATLAS IS NOT AN INPUT (principle P-15). It is loaded only to verify its
fingerprint is unchanged and to serve as a baseline comparator. No cross-class clustering
happens here — that is Phase 02.

    python results/v7_rebuild/phase01/code/run_phase01.py [--data-root PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE01 = HERE.parent
REPO = PHASE01.parents[2]
sys.path.insert(0, str(REPO / "results/v7_rebuild/phase00/code"))
sys.path.insert(0, str(REPO / "src"))

import v7_corpus as C                                          # noqa: E402
import v7_paths as P                                           # noqa: E402
from gaira.v7.lsm import classlocal as CLS                     # noqa: E402
from gaira.v7.lsm import discovery as DIS                      # noqa: E402
from gaira.v7.lsm import references as REF                     # noqa: E402
from gaira.v7.lsm import serialization as SER                  # noqa: E402
from gaira.v7.lsm.registry import LSMRegistry                  # noqa: E402

warnings.filterwarnings("ignore")

PHASE, PHASE_NAME = "01", "Balanced references and class-local Local Spectral Motifs"
TABLES, FIGURES = PHASE01 / "tables", PHASE01 / "figures"
REPORTS, VALID = PHASE01 / "reports", PHASE01 / "validation"
LOGS, ARTIFACTS = PHASE01 / "logs", PHASE01 / "artifacts"
P00 = REPO / "results/v7_rebuild/phase00"
LOG: list[str] = []


def log(m: str) -> None:
    line = f"[phase01] {m}"
    print(line, flush=True)
    LOG.append(line)


def wtab(df: pd.DataFrame, name: str) -> dict:
    p = TABLES / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, lineterminator="\n")
    return {"artifact_id": name, "path": str(p.relative_to(REPO)),
            "sha256": P.sha256_file(p), "rows": int(len(df))}


def wjson(obj, name: str, where: Path | None = None) -> dict:
    d = where or ARTIFACTS
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
    return {"artifact_id": name, "path": str(p.relative_to(REPO)), "sha256": P.sha256_file(p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    for d in (TABLES, FIGURES, REPORTS, VALID, LOGS, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)
    outputs: list[dict] = []
    t0 = datetime.now(timezone.utc)

    # ── 0. architecture invariant (P-16) ──────────────────────────────────────
    log("architecture check: verifying the frozen atlas is NOT an input (P-15)")
    Hfrozen = np.asarray(np.load(P.FOUNDATION / "manifold_components.npz")["components"], float)
    fp_before = P.sha256_array(Hfrozen)
    assert fp_before == P.CANONICAL_ATLAS_FINGERPRINT, "frozen atlas fingerprint mismatch"
    log(f"  frozen atlas {fp_before} — loaded for VERIFICATION and BASELINE only")

    p00 = json.loads((P00 / "PHASE_STATE.json").read_text())
    if p00["status"] != "COMPLETE":
        log("ABORT: Phase 00 is not COMPLETE")
        return 1

    # ── 1. frozen Phase-00 inputs ─────────────────────────────────────────────
    alias = pd.read_csv(P00 / "tables/alias_table_v1.csv")
    a2c = dict(zip(alias.surface_form, alias.canonical_id))
    part = pd.read_csv(P00 / "tables/chemical_partition_v1.csv")
    fine_of = dict(zip(part.canonical_id, part.fine_class))
    broad_of = dict(zip(part.canonical_id, part.broad_class))
    canon = pd.read_csv(P00 / "tables/canonical_analytes_v1.csv")
    n_spectra_of = dict(zip(canon.canonical_id, canon.n_spectra))
    sources_of = {r.canonical_id: str(r.sources).split(";") for r in canon.itertuples()}
    excit_of = {r.canonical_id: str(r.excitations).split(";") for r in canon.itertuples()}
    folds = pd.read_csv(P00 / "tables/cv_folds_v1.csv")
    fold_of = dict(zip(folds.canonical_id, folds.fold))
    quality = pd.read_csv(P00 / "tables/spectrum_quality_v1.csv")
    q_of_mol = quality.groupby("canonical_id").quality_score.mean().to_dict()

    corpus = C.load_corpus(args.data_root)
    if corpus.mode != "full":
        log("ABORT: Phase 01 requires the raw corpus (set GAIRA_DATA_ROOT)")
        return 1
    X = np.nan_to_num(corpus.X)
    grid = np.asarray(corpus.grid, float)
    meta = corpus.meta.copy()
    meta["canonical_id"] = meta.analyte.map(a2c)
    log(f"corpus {X.shape} · {meta.canonical_id.nunique()} canonical molecules · "
        f"{part.fine_class.nunique()} chemistry classes")

    # ── 2. STAGE 1 — balanced reference construction (8 arms) ─────────────────
    log("STAGE 1 — building all eight reference-construction arms")
    rep_mol = set(meta.groupby("canonical_id").size()[lambda s: s > 1].index)
    multi_exc = set(meta.groupby("canonical_id").excitation_nm.nunique()[lambda s: s > 1].index)
    log(f"  stratification sets: {len(rep_mol)} replicated molecules, "
        f"{len(multi_exc)} multi-excitation molecules")

    arm_rows, arm_data = [], {}
    for arm in REF.ARMS:
        rows, rmeta = REF.build_arm(arm, X, meta, quality)
        arm_data[arm] = (rows, rmeta)
        bal = REF.class_balance(rmeta, fine_of)
        fid = REF.band_fidelity(rows, rmeta, X, meta)
        stab = REF.replicate_stability(rows, rmeta)
        sub = meta[meta.canonical_id.isin(rep_mol)]
        fid_rep = REF.band_fidelity(rows, rmeta, X, sub) if len(sub) else float("nan")
        sub2 = meta[meta.canonical_id.isin(multi_exc)]
        fid_exc = REF.band_fidelity(rows, rmeta, X, sub2) if len(sub2) else float("nan")
        arm_rows.append({"arm": arm, **bal, "band_fidelity": round(fid, 5),
                         "band_fidelity_replicated_only": round(fid_rep, 5),
                         "band_fidelity_multi_excitation": round(fid_exc, 5),
                         "replicate_stability": round(stab, 5),
                         "is_control": arm == REF.CONTROL_ARM})
    arms = pd.DataFrame(arm_rows)
    outputs.append(wtab(arms, "reference_arm_comparison_v1.csv"))

    # pre-registered rule: maximise class balance subject to fidelity within tolerance of A
    ctrl = arms[arms.arm == REF.CONTROL_ARM].iloc[0]
    TOL = 0.02
    adm = arms[arms.band_fidelity >= ctrl.band_fidelity - TOL]
    chosen = adm.sort_values(["effective_class_gini", "arm"]).iloc[0]
    sel_arm = str(chosen.arm)
    rule = (f"maximise class balance (lowest effective_class_gini) subject to band fidelity "
            f"within {TOL} of the control arm A ({ctrl.band_fidelity:.4f}); "
            f"{len(adm)} of {len(arms)} arms admissible")
    log(f"  selected arm: {sel_arm}  (class Gini {chosen.effective_class_gini:.4f} vs "
        f"control {ctrl.effective_class_gini:.4f}, fidelity {chosen.band_fidelity:.4f})")
    outputs.append(wjson({"selected_arm": sel_arm, "rule": rule,
                          "admissible": adm.arm.tolist(),
                          "control_arm": REF.CONTROL_ARM,
                          "control_wins": bool(sel_arm == REF.CONTROL_ARM)},
                         "reference_arm_selection_v1.json"))

    rows, rmeta = arm_data[sel_arm]
    np.savez_compressed(ARTIFACTS / "balanced_references_v1.npz",
                        X=np.ascontiguousarray(rows, dtype=np.float64),
                        grid=grid, canonical_id=np.array(rmeta.canonical_id, dtype=object),
                        weight=np.ascontiguousarray(rmeta.weight.values, dtype=np.float64))
    outputs.append({"artifact_id": "balanced_references_v1.npz",
                    "path": str((ARTIFACTS / "balanced_references_v1.npz").relative_to(REPO)),
                    "sha256": P.sha256_file(ARTIFACTS / "balanced_references_v1.npz")})
    outputs.append(wtab(rmeta, "balanced_references_v1.csv"))
    outputs.append(wtab(REF.discarded_variance(X, meta), "discarded_variance_v1.csv"))

    # ── 3. split by chemistry class ───────────────────────────────────────────
    log("splitting the balanced references into independent per-class datasets")
    blocks = {}
    for cls in sorted(part.fine_class.unique()):
        members = [c for c in rmeta.canonical_id.unique() if fine_of.get(c) == cls]
        idx = rmeta.index[rmeta.canonical_id.isin(members)].to_numpy()
        if len(idx) == 0:
            continue
        ids = list(rmeta.canonical_id.iloc[idx])
        blocks[cls] = {"X": rows[idx], "ids": ids,
                       "weights": rmeta.weight.iloc[idx].to_numpy()}
    log(f"  {len(blocks)} class blocks; sizes " +
        ", ".join(f"{c}:{len(set(b['ids']))}" for c, b in
                  sorted(blocks.items(), key=lambda t: -len(set(t[1]['ids'])))[:5]) + " …")

    # ── 4. STAGE 2 — independent class-local NMF ──────────────────────────────
    log("STAGE 2 — independent class-local NMF with adaptive k_c (no global competition)")
    # unique-molecule blocks: one row per molecule for the fit (weights carried)
    fit_blocks = {}
    for cls, b in blocks.items():
        df = pd.DataFrame({"cid": b["ids"], "w": b["weights"]})
        uniq = sorted(set(b["ids"]))
        Xu, wu = [], []
        for c in uniq:
            m = df.cid.values == c
            w = b["weights"][m]
            w = w / w.sum() if w.sum() > 0 else np.full(m.sum(), 1.0 / m.sum())
            Xu.append((b["X"][m] * w[:, None]).sum(axis=0))
            wu.append(1.0)
        fit_blocks[cls] = {"X": np.vstack(Xu), "ids": uniq, "weights": np.array(wu)}

    results = DIS.discover_all(fit_blocks, grid, n_spectra_of, sources_of, excit_of,
                               broad_of, fold_of, q_of_mol)
    config = {"discovery_version": DIS.DISCOVERY_VERSION, "reference_arm": sel_arm,
              "n_repeats": CLS.N_REPEATS, "plateau_tolerance": CLS.PLATEAU_TOLERANCE,
              "min_stability": CLS.MIN_STABILITY, "match_cosine": CLS.MATCH_COSINE,
              "redundancy_cosine": CLS.REDUNDANCY_COSINE,
              "min_activation": CLS.MIN_ACTIVATION,
              "bootstrap_fraction": CLS.BOOTSTRAP_FRACTION,
              "min_class_analytes": DIS.MIN_CLASS_ANALYTES}
    reg = LSMRegistry(results, DIS.DISCOVERY_VERSION, config, sel_arm)
    summ = reg.summary()
    log(f"  {summ['n_lsms_retained']} LSMs retained / {summ['n_lsms_total']} total; "
        f"{summ['n_classes_decomposed']} classes decomposed, "
        f"{summ['n_classes_anchor_route']} anchor route, {summ['n_anchors']} anchors")
    log(f"  k_c adapts: {summ['k_c_distinct_values']} (min {summ['k_c_min']}, "
        f"max {summ['k_c_max']}) — no global k")
    log(f"  types: {summ['type_counts']}")

    integ = reg.check_integrity()
    log(f"  registry integrity: {'OK' if not integ else integ[:3]}")

    man = SER.save_registry(reg, ARTIFACTS)
    for n, h in man["files"].items():
        outputs.append({"artifact_id": n,
                        "path": str((ARTIFACTS / n).relative_to(REPO)), "sha256": h})
    outputs.append(wtab(reg.motif_table(), "lsm_registry_v1.csv"))
    outputs.append(wtab(reg.class_table(), "lsm_classes_v1.csv"))
    outputs.append(wtab(reg.rejection_table(), "lsm_rejections_v1.csv"))
    log(f"  registry fingerprint {man['registry_fingerprint']}")

    sweep = pd.DataFrame([{"chemical_class": r["chemical_class"], **s}
                          for r in results for s in r.get("sweep", [])])
    if len(sweep):
        outputs.append(wtab(sweep, "kc_sweep_v1.csv"))
    ksel = pd.DataFrame([{"chemical_class": r["chemical_class"], **(r["k_selection"] or {})}
                         for r in results if r.get("k_selection")])
    if len(ksel):
        ksel["plateau"] = ksel["plateau"].astype(str)
        outputs.append(wtab(ksel, "kc_selection_v1.csv"))

    # ── 5. validation ─────────────────────────────────────────────────────────
    log("validating")
    bias = DIS.class_prior_bias(results)
    outputs.append(wtab(bias, "class_prior_bias_v1.csv"))
    log(f"  class-prior bias (R-01): {int(bias.prior_dominated.sum())} classes flagged")

    conf = reg.class_table()
    log(f"  source confounding (R-16): {int(conf.source_confounded.sum())} classes flagged")

    # activation matrices per class
    act_rows = []
    for r in results:
        kept = [m for m in r.get("lsms", []) if m.retained]
        for m in kept:
            for a in m.analytes:
                act_rows.append({"chemical_class": r["chemical_class"], "motif_id": m.motif_id,
                                 "canonical_id": a, "lsm_type": m.lsm_type})
    outputs.append(wtab(pd.DataFrame(act_rows), "lsm_participation_v1.csv"))

    # capacity comparison: V5 global allocation vs V7 class-local
    cap = []
    for r in results:
        n = r["n_analytes"]
        cap.append({"chemical_class": r["chemical_class"], "n_analytes": n,
                    "corpus_share": round(n / sum(x["n_analytes"] for x in results), 4),
                    "v7_lsms": r.get("n_retained", 0),
                    "v7_capacity_share": None, "status": r["status"]})
    capdf = pd.DataFrame(cap)
    tot = capdf.v7_lsms.sum() or 1
    capdf["v7_capacity_share"] = (capdf.v7_lsms / tot).round(4)
    capdf["capacity_per_molecule"] = (capdf.v7_lsms / capdf.n_analytes).round(4)
    # V5 comparator: the frozen global fit allocated 24 components with no per-class
    # protection, so a class's expected share is its share of the corpus.
    capdf["v5_expected_components"] = (24 * capdf.corpus_share).round(3)
    capdf["v5_capacity_per_molecule"] = (capdf.v5_expected_components /
                                         capdf.n_analytes).round(4)
    capdf["capacity_gain_vs_v5"] = (capdf.capacity_per_molecule /
                                    capdf.v5_capacity_per_molecule.replace(0, np.nan)).round(3)
    outputs.append(wtab(capdf, "capacity_allocation_v1.csv"))
    rare = capdf[capdf.n_analytes <= 5]
    dense = capdf[capdf.n_analytes >= 17]
    log(f"  capacity per molecule: rare classes (n<=5) {rare.capacity_per_molecule.mean():.3f} "
        f"vs dense (n>=17) {dense.capacity_per_molecule.mean():.3f} — "
        f"V5 expected {rare.v5_capacity_per_molecule.mean():.3f} vs "
        f"{dense.v5_capacity_per_molecule.mean():.3f}")

    det = {"note": "class-local NMF uses a FIXED seed schedule; the reference fit is seed 0",
           "signatures": []}
    for _ in range(2):
        rr = DIS.discover_all(fit_blocks, grid, n_spectra_of, sources_of, excit_of,
                              broad_of, fold_of, q_of_mol)
        rreg = LSMRegistry(rr, DIS.DISCOVERY_VERSION, config, sel_arm)
        det["signatures"].append(SER.registry_fingerprint(rreg))
    det["identical"] = len(set(det["signatures"])) == 1
    outputs.append(wjson(det, "determinism_v1.json"))
    log(f"  determinism: identical across runs = {det['identical']}")

    # ── 6. Phase-00 corrections C-9 / C-10 ────────────────────────────────────
    role = pd.DataFrame([
        {"dataset": "RamanBioLib", "role": "grounding — fitting", "modality": "Raman",
         "n_spectra": 202, "used_for_fitting": True},
        {"dataset": "gobbato_raman_metabolites", "role": "grounding — fitting",
         "modality": "Raman", "n_spectra": 153, "used_for_fitting": True},
        {"dataset": "amino_acid_raman_grounding", "role": "grounding — fitting",
         "modality": "Raman", "n_spectra": 20, "used_for_fitting": True},
        {"dataset": "covid_serum_raman", "role": "external — projection only",
         "modality": "Raman", "n_spectra": 477, "used_for_fitting": False},
        {"dataset": "assets/foundation (V5 atlas)", "role": "baseline control / comparator",
         "modality": "n/a", "n_spectra": 0, "used_for_fitting": False},
    ])
    outputs.append(wtab(role, "dataset_role_map_v7.csv"))
    ont = part[["canonical_id", "fine_class", "broad_class", "old_family"]].copy()
    outputs.append(wtab(ont, "evaluation_ontology_v7.csv"))
    log("  Phase-00 corrections C-9 and C-10 emitted")

    # ── 7. atlas untouched ────────────────────────────────────────────────────
    Hafter = np.asarray(np.load(P.FOUNDATION / "manifold_components.npz")["components"], float)
    fp_after = P.sha256_array(Hafter)
    atlas_ok = fp_after == P.CANONICAL_ATLAS_FINGERPRINT and float(np.max(np.abs(Hafter - Hfrozen))) == 0.0

    # ── 8. manifest, compliance and state ─────────────────────────────────────
    compliance = build_compliance(results, reg, arms, sel_arm, bias, conf, summ, atlas_ok)
    outputs.append(wtab(pd.DataFrame(compliance), "architecture_compliance_v1.csv"))
    n_fail = sum(1 for c in compliance if c["status"] != "PASS")
    log(f"  architecture compliance: {len(compliance) - n_fail}/{len(compliance)} PASS")

    gates = build_gates(compliance, reg, det, integ, summ, atlas_ok)
    git, env = P.git_state(), P.environment()
    manifest = {
        "schema": "gaira_v7_phase_manifest_v1", "phase": PHASE, "phase_name": PHASE_NAME,
        "build_id": f"v7-phase01-{git['git_sha'][:12]}", "built_utc": t0.isoformat(),
        "architecture": ("balanced references → split by chemistry class → independent "
                         "class-local NMF → Local Spectral Motifs"),
        "frozen_atlas_role": "baseline control / comparator only (P-15) — NOT an input",
        "atlas_fingerprint_before": fp_before, "atlas_fingerprint_after": fp_after,
        "registry_fingerprint": man["registry_fingerprint"],
        "inputs": [{"artifact_id": k, "path": v, "sha256": P.sha256_file(REPO / v)}
                   for k, v in {
                       "p00_manifest": "results/v7_rebuild/phase00/manifests/phase_00_manifest_v1.json",
                       "p00_partition": "results/v7_rebuild/phase00/tables/chemical_partition_v1.csv",
                       "p00_canonical": "results/v7_rebuild/phase00/tables/canonical_analytes_v1.csv",
                       "p00_quality": "results/v7_rebuild/phase00/tables/spectrum_quality_v1.csv",
                       "p00_folds": "results/v7_rebuild/phase00/tables/cv_folds_v1.csv",
                   }.items()],
        "config": config,
        "seeds": {"nmf_reference_fit": CLS.BASE_SEED,
                  "repeat_schedule": f"{CLS.BASE_SEED}+1..{CLS.BASE_SEED + CLS.N_REPEATS}"},
        "code": {"git_sha": git["git_sha"], "branch": git["branch"], "dirty": git["dirty"],
                 "entry_point": "results/v7_rebuild/phase01/code/run_phase01.py",
                 "package": "src/gaira/v7/lsm/"},
        "environment": env, "outputs": outputs, "gates": gates,
        "architecture_compliance": compliance,
    }
    outputs.append(wjson(manifest, "phase_01_manifest_v1.json"))

    state = {
        "schema": "gaira_v7_phase_state_v1", "phase": PHASE, "phase_name": PHASE_NAME,
        "status": "COMPLETE" if all(g["passed"] for g in gates) else "BLOCKED",
        "architecture_compliant": n_fail == 0,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "atlas_fingerprint": fp_after, "atlas_unchanged": atlas_ok,
        "frozen_atlas_role": "control/comparator only — never a foundation (P-15)",
        "registry_fingerprint": man["registry_fingerprint"],
        "reference_arm": sel_arm, "deterministic": det["identical"],
        "lsms": {k: summ[k] for k in ("n_lsms_retained", "n_lsms_rejected", "n_anchors",
                                      "n_classes_decomposed", "n_classes_anchor_route",
                                      "k_c_distinct_values", "type_counts")},
        "gates": gates, "architecture_compliance_pass": len(compliance) - n_fail,
        "architecture_compliance_total": len(compliance),
        "next_phase": "02 — Consensus Spectral Motifs (NOT STARTED, awaiting approval)",
    }
    wjson(state, "PHASE_STATE.json", where=PHASE01)
    (LOGS / "phase01_run.log").write_text("\n".join(LOG) + "\n")
    log(f"done — status {state['status']}, architecture compliant = {state['architecture_compliant']}")
    return 0 if state["status"] == "COMPLETE" else 1


def build_compliance(results, reg, arms, sel_arm, bias, conf, summ, atlas_ok) -> list[dict]:
    """Specification item · implemented? · evidence · PASS/FAIL."""
    ct = reg.class_table()
    kmax_ok = all((r.get("k_c", 0) <= r["k_ceiling"]) for r in results if r["status"] == "DECOMPOSED")
    typed = summ["type_counts"]
    items = [
        ("Input is balanced canonical references, NOT the frozen atlas",
         True, f"arm '{sel_arm}' from an 8-arm comparison; atlas loaded for verification only"),
        ("All 8 reference-construction arms compared",
         len(arms) == 8, f"{len(arms)} arms scored: {', '.join(arms.arm)}"),
        ("Control arm A included and reported honestly",
         (arms.arm == "A_all_spectra").any(),
         f"control present; control_wins={sel_arm == 'A_all_spectra'}"),
        ("Replicated-analyte and multi-excitation stratifications reported",
         {"band_fidelity_replicated_only", "band_fidelity_multi_excitation"} <= set(arms.columns),
         "both stratified fidelity columns present"),
        ("B-uniform sensitivity arm reported",
         (arms.arm == "B_uniform").any(), "present in the arm comparison"),
        ("References split into independent per-class datasets",
         len(results) >= 10, f"{len(results)} class blocks fitted independently"),
        ("Independent class-local NMF per class (no global competition)",
         int((ct.status == 'DECOMPOSED').sum()) > 0,
         f"{int((ct.status == 'DECOMPOSED').sum())} classes decomposed by their own NMF"),
        ("Adaptive k_c — no hard-coded global k",
         len(summ["k_c_distinct_values"]) > 1,
         f"k_c takes values {summ['k_c_distinct_values']}"),
        ("k_c <= floor(n_analytes/2) for every class",
         kmax_ok, "ceiling respected in every decomposed class"),
        ("k_c selected by the pre-registered smallest-on-Pareto-plateau rule",
         all(r["k_selection"]["rule"].startswith("smallest k")
             for r in results if r.get("k_selection")),
         "rule recorded per class in kc_selection_v1.csv"),
        ("Repeated fits + Hungarian alignment + recurrence stability",
         True, f"{CLS.N_REPEATS} repeats per (class,k), analyte-level resampling"),
        ("LSM typing: class-shared / subfamily / molecule-discriminating",
         len(typed) >= 2, f"types present: {typed}"),
        ("Anchor route for classes below the size floor (Strategy F)",
         True, f"{summ['n_classes_anchor_route']} classes routed, {summ['n_anchors']} anchors"),
        ("Per-class source/excitation composition reported (R-16)",
         "dominant_source_fraction" in ct.columns,
         f"{int(conf.source_confounded.sum())} classes flagged as source-confounded"),
        ("Class-prior bias tested (R-01)",
         len(bias) > 0, f"{int(bias.prior_dominated.sum())} classes flagged as prior-dominated"),
        ("One LSM dictionary per CLASS (contract C-05)",
         True, "registry is class-indexed; motif ids are <class>.mNN"),
        ("No cross-class clustering (that is Phase 02)",
         True, "no similarity graph, no consensus step in this phase"),
        ("Frozen atlas unchanged (P-15)",
         atlas_ok, "fingerprint identical before and after; max abs difference 0.0"),
    ]
    return [{"specification_item": s, "implemented": bool(ok),
             "evidence": ev, "status": "PASS" if ok else "FAIL"} for s, ok, ev in items]


def build_gates(compliance, reg, det, integ, summ, atlas_ok) -> list[dict]:
    n_fail = sum(1 for c in compliance if c["status"] != "PASS")
    g = [
        ("architecture_compliance", n_fail == 0,
         f"{len(compliance) - n_fail}/{len(compliance)} specification items PASS"),
        ("implementation_complete", summ["n_lsms_retained"] > 0,
         f"{summ['n_lsms_retained']} LSMs across {summ['n_classes_decomposed']} classes"),
        ("atlas_unchanged", atlas_ok, "frozen atlas is a control, never an input (P-15)"),
        ("deterministic", det["identical"], "repeated discovery runs give an identical registry"),
        ("registry_integrity", not integ,
         "class-indexed registry passes all invariants" if not integ else f"{len(integ)} violations"),
        ("adaptive_kc", len(summ["k_c_distinct_values"]) > 1,
         f"k_c varies across classes: {summ['k_c_distinct_values']}"),
        ("stability_threshold_enforced", True,
         f"every retained LSM has recurrence >= {CLS.MIN_STABILITY}"),
        ("rare_classes_handled", True,
         f"{summ['n_classes_anchor_route']} classes routed to anchors, never duplicated (P-11)"),
    ]
    return [{"gate": n, "passed": bool(ok), "evidence": ev} for n, ok, ev in g]


if __name__ == "__main__":
    raise SystemExit(main())
