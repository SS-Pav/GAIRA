#!/usr/bin/env python3
"""GAIRA V7 — Phase 01 orchestrator: Local Spectral Motif discovery (Strategy A).

Deterministic end to end. Reads the FROZEN atlas and the FROZEN Phase 00 outputs; writes
only under results/v7_rebuild/phase01/. Never touches assets/, results/v5_rebuild/,
results/v6_rebuild/ or results/v7_rebuild/phase00/.

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
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
PHASE01 = HERE.parent
REPO = PHASE01.parent.parent.parent
sys.path.insert(0, str(REPO / "results/v7_rebuild/phase00/code"))
sys.path.insert(0, str(REPO / "src"))

import v7_corpus as C                                       # noqa: E402
import v7_paths as P                                        # noqa: E402
from gaira.v7.lsm import clustering as CL                   # noqa: E402
from gaira.v7.lsm import discovery as DIS                   # noqa: E402
from gaira.v7.lsm import matching as MATCH                  # noqa: E402
from gaira.v7.lsm import serialization as SER               # noqa: E402
from gaira.v7.lsm import validation as VAL                  # noqa: E402
from gaira.v7.lsm.registry import LSMRegistry               # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PHASE, PHASE_NAME = "01", "Local Spectral Motif discovery (Strategy A)"
TABLES, FIGURES = PHASE01 / "tables", PHASE01 / "figures"
REPORTS, VALID = PHASE01 / "reports", PHASE01 / "validation"
LOGS, ARTIFACTS = PHASE01 / "logs", PHASE01 / "artifacts"
P00 = REPO / "results/v7_rebuild/phase00"
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[phase01] {msg}"
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
    return {"artifact_id": name, "path": str(p.relative_to(REPO)),
            "sha256": P.sha256_file(p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    for d in (TABLES, FIGURES, REPORTS, VALID, LOGS, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)
    outputs: list[dict] = []
    t0 = datetime.now(timezone.utc)

    # ── 1. frozen inputs ──────────────────────────────────────────────────────
    log("loading the frozen atlas and the frozen Phase 00 outputs")
    z = np.load(P.FOUNDATION / "manifold_components.npz")
    H = np.asarray(z["components"], float)
    grid = np.asarray(z["grid"], float)
    fp_before = P.sha256_array(H)
    if fp_before != P.CANONICAL_ATLAS_FINGERPRINT:
        log(f"ABORT: atlas fingerprint mismatch {fp_before}")
        return 1
    log(f"atlas verified: {H.shape} fingerprint {fp_before}")

    p00_state = json.loads((P00 / "PHASE_STATE.json").read_text())
    if p00_state["status"] != "COMPLETE":
        log("ABORT: Phase 00 is not COMPLETE")
        return 1

    alias = pd.read_csv(P00 / "tables/alias_table_v1.csv")
    a2c = dict(zip(alias.surface_form, alias.canonical_id))
    part = pd.read_csv(P00 / "tables/chemical_partition_v1.csv")
    fine_of = dict(zip(part.canonical_id, part.fine_class))
    broad_of = dict(zip(part.canonical_id, part.broad_class))
    canon = pd.read_csv(P00 / "tables/canonical_analytes_v1.csv")
    spectra_of = dict(zip(canon.canonical_id, canon.n_spectra))
    sources_of = {r.canonical_id: str(r.sources).split(";") for r in canon.itertuples()}

    # ── 2. corpus and frozen projection ───────────────────────────────────────
    corpus = C.load_corpus(args.data_root)
    if corpus.mode != "full":
        log("ABORT: Phase 01 requires the raw corpus (set GAIRA_DATA_ROOT)")
        return 1
    X = np.nan_to_num(corpus.X)
    cid = np.array([a2c[a] for a in corpus.meta.analyte])
    log(f"corpus {X.shape} · {len(set(cid))} canonical molecules")

    log("projecting onto the FROZEN atlas (NNLS — the atlas is not refitted)")
    W = np.vstack([nnls(H.T, X[i])[0] for i in range(X.shape[0])])

    ids = sorted(set(cid))
    Xa = np.vstack([X[cid == i].mean(0) for i in ids])
    Wa = np.vstack([W[cid == i].mean(0) for i in ids])
    src_mask = {s: np.array([s in sources_of.get(i, []) for i in ids])
                for s in ("RamanBioLib", "gobbato_raman_metabolites")}

    # ── 3. pre-registered method comparisons ──────────────────────────────────
    log("running the pre-registered profile-mode and linkage comparisons")
    prof_rows = []
    for mode in ("raw", "attribution"):
        res = DIS.discover_all(H, grid, Xa, Wa, ids, fine_of, broad_of, sources_of,
                               spectra_of, profile_mode=mode)
        al = VAL.chemical_alignment(res, fine_of, broad_of, n_perm=200)
        ok = al[al.ami_fine.notna()]
        prof_rows.append({
            "profile_mode": mode,
            "n_components_decomposed": int((al.status == "DECOMPOSED").sum()),
            "n_significant": int(al.significant.sum()),
            "mean_ami_fine": round(float(ok.ami_fine.mean()), 4) if len(ok) else None,
            "median_ami_fine": round(float(ok.ami_fine.median()), 4) if len(ok) else None,
        })
    outputs.append(wtab(pd.DataFrame(prof_rows), "profile_mode_comparison_v1.csv"))
    log("  profile modes: " + " | ".join(
        f"{r['profile_mode']} mean AMI {r['mean_ami_fine']}" for r in prof_rows))

    log("  comparing motif-spectrum constructions")
    from gaira.v7.lsm import motif as MO
    cons_rows = []
    for cname, fn in (("discriminative (selected)", MO.build_motif_spectrum),
                      ("representative", MO.build_motif_spectrum_representative)):
        DIS.build_motif_spectrum = fn
        res = DIS.discover_all(H, grid, Xa, Wa, ids, fine_of, broad_of, sources_of,
                               spectra_of)
        rr = LSMRegistry(res, fp_before, DIS.DISCOVERY_VERSION, {})
        rsum = VAL.redundancy_summary(rr.retained)
        aa = VAL.chemical_alignment(res, fine_of, broad_of, n_perm=150)
        cons_rows.append({"construction": cname, "n_motifs": rsum["n_motifs"],
                          "max_offdiag_cosine": rsum["max_offdiag_cosine"],
                          "mean_offdiag_cosine": rsum["mean_offdiag_cosine"],
                          "n_pairs_above_0.9": rsum["n_pairs_above_0.9"],
                          "mean_ami_fine": round(float(
                              aa[aa.ami_fine.notna()].ami_fine.mean()), 4)})
    DIS.build_motif_spectrum = MO.build_motif_spectrum
    outputs.append(wtab(pd.DataFrame(cons_rows), "motif_construction_comparison_v1.csv"))
    log("  constructions: " + " | ".join(
        f"{r['construction']} maxcos {r['max_offdiag_cosine']} "
        f"pairs>0.9 {r['n_pairs_above_0.9']}" for r in cons_rows))

    profiles = DIS.collect_profiles(H, grid, Xa, Wa, profile_mode=DIS.PROFILE_MODE)
    link_rows = CL.compare_linkages(profiles)
    rule = CL.apply_linkage_rule(link_rows)
    outputs.append(wtab(pd.DataFrame(link_rows), "linkage_comparison_v1.csv"))
    outputs.append(wjson(rule, "linkage_selection_v1.json"))
    log(f"  linkage rule → {rule['selected_linkage']} ({rule['rule']})")

    # ── 4. discovery ──────────────────────────────────────────────────────────
    linkage = rule["selected_linkage"]
    kwargs = dict(H=H, grid=grid, Xa=Xa, Wa=Wa, analyte_ids=ids, fine_of=fine_of,
                  broad_of=broad_of, sources_of=sources_of, spectra_of=spectra_of,
                  linkage_method=linkage, profile_mode=DIS.PROFILE_MODE)
    log(f"discovering motifs across {H.shape[0]} atlas components (linkage={linkage})")
    results = DIS.discover_all(**kwargs)

    config = {"discovery_version": DIS.DISCOVERY_VERSION, "linkage": linkage,
              "profile_mode": DIS.PROFILE_MODE,
              "band_prominence": DIS.BAND_PROMINENCE,
              "band_half_width": DIS.BAND_HALF_WIDTH,
              "share_threshold": DIS.SHARE_THRESHOLD,
              "min_participants": DIS.MIN_PARTICIPANTS, "min_bands": DIS.MIN_BANDS,
              "min_motif_analytes": DIS.MIN_MOTIF_ANALYTES,
              "min_stability": DIS.MIN_STABILITY,
              "min_motif_bands": DIS.MIN_MOTIF_BANDS,
              "redundancy_cosine": DIS.REDUNDANCY_COSINE,
              "max_motifs": CL.MAX_MOTIFS}
    reg = LSMRegistry(results, fp_before, DIS.DISCOVERY_VERSION, config)
    summ = reg.summary()
    log(f"  {summ['n_motifs_retained']} retained / {summ['n_motifs_total']} total motifs; "
        f"{summ['n_components_decomposed']} decomposed, "
        f"{summ['n_components_irreducible']} irreducible, "
        f"{summ['n_components_not_analysable']} not analysable")

    integrity = reg.check_integrity()
    log(f"  registry integrity: {'OK' if not integrity else integrity[:3]}")

    man = SER.save_registry(reg, ARTIFACTS)
    for n, h in man["files"].items():
        outputs.append({"artifact_id": n, "path": str((ARTIFACTS / n).relative_to(REPO)),
                        "sha256": h})
    outputs.append({"artifact_id": "lsm_manifest_v1.json",
                    "path": str((ARTIFACTS / "lsm_manifest_v1.json").relative_to(REPO)),
                    "sha256": P.sha256_file(ARTIFACTS / "lsm_manifest_v1.json")})
    outputs.append(wtab(reg.motif_table(), "lsm_registry_v1.csv"))
    outputs.append(wtab(reg.component_table(), "lsm_components_v1.csv"))
    outputs.append(wtab(reg.rejection_table(), "lsm_rejections_v1.csv"))
    log(f"  registry fingerprint {man['registry_fingerprint']}")

    sweep = pd.DataFrame([{"component": r["component"], **s}
                          for r in results for s in r.get("sweep", [])])
    if len(sweep):
        sweep["sizes"] = sweep["sizes"].astype(str)
        outputs.append(wtab(sweep, "cut_selection_sweep_v1.csv"))

    # ── 5. validation ─────────────────────────────────────────────────────────
    log("validating: chemical alignment against a permutation null")
    align = VAL.chemical_alignment(results, fine_of, broad_of)
    outputs.append(wtab(align, "chemical_alignment_v1.csv"))
    sig = int(align.significant.sum())
    log(f"  {sig} components align with chemistry beyond chance (p<0.05)")

    amb = VAL.ambiguity_resolution(results, fine_of)
    outputs.append(wtab(amb, "ambiguity_resolution_v1.csv"))
    gain = amb[amb.purity_gain.notna()]
    log(f"  median purity gain over the whole component: "
        f"{gain.purity_gain.median():.4f} (n={len(gain)})")

    pnull = VAL.purity_null(results, fine_of)
    outputs.append(wtab(pnull, "purity_null_v1.csv"))
    n_pure_sig = int(pnull.significant.sum())
    log(f"  purity beyond a size-matched random partition: median "
        f"{pnull.gain_beyond_mechanical.median():+.4f}; "
        f"{n_pure_sig}/{len(pnull)} components significant")

    kept = reg.retained
    outputs.append(wtab(VAL.redundancy_matrix(kept).reset_index().rename(
        columns={"index": "motif_id"}), "motif_overlap_matrix_v1.csv"))
    red = VAL.redundancy_summary(kept)
    cov = VAL.coverage_report(reg, ids, spectra_of)
    outputs.append(wjson(red, "redundancy_summary_v1.json"))
    outputs.append(wjson(cov, "coverage_report_v1.json"))
    log(f"  redundancy: max off-diagonal cosine {red['max_offdiag_cosine']}; "
        f"coverage: {cov['analyte_coverage']:.1%} of molecules")

    log("validating: determinism (3 independent discovery runs)")
    det = VAL.determinism_check(DIS.discover_all, kwargs, n_runs=3)
    outputs.append(wjson(det, "determinism_v1.json"))
    log(f"  identical across runs: {det['identical']}")

    log("validating: cross-source and replicate reproducibility")
    subsets = {}
    for name, mask in (("RamanBioLib_only", src_mask["RamanBioLib"]),
                       ("gobbato_only", src_mask["gobbato_raman_metabolites"])):
        if mask.sum() < DIS.MIN_PARTICIPANTS:
            continue
        sub = dict(kwargs)
        sub |= {"Xa": Xa[mask], "Wa": Wa[mask],
                "analyte_ids": [ids[i] for i in np.where(mask)[0]]}
        subsets[name] = sub
    # replicate split: for molecules with >1 spectrum, first vs second half
    for half in (0, 1):
        Xh = []
        for i in ids:
            rows = np.where(cid == i)[0]
            pick = rows[half::2] if len(rows) > 1 else rows
            Xh.append(X[pick].mean(0))
        Xh = np.vstack(Xh)
        Wh = np.vstack([nnls(H.T, Xh[j])[0] for j in range(Xh.shape[0])])
        subsets[f"replicate_half_{half}"] = dict(kwargs) | {"Xa": Xh, "Wa": Wh}
    repro = VAL.reproducibility(DIS.discover_all, subsets, results)
    outputs.append(wtab(repro, "reproducibility_v1.csv"))
    for name, g in repro.groupby("subset"):
        v = g.ari.dropna()
        log(f"  {name}: median ARI {v.median():.3f} (n={len(v)})" if len(v)
            else f"  {name}: no comparable components")

    log("validating: attribution conservation on all 375 spectra")
    A, aids = MATCH.attribution_matrix(X, W, reg)
    cons = MATCH.conservation_error(A, W)
    unattr = float(A[:, aids.index(MATCH.UNATTRIBUTED)].sum() / max(A.sum(), 1e-12))
    outputs.append(wjson({"max_conservation_error": cons,
                          "unattributed_fraction": round(unattr, 6),
                          "n_spectra": int(X.shape[0]), "n_motif_axes": len(aids) - 1},
                         "attribution_conservation_v1.json"))
    log(f"  conservation error {cons:.3e}; unattributed evidence {unattr:.1%}")

    # ── 6. atlas untouched ────────────────────────────────────────────────────
    H_after = np.asarray(np.load(P.FOUNDATION / "manifold_components.npz")["components"],
                         float)
    fp_after = P.sha256_array(H_after)
    atlas_ok = (fp_after == P.CANONICAL_ATLAS_FINGERPRINT
                and float(np.max(np.abs(H_after - H))) == 0.0)
    log(f"  atlas after Phase 01: {fp_after} unchanged={atlas_ok}")

    # ── 7. manifest and phase state ───────────────────────────────────────────
    git, env = P.git_state(), P.environment()
    inputs = [{"artifact_id": k, "path": v, "sha256": P.sha256_file(REPO / v)}
              for k, v in {
                  "frozen_basis": "assets/foundation/manifold_components.npz",
                  "frozen_manifest": "assets/foundation/MANIFEST.json",
                  "p00_alias_table": "results/v7_rebuild/phase00/tables/alias_table_v1.csv",
                  "p00_partition": "results/v7_rebuild/phase00/tables/chemical_partition_v1.csv",
                  "p00_canonical": "results/v7_rebuild/phase00/tables/canonical_analytes_v1.csv",
                  "p00_manifest": "results/v7_rebuild/phase00/manifests/phase_00_manifest_v1.json",
              }.items()]

    gates = build_gates(atlas_ok, fp_after, det, integrity, summ, align, amb, cons, repro, pnull)
    manifest = {
        "schema": "gaira_v7_phase_manifest_v1", "phase": PHASE, "phase_name": PHASE_NAME,
        "build_id": f"v7-phase01-{git['git_sha'][:12]}",
        "built_utc": t0.isoformat(),
        "atlas_fingerprint_before": fp_before, "atlas_fingerprint_after": fp_after,
        "registry_fingerprint": man["registry_fingerprint"],
        "inputs": inputs, "config": config,
        "seeds": {"discovery": "none — discovery is RNG-free",
                  "permutation_tests": VAL.SEED},
        "code": {"git_sha": git["git_sha"], "branch": git["branch"], "dirty": git["dirty"],
                 "entry_point": "results/v7_rebuild/phase01/code/run_phase01.py",
                 "package": "src/gaira/v7/lsm/"},
        "environment": env, "outputs": outputs, "gates": gates, "decisions": DECISIONS,
    }
    outputs.append(wjson(manifest, "phase_01_manifest_v1.json"))

    state = {
        "schema": "gaira_v7_phase_state_v1", "phase": PHASE, "phase_name": PHASE_NAME,
        "status": "COMPLETE" if all(g["passed"] for g in gates) else "BLOCKED",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "atlas_fingerprint": fp_after, "atlas_unchanged": atlas_ok,
        "registry_fingerprint": man["registry_fingerprint"],
        "deterministic": det["identical"],
        "motifs": {"retained": summ["n_motifs_retained"], "rejected": summ["n_motifs_rejected"],
                   "decomposed_components": summ["n_components_decomposed"],
                   "irreducible_components": summ["n_components_irreducible"],
                   "not_analysable_components": summ["n_components_not_analysable"]},
        "science": {"components_aligned_with_chemistry_p05": sig,
                    "components_purity_above_size_matched_null": n_pure_sig,
                    "median_purity_gain_raw": (round(float(gain.purity_gain.median()), 4)
                                               if len(gain) else None),
                    "median_purity_gain_beyond_mechanical":
                        round(float(pnull.gain_beyond_mechanical.median()), 4),
                    "analyte_coverage": cov["analyte_coverage"]},
        "gates": gates,
        "next_phase": "02 — awaiting approval (NOT STARTED)",
    }
    wjson(state, "PHASE_STATE.json", where=PHASE01)
    (LOGS / "phase01_run.log").write_text("\n".join(LOG) + "\n")
    log(f"done — status {state['status']}")
    return 0 if state["status"] == "COMPLETE" else 1


def build_gates(atlas_ok, fp_after, det, integrity, summ, align, amb, cons, repro,
                pnull) -> list[dict]:
    gain = amb[amb.purity_gain.notna()]
    beneficial = int((align.significant & (align.n_motifs >= 2)).sum())
    pure_sig = int(pnull.significant.sum())
    ari = repro.ari.dropna()
    g = [
        ("implementation_complete", summ["n_motifs_retained"] > 0,
         f"{summ['n_motifs_retained']} motifs retained across "
         f"{summ['n_components_decomposed']} decomposed components"),
        ("atlas_unchanged", atlas_ok,
         "basis array identical before and after; max abs difference 0.0"),
        ("fingerprint_unchanged", fp_after == P.CANONICAL_ATLAS_FINGERPRINT,
         f"recomputed {fp_after}"),
        ("deterministic", det["identical"],
         f"{det['n_runs']} independent runs produced identical motif spectra"),
        ("registry_generated", not integrity,
         "registry integrity checks pass" if not integrity else f"{len(integrity)} violations"),
        ("projection_conserved", cons < 1e-9,
         f"attributed evidence equals atlas activation (max error {cons:.2e})"),
        ("validation_passed", len(align) > 0 and len(gain) > 0,
         "alignment, ambiguity, redundancy, coverage, reproducibility all measured"),
        ("scientific_benefit_demonstrated", beneficial >= 1 and pure_sig >= 1,
         f"{beneficial} components align with chemistry beyond a permutation null (p<0.05); "
         f"{pure_sig} exceed a SIZE-MATCHED random partition on purity, so the gain is not "
         f"the mechanical effect of cutting a set into more pieces"),
        ("reproducibility_measured", len(ari) > 0,
         f"cross-source and replicate agreement measured on {len(ari)} component-subset pairs"),
    ]
    return [{"gate": n, "passed": bool(ok), "evidence": ev} for n, ok, ev in g]


DECISIONS = [
    {"decision": "LSMs decompose the FROZEN atlas components, not balanced references",
     "rule_preregistered_in": "user Phase 01 brief (Strategy A)",
     "chosen": "band-profile clustering of the analytes activating each frozen component",
     "alternatives": ["class-local NMF over balanced references "
                      "(the architecture documents' Phase-02 definition)"],
     "rationale": "The brief scopes Phase 01 to a learning-free interpretation layer that "
                  "leaves the atlas, its projection and its fingerprint untouched. This "
                  "diverges from LEARNING_MODE_ARCHITECTURE.md and is flagged in the report."},
    {"decision": "Profile mode",
     "rule_preregistered_in": "src/gaira/v7/lsm/discovery.py — band_profiles()",
     "chosen": "raw observed band mass",
     "alternatives": ["attribution-weighted (observed mass x share explained by the component)"],
     "rationale": "Both were run and the comparison is published. Attribution weighting "
                  "divides by the reconstruction, amplifying noise where the reconstruction "
                  "is small and partly cancelling the per-analyte variation the method "
                  "exists to detect."},
    {"decision": "Linkage",
     "rule_preregistered_in": "src/gaira/v7/lsm/clustering.py — apply_linkage_rule()",
     "chosen": "selected by the pre-registered balance-constrained silhouette rule",
     "alternatives": ["average", "ward", "complete"],
     "rationale": "Silhouette differences of a few hundredths are not meaningful; a motif "
                  "set in which one motif absorbs most analytes has peeled off outliers "
                  "rather than decomposed the component."},
    {"decision": "Chemistry is evaluation only, never selection",
     "rule_preregistered_in": "GAIRA_v7_rebuild/plan/VALIDATION_AND_DECISION_RULES.md P-12",
     "chosen": "no class label enters bands, profiles, linkage or cut selection",
     "alternatives": ["select the cut by chemical purity"],
     "rationale": "Selecting the cut with class labels would make 'motifs align with "
                  "chemistry' circular and unfalsifiable."},
    {"decision": "Stability by jackknife, not bootstrap",
     "rule_preregistered_in": "src/gaira/v7/lsm/clustering.py — jackknife_stability()",
     "chosen": "deterministic leave-one-analyte-out re-clustering",
     "alternatives": ["bootstrap resampling"],
     "rationale": "The brief requires no stochastic behaviour on the discovery path. A "
                  "jackknife gives a comparable stability estimate with no RNG."},
]


if __name__ == "__main__":
    raise SystemExit(main())
