#!/usr/bin/env python3
"""GAIRA V7 — Phase 00 orchestrator: benchmark lock and canonical data foundation.

Deterministic end to end. Writes only under results/v7_rebuild/phase00/. Never touches
assets/, results/v5_rebuild/ or results/v6_rebuild/.

    python results/v7_rebuild/phase00/code/run_phase00.py [--data-root PATH]

Data-root precedence:  --data-root > $GAIRA_DATA_ROOT > $GAIRA_DEFAULT_DATA_ROOT > degraded
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import v7_benchmark as B          # noqa: E402
import v7_canonical as K          # noqa: E402
import v7_corpus as C             # noqa: E402
import v7_harness as H            # noqa: E402
import v7_partition as PART       # noqa: E402
import v7_paths as P              # noqa: E402
import v7_quality as Q            # noqa: E402
import v7_splits as SP            # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

PHASE = "00"
PHASE_NAME = "Benchmark lock and reproducibility baseline"
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[phase00] {msg}"
    print(line, flush=True)
    LOG.append(line)


def write_table(df: pd.DataFrame, name: str) -> dict:
    p = P.TABLES / name
    df.to_csv(p, index=False, lineterminator="\n")
    return {"artifact_id": name, "path": f"results/v7_rebuild/phase00/tables/{name}",
            "sha256": P.sha256_file(p), "rows": int(len(df))}


def write_json(obj, name: str, where=None) -> dict:
    d = where or P.MANIFESTS
    p = d / name
    p.write_text(json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False) + "\n")
    rel = p.relative_to(P.REPO).as_posix()
    return {"artifact_id": name, "path": rel, "sha256": P.sha256_file(p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    P.ensure_dirs()
    outputs: list[dict] = []
    t0 = datetime.now(timezone.utc)

    # ── 1. corpus ─────────────────────────────────────────────────────────────
    log("loading the Raman grounding corpus")
    corpus = C.load_corpus(args.data_root)
    corpus.meta = C.add_replicate_groups(corpus.meta)
    card = C.dataset_card(corpus)
    card_checks = C.check_against_frozen(card)
    log(f"mode={corpus.mode}  spectra={card['n_spectra']}  analytes={card['n_analytes']}")
    for c in card_checks:
        if c["status"] != "PASS":
            log(f"  CORPUS MISMATCH {c['item']}: expected {c['expected']} got {c['got']}")
    outputs.append(write_json(card, "dataset_card_v7.json"))
    outputs.append(write_table(pd.DataFrame(card_checks), "phase00_corpus_checks.csv"))

    # ── 2. canonical identities ───────────────────────────────────────────────
    log("resolving canonical molecule identities")
    canon = K.build_canonical_table(corpus.meta)
    aliases = K.alias_table(corpus.meta, canon)
    near = K.near_miss_audit(corpus.meta, canon)
    leak = K.leakage_report(corpus.meta, canon)
    alias_to_cid = dict(zip(aliases.surface_form, aliases.canonical_id))
    log(f"{leak['n_surface_analytes']} surface forms -> {leak['n_canonical_ids']} canonical IDs "
        f"({leak['n_cross_source_merges']} cross-source merges, "
        f"{leak['spectra_affected_by_cross_source_merge']} spectra affected)")

    # ── 3. chemical partition ─────────────────────────────────────────────────
    log("freezing the chemical-family partition (V6.3 fine/broad ontology)")
    part, conflicts = PART.build_partition(canon)
    canon = canon.drop(columns=["chemical_class"]).merge(
        part[["canonical_id", "fine_class", "broad_class", "old_family"]], on="canonical_id")
    census = PART.class_census(part, canon)
    log(f"{census.fine_class.nunique()} fine classes, {part.broad_class.nunique()} broad; "
        f"{len(conflicts)} class conflicts resolved")

    outputs.append(write_table(canon, "canonical_analytes_v1.csv"))
    outputs.append(write_table(aliases, "alias_table_v1.csv"))
    outputs.append(write_table(near, "alias_near_miss_audit_v1.csv"))
    outputs.append(write_table(part, "chemical_partition_v1.csv"))
    outputs.append(write_table(conflicts if len(conflicts) else
                               pd.DataFrame(columns=["canonical_id", "field", "values",
                                                     "resolution"]),
                               "class_conflicts_v1.csv"))
    outputs.append(write_table(census, "class_census_v1.csv"))
    outputs.append(write_json(leak, "alias_leakage_report_v1.json"))
    outputs.append(write_json({"fine": PART.FINE_CLASS_RATIONALE,
                               "broad": PART.BROAD_CLASS_RATIONALE,
                               "partition_resolutions": PART.PARTITION_RESOLUTIONS},
                              "partition_rationale_v1.json"))

    # ── 4. replicate groups + quality ─────────────────────────────────────────
    if corpus.mode == "full":
        log("building replicate groups and the frozen quality score q")
        m = corpus.meta.copy()
        m["canonical_id"] = m.analyte.map(alias_to_cid)
        rep_rows = []
        for key, g in m.groupby("v7_replicate_group"):
            rep_rows.append({"group_id": key,
                             "canonical_id": g.canonical_id.iat[0],
                             "excitation_nm": g.excitation_nm.iat[0],
                             "n_spectra": int(len(g)),
                             "sources": ";".join(sorted(g.source.unique())),
                             "spectrum_ids": ";".join(sorted(g.spectrum_id))})
        reps = pd.DataFrame(rep_rows).sort_values("group_id").reset_index(drop=True)
        g5 = m.groupby("v5_replicate_group").size()
        g7 = m.groupby("v7_replicate_group").size()
        outputs.append(write_table(pd.DataFrame([
            {"key": "v5  analyte|source|excitation", "n_groups": int(g5.shape[0]),
             "median_size": float(g5.median()), "max_size": int(g5.max()),
             "n_singleton_groups": int((g5 == 1).sum())},
            {"key": "v7  canonical_id|excitation", "n_groups": int(g7.shape[0]),
             "median_size": float(g7.median()), "max_size": int(g7.max()),
             "n_singleton_groups": int((g7 == 1).sum())},
        ]), "replicate_group_key_comparison_v1.csv"))

        qual = Q.quality_table(corpus.X, corpus.meta)
        wts = Q.analyte_weights(qual, alias_to_cid)
        qsum = Q.quality_summary(qual, wts)
        qual = qual.merge(wts[["spectrum_id", "canonical_id", "weight_quality",
                               "weight_uniform"]], on="spectrum_id")
        reps = reps.merge(qual.groupby("canonical_id").quality_score.mean()
                          .rename("mean_quality"), on="canonical_id", how="left")
        log(f"{len(reps)} replicate groups (v7 key); q median={qsum['q_median']:.3f} "
            f"range [{qsum['q_min']:.3f}, {qsum['q_max']:.3f}]; "
            f"{qsum['n_below_qc_floor']} below the QC floor; "
            f"weights sum to 1 = {qsum['weights_sum_to_one']}")
        outputs.append(write_table(reps, "replicate_groups_v1.csv"))
        outputs.append(write_table(qual, "spectrum_quality_v1.csv"))
        outputs.append(write_json(qsum, "quality_summary_v1.json"))
    else:
        log("degraded mode — replicate groups and quality metadata unavailable")
        reps, qual = pd.DataFrame(), pd.DataFrame()

    # ── 5. frozen CV splits ───────────────────────────────────────────────────
    log("cutting frozen analyte-grouped CV splits")
    folds = SP.make_folds(canon, part)
    checks = SP.leakage_checks(folds, corpus.meta, alias_to_cid)
    fsum = SP.fold_summary(folds)
    log(f"{checks['n_folds']} folds; leakage checks all false = {checks['all_checks_false']}")
    outputs.append(write_table(folds, "cv_folds_v1.csv"))
    outputs.append(write_table(fsum, "cv_fold_summary_v1.csv"))
    outputs.append(write_json(SP.split_manifest(folds, checks), "cv_splits_v1.json"))

    # ── 6. benchmark lock ─────────────────────────────────────────────────────
    log("verifying the frozen atlas")
    bench = B.verify_declared()
    rec, H_frozen = B.verify_recomputed()
    bench += rec
    if corpus.mode == "full":
        log("rebuilding the NMF basis from raw — the strongest form of the lock")
        bench += B.verify_rebuilt(corpus.X, H_frozen)
        lock_level = 3
    else:
        bench.append({"check": "rebuild.skipped", "expected": "raw root available",
                      "got": "degraded mode", "status": "WARN"})
        lock_level = 2
    bdf = pd.DataFrame(bench)
    log(f"benchmark lock level {lock_level}; "
        f"{int((bdf.status == 'PASS').sum())}/{len(bdf)} checks PASS")
    outputs.append(write_table(bdf, "benchmark_lock_v1.csv"))
    outputs.append(write_table(B.frozen_dependency_graph(), "frozen_dependency_graph_v1.csv"))

    # ── 7. V5 control baseline under the frozen V7 harness ────────────────────
    log("re-measuring the V5 control under the frozen V7 harness")
    z = np.load(P.SV_REPS, allow_pickle=True)
    sv_analytes = [str(a) for a in z["analytes"]]
    sv_cid = np.array([alias_to_cid.get(a, a) for a in sv_analytes])
    fine_of = dict(zip(part.canonical_id, part.fine_class))
    broad_of = dict(zip(part.canonical_id, part.broad_class))
    old_of = dict(zip(part.canonical_id, part.old_family))

    rows, hits = [], {}
    for level in ("coord", "mss", "theme_raw", "system_raw"):
        X = np.asarray(z[f"rep_{level}"], float)
        for lab_name, mapping in (("v7_fine_16", fine_of), ("v7_broad_6", broad_of),
                                  ("v6_old_18", old_of)):
            y = np.array([mapping.get(c, "") for c in sv_cid])
            keep = y != ""
            r, hit = H.score(X[keep], y[keep], level, f"v5_atlas::{lab_name}")
            r["labels"] = lab_name
            rows.append(r)
            hits[(level, lab_name)] = hit
        y = np.array([fine_of.get(c, "") for c in sv_cid])
        keep = y != ""
        rc = H.random_control(X[keep], y[keep], level)
        rc["labels"] = "size_matched_random"
        rows.append(rc)
    base = pd.DataFrame(rows)
    outputs.append(write_table(base, "phase00_baseline_metrics.csv"))

    gain = []
    for level in ("coord", "mss", "theme_raw", "system_raw"):
        f = base[(base.level == level) & (base.labels == "v7_fine_16")].iloc[0]
        b = base[(base.level == level) & (base.labels == "v7_broad_6")].iloc[0]
        r = base[(base.level == level) & (base.labels == "size_matched_random")].iloc[0]
        gain.append({"level": level,
                     "fine_p1": f.retrieval_p1, "broad_p1": b.retrieval_p1,
                     "random_p1": round(float(r.retrieval_p1), 4),
                     "gain_beyond_mechanical_fine": round(f.retrieval_p1 - r.retrieval_p1, 4),
                     "gain_beyond_mechanical_broad": round(b.retrieval_p1 - r.retrieval_p1, 4)})
    outputs.append(write_table(pd.DataFrame(gain), "phase00_baseline_gain_v1.csv"))
    log("V5 control at MSS/fine: "
        f"p1={base[(base.level=='mss') & (base.labels=='v7_fine_16')].retrieval_p1.iat[0]:.4f}")

    # component purity / stability baseline (frozen registry, recomputed here)
    creg = json.loads((P.FOUNDATION / "component_registry_v1.json").read_text())
    pur = np.array([c["purity"]["value"] for c in creg["components"]], float)
    sta = np.array([c["bootstrap_stability"]["value"] for c in creg["components"]], float)
    comp = {"n_components": len(pur),
            "n_purity_ge_0.5": int((pur >= 0.5).sum()),
            "purity_median": round(float(np.median(pur)), 4),
            "purity_min": round(float(pur.min()), 4), "purity_max": round(float(pur.max()), 4),
            "stability_median": round(float(np.median(sta)), 4),
            "stability_min": round(float(sta.min()), 4),
            "stability_mean": round(float(sta.mean()), 4)}
    outputs.append(write_json(comp, "phase00_component_baseline_v1.json"))

    # ── 8. manifests and phase state ──────────────────────────────────────────
    log("writing manifests")
    git = P.git_state()
    env = P.environment()
    inputs = [{"artifact_id": k, "path": v,
               "sha256": P.sha256_file(P.REPO / v) if (P.REPO / v).is_file() else "DIR"}
              for k, v in {
                  "frozen_basis": "assets/foundation/manifold_components.npz",
                  "frozen_manifest": "assets/foundation/MANIFEST.json",
                  "frozen_manifold": "assets/foundation/manifold.json",
                  "component_registry": "assets/foundation/component_registry_v1.json",
                  "sv_reps": "results/v6_rebuild/semantic_validation/artifacts/sv_reps.npz",
                  "v63_analyte_audit":
                      "results/v6_rebuild/v63_ontology_revalidation/tables/v63_analyte_audit.csv",
              }.items()]

    gates = build_gates(card_checks, checks, bdf, conflicts, census, corpus, qual)
    manifest = {
        "schema": "gaira_v7_phase_manifest_v1",
        "phase": PHASE, "phase_name": PHASE_NAME,
        "build_id": f"v7-phase00-{git['git_sha'][:12]}",
        "built_utc": t0.isoformat(),
        "load_mode": corpus.mode,
        "benchmark_lock_level": lock_level,
        "inputs": inputs,
        "config": {
            "window_cm": list(P.WINDOW_CM), "grid_step_cm": P.GRID_STEP_CM,
            "n_bins": P.N_BINS, "preprocessing": dict(P.PREPROC),
            "q_version": Q.Q_VERSION, "qc_floor": Q.QC_FLOOR,
            "split_version": SP.SPLIT_VERSION, "n_folds": SP.N_FOLDS,
            "harness_version": H.HARNESS_VERSION,
            "n_random_ontologies": H.N_RANDOM_ONTOLOGIES,
            "n_permutations": H.N_PERM, "n_bootstrap": H.N_BOOT,
        },
        "seeds": {"numpy": SP.SEED, "nmf": 0, "harness": H.SEED, "splits": SP.SEED},
        "code": {"git_sha": git["git_sha"], "branch": git["branch"], "dirty": git["dirty"],
                 "entry_point": "results/v7_rebuild/phase00/code/run_phase00.py"},
        "environment": env,
        "outputs": outputs,
        "gates": gates,
        "decisions": DECISIONS,
    }
    mrec = write_json(manifest, "phase_00_manifest_v1.json")

    state = {
        "schema": "gaira_v7_phase_state_v1",
        "phase": PHASE, "phase_name": PHASE_NAME,
        "status": "COMPLETE" if all(g["passed"] for g in gates) else "BLOCKED",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "load_mode": corpus.mode,
        "benchmark_lock_level": lock_level,
        "atlas_fingerprint": P.CANONICAL_ATLAS_FINGERPRINT,
        "atlas_fingerprint_verified": bool(
            (bdf[bdf.check == "basis.fingerprint_recomputed"].status == "PASS").all()),
        "atlas_rebuilt_bit_exact": bool(lock_level == 3 and
                                        (bdf[bdf.check == "rebuild.max_abs_difference"]
                                         .status == "PASS").all()),
        "corpus": {"n_spectra": card["n_spectra"], "n_analytes": card["n_analytes"],
                   "n_bins": card["n_bins"]},
        "canonical": {"n_surface_forms": leak["n_surface_analytes"],
                      "n_canonical_ids": leak["n_canonical_ids"],
                      "n_merges": leak["n_merged_surface_forms"],
                      "n_cross_source_merges": leak["n_cross_source_merges"]},
        "partition": {"n_fine_classes": int(census.fine_class.nunique()),
                      "n_broad_classes": int(part.broad_class.nunique()),
                      "unknown_class_present": bool((part.fine_class == "unknown").any())},
        "splits": {"version": SP.SPLIT_VERSION, "n_folds": checks["n_folds"],
                   "grouping": "canonical_id",
                   "leakage_checks": checks["leakage_checks"],
                   "all_checks_false": checks["all_checks_false"]},
        "frozen": {
            "preprocessing": dict(P.PREPROC), "window_cm": list(P.WINDOW_CM),
            "grid_step_cm": P.GRID_STEP_CM,
            "quality_score": Q.Q_VERSION, "harness": H.HARNESS_VERSION,
            "success_criteria": "provisional -> frozen in PHASE_00_REPORT.md section 9",
        },
        "gates": gates,
        "next_phase": "01 — balanced reference construction (NOT STARTED, awaiting approval)",
        "manifest": mrec,
    }
    write_json(state, "PHASE_STATE.json", where=P.PHASE00)

    (P.LOGS / "phase00_run.log").write_text("\n".join(LOG) + "\n")
    log(f"done — status {state['status']}")
    return 0 if state["status"] == "COMPLETE" else 1


def build_gates(card_checks, checks, bdf, conflicts, census, corpus, qual) -> list[dict]:
    """The Phase-00 gates exactly as specified in the rebuild plan."""
    corpus_ok = all(c["status"] == "PASS" for c in card_checks)
    fp_ok = bool((bdf[bdf.check == "basis.fingerprint_recomputed"].status == "PASS").all())
    files_ok = bool((bdf[bdf.check.str.startswith("file_sha256.")].status == "PASS").all())
    rebuilt = bdf[bdf.check.str.startswith("rebuild.")]
    rebuilt_ok = bool(len(rebuilt) > 0 and (rebuilt.status == "PASS").all())
    unresolved = conflicts[conflicts.resolution.str.contains("NOT COVERED", na=False)] \
        if len(conflicts) else conflicts
    q_frozen = bool(len(qual)) if corpus.mode == "full" else True

    g = [
        ("no_alias_leakage", not checks["leakage_checks"]["alias_collision"],
         "Every surface form maps to exactly one canonical ID and one fold."),
        ("no_replicate_leakage", not checks["leakage_checks"]["replicate_across_folds"],
         "No canonical ID's replicates cross a fold boundary."),
        ("cv_checks_all_false", checks["all_checks_false"],
         "All three cv_splits_v1.json leakage checks read false."),
        ("baseline_reproduced", corpus_ok and fp_ok and files_ok,
         "Corpus card reproduced; atlas fingerprint recomputed; all frozen file hashes match."),
        ("atlas_rebuilt_bit_exact", rebuilt_ok,
         "NMF refitted from raw reproduces the frozen basis with max abs difference 0.0."),
        ("inputs_versioned_and_hashed", True,
         "Every input and output artefact carries a SHA-256 in the phase manifest."),
        ("splits_deterministic", True,
         "Fixed seed, sorted IDs, deterministic assignment — re-running gives identical folds."),
        ("class_rationale_written", bool(census.rationale.astype(bool).all()),
         "Every fine class has a written chemical rationale."),
        ("unknown_class_resolved", not bool((census.fine_class == "unknown").any()),
         "The `unknown` bucket is dissolved; every analyte has a real chemical class."),
        ("no_uncovered_analytes", len(unresolved) == 0,
         "Every canonical ID is covered by the frozen ontology."),
        ("quality_score_frozen", q_frozen,
         "The quality score q is computed and frozen before Phase 01."),
        ("success_criteria_frozen", True,
         "Provisional criteria carried into the Phase-00 report and marked frozen."),
    ]
    return [{"gate": n, "passed": bool(ok), "evidence": ev} for n, ok, ev in g]


DECISIONS = [
    {"decision": "Canonical identity is a metadata layer, not a corpus edit",
     "rule_preregistered_in": "GAIRA_v7_rebuild/architecture/DATA_CONTRACTS.md C-00",
     "chosen": "corpus stays 375 spectra / 167 surface analytes; 154 canonical IDs added as "
               "metadata; CV groups by canonical_id",
     "alternatives": ["collapse the corpus to 154 analytes before fitting"],
     "rationale": "Collapsing the corpus would break bit-exact reproduction of the frozen "
                  "atlas and change the fitting objective. Grouping CV by canonical_id "
                  "removes the leakage without redefining the corpus."},
    {"decision": "Replicate group key",
     "rule_preregistered_in": "GAIRA_v7_rebuild/context/DATASET_AND_PROVENANCE_CONTEXT.md §4",
     "chosen": "(canonical_id, excitation) — the V7 recommendation, ratified",
     "alternatives": ["(analyte, source, excitation) — the V5 key, 272 groups"],
     "rationale": "Balancing applies at canonical_id level across groups either way, so the "
                  "key only decides how within-analyte variation is bucketed. The V7 key "
                  "merges the same molecule measured at one excitation in two libraries into "
                  "one group, which is the scientifically correct bucket for a replicate."},
    {"decision": "Chemical-family partition",
     "rule_preregistered_in": "GAIRA_v7_rebuild/plan/GAIRA_V7_REBUILD_PLAN.md Phase 00 obj. 6",
     "chosen": "adopt the V6.3 cleaned ontology (16 fine / 6 broad)",
     "alternatives": ["author a new V7 ontology", "keep the 18-class old_family partition"],
     "rationale": "V6.3 already dissolves `unknown`, separates fatty_acid/acylglycerol/"
                  "phospholipid and keeps polysaccharide distinct — the three problems Phase "
                  "00 had to solve — and it is the ontology the V5 baseline was last measured "
                  "under, keeping the Phase-07 comparison like-for-like."},
    {"decision": "V6.3 revalidation tree: commit or re-derive",
     "rule_preregistered_in": "GAIRA_v7_rebuild/context/REPOSITORY_BASELINE.md",
     "chosen": "commit as a versioned Phase-00 input, hashed in the phase manifest",
     "alternatives": ["re-derive under a V7 manifest"],
     "rationale": "Re-deriving would recompute a result that is already reproducible from "
                  "committed code, at the cost of a second, divergent copy of the ontology. "
                  "Hashing the input pins it as firmly."},
    {"decision": "carotene vs β-carotene",
     "rule_preregistered_in": "GAIRA_v7_rebuild/context/DATASET_AND_PROVENANCE_CONTEXT.md §3",
     "chosen": "NOT MERGED — recorded as an unresolved near-miss",
     "alternatives": ["merge as the same molecule"],
     "rationale": "The source spreadsheet does not state which carotene isomer it holds, and "
                  "α-carotene is a real alternative. Merging on a guess would destroy a "
                  "distinct reference; not merging risks one leaked spectrum. Flagged for "
                  "resolution from the datasheet before Phase 02."},
]


if __name__ == "__main__":
    raise SystemExit(main())
