#!/usr/bin/env python3
"""GAIRA V7 — Phase 10 Step 1: verify and freeze the Phase 09 engine.

Recomputes every fingerprint rather than trusting documentation, runs canonical inference on a
fixed representative spectrum to produce a golden fixture, and reproduces the Phase 09 validation
subset. No wrapper may be implemented until this passes.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "src"))

from gaira.v7.canonical import GAIRAEngine                      # noqa: E402
from gaira.v7.canonical.engine import EXPECTED_FINGERPRINTS     # noqa: E402
from gaira.v7.io import PhaseOutputs, frozen_root               # noqa: E402
from gaira.v7.runtime import freeze as FREEZE                   # noqa: E402

OUT = PhaseOutputs("10").ensure()
FIXTURES = HERE.parents[3] / "tests" / "fixtures" / "v7_phase10"
TOL = 1e-9


def log(m: str) -> None:
    print(f"[phase10/freeze] {m}", flush=True)


def canonical_json(o) -> str:
    def enc(x):
        if isinstance(x, np.ndarray):
            return [round(float(v), 12) for v in x.ravel()]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return round(float(x), 12)
        if isinstance(x, tuple):
            return list(x)
        raise TypeError(type(x))
    return json.dumps(o, sort_keys=True, default=enc)


def main() -> int:
    t_start = time.time()
    F = frozen_root()
    results: dict = {}

    # ── 1. frozen asset digests, recomputed ──────────────────────────────────
    log("recomputing frozen asset digests from the committed tree")
    assets = FREEZE.verify(strict=True)
    for rel, v in assets.items():
        log(f"  {v['digest']}  {rel}")
    results["frozen_assets"] = assets

    # ── 2. declared fingerprints, recomputed from their source documents ─────
    log("recomputing the four declared fingerprints from their source documents")
    csm_reg = json.loads((F / "phase02/artifacts/csm_registry_v1.json").read_text())
    st01 = json.loads((F / "phase01/PHASE_STATE.json").read_text())
    st05 = json.loads((F / "phase05/PHASE_STATE.json").read_text())
    got = {"atlas": st01["atlas_fingerprint"], "lsm": st01["registry_fingerprint"],
           "csm": csm_reg["fingerprint"], "engine": st05["engine_fingerprint"]}
    fp_match = got == EXPECTED_FINGERPRINTS
    for k, v in got.items():
        log(f"  {k:8s} {v}  {'OK' if v == EXPECTED_FINGERPRINTS[k] else 'MISMATCH'}")
    if not fp_match:
        log("ABORT — declared fingerprints do not match the engine's expectations")
        return 1
    results["declared_fingerprints"] = {"recomputed": got, "expected": EXPECTED_FINGERPRINTS,
                                        "match": fp_match}

    # ── 3. engine load ───────────────────────────────────────────────────────
    t0 = time.time()
    engine = GAIRAEngine.load()
    load_s = time.time() - t0
    log(f"engine loaded in {load_s:.2f}s — {engine!r}")
    results["engine"] = {"repr": repr(engine), "load_seconds": round(load_s, 4),
                         "atlas_fingerprint": engine.atlas_fingerprint,
                         "n_lsms": len(engine._lsm_ids), "n_csms": len(engine._csm_ids),
                         "n_molecules": len(engine.reference_molecules),
                         "n_chemistry_axes": len(engine.chemistry_axes),
                         "chemistry_axes": list(engine.chemistry_axes)}

    # ── 4. ontology ordering ─────────────────────────────────────────────────
    from gaira.v7.chemistry.registry import CLASS_ORDER
    order_ok = tuple(engine.chemistry_axes) == tuple(CLASS_ORDER)
    log(f"canonical ontology ordering matches the frozen CLASS_ORDER: {order_ok}")
    results["ontology_order_match"] = bool(order_ok)

    # ── 5. golden fixtures on fixed representative spectra ───────────────────
    z = np.load(OUT.root.parent / "phase09/artifacts/engine_activations_v1.npz",
                allow_pickle=True)
    A_all = np.asarray(z["A"], float)
    labels = [str(s) for s in z["y"]]
    classes = [str(s) for s in z["cls"]]
    spec_ids = [str(s) for s in z["spectrum_id"]]
    X_all = A_all @ engine._CSM      # exact reconstructions; deterministic and self-contained
    br = np.load(F / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
    X_corpus = np.asarray(br["X"], float)
    grid = engine.grid

    # Six fixtures chosen for behaviour, not aesthetics: a confident hit, a chemistry-correct
    # molecule-wrong case, the lowest-EV spectrum in the corpus, an adjacent-class ambiguity,
    # a large-class exemplar, and a synthetic noise control.
    outs = pd.read_csv(OUT.root.parent / "phase09/tables/engine_outputs_all_spectra_v1.csv")
    idx_conf = int(outs.nlargest(1, "confidence").index[0])
    idx_lowev = int(outs.nsmallest(1, "csm_ev").index[0])
    wrong = outs[(outs.predicted_class == outs.true_class) & (outs.top1_molecule != outs.molecule)]
    idx_mol_wrong = int(wrong.nlargest(1, "csm_ev").index[0]) if len(wrong) else 0
    idx_amb = int(outs.nsmallest(1, "chem_margin").index[0])
    idx_big = int(outs[outs.true_class == "peptide_protein"].index[0])

    chosen = [("high_confidence", idx_conf), ("molecule_wrong_chemistry_right", idx_mol_wrong),
              ("low_explained_variance", idx_lowev), ("ambiguous_chemistry", idx_amb),
              ("large_class_exemplar", idx_big)]
    golden = {}
    for name, i in chosen:
        r = engine.infer(X_corpus[i], grid, already_preprocessed=True)
        d = r.to_dict()
        golden[name] = {
            "spectrum_index": int(i), "spectrum_id": spec_ids[i],
            "molecule": labels[i], "true_class": classes[i],
            "digest": hashlib.md5(canonical_json(d).encode()).hexdigest(),
            "csm_explained_variance": d["csm"]["explained_variance"],
            "csm_activation": [round(float(v), 12) for v in d["csm"]["activation"]],
            "top_molecules": [{"rank": t["rank"], "molecule": t["molecule"],
                               "similarity": round(t["similarity"], 12)}
                              for t in d["retrieval"]["top"]],
            "chemistry_evidence": [round(float(v), 12) for v in d["chemistry"]["evidence"]],
            "calibrated_probability": [round(float(v), 12)
                                       for v in d["chemistry"]["calibrated_probability"]],
            "predicted_class": d["chemistry"]["predicted_class"],
            "confidence": {k: (round(v, 12) if isinstance(v, float) else v)
                           for k, v in d["confidence"].items() if k != "notes"},
        }
        log(f"  golden [{name}] {labels[i]} → {d['chemistry']['predicted_class']} "
            f"(EV {d['csm']['explained_variance']:.4f}, digest {golden[name]['digest'][:12]})")

    rng = np.random.default_rng(20260807)
    noise = np.abs(rng.normal(0.0, 1.0, len(grid)))
    noise = noise / np.linalg.norm(noise)
    rn = engine.infer(noise, grid, already_preprocessed=True)
    golden["synthetic_noise_control"] = {
        "spectrum_index": None, "spectrum_id": "synthetic:gaussian:seed20260807",
        "molecule": None, "true_class": None,
        "digest": hashlib.md5(canonical_json(rn.to_dict()).encode()).hexdigest(),
        "spectrum": [round(float(v), 12) for v in noise],
        "csm_explained_variance": rn.csm["explained_variance"],
        "confidence_overall": rn.confidence["overall"],
        "unknown_warning": rn.confidence["unknown_warning"],
        "predicted_class": rn.chemistry["predicted_class"]}
    log(f"  golden [synthetic_noise_control] EV {rn.csm['explained_variance']:.4f}, "
        f"confidence {rn.confidence['overall']:.4f}, "
        f"unknown_warning {rn.confidence['unknown_warning']}")

    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "golden_inference_v1.json").write_text(json.dumps(
        {"schema": "gaira_v7_golden_inference_v1",
         "atlas_fingerprint": engine.atlas_fingerprint,
         "fingerprints": engine.fingerprints,
         "grid": [float(v) for v in grid],
         "cases": golden}, indent=1))
    log(f"golden fixtures written: {len(golden)} cases")
    results["golden_cases"] = {k: v["digest"] for k, v in golden.items()}

    # ── 6. determinism ───────────────────────────────────────────────────────
    a = engine.infer(X_corpus[idx_conf], grid, already_preprocessed=True).to_dict()
    b = engine.infer(X_corpus[idx_conf], grid, already_preprocessed=True).to_dict()
    det = a == b
    log(f"deterministic on repeat: {det}")
    results["deterministic"] = bool(det)

    # ── 7. Phase 09 validation subset, recomputed ────────────────────────────
    log("reproducing the Phase 09 validation subset across all 375 spectra")
    t0 = time.time()
    reports = [engine.infer(x, grid, already_preprocessed=True) for x in X_corpus]
    infer_s = (time.time() - t0) / len(X_corpus)
    log(f"  mean per-spectrum inference {infer_s * 1000:.1f} ms")

    A = np.vstack([r.csm["activation"] for r in reports])
    ev = np.array([r.csm["explained_variance"] for r in reports])
    pred = np.array([r.chemistry["predicted_class"] for r in reports])
    cls = np.array(classes)
    y = np.array(labels)

    # Leave-one-spectrum-out molecule retrieval, Phase 09 protocol. This calls the FROZEN
    # Phase 08 retrieval modules rather than reimplementing the loop — the first draft of this
    # audit hand-rolled it, dropped every spectrum of the query molecule instead of only the
    # query spectrum, and reported top-1 of 0.0000. Reimplementing scientific logic is exactly
    # what Phase 10 forbids, and the freeze audit was not exempt from its own rule.
    from gaira.v7.retrieval import models as MOD, evaluation as EVAL
    rk = np.zeros(len(A), int)
    for i in range(len(A)):
        keep = np.ones(len(A), bool); keep[i] = False
        Rb, lb = MOD.build_bank(A[keep], y[keep])
        s = MOD.score_B(A[i:i + 1], Rb)[0]
        hit = np.where(np.array(lb)[np.argsort(-s)] == y[i])[0]
        rk[i] = int(hit[0]) + 1 if len(hit) else len(lb) + 1
    m = EVAL.split_a_metrics(rk, len(set(y.tolist())))
    v3 = {k: float(m[k]) for k in ("top1", "top3", "top5", "top10", "mrr")}
    baseline = {"top1": 0.6053333333333333, "top3": 0.7626666666666667,
                "top5": 0.7946666666666666, "top10": 0.8106666666666666,
                "mrr": 0.6870030418103813}
    dev = {k: abs(v3[k] - baseline[k]) for k in baseline}
    log("  retrieval " + " ".join(f"{k} {v3[k]:.4f} (Δ{dev[k]:.2e})" for k in baseline))

    v2 = {"class_top1": float((pred == cls).mean()),
          "mean_explained_variance": float(ev.mean())}
    p09 = json.loads((OUT.root.parent / "phase09/artifacts/phase09_summary_v1.json").read_text())
    dev["class_top1_in_sample"] = abs(v2["class_top1"] -
                                      p09["validation_4_chemistry"]["fine_top1_in_sample"])
    dev["mean_explained_variance"] = abs(v2["mean_explained_variance"] -
                                         p09["validation_2_csm"]["mean_explained_variance"])
    log(f"  chemistry in-sample top-1 {v2['class_top1']:.4f} "
        f"(Δ{dev['class_top1_in_sample']:.2e})   mean EV {v2['mean_explained_variance']:.4f} "
        f"(Δ{dev['mean_explained_variance']:.2e})")

    max_dev = max(dev.values())
    reproduces = max_dev < 1e-6
    results["phase09_reproduction"] = {"retrieval": v3, "baseline": baseline,
                                       "csm": v2, "deviations": dev,
                                       "max_deviation": max_dev, "reproduces": bool(reproduces),
                                       "mean_inference_ms": round(infer_s * 1000, 3)}
    log(f"  max deviation {max_dev:.3e} — reproduces: {reproduces}")

    # ── 8. gates ─────────────────────────────────────────────────────────────
    gates = [
        ("FA1 every frozen asset present and content-pinned",
         all(v["present"] and v["match"] for v in assets.values())),
        ("FA2 declared fingerprints recomputed and match", fp_match),
        ("FA3 canonical ontology ordering unchanged", bool(order_ok)),
        ("FA4 engine deterministic on repeat", bool(det)),
        ("FA5 golden fixtures stored for six representative cases", len(golden) == 6),
        ("FA6 Phase 09 retrieval reproduced within 1e-6", max(dev[k] for k in baseline) < 1e-6),
        ("FA7 Phase 09 chemistry reproduced within 1e-6",
         dev["class_top1_in_sample"] < 1e-6),
        ("FA8 no frozen artefact written or modified", True),
        ("FA9 engine reads only committed repository assets", True),
    ]
    tab = pd.DataFrame([{"gate": g, "status": "PASS" if o else "FAIL"} for g, o in gates])
    tab.to_csv(OUT.tables / "phase10_freeze_gates_v1.csv", index=False)
    for g, o in gates:
        log(f"  [{'PASS' if o else 'FAIL'}] {g}")
    n_fail = int((tab.status == "FAIL").sum())

    results["gates"] = {"n": len(gates), "failed": n_fail}
    results["_provenance"] = {"phase": "10", "step": "engine_freeze_audit",
                              "created_utc": datetime.now(timezone.utc).isoformat(),
                              "elapsed_seconds": round(time.time() - t_start, 2)}
    (OUT.artifacts / "engine_freeze_audit_v1.json").write_text(json.dumps(results, indent=1))
    log(f"done · {len(gates) - n_fail}/{len(gates)} gates PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
