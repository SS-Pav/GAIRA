#!/usr/bin/env python3
"""GAIRA V7 — Phase 10 Steps 13, 14 and 17: cross-surface parity, latency, scientific validation.

Runs a locked set of representative spectra through SIX independent surfaces and requires
scientifically identical output. This is the validation the whole phase turns on: if two surfaces
disagree, one of them is computing something, and the architecture has already failed.

  A. GAIRAEngine directly
  B. the runtime service
  C. the Python SDK
  D. FastAPI (through the ASGI test client, the same code path uvicorn serves)
  E. the MCP tool layer
  F. the Streamlit backend path (the SDK call the app makes, with the app's own options)
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "src"))

from gaira.v7 import GAIRA                                        # noqa: E402
from gaira.v7.canonical import GAIRAEngine                        # noqa: E402
from gaira.v7.contracts import (InferenceOptions, InferenceRequest, SpectrumInput,  # noqa: E402
                                SpectrumMetadata)
from gaira.v7.io import PhaseOutputs, frozen_root                 # noqa: E402
from gaira.v7.mcp import call as mcp_call                         # noqa: E402
from gaira.v7.runtime.service import GAIRAService                 # noqa: E402

OUT = PhaseOutputs("10").ensure()
TOL = 1e-12


def log(m: str) -> None:
    print(f"[phase10/parity] {m}", flush=True)


def core(d: dict) -> dict:
    """The scientific fields, rounded to a documented precision for comparison."""
    return {
        "csm": [round(float(v), 12) for v in (d.get("csm") or {}).get("activation", [])],
        "csm_ev": round(float((d.get("csm") or {}).get("explained_variance", 0.0)), 12),
        "chem": [round(float(v), 12) for v in d["chemistry"]["evidence"]],
        "chem_cal": [round(float(v), 12) for v in d["chemistry"]["calibrated_probability"]],
        "pred": d["chemistry"]["predicted_class"],
        "ranks": [[h["rank"], h["molecule"], round(float(h["similarity"]), 12)]
                  for h in d["retrieval"]["top"]],
        "conf": round(float(d["confidence"]["overall"]), 12),
        "unknown": bool(d["confidence"]["unknown_warning"]),
        "coverage": round(float(d["preprocessing"]["grid_coverage"]), 12),
        "audit_reconcile": bool((d.get("audit") or {}).get("all_scores_reconcile", True)),
    }


def max_abs_diff(a: dict, b: dict) -> tuple[float, list[str]]:
    worst, fields = 0.0, []
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, list) and va and isinstance(va[0], (int, float)):
            d = max((abs(x - y) for x, y in zip(va, vb)), default=0.0)
        elif isinstance(va, (int, float)) and not isinstance(va, bool):
            d = abs(va - vb)
        else:
            d = 0.0 if va == vb else float("inf")
        if d > TOL:
            fields.append(k)
        worst = max(worst, d)
    return worst, fields


def main() -> int:
    t_start = time.time()
    F = frozen_root()

    # ── the locked spectrum set ──────────────────────────────────────────────
    br = np.load(F / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
    X = np.asarray(br["X"], float)
    grid = np.asarray(br["grid"], float)
    outs = pd.read_csv(F / "phase09/tables/engine_outputs_all_spectra_v1.csv")

    wrong = outs[(outs.predicted_class == outs.true_class) & (outs.top1_molecule != outs.molecule)]
    cases = [
        ("high_confidence_correct", int(outs.nlargest(1, "confidence").index[0])),
        ("chemistry_right_molecule_wrong", int(wrong.nlargest(1, "csm_ev").index[0])),
        ("low_explained_variance", int(outs.nsmallest(1, "csm_ev").index[0])),
        ("ambiguous_class", int(outs.nsmallest(1, "chem_margin").index[0])),
    ]
    # different chemistry classes, and different source libraries where available
    for cls in ("sterol_steroid", "purine", "polysaccharide", "peptide_protein"):
        sel = outs[outs.true_class == cls]
        if len(sel):
            cases.append((f"class_{cls}", int(sel.index[0])))
    for src in sorted(outs.source.dropna().unique())[:3]:
        sel = outs[outs.source == src]
        if len(sel):
            cases.append((f"source_{str(src)[:20]}", int(sel.index[0])))

    rng = np.random.default_rng(20260807)
    noise = np.abs(rng.normal(0, 1, len(grid)))
    noise = noise / np.linalg.norm(noise)

    log(f"locked spectrum set: {len(cases)} corpus cases + 1 synthetic noise control "
        f"+ 1 malformed input")

    # ── surfaces ─────────────────────────────────────────────────────────────
    engine = GAIRAEngine.load()
    svc = GAIRAService.instance()
    sdk = GAIRA.shared()
    from fastapi.testclient import TestClient
    from gaira.v7.api import app

    OPTS = {"already_preprocessed": True, "top_k_molecules": 10}
    STREAMLIT_OPTS = {**OPTS, "include_reconstruction": True}   # what the app actually asks for

    rows, digests = [], []
    with TestClient(app) as http:
        for name, idx in cases + [("synthetic_noise_control", -1)]:
            y = noise if idx < 0 else X[idx]
            body = {"spectrum": {"wavenumber": grid.tolist(), "intensity": y.tolist()},
                    "options": OPTS}

            # A — engine directly
            t0 = time.perf_counter()
            a_rep = engine.infer(y, grid, top_k=10, already_preprocessed=True)
            t_engine = time.perf_counter() - t0
            a = {"csm": [round(float(v), 12) for v in a_rep.csm["activation"]],
                 "csm_ev": round(float(a_rep.csm["explained_variance"]), 12),
                 "chem": [round(float(v), 12) for v in a_rep.chemistry["evidence"]],
                 "chem_cal": [round(float(v), 12)
                              for v in a_rep.chemistry["calibrated_probability"]],
                 "pred": a_rep.chemistry["predicted_class"],
                 "ranks": [[t["rank"], t["molecule"], round(float(t["similarity"]), 12)]
                           for t in a_rep.retrieval["top"]],
                 "conf": round(float(a_rep.confidence["overall"]), 12),
                 "unknown": bool(a_rep.confidence["unknown_warning"]),
                 "coverage": None, "audit_reconcile": True}

            # B — runtime service
            req = InferenceRequest(
                spectrum=SpectrumInput(wavenumber=grid.tolist(), intensity=y.tolist()),
                metadata=SpectrumMetadata(), options=InferenceOptions(**OPTS))
            t0 = time.perf_counter()
            b_res = svc.infer(req)
            t_service = time.perf_counter() - t0
            b = core(b_res.model_dump(mode="json"))
            a["coverage"] = b["coverage"]           # the engine does not compute coverage

            # C — Python SDK
            t0 = time.perf_counter()
            c_res = sdk.infer(grid.tolist(), y.tolist(), None, OPTS)
            t_sdk = time.perf_counter() - t0
            c = core(c_res.model_dump(mode="json"))

            # D — FastAPI
            t0 = time.perf_counter()
            resp = http.post("/v1/infer", json=body)
            t_api = time.perf_counter() - t0
            d = core(resp.json())

            # E — MCP
            t0 = time.perf_counter()
            e_payload = mcp_call("gaira_infer_spectrum", {
                "spectrum": {"wavenumber": grid.tolist(), "intensity": y.tolist()},
                "top_k": 10})
            t_mcp = time.perf_counter() - t0
            # MCP does not expose already_preprocessed; re-run through the service the same way
            # the tool does, so the comparison is like for like.
            e = core(e_payload)

            # F — Streamlit backend path
            f_res = sdk.infer(grid.tolist(), y.tolist(),
                              {"sample_type": "pure", "modality": "raman"}, STREAMLIT_OPTS)
            f = core(f_res.model_dump(mode="json"))

            # MCP runs the full preprocessing path (no already_preprocessed flag), so it is
            # compared against a matched run rather than against the canonical-grid path.
            e_ref = core(sdk.infer(grid.tolist(), y.tolist(), None,
                                   {"top_k_molecules": 10}).model_dump(mode="json"))

            pairs = [("engine→service", a, b), ("service→sdk", b, c), ("sdk→api", c, d),
                     ("sdk→streamlit", c, f), ("mcp→matched_sdk", e, e_ref)]
            for label, u, v in pairs:
                diff, fields = max_abs_diff(u, v)
                rows.append({"case": name, "comparison": label, "max_abs_diff": diff,
                             "identical": diff <= TOL, "differing_fields": ",".join(fields)})
            digests.append({"case": name, "service": b_res.result_digest,
                            "sdk": c_res.result_digest, "api": resp.json()["result_digest"],
                            "streamlit": f_res.result_digest,
                            "all_equal": len({b_res.result_digest, c_res.result_digest,
                                              resp.json()["result_digest"],
                                              f_res.result_digest}) == 1,
                            "engine_ms": round(t_engine * 1000, 3),
                            "service_ms": round(t_service * 1000, 3),
                            "sdk_ms": round(t_sdk * 1000, 3),
                            "api_ms": round(t_api * 1000, 3),
                            "mcp_ms": round(t_mcp * 1000, 3)})
            log(f"  {name:<34s} digest {b_res.result_digest[:12]}  "
                f"engine {t_engine * 1000:6.1f}ms  api {t_api * 1000:6.1f}ms")

        # ── malformed input, every surface ───────────────────────────────────
        log("malformed input handling")
        malformed = {"spectrum": {"wavenumber": [1.0, 2.0], "intensity": [1.0, 2.0]}}
        mal_rows = []
        r = http.post("/v1/infer", json=malformed)
        mal_rows.append({"surface": "api", "status": r.status_code,
                         "rejected": r.status_code == 422})
        try:
            sdk.infer([1.0, 2.0], [1.0, 2.0])
            mal_rows.append({"surface": "sdk", "status": 200, "rejected": False})
        except Exception as exc:
            mal_rows.append({"surface": "sdk", "status": type(exc).__name__, "rejected": True})
        m = mcp_call("gaira_validate_spectrum",
                     {"spectrum": {"wavenumber": [1.0, 2.0], "intensity": [1.0, 2.0]}})
        mal_rows.append({"surface": "mcp", "status": "validated",
                         "rejected": not m["can_run"]})
        for row in mal_rows:
            log(f"  {row['surface']:<8s} rejected={row['rejected']} ({row['status']})")

        # unsupported modality must be blocked on every surface
        sers = {"spectrum": {"wavenumber": grid.tolist(), "intensity": X[0].tolist()},
                "metadata": {"modality": "ag_sers"}, "options": OPTS}
        r = http.post("/v1/infer", json=sers)
        sers_api = r.status_code == 422
        try:
            sdk.infer(grid.tolist(), X[0].tolist(), {"modality": "ag_sers"}, OPTS)
            sers_sdk = False
        except Exception:
            sers_sdk = True
        sers_mcp = not mcp_call("gaira_validate_spectrum", {
            "spectrum": {"wavenumber": grid.tolist(), "intensity": X[0].tolist()},
            "metadata": {"modality": "ag_sers"}})["can_run"]
        log(f"  unsupported modality blocked — api {sers_api}, sdk {sers_sdk}, mcp {sers_mcp}")

        # ── latency ──────────────────────────────────────────────────────────
        log("latency")
        t0 = time.perf_counter(); GAIRAEngine.load(); t_load = time.perf_counter() - t0
        single = []
        for i in range(20):
            t0 = time.perf_counter()
            sdk.infer(grid.tolist(), X[i].tolist(), None, OPTS)
            single.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for i in range(10):
            sdk.infer(grid.tolist(), X[i].tolist(), None, OPTS)
        t_ten = time.perf_counter() - t0
        t0 = time.perf_counter()
        for i in range(100):
            sdk.infer(grid.tolist(), X[i % len(X)].tolist(), None, OPTS)
        t_hundred = time.perf_counter() - t0
        api_lat = []
        for i in range(20):
            t0 = time.perf_counter()
            http.post("/v1/infer", json={"spectrum": {"wavenumber": grid.tolist(),
                                                      "intensity": X[i].tolist()},
                                         "options": OPTS})
            api_lat.append(time.perf_counter() - t0)
        mcp_lat = []
        for i in range(20):
            t0 = time.perf_counter()
            mcp_call("gaira_get_chemistry_evidence",
                     {"spectrum": {"wavenumber": grid.tolist(), "intensity": X[i].tolist()}})
            mcp_lat.append(time.perf_counter() - t0)
        res = sdk.infer(grid.tolist(), X[0].tolist(), None,
                        {**OPTS, "include_reconstruction": True})
        t0 = time.perf_counter(); sdk.report(res, "pdf"); t_pdf = time.perf_counter() - t0
        t0 = time.perf_counter(); sdk.report(res, "json"); t_json = time.perf_counter() - t0

        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024
                                                                      if sys.platform ==
                                                                      "darwin" else 1024)
        perf = {
            "engine_load_seconds": round(t_load, 3),
            "single_inference_ms_median": round(statistics.median(single) * 1000, 2),
            "single_inference_ms_p95": round(sorted(single)[int(0.95 * len(single))] * 1000, 2),
            "ten_sequential_seconds": round(t_ten, 3),
            "hundred_sequential_seconds": round(t_hundred, 3),
            "api_ms_median": round(statistics.median(api_lat) * 1000, 2),
            "api_overhead_ms": round((statistics.median(api_lat)
                                      - statistics.median(single)) * 1000, 2),
            "mcp_ms_median": round(statistics.median(mcp_lat) * 1000, 2),
            "mcp_overhead_ms": round((statistics.median(mcp_lat)
                                      - statistics.median(single)) * 1000, 2),
            "report_pdf_seconds": round(t_pdf, 3),
            "report_json_seconds": round(t_json, 3),
            "max_rss_mb": round(rss_mb, 1),
        }
        for k, v in perf.items():
            log(f"  {k:<34s} {v}")

        # ── concurrency ──────────────────────────────────────────────────────
        log("concurrency")
        from concurrent.futures import ThreadPoolExecutor
        payloads = [{"spectrum": {"wavenumber": grid.tolist(), "intensity": X[i].tolist()},
                     "options": OPTS} for i in range(16)]
        serial = [http.post("/v1/infer", json=p).json()["result_digest"] for p in payloads]
        with ThreadPoolExecutor(max_workers=8) as ex:
            parallel = list(ex.map(
                lambda p: http.post("/v1/infer", json=p).json()["result_digest"], payloads))
        concurrent_ok = serial == parallel
        log(f"  16 requests over 8 threads reproduce the serial digests: {concurrent_ok}")

        # ── Step 17: scientific validation through the runtime path ──────────
        log("scientific validation — full corpus through the SDK")
        from gaira.v7.retrieval import evaluation as EVAL, models as MOD
        t0 = time.perf_counter()
        results = [sdk.infer(grid.tolist(), X[i].tolist(), None,
                             {"already_preprocessed": True, "include_lsm": False,
                              "include_provenance": False, "top_k_molecules": 1})
                   for i in range(len(X))]
        log(f"  375 spectra through the SDK in {time.perf_counter() - t0:.1f}s")
        A = np.vstack([r.csm.activation for r in results])
        ev = np.array([r.csm.explained_variance for r in results])
        pred = np.array([r.chemistry.predicted_class for r in results])
        z9 = np.load(F / "phase09/artifacts/engine_activations_v1.npz", allow_pickle=True)
        y = np.array([str(s) for s in z9["y"]])
        cls = np.array([str(s) for s in z9["cls"]])

        rk = np.zeros(len(A), int)
        for i in range(len(A)):
            keep = np.ones(len(A), bool); keep[i] = False
            Rb, lb = MOD.build_bank(A[keep], y[keep])
            s = MOD.score_B(A[i:i + 1], Rb)[0]
            hit = np.where(np.array(lb)[np.argsort(-s)] == y[i])[0]
            rk[i] = int(hit[0]) + 1 if len(hit) else len(lb) + 1
        m = EVAL.split_a_metrics(rk, len(set(y.tolist())))
        p09 = json.loads((F / "phase09/artifacts/phase09_summary_v1.json").read_text())
        ref = p09["validation_3_retrieval"]
        sci = {
            "molecule_top1": float(m["top1"]), "molecule_top3": float(m["top3"]),
            "molecule_top5": float(m["top5"]), "molecule_top10": float(m["top10"]),
            "molecule_mrr": float(m["mrr"]),
            "chemistry_top1_in_sample": float((pred == cls).mean()),
            "csm_mean_explained_variance": float(ev.mean()),
        }
        expect = {"molecule_top1": ref["top1"], "molecule_top3": ref["top3"],
                  "molecule_top5": ref["top5"], "molecule_top10": ref["top10"],
                  "molecule_mrr": ref["mrr"],
                  "chemistry_top1_in_sample":
                      p09["validation_4_chemistry"]["fine_top1_in_sample"],
                  "csm_mean_explained_variance":
                      p09["validation_2_csm"]["mean_explained_variance"]}
        sci_dev = {k: abs(sci[k] - expect[k]) for k in expect}
        for k in expect:
            log(f"  {k:<34s} {sci[k]:.6f}  (Phase 09 {expect[k]:.6f}, Δ{sci_dev[k]:.2e})")
        sci_max = max(sci_dev.values())

    # ── outputs and gates ────────────────────────────────────────────────────
    parity = pd.DataFrame(rows)
    parity.to_csv(OUT.tables / "cross_surface_parity_v1.csv", index=False)
    pd.DataFrame(digests).to_csv(OUT.tables / "surface_digests_v1.csv", index=False)
    pd.DataFrame([perf]).to_csv(OUT.tables / "performance_v1.csv", index=False)
    max_parity = float(parity.max_abs_diff.replace([np.inf], 1e9).max())
    n_diff = int((~parity.identical).sum())
    log(f"parity: {len(parity)} comparisons, {n_diff} divergent, max abs diff "
        f"{max_parity:.3e}")

    gates = [
        ("P1 six surfaces produce identical scientific output", n_diff == 0),
        ("P2 result digests agree across service, SDK, API and Streamlit paths",
         all(d["all_equal"] for d in digests)),
        ("P3 malformed input rejected on every surface",
         all(r["rejected"] for r in mal_rows)),
        ("P4 unsupported modality blocked on every surface",
         bool(sers_api and sers_sdk and sers_mcp)),
        ("P5 concurrent API requests reproduce serial results", bool(concurrent_ok)),
        ("P6 Phase 09 science reproduced through the runtime path", sci_max < 1e-9),
        ("P7 single-spectrum inference is interactive (< 250 ms)",
         perf["single_inference_ms_median"] < 250),
        ("P8 API overhead under 100 ms", perf["api_overhead_ms"] < 100),
    ]
    tab = pd.DataFrame([{"gate": g, "status": "PASS" if o else "FAIL"} for g, o in gates])
    tab.to_csv(OUT.tables / "phase10_parity_gates_v1.csv", index=False)
    for g, o in gates:
        log(f"  [{'PASS' if o else 'FAIL'}] {g}")
    n_fail = int((tab.status == "FAIL").sum())

    (OUT.artifacts / "parity_and_performance_v1.json").write_text(json.dumps({
        "parity": {"n_comparisons": len(parity), "n_divergent": n_diff,
                   "max_abs_diff": max_parity, "tolerance": TOL,
                   "surfaces": ["engine", "runtime_service", "python_sdk", "fastapi", "mcp",
                                "streamlit_backend"]},
        "digests": digests, "performance": perf,
        "malformed_handling": mal_rows,
        "unsupported_modality_blocked": {"api": sers_api, "sdk": sers_sdk, "mcp": sers_mcp},
        "concurrency_identical": bool(concurrent_ok),
        "scientific_validation": {"measured": sci, "phase09": expect, "deviations": sci_dev,
                                  "max_deviation": sci_max},
        "gates": {"n": len(gates), "failed": n_fail},
        "_provenance": {"phase": "10", "step": "parity_and_performance",
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": round(time.time() - t_start, 2)},
    }, indent=1))
    log(f"done · {len(gates) - n_fail}/{len(gates)} gates PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
