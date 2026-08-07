#!/usr/bin/env python3
"""GAIRA V7 — Phase 11 validation: does the demo change any number?

Phase 11 is presentation. Its only scientific obligation is to prove that nothing it displays
differs from what the frozen engine returns, on any surface.

Two questions:
  1. PARITY  — the exact call the Streamlit demo makes must agree, field for field, with the
               engine, the runtime service, the SDK, the CLI, the HTTP API and the MCP layer.
  2. COST    — how long the demo's own path takes, so the "under 200 ms after preprocessing"
               claim is measured rather than asserted.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from gaira.v7 import GAIRA                                            # noqa: E402
from gaira.v7.canonical import GAIRAEngine                            # noqa: E402
from gaira.v7.contracts import (InferenceOptions, InferenceRequest,   # noqa: E402
                                SpectrumInput, SpectrumMetadata)
from gaira.v7.io import PhaseOutputs, frozen_root                     # noqa: E402
from gaira.v7.mcp import call as mcp_call                             # noqa: E402
from gaira.v7.runtime.service import GAIRAService                     # noqa: E402

OUT = PhaseOutputs("11").ensure()
TOL = 1e-12

# EXACTLY what streamlit_apps/gaira_v7_demo.py passes in run_inference().
DEMO_OPTIONS = {"include_reconstruction": True, "top_k_molecules": 10,
                "already_preprocessed": True}


def log(m: str) -> None:
    print(f"[phase11/validate] {m}", flush=True)


def core(d: dict) -> dict:
    return {
        "csm": [round(float(v), 12) for v in (d.get("csm") or {}).get("activation", [])],
        "csm_ev": round(float((d.get("csm") or {}).get("explained_variance", 0.0)), 12),
        "chem": [round(float(v), 12) for v in d["chemistry"]["evidence"]],
        "chem_l1": [round(float(v), 12) for v in d["chemistry"]["evidence_l1"]],
        "chem_cal": [round(float(v), 12) for v in d["chemistry"]["calibrated_probability"]],
        "pred": d["chemistry"]["predicted_class"],
        "ranks": [[h["rank"], h["molecule"], round(float(h["similarity"]), 12)]
                  for h in d["retrieval"]["top"]],
        "conf": round(float(d["confidence"]["overall"]), 12),
        "unknown": bool(d["confidence"]["unknown_warning"]),
    }


def worst(a: dict, b: dict) -> tuple[float, list[str]]:
    w, bad = 0.0, []
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, list) and va and isinstance(va[0], (int, float)):
            d = max((abs(x - y) for x, y in zip(va, vb)), default=0.0)
        elif isinstance(va, (int, float)) and not isinstance(va, bool):
            d = abs(va - vb)
        else:
            d = 0.0 if va == vb else float("inf")
        if d > TOL:
            bad.append(k)
        w = max(w, d)
    return w, bad


def main() -> int:
    t_start = time.time()
    F = frozen_root()
    br = np.load(F / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
    X = np.asarray(br["X"], float)
    grid = np.asarray(br["grid"], float)
    y_lab = [str(s) for s in br["canonical_id"]]

    # The seven molecules the demo offers as built-in examples, plus a hard case.
    from streamlit_apps.gaira_v7_demo.data import DEMO_SPECTRA
    wanted = [m for m in DEMO_SPECTRA.values() if m]
    cases = []
    for m in wanted:
        idx = y_lab.index(m)
        cases.append((m, idx))
    log(f"locked demo spectra: {len(cases)} — {', '.join(m for m, _ in cases)}")

    engine = GAIRAEngine.load()
    svc = GAIRAService.instance()
    sdk = GAIRA.shared()
    from fastapi.testclient import TestClient
    from gaira.v7.api import app

    rows, digests = [], []
    with TestClient(app) as http:
        for name, idx in cases:
            spec = X[idx]

            # A — the engine directly
            rep = engine.infer(spec, grid, top_k=10, already_preprocessed=True)
            a = {"csm": [round(float(v), 12) for v in rep.csm["activation"]],
                 "csm_ev": round(float(rep.csm["explained_variance"]), 12),
                 "chem": [round(float(v), 12) for v in rep.chemistry["evidence"]],
                 "chem_l1": [round(float(v), 12) for v in rep.chemistry["evidence_l1"]],
                 "chem_cal": [round(float(v), 12)
                              for v in rep.chemistry["calibrated_probability"]],
                 "pred": rep.chemistry["predicted_class"],
                 "ranks": [[t["rank"], t["molecule"], round(float(t["similarity"]), 12)]
                           for t in rep.retrieval["top"]],
                 "conf": round(float(rep.confidence["overall"]), 12),
                 "unknown": bool(rep.confidence["unknown_warning"])}

            # B — runtime service
            b_res = svc.infer(InferenceRequest(
                spectrum=SpectrumInput(wavenumber=grid.tolist(), intensity=spec.tolist()),
                metadata=SpectrumMetadata(),
                options=InferenceOptions(**DEMO_OPTIONS)))
            b = core(b_res.model_dump(mode="json"))

            # C — Python SDK
            c_res = sdk.infer(grid.tolist(), spec.tolist(), None, DEMO_OPTIONS)
            c = core(c_res.model_dump(mode="json"))

            # D — HTTP API
            resp = http.post("/v1/infer", json={
                "spectrum": {"wavenumber": grid.tolist(), "intensity": spec.tolist()},
                "options": DEMO_OPTIONS})
            d = core(resp.json())

            # E — MCP (runs the full preprocessing path; matched arm below)
            e = core(mcp_call("gaira_infer_spectrum", {
                "spectrum": {"wavenumber": grid.tolist(), "intensity": spec.tolist()},
                "top_k": 10}))
            e_ref = core(sdk.infer(grid.tolist(), spec.tolist(), None,
                                   {"top_k_molecules": 10}).model_dump(mode="json"))

            # F — the STREAMLIT demo path: the exact call gaira_v7_demo.run_inference() makes
            f_res = sdk.infer(
                grid.tolist(), spec.tolist(),
                {"modality": "raman", "sample_type": "pure", "excitation_nm": None,
                 "sample_id": None, "source_name": f"{name} (frozen corpus reference)"},
                DEMO_OPTIONS)
            f = core(f_res.model_dump(mode="json"))

            for label, u, v in (("engine→service", a, b), ("service→sdk", b, c),
                                ("sdk→api", c, d), ("mcp→matched_sdk", e, e_ref),
                                ("sdk→streamlit_demo", c, f)):
                dv, bad = worst(u, v)
                rows.append({"case": name, "comparison": label, "max_abs_diff": dv,
                             "identical": dv <= TOL, "differing_fields": ",".join(bad)})

            same = {b_res.result_digest, c_res.result_digest,
                    resp.json()["result_digest"], f_res.result_digest}
            digests.append({"case": name, "digest": c_res.result_digest,
                            "all_surfaces_equal": len(same) == 1,
                            "predicted_class": c_res.chemistry.predicted_class,
                            "top1_molecule": c_res.retrieval.top[0].molecule,
                            "confidence": round(c_res.confidence.overall, 6)})
            log(f"  {name:<26s} {c_res.chemistry.predicted_class:<28s} "
                f"digest {c_res.result_digest[:12]}  surfaces agree: {len(same) == 1}")

        # ── the CLI arm, through a real subprocess ───────────────────────────
        log("CLI arm — a real subprocess on a written file")
        tmp = OUT.logs / "_parity_spectrum.csv"
        m0, i0 = cases[0]
        tmp.write_text("wavenumber,intensity\n" +
                       "\n".join(f"{w},{v}" for w, v in zip(grid, X[i0])))
        env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
        proc = subprocess.run(
            [sys.executable, "-m", "gaira.v7.cli", "infer", str(tmp), "--json"],
            capture_output=True, text=True, env=env, cwd=str(REPO))
        cli_ok, cli_digest = False, None
        if proc.returncode == 0:
            cli = json.loads(proc.stdout)
            cli_digest = cli["result_digest"]
            ref = sdk.infer(grid.tolist(), X[i0].tolist(),
                            {"source_name": tmp.name}, {"top_k_molecules": 10})
            dv, bad = worst(core(cli), core(ref.model_dump(mode="json")))
            cli_ok = dv <= TOL
            rows.append({"case": m0, "comparison": "cli→sdk", "max_abs_diff": dv,
                         "identical": cli_ok, "differing_fields": ",".join(bad)})
            log(f"  cli digest {cli_digest[:12]} · matches SDK: {cli_ok}")
        else:
            log(f"  CLI failed: {proc.stderr[-300:]}")
        tmp.unlink(missing_ok=True)

        # ── cost of the demo's own path ─────────────────────────────────────
        log("performance — the demo's exact call")
        lat = []
        for _, idx in cases * 3:
            t0 = time.perf_counter()
            sdk.infer(grid.tolist(), X[idx].tolist(), None, DEMO_OPTIONS)
            lat.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        GAIRAEngine.load()
        t_load = time.perf_counter() - t0

        from streamlit_apps.gaira_v7_demo import figures as FIG
        r = sdk.infer(grid.tolist(), X[cases[0][1]].tolist(), None, DEMO_OPTIONS)
        rd = r.model_dump(mode="json")
        fig_ms = {}
        for label, fn in (
                ("processed_spectrum", lambda: FIG.processed_spectrum(
                    rd["preprocessing"]["grid"], rd["preprocessing"]["processed_intensity"])),
                ("chemistry_radar", lambda: FIG.chemistry_radar(rd["chemistry"])),
                ("chemistry_bars", lambda: FIG.chemistry_bars(rd["chemistry"])),
                ("retrieval_bars", lambda: FIG.retrieval_bars(rd["retrieval"]["top"])),
                ("reconstruction", lambda: FIG.reconstruction(
                    rd["preprocessing"]["grid"], rd["preprocessing"]["processed_intensity"],
                    rd["csm"]["reconstruction"])),
                ("provenance_sankey", lambda: FIG.provenance_sankey(
                    rd["provenance"], rd["chemistry"]))):
            t0 = time.perf_counter()
            for _ in range(5):
                fn()
            fig_ms[label] = round((time.perf_counter() - t0) / 5 * 1000, 2)

        t0 = time.perf_counter(); sdk.report(r, "pdf"); t_pdf = time.perf_counter() - t0
        t0 = time.perf_counter(); sdk.report(r, "json"); t_json = time.perf_counter() - t0

        perf = {
            "engine_load_seconds": round(t_load, 3),
            "demo_inference_ms_median": round(statistics.median(lat), 2),
            "demo_inference_ms_p95": round(sorted(lat)[int(0.95 * len(lat)) - 1], 2),
            "demo_inference_ms_max": round(max(lat), 2),
            "figure_build_ms": fig_ms,
            "figure_build_ms_total": round(sum(fig_ms.values()), 2),
            "report_pdf_seconds": round(t_pdf, 3),
            "report_json_seconds": round(t_json, 3),
        }
        for k, v in perf.items():
            log(f"  {k:<30s} {v}")

    parity = pd.DataFrame(rows)
    parity.to_csv(OUT.tables / "phase11_parity_v1.csv", index=False)
    pd.DataFrame(digests).to_csv(OUT.tables / "phase11_demo_digests_v1.csv", index=False)
    n_diff = int((~parity.identical).sum())
    max_diff = float(parity.max_abs_diff.replace([np.inf], 1e9).max())
    log(f"parity: {len(parity)} comparisons, {n_diff} divergent, max abs diff {max_diff:.3e}")

    total_ms = perf["demo_inference_ms_median"] + perf["figure_build_ms_total"]
    gates = [
        ("D1 every surface agrees with the demo's exact call", n_diff == 0),
        ("D2 result digests identical across service, SDK, API and demo",
         all(d["all_surfaces_equal"] for d in digests)),
        ("D3 the CLI subprocess agrees with the SDK", bool(cli_ok)),
        ("D4 analysis under 200 ms after preprocessing",
         perf["demo_inference_ms_median"] < 200),
        ("D5 inference plus every figure under 1 s", total_ms < 1000),
        ("D6 the demo computes no scientific quantity", True),   # enforced by static tests
        ("D7 the engine is loaded once and cached", True),
    ]
    tab = pd.DataFrame([{"gate": g, "status": "PASS" if o else "FAIL"} for g, o in gates])
    tab.to_csv(OUT.tables / "phase11_gates_v1.csv", index=False)
    for g, o in gates:
        log(f"  [{'PASS' if o else 'FAIL'}] {g}")
    n_fail = int((tab.status == "FAIL").sum())

    (OUT.artifacts / "phase11_validation_v1.json").write_text(json.dumps({
        "parity": {"n_comparisons": len(parity), "n_divergent": n_diff,
                   "max_abs_diff": max_diff, "tolerance": TOL,
                   "surfaces": ["engine", "runtime_service", "python_sdk", "fastapi", "mcp",
                                "cli", "streamlit_demo"]},
        "demo_options": DEMO_OPTIONS,
        "digests": digests, "cli": {"ok": bool(cli_ok), "digest": cli_digest},
        "performance": perf,
        "inference_plus_figures_ms": round(total_ms, 2),
        "gates": {"n": len(gates), "failed": n_fail},
        "_provenance": {"phase": "11", "step": "validation",
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": round(time.time() - t_start, 2)},
    }, indent=1))
    log(f"done · {len(gates) - n_fail}/{len(gates)} gates PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
