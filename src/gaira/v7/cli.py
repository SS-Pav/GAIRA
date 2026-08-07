"""GAIRA V7 — Phase 10: the command-line interface.

    gaira info                      engine metadata and validated performance
    gaira validate sample.csv       input checks without running inference
    gaira infer sample.csv          inference; --json, --report FILE
    gaira compare a.csv b.csv       two spectra through the engine independently
    gaira serve                     the FastAPI service
    gaira streamlit                 the Streamlit client
    gaira mcp                       the MCP tool server

Everything routes through `GAIRAService`. The CLI computes nothing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from gaira.v7 import __version__


def _fmt_diag(d) -> str:
    mark = {"error": "ERROR  ", "warning": "WARNING", "info": "info   "}[d.severity.value]
    return f"  {mark} [{d.code}] {d.message}"


def _cmd_info(a) -> int:
    from gaira.v7 import GAIRA
    i = GAIRA.load().engine_info()
    if a.json:
        print(json.dumps(i.model_dump(mode="json"), indent=2)); return 0
    print(f"GAIRA {i.gaira_version} — {i.engine_version}")
    print(f"  atlas fingerprint  {i.atlas_fingerprint}")
    for k, v in i.fingerprints.items():
        print(f"  frozen {k:<8s}    {v}")
    print(f"  frozen assets verified: {i.frozen_assets_verified}")
    print(f"  atlas: {i.n_lsms} LSMs · {i.n_csms} CSMs · {i.n_molecules} molecules · "
          f"{i.n_chemistry_axes} chemistry axes")
    print(f"  grid: {i.grid['low_cm']:.0f}–{i.grid['high_cm']:.0f} cm-1, "
          f"{i.grid['step_cm']:.1f} step, {int(i.grid['n_bins'])} bins")
    print(f"  corpus: {i.corpus['n_spectra']} spectra · "
          f"{i.corpus['n_canonical_molecules']} molecules · {i.corpus['scope']}")
    p = i.validated_performance
    print(f"  validated: molecule top-1 {p['molecule_top1']:.4f} · top-5 "
          f"{p['molecule_top5']:.4f} · MRR {p['molecule_mrr']:.4f}")
    print(f"             chemistry top-1 (held out) {p['chemistry_top1_heldout']:.4f} · "
          f"top-3 {p['chemistry_top3_heldout']:.4f}")
    print(f"  supported modalities: {', '.join(i.supported_modalities)}")
    print("  known limitations:")
    for lim in i.known_limitations:
        print(f"    — {lim}")
    return 0


def _cmd_validate(a) -> int:
    from gaira.v7 import GAIRA
    g = GAIRA.load()
    x, y, diags = g.read(a.path)
    v = g.validate(x, y, {"modality": a.modality, "sample_type": a.sample_type})
    all_d = list(diags) + list(v.diagnostics)
    if a.json:
        print(json.dumps({"can_run": v.can_run, "n_points": v.n_points,
                          "range_cm": v.range_cm, "grid_coverage": v.grid_coverage,
                          "diagnostics": [d.model_dump(mode="json") for d in all_d]}, indent=2))
        return 0 if v.can_run else 2
    print(f"{Path(a.path).name}: {v.n_points} points, "
          f"{(v.range_cm or (0, 0))[0]:.0f}–{(v.range_cm or (0, 0))[1]:.0f} cm-1, "
          f"coverage {(v.grid_coverage or 0):.1%}")
    for d in all_d:
        print(_fmt_diag(d))
    print(f"  → {'can run' if v.can_run else 'CANNOT RUN'}")
    return 0 if v.can_run else 2


def _cmd_infer(a) -> int:
    from gaira.v7 import GAIRA
    from gaira.v7.sdk import SpectrumRejected
    g = GAIRA.load()
    md = {"modality": a.modality, "sample_type": a.sample_type, "sample_id": a.sample_id,
          "source_name": Path(a.path).name}
    if a.excitation:
        md["excitation_nm"] = a.excitation
    try:
        r = g.infer_file(a.path, md, {"top_k_molecules": a.top_k,
                                      "include_reconstruction": bool(a.report)})
    except SpectrumRejected as rejected:
        print(f"spectrum rejected: {rejected}", file=sys.stderr)
        for d in rejected.validation.diagnostics:
            print(_fmt_diag(d), file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r.model_dump(mode="json"), indent=2))
    else:
        print(f"GAIRA V7 — {Path(a.path).name}   digest {r.result_digest[:16]}")
        print(f"\nChemistry Evidence (relative — not a concentration)")
        for ax in r.chemistry.top:
            bar = "█" * max(1, int(round(ax.share * 40)))
            print(f"  {ax.axis:<28s} {ax.share:6.1%}  {bar}")
        print(f"\nGrounded Evidence Retrieval (reference analogues, not identifications)")
        for h in r.retrieval.top[:5]:
            print(f"  {h.rank}. {h.molecule:<38s} {h.similarity:.4f}  [{h.chemistry_class}]")
        print(f"\nConfidence {r.confidence.overall:.4f} · CSM explained variance "
              f"{r.confidence.reconstruction_explained_variance:.4f} · "
              f"margin {r.confidence.retrieval_margin:.4f}")
        if r.confidence.unknown_warning or r.confidence.outlier_warning:
            flags = [n for n, f in (("unknown", r.confidence.unknown_warning),
                                    ("outlier", r.confidence.outlier_warning)) if f]
            print(f"  warnings: {', '.join(flags)}")
        for d in r.diagnostics:
            if d.severity.value in ("warning", "error"):
                print(_fmt_diag(d))
        print(f"\n{r.interpretation}")
    if a.report:
        out = Path(a.report)
        fmt = out.suffix.lstrip(".").lower() or "pdf"
        payload = g.report(r, fmt=fmt)
        out.write_bytes(payload if isinstance(payload, bytes) else payload.encode("utf-8"))
        print(f"\nreport written: {out}", file=sys.stderr)
    return 0


def _cmd_compare(a) -> int:
    from gaira.v7 import GAIRA
    g = GAIRA.load()
    xa, ya, _ = g.read(a.a)
    xb, yb, _ = g.read(a.b)
    c = g.compare((xa, ya), (xb, yb), label_a=Path(a.a).stem, label_b=Path(a.b).stem)
    if a.json:
        print(json.dumps(c.model_dump(mode="json"), indent=2)); return 0
    print(f"{c.label_a}  vs  {c.label_b}")
    print(f"  CSM cosine        {c.csm_cosine:.4f}")
    print(f"  chemistry cosine  {c.chemistry_cosine:.4f}")
    print(f"  top-10 overlap    {c.rank_agreement:.2f} "
          f"({len(c.shared_top_molecules)} shared)")
    print("\n  largest chemistry differences")
    for d in sorted(c.chemistry_delta, key=lambda d: -abs(d.delta))[:6]:
        print(f"    {d.axis:<28s} {d.a:.4f} → {d.b:.4f}   {d.delta:+.4f}")
    print(f"\n{c.interpretation}")
    return 0


def _cmd_serve(a) -> int:
    import uvicorn
    uvicorn.run("gaira.v7.api.app:app", host=a.host, port=a.port, log_level=a.log_level)
    return 0


def _cmd_mcp(a) -> int:
    from gaira.v7.mcp.__main__ import main as mcp_main
    return mcp_main([])


def _cmd_streamlit(a) -> int:
    from gaira.v7.io import repo_root
    app = repo_root() / "streamlit_apps" / "gaira_v7_console.py"
    if not app.exists():
        print(f"Streamlit app not found at {app}", file=sys.stderr); return 1
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           "--server.port", str(a.port), "--server.address", a.host]
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gaira", description=(
        "GAIRA V7 — Grounded Raman Biochemical Inference. Project a Raman spectrum into a "
        "frozen biochemical motif atlas, retrieve grounded reference evidence, and obtain an "
        "interpretable Chemistry Evidence profile."))
    p.add_argument("--version", action="version", version=f"GAIRA {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--modality", default="raman",
                        help="raman (the only supported value); others are rejected with a "
                             "scope error rather than run silently")
        sp.add_argument("--sample-type", dest="sample_type", default="pure",
                        help="pure | mixture | serum | plasma | EV | bacteria | tissue | other; "
                             "recorded as metadata, never applied to the calculation")
        sp.add_argument("--json", action="store_true", help="machine-readable output")

    s = sub.add_parser("info", help="engine metadata and validated performance")
    s.add_argument("--json", action="store_true"); s.set_defaults(func=_cmd_info)

    s = sub.add_parser("validate", help="check an input spectrum without running inference")
    s.add_argument("path"); common(s); s.set_defaults(func=_cmd_validate)

    s = sub.add_parser("infer", help="run the frozen engine on a spectrum file")
    s.add_argument("path")
    s.add_argument("--top-k", type=int, default=10, dest="top_k")
    s.add_argument("--sample-id", dest="sample_id", default=None)
    s.add_argument("--excitation", type=float, default=None, help="excitation wavelength in nm")
    s.add_argument("--report", default=None, metavar="FILE",
                   help="also write a report; format from the extension (.pdf/.html/.json)")
    common(s); s.set_defaults(func=_cmd_infer)

    s = sub.add_parser("compare", help="run two spectra independently and compare")
    s.add_argument("a"); s.add_argument("b"); common(s); s.set_defaults(func=_cmd_compare)

    s = sub.add_parser("serve", help="run the FastAPI service")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    s.add_argument("--log-level", dest="log_level", default="info")
    s.set_defaults(func=_cmd_serve)

    s = sub.add_parser("streamlit", help="run the Streamlit scientific client")
    s.add_argument("--host", default="localhost"); s.add_argument("--port", type=int, default=8501)
    s.set_defaults(func=_cmd_streamlit)

    s = sub.add_parser("mcp", help="run the MCP tool server on stdio")
    s.set_defaults(func=_cmd_mcp)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
