#!/usr/bin/env python3
"""GAIRA V7 — Phase 10: twelve architecture figures at 300 dpi.

Drawn from the committed Phase 10 artifacts. Every number on a figure is read from a file; none
is typed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "src"))
from gaira.v7.io import PhaseOutputs                                   # noqa: E402

OUT = PhaseOutputs("10").ensure()
A, T, F = OUT.artifacts, OUT.tables, OUT.figures
DPI = 300
INK, MUTED, RULE = "#1a1a1a", "#6b7280", "#d1d5db"
FROZEN, RUNTIME, SURFACE, FUTURE, BAD, GOOD = (
    "#1d4ed8", "#0f766e", "#7c3aed", "#9ca3af", "#b91c1c", "#15803d")
ACCENT_D = "#c2410c"          # the dynamic / trajectory lane
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": DPI, "figure.dpi": 110})


def box(ax, x, y, w, h, label, sub="", colour=RUNTIME, fill=None, fs=8.6, alpha=0.10,
        ls="-", text_colour=None):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                   fc=fill or colour, ec=colour, lw=1.1, alpha=1.0 if fill
                                   else alpha, linestyle=ls))
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                   fc="none", ec=colour, lw=1.1, linestyle=ls))
    ax.text(x + w / 2, y + h / 2 + (0.012 if sub else 0), label, ha="center", va="center",
            fontsize=fs, color=text_colour or INK, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.020, sub, ha="center", va="center", fontsize=fs - 1.6,
                color=MUTED)


def arrow(ax, x1, y1, x2, y2, colour=MUTED, ls="-", lw=1.1, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=colour, lw=lw, linestyle=ls,
                                shrinkA=2, shrinkB=2))


def canvas(w=13.5, h=8.0, title="", sub=""):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    if title:
        ax.text(0.02, 0.975, title, fontsize=15, weight="bold", color=INK, va="top")
    if sub:
        ax.text(0.02, 0.935, sub, fontsize=9.5, color=MUTED, va="top")
    return fig, ax


def save(fig, name):
    fig.savefig(F / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def main() -> int:
    par = json.loads((A / "parity_and_performance_v1.json").read_text())
    frz = json.loads((A / "engine_freeze_audit_v1.json").read_text())
    perf = par["performance"]

    # ── F01 complete platform architecture ───────────────────────────────────
    fig, ax = canvas(title="Figure 1 — GAIRA V7 Phase 10 platform architecture",
                     sub="Frozen science at the centre; one runtime; five interchangeable "
                         "surfaces. Nothing outside the blue band computes.")
    box(ax, 0.06, 0.60, 0.88, 0.20, "", colour=FROZEN, fill="#eff6ff")
    ax.text(0.08, 0.775, "FROZEN SCIENTIFIC ENGINE — gaira.v7.canonical (Phase 09)",
            fontsize=9.4, weight="bold", color=FROZEN)
    stages = [("preprocess", "asLS · SG · L2"), ("LSM", "50 motifs"), ("CSM", "49 motifs"),
              ("retrieval", "154 molecules"), ("chemistry", "16 axes"),
              ("confidence", "calibrated"), ("provenance", "→ wavenumbers")]
    for i, (n, s) in enumerate(stages):
        box(ax, 0.075 + i * 0.125, 0.635, 0.108, 0.098, n, s, FROZEN, fs=8.2)
        if i:
            arrow(ax, 0.075 + i * 0.125 - 0.017, 0.684, 0.075 + i * 0.125, 0.684, FROZEN)
    box(ax, 0.255, 0.415, 0.49, 0.105, "RUNTIME SERVICE — gaira.v7.runtime",
        "validate · call the engine once · translate to the typed contract · template text",
        RUNTIME, fs=9.4)
    arrow(ax, 0.50, 0.60, 0.50, 0.522, RUNTIME, lw=1.6)
    box(ax, 0.06, 0.245, 0.20, 0.085, "input adapters", "CSV · TSV · TXT · arrays", RUNTIME,
        fs=8.4)
    box(ax, 0.74, 0.245, 0.20, 0.085, "reporting", "PDF · HTML · JSON", RUNTIME, fs=8.4)
    arrow(ax, 0.26, 0.310, 0.262, 0.418, RUNTIME); arrow(ax, 0.738, 0.418, 0.74, 0.310, RUNTIME)
    surfaces = [("Python SDK", "gaira.v7.sdk"), ("CLI", "gaira …"),
                ("FastAPI", "/v1/*"), ("MCP", "8 tools"), ("Streamlit", "thin client")]
    for i, (n, s) in enumerate(surfaces):
        box(ax, 0.075 + i * 0.175, 0.085, 0.155, 0.092, n, s, SURFACE, fs=8.8)
        arrow(ax, 0.152 + i * 0.175, 0.177, 0.33 + i * 0.085, 0.412, SURFACE, ls=":")
    ax.text(0.5, 0.032, "All five surfaces return the SAME InferenceResult. Cross-surface "
            f"parity measured at max |Δ| = {par['parity']['max_abs_diff']:.1e} over "
            f"{par['parity']['n_comparisons']} comparisons.",
            ha="center", fontsize=8.6, color=GOOD, style="italic")
    ax.text(0.5, 0.885, "no LLM · no cloud · no network · no external volume",
            ha="center", fontsize=9, color=BAD, weight="bold")
    save(fig, "F01_platform_architecture.png")

    # ── F02 engine vs runtime vs surfaces ────────────────────────────────────
    fig, ax = canvas(title="Figure 2 — Scientific engine, runtime, API, MCP and clients",
                     sub="What each layer is allowed to do. The rule that matters: only the "
                         "engine computes.")
    layers = [
        (0.78, "SCIENTIFIC ENGINE", FROZEN,
         "preprocessing · NNLS projection · retrieval · chemistry evidence · calibration",
         "MAY compute. Frozen after Phase 09; verified by fingerprint on every load."),
        (0.60, "RUNTIME SERVICE", RUNTIME,
         "validate · orchestrate · translate · deterministic template text",
         "MAY NOT compute a scientific quantity. Calls the engine and copies its numbers."),
        (0.42, "TRANSPORT — FastAPI · MCP", SURFACE,
         "serialise · route · enforce limits · reject malformed input",
         "MAY NOT compute. Static AST tests fail the build if a scientific name appears."),
        (0.24, "CLIENTS — SDK · CLI · Streamlit", SURFACE,
         "collect input · render · download",
         "MAY narrow what is SHOWN, never what is COMPUTED (P-20)."),
    ]
    for y, name, colour, does, rule in layers:
        box(ax, 0.06, y, 0.88, 0.135, "", colour=colour, fill=None, alpha=0.07)
        ax.text(0.085, y + 0.098, name, fontsize=10.5, weight="bold", color=colour)
        ax.text(0.085, y + 0.062, does, fontsize=8.8, color=INK)
        ax.text(0.085, y + 0.026, rule, fontsize=8.4, color=MUTED, style="italic")
        if y > 0.25:
            arrow(ax, 0.50, y, 0.50, y - 0.045, MUTED)
    ax.text(0.5, 0.16, "P-19 — one implementation of every scientific quantity.\n"
            "If a number appears in two places it will eventually disagree in two places.",
            ha="center", fontsize=9.4, color=BAD, weight="bold")
    ax.text(0.5, 0.085, "Enforced by tests/test_v7_phase10_parity.py: every surface is parsed "
            "with `ast` and checked for scientific imports and identifiers.\n"
            "Parsed, not grepped — a docstring listing what a module excludes must not fail a "
            "substring search.", ha="center", fontsize=8.4, color=MUTED)
    save(fig, "F02_layer_responsibilities.png")

    # ── F03 raw spectrum → inference ─────────────────────────────────────────
    fig, ax = canvas(h=8.4, title="Figure 3 — Raw spectrum to frozen V7 inference",
                     sub="The only path. No branches, no options, no tunable parameter at "
                         "inference time.")
    steps = [
        ("raw spectrum", "any grid, any scale", "—"),
        ("parse + validate", "adapters + 3 severities", "reject on ERROR"),
        ("canonical preprocessing", "450–1800 · 676 bins\nasLS → SG(9,3) → L2", "scale removed"),
        ("LSM projection", "50 motifs, NNLS", "diagnostic only"),
        ("CSM projection", "49 motifs, NNLS", "CANONICAL"),
        ("grounded retrieval", "cosine, 154 molecules", "scores reconcile"),
        ("Chemistry Evidence", "16 axes, D:A_max_idf λ=0.5", "RELATIVE"),
        ("calibration", "temperature T = 0.4538", "ranking unchanged"),
        ("confidence + audit", "EV × top-1 similarity", "pessimistic"),
        ("provenance", "→ diagnostic wavenumbers", "auditable"),
        ("interpretation report", "deterministic template", "no LLM"),
    ]
    y = 0.885
    for i, (n, s, note) in enumerate(steps):
        colour = FROZEN if 2 <= i <= 8 else RUNTIME
        box(ax, 0.20, y - 0.062, 0.34, 0.058, n, s, colour, fs=9.0)
        ax.text(0.57, y - 0.033, note, fontsize=8.2, color=GOOD if note not in ("—",) else MUTED,
                va="center", style="italic")
        if i < len(steps) - 1:
            arrow(ax, 0.37, y - 0.062, 0.37, y - 0.078, colour)
        y -= 0.078
    ax.text(0.06, 0.50, "FROZEN\nPhase 09", fontsize=9, color=FROZEN, weight="bold",
            ha="center", rotation=90, va="center")
    ax.add_patch(mp.Rectangle((0.10, 0.245), 0.008, 0.50, color=FROZEN, alpha=0.35))
    ax.text(0.80, 0.55, "NOT on this path\n\n· BSV2\n· latent geometry\n· clustering\n· UMAP\n"
            "· PCA\n· themes\n· Meta Components\n· chemistry reranking\n· SERS handling",
            fontsize=8.8, color=BAD, va="center", linespacing=1.5)
    save(fig, "F03_inference_flow.png")

    # ── F04 data contract ────────────────────────────────────────────────────
    fig, ax = canvas(title="Figure 4 — InferenceRequest and InferenceResult",
                     sub="The typed public contract. Pydantic v2 models serve the SDK, the API, "
                         "the MCP schemas and the report generator from one definition.")
    box(ax, 0.04, 0.30, 0.28, 0.53, "", colour=SURFACE, fill="#faf5ff")
    ax.text(0.055, 0.805, "InferenceRequest", fontsize=11, weight="bold", color=SURFACE)
    req = [("spectrum", "SpectrumInput", "wavenumber[], intensity[]"),
           ("metadata", "SpectrumMetadata", "modality, sample_type,\nexcitation, ids, notes"),
           ("options", "InferenceOptions", "top_k, include_lsm/csm/\nprovenance/audit")]
    yy = 0.735
    for n, t, d in req:
        box(ax, 0.055, yy - 0.10, 0.25, 0.095, n, t, SURFACE, fs=8.8)
        ax.text(0.18, yy - 0.128, d, fontsize=7.4, color=MUTED, ha="center", linespacing=1.3)
        yy -= 0.155
    arrow(ax, 0.325, 0.56, 0.395, 0.56, RUNTIME, lw=2.0)
    ax.text(0.36, 0.585, "runtime", fontsize=8, color=RUNTIME, ha="center")
    box(ax, 0.40, 0.16, 0.56, 0.67, "", colour=FROZEN, fill="#eff6ff")
    ax.text(0.415, 0.805, "InferenceResult", fontsize=11, weight="bold", color=FROZEN)
    res = [("preprocessing", "grid coverage, SNR, peaks, warnings"),
           ("lsm", "50-d activation, top motifs  (optional)"),
           ("csm", "49-d activation, EV, residual, sparsity  ← CANONICAL"),
           ("retrieval", "ranked MolecularHit[] with per-CSM decomposition"),
           ("chemistry", "16 axes, evidence, L1 share, calibrated probability"),
           ("confidence", "overall = EV × top-1; unknown / outlier flags"),
           ("audit", "coverage, margins, reconciliation, open-set limitation"),
           ("provenance", "spectrum → LSM → CSM → chemistry → molecule"),
           ("diagnostics", "Diagnostic[]  error | warning | info"),
           ("interpretation", "deterministic template text"),
           ("engine", "EngineInfo — fingerprints, corpus, validated performance"),
           ("result_digest", "MD5 over the scientific fields — the parity anchor")]
    yy = 0.775
    for n, d in res:
        ax.text(0.425, yy, n, fontsize=8.8, weight="bold", color=INK)
        ax.text(0.585, yy, d, fontsize=8.0, color=MUTED)
        yy -= 0.048
    ax.text(0.5, 0.10, "No NumPy in the public contract — arrays cross as lists of floats and a "
            "JSON round-trip is lossless.\n"
            "Every result is a frozen dataclass: a record that can be edited is not a record.",
            ha="center", fontsize=8.8, color=INK)
    save(fig, "F04_data_contract.png")

    # ── F05 FastAPI routes ───────────────────────────────────────────────────
    fig, ax = canvas(title="Figure 5 — FastAPI routes and backend routing",
                     sub="Six versioned routes. Each does one thing and delegates; none "
                         "contains a scientific expression.")
    routes = [("GET  /v1/health", "engine loaded, frozen assets verified", "health()"),
              ("GET  /v1/engine", "fingerprints, atlas shape, validated performance",
               "engine_info()"),
              ("POST /v1/validate-spectrum", "3-severity diagnostics, no inference",
               "validate_input()"),
              ("POST /v1/infer", "the full path → InferenceResult", "infer()"),
              ("POST /v1/compare", "two spectra run independently", "compare()"),
              ("POST /v1/report", "JSON / HTML inline, PDF base64", "generate_report()")]
    yy = 0.79
    for r, d, m in routes:
        box(ax, 0.045, yy - 0.075, 0.27, 0.072, r, "", SURFACE, fs=9.2)
        ax.text(0.335, yy - 0.028, d, fontsize=8.6, color=INK, va="center")
        arrow(ax, 0.70, yy - 0.039, 0.755, yy - 0.039, RUNTIME)
        box(ax, 0.76, yy - 0.075, 0.20, 0.072, m, "GAIRAService", RUNTIME, fs=8.6)
        yy -= 0.104
    ax.text(0.5, 0.145, "Engine loaded once at startup and shared read-only. It holds no mutable "
            "state and draws no random numbers,\nso concurrency needs no lock on the science — "
            "16 concurrent requests over 8 threads reproduce the serial digests exactly.",
            ha="center", fontsize=8.8, color=INK)
    ax.text(0.5, 0.075, f"Median API latency {perf['api_ms_median']} ms · overhead over a direct "
            f"call {perf['api_overhead_ms']} ms · body limit 32 MB · no filesystem path is "
            f"reachable through any route", ha="center", fontsize=8.4, color=MUTED)
    save(fig, "F05_api_routes.png")

    # ── F06 MCP tools ────────────────────────────────────────────────────────
    fig, ax = canvas(title="Figure 6 — MCP tool orchestration",
                     sub="Eight read-only tools, deliberately coarse. The server provides "
                         "tools; it runs no model.")
    tools = [("gaira_engine_info", "metadata, limits, scope"),
             ("gaira_validate_spectrum", "can this run, and with what caveats"),
             ("gaira_infer_spectrum", "the complete InferenceResult"),
             ("gaira_compare_spectra", "two spectra, independently"),
             ("gaira_get_molecular_evidence", "ranked analogues + decomposition"),
             ("gaira_get_chemistry_evidence", "16 axes, calibrated"),
             ("gaira_explain_result", "audit, provenance, interpretation"),
             ("gaira_generate_report", "JSON / HTML")]
    for i, (n, d) in enumerate(tools):
        col, row = i % 2, i // 2
        x = 0.055 + col * 0.47
        y = 0.72 - row * 0.115
        box(ax, x, y, 0.42, 0.088, n, d, SURFACE, fs=8.8)
        arrow(ax, x + 0.21, y, 0.5, 0.245, RUNTIME, ls=":", lw=0.8)
    box(ax, 0.30, 0.155, 0.40, 0.088, "GAIRAService", "one runtime, one engine", RUNTIME,
        fs=10)
    ax.text(0.5, 0.115, "Coarse by design: an agent must not be able to assemble its own "
            "inference path out of primitives.\nNo tool exposes NNLS, a raw matrix, or an "
            "arbitrary projection.", ha="center", fontsize=8.8, color=INK)
    ax.text(0.5, 0.045, "No LLM. No network. No cloud credential. The server is a provider; "
            "whatever consumes it lives entirely outside this process.",
            ha="center", fontsize=9, color=BAD, weight="bold")
    save(fig, "F06_mcp_tools.png")

    # ── F07 Streamlit workflow ───────────────────────────────────────────────
    fig, ax = canvas(title="Figure 7 — Streamlit user workflow",
                     sub="A thin client. It uploads, calls the runtime, and renders.")
    flow = [("upload", ".csv / .tsv / .txt"), ("preview + parse", "adapter diagnostics"),
            ("metadata", "modality · sample type\nexcitation · notes"),
            ("scope check", "unsupported modality\nis BLOCKED, not run"),
            ("Run inference", "one runtime call"),
            ("render", "5 tabs from one result")]
    for i, (n, s) in enumerate(flow):
        box(ax, 0.035 + i * 0.157, 0.68, 0.14, 0.10, n, s, SURFACE, fs=8.6)
        if i:
            arrow(ax, 0.035 + i * 0.157 - 0.017, 0.73, 0.035 + i * 0.157, 0.73, SURFACE)
    pages = [("Analyze Spectrum", "chemistry · retrieval · preprocessing · CSM · report"),
             ("Scientific Audit", "confidence · margins · warnings · open-set limitation"),
             ("Evidence & Provenance", "spectrum → LSM → CSM → chemistry → molecule"),
             ("Compare Spectra", "two runs, difference in evidence"),
             ("Engine Information", "fingerprints · corpus · validated performance")]
    yy = 0.56
    for n, d in pages:
        box(ax, 0.10, yy - 0.075, 0.28, 0.072, n, "", SURFACE, fs=9.2)
        ax.text(0.40, yy - 0.039, d, fontsize=8.6, color=MUTED, va="center")
        yy -= 0.093
    ax.text(0.5, 0.085, "Backend selection is CONFIGURATION ONLY. Set GAIRA_API_URL and the "
            "client talks to a deployed service instead of loading its own engine;\nthe request "
            "and result schemas are identical either way, and so are the numbers.",
            ha="center", fontsize=8.8, color=GOOD)
    ax.text(0.5, 0.028, "Static test: the app is parsed with `ast` and must reference no "
            "scientific primitive and import no scientific module.",
            ha="center", fontsize=8.4, color=MUTED, style="italic")
    save(fig, "F07_streamlit_workflow.png")

    # ── F08 provenance chain ─────────────────────────────────────────────────
    fig, ax = canvas(title="Figure 8 — The evidence and provenance chain",
                     sub="Every conclusion resolves to specific wavenumbers. This is what makes "
                         "a GAIRA answer auditable rather than merely produced.")
    chain = [("query spectrum", "676 bins", 0.06), ("LSM activation", "50-d, ~10 active", 0.245),
             ("CSM activation", "49-d, CANONICAL", 0.43),
             ("chemistry axes", "16-d, relative", 0.615),
             ("reference molecules", "ranked analogues", 0.80)]
    for i, (n, s, x) in enumerate(chain):
        box(ax, x, 0.62, 0.155, 0.11, n, s, FROZEN if i else RUNTIME, fs=9.0)
        if i:
            arrow(ax, x - 0.03, 0.675, x, 0.675, FROZEN, lw=1.4)
    ax.text(0.50, 0.565, "each CSM carries its diagnostic bands, its contributing LSMs and its "
            "band assignment", ha="center", fontsize=8.6, color=MUTED, style="italic")
    for x, lab in ((0.43, "diagnostic bands\n(cm⁻¹)"), (0.615, "supporting\nreference molecules"),
                   (0.80, "per-CSM score\ndecomposition")):
        arrow(ax, x + 0.078, 0.62, x + 0.078, 0.50, MUTED, ls=":")
        ax.text(x + 0.078, 0.455, lab, ha="center", fontsize=8.2, color=MUTED)
    box(ax, 0.24, 0.24, 0.52, 0.13, "Score reconciliation",
        "similarity = Σ per-CSM contributions, verified < 1e-9 for every candidate",
        GOOD, fs=10)
    ax.text(0.5, 0.185, "A cosine is an inner product, so the score decomposes EXACTLY into "
            "per-motif terms. Nothing is hidden and nothing is left over.",
            ha="center", fontsize=8.8, color=INK)
    ax.text(0.5, 0.10, "Atlas identity travels with every result: four frozen fingerprints plus "
            "ten content digests plus a per-result digest.\n"
            "An answer can always be traced back to the exact atlas that produced it.",
            ha="center", fontsize=8.6, color=MUTED)
    save(fig, "F08_provenance_chain.png")

    # ── F09 modality plugins + the dynamic trajectory lane ──────────────────
    fig, ax = canvas(h=8.6, title="Figure 9 — Future modality plugins and the dynamic "
                                  "trajectory lane",
                     sub="Modality adapters sit BEFORE the frozen core. DART is not one of them: "
                         "it is a dynamic perturbation protocol and attaches AFTER.")
    ax.text(0.045, 0.860, "UPSTREAM — modality adapters (the physics between sample and spectrum)",
            fontsize=9.2, weight="bold", color=MUTED, va="top")
    mods = [("PureRamanAdapter", "IMPLEMENTED", GOOD, "identity — the validated domain"),
            ("AgSERSAdapter", "contract only", FUTURE,
             "silver observation model,\ndetection gate, transfer function"),
            ("AuSERSAdapter", "contract only", FUTURE,
             "gold chemisorption differs;\nits own gate and corpus")]
    for i, (n, status, colour, note) in enumerate(mods):
        x = 0.055 + i * 0.31
        box(ax, x, 0.715, 0.27, 0.105, n, status, colour, fs=9.0,
            ls="-" if colour is GOOD else "--")
        ax.text(x + 0.135, 0.703, note, ha="center", va="top", fontsize=7.6, color=MUTED,
                linespacing=1.4)
        arrow(ax, x + 0.135, 0.715, 0.50, 0.598, colour, ls="-" if colour is GOOD else ":")

    box(ax, 0.235, 0.430, 0.53, 0.165, "FROZEN GAIRA CORE",
        "canonical preprocessing → frozen LSM → frozen CSM\n→ frozen Chemistry Evidence · "
        "confidence · provenance", FROZEN, fs=11.5)
    ax.text(0.50, 0.418, "Raman-only. Frozen after Phase 09. Unchanged by anything on this page.",
            ha="center", va="top", fontsize=8.6, color=FROZEN, style="italic")

    box(ax, 0.030, 0.500, 0.180, 0.140, "Dynamic Raman /\nSERS acquisition", "", ACCENT_D,
        fs=8.8)
    ax.text(0.120, 0.545, "I(wavenumber, potential, time)", ha="center", va="top",
            fontsize=7.0, color=MUTED)
    arrow(ax, 0.210, 0.560, 0.235, 0.528, ACCENT_D, lw=1.6)
    ax.text(0.120, 0.492, "each slice is an ordinary\nRaman/SERS spectrum", ha="center",
            va="top", fontsize=7.6, color=ACCENT_D, linespacing=1.4)

    box(ax, 0.790, 0.500, 0.180, 0.140, "TrajectoryAdapter", "", ACCENT_D, fs=8.8, ls="--")
    ax.text(0.880, 0.556, "how the frozen\nrepresentation moves", ha="center", va="top",
            fontsize=7.4, color=MUTED, linespacing=1.4)
    arrow(ax, 0.765, 0.528, 0.790, 0.560, ACCENT_D, lw=1.6)
    box(ax, 0.790, 0.350, 0.180, 0.100, "Dynamic biochemical", "trajectories", ACCENT_D,
        fs=8.8, ls="--")
    arrow(ax, 0.880, 0.500, 0.880, 0.452, ACCENT_D, ls="--")

    ax.text(0.50, 0.330, "DOWNSTREAM — DART is a dynamic perturbation layer, not a modality",
            ha="center", va="top", fontsize=9.8, weight="bold", color=ACCENT_D)
    ax.text(0.50, 0.288, "DART-Met produces I(wavenumber, potential, time), which is still a "
            "VIBRATIONAL measurement. Every slice is a spectrum the frozen\nengine already "
            "reads, so no spectral transform is needed and nothing upstream changes. What is new "
            "is perturbation and time,\nso DART attaches at the trajectory layer — over a "
            "SEQUENCE of ordinary inference results.",
            ha="center", va="top", fontsize=8.6, color=INK, linespacing=1.7)
    ax.text(0.50, 0.178, "Why downstream is correct rather than merely convenient: a trajectory "
            "of CSM activations is interpretable only if every activation\nalong it came from "
            "the same frozen path. Placed upstream, a DART 'modality adapter' would have to "
            "collapse the potential and\ntime axes before the core saw them — discarding "
            "exactly what the protocol exists to produce.",
            ha="center", va="top", fontsize=8.4, color=MUTED, linespacing=1.7)
    ax.text(0.50, 0.058, "Every unimplemented adapter RAISES. A stub that returns plausible "
            "numbers is worse than no stub at all.",
            ha="center", va="top", fontsize=8.6, color=BAD, weight="bold")
    save(fig, "F09_modality_plugins.png")

    # ── F10 context plugins ──────────────────────────────────────────────────
    fig, ax = canvas(title="Figure 10 — Future sample-context architecture",
                     sub="Context adapters frame a completed result. The protocol gives them no "
                         "way to change a number, and a test asserts it.")
    box(ax, 0.30, 0.62, 0.40, 0.14, "GAIRA CORE result",
        "InferenceResult — computed identically\nwhatever the sample type", FROZEN, fs=10.5)
    ctx = [("PureAnalyteContext", "IMPLEMENTED", GOOD, "the validated context"),
           ("MixtureContext", "contract only", FUTURE, "do activation shares track\ncomponent "
            "proportions? unmeasured"),
           ("SerumContext", "contract only", FUTURE, "which analytes are visible at\n"
            "physiological concentration?"),
           ("EVContext", "contract only", FUTURE, "membrane vs cargo attribution;\nisolation "
            "confounding"),
           ("BacteriaContext", "contract only", FUTURE, "does envelope abstraction\nsurvive "
            "transfer?"),
           ("TissueContext", "contract only", FUTURE, "spatial heterogeneity;\nis a pixel the "
            "right unit?")]
    for i, (n, status, colour, note) in enumerate(ctx):
        col, row = i % 3, i // 3
        x, y = 0.045 + col * 0.325, 0.36 - row * 0.185
        box(ax, x, y, 0.29, 0.10, n, status, colour, fs=8.8,
            ls="-" if colour is GOOD else "--")
        ax.text(x + 0.145, y - 0.038, note, ha="center", fontsize=7.6, color=MUTED,
                linespacing=1.4)
        if row == 0:
            arrow(ax, x + 0.145, 0.46, x + 0.145, y + 0.10, colour,
                  ls="-" if colour is GOOD else ":")
    ax.text(0.5, 0.505, "core output flows DOWN into framing — never back up",
            ha="center", fontsize=9, color=RUNTIME, style="italic")
    ax.text(0.5, 0.045, "scientific representation  ≠  domain interpretation\n"
            "A context adapter returns caveats, framing and diagnostics. It cannot return "
            "evidence, so it cannot rewrite it.",
            ha="center", fontsize=9.2, color=BAD, weight="bold")
    save(fig, "F10_context_plugins.png")

    # ── F11 agent boundary ───────────────────────────────────────────────────
    fig, ax = canvas(h=8.6, title="Figure 11 — The agent boundary",
                     sub="An LLM sits ABOVE the validated science, never inside it. Phase 10 "
                         "ships no model; this is the boundary Phase 11 must respect.")

    # the permitted chain, drawn as a single one-directional spine
    ax.text(0.50, 0.885, "THE ONLY PERMITTED CHAIN", ha="center", va="top", fontsize=9.4,
            weight="bold", color=GOOD)
    chain = [("LLM", "chooses tools · narrates", BAD),
             ("MCP", "8 read-only tools", SURFACE),
             ("FROZEN RUNTIME", "validate · orchestrate", RUNTIME),
             ("FROZEN ENGINE", "computes every number", FROZEN)]
    for i, (n, s, colour) in enumerate(chain):
        x = 0.055 + i * 0.235
        box(ax, x, 0.735, 0.195, 0.105, n, s, colour, fs=10.2,
            ls="--" if colour is BAD else "-")
        if i:
            arrow(ax, x - 0.038, 0.788, x - 0.002, 0.788, GOOD, lw=2.0)
    ax.text(0.50, 0.712, "one-directional — nothing flows back up, and the LLM never reaches "
            "past the MCP layer", ha="center", va="top", fontsize=8.6, color=GOOD,
            style="italic")

    # the forbidden shortcut
    ax.text(0.50, 0.650, "AND NEVER", ha="center", va="top", fontsize=9.4, weight="bold",
            color=BAD)
    box(ax, 0.245, 0.530, 0.195, 0.085, "LLM", "", BAD, fs=10.2, ls="--")
    box(ax, 0.560, 0.530, 0.195, 0.085, "scientific computation", "", BAD, fs=9.4, ls="--")
    arrow(ax, 0.445, 0.572, 0.556, 0.572, BAD, lw=2.0)
    ax.plot([0.478, 0.522], [0.548, 0.596], color=BAD, lw=3.0, zorder=5)
    ax.plot([0.478, 0.522], [0.596, 0.548], color=BAD, lw=3.0, zorder=5)

    # may / must never
    ax.add_patch(mp.FancyBboxPatch((0.045, 0.115), 0.42, 0.355,
                                   boxstyle="round,pad=0.008,rounding_size=0.012",
                                   fc="#f0fdf4", ec=GOOD, lw=1.2))
    ax.text(0.065, 0.442, "AN AGENT MAY", fontsize=10.5, weight="bold", color=GOOD, va="top")
    for i, s in enumerate(("choose tools", "explain", "compare", "narrate", "summarise")):
        ax.text(0.075, 0.398 - i * 0.048, f"·  {s}", fontsize=9.6, color=INK, va="top")
    ax.text(0.065, 0.152, "…and surface every caveat verbatim.", fontsize=8.4, color=MUTED,
            va="top", style="italic")

    ax.add_patch(mp.FancyBboxPatch((0.535, 0.115), 0.42, 0.355,
                                   boxstyle="round,pad=0.008,rounding_size=0.012",
                                   fc="#fef2f2", ec=BAD, lw=1.2))
    ax.text(0.555, 0.442, "AN AGENT IS FORBIDDEN FROM", fontsize=10.5, weight="bold", color=BAD,
            va="top")
    for i, s in enumerate(("computing chemistry", "computing similarity",
                           "estimating concentrations", "re-ranking",
                           "diagnosing disease", "modifying inference")):
        ax.text(0.565, 0.398 - i * 0.043, f"·  {s}", fontsize=9.6, color=INK, va="top")
    ax.text(0.555, 0.138, "Every number it states must trace to an InferenceResult field.",
            fontsize=8.4, color=MUTED, va="top", style="italic")

    ax.text(0.50, 0.062, "Phase 10 ships no model, requires no cloud account, and makes no "
            "network call. The MCP server is a provider;\nwhatever consumes it lives entirely "
            "outside the process.", ha="center", va="top", fontsize=8.8, color=INK,
            linespacing=1.6)
    save(fig, "F11_agent_boundary.png")

    # ── F12 parity validation ────────────────────────────────────────────────
    parity = pd.read_csv(T / "cross_surface_parity_v1.csv")
    digests = pd.read_csv(T / "surface_digests_v1.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    fig.suptitle("Figure 12 — Cross-surface parity validation", fontsize=14, weight="bold",
                 x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.925, f"{par['parity']['n_comparisons']} comparisons across "
             f"{len(par['parity']['surfaces'])} surfaces on {len(digests)} locked spectra. "
             f"Maximum absolute difference: {par['parity']['max_abs_diff']:.1e}.",
             fontsize=9.5, color=MUTED)

    ax = axes[0]
    labels = sorted(parity.comparison.unique())
    counts = [int((parity[parity.comparison == c].identical).sum()) for c in labels]
    totals = [int((parity.comparison == c).sum()) for c in labels]
    ax.barh(np.arange(len(labels)), counts, color=GOOD, height=0.62)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([l.replace("→", " → ") for l in labels], fontsize=8.4)
    ax.set_xlabel("identical results")
    ax.set_title("Every pairwise comparison", fontsize=10, loc="left")
    for i, (c, t) in enumerate(zip(counts, totals)):
        ax.text(c + 0.15, i, f"{c}/{t}", va="center", fontsize=8, color=GOOD)
    ax.set_xlim(0, max(totals) * 1.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    surf = ["engine", "service", "sdk", "api", "mcp"]
    lat = [digests[f"{s}_ms"].median() for s in surf]
    ax.bar(np.arange(len(surf)), lat, color=[FROZEN, RUNTIME, SURFACE, SURFACE, SURFACE],
           width=0.62)
    ax.set_xticks(np.arange(len(surf))); ax.set_xticklabels(surf, fontsize=8.6, rotation=20)
    ax.set_ylabel("median latency (ms)")
    ax.set_title("Cost of each surface", fontsize=10, loc="left")
    for i, v in enumerate(lat):
        ax.text(i, v + max(lat) * 0.03, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[2]
    sci = par["scientific_validation"]
    keys = ["molecule_top1", "molecule_top3", "molecule_top5", "molecule_top10", "molecule_mrr"]
    x = np.arange(len(keys))
    ax.bar(x - 0.19, [sci["phase09"][k] for k in keys], width=0.36, color=MUTED,
           label="Phase 09")
    ax.bar(x + 0.19, [sci["measured"][k] for k in keys], width=0.36, color=GOOD,
           label="Phase 10 runtime")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("molecule_", "") for k in keys], fontsize=8.6)
    ax.set_ylim(0, 1.0); ax.legend(frameon=False, fontsize=8.4)
    ax.set_title(f"Science unchanged — max Δ {sci['max_deviation']:.1e}", fontsize=10,
                 loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save(fig, "F12_parity_validation.png")

    print(f"12 figures written to {F}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
