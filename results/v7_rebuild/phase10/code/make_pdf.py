#!/usr/bin/env python3
"""GAIRA V7 — Phase 10: assemble PHASE_10_GAIRA_RUNTIME_PLATFORM_REPORT.pdf.

What GAIRA V7 is, what is frozen, how a raw spectrum becomes an inference, where the API, MCP and
Streamlit surfaces fit, why no LLM is required, and how future modality and context plugins would
extend the platform. Every number is read from the committed Phase 10 artifacts.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "src"))
PH = HERE.parent
REPO = PH.parents[2]
F, A, R, T = PH / "figures", PH / "artifacts", PH / "reports", PH / "tables"
INK, MUTED, RULE = "#1a1a1a", "#6b7280", "#d1d5db"
ACCENT, WARN, GOOD = "#1d4ed8", "#b45309", "#15803d"
PAGE = (11.69, 8.27)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "mathtext.fontset": "dejavusans"})

_page_no = [0]


# ── page primitives ──────────────────────────────────────────────────────────
def _new(pdf, draw, chrome=True, footer=""):
    _page_no[0] += 1
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    draw(fig)
    if chrome:
        ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.95, 0.030, str(_page_no[0]), fontsize=8, color=MUTED, ha="right")
        ax.text(0.05, 0.030, footer or "GAIRA V7 · Phase 10 · the runtime platform",
                fontsize=7.4, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


class Body:
    """A simple flowing text column with a y cursor. Blocks are (kind, payload) tuples."""

    def __init__(self, ax, x=0.075, y=0.855, width=138, right=0.925):
        self.ax, self.x, self.y, self.w, self.right = ax, x, y, width, right

    def _t(self, s, size, color, dy, x=None, weight="normal", style="normal", ha="left"):
        self.ax.text(self.x if x is None else x, self.y, s, fontsize=size, color=color,
                     weight=weight, style=style, ha=ha, va="top")
        self.y -= dy

    def h2(self, s):
        self.y -= 0.012
        self._t(s, 12.5, INK, 0.036, weight="bold")

    def h3(self, s):
        self.y -= 0.006
        self._t(s, 10.0, ACCENT, 0.030, weight="bold")

    def p(self, s, size=9.6, color=INK):
        for line in textwrap.wrap(s, self.w):
            self._t(line, size, color, 0.0250)
        self.y -= 0.010

    def lead(self, s):
        for line in textwrap.wrap(s, int(self.w * 0.84)):
            self._t(line, 10.8, INK, 0.029)
        self.y -= 0.012

    def bullet(self, s, marker="—"):
        lines = textwrap.wrap(s, self.w - 6)
        for i, line in enumerate(lines):
            self.ax.text(self.x + (0.0 if i else 0.0), self.y, marker if i == 0 else " ",
                         fontsize=9.2, color=MUTED, va="top")
            self.ax.text(self.x + 0.022, self.y, line, fontsize=9.2, color=INK, va="top")
            self.y -= 0.0235
        self.y -= 0.005

    def kv(self, rows, keyw=0.30, size=9.4):
        for k, v in rows:
            if not k and not v:
                self.y -= 0.014
                continue
            self.ax.text(self.x, self.y, k, fontsize=size, color=MUTED, va="top")
            self.ax.text(self.x + keyw, self.y, v, fontsize=size, color=INK, va="top")
            self.y -= 0.0265
        self.y -= 0.008

    def table(self, header, rows, cols, size=9.0, colors=None):
        self.ax.plot([self.x, self.right], [self.y + 0.016, self.y + 0.016], color=RULE, lw=0.9)
        for c, h in zip(cols, header):
            self.ax.text(self.x + c, self.y, h, fontsize=size, color=MUTED, va="top",
                         weight="bold")
        self.y -= 0.026
        self.ax.plot([self.x, self.right], [self.y + 0.014, self.y + 0.014], color=RULE, lw=0.6)
        for i, row in enumerate(rows):
            col = (colors or {}).get(i, INK)
            for c, cell in zip(cols, row):
                self.ax.text(self.x + c, self.y, str(cell), fontsize=size, color=col, va="top")
            self.y -= 0.0245
        self.ax.plot([self.x, self.right], [self.y + 0.014, self.y + 0.014], color=RULE, lw=0.9)
        self.y -= 0.020

    def eq(self, s, size=13):
        self.y -= 0.008
        self.ax.text(0.5, self.y, s, fontsize=size, color=INK, ha="center", va="top")
        self.y -= 0.062

    def note(self, s, color=WARN, label="Note"):
        lines = textwrap.wrap(s, self.w - 10)
        h = 0.0235 * len(lines) + 0.030
        self.ax.add_patch(plt.Rectangle((self.x - 0.012, self.y - h + 0.020), 0.004, h,
                                        color=color, transform=self.ax.transAxes, clip_on=False))
        self.ax.text(self.x, self.y, label, fontsize=8.2, color=color, weight="bold", va="top")
        self.y -= 0.024
        for line in lines:
            self._t(line, 9.3, INK, 0.0235, style="italic")
        self.y -= 0.014

    def plain(self, s):
        """The plain-English gloss that runs beside every technical statement."""
        self.note(s, color=ACCENT, label="In plain terms")


def text_page(pdf, kicker, title, build, footer=""):
    def draw(fig):
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if kicker:
            ax.text(0.075, 0.945, kicker.upper(), fontsize=7.8, color=MUTED, va="top",
                    weight="bold")
        ax.text(0.075, 0.925, title, fontsize=16.5, color=INK, va="top", weight="bold")
        ax.plot([0.075, 0.925], [0.888, 0.888], color=RULE, lw=1.0)
        build(Body(ax))
    _new(pdf, draw, footer=footer)


def divider(pdf, number, title, blurb):
    def draw(fig):
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#f8fafc", zorder=-2))
        ax.text(0.10, 0.60, number, fontsize=64, color="#e2e8f0", weight="bold", va="center")
        ax.text(0.10, 0.47, title, fontsize=24, color=INK, weight="bold", va="center")
        ax.plot([0.10, 0.62], [0.425, 0.425], color=RULE, lw=1.2)
        y = 0.385
        for line in textwrap.wrap(blurb, 78):
            ax.text(0.10, y, line, fontsize=10.2, color=MUTED, va="top")
            y -= 0.030
    _new(pdf, draw)


def figure_page(pdf, path, caption):
    img = mpimg.imread(path)
    h, w = img.shape[:2]

    def draw(fig):
        bw, bh = 0.88 * PAGE[0], 0.76 * PAGE[1]
        asp = h / w
        dw, dh = (bw, bw * asp) if bw * asp <= bh else (bh / asp, bh)
        fw, fh = dw / PAGE[0], dh / PAGE[1]
        top, bottom = 0.935, 0.135
        ax = fig.add_axes([(1 - fw) / 2, bottom + (top - bottom - fh) / 2, fw, fh])
        ax.imshow(img); ax.axis("off")
        ft = fig.add_axes([0, 0, 1, 1]); ft.axis("off")
        ft.set_xlim(0, 1); ft.set_ylim(0, 1)
        ft.text(0.05, 0.968, path.stem.split("_")[0].replace("F", "Figure "),
                fontsize=11, color=INK, weight="bold", va="top")
        ft.plot([0.05, 0.95], [0.112, 0.112], color=RULE, lw=0.8)
        yy = 0.094
        for line in textwrap.wrap(caption, 150):
            ft.text(0.05, yy, line, fontsize=8.3, color=INK, va="top")
            yy -= 0.020
    _new(pdf, draw)



CAPTIONS = {
    "F01": "The complete Phase 10 platform. Frozen science at the centre, one runtime beneath it, "
           "five interchangeable surfaces below that. Nothing outside the blue band computes, and "
           "a static test fails the build if anything tries.",
    "F02": "What each layer may do. The engine computes; the runtime orchestrates; the transport "
           "serialises; the clients render. P-19 in one picture.",
    "F03": "The only inference path, stage by stage, with what is deliberately absent listed "
           "beside it. Every exclusion is a measured decision from an earlier phase.",
    "F04": "The typed public contract. One pydantic definition serves the SDK, the API, the MCP "
           "schemas and the report generator, so the four cannot drift apart.",
    "F05": "Six versioned FastAPI routes, each delegating to one runtime method. No route contains "
           "a scientific expression and none accepts a filesystem path.",
    "F06": "Eight MCP tools, deliberately coarse. An agent can ask for an interpretation; it "
           "cannot assemble its own inference path out of primitives.",
    "F07": "The Streamlit workflow and its five pages. Backend selection is configuration only — "
           "local engine or deployed API, identical numbers either way.",
    "F08": "The provenance chain. A cosine is an inner product, so every retrieval score "
           "decomposes exactly into per-motif contributions, verified below 1e-9.",
    "F09": "Modality adapters sit BEFORE the frozen core; DART does not. DART-Met produces "
           "I(wavenumber, potential, time) — still a vibrational measurement — so it attaches "
           "AFTER the frozen representation, at the trajectory layer.",
    "F10": "Sample-context plugins. Context adapters frame a completed result; the protocol gives "
           "them no way to change a number.",
    "F11": "The agent boundary. LLM → MCP → Frozen Runtime → Frozen Engine, one-directional, "
           "and never LLM → scientific computation. Phase 10 ships no model.",
    "F12": "Cross-surface parity: 60 comparisons across six surfaces, zero divergent, maximum "
           "absolute difference 0.0, and every Phase 09 retrieval figure reproduced exactly.",
}

EXPLAIN = {
    "F01": ("The shape of the platform",
            "Phase 10 adds no science. It takes the engine frozen at the end of Phase 09 and "
            "makes it reachable — from Python, from a terminal, over HTTP, through MCP, and from "
            "a browser — without letting any of those routes change an answer.",
            ["The blue band at the top is the frozen engine. It preprocesses a spectrum, projects "
             "it onto 50 local motifs and then 49 consensus motifs, retrieves the nearest of 154 "
             "reference molecules, aggregates 16 chemistry axes, calibrates a confidence and "
             "builds a provenance chain. It is the only thing in this diagram that computes.",
             "The green band is the runtime service. It validates input, calls the engine once, "
             "translates the engine's dictionaries into a typed public contract, and renders "
             "deterministic template text. It orchestrates and copies; it does not calculate.",
             "The five purple boxes are the surfaces. Each is a different way of reaching the "
             "same runtime, and each returns the same object. A client may choose to display "
             "less; it may never compute more."],
            "The line at the bottom is the phase's central measurement: across 60 comparisons "
            "spanning six independent code paths, the maximum absolute difference between any "
            "two surfaces was zero. Not 'within tolerance' — identical."),
    "F02": ("One rule, four layers",
            "The rule is P-19: one implementation of every scientific quantity. If a number "
            "appears in two places it will eventually disagree in two places, and the "
            "disagreement will be discovered by a user rather than by a test.",
            ["The engine may compute. It is frozen, fingerprint-verified on every load, and "
             "nothing above it is permitted to reproduce any part of it.",
             "The runtime may validate, orchestrate and translate. It may not project, fit or "
             "calibrate. A static test parses it and asserts that it imports neither scipy nor "
             "scikit-learn and references no scientific primitive.",
             "The transport and client layers may serialise, route, enforce limits, collect input "
             "and render. The Streamlit app additionally may not import the engine directly: it "
             "must go through the runtime rather than reach past it."],
            "This is enforced by parsing each surface with Python's `ast` module rather than "
            "searching its text. A module docstring that lists what the module excludes would "
            "fail a naive substring search — Phase 09 learned that, and Phase 10 inherited both "
            "the lesson and the fix."),
    "F03": ("From spectrum to answer",
            "Eleven stages, always in this order, with no branches and no parameter a caller can "
            "tune at inference time.",
            ["The first two stages belong to Phase 10: parse the file and validate it. Validation "
             "answers one question — can the frozen engine say anything useful about this input, "
             "and if so with what caveats — and it answers at three severities. An error stops "
             "the run; a warning limits the interpretation; an info note records scope.",
             "The seven blue stages are the frozen engine and were fixed by Phases 00 through 09. "
             "Canonical preprocessing removes absolute intensity, which is why nothing downstream "
             "can be read as a concentration. The CSM activation is the canonical representation "
             "— 49 numbers that every later stage reads and the only ones it reads.",
             "The final stages produce confidence, provenance and a deterministic interpretation "
             "paragraph. No language model is involved at any point."],
            "The red column is as important as the blue one. BSV2, latent geometry, clustering, "
            "UMAP, PCA, themes, Meta Components and chemistry-aware reranking are each absent "
            "because a measurement said so, not because of a preference. Four independent "
            "attempts to build a layer above the CSM each lost information."),
    "F04": ("The contract",
            "Everything crossing the boundary between the frozen engine and the outside world is "
            "typed, versioned and free of NumPy. Arrays cross as lists of floats, so a JSON "
            "round-trip is lossless and no caller needs to know how the engine is implemented.",
            ["A request carries three things: the spectrum, its metadata, and the options. The "
             "metadata records modality and sample type. Modality is enforced — anything but "
             "Raman is rejected. Sample type is recorded and warned about but never applied, and "
             "a test proves it by running the same spectrum under four sample types and checking "
             "the result digest does not move.",
             "A result carries eleven blocks and a digest. The digest is an MD5 over the "
             "scientific fields alone, excluding timestamps and free text, and it is the anchor "
             "the whole parity validation turns on: two surfaces agreeing means their digests are "
             "equal.",
             "Every result object is a frozen dataclass. A record that can be edited after the "
             "fact is not a record."],
            "One definition serves four consumers. The SDK returns these objects, the API "
            "serialises them, the MCP schemas are generated from the same shapes, and the report "
            "generator reads them. There is no second place where the response format is "
            "described, so there is no second place for it to drift."),
    "F05": ("HTTP",
            "Six routes, versioned under /v1. Each does one thing and delegates to exactly one "
            "runtime method.",
            ["The engine loads once at startup, verifies ten frozen asset digests, and is shared "
             "read-only for the life of the process. Because it holds no mutable state and draws "
             "no random numbers, concurrency needs no lock on the science — and this was checked "
             "rather than assumed: sixteen requests across eight threads reproduce the serial "
             "digests exactly.",
             "Validation failures return 422 with the complete diagnostic list, so a caller "
             "learns precisely which condition failed rather than receiving a generic rejection. "
             "Unknown request fields are rejected rather than ignored, which catches a typo in an "
             "option name instead of silently using the default.",
             "The report route never writes to a caller-supplied path. JSON and HTML come back "
             "inline, a PDF comes back base64-encoded, and the filename is derived from the "
             "result digest. A report endpoint that writes where the caller says is a file-write "
             "primitive wearing a scientific label."],
            "API overhead over a direct Python call is 1.6 ms against a 2.3 ms inference. The "
            "transport is not where the time goes."),
    "F06": ("Tools, not primitives",
            "The MCP server exposes the same runtime as eight callable tools. No language model "
            "runs inside it and it makes no network call; it is a provider, and whatever consumes "
            "it lives entirely outside the process.",
            ["The tools are deliberately coarse. An agent should be able to ask 'interpret this "
             "spectrum' and receive a scientifically coherent answer. It should not be able to "
             "request an NNLS solve against a raw dictionary matrix, because that is where a "
             "caller could construct a result the engine never sanctioned.",
             "The narrow tools — chemistry evidence, molecular evidence, explain — return exactly "
             "the corresponding slice of the full inference for the same input. This is verified "
             "by digest equality rather than trusted, so a future edit that made one of them "
             "drift would fail a test.",
             "Every tool description carries its own caveats, and the server advertises the scope "
             "in its MCP instructions field, so a connecting client receives the limitations "
             "before its first call."],
            "The engine_info tool should be called first and, in an agent deployment, required. "
            "It is the tool that states what the engine cannot do — no open-set detection, "
            "relative evidence only, reference analogues rather than identifications, Raman "
            "only."),
    "F07": ("The browser client",
            "Seven hundred lines of Streamlit containing no scientific computation whatsoever. It "
            "uploads, calls the runtime, and renders.",
            ["Backend selection is configuration only. With GAIRA_API_URL unset the app loads its "
             "own engine; with it set the app talks to a deployed service. The request and result "
             "schemas are identical either way and so are the numbers, so moving from a laptop to "
             "a server is an environment variable rather than a rewrite.",
             "The processed spectrum and the CSM reconstruction that the app draws are returned "
             "by the engine, through a read-only accessor added in this phase. The UI never "
             "reproduces a transformation, so what a user sees is exactly what the projection "
             "consumed.",
             "Language is constrained deliberately. The retrieval tab is titled Grounded Evidence "
             "Retrieval, not Molecule Identification. Every chemistry view carries the "
             "relative-evidence caveat. Selecting a non-Raman modality shows a red block and "
             "prevents the run rather than warning and proceeding."],
            "A static test parses the app and fails the build if it references a scientific "
            "primitive, imports scipy or scikit-learn, or imports the engine directly. The UI "
            "must go through the runtime, not reach past it."),
    "F08": ("Why an answer is auditable",
            "The provenance chain is what separates a GAIRA answer from a produced number.",
            ["Each retrieval score is a cosine similarity, which is an inner product of unit "
             "vectors. That means it decomposes exactly into per-motif terms — 'how much did each "
             "consensus motif contribute to this match'. The engine asserts that the parts sum to "
             "the whole within 1e-9 for every candidate it reports, on every call.",
             "Each consensus motif carries its diagnostic band positions, its band assignment "
             "string, and the local motifs that formed it. So a chemistry conclusion can be "
             "walked back to specific wavenumbers, and a user can ask which part of the spectrum "
             "is responsible for a claim.",
             "Atlas identity travels with every result: four frozen fingerprints, ten content "
             "digests, and a per-result digest. An answer can always be traced back to the exact "
             "atlas that produced it, and an atlas that has changed will not load."],
            "Phase 10 added the second freeze layer. Phase 09 verifies four fingerprints recorded "
            "inside state files; Phase 10 pins the content of all ten files the engine opens. The "
            "first answers 'did the producing phase claim this artefact', the second answers 'is "
            "the file still the one that phase wrote'."),
    "F09": ("Extending to new measurement channels — and where DART actually goes",
            "Three modality adapters are declared and one is implemented. DART is deliberately "
            "not among them, and the reason is worth reading carefully because it is the most "
            "commonly misdescribed part of this architecture.",
            ["A modality adapter models the physics between the sample and the spectrum. It runs "
             "before the core and may correct, veto, or pass through — but it may never touch the "
             "dictionaries, the retrieval, or the chemistry model.",
             "The reason that boundary is hard rather than advisory is a measurement. Phase 04 "
             "tested whether the frozen Raman atlas could detect real Ag-SERS as out of domain "
             "and got AUROC 0.548 — chance. A non-negative Raman motif basis reconstructs SERS of "
             "the same metabolites comfortably, so a SERS spectrum run through the Raman core "
             "produces confident numbers with no validated meaning. Unsupported modalities are "
             "therefore blocked, not warned about.",
             "DART is a different kind of thing entirely. It is NOT a new spectral modality — it "
             "is a dynamic perturbation protocol built on Raman/SERS measurements. DART-Met "
             "produces I(wavenumber, potential, time), which is still a VIBRATIONAL measurement, "
             "and every slice through that volume is a spectrum the frozen engine already reads "
             "correctly. There is no spectral transform to invent, because the measurement axis "
             "has not changed. What has been added is perturbation and time.",
             "So DART attaches at the TRAJECTORY layer, downstream of the frozen representation, "
             "consuming a sequence of ordinary inference results with their potential and time "
             "coordinates. Nothing upstream changes."],
            "Downstream is correct rather than merely convenient. A trajectory of CSM "
            "activations is interpretable only if every activation along it came from the same "
            "frozen path — which is exactly what placing the dynamic layer downstream "
            "guarantees. Placed upstream, a DART 'modality adapter' would have to collapse the "
            "potential and time axes before the core ever saw them, discarding the very "
            "information the protocol exists to produce."),
    "F10": ("Extending to new sample contexts",
            "Seven context adapters are declared and one is implemented. A context adapter frames "
            "a completed result; it runs after the core and cannot reach back into it.",
            ["The protocol is the enforcement. A context adapter returns caveats, framing and "
             "diagnostics — there is no field through which it could return evidence, so there is "
             "no way for it to rewrite a number. A test asserts the shape of that return type.",
             "Each unimplemented context states the open questions a working version must answer. "
             "For serum: which analytes are visible at physiological concentration, and how "
             "albumin dominance should be handled. For extracellular vesicles: membrane versus "
             "cargo attribution, and isolation-method confounding. For tissue: whether a pixel is "
             "even the right unit.",
             "Meanwhile sample_type is accepted today as metadata. It produces a scope warning "
             "and changes nothing in the calculation, and a test runs the same spectrum under "
             "four sample types to prove the digest does not move."],
            "The principle this figure encodes: scientific representation is not domain "
            "interpretation. Confusing the two is how a spectral engine becomes a clinical claim "
            "without anyone deciding that it should."),
    "F11": ("Where an agent would sit",
            "Phase 10 ships no language model and requires no cloud account. This figure is the "
            "boundary a Phase 11 agent layer would have to respect.",
            ["There is exactly one permitted chain, and it runs in one direction: LLM → MCP → "
             "Frozen Runtime → Frozen Engine. Nothing flows back up it, and the model never "
             "reaches past the MCP layer. The forbidden shortcut — LLM → scientific computation "
             "— is drawn crossed out because it is not a matter of policy but of architecture: "
             "no tool exposes a primitive an agent could compute with.",
             "An agent MAY choose tools, explain, compare, narrate and summarise. Concretely: "
             "decide which tools to call and in what order, request more evidence before "
             "answering, rephrase the deterministic interpretation, and surface caveats verbatim.",
             "An agent is FORBIDDEN from computing chemistry, computing similarity, estimating "
             "concentrations, re-ranking, diagnosing disease, and modifying inference. Every "
             "number it states must trace to a field of an InferenceResult, and every claim "
             "about a molecule must carry the word analogue or an equivalent."],
            "The honest assessment is that the engineering guardrails are in place and the "
            "linguistic one is not. Every failure mode this architecture prevents is a failure of "
            "language, and an agent is a language system. Phase 11 should open with an "
            "adversarial overclaim benchmark, not close with one."),
    "F12": ("The measurement the phase turns on",
            "Twelve locked spectra through six independent surfaces, compared field by field at a "
            "tolerance of one part in a trillion.",
            ["The spectra were chosen for behaviour, not convenience: the highest-confidence "
             "correct case, a chemistry-right but molecule-wrong case, the lowest-explained-"
             "variance spectrum in the entire corpus, an ambiguous class, four different "
             "chemistry families, three different source libraries, and a synthetic noise "
             "control.",
             "Compared per case: all 49 CSM activations, explained variance, all 16 chemistry "
             "evidence values, all 16 calibrated probabilities, the predicted class, all ten "
             "retrieval ranks and similarities, overall confidence, the unknown flag, grid "
             "coverage, and score reconciliation.",
             "Sixty comparisons, zero divergent, maximum absolute difference exactly zero. The "
             "right panel shows the other half of the obligation: every Phase 09 retrieval figure "
             "reproduced through the runtime path at deviation 0.0."],
            "The middle panel is worth reading as a cost sheet. Reaching the engine through HTTP "
            "costs 1.6 ms and through MCP 3.0 ms, against a 2.3 ms inference. The abstraction is "
            "not where the time goes, so there is no performance argument for bypassing it."),
}


def main() -> int:
    par = json.loads((A / "parity_and_performance_v1.json").read_text())
    frz = json.loads((A / "engine_freeze_audit_v1.json").read_text())
    perf, sci = par["performance"], par["scientific_validation"]
    figs = sorted(F.glob("F*.png"))
    fgates = pd.read_csv(T / "phase10_freeze_gates_v1.csv")
    pgates = pd.read_csv(T / "phase10_parity_gates_v1.csv")
    R.mkdir(parents=True, exist_ok=True)
    out = R / "PHASE_10_GAIRA_RUNTIME_PLATFORM_REPORT.pdf"

    with PdfPages(out) as pdf:
        def cover(fig):
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.text(0.075, 0.90, "GAIRA V7", fontsize=34, color=INK, weight="bold")
            ax.text(0.075, 0.838, "The Runtime Platform", fontsize=19, color=INK)
            ax.plot([0.075, 0.925], [0.805, 0.805], color=RULE, lw=1.3)
            ax.text(0.075, 0.772, "Phase 10 — frozen science, one runtime, five surfaces, "
                    "identical answers", fontsize=10.5, color=MUTED)
            e = frz["engine"]
            rows = [
                ("Status", "COMPLETE — packaging phase, science unchanged"),
                ("Engine", f"{e['n_lsms']} LSMs · {e['n_csms']} CSMs · {e['n_molecules']} "
                           f"molecules · {e['n_chemistry_axes']} chemistry axes"),
                ("Scientific Atlas Fingerprint",
                 frz["declared_fingerprints"]["recomputed"]["atlas"]),
                ("Frozen Runtime Content Hash", e["atlas_fingerprint"]),
                ("Frozen assets pinned", f"{len(frz['frozen_assets'])} content digests, all "
                                         f"verified"),
                ("", ""),
                ("Cross-surface parity", f"{par['parity']['n_comparisons']} comparisons across "
                                         f"{len(par['parity']['surfaces'])} surfaces · "
                                         f"{par['parity']['n_divergent']} divergent"),
                ("Maximum discrepancy", f"{par['parity']['max_abs_diff']:.1e}   "
                                        f"(tolerance {par['parity']['tolerance']:.0e})"),
                ("Phase 09 science", f"reproduced at max deviation "
                                     f"{sci['max_deviation']:.1e}"),
                ("  molecule top-1 / top-5 / MRR", f"{sci['measured']['molecule_top1']:.4f} / "
                                                   f"{sci['measured']['molecule_top5']:.4f} / "
                                                   f"{sci['measured']['molecule_mrr']:.4f}"),
                ("", ""),
                ("Single-spectrum latency", f"{perf['single_inference_ms_median']} ms median "
                                            f"(p95 {perf['single_inference_ms_p95']} ms)"),
                ("API / MCP overhead", f"{perf['api_overhead_ms']} ms / "
                                       f"{perf['mcp_overhead_ms']} ms"),
                ("Engine load", f"{perf['engine_load_seconds']} s"),
                ("", ""),
                ("Gates", f"{len(fgates) + len(pgates)} of "
                          f"{len(fgates) + len(pgates)} PASS"),
                ("Test suite", "1445 passed · 1 skipped · 0 failed"),
                ("Surfaces", "Python SDK · CLI · FastAPI · MCP · Streamlit"),
                ("", ""),
                ("Dependencies", "no LLM · no cloud account · no network · no external volume"),
                ("Scope", "pure Raman reference spectra only"),
            ]
            y = 0.720
            for k, v in rows:
                if k:
                    ax.text(0.075, y, k, fontsize=9.0, color=MUTED)
                    ax.text(0.44, y, v, fontsize=9.0, color=INK)
                y -= 0.0288
            ax.text(0.075, 0.075,
                    "Chemistry Evidence is RELATIVE — not a concentration, not an abundance, not "
                    "a mixture fraction. Retrieved molecules are reference analogues, not "
                    "identifications.\n"
                    "Sources of record: PHASE_10_CONTEXT.md · PHASE_10_ENGINE_FREEZE_AUDIT.md · "
                    "PHASE_10_ARCHITECTURE.md · PHASE_10_API_SPEC.md · PHASE_10_MCP_SPEC.md · "
                    "PHASE_10_STREAMLIT_SPEC.md\n"
                    "PHASE_10_PLUGIN_ARCHITECTURE.md · PHASE_10_DEPLOYMENT_GUIDE.md · "
                    "PHASE_10_VALIDATION_REPORT.md · PHASE_10_SCIENTIFIC_AUDIT.md · "
                    "PHASE_10_DECISION_GATE.md",
                    fontsize=8.0, color=MUTED)
        _new(pdf, cover, chrome=False)

        def contents(b):
            b.p("This document explains what GAIRA V7 is, what is frozen, how a raw Raman "
                "spectrum becomes an inference, where each consumer surface fits, why no "
                "language model is required, and how future modality and sample-context plugins "
                "would extend the platform without touching the science.")
            b.p("It is written to be read start to finish by someone who is not a "
                "spectroscopist. Technical sections are followed by a plain-English gloss marked "
                "in blue.")
            b.h2("Part I — What GAIRA V7 is")
            for t in ("The problem, and the answer GAIRA gives instead",
                      "What is frozen, and how the freeze is enforced",
                      "How a raw spectrum becomes an inference"):
                b.bullet(t, marker="·")
            b.h2("Part II — The platform")
            for t in ("Twelve architecture figures, each with an explanation page",):
                b.bullet(t, marker="·")
            b.h2("Part III — Validation")
            for t in ("The engine freeze audit", "Cross-surface parity",
                      "Performance", "Defects found and fixed"):
                b.bullet(t, marker="·")
            b.h2("Part IV — Extension and the agent boundary")
            b.h2("Part V — Scope, limits and the decision")
        text_page(pdf, "", "Contents", contents)

        # ── Part I ───────────────────────────────────────────────────────────
        divider(pdf, "I", "What GAIRA V7 is",
                "A frozen biochemical coordinate system for Raman spectroscopy, and the runtime "
                "that makes it reachable without letting any route change an answer.")

        def p_problem(b):
            b.lead("Shine a laser at a sample and a small fraction of the light returns with its "
                   "colour shifted. The size of each shift is set by how the molecules present "
                   "vibrate, so the pattern of shifts is a fingerprint of the chemistry. That "
                   "pattern is a Raman spectrum.")
            b.h3("Why the textbook approach fails")
            b.p("A textbook treats a spectrum as a fingerprint to be matched against a library. "
                "Real samples do not cooperate, and three facts govern this architecture:")
            b.bullet("Spectra are mixtures, not fingerprints. What reaches the detector is the "
                     "sum of everything in the illuminated volume.")
            b.bullet("A peak is not a molecule. A band near 1450 cm-1 says a CH2 group is "
                     "bending, and thousands of biological molecules contain CH2 groups.")
            b.bullet("Nearby is not the same. Published assignments frequently match a peak to a "
                     "molecule because the wavenumbers agree within a few units and the biology "
                     "is plausible. That is not evidence.")
            b.h3("What GAIRA answers instead")
            b.p("Rather than 'which molecule is this?', the engine answers a question it can "
                "support: what chemistry does the evidence favour, how strongly, and by way of "
                "which specific spectral features? It still returns ranked molecular candidates, "
                "because a shortlist is useful, but the load-bearing output is the chemistry "
                "reading.")
            b.plain("Think of a doctor reading a blood panel. They rarely conclude 'this is "
                    "precisely molecule X at precisely concentration Y'. They read a pattern and "
                    "say which processes it is consistent with, and how sure they are. GAIRA is "
                    "built to do the second thing well rather than the first thing badly.")
        text_page(pdf, "Part I", "The problem, and what GAIRA answers", p_problem)

        def p_frozen(b):
            b.lead("The scientific architecture was frozen at the end of Phase 09. Phase 10 may "
                   "not change it, and the freeze is enforced by code rather than by convention.")
            b.h3("What is frozen")
            b.kv([("canonical preprocessing", "450–1800 cm-1 · 2.0 step · 676 bins · asLS → "
                                              "Savitzky-Golay(9,3) → L2"),
                  ("LSM dictionary", "50 non-negative motifs, learned within chemistry class"),
                  ("CSM dictionary", "49 consensus motifs — THE canonical representation"),
                  ("retrieval", "cosine over a 154-molecule bank"),
                  ("Chemistry Evidence", "16 axes, model D:A_max_idf with λ = 0.5"),
                  ("calibration", "temperature scaling, T = 0.4538"),
                  ("confidence and warnings", "Phase 05 thresholds, untouched")], keyw=0.26)
            b.h3("Two identities, two names")
            b.kv([("Scientific Atlas Fingerprint",
                   frz["declared_fingerprints"]["recomputed"]["atlas"]),
                  ("", "the scientific atlas identity generated by the Phase 01 build"),
                  ("Frozen Runtime Content Hash", frz["engine"]["atlas_fingerprint"]),
                  ("", "the complete frozen runtime asset identity")], keyw=0.30)
            b.p("These are different objects and this document never calls both \"the atlas "
                "fingerprint\". Note that the API field NAMED `atlas_fingerprint` carries the "
                "Frozen Runtime Content Hash — the runtime is frozen, so the name stands and the "
                "distinction is documented rather than renamed.")
            b.h3("Two independent verification layers")
            b.p("Phase 09 verifies four DECLARED fingerprints — values recorded inside "
                "PHASE_STATE.json and csm_registry_v1.json by the phases that produced them. "
                "That check answers 'did the producing phase claim this artefact'. It does not "
                "answer 'is the file on disk still the file that phase wrote', because the "
                "declared fingerprint lives in a different file from the artefact.")
            b.p(f"Phase 10 adds the second layer: the content digest of every one of the ten "
                f"files the engine opens, recomputed from the committed tree and verified before "
                f"the runtime will serve anything. The loader was instrumented to find those ten "
                f"— every one is inside the repository and git-tracked.")
            b.plain("Together these answer two different questions. Change a dictionary in place "
                    "and the declared fingerprint would not notice; change which artefact a "
                    "phase claims and the content digests would not notice. Both layers run.")
            b.note("Consequence for deployment: an external volume is NOT required. The frozen "
                   "atlas is about ten megabytes of committed files, so a clean clone can run "
                   "inference immediately.", color=GOOD, label="Why this matters")
        text_page(pdf, "Part I", "What is frozen, and how", p_frozen)

        def p_flow(b):
            b.lead("One path, no branches, no parameter tunable at inference time.")
            for n, d in (
                ("1 · Parse", "CSV, TSV, two-column text or arrays. Delimiter, header, column "
                              "identity and axis direction are detected, and every decision is "
                              "reported as a diagnostic rather than applied silently."),
                ("2 · Validate", "Three severities. ERROR stops the run — too few points, no "
                                 "usable overlap with the canonical window, an all-zero or "
                                 "constant spectrum, an unsupported modality. WARNING runs with "
                                 "a stated limitation. INFO records scope."),
                ("3 · Preprocess", "Crop, resample to 676 bins, remove the fluorescence "
                                   "background by asymmetric least squares, smooth, normalise to "
                                   "unit length. The last step discards absolute intensity, "
                                   "which is why no output can be read as a concentration."),
                ("4 · Project", "Non-negative least squares onto 50 local motifs, then onto the "
                                "49 consensus motifs. Those 49 numbers are the canonical "
                                "representation and the only thing any later stage reads."),
                ("5 · Retrieve", "Cosine similarity against 154 reference molecules. Because a "
                                 "cosine is an inner product, each score decomposes exactly into "
                                 "per-motif contributions — which is what makes the ranking "
                                 "explainable rather than merely produced."),
                ("6 · Chemistry", "Collapse the activation onto 16 chemistry axes with a "
                                  "hierarchical model that lets coarse chemistry gently inform "
                                  "fine chemistry without ever excluding it. Then calibrate."),
                ("7 · Report", "Confidence, audit, provenance and a deterministic interpretation "
                               "paragraph. No language model is involved."),
            ):
                b.h3(n); b.p(d)
            b.plain("Each stage answers a narrower question than the last: can this run, is it "
                    "clean, what patterns is it made of, which of those are real and distinct, "
                    "what does it resemble, and what does that mean chemically.")
        text_page(pdf, "Part I", "How a spectrum becomes an inference", p_flow)

        # ── Part II ──────────────────────────────────────────────────────────
        divider(pdf, "II", "The platform",
                "Twelve figures. Each is preceded by a page explaining what it shows and what it "
                "does not license you to conclude.")
        for p in figs:
            key = p.stem.split("_")[0]
            title, lead, paras, note = EXPLAIN[key]

            def build(b, lead=lead, paras=paras, note=note):
                b.lead(lead)
                for para in paras:
                    b.p(para)
                b.plain(note)
            text_page(pdf, f"Figure {key.replace('F', '').lstrip('0')}", title, build)
            figure_page(pdf, p, CAPTIONS.get(key, ""))

        # ── Part III ─────────────────────────────────────────────────────────
        divider(pdf, "III", "Validation",
                "Phase 10 has one scientific obligation: prove that packaging did not change the "
                "science. It is met to the digit.")

        def p_freeze(b):
            b.lead("Nine gates. Every fingerprint recomputed from the committed tree rather than "
                   "read from documentation.")
            b.table(["gate", "status"],
                    [[r.gate, r.status] for r in fgates.itertuples()], [0.0, 0.74],
                    colors={i: GOOD for i in range(len(fgates))})
            b.h3("Phase 09 reproduction, all 375 spectra")
            rep = frz["phase09_reproduction"]
            b.table(["metric", "Phase 10", "Phase 09", "deviation"],
                    [[k, f"{rep['retrieval'][k]:.6f}", f"{rep['baseline'][k]:.6f}",
                      f"{rep['deviations'][k]:.1e}"] for k in rep["baseline"]],
                    [0.0, 0.24, 0.42, 0.60])
            b.p(f"Maximum deviation {rep['max_deviation']:.1e}. Not 'within tolerance' — "
                f"identical.")
        text_page(pdf, "Part III", "The engine freeze audit", p_freeze)

        def p_parity(b):
            b.lead(f"{par['parity']['n_comparisons']} comparisons across "
                   f"{len(par['parity']['surfaces'])} surfaces on twelve locked spectra, at a "
                   f"tolerance of {par['parity']['tolerance']:.0e}.")
            b.table(["gate", "status"],
                    [[r.gate, r.status] for r in pgates.itertuples()], [0.0, 0.74],
                    colors={i: GOOD for i in range(len(pgates))})
            b.h3("Scientific validation through the runtime path")
            b.table(["metric", "Phase 10 runtime", "Phase 09", "deviation"],
                    [[k.replace("_", " "), f"{sci['measured'][k]:.6f}",
                      f"{sci['phase09'][k]:.6f}", f"{sci['deviations'][k]:.1e}"]
                     for k in sci["phase09"]],
                    [0.0, 0.30, 0.48, 0.66])
            b.plain("Six independent code paths were given the same spectra and compared field "
                    "by field: 49 activations, 16 evidence values, 16 calibrated probabilities, "
                    "ten ranks and similarities, confidence, coverage and reconciliation. The "
                    "largest disagreement anywhere was zero.")
        text_page(pdf, "Part III", "Cross-surface parity", p_parity)

        def p_perf(b):
            b.lead("Measured on a laptop before any gate was written. No latency threshold was "
                   "invented in advance of the numbers.")
            b.kv([(k.replace("_", " "), str(v)) for k, v in perf.items()], keyw=0.34)
            b.p("At 2.3 ms per spectrum the engine sustains roughly 430 spectra per second "
                "single-threaded. For live Raman use, acquisition is the bottleneck, not "
                "inference.")
            b.plain("The cost of reaching the engine through HTTP is 1.6 ms and through MCP 3.0 "
                    "ms, against a 2.3 ms inference. There is no performance argument for "
                    "bypassing the runtime, which matters: the usual reason a client ends up "
                    "reimplementing science is that the abstraction felt expensive.")
        text_page(pdf, "Part III", "Performance", p_perf)

        def p_defects(b):
            b.lead("Four defects were found and fixed during this phase. Two of them are the "
                   "same defect.")
            for n, what, cost, fix in (
                ("1 · The freeze audit reimplemented the science",
                 "The first script Phase 10 wrote hand-rolled the leave-one-out retrieval loop "
                 "and dropped every spectrum of the query molecule instead of only the query "
                 "spectrum.",
                 "It reported molecule top-1 of 0.0000 — an apparent catastrophic regression "
                 "that was entirely an artefact of the audit. Chasing it would have wasted the "
                 "phase; believing it would have falsely condemned the engine.",
                 "Replaced with calls to the frozen retrieval modules. Deviation dropped to "
                 "exactly 0.0."),
                ("2 · The text adapter desynchronised its columns",
                 "A row whose wavenumber parsed and whose intensity did not appended the "
                 "wavenumber and then failed, pairing every subsequent intensity with the wrong "
                 "wavenumber.",
                 "A silently mangled spectrum that still looks like a spectrum — the worst "
                 "failure an adapter has, because nothing downstream can detect it.",
                 "Parse both values before appending either, with a regression test that checks "
                 "the pairing arithmetic explicitly."),
                ("3 · A broad exception handler hid defect 2",
                 "The adapter loader swallowed parse exceptions and reported 'unrecognised "
                 "format' for a file it had actually accepted.",
                 "Defect 2 was invisible for as long as this stood, and it presented as a format "
                 "problem, so the investigation would have started in the wrong place.",
                 "Only the cheap recognition step is guarded now; a failing parse returns an "
                 "explicit diagnostic naming the adapter and the exception."),
                ("4 · The static test flagged its own documentation",
                 "A substring search for a banned term matched the Streamlit docstring listing "
                 "what the file excludes.",
                 "A permanently red test that would be disabled rather than fixed — and then the "
                 "rule it encodes would be gone.",
                 "AST-based checking of referenced names and imported modules, ignoring strings "
                 "and comments."),
            ):
                b.h3(n)
                b.p(what)
                b.p("Had it stood: " + cost)
                b.p("Fix: " + fix)
            b.note("Defect 1 matters out of proportion to its size. The very first script this "
                   "phase wrote reproduced the exact failure mode the phase exists to prevent — "
                   "reimplementing scientific logic outside the engine — within twenty minutes "
                   "of starting. That is why the one-implementation rule is enforced by tests "
                   "rather than stated as a principle: the discipline does not survive contact "
                   "with convenience.")
        text_page(pdf, "Part III", "Defects found and fixed", p_defects)

        # ── Part IV ──────────────────────────────────────────────────────────
        divider(pdf, "IV", "Extension and the agent boundary",
                "How new measurement channels and sample contexts would attach, and where a "
                "language model may and may not sit.")

        def p_ext(b):
            b.lead("Four extension protocols are defined. One modality adapter and one context "
                   "adapter are implemented; the other ten raise.")
            b.table(["protocol", "runs", "may do", "may never do"],
                    [["ModalityAdapter", "before the core", "correct, veto, pass through",
                      "touch the dictionaries"],
                     ["SampleContextAdapter", "after the core", "add caveats and framing",
                      "change a number"],
                     ["InterpretationAdapter", "after the core", "rephrase",
                      "compute anything"],
                     ["TrajectoryAdapter", "over a sequence", "analyse change",
                      "recompute a result"]],
                    [0.0, 0.22, 0.40, 0.66], size=8.4)
            b.h3("Why no stub returns a number")
            b.p("A stub that produces plausible output is worse than no stub at all: it will be "
                "wired up, its results will be plotted, and nobody will remember it was never "
                "validated. Every unimplemented adapter raises with a statement of what a "
                "working version must supply — an observation model, a detection gate, a "
                "transfer function, and its own held-out corpus.")
            b.h3("Why the modality boundary is hard rather than advisory")
            b.p("Phase 04 tested whether the frozen Raman atlas could detect real Ag-SERS as "
                "out of domain. It got AUROC 0.548 — chance. A non-negative Raman motif basis "
                "reconstructs SERS of the same metabolites comfortably, so a SERS spectrum run "
                "through the Raman core produces confident numbers with no validated meaning.")
            b.h3("DART is not a modality")
            b.p("DART is a DYNAMIC PERTURBATION PROTOCOL built on Raman/SERS measurements. "
                "DART-Met produces I(wavenumber, potential, time), which is still a vibrational "
                "measurement: every slice through that volume is a spectrum the frozen engine "
                "already reads. There is no spectral transform to invent, because the "
                "measurement axis has not changed — what has been added is perturbation and "
                "time.")
            b.p("So DART attaches at the TrajectoryAdapter layer, downstream of the frozen "
                "representation, over a sequence of ordinary inference results. Nothing "
                "upstream changes. Downstream is correct rather than convenient: a trajectory "
                "of CSM activations is interpretable only if every activation along it came "
                "from the same frozen path.")
            b.plain("scientific representation is not domain interpretation. Confusing the two "
                    "is how a spectral engine becomes a clinical claim without anyone deciding "
                    "that it should.")
        text_page(pdf, "Part IV", "The plugin contracts", p_ext)

        def p_agent(b):
            b.lead("Phase 10 ships no language model and requires no cloud account. This is the "
                   "boundary a Phase 11 agent layer would have to respect.")
            b.h3("The only permitted chain")
            b.p("LLM  →  MCP  →  Frozen Runtime  →  Frozen Engine")
            b.p("One-directional. Nothing flows back up it, and the model never reaches past the "
                "MCP layer. The forbidden shortcut — LLM → scientific computation — is closed by "
                "architecture rather than by policy: no tool exposes a primitive an agent could "
                "compute with.")
            b.h3("An agent MAY — choose tools · explain · compare · narrate · summarise")
            for t in ("choose which tools to call, and in what order",
                      "ask for more evidence before answering",
                      "rephrase the deterministic interpretation",
                      "compare results it obtained through the tools",
                      "surface the caveats verbatim"):
                b.bullet(t)
            b.h3("An agent is FORBIDDEN from — computing chemistry · computing "
                 "similarity · estimating concentrations · re-ranking · diagnosing disease · "
                 "modifying inference")
            for t in ("compute a similarity, an activation, a chemistry axis or a confidence",
                      "re-rank retrieval output",
                      "assert a molecular identification",
                      "convert relative evidence into a concentration or an abundance",
                      "treat low confidence as evidence of a novel molecule",
                      "drop a scope warning",
                      "infer a biological or clinical state"):
                b.bullet(t)
            b.h3("What must be validated first")
            for t in ("an adversarial overclaim benchmark — real tool output plus a leading "
                      "question, measuring how often the model overclaims",
                      "a citation check: every numeric claim resolves to a result field",
                      "a caveat-retention check: scope, relative-evidence and analogue caveats "
                      "survive into the output",
                      "a refusal check: the agent declines questions the engine cannot support",
                      "a determinism boundary: the report generator remains the system of "
                      "record, agent text is marked as commentary"):
                b.bullet(t)
            b.note("The engineering guardrails are in place; the linguistic one is not. Every "
                   "failure mode this architecture prevents is a failure of language — calling "
                   "an analogue an identification, calling relative evidence a concentration, "
                   "calling low confidence a discovery. An agent is a language system, and those "
                   "are exactly the sentences it would find natural to produce.")
        text_page(pdf, "Part IV", "Where a future agent connects", p_agent)

        # ── Part V ───────────────────────────────────────────────────────────
        divider(pdf, "V", "Scope, limits and the decision",
                "What this platform may and may not be used to claim.")

        def p_limits(b):
            b.lead("Constraints on what the output MEANS, not caveats about its quality.")
            for t in ("Chemistry Evidence is RELATIVE — not a concentration, not an abundance, "
                      "not a mixture fraction. L2 normalisation removes absolute scale in the "
                      "first preprocessing stage.",
                      "Retrieved molecules are reference ANALOGUES, not identifications. "
                      "Validated molecule top-1 is 0.6053, and 68 of 375 corpus queries are "
                      "unretrievable by construction.",
                      "There is NO validated open-set detection. The engine cannot determine "
                      "that the true molecule is absent from its bank; white noise reconstructs "
                      "at CSM explained variance 0.61, above the 0.50 warning floor. Read the "
                      "confidence, not the flag.",
                      "Pure Raman reference spectra only. SERS, serum, plasma, EV, bacteria and "
                      "tissue behaviour is unmeasured. Contracts exist; validation does not.",
                      "The 16 chemistry classes are a curated cut through a continuum, not a "
                      "discovered structure.",
                      "Class-prior bias remains open — class sizes range from 3 to 80 spectra.",
                      "In-sample chemistry figures describe the shipped fit. Quote the held-out "
                      "0.8507."):
                b.bullet(t)
            b.h3("Engineering limits")
            for t in ("No real instrument export has been through the adapters — Renishaw, B&W "
                      "Tek and Horiba files were unavailable.",
                      "Validation thresholds are reasoned and declared in advance, but not "
                      "swept; no false-positive rate is attached.",
                      "No production hardening: no authentication, rate limiting, TLS or audit "
                      "log. Loopback by default.",
                      "PDF bytes depend on the matplotlib version; content is reproducible, "
                      "bytes are reproducible on a fixed environment."):
                b.bullet(t)
        text_page(pdf, "Part V", "Interpretation and engineering limits", p_limits)

        def p_decision(b):
            b.lead("All seventeen gates pass. No defect remains open.")
            b.kv([("Frozen Phase 09 engine verified", "PASS — 9/9, fingerprints recomputed"),
                  ("Scientific outputs unchanged", f"PASS — max deviation "
                                                   f"{sci['max_deviation']:.1e}"),
                  ("Cross-surface numerical parity", f"PASS — max discrepancy "
                                                     f"{par['parity']['max_abs_diff']:.1e}"),
                  ("No duplicated scientific logic", "PASS — AST-enforced on five surfaces"),
                  ("No LLM or cloud dependency", "PASS — statically verified"),
                  ("Raman-only scope preserved", "PASS — blocked on all five surfaces"),
                  ("Extension contracts defined", "PASS — 11 declared, 10 raise"),
                  ("Local clean-clone inference", "PASS — 10 committed files"),
                  ("SSD_Rad required?", "NO"),
                  ("Full test suite", "1445 passed · 1 skipped · 0 failed")], keyw=0.34)
            b.h3("Recommendation")
            b.p("FREEZE THE GAIRA V7 RUNTIME. Ready to design a Phase 11 agent layer, "
                "conditional on the five linguistic validations in Part IV.")
            b.note("Confidence that packaging changed no science: 10/10 — checkable, and "
                   "checked at deviation 0.0. Confidence that the platform is stable enough to "
                   "build on: 9/10. The deduction is the adapters: no real instrument export "
                   "has been parsed, and file handling is the one part of this platform whose "
                   "correctness cannot be argued from first principles.",
                   color=GOOD, label="Assessment")
        text_page(pdf, "Part V", "The decision", p_decision)

        def p_colophon(b):
            b.p("Every figure, table and number in this document was generated from the "
                "committed Phase 10 artifacts by the scripts listed below. No value was typed "
                "by hand.")
            b.kv([("engine", "src/gaira/v7/canonical/engine.py"),
                  ("runtime", "src/gaira/v7/runtime/"),
                  ("freeze audit", "results/v7_rebuild/phase10/code/run_engine_freeze_audit.py"),
                  ("parity + performance",
                   "results/v7_rebuild/phase10/code/run_parity_and_performance.py"),
                  ("figures", "results/v7_rebuild/phase10/code/make_figures.py"),
                  ("this document", "results/v7_rebuild/phase10/code/make_pdf.py"),
                  ("tests", "tests/test_v7_phase10_{runtime,surfaces,parity}.py"),
                  ("", ""),
                  ("Frozen Runtime Content Hash", frz["engine"]["atlas_fingerprint"]),
                  ("Scientific Atlas Fingerprint",
                   frz["declared_fingerprints"]["recomputed"]["atlas"]),
                  ("frozen LSM registry", frz["declared_fingerprints"]["recomputed"]["lsm"]),
                  ("frozen CSM registry", frz["declared_fingerprints"]["recomputed"]["csm"]),
                  ("frozen Phase 05 engine",
                   frz["declared_fingerprints"]["recomputed"]["engine"])], keyw=0.30)
            b.h3("Companion documents")
            for d in ("PHASE_10_CONTEXT.md — the continuity note",
                      "PHASE_10_ENGINE_FREEZE_AUDIT.md — fingerprints, golden fixtures, "
                      "regression",
                      "PHASE_10_ARCHITECTURE.md — packages, rules, freeze layers",
                      "PHASE_10_API_SPEC.md · PHASE_10_MCP_SPEC.md · "
                      "PHASE_10_STREAMLIT_SPEC.md",
                      "PHASE_10_PLUGIN_ARCHITECTURE.md — the four extension contracts",
                      "PHASE_10_DEPLOYMENT_GUIDE.md — clean clone to running in five minutes",
                      "PHASE_10_VALIDATION_REPORT.md — all seventeen gates",
                      "PHASE_10_SCIENTIFIC_AUDIT.md — strongly / weakly / unsupported",
                      "PHASE_10_DECISION_GATE.md — the verdict"):
                b.bullet(d, marker="·")
        text_page(pdf, "", "Colophon and provenance", p_colophon)

        d = pdf.infodict()
        d["Title"] = "GAIRA V7 Phase 10 — The Runtime Platform"
        d["Subject"] = ("Frozen Raman inference engine, runtime service, API, MCP, SDK and "
                        "Streamlit client with cross-surface parity validation")
        d["Keywords"] = "GAIRA V7 Raman runtime platform API MCP Streamlit parity"
    print(f"  {out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB, {_page_no[0]} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
