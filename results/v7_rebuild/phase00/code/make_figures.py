#!/usr/bin/env python3
"""GAIRA V7 Phase 00 — publication figures (SVG vector + PNG preview).

Reads only the Phase-00 tables; performs no science of its own. Deterministic: no RNG,
no timestamps. Repo policy gitignores *.pdf, so SVG is the vector format.

    python results/v7_rebuild/phase00/code/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_paths as P                                                    # noqa: E402

T = P.TABLES
OUT = P.FIGURES

INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.18, "svg.fonttype": "none",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", format="svg")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}.svg + {name}.png")


def canvas(w, h, title, sub=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(0, 99, title, fontsize=12, weight="bold", color=INK, va="top")
    if sub:
        ax.text(0, 94.4, sub, fontsize=8, color=MUTED, va="top")
    return fig, ax


def box(ax, x, y, w, h, text, *, fc="white", ec=GREY, fs=8.0, weight="normal", lw=1.1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=3, linespacing=1.45, weight=weight)


def arrow(ax, p0, p1, color=LINE, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color,
                                 linewidth=lw, linestyle=ls, shrinkA=1.5, shrinkB=1.5, zorder=1))


# ── 1 canonical resolution workflow ───────────────────────────────────────────
def f01():
    leak = json.loads((P.MANIFESTS / "alias_leakage_report_v1.json").read_text())
    near = pd.read_csv(T / "alias_near_miss_audit_v1.csv")
    fig, ax = canvas(10.4, 7.2, "1 — Canonical molecule resolution workflow",
                     "Identity is a metadata layer: the corpus stays 375/167, CV groups by canonical_id.")
    steps = [
        (f"observed surface forms\n{leak['n_surface_analytes']} analyte names", "white", GREY),
        ("step 1  MECHANICAL\nUnicode NFKC + whitespace + case\n→ collapses the U+FB02 'ﬂ' ligature",
         "#dbeafe", BLUE),
        ("step 2  DECLARED MERGES\n13 duplicate pairs from the V6.3 audit,\neach re-audited with a chemical justification",
         "#dbeafe", BLUE),
        ("step 3  PROTECTED\nenantiomers & anomers NEVER merged\n(+)-arabinose ≠ (−)-arabinose", "#fee2e2", RED),
        ("step 4  FLAGGED\nnear-misses recorded as UNRESOLVED,\nnever merged silently", "#fef3c7", AMBER),
        (f"canonical molecule IDs\n{leak['n_canonical_ids']} molecules", "#dcfce7", GREEN),
    ]
    y = 82
    for i, (t, fc, ec) in enumerate(steps):
        box(ax, 6, y, 52, 8.4, t, fc=fc, ec=ec, fs=7.8,
            weight="bold" if i in (0, len(steps) - 1) else "normal")
        if i < len(steps) - 1:
            arrow(ax, (32, y), (32, y - 3.6), color=BLUE)
        y -= 12.0

    n_prot = int((near.decision == "NOT_MERGED_PROTECTED").sum())
    n_unres = int((near.decision == "NOT_MERGED_UNRESOLVED").sum())
    box(ax, 63, 52, 35, 38,
        "WHAT THE MERGE REMOVED\n\n"
        f"surface forms          {leak['n_surface_analytes']}\n"
        f"canonical molecules    {leak['n_canonical_ids']}\n"
        f"merged forms           {leak['n_merged_surface_forms']}\n\n"
        f"CROSS-SOURCE merges    {leak['n_cross_source_merges']}\n"
        f"spectra affected       {leak['spectra_affected_by_cross_source_merge']}\n\n"
        "A cross-source merge is one molecule\nthat appeared in two reference libraries\n"
        "under two spellings. Grouping CV by\nsurface name would score it against itself.",
        fc="#eff6ff", ec=BLUE, fs=7.5)
    box(ax, 63, 22, 35, 26,
        "EXPLICIT NON-MERGES\n\n"
        f"protected (enantiomer / anomer)   {n_prot}\n"
        f"unresolved, flagged               {n_unres}\n\n"
        "Every non-merge is a recorded decision\nwith a reason, so a reader can disagree\nwith it.",
        fc="#f9fafb", ec=LINE, fs=7.5)
    save(fig, "fig01_canonical_resolution_workflow")


# ── 2 alias graph ─────────────────────────────────────────────────────────────
def f02():
    canon = pd.read_csv(T / "canonical_analytes_v1.csv")
    near = pd.read_csv(T / "alias_near_miss_audit_v1.csv")
    multi = canon[canon.n_surface_forms > 1].sort_values(
        ["cross_source_merge", "n_spectra"], ascending=[False, False])
    fig, ax = canvas(10.4, 7.4, "2 — Alias graph: surface forms → canonical molecules",
                     "Blue = merged (one molecule, several spellings). Red = deliberately NOT merged.")
    y = 86
    for _, r in multi.iterrows():
        forms = r.surface_forms.split(";")
        cross = bool(r.cross_source_merge)
        col = BLUE if cross else GREY
        ax.text(30, y, r.canonical_id, ha="right", va="center", fontsize=7.6,
                weight="bold", color=INK)
        ax.plot([31, 34], [y, y], color=col, lw=1.0)
        ax.plot(34, y, "o", ms=4, color=col)
        for k, f in enumerate(sorted(f for f in forms if f != r.canonical_id)):
            yy = y + (k - (len(forms) - 2) / 2) * 2.4
            ax.annotate("", xy=(35, y), xytext=(46, yy),
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
            # a Unicode ligature renders identically to its decomposition in most fonts,
            # so name the codepoint rather than showing two apparently identical strings
            note = ""
            if any(ord(ch) > 127 for ch in f):
                cps = " ".join(f"U+{ord(ch):04X}" for ch in f if ord(ch) > 127)
                note = f"   [{cps} ligature]"
            ax.text(47, yy, f + note, ha="left", va="center", fontsize=7.0, color=MUTED)
        ax.text(97, y, f"{int(r.n_spectra)} spec · {int(r.n_sources)} src", ha="right",
                va="center", fontsize=6.6, color=(BLUE if cross else MUTED))
        y -= 5.4

    nm = near[near.decision.str.startswith("NOT_MERGED")]
    y -= 2.0
    ax.plot([4, 96], [y + 2.0, y + 2.0], color="#e5e7eb", lw=1.0)
    ax.text(4, y - 0.5, "explicitly NOT merged", fontsize=7.6, weight="bold", color=RED,
            va="top")
    y -= 5.0
    for _, r in nm.iterrows():
        tag = "protected" if r.decision.endswith("PROTECTED") else "unresolved"
        ax.text(30, y, r.form_a, ha="right", va="center", fontsize=7.2, color=INK)
        ax.plot([32, 44], [y, y], color=RED, lw=1.0, ls=(0, (3, 2)))
        ax.text(38, y + 1.4, "✗", ha="center", va="center", fontsize=8, color=RED)
        ax.text(46, y, r.form_b, ha="left", va="center", fontsize=7.2, color=INK)
        ax.text(97, y, tag, ha="right", va="center", fontsize=6.6,
                color=(RED if tag == "protected" else AMBER))
        y -= 4.4
    save(fig, "fig02_alias_graph")


# ── 3 replicate grouping ──────────────────────────────────────────────────────
def f03():
    cmp_ = pd.read_csv(T / "replicate_group_key_comparison_v1.csv")
    reps = pd.read_csv(T / "replicate_groups_v1.csv")
    fig, ax = canvas(10.4, 6.4, "3 — Replicate grouping: which spectra are one measurement?",
                     "Ratified key: (canonical_id, excitation). Balancing then applies per molecule ACROSS groups.")
    sizes = reps.n_spectra.value_counts().sort_index()
    x0, w = 8.0, 8.0
    mx = float(sizes.max())
    for i, (sz, n) in enumerate(sizes.items()):
        h = 42.0 * n / mx
        ax.add_patch(Rectangle((x0 + i * (w + 3), 30), w, h, fc="#dbeafe", ec=BLUE, lw=1.0))
        ax.text(x0 + i * (w + 3) + w / 2, 30 + h + 2.0, str(int(n)), ha="center",
                fontsize=7.4, color=BLUE, weight="bold")
        ax.text(x0 + i * (w + 3) + w / 2, 27, f"{int(sz)}", ha="center", fontsize=7.4, color=INK)
    ax.text(x0 + 2.5 * (w + 3), 21.5, "spectra per replicate group", ha="center",
            fontsize=7.6, color=MUTED)
    ax.text(x0, 80, "replicate-group size distribution  (v7 key)", fontsize=8.4,
            weight="bold", color=INK)

    rows = ["key                             groups  median  max  singletons"]
    for _, r in cmp_.iterrows():
        rows.append(f"{r.key:31s} {int(r.n_groups):6d} {r.median_size:7.1f} {int(r.max_size):4d}"
                    f" {int(r.n_singleton_groups):11d}")
    box(ax, 58, 56, 40, 26, "\n".join(rows), fc="#f9fafb", ec=LINE, fs=6.6)
    box(ax, 58, 16, 40, 36,
        "WHY (canonical_id, excitation)\n\n"
        "The V5 key also split on SOURCE, so the same\n"
        "molecule measured at 785 nm in two reference\n"
        "libraries formed two groups.\n\n"
        "Under the V7 key those are one replicate group —\n"
        "which is what a replicate is: the same molecule\n"
        "under the same measurement condition.\n\n"
        "Excitation stays in the key because it is a\n"
        "tracked nuisance factor: peak POSITION is\n"
        "excitation-invariant, relative INTENSITY is not.\n\n"
        "Balancing applies per canonical molecule ACROSS\n"
        "its groups, so the 41 multi-excitation molecules\n"
        "buy no extra weight.",
        fc="#eff6ff", ec=BLUE, fs=7.2)
    save(fig, "fig03_replicate_grouping")


# ── 4 dataset composition ─────────────────────────────────────────────────────
def f04():
    card = json.loads((P.MANIFESTS / "dataset_card_v7.json").read_text())
    census = pd.read_csv(T / "class_census_v1.csv").sort_values("n_canonical_analytes")
    fig, ax = canvas(11.0, 7.4, "4 — Frozen dataset composition after canonicalisation",
                     f"{card['n_spectra']} spectra · {card['n_analytes']} surface forms · "
                     f"{int(census.n_canonical_analytes.sum())} canonical molecules · Raman only")
    x0, bh, sc = 22.0, 3.5, 1.35
    ax.text(x0 + 20, 88, "canonical molecules per fine chemical class", fontsize=8.2,
            color=INK, ha="center")
    for i, (_, r) in enumerate(census.iterrows()):
        y = 12 + i * (bh + 1.0)
        n = int(r.n_canonical_analytes)
        col = RED if n <= 3 else (AMBER if n <= 8 else GREY)
        ax.add_patch(Rectangle((x0, y), n * sc, bh, fc=col, ec="none", alpha=.85))
        ax.text(x0 - 1.2, y + bh / 2, r.fine_class.replace("_", " "), ha="right",
                va="center", fontsize=6.9, color=INK)
        ax.text(x0 + n * sc + 1.0, y + bh / 2, f"{n}  (k_c ≤ {int(r.k_c_ceiling)})",
                ha="left", va="center", fontsize=6.6, color=MUTED)
        if bool(r.source_confounded):
            ax.text(x0 + n * sc + 14.5, y + bh / 2, "⚠ source-confounded", ha="left",
                    va="center", fontsize=6.2, color=AMBER)

    src = card["sources"]
    exc = card["excitations"]
    box(ax, 66, 52, 32, 32,
        "SOURCES\n" + "\n".join(f"{k:28s}{v:4d}" for k, v in src.items()) +
        "\n\nEXCITATIONS (nm)\n" +
        "\n".join(f"{k:>10s}{v:22d}" for k, v in list(exc.items())[:5]) +
        f"\n{'other':>10s}{sum(list(exc.values())[5:]):22d}",
        fc="#f9fafb", ec=LINE, fs=6.6)
    box(ax, 66, 20, 32, 29,
        "EXCLUDED BY CONSTRUCTION\n\n" + "\n".join(f"· {d}" for d in card["excluded_domains"][:6]) +
        "\n\nSERS is a measurement channel applied\nto the Raman latent state, never a\n"
        "training domain for it.",
        fc="#fee2e2", ec=RED, fs=6.8)
    save(fig, "fig04_dataset_composition")


# ── 5 provenance flow ─────────────────────────────────────────────────────────
def f05():
    fig, ax = canvas(9.6, 8.0, "5 — Phase 00 provenance flow",
                     "Every artefact records the IDs and SHA-256 of its inputs one level down.")
    steps = [
        ("raw source datasets\nRamanBioLib · Gobbato · amino-acid grounding", "white", GREY, "GAIRA_DATA_ROOT"),
        ("canonical preprocessing\ncrop → resample → asls → savgol → L2", "#f3f4f6", GREY, "UNCHANGED from V5"),
        ("corpus  375 × 676\nverified against the frozen V5 card", "#dbeafe", BLUE, "dataset_card_v7.json"),
        ("canonical molecule IDs\n167 surface forms → 154 molecules", "#dbeafe", BLUE, "canonical_analytes_v1.csv"),
        ("replicate groups + quality q\n231 groups · q frozen before Phase 01", "#dbeafe", BLUE, "spectrum_quality_v1.csv"),
        ("chemical partition\n16 fine / 6 broad, rationale per class", "#dbeafe", BLUE, "chemical_partition_v1.csv"),
        ("frozen CV splits\n5 folds grouped by canonical_id", "#dbeafe", BLUE, "cv_splits_v1.json"),
        ("frozen evaluation harness\nmetrics · nulls · CIs · paired tests", "#dbeafe", BLUE, "v7_harness.py"),
        ("V5 CONTROL BASELINE\nmeasured under the frozen harness", "#dcfce7", GREEN, "phase00_baseline_metrics.csv"),
    ]
    y, h, gap = 84.0, 6.2, 8.6
    for i, (t, fc, ec, note) in enumerate(steps):
        box(ax, 5, y, 62, h, t, fc=fc, ec=ec, fs=7.6,
            weight="bold" if i == len(steps) - 1 else "normal")
        ax.text(68.5, y + h / 2, note, fontsize=6.5, color=MUTED, va="center")
        if i < len(steps) - 1:
            arrow(ax, (36, y), (36, y - (gap - h)), color=BLUE)
        y -= gap
    ax.text(0, 6.0,
            "Reverse traversal answers 'what breaks if this dataset is retracted?'.\n"
            "Forward traversal answers 'which chemistry supports this axis?'.",
            fontsize=7.0, color=MUTED, va="bottom", linespacing=1.5)
    save(fig, "fig05_provenance_flow")


# ── 6 benchmark lock ──────────────────────────────────────────────────────────
def f06():
    b = pd.read_csv(T / "benchmark_lock_v1.csv")
    fig, ax = canvas(10.4, 6.8, "6 — Benchmark lock: three levels of verification",
                     "Level 3 is the real lock — the basis is REFITTED from raw and compared element-wise.")
    lv = [
        (1, "DECLARED", "the fingerprint recorded in MANIFEST.json / manifold.json",
         b[b.check.str.startswith(("MANIFEST", "manifold", "preprocessing"))], "#f3f4f6", GREY),
        (2, "RECOMPUTED", "fingerprint recomputed from the basis array; every frozen file re-hashed",
         b[b.check.str.startswith(("basis.", "file_sha256."))], "#dbeafe", BLUE),
        (3, "REBUILT", "NMF refitted from raw through canonical preprocessing, compared element-wise",
         b[b.check.str.startswith("rebuild.")], "#dcfce7", GREEN),
    ]
    y = 76
    for n, name, desc, sub, fc, ec in lv:
        npass = int((sub.status == "PASS").sum())
        box(ax, 5, y, 90, 15.5, "", fc=fc, ec=ec, lw=1.3)
        ax.text(9, y + 11.6, f"LEVEL {n} — {name}", fontsize=9.2, weight="bold", color=ec)
        ax.text(9, y + 7.6, desc, fontsize=7.4, color=INK)
        ax.text(9, y + 3.6, f"{npass}/{len(sub)} checks PASS", fontsize=7.6,
                color=(GREEN if npass == len(sub) and len(sub) else RED), weight="bold")
        if n == 3 and len(sub):
            md = sub[sub.check == "rebuild.max_abs_difference"]
            if len(md):
                ax.text(52, y + 3.6,
                        f"max |H_rebuilt − H_frozen| = {float(md.iloc[0].got):g}",
                        fontsize=8.0, color=GREEN, weight="bold")
        y -= 18.5
    box(ax, 5, 6, 90, 12,
        f"ATLAS FINGERPRINT   {P.CANONICAL_ATLAS_FINGERPRINT}\n"
        "unchanged before and after Phase 00 · recomputed from the array, not read from the manifest\n"
        "The V7 corpus loader, the canonical preprocessing chain and the frozen atlas are the same object.",
        fc="#f9fafb", ec=LINE, fs=7.4)
    save(fig, "fig06_benchmark_lock")


# ── 7 frozen artifact dependency graph ────────────────────────────────────────
def f07():
    dep = pd.read_csv(T / "frozen_dependency_graph_v1.csv")
    fig, ax = canvas(10.6, 7.6, "7 — Frozen artefact dependency graph",
                     "Phase 00 READS every asset below and WRITES none of them.")
    used = dep[dep.phase00_dependency != "not used in Phase 00"]
    unused = dep[dep.phase00_dependency == "not used in Phase 00"]
    ax.text(4, 88, "READ and used by Phase 00", fontsize=8.6, weight="bold", color=BLUE)
    y = 82
    for _, r in used.iterrows():
        box(ax, 4, y, 45, 4.6, r.frozen_asset.split("/")[-1], fc="#dbeafe", ec=BLUE, fs=6.9)
        ax.text(50, y + 2.3, r.phase00_dependency, fontsize=6.6, color=MUTED, va="center")
        y -= 5.6
    ax.text(4, y - 1.0, "READ-ONLY, not used by Phase 00", fontsize=8.6, weight="bold",
            color=MUTED)
    y -= 7.0
    for _, r in unused.iterrows():
        box(ax, 4, y, 45, 4.6, r.frozen_asset.split("/")[-1], fc="white", ec=LINE, fs=6.9)
        y -= 5.6
    box(ax, 62, 14, 36, 26,
        "WRITE TARGETS\n\n"
        "results/v7_rebuild/phase00/\n"
        "  tables/ · manifests/ · figures/\n"
        "  reports/ · logs/ · validation/\n\n"
        "NEVER WRITTEN\n"
        "  assets/foundation/\n"
        "  results/v5_rebuild/\n"
        "  results/v6_rebuild/\n"
        "  src/gaira/",
        fc="#dcfce7", ec=GREEN, fs=7.2)
    save(fig, "fig07_frozen_dependency_graph")


# ── 8 V5 control baseline ─────────────────────────────────────────────────────
def f08():
    b = pd.read_csv(T / "phase00_baseline_metrics.csv")
    g = pd.read_csv(T / "phase00_baseline_gain_v1.csv")
    fig, ax = canvas(10.6, 7.0, "8 — The V5 control baseline, re-measured under the frozen V7 harness",
                     "This is the number Phase 07 must beat. Measured on the frozen atlas — no V7 model exists.")
    levels = ["coord", "mss", "theme_raw", "system_raw"]
    labels = [("v7_fine_16", BLUE, "fine (16)"), ("v7_broad_6", GREEN, "broad (6)"),
              ("size_matched_random", GREY, "random control")]
    y0, hmax = 40.0, 42.0                     # plot floor and full-scale height
    x0, gw, bw = 14.0, 20.0, 5.0
    for gy in (0.0, 0.25, 0.5, 0.75, 1.0):    # gridlines
        ax.plot([x0 - 2, x0 + 4 * gw - 4], [y0 + gy * hmax] * 2, color="#eef0f3", lw=.8, zorder=0)
        ax.text(x0 - 3, y0 + gy * hmax, f"{gy:.2f}", ha="right", va="center",
                fontsize=6.4, color=MUTED)
    for i, lv in enumerate(levels):
        for j, (lab, col, _) in enumerate(labels):
            r = b[(b.level == lv) & (b.labels == lab)]
            if not len(r):
                continue
            v = float(r.retrieval_p1.iat[0])
            x = x0 + i * gw + j * (bw + 0.8)
            ax.add_patch(Rectangle((x, y0), bw, v * hmax, fc=col, ec="none", alpha=.88, zorder=2))
            ax.text(x + bw / 2, y0 + v * hmax + 1.2, f"{v:.3f}", ha="center", fontsize=6.2,
                    color=col, va="bottom", zorder=3)
            if lab != "size_matched_random":
                lo, hi = float(r.ci95_low.iat[0]), float(r.ci95_high.iat[0])
                ax.plot([x + bw / 2] * 2, [y0 + lo * hmax, y0 + hi * hmax], color=INK,
                        lw=.9, zorder=3)
        ax.text(x0 + i * gw + (3 * bw) / 2, y0 - 3.0, lv, ha="center", fontsize=7.6, color=INK)
    for k, (_, col, name) in enumerate(labels):
        ax.add_patch(Rectangle((14 + k * 24, 88.5), 3.2, 2.4, fc=col, ec="none"))
        ax.text(18.2 + k * 24, 89.7, name, fontsize=7.2, va="center", color=INK)
    ax.text(6.5, y0 + hmax / 2, "retrieval@1", rotation=90, fontsize=7.4, color=MUTED,
            ha="center", va="center")

    rows = ["level         fine    broad   random   gain(fine)  gain(broad)"]
    for _, r in g.iterrows():
        rows.append(f"{r.level:12s} {r.fine_p1:.3f}   {r.broad_p1:.3f}   {r.random_p1:.3f}"
                    f"      {r.gain_beyond_mechanical_fine:+.3f}      "
                    f"{r.gain_beyond_mechanical_broad:+.3f}")
    box(ax, 5, 6, 90, 24, "\n".join(rows) +
        "\n\nGain beyond mechanical = observed − size-matched random.\n"
        "Coarse chemistry is genuinely present; the fine ceiling is what V7 targets.",
        fc="#f9fafb", ec=LINE, fs=6.8)
    save(fig, "fig08_v5_control_baseline")


if __name__ == "__main__":
    print(f"writing Phase 00 figures to {OUT}")
    for f in (f01, f02, f03, f04, f05, f06, f07, f08):
        f()
    print("done — 8 figures (SVG vector + PNG preview)")
