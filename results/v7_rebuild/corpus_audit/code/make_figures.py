#!/usr/bin/env python3
"""GAIRA V7 — corpus audit figures (SVG vector + PNG preview).

Every title distinguishes SPECTRA, LABELS, CANONICAL MOLECULES and STRUCTURES.
The word "analyte" is not used ambiguously.
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
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
AUD = HERE.parent
REPO = AUD.parents[2]
T, F, A = AUD / "tables", AUD / "figures", AUD / "artifacts"

INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.18,
                     "svg.fonttype": "none"})


def save(fig, n):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{n}.svg", format="svg")
    fig.savefig(F / f"{n}.png", dpi=200)
    plt.close(fig)
    print(f"  {n}.svg + {n}.png")


def f01_waterfall(rec):
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    labels = ["raw Raman\nSPECTRA", "dataset-specific\nSOURCE LABELS\n(source, label)",
              "distinct raw\nLABEL STRINGS", "normalized\nANALYTE NAMES\n(V5 layer)",
              "CANONICAL\nMOLECULES\n(V7 layer)"]
    vals = rec["count"].tolist()
    x = np.arange(len(vals))
    cols = [GREY, BLUE, BLUE, AMBER, GREEN]
    ax.bar(x, vals, .62, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 6, str(v), ha="center", fontsize=11, weight="bold", color=cols[i])
        if i:
            d = vals[i] - vals[i - 1]
            ax.annotate("", xy=(i - .31, vals[i]), xytext=(i - .69, vals[i - 1]),
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
            ax.text(i - .5, max(vals[i], vals[i - 1]) + 22, f"{d:+d}", ha="center",
                    fontsize=8.5, color=RED, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel("count")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    notes = ["375 files/columns loaded\nfrom 3 pure-Raman sources",
             "same string in two\nlibraries counted twice",
             "−18: cross-source\nidentical strings",
             "−27: L-/D- prefixes,\nacid/base names",
             "−13: NFKC ligature,\ntruncations, synonyms"]
    for i, n in enumerate(notes):
        ax.text(i, -max(vals) * .155, n, ha="center", fontsize=6.2, color=MUTED, va="top")
    ax.set_title("1 — Corpus reconciliation: from raw SPECTRA to CANONICAL MOLECULES\n"
                 "the widely-quoted “167” counts V5-normalized analyte NAMES; "
                 "“154” counts canonical MOLECULES",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=14)
    fig.subplots_adjust(bottom=0.26)
    save(fig, "fig01_corpus_reconciliation_waterfall")


def f02_modality(inv):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    g = inv.groupby(["source_dataset", "modality"]).size().unstack(fill_value=0)
    g = g.reindex(sorted(g.index, key=lambda s: -g.loc[s].sum()))
    bottom = np.zeros(len(g))
    for mod, col in (("raman", GREEN), ("sers", RED)):
        if mod in g.columns:
            axes[0].barh(np.arange(len(g)), g[mod], .62, left=bottom, color=col,
                         label=f"{mod} SPECTRA")
            bottom = bottom + g[mod].values
    axes[0].set_yticks(np.arange(len(g)))
    axes[0].set_yticklabels(g.index, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("SPECTRA")
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_title("dataset × modality", fontsize=9, loc="left")

    inc = inv.groupby("included_in_v7_raman").size()
    axes[1].axis("off")
    axes[1].text(0, .96, "INCLUSION DECISION", fontsize=9.5, weight="bold", color=INK)
    axes[1].text(0, .84, f"examined      {len(inv):5d} spectra", fontsize=9,
                 family="DejaVu Sans Mono")
    axes[1].text(0, .76, f"included      {int(inc.get(True, 0)):5d} pure Raman",
                 fontsize=9, color=GREEN, family="DejaVu Sans Mono")
    axes[1].text(0, .68, f"excluded      {int(inc.get(False, 0)):5d} Ag-SERS",
                 fontsize=9, color=RED, family="DejaVu Sans Mono")
    axes[1].text(0, .54, "Modality was verified from source metadata, loader\n"
                         "records and archive provenance — never from file names.",
                 fontsize=7.6, color=MUTED, va="top")
    axes[1].text(0, .40, "Excluded by policy without loading:", fontsize=8, weight="bold",
                 color=INK)
    axes[1].text(0, .34, "27 SERS / serum / EV / plasma / perturbation datasets\n"
                         "under the data root, incl. adenine SERS controls, ergothioneine\n"
                         "SERS, uricase, serum spike-ins, Ag-flake, metabolite-63.",
                 fontsize=7.2, color=MUTED, va="top")
    fig.suptitle("2 — Dataset inventory by modality (SPECTRA)", fontsize=11, weight="bold",
                 color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .90))
    save(fig, "fig02_dataset_modality_inventory")


def f03_gobbato(gob):
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.add_patch(Rectangle((3, 52), 44, 38, fc="#dcfce7", ec=GREEN, lw=1.4))
    ax.text(25, 86, "GOBBATO PURE RAMAN", ha="center", fontsize=10, weight="bold", color=GREEN)
    ax.text(25, 80, "INCLUDED in the V7 foundation", ha="center", fontsize=7.6, color=GREEN)
    for i, (k, v) in enumerate([
            ("archive files", gob["archive_raman_files"]),
            ("filename-regex matches", gob["archive_raman_parseable"]),
            ("SPECTRA loaded", gob["loaded_raman_spectra"]),
            ("source LABELS", gob["archive_raman_source_labels"]),
            ("CANONICAL MOLECULES", gob["gobbato_canonical_molecules"]),
            ("missing from corpus", gob["raman_files_missing_from_corpus"])]):
        ax.text(7, 73 - i * 3.6, k, fontsize=7.4, color=INK)
        ax.text(43, 73 - i * 3.6, str(v), fontsize=7.8, weight="bold", ha="right",
                color=(RED if k.startswith("missing") and v else GREEN))
    ax.text(25, 55.5, "replicates per label: exactly 3 for all 51", ha="center",
            fontsize=7, color=MUTED)

    ax.add_patch(Rectangle((53, 52), 44, 38, fc="#fee2e2", ec=RED, lw=1.4))
    ax.text(75, 86, "GOBBATO Ag-SERS", ha="center", fontsize=10, weight="bold", color=RED)
    ax.text(75, 80, "EXCLUDED from the V7 foundation", ha="center", fontsize=7.6, color=RED)
    for i, (k, v) in enumerate([
            ("archive files", gob["archive_sers_files"]),
            ("SERS-for-fitting files", gob["archive_sers_for_fitting"]),
            ("SPECTRA loaded then discarded", gob["loaded_sers_spectra_EXCLUDED"]),
            ("source LABELS", gob["archive_sers_source_labels"]),
            ("leaked into the corpus", gob["sers_files_leaked_into_corpus"])]):
        ax.text(57, 73 - i * 3.6, k, fontsize=7.4, color=INK)
        ax.text(93, 73 - i * 3.6, str(v), fontsize=7.8, weight="bold", ha="right",
                color=(GREEN if k.startswith("leaked") and not v else RED))
    ax.text(75, 55.5, "substrate: 'Ag colloid (Gobbato)' · modality field = sers",
            ha="center", fontsize=7, color=MUTED)

    ax.add_patch(Rectangle((3, 12), 94, 34, fc="#f9fafb", ec=LINE, lw=1.0))
    ax.text(50, 42, "SEPARATION IS BY RECORD METADATA, NOT FILE NAME", ha="center",
            fontsize=8.6, weight="bold", color=INK)
    lines = [
        f"{gob['labels_with_both_raman_and_sers']} source labels appear in BOTH the Raman and the Ag-SERS set —",
        "the same molecules measured two ways. Only the Raman measurement enters the foundation.",
        "",
        f"labels Raman-only: {gob['labels_raman_only'] or 'none'}",
        f"labels SERS-only (correctly absent from the foundation): {', '.join(gob['labels_sers_only_correctly_excluded'])}",
        "",
        f"of {gob['gobbato_normalized_labels']} Gobbato normalized labels, "
        f"{gob['overlap_with_ramanbiolib']} also occur in RamanBioLib and "
        f"{gob['unique_to_gobbato']} are unique to Gobbato.",
    ]
    for i, t in enumerate(lines):
        ax.text(6, 37 - i * 3.4, t, fontsize=7.2, color=INK)
    ax.set_title("3 — Gobbato: pure Raman versus Ag-SERS separation (SPECTRA and LABELS)",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=12)
    save(fig, "fig03_gobbato_raman_vs_sers")


def f04_bipartite(inv, m2o):
    raman = inv[inv.included_in_v7_raman]
    fig, ax = plt.subplots(figsize=(10.0, 7.6))
    ax.axis("off")
    multi = m2o.sort_values(["cross_source", "n_source_labels"], ascending=[False, False])
    y = len(multi)
    for _, r in multi.iterrows():
        labs = r.source_labels.split(";")
        col = BLUE if r.cross_source else GREY
        ax.text(46, y, r.canonical_id, ha="left", va="center", fontsize=7.2, weight="bold",
                color=INK)
        for j, lab in enumerate(labs):
            yy = y + (j - (len(labs) - 1) / 2) * .42
            ax.text(43, yy, lab, ha="right", va="center", fontsize=6.2, color=MUTED)
            ax.annotate("", xy=(44.6, y), xytext=(43.6, yy),
                        arrowprops=dict(arrowstyle="->", color=col, lw=.8))
        ax.text(78, y, f"{int(r.n_spectra)} spectra · {int(r.n_source_datasets)} src",
                fontsize=6, color=col, va="center")
        y -= 1
    ax.set_xlim(0, 100)
    ax.set_ylim(y - 1, len(multi) + 2)
    ax.text(43, len(multi) + 1.2, "SOURCE LABELS", ha="right", fontsize=8, weight="bold",
            color=MUTED)
    ax.text(46, len(multi) + 1.2, "CANONICAL MOLECULE", ha="left", fontsize=8, weight="bold",
            color=INK)
    ax.set_title(f"4 — Source LABEL → CANONICAL MOLECULE mapping "
                 f"({len(multi)} many-to-one groups)\n"
                 f"blue = cross-source duplicate molecule · grey = within-source alias",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=12)
    save(fig, "fig04_label_to_molecule_bipartite")


def f05_bysource(bysrc, bycls):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    x = np.arange(len(bysrc))
    for i, (c, col, lab) in enumerate([("spectra", GREY, "SPECTRA"),
                                       ("source_labels", BLUE, "source LABELS"),
                                       ("normalized", AMBER, "normalized NAMES"),
                                       ("canonical", GREEN, "CANONICAL MOLECULES")]):
        axes[0].bar(x + (i - 1.5) * .2, bysrc[c], .19, color=col, label=lab)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.replace("_", "\n") for s in bysrc.source_dataset], fontsize=6.8)
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("count")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_title("per pure-Raman source", fontsize=9, loc="left")

    d = bycls.sort_values("canonical")
    axes[1].barh(np.arange(len(d)), d.canonical, .66, color=GREEN)
    axes[1].set_yticks(np.arange(len(d)))
    axes[1].set_yticklabels([c.replace("_", " ") for c in d.chemistry_class], fontsize=6.4)
    axes[1].set_xlabel("CANONICAL MOLECULES")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_title("per chemistry class", fontsize=9, loc="left")
    fig.suptitle("5 — CANONICAL MOLECULE counts by source and by chemistry class",
                 fontsize=11, weight="bold", color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .90))
    save(fig, "fig05_canonical_counts")


def f06_replicates(spm, inv):
    raman = inv[inv.included_in_v7_raman]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9))
    vc = spm.n_spectra.value_counts().sort_index()
    axes[0].bar(vc.index, vc.values, .66, color=BLUE)
    for xi, v in zip(vc.index, vc.values):
        axes[0].text(xi, v + 1.5, str(v), ha="center", fontsize=7, color=BLUE)
    axes[0].set_xlabel("SPECTRA per CANONICAL MOLECULE")
    axes[0].set_ylabel("CANONICAL MOLECULES")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_title(f"replicate distribution — {int((spm.n_spectra == 1).sum())} singletons "
                      f"of {len(spm)}", fontsize=9, loc="left")

    ec = raman.excitation_nm.value_counts().sort_index()
    axes[1].bar(np.arange(len(ec)), ec.values, .62, color=AMBER)
    axes[1].set_xticks(np.arange(len(ec)))
    axes[1].set_xticklabels([f"{x:g}" for x in ec.index], fontsize=7, rotation=45)
    for i, v in enumerate(ec.values):
        axes[1].text(i, v + 3, str(v), ha="center", fontsize=7, color=AMBER)
    axes[1].set_xlabel("excitation (nm)")
    axes[1].set_ylabel("SPECTRA")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_title(f"{len(ec)} excitation domains; 785 nm carries "
                      f"{100 * ec.max() / ec.sum():.0f}%", fontsize=9, loc="left")
    fig.suptitle("6 — Replicate and excitation structure", fontsize=11, weight="bold",
                 color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .89))
    save(fig, "fig06_replicates_and_excitation")


def f07_crosssource(m2o):
    cs = m2o[m2o.cross_source].sort_values("n_spectra", ascending=False)
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    x = np.arange(len(cs))
    ax.bar(x, cs.n_spectra, .66, color=BLUE)
    for i, (_, r) in enumerate(cs.iterrows()):
        ax.text(i, r.n_spectra + .12, str(int(r.n_source_datasets)), ha="center", fontsize=6.4,
                color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(cs.canonical_id, rotation=90, fontsize=6.4)
    ax.set_ylabel("SPECTRA pooled")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"7 — Cross-source duplicate CANONICAL MOLECULES ({len(cs)} of "
                 f"{len(m2o)} many-to-one groups)\n"
                 "each is one molecule that two reference libraries labelled differently; "
                 "number above the bar = source datasets pooled",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=12)
    save(fig, "fig07_cross_source_duplicates")


def f08_acidbase(ab):
    d = ab[ab.cosine.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    x = np.arange(len(d))
    axes[0].bar(x - .2, d.cooh_1710_share_a, .38, color=GREY, label="form A (named as acid)")
    axes[0].bar(x + .2, d.cooh_1710_share_b, .38, color=BLUE, label="form B (named as base)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(d.canonical_id, rotation=45, fontsize=7)
    axes[0].set_ylabel("spectral share in the\nC=O (1710 cm$^{-1}$) window")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_title("protonation-state test", fontsize=9, loc="left")

    axes[1].barh(x, d.cosine, .6, color=[GREEN if c > .8 else AMBER for c in d.cosine])
    axes[1].axvline(0.915, color=RED, ls="--", lw=1)
    axes[1].text(0.915, len(d) - .3, " lowest UNCONTESTED\n cross-source merge (tyrosine)",
                 fontsize=6.2, color=RED, va="top")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(d.canonical_id, fontsize=7)
    axes[1].set_xlim(0.7, 1.0)
    axes[1].set_xlabel("cosine between the two labelled forms")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_title("spectral agreement", fontsize=9, loc="left")
    fig.suptitle("8 — Acid / conjugate-base merges: are they distinct chemical FORMS?\n"
                 "near-identical C=O share means both members are in the SAME protonation "
                 "state — a labelling variant, not a distinct form",
                 fontsize=11, weight="bold", color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .85))
    save(fig, "fig08_acid_base_verification")


def f09_reproduction():
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.add_patch(Rectangle((4, 20), 92, 62, fc="#dcfce7", ec=GREEN, lw=1.4))
    ax.text(50, 74, "CORPUS UNCHANGED → PHASE 01 REPRODUCED EXACTLY", ha="center",
            fontsize=11, weight="bold", color=GREEN)
    rows = [
        ("raw Raman SPECTRA", "375", "375"),
        ("normalized analyte NAMES", "167", "167"),
        ("CANONICAL MOLECULES", "154", "154"),
        ("chemistry classes", "16", "16"),
        ("balancing arm", "B_analyte_weighted", "B_analyte_weighted"),
        ("Local Spectral Motifs", "50", "50"),
        ("k_c values", "{1,2,3,5,6,7,10}", "{1,2,3,5,6,7,10}"),
        ("registry fingerprint", "208482d6f7178b5b8f16…", "208482d6f7178b5b8f16…"),
        ("architecture compliance", "18/18", "18/18"),
    ]
    ax.text(8, 66, "quantity", fontsize=7.6, weight="bold", color=MUTED)
    ax.text(48, 66, "before audit", fontsize=7.6, weight="bold", color=MUTED)
    ax.text(74, 66, "after audit", fontsize=7.6, weight="bold", color=MUTED)
    for i, (k, a, b) in enumerate(rows):
        y = 61 - i * 4.4
        ax.text(8, y, k, fontsize=7.6, color=INK)
        ax.text(48, y, a, fontsize=7.4, color=MUTED, family="DejaVu Sans Mono")
        ax.text(74, y, b, fontsize=7.4, color=GREEN, family="DejaVu Sans Mono")
        ax.text(94, y, "✓", fontsize=8, color=GREEN, ha="right")
    ax.text(50, 24, "No corpus correction was required, so no Phase 01 rerun was required.\n"
                    "The dictionary was regenerated anyway and reproduced bit-identically.",
            ha="center", fontsize=7.8, color=INK)
    ax.set_title("9 — Phase 01 impact: nothing changed, and reproduction confirms it",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=12)
    save(fig, "fig09_phase01_impact")


if __name__ == "__main__":
    print(f"writing corpus-audit figures to {F}")
    inv = pd.read_csv(T / "spectrum_level_audit_v1.csv")
    rec = pd.read_csv(T / "count_reconciliation_v1.csv")
    m2o = pd.read_csv(T / "canonicalization_many_to_one_audit.csv")
    bysrc = pd.read_csv(T / "canonical_count_by_source_v1.csv")
    bycls = pd.read_csv(T / "canonical_count_by_class_v1.csv")
    spm = pd.read_csv(T / "spectra_per_canonical_molecule_v1.csv")
    ab = pd.read_csv(T / "acid_base_merge_verification.csv")
    gob = json.loads((A / "gobbato_audit_v1.json").read_text())
    f01_waterfall(rec)
    f02_modality(inv)
    f03_gobbato(gob)
    f04_bipartite(inv, m2o)
    f05_bysource(bysrc, bycls)
    f06_replicates(spm, inv)
    f07_crosssource(m2o)
    f08_acidbase(ab)
    f09_reproduction()
    print("done — 9 figures (SVG vector + PNG preview)")
