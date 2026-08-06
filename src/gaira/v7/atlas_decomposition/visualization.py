"""GAIRA V7 — figures for the Atlas Component Substructure layer.

Plotting primitives only: every function takes already-computed objects and draws them.
No science happens here, and no RNG. SVG is the vector format (repo policy gitignores PDF).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
PALETTE = ["#2563eb", "#15803d", "#b45309", "#7c3aed", "#0891b2", "#be123c",
           "#4d7c0f", "#a16207", "#4b5563"]

RC = {"font.family": "DejaVu Sans", "font.size": 8.5,
      "figure.facecolor": "white", "savefig.facecolor": "white",
      "savefig.bbox": "tight", "savefig.pad_inches": 0.18, "svg.fonttype": "none"}


def apply_style() -> None:
    plt.rcParams.update(RC)


def save(fig, out_dir, name: str) -> None:
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.svg", format="svg")
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    plt.close(fig)


def component_motif_tree(comp: pd.DataFrame, ax=None):
    """Component → motif tree, coloured by decomposition status."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))
    ax.axis("off")
    y = len(comp)
    for _, r in comp.iterrows():
        col = {"DECOMPOSED": BLUE, "IRREDUCIBLE": GREY,
               "NOT_ANALYSABLE": "#d1d5db"}[r.status]
        ax.text(0, y, f"c{int(r.component):02d}", fontsize=8, weight="bold",
                color=INK, va="center", ha="right")
        ax.plot([0.4, 1.2], [y, y], color=col, lw=1.1)
        n = int(r.n_retained_motifs)
        if n == 0:
            ax.text(1.5, y, r.status.lower().replace("_", " "), fontsize=7,
                    color=MUTED, va="center")
        else:
            for j in range(n):
                yy = y + (j - (n - 1) / 2) * 0.42
                ax.plot([1.2, 2.0], [y, yy], color=col, lw=0.8)
                ax.plot(2.05, yy, "o", ms=3.4, color=PALETTE[j % len(PALETTE)])
                ax.text(2.2, yy, f"m{j:02d}", fontsize=6.2, color=MUTED, va="center")
        y -= 1
    ax.set_xlim(-0.6, 4.2)
    ax.set_ylim(-0.5, len(comp) + 1)
    return ax


def motif_spectra(motifs, grid, parent=None, ax=None, max_show: int = 6,
                  normalise: bool = True):
    """Overlay a component's motif spectra on the parent component.

    With `normalise` (the default) every trace is scaled to unit maximum. Motif spectra are
    masked restrictions of the parent, so their absolute amplitude is always smaller; showing
    raw amplitudes makes them look negligible and invites the wrong conclusion. Normalising
    puts the comparison where it belongs — on SHAPE, i.e. which bands each motif carries.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.4))

    def nz(v):
        v = np.asarray(v, float)
        mx = float(v.max())
        return v / mx if (normalise and mx > 0) else v

    if parent is not None:
        ax.fill_between(grid, nz(parent), color="#e5e7eb", lw=0, label="parent component")
    for j, m in enumerate(motifs[:max_show]):
        ax.plot(grid, nz(m.spectrum), lw=1.2, color=PALETTE[j % len(PALETTE)],
                label=f"{m.motif_id} · {m.dominant_class} (n={m.n_analytes})")
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("normalised intensity" if normalise else "intensity")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.2, frameon=False, loc="upper right")
    return ax


def overlap_graph(C: pd.DataFrame, motifs, ax=None, threshold: float = 0.5):
    """Motif overlap graph laid out deterministically on a circle."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.axis("off")
    ids = list(C.index)
    n = len(ids)
    if n == 0:
        return ax
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs, ys = np.cos(ang), np.sin(ang)
    comp_of = {m.motif_id: m.parent_component for m in motifs}
    V = C.values
    for i in range(n):
        for j in range(i + 1, n):
            w = V[i, j]
            if w < threshold:
                continue
            same = comp_of.get(ids[i]) == comp_of.get(ids[j])
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                    color=(GREY if same else RED), alpha=min(1.0, (w - threshold) * 2.2),
                    lw=0.5 + 2.0 * (w - threshold), zorder=1)
    cols = [PALETTE[comp_of.get(m, 0) % len(PALETTE)] for m in ids]
    ax.scatter(xs, ys, s=26, c=cols, zorder=3, edgecolors="white", linewidths=0.5)
    for i, mid in enumerate(ids):
        r = 1.09
        ax.text(xs[i] * r, ys[i] * r, mid, fontsize=5.0, ha="center", va="center",
                color=MUTED, rotation=np.degrees(ang[i]) - 90 if ys[i] < 0 else
                np.degrees(ang[i]) + 90)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    return ax


def score_distribution(df: pd.DataFrame, column: str, ax=None, color=BLUE,
                       threshold: float | None = None, label: str = ""):
    """Histogram of a motif score with an optional rejection threshold marked."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 3.0))
    v = pd.to_numeric(df[column], errors="coerce").dropna()
    ax.hist(v, bins=18, color=color, alpha=.85, edgecolor="white", linewidth=.5)
    if threshold is not None:
        ax.axvline(threshold, color=RED, lw=1.1, ls="--")
        ax.text(threshold, ax.get_ylim()[1] * .95, f" reject < {threshold}",
                fontsize=6.4, color=RED, va="top")
    ax.set_xlabel(label or column)
    ax.set_ylabel("motifs")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def participation_heatmap(M: pd.DataFrame, class_of: dict, ax=None, max_analytes: int = 80):
    """Analyte × motif participation, analytes ordered by chemical class."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    order = sorted(M.index, key=lambda a: (class_of.get(a, ""), a))
    keep = [a for a in order if M.loc[a].sum() > 0][:max_analytes]
    D = M.loc[keep]
    ax.imshow(D.values, aspect="auto", cmap="Blues", interpolation="nearest",
              vmin=0, vmax=1)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([f"{a[:26]}  [{class_of.get(a, '')[:16]}]" for a in keep],
                       fontsize=4.6)
    ax.set_xticks(range(0, D.shape[1], max(1, D.shape[1] // 30)))
    ax.set_xticklabels([D.columns[i] for i in range(0, D.shape[1],
                                                    max(1, D.shape[1] // 30))],
                       rotation=90, fontsize=4.6)
    ax.set_xlabel("retained motifs")
    return ax


def ambiguity_waterfall(amb: pd.DataFrame, ax=None):
    """Per component: whole-component purity vs weighted motif purity."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.2))
    d = amb[amb.n_retained_motifs >= 2].sort_values("purity_gain", ascending=False)
    x = np.arange(len(d))
    ax.bar(x, d.component_dominant_share, .74, color="#d1d5db", label="component as a whole")
    ax.bar(x, d.weighted_motif_purity, .38, color=BLUE, label="motif layer (size-weighted)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{int(c):02d}" for c in d.component], fontsize=6.4, rotation=90)
    ax.set_ylabel("chemical purity")
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    return ax
