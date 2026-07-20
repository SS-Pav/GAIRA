"""Publication figures for the V6 demo (matplotlib, position-first, one accent).

Every figure is a pure function of engine outputs — no hidden state, no fitting.
The MSS-hierarchy figure is the demo's centerpiece: it shows how a biochemical
state decomposes Radar -> MSS motifs -> latent components.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams

from . import theme as T

rcParams.update(T.mpl_rc())

# short display labels for the 11 biochemical themes (radar axes)
THEME_SHORT = {
    "nucleic_purine": "Purine", "nucleic_pyrimidine": "Pyrimidine",
    "protein_peptide": "Protein", "aromatic_amino_acid": "Aromatic AA",
    "lipid_acyl": "Lipid acyl", "sterol_membrane": "Sterol",
    "saccharide_glycan": "Glycan", "organic_acid_metabolism": "Organic acid",
    "sulfur_antioxidant": "Sulfur", "heme_porphyrin": "Heme", "redox_broad": "Redox",
}


# ── Page 1: architecture flow ──
def architecture_diagram():
    stages = [
        ("Reference spectra", "375 pure Raman · 167 analytes"),
        ("Frozen Raman Atlas", "NMF k=24 · dictionary held fixed"),
        ("Latent Raman motifs", "24 components (mathematical)"),
        ("Molecular Spectral Signatures", "13 validated motifs (chemistry)"),
        ("Biochemical State Vector", "13 themes · elevation + confidence"),
        ("Domain-aware interpretation", "serum / EV / buffer / tissue"),
        ("Evidence report + radar", "provenance · OOD · caveats"),
    ]
    highlight = 3  # the MSS layer — the interpretive centrepiece
    fig, ax = plt.subplots(figsize=(6.4, 8.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, len(stages) * 1.5 + 1.15); ax.axis("off")
    n = len(stages)
    for i, (title, sub) in enumerate(stages):
        y = (n - i - 1) * 1.5 + 0.6
        is_hi = i == highlight
        face = "#eef5fa" if not is_hi else "#fdeef0"
        edge = T.PRIMARY if not is_hi else T.UP
        box = FancyBboxPatch((1.2, y), 7.6, 1.02, boxstyle="round,pad=0.02,rounding_size=0.12",
                             facecolor=face, edgecolor=edge, linewidth=1.6 if is_hi else 1.1)
        ax.add_patch(box)
        ax.text(5.0, y + 0.66, title, ha="center", va="center", fontsize=11.5,
                fontweight="700", color=T.INK)
        ax.text(5.0, y + 0.28, sub, ha="center", va="center", fontsize=8.6, color=T.MUTED)
        if is_hi:
            ax.text(8.95, y + 0.51, "centerpiece", ha="left", va="center", fontsize=8,
                    color=T.UP, fontweight="700", rotation=90)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((5.0, y), (5.0, y - 0.48), arrowstyle="-|>",
                         mutation_scale=14, color=T.FAINT, linewidth=1.3))
    ax.text(5.0, n * 1.5 + 0.7, "Frozen coordinate system · deterministic · provenance-preserving",
            ha="center", va="center", fontsize=8.8, color=T.FAINT, style="italic")
    fig.tight_layout()
    return fig


# ── biochemical theme radar (the engine's canonical radar backend) ──
def radar(radar_axes, title="Biochemical State Vector", score_key="score"):
    """Plot the engine's radar (out.radar['axes']). Uses composition share by
    default — it stays informative in- AND out-of-domain, unlike the tanh display
    value which saturates for far-OOD SERS inputs."""
    vals = [a[score_key] for a in radar_axes]
    labels = [THEME_SHORT.get(a["theme"], a["theme"]) for a in radar_axes]
    ang = np.linspace(0, 2 * np.pi, len(vals), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]]); v_c = np.array(vals + vals[:1])
    vmax = max(vals) * 1.18 if max(vals) > 0 else 1.0
    fig, ax = plt.subplots(figsize=(5.6, 5.6), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, vmax)
    ax.set_yticks([vmax / 2]); ax.set_yticklabels([f"{vmax/2:.2f}"], color=T.FAINT, fontsize=8)
    ax.set_xticks(ang); ax.set_xticklabels(labels, fontsize=9, color=T.INK)
    ax.plot(ang_c, v_c, color=T.PRIMARY, linewidth=2.0)
    ax.fill(ang_c, v_c, color=T.PRIMARY, alpha=0.16)
    ax.scatter(ang, vals, color=T.PRIMARY, s=22, zorder=5)
    ax.grid(color=T.GRID, linewidth=0.7)
    ax.set_title(title + "  ·  evidence share", fontsize=12.0, color=T.INK, pad=18)
    ax.spines["polar"].set_color(T.PANEL_EDGE)
    fig.tight_layout()
    return fig


# ── MSS hierarchy — the centerpiece ──
def mss_hierarchy(activations, title="Molecular Spectral Signatures"):
    """Horizontal motif bars by elevation; diverging colour = increase/decrease."""
    acts = [a for a in activations if not a.non_biochemical]
    acts = sorted(acts, key=lambda a: a.elevation)
    names = [a.name for a in acts]
    elev = np.array([a.elevation for a in acts])
    conf = np.array([a.confidence for a in acts])
    colors = [T.UP if e >= 0 else T.DOWN for e in elev]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    y = np.arange(len(acts))
    ax.barh(y, elev, color=colors, alpha=0.9, height=0.62,
            edgecolor=T.SURFACE, linewidth=1.2)
    ax.axvline(0, color=T.MUTED, linewidth=1.0)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5, color=T.INK)
    ax.set_xlabel("Motif elevation vs pure-Raman reference (signed z)")
    ax.set_title(title, fontsize=12.5, color=T.INK, pad=10)
    # annotate confidence as a faint marker at the bar end
    for yi, (e, c) in enumerate(zip(elev, conf)):
        ax.text(e + (0.02 if e >= 0 else -0.02) * max(abs(elev).max(), 1),
                yi, f"c={c:.2f}", va="center", ha="left" if e >= 0 else "right",
                fontsize=7.4, color=T.FAINT)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.16)
    fig.tight_layout()
    return fig


# ── 24-component fingerprint ──
def component_fingerprint(coord, highlight=None, title="Latent Raman fingerprint (24 components)"):
    coord = np.asarray(coord, float)
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    x = np.arange(len(coord))
    base = [T.PRIMARY_SOFT] * len(coord)
    if highlight:
        for j in highlight:
            base[j] = T.PRIMARY
    ax.bar(x, coord, color=base, width=0.78)
    ax.set_xticks(x); ax.set_xticklabels([f"c{j}" for j in x], fontsize=6.8, color=T.MUTED)
    ax.set_ylabel("evidence share")
    ax.set_title(title, fontsize=11.5, pad=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# ── a component's frozen basis spectrum ──
def basis_spectrum(grid, spectrum, bands=None, title="Basis spectrum"):
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.plot(grid, spectrum, color=T.PRIMARY, linewidth=1.4)
    ax.fill_between(grid, 0, spectrum, color=T.PRIMARY, alpha=0.10)
    if bands:
        for b in bands:
            ax.axvline(b, color=T.UP, linewidth=0.8, alpha=0.55, linestyle="--")
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("loading")
    ax.set_title(title, fontsize=11.5, pad=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# ── spectral collisions map ──
def band_collision_map(motifs, tol=16.0, title="Spectral collisions: where motifs share bands"):
    """Each biochemical motif's characteristic bands as ticks on a shared cm⁻¹ axis.
    Regions claimed by >=3 motifs are shaded — the physical reason multiple motifs
    contribute to one spectral region ('peak != molecule')."""
    mot = [m for m in motifs if not m.non_biochemical]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    all_bands = []
    for i, m in enumerate(mot):
        for b in m.bands_cm:
            ax.scatter(b, i, s=26, color=T.PRIMARY, zorder=3)
            all_bands.append(b)
    all_bands = np.array(all_bands)
    # find collision hotspots: grid points where >=3 DISTINCT motifs have a band within tol,
    # then merge contiguous hotspot cells into single shaded spans (avoids striping).
    grid = np.arange(450, 1801, 4)
    def n_motifs_near(g):
        return sum(any(abs(b - g) <= tol for b in m.bands_cm) for m in mot)
    hot = np.array([n_motifs_near(g) >= 4 for g in grid])   # genuinely congested regions only
    spans, start = [], None
    for gi, h in enumerate(list(hot) + [False]):
        if h and start is None:
            start = grid[gi]
        elif not h and start is not None:
            spans.append((start, grid[gi - 1])); start = None
    for lo, hi in spans:
        ax.axvspan(lo - tol, hi + tol, color=T.UP, alpha=0.12, zorder=0)
    ax.set_yticks(range(len(mot)))
    ax.set_yticklabels([m.name for m in mot], fontsize=8.6, color=T.INK)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_xlim(440, 1810)
    ax.set_title(title, fontsize=12.0, pad=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ── calibration dose-response ──
def dose_response(levels, scores, xlabel="dose", ylabel="theme composition",
                  title="Dose-response"):
    """Faint replicate cloud + bold per-dose mean line (handles many replicates/dose)."""
    levels = np.asarray(levels, float); scores = np.asarray(scores, float)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.scatter(levels, scores, s=10, color=T.PRIMARY_SOFT, alpha=0.5, zorder=1)
    uniq = np.array(sorted(np.unique(levels)))
    mean = np.array([scores[levels == u].mean() for u in uniq])
    ax.plot(uniq, mean, "-o", color=T.PRIMARY, markersize=5, linewidth=1.8, zorder=3)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11.5, pad=8)
    fig.tight_layout()
    return fig
