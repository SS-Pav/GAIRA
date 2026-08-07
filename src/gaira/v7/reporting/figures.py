"""GAIRA V7 — Phase 10: report figures.

Drawn from an `InferenceResult` and nothing else. No engine call, no recomputation — if a value
is not in the result, it is not plotted (P-20).
"""
from __future__ import annotations

import io

import numpy as np

INK, MUTED, RULE = "#1a1a1a", "#6b7280", "#d1d5db"
ACCENT, WARM, GOOD = "#1d4ed8", "#b45309", "#15803d"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    return plt


def _png(fig, dpi=200) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


def spectrum_panel(result) -> bytes:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 3.0))
    pre = result.preprocessing
    if pre.grid and pre.processed_intensity:
        ax.plot(pre.grid, pre.processed_intensity, lw=1.0, color=INK)
        ax.set_xlabel("wavenumber (cm$^{-1}$)"); ax.set_ylabel("normalised intensity")
        ax.set_title("Canonical preprocessed spectrum", fontsize=10, loc="left", color=INK)
    else:
        ax.text(0.5, 0.5, "processed spectrum not requested\n(set include_reconstruction=True)",
                ha="center", va="center", color=MUTED, fontsize=9)
        ax.axis("off")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _png(fig)


def reconstruction_panel(result) -> bytes:
    plt = _mpl()
    pre, csm = result.preprocessing, result.csm
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.4]})
    if pre.grid and pre.processed_intensity and csm and csm.reconstruction:
        g = np.asarray(pre.grid); x = np.asarray(pre.processed_intensity)
        r = np.asarray(csm.reconstruction)
        axes[0].plot(g, x, lw=1.0, color=INK, label="query")
        axes[0].plot(g, r, lw=1.0, color=ACCENT, alpha=0.85, label="CSM reconstruction")
        axes[0].legend(frameon=False, fontsize=8, loc="upper right")
        axes[0].set_ylabel("intensity")
        axes[0].set_title(f"Reconstruction — explained variance "
                          f"{csm.explained_variance:.3f}, residual fraction "
                          f"{csm.residual_fraction:.3f}", fontsize=10, loc="left", color=INK)
        axes[1].plot(g, x - r, lw=0.8, color=WARM)
        axes[1].axhline(0, color=RULE, lw=0.6)
        axes[1].set_ylabel("residual"); axes[1].set_xlabel("wavenumber (cm$^{-1}$)")
    else:
        for a in axes:
            a.axis("off")
        axes[0].text(0.5, 0.5, "reconstruction not requested", ha="center", color=MUTED)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    return _png(fig)


def csm_panel(result) -> bytes:
    plt = _mpl()
    csm = result.csm
    fig, ax = plt.subplots(figsize=(9, 2.8))
    if csm:
        a = np.asarray(csm.activation)
        ax.bar(np.arange(len(a)), a, color=[ACCENT if v > 0 else RULE for v in a], width=0.8)
        ax.set_xlabel("CSM index"); ax.set_ylabel("activation")
        ax.set_title(f"CSM activation — {csm.n_active} of {len(a)} active, sparsity "
                     f"{csm.sparsity:.3f}", fontsize=10, loc="left", color=INK)
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "CSM not requested", ha="center", color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _png(fig)


def chemistry_panel(result) -> bytes:
    """Radar and ordered bars side by side. The bars are the precision default."""
    plt = _mpl()
    chem = result.chemistry
    n = len(chem.axis_names)
    vals = np.asarray(chem.evidence_l1)
    names = [a.replace("_", " ") for a in chem.axis_names]
    fig = plt.figure(figsize=(10.5, 5.2))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.concatenate([vals, vals[:1]]); a = np.concatenate([ang, ang[:1]])
    ax1.plot(a, v, color=ACCENT, lw=1.4)
    ax1.fill(a, v, color=ACCENT, alpha=0.18)
    ax1.set_xticks(ang)
    ax1.set_xticklabels([s.replace(" ", "\n") for s in names], fontsize=7.0)
    ax1.tick_params(pad=6)
    ax1.set_yticklabels([])
    ax1.set_title("Relative Chemistry Evidence", fontsize=10, color=INK, pad=16)

    ax2 = fig.add_subplot(1, 2, 2)
    order = np.argsort(vals)
    ax2.barh(np.arange(n), vals[order], color=ACCENT, alpha=0.85, height=0.72)
    ax2.set_yticks(np.arange(n))
    ax2.set_yticklabels([names[i] for i in order], fontsize=8.4)
    ax2.set_xlabel("relative evidence (share of total)")
    ax2.set_title("Ordered evidence", fontsize=10, loc="left", color=INK)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    fig.text(0.5, -0.02, "RELATIVE BIOCHEMICAL EVIDENCE — not a concentration, not an "
             "abundance, not a mixture fraction.", ha="center", fontsize=7.6, color=WARM)
    fig.tight_layout()
    return _png(fig)


def retrieval_panel(result) -> bytes:
    plt = _mpl()
    hits = result.retrieval.top
    fig, ax = plt.subplots(figsize=(9, max(2.4, 0.32 * len(hits) + 1.0)))
    sims = [h.similarity for h in hits][::-1]
    labs = [f"{h.rank}. {h.molecule}" for h in hits][::-1]
    ax.barh(np.arange(len(sims)), sims, color=ACCENT, alpha=0.85, height=0.7)
    ax.set_yticks(np.arange(len(sims))); ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel("CSM cosine similarity")
    ax.set_xlim(0, 1.0)
    ax.set_title("Grounded Evidence Retrieval — reference analogues, not identifications",
                 fontsize=10, loc="left", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _png(fig)


def comparison_panel(comparison) -> bytes:
    plt = _mpl()
    d = comparison.chemistry_delta
    names = [x.axis.replace("_", " ") for x in d]
    delta = np.asarray([x.delta for x in d])
    order = np.argsort(delta)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(np.arange(len(d)), delta[order],
            color=[GOOD if v >= 0 else WARM for v in delta[order]], height=0.72)
    ax.axvline(0, color=RULE, lw=0.8)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([names[i] for i in order], fontsize=7.5)
    ax.set_xlabel(f"chemistry evidence:  {comparison.label_b} − {comparison.label_a}")
    ax.set_title("Difference in relative chemistry evidence", fontsize=10, loc="left", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _png(fig)
