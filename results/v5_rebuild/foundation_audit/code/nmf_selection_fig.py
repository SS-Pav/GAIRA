"""Foundation audit — Part 4: k / representation selection figure from the reproduced
benchmark, plus a per-criterion breakdown of the winner NMF k=24 vs raw-top ICA k=32."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUD = Path("/Users/surajpg/projects/GAIRA/results/v5_rebuild/foundation_audit")
df = pd.read_csv(AUD / "tables/c1_representation_benchmark_repro.csv")
INK = "#1b2430"
COL = {"NMF": "#2a6f97", "ICA": "#b2182b", "PCA": "#2f7d4f",
       "SparseDict": "#8a6d3b", "Autoencoder": "#7a3b8a"}

# 1. total_score vs k per representation, tie band shaded
fig, ax = plt.subplots(figsize=(7.6, 4.8))
top = df.total_score.max()
ax.axhspan(top - 0.02, top, color="#ffe08a", alpha=0.4, zorder=0, label="tie band (0.02)")
for name, g in df.groupby("representation"):
    g = g.sort_values("k")
    ax.plot(g.k, g.total_score, "-o", color=COL.get(name, "#555"), label=name, lw=1.6, ms=5)
# mark the selected NMF k=24
sel = df[(df.representation == "NMF") & (df.k == 24)].iloc[0]
ax.scatter([24], [sel.total_score], s=220, facecolor="none", edgecolor=INK, linewidth=2.0, zorder=5)
ax.annotate("SELECTED\nNMF k=24", (24, sel.total_score), textcoords="offset points",
            xytext=(6, -38), fontsize=8.5, color=INK, fontweight="bold")
ax.set_xlabel("latent dimension k"); ax.set_ylabel("multi-criteria selection score")
ax.set_title("Representation benchmark — score vs k (reproduced, seed 0)", color=INK, fontsize=11)
ax.legend(fontsize=8, frameon=False, ncol=2); ax.set_xticks([4, 8, 12, 16, 24, 32])
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#d7dce3", lw=0.6)
fig.tight_layout(); fig.savefig(AUD / "figures/nmf_selection_score_vs_k.png", dpi=130); plt.close(fig)

# 2. per-criterion radar-ish bar: NMF24 vs ICA32 vs PCA24 on the six sub-scores
def sub(df):
    d = df.copy()
    def n01(x, inv=False):
        x = np.asarray(x, float); lo, hi = np.nanmin(x), np.nanmax(x)
        v = (x - lo) / (hi - lo + 1e-12); return 1 - v if inv else v
    d["nbr"] = n01(d.neighbourhood_preservation); d["rep"] = n01(d.replicate_robustness)
    d["stab"] = n01(d.component_stability)
    d["interp"] = 0.5 * n01(d.loading_sparsity) + 0.5 * n01(d.band_localisation, inv=True)
    d["recon"] = n01(d.recon_rel_error, inv=True)
    d["nuis"] = 0.5 * n01(np.abs(d.excitation_leakage), inv=True) + 0.5 * n01(np.abs(d.source_leakage), inv=True)
    return d
d = sub(df)
crit = ["nbr", "rep", "stab", "interp", "recon", "nuis"]
labels = ["neighbourhood\n(0.25)", "replicate\n(0.25)", "stability\n(0.20)",
          "interpret\n(0.15)", "reconstruct\n(0.10)", "nuisance\n(0.05)"]
picks = [("NMF", 24), ("ICA", 32), ("PCA", 24)]
x = np.arange(len(crit)); w = 0.26
fig, ax = plt.subplots(figsize=(8.4, 4.4))
for i, (nm, k) in enumerate(picks):
    r = d[(d.representation == nm) & (d.k == k)].iloc[0]
    ax.bar(x + (i - 1) * w, [r[c] for c in crit], w, label=f"{nm} k={k}", color=COL[nm], zorder=3)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("normalized sub-score (0–1)"); ax.legend(fontsize=8, frameon=False)
ax.set_title("Per-criterion breakdown — why NMF k=24 wins the parts-based tie", color=INK, fontsize=11)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#d7dce3", lw=0.6)
fig.tight_layout(); fig.savefig(AUD / "figures/nmf_selection_criteria.png", dpi=130); plt.close(fig)

# dump the exact comparison numbers
out = {}
for nm, k in [("NMF", 24), ("ICA", 32), ("PCA", 24), ("Autoencoder", 24), ("SparseDict", 24)]:
    r = df[(df.representation == nm) & (df.k == k)]
    if len(r):
        r = r.iloc[0]
        out[f"{nm}_k{k}"] = {c: round(float(r[c]), 4) for c in
                             ["total_score", "recon_rel_error", "neighbourhood_preservation",
                              "replicate_robustness", "component_stability", "loading_sparsity",
                              "band_localisation", "excitation_leakage", "source_leakage", "nonneg"]}
(AUD / "tables/nmf_selection_comparison.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
