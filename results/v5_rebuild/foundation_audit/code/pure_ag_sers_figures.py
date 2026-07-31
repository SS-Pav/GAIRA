"""Figures for the pure Ag-SERS validation: the analyte recoverability ranking (the
headline figure) and the per-family transfer summary. Reads the committed artifact."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUD = Path("/Users/surajpg/projects/GAIRA/results/v5_rebuild/foundation_audit")
df = pd.read_csv(AUD / "tables/pure_ag_sers_per_analyte.csv")
summ = json.loads((AUD / "tables/pure_ag_sers_validation.json").read_text())["summary"]
INK = "#16202c"
TIER_COLOR = {"Excellent": "#1a5e3a", "Good": "#2f7d4f", "Moderate": "#b8862a",
              "Weak": "#c0603a", "Poor": "#b2182b"}
ORDER = ["Excellent", "Good", "Moderate", "Weak", "Poor"]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


# 1 · recoverability ranking (all 51 analytes) — the headline
d = df.sort_values("coord_cosine")
colors = [TIER_COLOR[t] for t in d.recoverability_tier]
fig, ax = plt.subplots(figsize=(7.8, 11))
y = np.arange(len(d))
ax.barh(y, d.coord_cosine, color=colors, height=0.72, edgecolor="white", linewidth=0.4, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([f"{a}" for a in d.analyte], fontsize=7.4)
ax.set_xlim(0, 1)
for name, thr in [("0.80", 0.80), ("0.65", 0.65), ("0.45", 0.45), ("0.25", 0.25)]:
    ax.axvline(thr, color="#c9cfd6", lw=0.8, ls=":", zorder=1)
ax.set_xlabel("Raman ↔ Ag-SERS coordinate cosine  (1 = signature preserved)", fontsize=9)
ax.set_title(f"Pure Ag-SERS recoverability — 51 matched analytes\n"
             f"median {summ['median_coord_cosine']:.2f} · theme preserved "
             f"{summ['n_theme_preserved']}/51", fontsize=11, color=INK)
handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLOR[t]) for t in ORDER]
labels = [f"{t} ({summ['tier_counts'].get(t,0)})" for t in ORDER]
ax.legend(handles, labels, fontsize=8, loc="lower right", frameon=False, title="tier")
style(ax)
fig.tight_layout()
fig.savefig(AUD / "figures/pure_ag_sers_ranking.png", dpi=130)
plt.close(fig)

# 2 · per-family transfer summary
fam = summ["family_coord_cosine"]
famdf = pd.DataFrame([{"family": k, "mean": v["mean"], "n": v["n"]} for k, v in fam.items()])
famdf = famdf.sort_values("mean")
fig, ax = plt.subplots(figsize=(7.2, 4.6))
bar_c = ["#2f7d4f" if m >= 0.55 else "#b8862a" if m >= 0.4 else "#b2182b" for m in famdf["mean"]]
ax.barh(famdf.family, famdf["mean"], color=bar_c, zorder=3, edgecolor="white", linewidth=0.4)
for i, (m, n) in enumerate(zip(famdf["mean"], famdf["n"])):
    ax.text(m + 0.01, i, f"{m:.2f}  (n={n})", va="center", fontsize=7.6, color=INK)
ax.set_xlim(0, 1)
ax.set_xlabel("mean Raman↔Ag-SERS coordinate cosine", fontsize=9)
ax.set_title("Transfer by chemical family — adsorption affinity ranking", fontsize=11, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(AUD / "figures/pure_ag_sers_by_family.png", dpi=130)
plt.close(fig)

print("wrote pure_ag_sers_ranking.png + pure_ag_sers_by_family.png")
print("tier counts:", summ["tier_counts"])
