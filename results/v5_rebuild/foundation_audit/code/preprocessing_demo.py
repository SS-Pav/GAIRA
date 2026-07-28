"""Foundation audit — Part 3: visualize the exact preprocessing pipeline stage by
stage on a representative spectrum, and quantify the negative-value clipping that the
NMF fit applies (crop -> ASLS baseline -> Savitzky-Golay -> resample -> L2 -> clip>=0).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.foundation import dataset as DS
from gaira.preprocessing import pipeline as pp
from gaira.data import gobbato

AUD = REPO / "results/v5_rebuild/foundation_audit"
FIG, TAB = AUD / "figures", AUD / "tables"
GRID, WINDOW, PREPROC = DS.GRID, DS.WINDOW, DS.PREPROC
INK = "#1b2430"

# pick one raw Gobbato Raman spectrum (adenine) to show the stages
raw = None
for s in gobbato.load_gobbato_785():
    if s.record.modality.value == "raman" and "aden" in s.record.canonical_analyte_name.lower():
        raw = (np.asarray(s.wavenumber, float), np.asarray(s.intensity, float)); break
if raw is None:                                   # fallback: first raman spectrum
    for s in gobbato.load_gobbato_785():
        if s.record.modality.value == "raman":
            raw = (np.asarray(s.wavenumber, float), np.asarray(s.intensity, float)); break

wn, y = raw
wn2, y2 = pp.crop(wn, y, *WINDOW)
z = pp.baseline_asls(y2)
yb = y2 - z
ys = pp.smooth_savgol(yb)
yg = pp.resample(wn2, ys, GRID)
finite = np.isfinite(yg)
yl = np.full(len(GRID), np.nan); yl[finite] = pp.norm_l2(yg[finite])
yclip = np.clip(np.nan_to_num(yl), 0, None)

fig, ax = plt.subplots(3, 2, figsize=(11, 8))
ax = ax.ravel()
ax[0].plot(wn2, y2, color=INK, lw=0.8); ax[0].plot(wn2, z, color="#b2182b", lw=1.2, label="ASLS baseline")
ax[0].set_title("1. Cropped raw + ASLS baseline (λ=1e5, p=0.01, 8 iter)"); ax[0].legend(fontsize=8, frameon=False)
ax[1].plot(wn2, yb, color=INK, lw=0.8); ax[1].set_title("2. Baseline-subtracted")
ax[2].plot(wn2, ys, color="#2a6f97", lw=0.9); ax[2].set_title("3. Savitzky-Golay smoothed (win=9, poly=3)")
ax[3].plot(GRID, yg, color="#2a6f97", lw=0.9); ax[3].set_title("4. Resampled to 676-bin 2 cm⁻¹ grid")
ax[4].plot(GRID, yl, color="#2f7d4f", lw=0.9); ax[4].axhline(0, color="#999", lw=0.6)
ax[4].set_title("5. L2-normalized (note negative lobes)")
ax[5].plot(GRID, yclip, color="#2f7d4f", lw=0.9)
ax[5].set_title("6. Clip ≥0 for NMF (what the basis is fit on)")
for a in ax:
    a.set_xlabel("Raman shift (cm⁻¹)", fontsize=8); a.tick_params(labelsize=7)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.suptitle("GAIRA canonical preprocessing — stage by stage (adenine, 785 nm)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIG / "preprocessing_stages.png", dpi=130); plt.close(fig)

# quantify negative clipping across the WHOLE corpus (fraction of mass/points clipped)
c = DS.load_reference_corpus()
X = c.X
neg_frac_points = float(np.mean(np.nan_to_num(X) < 0))
neg_mass = float(np.abs(np.clip(np.nan_to_num(X), None, 0)).sum() / (np.abs(np.nan_to_num(X)).sum() + 1e-12))
finite_frac = float(np.mean(np.isfinite(X)))
stats = {
    "example_analyte": "adenine",
    "n_bins": int(X.shape[1]), "window_cm": list(WINDOW), "grid_step_cm": 2.0,
    "pipeline": PREPROC,
    "asls_params": {"lam": 1e5, "p": 0.01, "n_iter": 8},
    "savgol_params": {"window": 9, "poly": 3},
    "frac_points_negative_before_clip": round(neg_frac_points, 4),
    "frac_absolute_mass_clipped_to_zero": round(neg_mass, 4),
    "frac_grid_finite_mean": round(finite_frac, 4),
}
(TAB / "preprocessing_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
