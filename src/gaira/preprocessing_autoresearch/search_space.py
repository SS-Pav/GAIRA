"""Stage B0 — the PREDECLARED search space and the controlled experimental arms.

The search is structured (arm-by-arm), not a brute-force Cartesian product. Every
option is physically motivated and bounded. Deterministic candidate ids.
"""
from __future__ import annotations
from .pipeline import Candidate

# ── bounded stage options ──
BASELINES = [
    ("none", {}),
    ("asls", {"lam": 1e4}), ("asls", {"lam": 1e5}), ("asls", {"lam": 1e6}),
    ("airpls", {"lam": 1e5}), ("arpls", {"lam": 1e5}),
    ("rubberband", {}), ("poly3", {}), ("morph", {"size": 61}),
]
SMOOTHERS = [
    ("none", {}),
    ("savgol", {"window": 5, "poly": 2}), ("savgol", {"window": 7, "poly": 3}),
    ("savgol", {"window": 9, "poly": 3}), ("savgol", {"window": 11, "poly": 3}),
    ("savgol", {"window": 15, "poly": 3}), ("savgol", {"window": 21, "poly": 3}),
    ("gaussian", {"sigma": 1.0}), ("gaussian", {"sigma": 2.0}), ("gaussian", {"sigma": 3.0}),
    ("whittaker", {"lam": 5.0}), ("whittaker", {"lam": 50.0}),
    ("wavelet", {"level": 2}),
]
NORMS = ["l2", "area", "robust", "p95", "max", "snv"]   # snv = declared control
AGGREGATORS = ["mean", "median", "ivw", "huber", "consensus"]
DERIVATIVES = [0, 1, 2, "concat"]
PEAK_TRANSFORMS = ["full", "peak_likelihood", "prominence", "ridge"]

from . import background_models as BG
BACKGROUNDS = BG.CANDIDATES


def _mk(cid, arm, *, base=("asls", {"lam": 1e5}), smooth=("none", {}),
        base_s=None, smooth_s=None, bg=("none", {}), agg="mean",
        deriv=0, nr="l2", ns=None, pk="full"):
    base_s = base_s or base; smooth_s = smooth_s or smooth; ns = ns or nr
    return Candidate(
        cid=cid, arm=arm,
        raman=dict(baseline=base[0], baseline_params=base[1],
                   smooth=smooth[0], smooth_params=smooth[1]),
        sers=dict(baseline=base_s[0], baseline_params=base_s[1],
                  smooth=smooth_s[0], smooth_params=smooth_s[1]),
        background=bg, aggregate=agg, derivative=deriv,
        norm_raman=nr, norm_sers=ns, peak_transform=pk)


# ───────────────────────── frozen baselines (reproduced exactly) ─────────────────────────
def baseline_arm():
    """Prior baselines reproduced unchanged, for comparability with Stage A/B."""
    out = [
        _mk("BASE_raw_l2", "baseline", base=("none", {}), smooth=("none", {}), nr="l2"),
        _mk("BASE_asls_sg_l2", "baseline", base=("asls", {"lam": 1e5}),
            smooth=("savgol", {"window": 9, "poly": 3}), nr="l2"),
        _mk("BASE_asls_sg_snv", "baseline", base=("asls", {"lam": 1e5}),
            smooth=("savgol", {"window": 9, "poly": 3}), nr="snv"),
        _mk("BASE_asls_sg_area", "baseline", base=("asls", {"lam": 1e5}),
            smooth=("savgol", {"window": 9, "poly": 3}), nr="area"),
        _mk("BASE_asls_sg_l2_d1", "baseline", base=("asls", {"lam": 1e5}),
            smooth=("savgol", {"window": 9, "poly": 3}), nr="l2", deriv=1),
        _mk("BASE_asls_sg_l2_bgmean", "baseline", base=("asls", {"lam": 1e5}),
            smooth=("savgol", {"window": 9, "poly": 3}), nr="l2", bg=("mean", {})),
    ]
    return out


# ───────────────────────── arms ─────────────────────────
def arm_A():
    """Baseline correction x normalization (no smoothing, no background)."""
    out = []
    for bi, (bm, bp) in enumerate(BASELINES):
        for nm in NORMS:
            out.append(_mk(f"A_{bm}{bi}_{nm}", "A_baseline_norm",
                           base=(bm, bp), smooth=("none", {}), nr=nm))
    return out


def arm_B(best_base):
    """Smoothing study on the best baselines."""
    out = []
    for bi, b in enumerate(best_base):
        for si, (sm, sp) in enumerate(SMOOTHERS):
            out.append(_mk(f"B_{bi}_{sm}{si}", "B_smoothing", base=b, smooth=(sm, sp), nr="l2"))
    return out


def arm_C(best_base, best_smooth):
    return [_mk(f"C_{a}", "C_aggregation", base=best_base, smooth=best_smooth, agg=a, nr="l2")
            for a in AGGREGATORS]


def arm_D(best_base, best_smooth, agg="mean"):
    """Ag-SERS common-background correction — the primary component."""
    out = []
    for i, (m, p) in enumerate(BACKGROUNDS):
        out.append(_mk(f"D_{m}{i}", "D_background", base=best_base, smooth=best_smooth,
                       agg=agg, bg=(m, p), nr="l2"))
    return out


def arm_E(best_base, best_smooth, bg, agg="mean"):
    return [_mk(f"E_d{d}", "E_derivative", base=best_base, smooth=best_smooth,
                agg=agg, bg=bg, deriv=d, nr="l2") for d in DERIVATIVES]


def arm_F(best_r, best_s, best_smooth_r, best_smooth_s, bg, agg="mean"):
    """Modality-specific vs global pipelines."""
    out = [
        _mk("F_global", "F_modality", base=best_r, smooth=best_smooth_r, bg=bg, agg=agg, nr="l2"),
        _mk("F_modspec", "F_modality", base=best_r, smooth=best_smooth_r,
            base_s=best_s, smooth_s=best_smooth_s, bg=bg, agg=agg, nr="l2"),
        _mk("F_modspec_norm", "F_modality", base=best_r, smooth=best_smooth_r,
            base_s=best_s, smooth_s=best_smooth_s, bg=bg, agg=agg, nr="l2", ns="area"),
    ]
    for pk in PEAK_TRANSFORMS[1:]:
        out.append(_mk(f"F_pk_{pk}", "F_modality", base=best_r, smooth=best_smooth_r,
                       base_s=best_s, smooth_s=best_smooth_s, bg=bg, agg=agg, nr="l2", pk=pk))
    return out


def arm_G(components):
    """A small number of rational combined pipelines (NOT the Cartesian product)."""
    out = []
    for i, c in enumerate(components):
        out.append(_mk(f"G_combo{i}", "G_combined", **c))
    return out
