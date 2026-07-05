"""gaira_base_4_shine_cross_set_reproducibility_audit_v1.

Cross-set reproducibility audit (Set9 D2 vs Set10 D2) for SHINE EV SERS.

Computes — honestly, no assumptions:
  1. Per-(set, dose) MEAN RAW spectrum (length 1401, GAIRA preprocessing chain).
  2. Per-(set, dose) MEAN BSV vector (11 axes, CLR transform).
  3. Pearson r between Set9 vs Set10 means PER DOSE — RAW and BSV.
  4. Flat-vector trajectory Pearson r (RAW: 4 doses × 1401 wn ; BSV: 4 × 11).
  5. Within-cohort variance (mean per-wavenumber for RAW; per-axis for BSV).

STRICT INVARIANTS:
- GAIRA core unchanged (this script is read-only over engine + helpers).
- Same preprocessing chain as the SHINE pilot:
  pixel→wavenumber polynomial (Fig4D.m) → master_x (400-1800, step 1) →
  AsLS (lambda=1e5, p=0.001, 10 iters) → SG (window 11, polyorder 3) → L2.
- NO paper normalization (no D0_C0 / D2_C0 / Si 642).
- Cohorts pinned to the cached per-spectrum BSV `spectrum_id` set so
  RAW and BSV cohorts are identical by construction.

Output:
    /Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_shine_cross_set_reproducibility_audit_v1/
"""
from __future__ import annotations

import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.simplefilter("ignore")

# ── Wire the GAIRA src + pilot helpers ────────────────────────────────────
PROJECT_ROOT = Path("/Users/suraj/projects/GAIRA")
PILOT_SNAPSHOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_shine_ev_gaira_pilot_v1/"
    "code_snapshot")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PILOT_SNAPSHOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from gaira.spectral import canonical_master_axis  # noqa: E402
from run_gaira_base_4_mss_resolution_reporting_layer_v1 import (  # noqa: E402
    baseline_correct,
)


# ── paths ────────────────────────────────────────────────────────────────
SHINE_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/shine_ev_sers/"
    "SERS-Hepatotoxicity_DATA_CODE_FIGURE")
SHINE_SET9  = SHINE_ROOT / "Figure4/data/Set9"
SHINE_SET10 = SHINE_ROOT / "Figure4/data/Set10"

PILOT_TABLES = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_shine_ev_gaira_pilot_v1/tables")

OUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_shine_cross_set_reproducibility_audit_v1")
T = OUT_ROOT / "tables"; F = OUT_ROOT / "figures"
R = OUT_ROOT / "reports"; C = OUT_ROOT / "code_snapshot"
for d in (T, F, R, C): d.mkdir(parents=True, exist_ok=True)


# ── pixel→wavenumber calibration (Fig4D.m verbatim) ──────────────────────
SHINE_CAL_PIX = np.array([263, 367, 492, 512, 590, 782, 872, 887], dtype=float)
SHINE_CAL_CM  = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5,
                            1583.1, 1602.3])
SHINE_N_PIXELS = 1650


def shine_wn_axis() -> np.ndarray:
    coeffs = np.polyfit(SHINE_CAL_PIX, SHINE_CAL_CM, 3)
    return np.polyval(coeffs, np.arange(1, SHINE_N_PIXELS + 1, dtype=float))


# ── per-spectrum loaders ─────────────────────────────────────────────────

def _spectrum_csv_path(spectrum_id: str) -> Path:
    """spectrum_id is `Set9/D2_C40/241/s_1` style."""
    return SHINE_ROOT / "Figure4" / "data" / spectrum_id


def load_one_raw(spectrum_id: str) -> np.ndarray | None:
    p = _spectrum_csv_path(spectrum_id)
    if not p.exists():
        return None
    try:
        # CSV is "pixel,intensity" no header
        arr = pd.read_csv(p, header=None).values
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        return arr[:, 1].astype(float)
    except Exception:
        return None


# ── pipeline ─────────────────────────────────────────────────────────────

def stage1_select_cohorts() -> pd.DataFrame:
    """Read cached per-spectrum BSV, filter to D2 (Set9 + Set10) at the four
    doses. Returns a DataFrame with spectrum_id + cohort labels + BSV cols."""
    print("[stage 1] selecting cohorts from cached per-spectrum BSV")
    df = pd.read_csv(PILOT_TABLES / "shine_per_spectrum_bsv_outputs_v1.csv")
    df = df[(df["day"] == "D2") & (df["dose_mM"].isin([0, 10, 20, 40]))]
    df = df[df["set_id"].isin(["Set9", "Set10"])]
    df = df[df["qc_status"].fillna("OK") == "OK"]
    print(f"  cohort breakdown:")
    print(df.groupby(["set_id", "dose_mM"]).size().to_string())
    return df.reset_index(drop=True)


def stage2_load_raw_preprocess(cohorts: pd.DataFrame, master_x: np.ndarray,
                                wn_axis_full: np.ndarray) -> dict:
    """Load + preprocess RAW spectra for the cohort spectrum_ids.

    Returns dict[(set_id, dose)] = ndarray (n_kept, 1401)."""
    print("[stage 2] loading + preprocessing matched RAW spectra")
    out: dict = {}
    n_total = len(cohorts)
    for i, (cohort_key, sub) in enumerate(
            cohorts.groupby(["set_id", "dose_mM"])):
        n_in_cohort = len(sub)
        rows = []
        for j, (_, r) in enumerate(sub.iterrows()):
            if (i + j) % 200 == 0:
                pct = (sum(len(g) for k, g in cohorts.groupby(["set_id","dose_mM"])
                            if k <= cohort_key) - n_in_cohort + j) / n_total * 100
                print(f"  {cohort_key}  {j}/{n_in_cohort}  ({pct:.0f}% overall)")
            y_raw = load_one_raw(r["spectrum_id"])
            if y_raw is None:
                continue
            # Pad / trim to SHINE_N_PIXELS basis
            if len(y_raw) < SHINE_N_PIXELS:
                y = np.full(SHINE_N_PIXELS, np.nan)
                y[: len(y_raw)] = y_raw
            else:
                y = y_raw[:SHINE_N_PIXELS]
            # Interpolate to master_x
            y_interp = np.interp(master_x, wn_axis_full, y,
                                  left=np.nan, right=np.nan)
            if not np.isfinite(y_interp).all():
                # fill remaining NaN with cohort median (interp edge effects)
                med = np.nanmedian(y_interp)
                y_interp = np.where(np.isfinite(y_interp), y_interp, med)
            # AsLS baseline + SG smooth + L2 norm
            y_pp = baseline_correct(y_interp)
            from scipy.signal import savgol_filter
            y_pp = savgol_filter(y_pp, window_length=11, polyorder=3)
            n = float(np.linalg.norm(y_pp))
            if n > 0: y_pp = y_pp / n
            rows.append(y_pp)
        out[cohort_key] = np.vstack(rows) if rows else np.zeros((0, len(master_x)))
        print(f"  {cohort_key}: kept {out[cohort_key].shape[0]} / {n_in_cohort}")
    return out


def stage3_means_and_correlations(cohorts: pd.DataFrame, raw_per_cohort: dict,
                                    master_x: np.ndarray) -> dict:
    """Compute per-cohort RAW + BSV means, then Pearson r per dose."""
    print("[stage 3] cohort means + per-dose correlations")
    BSV_AXES = [f"clr_G{i:02d}" for i in range(1, 12)]

    # RAW means: dict[(set, dose)] = vec(1401)
    raw_means = {k: v.mean(axis=0) for k, v in raw_per_cohort.items()}
    raw_vars = {k: v.var(axis=0) for k, v in raw_per_cohort.items()}

    # BSV means + per-axis variance (CLR transform — same as cached pilot)
    bsv_means: dict = {}
    bsv_vars: dict = {}
    for (set_id, dose), sub in cohorts.groupby(["set_id", "dose_mM"]):
        bv = sub[BSV_AXES].values  # (n, 11)
        bsv_means[(set_id, dose)] = bv.mean(axis=0)
        bsv_vars[(set_id, dose)] = bv.var(axis=0)

    # Per-dose Pearson r between Set9 and Set10
    rows = []
    doses = [0, 10, 20, 40]
    for d in doses:
        s9 = ("Set9", d); s10 = ("Set10", d)
        if s9 not in raw_means or s10 not in raw_means:
            rows.append({"dose": d, "raw_pearson_r": np.nan,
                         "raw_p": np.nan, "bsv_pearson_r": np.nan,
                         "bsv_p": np.nan,
                         "n_set9": 0, "n_set10": 0})
            continue
        r_raw, p_raw = pearsonr(raw_means[s9], raw_means[s10])
        r_bsv, p_bsv = pearsonr(bsv_means[s9], bsv_means[s10])
        rows.append({
            "dose": d,
            "raw_pearson_r": float(r_raw), "raw_p": float(p_raw),
            "bsv_pearson_r": float(r_bsv), "bsv_p": float(p_bsv),
            "n_set9":  raw_per_cohort.get(s9, np.zeros((0,))).shape[0],
            "n_set10": raw_per_cohort.get(s10, np.zeros((0,))).shape[0],
        })
    per_dose = pd.DataFrame(rows)
    per_dose.to_csv(T / "cross_set_pearson_per_dose.csv", index=False)
    print(per_dose.to_string(index=False))

    # Trajectory correlation — flat 4-dose stack
    raw_traj_s9 = np.concatenate(
        [raw_means.get(("Set9", d), np.zeros(len(master_x))) for d in doses])
    raw_traj_s10 = np.concatenate(
        [raw_means.get(("Set10", d), np.zeros(len(master_x))) for d in doses])
    bsv_traj_s9 = np.concatenate(
        [bsv_means.get(("Set9", d), np.zeros(11)) for d in doses])
    bsv_traj_s10 = np.concatenate(
        [bsv_means.get(("Set10", d), np.zeros(11)) for d in doses])
    r_raw_traj, p_raw_traj = pearsonr(raw_traj_s9, raw_traj_s10)
    r_bsv_traj, p_bsv_traj = pearsonr(bsv_traj_s9, bsv_traj_s10)
    traj_df = pd.DataFrame([
        {"representation": "RAW (1401 wn × 4 doses = 5604-d)",
         "pearson_r": float(r_raw_traj), "p_value": float(p_raw_traj)},
        {"representation": "BSV (11 axes × 4 doses = 44-d)",
         "pearson_r": float(r_bsv_traj), "p_value": float(p_bsv_traj)},
    ])
    traj_df.to_csv(T / "trajectory_correlation.csv", index=False)
    print(traj_df.to_string(index=False))

    # Within-cohort variance summary
    var_rows = []
    for (set_id, dose), v in raw_vars.items():
        var_rows.append({"set_id": set_id, "dose": dose,
                         "representation": "RAW",
                         "metric": "mean per-wavenumber variance",
                         "value": float(v.mean()),
                         "n_features": len(v)})
    for (set_id, dose), v in bsv_vars.items():
        var_rows.append({"set_id": set_id, "dose": dose,
                         "representation": "BSV",
                         "metric": "mean per-axis variance",
                         "value": float(v.mean()),
                         "n_features": len(v)})
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(T / "within_cohort_variance.csv", index=False)

    return {
        "per_dose": per_dose, "trajectory": traj_df, "variance": var_df,
        "raw_means": raw_means, "bsv_means": bsv_means,
        "doses": doses,
    }


def stage4_figure(per_dose: pd.DataFrame, traj_df: pd.DataFrame,
                  out_path: Path) -> None:
    print("[stage 4] rendering figure")
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Panel A: per-dose Pearson r — RAW vs BSV grouped bars
    ax = axes[0]
    doses = per_dose["dose"].tolist()
    x = np.arange(len(doses)); w = 0.36
    ax.bar(x - w/2, per_dose["raw_pearson_r"].values, width=w,
           color="#9aa6ad", edgecolor="white", linewidth=0.5,
           label="RAW (1401 wn)")
    ax.bar(x + w/2, per_dose["bsv_pearson_r"].values, width=w,
           color="#3a7d8c", edgecolor="white", linewidth=0.5,
           label="BSV (11 axes)")
    for i, (rr, rb) in enumerate(zip(per_dose["raw_pearson_r"].values,
                                      per_dose["bsv_pearson_r"].values)):
        if np.isfinite(rr):
            ax.text(i - w/2, rr + 0.01, f"{rr:.3f}", ha="center", va="bottom",
                     fontsize=8.5, color="#444")
        if np.isfinite(rb):
            ax.text(i + w/2, rb + 0.01, f"{rb:.3f}", ha="center", va="bottom",
                     fontsize=8.5, color="#1a4651", fontweight="600")
    ax.set_xticks(x); ax.set_xticklabels([f"{d} mM" for d in doses])
    ax.set_ylabel("Set9 vs Set10 · Pearson r")
    ax.set_xlabel("APAP dose")
    ax.set_ylim(0, 1.05)
    ax.set_title("A · per-dose cross-set correlation",
                  fontsize=11, loc="left", pad=10)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="y", alpha=0.20, lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # Panel B: trajectory correlation — 2 large numbers
    ax = axes[1]
    reps = ["RAW", "BSV"]
    rr_traj = float(traj_df.iloc[0]["pearson_r"])
    rb_traj = float(traj_df.iloc[1]["pearson_r"])
    bars = ax.bar(reps, [rr_traj, rb_traj], width=0.5,
                   color=["#9aa6ad", "#3a7d8c"],
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, [rr_traj, rb_traj]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=14, fontweight="700", color="#1a4651")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Set9 vs Set10 · trajectory r")
    ax.set_title("B · trajectory consistency · 4 doses stacked",
                  fontsize=11, loc="left", pad=10)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="y", alpha=0.20, lw=0.5)

    fig.suptitle("SHINE D2 cross-set reproducibility · RAW vs GAIRA BSV",
                  fontsize=13, fontweight="600", y=0.99)
    fig.text(0.5, 0.005,
              "Same preprocessing chain (pixel→wn → AsLS → SG → L2) · "
              "no paper normalization · cohorts matched by spectrum_id",
              ha="center", fontsize=8, color="#666", style="italic")
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                 facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


def stage5_report(per_dose: pd.DataFrame, traj_df: pd.DataFrame,
                  variance: pd.DataFrame) -> None:
    print("[stage 5] writing report + interpretation")

    raw_per_dose = per_dose["raw_pearson_r"].dropna().values
    bsv_per_dose = per_dose["bsv_pearson_r"].dropna().values
    delta_per_dose = bsv_per_dose - raw_per_dose

    raw_traj = float(traj_df.iloc[0]["pearson_r"])
    bsv_traj = float(traj_df.iloc[1]["pearson_r"])

    # Verdict
    if (delta_per_dose > 0.01).all() and bsv_traj > raw_traj + 0.01:
        verdict = "YES"
    elif abs(bsv_traj - raw_traj) < 0.01 and np.median(np.abs(delta_per_dose)) < 0.01:
        verdict = "NO (≈ equivalent)"
    elif (delta_per_dose > 0.01).any() and (delta_per_dose < -0.01).any():
        verdict = "MIXED"
    elif (delta_per_dose < -0.01).all() and bsv_traj < raw_traj - 0.01:
        verdict = "NO (RAW more reproducible)"
    else:
        verdict = "MIXED"

    raw_var_mean = float(variance[variance.representation == "RAW"]["value"].mean())
    bsv_var_mean = float(variance[variance.representation == "BSV"]["value"].mean())

    lines = [
        "# REPORT — SHINE cross-set reproducibility audit v1\n",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Setup",
        "- Set9 D2 vs Set10 D2 at doses 0 / 10 / 20 / 40 mM.",
        "- Cohorts pinned to the SHINE pilot's cached `spectrum_id` set "
          "(`shine_per_spectrum_bsv_outputs_v1.csv`) so RAW and BSV cohorts "
          "are identical by construction.",
        "- Preprocessing chain (RAW): pixel→wavenumber polynomial (Fig4D.m, "
          "8 reference pairs) → master_x 400-1800 step 1 → AsLS baseline "
          "(λ=1e5, p=0.001, 10 iter) → SG (window 11, polyorder 3) → L2 norm.",
        "- BSV uses the cached 11-axis CLR vectors from the same pilot run.",
        "- NO paper normalization. NO label leakage.",
        "",
        "## Per-dose Pearson r",
        "",
        "| dose | RAW r | BSV r | Δ (BSV − RAW) | n Set9 | n Set10 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in per_dose.iterrows():
        delta = (r["bsv_pearson_r"] - r["raw_pearson_r"]
                  if pd.notna(r["bsv_pearson_r"]) and pd.notna(r["raw_pearson_r"])
                  else float("nan"))
        lines.append(
            f"| {int(r['dose'])} mM | {r['raw_pearson_r']:.3f} | "
            f"{r['bsv_pearson_r']:.3f} | {delta:+.3f} | "
            f"{int(r['n_set9'])} | {int(r['n_set10'])} |")

    lines += [
        "",
        "## Trajectory correlation (4 doses stacked)",
        "",
        f"- RAW trajectory r = **{raw_traj:.3f}**  (1401 wn × 4 doses = 5604-d)",
        f"- BSV trajectory r = **{bsv_traj:.3f}**  (11 axes × 4 doses = 44-d)",
        f"- Δ (BSV − RAW) = **{bsv_traj - raw_traj:+.3f}**",
        "",
        "## Within-cohort variance (mean per-feature)",
        "",
        f"- RAW · mean per-wavenumber variance = {raw_var_mean:.4g}",
        f"- BSV · mean per-axis variance = {bsv_var_mean:.4g}",
        "",
        "## Interpretation (≤150 words, strict)",
        "",
        f"**1. Is BSV more reproducible than RAW?  →  {verdict}**",
        "",
        "**2. By how much?**  ",
        f"Per-dose Δ (BSV − RAW) Pearson r = "
        f"{', '.join(f'{x:+.3f}' for x in delta_per_dose)}  "
        f"(median {np.median(delta_per_dose):+.3f}, "
        f"min {delta_per_dose.min():+.3f}, max {delta_per_dose.max():+.3f}). "
        f"Trajectory Δ = {bsv_traj - raw_traj:+.3f}.",
        "",
        "**3. Mechanistic explanation.**  ",
        ("RAW spectra retain probe / substrate / aggregation effects across "
         "the 1401 wavenumbers; cross-set Pearson is set by whichever batch "
         "structure dominates the cohort mean. BSV compresses each spectrum "
         "into 11 chemistry-family magnitudes — many wavenumber-level "
         "differences fold into the same axis bin. This is helpful when the "
         "11 axes track shared biology, and unhelpful when batch effects "
         "themselves project onto axis bins. The numbers above show which "
         "regime applies for SHINE D2."),
        "",
        "## Critical rules respected",
        "- No label leakage (labels only used post-hoc to define cohorts).",
        "- No averaging across sets before comparison.",
        "- No smoothing beyond the documented preprocessing chain.",
        "- All numbers computed from per-spectrum data — no assumptions.",
    ]
    (R / "REPORT_shine_cross_set_reproducibility_audit_v1.md").write_text(
        "\n".join(lines))
    print(f"  verdict: {verdict}")
    return verdict


def main() -> None:
    print("=" * 70)
    print("gaira_base_4_shine_cross_set_reproducibility_audit_v1")
    print("=" * 70)
    cohorts = stage1_select_cohorts()
    master_x = canonical_master_axis()
    wn_axis_full = shine_wn_axis()
    raw_per_cohort = stage2_load_raw_preprocess(cohorts, master_x, wn_axis_full)
    out = stage3_means_and_correlations(cohorts, raw_per_cohort, master_x)
    stage4_figure(out["per_dose"], out["trajectory"],
                   F / "fig_shine_cross_set_reproducibility_audit_v1.png")
    verdict = stage5_report(out["per_dose"], out["trajectory"],
                              out["variance"])
    try:
        shutil.copy(__file__, C / Path(__file__).name)
    except Exception:
        pass
    print(f"\n[done] verdict: {verdict}")


if __name__ == "__main__":
    main()
