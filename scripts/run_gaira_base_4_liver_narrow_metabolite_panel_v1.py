"""gaira_base_4 liver narrow-metabolite panel v1.

Explicit paper-band scoring (Gurian/Bonifacio 2020) for UA / HX / ERG / GSH
+ lactate as missing-anchor placeholder.

NO engine / MSS / motif / taxonomy / weight changes. NO label-driven tuning.
Analysis/calibration layer only.
"""
from __future__ import annotations

import shutil
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis
from run_gaira_base_4_hybrid_bsv_build_v1 import _band_max
from run_gaira_base_4_hybrid_bsv_controlled_calibration_v2 import (
    load_erg_calibration, load_uricase, load_sers_fitting, load_isotopic,
)


ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_liver_narrow_metabolite_panel_v1"
)
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
REGISTRY = ROOT / "registry"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"

P1_CSV = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/hcc_serum/data.csv")
P2_ZIP = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/cca_hcc_lm_serum_sers/"
    "Combination of label-free SERS-based nanosensor an.zip"
)
P3_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/covid_serum_raman")


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — narrow MSS template panel registry (metadata)
# ─────────────────────────────────────────────────────────────────────

NARROW_TEMPLATES = [
    {
        "molecule": "uric_acid",
        "global_mss_present": True, "global_mss_name": "uric acid",
        "raman_anchor_cm1": "638;888;1132", "sers_anchor_cm1": "594;638;812;888;1132",
        "support_cm1": "1497", "anti_evidence_cm1": "725 (overlap with HX)",
        "regime_support": "Raman+SERS",
        "calibration_data_present": "uricase depletion + isotopic UA + UAfree/UAbound",
        "reliability_tier": "STRONG (SERS) / MODERATE (Raman)",
        "notes": "Classic SERS purine fingerprint; ring stretching 638 + ring breathing 1132",
    },
    {
        "molecule": "hypoxanthine",
        "global_mss_present": False, "global_mss_name": "(MISSING in MSS v4.3)",
        "raman_anchor_cm1": "725;640", "sers_anchor_cm1": "725;1330",
        "support_cm1": "888", "anti_evidence_cm1": "1132 (UA-distinctive)",
        "regime_support": "Raman+SERS",
        "calibration_data_present": "serum_ag_colloids/SERS_fitting Hypox × 10 + uricase Serumspiked-with-HX",
        "reliability_tier": "MODERATE (calibration data exists; MSS template missing)",
        "notes": "Critical gap in MSS v4.3 — paper-band scoring used as workaround; 725 cm⁻¹ is dominant purine ring breathing",
    },
    {
        "molecule": "xanthine",
        "global_mss_present": True, "global_mss_name": "xanthine",
        "raman_anchor_cm1": "640;750", "sers_anchor_cm1": "750;1280",
        "support_cm1": "950", "anti_evidence_cm1": "725 (HX)",
        "regime_support": "Raman+SERS",
        "calibration_data_present": "Gobbato pure xanthine (Raman + SERS via metabolite_63)",
        "reliability_tier": "MODERATE",
        "notes": "Sister-purine to UA + HX; band pattern shifts with C8-OH",
    },
    {
        "molecule": "ergothioneine",
        "global_mss_present": True, "global_mss_name": "ergothioneine",
        "raman_anchor_cm1": "1442;1582", "sers_anchor_cm1": "480;1220;1442;1582",
        "support_cm1": "1302", "anti_evidence_cm1": "(none)",
        "regime_support": "Raman+SERS",
        "calibration_data_present": "ERG_calibration 11 conc × 5 rep on cAg + CSPP fig7 ERG spike",
        "reliability_tier": "STRONG (SERS calibration confirms 480 Ag-S anchor)",
        "notes": "Data-driven SERS template from ERG_calibration (v3 fix phase): 480 Ag-S bond stretch is primary, imidazole ring 1220/1582 secondary",
    },
    {
        "molecule": "glutathione",
        "global_mss_present": True, "global_mss_name": "glutathione",
        "raman_anchor_cm1": "664;912", "sers_anchor_cm1": "664;1295;1418",
        "support_cm1": "1645", "anti_evidence_cm1": "(none)",
        "regime_support": "Raman+SERS",
        "calibration_data_present": "Gobbato + RamanBioLib pure GSH",
        "reliability_tier": "MODERATE",
        "notes": "664 cm⁻¹ C-S stretch is GSH-specific thiolate marker on Ag",
    },
    {
        "molecule": "lactate",
        "global_mss_present": False, "global_mss_name": "(MISSING in MSS v4.3)",
        "raman_anchor_cm1": "830;1043;1453", "sers_anchor_cm1": "830;1043",
        "support_cm1": "(none)", "anti_evidence_cm1": "(none)",
        "regime_support": "Raman+SERS",
        "calibration_data_present": "Gobbato pure lactate (Raman); no dedicated SERS calibration",
        "reliability_tier": "WEAK_TEMPLATE_MISSING (paper-band scoring used as workaround)",
        "notes": "Common metabolite anchor; not in MSS v4.3 registry; paper-band scoring possible from 830 + 1043 reference bands",
    },
]


def stage1_panel_registry():
    print("\n[STAGE 1] Build narrow MSS template panel registry")
    pd.DataFrame(NARROW_TEMPLATES).to_csv(
        REGISTRY / "narrow_metabolite_mss_panel_v1.csv", index=False,
    )
    print(f"  emitted {len(NARROW_TEMPLATES)} template entries")


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — Paper-band feature scoring
# ─────────────────────────────────────────────────────────────────────

# Paper bands (Gurian / Bonifacio 2020)
PAPER_BANDS = {
    "uric_acid":     [594.0, 638.0, 812.0, 888.0, 1132.0],
    "hypoxanthine":  [724.0],
    "xanthine":      [640.0, 950.0, 1280.0],
    "ergothioneine": [480.0, 1220.0, 1442.0, 1582.0],
    "glutathione":   [664.0, 912.0],
    "lactate":       [830.0, 1043.0, 1453.0],
}


def _local_prominence(spectrum, master_x, cm, half=8.0, baseline_window=40.0):
    """band intensity − local baseline (median over wider window)."""
    if not np.any(np.isfinite(spectrum)): return 0.0
    idx = int(np.argmin(np.abs(master_x - cm)))
    bh = int(half / max(np.diff(master_x).mean(), 1e-3))
    bw = int(baseline_window / max(np.diff(master_x).mean(), 1e-3))
    win = spectrum[max(0, idx - bh): idx + bh + 1]
    base_lo = spectrum[max(0, idx - bw): max(0, idx - bh)]
    base_hi = spectrum[idx + bh + 1: min(len(spectrum), idx + bw + 1)]
    base = np.concatenate([base_lo, base_hi]) if (len(base_lo) + len(base_hi)) > 0 else win
    base = base[np.isfinite(base)]
    win = win[np.isfinite(win)]
    if len(win) == 0: return 0.0
    peak = float(np.max(win))
    bl   = float(np.median(base)) if len(base) > 0 else 0.0
    return peak - bl


def paper_score_spectrum(spectrum, master_x, sp_max=None):
    """Per-spectrum paper-band feature scores per molecule (height + prominence)."""
    fin = np.isfinite(spectrum)
    if sp_max is None:
        sp_max = float(np.max(spectrum[fin])) if fin.any() else 1.0
    scores = {}
    for mol, bands in PAPER_BANDS.items():
        heights = []
        prominences = []
        contrasts = []
        for cm in bands:
            h = float(_band_max(spectrum, master_x, cm, half=8.0)) / max(sp_max, 1e-9)
            p = _local_prominence(spectrum, master_x, cm) / max(sp_max, 1e-9)
            # Contrast: peak / (local min in window)
            idx = int(np.argmin(np.abs(master_x - cm)))
            win = spectrum[max(0, idx-8):idx+9]
            win = win[np.isfinite(win)]
            if len(win) > 1 and np.min(win) != 0:
                c = float(np.max(win)) / max(abs(float(np.min(win))) + 1e-9, 1e-9)
            else: c = 0.0
            heights.append(h); prominences.append(p); contrasts.append(c)
        # Co-band completeness: fraction of bands with prominence > 0.02
        completeness = float(np.mean([p > 0.02 for p in prominences]))
        # Molecule panel score: mean height + mean prominence + 0.2 × completeness
        panel_score = (float(np.mean(heights)) + float(np.mean(prominences)) +
                        0.2 * completeness)
        scores[f"paper_{mol}_height_mean"]      = round(float(np.mean(heights)), 4)
        scores[f"paper_{mol}_prom_mean"]        = round(float(np.mean(prominences)), 4)
        scores[f"paper_{mol}_contrast_mean"]    = round(float(np.mean(contrasts)), 4)
        scores[f"paper_{mol}_completeness"]     = round(completeness, 3)
        scores[f"paper_{mol}_panel_score"]      = round(panel_score, 4)
    # Combined signed paper score: UA − HX − ERG − GSH
    scores["paper_signed_score"] = round(
        scores["paper_uric_acid_panel_score"]
        - scores["paper_hypoxanthine_panel_score"]
        - scores["paper_ergothioneine_panel_score"]
        - scores["paper_glutathione_panel_score"], 4)
    return scores


# ─────────────────────────────────────────────────────────────────────
# Loaders for all relevant datasets
# ─────────────────────────────────────────────────────────────────────

def load_p1_raw(master_x):
    df = pd.read_csv(P1_CSV, low_memory=False)
    meta = ["acquisition_date", "substrate_batch", "class", "sample_code"]
    wn_cols = [c for c in df.columns if c not in meta]
    wn = np.array([float(c) for c in wn_cols]); order = np.argsort(wn)
    refs = []
    for i, row in df.iterrows():
        y = row[wn_cols].values.astype(float)
        y_rs = np.interp(master_x, wn[order], y[order], left=np.nan, right=np.nan)
        refs.append({
            "spectrum_id": f"p1::{row['sample_code']}",
            "sample_id": row["sample_code"],
            "class_label": row["class"],
            "dataset": "P1",
            "spectrum": y_rs,
        })
    return refs


def load_p2_raw(master_x):
    refs = []
    with zipfile.ZipFile(P2_ZIP) as z:
        for info in z.infolist():
            if not info.filename.endswith(".txt"): continue
            parts = info.filename.split("/")
            if len(parts) < 4: continue
            patient_folder = parts[2]
            if not patient_folder.startswith("SER-"): continue
            toks = patient_folder.split("-")
            if len(toks) < 3: continue
            cls = toks[1]
            data = z.read(info).decode("utf-8", errors="ignore").splitlines()
            if len(data) < 2: continue
            try: wn = np.array([float(x) for x in data[0].split("\t") if x.strip()])
            except: continue
            arrs = []
            for line in data[1:]:
                vals = line.split("\t")
                try: f = [float(v) for v in vals if v.strip()]
                except ValueError: continue
                if len(f) >= len(wn) + 2: arrs.append(np.asarray(f[2:2+len(wn)]))
            if not arrs: continue
            mean_y = np.mean(arrs, 0); order = np.argsort(wn)
            y_rs = np.interp(master_x, wn[order], mean_y[order], left=np.nan, right=np.nan)
            refs.append({
                "spectrum_id": f"p2::{patient_folder}",
                "sample_id": patient_folder,
                "class_label": cls,
                "dataset": "P2",
                "spectrum": y_rs,
            })
    return refs


def load_p3_raw(master_x):
    """COVID Raman per-spectrum, dropping boundary zeros."""
    refs = []
    wn_full = np.loadtxt(P3_DIR / "wave_number.txt")
    files = {"Healthy": "raw_Helthy.txt", "COVID": "raw_COVID.txt",
             "Suspected": "raw_Suspected.txt"}
    for cls, fname in files.items():
        arr = np.loadtxt(P3_DIR / fname)  # 900 × N
        keep = np.arange(1, arr.shape[0] - 1)
        arr = arr[keep]; wn = wn_full[keep]; order = np.argsort(wn)
        for i in range(arr.shape[1]):
            y_rs = np.interp(master_x, wn[order], arr[:, i][order], left=np.nan, right=np.nan)
            refs.append({
                "spectrum_id": f"p3::{cls}_{i+1:03d}",
                "sample_id": f"{cls}_{(i // 3) + 1}",
                "class_label": cls, "dataset": "P3",
                "spectrum": y_rs,
            })
    return refs


# ─────────────────────────────────────────────────────────────────────
# Score datasets + helpers
# ─────────────────────────────────────────────────────────────────────

def score_refs(refs, master_x):
    rows = []
    for r in refs:
        sc = paper_score_spectrum(r["spectrum"], master_x)
        row = {k: r[k] for k in ("spectrum_id", "sample_id", "class_label", "dataset")
                 if k in r}
        # Calibration loaders use 'cohort' instead of 'class_label'; mirror it
        if "class_label" not in row and "cohort" in r:
            row["class_label"] = r["cohort"]
        row.update(sc)
        rows.append(row)
    return pd.DataFrame(rows)


def _cohens_d(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2: return 0.0
    pooled = np.sqrt(((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1))
                       / max(len(x)+len(y)-2, 1))
    return (np.mean(x) - np.mean(y)) / (pooled if pooled > 0 else 1.0)


def _bootstrap_ci(x, y, n=500, seed=42):
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        ds.append(_cohens_d(rng.choice(x, len(x), replace=True),
                              rng.choice(y, len(y), replace=True)))
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def _spearman(x, y):
    rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _auc(v, y):
    order = np.argsort(v); rank = np.empty_like(order); rank[order] = np.arange(len(v))
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    return float((rank[y == 1].sum() - n_pos*(n_pos-1)/2) / (n_pos * n_neg))


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Calibration validation
# ─────────────────────────────────────────────────────────────────────

def stage3_calibration(master_x):
    print("\n[STAGE 3] Calibration validation")
    rows = []

    # ERG concentration series
    print("  ERG calibration (cAg, 11 conc × 5 rep)")
    erg = load_erg_calibration(master_x)
    erg_df = score_refs(erg, master_x)
    erg_df["conc_M"] = [r["conc_M"] for r in erg]
    log_c = np.log10(erg_df["conc_M"].replace(0, 1e-13))
    rho_erg = _spearman(log_c, erg_df["paper_ergothioneine_panel_score"])
    rho_erg_h = _spearman(log_c, erg_df["paper_ergothioneine_height_mean"])
    rho_erg_p = _spearman(log_c, erg_df["paper_ergothioneine_prom_mean"])
    # Per-conc replicate CV at 2.0 µM
    sub_top = erg_df[erg_df["conc_M"] >= 1e-6]["paper_ergothioneine_panel_score"]
    cv_erg = float(sub_top.std(ddof=1) / max(sub_top.mean(), 1e-9)) if len(sub_top) > 1 else None
    rows.append({
        "molecule": "ergothioneine",
        "calibration_dataset": "ERG_calibration (cAg, 11 conc × 5 rep)",
        "test_metric": "Spearman ρ(log[ERGO], ERG paper score)",
        "value": round(rho_erg, 3) if rho_erg == rho_erg else None,
        "secondary_metrics": f"ρ_height={rho_erg_h:+.2f} ρ_prom={rho_erg_p:+.2f}; CV at ≥1µM={cv_erg:.3f}" if cv_erg is not None else None,
        "expected_direction": "+",
        "passes": (rho_erg is not None and rho_erg >= 0.4),
        "reliability_tier": "STRONG" if (rho_erg or 0) >= 0.5 else ("MODERATE" if (rho_erg or 0) >= 0.3 else "WEAK"),
    })
    print(f"    ERG paper score ρ(log-conc) = {rho_erg:+.3f} (height ρ={rho_erg_h:+.2f}, prom ρ={rho_erg_p:+.2f})")

    # Hypoxanthine SERS fitting
    print("  Hypoxanthine via SERS metabolites for fitting")
    fit = load_sers_fitting(master_x)
    fit_df = score_refs(fit, master_x)
    hyp = fit_df[fit_df["class_label"] == "Hypoxanthine"]["paper_hypoxanthine_panel_score"]
    other = fit_df[fit_df["class_label"] != "Hypoxanthine"]["paper_hypoxanthine_panel_score"]
    if len(hyp) >= 2 and len(other) >= 2:
        d_hyp = _cohens_d(hyp.values, other.values)
        rows.append({
            "molecule": "hypoxanthine",
            "calibration_dataset": "SERS_fitting (Hypox × 10 vs UA_free/UA_bound × 20)",
            "test_metric": "Cohen's d (Hypox vs UA cohorts) on HX paper score",
            "value": round(float(d_hyp), 3),
            "secondary_metrics": f"Hypox mean={hyp.mean():.3f}, UA cohorts mean={other.mean():.3f}",
            "expected_direction": "Hypox >> UA cohorts on HX 724 cm⁻¹",
            "passes": d_hyp >= 0.5,
            "reliability_tier": "STRONG" if d_hyp >= 1.0 else ("MODERATE" if d_hyp >= 0.5 else "WEAK"),
        })
        print(f"    HX score Hypox vs UA cohorts: d={d_hyp:+.3f}")

    # Uricase enzymatic depletion
    print("  Uricase enzymatic depletion")
    uri = load_uricase(master_x)
    uri_df = score_refs(uri, master_x)
    sigma_no = uri_df[uri_df["class_label"] == "SerumSigma"]["paper_uric_acid_panel_score"]
    sigma_e  = uri_df[uri_df["class_label"] == "SerumSigma+Enzyme"]["paper_uric_acid_panel_score"]
    if len(sigma_no) >= 2 and len(sigma_e) >= 2:
        d_uri = _cohens_d(sigma_e.values, sigma_no.values)
        # Paper expects depletion → negative
        rows.append({
            "molecule": "uric_acid",
            "calibration_dataset": "Uricase depletion (SerumSigma vs SerumSigma+Enzyme, 5×5)",
            "test_metric": "Cohen's d (+Enzyme vs −Enzyme) on UA paper score",
            "value": round(float(d_uri), 3),
            "secondary_metrics": f"−Enzyme mean={sigma_no.mean():.3f}, +Enzyme mean={sigma_e.mean():.3f}",
            "expected_direction": "− (UA depletion → score drops with enzyme)",
            "passes": d_uri <= -0.3,
            "reliability_tier": "STRONG" if d_uri <= -0.5 else ("MODERATE" if d_uri <= -0.2 else "WEAK"),
        })
        print(f"    UA score +Enzyme vs −Enzyme: d={d_uri:+.3f}")

    # Isotopic UA
    print("  Isotopic UA (UA vs UAiso)")
    iso = load_isotopic(master_x)
    iso_df = score_refs(iso, master_x)
    ua_only = iso_df[iso_df["class_label"] == "UA"]["paper_uric_acid_panel_score"]
    uaiso   = iso_df[iso_df["class_label"] == "UAiso"]["paper_uric_acid_panel_score"]
    if len(ua_only) >= 2 and len(uaiso) >= 2:
        d_iso = _cohens_d(uaiso.values, ua_only.values)
        rows.append({
            "molecule": "uric_acid",
            "calibration_dataset": "Isotopic UA (UA vs UAiso)",
            "test_metric": "Cohen's d (UAiso vs UA) on UA paper score",
            "value": round(float(d_iso), 3),
            "secondary_metrics": f"UA mean={ua_only.mean():.3f}, UAiso mean={uaiso.mean():.3f}",
            "expected_direction": "small (isotope shift → some band-position drift but score should be similar)",
            "passes": abs(d_iso) <= 0.5,
            "reliability_tier": "BAND_SENSITIVITY_CHECK",
        })
        print(f"    UA score UAiso vs UA: d={d_iso:+.3f}")

    # Lactate — no calibration
    rows.append({
        "molecule": "lactate",
        "calibration_dataset": "(none in current corpus)",
        "test_metric": "n/a",
        "value": None, "secondary_metrics": None,
        "expected_direction": "n/a",
        "passes": False,
        "reliability_tier": "UNVALIDATED",
    })

    # Glutathione — no dedicated calibration
    rows.append({
        "molecule": "glutathione",
        "calibration_dataset": "(no dedicated SERS calibration; pure GSH in Gobbato/RamanBioLib only)",
        "test_metric": "n/a",
        "value": None, "secondary_metrics": None,
        "expected_direction": "n/a",
        "passes": False,
        "reliability_tier": "UNVALIDATED_FOR_PAPER_SCORE",
    })

    pd.DataFrame(rows).to_csv(
        TABLES / "narrow_metabolite_calibration_validation_v1.csv", index=False,
    )
    return rows, erg_df


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — Pilot 1 paper replication
# ─────────────────────────────────────────────────────────────────────

PAPER_CLAIMS = {
    "uric_acid":     ("HCC > CTR", "+"),
    "hypoxanthine":  ("CTR > HCC", "-"),
    "ergothioneine": ("CTR > HCC", "-"),
    "glutathione":   ("CTR > HCC", "-"),
}


def stage4_p1_replication(p1_df):
    print("\n[STAGE 4] Pilot 1 paper-claim replication using paper-band scores")
    rows = []
    p1_hcc = p1_df[p1_df["class_label"] == "H0T"]
    p1_ctr = p1_df[p1_df["class_label"] == "CTR"]
    for mol, (claim, sign) in PAPER_CLAIMS.items():
        col = f"paper_{mol}_panel_score"
        x = p1_hcc[col].values; y = p1_ctr[col].values
        d = _cohens_d(x, y)
        ci_lo, ci_hi = _bootstrap_ci(x, y)
        # AUC: positive class = HCC if paper expects HCC>; otherwise CTR
        if sign == "+":
            yb = (p1_df["class_label"].values == "H0T").astype(int)
            v = p1_df[col].values
        else:
            yb = (p1_df["class_label"].values == "CTR").astype(int)
            v = p1_df[col].values
        auc = _auc(v, yb)
        obs_sign = "+" if d > 0 else ("-" if d < 0 else "0")
        agrees = (obs_sign == sign and abs(d) >= 0.15)
        rows.append({
            "metabolite": mol, "paper_claim": claim, "expected_sign": sign,
            "p1_HCC_mean": round(float(np.mean(x)), 4),
            "p1_CTR_mean": round(float(np.mean(y)), 4),
            "cohens_d_HCC_vs_CTR": round(float(d), 3),
            "ci95_low": round(ci_lo, 3), "ci95_high": round(ci_hi, 3),
            "ci_excludes_zero": (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0),
            "auc_paper_direction": round(float(auc), 3),
            "observed_sign": obs_sign,
            "agrees_with_paper": agrees,
        })
        print(f"  {mol:14s} d={d:+.3f}  CI=[{ci_lo:+.2f},{ci_hi:+.2f}]  AUC={auc:.2f}  "
              f"paper={sign}  observed={obs_sign}  agrees={agrees}")

    # Combined signed score
    x = p1_hcc["paper_signed_score"].values; y = p1_ctr["paper_signed_score"].values
    d = _cohens_d(x, y); ci_lo, ci_hi = _bootstrap_ci(x, y)
    yb = (p1_df["class_label"].values == "H0T").astype(int)
    auc = _auc(p1_df["paper_signed_score"].values, yb)
    rows.append({
        "metabolite": "COMBINED_PAPER_SIGNED",
        "paper_claim": "HCC > CTR (UA − HX − ERG − GSH)",
        "expected_sign": "+",
        "p1_HCC_mean": round(float(np.mean(x)), 4),
        "p1_CTR_mean": round(float(np.mean(y)), 4),
        "cohens_d_HCC_vs_CTR": round(float(d), 3),
        "ci95_low": round(ci_lo, 3), "ci95_high": round(ci_hi, 3),
        "ci_excludes_zero": (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0),
        "auc_paper_direction": round(float(auc), 3),
        "observed_sign": "+" if d > 0 else "-",
        "agrees_with_paper": (d > 0 and abs(d) >= 0.15),
    })
    print(f"  COMBINED       d={d:+.3f}  CI=[{ci_lo:+.2f},{ci_hi:+.2f}]  AUC={auc:.2f}")
    pd.DataFrame(rows).to_csv(
        TABLES / "pilot1_paper_feature_replication_v2.csv", index=False,
    )
    return rows


# ─────────────────────────────────────────────────────────────────────
# Stage 5 — Cross-pilot transfer
# ─────────────────────────────────────────────────────────────────────

def stage5_cross_pilot(p1_df, p2_df, p3_df):
    print("\n[STAGE 5] Cross-pilot transfer")
    cross_rows = []
    # P1 HCC vs CTR is already in stage 4 — repeat for canonical comparison labeling
    cohorts_per_dataset = {
        "P1": [("H0T", "CTR", "P1_HCC_vs_CTR")],
        "P2": [("HCC", "NC", "P2_HCC_vs_NC"),
                ("CCA", "NC", "P2_CCA_vs_NC"),
                ("LM",  "NC", "P2_LM_vs_NC")],
        "P3": [("COVID", "Healthy", "P3_COVID_vs_Healthy"),
                ("Suspected", "Healthy", "P3_Suspected_vs_Healthy")],
    }
    metab_cols = [f"paper_{m}_panel_score" for m in PAPER_BANDS] + ["paper_signed_score"]
    pilot_dfs = {"P1": p1_df, "P2": p2_df, "P3": p3_df}
    for pilot, comps in cohorts_per_dataset.items():
        df = pilot_dfs[pilot]
        for a, b, label in comps:
            x_df = df[df["class_label"] == a]; y_df = df[df["class_label"] == b]
            if len(x_df) < 2 or len(y_df) < 2: continue
            for col in metab_cols:
                d = _cohens_d(x_df[col].values, y_df[col].values)
                ci_lo, ci_hi = _bootstrap_ci(x_df[col].values, y_df[col].values)
                cross_rows.append({
                    "pilot": pilot, "comparison": label,
                    "feature": col,
                    "n_target": len(x_df), "n_control": len(y_df),
                    "cohens_d": round(float(d), 3),
                    "ci95_low": round(ci_lo, 3), "ci95_high": round(ci_hi, 3),
                    "ci_excludes_zero": (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0),
                    "direction": "+" if d > 0 else ("-" if d < 0 else "0"),
                })
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(TABLES / "narrow_metabolite_cross_pilot_v1.csv", index=False)

    # Diagnostic classification per metabolite
    diag_rows = []
    for col in metab_cols:
        sub = cross_df[cross_df.feature == col]
        def _g(comp):
            row = sub[sub.comparison == comp]
            return float(row["cohens_d"].iloc[0]) if len(row) else None
        d_p1 = _g("P1_HCC_vs_CTR"); d_p2hcc = _g("P2_HCC_vs_NC")
        d_p2cca = _g("P2_CCA_vs_NC"); d_p2lm = _g("P2_LM_vs_NC")
        d_p3cov = _g("P3_COVID_vs_Healthy")
        # Classify
        if d_p1 is None or d_p2hcc is None: cat = "MISSING_DATA"
        elif (abs(d_p1) >= 0.20 and abs(d_p2hcc) >= 0.20
                and np.sign(d_p1) == np.sign(d_p2hcc)):
            cat = "TRANSFERS"
        elif (abs(d_p1) >= 0.30 and abs(d_p2hcc) < 0.20):
            cat = "SUBSTRATE_LOCKED"
        elif (d_p2cca is not None and d_p2lm is not None
                and abs(d_p2cca) >= 0.50 and abs(d_p2lm) >= 0.50
                and np.sign(d_p2cca) == np.sign(d_p2lm) and abs(d_p2hcc) < 0.30):
            cat = "ADVANCED_CANCER_ONLY"
        elif d_p3cov is not None and abs(d_p3cov) >= 0.30 and abs(d_p1 or 0) >= 0.20:
            cat = "SYSTEMIC_ILLNESS"
        elif abs(d_p1 or 0) < 0.15 and abs(d_p2hcc or 0) < 0.15:
            cat = "INDETERMINATE"
        else: cat = "INDETERMINATE"
        diag_rows.append({
            "feature": col,
            "P1_HCC_vs_CTR_d": d_p1, "P2_HCC_vs_NC_d": d_p2hcc,
            "P2_CCA_vs_NC_d": d_p2cca, "P2_LM_vs_NC_d": d_p2lm,
            "P3_COVID_vs_Healthy_d": d_p3cov,
            "category": cat,
        })
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(TABLES / "narrow_metabolite_transfer_diagnostic_v2.csv", index=False)
    print("Per-feature classification:")
    for _, r in diag_df.iterrows():
        d_p1 = f"{r['P1_HCC_vs_CTR_d']:+.2f}" if r['P1_HCC_vs_CTR_d'] is not None else "—"
        d_p2 = f"{r['P2_HCC_vs_NC_d']:+.2f}" if r['P2_HCC_vs_NC_d'] is not None else "—"
        d_p3 = f"{r['P3_COVID_vs_Healthy_d']:+.2f}" if r['P3_COVID_vs_Healthy_d'] is not None else "—"
        print(f"  {r['feature']:32s} P1={d_p1:>6s} P2HCC={d_p2:>6s} P3COV={d_p3:>6s}  → {r['category']}")
    return cross_df, diag_df


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_4_liver_narrow_metabolite_panel_v1")
    print("=" * 78)
    for d in (TABLES, FIGS, REPORTS, REGISTRY, AUDIT, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    master_x = canonical_master_axis()
    stage1_panel_registry()

    # ── Stage 2: score all spectra in P1 + P2 + P3 + calibration ──
    print("\n[STAGE 2] Score Pilot 1 + Pilot 2 + Pilot 3 spectra")
    p1_refs = load_p1_raw(master_x)
    p2_refs = load_p2_raw(master_x)
    p3_refs = load_p3_raw(master_x)
    p1_df = score_refs(p1_refs, master_x); p1_df["pilot"] = "P1"
    p2_df = score_refs(p2_refs, master_x); p2_df["pilot"] = "P2"
    p3_df = score_refs(p3_refs, master_x); p3_df["pilot"] = "P3"
    pd.concat([p1_df, p2_df, p3_df], ignore_index=True).to_csv(
        TABLES / "paper_feature_scores_per_spectrum_v1.csv", index=False,
    )
    print(f"  P1: {len(p1_df)}, P2: {len(p2_df)}, P3: {len(p3_df)}")

    # ── Stage 3-5 ──
    cal_rows, erg_df = stage3_calibration(master_x)
    p1_rep = stage4_p1_replication(p1_df)
    cross_df, diag_df = stage5_cross_pilot(p1_df, p2_df, p3_df)

    # ── Figures ──
    print("\n[figures]")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Paper-feature band-score heatmap (raw band heights for top metabolites)
        fig, ax = plt.subplots(figsize=(11, 5))
        cohorts = [("P1 CTR", p1_df[p1_df.class_label == "CTR"]),
                    ("P1 HCC", p1_df[p1_df.class_label == "H0T"]),
                    ("P2 NC",  p2_df[p2_df.class_label == "NC"]),
                    ("P2 HCC", p2_df[p2_df.class_label == "HCC"]),
                    ("P2 CCA+LM", p2_df[p2_df.class_label.isin(["CCA","LM"])]),
                    ("P3 Healthy", p3_df[p3_df.class_label == "Healthy"]),
                    ("P3 COVID", p3_df[p3_df.class_label == "COVID"])]
        feats = [f"paper_{m}_panel_score" for m in PAPER_BANDS]
        mat = np.array([[float(sub[f].mean()) for sub, _ in [(s[1], s[0])]
                            for f in feats][0:len(feats)] for s in cohorts])
        # Fix matrix correctly
        mat = np.zeros((len(cohorts), len(feats)))
        for i, (label, sub) in enumerate(cohorts):
            for j, f in enumerate(feats):
                mat[i, j] = float(sub[f].mean()) if f in sub.columns and len(sub) else 0.0
        vmax = float(np.abs(mat).max()) or 0.1
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(cohorts))); ax.set_yticklabels([c[0] for c in cohorts])
        ax.set_xticks(range(len(feats))); ax.set_xticklabels([f.replace("paper_","").replace("_panel_score","") for f in feats], rotation=20)
        ax.set_title("Paper-band panel scores per cohort (cohort means)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(mat[i,j]) > vmax*0.5 else "black")
        fig.colorbar(im, ax=ax, label="mean panel score")
        fig.tight_layout()
        fig.savefig(FIGS / "fig1_paper_feature_band_score_heatmap.png", dpi=150)
        plt.close(fig)

        # 2. Pilot 1 UA/HX/ERG/GSH bar
        fig, ax = plt.subplots(figsize=(9, 4.5))
        labels = ["uric_acid", "hypoxanthine", "ergothioneine", "glutathione",
                    "COMBINED_paper_signed"]
        ds = []; sems_lo = []; sems_hi = []; agrees = []
        for r in p1_rep:
            ds.append(r["cohens_d_HCC_vs_CTR"])
            sems_lo.append(r["ci95_low"]); sems_hi.append(r["ci95_high"])
            agrees.append(r["agrees_with_paper"])
        colors = ["#2ca02c" if a else "#d62728" for a in agrees]
        x = np.arange(len(labels))
        ax.bar(x, ds, color=colors)
        for i, d in enumerate(ds):
            ax.plot([i, i], [sems_lo[i], sems_hi[i]], color="black", lw=1.2)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Cohen's d HCC vs CTR (P1)")
        ax.set_title("Pilot 1 paper-feature replication (green=agrees, red=disagrees)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig2_pilot1_paper_score_bar.png", dpi=150)
        plt.close(fig)

        # 3. Calibration response: ERG dose-response
        fig, ax = plt.subplots(figsize=(8, 5))
        log_c = np.log10(erg_df["conc_M"].replace(0, 1e-13))
        ax.scatter(log_c, erg_df["paper_ergothioneine_panel_score"], s=25, alpha=0.7)
        rho_erg = _spearman(log_c, erg_df["paper_ergothioneine_panel_score"])
        ax.set_xlabel("log10([ERGO], M)"); ax.set_ylabel("ERG paper panel score")
        ax.set_title(f"ERG calibration (cAg) — paper panel score vs concentration "
                       f"(ρ={rho_erg:+.2f})")
        fig.tight_layout()
        fig.savefig(FIGS / "fig3_calibration_erg_dose_response.png", dpi=150)
        plt.close(fig)

        # 4. Cross-pilot transfer heatmap
        feats2 = [f"paper_{m}_panel_score" for m in PAPER_BANDS] + ["paper_signed_score"]
        comps2 = ["P1_HCC_vs_CTR", "P2_HCC_vs_NC", "P2_CCA_vs_NC",
                   "P2_LM_vs_NC", "P3_COVID_vs_Healthy"]
        mat2 = np.zeros((len(feats2), len(comps2)))
        ci2 = np.zeros((len(feats2), len(comps2)), dtype=bool)
        for i, f in enumerate(feats2):
            for j, c in enumerate(comps2):
                row = cross_df[(cross_df.feature == f) & (cross_df.comparison == c)]
                if len(row):
                    mat2[i, j] = float(row["cohens_d"].iloc[0])
                    ci2[i, j] = bool(row["ci_excludes_zero"].iloc[0])
        fig, ax = plt.subplots(figsize=(9, 5))
        vmax = float(np.abs(mat2).max()) or 0.5
        im = ax.imshow(mat2, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(feats2))); ax.set_yticklabels([f.replace("paper_","").replace("_panel_score","") for f in feats2])
        ax.set_xticks(range(len(comps2))); ax.set_xticklabels(comps2, rotation=20, ha="right")
        ax.set_title("Cross-pilot transfer: paper panel score Cohen's d (* = CI ✓)")
        for i in range(mat2.shape[0]):
            for j in range(mat2.shape[1]):
                v = mat2[i,j]; star = "*" if ci2[i,j] else ""
                ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=8,
                         color="white" if abs(v) > vmax*0.5 else "black")
        fig.colorbar(im, ax=ax, label="d")
        fig.tight_layout()
        fig.savefig(FIGS / "fig4_cross_pilot_transfer_heatmap.png", dpi=150)
        plt.close(fig)

        # 5. Narrow vs broad schematic
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.axis("off")
        ax.text(0.5, 0.92, "GAIRA narrow paper-band panel vs broad BSV", fontsize=14, fontweight="bold", ha="center")
        layers = [
            (0.05, 0.55, "BROAD BSV (sumnorm/CLR)\n• 11 chemistry families\n• G09 Sterol-lipid ↓ replicates 5/5\n• Cross-pilot stable", "#cfe2ff"),
            (0.40, 0.55, "NARROW MSS (template-based)\n• 0/3 paper claims reproduce\n• MSS missing for HX, lactate\n• No compositional norm at MSS layer", "#ffe8cc"),
            (0.72, 0.55, "NARROW PAPER-BAND PANEL (this phase)\n• Direct band intensity + prominence\n• HX scoreable without MSS template\n• Paper-claim direct test", "#cce6cc"),
        ]
        for x, y, text, color in layers:
            ax.text(x, y, text, fontsize=9, ha="left", va="center",
                      bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="black"))
        ax.text(0.5, 0.18,
                  "Three complementary layers — each tests a different scientific claim.\n"
                  "Broad BSV = portable biochemistry. Narrow MSS = molecule-level decision.\n"
                  "Narrow paper-band = literature-claim verification.",
                  fontsize=10, ha="center", style="italic")
        fig.tight_layout()
        fig.savefig(FIGS / "fig5_narrow_vs_broad_schematic.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  figure issue: {e}")

    # ── Final decision ──
    n_agree = sum(1 for r in p1_rep if r["agrees_with_paper"])
    n_total_testable = sum(1 for r in p1_rep)  # paper signed is always testable
    n_transfers = int((diag_df.category == "TRANSFERS").sum())
    n_sub_locked = int((diag_df.category == "SUBSTRATE_LOCKED").sum())
    n_advanced_only = int((diag_df.category == "ADVANCED_CANCER_ONLY").sum())
    cal_passes = sum(1 for r in cal_rows if r.get("passes"))
    cal_total = sum(1 for r in cal_rows if r.get("test_metric") not in (None, "n/a"))

    if n_agree >= 3 and n_transfers >= 1 and cal_passes >= 2:
        decision = "PAPER_FEATURES_REPLICATE_AND_TRANSFER"
    elif n_agree >= 2 and n_transfers == 0:
        decision = "PAPER_FEATURES_REPLICATE_BUT_SUBSTRATE_LOCKED"
    elif n_agree <= 1:
        if cal_passes < 2:
            decision = "INSUFFICIENT_CALIBRATION_SUPPORT"
        else:
            decision = "PAPER_FEATURES_DO_NOT_REPLICATE"
    else:
        decision = "PAPER_FEATURES_REPLICATE_BUT_SUBSTRATE_LOCKED"

    # ── Reports ──
    lines = [
        "# Narrow metabolite paper-feature panel — final report",
        "",
        f"## Decision: **{decision}**",
        "",
        "## Calibration validation",
        "",
        "| molecule | dataset | metric | value | passes | tier |",
        "|---|---|---|---:|---|---|",
    ]
    for r in cal_rows:
        v = r.get("value"); v_s = f"{v:+.3f}" if v is not None else "—"
        lines.append(f"| {r['molecule']} | {r['calibration_dataset']} | "
                     f"{r['test_metric']} | {v_s} | "
                     f"{'✓' if r.get('passes') else '✗'} | {r['reliability_tier']} |")
    lines += [
        "",
        f"## Pilot 1 paper-claim replication ({n_agree}/{n_total_testable} agree)",
        "",
        "| metabolite | paper expects | observed d | CI excl 0 | AUC | observed sign | agrees? |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for r in p1_rep:
        ci_str = "✓" if r["ci_excludes_zero"] else "✗"
        lines.append(f"| {r['metabolite']} | {r['expected_sign']} | "
                     f"{r['cohens_d_HCC_vs_CTR']:+.3f} | {ci_str} | "
                     f"{r['auc_paper_direction']:.2f} | {r['observed_sign']} | "
                     f"{'YES' if r['agrees_with_paper'] else 'no'} |")
    lines += [
        "",
        "## Cross-pilot transfer classification",
        "",
        "| feature | P1 HCC | P2 HCC | P2 CCA | P2 LM | P3 COVID | category |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in diag_df.iterrows():
        def _f(v): return f"{v:+.2f}" if v is not None and v == v else "—"
        lines.append(f"| {r['feature']} | {_f(r['P1_HCC_vs_CTR_d'])} | "
                     f"{_f(r['P2_HCC_vs_NC_d'])} | {_f(r['P2_CCA_vs_NC_d'])} | "
                     f"{_f(r['P2_LM_vs_NC_d'])} | {_f(r['P3_COVID_vs_Healthy_d'])} | "
                     f"**{r['category']}** |")
    lines += [
        "",
        "## Required answers",
        "",
        "### 1. Do explicit paper bands reproduce Pilot 1 better than MSS?",
        f"- MSS subaxis (prior phase): 0/3 claims reproduced",
        f"- Paper-feature panel (this phase): **{n_agree}/{n_total_testable}** claims reproduced",
        "",
        "### 2. Are paper bands calibration-supported?",
        f"- {cal_passes}/{cal_total} calibration tests pass at the meaningful threshold",
        "",
        "### 3. Do they transfer to Pilot 2?",
        f"- TRANSFERS: {n_transfers}",
        f"- SUBSTRATE_LOCKED: {n_sub_locked}",
        f"- ADVANCED_CANCER_ONLY: {n_advanced_only}",
        "",
        "### 4. Which signals are true chemistry but substrate-specific?",
    ]
    for _, r in diag_df[diag_df.category == "SUBSTRATE_LOCKED"].iterrows():
        lines.append(f"- {r['feature']}: P1 d={r['P1_HCC_vs_CTR_d']:+.2f}, P2 d={r['P2_HCC_vs_NC_d']:+.2f}")
    lines += [
        "",
        "### 5. Which are candidates for portable biology?",
    ]
    for _, r in diag_df[diag_df.category == "TRANSFERS"].iterrows():
        lines.append(f"- {r['feature']}: P1 d={r['P1_HCC_vs_CTR_d']:+.2f}, P2 d={r['P2_HCC_vs_NC_d']:+.2f}")
    if not (diag_df.category == "TRANSFERS").any():
        lines.append("- (none meet TRANSFERS criterion)")
    lines += [
        "",
        "### 6. What should be in GAIRA demo?",
        "",
        "- **Broad BSV layer** with cross-pilot G09 ↓ replication (5/5 cohorts)",
        "- **Narrow paper-band panel** as literature-claim verification layer (this phase)",
        f"- **Calibration evidence** that supports {cal_passes} of {cal_total} narrow scores",
        "- **Honest substrate caveats** for narrow signals that fail cross-pilot transfer",
    ]
    (REPORTS / "REPORT_narrow_metabolite_panel_v1.md").write_text("\n".join(lines))

    # Audit log
    lines = [
        "# gaira_base_4 liver narrow-metabolite panel v1 — Audit Log",
        "",
        "## Layers",
        "1. Narrow MSS template panel (registry of 6 metabolites)",
        "2. Paper-band feature scoring (height + prominence + completeness)",
        "3. Calibration validation (ERG / HX / Uricase / Isotopic)",
        "4. Pilot 1 paper-claim replication via paper-band scores",
        "5. Cross-pilot transfer (P1 + P2 + P3 COVID)",
        "",
        "## Datasets used",
        "- P1 Gurian HCC SERS (144 spectra)",
        "- P2 label-free SERS nanosensor (195 patient-mean spectra)",
        "- P3 COVID Raman serum (465 clinical + 12 tube)",
        "- ERG_calibration (55 spectra), Uricase (20), SERS-fitting (30), Isotopic (73)",
        "",
        "## Results",
        f"- Pilot 1 paper-claim replication: {n_agree}/{n_total_testable} agree",
        f"- Calibration: {cal_passes}/{cal_total} tests pass",
        f"- Cross-pilot: TRANSFERS={n_transfers}, SUBSTRATE_LOCKED={n_sub_locked}, "
        f"ADVANCED_CANCER_ONLY={n_advanced_only}",
        "",
        f"## Final decision: **{decision}**",
        "",
        "## Invariants",
        "- engine v4.5 / taxonomy / motif / MSS v4.3 / substrate physics v1.2: unchanged",
        "- narrow MSS panel registry is calibration-only (not merged into global MSS)",
        "- paper-band scoring is analysis layer (no engine modification)",
        "- no classifier training, no threshold tuning, no label-driven feature select",
        "- no DART-Met",
    ]
    (AUDIT / "gaira_base_4_liver_narrow_metabolite_panel_v1_audit_log.md"
     ).write_text("\n".join(lines))

    p = Path(__file__)
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)

    print(f"\n[final decision] {decision}")
    print(f"  paper-claim agree: {n_agree}/{n_total_testable}")
    print(f"  calibration passes: {cal_passes}/{cal_total}")
    print(f"  cross-pilot: TRANSFERS={n_transfers}, LOCKED={n_sub_locked}, ADV_ONLY={n_advanced_only}")


if __name__ == "__main__":
    main()
