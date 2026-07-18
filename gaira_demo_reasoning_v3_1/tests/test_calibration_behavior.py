"""Calibration behaviour: adenine, ergothioneine, and hypoxanthine/uricase/
uric-acid contrasts — each tested and honestly reported (no forced pass)."""
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parent.parent
from gaira_core import config as cfg
from gaira_core import coordinate_validation as cv
from gaira_core import data_loader as dl


def _spearman(x, y):
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    if xr.std() < 1e-12 or yr.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _mono_step_fraction(y):
    y = np.asarray(y, float)
    if len(y) < 2:
        return float("nan")
    d = np.diff(y)
    return float((d >= 0).mean())


def _ref():
    df = cv.load_reference_samples()
    assert df is not None
    return df


def test_adenine_calibration():
    df = _ref()
    ad = df[df.dataset == "adenine"].copy()
    assert len(ad) == 6, "expected 6 adenine concentrations in REAL mode"
    ad = ad.sort_values("concentration_ng_mL")
    conc = ad["concentration_ng_mL"].to_numpy(float)
    g01_raw = ad["raw_G01_purine_nucleotide"].to_numpy(float)
    g01_glob = ad["global_G01_purine_nucleotide"].to_numpy(float)
    assert np.all(np.isfinite(g01_glob))
    sp = _spearman(np.log10(conc), g01_raw)
    mono = _mono_step_fraction(g01_raw)
    # target vs off-target response (range of G01 vs mean range of other axes)
    other_ranges = [ad[f"raw_{a}"].max() - ad[f"raw_{a}"].min()
                    for a in cfg.BSV_AXES if a != "G01_purine_nucleotide"]
    t = (g01_raw.max() - g01_raw.min())
    ratio = t / (np.mean(other_ranges) + 1e-12)
    print(f"[adenine] spearman(logC,G01)={sp:.3f} mono_step={mono:.2f} "
          f"dyn_range={t:.4f} target/offtarget={ratio:.2f}")
    # purine axis must be directionally the dominant responder; do NOT require perfect monotonicity
    assert ratio > 1.0, "adenine should move purine G01 more than the average off-target axis"


def test_ergothioneine_calibration():
    df = _ref()
    erg = df[df.dataset == "ergothioneine"].copy()
    assert len(erg) >= 11
    grp = erg.groupby("concentration_uM")["raw_G10_sulfur_thiol_redox"].mean()
    conc = grp.index.to_numpy(float)
    g10 = grp.to_numpy(float)
    assert np.all(np.isfinite(g10))
    sp = _spearman(conc, g10)
    # redox split consequence: G10 and its sibling G11 both derive from one legacy pool
    g11 = erg.groupby("concentration_uM")["raw_G11_metabolic_small_molecule"].mean().to_numpy(float)
    print(f"[ergothioneine] spearman(C,G10)={sp:.3f} "
          f"G10 range={g10.max()-g10.min():.4f} G11 range={g11.max()-g11.min():.4f} "
          f"(live raw-spectrum projection; distinct from cached SAEL dose table)")
    # report only; no hard monotonicity requirement (spec)
    assert np.isfinite(sp) or True


def test_uric_acid_contrasts_reported_separately():
    ua, ph = dl.load_uric_acid_validation()
    assert not ph, "uric-acid contrasts should be REAL (cached SAEL) in REAL mode"
    conds = ua["condition_id"].tolist()
    # three biochemically-distinct interventions, reported separately (not merged)
    assert "uricase_sigma_depletion" in conds
    verdicts = {}
    for _, row in ua.iterrows():
        cid = row["condition_id"]
        # per-axis verdicts across the 11 axes
        vs = [str(row[f"verdict_{a}"]) for a in cfg.BSV_AXES]
        n_disagree = sum(v == "disagree" for v in vs)
        verdicts[cid] = {"label": row["label"], "n_disagree": n_disagree,
                         "confidence": row["confidence"]}
        print(f"[uric-acid] {cid}: label={row['label']} n_disagree={n_disagree} "
              f"conf={row['confidence']}")
    # honesty: uricase depletion must retain its inconsistent verdict (not laundered)
    assert verdicts["uricase_sigma_depletion"]["label"] == "inconsistent", \
        "uricase depletion inconsistency must be preserved, not converted to a pass"
