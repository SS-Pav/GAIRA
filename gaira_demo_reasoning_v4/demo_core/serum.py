"""Serum-spike stress-test analysis (Page 5).

All recoverability tiers derive from the committed Spike Validation output
`phase7_serum_vs_pure.csv` (53 analytes) using documented quantitative criteria —
examples are NOT hand-picked for attractiveness. Before/after states come from the
committed serum-baseline and spiked-serum frozen-atlas projections, driven live
through the V6 engine.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .engine_bridge import REPO, K

TAB = REPO / "results/v5_rebuild/spike_validation/tables"

# recoverability tiers from cos(spike direction, pure-SERS fingerprint of that analyte)
STRONG, PARTIAL = 0.35, 0.10


def load_recoverability():
    """The 53-analyte serum-vs-pure recoverability table + derived tier."""
    df = pd.read_csv(TAB / "phase7_serum_vs_pure.csv")
    df = df.sort_values("cos_spike_vs_pureSERS", ascending=False).reset_index(drop=True)
    df["tier"] = np.where(df.cos_spike_vs_pureSERS >= STRONG, "strong",
                          np.where(df.cos_spike_vs_pureSERS >= PARTIAL, "partial", "poor"))
    return df


def recoverability_summary():
    s = json.loads((TAB / "phase7_summary.json").read_text())
    df = load_recoverability()
    return {
        "n_analytes": int(s["n_analytes"]),
        "n_above_null_p05": int(s.get("n_analytes_cos_above_null_p05", 0)),
        "median_angle_deg": float(s["median_angle_vs_pureSERS_deg"]),
        "median_replicate_dir_cos": float(s["median_replicate_direction_cos"]),
        "n_strong": int((df.tier == "strong").sum()),
        "n_partial": int((df.tier == "partial").sum()),
        "n_poor": int((df.tier == "poor").sum()),
        "strong_analytes": list(df[df.tier == "strong"].analyte),
    }


def _coords(name):
    df = pd.read_csv(TAB / f"phase3_projection_{name}.csv")
    return df, df[[f"c{j}" for j in range(K)]].values.astype(float)


def baseline_coord():
    _, Z = _coords("serum_baseline")
    return Z.mean(0)


def analyte_after(analyte):
    df, Z = _coords("spiked_serum")
    m = (df["analyte"] == analyte).values
    if not m.any():
        return None
    return Z[m].mean(0)


def before_after(bridge, analyte):
    """Full before/after V6 inference for one spiked analyte."""
    before = baseline_coord()
    after = analyte_after(analyte)
    if after is None:
        return None
    ob, _ = bridge.bsv_and_mss(before, domain="serum")
    oa, acts_a = bridge.bsv_and_mss(after, domain="serum")
    _, acts_b = bridge.bsv_and_mss(before, domain="serum")
    return {
        "before_coord": before, "after_coord": after,
        "bsv_before": ob.bsv, "bsv_after": oa.bsv,
        "radar_before": ob.radar["axes"], "radar_after": oa.radar["axes"],
        "mss_before": {a.id: a.composition for a in acts_b},
        "mss_after": {a.id: a.composition for a in acts_a},
        "ood": oa.bsv.ood_score, "confidence": oa.bsv.overall_confidence,
        "background": oa.bsv.non_biochemical.get("background_matrix", 0.0),
    }


def theme_delta_matrix(bridge, analytes):
    """analytes × biochemical-theme signed ΔBSV (after − baseline)."""
    themes = bridge.bio_themes
    before = baseline_coord()
    b_bsv = bridge.infer(before, domain="serum").bsv
    rows = []
    for a in analytes:
        after = analyte_after(a)
        if after is None:
            rows.append([0.0] * len(themes)); continue
        a_bsv = bridge.infer(after, domain="serum").bsv
        rows.append([a_bsv.composition[t] - b_bsv.composition[t] for t in themes])
    return np.array(rows), themes


def confidence_recoverability(bridge, df):
    """Per-analyte (cos_spike_vs_pureSERS, engine confidence, OOD) — the limitation view."""
    out = []
    for _, r in df.iterrows():
        after = analyte_after(r.analyte)
        if after is None:
            continue
        bsv = bridge.infer(after, domain="serum").bsv
        out.append({"analyte": r.analyte, "cos": r.cos_spike_vs_pureSERS,
                    "tier": r.tier, "confidence": bsv.overall_confidence, "ood": bsv.ood_score})
    return pd.DataFrame(out)
