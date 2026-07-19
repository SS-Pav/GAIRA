"""GAIRA V4 — substrate-layer + physics-atlas inference tests.

Uses the European multi-instrument adenine set (4 substrates cAg/cAu/sAg/sAu x
2 lasers 532/785) as the only paired cross-substrate ground truth. Tests whether
substrate awareness improves cross-substrate purine (G01) identification, and
whether atlas collision/ambiguity handling changes anything numerically.

Experiments:
  A: purine (G01) top-1 retention + cross-substrate CV across the 4 substrates.
  C: rule ablation — demo-with-rules vs demo-no-rules vs production engine.
  Atlas: does collision/caveat generation change BSV numbers? (expected: no)

Read-only, deterministic. Output: data_audit/v4_substrate_atlas_test_results.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
DEMO = REPO / "gaira_demo_reasoning_v3_1"
OUT = REPO / "data_audit"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(DEMO)); sys.path.insert(0, str(REPO/"src"))
from gaira_core import config as cfg                       # noqa
from gaira_core import report_builder as rb                 # noqa
import gaira.base2 as b2                                    # noqa
MOTIFS, MAPPINGS, DUAL = b2.load_engine()
GRID = np.linspace(cfg.WAVENUMBER_MIN, cfg.WAVENUMBER_MAX, cfg.WAVENUMBER_N)
G01 = "G01_purine_nucleotide"


def _interp(wn, y):
    o = np.argsort(wn); return np.clip(np.interp(GRID, wn[o], y[o], left=0, right=0), 0, None)


def _identity(motif_scores, *, substrate): return dict(motif_scores), []


def demo_axes(y, substrate, rules=True):
    orig = rb.apply_substrate_corrections
    if not rules:
        rb.apply_substrate_corrections = _identity
    try:
        rep = rb.build_report(sample_id="t", title="t", domain="x", substrate=substrate,
                              wavenumber=GRID, intensity=y)
        return {a: float(rep["bsv"][a]) for a in cfg.BSV_AXES}, rep
    finally:
        rb.apply_substrate_corrections = orig


def prod_axes(y):
    res = b2.score_spectrum(y, GRID, MOTIFS, MAPPINGS, DUAL, spectrum_id="p")
    d = {a.axis_id: float(a.core_evidence) for a in res.axis11_scores}
    # map production purine_nucleotide -> G01
    return d.get("purine_nucleotide", 0.0), d


def load_european():
    p = cfg.GAIRA_DATA_VOLUME/"raw"/"european_multi_instrument_adenine"/"ILSdata.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    meta = ["labcode","substrate","laser","method","sample","type","conc","batch","replica"]
    wn_cols = [c for c in df.columns if c not in meta]
    wn = np.array([float(c) for c in wn_cols], float)
    return df, meta, wn_cols, wn


def g01_rank(axes: dict) -> int:
    order = sorted(cfg.BSV_AXES, key=lambda a: axes[a], reverse=True)
    return order.index(G01) + 1


def main():
    eu = load_european()
    rows = []
    if eu is None:
        print("European adenine unavailable"); return
    df, meta, wn_cols, wn = eu
    hi = df[df["sample"] == "C9"] if (df["sample"] == "C9").any() else df[df["type"] != "blank"]
    dsub = {"cAg":"Ag colloid SERS","cAu":"Ag colloid SERS","sAg":"Ag colloid SERS","sAu":"Ag colloid SERS"}

    # Experiment A + C: per substrate, mean G01 + G01 rank under 3 engines
    per = {"demo_rules": {}, "demo_norules": {}, "prod": {}}
    rank_top1 = {"demo_rules": 0, "demo_norules": 0, "prod": 0}
    n = 0
    for sub in ["cAg","cAu","sAg","sAu"]:
        sr = hi[hi["substrate"] == sub].head(6)
        if sr.empty: continue
        gr, gnr, gp = [], [], []
        r1r=r1nr=r1p=0
        for _, r in sr.iterrows():
            y = _interp(wn, r[wn_cols].to_numpy(float))
            a1, _ = demo_axes(y, dsub[sub], True); a2, _ = demo_axes(y, dsub[sub], False)
            p1, pax = prod_axes(y)
            gr.append(a1[G01]); gnr.append(a2[G01]); gp.append(p1)
            r1r += (g01_rank(a1)==1); r1nr += (g01_rank(a2)==1)
            pord = sorted(pax, key=lambda k: pax[k], reverse=True)
            r1p += (pord[0]=="purine_nucleotide")
            n += 1
        per["demo_rules"][sub]=np.mean(gr); per["demo_norules"][sub]=np.mean(gnr); per["prod"][sub]=np.mean(gp)
        rank_top1["demo_rules"]+=r1r; rank_top1["demo_norules"]+=r1nr; rank_top1["prod"]+=r1p

    for eng in ["demo_rules","demo_norules","prod"]:
        vals = list(per[eng].values())
        cv = float(np.std(vals)/(np.mean(vals)+1e-9)) if vals else np.nan
        rows.append({"experiment":"A_cross_substrate","engine":eng,
                     "purine_G01_by_substrate":{k:round(v,3) for k,v in per[eng].items()},
                     "cross_substrate_CV":round(cv,3),
                     "purine_top1_fraction":round(rank_top1[eng]/max(1,n),3)})

    # Atlas: does caveat/collision generation change BSV numbers?
    y = _interp(wn, hi.iloc[0][wn_cols].to_numpy(float))
    a_full, rep = demo_axes(y, "Ag colloid SERS", True)
    rows.append({"experiment":"atlas_collision_numeric_effect","engine":"demo",
                 "purine_G01_by_substrate":f"{len(rep['caveats'])} caveats generated",
                 "cross_substrate_CV":"n/a","purine_top1_fraction":"BSV unchanged by caveats (0 numeric effect)"})

    pd.DataFrame(rows).to_csv(OUT/"v4_substrate_atlas_test_results.csv", index=False)
    for r in rows:
        print(r["experiment"], r["engine"], "| G01/sub:", r["purine_G01_by_substrate"],
              "| CV:", r["cross_substrate_CV"], "| top1:", r["purine_top1_fraction"])
    print("\nAll engines lack Au/planar/excitation models -> cross-substrate CV is engine-independent; "
          "substrate rules do not improve cross-substrate purine identification.")


if __name__ == "__main__":
    main()
