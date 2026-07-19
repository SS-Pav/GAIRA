"""GAIRA V4 — demo <-> production reconciliation adapter.

Runs the SAME reference/biological spectra through BOTH:
  * demo heuristic engine   (gaira_demo_reasoning_v3_1/gaira_core, build_report)
  * production deterministic engine (src/gaira/base2, score_spectrum)
and emits a comparison packet: per-axis BSV, motif counts, agreement, provenance.

Both are DETERMINISTIC band/motif scorers (no learned encoder). Read-only.
Output: data_audit/v4_demo_production_comparison.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
DEMO = REPO / "gaira_demo_reasoning_v3_1"
OUT = REPO / "data_audit"; OUT.mkdir(exist_ok=True)

# demo engine
sys.path.insert(0, str(DEMO))
from gaira_core import config as cfg                      # noqa
from gaira_core.report_builder import build_report         # noqa
# production engine
sys.path.insert(0, str(REPO / "src"))
import gaira.base2 as b2                                    # noqa

GRID = np.linspace(cfg.WAVENUMBER_MIN, cfg.WAVENUMBER_MAX, cfg.WAVENUMBER_N)
MOTIFS, MAPPINGS, DUAL = b2.load_engine()

# demo G-axis <-> production axis11 name
AXMAP = {
 "G01_purine_nucleotide": "purine_nucleotide", "G02_purine_metabolite": "purine_metabolite",
 "G03_pyrimidine_nucleotide": "pyrimidine_nucleotide", "G04_nucleic_acid_phosphate": "phosphate_nucleic_adjacent",
 "G05_glycan_carbohydrate": "glycan_carbohydrate", "G06_protein_peptide_backbone": "protein_peptide_backbone",
 "G07_aromatic_residue": "aromatic_residue", "G08_lipid_acyl_membrane": "lipid_acyl_membrane",
 "G09_sterol_neutral_lipid": "sterol_neutral_lipid", "G10_sulfur_thiol_redox": "sulfur_thiol_redox",
 "G11_metabolic_small_molecule": "metabolic_small_molecule"}


def _interp(wn, y):
    o = np.argsort(wn)
    return np.clip(np.interp(GRID, np.asarray(wn)[o], np.asarray(y)[o], left=0, right=0), 0, None)


def demo_bsv(y, substrate):
    rep = build_report(sample_id="d", title="d", domain="x", substrate=substrate,
                       wavenumber=GRID, intensity=y)
    return {a: float(rep["bsv"][a]) for a in cfg.BSV_AXES}, rep


def prod_axes(y):
    res = b2.score_spectrum(y, GRID, MOTIFS, MAPPINGS, DUAL, spectrum_id="p")
    out = {a.axis_id: float(a.core_evidence) for a in res.axis11_scores}
    fd = b2.result_to_flat_dict(res)
    n_motif = sum(1 for k, v in fd.items() if k.startswith("motif_core.") and float(v) > 0.01)
    return out, n_motif


def load_cases():
    cases = []
    # adenine Ag-SERS (real)
    from gaira_core.data_loader import _ADENINE_FILES, _read_adenine_csv, _crop_and_interp
    for fname, label, _ in _ADENINE_FILES[:2] + _ADENINE_FILES[-1:]:
        p = cfg.ADENINE_RAW_DIR / fname
        pr = _read_adenine_csv(p) if p.exists() else None
        if pr:
            wn, y = _crop_and_interp(*pr)
            cases.append((f"adenine::{label}", y, "Ag colloid SERS", "adenine_agsers"))
    # serum liver mean spectrum (first HA)
    sp = cfg.LIVER_PATIENT_TABLES / "patient_level_mean_spectra.csv"
    if sp.exists():
        df = pd.read_csv(sp)
        wn_cols = [c for c in df.columns if c.startswith("wn_")]
        wn = np.array([int(c[3:]) for c in wn_cols], float)
        r = df.iloc[0]
        cases.append((f"serum_liver::{r['class_label_display']}", np.clip(r[wn_cols].to_numpy(float), 0, None),
                      "Ag colloid SERS", "serum_liver"))
    # EV diabetes (first)
    ev = cfg.EV_DIABETES_TABLES / "sample_query_spectra.csv"
    if ev.exists():
        e = pd.read_csv(ev).iloc[0]
        wn = np.asarray(json.loads(e["wavenumbers_json"]), float)
        y = np.asarray(json.loads(e["intensity_json"]), float)
        cases.append((f"ev::{e['class_label']}", _interp(wn, y), "Ag colloid SERS", "ev_diabetes"))
    # synthetic Raman reference (Phe + purine)
    y = np.exp(-((GRID-1003)**2)/50) + 0.4*np.exp(-((GRID-725)**2)/40) + 0.3*np.exp(-((GRID-1440)**2)/80)
    cases.append(("synthetic::phe+purine+lipid", y, "Raman", "synthetic_raman"))
    return cases


def main():
    rows = []
    for name, y, substrate, domain in load_cases():
        db, rep = demo_bsv(y, substrate)
        pa, pn = prod_axes(y)
        dvec = np.array([db[a] for a in cfg.BSV_AXES])
        pvec = np.array([pa.get(AXMAP[a], 0.0) for a in cfg.BSV_AXES])
        # normalize each to unit sum for shape comparison
        dn = dvec/ (dvec.sum()+1e-9); pn_ = pvec/(pvec.sum()+1e-9)
        corr = float(np.corrcoef(dvec, pvec)[0, 1]) if dvec.std()>1e-9 and pvec.std()>1e-9 else np.nan
        cos = float(np.dot(dn, pn_)/((np.linalg.norm(dn)*np.linalg.norm(pn_))+1e-9))
        row = {"case": name, "substrate": substrate, "domain": domain,
               "demo_top_axis": cfg.BSV_AXES[int(dvec.argmax())],
               "prod_top_axis": cfg.BSV_AXES[int(pvec.argmax())],
               "top_axis_agree": cfg.BSV_AXES[int(dvec.argmax())] == cfg.BSV_AXES[int(pvec.argmax())],
               "pearson_demo_vs_prod": round(corr, 3), "cosine_shape": round(cos, 3),
               "demo_n_motifs": sum(1 for v in rep["motif_scores_adjusted"].values() if v > 0.01),
               "prod_n_motifs": pn, "demo_n_caveats": len(rep["caveats"])}
        for a in cfg.BSV_AXES:
            row[f"demo_{a}"] = round(db[a], 4); row[f"prod_{AXMAP[a]}"] = round(pa.get(AXMAP[a], 0.0), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"v4_demo_production_comparison.csv", index=False)
    print(df[["case", "demo_top_axis", "prod_top_axis", "top_axis_agree",
              "pearson_demo_vs_prod", "cosine_shape", "demo_n_motifs", "prod_n_motifs"]].to_string(index=False))
    print("\nboth engines are deterministic motif scorers; axis constants curated in both "
          "(demo config.BSV_AXES; production base2.BIOLOGY_AXES_V11).")


if __name__ == "__main__":
    main()
