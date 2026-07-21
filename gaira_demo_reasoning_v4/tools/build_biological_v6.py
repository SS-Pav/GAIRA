"""Generate GENUINE V6 biochemical-state artifacts for biological cohorts.

Reads raw biological spectra from the data volume, runs each through the FROZEN V6
engine (gaira.engine.GAIRAEngine -> component coordinates -> BSV -> MSS), and writes
SANITIZED, committed artifacts to ../biological_artifacts/<dataset>.json so the demo
runs on a fresh checkout without the private volume (DEGRADED mode).

Sanitisation is mandatory:
  - NO demographics (age, sex, race, bmi, hba1c, …) are read or stored.
  - Diabetes cohort labels come from the .mat FILE (Impact / Strong-D), never from
    the demographic patient_data.csv; sample identifiers are anonymised indices.

Every number is a real V6 engine output. Nothing is invented. If a dataset's raw
files are absent, it is skipped and marked UNAVAILABLE (no fabricated output).

    python tools/build_biological_v6.py            # build all available cohorts
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]      # gaira_demo_reasoning_v4/
REPO = HERE.parent                              # repo root
sys.path.insert(0, str(REPO / "src"))
from gaira.engine import GAIRAEngine                 # noqa: E402
from gaira.engine.mss import MSSLayer                # noqa: E402
from gaira.engine.versioning import VERSIONS         # noqa: E402

VOL = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw")
OUT = HERE / "biological_artifacts"
OUT.mkdir(exist_ok=True)


# ── raw loaders: each yields (label, wavenumber, intensity) units ──
def load_covid():
    base = VOL / "covid_serum_raman"
    wn = np.loadtxt(base / "wave_number.txt")
    files = {"COVID": "raw_COVID.txt", "Suspected": "raw_Suspected.txt", "Healthy": "raw_Helthy.txt"}
    for grp, fn in files.items():
        A = np.loadtxt(base / fn)                     # (900, n_spectra)
        for j in range(A.shape[1]):
            yield {"group": grp, "id": f"{grp[:3].upper()}-{j+1:03d}", "wn": wn, "y": A[:, j]}


def load_hcc():
    import pandas as pd
    df = pd.read_csv(VOL / "hcc_serum" / "data.csv")
    wn_cols = [c for c in df.columns[4:]]
    wn = np.array([float(c) for c in wn_cols])
    keep = wn >= 200                                  # drop negative-shift instrument padding
    for i, r in df.iterrows():
        grp = "control" if str(r["class"]).upper() == "CTR" else "HCC"
        yield {"group": grp, "id": f"HCC-{i+1:03d}", "wn": wn[keep],
               "y": r[wn_cols].values.astype(float)[keep]}


def load_diabetes():
    import scipy.io as sio
    base = VOL / "diabetes_plasma_ev_sers" / "extracted"
    # wavenumber reconstruction (documented in the dataset's Figure3.m; recon-verified)
    pix = [263, 367, 492, 512, 590, 782, 872, 887]
    cal = [620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3]
    x = np.polyval(np.polyfit(pix, cal, 3), np.arange(1, 1651))
    wn = x[161:898]                                   # MATLAB range 162:898 -> 737 pts
    idx = 1
    for fn, grp in [("RawDataImpact.mat", "Impact"), ("RawDataStrong.mat", "Strong-D")]:
        ss = sio.loadmat(base / fn)["smoothed_spectra"]
        for p in range(ss.shape[1]):
            patient = ss[0, p]                        # (737, n_scans)
            yield {"group": grp, "id": f"P{idx:03d}",  # anonymised; NO demographics
                   "wn": wn, "y": np.asarray(patient, float).mean(1)}   # patient-level mean
            idx += 1


def _subsample(items, per_group, key, seed=0):
    """Deterministic stratified subsample: <=per_group units per stratum."""
    rng = np.random.default_rng(seed)
    by = {}
    for it in items:
        by.setdefault(key(it), []).append(it)
    out = []
    for g, lst in sorted(by.items()):
        idx = np.arange(len(lst))
        if len(lst) > per_group:
            idx = np.sort(rng.choice(idx, per_group, replace=False))
        out.extend(lst[i] for i in idx)
    return out


def load_shine():
    """SHINE EV-SERS (APAP hepatotoxicity cell-culture EV). Consolidated, QC'd 737-ch
    matrices; cohort labels = day × dose. Wavenumber from the dataset's Fig4D cubic
    calibration (recon-verified). Cell-culture EV → no subjects; spectrum-level."""
    import scipy.io as sio
    base = VOL / "shine_ev_sers/SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/data"
    pix = np.array([263, 367, 492, 512, 590, 782, 872, 887.])
    cal = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3])
    wn = np.polyval(np.polyfit(pix, cal, 3), np.arange(1, 1651))[161:898]   # 737 ch
    order = [f"D{d}_C{c}" for d in (0, 1, 2) for c in (0, 10, 20, 40)]      # cell order
    cells = sio.loadmat(base / "RawDataSet91.mat")["clustered"]
    units = []
    for ci, lab in enumerate(order):
        day, dose = lab.split("_")
        if day == "D1":                                # D1 has no sample structure — skip
            continue
        M = cells[0, ci]
        if M is None or np.size(M) == 0:
            continue
        M = np.asarray(M)                              # (737, n_spectra)
        for j in range(M.shape[1]):
            units.append({"group": day, "strata": dose, "id": f"{lab}-{j+1:04d}",
                          "wn": wn, "y": M[:, j].astype(float)})
    return _subsample(units, per_group=90, key=lambda u: (u["group"], u["strata"]))


def load_small2023():
    """small2023 EV (Parlatan et al.) — Probe-1 exosome Raman, 1131 ch. Axis from the
    SI Fig_S7 spreadsheet (recon-verified). A probe-CONCENTRATION titration of cell-line
    exosomes: no subject/disease labels → EV state-space characterization only."""
    import scipy.io as sio, zipfile, re
    from xml.etree import ElementTree as ET
    base = VOL / "small2023_ev"
    z = zipfile.ZipFile(base / "Fig_S7 (1).xlsx")
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    axis = []
    for r in root.findall(".//a:sheetData/a:row", ns):
        for c in r.findall("a:c", ns):
            if re.match(r"[A-Z]+", c.get("r")).group() == "A" and c.get("t") != "s":
                v = c.find("a:v", ns)
                if v is not None:
                    try:
                        axis.append(float(v.text))
                    except ValueError:
                        pass
    wn = np.array(axis)                                # 1131, 670–1800 cm⁻¹
    s = sio.loadmat(base / "NormedProbe1.mat", struct_as_record=False, squeeze_me=True)["normed1"]
    units = []
    for lvl in ["c00", "c01", "c10", "c25", "c50", "c100"]:
        M = np.atleast_2d(getattr(s, lvl))             # (n_spectra, 1131)
        for j in range(M.shape[0]):
            if M[j].shape[0] != wn.shape[0]:
                continue
            units.append({"group": lvl, "strata": None, "id": f"{lvl}-{j+1:04d}",
                          "wn": wn, "y": M[j].astype(float)})
    return _subsample(units, per_group=100, key=lambda u: u["group"])


DATASETS = {
    "covid_serum_raman": dict(
        loader=load_covid, display="COVID serum Raman", domain="serum", modality="Raman",
        substrate="none (Raman)", excitation="785", aggregation="spectrum",
        source="raw/covid_serum_raman (Raman serum; 3 clinical groups)",
        caveats=["Subject-to-spectrum mapping is not documented in the source; analyses are "
                 "spectrum-level and exploratory (not patient-level inference).",
                 "Serum Raman is nearer the atlas domain than SERS, but still a complex matrix."]),
    "hcc_serum": dict(
        loader=load_hcc, display="HCC serum SERS", domain="serum", modality="SERS",
        substrate="Ag", excitation="785", aggregation="spectrum",
        source="raw/hcc_serum/data.csv (serum SERS; CTR vs H0T)",
        caveats=["'H0T' is the source's cancer label (control = CTR); exact clinical definition "
                 "should be confirmed from the dataset's R script.",
                 "Spectrum-level; subject mapping not documented — exploratory."]),
    "diabetes_plasma_ev_sers": dict(
        loader=load_diabetes, display="Diabetes plasma-EV SERS", domain="ev", modality="SERS",
        substrate="Ag", excitation="785", aggregation="patient",
        source="raw/diabetes_plasma_ev_sers (plasma EV SERS; Impact vs Strong-D)",
        caveats=["PATIENT-LEVEL: one mean spectrum per patient (39 Impact + 24 Strong-D) — no "
                 "per-scan pseudoreplication.",
                 "Cohort labels come from the .mat file (Impact / Strong-D), NOT the demographic "
                 "CSV; all demographics are excluded and identifiers are anonymised.",
                 "Impact vs Strong-D are study-specific cohort labels, not a clean disease/control "
                 "contrast."]),
    "shine_ev_sers": dict(
        loader=load_shine, display="SHINE EV-SERS (hepatotoxicity)", domain="ev", modality="SERS",
        substrate="Ag", excitation="785", aggregation="spectrum",
        source="raw/shine_ev_sers (APAP-dosed cell-culture EV; Day 0 vs Day 2 x dose)",
        caveats=["Cell-culture secreted EV (APAP 0/10/20/40 mM) — NO human subjects; spectra are "
                 "cohort (day x dose) grouped, aggregated at cohort level.",
                 "Consolidated QC'd 737-ch matrices (RawDataSet91); Day 1 omitted (no sample "
                 "structure). Pairing is day-vs-day WITHIN a dose cohort, not per-well.",
                 "Group = timepoint (D0/D2); strata = APAP dose."]),
    "small2023_ev": dict(
        loader=load_small2023, display="EV single-vesicle Raman (small2023)", domain="ev",
        modality="Raman", substrate="none (Raman)", excitation="785", aggregation="spectrum",
        source="raw/small2023_ev (Parlatan et al.; probe-concentration titration of cell-line EV)",
        caveats=["A probe-CONCENTRATION titration (c00..c100) of cell-line exosomes — NOT a disease "
                 "or subject cohort. Used for EV biochemical-state-space characterization only.",
                 "Weak labels: the only grouping is probe concentration; no biological contrast is "
                 "claimed.",
                 "Wavenumber axis recovered from the SI (Fig_S7); Probe-1 (1131 ch, 670-1800)."]),
}


def build_one(key, spec, eng, mss):
    if not VOL.exists():
        return None
    try:
        units = list(spec["loader"]())
    except FileNotFoundError:
        return None
    if not units:
        return None
    themes = eng.builder.onto.theme_ids
    motif_ids = [m.id for m in mss.motifs]
    records = []
    for u in units:
        out = eng.infer(wavenumber=u["wn"], intensity=u["y"], domain=spec["domain"])
        acts = {a.id: a.composition for a in mss.activate(out.bsv)}
        records.append({
            "id": u["id"], "group": u["group"], "strata": u.get("strata"),
            "coord": [round(float(x), 5) for x in out.bsv.component_coord],
            "themes": [round(float(out.bsv.composition[t]), 5) for t in themes],
            "motifs": [round(float(acts[m]), 5) for m in motif_ids],
            "ood": round(float(out.bsv.ood_score), 4),
            "confidence": round(float(out.bsv.overall_confidence), 4),
            "background": round(float(out.bsv.non_biochemical.get("background_matrix", 0.0)), 4),
        })
    groups = sorted({r["group"] for r in records})
    art = {
        "dataset_id": key, "display_name": spec["display"], "domain": spec["domain"],
        "modality": spec["modality"], "substrate": spec["substrate"],
        "excitation_nm": spec["excitation"], "aggregation": spec["aggregation"],
        "source": spec["source"], "caveats": spec["caveats"],
        "atlas_fingerprint": VERSIONS.atlas_fingerprint, "engine_versions": VERSIONS.as_dict(),
        "theme_ids": themes, "motif_ids": motif_ids,
        "groups": groups, "n_units": len(records),
        "n_by_group": {g: sum(r["group"] == g for r in records) for g in groups},
        "records": records,
    }
    (OUT / f"{key}.json").write_text(json.dumps(art))
    return art


def main():
    eng = GAIRAEngine()
    mss = MSSLayer.from_engine(eng)
    manifest = {"atlas_fingerprint": VERSIONS.atlas_fingerprint, "datasets": {}}
    for key, spec in DATASETS.items():
        art = build_one(key, spec, eng, mss)
        status = "REAL" if art else "UNAVAILABLE"
        manifest["datasets"][key] = {
            "display_name": spec["display"], "status": status,
            "aggregation": spec["aggregation"], "domain": spec["domain"],
            "n_units": art["n_units"] if art else 0,
            "n_by_group": art["n_by_group"] if art else {},
        }
        print(f"  {key:<28} {status:<12} "
              + (f"{art['n_units']} units {art['n_by_group']}" if art else "(raw data absent)"))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT.relative_to(REPO)}/  (atlas {VERSIONS.atlas_fingerprint[:12]}…)")


if __name__ == "__main__":
    main()
