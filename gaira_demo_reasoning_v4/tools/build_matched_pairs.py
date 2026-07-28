"""Build the matched pure-Raman <-> pure-Ag-SERS reference artifact (Gobbato 2025).

The Gobbato archive holds, for 51 analytes, BOTH a pure 785 nm Raman powder spectrum
(already in the frozen atlas as source `gobbato_raman_metabolites`) AND a pure 785 nm
Ag-SERS spectrum (the `SERS metabolites/` twin). This is the matched set that lets us
measure — per analyte — how far the Ag-SERS representation moves from the Raman
representation in the SAME frozen component space.

We do NOT refit anything. Both sides are projected through the FROZEN V6 engine
(domain="buffer", the closest-to-reference regime) into the 24-dim component
coordinate, averaged per analyte, and compared by cosine. The result is the empirical
"observation-model" gap the DART / Au-SERS page describes as future work — quantified
here from data already on disk.

Output: ../reference_artifacts/matched_raman_sers_pairs.json  (committed, sanitized).

    python tools/build_matched_pairs.py
"""
from __future__ import annotations
import sys
import json
import zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]      # gaira_demo_reasoning_v4/
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
from gaira.engine import GAIRAEngine                       # noqa: E402
from gaira.engine.versioning import VERSIONS               # noqa: E402
from gaira.data.synonyms import GOBBATO_ABBREV, canonical  # noqa: E402

VOL = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw")
ZIP = VOL / "serum_ag_colloids" / "dataset_spectral_data.zip"
OUT = HERE / "reference_artifacts"
OUT.mkdir(exist_ok=True)


def _bwtek(text):
    lines = text.splitlines()
    hdr = next((i for i, l in enumerate(lines)
                if "Raman Shift" in l and "Dark Subtracted" in l), None)
    if hdr is None:
        return None
    cols = [c.strip() for c in lines[hdr].split(";")]
    try:
        iw = cols.index("Raman Shift")
        iy = next(i for i, c in enumerate(cols) if c.startswith("Dark Subtracted"))
    except (ValueError, StopIteration):
        return None
    wn, y = [], []
    for l in lines[hdr + 1:]:
        p = l.split(";")
        if len(p) <= max(iw, iy):
            continue
        try:
            wn.append(float(p[iw].replace(",", ".")))
            y.append(float(p[iy].replace(",", ".")))
        except ValueError:
            continue
    if len(wn) < 100:
        return None
    o = np.argsort(wn)
    return np.array(wn)[o], np.array(y)[o]


def _project_folder(eng, prefix, abbr_re):
    """{canonical_analyte: [component_coord vectors]} for every spectrum in a folder."""
    import re
    out = {}
    with zipfile.ZipFile(ZIP) as z:
        for n in z.namelist():
            if not (n.startswith(prefix) and n.endswith(".txt")):
                continue
            m = re.match(abbr_re, n.split("/")[-1])
            if not m:
                continue
            analyte = canonical(GOBBATO_ABBREV.get(m.group(1), m.group(1).lower()))
            parsed = _bwtek(z.read(n).decode("cp1252", errors="replace"))
            if parsed is None:
                continue
            wn, y = parsed
            bsv = eng.infer(wavenumber=wn, intensity=y, domain="buffer").bsv
            out.setdefault(analyte, []).append(np.asarray(bsv.component_coord, float))
    return out


def _dominant_theme(eng, coord):
    """BSV dominant biochemical theme name for a mean component coordinate."""
    bsv = eng.infer(coordinates=coord, domain="buffer").bsv
    themes = {t: bsv.composition[t] for t in eng.builder.onto.theme_ids
              if t not in ("background_matrix", "unknown_mixed")}
    return max(themes, key=themes.get), float(max(themes.values()))


def main():
    if not ZIP.exists():
        print(f"UNAVAILABLE — Gobbato archive not mounted at {ZIP}")
        return
    eng = GAIRAEngine()
    raman = _project_folder(eng, "Raman metabolites/", r"Raman_pwd_(.+?)_")
    sers = _project_folder(eng, "SERS metabolites/", r"SERS_met_(.+?)_")
    both = sorted(set(raman) & set(sers))
    pairs = []
    for a in both:
        r = np.mean(raman[a], 0)
        s = np.mean(sers[a], 0)
        cos = float(np.dot(r, s) / (np.linalg.norm(r) * np.linalg.norm(s) + 1e-12))
        rt, rw = _dominant_theme(eng, r)
        stt, sw = _dominant_theme(eng, s)
        pairs.append({
            "analyte": a, "n_raman": len(raman[a]), "n_sers": len(sers[a]),
            "coord_cosine": round(cos, 4),
            "raman_dominant_theme": rt, "sers_dominant_theme": stt,
            "theme_preserved": rt == stt,
            "raman_coord": [round(float(x), 5) for x in r],
            "sers_coord": [round(float(x), 5) for x in s],
        })
    pairs.sort(key=lambda p: -p["coord_cosine"])
    cosines = [p["coord_cosine"] for p in pairs]
    art = {
        "artifact": "matched_raman_sers_pairs",
        "description": "Per-analyte pure-Raman vs pure-Ag-SERS in the frozen V6 component "
                       "space (Gobbato 2025, 785 nm B&WTek, same instrument/group).",
        "source": "raw/serum_ag_colloids/dataset_spectral_data.zip -> Raman metabolites/ + "
                  "SERS metabolites/ (Gobbato 2025, DOI 10.1007/s00216-025-06192-5)",
        "atlas_fingerprint": VERSIONS.atlas_fingerprint, "engine_versions": VERSIONS.as_dict(),
        "domain": "buffer", "n_pairs": len(pairs),
        "median_coord_cosine": round(float(np.median(cosines)), 4),
        "n_theme_preserved": int(sum(p["theme_preserved"] for p in pairs)),
        "pairs": pairs,
    }
    (OUT / "matched_raman_sers_pairs.json").write_text(json.dumps(art))
    print(f"wrote {(OUT / 'matched_raman_sers_pairs.json').relative_to(REPO)}")
    print(f"  {len(pairs)} matched analytes | median coord-cosine "
          f"{art['median_coord_cosine']:.3f} | theme preserved "
          f"{art['n_theme_preserved']}/{len(pairs)}")
    print("  MOST preserved:", ", ".join(f"{p['analyte']}({p['coord_cosine']:.2f})" for p in pairs[:5]))
    print("  LEAST preserved:", ", ".join(f"{p['analyte']}({p['coord_cosine']:.2f})" for p in pairs[-5:]))


if __name__ == "__main__":
    main()
