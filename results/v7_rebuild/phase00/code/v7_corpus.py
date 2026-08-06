"""GAIRA V7 Phase 00 — the frozen Raman grounding corpus, loaded reproducibly.

Reproduces the V5 reference corpus exactly (375 spectra / 167 analytes / 676 bins) but
without any hard-coded lab path: the raw root resolves through GAIRA_DATA_ROOT. The
canonical preprocessing chain is UNCHANGED — it calls the same `gaira.preprocessing`
primitives and the same loaders the frozen atlas was built from.

Two modes:
  FULL      raw root available → corpus loaded from raw, card recomputed, NMF
            reproduction possible.
  DEGRADED  raw root unavailable → per-analyte structure recovered from committed
            artefacts (sv_reps.npz, V5 tables). Recorded in every manifest.

Nothing here writes to assets/ or results/v5_rebuild/ or results/v6_rebuild/.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import v7_paths as P

# The Ag-SERS / Au-SERS / DART exclusion list, carried verbatim from the V5 corpus card.
EXCLUDED_DOMAINS = [
    "Ag-SERS", "Au-SERS", "DART", "serum Ag-colloid",
    "metabolite-63 (633 nm Ag-SERS)", "adenine Ag-SERS series",
    "european_multi_instrument_adenine (cAg/sAg/cAu substrates)",
]

# The frozen V5 corpus card values every load is checked against.
EXPECTED = {
    "n_spectra": 375,
    "n_analytes": 167,
    "n_bins": 676,
    "sources": {"RamanBioLib": 202, "gobbato_raman_metabolites": 153,
                "amino_acid_raman_grounding": 20},
    "excitations": {"785.0": 234, "1064.0": 55, "532.0": 50, "488.0": 29,
                    "514.5": 3, "632.8": 1, "457.9": 1, "850.0": 1, "633.0": 1},
    "n_analytes_multi_excitation": 41,
    "replicate_groups_v5": 272,
    "analytes_with_replicates": 87,
}


@dataclass
class Corpus:
    X: np.ndarray            # (n_spectra, 676) preprocessed on the canonical grid
    grid: np.ndarray
    meta: pd.DataFrame       # spectrum_id, analyte, source, excitation_nm, replicate
    mode: str                # "full" | "degraded"
    notes: list = field(default_factory=list)

    @property
    def analytes(self) -> list[str]:
        return sorted(self.meta.analyte.unique().tolist())


def nfkc(s: str) -> str:
    """Unicode NFKC + whitespace + case normalisation.

    NFKC is what collapses the U+FB02 'ﬂ' ligature in `riboﬂavin` onto `riboflavin`;
    without it those are two distinct analytes and a direct CV-leakage path (risk R-09).
    """
    s = unicodedata.normalize("NFKC", str(s))
    s = " ".join(s.split())
    return s.strip().lower()


def load_full(root: Path) -> Corpus:
    """Load the corpus from raw, using the same loaders and preprocessing as V5."""
    P.add_src_to_path()
    from gaira.data import gobbato, loader                     # noqa: E402
    from gaira.data.synonyms import canonical                  # noqa: E402
    from gaira.foundation.dataset import AA_NAME_FIX           # noqa: E402
    from gaira.preprocessing import pipeline as pp             # noqa: E402

    grid = pp.common_grid(*P.WINDOW_CM, P.GRID_STEP_CM)

    def prep(wn, y):
        return pp.preprocess(np.asarray(wn, float), np.asarray(y, float),
                             P.PREPROC, grid, P.WINDOW_CM)

    rows, recs = [], []
    for s in loader.load_ramanbiolib():
        v = prep(s.wavenumber, s.intensity)
        if not np.isfinite(v).any():
            continue
        rows.append(v)
        recs.append({"spectrum_id": s.record.spectrum_id,
                     "analyte": canonical(s.record.canonical_analyte_name),
                     "source": "RamanBioLib",
                     "excitation_nm": float(s.record.excitation_nm),
                     "replicate": str(s.record.replicate)})
    for s in gobbato.load_gobbato_785():
        if s.record.modality.value != "raman":                 # Ag-SERS EXCLUDED
            continue
        v = prep(s.wavenumber, s.intensity)
        if not np.isfinite(v).any():
            continue
        rows.append(v)
        recs.append({"spectrum_id": s.record.spectrum_id,
                     "analyte": canonical(s.record.canonical_analyte_name),
                     "source": "gobbato_raman_metabolites",
                     "excitation_nm": 785.0,
                     "replicate": str(s.record.replicate)})

    aa = root / "amino_acid_raman_grounding/aa.xlsx"
    if aa.exists():
        df = pd.read_excel(aa)
        wn = df.iloc[:, 0].values.astype(float)
        for col in df.columns[1:]:
            name = canonical(AA_NAME_FIX.get(str(col).strip().lower(), str(col).strip().lower()))
            y = df[col].values.astype(float)
            ok = np.isfinite(y)
            if ok.sum() < 100:
                continue
            v = prep(wn[ok], y[ok])
            if not np.isfinite(v).any():
                continue
            rows.append(v)
            recs.append({"spectrum_id": f"amino_acid_raman::{name}", "analyte": name,
                         "source": "amino_acid_raman_grounding", "excitation_nm": 785.0,
                         "replicate": "1"})

    X = np.vstack(rows)
    meta = pd.DataFrame(recs)
    meta = _disambiguate_ids(meta)
    return Corpus(X=X, grid=np.asarray(grid, float), meta=meta, mode="full")


def _disambiguate_ids(meta: pd.DataFrame) -> pd.DataFrame:
    """Make spectrum_id unique.

    The amino-acid grounding sheet holds two columns (`l-glu`, `glutamic acid`) that both
    canonicalise to `glutamate`, so the V5 id scheme emits `amino_acid_raman::glutamate`
    twice. Two distinct measured spectra sharing one id would silently collapse in any
    id-keyed join — including the quality table and the weight normalisation.
    """
    m = meta.copy()
    dup = m.spectrum_id.duplicated(keep=False)
    if dup.any():
        for sid, g in m[dup].groupby("spectrum_id"):
            for k, idx in enumerate(g.index, start=1):
                m.loc[idx, "spectrum_id"] = f"{sid}#{k}"
                m.loc[idx, "replicate"] = str(k)
    return m


def load_degraded() -> Corpus:
    """Per-analyte structure from committed artefacts. No raw volume required."""
    z = np.load(P.SV_REPS, allow_pickle=True)
    analytes = [str(a) for a in z["analytes"]]
    Xa = np.asarray(z["corpusX"], float)                       # (167, 676) analyte means
    grid = np.asarray(z["grid"], float)
    meta = pd.DataFrame({
        "spectrum_id": [f"degraded::{a}" for a in analytes],
        "analyte": analytes,
        "source": "committed_artifact",
        "excitation_nm": np.nan,
        "replicate": "1",
    })
    return Corpus(X=Xa, grid=grid, meta=meta, mode="degraded",
                  notes=["raw root unavailable — per-spectrum structure not recoverable; "
                         "replicate grouping and quality metadata are unavailable in this mode"])


def load_corpus(explicit_root: str | None = None) -> Corpus:
    root = P.data_root(explicit_root)
    if root is None:
        return load_degraded()
    try:
        return load_full(root)
    except Exception as exc:                                    # pragma: no cover
        c = load_degraded()
        c.notes.append(f"full load failed ({type(exc).__name__}: {exc}); fell back to degraded")
        return c


# ── replicate grouping ────────────────────────────────────────────────────────
def add_replicate_groups(meta: pd.DataFrame) -> pd.DataFrame:
    """Attach both candidate grouping keys so Phase 00 can decide with evidence.

    v5_group   analyte | source | excitation   (what the frozen atlas used, 272 groups)
    v7_group   analyte | excitation            (the V7 specification's recommendation)

    Analyte balancing applies at the analyte level ACROSS groups either way, so the two
    differ only in how within-analyte variation is bucketed.
    """
    m = meta.copy()
    exc = m.excitation_nm.astype(str)
    m["v5_replicate_group"] = m.analyte + "|" + m.source + "|" + exc
    m["v7_replicate_group"] = m.analyte + "|" + exc
    return m


def dataset_card(c: Corpus) -> dict:
    m = c.meta
    card = {
        "domain": "Raman only (canonical observation domain)",
        "excluded_domains": EXCLUDED_DOMAINS,
        "load_mode": c.mode,
        "n_spectra": int(len(m)),
        "n_analytes": int(m.analyte.nunique()),
        "n_bins": int(c.X.shape[1]),
        "window_cm": list(P.WINDOW_CM),
        "grid_step_cm": P.GRID_STEP_CM,
        "preprocessing": dict(P.PREPROC),
        "sources": {k: int(v) for k, v in m.source.value_counts().items()},
    }
    if c.mode == "full":
        multi = m.groupby("analyte").excitation_nm.nunique()
        g5 = m.groupby("v5_replicate_group").size() if "v5_replicate_group" in m else None
        g7 = m.groupby("v7_replicate_group").size() if "v7_replicate_group" in m else None
        card |= {
            "excitations": {str(k): int(v) for k, v in m.excitation_nm.value_counts().items()},
            "n_analytes_multi_excitation": int((multi > 1).sum()),
            "analytes_with_replicates": int((m.groupby("analyte").size() > 1).sum()),
        }
        if g5 is not None:
            card["replicate_groups_v5"] = {"n": int(g5.shape[0]),
                                           "median_size": float(g5.median()),
                                           "max_size": int(g5.max())}
        if g7 is not None:
            card["replicate_groups_v7"] = {"n": int(g7.shape[0]),
                                           "median_size": float(g7.median()),
                                           "max_size": int(g7.max())}
    card["notes"] = list(c.notes)
    return card


def check_against_frozen(card: dict) -> list[dict]:
    """Compare a freshly computed card against the frozen V5 corpus card."""
    checks = []

    def add(item, expected, got):
        checks.append({"item": item, "expected": expected, "got": got,
                       "status": "PASS" if expected == got else "FAIL"})

    add("n_spectra", EXPECTED["n_spectra"], card.get("n_spectra"))
    add("n_analytes", EXPECTED["n_analytes"], card.get("n_analytes"))
    add("n_bins", EXPECTED["n_bins"], card.get("n_bins"))
    add("sources", EXPECTED["sources"], card.get("sources"))
    if card.get("load_mode") == "full":
        add("excitations", EXPECTED["excitations"], card.get("excitations"))
        add("n_analytes_multi_excitation", EXPECTED["n_analytes_multi_excitation"],
            card.get("n_analytes_multi_excitation"))
        add("analytes_with_replicates", EXPECTED["analytes_with_replicates"],
            card.get("analytes_with_replicates"))
        add("replicate_groups_v5.n", EXPECTED["replicate_groups_v5"],
            (card.get("replicate_groups_v5") or {}).get("n"))
    return checks
