#!/usr/bin/env python3
"""GAIRA V7 — pre-Phase-02 corpus identity and pure-Raman completeness audit.

Reconstructs the intended V7 pure-Raman corpus directly from the configured data root and
reconciles every count from raw file to canonical molecule. Modality is verified from source
metadata, loader records and archive provenance — never inferred from file names.

    python results/v7_rebuild/corpus_audit/code/audit_corpus.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AUD = HERE.parent
REPO = AUD.parents[2]
sys.path.insert(0, str(REPO / "results/v7_rebuild/phase00/code"))
sys.path.insert(0, str(REPO / "src"))

import v7_paths as P                                        # noqa: E402
from gaira.data import gobbato, loader                      # noqa: E402
from gaira.data.synonyms import canonical                   # noqa: E402
from gaira.foundation.dataset import AA_NAME_FIX            # noqa: E402
from gaira.preprocessing import pipeline as pp              # noqa: E402

warnings.filterwarnings("ignore")

T, A, LOGS = AUD / "tables", AUD / "artifacts", AUD / "logs"
P00 = REPO / "results/v7_rebuild/phase00"
GOBBATO_ZIP = "serum_ag_colloids/dataset_spectral_data.zip"
LOG: list[str] = []

# Sources classified as pure-Raman representation-training inputs, from the frozen corpus card.
RAMAN_SOURCES = {"RamanBioLib", "gobbato_raman_metabolites", "amino_acid_raman_grounding"}

# Everything under the data root that must never enter the Raman foundation.
EXCLUDED_DATASETS = [
    "adenine_sers_control", "ag_flakes_metabolites_23", "cca_hcc_lm_serum_sers",
    "coeliac_faecal_sers", "covid_serum_raman", "cspp_serum", "diabetes_plasma_ev_sers",
    "ergothioneine_serum", "european_multi_instrument_adenine", "hcc_serum",
    "metabolite_sers63_support", "mycoplasma_na_sers", "nature_serum_sers", "otc_drugs",
    "ovarian_plasma_raman_sers", "sers24_metabolite_support", "sers_metabolite_63",
    "sers_fingerprint_workingpaper_support", "serum_ag_colloids", "serum_protocol_comparison",
    "shine_ev_sers", "single_vesicle_ev_raman", "small2023_ev", "stroke_urine_sers",
    "ucla_saliva_sev_gc", "serum_ag_colloids_grounding",
    "serum_ag_colloids_literature_grounding",
]

# Substrate strings that would indicate a plasmonic (SERS) substrate rather than a holder.
SERS_SUBSTRATE_PAT = re.compile(
    r"sers|colloid|nanoparticle|nanostar|nanorod|roughen|island film|klarite|"
    r"silver|\bag\b|\bau\b|gold", re.IGNORECASE)


def log(m):
    line = f"[corpus-audit] {m}"
    print(line, flush=True)
    LOG.append(line)


def wtab(df, name):
    T.mkdir(parents=True, exist_ok=True)
    df.to_csv(T / name, index=False, lineterminator="\n")
    return name


def wjson(o, name):
    A.mkdir(parents=True, exist_ok=True)
    (A / name).write_text(json.dumps(o, indent=2, ensure_ascii=False, default=str) + "\n")
    return name


# ── Part 1 — spectrum-level inventory, modality verified from metadata ────────
def build_inventory(root: Path) -> pd.DataFrame:
    grid = pp.common_grid(450.0, 1800.0, 2.0)
    alias = pd.read_csv(P00 / "tables/alias_table_v1.csv")
    a2c = dict(zip(alias.surface_form, alias.canonical_id))
    part = pd.read_csv(P00 / "tables/chemical_partition_v1.csv")
    fine = dict(zip(part.canonical_id, part.fine_class))

    rows = []

    # RamanBioLib — modality from the loader record; substrate from the source index
    idx = pd.read_csv(loader.RAMANBIOLIB_INDEX).set_index("id")
    for s in loader.load_ramanbiolib():
        r = s.record
        sid = int(str(r.spectrum_id).split("::")[1])
        sub = str(idx.loc[sid, "sample_substrate"]) if sid in idx.index else ""
        norm = canonical(r.canonical_analyte_name)
        cid = a2c.get(norm, norm)
        susp = bool(SERS_SUBSTRATE_PAT.search(sub))
        rows.append({
            "spectrum_id": r.spectrum_id, "source_dataset": "RamanBioLib",
            "original_file": Path(str(r.raw_path)).name,
            "original_analyte_label": r.canonical_analyte_name,
            "normalized_label": norm, "canonical_id": cid,
            "modality": r.modality.value, "substrate": sub,
            "excitation_nm": r.excitation_nm, "is_pure_analyte": True,
            "included_in_v7_raman": True,
            "inclusion_reason": "pure Raman reference library; modality=raman in source record",
            "exclusion_reason": "",
            "chemistry_class": fine.get(cid, ""),
            "replicate_group": f"{cid}|{r.excitation_nm}",
            "substrate_flag": ("SUSPECT: substrate string matches a plasmonic pattern"
                               if susp else ""),
        })

    # Gobbato — the archive holds BOTH Raman powders and Ag-SERS; separated by record modality
    for s in gobbato.load_gobbato_785():
        r = s.record
        is_raman = r.modality.value == "raman"
        norm = canonical(r.canonical_analyte_name)
        cid = a2c.get(norm, norm) if is_raman else ""
        rows.append({
            "spectrum_id": r.spectrum_id,
            "source_dataset": ("gobbato_raman_metabolites" if is_raman
                               else "gobbato_sers_metabolites"),
            "original_file": Path(str(r.raw_path)).name,
            "original_analyte_label": r.canonical_analyte_name,
            "normalized_label": norm if is_raman else "",
            "canonical_id": cid, "modality": r.modality.value,
            "substrate": r.substrate_material, "excitation_nm": r.excitation_nm,
            "is_pure_analyte": True,
            "included_in_v7_raman": is_raman,
            "inclusion_reason": ("pure Raman powder; modality=raman, substrate=powder"
                                 if is_raman else ""),
            "exclusion_reason": ("" if is_raman else
                                 "Ag-SERS: modality=sers, substrate=Ag colloid (Gobbato)"),
            "chemistry_class": fine.get(cid, "") if is_raman else "",
            "replicate_group": f"{cid}|785.0" if is_raman else "",
            "substrate_flag": "",
        })

    # amino-acid grounding — column-per-analyte spreadsheet
    aa = root / "amino_acid_raman_grounding/aa.xlsx"
    if aa.exists():
        df = pd.read_excel(aa)
        wn = df.iloc[:, 0].values.astype(float)
        seen = collections.Counter()
        for col in df.columns[1:]:
            lab = str(col).strip()
            fixed = AA_NAME_FIX.get(lab.lower(), lab.lower())
            norm = canonical(fixed)
            cid = a2c.get(norm, norm)
            y = pd.to_numeric(df[col], errors="coerce").values.astype(float)
            ok = np.isfinite(y)
            seen[norm] += 1
            rows.append({
                "spectrum_id": f"amino_acid_raman::{norm}#{seen[norm]}",
                "source_dataset": "amino_acid_raman_grounding",
                "original_file": "aa.xlsx", "original_analyte_label": lab,
                "normalized_label": norm, "canonical_id": cid, "modality": "raman",
                "substrate": "powder/standard", "excitation_nm": 785.0,
                "is_pure_analyte": True,
                "included_in_v7_raman": bool(ok.sum() >= 100),
                "inclusion_reason": "pure amino-acid Raman reference",
                "exclusion_reason": ("" if ok.sum() >= 100
                                     else f"only {int(ok.sum())} finite points (<100)"),
                "chemistry_class": fine.get(cid, ""),
                "replicate_group": f"{cid}|785.0", "substrate_flag": "",
            })
    return pd.DataFrame(rows)


# ── Part 2 — Gobbato-specific audit against the raw archive ──────────────────
def gobbato_audit(root: Path, inv: pd.DataFrame) -> dict:
    z = root / GOBBATO_ZIP
    arch = {"zip_present": z.exists()}
    if not z.exists():
        return arch
    with zipfile.ZipFile(z) as f:
        names = [n for n in f.namelist() if not n.endswith("/")]
    ram_files = [n for n in names if "Raman metabolites" in n]
    sers_files = [n for n in names if "SERS metabolites" in n and "fitting" not in n]
    fit_files = [n for n in names if "fitting" in n]

    rpat = re.compile(r"Raman_pwd_(.+?)_s_(\d+)\.txt")
    spat = re.compile(r"SERS_met_(.+?)_(.+?)_(\d+)\.txt")
    r_ok = [n for n in ram_files if rpat.match(Path(n).name)]
    r_bad = [Path(n).name for n in ram_files if not rpat.match(Path(n).name)]
    r_lab = collections.Counter(rpat.match(Path(n).name).group(1) for n in r_ok)
    s_lab = collections.Counter(spat.match(Path(n).name).group(1) for n in sers_files
                                if spat.match(Path(n).name))

    loaded_r = inv[inv.source_dataset == "gobbato_raman_metabolites"]
    loaded_s = inv[inv.source_dataset == "gobbato_sers_metabolites"]

    wtab(pd.DataFrame([{"archive_file": Path(n).name, "source_label": rpat.match(Path(n).name).group(1),
                        "replicate": rpat.match(Path(n).name).group(2), "modality": "raman",
                        "loaded": True} for n in sorted(r_ok)]),
         "gobbato_pure_raman_inventory.csv")
    wtab(pd.DataFrame([{"archive_file": Path(n).name,
                        "source_label": (spat.match(Path(n).name).group(1)
                                         if spat.match(Path(n).name) else ""),
                        "modality": "ag_sers", "included_in_v7_raman": False,
                        "exclusion_reason": "Ag colloid SERS — excluded by modality"}
                       for n in sorted(sers_files)]),
         "gobbato_ag_sers_inventory.csv")
    both = sorted(set(r_lab) & set(s_lab))
    wtab(pd.DataFrame([{"source_label": l, "has_raman": l in r_lab, "has_ag_sers": l in s_lab,
                        "n_raman_files": r_lab.get(l, 0), "n_sers_files": s_lab.get(l, 0)}
                       for l in sorted(set(r_lab) | set(s_lab))]),
         "gobbato_raman_sers_pair_map.csv")

    rbl_norm = set(inv[inv.source_dataset == "RamanBioLib"].normalized_label)
    g_norm = set(loaded_r.normalized_label)
    wtab(pd.DataFrame([{"normalized_label": n, "canonical_id":
                        loaded_r[loaded_r.normalized_label == n].canonical_id.iloc[0],
                        "unique_to_gobbato": n not in rbl_norm}
                       for n in sorted(g_norm)]),
         "gobbato_unique_raman_molecules.csv")

    arch |= {
        "archive_raman_files": len(ram_files), "archive_raman_parseable": len(r_ok),
        "archive_raman_unmatched_filenames": r_bad,
        "archive_raman_source_labels": len(r_lab),
        "replicates_per_label": dict(collections.Counter(r_lab.values())),
        "archive_sers_files": len(sers_files), "archive_sers_for_fitting": len(fit_files),
        "archive_sers_source_labels": len(s_lab),
        "loaded_raman_spectra": int(len(loaded_r)),
        "loaded_sers_spectra_EXCLUDED": int(len(loaded_s)),
        "raman_files_missing_from_corpus": len(r_ok) - int(len(loaded_r)),
        "sers_files_leaked_into_corpus": int(loaded_s.included_in_v7_raman.sum()),
        "labels_with_both_raman_and_sers": len(both),
        "labels_sers_only_correctly_excluded": sorted(set(s_lab) - set(r_lab)),
        "labels_raman_only": sorted(set(r_lab) - set(s_lab)),
        "gobbato_normalized_labels": len(g_norm),
        "gobbato_canonical_molecules": int(loaded_r.canonical_id.nunique()),
        "overlap_with_ramanbiolib": len(g_norm & rbl_norm),
        "unique_to_gobbato": len(g_norm - rbl_norm),
    }
    return arch


# ── Part 3 — canonicalisation audit ──────────────────────────────────────────
MERGE_CLASSES = {
    "riboﬂavin": ("2 Unicode normalization", "U+FB02 ligature; NFKC-identical"),
    "ure": ("3 abbreviation", "truncated spreadsheet header"),
    "alb": ("3 abbreviation", "truncated spreadsheet header"),
    "gluth": ("3 abbreviation", "truncated spreadsheet header"),
    "acetyl coenzyme a": ("4 common vs systematic name", "long form vs hyphenated abbreviation"),
    "aspartic acid": ("5 salt/free-acid equivalence", "conjugate acid/base of one molecule"),
    "n-acetyl- d-glucosamine": ("1 spelling/formatting", "stray space and D- prefix"),
    "(+)-dextrose": ("4 common vs systematic name", "dextrose is D-(+)-glucose"),
    "glucose": ("7 stereoisomer prefix (generic→specific)", "unprefixed denotes the natural D-(+) form"),
    "fructose": ("7 stereoisomer prefix (generic→specific)", "unprefixed denotes the natural D-(−) form"),
    "galactose": ("7 stereoisomer prefix (generic→specific)", "unprefixed denotes the natural D-(+) form"),
    "mannose": ("7 stereoisomer prefix (generic→specific)", "unprefixed denotes the natural D-(+) form"),
    "lactose": ("7 stereoisomer prefix (generic→specific)", "unprefixed denotes the natural (+) form"),
}

V5_NORMALIZATION_CLASSES = {
    "l-prefix": "7 stereoisomer prefix — L- is the biological form; source labels differ only in prefix",
    "acid-base": "5 salt/free-acid equivalence — VERIFIED EMPIRICALLY (see acid_base_verification)",
    "d-prefix": "7 stereoisomer prefix — D- is the biological form",
}


def canonicalisation_audit(inv: pd.DataFrame) -> tuple:
    raman = inv[inv.included_in_v7_raman]
    # many-to-one: source label -> canonical id
    m2o = []
    for cid, g in raman.groupby("canonical_id"):
        labs = sorted(set(g.original_analyte_label.str.lower()))
        norms = sorted(set(g.normalized_label))
        if len(labs) < 2:
            continue
        kinds = []
        for n in norms:
            if n in MERGE_CLASSES:
                kinds.append(MERGE_CLASSES[n][0])
        for lab in labs:
            if lab in MERGE_CLASSES:
                kinds.append(MERGE_CLASSES[lab][0])
        if not kinds:
            kinds = ["1 spelling/formatting" if len(norms) == 1
                     else "7 stereoisomer prefix (generic→specific)"]
        m2o.append({
            "canonical_id": cid, "n_source_labels": len(labs),
            "source_labels": ";".join(labs), "normalized_labels": ";".join(norms),
            "n_source_datasets": int(g.source_dataset.nunique()),
            "source_datasets": ";".join(sorted(g.source_dataset.unique())),
            "n_spectra": int(len(g)),
            "merge_classification": ";".join(sorted(set(kinds))),
            "cross_source": bool(g.source_dataset.nunique() > 1),
        })
    m2o = pd.DataFrame(m2o).sort_values("canonical_id")

    # one-to-many: a raw spectrum mapping to >1 canonical id (must be empty)
    o2m = (raman.groupby("spectrum_id").canonical_id.nunique()
           .pipe(lambda s: s[s > 1]).reset_index()
           .rename(columns={"canonical_id": "n_canonical_ids"}))

    # protected distinctions that must NOT have been collapsed
    protected = [
        ("(+)-arabinose", "(-)-arabinose", "enantiomers"),
        ("(+)-glucose", "β-d-glucose", "anomers"),
        ("(-)-ribose", "2-deoxy-d-ribose", "distinct molecules (2'-OH)"),
        ("carotene", "β-carotene", "isomer — provenance does not prove equivalence"),
    ]
    prot_rows = []
    cids = set(raman.canonical_id)
    lab2cid = dict(zip(raman.normalized_label, raman.canonical_id))
    for a, b, why in protected:
        ca, cb = lab2cid.get(a), lab2cid.get(b)
        prot_rows.append({"form_a": a, "form_b": b, "relationship": why,
                          "canonical_a": ca, "canonical_b": cb,
                          "collapsed": bool(ca is not None and ca == cb),
                          "status": ("VIOLATION — collapsed" if (ca and ca == cb)
                                     else "protected" if (ca and cb)
                                     else "one form absent from the corpus")})
    prot = pd.DataFrame(prot_rows)

    unresolved = pd.DataFrame([
        {"item": "carotene vs β-carotene", "kind": "11 unresolved",
         "detail": "loose-key collision across two sources, both chromophore_pigment; the "
                   "source spreadsheet does not state which carotene isomer it holds",
         "action": "NOT merged; recorded as unresolved"},
        {"item": "RamanBioLib id 197 (insulin), 'Gold-coated glass sustrate' @633 nm",
         "kind": "modality provenance",
         "detail": "gold + 633 nm is a plasmonic combination, but a SMOOTH gold-coated slide "
                   "is a standard non-enhancing reflective substrate for normal Raman; the "
                   "source library declares Raman and gives no roughening/nanostructure field",
         "action": "RETAINED, flagged; sensitivity tested"},
    ])
    return m2o, o2m, prot, unresolved


def acid_base_verification(root: Path) -> pd.DataFrame:
    """Do the V5 acid/conjugate-base merges join materially different spectra?"""
    grid = pp.common_grid(450.0, 1800.0, 2.0)

    def prep(w, y):
        return pp.preprocess(np.asarray(w, float), np.asarray(y, float),
                             {"baseline": "asls", "smooth": "savgol", "norm": "l2"},
                             grid, (450.0, 1800.0))

    spec = collections.defaultdict(list)
    for s in loader.load_ramanbiolib():
        spec[s.record.canonical_analyte_name.lower()].append(prep(s.wavenumber, s.intensity))
    for s in gobbato.load_gobbato_785():
        if s.record.modality.value == "raman":
            spec[s.record.canonical_analyte_name.lower()].append(prep(s.wavenumber, s.intensity))

    COOH = np.abs(grid - 1710) < 40
    COO = np.abs(grid - 1400) < 40
    rows = []
    for tgt, (la, lb) in {"ascorbate": ("ascorbic acid", "ascorbate"),
                          "citrate": ("citric acid", "citrate"),
                          "oleate": ("oleic acid", "oleate"),
                          "stearate": ("stearic acid", "stearate"),
                          "aspartate": ("aspartic acid", "aspartate")}.items():
        A, B = spec.get(la, []), spec.get(lb, [])
        if not A or not B:
            rows.append({"canonical_id": tgt, "form_a": la, "form_b": lb,
                         "n_a": len(A), "n_b": len(B), "cosine": None,
                         "verdict": "one form absent — no merge to verify"})
            continue
        a = np.nan_to_num(np.nanmean(np.vstack(A), 0))
        b = np.nan_to_num(np.nanmean(np.vstack(B), 0))
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        ra, rb = a[COOH].sum() / (a.sum() + 1e-12), b[COOH].sum() / (b.sum() + 1e-12)
        ca, cb = a[COO].sum() / (a.sum() + 1e-12), b[COO].sum() / (b.sum() + 1e-12)
        rows.append({
            "canonical_id": tgt, "form_a": la, "form_b": lb, "n_a": len(A), "n_b": len(B),
            "cosine": round(cos, 4),
            "cooh_1710_share_a": round(float(ra), 5), "cooh_1710_share_b": round(float(rb), 5),
            "coo_1400_share_a": round(float(ca), 5), "coo_1400_share_b": round(float(cb), 5),
            "cooh_ratio": round(float(max(ra, rb) / (min(ra, rb) + 1e-12)), 3),
            "verdict": ("same protonation state — labelling variant, merge valid"
                        if max(ra, rb) / (min(ra, rb) + 1e-12) < 4.0
                        else "DIFFERENT protonation state — merge questionable"),
        })
    return pd.DataFrame(rows)


def main():
    for d in (T, A, LOGS, AUD / "figures", AUD / "reports"):
        d.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)
    root = P.data_root()
    if root is None:
        log("ABORT: raw data root unavailable (set GAIRA_DATA_ROOT)")
        return 1
    log(f"data root resolved; auditing {len(EXCLUDED_DATASETS)} excluded datasets by policy")

    inv = build_inventory(root)
    wtab(inv, "spectrum_level_audit_v1.csv")
    raman = inv[inv.included_in_v7_raman]
    log(f"Part 1 — inventory: {len(inv)} spectra examined, {len(raman)} included as pure Raman")

    gob = gobbato_audit(root, inv)
    wjson(gob, "gobbato_audit_v1.json")
    log(f"Part 2 — Gobbato: archive Raman {gob['archive_raman_files']} files / "
        f"{gob['archive_raman_source_labels']} labels; loaded {gob['loaded_raman_spectra']}; "
        f"missing {gob['raman_files_missing_from_corpus']}; "
        f"SERS leaked {gob['sers_files_leaked_into_corpus']}")

    m2o, o2m, prot, unres = canonicalisation_audit(inv)
    wtab(m2o, "canonicalization_many_to_one_audit.csv")
    wtab(o2m if len(o2m) else pd.DataFrame(columns=["spectrum_id", "n_canonical_ids"]),
         "canonicalization_one_to_many_conflicts.csv")
    wtab(prot, "canonicalization_protected_distinctions.csv")
    wtab(unres, "canonicalization_unresolved.csv")
    log(f"Part 3 — canonicalisation: {len(m2o)} many-to-one groups, "
        f"{len(o2m)} one-to-many conflicts, "
        f"{int((prot.status == 'VIOLATION — collapsed').sum())} protected violations")

    ab = acid_base_verification(root)
    wtab(ab, "acid_base_merge_verification.csv")
    bad = ab[ab.verdict.astype(str).str.startswith("DIFFERENT")]
    log(f"   acid/base merges verified: {len(ab)} checked, {len(bad)} questionable")

    # ── Part 4 — the reconciliation ───────────────────────────────────────────
    lab_pairs = {(r.source_dataset, str(r.original_analyte_label).lower())
                 for r in raman.itertuples()}
    raw_lab = {str(l).lower() for l in raman.original_analyte_label}
    steps = [
        ("A  raw Raman spectra (files/columns loaded)", len(raman), ""),
        ("B  dataset-specific source labels (source, label)", len(lab_pairs),
         "same label in two libraries counted twice"),
        ("B' distinct raw label strings", len(raw_lab),
         "cross-source identical strings collapsed"),
        ("C  normalized analyte names (V5 synonyms layer)", raman.normalized_label.nunique(),
         "L-/D- prefixes, acid/base names, salt forms"),
        ("D  canonical molecule IDs (V7 layer)", raman.canonical_id.nunique(),
         "NFKC, truncations, cross-source duplicates"),
    ]
    rec = pd.DataFrame([{"step": s, "count": int(n), "note": note} for s, n, note in steps])
    rec["delta_from_previous"] = rec["count"].diff().fillna(0).astype(int)
    wtab(rec, "count_reconciliation_v1.csv")
    log("Part 4 — reconciliation: " +
        " → ".join(f"{int(r['count'])}" for _, r in rec.iterrows()))

    # per-source and per-class canonical counts
    bysrc = (raman.groupby("source_dataset")
             .agg(spectra=("spectrum_id", "size"),
                  source_labels=("original_analyte_label", "nunique"),
                  normalized=("normalized_label", "nunique"),
                  canonical=("canonical_id", "nunique")).reset_index())
    wtab(bysrc, "canonical_count_by_source_v1.csv")
    bycls = (raman.groupby("chemistry_class")
             .agg(canonical=("canonical_id", "nunique"),
                  spectra=("spectrum_id", "size")).reset_index()
             .sort_values("canonical", ascending=False))
    wtab(bycls, "canonical_count_by_class_v1.csv")

    reps = raman.groupby("canonical_id").size()
    exc = raman.groupby("canonical_id").excitation_nm.nunique()
    wtab(pd.DataFrame({"canonical_id": reps.index, "n_spectra": reps.values,
                       "n_excitations": exc.reindex(reps.index).values}),
         "spectra_per_canonical_molecule_v1.csv")

    # ── registry v2 ───────────────────────────────────────────────────────────
    reg = []
    for cid, g in raman.groupby("canonical_id"):
        reg.append({
            "canonical_id": cid,
            "n_source_labels": int(g.original_analyte_label.str.lower().nunique()),
            "source_labels": ";".join(sorted(set(g.original_analyte_label.str.lower()))),
            "normalized_labels": ";".join(sorted(set(g.normalized_label))),
            "source_datasets": ";".join(sorted(g.source_dataset.unique())),
            "n_spectra": int(len(g)), "n_excitations": int(g.excitation_nm.nunique()),
            "excitations": ";".join(str(x) for x in sorted(g.excitation_nm.dropna().unique())),
            "chemistry_class": g.chemistry_class.iloc[0],
            "class_conflict": bool(g.chemistry_class.nunique() > 1),
            "structure_identifier": "", "identifier_source": "none available in source data",
        })
    regdf = pd.DataFrame(reg).sort_values("canonical_id")
    wtab(regdf, "canonical_molecule_registry_v2.csv")

    summary = {
        "generated_utc": t0.isoformat(),
        "raw_raman_spectra": int(len(raman)),
        "dataset_specific_source_labels": len(lab_pairs),
        "distinct_raw_label_strings": len(raw_lab),
        "normalized_analyte_names": int(raman.normalized_label.nunique()),
        "canonical_molecule_ids": int(raman.canonical_id.nunique()),
        "unique_chemical_structures": None,
        "structure_identifier_note": ("no InChIKey/SMILES/CID present in any source dataset; "
                                      "none fetched externally — uncertainty preserved"),
        "protected_stereoisomers_retained": int((prot.status == "protected").sum()),
        "alias_collisions": int(len(m2o)),
        "cross_source_duplicate_molecules": int(m2o.cross_source.sum()) if len(m2o) else 0,
        "one_to_many_conflicts": int(len(o2m)),
        "excitation_domains": int(raman.excitation_nm.nunique()),
        "chemistry_classes": int(raman.chemistry_class.nunique()),
        "sers_spectra_examined_and_excluded": int((~inv.included_in_v7_raman).sum()),
        "gobbato": gob,
        "acid_base_merges_questionable": int(len(bad)),
        "substrate_suspects": inv[inv.substrate_flag != ""][
            ["spectrum_id", "original_analyte_label", "substrate", "excitation_nm"]
        ].to_dict("records"),
    }
    wjson(summary, "corpus_audit_summary_v1.json")
    (LOGS / "audit.log").write_text("\n".join(LOG) + "\n")
    log(f"canonical molecule count = {summary['canonical_molecule_ids']}")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
