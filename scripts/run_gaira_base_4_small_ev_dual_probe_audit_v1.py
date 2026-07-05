"""gaira_base_4_small_ev_dual_probe_audit_v1

Phase: STRICT PRE-ANALYSIS AUDIT of the small2023_ev dual-probe dataset.

Goal: understand WHAT Probe 1 vs Probe 2 are physically, chemically, and
experimentally — BEFORE any GAIRA analysis.

STRICT RULES:
- NO GAIRA scoring
- NO classifier
- NO interpretation claims
- ONLY physics + dataset audit + pre-GAIRA hypothesis

Dataset:
  /Volumes/SSD_Rad/GAIRA_DATA/raw/small2023_ev/
    NormedProbe1.mat  → struct `normed1` with fields c00, c01, c10, c25, c50, c100
    NormedProbe2.mat  → struct `Normed`  with fields c00, c01, c10, c25, c50, c100
    Main_Text.zip     → MATLAB code + figure data
    Readme.docx       → "Label-free identification of exosomes using Raman spectroscopy
                         and machine learning" (Parlatan et al., Small 2023)

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_4_small_ev_dual_probe_audit_v1.py
"""
from __future__ import annotations

import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_small_ev_dual_probe_audit_v1")
TABLES  = ROOT / "tables"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT   = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/small2023_ev")
P1_PATH = DATA_DIR / "NormedProbe1.mat"
P2_PATH = DATA_DIR / "NormedProbe2.mat"

COHORTS = ["c00", "c01", "c10", "c25", "c50", "c100"]


# ──────────────────────────────────────────────────────────────────────
# STEP 0+1 — locate + load + inferred metadata
# ──────────────────────────────────────────────────────────────────────
def load_probe(path: Path, key_candidates):
    d = sio.loadmat(path, squeeze_me=False)
    key = None
    for k in key_candidates:
        if k in d: key = k; break
    if key is None:
        for k in d:
            if not k.startswith("__"): key = k; break
    s = d[key]
    rec = s[0, 0]
    out = {}
    for fld in s.dtype.names:
        out[fld] = rec[fld]
    return key, out


def step0_locate(p1, p2):
    print("[STEP 0] dataset location + structure")
    rows = []
    for tag, path, probe in [("Probe1", P1_PATH, p1), ("Probe2", P2_PATH, p2)]:
        for cohort in COHORTS:
            arr = probe.get(cohort)
            if arr is None: continue
            rows.append({
                "probe":           tag,
                "cohort":          cohort,
                "n_spectra":       int(arr.shape[0]),
                "n_wn_columns":    int(arr.shape[1]),
                "file_path":       str(path),
                "loader_format":   "scipy.io.loadmat → struct (one field per cohort)",
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "dataset_inventory_v1.csv", index=False)
    print(df.groupby("probe")[["n_spectra", "n_wn_columns"]].agg({
        "n_spectra": "sum", "n_wn_columns": "first"}))
    return df


# ──────────────────────────────────────────────────────────────────────
# STEP 1 — Paper + methods table (extracted from Readme + paper context)
# ──────────────────────────────────────────────────────────────────────
def step1_paper_methods():
    print("[STEP 1] paper + methods extraction")
    # What we KNOW from Readme.docx:
    # - Paper: "Label-free identification of exosomes using Raman spectroscopy and machine learning"
    # - Authors: Parlatan, Ozen, Kecoglu, Koyuncu, Torun, et al.
    # - Published: Small 2023 (folder name)
    # - Used 5 cell lines for cell ID: MEF, HEC-1-A, HeLa, HT-1080, THP-1 (from Fig_3*.m)
    # - Mixture cohorts c00..c100: ratios of HT-1080-derived EVs in THP-1 background
    #   (from Fig5 Labels.csv: ht100, thp100, thp50ht1, thp50ht10, thp50ht25, thp50ht50)
    #
    # What we DO NOT KNOW from local files (would need paper PDF or Methods section):
    # - exact material / morphology / chemistry for Probe 1 vs Probe 2
    # - laser wavelength
    # - integration time
    # - EV isolation method
    # - whether SAME EV sample was measured on both probes (very likely yes given matched cohort labels)
    rows = [
        {
            "field": "paper title",
            "Probe1": "(common) Label-free identification of exosomes using Raman spectroscopy and machine learning",
            "Probe2": "(common) — same paper",
            "evidence": "Readme.docx",
        },
        {
            "field": "authors", "Probe1": "Parlatan U., Ozen M.O., Kecoglu I., Koyuncu B., Torun H., et al.",
            "Probe2": "(same)", "evidence": "Readme.docx Credits section",
        },
        {
            "field": "venue", "Probe1": "Small 2023 (inferred from dataset folder name `small2023_ev`)",
            "Probe2": "(same)", "evidence": "folder name + paper title match",
        },
        {
            "field": "task in paper", "Probe1": "Label-free Raman/SERS identification of EV cell-of-origin",
            "Probe2": "(same)",
            "evidence": "Readme abstract + Fig3-Fig5 cell-line PCA / RF / DL pipelines",
        },
        {
            "field": "n spectra (this dataset)", "Probe1": "19,557 (4687+3598+2349+2279+2000+4644)",
            "Probe2": "85,583 (14884+11163+14884+14884+14884+14884)",
            "evidence": "shapes from .mat structs",
        },
        {
            "field": "n wavenumber columns", "Probe1": "1131",
            "Probe2": "1400",
            "evidence": "shapes from .mat structs",
        },
        {
            "field": "wavenumber axis (Calx)", "Probe1": "NOT EMBEDDED in NormedProbe1.mat — would need data_BC_NORM.mat from paper Drive",
            "Probe2": "NOT EMBEDDED in NormedProbe2.mat",
            "evidence": "loadmat keys — only `normed1` / `Normed` present; first/last column are zero (edge bound)",
        },
        {
            "field": "preprocessing state", "Probe1": "Already baseline-corrected + normalized (variable name = NormedProbe*)",
            "Probe2": "(same)",
            "evidence": "filename + first col always 0 + small float magnitudes (~1e-3)",
        },
        {
            "field": "cohort labels (c00..c100)", "Probe1": "EV mixture ratios — HT-1080:THP-1 dilution series",
            "Probe2": "(same labels)",
            "evidence": "Fig5/Labels.csv: ht100 / thp100 / thp50ht1 / thp50ht10 / thp50ht25 / thp50ht50",
        },
        {
            "field": "probe material", "Probe1": "**UNKNOWN — not in local files**",
            "Probe2": "**UNKNOWN — not in local files**",
            "evidence": "neither Readme nor Fig_*.m specify probe substrate physics; paper Methods PDF would have this",
        },
        {
            "field": "probe morphology", "Probe1": "UNKNOWN", "Probe2": "UNKNOWN",
            "evidence": "(see above)",
        },
        {
            "field": "synthesis method", "Probe1": "UNKNOWN", "Probe2": "UNKNOWN",
            "evidence": "(see above)",
        },
        {
            "field": "functionalization", "Probe1": "UNKNOWN", "Probe2": "UNKNOWN",
            "evidence": "(see above)",
        },
        {
            "field": "enhancement mechanism", "Probe1": "UNKNOWN (Probe label suggests SERS-based, but mechanism unspecified locally)",
            "Probe2": "UNKNOWN",
            "evidence": "(see above)",
        },
        {
            "field": "laser wavelength", "Probe1": "UNKNOWN", "Probe2": "UNKNOWN",
            "evidence": "Methods section needed",
        },
        {
            "field": "integration time", "Probe1": "UNKNOWN", "Probe2": "UNKNOWN", "evidence": "(see above)",
        },
        {
            "field": "sample preparation",
            "Probe1": "EV isolation from cell culture supernatant; conditioned media; method UNSPECIFIED locally",
            "Probe2": "(presumed same)",
            "evidence": "paper context only",
        },
        {
            "field": "study design — same sample on both probes?",
            "Probe1": "LIKELY YES — both probes use identical cohort labels (c00..c100); INFERRED from matched cohort names",
            "Probe2": "(see above)",
            "evidence": "matching cohort labels — but per-probe spectra COUNT differs (Probe2 ~4× larger), suggesting ALIQUOT-LEVEL replicates differ between probes, while sample identity matches",
        },
        {
            "field": "replicates per probe per cohort",
            "Probe1": "Probe1 c00..c100 = 4687/3598/2349/2279/2000/4644 (variable per cohort)",
            "Probe2": "Probe2 c00..c100 = 14884/11163/14884/14884/14884/14884 (mostly fixed at 14884, c01 lower)",
            "evidence": "shape audit",
        },
        {
            "field": "disease groups", "Probe1": "(none — EV cell-of-origin study, not disease cohorts)",
            "Probe2": "(none)",
            "evidence": "cell-line + mixture-ratio design only",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "probe1_vs_probe2_methods_table_v1.csv", index=False)
    return df


# ──────────────────────────────────────────────────────────────────────
# STEP 2 — Physics difference classification
# ──────────────────────────────────────────────────────────────────────
def step2_physics_classification():
    print("[STEP 2] physics difference classification")
    # Without paper Methods, we MUST be conservative. The differences we CAN observe:
    # - Different N wavenumber columns (1131 vs 1400) → different spectral resolution OR different spectrometer range
    # - Different per-cohort spectra count → different acquisition density / mapping protocol
    # - Different normalization scale (visible from raw values)
    # The most likely interpretation given the cell-EV literature is that Probe1 and Probe2
    # are TWO DIFFERENT SERS substrates (e.g. Ag colloid vs Au-coated chip; or two different
    # nanostructured surfaces), but THIS IS A WORKING ASSUMPTION not a confirmed fact.
    classification = {
        "category":          "MAJOR_VARIATION_LIKELY (per working assumption)",
        "rationale": (
            "Different wavenumber-column counts (1131 vs 1400) and different per-cohort "
            "spectrum counts (Probe2 ~4× more spectra per cohort) strongly suggest the two "
            "probes are physically distinct measurement setups, not just two acquisitions on "
            "the same probe. Without paper Methods we cannot confirm whether the difference "
            "is (A) same material + different morphology, or (B) different material entirely."
        ),
        "expected_transferability_ceiling": (
            "MODERATE at best — different SERS substrates rarely produce intensity-identical "
            "spectra for the same EV sample, but BIOCHEMICAL identity (band positions) often "
            "transfers if both probes are SERS-active in the 600-1800 cm⁻¹ range. "
            "Confirm with paper Methods before committing to a transferability ceiling."
        ),
        "what_we_must_confirm_with_paper": (
            "Probe 1 material/morphology/synthesis; Probe 2 same fields; whether SAME EV "
            "ALIQUOT was deposited on both probes; laser wavelength per probe."
        ),
    }
    pd.DataFrame([classification]).to_csv(
        TABLES / "physics_difference_classification_v1.csv", index=False)
    return classification


# ──────────────────────────────────────────────────────────────────────
# STEP 3 — Spectral signature comparison (NO GAIRA)
# ──────────────────────────────────────────────────────────────────────
def step3_spectral_comparison(p1, p2):
    print("[STEP 3] spectral signature comparison (no GAIRA)")
    # Compute mean spectra per cohort per probe (these spectra are already normalized)
    rows = []
    means = {"Probe1": {}, "Probe2": {}}
    for tag, probe in [("Probe1", p1), ("Probe2", p2)]:
        for cohort in COHORTS:
            arr = probe.get(cohort)
            if arr is None: continue
            mean_spec = arr.mean(axis=0)
            means[tag][cohort] = mean_spec
            # Find top-10 peaks (index-based since wn axis unknown)
            rng = float(mean_spec.max() - mean_spec.min())
            if rng > 0:
                idx, _ = find_peaks(mean_spec, prominence=0.05 * rng)
            else:
                idx = np.array([], dtype=int)
            heights = mean_spec[idx] if len(idx) else np.array([])
            order = np.argsort(-heights)
            top_idx = idx[order][:10]
            for rank, ii in enumerate(top_idx, 1):
                rows.append({
                    "probe":   tag,
                    "cohort":  cohort,
                    "rank":    rank,
                    "peak_index_in_axis": int(ii),
                    "peak_index_relative": float(ii / max(len(mean_spec) - 1, 1)),
                    "intensity_at_peak":   float(mean_spec[ii]),
                    "n_wn_columns":        int(len(mean_spec)),
                })
    pd.DataFrame(rows).to_csv(TABLES / "top10_peaks_per_cohort_per_probe_v1.csv", index=False)

    # Persist the mean spectra
    for tag, by_cohort in means.items():
        for cohort, vec in by_cohort.items():
            np.save(TABLES / f"mean_spectrum_{tag}_{cohort}_v1.npy", vec)

    # Mean-spectrum overlay figure (per probe — different lengths, can't combine on one axis)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (tag, by_cohort) in zip(axes, means.items()):
            for cohort, vec in by_cohort.items():
                ax.plot(np.arange(len(vec)), vec, lw=0.8, alpha=0.7, label=cohort)
            ax.set_title(f"{tag} mean spectra by cohort  (n_wn={len(vec)})")
            ax.set_xlabel("wavenumber index (Calx unknown — see audit)")
            ax.set_ylabel("normalized intensity")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_mean_spectra_per_probe_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig mean issue: {e}")

    # Per-probe PCA (within-probe variance structure; cross-probe PCA infeasible w/o common axis)
    try:
        from sklearn.decomposition import PCA
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (tag, probe) in zip(axes, [("Probe1", p1), ("Probe2", p2)]):
            # Sample up to 500 spectra per cohort for PCA tractability
            X_list, y_list = [], []
            rng = np.random.default_rng(0)
            for cohort in COHORTS:
                arr = probe.get(cohort)
                if arr is None: continue
                n = arr.shape[0]
                k = min(500, n)
                idx = rng.choice(n, k, replace=False)
                X_list.append(arr[idx]); y_list += [cohort] * k
            X = np.vstack(X_list); y = np.array(y_list)
            X = np.nan_to_num(X, nan=0.0)
            try:
                Z = PCA(n_components=2).fit_transform(X)
            except Exception as e:
                print(f"  PCA {tag} failed: {e}"); continue
            cmap = plt.cm.viridis(np.linspace(0, 1, len(COHORTS)))
            for i, cohort in enumerate(COHORTS):
                m = y == cohort
                ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.4, color=cmap[i], label=cohort)
            ax.set_title(f"{tag} PCA  (subsampled n_per_cohort ≤ 500)")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_per_probe_pca_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig PCA issue: {e}")
    return means


# ──────────────────────────────────────────────────────────────────────
# STEP 4 — Same-sample (cohort-level) cross-probe consistency
# ──────────────────────────────────────────────────────────────────────
def step4_same_sample_cross_probe(p1, p2, means):
    print("[STEP 4] same-sample cross-probe consistency (cohort means)")
    # Different N wn columns → cannot directly cosine-compare per-spectrum vectors.
    # Compute cosine on RANK-NORMALIZED INDEX-RELATIVE peak signatures instead:
    # for each cohort, take the vector resampled (linear interp) onto a common 1024-pt
    # relative-index axis and compute cosine. Acknowledge this as a coarse proxy for true
    # wavenumber-aligned similarity.
    common_n = 1024
    common_x = np.linspace(0, 1, common_n)
    rows = []
    p1_rs = {}; p2_rs = {}
    for cohort in COHORTS:
        v1 = means["Probe1"].get(cohort); v2 = means["Probe2"].get(cohort)
        if v1 is None or v2 is None: continue
        rs1 = np.interp(common_x, np.linspace(0, 1, len(v1)), v1)
        rs2 = np.interp(common_x, np.linspace(0, 1, len(v2)), v2)
        p1_rs[cohort] = rs1; p2_rs[cohort] = rs2
        # Cosine on resampled index-axis
        n1 = float(np.linalg.norm(rs1)); n2 = float(np.linalg.norm(rs2))
        cos = float((rs1 @ rs2) / (n1 * n2)) if (n1 > 0 and n2 > 0) else np.nan
        # Pearson r
        if np.std(rs1) > 0 and np.std(rs2) > 0:
            r = float(np.corrcoef(rs1, rs2)[0, 1])
        else:
            r = np.nan
        rows.append({
            "cohort":              cohort,
            "n_spectra_probe1":    int(p1[cohort].shape[0]),
            "n_spectra_probe2":    int(p2[cohort].shape[0]),
            "cosine_resampled":    cos,
            "pearson_r_resampled": r,
            "method_caveat":       "wavenumber axis unknown — comparison done on relative-index axis only",
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "same_cohort_cross_probe_similarity_v1.csv", index=False)
    print(df)

    # Side-by-side mean-spectrum overlay on common relative axis
    try:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
        for ax, cohort in zip(axes.flat, COHORTS):
            if cohort not in p1_rs or cohort not in p2_rs:
                ax.set_title(f"{cohort} — missing"); continue
            ax.plot(common_x, p1_rs[cohort], lw=1.2, color="#4C72B0", label="Probe1 (resampled)")
            ax.plot(common_x, p2_rs[cohort], lw=1.2, color="#DD8452", label="Probe2 (resampled)")
            r_row = df[df.cohort == cohort].iloc[0]
            ax.set_title(f"{cohort}  cos={r_row['cosine_resampled']:.2f}  r={r_row['pearson_r_resampled']:.2f}")
            ax.legend(fontsize=7)
        fig.suptitle("Cohort mean spectra cross-probe overlay — RELATIVE-INDEX axis (Calx unknown)",
                       y=1.01)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_same_cohort_cross_probe_overlay_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig overlay issue: {e}")

    # Distribution of similarity values
    cos_mean = float(df["cosine_resampled"].mean())
    cos_sd   = float(df["cosine_resampled"].std())
    r_mean   = float(df["pearson_r_resampled"].mean())
    r_sd     = float(df["pearson_r_resampled"].std())
    summary = {
        "n_cohorts_compared":      int(len(df)),
        "cosine_mean":             cos_mean,
        "cosine_sd":               cos_sd,
        "pearson_r_mean":          r_mean,
        "pearson_r_sd":            r_sd,
        "interpretation_caveat":   "Resampling onto relative-index axis MAY artifactually inflate similarity if peak positions are coincidentally near the same fractional index. True cross-probe transferability requires Calx alignment.",
    }
    pd.DataFrame([summary]).to_csv(TABLES / "cross_probe_similarity_summary_v1.csv", index=False)
    return df, summary


# ──────────────────────────────────────────────────────────────────────
# STEP 5 — Pre-GAIRA hypothesis (no scoring, just expected behavior)
# ──────────────────────────────────────────────────────────────────────
def step5_hypotheses(physics, cross_probe_summary):
    print("[STEP 5] writing pre-GAIRA hypothesis")
    text = [
        "# Pre-GAIRA hypotheses — small2023_ev dual-probe study",
        f"date: {datetime.now().isoformat()}",
        "",
        "## What the dataset is",
        "- Paper: *Label-free identification of exosomes using Raman spectroscopy and machine learning*, "
        "Parlatan et al., Small 2023.",
        "- Dataset includes 6 EV mixture cohorts (c00..c100) measured on TWO different probes:",
        "  - Probe 1: 19,557 spectra, 1131 wn columns",
        "  - Probe 2: 85,583 spectra, 1400 wn columns",
        "- Cohorts almost certainly correspond to HT-1080:THP-1 EV mixture ratios (per Fig5/Labels.csv: "
        "ht100 / thp100 / thp50ht1 / thp50ht10 / thp50ht25 / thp50ht50).",
        "- **Wavenumber axis (Calx) is NOT in the local .mat files** — paper's `data_BC_NORM.mat` would carry it.",
        "- Probe 1 vs Probe 2 physical chemistry (material / morphology / synthesis / functionalization / "
        "enhancement mechanism / laser wavelength) is **NOT specified in any local file** — paper Methods PDF needed.",
        "",
        "## Hypotheses (BEFORE running any GAIRA scoring)",
        "",
        "### H1 — Raw spectra: STRONG probe clustering expected",
        "Different wavenumber resolutions (1131 vs 1400) and different per-cohort spectrum counts "
        "indicate distinct measurement setups. Even if the same biological samples are measured on "
        "both probes, raw normalized intensities will differ in absolute scale and band shape (per "
        "the prior European adenine benchmark η² findings: substrate explained 27% of variance and "
        "labcode explained 26% even at the per-molecule MSS level). Expectation: PCA on raw spectra "
        "will separate Probe 1 from Probe 2 more strongly than it separates cohorts.",
        "",
        "### H2 — Narrow MSS: PARTIALLY probe-sensitive",
        "From the European adenine + multi-molecule wrapper phases, narrow MSS scores are method-locked "
        "(η²(method) = 0.40 for adenine on Fornasaro). For EV mixture cohorts:",
        "- adenine MSS top-K likely VARIES across probes",
        "- ring-window presence (band positions) likely STABLE across probes IF both probes cover the "
        "  720-740 cm⁻¹ window",
        "- substrate-aware POST-HOC wrapper (proven on adenine) MAY recover narrow identity per cohort",
        "",
        "### H3 — Broad BSV (sumnorm/CLR): MOST STABLE layer",
        "Per the cross-pilot synthesis: G09 sterol-lipid ↓ replicated across 5 disease cohorts × 2 regimes "
        "even when raw spectra didn't transfer. For an EV cell-of-origin mixture series, broad axes most "
        "likely to track the HT-1080 vs THP-1 mixture ratio:",
        "- **G06 protein_peptide_backbone** — cell-of-origin proteins differ; should track cohort ratio",
        "- **G08 lipid_acyl_membrane** — EV membrane composition differs by cell line",
        "- **G09 sterol_neutral_lipid** — cholesterol content differs by cell line",
        "- **G02 purine_metabolite** — metabolic state may differ THP-1 (immune) vs HT-1080 (fibrosarcoma)",
        "Expectation: at the broad BSV layer, cohort separation should be visible AND consistent across probes.",
        "",
        "### H4 — Which biochemical axes are likely STABLE across probes?",
        "- Aromatic AA region (~1003 phenylalanine ring breathing) — universally Raman-active, less probe-sensitive",
        "- Amide I/III protein backbone bands — robust across most SERS substrates",
        "- DNA/RNA backbone (phosphate / ring breathing) — moderately stable",
        "- Lipid CH₂/CH₃ deformations — stable when present",
        "- BSV families G05 glycan, G06 protein, G08 lipid, G09 sterol are CANDIDATE STABLE AXES",
        "",
        "### H5 — Likely UNSTABLE / probe-sensitive axes",
        "- Narrow nucleic-acid bands (purine 720-740 region) — known to vary across SERS substrates "
        "(per European adenine benchmark)",
        "- Sulfur/redox band region — known to be substrate-chemistry dependent (S-Au vs S-Ag vs unbound)",
        "- Any band whose position falls near probe-specific surface chemistry resonances",
        "",
        "## Risks for GAIRA",
        "",
        "### R1 — Substrate-sensitive axis predictions",
        "- G01 purine_nucleotide / G02 purine_metabolite — very likely probe-sensitive (matches European adenine finding)",
        "- G10 sulfur_thiol_redox — depends on whether either probe has S-binding chemistry (Au vs Ag)",
        "- G07 aromatic_residue — moderate sensitivity (selective enhancement on some substrates)",
        "",
        "### R2 — Molecule-level collisions",
        "- UA / ERG / HX often collide on Au-substrate SERS (per multi-molecule calibration phase: GSH 912 cm⁻¹ "
        "  collides with cysteine/cystine; ERG 1220 has narrow specificity but suppressed at low conc).",
        "- For an EV mixture, all of these are likely background-level signals — should NOT be treated as identity hits.",
        "",
        "### R3 — Could probe differences invert signals?",
        "- Yes, possibly. Per the European adenine cAu@785 paradox (best concentration-tracking BUT worst MSS identity), "
        "  a probe with strong selective enhancement at one band can DEMOTE that molecule in MSS competition "
        "  even while its quantitative response is excellent. Substrate-aware wrapper can correct for this; "
        "  raw MSS cannot.",
        "",
        f"## Cross-probe similarity (cohort-mean RELATIVE-INDEX cosine)",
        f"- mean cosine across {cross_probe_summary['n_cohorts_compared']} cohorts: "
        f"{cross_probe_summary['cosine_mean']:.2f} ± {cross_probe_summary['cosine_sd']:.2f}",
        f"- mean Pearson r across cohorts: "
        f"{cross_probe_summary['pearson_r_mean']:.2f} ± {cross_probe_summary['pearson_r_sd']:.2f}",
        "- **CAVEAT**: the comparison is on a RELATIVE-INDEX axis (probe 1 has 1131 cols, probe 2 has 1400) "
        "because Calx is not in the .mat files. True cross-probe similarity requires aligned wavenumber axes; "
        "the resampled cosine should be treated as an ORDER-OF-MAGNITUDE proxy only.",
        "",
        f"## Physics difference category (working assumption)",
        f"**{physics['category']}**",
        f"- {physics['rationale']}",
        f"- Expected transferability ceiling: {physics['expected_transferability_ceiling']}",
        f"- Must confirm with paper: {physics['what_we_must_confirm_with_paper']}",
        "",
        "## Pre-analysis go/no-go for GAIRA",
        "- The dataset IS GAIRA-runnable in principle (large N spectra per cohort, normalized format).",
        "- TWO BLOCKERS before meaningful GAIRA interpretation:",
        "  1. **Calx (wavenumber axis)** must be obtained from the paper Drive (data_BC_NORM.mat) before MSS / BSV scoring",
        "     — without it, the find_peaks output indices cannot be mapped to Raman shifts and MSS templates cannot fire.",
        "  2. **Probe 1 / Probe 2 substrate physics** must be obtained from the paper Methods PDF before "
        "     applying substrate-aware wrappers — current GAIRA substrate physics rules know cAg / cAu / sAg / sAu "
        "     blocks; if these probes are different (e.g. silicon-supported nanostructures), inference must be GATED.",
        "- Once both are obtained, the natural GAIRA flow is:",
        "  (a) score per spectrum → 11-axis BSV (sumnorm + CLR) per probe per cohort",
        "  (b) cross-probe BSV-axis correlation per cohort (the true cross-probe transferability test)",
        "  (c) substrate-aware wrapper applied per probe (if substrate-physics rule available)",
        "  (d) cohort-mixture ratio response: does ΔBSV monotonically track HT:THP ratio?",
    ]
    (REPORTS / "REPORT_pre_gaira_hypothesis_v1.md").write_text("\n".join(text))


# ──────────────────────────────────────────────────────────────────────
# Audit log
# ──────────────────────────────────────────────────────────────────────
def write_audit():
    txt = [
        "# gaira_base_4_small_ev_dual_probe_audit_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Strict scope",
        "- This phase is a **PRE-ANALYSIS audit only**.",
        "- NO GAIRA scoring was performed.",
        "- NO classifier was built.",
        "- NO interpretation claims were made about probe physics.",
        "",
        "## Source dataset",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/small2023_ev/",
        "  - NormedProbe1.mat (struct `normed1`: c00 4687, c01 3598, c10 2349, c25 2279, c50 2000, c100 4644 spectra; 1131 wn columns)",
        "  - NormedProbe2.mat (struct `Normed`:  c00 14884, c01 11163, c10 14884, c25 14884, c50 14884, c100 14884 spectra; 1400 wn columns)",
        "  - Readme.docx (paper title + authors + reproduction guide)",
        "  - Main_Text.zip (MATLAB code + figure data; NO probe physics description in extracted .m files)",
        "",
        "## Strict negative invariants",
        "- NO engine, MSS, motif, BSV, preprocessing changes",
        "- NO scoring of any kind in this phase",
        "- NO hypothesis was tested with disease labels (none exist in this dataset anyway)",
        "",
        "## Outputs",
        "- tables/dataset_inventory_v1.csv",
        "- tables/probe1_vs_probe2_methods_table_v1.csv",
        "- tables/physics_difference_classification_v1.csv",
        "- tables/top10_peaks_per_cohort_per_probe_v1.csv",
        "- tables/same_cohort_cross_probe_similarity_v1.csv",
        "- tables/cross_probe_similarity_summary_v1.csv",
        "- tables/mean_spectrum_Probe[12]_c[00..100]_v1.npy (12 mean-spectrum vectors)",
        "- figures/fig_mean_spectra_per_probe_v1.png",
        "- figures/fig_per_probe_pca_v1.png",
        "- figures/fig_same_cohort_cross_probe_overlay_v1.png",
        "- reports/REPORT_pre_gaira_hypothesis_v1.md",
        "",
        "## Acquisition gaps that BLOCK downstream GAIRA",
        "1. Wavenumber axis (Calx) is NOT in NormedProbe[12].mat. Acquire from paper Drive `data_BC_NORM.mat` "
        "or paper Methods PDF. Both probes' Calx required (likely different).",
        "2. Probe 1 / Probe 2 physical chemistry is NOT in any local file. Acquire from paper Methods PDF "
        "(material / morphology / synthesis / laser / integration time / functionalization).",
        "3. Whether the SAME EV ALIQUOT was deposited on both probes is INFERRED but not confirmed locally.",
    ]
    (AUDIT / "gaira_base_4_small_ev_dual_probe_audit_v1_audit_log.md").write_text("\n".join(txt))


# ──────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("gaira_base_4_small_ev_dual_probe_audit_v1 — STRICT PRE-ANALYSIS AUDIT")
    print("=" * 78)

    print("[load] NormedProbe1.mat")
    _, p1 = load_probe(P1_PATH, ["normed1"])
    print("[load] NormedProbe2.mat")
    _, p2 = load_probe(P2_PATH, ["Normed", "normed2"])

    inv_df = step0_locate(p1, p2)
    methods_df = step1_paper_methods()
    physics = step2_physics_classification()
    means = step3_spectral_comparison(p1, p2)
    cross_df, cross_sum = step4_same_sample_cross_probe(p1, p2, means)
    step5_hypotheses(physics, cross_sum)
    write_audit()

    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print("[done] pre-analysis audit complete")


if __name__ == "__main__":
    main()
