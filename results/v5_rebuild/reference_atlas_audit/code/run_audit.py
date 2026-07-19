"""GAIRA Raman Reference Atlas v0.1 — audit driver (READ-ONLY).

Loads the FROZEN NMF k=24 atlas and audits it. The atlas is never refitted;
its fingerprint is verified on load and re-verified at the end.
"""
from __future__ import annotations
import sys, json, time, warnings, zipfile, tempfile, re
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from gaira.foundation import dataset as DS, serialization as SER
from gaira.foundation.families_raman import family_of
from gaira.data import loader
from gaira.preprocessing import pipeline as pp
import atlas_audit as AA

OUT = REPO / "results/v5_rebuild/reference_atlas_audit"
TAB, ART, FIG = OUT / "tables", OUT / "artifacts", OUT / "figures"
for p in (TAB, ART, FIG): p.mkdir(parents=True, exist_ok=True)
FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ───────── P13 out-of-domain loaders (Ag-SERS — stress test only) ─────────
def load_ood():
    """Ag-SERS calibration sets projected as a STRESS TEST. Out of domain."""
    grid, win = DS.GRID, DS.WINDOW
    sets = {}

    # adenine Ag-SERS concentration series
    try:
        conc_map = {"10pg": 1e-11, "100pg": 1e-10, "10nano": 1e-8, "100nano": 1e-7,
                    "1micro": 1e-6, "10micro": 1e-5}
        rows, recs = [], []
        for s in loader.load_adenine():
            v = pp.preprocess(s.wavenumber, s.intensity, DS.PREPROC, grid, win)
            if np.isfinite(v).any():
                rows.append(v)
                recs.append({"dataset": "adenine_AgSERS_series", "label": s.record.concentration,
                             "conc_M": conc_map.get(str(s.record.concentration), np.nan)})
        if rows:
            sets["adenine_series"] = (np.vstack(rows), pd.DataFrame(recs))
    except Exception as e:
        log(f"  adenine OOD unavailable: {e}")

    # ergothioneine Ag-SERS calibration
    try:
        p = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/ergothioneine_serum/ERG_calibration.csv")
        if p.exists():
            df = pd.read_csv(p)
            wn = np.array([float(c) for c in df.columns[4:]])
            keep = wn > 0
            rows, recs = [], []
            for _, r in df.iterrows():
                y = r.values[4:].astype(float)
                v = pp.preprocess(wn[keep], y[keep], DS.PREPROC, grid, win)
                if np.isfinite(v).any():
                    rows.append(v)
                    recs.append({"dataset": "ergothioneine_AgSERS_calibration",
                                 "label": str(r.get("c", "")), "conc_M": float(r.get("c", np.nan))})
            if rows:
                sets["ergothioneine_calibration"] = (np.vstack(rows), pd.DataFrame(recs))
    except Exception as e:
        log(f"  ergothioneine OOD unavailable: {e}")

    # uricase depletion (serum spiked +/- enzyme) — Ag-SERS
    try:
        zp = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/serum_ag_colloids/dataset_spectral_data.zip")
        if zp.exists():
            cache = Path(tempfile.gettempdir()) / "gaira_atlas_audit_uricase"
            if not cache.exists():
                with zipfile.ZipFile(zp) as z:
                    for n in z.namelist():
                        if n.startswith("dataset uricase/") and n.endswith(".txt"):
                            z.extract(n, cache)
            rows, recs = [], []
            for f in sorted((cache / "dataset uricase").glob("*.txt")):
                nm = f.name
                grp = ("serum_spiked+uricase" if "Enzyme" in nm else
                       "serum_spiked" if "Serumspiked" in nm else "serum_reference")
                txt = f.read_text(encoding="cp1252", errors="replace").splitlines()
                hdr = next((i for i, l in enumerate(txt)
                            if "Raman Shift" in l and "Dark Subtracted" in l), None)
                if hdr is None:
                    continue
                cols = [c.strip() for c in txt[hdr].split(";")]
                iw = cols.index("Raman Shift")
                iy = next(i for i, c in enumerate(cols) if c.startswith("Dark Subtracted"))
                wn, y = [], []
                for l in txt[hdr + 1:]:
                    p2 = l.split(";")
                    if len(p2) <= max(iw, iy):
                        continue
                    try:
                        wn.append(float(p2[iw].replace(",", "."))); y.append(float(p2[iy].replace(",", ".")))
                    except ValueError:
                        continue
                if len(wn) < 100:
                    continue
                o = np.argsort(wn)
                v = pp.preprocess(np.array(wn)[o], np.array(y)[o], DS.PREPROC, grid, win)
                if np.isfinite(v).any():
                    rows.append(v); recs.append({"dataset": "uricase_AgSERS", "label": grp, "conc_M": np.nan})
            if rows:
                sets["uricase_depletion"] = (np.vstack(rows), pd.DataFrame(recs))
    except Exception as e:
        log(f"  uricase OOD unavailable: {e}")
    return sets


def main():
    t0 = time.time()
    atlas = SER.load_frozen_manifold(FROZEN)
    log(f"FROZEN atlas loaded: {atlas.name} k={atlas.k} fingerprint {atlas.meta['fingerprint']}")
    corpus = DS.load_reference_corpus()
    W, grid = atlas.components, atlas.grid
    Z = atlas.coordinates(corpus.X, normalise=True)
    A = AA.analyte_activation(Z, corpus.meta)                  # analytes x components
    Xa = pd.DataFrame(np.nan_to_num(corpus.X)).assign(a=corpus.meta.analyte.values) \
        .groupby("a").mean().loc[A.index].values
    peaks_df = DS.load_peak_assignments()
    stats = atlas.meta["stats"]
    stab = np.array(stats["bootstrap_component_stability"]["per_component"])
    pcv = np.array(stats["per_component_variance"])
    log(f"corpus {corpus.X.shape} | analyte-level atlas {A.shape}")

    # ── P5 spectral ──
    bands = AA.component_bands(W, grid)
    uniq = AA.band_uniqueness(bands)

    # ── P1 inventory + P2 composition + P4 coherence ──
    inv_rows, comp_all, coh_rows, spec_rows = [], [], [], []
    for j in range(atlas.k):
        comp = AA.composition(A, j)
        comp_all.append(comp)
        coh = AA.coherence(A, j, Xa)
        coh_rows.append(coh)
        b = bands[j]
        lit = AA.match_literature(b.position.values, peaks_df)
        lit_groups = pd.Series([h["group"] for h in lit]).value_counts().to_dict() if lit else {}
        v = A.values[:, j]; tot = v.sum() + 1e-12
        dom = comp[comp.contribution_pct >= 5.0]
        prim = coh["dominant_family"]
        sec = (comp.chemical_family.value_counts().index[1]
               if comp.chemical_family.nunique() > 1 else "unavailable")
        # confidence: purity + enrichment + stability + literature corroboration
        score = (0.4 * min(coh["class_purity"] / 0.6, 1) + 0.25 * min(coh["enrichment_vs_corpus"] / 3, 1)
                 + 0.25 * min(stab[j] / 0.85, 1) + 0.10 * (1.0 if lit_groups else 0.0))
        conf = "high" if score >= 0.75 else ("moderate" if score >= 0.55 else "low")
        inv_rows.append({
            "component": j, "variance_explained": float(pcv[j]),
            "sparsity_gini": AA.gini(W[j]), "bootstrap_stability": float(stab[j]),
            "n_analytes_contributing": int((v > 1e-4 * tot).sum()),
            "n_dominant_analytes": int(len(dom)),
            "median_loading": float(np.median(v[v > 0])) if (v > 0).any() else 0.0,
            "entropy_analyte": coh["shannon_entropy_analyte"],
            "representative_spectrum": comp.analyte.iloc[0] if len(comp) else "n/a",
            "dominant_raman_peaks_cm": [round(float(x), 0) for x in b.position.values[:8]],
            "spectral_bandwidth_cm": float(b.width_cm.median()) if len(b) else np.nan,
            "band_uniqueness": float(uniq[j]),
            "primary_interpretation": prim, "secondary_interpretation": sec,
            "dominant_class": coh["dominant_class"],
            "class_purity": coh["class_purity"], "enrichment": coh["enrichment_vs_corpus"],
            "literature_groups": lit_groups, "confidence": conf, "confidence_score": round(score, 3),
            "comments": ("chemically focused" if coh["class_purity"] >= 0.5 else
                         "mixed composition — treat as a mathematical mixture unless "
                         "band evidence supports a shared chemical motif"),
        })
        for _, r in b.iterrows():
            spec_rows.append({"component": j, "position_cm": float(r.position),
                              "prominence": float(r.prominence), "width_cm": float(r.width_cm),
                              "importance": float(r.importance), "uniqueness": float(uniq[j]),
                              "literature_groups": json.dumps(lit_groups)})

    inv = pd.DataFrame(inv_rows)
    inv.to_csv(TAB / "p1_component_inventory.csv", index=False)
    pd.concat(comp_all, ignore_index=True).to_csv(TAB / "p2_full_analyte_composition.csv", index=False)
    coh_df = pd.DataFrame(coh_rows)
    coh_df["coherence_rank"] = coh_df.class_purity.rank(ascending=False).astype(int)
    coh_df.sort_values("class_purity", ascending=False).to_csv(TAB / "p4_chemical_coherence.csv", index=False)
    pd.DataFrame(spec_rows).to_csv(TAB / "p5_spectral_interpretation.csv", index=False)
    log(f"P1/P2/P4/P5 done | mean class purity {coh_df.class_purity.mean():.3f}")

    # ── P6 relationships ──
    rel = AA.relationships(A, W, bands)
    np.savez(ART / "p6_relationship_matrices.npz", **rel)
    pd.DataFrame(rel["spectral_cosine"]).to_csv(TAB / "p6_spectral_cosine.csv", index=False)
    pd.DataFrame(rel["activation_corr"]).to_csv(TAB / "p6_activation_correlation.csv", index=False)

    # ── P7 grouping study ──
    gdf, assignments, Zl, D = AA.grouping_study(A, W, rel, Xa)
    gdf.to_csv(TAB / "p7_grouping_study.csv", index=False)
    best_k = int(gdf.iloc[0].n_groups)
    np.savez(ART / "p7_grouping.npz", linkage=Zl, distance=D,
             **{f"labels_k{k}": v for k, v in assignments.items()})
    log(f"P7 grouping: best composite k={best_k} "
        f"(sil {gdf.iloc[0].silhouette:.3f}, boot {gdf.iloc[0].bootstrap_reproducibility:.3f}, "
        f"chem {gdf.iloc[0].chemical_coherence:.3f})")

    # group composition at the recommended k
    lab = assignments[best_k]
    grp_rows = []
    for g in sorted(set(lab)):
        members = [j for j in range(atlas.k) if lab[j] == g]
        act = A.values[:, members].sum(axis=1)
        fam = pd.Series(act, index=[family_of(a) for a in A.index]).groupby(level=0).sum()
        fam = (fam / fam.sum()).sort_values(ascending=False)
        top = [A.index[i] for i in np.argsort(-act)[:8]]
        grp_rows.append({"group": int(g), "n_components": len(members), "components": members,
                         "dominant_family": fam.index[0], "dominant_fraction": round(float(fam.iloc[0]), 3),
                         "second_family": fam.index[1] if len(fam) > 1 else "n/a",
                         "top_analytes": top,
                         "share_of_atlas": round(float(act.sum() / A.values.sum()), 3)})
    pd.DataFrame(grp_rows).to_csv(TAB / f"p7_group_composition_k{best_k}.csv", index=False)

    # ── P9 biological plausibility ──
    plaus = {}
    for fam_name in ("triglyceride", "fatty_acid", "sterol", "saccharide", "polysaccharide",
                     "protein", "amino_acid", "purine", "pyrimidine", "cofactor", "organic_acid"):
        mem = [a for a in A.index if family_of(a) == fam_name]
        if len(mem) < 2:
            continue
        idx = [list(A.index).index(m) for m in mem]
        sub = A.values[idx] / (A.values[idx].sum(axis=1, keepdims=True) + 1e-12)
        U = AA._unit(sub); S = U @ U.T
        iu = np.triu_indices(len(idx), 1)
        # null: random analytes of the same count
        rng = np.random.default_rng(0)
        nulls = []
        for _ in range(200):
            r = rng.choice(len(A.index), len(idx), replace=False)
            sr = A.values[r] / (A.values[r].sum(axis=1, keepdims=True) + 1e-12)
            Ur = AA._unit(sr); Sr = Ur @ Ur.T
            nulls.append(float(Sr[iu].mean()))
        obs = float(S[iu].mean())
        dom = pd.Series(sub.sum(axis=0)).idxmax()
        plaus[fam_name] = {"n_analytes": len(idx), "within_family_bsv_cos": round(obs, 3),
                           "null_mean": round(float(np.mean(nulls)), 3),
                           "p_value": round(float((np.sum(np.array(nulls) >= obs) + 1) / 201), 4),
                           "dominant_component": int(dom),
                           "coheres": bool(obs > np.percentile(nulls, 95))}
    (TAB / "p9_biological_plausibility.json").write_text(json.dumps(plaus, indent=2))
    log("P9 family coherence: " + ", ".join(f"{k}={'Y' if v['coheres'] else 'n'}" for k, v in plaus.items()))

    # ── P12 MSS readiness ──
    mss = AA.mss_readiness(A)
    mss.to_csv(TAB / "p12_mss_readiness.csv", index=False)
    log(f"P12 MSS: median uniqueness {mss.signature_uniqueness.median():.3f} | "
        f"confidence {mss.assignment_confidence.value_counts().to_dict()}")

    # ── P13 out-of-domain stress test ──
    log("P13 OUT-OF-DOMAIN stress test (Ag-SERS projected into a Raman atlas) …")
    ood_sets = load_ood()
    ood_rows = []
    for name, (Xo, mo) in ood_sets.items():
        Zo = atlas.coordinates(Xo, normalise=True)
        # OOD distance to the reference support
        U = AA._unit(np.clip(Z, 0, None)); Uo = AA._unit(np.clip(Zo, 0, None))
        S = Uo @ U.T
        for i in range(len(Zo)):
            top = np.argsort(-S[i])[:3]
            ood_rows.append({"dataset": name, "label": mo.label.iloc[i],
                             "conc_M": mo.conc_M.iloc[i],
                             "ood_distance": float(1 - np.sort(S[i])[-5:].mean()),
                             "max_sim_to_reference": float(S[i].max()),
                             "nearest_reference_analytes": [A.index[t] for t in
                                                            np.argsort(-(Zo[i] @ AA._unit(A.values).T))[:3]],
                             "dominant_components": [int(c) for c in np.argsort(-Zo[i])[:3]],
                             "top_component_share": float(np.sort(Zo[i])[::-1][:3].sum())})
        log(f"  {name}: n={len(Zo)} median OOD {np.median([r['ood_distance'] for r in ood_rows if r['dataset']==name]):.3f}")
    if ood_rows:
        pd.DataFrame(ood_rows).to_csv(TAB / "p13_out_of_domain_stress_test.csv", index=False)

    # ── verify the atlas was never altered ──
    import hashlib
    fp_now = hashlib.sha256(np.ascontiguousarray(atlas.components).tobytes()).hexdigest()[:32]
    assert fp_now == atlas.meta["fingerprint"], "ATLAS MUTATED — audit invalid"

    (ART / "audit_manifest.json").write_text(json.dumps({
        "atlas": {"representation": atlas.name, "k": atlas.k,
                  "fingerprint": atlas.meta["fingerprint"], "verified_unchanged": True},
        "corpus": {"n_spectra": int(corpus.X.shape[0]), "n_analytes": int(A.shape[0])},
        "grouping_recommendation_k": best_k,
        "grouping_ranking": gdf.to_dict("records"),
        "mean_class_purity": float(coh_df.class_purity.mean()),
        "family_coherence": plaus,
        "out_of_domain_sets": {k: int(len(v[0])) for k, v in ood_sets.items()},
        "runtime_s": round(time.time() - t0, 1),
    }, indent=2, default=str))
    log(f"atlas fingerprint re-verified unchanged | runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
