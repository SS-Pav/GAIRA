"""BSV Validation — Parts 2-13 (READ-ONLY; drives calibration data through V6)."""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import bsv_val_lib as L
from gaira.foundation import dataset as DS

OUT = REPO / "results/v5_rebuild/bsv_validation"
TAB, ART = OUT / "tables", OUT / "artifacts"
for p in (TAB, ART): p.mkdir(parents=True, exist_ok=True)


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    t0 = time.time()
    H = L.Harness()
    bio = H.bio
    log(f"V6 engine loaded | {len(bio)} biochemical themes | atlas {H.eng.atlas.meta['fingerprint']}")

    # ── Part 2: complete inference for every calibration dataset ──
    log("Part 2 — full V6 inference per dataset …")
    frames = {}
    for name, dom in [("ils_adenine", "buffer"), ("ergothioneine", "buffer"),
                      ("uricase", "serum"), ("spiked_serum", "serum"),
                      ("serum_baseline", "serum"), ("pure_sers", "buffer")]:
        Z, meta = H.project_dataset(name)
        fr = H.bsv_frame(Z, meta, domain=dom)
        frames[name] = fr
        fr.to_csv(TAB / f"part2_bsv_{name}.csv", index=False)
        log(f"  {name}: {len(fr)} inferences | median OOD {fr.ood.median():.3f} | "
            f"median conf {fr.overall_confidence.median():.3f}")

    # pure Raman references through the engine (in-domain control)
    corpus = DS.load_reference_corpus()
    Zr = H.eng.atlas.coordinates(corpus.X, normalise=True)
    ref = H.bsv_frame(Zr, corpus.meta[["analyte", "source", "excitation_nm"]], domain="buffer")
    ref.to_csv(TAB / "part2_bsv_pure_raman.csv", index=False)
    log(f"  pure_raman (in-domain): {len(ref)} | median OOD {ref.ood.median():.3f}")

    # ── Part 3: monotonicity (dose-response) ──
    log("Part 3 — monotonicity …")
    mono_rows = []
    dose_defs = []
    fr = frames["ils_adenine"]
    for (sub, las), g in fr.groupby(["substrate", "laser_nm"]):
        if g.conc_uM.nunique() < 4:
            continue
        dose_defs.append((f"adenine::{sub}@{las}", g, "conc_uM", "nucleic_purine"))
    dose_defs.append(("ergothioneine", frames["ergothioneine"], "conc_uM", "sulfur_antioxidant"))
    for exp, g, lvl, target in dose_defs:
        m = L.monotonicity(g[lvl], g[f"theme_{target}"])
        if m.get("insufficient"):
            continue
        m["permutation_p"] = L.permutation_p(g[lvl].values, g[f"theme_{target}"].values, n=500)
        m.update({"experiment": exp, "target_theme": target,
                  "ood_median": float(g.ood.median()), "conf_median": float(g.overall_confidence.median())})
        mono_rows.append(m)
    mono = pd.DataFrame(mono_rows)
    mono.to_csv(TAB / "part3_monotonicity.csv", index=False)
    log(f"  {len(mono)} dose series | target spearman median {mono.spearman.median():.3f}")

    # ── Part 4/11: theme cross-talk + orthogonality ──
    log("Part 4/11 — theme specificity, cross-talk, orthogonality …")
    ct_rows = []
    for exp, g, lvl, target in dose_defs:
        row = L.crosstalk_row(g, lvl, bio)
        row = {"experiment": exp, "target": target} | {f"rho_{t}": row[t] for t in bio}
        ct_rows.append(row)
    ct = pd.DataFrame(ct_rows)
    ct.to_csv(TAB / "part4_crosstalk.csv", index=False)
    # specificity: |target rho| minus mean |off-target rho|
    spec_rows = []
    for _, r in ct.iterrows():
        tgt = abs(r[f"rho_{r.target}"])
        off = np.mean([abs(r[f"rho_{t}"]) for t in bio if t != r.target])
        spec_rows.append({"experiment": r.experiment, "target": r.target,
                          "target_rho_abs": round(tgt, 3), "mean_offtarget_rho_abs": round(off, 3),
                          "specificity_margin": round(tgt - off, 3),
                          "leakage_ratio": round(off / (tgt + 1e-9), 3)})
    pd.DataFrame(spec_rows).to_csv(TAB / "part4_specificity.csv", index=False)

    # theme orthogonality on the pure-Raman reference cloud (the intrinsic geometry)
    Tref = ref[[f"theme_{t}" for t in bio]].values
    corr = np.corrcoef(Tref.T)
    cov = np.cov(Tref.T)
    pd.DataFrame(corr, index=bio, columns=bio).to_csv(TAB / "part11_theme_correlation.csv")
    # mutual information between themes (discretised)
    from sklearn.metrics import mutual_info_score
    def disc(x): return pd.qcut(pd.Series(x).rank(method="first"), 5, labels=False)
    mi = np.zeros((len(bio), len(bio)))
    for i in range(len(bio)):
        for j in range(len(bio)):
            mi[i, j] = mutual_info_score(disc(Tref[:, i]), disc(Tref[:, j]))
    pd.DataFrame(mi, index=bio, columns=bio).to_csv(TAB / "part11_theme_mutual_information.csv")
    off_corr = corr[~np.eye(len(bio), dtype=bool)]
    log(f"  theme correlation off-diagonal: mean |r| {np.abs(off_corr).mean():.3f}, "
        f"max |r| {np.abs(off_corr).max():.3f}")

    # ── Part 5: component contributions + turnover ──
    log("Part 5 — component contributions and turnover …")
    comp_rows = []
    for exp, g, lvl, target in dose_defs:
        keys = np.array(sorted(g[lvl].unique()))
        coords = np.vstack([g[g[lvl] == k][[f"coord_c{j}" for j in range(L.K)]].mean().values for k in keys])
        # turnover: change in the top-5 component set from lowest to highest dose
        r0 = set(np.argsort(-coords[0])[:5]); r1 = set(np.argsort(-coords[-1])[:5])
        turnover = 1 - len(r0 & r1) / 5
        # scaling vs redistribution: correlation of the coordinate profile low vs high
        prof_corr = np.corrcoef(coords[0], coords[-1])[0, 1]
        driver = int(np.argmax(np.abs(coords[-1] - coords[0])))
        comp_rows.append({"experiment": exp, "target_theme": target,
                          "component_turnover_low_to_high": round(float(turnover), 3),
                          "profile_corr_low_high": round(float(prof_corr), 3),
                          "dominant_driver_component": driver,
                          "mode": "scaling (same components)" if prof_corr > 0.8
                                  else "redistribution (different components)"})
    pd.DataFrame(comp_rows).to_csv(TAB / "part5_component_contributions.csv", index=False)

    # ── Part 6/7: trajectories + inter-analyte geometry ──
    log("Part 6/7 — trajectories and inter-analyte geometry …")
    # per-analyte BSV centroid (biochemical themes) for geometry
    anchors = {}
    # pure references for the requested analytes
    for a in ["adenine", "ergothioneine", "glucose", "(+)-glucose", "cholesterol",
              "hypoxanthine", "xanthine", "guanine", "urate", "phenylalanine", "albumin"]:
        m = ref.analyte == a
        if m.any():
            anchors[a] = ref[m][[f"theme_{t}" for t in bio]].mean().values
    names = list(anchors); M = np.vstack([anchors[a] for a in names])
    U = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    C = U @ U.T
    pd.DataFrame(C, index=names, columns=names).round(3).to_csv(TAB / "part7_analyte_cosine.csv")
    D = np.linalg.norm(M[:, None] - M[None, :], axis=2)
    pd.DataFrame(D, index=names, columns=names).round(3).to_csv(TAB / "part7_analyte_euclidean.csv")
    # nearest neighbour per analyte
    nn = []
    for i, a in enumerate(names):
        order = np.argsort(-C[i]); order = [j for j in order if j != i]
        nn.append({"analyte": a, "nearest": names[order[0]], "cosine": round(float(C[i, order[0]]), 3),
                   "second": names[order[1]] if len(order) > 1 else None})
    pd.DataFrame(nn).to_csv(TAB / "part7_nearest_neighbours.csv", index=False)

    # trajectory geometry (dose series through biochemical-theme space)
    traj_rows = {}
    for exp, g, lvl, target in dose_defs:
        keys = np.array(sorted(g[lvl].unique()))
        T = np.vstack([g[g[lvl] == k][[f"theme_{t}" for t in bio]].mean().values for k in keys])
        steps = np.diff(T, axis=0); sn = np.linalg.norm(steps, axis=1)
        net = T[-1] - T[0]
        curv = [np.degrees(np.arccos(np.clip(np.dot(steps[i], steps[i+1]) /
                (np.linalg.norm(steps[i])*np.linalg.norm(steps[i+1])+1e-12), -1, 1)))
                for i in range(len(steps)-1)]
        traj_rows[exp] = {"path_length": float(sn.sum()), "net_displacement": float(np.linalg.norm(net)),
                          "straightness": float(np.linalg.norm(net)/(sn.sum()+1e-12)),
                          "mean_curvature_deg": float(np.mean(curv)) if curv else None,
                          "dominant_theme_shift": bio[int(np.argmax(np.abs(net)))]}
    (ART / "part6_trajectories.json").write_text(json.dumps(traj_rows, indent=2, default=float))

    # ── Part 8: replicate stability ──
    log("Part 8 — replicate stability …")
    stab_rows = []
    fr = frames["ils_adenine"]
    for (sub, las), g in fr.groupby(["substrate", "laser_nm"]):
        groups = [gg[f"theme_nucleic_purine"].values for _, gg in g.groupby("conc_uM") if len(gg) >= 2]
        if len(groups) < 2:
            continue
        within = np.median([L.cv(x) for x in groups if len(x) >= 2])
        stab_rows.append({"experiment": f"adenine::{sub}@{las}", "target": "nucleic_purine",
                          "icc_purine": round(L.icc(groups), 3),
                          "within_dose_cv_median": round(float(within), 3),
                          "ood_cv": round(L.cv(g.ood.values), 3),
                          "conf_cv": round(L.cv(g.overall_confidence.values), 3)})
    pd.DataFrame(stab_rows).to_csv(TAB / "part8_replicate_stability.csv", index=False)

    # ── Part 9: confidence system ──
    log("Part 9 — confidence system …")
    # confidence vs known reliability: pure Raman (clean) vs SERS (OOD) vs weak adsorbers
    conf_rows = []
    conf_rows.append({"group": "pure_raman_reference", "n": len(ref),
                      "median_ood": round(float(ref.ood.median()), 3),
                      "median_confidence": round(float(ref.overall_confidence.median()), 3)})
    for name in ("pure_sers", "spiked_serum", "serum_baseline"):
        f = frames[name]
        conf_rows.append({"group": name, "n": len(f),
                          "median_ood": round(float(f.ood.median()), 3),
                          "median_confidence": round(float(f.overall_confidence.median()), 3)})
    # strong vs weak Ag adsorbers among serum spikes
    sp7 = pd.read_csv(REPO / "results/v5_rebuild/spike_validation/tables/phase7_serum_vs_pure.csv")
    strong = set(sp7.nlargest(6, "cos_spike_vs_pureSERS").analyte)
    fs = frames["spiked_serum"]
    fs_strong = fs[fs.analyte.isin(strong)]; fs_weak = fs[~fs.analyte.isin(strong)]
    conf_rows.append({"group": "serum_spike_STRONG_adsorbers", "n": len(fs_strong),
                      "median_ood": round(float(fs_strong.ood.median()), 3),
                      "median_confidence": round(float(fs_strong.overall_confidence.median()), 3)})
    conf_rows.append({"group": "serum_spike_WEAK_adsorbers", "n": len(fs_weak),
                      "median_ood": round(float(fs_weak.ood.median()), 3),
                      "median_confidence": round(float(fs_weak.overall_confidence.median()), 3)})
    pd.DataFrame(conf_rows).to_csv(TAB / "part9_confidence_system.csv", index=False)
    # correlation: does confidence track cleanliness (inverse OOD)?
    allf = pd.concat([ref.assign(grp="ref")] + [frames[n].assign(grp=n) for n in
                     ("pure_sers", "spiked_serum")], ignore_index=True)
    from scipy.stats import spearmanr
    r_conf_ood = float(spearmanr(allf.overall_confidence, allf.ood)[0])
    log(f"  confidence vs OOD spearman: {r_conf_ood:+.3f} (should be negative)")

    # ── Part 12: BSV state-space geometry ──
    log("Part 12 — BSV state-space geometry …")
    from sklearn.decomposition import PCA
    Tall = ref[[f"theme_{t}" for t in bio]].values
    pca = PCA().fit(Tall - Tall.mean(0))
    evr = pca.explained_variance_ratio_
    eff_dim = float(np.exp(-np.sum(evr * np.log(evr + 1e-12))))   # participation entropy
    n90 = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
    geom = {"n_biochemical_themes": len(bio),
            "effective_dimensionality_entropy": round(eff_dim, 2),
            "n_components_90pct_variance": n90,
            "explained_variance_ratio": [round(float(x), 3) for x in evr],
            "theme_correlation_mean_abs_offdiag": round(float(np.abs(off_corr).mean()), 3),
            "theme_correlation_max_abs_offdiag": round(float(np.abs(off_corr).max()), 3)}
    (ART / "part12_state_space.json").write_text(json.dumps(geom, indent=2))
    log(f"  effective dimensionality {eff_dim:.1f} of {len(bio)} themes; 90% var by {n90}")

    # ── manifest / verdict ──
    summary = {
        "part1_implementation_verified": True,
        "atlas_fingerprint": H.eng.atlas.meta["fingerprint"],
        "monotonicity_target_spearman_median": float(mono.spearman.median()),
        "specificity_margin_median": float(pd.DataFrame(spec_rows).specificity_margin.median()),
        "theme_orthogonality_mean_abs_r": round(float(np.abs(off_corr).mean()), 3),
        "effective_dimensionality": geom["effective_dimensionality_entropy"],
        "confidence_tracks_ood": r_conf_ood,
        "confidence_strong_vs_weak_adsorbers": {
            "strong": conf_rows[-2]["median_confidence"], "weak": conf_rows[-1]["median_confidence"]},
        "runtime_s": round(time.time() - t0, 1),
    }
    (ART / "validation_manifest.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"\nVERDICT: monotonicity {summary['monotonicity_target_spearman_median']:.2f}, "
        f"specificity margin {summary['specificity_margin_median']:.2f}, "
        f"theme mean|r| {summary['theme_orthogonality_mean_abs_r']}, "
        f"eff-dim {summary['effective_dimensionality']}")
    log(f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
