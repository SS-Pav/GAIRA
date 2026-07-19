"""Perturbation Response Audit — main driver (READ-ONLY on the frozen atlas)."""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import response_lib as RL

OUT = REPO / "results/v5_rebuild/perturbation_response"
TAB, ART = OUT / "tables", OUT / "artifacts"
for p in (TAB, ART): p.mkdir(parents=True, exist_ok=True)
K = RL.K


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    t0 = time.time()
    ctx = RL.load_atlas_context()
    themes, tclass, tconf, stab = ctx["themes"], ctx["theme_class"], ctx["theme_conf"], ctx["stability"]
    log(f"frozen atlas fingerprint {ctx['fingerprint']} | 24 components, 12 themes")

    proj = {d: RL.load_projection(d) for d in RL.DATASETS}
    proj = {k: v for k, v in proj.items() if v is not None}

    # ── Part 1: component dose-response for each dose experiment ──
    log("Part 1 — component dose-response …")
    dose_rows = []
    dose_experiments = {}
    Z, m, _ = proj["ils_adenine"]
    for (sub, las), idx in m.groupby(["substrate", "laser_nm"]).groups.items():
        pos = [m.index.get_loc(i) for i in idx]
        concs = m.iloc[pos].conc_uM.values
        if len(np.unique(concs)) < 4:
            continue
        exp = f"adenine::{sub}@{las}nm"
        rc = RL.responsive_components(Z[pos], concs)
        rc["experiment"] = exp
        dose_rows.append(rc); dose_experiments[exp] = (Z[pos], concs)
    Ze, me, _ = proj["ergothioneine"]
    rc = RL.responsive_components(Ze, me.conc_uM.values); rc["experiment"] = "ergothioneine"
    dose_rows.append(rc); dose_experiments["ergothioneine"] = (Ze, me.conc_uM.values)
    dose = pd.concat(dose_rows, ignore_index=True)
    dose.to_csv(TAB / "part1_component_dose_response.csv", index=False)
    resp_counts = dose[dose.responsive].groupby("experiment").component.apply(list).to_dict()
    log("  responsive components per experiment: " +
        ", ".join(f"{k}={v}" for k, v in resp_counts.items()))

    # ── Part 2: response fingerprints (per analyte, from serum spikes) ──
    log("Part 2 — response fingerprints …")
    Zs, ms, _ = proj["spiked_serum"]
    Zb, mb, _ = proj["serum_baseline"]
    fps = {}
    fp_rows = []
    for a, idx in ms.groupby("analyte").groups.items():
        pos = [ms.index.get_loc(i) for i in idx]
        fp = RL.response_fingerprint(Zs[pos], Zb)
        fps[a] = fp
        fp_rows.append({"analyte": a, "family": RL.analyte_family(a),
                        "n_significant_components": fp["n_significant"],
                        "response_entropy": fp["entropy"],
                        "top_up": fp["top_up"], "top_down": fp["top_down"],
                        **{f"d{j}": float(fp["delta"][j]) for j in range(K)}})
    fpdf = pd.DataFrame(fp_rows)
    fpdf.to_csv(TAB / "part2_response_fingerprints.csv", index=False)
    # distinctness: mean pairwise cosine of fingerprints
    F = np.vstack([fps[a]["delta"] for a in fps])
    U = RL._unit(F); C = U @ U.T; iu = np.triu_indices(len(U), 1)
    log(f"  {len(fps)} fingerprints | mean pairwise cosine {C[iu].mean():.3f} "
        f"(low = distinct) | median significant components {fpdf.n_significant_components.median():.0f}")

    # ── Part 3 + Part 4: specificity and theme-match ──
    log("Part 3/4 — component specificity and theme match …")
    spec = RL.component_specificity(fps, themes, tclass)
    spec.to_csv(TAB / "part4_component_specificity.csv", index=False)

    # theme match: does a component's dose-response match its audit theme AND, more
    # tellingly, does the perturbed analyte actually load the responding component
    # in the reference atlas (a label-independent identity test)?
    comp_load = RL.load_component_reference_loadings()
    match_rows = []
    for exp, (Zx, cx) in dose_experiments.items():
        analyte = "adenine" if "adenine" in exp else "ergothioneine"
        rc = dose[dose.experiment == exp]
        # rank responsive components by |effect| and record the single strongest riser
        rr = rc[rc.responsive].reindex(rc[rc.responsive].effect_size.abs().sort_values(ascending=False).index)
        for rank, (_, r) in enumerate(rr.iterrows()):
            j = int(r.component)
            match_rows.append({"experiment": exp, "analyte": analyte, "component": j,
                               "effect_rank": rank, "theme": themes.get(j), "theme_class": tclass.get(j),
                               "theme_confidence": tconf.get(j), "rho": r.spearman_rho,
                               "effect_size": r.effect_size, "direction": r.direction,
                               "theme_matches_analyte": _theme_matches(analyte, themes.get(j)),
                               "analyte_loads_this_component": RL.component_encodes(analyte, comp_load, j)})
    tm = pd.DataFrame(match_rows)
    tm.to_csv(TAB / "part3_theme_match.csv", index=False)
    if len(tm):
        # does the analyte's OWN encoding component appear among its top responders?
        own = tm[tm.analyte_loads_this_component]
        own_top3 = own[own.effect_rank < 3]
        log(f"  theme-label matches {int(tm.theme_matches_analyte.sum())}/{len(tm)}; "
            f"identity matches (analyte loads the responding component) "
            f"{int(tm.analyte_loads_this_component.sum())}/{len(tm)}; "
            f"of which in the top-3 strongest responders: {len(own_top3)}")

    # ── Part 5: analyte consistency (pure vs spike) ──
    log("Part 5 — analyte consistency pure vs spike …")
    Zp, mp, _ = proj["pure_sers"]
    pure_centroid = np.nan_to_num(Zp).mean(0)
    cons_rows = []
    for a in fps:
        pidx = mp.index[mp.analyte == a]
        if not len(pidx):
            continue
        ppos = [mp.index.get_loc(i) for i in pidx]
        pure_fp = np.nan_to_num(Zp[ppos]).mean(0) - pure_centroid
        spike_fp = fps[a]["delta"]
        cons_rows.append({"analyte": a, "family": RL.analyte_family(a),
                          "consistency_cosine": RL.cos(spike_fp, pure_fp),
                          "spike_norm": float(np.linalg.norm(spike_fp)),
                          "pure_norm": float(np.linalg.norm(pure_fp))})
    cons = pd.DataFrame(cons_rows)
    cons.to_csv(TAB / "part5_analyte_consistency.csv", index=False)
    log(f"  consistency cosine: median {cons.consistency_cosine.median():.3f}")

    # ── Part 6: purine case study ──
    log("Part 6 — purine case study …")
    purines = ["adenine", "hypoxanthine", "xanthine", "guanine", "urate"]
    pur = {a: fps[a]["delta"] for a in purines if a in fps}
    pur_mat = np.vstack(list(pur.values())) if pur else np.zeros((0, K))
    Up = RL._unit(pur_mat)
    purine_sim = pd.DataFrame(Up @ Up.T, index=list(pur), columns=list(pur))
    purine_sim.to_csv(TAB / "part6_purine_similarity.csv")
    purine_summary = {a: {"top_up": fps[a]["top_up"], "top_down": fps[a]["top_down"],
                          "activates_c15_purine": float(fps[a]["delta"][15]),
                          "n_significant": fps[a]["n_significant"]}
                      for a in pur}
    (TAB / "part6_purine_summary.json").write_text(json.dumps(purine_summary, indent=2, default=float))
    log(f"  purine fingerprint mean pairwise cos: "
        f"{purine_sim.values[np.triu_indices(len(pur),1)].mean():.3f}" if len(pur) > 1 else "  n<2")

    # ── Part 7: ergothioneine ──
    log("Part 7 — ergothioneine case study …")
    erg_rc = dose[dose.experiment == "ergothioneine"].sort_values("effect_size", key=abs, ascending=False)
    erg = {"responsive_components": erg_rc[erg_rc.responsive][["component", "spearman_rho",
                                                              "effect_size", "direction"]].to_dict("records"),
           "top_by_effect": erg_rc.head(5)[["component", "spearman_rho", "effect_size",
                                            "saturating_r2" if "saturating_r2" in erg_rc else "linear_r2"]].to_dict("records"),
           "serum_spike_top_up": fps.get("ergothioneine", {}).get("top_up") if "ergothioneine" in fps else None}
    (TAB / "part7_ergothioneine.json").write_text(json.dumps(erg, indent=2, default=float))

    # ── Part 8: uricase depletion ──
    log("Part 8 — uricase depletion …")
    Zu, mu, _ = proj["uricase"]
    g = {c: np.nan_to_num(Zu[[mu.index.get_loc(i) for i in idx]]).mean(0)
         for c, idx in mu.groupby("condition").groups.items()}
    uricase = {}
    if "spiked" in g and "spiked+uricase" in g:
        d = g["spiked+uricase"] - g["spiked"]
        uricase = {"delta_components": {int(j): float(d[j]) for j in range(K)},
                   "components_decreased": [int(j) for j in np.argsort(d)[:5]],
                   "components_increased": [int(j) for j in np.argsort(-d)[:5]],
                   "purine_component_c15_change": float(d[15]),
                   "pyrimidine_c17_change": float(d[17]),
                   "global_change_norm": float(np.linalg.norm(d)),
                   "selective": bool(abs(d[15]) > np.median(np.abs(d)) * 2)}
    (TAB / "part8_uricase.json").write_text(json.dumps(uricase, indent=2, default=float))
    if uricase:
        log(f"  uricase: c15(purine) change {uricase['purine_component_c15_change']:+.4f}, "
            f"selective={uricase['selective']}")

    # ── Part 9: serum spikes — responders vs non-responders ──
    log("Part 9 — serum spike component activation …")
    sp7 = pd.read_csv(RL.SPIKE / "tables/phase7_serum_vs_pure.csv")
    resp = set(sp7.nlargest(6, "cos_spike_vs_pureSERS").analyte)
    p9_rows = []
    for a in fps:
        fp = fps[a]
        p9_rows.append({"analyte": a, "family": RL.analyte_family(a),
                        "responder": a in resp,
                        "activation_norm": float(np.linalg.norm(fp["delta"])),
                        "n_significant": fp["n_significant"],
                        "response_entropy": fp["entropy"],
                        "consistency_with_pure": float(cons.set_index("analyte").consistency_cosine.get(a, np.nan))})
    p9 = pd.DataFrame(p9_rows)
    p9.to_csv(TAB / "part9_serum_responders.csv", index=False)
    log(f"  responders activation norm {p9[p9.responder].activation_norm.median():.4f} vs "
        f"non {p9[~p9.responder].activation_norm.median():.4f} | "
        f"entropy {p9[p9.responder].response_entropy.median():.2f} vs {p9[~p9.responder].response_entropy.median():.2f}")

    # ── Part 10: response families (cluster on fingerprints, not spectra) ──
    log("Part 10 — response families …")
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    D = 1 - (U @ U.T); D = np.clip((D + D.T) / 2, 0, None); np.fill_diagonal(D, 0)
    Zl = linkage(squareform(D, checks=False), method="average")
    fams = np.array([RL.analyte_family(a) for a in fps])
    # ARI of response-fingerprint clusters vs chemical family, swept over k
    best = {"k": None, "ari": -1}
    for k in range(4, 13):
        lab = fcluster(Zl, t=k, criterion="maxclust")
        ari = adjusted_rand_score(fams, lab)
        if ari > best["ari"]:
            best = {"k": k, "ari": float(ari), "labels": lab.tolist()}
    # compare against clustering the RAW spike spectra (mean per analyte)
    blob = np.load(RL.SPIKE / "artifacts/processed_spectra.npz")
    Xs = blob["X_spiked_serum"]
    Xmean = np.vstack([np.nan_to_num(Xs[[ms.index.get_loc(i) for i in ms.index[ms.analyte == a]]]).mean(0)
                       for a in fps])
    Dx = 1 - RL._unit(Xmean) @ RL._unit(Xmean).T; Dx = np.clip((Dx + Dx.T) / 2, 0, None); np.fill_diagonal(Dx, 0)
    Zx = linkage(squareform(Dx, checks=False), method="average")
    best_x = max((adjusted_rand_score(fams, fcluster(Zx, t=k, criterion="maxclust"))
                  for k in range(4, 13)))
    fam_recovery = {"response_fingerprint_best_ari": best["ari"], "response_fingerprint_k": best["k"],
                    "raw_spectrum_best_ari": float(best_x),
                    "response_better_than_spectra": best["ari"] > best_x}
    (TAB / "part10_response_families.json").write_text(json.dumps(fam_recovery, indent=2))
    pd.DataFrame({"analyte": list(fps), "family": fams,
                  "response_cluster": best["labels"]}).to_csv(TAB / "part10_clusters.csv", index=False)
    np.savez(ART / "fingerprint_linkage.npz", linkage=Zl, distance=D, analytes=list(fps))
    log(f"  family recovery ARI — response {best['ari']:.3f} vs raw spectra {best_x:.3f}")

    # ── Part 11: component robustness under perturbation ──
    log("Part 11 — component robustness under perturbation …")
    rob_rows = []
    for j in range(K):
        # variability of a component across control replicates AND its responsiveness
        ctrl_var = float(np.nan_to_num(Zb[:, j]).std())
        resp = dose[dose.component == j]
        n_resp = int(resp.responsive.sum())
        spike_var = float(np.abs(F[:, j]).mean())
        rob_rows.append({"component": j, "theme": themes.get(j), "confidence": tconf.get(j),
                         "math_stability": stab.get(j),
                         "control_replicate_std": ctrl_var,
                         "mean_abs_spike_response": spike_var,
                         "n_dose_experiments_responsive": n_resp,
                         "responsiveness_ratio": spike_var / (ctrl_var + 1e-9)})
    rob = pd.DataFrame(rob_rows)
    # anchor score: mathematically stable AND meaningfully responsive AND interpretable
    rob["anchor_score"] = (rob.math_stability.rank(pct=True) * 0.4 +
                           (rob.n_dose_experiments_responsive > 0).astype(float) * 0.3 +
                           (rob.confidence == "high").astype(float) * 0.3)
    rob.sort_values("anchor_score", ascending=False).to_csv(TAB / "part11_component_robustness.csv", index=False)
    anchors = rob.sort_values("anchor_score", ascending=False).head(6)
    log(f"  candidate BSV anchor components: {anchors.component.tolist()}")

    # ── Part 12: trajectory library ──
    log("Part 12 — trajectory library …")
    traj_lib = {}
    for exp, (Zx, cx) in dose_experiments.items():
        traj_lib[exp] = RL.trajectory_fingerprint(Zx, cx)
    # uricase 'trajectory' = spiked -> spiked+uricase (2 states)
    (ART / "trajectory_library.json").write_text(json.dumps(traj_lib, indent=2, default=float))
    log(f"  {len(traj_lib)} dose trajectories stored")

    # ── Part 14: bipartite component-analyte network ──
    log("Part 14 — bipartite network …")
    edges = []
    for a in fps:
        d = fps[a]["delta"]
        for j in np.argsort(-np.abs(d))[:4]:
            if abs(d[j]) > 1e-4:
                edges.append({"analyte": a, "family": RL.analyte_family(a), "component": int(j),
                              "theme": themes.get(int(j)), "weight": float(d[j])})
    ed = pd.DataFrame(edges)
    ed.to_csv(TAB / "part14_bipartite_edges.csv", index=False)
    comp_hub = ed.groupby("component").analyte.nunique().sort_values(ascending=False)
    analyte_hub = ed.groupby("analyte").component.nunique().sort_values(ascending=False)
    (TAB / "part14_hubs.json").write_text(json.dumps({
        "component_hubs": {int(k): int(v) for k, v in comp_hub.head(6).items()},
        "analyte_hubs": {k: int(v) for k, v in analyte_hub.head(6).items()}}, indent=2))

    # ── manifest ──
    import hashlib
    W = np.load(RL.FROZEN / "manifold_components.npz")["components"]
    fp1 = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()[:32]
    assert fp1 == ctx["fingerprint"], "ATLAS MUTATED"
    (ART / "response_audit_manifest.json").write_text(json.dumps({
        "atlas_fingerprint": ctx["fingerprint"], "verified_unchanged": True,
        "n_analytes_fingerprinted": len(fps),
        "fingerprint_mean_pairwise_cosine": float(C[iu].mean()),
        "consistency_median": float(cons.consistency_cosine.median()),
        "family_recovery": fam_recovery,
        "candidate_anchor_components": anchors.component.tolist(),
        "responsive_components_by_experiment": {k: [int(x) for x in v] for k, v in resp_counts.items()},
        "runtime_s": round(time.time() - t0, 1),
    }, indent=2, default=str))
    log(f"atlas fingerprint re-verified unchanged | runtime {time.time()-t0:.0f}s")


def _theme_matches(analyte, theme):
    m = {"adenine": {"purine", "pyrimidine"}, "hypoxanthine": {"purine"},
         "guanine": {"purine"}, "xanthine": {"purine"}, "urate": {"purine"},
         "ergothioneine": {"cofactor", "amino_acid"}}
    return theme in m.get(analyte, set())


if __name__ == "__main__":
    main()
