#!/usr/bin/env python3
"""GAIRA V7 — Phase 02.5: latent geometry of spectral motif space.

Analysis only. Nothing frozen is refitted; no themes are created.

    python GAIRA_v7_rebuild/results/phase_02_5_latent_geometry/code/run_phase02_5.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PH = HERE.parent
REPO = PH.parents[2]
sys.path.insert(0, str(REPO / "results/v7_rebuild/phase00/code"))
sys.path.insert(0, str(REPO / "src"))

import v7_paths as P                                        # noqa: E402
from gaira.v7.geometry import embedding as EMB              # noqa: E402
from gaira.v7.geometry import fusion as FUS                 # noqa: E402
from gaira.v7.geometry import metrics as MET                # noqa: E402
from gaira.v7.geometry import neighbourhoods as NBH         # noqa: E402
from gaira.v7.geometry import nulls as NUL                  # noqa: E402
from gaira.v7.geometry import representations as REP        # noqa: E402
from gaira.v7.geometry import structure as STR              # noqa: E402

warnings.filterwarnings("ignore")

PHASE, PHASE_NAME = "02.5", "Latent geometry of spectral motif space"
TABLES, FIGURES = PH / "tables", PH / "figures"
REPORTS, VALID = PH / "reports", PH / "validation"
LOGS, ARTIFACTS = PH / "logs", PH / "artifacts"
INTERACTIVE = PH / "interactive"
P00 = REPO / "results/v7_rebuild/phase00"
P01 = REPO / "results/v7_rebuild/phase01"
P02 = REPO / "results/v7_rebuild/phase02"
P01_FP = "208482d6f7178b5b8f16cace91be55b0"
P02_FP = "0b4aa550ccefed3edabdbde5bae11c8d"
SEED = 0
K_NN = 5
LOG: list[str] = []


def log(m):
    line = f"[phase02.5] {m}"
    print(line, flush=True)
    LOG.append(line)


def wtab(df, name, where=None):
    p = (where or TABLES) / name
    df.to_csv(p, index=False)
    return {"artifact_id": name, "path": str(p.relative_to(REPO)),
            "sha256": P.sha256_file(p), "rows": len(df)}


def wjson(obj, name, where=None):
    p = (where or ARTIFACTS) / name
    p.write_text(json.dumps(obj, indent=2, default=_ser))
    return {"artifact_id": name, "path": str(p.relative_to(REPO)),
            "sha256": P.sha256_file(p)}


def _ser(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main() -> int:
    for d in (TABLES, FIGURES, REPORTS, VALID, LOGS, ARTIFACTS, INTERACTIVE):
        d.mkdir(parents=True, exist_ok=True)
    outputs: list[dict] = []
    t0 = datetime.now(timezone.utc)

    # ── STEP 0 — architecture compliance ─────────────────────────────────────
    log("architecture check — Phase 02.5 is an INSERTED analysis phase, not in the 00–08 "
        "sequence; it refits nothing and creates no themes")
    Hfrozen = np.asarray(np.load(P.FOUNDATION / "manifold_components.npz")["components"], float)
    fp_atlas = P.sha256_array(Hfrozen)
    assert fp_atlas == P.CANONICAL_ATLAS_FINGERPRINT
    s01 = json.loads((P01 / "PHASE_STATE.json").read_text())
    s02 = json.loads((P02 / "PHASE_STATE.json").read_text())
    assert s01["registry_fingerprint"] == P01_FP, "Phase 01 fingerprint mismatch"
    assert s02["csm_fingerprint"] == P02_FP, "Phase 02 fingerprint mismatch"
    log(f"  frozen: atlas {fp_atlas} · LSM {P01_FP} · CSM {P02_FP}")

    # ── STEP 1 — representation panel ────────────────────────────────────────
    z = np.load(P01 / "artifacts/lsm_dictionary_v1.npz", allow_pickle=True)
    H = np.asarray(z["H"], float)
    ids = [str(s) for s in z["motif_ids"]]
    reg01 = pd.read_csv(P01 / "artifacts/lsm_registry_v1.csv").set_index("motif_id").loc[ids]
    classes = reg01.chemical_class.tolist()              # revealed only after fitting
    lsm_meta = [{"motif_id": m, "chemical_class": r.chemical_class, "lsm_type": r.lsm_type,
                 "stability": float(r.stability),
                 "analytes": str(r.analytes).split(";") if pd.notna(r.analytes) else [],
                 "bands": [float(b) for b in str(r.band_centers_cm).split(";")
                           if b not in ("", "nan")]}
                for m, r in zip(ids, reg01.itertuples())]
    bands = [m["bands"] for m in lsm_meta]

    br = np.load(P01 / "artifacts/balanced_references_v1.npz", allow_pickle=True)
    X = np.asarray(br["X"], float)
    grid = np.asarray(br["grid"], float)
    canonical_id = np.array([str(s) for s in br["canonical_id"]])
    weight = np.asarray(br["weight"], float)

    ef = np.load(P02 / "artifacts/edge_features_v1.npz", allow_pickle=True)
    feat = {f: ef[f] for f in ("spectral_cosine", "band_overlap", "peak_agreement",
                               "bootstrap_cooccurrence", "activation_cooccurrence",
                               "provenance_overlap", "substitutability")}
    W02 = np.asarray(ef["W"], float)
    A_mol = np.asarray(ef["A_mol"], float)

    canon = pd.read_csv(P00 / "tables/canonical_analytes_v1.csv")
    sources_of = {r.canonical_id: str(r.sources).split(";") for r in canon.itertuples()}
    excit_of = {r.canonical_id: str(r.excitations).split(";") for r in canon.itertuples()}
    nspec_of = dict(zip(canon.canonical_id, canon.n_spectra))

    log("STEP 1 — building seven representations")
    panel = REP.build_panel(H, grid, A_mol, X, canonical_id, classes, lsm_meta,
                            sources_of, excit_of, nspec_of, feat)
    views = panel["views"]
    for k, v in views.items():
        log(f"  {k:30s} {v.shape}")
    outputs.append(wjson({"views": panel["manifest"], "n_motifs": len(ids),
                          "chemistry_label_used_in_construction": False,
                          "source_label_used_in_construction": False},
                         "representation_manifest_v1.json"))

    peak_vec = REP.peak_vector(panel["peaks"], grid)
    band_vec, band_cols = REP.band_family(H, grid)

    # ── STEP 2 — distance benchmark ──────────────────────────────────────────
    log("STEP 2 — benchmarking ten distance metrics")
    Ds = MET.all_distances(H, grid, peak_vec, band_vec, A_mol, W02)
    null_D = {m: NUL.null_distance_ensemble(
        H, grid, lambda Hn, m=m: MET.DISPATCH[m](
            Hn, grid=grid, peak_vec=REP.peak_vector(REP.peak_records(Hn, grid), grid),
            band_vec=REP.band_family(Hn, grid)[0], act=A_mol, W=W02),
        "band_position", n=25, seed=SEED)
        for m in ("spectral_cosine", "pearson", "spearman", "euclidean_l2",
                  "jensen_shannon", "wasserstein", "peak_set", "band_overlap")}

    def _boot(metric):
        def build(rng):
            if rng is None:
                return Ds[metric]
            keep = np.sort(rng.choice(H.shape[1], H.shape[1], replace=True))
            Hb = H[:, keep]
            gb = grid[keep]
            return np.asarray(MET.DISPATCH[metric](
                Hb, grid=gb, peak_vec=peak_vec, band_vec=band_vec, act=A_mol, W=W02), float)
        return MET.bootstrap_stability(build, len(ids), n_boot=15, k=K_NN, seed=SEED)

    rows = []
    for m in MET.METRICS:
        fn = MET.DISPATCH[m]
        probeable = m in MET.PROBEABLE
        pr = MET.scale_free_probes(fn, grid, Ds[m]) if probeable else {}
        r = {"metric": m, "probeable": probeable,
             "median_observed_distance": pr.get("median_observed_distance", np.nan),
             "amplitude_invariance": 1.0 - pr.get("amplitude_leakage", np.nan),
             "peak_shift_tolerance": 1.0 - pr.get("peak_shift_cost", np.nan),
             "width_sensitivity": pr.get("width_discrimination", np.nan),
             "background_separation": pr.get("background_separation", np.nan),
             "knn_chemical_coherence": MET.knn_label_coherence(Ds[m], classes, K_NN),
             "bootstrap_stability": _boot(m) if m in ("spectral_cosine", "pearson", "spearman",
                                                      "euclidean_l2", "jensen_shannon",
                                                      "wasserstein") else float("nan"),
             "null_separation": (MET.null_separation(Ds[m], null_D[m])
                                 if m in null_D else float("nan"))}
        rows.append(r)
        log(f"  {m:22s} ampl-inv {r['amplitude_invariance']:6.3f}  shift-tol "
            f"{r['peak_shift_tolerance']:6.3f}  bg-sep {r['background_separation']:6.3f}  "
            f"kNN-coh {r['knn_chemical_coherence']:.3f}  null-z {r['null_separation']:6.2f}"
            + ("" if probeable else "   [not probeable on synthetic spectra]"))
    metric_tab = pd.DataFrame(rows)
    outputs.append(wtab(metric_tab, "metric_comparison_v1.csv"))

    # Pre-declared selection: spectral geometry maximises background separation x null
    # separation subject to amplitude invariance >= 0.95 and shift tolerance >= 0.90.
    # Selection rule, applied to the SCALE-FREE probes: among metrics that leak < 5% of their
    # own scale to amplitude and cost < 10% of it for a 6 cm-1 shift, maximise
    # background separation x null separation.
    adm = metric_tab[(metric_tab.amplitude_invariance >= 0.95)
                     & (metric_tab.peak_shift_tolerance >= 0.90)
                     & metric_tab.null_separation.notna()]
    adm = adm[adm.metric.isin(("spectral_cosine", "pearson", "spearman", "euclidean_l2",
                               "jensen_shannon", "wasserstein", "peak_set", "band_overlap"))]
    if adm.empty:
        adm = metric_tab[metric_tab.null_separation.notna()]
    primary = str(adm.assign(s=adm.background_separation * adm.null_separation)
                  .sort_values("s", ascending=False).iloc[0]["metric"])
    activation_metric = "activation_profile"
    log(f"  primary spectral metric: {primary}; activation metric: {activation_metric}")

    # ── STEP 3 — nulls (recorded) ────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    null_summary = {
        "band_position": {"n": 25, "destroys": "relative band positions between motifs"},
        "intensity_permutation": {"n": 25,
                                  "destroys": "relative intensity among a motif's own peaks"},
        "class_label": {"n": 500, "destroys": "the association between geometry and chemistry"},
        "molecule_activation": {"n": 200, "destroys": "which molecules a motif responds to"},
        "source_label": {"n": 500, "destroys": "the association between geometry and source"},
        "degree_preserving_graph": {"n": 50, "destroys": "community structure at fixed degree"},
    }
    outputs.append(wjson(null_summary, "null_models_v1.json"))

    # ── STEP 4 — linear geometry ─────────────────────────────────────────────
    log("STEP 4 — PCA on four representations")
    pca_rows, pca_store = [], {}
    for name, V in (("spectral_profile", views["spectral_profile"]),
                    ("band_family", views["band_family"]),
                    ("activation", views["activation"]),
                    ("multiview_concat", np.hstack([
                        (v - v.mean(0)) / (v.std(0) + 1e-12) / np.sqrt(v.shape[1])
                        for v in views.values()]))):
        p = EMB.fit_pca(V, 10)
        stab = EMB.pca_stability(V, 6, n_boot=30)
        pca_store[name] = p
        for j in range(min(6, p["loadings"].shape[0])):
            drivers = (EMB.band_drivers(p["loadings"][j], grid)
                       if name == "spectral_profile" else [])
            pca_rows.append({
                "representation": name, "pc": j + 1,
                "explained_variance_ratio": float(p["explained_variance_ratio"][j]),
                "cumulative": float(p["cumulative"][j]),
                "loading_stability": float(stab[j]) if j < stab.size else float("nan"),
                "top_bands_cm1": ";".join(f"{d['direction']}{d['cm1']:.0f}" for d in drivers),
                "extreme_low": ids[int(np.argmin(p["scores"][:, j]))],
                "extreme_high": ids[int(np.argmax(p["scores"][:, j]))],
            })
        log(f"  {name:20s} PC1–3 EV {p['explained_variance_ratio'][:3].round(3)} "
            f"stability {stab[:3].round(3)}")
    pca_tab = pd.DataFrame(pca_rows)
    outputs.append(wtab(pca_tab, "pca_components_v1.csv"))
    np.savez_compressed(ARTIFACTS / "pca_v1.npz",
                        **{f"{k}_scores": v["scores"] for k, v in pca_store.items()},
                        **{f"{k}_loadings": v["loadings"] for k, v in pca_store.items()},
                        grid=grid, motif_ids=np.array(ids, dtype=object))

    # PC association with source / excitation
    def _dominant(m, table):
        """Most frequent source/excitation behind a motif, ties broken alphabetically.

        Sorted, not `max` over a set: Python randomises string hashing per process, so set
        iteration order — and therefore the winner of a tie — changes between runs. That made
        the excitation PERMANOVA move from p = 0.206 to p = 0.005 with no change to the data.
        """
        cnt = {}
        for a in m["analytes"]:
            for v in table.get(a, []):
                cnt[v] = cnt.get(v, 0) + 1
        if not cnt:
            return "?"
        return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    src_of = [_dominant(m, sources_of) for m in lsm_meta]
    exc_of = [_dominant(m, excit_of) for m in lsm_meta]
    pc_conf = []
    S = pca_store["spectral_profile"]["scores"]
    for j in range(6):
        pc_conf.append({
            "pc": j + 1,
            "source_permanova_p": NBH.permanova(
                np.abs(S[:, [j]] - S[:, [j]].T), src_of, n_perm=499, seed=SEED)["p"],
            "excitation_permanova_p": NBH.permanova(
                np.abs(S[:, [j]] - S[:, [j]].T), exc_of, n_perm=499, seed=SEED)["p"],
            "class_permanova_p": NBH.permanova(
                np.abs(S[:, [j]] - S[:, [j]].T), classes, n_perm=499, seed=SEED)["p"],
        })
    outputs.append(wtab(pd.DataFrame(pc_conf), "pc_confounding_v1.csv", VALID))

    # ── STEP 5 — nonlinear geometry ──────────────────────────────────────────
    log("STEP 5 — nonlinear embeddings")
    Dprim = Ds[primary]
    umap_rows = EMB.umap_stability_sweep(Dprim, k=K_NN)
    umap_tab = pd.DataFrame(umap_rows)
    outputs.append(wtab(umap_tab, "umap_stability_sweep_v1.csv"))
    best_umap = umap_tab.sort_values("knn_jaccard_across_seeds", ascending=False).iloc[0]
    log(f"  UMAP: kNN Jaccard vs high-dim {umap_tab.knn_jaccard_vs_highdim.mean():.3f} "
        f"(best {umap_tab.knn_jaccard_vs_highdim.max():.3f}); across seeds "
        f"{umap_tab.knn_jaccard_across_seeds.mean():.3f}")

    dm = EMB.fit_diffusion_map(Dprim, 5)
    spec_emb = EMB.fit_spectral_embedding(Dprim, 3)
    E_umap = EMB.fit_umap(Dprim, int(best_umap.n_neighbors), float(best_umap.min_dist))
    emb_rows = []
    for name, E in (("umap", E_umap), ("diffusion_map", dm["coordinates"][:, :2]),
                    ("spectral_embedding", spec_emb[:, :2]),
                    ("pca", pca_store["spectral_profile"]["scores"][:, :2])):
        emb_rows.append({
            "embedding": name,
            "trustworthiness": float(EMB.trustworthiness(Dprim, E, n_neighbors=K_NN,
                                                         metric="precomputed")),
            "continuity": EMB.continuity(Dprim, E, K_NN),
            "knn_preservation": EMB.knn_preservation(Dprim, E, K_NN),
        })
    emb_rows[0]["procrustes_disparity"] = EMB.procrustes_stability(
        lambda s: EMB.fit_umap(Dprim, int(best_umap.n_neighbors), float(best_umap.min_dist), s),
        n_rep=4)
    emb_tab = pd.DataFrame(emb_rows)
    outputs.append(wtab(emb_tab, "embedding_quality_v1.csv"))
    log(f"  diffusion map: spectral gap {dm['spectral_gap']:.4f}, "
        f"eigenvalues {np.round(dm['eigenvalues'][:4], 3)}")
    np.savez_compressed(ARTIFACTS / "embeddings_v1.npz", umap=E_umap,
                        diffusion=dm["coordinates"], spectral=spec_emb,
                        eigenvalues=dm["eigenvalues"], motif_ids=np.array(ids, dtype=object))

    # ── STEP 6 — hierarchy and graph ─────────────────────────────────────────
    log("STEP 6 — hierarchical, spectral and graph structure")
    sweep = STR.cluster_sweep(Dprim, views["spectral_profile"])
    sweep_tab = pd.DataFrame(sweep)
    outputs.append(wtab(sweep_tab, "cluster_sweep_v1.csv"))
    best = sweep_tab.sort_values("silhouette", ascending=False).iloc[0]
    boot = STR.cluster_bootstrap_ari(Dprim, int(best.k), str(best.method))
    log(f"  best silhouette {best.silhouette:.3f} at k={int(best.k)} ({best.method}); "
        f"bootstrap ARI {boot['mean_ari']:.3f}")

    G = STR.knn_graph(Dprim, ids, k=K_NN)
    Gm = STR.knn_graph(Dprim, ids, k=K_NN, mutual=True)
    roles = STR.graph_roles(G, Dprim, ids, classes)
    outputs.append(wtab(roles, "graph_roles_v1.csv"))
    mod = STR.modularity_vs_null(G, W02, n_null=40, seed=SEED)
    log(f"  modularity {mod['observed_modularity']:.3f} vs degree-preserving null "
        f"{mod['null_mean']:.3f} ± {mod['null_sd']:.3f} (z = {mod['z']:.2f}, "
        f"p = {mod['p_empirical']:.3f})")
    mst = STR.minimum_spanning_tree(Dprim, ids)
    outputs.append(wjson({"modularity_vs_null": mod,
                          "knn_graph_edges": G.number_of_edges(),
                          "mutual_knn_edges": Gm.number_of_edges(),
                          "mst_edges": mst.number_of_edges(),
                          "cluster_bootstrap": boot,
                          "best_k": int(best.k), "best_linkage": str(best.method)},
                         "graph_structure_v1.json"))

    # ── STEP 7 — discrete vs continuous ──────────────────────────────────────
    log("STEP 7 — discrete islands or continua")
    lid = STR.local_intrinsic_dimension(Dprim, k=10)
    gap = STR.density_gap_statistic(Dprim)
    import networkx as nx
    comms = nx.community.louvain_communities(G, weight="weight", seed=0)
    groups = [sorted(ids.index(u) for u in c) for c in comms]
    cond = STR.graph_conductance(Dprim, groups)
    region_rows = []
    for k, (g, c) in enumerate(zip(groups, cond)):
        gb = STR.cluster_bootstrap_ari(Dprim[np.ix_(g, g)], max(2, min(3, len(g) - 1)),
                                       n_boot=25) if len(g) > 3 else {"mean_ari": 1.0}
        region_rows.append({
            "region": f"region{k:02d}", "n_motifs": len(g),
            "members": ";".join(ids[i] for i in g),
            "conductance": float(c), "mean_local_dimension": float(np.nanmean(lid[g])),
            "internal_stability": float(gb["mean_ari"]),
            "geometry_type": STR.classify_region(c, float(np.nanmean(lid[g])),
                                                 gap["bimodal"], float(gb["mean_ari"])),
            "classes": ";".join(sorted({classes[i] for i in g})),
        })
    region_tab = pd.DataFrame(region_rows).sort_values("n_motifs", ascending=False)
    outputs.append(wtab(region_tab, "geometry_regions_v1.csv"))
    outputs.append(wjson({"local_intrinsic_dimension": {"mean": float(np.nanmean(lid)),
                                                        "median": float(np.nanmedian(lid)),
                                                        "per_motif": dict(zip(ids, lid))},
                          "density": gap,
                          "verdict": ("mixed" if len(set(region_tab.geometry_type)) > 1
                                      else region_tab.geometry_type.iloc[0])},
                         "continuum_analysis_v1.json"))
    log(f"  local dimension mean {np.nanmean(lid):.2f}; density modes {gap['n_modes']}, "
        f"valley depth {gap['valley_depth']:.3f}, bimodal {gap['bimodal']}")
    log(f"  region types: {dict(region_tab.geometry_type.value_counts())}")

    # ── STEP 8 — neighbourhoods (labels revealed here, not before) ───────────
    log("STEP 8 — neighbourhood discovery (chemistry revealed AFTER the geometry)")
    cards = NBH.nearest_neighbour_cards(Dprim, ids, lsm_meta, bands, K_NN)
    pv = np.asarray(ef["pvalues"], float)
    idx_of = {m: i for i, m in enumerate(ids)}
    csm_reg = json.loads((P02 / "artifacts/csm_registry_v1.json").read_text())
    merged_pairs = {frozenset((a["lsm_id"], b["lsm_id"]))
                    for c in csm_reg["csms"] if c["n_lsms"] > 1
                    for a in c["contributing_lsms"] for b in c["contributing_lsms"]
                    if a["lsm_id"] != b["lsm_id"]}
    cards["edge_weight"] = [W02[idx_of[r.motif], idx_of[r.neighbour]] for r in cards.itertuples()]
    cards["null_p"] = [pv[idx_of[r.motif], idx_of[r.neighbour]] for r in cards.itertuples()]
    cards["band_overlap"] = [feat["band_overlap"][idx_of[r.motif], idx_of[r.neighbour]]
                             for r in cards.itertuples()]
    cards["relationship_tier"] = [
        NBH.classify_relationship(r._asdict(), merged_pairs, r.edge_weight, r.null_p,
                                  r.band_overlap) for r in cards.itertuples()]
    outputs.append(wtab(cards, "nearest_neighbour_cards_v1.csv"))
    tiers = cards.relationship_tier.value_counts().to_dict()
    log(f"  relationship tiers over {len(cards)} neighbour links: {tiers}")

    # class enrichment of neighbourhoods, against the label-permutation null
    def coherence_stat(lab):
        return MET.knn_label_coherence(Dprim, list(lab), K_NN)
    obs_coh = coherence_stat(classes)
    p_coh, mu, sd = NUL.enrichment_pvalue(obs_coh, classes, coherence_stat, 500, SEED)
    log(f"  kNN chemical coherence {obs_coh:.3f} vs label-permutation null {mu:.3f} ± {sd:.3f} "
        f"(p = {p_coh:.4f})")

    # ── STEP 9 — the three rejected Phase 02 groups, as geometry ─────────────
    log("STEP 9 — re-examining the three rejected Phase 02 proposals as neighbourhoods")
    rej = pd.read_csv(P02 / "tables/rejected_consensus_motifs_v1.csv")
    prop_rows = []
    for r in rej.itertuples():
        mem = [idx_of[m] for m in r.contributing_lsms.split(";")]
        sub = Dprim[np.ix_(mem, mem)]
        others = [i for i in range(len(ids)) if i not in mem]
        c = STR.graph_conductance(Dprim, [mem])[0]
        gb = STR.cluster_bootstrap_ari(sub, max(2, min(3, len(mem) - 1)), n_boot=25) \
            if len(mem) > 3 else {"mean_ari": 1.0}
        lidm = float(np.nanmean(lid[mem]))
        gtype = STR.classify_region(c, lidm, gap["bimodal"], float(gb["mean_ari"]))
        prop_rows.append({
            "proposal": r.proposed_group, "n_motifs": len(mem),
            "classes": r.supporting_classes,
            "mean_internal_distance": float(sub[np.triu_indices(len(mem), 1)].mean()),
            "mean_external_distance": float(Dprim[np.ix_(mem, others)].mean()),
            "separation_ratio": float(Dprim[np.ix_(mem, others)].mean()
                                      / (sub[np.triu_indices(len(mem), 1)].mean() + 1e-12)),
            "conductance": float(c), "mean_local_dimension": lidm,
            "internal_stability": float(gb["mean_ari"]),
            "geometry_type": gtype,
            "shared_bands_cm1": ";".join(f"{b:.0f}" for b in
                                         NBH._shared_bands(H[mem], grid)),
        })
    prop_tab = pd.DataFrame(prop_rows)
    outputs.append(wtab(prop_tab, "rejected_proposal_geometry_v1.csv"))
    for r in prop_tab.itertuples():
        log(f"  {r.proposal}: sep-ratio {r.separation_ratio:.2f}, conductance "
            f"{r.conductance:.3f}, local dim {r.mean_local_dimension:.2f} → {r.geometry_type}")

    # gradient tests inside each proposal, on the leading diffusion coordinate
    grad_rows = []
    for r in rej.itertuples():
        mem = [idx_of[m] for m in r.contributing_lsms.split(";")]
        dsub = EMB.fit_diffusion_map(Dprim[np.ix_(mem, mem)], 2)
        c1 = dsub["coordinates"][:, 0]
        order = np.argsort(c1)
        for rank, o in enumerate(order):
            i = mem[o]
            grad_rows.append({"proposal": r.proposed_group, "rank": rank,
                              "motif": ids[i], "chemical_class": classes[i],
                              "diffusion_coord_1": float(c1[o]),
                              "n_bands": len(bands[i]),
                              "bands_cm1": ";".join(f"{b:.0f}" for b in bands[i])})
    outputs.append(wtab(pd.DataFrame(grad_rows), "proposal_gradients_v1.csv"))

    # ── STEP 10 — source and excitation confounding ──────────────────────────
    log("STEP 10 — source and excitation confounding")
    conf_rows = []
    for lab_name, lab in (("chemistry_class", classes), ("source", src_of),
                          ("excitation", exc_of)):
        pm = NBH.permanova(Dprim, lab, n_perm=999, seed=SEED)
        kp = NBH.knn_label_predictability(Dprim, lab, K_NN)
        conf_rows.append({"label": lab_name, "n_groups": pm["n_groups"], "permanova_F": pm["F"],
                          "permanova_p": pm["p"], "permanova_R2": pm.get("R2", float("nan")),
                          **kp})
        log(f"  {lab_name:16s} PERMANOVA F={pm['F']:.2f} p={pm['p']:.4f} R2={pm.get('R2', 0):.3f}"
            f" · kNN acc {kp['knn_accuracy']:.3f} vs chance {kp['chance']:.3f}")
    conf_tab = pd.DataFrame(conf_rows)
    outputs.append(wtab(conf_tab, "confounding_v1.csv", VALID))

    single_src = NBH.single_source_motifs(lsm_meta, sources_of)
    src_groups = {}
    for s in sorted(set(src_of)):
        src_groups[s] = [i for i, v in enumerate(src_of) if v == s]

    def rebuild(keep):
        return np.asarray(MET.DISPATCH[primary](
            H[keep], grid=grid, peak_vec=peak_vec[keep], band_vec=band_vec[keep],
            act=A_mol[:, keep], W=W02[np.ix_(keep, keep)]), float)

    loso = NBH.leave_one_out_geometry(rebuild, src_groups, ids, Dprim, K_NN)
    exc_groups = {e: [i for i, v in enumerate(exc_of) if v == e] for e in sorted(set(exc_of))}
    loeo = NBH.leave_one_out_geometry(rebuild, exc_groups, ids, Dprim, K_NN)
    outputs.append(wtab(loso.assign(kind="source"), "leave_one_source_out_v1.csv", VALID))
    outputs.append(wtab(loeo.assign(kind="excitation"), "leave_one_excitation_out_v1.csv", VALID))
    log(f"  {len(single_src)} of {len(ids)} motifs are single-source → source-untestable")

    # ── STEP 11 — multi-view integration ─────────────────────────────────────
    log("STEP 11 — multi-view integration, five candidates")
    view_D = {
        "spectral": Ds[primary],
        "peaks": Ds["peak_set"],
        "bands": Ds["band_overlap"],
        "activation": Ds["activation_profile"],
        "edges": Ds["phase02_composite"],
    }
    cands = {
        "weighted_similarity": FUS.weighted_similarity(
            view_D, {"spectral": 0.3, "peaks": 0.2, "bands": 0.2,
                     "activation": 0.15, "edges": 0.15}),
        "similarity_network_fusion": FUS.similarity_network_fusion(view_D, k=8),
        "multiple_kernel_embedding": FUS.multiple_kernel_embedding(view_D),
        "concatenated_features": FUS.concatenated_features(views),
        "graph_consensus": FUS.graph_consensus(view_D, k=K_NN),
    }
    cands["single_view_" + primary] = Dprim          # the do-nothing control

    def boot_fn(Dm):
        rng = np.random.default_rng(SEED)
        base = [set(np.argsort(Dm[i])[1:K_NN + 1]) for i in range(len(ids))]
        js = []
        for _ in range(15):
            noise = rng.normal(0, 0.02, Dm.shape)
            noise = (noise + noise.T) / 2
            Db = np.clip(Dm + noise, 0, None)
            np.fill_diagonal(Db, 0)
            for i in range(len(ids)):
                nb = set(np.argsort(Db[i])[1:K_NN + 1])
                js.append(len(nb & base[i]) / len(nb | base[i]))
        return float(np.mean(js))

    INTERP = {"weighted_similarity": 0.8, "similarity_network_fusion": 0.5,
              "multiple_kernel_embedding": 0.3, "concatenated_features": 0.6,
              "graph_consensus": 0.9, "single_view_" + primary: 1.0}
    NFEAT = {"weighted_similarity": 5, "similarity_network_fusion": 5,
             "multiple_kernel_embedding": 5, "concatenated_features":
             sum(v.shape[1] for v in views.values()), "graph_consensus": 5,
             "single_view_" + primary: 1}

    # Each candidate needs its OWN null. Scoring a fused geometry against the single-view
    # null compares distances on different scales, and null separation is scale-dependent —
    # it would reward whichever fusion happens to compress its distance range hardest.
    log("  building a matched band-permutation null for every candidate geometry")
    rng_null = np.random.default_rng(SEED)
    cand_nulls = {name: [] for name in cands}
    for _ in range(10):
        Hn = NUL.band_position_null(H, rng_null)
        pkn = REP.peak_vector(REP.peak_records(Hn, grid), grid)
        bfn = REP.band_family(Hn, grid)[0]
        Dn = {"spectral": np.asarray(MET.DISPATCH[primary](
                  Hn, grid=grid, peak_vec=pkn, band_vec=bfn, act=A_mol, W=W02), float),
              "peaks": np.asarray(MET.DISPATCH["peak_set"](Hn, peak_vec=pkn), float),
              "bands": np.asarray(MET.DISPATCH["band_overlap"](Hn, band_vec=bfn), float),
              "activation": Ds["activation_profile"],
              "edges": Ds["phase02_composite"]}
        viewsn = dict(views)
        viewsn["spectral_profile"] = REP.spectral_profile(Hn)
        viewsn["peak_representation"] = np.hstack([
            pkn / (np.linalg.norm(pkn, axis=1, keepdims=True) + 1e-12),
            REP.peak_summary(REP.peak_records(Hn, grid))])
        viewsn["band_family"] = bfn
        cand_nulls["weighted_similarity"].append(FUS.weighted_similarity(
            Dn, {"spectral": 0.3, "peaks": 0.2, "bands": 0.2,
                 "activation": 0.15, "edges": 0.15}))
        cand_nulls["similarity_network_fusion"].append(
            FUS.similarity_network_fusion(Dn, k=8))
        cand_nulls["multiple_kernel_embedding"].append(FUS.multiple_kernel_embedding(Dn))
        cand_nulls["concatenated_features"].append(FUS.concatenated_features(viewsn))
        cand_nulls["graph_consensus"].append(FUS.graph_consensus(Dn, k=K_NN))
        cand_nulls["single_view_" + primary].append(Dn["spectral"])

    fus_rows = []
    for name, Dm in cands.items():
        sc = FUS.score_geometry(Dm, classes, src_of, cand_nulls[name], boot_fn, K_NN,
                                NFEAT[name])
        sc["interpretability"] = INTERP[name]
        sc["method"] = name
        fus_rows.append(sc)
        log(f"  {name:32s} stab {sc['neighbourhood_stability']:.3f}  null-z "
            f"{sc['null_separation']:.2f}  coh {sc['spectroscopic_coherence']:.3f}  "
            f"src-rob {sc['source_robustness']:.3f}")
    for sc, (name, Dm) in zip(fus_rows, cands.items()):
        off = Dm[~np.eye(Dm.shape[0], dtype=bool)]
        sc["distance_range"] = float(off.max() - off.min())
        sc["degenerate"] = bool(sc["distance_range"] < 0.05
                                or not np.isfinite(sc["null_separation"])
                                or abs(sc["null_separation"]) > 1e3)
    live = [s for s in fus_rows if not s["degenerate"]]
    dead = [s["method"] for s in fus_rows if s["degenerate"]]
    if dead:
        log(f"  excluded from the Pareto as degenerate: {', '.join(dead)} — a collapsed "
            f"distance range makes null separation unbounded and would flatten the criterion "
            f"for every other candidate under min-max normalisation")
    winner, comp_live = FUS.pareto_select(live)
    comp_map = {s["method"]: c for s, c in zip(live, comp_live)}
    fus_tab = pd.DataFrame(fus_rows).assign(
        pareto_composite=[comp_map.get(s["method"], float("nan")) for s in fus_rows]
    ).sort_values("pareto_composite", ascending=False)
    outputs.append(wtab(fus_tab, "multiview_comparison_v1.csv"))
    log(f"  PRIMARY GEOMETRY: {winner}")
    Dfinal = cands[winner]
    # TWO distinct objects, named so they cannot be confused:
    #   D_primary_metric   — the selected single-view spectral geometry (steps 4-10 and the
    #                        neighbourhood cards are computed on this)
    #   D_primary_geometry — the Pareto-winning multi-view fusion
    # An earlier version stored the fused matrix under a key the cards were not built from.
    nb_agree = float(np.mean([
        len(set(np.argsort(Dprim[i])[1:K_NN + 1]) & set(np.argsort(Dfinal[i])[1:K_NN + 1]))
        / K_NN for i in range(len(ids))]))
    log(f"  metric vs fused geometry: {nb_agree:.3f} mean k-NN agreement")
    np.savez_compressed(ARTIFACTS / "geometry_v1.npz",
                        **{f"D_{k}": v for k, v in Ds.items()},
                        **{f"fused_{k}": v for k, v in cands.items()},
                        D_primary_metric=Dprim, D_primary_geometry=Dfinal,
                        motif_ids=np.array(ids, dtype=object))

    # ── STEP 4b — CSM sensitivity control ────────────────────────────────────
    log("CSM sensitivity control (49 CSMs)")
    zc = np.load(P02 / "artifacts/csm_dictionary_v1.npz", allow_pickle=True)
    Hc = np.asarray(zc["CSM"], float)
    cids = [str(s) for s in zc["csm_ids"]]
    pk_c = REP.peak_vector(REP.peak_records(Hc, grid), grid)
    bf_c, _ = REP.band_family(Hc, grid)
    Dc = np.asarray(MET.DISPATCH[primary](Hc, grid=grid, peak_vec=pk_c, band_vec=bf_c,
                                          act=None, W=None), float)
    lsm2csm = {l["lsm_id"]: c["csm_id"] for c in csm_reg["csms"] for l in c["contributing_lsms"]}
    csm_class = []
    for c in csm_reg["csms"]:
        cs = c["supporting_classes"]
        csm_class.append(cs[0] if len(cs) == 1 else "multi")
    csm_rows = {
        "n_csms": len(cids),
        "knn_chemical_coherence_lsm": MET.knn_label_coherence(Dprim, classes, K_NN),
        "knn_chemical_coherence_csm": MET.knn_label_coherence(Dc, csm_class, K_NN),
        "local_dimension_lsm": float(np.nanmean(lid)),
        "local_dimension_csm": float(np.nanmean(STR.local_intrinsic_dimension(Dc, 10))),
        "density_lsm": gap, "density_csm": STR.density_gap_statistic(Dc),
        "neighbour_agreement_lsm_vs_csm": float(np.mean([
            len(set(lsm2csm.get(ids[j], "") for j in np.argsort(Dprim[i])[1:K_NN + 1])
                & set(cids[j] for j in np.argsort(Dc[cids.index(lsm2csm[ids[i]])])[1:K_NN + 1]))
            / K_NN for i in range(len(ids)) if ids[i] in lsm2csm])),
    }
    outputs.append(wjson(csm_rows, "csm_sensitivity_v1.json"))
    log(f"  CSM coherence {csm_rows['knn_chemical_coherence_csm']:.3f} vs LSM "
        f"{csm_rows['knn_chemical_coherence_lsm']:.3f}; neighbour agreement "
        f"{csm_rows['neighbour_agreement_lsm_vs_csm']:.3f}")

    # ── STEP 12 — Phase 03 priors ────────────────────────────────────────────
    log("STEP 12 — deriving provisional Phase 03 priors")
    priors = build_priors(ids, idx_of, lsm_meta, classes, H, grid, Dfinal, Dprim, lid, gap,
                          prop_tab, rej, region_tab, roles, lsm2csm, conf_tab, single_src,
                          merged_pairs, csm_reg)
    outputs.append(wjson({"schema": "gaira_v7_phase03_geometry_priors_v1",
                          "status": "PROVISIONAL — priors only; no themes are created here",
                          "source_phase": "02.5", "primary_geometry": winner,
                          "primary_spectral_metric": primary,
                          "n_priors": len(priors), "priors": priors},
                         "phase03_geometry_priors.json"))
    log(f"  {len(priors)} priors: " + ", ".join(p["prior_id"] for p in priors))

    # ── manifest, state ──────────────────────────────────────────────────────
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())
    manifest = {
        "schema": "gaira_v7_phase_manifest_v1", "phase": PHASE, "phase_name": PHASE_NAME,
        "built_utc": t0.isoformat(),
        "nature": "ANALYSIS ONLY — refits nothing, creates no themes",
        "frozen_inputs": {
            "atlas_fingerprint": fp_atlas,
            "lsm_registry_fingerprint": P01_FP,
            "csm_dictionary_fingerprint": P02_FP,
            "lsm_dictionary_sha256": P.sha256_file(P01 / "artifacts/lsm_dictionary_v1.npz"),
            "csm_dictionary_sha256": P.sha256_file(P02 / "artifacts/csm_dictionary_v1.npz"),
        },
        "firewalls": {"chemistry_labels_used_in_fitting": False,
                      "source_labels_used_in_fitting": False,
                      "revealed_at_step": 8},
        "primary_spectral_metric": primary, "activation_metric": activation_metric,
        "primary_geometry": winner, "seed": SEED, "k_nn": K_NN,
        "neighbourhoods_computed_on": "D_primary_metric",
        "metric_vs_fused_knn_agreement": nb_agree,
        "outputs": outputs, "code_dirty": dirty,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pandas": pd.__version__},
    }
    wjson(manifest, "phase_02_5_manifest_v1.json")
    (PH / "PHASE_STATE.json").write_text(json.dumps({
        "schema": "gaira_v7_phase_state_v1", "phase": PHASE, "phase_name": PHASE_NAME,
        "status": "COMPLETE", "analysis_only": True, "themes_created": False,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "atlas_fingerprint": fp_atlas,
        "lsm_registry_fingerprint": P01_FP, "csm_dictionary_fingerprint": P02_FP,
        "primary_spectral_metric": primary, "primary_geometry": winner,
        "neighbourhoods_computed_on": "D_primary_metric",
        "metric_vs_fused_knn_agreement": nb_agree,
        "n_priors": len(priors),
        "geometry_verdict": ("mixed" if len(set(region_tab.geometry_type)) > 1
                             else str(region_tab.geometry_type.iloc[0])),
        "modularity_z": mod["z"], "knn_coherence": obs_coh, "knn_coherence_p": p_coh,
        "n_single_source_motifs": len(single_src),
    }, indent=2))
    (LOGS / "phase02_5_run.log").write_text("\n".join(LOG))
    log("PHASE 02.5 COMPLETE")
    return 0


def build_priors(ids, idx_of, meta, classes, H, grid, Dfinal, Dprim, lid, gap, prop_tab,
                 rej, region_tab, roles, lsm2csm, conf_tab, single_src, merged_pairs,
                 csm_reg) -> list[dict]:
    """Turn validated geometry into provisional priors — never into themes."""
    from gaira.v7.geometry import neighbourhoods as N
    src_row = conf_tab[conf_tab.label == "source"].iloc[0]
    priors = []

    # (a) one prior per rejected Phase 02 proposal, now described as geometry
    names = {"proposal00": "lipid_superfamily", "proposal03": "polar_skeletal_backbone",
             "proposal16": "heterocyclic_ring_system"}
    for r in rej.itertuples():
        mem = [idx_of[m] for m in r.contributing_lsms.split(";")]
        g = prop_tab[prop_tab.proposal == r.proposed_group].iloc[0]
        srcs = {s for i in mem for s in [ids[i]] if s in single_src}
        conf = float(np.clip(0.35 + 0.25 * (g.separation_ratio - 1.0)
                             + 0.2 * g.internal_stability, 0.05, 0.95))
        priors.append(N.build_prior(
            f"prior_{names.get(r.proposed_group, r.proposed_group)}",
            names.get(r.proposed_group, r.proposed_group), mem, ids, lsm2csm, meta, H, grid,
            str(g.geometry_type),
            {"tier": "strong" if g.separation_ratio > 1.15 else "moderate",
             "separation_ratio": float(g.separation_ratio),
             "conductance": float(g.conductance),
             "mean_local_dimension": float(g.mean_local_dimension),
             "internal_stability": float(g.internal_stability),
             "phase02_verdict": "rejected as a merge — reconstruction cost exceeded tolerance",
             "phase02_isolated_ev_cost": float(r.isolated_ev_cost),
             "source_confounding": (f"{len(srcs)}/{len(mem)} members single-source"),
             "confidence": conf},
            [[a, b] for k, a in enumerate(r.contributing_lsms.split(";"))
             for b in r.contributing_lsms.split(";")[k + 1:]],
            (f"Phase 02 rejected this as a single motif at a cost of "
             f"{-r.isolated_ev_cost:.3f} EV. Phase 02.5 finds it is a coherent "
             f"{g.geometry_type} neighbourhood. It is a THEME candidate, not a motif: soft "
             f"overlapping membership does not incur the reconstruction cost a merge does.")))

    # (b) the accepted Phase 02 equivalence, carried forward as a fixed anchor
    for c in csm_reg["csms"]:
        if c["n_lsms"] > 1:
            mem = [idx_of[l["lsm_id"]] for l in c["contributing_lsms"]]
            priors.append(N.build_prior(
                "prior_cis_unsaturation", "cis_unsaturation", mem, ids, lsm2csm, meta, H, grid,
                "discrete",
                {"tier": "established", "phase02_verdict": "ACCEPTED equivalence",
                 "cohesion": c["cohesion"], "uncertainty": c["uncertainty"],
                 "source_confounding": "flagged in Phase 02; defeated by the 532/1064 nm split",
                 "confidence": 0.9},
                [], "The one Phase 02 equivalence. Already a single CSM; Phase 03 must not "
                    "split it."))

    # (c) data-driven regions that are not one of the above
    covered = {m for p in priors for m in p["supporting_lsms"]}
    for r in region_tab.itertuples():
        mem_ids = r.members.split(";")
        if len(mem_ids) < 3 or len(set(mem_ids) - covered) < 3:
            continue
        mem = [idx_of[m] for m in mem_ids]
        cls = sorted({classes[i] for i in mem})
        nm = ("mixed_" + "_".join(c.split("_")[0] for c in cls[:2])) if len(cls) > 1 else cls[0]
        priors.append(N.build_prior(
            f"prior_region_{r.region}", f"geometry_region_{nm}", mem, ids, lsm2csm, meta, H,
            grid, str(r.geometry_type),
            {"tier": "exploratory", "conductance": float(r.conductance),
             "mean_local_dimension": float(r.mean_local_dimension),
             "internal_stability": float(r.internal_stability),
             "source_confounding": f"source PERMANOVA p = {src_row.permanova_p:.3f}",
             "confidence": float(np.clip(0.2 + 0.4 * r.internal_stability, 0.05, 0.7))},
            [], "Discovered by unsupervised community detection on the primary geometry; "
                "chemistry revealed only afterwards. Exploratory."))

    # (d) isolated motifs — explicitly NOT theme material
    iso = roles[roles.is_isolated].motif.tolist()
    if iso:
        priors.append(N.build_prior(
            "prior_isolated_diagnostic", "isolated_diagnostic_motifs",
            [idx_of[m] for m in iso], ids, lsm2csm, meta, H, grid, "discrete",
            {"tier": "established", "criterion": "kNN isolation above the 90th percentile",
             "source_confounding": "per-motif; see confounding_v1.csv", "confidence": 0.8},
            [[a, b] for k, a in enumerate(iso) for b in iso[k + 1:]],
            "These motifs have no close neighbour under the primary geometry. They are "
            "candidates for singleton themes or for exclusion from soft membership entirely — "
            "forcing them into a theme would be the L-03 failure mode, a motif borrowing "
            "foreign mass."))
    return priors


if __name__ == "__main__":
    raise SystemExit(main())
