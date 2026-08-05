"""GAIRA V6.2 — Parts 4, 5, 7, 8, 9, 11, 12, 13.

Continuous theme space, information-bottleneck optimisation, the 17x17 recoverability
matrix, the theme manifold, the information-loss audit, the ontology graph, Bayesian
uncertainty propagation, and the multi-objective Pareto front.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "results/v6_rebuild/code"))
from v62 import core as C

OUT = REPO / "results/v6_rebuild"


def partition(D, K, ids):
    if K >= len(ids):
        return [[i] for i in ids]
    lab = fcluster(linkage(squareform(D, checks=False), method="average"), K, criterion="maxclust")
    g = {}
    for i, l in enumerate(lab):
        g.setdefault(int(l), []).append(ids[i])
    return [sorted(v) for v in g.values()]


def main():
    ctx = C.load_context()
    ids = ctx.motif_ids
    Z = np.load(C.art("v62_membership.npz"), allow_pickle=True)
    S2, S3 = Z["S_L2"], Z["S_L3"]
    L2n, L3n = list(Z["L2_names"]), list(Z["L3_names"])
    Dh = Z["D_hybrid"]
    A, fams, analytes = ctx.A, ctx.families, ctx.analytes
    sup = json.loads((C.art("v62_soft_hierarchy.json")).read_text())
    print(f"V6.2 information & graph · atlas {C.CANON}")

    # ══ PART 4 + 8 · continuous biochemical theme space ══
    E2 = A @ S2                                        # (n_analytes, K2) continuous
    E3 = A @ S3
    np.save(C.art("theme_embedding.npy"), E2)
    pca = PCA(n_components=min(4, E2.shape[1]), random_state=0).fit(E2)
    P2 = pca.transform(E2)
    try:
        import umap
        U2 = umap.UMAP(n_neighbors=15, min_dist=0.15, random_state=0,
                       n_components=2).fit_transform(E2)
        umap_ok = True
    except Exception as e:                                             # noqa: BLE001
        print("  umap unavailable:", e); U2 = P2[:, :2]; umap_ok = False
    # distance matrix in theme space
    En = E2 / (np.linalg.norm(E2, axis=1, keepdims=True) + C.EPS)
    Dtheme = 1 - En @ En.T

    # ══ PART 5 + 9 · information bottleneck over K ══
    rows = []
    for K in range(2, len(ids) + 1):
        g = partition(Dh, K, ids)
        S, _ = C.soft_membership(A, g, ids, temperature=sup["soft_parameters"]["temperature"])
        T = A @ S
        Ahat = C.reconstruct_from_theme(A, S)
        rows.append({
            "K": K,
            "explained_variance_motif": round(C.explained_variance(A, Ahat), 4),
            "reconstruction_error": round(float(np.linalg.norm(A - Ahat) /
                                                (np.linalg.norm(A) + C.EPS)), 4),
            "kl_divergence": round(C.kl_divergence(A, Ahat), 4),
            "mi_family_theme": round(C.mutual_information(fams, T), 4),
            "mi_family_motif": round(C.mutual_information(fams, A), 4),
            "compression_ratio": round(len(ids) / K, 3),
            "theme_entropy": round(float(C.norm_entropy(T, axis=1).mean()), 4),
            "membership_entropy": round(float(C.norm_entropy(S, axis=1).mean()), 4),
            "mean_support": round(float((S > 0).sum(1).mean()), 3),
            "chemically_admissible": bool(C.admissible(g, ctx.class_of)),
            "themes": " | ".join(sorted({m for gg in g for m in gg}) and
                                 [", ".join(x) for x in g])[:200],
        })
    # the ACTUAL V6.2 L2 grouping (chemical superclasses) is not produced by the hybrid
    # clustering at any K, so evaluate it explicitly as a labelled point in the sweep.
    g_l2 = [[m for m in ids if int(np.argmax(S2[ids.index(m)])) == t] for t in range(S2.shape[1])]
    g_l2 = [g for g in g_l2 if g]
    T_l2 = A @ S2
    Ahat_l2 = C.reconstruct_from_theme(A, S2)
    rows.append({
        "K": S2.shape[1],
        "explained_variance_motif": round(C.explained_variance(A, Ahat_l2), 4),
        "reconstruction_error": round(float(np.linalg.norm(A - Ahat_l2) /
                                            (np.linalg.norm(A) + C.EPS)), 4),
        "kl_divergence": round(C.kl_divergence(A, Ahat_l2), 4),
        "mi_family_theme": round(C.mutual_information(fams, T_l2), 4),
        "mi_family_motif": round(C.mutual_information(fams, A), 4),
        "compression_ratio": round(len(ids) / S2.shape[1], 3),
        "theme_entropy": round(float(C.norm_entropy(T_l2, axis=1).mean()), 4),
        "membership_entropy": round(float(C.norm_entropy(S2, axis=1).mean()), 4),
        "mean_support": round(float((S2 > 0).sum(1).mean()), 3),
        "chemically_admissible": True,
        "themes": "L2 chemical superclasses: " + " | ".join(L2n),
        "grouping": "L2_superclass",
    })
    ib = pd.DataFrame(rows)
    ib["grouping"] = ib.grouping.fillna("hybrid_clustering") if "grouping" in ib else "hybrid_clustering"
    ib["mi_retained"] = (ib.mi_family_theme / (ib.mi_family_motif + C.EPS)).round(4)
    # elbow: maximum curvature of explained variance vs K, computed on the SWEEP rows only
    # (the appended L2 point is a labelled comparison, not part of the monotone curve)
    hy = ib[ib.grouping == "hybrid_clustering"].sort_values("K")
    ev = hy.explained_variance_motif.values
    d2 = np.gradient(np.gradient(ev))
    elbow_K = int(hy.K.values[int(np.argmax(np.abs(d2[1:-1])) + 1)])
    ib.to_csv(C.tab("v62_information_bottleneck.csv"), index=False)

    # ══ PART 7 · 17 x 17 recoverability / confusion at the MOTIF level ══
    # For each analyte, does its expected motif rank first? Where does the mass go instead?
    from v6_semantic.mss_v6 import name_matches
    hits = [{i for i, a in enumerate(analytes) if any(name_matches(e, a) for e in m.exemplars)}
            for m in ctx.motifs]
    prim = []
    for i in range(len(analytes)):
        ms = [k for k in range(len(ids)) if i in hits[k]]
        prim.append(min(ms, key=lambda k: len(ctx.motifs[k].exemplars)) if ms else None)
    lab = np.array([p is not None for p in prim])
    R = np.zeros((len(ids), len(ids)))
    for i in np.where(lab)[0]:
        R[prim[i], int(np.argmax(A[i]))] += 1
    Rn = R / np.maximum(R.sum(1, keepdims=True), 1)
    pd.DataFrame(R.astype(int), index=ids, columns=ids).to_csv(C.tab("v62_recoverability_matrix.csv"))

    # why do motifs confuse? spectral + component overlap
    Sp = ctx.M.T @ ctx.H
    Sp = Sp / (np.linalg.norm(Sp, axis=1, keepdims=True) + C.EPS)
    spec_cos = Sp @ Sp.T
    Mn = ctx.M / (np.linalg.norm(ctx.M, axis=0, keepdims=True) + C.EPS)
    comp_cos = Mn.T @ Mn
    conf_rows = []
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j or R[i, j] == 0:
                continue
            bi = set(np.round(ctx.motifs[i].bands_cm).astype(int))
            bj = set(np.round(ctx.motifs[j].bands_cm).astype(int))
            close = sorted({b for b in bi for t in bj if abs(b - t) <= 25})
            conf_rows.append({
                "expected": ids[i], "predicted": ids[j], "n": int(R[i, j]),
                "rate": round(float(Rn[i, j]), 3),
                "spectral_cosine": round(float(spec_cos[i, j]), 3),
                "component_overlap": round(float(comp_cos[i, j]), 3),
                "shared_components": ", ".join(
                    f"c{k}" for k in np.argsort(-(ctx.M[:, i] * ctx.M[:, j]))[:3]
                    if ctx.M[k, i] > 0 and ctx.M[k, j] > 0) or "—",
                "overlapping_bands_cm": ", ".join(str(b) for b in close[:6]) or "—",
            })
    conf = pd.DataFrame(conf_rows).sort_values("n", ascending=False)
    conf.to_csv(C.tab("v62_confusion_explained.csv"), index=False)

    # ══ PART 12 · Bayesian uncertainty propagation ══
    prop = []
    for i, a in enumerate(analytes):
        cov_c = C.replicate_covariance(ctx.Zs, ctx.spec_analyte, a)
        if not cov_c.any():
            continue
        cov_m, cov_t = C.propagate(cov_c, ctx.M, S2)
        post = C.theme_posterior(A[i], S2)
        prop.append({
            "analyte": a, "family": fams[i],
            "n_replicates": int((ctx.spec_analyte == a).sum()),
            "coord_total_var": round(float(np.trace(cov_c)), 6),
            "mss_total_var": round(float(np.trace(cov_m)), 6),
            "theme_total_var": round(float(np.trace(cov_t)), 6),
            "coord_to_mss_ratio": round(float(np.trace(cov_m) / (np.trace(cov_c) + C.EPS)), 4),
            "mss_to_theme_ratio": round(float(np.trace(cov_t) / (np.trace(cov_m) + C.EPS)), 4),
            "theme_entropy": round(float(post["entropy"][0]), 4),
            "theme_confidence": round(float(post["confidence"][0]), 4),
            "theme_margin": round(float(post["margin"][0]), 4),
            "dominant_theme": L2n[int(post["top"][0])],
        })
    pr = pd.DataFrame(prop)
    pr.to_csv(C.tab("v62_uncertainty_propagation.csv"), index=False)

    # ══ PART 11 · ontology graph (a DAG with multiple parents) ══
    G = nx.DiGraph()
    n2 = [f"L2·{t}" for t in L2n]                 # namespaced: L2 and L3 can share a label
    n3 = [f"L3·{t}" for t in L3n]
    for m in ids:
        G.add_node(m, kind="motif", level=1, label=m, chemical_class=ctx.class_of[m])
    for t, lab in zip(n2, L2n):
        G.add_node(t, kind="theme_L2", level=2, label=lab)
    for t, lab in zip(n3, L3n):
        G.add_node(t, kind="system_L3", level=3, label=lab)
    for i, m in enumerate(ids):
        for t in range(S2.shape[1]):
            if S2[i, t] > 0:
                G.add_edge(m, n2[t], weight=float(S2[i, t]))
        src2 = n2[int(np.argmax(S2[i]))]
        for t in range(S3.shape[1]):
            if S3[i, t] > 0:
                prev = G.get_edge_data(src2, n3[t], {"weight": 0.0})["weight"]
                G.add_edge(src2, n3[t], weight=float(max(prev, S3[i, t])))
    und = G.to_undirected()
    btw = nx.betweenness_centrality(und, weight=None)
    eig = nx.eigenvector_centrality_numpy(und, weight="weight")
    try:
        comms = list(nx.community.greedy_modularity_communities(und, weight="weight"))
    except Exception:                                                  # noqa: BLE001
        comms = []
    comm_of = {n: i for i, c in enumerate(comms) for n in c}
    ent = {m: float(C.norm_entropy(S2[i])) for i, m in enumerate(ids)}
    gnodes = []
    for n, d in G.nodes(data=True):
        gnodes.append({"node": n, "label": d.get("label", n),
                       "kind": d["kind"], "level": d["level"],
                       "degree": int(G.degree(n)),
                       "in_degree": int(G.in_degree(n)), "out_degree": int(G.out_degree(n)),
                       "betweenness": round(float(btw.get(n, 0)), 4),
                       "eigenvector": round(float(eig.get(n, 0)), 4),
                       "community": int(comm_of.get(n, -1)),
                       "uncertainty": round(ent.get(n, float("nan")), 4)
                       if n in ent else None})
    gn = pd.DataFrame(gnodes).sort_values(["level", "betweenness"], ascending=[True, False])
    gn.to_csv(C.tab("v62_graph_nodes.csv"), index=False)
    ge = pd.DataFrame([{"source": u, "target": v, "weight": round(d["weight"], 4)}
                       for u, v, d in G.edges(data=True)])
    ge.to_csv(C.tab("v62_graph_edges.csv"), index=False)
    multi_parent = [m for m in ids if G.out_degree(m) > 1]

    # ══ PART 13 · multi-objective Pareto ══
    prows = []
    for _, r in ib[ib.grouping == 'hybrid_clustering'].sort_values('K').iterrows():
        K = int(r.K)
        g = partition(Dh, K, ids)
        S, _ = C.soft_membership(A, g, ids, temperature=sup["soft_parameters"]["temperature"])
        T = A @ S
        # interpretability: chemical coherence of each theme x resolution
        coh = []
        for gg in g:
            sc = {C.SUPERCLASS.get(ctx.class_of[m], "?") for m in gg}
            sc.discard("BRIDGING")
            coh.append(1.0 / max(1, len(sc)))
        interp = 0.7 * float(np.mean(coh)) + 0.3 * float(np.log(K) / np.log(len(ids)))
        # recoverability: does the analyte's expected theme rank first?
        idxm = {m: k for k, m in enumerate(ids)}
        ok, tot = 0, 0
        for i in np.where(lab)[0]:
            exp_t = {int(np.argmax(S[prim[i]]))}
            if int(np.argmax(T[i])) in exp_t:
                ok += 1
            tot += 1
        rec = ok / max(tot, 1)
        # stability: membership under a 20-analyte jackknife
        rng = np.random.default_rng(0)
        sims = []
        for _ in range(12):
            keep = rng.choice(len(analytes), int(0.8 * len(analytes)), replace=False)
            Sj, _ = C.soft_membership(A[keep], g, ids,
                                      temperature=sup["soft_parameters"]["temperature"])
            sims.append(float(np.mean(np.sum(S * Sj, 1) /
                                      (np.linalg.norm(S, axis=1) * np.linalg.norm(Sj, axis=1) + C.EPS))))
        stab = float(np.mean(sims))
        # calibration of the theme posterior
        post = C.theme_posterior(A, S)
        corr = np.array([int(np.argmax(T[i]) == int(np.argmax(S[prim[i]])))
                         for i in np.where(lab)[0]], float)
        cf = post["posterior"].max(1)[lab]
        bins = np.linspace(0, 1, 11); ece = 0.0
        for b in range(10):
            mm = (cf > bins[b]) & (cf <= bins[b + 1])
            if mm.sum():
                ece += mm.sum() / len(cf) * abs(corr[mm].mean() - cf[mm].mean())
        prows.append({"K": K, "interpretability": round(interp, 4),
                      "information_retained": float(r.explained_variance_motif),
                      "mi_retained": float(r.mi_retained),
                      "recoverability": round(rec, 4), "stability": round(stab, 4),
                      "compression": float(r.compression_ratio),
                      "calibration_ece": round(float(ece), 4),
                      "chemically_admissible": bool(r.chemically_admissible)})
    P = pd.DataFrame(prows)
    obj = ["interpretability", "information_retained", "recoverability", "stability"]
    pts = P[obj].values
    par = np.ones(len(P), bool)
    for i in range(len(P)):
        for j in range(len(P)):
            if i != j and (pts[j] >= pts[i]).all() and (pts[j] > pts[i]).any():
                par[i] = False; break
    P["pareto"] = par
    P.to_csv(C.tab("v62_pareto.csv"), index=False)

    summary = {
        "atlas_fingerprint": C.CANON,
        "continuous_space": {
            "dim": int(E2.shape[1]), "themes": L2n,
            "pca_explained_variance": [round(float(x), 4) for x in pca.explained_variance_ratio_],
            "umap_available": bool(umap_ok),
            "mean_pairwise_theme_distance": round(float(Dtheme[np.triu_indices(len(analytes), 1)].mean()), 4),
        },
        "information_bottleneck": {
            "elbow_K": elbow_K,
            "mi_family_motif": float(ib.mi_family_motif.iloc[0]),
            "at_elbow": ib[ib.K == elbow_K].to_dict("records")[0],
            "at_L2": ib[ib.K == S2.shape[1]].to_dict("records")[0] if (ib.K == S2.shape[1]).any() else None,
        },
        "recoverability": {
            "motif_top1": round(float(np.trace(R) / max(R.sum(), 1)), 4),
            "n_scored": int(R.sum()),
            "worst_motifs": [ids[i] for i in np.argsort(np.diag(Rn))[:5]],
            "top_confusions": conf.head(8).to_dict("records"),
        },
        "uncertainty_propagation": {
            "n_analytes": int(len(pr)),
            "median_coord_to_mss_ratio": round(float(pr.coord_to_mss_ratio.median()), 5),
            "median_mss_to_theme_ratio": round(float(pr.mss_to_theme_ratio.median()), 5),
            "median_theme_confidence": round(float(pr.theme_confidence.median()), 4),
            "note": "linear (delta-method) propagation of the empirical replicate covariance "
                    "through the two frozen linear maps",
        },
        "ontology_graph": {
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
            "n_multi_parent_motifs": len(multi_parent), "multi_parent_motifs": multi_parent,
            "is_dag": bool(nx.is_directed_acyclic_graph(G)),
            "note": "node ids are namespaced by level (L2·/L3·) because a chemical theme name can survive into the coarse level",
            "n_communities": len(comms),
            "highest_betweenness": gn.head(5)[["node", "kind", "betweenness"]].to_dict("records"),
        },
        "pareto": {
            "n_pareto": int(par.sum()),
            "front": P[P.pareto][["K", "interpretability", "information_retained",
                                  "recoverability", "stability", "compression",
                                  "chemically_admissible"]].to_dict("records"),
        },
    }
    C.dump_json(summary, "v62_information_graph.json")
    np.savez(C.art("v62_spaces.npz"), E2=E2, E3=E3, P2=P2, U2=U2, Dtheme=Dtheme,
             R=R, Rn=Rn, spec_cos=spec_cos, comp_cos=comp_cos,
             analytes=np.array(analytes), families=fams,
             L2_names=np.array(L2n), L3_names=np.array(L3n), motif_ids=np.array(ids),
             S2=S2, S3=S3, A=A, labelled=lab,
             primary=np.array([p if p is not None else -1 for p in prim]))

    pd.set_option("display.width", 250)
    print(f"\nPART 4/8 — continuous theme space: {E2.shape}, PCA var "
          f"{[round(float(x),3) for x in pca.explained_variance_ratio_]}, UMAP={umap_ok}")
    print(f"\nPART 5/9 — information bottleneck (elbow at K={elbow_K}):")
    print(ib[["K", "explained_variance_motif", "reconstruction_error", "kl_divergence",
              "mi_retained", "compression_ratio", "mean_support",
              "chemically_admissible"]].to_string(index=False))
    print(f"\nPART 7 — motif recoverability: diag {np.trace(R)/max(R.sum(),1):.3f} over "
          f"{int(R.sum())} analytes. Top confusions:")
    print(conf.head(8)[["expected", "predicted", "n", "spectral_cosine",
                        "component_overlap", "overlapping_bands_cm"]].to_string(index=False))
    print(f"\nPART 12 — uncertainty: median coord→MSS var ratio "
          f"{pr.coord_to_mss_ratio.median():.4f}, MSS→theme {pr.mss_to_theme_ratio.median():.4f}")
    print(f"\nPART 11 — graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(multi_parent)} multi-parent motifs, {len(comms)} communities")
    print(f"\nPART 13 — Pareto: {int(par.sum())} of {len(P)} non-dominated")
    print(P[P.pareto][["K", "interpretability", "information_retained", "recoverability",
                       "stability", "compression", "chemically_admissible"]].to_string(index=False))


if __name__ == "__main__":
    main()
