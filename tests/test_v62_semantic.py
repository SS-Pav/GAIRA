"""GAIRA V6.2 regression tests.

Prove (a) the frozen atlas and the V6 MSS layer are untouched, and (b) every V6.2
module satisfies the contract it claims: soft, non-negative, row-stochastic, sparse,
deterministic, and non-circular.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v6_rebuild/code"))

CANON = "09ed804a40836f4a05a91ba10900cded"
V6 = REPO / "results/v6_rebuild"
ASSETS = REPO / "assets/foundation"
needs = pytest.mark.skipif(not (V6 / "artifacts/v62_soft_hierarchy.json").exists(),
                           reason="V6.2 artifacts not built")


# ── nothing frozen moved ──
def test_atlas_fingerprint_unchanged():
    c = np.load(ASSETS / "manifold_components.npz")["components"]
    assert hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest()[:32] == CANON


def test_frozen_assets_match_manifest():
    man = json.loads((ASSETS / "MANIFEST.json").read_text())
    for name, rec in man["files"].items():
        b = (ASSETS / name).read_bytes()
        assert len(b) == rec["bytes"] and hashlib.sha256(b).hexdigest() == rec["sha256"], name


def test_v6_mss_layer_is_frozen_input_not_rewritten():
    """V6.2 must read the V6 motif spec and never modify it."""
    spec = yaml.safe_load((V6 / "artifacts/mss_motifs_v6.yaml").read_text())
    assert spec["version"] == "mss_v6"
    assert spec["derivation"]["theme_evidence_used"] is False
    assert len([m for m in spec["motifs"] if not m.get("non_biochemical")]) == 17
    # no V6.2 module may open the motif spec for writing
    for f in (V6 / "code/v62").glob("*.py"):
        t = f.read_text()
        assert 'mss_motifs_v6.yaml", "w' not in t and "mss_motifs_v6.yaml', 'w" not in t, f.name
    assert "mss_motifs_v6.yaml" in (V6 / "code/v62/core.py").read_text()   # it is read
    # and the spec's own hash is stable against the committed copy
    import subprocess
    r = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                        "results/v6_rebuild/artifacts/mss_motifs_v6.yaml"],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "", "the V6 motif spec was modified"


def test_v6_explorer_apps_untouched():
    for v in ("", "_v2", "_v3", "_v4", "_v5", "_v6"):
        assert (REPO / f"gaira_foundation_explorer{v}/app.py").exists()
    assert "detection gate" in (REPO / "gaira_foundation_explorer_v6/app.py").read_text()[:400].lower()


# ── the soft membership contract ──
@needs
def test_membership_is_nonnegative_rowstochastic_and_sparse():
    Z = np.load(V6 / "artifacts/v62_membership.npz", allow_pickle=True)
    for key in ("S_L1", "S_L2", "S_L3", "S_learned"):
        S = Z[key]
        assert (S >= 0).all(), key
        assert np.allclose(S.sum(1), 1.0, atol=1e-6), key
        assert ((S > 0) & (S < 0.02)).sum() == 0, f"{key}: weights below the floor survived"
    S2 = Z["S_L2"]
    assert float((S2 > 0).sum(1).mean()) <= 2.5, "membership is not sparse"


@needs
def test_membership_yaml_matches_the_arrays():
    y = yaml.safe_load((V6 / "artifacts/theme_membership.yaml").read_text())
    assert y["atlas_fingerprint"] == CANON
    Z = np.load(V6 / "artifacts/v62_membership.npz", allow_pickle=True)
    S2, names = Z["S_L2"], [str(x) for x in Z["L2_names"]]
    ids = [str(x) for x in Z["motif_ids"]]
    lvl = y["levels"]["L2_medium"]
    assert lvl["K"] == S2.shape[1]
    for i, m in enumerate(ids):
        rec = lvl["motifs"][m]
        assert abs(sum(rec["theme_weights"].values()) - 1.0) < 0.01, m
        assert rec["dominant_theme"] == names[int(np.argmax(S2[i]))], m


@needs
def test_soft_membership_is_deterministic():
    from v62 import core as C
    Z = np.load(V6 / "artifacts/v62_membership.npz", allow_pickle=True)
    A, ids = Z["A"], [str(x) for x in Z["motif_ids"]]
    y = yaml.safe_load((V6 / "artifacts/theme_membership.yaml").read_text())
    groups = [t["seed_motifs"] for t in y["levels"]["L2_medium"]["themes"]]
    tau = y["temperature"]
    S1, _ = C.soft_membership(A, groups, ids, temperature=tau)
    S2, _ = C.soft_membership(A, groups, ids, temperature=tau)
    assert np.allclose(S1, S2)
    assert np.allclose(S1, Z["S_L2"], atol=1e-9), "stored membership is not reproducible"


# ── the hierarchy contract ──
@needs
def test_levels_cover_every_motif_exactly_once_as_seeds():
    j = json.loads((V6 / "artifacts/v62_soft_hierarchy.json").read_text())
    ids = set()
    for lv in j["levels"].values():
        members = [m for g in lv["groups"] for m in g]
        assert len(members) == len(set(members)), lv
        ids = ids or set(members)
        assert set(members) == ids, "levels do not cover the same motif set"
    assert len(ids) == 17


@needs
def test_theme_chain_composes_as_two_linear_maps():
    from v62 import core as C
    Z = np.load(V6 / "artifacts/v62_membership.npz", allow_pickle=True)
    A, S2, M = Z["A"], Z["S_L2"], Z["M"]
    rng = np.random.default_rng(0)
    coord = rng.random(24); coord /= coord.sum()
    mss = M.T @ coord
    theme = S2.T @ mss
    assert (theme >= 0).all()
    assert np.allclose(theme, (M @ S2).T @ coord), "theme != S^T M^T coord"
    post = C.theme_posterior(mss, S2)
    assert abs(post["posterior"].sum() - 1.0) < 1e-6
    assert 0.0 <= float(post["entropy"][0]) <= 1.0


# ── information + graph contracts ──
@needs
def test_information_bottleneck_is_monotone_and_complete():
    import pandas as pd
    ib = pd.read_csv(V6 / "tables/v62_information_bottleneck.csv")
    hy = ib[ib.grouping == "hybrid_clustering"].sort_values("K")
    assert list(hy.K) == list(range(2, 18))
    ev = hy.explained_variance_motif.values
    assert np.all(np.diff(ev) > -1e-9), "variance retained must not decrease with K"
    assert abs(ev[-1] - 1.0) < 1e-6, "K=17 must reconstruct the motif layer exactly"
    assert (hy.reconstruction_error >= 0).all()


@needs
def test_ontology_graph_is_a_multi_parent_dag():
    import pandas as pd
    import networkx as nx
    e = pd.read_csv(V6 / "tables/v62_graph_edges.csv")
    G = nx.DiGraph()
    for _, r in e.iterrows():
        G.add_edge(r.source, r.target, weight=r.weight)
    assert nx.is_directed_acyclic_graph(G)
    j = json.loads((V6 / "artifacts/v62_information_graph.json").read_text())
    assert j["ontology_graph"]["n_multi_parent_motifs"] > 0, "a graph with no multi-parent node is a tree"
    assert (e.weight > 0).all()


@needs
def test_uncertainty_propagation_variances_are_psd():
    import pandas as pd
    p = pd.read_csv(V6 / "tables/v62_uncertainty_propagation.csv")
    assert (p.coord_total_var >= 0).all()
    assert (p.mss_total_var >= 0).all()
    assert (p.theme_total_var >= 0).all()
    assert (p.n_replicates >= 2).all()


@needs
def test_continuous_embedding_shape_and_nonnegativity():
    E = np.load(V6 / "artifacts/theme_embedding.npy")
    Z = np.load(V6 / "artifacts/v62_membership.npz", allow_pickle=True)
    assert E.shape == (len(Z["analytes"]), Z["S_L2"].shape[1])
    assert (E >= 0).all(), "theme coordinates must be non-negative"


@needs
def test_pareto_front_is_nondominated():
    import pandas as pd
    P = pd.read_csv(V6 / "tables/v62_pareto.csv")
    obj = ["interpretability", "information_retained", "recoverability", "stability"]
    pts = P[obj].values
    for i in np.where(P.pareto.values)[0]:
        for j in range(len(P)):
            if i == j:
                continue
            assert not ((pts[j] >= pts[i]).all() and (pts[j] > pts[i]).any()), \
                f"row {i} marked Pareto but dominated by {j}"


@needs
def test_all_declared_artifacts_exist():
    for p in ("artifacts/theme_membership.yaml", "artifacts/theme_embedding.npy",
              "artifacts/v62_membership.npz", "artifacts/v62_spaces.npz",
              "artifacts/v62_soft_hierarchy.json", "artifacts/v62_information_graph.json",
              "figures_v62/v62_ontology_graph.html"):
        assert (V6 / p).exists(), p
    for i in range(1, 11):
        hits = list((V6 / "reports").glob(f"V62_{i:02d}_*.pdf"))
        assert hits, f"report {i} missing"
