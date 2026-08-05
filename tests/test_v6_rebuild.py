"""GAIRA V6 regression tests.

Two jobs: prove V6 changed nothing frozen, and prove the V6 layers are what they claim
(leakage-free, deterministic, non-circular).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v6_rebuild/code"))

CANON = "09ed804a40836f4a05a91ba10900cded"
V6 = REPO / "results/v6_rebuild"
ASSETS = REPO / "assets/foundation"
needs_v6 = pytest.mark.skipif(not (V6 / "artifacts/p7_evaluation.json").exists(),
                              reason="V6 artifacts not built")


# ── the frozen foundation is untouched ──
def test_atlas_fingerprint_unchanged():
    c = np.load(ASSETS / "manifold_components.npz")["components"]
    fp = hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest()[:32]
    assert fp == CANON
    assert json.loads((ASSETS / "manifold.json").read_text())["fingerprint"] == CANON


def test_every_frozen_asset_matches_its_manifest_hash():
    man = json.loads((ASSETS / "MANIFEST.json").read_text())
    assert man["atlas_fingerprint"] == CANON
    for name, rec in man["files"].items():
        b = (ASSETS / name).read_bytes()
        assert len(b) == rec["bytes"], name
        assert hashlib.sha256(b).hexdigest() == rec["sha256"], name


def test_engine_still_reproduces_committed_examples():
    from gaira.engine import GAIRAEngine
    from gaira.engine.ontology import NON_BIOCHEMICAL_THEMES
    ex = json.loads((REPO / "results/v5_rebuild/engine_v1/artifacts/example_inferences.json").read_text())
    eng = GAIRAEngine()
    assert eng.atlas.meta["fingerprint"] == CANON

    def top_bio(entry):
        comp = entry["biochemical_state_vector"]["composition"]
        bio = {t: s for t, s in comp.items() if t not in NON_BIOCHEMICAL_THEMES}
        return max(bio, key=bio.get)
    assert top_bio(ex["pure::adenine"]) == "nucleic_purine"
    assert top_bio(ex["pure::(+)-glucose"]) == "saccharide_glycan"


def test_v1_mss_layer_is_unmodified():
    """V6 must not have edited the shipped MSS layer."""
    src = (REPO / "src/gaira/engine/mss.py").read_text()
    assert "raw = self.wb * band + self.we * exemplar + self.wt * theme" in src
    spec = (ASSETS / "mss_motifs_v1.yaml").read_text()
    assert "band: 0.40, exemplar: 0.35, theme: 0.25" in spec.replace("{", "").replace("}", "")


def test_v6_writes_nothing_into_assets():
    names = {p.name for p in ASSETS.iterdir()}
    assert "mss_motifs_v6.yaml" not in names
    assert names == set(json.loads((ASSETS / "MANIFEST.json").read_text())["files"]) | \
        {"MANIFEST.json", "README.md"}


# ── the V6 layers are what they claim ──
@needs_v6
def test_v6_mss_has_no_theme_input():
    reg = json.loads((V6 / "artifacts/mss_registry_v6.json").read_text())
    assert reg["derivation"]["theme_evidence_used"] is False
    assert reg["atlas_fingerprint"] == CANON
    # check EXECUTABLE code, not the docstrings that explain what V1 did wrong
    import ast, io, tokenize
    path = V6 / "code/v6_semantic/mss_v6.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    code = ast.unparse(ast.fix_missing_locations(tree))
    for forbidden in ("parent_theme", "Ontology", "onto.W", "theme_index", "biochemical_ontology"):
        assert forbidden not in code, f"V6 MSS must not reference {forbidden} in executable code"


@needs_v6
def test_v6_mss_weights_normalised_and_sparse():
    reg = json.loads((V6 / "artifacts/mss_registry_v6.json").read_text())
    for m in reg["motifs"]:
        w = [c["weight"] for c in m["contributors"]]
        assert abs(sum(w) - 1.0) < 0.01, m["id"]
        assert len(w) <= reg["derivation"]["max_contributors"], m["id"]


@needs_v6
def test_v6_breadth_is_no_longer_constant():
    """The V1 bug (np.bool_ OR) pinned breadth at 1/3 for every motif."""
    reg = json.loads((V6 / "artifacts/mss_registry_v6.json").read_text())
    b = {round(m["evidence_breadth"], 4) for m in reg["motifs"]}
    assert len(b) > 1
    assert all(x > 0.34 for x in b)


@needs_v6
def test_theme_layer_is_a_partition_of_the_motifs():
    o = json.loads((V6 / "artifacts/p4_theme_optimisation.json").read_text())
    members = [m for t in o["selected_partition"]["themes"] for m in t["motifs"]]
    assert len(members) == len(set(members)), "a motif appears in two themes"
    assert set(members) == set(o["motif_ids"]), "partition does not cover the motifs"
    assert o["selected"]["chemically_admissible"] is True


@needs_v6
def test_v6_hierarchy_beats_its_permutation_null():
    o = json.loads((V6 / "artifacts/p4_theme_optimisation.json").read_text())
    s = o["selected"]
    assert s["top1"] > s["null_top1"]
    assert s["kappa"] > 0.4


@needs_v6
def test_v6_pipeline_is_deterministic_and_composes():
    from gaira.engine import GAIRAEngine
    from v6_semantic.mss_v6 import MSSLayerV6
    from v6_semantic import themes_v6 as TV
    eng = GAIRAEngine()
    v6 = MSSLayerV6(V6 / "artifacts/mss_motifs_v6.yaml", eng.builder.reg,
                    eng.atlas.components, eng.atlas.grid)
    o = json.loads((V6 / "artifacts/p4_theme_optimisation.json").read_text())
    bio_idx = [i for i, m in enumerate(v6.motifs) if not m.non_biochemical]
    L = TV.ThemeLayer([t["motifs"] for t in o["selected_partition"]["themes"]],
                      [v6.motifs[i].id for i in bio_idx])
    rng = np.random.default_rng(0)
    coord = rng.random(24); coord /= coord.sum()
    m1 = v6.activate(coord)[bio_idx]
    m2 = v6.activate(coord)[bio_idx]
    assert np.allclose(m1, m2)
    # theme = T^T M^T coord, and everything is non-negative
    th = L.compose(m1)[0]
    assert (th >= 0).all() and th.shape == (L.K,)
    assert np.allclose(th, m1 @ L.T)


@needs_v6
def test_v6_evaluation_covers_the_whole_corpus():
    e = json.loads((V6 / "artifacts/p7_evaluation.json").read_text())
    assert e["atlas_fingerprint"] == CANON
    assert e["n_analytes"] == 167
    assert e["n_labelled"] >= 160
    assert e["hierarchy"] == {"components": 24, "mss_motifs": 17, "chemical_themes": 13,
                              "themes": e["hierarchy"]["themes"]}


@needs_v6
def test_v6_explorer_pages_import():
    sys.path.insert(0, str(REPO / "gaira_semantic_explorer_v6"))
    from v6_ui.pages import PAGES
    from v6_ui import data as D
    assert len(PAGES) == 12
    h = D.headline()
    assert h["fingerprint"] == CANON and h["n_themes"] == 13


def test_legacy_explorers_untouched():
    """V6's semantic explorer is a NEW app; the six Foundation Explorers must be intact."""
    for v in ("", "_v2", "_v3", "_v4", "_v5", "_v6"):
        assert (REPO / f"gaira_foundation_explorer{v}/app.py").exists()
    # the pre-existing V6 detection-gate explorer must still be the detection-gate app
    head = (REPO / "gaira_foundation_explorer_v6/app.py").read_text()[:400]
    assert "detection gate" in head.lower()
    assert (REPO / "gaira_semantic_explorer_v6/app.py").exists()
