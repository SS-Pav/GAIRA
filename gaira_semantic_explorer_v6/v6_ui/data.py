"""Data access for Foundation Explorer V6. Reads only committed V6 artifacts."""
from __future__ import annotations
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "results/v6_rebuild"
FIGS = BASE / "figures"
FP = "09ed804a40836f4a05a91ba10900cded"


@lru_cache(maxsize=None)
def js(name):
    return json.loads((BASE / "artifacts" / name).read_text())


@lru_cache(maxsize=None)
def tb(name):
    return pd.read_csv(BASE / "tables" / name)


@lru_cache(maxsize=None)
def vectors():
    return np.load(BASE / "artifacts/p7_vectors.npz", allow_pickle=True)


@lru_cache(maxsize=None)
def mss_registry():
    return js("mss_registry_v6.json")


@lru_cache(maxsize=None)
def motif_spec():
    import yaml
    return yaml.safe_load((BASE / "artifacts/mss_motifs_v6.yaml").read_text())


def fig(name):
    p = FIGS / name
    return str(p) if p.exists() else None


@lru_cache(maxsize=None)
def engine():
    """The live frozen engine — used only by the interactive pipeline page."""
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(BASE / "code"))
    from gaira.engine import GAIRAEngine
    from v6_semantic.mss_v6 import MSSLayerV6
    from v6_semantic import themes_v6 as TV
    eng = GAIRAEngine()
    v6 = MSSLayerV6(BASE / "artifacts/mss_motifs_v6.yaml", eng.builder.reg,
                    eng.atlas.components, eng.atlas.grid)
    sel = js("p4_theme_optimisation.json")["selected_partition"]
    bio_idx = [i for i, m in enumerate(v6.motifs) if not m.non_biochemical]
    L = TV.ThemeLayer([t["motifs"] for t in sel["themes"]],
                      [v6.motifs[i].id for i in bio_idx])
    return eng, v6, L, bio_idx


def headline():
    e = js("p7_evaluation.json")
    o = js("p4_theme_optimisation.json")
    a = js("p0_p1_audit.json")
    return {
        "fingerprint": FP,
        "n_components": 24,
        "n_motifs": e["hierarchy"]["mss_motifs"],
        "n_themes": e["hierarchy"]["chemical_themes"],
        "theme_top1": e["theme_top1"], "theme_top3": e["theme_top3"],
        "motif_top1": e["motif_top1"], "motif_top3": e["motif_top3"],
        "ece": e["ece"], "n_analytes": e["n_analytes"], "n_labelled": e["n_labelled"],
        "method": o["selected"]["method"], "kappa": o["selected"]["kappa"],
        "null": o["selected"]["null_top1"],
        "interp": o["selected"]["interpretability"],
        "leak_mean": a["mean_theme_share_of_raw_score"],
        "leak_edges": a["n_edges_that_would_drop_below_keep_threshold"],
        "leak_total": a["n_contributor_edges"],
    }
