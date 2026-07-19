"""Emit the versions manifest and example Part-12 inference outputs."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.engine import GAIRAEngine, VERSIONS
from gaira.engine.versioning import LAYER_INDEPENDENCE
from gaira.foundation import dataset as DS

OUT = REPO / "results/v5_rebuild/engine_v1/artifacts"
PROJ = REPO / "results/v5_rebuild/spike_validation/tables"
K = 24


def main():
    eng = GAIRAEngine()
    (OUT / "versions_manifest.json").write_text(json.dumps(
        {"versions": VERSIONS.as_dict(), "layer_independence": LAYER_INDEPENDENCE,
         "principle": "The ontology can evolve without changing the frozen Raman coordinates."},
        indent=2))

    corpus = DS.load_reference_corpus()
    examples = {}
    # pure reference examples
    for a, dom in [("adenine", "buffer"), ("cholesterol", "buffer"), ("(+)-glucose", "buffer"),
                   ("albumin", "buffer")]:
        mask = corpus.meta.analyte.values == a
        if not mask.any():
            continue
        coords = eng.atlas.coordinates(corpus.X[mask], normalise=True).mean(0)
        examples[f"pure::{a}"] = eng.infer(coordinates=coords, domain=dom).as_dict()
    # a serum spike example, read in serum domain
    df = pd.read_csv(PROJ / "phase3_projection_spiked_serum.csv")
    Zs = df[[f"c{j}" for j in range(K)]].values
    for a in ["hypoxanthine", "phenylalanine"]:
        idx = df.index[df.analyte == a]
        if len(idx):
            coords = np.nan_to_num(Zs[idx]).mean(0)
            examples[f"serum_spike::{a}"] = eng.infer(coordinates=coords, domain="serum").as_dict()

    (OUT / "example_inferences.json").write_text(json.dumps(examples, indent=2, default=float))
    print("versions_manifest.json + example_inferences.json written")
    for k, v in examples.items():
        top = sorted(v["biochemical_state_vector"]["composition"].items(), key=lambda x: -x[1])
        top_bio = [(t, round(s, 2)) for t, s in top if t not in ("background_matrix", "unknown_mixed")][:3]
        print(f"  {k:26s} OOD {v['ood_score']:.2f} conf {v['overall_confidence']:.2f} | top {top_bio}")


if __name__ == "__main__":
    main()
