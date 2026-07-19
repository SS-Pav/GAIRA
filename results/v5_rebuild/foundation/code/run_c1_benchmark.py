"""Phase C1 — Raman-only representation benchmark."""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.foundation import dataset as DS, benchmark as BM

OUT = REPO / "results/v5_rebuild/foundation"
TAB, CFG = OUT / "tables", OUT / "configs"
for p in (TAB, CFG): p.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    c = DS.load_reference_corpus()
    card = DS.dataset_card(c)
    (TAB / "raman_dataset_card.json").write_text(json.dumps(card, indent=2, default=str))
    print(f"Raman corpus: {card['n_spectra']} spectra | {card['n_analytes']} analytes | "
          f"{card['n_bins']} bins | window {card['window_cm']}", flush=True)

    df = BM.run_benchmark(c, ks=(4, 8, 12, 16, 24, 32), n_splits=4, seed=0)
    scored = BM.score(df)
    scored.to_csv(TAB / "c1_representation_benchmark.csv", index=False)
    (CFG / "c1_selection_weights.json").write_text(json.dumps(BM.SELECTION_WEIGHTS, indent=2))

    best = scored.iloc[0]
    print("\n=== TOP 8 (multi-criteria; reconstruction deliberately only 10%) ===")
    cols = ["representation", "k", "total_score", "recon_rel_error", "neighbourhood_preservation",
            "replicate_robustness", "component_stability", "loading_sparsity", "excitation_leakage"]
    print(scored[cols].head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== best per representation ===")
    print(scored.sort_values("total_score", ascending=False).groupby("representation").head(1)[cols]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    sel = {"representation": best.representation, "k": int(best.k),
           "total_score": float(best.total_score),
           "metrics": {k: float(best[k]) for k in
                       ("recon_rel_error", "neighbourhood_preservation", "replicate_robustness",
                        "component_stability", "loading_sparsity", "band_localisation",
                        "excitation_leakage", "source_leakage")}}
    (TAB / "c1_selection.json").write_text(json.dumps(sel, indent=2))
    print(f"\nSELECTED: {sel['representation']} k={sel['k']}  (score {sel['total_score']:.3f})")
    print(f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
