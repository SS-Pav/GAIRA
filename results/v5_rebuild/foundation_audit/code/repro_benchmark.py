"""Foundation audit — independently RE-RUN the C1 representation benchmark on the
Raman-only corpus and reproduce the k / representation selection.

Writes to results/v5_rebuild/foundation_audit/tables/ (NEVER overwrites the committed
foundation outputs). Compares the recomputed benchmark to the committed one and
re-applies the pre-stated tie-break to confirm NMF k=24 is selected.

    python results/v5_rebuild/foundation_audit/code/repro_benchmark.py
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.foundation import dataset as DS, benchmark as BM

OUT = REPO / "results/v5_rebuild/foundation_audit/tables"
OUT.mkdir(parents=True, exist_ok=True)
COMMITTED = REPO / "results/v5_rebuild/foundation/tables/c1_representation_benchmark.csv"


def main():
    t0 = time.time()
    c = DS.load_reference_corpus()
    print(f"corpus: {len(c.X)} spectra / {c.meta.analyte.nunique()} analytes / {c.X.shape[1]} bins",
          flush=True)
    df = BM.run_benchmark(c, ks=(4, 8, 12, 16, 24, 32),
                          names=("PCA", "SparseDict", "NMF", "ICA", "Autoencoder"),
                          n_splits=4, seed=0, verbose=True)
    scored = BM.score(df)
    scored.to_csv(OUT / "c1_representation_benchmark_repro.csv", index=False)

    pick, tied, why = BM.select_with_tiebreak(scored)
    sel = {"representation": pick.representation, "k": int(pick.k),
           "total_score": float(pick.total_score), "tie_break": why,
           "n_tied_within_tol": int(len(tied)),
           "raw_top": {"representation": scored.iloc[0].representation, "k": int(scored.iloc[0].k),
                       "total_score": float(scored.iloc[0].total_score)}}
    (OUT / "c1_selection_repro.json").write_text(json.dumps(sel, indent=2))

    # compare to committed benchmark
    verdict = {}
    if COMMITTED.exists():
        old = pd.read_csv(COMMITTED)
        key = ["representation", "k"]
        m = scored.merge(old, on=key, suffixes=("_new", "_old"))
        cols = ["total_score", "recon_rel_error", "neighbourhood_preservation",
                "replicate_robustness", "component_stability", "loading_sparsity"]
        maxdiff = {c: float(np.nanmax(np.abs(m[f"{c}_new"] - m[f"{c}_old"]))) for c in cols}
        # rank agreement on total_score
        rn = scored.sort_values("total_score", ascending=False).reset_index()[key]
        ro = old.sort_values("total_score", ascending=False).reset_index()[key]
        rank_match = bool((rn.values == ro.values).all())
        verdict = {"max_abs_diff_vs_committed": maxdiff, "full_ranking_identical": rank_match}
    (OUT / "c1_repro_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n=== SELECTION (reproduced) ===")
    print(f"  raw top: {sel['raw_top']['representation']} k={sel['raw_top']['k']} "
          f"({sel['raw_top']['total_score']:.4f})")
    print(f"  SELECTED after tie-break: {sel['representation']} k={sel['k']} "
          f"({sel['total_score']:.4f})")
    print(f"  tied within tol: {sel['n_tied_within_tol']}")
    if verdict:
        print(f"  full ranking identical to committed: {verdict['full_ranking_identical']}")
        print(f"  max abs diff vs committed: "
              f"{max(verdict['max_abs_diff_vs_committed'].values()):.2e}")
    print(f"runtime {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
