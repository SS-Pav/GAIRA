"""Does the shared Ag-SERS background explain the cross-modal failure? (READ-ONLY)

The audit established that Ag-SERS spectra carry real analyte identity (LOO 1-NN
0.73 vs 0.02 chance) but as a small residual on a dominant common component that
has no Raman counterpart. This script tests the direct implication: if the common
component is projected out of each modality, does MATCHED cross-modal similarity
separate from MISMATCHED?

This is diagnostic linear algebra on already-computed spectra — no model is fitted,
no preprocessing is altered, nothing is written back into GAIRA.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.evidence import datasets as D

AUD = REPO / "results/v5_rebuild/spectral_audit"; TAB = AUD / "tables"


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def strip_common(X, n_comp=1):
    """Project out the top-n shared directions (corpus mean + leading PCs)."""
    Xn = unit(X)
    m = Xn.mean(0); m /= (np.linalg.norm(m) + 1e-12)
    Y = Xn - np.outer(Xn @ m, m)
    if n_comp > 1:
        U, S, Vt = np.linalg.svd(Y - Y.mean(0), full_matrices=False)
        for k in range(n_comp - 1):
            v = Vt[k] / (np.linalg.norm(Vt[k]) + 1e-12)
            Y = Y - np.outer(Y @ v, v)
    return Y


def retrieval(Rm, Sm):
    """Top-1 / MRR for Raman->SERS retrieval among the matched analyte set."""
    S = unit(Rm) @ unit(Sm).T
    n = S.shape[0]
    ranks = np.array([1 + int((S[i] > S[i, i]).sum()) for i in range(n)])
    return {"top1": float(np.mean(ranks == 1)), "mrr": float(np.mean(1.0 / ranks)),
            "matched_cos": float(np.mean(np.diag(S))),
            "mismatched_cos": float((S.sum() - np.trace(S)) / (n * n - n)),
            "chance_top1": 1.0 / n}


def main():
    out = {}
    for pipe in ("A1_asls_savgol_l2", "A2_asls_savgol_snv"):
        d = D.build(pipe)
        A = d.matched_analytes
        Rm = np.vstack([np.nan_to_num(d.X[((d.meta.analyte == a) & (d.meta.modality == "raman")).values]).mean(0) for a in A])
        Sm = np.vstack([np.nan_to_num(d.X[((d.meta.analyte == a) & (d.meta.modality == "sers")).values]).mean(0) for a in A])
        block = {"as_evaluated": retrieval(Rm, Sm)}
        for k in (1, 2, 3, 5):
            block[f"common_removed_{k}"] = retrieval(strip_common(Rm, k), strip_common(Sm, k))
        out[pipe] = block
    (TAB / "background_removal_test.json").write_text(json.dumps(out, indent=2))

    print("=== CROSS-MODAL RETRIEVAL BEFORE/AFTER REMOVING THE SHARED COMPONENT ===")
    print("(51 matched analytes; chance top-1 = 0.020)")
    for pipe, block in out.items():
        print(f"\n[{pipe}]")
        print(f"  {'condition':22s} {'top1':>7s} {'MRR':>7s} {'matched':>9s} {'mismatch':>9s} {'sep':>8s}")
        for cond, r in block.items():
            print(f"  {cond:22s} {r['top1']:7.3f} {r['mrr']:7.3f} {r['matched_cos']:9.3f} "
                  f"{r['mismatched_cos']:9.3f} {r['matched_cos']-r['mismatched_cos']:+8.3f}")
    return out


if __name__ == "__main__":
    main()
