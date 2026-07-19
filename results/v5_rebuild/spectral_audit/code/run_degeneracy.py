"""Ag-SERS analyte-discriminability test (READ-ONLY).

The audit found that Ag-SERS spectra of chemically distinct metabolites are ~95%
identical to one another (a dominant common component, consistent with the
citrate-capped Ag-colloid background / surface species). This script tests whether
ANY analyte-specific information survives:

  (1) leave-one-out 1-NN analyte identification from replicate spectra, per modality;
  (2) the same after projecting out the corpus-mean (common) component;
  (3) same-analyte vs different-analyte replicate similarity in the residual space.

If (2)/(3) show same-analyte >> different-analyte, analyte signal exists but is
buried; if not, the Ag-SERS arm is analytically degenerate.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.evidence import datasets as D

AUD = REPO / "results/v5_rebuild/spectral_audit"; TAB = AUD / "tables"


def cos_mat(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


def loo_1nn(X, labels):
    """Leave-one-out 1-NN analyte identification accuracy."""
    S = cos_mat(X, X); np.fill_diagonal(S, -np.inf)
    pred = labels[S.argmax(axis=1)]
    return float(np.mean(pred == labels))


def remove_common(X):
    """Project out the corpus-mean direction (the shared background component)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    m = Xn.mean(0); m /= (np.linalg.norm(m) + 1e-12)
    return Xn - np.outer(Xn @ m, m)


def same_vs_diff(X, labels):
    S = cos_mat(X, X); np.fill_diagonal(S, np.nan)
    same = labels[:, None] == labels[None, :]
    return float(np.nanmedian(S[same])), float(np.nanmedian(S[~same]))


def main():
    res = {}
    for pipe in ("A1_asls_savgol_l2", "A2_asls_savgol_snv"):
        d = D.build(pipe)
        sel = d.meta.analyte.isin(d.matched_analytes).values
        block = {}
        for mod in ("raman", "sers"):
            m = sel & (d.meta.modality == mod).values
            X = np.nan_to_num(d.X[m]); lab = d.meta[m].analyte.values
            n_cls = len(np.unique(lab))
            acc = loo_1nn(X, lab)
            Xr = remove_common(X)
            acc_r = loo_1nn(Xr, lab)
            s_raw, d_raw = same_vs_diff(X, lab)
            s_res, d_res = same_vs_diff(Xr, lab)
            block[mod] = {
                "n_spectra": int(m.sum()), "n_analytes": int(n_cls),
                "chance_acc": 1.0 / n_cls,
                "loo_1nn_acc": acc, "loo_1nn_acc_common_removed": acc_r,
                "same_analyte_cos": s_raw, "diff_analyte_cos": d_raw,
                "separation_raw": s_raw - d_raw,
                "same_analyte_cos_residual": s_res, "diff_analyte_cos_residual": d_res,
                "separation_residual": s_res - d_res,
            }
        res[pipe] = block
    (TAB / "sers_degeneracy_test.json").write_text(json.dumps(res, indent=2))

    print("=== Ag-SERS vs Raman ANALYTE DISCRIMINABILITY ===")
    for pipe, block in res.items():
        print(f"\n[{pipe}]")
        print(f"  {'':6s} {'1NN acc':>9s} {'1NN(res)':>9s} {'chance':>8s} "
              f"{'same-cos':>9s} {'diff-cos':>9s} {'sep':>7s} {'sep(res)':>9s}")
        for mod, b in block.items():
            print(f"  {mod:6s} {b['loo_1nn_acc']:9.3f} {b['loo_1nn_acc_common_removed']:9.3f} "
                  f"{b['chance_acc']:8.3f} {b['same_analyte_cos']:9.3f} {b['diff_analyte_cos']:9.3f} "
                  f"{b['separation_raw']:+7.3f} {b['separation_residual']:+9.3f}")
    return res


if __name__ == "__main__":
    main()
