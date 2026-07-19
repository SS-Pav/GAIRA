"""Sparse evidence projection (§14, §18).

Two uses:
  (1) hybrid: project an encoder embedding onto a sparse, transparent set of
      interpretable evidence activations;
  (2) interpretability: fit a sparse LINEAR probe from an embedding to interpretable
      features (e.g. region activations), quantifying how much interpretable evidence
      the embedding linearly encodes (R^2), with sparse, inspectable coefficients.
Fitted on TRAINING data only.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score


def sparse_linear_probe(Z_train, T_train, Z_eval, T_eval, alpha=0.01):
    """Predict interpretable target T from embedding Z with a sparse linear map.
    Returns per-target R^2 on eval + coefficient sparsity."""
    m = Lasso(alpha=alpha, max_iter=5000)
    m.fit(Z_train, T_train)
    pred = m.predict(Z_eval)
    if pred.ndim == 1:
        pred = pred[:, None]; T_eval = np.atleast_2d(T_eval).reshape(len(T_eval), -1)
    r2 = float(r2_score(T_eval, pred, multioutput="variance_weighted"))
    coef = np.atleast_2d(m.coef_)
    nz = float(np.mean(np.abs(coef) > 1e-8))
    return {"r2": r2, "nonzero_fraction": nz, "alpha": alpha,
            "coef_shape": list(coef.shape)}
