"""Encoder interpretability (§18). Two complementary attribution methods plus a
stability assessment. An encoder is an OBSERVATION representation; attribution
indicates which spectral regions drive the embedding, NOT molecular assignments.
"""
from __future__ import annotations
import numpy as np
import torch

MOD = {"raman": 0, "sers": 1}


def _forward(model, arch, x, modality):
    if arch == "shared":
        return model(x, normalize=False)
    mod = torch.tensor([MOD[modality]] * x.shape[0], dtype=torch.long)
    return model(x, mod, normalize=False)


def input_gradient_attribution(rep, X, modality):
    """|d(embedding energy)/d(input)| per wavenumber, averaged over rows. Label-free."""
    model = rep.model
    xt = torch.from_numpy(np.nan_to_num(np.atleast_2d(X)).astype(np.float32)).clone().requires_grad_(True)
    z = _forward(model, rep.arch, xt, modality)
    energy = (z ** 2).sum()
    energy.backward()
    g = xt.grad.abs().mean(dim=0).detach().numpy()
    return g / (g.max() + 1e-12)


def occlusion_attribution(rep, X, modality, window=16, stride=8):
    """Embedding shift (1 - cos) when a contiguous spectral window is zeroed."""
    import torch.nn.functional as F
    model = rep.model
    xt = torch.from_numpy(np.nan_to_num(np.atleast_2d(X)).astype(np.float32))
    with torch.no_grad():
        z0 = F.normalize(_forward(model, rep.arch, xt, modality), dim=1)
        L = xt.shape[1]
        imp = np.zeros(L)
        for st in range(0, L, stride):
            xo = xt.clone(); xo[:, st:st + window] = 0.0
            zo = F.normalize(_forward(model, rep.arch, xo, modality), dim=1)
            shift = (1 - (z0 * zo).sum(dim=1)).mean().item()
            imp[st:st + window] += shift
    return imp / (imp.max() + 1e-12)


def attribution_stability(rep, X_by_group, modality, method="occlusion"):
    """Given a dict analyte->stacked replicate spectra, measure attribution agreement
    (mean pairwise correlation) across replicates of the same analyte."""
    fn = occlusion_attribution if method == "occlusion" else input_gradient_attribution
    corrs = []
    for a, Xg in X_by_group.items():
        if len(Xg) < 2:
            continue
        atts = np.vstack([fn(rep, Xg[i:i + 1], modality) for i in range(len(Xg))])
        cc = np.corrcoef(atts)
        iu = np.triu_indices(len(Xg), 1)
        corrs.append(float(np.nanmean(cc[iu])))
    return {"mean_replicate_attribution_corr": float(np.mean(corrs)) if corrs else None,
            "n_groups": len(corrs), "method": method}
