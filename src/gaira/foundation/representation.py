"""GAIRA V5 Foundation — candidate Raman representations (Phase C1).

One interface for every candidate: fit(X_train) -> transform(X) / reconstruct(X)
/ components. PCA is NOT privileged; the benchmark decides.

Reuses sklearn decompositions; the autoencoder is a small regularised 1-D model
included ONLY as a benchmark comparator (not a foundation-model claim).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from sklearn.decomposition import PCA, NMF, FastICA, MiniBatchDictionaryLearning, sparse_encode


@dataclass
class Representation:
    name: str
    k: int
    model: object = None
    components_: np.ndarray = None       # (k, n_bins) — spectral loadings
    mean_: np.ndarray = None
    params: dict = field(default_factory=dict)
    nonneg: bool = False

    # ── API ──
    def transform(self, X):
        raise NotImplementedError

    def reconstruct(self, X):
        Z = self.transform(X)
        return Z @ self.components_ + (self.mean_ if self.mean_ is not None else 0.0)

    def to_dict(self):
        return {"name": self.name, "k": int(self.k), "params": self.params,
                "nonneg": bool(self.nonneg),
                "n_components": int(0 if self.components_ is None else len(self.components_))}


class _PCA(Representation):
    def transform(self, X):
        return self.model.transform(np.nan_to_num(X))


class _ICA(Representation):
    def transform(self, X):
        return self.model.transform(np.nan_to_num(X))


class _NMF(Representation):
    def transform(self, X):
        return self.model.transform(np.clip(np.nan_to_num(X), 0, None))

    def reconstruct(self, X):
        return self.transform(X) @ self.components_


class _Dict(Representation):
    def transform(self, X):
        return sparse_encode(np.nan_to_num(X), self.components_,
                             algorithm="lasso_lars", alpha=self.params.get("alpha", 0.5))

    def reconstruct(self, X):
        return self.transform(X) @ self.components_


class _AE(Representation):
    def transform(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(torch.from_numpy(np.nan_to_num(X).astype(np.float32)))
        return z.numpy()

    def reconstruct(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(np.nan_to_num(X).astype(np.float32))
            return self.model(xt).numpy()


# ───────────────────────── fitting ─────────────────────────
def fit_pca(X, k, seed=0):
    m = PCA(n_components=k, random_state=seed).fit(np.nan_to_num(X))
    return _PCA("PCA", k, m, m.components_, m.mean_, {})


def fit_ica(X, k, seed=0):
    m = FastICA(n_components=k, random_state=seed, max_iter=1000, whiten="unit-variance")
    m.fit(np.nan_to_num(X))
    return _ICA("ICA", k, m, m.mixing_.T, m.mean_, {})


def fit_nmf(X, k, seed=0):
    Xc = np.clip(np.nan_to_num(X), 0, None)
    m = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=1500).fit(Xc)
    return _NMF("NMF", k, m, m.components_, None, {}, nonneg=True)


def fit_dictionary(X, k, seed=0, alpha=0.5):
    m = MiniBatchDictionaryLearning(n_components=k, alpha=alpha, max_iter=300,
                                    random_state=seed, transform_algorithm="lasso_lars",
                                    transform_alpha=alpha).fit(np.nan_to_num(X))
    return _Dict("SparseDict", k, m, m.components_, None, {"alpha": alpha})


def fit_autoencoder(X, k, seed=0, epochs=400, lr=2e-3, wd=1e-4):
    """Small regularised autoencoder — benchmark comparator only."""
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    d = X.shape[1]
    Xt = torch.from_numpy(np.nan_to_num(X).astype(np.float32))

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, k))
            self.dec = nn.Sequential(nn.Linear(k, 256), nn.GELU(), nn.Linear(256, d))

        def encode(self, x):
            return self.enc(x)

        def forward(self, x):
            return self.dec(self.enc(x))

    m = AE()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        loss = ((m(Xt) - Xt) ** 2).mean()
        loss.backward(); opt.step()
        L = float(loss.detach())
        if L < best - 1e-6:
            best, wait = L, 0
            best_state = {kk: v.detach().clone() for kk, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 40:
                break
    if best_state:
        m.load_state_dict(best_state)
    # decoder weights of the last linear layer act as spectral loadings
    W = m.dec[-1].weight.detach().numpy().T          # (k, d)
    return _AE("Autoencoder", k, m, W, None, {"epochs": ep + 1, "final_mse": best})


FITTERS = {"PCA": fit_pca, "SparseDict": fit_dictionary, "NMF": fit_nmf,
           "ICA": fit_ica, "Autoencoder": fit_autoencoder}
