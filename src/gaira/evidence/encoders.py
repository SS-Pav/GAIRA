"""Small, regularized 1D-CNN encoders (§8). NO transformer, NO pretraining.

Architecture sized for a tiny corpus: 2-3 conv blocks, modest channels, global
average pooling, linear projection to a small latent dim, L2-normalized output.
Strong weight decay + dropout + early stopping are applied by the trainer.

E1 shared      : one encoder for both modalities.
E2 dual        : separate Raman and Ag-SERS encoders into one shared latent dim.
E3 modality-specific: same as E2 but trained without any cross-modal objective.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class Encoder1D(nn.Module):
    def __init__(self, n_bins, latent=16, channels=(16, 32), kernel=9, dropout=0.1):
        super().__init__()
        layers = []
        c_in = 1
        for c in channels:
            layers += [nn.Conv1d(c_in, c, kernel, padding=kernel // 2),
                       nn.BatchNorm1d(c), nn.GELU(),
                       nn.MaxPool1d(2), nn.Dropout(dropout)]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(c_in, latent))
        self.latent = latent

    def forward(self, x, normalize=True):
        if x.dim() == 2:
            x = x.unsqueeze(1)                 # (B,1,L)
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        z = self.head(h)
        if normalize:
            z = nn.functional.normalize(z, dim=1)
        return z


class DualEncoder(nn.Module):
    """Two encoders (raman, sers) mapping into a shared latent dim."""
    def __init__(self, n_bins, latent=16, **kw):
        super().__init__()
        self.raman = Encoder1D(n_bins, latent, **kw)
        self.sers = Encoder1D(n_bins, latent, **kw)
        self.latent = latent

    def forward(self, x, modality, normalize=True):
        # modality: (B,) 0=raman 1=sers
        out = torch.empty(x.shape[0], self.latent, device=x.device)
        ram = modality == 0; ser = modality == 1
        if ram.any():
            out[ram] = self.raman(x[ram], normalize=normalize)
        if ser.any():
            out[ser] = self.sers(x[ser], normalize=normalize)
        return out


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))
