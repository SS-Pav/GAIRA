"""Encoder training (§8-9). Deterministic, small, regularized, early-stopped.

Full-batch training on the (tiny) training split with per-epoch bounded augmentation.
Objectives are a weighted sum from losses.py. Early stopping on validation loss.
Returns an EncoderRepresentation (wraps the trained torch model, transforms numpy).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch

from .base import Representation
from .encoders import Encoder1D, DualEncoder, n_params
from . import losses as L
from .augmentations import AugConfig, augment

MOD = {"raman": 0, "sers": 1}


@dataclass
class EncoderConfig:
    name: str
    arch: str = "dual"                     # "shared" | "dual"
    latent: int = 16
    channels: tuple = (16, 32)
    dropout: float = 0.1
    epochs: int = 70
    lr: float = 2e-3
    weight_decay: float = 1e-3
    patience: int = 12
    temp: float = 0.1
    w_supcon: float = 1.0
    w_infonce: float = 1.0
    w_triplet: float = 0.0
    w_vicreg: float = 0.0
    cross_modal: bool = True               # False → E3 (no cross-modal objective)
    augment: bool = True
    seed: int = 0


class EncoderRepresentation(Representation):
    def __init__(self, model, arch, grid, latent, cfg_dict, history):
        super().__init__(name=cfg_dict["name"], branch="encoder", grid=grid,
                         modality_specific=(arch != "shared"),
                         params={"arch": arch, "latent": latent, "n_params": n_params(model), **cfg_dict})
        self.model = model.eval()
        self.arch = arch
        self.n_features = latent
        self.history = history

    @torch.no_grad()
    def transform(self, X, modality=None):
        X = np.nan_to_num(np.atleast_2d(X)).astype(np.float32)
        xt = torch.from_numpy(X)
        if self.arch == "shared":
            z = self.model(xt)
        else:
            if modality is None:
                raise ValueError("dual/modality-specific encoder requires modality per row")
            mod = torch.tensor([MOD[m] for m in np.atleast_1d(modality)], dtype=torch.long)
            z = self.model(xt, mod)
        return z.numpy()


def _set_determinism(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # small CPU model; keep BN/GELU fast


def train_encoder(cfg: EncoderConfig, X_tr, meta_tr, grid, X_val=None, meta_val=None):
    _set_determinism(cfg.seed)
    n_bins = X_tr.shape[1]
    model = (Encoder1D(n_bins, cfg.latent, cfg.channels, dropout=cfg.dropout) if cfg.arch == "shared"
             else DualEncoder(n_bins, cfg.latent, channels=cfg.channels, dropout=cfg.dropout))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    analyte_ids_all = {a: i for i, a in enumerate(sorted(set(meta_tr.analyte) |
                                                        (set(meta_val.analyte) if meta_val is not None else set())))}
    yr = torch.tensor([analyte_ids_all[a] for a in meta_tr.analyte], dtype=torch.long)
    mr = torch.tensor([MOD[m] for m in meta_tr.modality], dtype=torch.long)
    Xtr = np.nan_to_num(X_tr).astype(np.float32)
    aug_cfg = AugConfig()
    rng = np.random.default_rng(cfg.seed)

    def make_batch():
        if cfg.augment:
            v = np.vstack([augment(Xtr[i], grid, aug_cfg, rng) for i in range(len(Xtr))]).astype(np.float32)
        else:
            v = Xtr
        return torch.from_numpy(v)

    def forward(xt):
        return model(xt) if cfg.arch == "shared" else model(xt, mr)

    def total_loss(z):
        loss = torch.zeros((), dtype=z.dtype)
        if cfg.w_supcon:
            loss = loss + cfg.w_supcon * L.supcon_loss(z, yr, cfg.temp)
        if cfg.w_infonce and cfg.cross_modal:
            loss = loss + cfg.w_infonce * L.cross_modal_infonce(z, yr, mr, cfg.temp)
        if cfg.w_triplet:
            loss = loss + cfg.w_triplet * L.triplet_loss(z, yr)
        if cfg.w_vicreg:
            loss = loss + cfg.w_vicreg * L.vicreg_reg(model_raw_embed(model, cfg, Xtr, mr))
        return loss

    @torch.no_grad()
    def val_loss():
        if X_val is None or len(X_val) < 4:
            return None
        model.eval()
        xv = torch.from_numpy(np.nan_to_num(X_val).astype(np.float32))
        yv = torch.tensor([analyte_ids_all[a] for a in meta_val.analyte], dtype=torch.long)
        mv = torch.tensor([MOD[m] for m in meta_val.modality], dtype=torch.long)
        z = model(xv) if cfg.arch == "shared" else model(xv, mv)
        v = L.supcon_loss(z, yv, cfg.temp)
        if cfg.cross_modal:
            v = v + L.cross_modal_infonce(z, yv, mv, cfg.temp)
        model.train()
        return float(v)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, wait = np.inf, None, 0
    for ep in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        z = forward(make_batch())
        loss = total_loss(z)
        loss.backward(); opt.step()
        history["train_loss"].append(float(loss.detach()))
        vl = val_loss(); history["val_loss"].append(vl)
        crit = vl if vl is not None else float(loss)
        if crit < best_val - 1e-4:
            best_val, wait = crit, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["stopped_epoch"] = ep
    history["best_val"] = float(best_val)
    return EncoderRepresentation(model, cfg.arch, grid, cfg.latent,
                                 {"name": cfg.name, **{k: getattr(cfg, k) for k in
                                  ("arch", "latent", "epochs", "lr", "weight_decay", "dropout",
                                   "temp", "w_supcon", "w_infonce", "w_triplet", "w_vicreg",
                                   "cross_modal", "seed")}}, history)


def model_raw_embed(model, cfg, Xtr, mr):
    """Un-normalized embeddings for the VICReg term (anti-collapse on raw geometry)."""
    xt = torch.from_numpy(np.nan_to_num(Xtr).astype(np.float32))
    if cfg.arch == "shared":
        return model(xt, normalize=False)
    return model(xt, mr, normalize=False)
