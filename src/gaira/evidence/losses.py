"""Stage B training objectives (§9). Compact, predeclared set.

- supervised contrastive (SupCon) on analyte identity: pulls same-analyte spectra
  together INCLUDING across modality (labels are analyte-only) and pushes others apart.
- cross-modal InfoNCE: explicit Raman<->Ag-SERS matched-analyte alignment.
- triplet/margin: anchor/positive(same analyte)/negative(other analyte).
- VICReg variance+covariance regularizers: anti-collapse (std>=1 per dim, decorrelate).
Reconstruction is available only as an auxiliary term, never the primary objective.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def supcon_loss(z, analyte_ids, temp=0.1):
    """Supervised contrastive loss. z: (B,d) L2-normalized. analyte_ids: (B,) long.
    Positives = same analyte (any modality); handles within+cross-modal jointly."""
    B = z.shape[0]
    sim = z @ z.t() / temp
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    pos = (analyte_ids[:, None] == analyte_ids[None, :]) & ~eye
    exp = torch.exp(sim).masked_fill(eye, 0.0)
    denom = exp.sum(dim=1, keepdim=True) + 1e-12
    log_prob = sim - torch.log(denom)
    npos = pos.sum(dim=1)
    valid = npos > 0
    if valid.sum() == 0:
        return z.sum() * 0.0
    loss = -(log_prob.masked_fill(~pos, 0.0).sum(dim=1)[valid] / npos[valid].clamp(min=1))
    return loss.mean()


def cross_modal_infonce(z, analyte_ids, modality, temp=0.1):
    """InfoNCE between the two modalities for matched analytes present in the batch."""
    ram = modality == 0
    ser = modality == 1
    if ram.sum() == 0 or ser.sum() == 0:
        return z.sum() * 0.0
    zr, ar = z[ram], analyte_ids[ram]
    zs, as_ = z[ser], analyte_ids[ser]
    sim = zr @ zs.t() / temp                       # (nr, ns)
    pos = (ar[:, None] == as_[None, :]).float()
    npos = pos.sum(dim=1)
    valid = npos > 0
    if valid.sum() == 0:
        return z.sum() * 0.0
    logp = F.log_softmax(sim, dim=1)
    loss_r = -(logp * pos).sum(dim=1)[valid] / npos[valid].clamp(min=1)
    # symmetric direction
    logp2 = F.log_softmax(sim.t(), dim=1)
    pos2 = pos.t(); npos2 = pos2.sum(dim=1); valid2 = npos2 > 0
    loss_s = -(logp2 * pos2).sum(dim=1)[valid2] / npos2[valid2].clamp(min=1)
    return 0.5 * (loss_r.mean() + loss_s.mean())


def triplet_loss(z, analyte_ids, margin=0.3):
    """Batch-hard triplet on cosine distance (z L2-normalized)."""
    B = z.shape[0]
    d = 1.0 - z @ z.t()
    same = analyte_ids[:, None] == analyte_ids[None, :]
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    pos = same & ~eye
    if pos.sum() == 0:
        return z.sum() * 0.0
    hardest_pos = (d.masked_fill(~pos, -1.0)).max(dim=1).values
    hardest_neg = (d.masked_fill(same, 1e9)).min(dim=1).values
    valid = pos.any(dim=1)
    loss = F.relu(hardest_pos - hardest_neg + margin)[valid]
    return loss.mean() if loss.numel() else z.sum() * 0.0


def vicreg_reg(z, std_target=1.0, cov_w=1.0, var_w=1.0):
    """Variance + covariance anti-collapse regularizers on a batch (z NOT normalized
    for this term). Returns a scalar to ADD to the loss."""
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0) + 1e-4)
    var_loss = F.relu(std_target - std).mean()
    B, d = zc.shape
    cov = (zc.t() @ zc) / max(1, B - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off ** 2).sum() / d
    return var_w * var_loss + cov_w * cov_loss
