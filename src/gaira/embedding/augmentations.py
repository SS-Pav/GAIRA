from __future__ import annotations

import random

import torch.nn.functional as F
import torch


def add_noise(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    noise = torch.randn_like(x) * sigma
    return x + noise


def shift_wavenumber(x: torch.Tensor, max_shift: int = 3) -> torch.Tensor:
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return x.clone()
    shifted = torch.roll(x, shifts=shift, dims=-1)
    if shift > 0:
        shifted[..., :shift] = 0.0
    else:
        shifted[..., shift:] = 0.0
    return shifted


def intensity_scaling(x: torch.Tensor, scale_range: tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
    scale = random.uniform(*scale_range)
    return x * float(scale)


def local_intensity_warp(x: torch.Tensor, amplitude: float = 0.04, kernel_width: int = 41) -> torch.Tensor:
    noise = torch.randn(1, 1, x.shape[-1], device=x.device, dtype=x.dtype)
    kernel = torch.ones(1, 1, kernel_width, device=x.device, dtype=x.dtype) / float(kernel_width)
    smoothed = F.conv1d(noise, kernel, padding=kernel_width // 2).squeeze(0).squeeze(0)
    warp = 1.0 + amplitude * smoothed / (torch.std(smoothed) + 1e-6)
    return x * warp.clamp(min=0.85, max=1.15)


def mild_smoothing_perturbation(x: torch.Tensor, kernel_width: int = 7) -> torch.Tensor:
    kernel = torch.ones(1, 1, kernel_width, device=x.device, dtype=x.dtype) / float(kernel_width)
    smoothed = F.conv1d(x.view(1, 1, -1), kernel, padding=kernel_width // 2).view(-1)
    mix = random.uniform(0.0, 0.15)
    return (1.0 - mix) * x + mix * smoothed


def random_band_mask(x: torch.Tensor, max_width: int = 18, attenuation_range: tuple[float, float] = (0.1, 0.4)) -> torch.Tensor:
    width = random.randint(6, max_width)
    start = random.randint(0, max(0, x.shape[-1] - width))
    attenuation = random.uniform(*attenuation_range)
    y = x.clone()
    y[start : start + width] *= attenuation
    return y


def local_peak_dropout(x: torch.Tensor, max_width: int = 10, dropout_range: tuple[float, float] = (0.0, 0.25)) -> torch.Tensor:
    width = random.randint(3, max_width)
    start = random.randint(0, max(0, x.shape[-1] - width))
    factor = random.uniform(*dropout_range)
    y = x.clone()
    y[start : start + width] *= factor
    return y


def augment_spectrum(x: torch.Tensor, *, mode: str = "pass2", strength: float = 1.0) -> torch.Tensor:
    strength = float(max(0.25, min(2.0, strength)))
    basic_ops = [
        lambda v: add_noise(v, sigma=0.01 * strength),
        lambda v: shift_wavenumber(v, max_shift=max(1, int(round(3 * strength)))),
        lambda v: intensity_scaling(v, scale_range=(1.0 - 0.1 * strength, 1.0 + 0.1 * strength)),
        lambda v: local_intensity_warp(v, amplitude=0.04 * strength, kernel_width=41),
        lambda v: mild_smoothing_perturbation(v, kernel_width=7),
    ]
    region_ops = [
        lambda v: random_band_mask(v, max_width=max(8, int(round(18 * strength)))),
        lambda v: local_peak_dropout(v, max_width=max(5, int(round(10 * strength)))),
    ]
    operations = basic_ops if mode == "pass2" else basic_ops + region_ops
    augmented = x.clone()
    n_ops = 1 if mode == "pass2" else 2
    for operation in random.sample(operations, k=min(n_ops, len(operations))):
        augmented = operation(augmented)
    return augmented.to(dtype=torch.float32)
