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


def augment_spectrum(x: torch.Tensor) -> torch.Tensor:
    operations = [
        lambda v: add_noise(v, sigma=0.01),
        lambda v: shift_wavenumber(v, max_shift=3),
        lambda v: intensity_scaling(v, scale_range=(0.9, 1.1)),
        lambda v: local_intensity_warp(v, amplitude=0.04, kernel_width=41),
        lambda v: mild_smoothing_perturbation(v, kernel_width=7),
    ]
    augmented = random.choice(operations)(x.clone())
    return augmented.to(dtype=torch.float32)
