from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RamanEncoder(nn.Module):
    def __init__(self, input_len: int):
        super().__init__()
        self.input_len = int(input_len)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x).squeeze(-1)
        return F.normalize(self.fc(x), dim=1)
