"""GAIRA V7 — Phase 10: in-memory array adapter (NumPy arrays, Python lists)."""
from __future__ import annotations

import numpy as np

from .base import ParsedSpectrum, err
from .text import finalise


class ArrayAdapter:
    name = "arrays"
    extensions = ()

    def sniff(self, payload, filename=None) -> bool:
        return isinstance(payload, (tuple, list)) and len(payload) == 2

    def parse(self, payload, filename=None) -> ParsedSpectrum:
        try:
            xs, ys = payload
            x = np.asarray(xs, dtype=float).ravel()
            y = np.asarray(ys, dtype=float).ravel()
        except Exception as exc:
            return ParsedSpectrum(np.array([]), np.array([]),
                                  [err("input.not_numeric",
                                       f"could not read the pair as numeric arrays: {exc}")],
                                  self.name)
        if x.shape != y.shape:
            return ParsedSpectrum(x, y, [err("input.length_mismatch",
                                             f"wavenumber has {x.size} points but intensity has "
                                             f"{y.size}")], self.name)
        if x.size < 2:
            return ParsedSpectrum(x, y, [err("input.no_data",
                                             f"only {x.size} points supplied")], self.name)
        return finalise(x, y, [], self.name, {"n_points_input": int(x.size)})
