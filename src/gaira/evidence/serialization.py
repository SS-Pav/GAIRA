"""Serialize / restore Stage B representation artifacts (§ Part 8 / Part 24).

Interpretable reps → .npz (arrays) + .json (params). Encoder reps → torch
state_dict (.pt) + .json. Keeps only necessary artifacts (small)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def save_representation(rep, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta = rep.to_dict()
    if rep.branch == "encoder":
        import torch
        torch.save(rep.model.state_dict(), out_dir / f"{rep.name}.pt")
        meta["artifact"] = f"{rep.name}.pt"
        meta["history"] = rep.history
    else:
        arrays = {}
        for attr in ("edges", "atoms", "basis", "scales"):
            if hasattr(rep, attr):
                arrays[attr] = np.asarray(getattr(rep, attr))
        if arrays:
            np.savez(out_dir / f"{rep.name}.npz", **arrays)
            meta["artifact"] = f"{rep.name}.npz"
    (out_dir / f"{rep.name}.json").write_text(json.dumps(meta, indent=2, default=float))
    return out_dir / f"{rep.name}.json"
