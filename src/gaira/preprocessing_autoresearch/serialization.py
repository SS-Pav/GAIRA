"""Stage B0 — deterministic pipeline serialization."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from .pipeline import Candidate


def candidate_to_json(cand) -> dict:
    d = cand.to_dict()
    d["fingerprint"] = hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:32]
    return d


def save_candidate(cand, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    d = candidate_to_json(cand)
    p.write_text(json.dumps(d, indent=2, default=str))
    return d


def load_candidate(path) -> Candidate:
    d = json.loads(Path(path).read_text())
    return Candidate(cid=d["cid"], arm=d["arm"], raman=d["raman"], sers=d["sers"],
                     background=(d["background"]["method"], d["background"]["params"]),
                     aggregate=d["aggregate"], derivative=d["derivative"],
                     norm_raman=d["norm_raman"], norm_sers=d["norm_sers"],
                     peak_transform=d["peak_transform"])
