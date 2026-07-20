"""Generate the MSS registry artifact from the frozen atlas (deterministic).

Writes results/v5_rebuild/engine_v1/artifacts/mss_registry_v1.json — the derived,
provenance-carrying Molecular Spectral Signatures registry. Re-runnable; the output
is a pure function of the frozen Component Registry + ontology weights, so it is
reproducible byte-for-byte. Nothing frozen is modified.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
from gaira.engine.mss import MSSLayer

OUT = REPO / "results/v5_rebuild/engine_v1/artifacts/mss_registry_v1.json"


def main():
    layer = MSSLayer()
    reg = layer.registry()
    OUT.write_text(json.dumps(reg, indent=2))
    print(f"wrote {OUT.relative_to(REPO)}  ({reg['n_motifs']} motifs, "
          f"atlas {reg['atlas_fingerprint'][:12]}…)")
    for m in reg["motifs"]:
        cs = ", ".join(f"c{c['component']}={c['weight']:.2f}" for c in m["contributors"][:4])
        pert = m["perturbation"]
        flag = " [non-biochemical]" if m["non_biochemical"] else ""
        print(f"  {m['id']:<26} -> {m['parent_theme']:<22} conf={m['confidence']:.2f}"
              f"  dose={len(pert['dose_responsive_components'])} spike={len(pert['serum_spike_matches'])}"
              f" depl={len(pert['depletion_matches'])}{flag}")
        print(f"      {cs}")


if __name__ == "__main__":
    main()
