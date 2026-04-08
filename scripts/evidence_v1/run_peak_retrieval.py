from __future__ import annotations

import argparse
import json

from gaira.evidence_v1.constants import DB_PATH
from gaira.evidence_v1.retrieval import PeakListRetrievalEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAIRA evidence v1 peak-list retrieval.")
    parser.add_argument("--peaks", required=True, help="Comma-separated peak list in cm^-1.")
    parser.add_argument("--domain", default="", help="Optional domain hint: ev, serum, plasma, pathogen.")
    parser.add_argument("--modality", default="", help="Optional modality hint: raman or sers.")
    parser.add_argument("--tolerance", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    query_peaks = [float(item.strip()) for item in args.peaks.split(",") if item.strip()]
    engine = PeakListRetrievalEngine(str(DB_PATH))
    result = engine.search(
        query_peaks=query_peaks,
        domain_hint=args.domain or None,
        modality_hint=args.modality or None,
        tolerance_cm=args.tolerance,
        top_k=args.top_k,
    )
    run_id = engine.persist_run(result, top_k=args.top_k)
    result["run_id"] = run_id
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

