from __future__ import annotations

import json

from gaira.evidence_v1.ontology_aligned_motif_summary import run_ontology_aligned_motif_summary


def main() -> None:
    result = run_ontology_aligned_motif_summary()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
