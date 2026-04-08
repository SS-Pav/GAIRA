from __future__ import annotations

import json

from gaira.evidence_v1.paper_targeted_enrichment import run_targeted_enrichment


if __name__ == "__main__":
    print(json.dumps(run_targeted_enrichment(), indent=2, sort_keys=True))
