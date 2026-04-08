from __future__ import annotations

import json

from gaira.evidence_v1.remaining_paper_controlled_ingest import run_remaining_paper_controlled_ingest


if __name__ == "__main__":
    print(json.dumps(run_remaining_paper_controlled_ingest(), indent=2, sort_keys=True))
