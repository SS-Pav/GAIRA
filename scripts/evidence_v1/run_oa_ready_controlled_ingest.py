from __future__ import annotations

import json
import sys

from gaira.evidence_v1.oa_ready_controlled_ingest import run_oa_ready_controlled_ingest


if __name__ == "__main__":
    paper_ids = sys.argv[1:]
    print(json.dumps(run_oa_ready_controlled_ingest(paper_ids=paper_ids or None), indent=2, sort_keys=True))
