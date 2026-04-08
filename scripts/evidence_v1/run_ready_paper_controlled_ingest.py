from __future__ import annotations

import json
import sys

from gaira.evidence_v1.ready_paper_controlled_ingest import run_ready_paper_controlled_ingest


def main() -> None:
    summary = run_ready_paper_controlled_ingest(paper_ids=sys.argv[1:] or None)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
