from __future__ import annotations

import json

from gaira.evidence_v1.paper_evidence_qc import run_paper_evidence_qc


def main() -> None:
    print(json.dumps(run_paper_evidence_qc(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
