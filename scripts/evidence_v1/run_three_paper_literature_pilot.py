from __future__ import annotations

import json

from gaira.evidence_v1.pilot_literature_integration import run_three_paper_literature_pilot


def main() -> None:
    result = run_three_paper_literature_pilot()
    print(json.dumps(result["build_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
