from __future__ import annotations

import json

from gaira.evidence_v1.constants import DB_PATH
from gaira.evidence_v1.qa import generate_qa_artifacts


def main() -> None:
    counts = generate_qa_artifacts(DB_PATH)
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

