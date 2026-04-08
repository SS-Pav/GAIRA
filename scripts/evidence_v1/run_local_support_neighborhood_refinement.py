from __future__ import annotations

import json

from gaira.evidence_v1.local_support_neighborhoods import build_local_support_neighborhoods


def main() -> None:
    summary = build_local_support_neighborhoods()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
