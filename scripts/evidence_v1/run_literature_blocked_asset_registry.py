from __future__ import annotations

import json

from gaira.evidence_v1.literature_blocked_asset_registry import build_blocked_asset_registry


def main() -> None:
    summary = build_blocked_asset_registry()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
