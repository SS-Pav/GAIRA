from __future__ import annotations

import json

from gaira.evidence_v1.literature_asset_resolver import run_literature_asset_resolver


if __name__ == "__main__":
    print(json.dumps(run_literature_asset_resolver(), indent=2, sort_keys=True))
