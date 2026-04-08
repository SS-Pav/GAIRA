from __future__ import annotations

import json

from gaira.evidence_v1.literature_asset_truth_oa import run_asset_truth_oa_validation


def main() -> None:
    summary = run_asset_truth_oa_validation()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
