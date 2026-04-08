from __future__ import annotations

import json

from gaira.evidence_v1.oa_phase1_rerun import run_oa_phase1_rerun


if __name__ == "__main__":
    print(json.dumps(run_oa_phase1_rerun(), indent=2, sort_keys=True))
