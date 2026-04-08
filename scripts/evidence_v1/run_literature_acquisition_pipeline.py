from __future__ import annotations

import json

from gaira.evidence_v1.literature_acquisition_pipeline import run_literature_acquisition_pipeline


if __name__ == "__main__":
    print(json.dumps(run_literature_acquisition_pipeline(), indent=2, sort_keys=True))
