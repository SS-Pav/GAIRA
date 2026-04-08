from __future__ import annotations

import json

from gaira.evidence_v1.condition_ontology_layer import run_condition_ontology_layer


if __name__ == "__main__":
    print(json.dumps(run_condition_ontology_layer(), indent=2, sort_keys=True))
