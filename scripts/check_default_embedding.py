from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.embedding_registry import get_default_embedding_entry, list_embedding_entries

    dataset_id = "small2023_ev"
    default_entry = get_default_embedding_entry(dataset_id)
    rows = list_embedding_entries(dataset_id)
    df = pd.DataFrame(rows)

    print(f"Default embedding for {dataset_id}:")
    print(pd.DataFrame([default_entry]).to_string(index=False))
    print()
    print("Registered embedding roles:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
