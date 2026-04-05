from __future__ import annotations

import argparse
from pathlib import Path

from gaira.autoresearch_storage import (
    DEFAULT_STORAGE_CONFIG_PATH,
    initialize_autoresearch_sprint,
    load_autoresearch_storage_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and initialize the GAIRAv3 autoresearch SSD storage layout.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_STORAGE_CONFIG_PATH)
    parser.add_argument("--sprint-id", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_autoresearch_storage_config(args.config_path)
    sprint_id = args.sprint_id.strip() or config.sprint_id
    paths = initialize_autoresearch_sprint(args.config_path, sprint_id=sprint_id)

    print("GAIRA autoresearch storage initialized")
    print(f"output_root={paths.output_root}")
    print(f"sprint_root={paths.sprint_root}")
    print(f"runs_dir={paths.runs_dir}")
    print(f"figures_dir={paths.figures_dir}")
    print(f"tables_dir={paths.tables_dir}")
    print(f"logs_dir={paths.logs_dir}")
    print(f"report_dir={paths.report_dir}")
    print(f"manifest_path={paths.manifest_path}")


if __name__ == "__main__":
    main()
