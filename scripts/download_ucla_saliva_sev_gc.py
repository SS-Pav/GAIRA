from __future__ import annotations

import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tmp" / "ucla_saliva_download_manifest.csv"

sys.path.insert(0, str(ROOT / "src"))
from gaira.config import ensure_storage_dirs, resolve_storage_path  # noqa: E402


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    storage = ensure_storage_dirs()
    raw_root = resolve_storage_path(storage.get("raw_data"))
    if raw_root is None:
        raise RuntimeError("Could not resolve GAIRA raw_data storage root.")

    dataset_root = raw_root / "ucla_saliva_sev_gc"
    dataset_root.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(MANIFEST_PATH.open()))
    def download_one(row: dict) -> dict:
        article_id = row["article_id"]
        file_name = row["file_name"]
        dest_dir = dataset_root / f"article_{article_id}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file_name

        recovered = False
        error = ""
        if dest_path.exists() and dest_path.stat().st_size > 0:
            recovered = True
        else:
            try:
                subprocess.run(
                    ["curl", "-L", row["download_url"], "-o", str(dest_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                recovered = dest_path.exists() and dest_path.stat().st_size > 0
            except subprocess.CalledProcessError as exc:
                error = exc.stderr.strip()[:400]

        return {
            "article_id": article_id,
            "title": row["title"],
            "source_record_url": f"https://figshare.com/articles/dataset/x/{article_id}",
            "file_name": file_name,
            "bytes_expected": row["bytes"],
            "bytes_recovered": dest_path.stat().st_size if dest_path.exists() else 0,
            "download_url": row["download_url"],
            "local_relpath": str(dest_path.relative_to(dataset_root)) if dest_path.exists() else "",
            "recovered": recovered,
            "error": error,
        }

    manifest_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(download_one, row) for row in rows]
        for future in as_completed(futures):
            manifest_rows.append(future.result())

    shard_manifest = dataset_root / "shard_manifest.csv"
    with shard_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    recovered_count = sum(1 for row in manifest_rows if row["recovered"])
    print(f"Wrote shard manifest: {shard_manifest}")
    print(f"Recovered files: {recovered_count} / {len(manifest_rows)}")


if __name__ == "__main__":
    main()
