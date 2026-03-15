from pathlib import Path

import requests


def download_github_repo_zip(repo_url: str, target_folder: Path) -> tuple[Path | None, str | None]:
    """Try downloading a GitHub repository archive from common default branches."""
    base_url = repo_url.rstrip("/")
    branch_urls = {
        "main": f"{base_url}/archive/refs/heads/main.zip",
        "master": f"{base_url}/archive/refs/heads/master.zip",
    }

    for branch_name, archive_url in branch_urls.items():
        zip_path = target_folder / f"{target_folder.name}_{branch_name}.zip"

        try:
            with requests.get(archive_url, stream=True, timeout=60) as response:
                if response.status_code == 404:
                    print(f"Branch '{branch_name}' archive not found at: {archive_url}")
                    continue

                response.raise_for_status()
                with zip_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            handle.write(chunk)

            return zip_path, branch_name
        except requests.RequestException as exc:
            print(f"Could not download GitHub archive from branch '{branch_name}': {exc}")

    return None, None
