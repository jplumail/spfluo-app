import hashlib
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path

import requests

from spfluo.data.constants import (
    ARCHIVE_NAME,
    REPO_ID,
    SEAFILE_URL,
    UNISTRA_PASSWORD,
    UNISTRA_USERNAME,
)

if sys.version_info >= (3, 9):
    from importlib.resources import as_file, files

    @contextmanager
    def _get_registry_file():
        with as_file(files("spfluo").joinpath("data/registry.txt")) as registry:
            yield registry
else:

    @contextmanager
    def _get_registry_file():
        registry = Path(__file__).parent.joinpath("data/registry.txt")
        yield registry


def _file_hash(file: Path):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def get_token():
    # Step 1: Authenticate and get token
    auth_response = requests.post(
        f"{SEAFILE_URL}/api2/auth-token/",
        data={
            "username": f"{UNISTRA_USERNAME}@unistra.fr",
            "password": UNISTRA_PASSWORD,
        },
    )
    auth_response.raise_for_status()  # This will raise an error if the request failed
    return auth_response.json().get("token")


def upload_archive():
    # Step 0: Create a zip archive from registry.txt
    with zipfile.ZipFile(ARCHIVE_NAME, "w") as zipf:
        with _get_registry_file() as registry_path:
            with open(registry_path, "r") as registry:
                for line in registry:
                    # Assuming space separates the path and the hash
                    file, hash = line.strip().split(" ")
                    file_path = registry_path.parent / file
                    print(f"Adding {file_path} to zip")
                    if file_path.exists() and _file_hash(file_path) == hash:
                        zipf.write(file_path, file_path.relative_to(registry_path.parent))
                    else:
                        raise FileNotFoundError(
                            f"{file_path} was not found or hash was wrong"
                        )

    # Step 2: Get upload link
    headers = {"Authorization": f"Token {get_token()}"}
    upload_link_response = requests.get(
        f"{SEAFILE_URL}/api2/repos/{REPO_ID}/upload-link/", headers=headers
    )
    upload_link_response.raise_for_status()
    # Removing quotes from the response
    upload_link = upload_link_response.text.strip('"')

    # Step 3: Upload file
    with open(ARCHIVE_NAME, "rb") as file:
        f = {"file": file, "parent_dir": ("", "/"), "replace": (None, "1")}
        print("Uploading file to seafile...")
        upload_response = requests.post(upload_link, headers=headers, files=f)
        upload_response.raise_for_status()

    print("File uploaded successfully.")


if __name__ == "__main__":
    upload_archive()
