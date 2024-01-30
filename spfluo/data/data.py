import csv
import hashlib
import io
import tempfile
import zipfile
from importlib.resources import as_file, files
from pathlib import Path
from typing import Dict

import numpy as np
import requests
import tifffile


def _download_data(d: Path):
    url = "https://seafile.unistra.fr/f/ce50bfac098a45c38c99/?dl=1"
    response = requests.get(url)

    if response.status_code == 200:
        # Write the content of the response to a file
        with tempfile.TemporaryFile() as fp:
            fp.write(response.content)
            fp.seek(0)
            with zipfile.ZipFile(io.BytesIO(fp.read()), "r") as zip_ref:
                zip_ref.extractall(d)
            return True
    return False


def _check_data(d: Path):
    registry = d / "registry.txt"
    for line in registry.read_text().strip().split("\n"):
        path, hash = line.split(" ")
        path = d / path
        if (not path.exists()) or hashlib.sha256(path.read_bytes()).hexdigest() != hash:
            return False
    return True


def _fetch_dataset(dataset_dir: str) -> Dict[str, np.ndarray]:
    # Download if necessary
    with as_file(files("spfluo").joinpath("data")) as data_dir:
        if not _check_data(data_dir):
            _download_data(data_dir)
            if not _check_data(data_dir):
                raise RuntimeError("Download failed")

        # parse data
        root_dir = data_dir / "generated" / dataset_dir
        poses_path = root_dir / "poses.csv"
        content = csv.reader(poses_path.read_text().split("\n"))
        next(content)  # skip header
        data = {}
        for row in content:
            if len(row) == 7:
                particle_path = (root_dir / row[0]).as_posix()
                data[row[0]] = {
                    "array": tifffile.imread(particle_path),
                    "rot": np.array(row[1:4], dtype=float),
                    "trans": np.array(row[4:7], dtype=float),
                }

        N = len(data)
        p0 = next(iter(data))
        D = data[p0]["array"].shape[0]
        dtype = data[p0]["array"].dtype
        volumes = np.empty((N, D, D, D), dtype=dtype)
        poses = np.empty((len(data), 6))
        for i, k in enumerate(data):
            p = data[k]
            rot = p["rot"]
            trans = p["trans"]
            poses[i, :3] = rot
            poses[i, 3:] = trans
            volumes[i] = p["array"]

        psf = tifffile.imread((root_dir / "psf.tiff").as_posix())
        gt = tifffile.imread((root_dir / "gt.tiff").as_posix())

        return {
            "volumes": volumes,
            "poses": poses,
            "psf": psf,
            "gt": gt,
            "rootdir": poses_path.parent.absolute(),
        }


def generated_isotropic():
    return _fetch_dataset("isotropic-1.0")


def generated_anisotropic():
    return _fetch_dataset("anisotropic-5.0-1.0-1.0")
