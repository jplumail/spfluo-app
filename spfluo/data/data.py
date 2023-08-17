import csv
import importlib.metadata
from pathlib import Path
from typing import Dict

import numpy as np
import pkg_resources
import pooch
import tifffile

version = importlib.metadata.version("spfluo")

POOCH = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("spfluo"),
    # The remote data is on Github
    base_url="https://github.com/jplumail/spfluo/raw/{version}/spfluo/data/",
    version=version,
    # If this is a development version, get the data from the "main" branch
    version_dev="main",
    registry=None,
)

registry_file = pkg_resources.resource_stream("spfluo", "data/registry.txt")
POOCH.load_registry(registry_file)


def _fetch_dataset(dataset_dir) -> Dict[str, np.ndarray]:
    root_dir = Path(f"generated/{dataset_dir}")
    poses_path = Path(POOCH.fetch((root_dir / "poses.csv").as_posix()))
    content = csv.reader(poses_path.read_text().split("\n"))
    next(content)  # skip header
    data = {}
    for row in content:
        if len(row) == 7:
            particle_path = POOCH.fetch((root_dir / row[0]).as_posix())
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

    psf = tifffile.imread(POOCH.fetch((root_dir / "psf.tiff").as_posix()))
    gt = tifffile.imread(POOCH.fetch((root_dir / "gt.tiff").as_posix()))

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
