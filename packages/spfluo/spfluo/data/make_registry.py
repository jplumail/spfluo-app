from pathlib import Path

from spfluo.data.upload import _file_hash


def make_registry(directory: Path, output: Path, extensions: list[str] = ["*"]):
    """taken from pooch"""

    files = sorted(
        str(path.relative_to(directory))
        for ext in extensions
        for path in directory.glob(f"**/*{ext}")
        if path.is_file()
    )

    hashes = [_file_hash(directory / fname) for fname in files]

    with open(output, "w") as outfile:
        for fname, fhash in zip(files, hashes):
            # Only use Unix separators for the registry so that we don't go
            # insane dealing with file paths.
            outfile.write("{} {}\n".format(fname.replace("\\", "/"), fhash))


if __name__ == "__main__":
    data_dir = Path(__file__).parent
    make_registry(data_dir, data_dir / "registry.txt", extensions=[".csv", ".tiff", ".tif"])
