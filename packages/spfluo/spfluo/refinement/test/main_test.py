import tempfile
from pathlib import Path

from spfluo import data
from spfluo.refinement.__main__ import create_parser, main


def test_main_refinement():
    generated_root_dir = data.generated_anisotropic()["rootdir"]
    parser = create_parser()
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        reconstruction_path = tmpdir / "reconstruction.tiff"
        poses_path = tmpdir / "poses.csv"
        args = parser.parse_args(
            [
                "--particles_dir",
                str(generated_root_dir / "particles"),
                "--psf_path",
                str(generated_root_dir / "psf.tiff"),
                "--guessed_poses_path",
                str(generated_root_dir / "poses.csv"),
                "--ranges",
                "0",
                "10",
                "--steps",
                "(1,1)",
                "2",
                "--output_reconstruction_path",
                str(reconstruction_path),
                "--output_poses_path",
                str(poses_path),
            ]
        )
        main(args)
        assert reconstruction_path.exists()
        assert poses_path.exists()
