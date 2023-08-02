from pathlib import Path

from spfluo.ab_initio_reconstruction.__main__ import create_parser, main


def test_main_ab_initio(generated_root_dir: Path, tmpdir):
    parser = create_parser()
    tmpdir = Path(tmpdir)
    args = parser.parse_args(
        [
            "--particles_dir",
            str(generated_root_dir / "particles"),
            "--psf_path",
            str(generated_root_dir / "psf.tiff"),
            "--output_dir",
            str(tmpdir),
            "--N_iter_max",
            str(1),
            "--N_axes",
            str(1),
            "--N_rot",
            str(1),
        ]
    )
    main(args)
    reconstruction_path = tmpdir / "final_recons.tif"
    poses_path = tmpdir / "poses.csv"
    assert reconstruction_path.exists()
    assert poses_path.exists()
