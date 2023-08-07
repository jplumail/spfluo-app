import numpy as np
import pytest

import spfluo
from spfluo.ab_initio_reconstruction.api import AbInitioReconstruction
from spfluo.utils.volume import interpolate_to_size

SEED = 123


def minimal_run(gpu, generated_data_anisotropic, tmpdir):
    volumes_arr, poses_arr, psf_array, groundtruth_array = generated_data_anisotropic
    np.random.seed(SEED)
    ab_initio = AbInitioReconstruction(N_iter_max=1, interp_order=1, N_axes=2, N_rot=1)
    psf_array = interpolate_to_size(psf_array, volumes_arr.shape[1:])
    ab_initio.fit(volumes_arr, psf=psf_array, gpu=gpu, output_dir=tmpdir)

    return ab_initio, tmpdir


def long_run(gpu, generated_data_anisotropic, tmpdir):
    volumes_arr, poses_arr, psf_array, groundtruth_array = generated_data_anisotropic
    np.random.seed(SEED)
    ab_initio = AbInitioReconstruction(N_iter_max=10, interp_order=1)
    psf_array = interpolate_to_size(psf_array, volumes_arr.shape[1:])
    ab_initio.fit(volumes_arr, psf=psf_array, gpu=gpu, output_dir=tmpdir)

    return ab_initio, tmpdir


@pytest.fixture()
def minimal_run_cucim(generated_data_anisotropic, tmpdir):
    ab_initio, _ = minimal_run("cucim", generated_data_anisotropic, tmpdir)
    return ab_initio, tmpdir


@pytest.fixture()
def minimal_run_pytorch(generated_data_anisotropic, tmpdir):
    ab_initio, _ = minimal_run("pytorch", generated_data_anisotropic, tmpdir)
    return ab_initio, tmpdir


def test_ab_initio_files_exist(generated_data_anisotropic, tmpdir):
    ab_initio, _ = minimal_run(None, generated_data_anisotropic, tmpdir)
    files = [
        "distributions_axes.npy",
        "distributions_rot.npy",
        "energies_each_view.npy",
        "energies.csv",
        "params_learning_alg.json",
        "final_recons.tif",
    ]
    for f in files:
        assert (tmpdir / f).exists()
    assert (tmpdir / "intermediar_results").exists()
    for i in range(1, ab_initio._num_iter):
        assert (
            tmpdir / "intermediar_results" / f"estimated_poses_epoch_{i}.csv"
        ).exists()
        assert (tmpdir / "intermediar_results" / f"recons_epoch_{i}.tif").exists()
    assert ab_initio._num_iter == len(ab_initio._energies)


@pytest.mark.skipif(not spfluo.has_cupy, reason="skipping cupy test")
def test_ab_initio_same_results_cucim(generated_data_anisotropic, tmpdir):
    ab_initio_numpy, _ = minimal_run(None, generated_data_anisotropic, tmpdir)
    ab_initio, _ = minimal_run("cucim", generated_data_anisotropic, tmpdir)
    assert np.isclose(ab_initio_numpy._energies, ab_initio._energies).all()


@pytest.mark.skipif(
    not (spfluo.has_torch and spfluo.has_torch_cuda),
    reason="Cannot run ab initio pytorch without gpu",
)
def test_ab_initio_same_results_pytorch(generated_data_anisotropic, tmpdir):
    ab_initio_numpy, _ = minimal_run(None, generated_data_anisotropic, tmpdir)
    ab_initio, _ = minimal_run("pytorch", generated_data_anisotropic, tmpdir)
    assert np.isclose(ab_initio_numpy._energies, ab_initio._energies, rtol=0.001).all()


@pytest.mark.skipif(
    not (spfluo.has_torch and spfluo.has_torch_cuda),
    reason="Too long to test if CUDA is not available",
)
def test_long_run(generated_data_all, tmpdir):
    ab_initio, _ = long_run("pytorch", generated_data_all, tmpdir)
    assert ab_initio._energies[-1] < 200
