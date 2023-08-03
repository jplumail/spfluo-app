import numpy as np
import pytest

from spfluo.ab_initio_reconstruction.api import AbInitioReconstruction
from spfluo.utils.volume import interpolate_to_size

SEED = 123


@pytest.fixture(params=["cucim", "pytorch"])
def gpu(request):
    return request.param


def minimal_run(gpu, generated_data_arrays, tmpdir):
    volumes_arr, poses_arr, psf_array, groundtruth_array = generated_data_arrays
    np.random.seed(SEED)
    ab_initio = AbInitioReconstruction(N_iter_max=1, interp_order=1, N_axes=2, N_rot=1)
    psf_array = interpolate_to_size(psf_array, volumes_arr.shape[1:])
    ab_initio.fit(volumes_arr, psf=psf_array, gpu=gpu, output_dir=tmpdir)

    return ab_initio, tmpdir


def long_run(gpu, generated_data_arrays, tmpdir):
    volumes_arr, poses_arr, psf_array, groundtruth_array = generated_data_arrays
    np.random.seed(SEED)
    ab_initio = AbInitioReconstruction(N_iter_max=10, interp_order=1)
    psf_array = interpolate_to_size(psf_array, volumes_arr.shape[1:])
    ab_initio.fit(volumes_arr, psf=psf_array, gpu=gpu, output_dir=tmpdir)

    return ab_initio, tmpdir


@pytest.fixture()
def minimal_run_cucim(generated_data_arrays, tmpdir):
    ab_initio, _ = minimal_run("cucim", generated_data_arrays, tmpdir)
    return ab_initio, tmpdir


@pytest.fixture()
def minimal_run_pytorch(generated_data_arrays, tmpdir):
    ab_initio, _ = minimal_run("pytorch", generated_data_arrays, tmpdir)
    return ab_initio, tmpdir


@pytest.fixture()
def minimal_run_numpy(generated_data_arrays, tmpdir):
    ab_initio, _ = minimal_run(None, generated_data_arrays, tmpdir)
    return ab_initio, tmpdir


def test_ab_initio_files_exist(minimal_run_numpy):
    ab_initio, output_dir = minimal_run_numpy
    files = [
        "distributions_axes.npy",
        "distributions_rot.npy",
        "energies_each_view.npy",
        "energies.csv",
        "params_learning_alg.json",
        "final_recons.tif",
    ]
    for f in files:
        assert (output_dir / f).exists()
    assert (output_dir / "intermediar_results").exists()
    for i in range(1, ab_initio._num_iter):
        assert (
            output_dir / "intermediar_results" / f"estimated_poses_epoch_{i}.csv"
        ).exists()
        assert (output_dir / "intermediar_results" / f"recons_epoch_{i}.tif").exists()
    assert ab_initio._num_iter == len(ab_initio._energies)


def test_ab_initio_same_results(
    minimal_run_numpy, minimal_run_cucim, minimal_run_pytorch
):
    assert np.isclose(
        minimal_run_numpy[0]._energies, minimal_run_cucim[0]._energies, rtol=0.001
    ).all()
    assert np.isclose(
        minimal_run_numpy[0]._energies, minimal_run_pytorch[0]._energies, rtol=0.001
    ).all()


@pytest.fixture()
def long_run_pytorch(generated_data_arrays, tmpdir):
    ab_initio, _ = long_run("pytorch", generated_data_arrays, tmpdir)
    return ab_initio, tmpdir


def test_long_run(long_run_pytorch):
    ab_initio, tmpdir = long_run_pytorch
    assert ab_initio._energies[-1] < 200
