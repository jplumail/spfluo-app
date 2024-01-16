import numpy as np

import pytest
import spfluo
from spfluo.ab_initio_reconstruction.api import AbInitioReconstruction
from spfluo.utils.transform import distance_family_poses
from spfluo.utils.volume import interpolate_to_size

SEED = 123


params_minimal_run = {
    "N_iter_max": 1,
    "interp_order": 1,
    "N_axes": 2,
    "N_rot": 1,
}

params_long_run = {
    "N_iter_max": 10,
    "interp_order": 1,
}


def run(long, gpu, generated_data_anisotropic, tmpdir):
    volumes_arr, poses_arr, psf_array, groundtruth_array = generated_data_anisotropic
    np.random.seed(SEED)
    args = params_long_run if long else params_minimal_run
    ab_initio = AbInitioReconstruction(**args)
    psf_array = interpolate_to_size(psf_array, volumes_arr.shape[1:])
    ab_initio.fit(volumes_arr, psf=psf_array, gpu=gpu, output_dir=tmpdir)

    return ab_initio, tmpdir


@pytest.fixture(scope="module")
def gpu_run(request, generated_data_anisotropic, tmp_path_factory):
    gpu, long = request.param
    run_name = "long-run" if long else "minimal-run"
    run_name += f"_{gpu}"
    tmpdir = tmp_path_factory.mktemp(run_name)
    return run(long, gpu, generated_data_anisotropic, tmpdir)


@pytest.fixture(scope="module")
def numpy_run(request, generated_data_anisotropic, tmp_path_factory):
    long = request.param
    run_name = "long-run_numpy" if long else "minimal-run_numpy"
    tmpdir = tmp_path_factory.mktemp(run_name)
    return run(long, None, generated_data_anisotropic, tmpdir)


@pytest.mark.parametrize("numpy_run", [False], indirect=True)
def test_files_exist(numpy_run):
    ab_initio, tmpdir = numpy_run
    files = [
        "distributions_axes.npy",
        "distributions_rot.npy",
        "energies_each_view.npy",
        "energies.csv",
        "params_learning_alg.json",
        "final_recons.tif",
        "poses.csv",
        "ssims.csv",
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


GPU_LIBS = []
if spfluo.has_torch:
    GPU_LIBS.append("pytorch")
if spfluo.has_cupy:
    GPU_LIBS.append("cucim")


@pytest.mark.parametrize(
    "gpu_run, numpy_run", [((lib, False), False) for lib in GPU_LIBS], indirect=True
)
def test_same_results_gpu(gpu_run, numpy_run):
    ab_initio_gpu, _ = gpu_run
    ab_initio_numpy, _ = numpy_run
    assert np.isclose(
        ab_initio_numpy._energies, ab_initio_gpu._energies, rtol=0.001
    ).all()


@pytest.mark.skipif(
    not (spfluo.has_torch and spfluo.has_torch_cuda),
    reason="Too long to test if CUDA is not available",
)
@pytest.mark.parametrize("gpu_run", [("pytorch", True)], indirect=True)
def test_energy_threshold(gpu_run):
    ab_initio, _ = gpu_run
    assert ab_initio._energies[-1] < 200


@pytest.mark.skip(reason="not finished")
@pytest.mark.parametrize("gpu_run", [("pytorch", True)], indirect=True)
def test_poses_aligned(gpu_run):
    ab_initio, folder = gpu_run
    distance_family_poses
