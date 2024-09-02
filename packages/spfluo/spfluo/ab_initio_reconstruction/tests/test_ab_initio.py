from typing import TYPE_CHECKING

import numpy as np
import pytest

import spfluo
from spfluo.ab_initio_reconstruction.api import AbInitioReconstruction
from spfluo.tests.helpers import assert_allclose, ids, testing_libs
from spfluo.utils.array import get_prefered_namespace_device, numpy
from spfluo.utils.transform import distance_family_poses
from spfluo.utils.volume import interpolate_to_size

if TYPE_CHECKING:
    from spfluo.utils.array import array_api_module

SEED = 123
GPU_LIBS = []
for x in ["torch-cuda", "cupy"]:
    try:
        GPU_LIBS.append(testing_libs[ids.index(x)][0])
    except ValueError:
        pass

params_minimal_run = {
    "N_iter_max": 1,
    "interp_order": 1,
    "N_axes": 2,
    "N_rot": 1,
}

params_long_run = {
    "N_iter_max": 4,
    "interp_order": 1,
}


def run(
    long: bool,
    xp: "array_api_module",
    gpu: bool,
    generated_data_anisotropic,
    tmpdir: str,
):
    volumes_arr, poses_arr, psf_array, groundtruth_array = generated_data_anisotropic
    np.random.seed(SEED)
    args = params_long_run if long else params_minimal_run
    ab_initio = AbInitioReconstruction(**args)
    psf_array = interpolate_to_size(psf_array, volumes_arr.shape[1:])
    _, device = get_prefered_namespace_device(xp=xp, gpu=gpu)
    ab_initio.fit(
        xp.asarray(volumes_arr, device=device),
        psf=xp.asarray(psf_array, device=device),
        output_dir=tmpdir,
    )

    return ab_initio, tmpdir


@pytest.fixture(scope="module")
def gpu_run(request, generated_data_anisotropic, tmp_path_factory):
    xp, long = request.param
    run_name = "long-run" if long else "minimal-run"
    run_name += f"_{xp.__name__}"
    tmpdir = tmp_path_factory.mktemp(run_name)
    return run(long, xp, True, generated_data_anisotropic, tmpdir)


@pytest.fixture(scope="module")
def numpy_run(request, generated_data_anisotropic, tmp_path_factory):
    long = request.param
    run_name = "long-run_numpy" if long else "minimal-run_numpy"
    tmpdir = tmp_path_factory.mktemp(run_name)
    return run(long, numpy, False, generated_data_anisotropic, tmpdir)


@pytest.mark.parametrize("numpy_run", [False], indirect=True)
def test_files_exist(numpy_run):
    ab_initio, tmpdir = numpy_run
    files = [
        "energies.csv",
        "params_learning_alg.json",
        "final_recons.ome.tiff",
        "poses.csv",
    ]
    for f in files:
        assert (tmpdir / f).exists()
    assert (tmpdir / "intermediar_results").exists()
    assert (tmpdir / "distributions_angles").exists()
    assert (tmpdir / "energies").exists()
    for i in range(1, ab_initio._num_iter):
        assert (
            tmpdir / "intermediar_results" / f"estimated_poses_epoch_{i}.csv"
        ).exists()
        assert (tmpdir / "intermediar_results" / f"recons_epoch_{i}.tif").exists()
        assert (tmpdir / "distributions_angles" / f"iter={i:04}_axes.npy").exists()
        assert (tmpdir / "distributions_angles" / f"iter={i:04}_rot.npy").exists()
        assert (tmpdir / "energies" / "energies_each_view_iter=0001.npy").exists()
    assert ab_initio._num_iter == len(ab_initio._energies)


@pytest.mark.parametrize(
    "gpu_run, numpy_run", [((lib, False), False) for lib in GPU_LIBS], indirect=True
)
def test_same_results_gpu(gpu_run, numpy_run):
    ab_initio_gpu, _ = gpu_run
    ab_initio_numpy, _ = numpy_run
    assert_allclose(ab_initio_numpy._energies, ab_initio_gpu._energies, rtol=0.002)


@pytest.mark.slow
@pytest.mark.skipif(
    not (spfluo.has_torch() and spfluo.has_torch_cuda()),
    reason="Too long to test if CUDA is not available",
)
@pytest.mark.parametrize("gpu_run", [(GPU_LIBS[0], True)], indirect=True)
def test_energy_threshold(gpu_run):
    ab_initio, _ = gpu_run
    assert ab_initio._energies[-1] < 210


@pytest.mark.skip(reason="not finished")
@pytest.mark.parametrize("gpu_run", [(GPU_LIBS[0], True)], indirect=True)
def test_poses_aligned(gpu_run):
    ab_initio, folder = gpu_run
    distance_family_poses
