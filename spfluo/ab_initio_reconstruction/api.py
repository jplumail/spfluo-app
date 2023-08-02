import array_api_compat.numpy
import numpy as np

from spfluo.utils.volume import (
    discretize_sphere_uniformly,
)

from .learning_algorithms.gradient_descent_importance_sampling import (
    gd_importance_sampling_3d,
)
from .params import ParametersMainAlg
from .volume_representation.pixel_representation import Fourier_pixel_representation


class AbInitioReconstruction:
    def __init__(self, **params):
        self.params = params

    def fit(self, X, psf=None, output_dir=None, gpu=None):
        """Reconstruct a volume based on views of particles"""
        if psf is None:
            raise NotImplementedError  # TODO : default psf to gaussian
        if output_dir is None:
            output_dir = "./ab-initio-output"

        params_learning_alg = ParametersMainAlg(**self.params)
        fourier_volume = Fourier_pixel_representation(
            3,
            psf.shape[0],
            psf,
            init_vol=None,
            random_init=True,
            dtype=params_learning_alg.dtype,
        )

        N = X.shape[0]

        uniform_sphere_discretization = discretize_sphere_uniformly(
            array_api_compat.numpy,
            params_learning_alg.M_axes,
            params_learning_alg.M_rot,
            dtype=np.float64,
        )
        imp_distrs_axes = (
            np.ones((N, params_learning_alg.M_axes)) / params_learning_alg.M_axes
        )
        imp_distrs_rot = (
            np.ones((N, params_learning_alg.M_rot)) / params_learning_alg.M_rot
        )

        (
            imp_distrs_rot_recorded,
            imp_distrs_axes_recorded,
            recorded_energies,
            recorded_shifts,
            unif_prop,
            volume_representation,
            itr,
            energies_each_view,
            views,
            file_names,
            ests_poses,
        ) = gd_importance_sampling_3d(
            fourier_volume,
            uniform_sphere_discretization,
            None,
            X,
            imp_distrs_axes,
            imp_distrs_rot,
            params_learning_alg.init_unif_prop,
            0,
            params_learning_alg,
            output_dir,
            ground_truth=None,
            file_names=None,
            folder_views_selected=None,
            gpu=gpu,
        )
        self._volume = volume_representation.get_image_from_fourier_representation()
        self._energies = np.mean(energies_each_view, axis=0)
        self._num_iter = itr

        return self
