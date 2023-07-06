from .params import ParametersMainAlg
from .learning_algorithms.gradient_descent_importance_sampling import (
    gd_importance_sampling_3d,
)
from .volume_representation.pixel_representation import Fourier_pixel_representation
from .volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import (
    make_grid,
    nd_gaussian,
)
from .common_image_processing_methods.rotation_translation import (
    discretize_sphere_uniformly,
)

import numpy as np


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
        self.fourier_volume = Fourier_pixel_representation(
            3,
            psf.shape[0],
            psf,
            init_vol=None,
            random_init=True,
            dtype=params_learning_alg.dtype,
        )

        N = X.shape[0]
        uniform_sphere_discretization = discretize_sphere_uniformly(
            params_learning_alg.M_axes, params_learning_alg.M_rot
        )
        imp_distrs_axes = (
            np.ones((N, params_learning_alg.M_axes)) / params_learning_alg.M_axes
        )
        imp_distrs_rot = (
            np.ones((N, params_learning_alg.M_rot)) / params_learning_alg.M_rot
        )

        gd_importance_sampling_3d(
            self.fourier_volume,
            uniform_sphere_discretization,
            None,
            X,
            imp_distrs_axes,
            imp_distrs_rot,
            params_learning_alg.init_unif_prop,
            0,
            params_learning_alg,
            False,
            output_dir,
            ground_truth=None,
            file_names=None,
            folder_views_selected=None,
            gpu=gpu,
        )

        return self
