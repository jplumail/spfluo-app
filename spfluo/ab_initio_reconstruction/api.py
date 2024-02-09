from typing import Any, Callable, Optional

from spfluo.ab_initio_reconstruction.common_image_processing_methods.others import (
    normalize,
)
from spfluo.utils.array import numpy as np
from spfluo.utils.volume import (
    discretize_sphere_uniformly,
)

from .learning_algorithms.gradient_descent_importance_sampling import (
    gd_importance_sampling_3d,
)
from .params import ParametersMainAlg
from .volume_representation.pixel_representation import Fourier_pixel_representation


class AbInitioReconstruction:
    def __init__(
        self, callback: Callable[[np.ndarray, int], Any] | None = None, **params
    ):
        self.params = params
        self._volume = None
        self._energies = None
        self._num_iter = None
        self._poses = None
        self.callback = callback

    def fit(
        self,
        X: np.ndarray,
        psf: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None,
        gpu: Optional[str] = None,
        minibatch_size: Optional[int] = None,
        particles_names: Optional[list[str]] = None,
    ):
        """Reconstruct a volume based on views of particles"""
        if psf is None:
            raise NotImplementedError  # TODO : default psf to gaussian
        if output_dir is None:
            output_dir = "./ab-initio-output"

        if gpu == "pytorch":
            from spfluo.utils.array import torch

            xp = torch
            device = "cuda"
        elif gpu == "cupy":
            from spfluo.utils.array import cupy

            xp = cupy
            device = None
        elif gpu is None:
            xp = np
            device = None
        else:
            raise ValueError(f"Found {gpu=}")

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
        # normalize views
        X = np.stack([normalize(X[i]) for i in range(N)])

        uniform_sphere_discretization = discretize_sphere_uniformly(
            np,
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
            xp=xp,
            device=device,
            minibatch_size=minibatch_size,
            callback=self.callback,
            particles_names=particles_names,
        )
        self._volume = volume_representation.get_image_from_fourier_representation()
        self._energies = np.mean(energies_each_view, axis=0)
        self._num_iter = itr
        self._poses = ests_poses

        return self
