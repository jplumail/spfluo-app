from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from spfluo.ab_initio_reconstruction.common_image_processing_methods.others import (
    normalize,
)
from spfluo.utils.array import array_namespace, get_device
from spfluo.utils.volume import (
    discretize_sphere_uniformly,
)

from .learning_algorithms.gradient_descent_importance_sampling import (
    gd_importance_sampling_3d,
)
from .params import ParametersMainAlg
from .volume_representation.pixel_representation import Fourier_pixel_representation

if TYPE_CHECKING:
    from spfluo.utils.array import Array


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
        X: "Array",
        psf: Optional["Array"] = None,
        output_dir: Optional[str] = None,
        minibatch_size: Optional[int] = None,
        particles_names: Optional[list[str]] = None,
    ):
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
        # normalize views
        xp = array_namespace(X)
        X = xp.stack([normalize(X[i]) for i in range(N)])

        uniform_sphere_discretization = discretize_sphere_uniformly(
            xp,
            params_learning_alg.M_axes,
            params_learning_alg.M_rot,
            dtype=xp.float64,
        )
        imp_distrs_axes = (
            xp.ones((N, params_learning_alg.M_axes)) / params_learning_alg.M_axes
        )
        imp_distrs_rot = (
            xp.ones((N, params_learning_alg.M_rot)) / params_learning_alg.M_rot
        )

        (
            recorded_energies,
            recorded_shifts,
            unif_prop,
            reconstruction,
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
            folder_views_selected=None,
            xp=xp,
            device=get_device(X),
            minibatch_size=minibatch_size,
            callback=self.callback,
            particles_names=particles_names,
        )
        self._volume = xp.asarray(reconstruction)
        self._energies = xp.mean(xp.asarray(energies_each_view), axis=0)
        self._num_iter = itr
        self._poses = xp.asarray(ests_poses)

        return self
