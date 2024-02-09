import numpy as np

from .volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import (
    make_grid,
    nd_gaussian,
)


class ParametersMainAlg:
    def __init__(
        self,
        M_axes=360**2,
        M_rot=360,
        dec_prop=1.2,
        init_unif_prop=(1, 1),
        coeff_kernel_axes=50.0,
        coeff_kernel_rot=5.0,
        eps=0,
        lr=0.1,
        N_axes=25,
        N_rot=20,
        prop_min=0,
        interp_order=3,
        N_iter_max=20,
        gaussian_kernel=True,
        N_iter_with_unif_distr=None,
        epochs_of_suppression=None,
        proportion_of_views_suppressed=None,
        convention="XZX",
        dtype="float32",
        reg_coeff=0,
        beta_sampling=0,
        epoch_length=None,
        batch_size=1,
        beta_grad=0,
        random_sampling=False,
    ):
        self.params = locals()
        self.M_axes = M_axes
        self.M_rot = M_rot
        self.dec_prop = dec_prop
        self.coeff_kernel_axes = coeff_kernel_axes
        self.coeff_kernel_rot = coeff_kernel_rot
        self.eps = eps
        self.lr = lr
        self.N_axes = N_axes
        self.N_rot = N_rot
        self.prop_min = prop_min
        self.interp_order = interp_order
        self.gaussian_kernel = gaussian_kernel
        self.N_iter_with_unif_distr = N_iter_with_unif_distr
        self.epochs_of_suppression = epochs_of_suppression
        self.proportions_of_views_suppressed = proportion_of_views_suppressed
        self.N_iter_max = N_iter_max
        self.convention = convention
        self.init_unif_prop = init_unif_prop
        self.dtype = dtype
        self.reg_coeff = reg_coeff
        self.beta_sampling = beta_sampling
        self.epoch_length = epoch_length
        self.batch_size = batch_size
        self.beta_grad = beta_grad
        self.random_sampling = random_sampling


class ParametersDataGeneration:
    def __init__(
        self,
        nb_dim=3,
        nb_views=10,
        sig_z=5,
        sig_xy=1,
        snr=100,
        order=3,
        sigma_trans_ker=0,
        size=50,
        psf=None,
        projection=False,
        rot_vecs=None,
        convention="zxz",
        partial_labelling=False,
        **partial_labelling_args,
    ):
        self.params = locals()
        self.nb_views = nb_views
        self.sig_z = sig_z
        self.sig_xy = sig_xy
        self.snr = snr
        self.sigma_trans_ker = sigma_trans_ker
        self.size = size
        self.grid_step = 2 / (size - 1)
        self.psf = psf
        self.partial_labelling = partial_labelling
        self.partial_labelling_args = partial_labelling_args
        self.projection = projection
        self.rot_vecs = rot_vecs
        self.convention = convention
        self.nb_dim = nb_dim
        self.order = order

    def get_cov_psf(self):
        cov_PSF = self.grid_step**2 * np.eye(self.nb_dim)
        cov_PSF[0, 0] *= self.sig_z**2
        for i in range(1, self.nb_dim):
            cov_PSF[i, i] *= self.sig_xy**2
        return cov_PSF

    def get_psf(self):
        if self.psf is None:
            cov_psf = self.get_cov_psf()
            grid = make_grid(self.size, self.nb_dim)
            psf = nd_gaussian(grid, np.zeros(self.nb_dim), cov_psf, self.nb_dim)
            psf /= np.sum(psf)
            return psf
        else:
            return self.psf / np.sum(self.psf)


class ParametersGMM:
    def __init__(
        self,
        nb_gaussians_init=10,
        nb_gaussians_ratio=4,
        sigma_init=0.2,
        sigma_ratio=1.4,
        nb_steps=4,
        threshold_gaussians=0.01,
        unif_prop_mins=[0.5, 0.25, 0.125, 0],
        init_with_views=True,
    ):
        self.nb_gaussians_init = nb_gaussians_init
        self.nb_gaussians_ratio = nb_gaussians_ratio
        self.sigma_init = sigma_init
        self.sigma_ratio = sigma_ratio
        self.nb_steps = nb_steps
        self.threshold_gaussians = threshold_gaussians
        self.unif_prop_mins = unif_prop_mins
        self.init_with_views = init_with_views
