import numpy as np


def gaussian_mixture_isotrop_identical_gaussians(
    grid, coeffs, centers, sigma, nb_dim, alpha
):
    covs = [sigma**2 * np.eye(nb_dim) for _ in range(len(coeffs))]
    gauss_mixture, _ = gaussian_mixture(grid, coeffs, centers, covs, nb_dim, alpha)
    return gauss_mixture


def gaussian_mixture(
    grid, coeffs, centers, covs, nb_dim, alpha, return_gaussians=False
):
    """computes the values of a multivariate gaussian mixture on a 3d grid.
    It truncates the gaussians around their centers
    IN :
    - grid : 4d array of shape  (d, h, w, 3) set of 3d voxels organised in a 3d volume.
             Values are computed on these voxels
    - centers : array of shape (Nb_gaussians, 3,1)
    - covs : covariance matrices of the gaussians, shape (Nb_gaussians, 3,3)
    - coeffs : linear coefficient of each gaussian
    OUT : array of shape (d, h, w) containing the values taken
          by the gaussian mixture at the voxels on the grid
    """
    shp = tuple([grid.shape[d] for d in range(nb_dim)])
    gauss_mixture = np.zeros(shp)
    gaussians = []
    for i in range(coeffs.shape[0]):
        gaussian = truncated_nd_gaussian(
            grid,
            gauss_mixture,
            coeffs[i],
            centers[i],
            covs[i],
            nb_dim,
            alpha,
            return_gaussians,
        )
        if return_gaussians:
            gaussians.append(gaussian)
    return gauss_mixture, gaussians


def truncated_nd_gaussian(
    grid, gaussian_mixture, coeff, center, cov, nb_dim, alpha, return_gaussian=False
):
    """computes the values of a multivariate gaussian on a 3d grid.
    It truncates the gaussian around the center
    IN :
    - grid : 4d array of shape (d, h, w, 3) set of 3d voxels organised in a 3d volume.
             Values are computed on these voxels
    - center : array of shape (3)
    - cov : covariance matrix of the gaussian, shape (3,3)
    - half_size_cut : half_size of the cut
    OUT : array of shape (d, h, w) containing the values taken by the gaussian at
          the voxels on the grid
    """
    grid_step = 2 / (len(grid) - 1)
    center_int = [int((center[d] - 1) / grid_step + len(grid)) for d in range(nb_dim)]
    # center_int = [int(grid_reverse_transform(center[d])) for d in range(nb_dim)]
    mins = []
    maxs = []
    lenghts = []
    slices = []
    for d in range(nb_dim):
        half_size_cut = int(alpha * np.sqrt(cov[d, d]) / grid_step + 1)
        mi = max(0, center_int[d] - half_size_cut)
        mi = min(mi, grid.shape[d])
        mins.append(mi)
        ma = min(grid.shape[d], center_int[d] + half_size_cut)
        ma = max(0, ma)
        maxs.append(ma)
        lenghts.append(ma - mi)
        slices.append(slice(mi, ma, 1))

    slices = tuple(slices)
    cut_grid = grid[slices]
    res_cut_grid = nd_gaussian(cut_grid, center, cov, nb_dim)
    shp = tuple([grid.shape[d] for d in range(nb_dim)])
    gaussian_mixture[slices] += coeff * res_cut_grid
    if return_gaussian:
        gaussian = np.zeros(shp)
        gaussian[slices] = res_cut_grid
        return gaussian
    return 0


def one_d_gaussian(points, center, sig):
    centered_grid = points - center
    A = centered_grid**2 / sig**2
    return np.exp(-A / 2)


def nd_gaussian_with_points(points, center, cov):
    centered_grid = points - center
    centered_grid = centered_grid.T
    inv_cov = np.linalg.inv(cov)
    center_grid_dot_cov = inv_cov @ centered_grid
    A = (center_grid_dot_cov * centered_grid).sum(axis=0)
    gauss = np.exp(-A / 2)
    return gauss


def flattens_grid(grid, nb_dim):
    shp = grid.shape
    size_flatten = 1
    for i in range(nb_dim):
        size_flatten *= shp[i]
    flatten_grid = grid.reshape((size_flatten, nb_dim))
    return flatten_grid, shp


def nd_gaussian(grid, center, cov, nb_dim):
    flatten_grid, shp = flattens_grid(grid, nb_dim)
    gauss = nd_gaussian_with_points(flatten_grid, center, cov)
    shape_unflatten = tuple([shp[i] for i in range(nb_dim)])
    return gauss.reshape(shape_unflatten)


def make_grid(size, nb_dim, mi=-1, ma=1):
    slices = []
    for d in range(nb_dim):
        slices.append(slice(0, size, 1))
    slices = tuple(slices)
    transpose_idx = list(range(1, nb_dim + 1))
    transpose_idx.append(0)
    grid = np.mgrid[slices].transpose(*transpose_idx)
    if mi is not None and ma is not None:
        grid_step = (ma - mi) / (size - 1)
        grid = grid_step * (grid - size + 1) + ma
    return grid
