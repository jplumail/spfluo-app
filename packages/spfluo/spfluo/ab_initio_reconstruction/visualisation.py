import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.interpolate import griddata


def plot_map(
    vals_to_plot: np.ndarray,
    thetas: np.ndarray,
    phis: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: Axes | None = None,
):
    if ax is None:
        ax: Axes = plt.subplots()[1]
    min_theta, max_theta, min_phi, max_phi = (
        np.min(thetas),
        np.max(thetas),
        np.min(phis),
        np.max(phis),
    )
    # min_theta, max_theta, min_phi, max_phi = 0,360,0,180
    grid_theta, grid_phi = np.mgrid[min_theta:max_theta:500j, min_phi:max_phi:500j]
    grid_z = griddata(
        np.array([thetas, phis]).T,
        vals_to_plot,
        (grid_theta, grid_phi),
        method="nearest",
    )
    mean_non_nan = np.nanmean(grid_z)
    ax.imshow(
        np.nan_to_num(grid_z.T, nan=mean_non_nan),
        extent=(min_theta, max_theta, min_phi, max_phi),
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )  # , origin='lower', extent=(0, 1, 0, 1))
    ax.set_xlabel("φ_1 (°)")
    ax.set_ylabel("φ_2 (°)")
    return ax
