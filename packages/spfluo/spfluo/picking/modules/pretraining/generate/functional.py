"""This file defines functions to add random augmentation to a 3d point cloud in order to
generate data. It expects torch.Tensor point clouds and should operate indifferently on cpu
and gpu.
"""

from typing import Tuple, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

# _______________________________________________________________________________________________ #


def get_center_and_radius(pointcloud: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Returns the center and the radius of a given point cloud.
    The center is the mean coordinate.
    The radius is given by the max distance between the center and any point from the point cloud.

    Args:
        pointcloud (torch.Tensor): Point cloud tensor, of shape (N, 3).

    Returns:
        Tuple[torch.Tensor, float]: * The center coordinates (x, y, z), of shape (3, ).
                                    * The radius as a single float.
    """
    center = pointcloud.mean(dim=0)
    distances = torch.norm(pointcloud - center, dim=1)
    return center, distances.max().cpu().numpy().item()


# _______________________________________________________________________________________________ #


def get_FOV(pointcloud: Union[torch.Tensor, np.ndarray]) -> Tuple[float]:
    """Get the Field Of View of a point cloud.

    Args:
        pointcloud (Union[torch.Tensor, np.ndarray]): Point cloud array like, of shape (N, 3).

    Returns:
        Tuple[float]: x_min, x_max, y_min, y_max, z_min, z_max.
    """
    return (
        pointcloud[:, 0].min(),
        pointcloud[:, 0].max(),
        pointcloud[:, 1].min(),
        pointcloud[:, 1].max(),
        pointcloud[:, 2].min(),
        pointcloud[:, 2].max(),
    )


# _______________________________________________________________________________________________ #


def shrink(pointcloud: torch.Tensor, factor: float = 1500.0) -> torch.Tensor:
    """Shrink a point cloud tensor by a given factor and center it.

    Args:
        pointcloud (torch.Tensor): Point cloud tensor to be shrinked, of shape (N, 3). SHOULD BE CENTERED!
        factor (float, optional): Shrink factor. Defaults to 1500.0.

    Returns:
        torch.Tensor: Shrinked and centered point cloud tensor.
    """
    return pointcloud / factor


# _______________________________________________________________________________________________ #


def rotate(
    pointcloud: torch.Tensor,
    rotation_angles: Tuple[float] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Randomly rotate a pointcloud with a given probability. If no angles are provided, the
        rotation is determined by uniformly sampling 3 Euler's angles.

    Args:
        pointcloud (torch.Tensor): Point cloud tensor, of shape (N, 3).
        rotation_angles (Tuple[int]): zyx Euler's angles. If None, will be sampled uniformely in
                                      [0, 360]. Defaults to None.
        p (float, optional): Rotation probability. Defaults to 0.5.
        device (torch.device, optional): CPU or GPU. Defaults to torch.device('cpu').

    Returns:
        torch.Tensor: * Rotated point cloud tensor, of shape (N, 3).
    """
    if rotation_angles is None:
        rotation_angles = np.random.uniform(low=0, high=360, size=3)
    rot = Rotation.from_euler("XZX", rotation_angles, degrees=True)
    rot_mat = torch.as_tensor(rot.as_matrix(), dtype=pointcloud.dtype, device=device)
    return (rot_mat @ pointcloud.T).T


# _______________________________________________________________________________________________ #


def translate(
    pointcloud: torch.Tensor,
    translation: Tuple[float] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly rotate a pointcloud with a given probability. If no angles are provided, the
        rotation is determined by uniformly sampling 3 Euler's angles.

    Args:
        pointcloud (torch.Tensor): Point cloud tensor, of shape (N, 3).
        rotation_angles (Tuple[int]): zyx Euler's angles. If None, will be sampled uniformely in
                                      [0, 360]. Defaults to None.
        p (float, optional): Rotation probability. Defaults to 0.5.
        device (torch.device, optional): CPU or GPU. Defaults to torch.device('cpu').

    Returns:
        torch.Tensor: * Rotated point cloud tensor, of shape (N, 3).
    """
    if translation is None:
        return pointcloud
    t = torch.as_tensor(translation, dtype=pointcloud.dtype, device=pointcloud.device)
    return pointcloud + t


# _______________________________________________________________________________________________ #


def uniform_in_sphere(
    center: torch.Tensor,
    radius: float,
    N: int = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate N points uniformly in the sphere(center, radius).
    Do so by generating points in a cube(center, radius) and discarding points outside the
    sphere.

    Args:
        center (torch.Tensor): Sphere's center coordinate tensor, of shape (3, ).
        radius (float): Sphere's radius.
        N (int, optional): Number of points generated. Defaults to 1.

    Returns:
        torch.Tensor: Generated point cloud, of shape (N, 3).
    """
    points = torch.zeros((N, 3)).to(device)
    selected = 0
    while selected < N:
        remainder = N - selected
        sel = min(remainder, N)
        point = (
            2 * (torch.rand(2 * sel, 3, device=points.device) - 0.5) * radius + center
        )
        point_distance = torch.norm(point - center, dim=1, keepdim=True)
        point = point[(point_distance <= radius).squeeze()]
        sel = min(sel, point.size(0))
        points[selected : (selected + sel)] = point[:sel]
        selected += sel
    return points


# _______________________________________________________________________________________________ #


def gaussian_digger(
    pointcloud: torch.Tensor, center: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Randomly remove points from a point cloud following a gaussian probability with respect
    to a given center.

    Args:
        pointcloud (torch.Tensor): Point cloud tensor, of shape (N, 3).
        center (torch.Tensor): Gaussian distribution center. Tensor of shape (3, ).
        sigma (torch.Tensor): Gaussian distribution standard deviation. Tensor of shape (1, ).

    Returns:
        torch.Tensor: Randomly filtered point cloud, of shape (N', 3), where N' < N.
    """
    sigma_ = sigma.cpu().numpy()
    center_ = center.cpu().numpy()
    pc_ = pointcloud.detach().cpu().numpy()
    cov = sigma_**2 * np.eye(3)
    norm_factor = multivariate_normal.pdf(center_, mean=center_, cov=cov)
    thresholds = multivariate_normal.pdf(pc_, mean=center_, cov=cov) / norm_factor
    dices = np.random.uniform(low=0, high=1, size=(pointcloud.size(0)))
    choice_keep = dices >= thresholds
    return pointcloud[choice_keep]


# _______________________________________________________________________________________________ #


def non_uniform_density(
    pointcloud: torch.Tensor,
    nb_holes_min: int = 0,
    nb_holes_max: int = 10,
    mean_ratio: float = 0.5,
    std_ratio: float = 0.25,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Simulate non uniform density of point cloud.
    Do so by randomly picking center and apply gaussian_digger onto them.

    Args:
        pointcloud (torch.Tensor): Point cloud tensor, of shape (N, 3).
        nb_holes_min (int, optional): Min number of intensity holes. Defaults to 0.
        nb_holes_max (int, optional): Max number of intensity holes. Defaults to 10.
        mean_ratio (float, optional): Holes will be from sphere following a gaussian proba
                                      of mean radius * mean_ratio. Defaults to 0.5.
        std_ratio (float, optional): Holes will be from sphere following a gaussian proba
                                      of std radius * mean_std. Defaults to 0.25.

    Returns:
        torch.Tensor: Randomly filtered point cloud, of shape (N', 3), where N' < N.
    """
    nb_holes = torch.randint(low=nb_holes_min, high=nb_holes_max, size=(1, 1)).squeeze()
    if nb_holes == 0:
        return pointcloud
    else:
        center, radius = get_center_and_radius(pointcloud)
        radius = torch.Tensor([radius]).to(device)
        hole_centers = uniform_in_sphere(center, radius, N=nb_holes, device=device)
        mean, std = radius * mean_ratio, radius * std_ratio
        if nb_holes == 1:
            sigma = torch.clamp(torch.normal(mean=mean, std=std), min=1e-3)
            return gaussian_digger(
                pointcloud, center=hole_centers.squeeze(), sigma=sigma
            )
        else:
            for hole_center in hole_centers:
                sigma = torch.clamp(torch.normal(mean=mean, std=std), min=1e-3)
                pointcloud = gaussian_digger(
                    pointcloud, center=hole_center, sigma=sigma
                )
            return pointcloud
