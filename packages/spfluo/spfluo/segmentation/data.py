import os
from abc import abstractmethod
from copy import deepcopy
from time import time
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pyfigtree import figtree
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from scipy.spatial.transform import Rotation
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F

from spfluo.picking.modules.pretraining.generate.data_generator import DataGenerator
from spfluo.segmentation.create_centriole import get_radius, simu_cylinder_rev
from spfluo.utils.loading import load_array, load_pointcloud
from spfluo.utils.volume import get_random_3d_vector


class SamplingStrategy:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return torch.as_tensor(self.func(*self.args, **self.kwargs))


class Uniform3DAngle(SamplingStrategy):
    def __init__(self):
        super().__init__(self.func)

    def func(self):
        a = torch.rand(size=(3,))
        a[[0, 2]] *= 360
        a[1] *= 180
        return a


class Normal(SamplingStrategy):
    def __init__(self, size=(1,), loc=0, scale=1):
        super().__init__(self.func, size, loc, scale)

    def func(self, size, loc, scale):
        return torch.randn(size=size) * scale + loc


class DiscreteUniform(SamplingStrategy):
    def __init__(self, low, high):
        super().__init__(torch.randint, size=(1,), low=low, high=high)


class Uniform(SamplingStrategy):
    def __init__(self, low=0, high=1):
        super().__init__(self.func, low, high)

    def func(self, low, high):
        return torch.rand(1) * (high - low) + low


class Parameter(torch.nn.parameter.Parameter):
    def __new__(cls, arg, size=(1,), dtype=None, min=None, max=None):
        if isinstance(arg, SamplingStrategy):
            data = torch.empty(size, dtype=dtype)
        else:
            data = torch.tensor(arg)
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self, arg, size=(1,), dtype=None, min=None, max=None):
        if isinstance(arg, SamplingStrategy):
            self.sampling_strategy = arg
        else:
            self.sampling_strategy = None
        self.clamp_params = {"min": min, "max": max}
        if min is not None and max is not None:
            self.do_clamp = True
        else:
            self.do_clamp = False

    def sample(self):
        if self.sampling_strategy is None:
            return self.data
        else:
            self.data = torch.as_tensor(
                self.sampling_strategy(), dtype=self.dtype, device=self.device
            )
            if self.do_clamp:
                self.data.clamp(**self.clamp_params)
            return self.data


class _AbstractBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad_(False)

    @abstractmethod
    def transform(self, data):
        pass

    def get_info(self):
        return dict(
            map(
                lambda x: (x[0], x[1].clone().float().squeeze().numpy()),
                self.named_parameters(recurse=False),
            )
        )

    def update_info(self, info):
        if self.__class__.__name__ not in info:
            info[self.__class__.__name__] = {}
        info[self.__class__.__name__].update(self.get_info())
        info[self.__class__.__name__].update({"applied": True})


class _AbstractAugment(_AbstractBlock):
    def __init__(self):
        super().__init__()

    def sample_params(self):
        return dict(
            map(lambda x: (x[0], x[1].sample()), self.named_parameters(recurse=False))
        )

    def forward(self, data):
        # t0 = time()
        self.sample_params()
        data = self.transform(data)
        # print(f"{self.__class__.__name__}: {time()-t0:.2f}s")
        return data


class _AbstractTransform(_AbstractBlock):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        return self.transform(data)


class Sometimes(torch.nn.Module):
    def __init__(self, block: _AbstractBlock, p: float):
        super().__init__()
        self.block = block
        self.block_name = self.block.__class__.__name__
        self.p = p

    def forward(self, data):
        transformation_applied = torch.rand(1) < self.p
        if transformation_applied:
            data = self.block(data)
            info = data[-1]
        else:
            info = data[-1]
            self.block.update_info(info)
            for k in info[self.block_name]:
                info[self.block_name][k] += np.nan
            info[self.block_name]["applied"] = False
        return data


class _Generator(torch.nn.Sequential):
    def __init__(self):
        self.blocks = [
            CreatePointcloud,
            Rotate,
            Shrink,
            MakeHoles,
            Translate,
            DropPointcloud,
            Voxelise,
            CreateMask,
            AddRandomParticle,
            AddOutlierClusters,
            ApplyPSF,
            AddGaussianNoise,
            AddPoissonNoise,
        ]
        self.params = {c.__name__: {"p": 1, "args": {}} for c in self.blocks}

    def build_module(self):
        super().__init__(*self.build_pipeline(self.blocks, self.params))
        groundtruth_gen = torch.nn.Sequential(
            *self.build_pipeline(
                [CreatePointcloud, Voxelise],
                {
                    "CreatePointcloud": self.params["CreatePointcloud"],
                    "Voxelise": self.params["Voxelise"],
                },
            )
        )
        self.groundtruth = groundtruth_gen({})

    def build_pipeline(self, blocks, params):
        initialized_blocks = []
        for b_class in blocks:
            b = b_class(**params[b_class.__name__]["args"])
            p = params[b_class.__name__]["p"]
            if p < 1:
                initialized_blocks.append(Sometimes(b, p))
            else:
                initialized_blocks.append(b)
        return initialized_blocks

    def get_groundtruth(self):
        return self.groundtruth[0], self.groundtruth[2]


class _RandomParticle(_Generator):
    def __init__(self):
        super().__init__()

        self.params["CreatePointcloud"]["args"] = dict(
            a=-0.5, b=5.4, c=103.253, r=15, L=600, s=15, angle=110, N=1000
        )
        self.params["Rotate"]["args"]["rotation_angles"] = Uniform3DAngle()
        self.params["Shrink"]["args"]["factor"] = Normal(loc=1, scale=0.1)
        self.params["MakeHoles"]["args"] = dict(
            nb_holes=DiscreteUniform(0, 10), mean_ratio=0.5, std_ratio=0.25
        )
        self.params["Translate"]["args"]["tvec_norm"] = Normal(loc=100, scale=10)
        self.params["DropPointcloud"]["args"]["drop"] = True
        self.params["Voxelise"]["args"] = dict(
            center=(0, 0, 0), img_shape=(64, 64, 64), resolution=20
        )

        self.params["Shrink"]["p"] = 0.5
        self.params["MakeHoles"]["p"] = 0.5
        self.params["Translate"]["p"] = 0.5

        self.blocks = self.blocks[:8]
        self.blocks.pop(5)


class _DefaultGenerator(_Generator):
    def __init__(self):
        super().__init__()

        self.params["CreatePointcloud"]["args"] = dict(
            a=-0.5, b=5.4, c=103.253, r=15, L=600, s=15, angle=110, N=1000
        )
        self.params["Rotate"]["args"]["rotation_angles"] = Uniform3DAngle()
        self.params["Shrink"]["args"]["factor"] = Normal(loc=0.6, scale=0.01)
        self.params["MakeHoles"]["args"] = dict(
            nb_holes=DiscreteUniform(0, 10), mean_ratio=0.5, std_ratio=0.25
        )
        self.params["Translate"]["args"]["tvec_norm"] = Normal(loc=100, scale=10)
        self.params["DropPointcloud"]["args"]["drop"] = True
        self.params["Voxelise"]["args"] = dict(
            center=(0, 0, 0), img_shape=(64, 64, 64), resolution=20
        )
        rp = _RandomParticle()
        rp.build_module()
        self.params["AddRandomParticle"]["args"] = dict(
            create_particle=rp, distance=Normal(loc=5, scale=2), size=(64, 64, 64)
        )
        self.params["AddOutlierClusters"]["args"]["nb_clusters"] = DiscreteUniform(
            0, 10
        )
        self.params["AddGaussianNoise"]["args"] = dict(
            target_snr=Normal(loc=15, scale=2), gaussian_noise_mean=Uniform()
        )
        self.params["ApplyPSF"]["args"] = dict(
            cov_coef_sampling_strategy=Normal(loc=2, scale=1),
            anisotropy_sampling_strategy=Normal(loc=2, scale=0.5),
        )

        # set default prob
        self.params["Shrink"]["p"] = 0.7
        self.params["MakeHoles"]["p"] = 0.5
        self.params["Translate"]["p"] = 0.7
        self.params["DropPointcloud"]["p"] = 0.1
        self.params["AddRandomParticle"]["p"] = 0.7
        self.params["AddOutlierClusters"]["p"] = 0.8
        self.params["AddGaussianNoise"]["p"] = 0.6
        self.params["AddPoissonNoise"]["p"] = 0.6
        self.params["ApplyPSF"]["p"] = 1


class NoAugGenerator(_DefaultGenerator):
    def __init__(self, psf_coef=(0.1, 0.1, 0.1)):
        super().__init__()
        aug_blocks = [
            "Shrink",
            "Translate",
            "MakeHoles",
            "DropPointcloud",
            "AddRandomParticle",
            "AddOutlierClusters",
            "AddGaussianNoise",
            "AddPoissonNoise",
        ]
        for b in aug_blocks:
            self.params[b]["p"] = 0

        self.params["ApplyPSF"]["args"].update(
            dict(
                cov=psf_coef,
                cov_coef_sampling_strategy=None,
                anisotropy_sampling_strategy=None,
            )
        )

        self.build_module()


class SingleParticleGenerator_translation_holes(_DefaultGenerator):
    def __init__(
        self,
        image_shape=(32, 32, 32),
        voxelise_res=30,
        psf_coef=(0.1, 0.1, 0.1),
        nb_holes=10,
        tvec_norm=Normal(loc=100, scale=10),
    ):
        super().__init__()
        aug_blocks = [
            "Shrink",
            "DropPointcloud",
            "AddRandomParticle",
            "AddOutlierClusters",
            "AddGaussianNoise",
            "AddPoissonNoise",
        ]
        for b in aug_blocks:
            self.params[b]["p"] = 0

        self.params["Voxelise"]["args"]["resolution"] = voxelise_res

        self.params["Translate"]["args"]["tvec_norm"] = tvec_norm
        if nb_holes == 0:
            self.params["MakeHoles"]["args"] = dict(
                nb_holes=0, mean_ratio=0.5, std_ratio=0.25
            )
        else:
            self.params["MakeHoles"]["args"] = dict(
                nb_holes=DiscreteUniform(0, nb_holes), mean_ratio=0.5, std_ratio=0.25
            )
        self.params["Voxelise"]["args"].update({"img_shape": image_shape})
        self.params["Translate"]["p"] = 1
        self.params["MakeHoles"]["p"] = 1

        self.params["ApplyPSF"]["args"].update(
            dict(
                cov=psf_coef,
                cov_coef_sampling_strategy=None,
                anisotropy_sampling_strategy=None,
            )
        )

        self.build_module()


class RandomParticle_variableSize(_RandomParticle):
    def __init__(self, low, high=None):
        super().__init__()

        if high is None:
            high = low
        self.params["CreatePointcloud"]["args"].update(
            dict(
                L=Uniform(low, high),
            )
        )
        self.params["Shrink"]["args"].update(
            dict(
                factor=1.0,
            )
        )
        self.build_module()


class Generator_variableSize(_DefaultGenerator):
    def __init__(self, low, high=None):
        super().__init__()

        if high is None:
            high = low

        self.params["CreatePointcloud"]["args"].update(
            dict(
                L=Uniform(low, high),
            )
        )
        self.params["Shrink"]["args"].update(
            dict(
                factor=1.0,
            )
        )
        self.params["AddRandomParticle"]["args"].update(
            dict(create_particle=RandomParticle_variableSize(low, high))
        )

        self.build_module()


class ISIMParticle(_RandomParticle):
    def __init__(self):
        super().__init__()

        # Change params
        self.params["CreatePointcloud"]["args"].update(
            dict(
                a=Uniform(-0.535, 0),
                b=Uniform(5.4, 6),
                c=Uniform(103.253, 105),
                r=Uniform(10, 20),
                L=900,
                s=Uniform(10, 20),
                angle=Uniform(0, 180),
                N=1000,
            )
        )

        self.params["Voxelise"]["args"].update(
            {"img_shape": (64, 64, 64), "resolution": 20}
        )

        self.params["Shrink"]["args"]["factor"] = Normal(loc=1, scale=0.1)

        self.params["MakeHoles"]["args"] = dict(
            nb_holes=DiscreteUniform(100, 200), mean_ratio=0.1, std_ratio=0.02
        )

        self.build_module()


class ISIMGenerator(_DefaultGenerator):
    def __init__(self):
        super().__init__()

        # Change pointcloud loading
        # blocks[0] = CreatePointcloud
        # params.update({CreatePointcloud.__name__: {'p': 1, 'args': {}}})

        # Change params
        self.params["CreatePointcloud"]["args"].update(
            dict(
                a=Uniform(-0.535, 0),
                b=Uniform(5.4, 6),
                c=Uniform(103.253, 105),
                r=Uniform(10, 20),
                L=900,
                s=Uniform(10, 20),
                angle=Uniform(0, 180),
                N=1000,
            )
        )

        self.params["Shrink"]["args"]["factor"] = Normal(loc=1, scale=0.1)

        self.params["MakeHoles"]["args"] = dict(
            nb_holes=DiscreteUniform(100, 200), mean_ratio=0.1, std_ratio=0.02
        )

        self.params["Voxelise"]["args"].update(
            {"img_shape": (64, 64, 64), "resolution": 20}
        )

        self.params["AddRandomParticle"]["args"].update(
            dict(
                distance=Normal(loc=5, scale=2),
                size=(64, 64, 64),
                create_particle=ISIMParticle(),
            )
        )

        self.params["AddGaussianNoise"]["args"] = dict(
            target_snr=Normal(loc=20, scale=2),
            gaussian_noise_mean=Uniform(low=0.001, high=0.01),
        )

        self.params["ApplyPSF"]["args"] = dict(
            psf_path="/data/plumail/real_data/inputs/ISIM/psf.tif"
        )

        self.params["AddOutlierClusters"]["args"] = dict(
            nb_clusters=DiscreteUniform(100, 200),
            radius_ranges=((1, 6), (1, 6), (1, 6)),
            nb_points_range=(0, 20),
            intensity_range=(0.1, 0.5),
        )

        self.params["Shrink"]["p"] = 0.7
        self.params["MakeHoles"]["p"] = 1.0
        self.params["Translate"]["p"] = 0.8
        self.params["DropPointcloud"]["p"] = 0.0
        self.params["AddRandomParticle"]["p"] = 1.0
        self.params["AddOutlierClusters"]["p"] = 1.0
        self.params["AddGaussianNoise"]["p"] = 1.0
        self.params["AddPoissonNoise"]["p"] = 1.0
        self.params["ApplyPSF"]["p"] = 1

        self.build_module()


class Shrink(_AbstractAugment):
    def __init__(self, factor=1):
        super().__init__()
        self.factor = Parameter(factor, min=0.1)

    def transform(self, data):
        pointcloud, true_pointcloud, info = data
        self.update_info(info)
        if self.factor != 1:
            m = true_pointcloud.mean(dim=0)

            def shrink(x):
                return m + (x - m) / self.factor

            pointcloud = shrink(pointcloud)
            true_pointcloud = shrink(true_pointcloud)
        return pointcloud, true_pointcloud, info


class DropPointcloud(_AbstractAugment):
    def __init__(self, drop=True):
        super().__init__()
        self.empty_pointcloud = torch.Tensor([[], [], []]).permute(1, 0)
        self.pointcloud_dropped = Parameter(drop, dtype=torch.bool)

    def transform(self, data):
        pointcloud, true_pointcloud, info = data
        self.update_info(info)
        if self.pointcloud_dropped == 1:
            return self.empty_pointcloud.clone(), self.empty_pointcloud.clone(), info
        else:
            return pointcloud, true_pointcloud, info


class Translate(_AbstractAugment):
    def __init__(self, tvec_norm=0):
        super().__init__()
        if callable(tvec_norm):
            tvec_sampling_strategy = SamplingStrategy(
                lambda: get_random_3d_vector(tvec_norm())
            )
        else:
            tvec_sampling_strategy = SamplingStrategy(
                lambda: get_random_3d_vector(tvec_norm)
            )
        self.tvec = Parameter(tvec_sampling_strategy, size=(3,))

    def transform(self, data):
        pointcloud, true_pointcloud, info = data
        self.update_info(info)
        pointcloud += self.tvec
        true_pointcloud += self.tvec
        return pointcloud, true_pointcloud, info


class Rotate(_AbstractAugment):
    def __init__(self, rotation_angles=Uniform3DAngle()):
        super().__init__()
        self.rotation_angles = Parameter(rotation_angles, size=(3,))

    def transform(self, data):
        pointcloud, true_pointcloud, info = data
        self.update_info(info)

        rot = Rotation.from_euler(
            "ZXZ", self.rotation_angles.cpu().numpy(), degrees=True
        )
        rot_mat = torch.as_tensor(
            rot.as_matrix(), dtype=pointcloud.dtype, device=pointcloud.device
        )
        pointcloud = (rot_mat @ pointcloud[:, [2, 1, 0]].T).T
        true_pointcloud = (rot_mat @ true_pointcloud[:, [2, 1, 0]].T).T

        return pointcloud[:, [2, 1, 0]], true_pointcloud[:, [2, 1, 0]], info


class MakeHoles(_AbstractAugment):
    def __init__(self, nb_holes=0, mean_ratio=0, std_ratio=0):
        super().__init__()
        self.nb_holes = Parameter(nb_holes)
        self.mean_ratio = Parameter(mean_ratio)
        self.std_ratio = Parameter(std_ratio)

    def get_center_and_radius(
        self, pointcloud: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Returns the center and the radius of a given point cloud.
        The center is the mean coordinate.
        The radius is given by the max distance between
        the center and any point from the point cloud.

        Args:
            pointcloud (torch.Tensor): Point cloud tensor, of shape (N, 3).

        Returns:
            Tuple[torch.Tensor, float]:
                * The center coordinates (x, y, z), of shape (3, ).
                * The radius as a single float.
        """
        center = pointcloud.mean(dim=0)
        distances = torch.norm(pointcloud - center, dim=1)
        return center, distances.max()

    def uniform_in_sphere(
        self, center: torch.Tensor, radius: float, N: int = 1
    ) -> torch.Tensor:
        """Generate N points uniformly in the sphere(center, radius).
        Do so by generating points in a cube(center, radius) and discarding points
        outside the sphere.

        Args:
            center (torch.Tensor): Sphere's center coordinate tensor, of shape (3, ).
            radius (float): Sphere's radius.
            N (int, optional): Number of points generated. Defaults to 1.

        Returns:
            torch.Tensor: Generated point cloud, of shape (N, 3).
        """
        N = int(N)
        points = center.new_zeros((N, 3))
        selected = 0
        while selected < N:
            remainder = N - selected
            sel = min(remainder, N)
            point = (
                2 * (torch.rand(2 * sel, 3, device=points.device) - 0.5) * radius
                + center
            )
            point_distance = torch.norm(point - center, dim=1, keepdim=True)
            point = point[(point_distance <= radius).squeeze()]
            sel = min(sel, point.size(0))
            points[selected : (selected + sel)] = point[:sel]
            selected += sel
        return points

    def get_params(self, pointcloud):
        center, radius = self.get_center_and_radius(pointcloud)
        holes_centers = self.uniform_in_sphere(center, radius, N=self.nb_holes)
        mean, std = radius * self.mean_ratio, radius * self.std_ratio
        return holes_centers, mean, std

    def transform(self, data: Tuple[torch.Tensor]) -> torch.Tensor:
        """Simulate non uniform density of point cloud.
        Do so by randomly picking center and apply gaussian_digger onto them.

        Args:
            pointcloud (torch.Tensor): Point cloud tensor, of shape (N, 3).
            nb_holes_min (int, optional): Min number of intensity holes.
                Defaults to 0.
            nb_holes_max (int, optional): Max number of intensity holes.
                Defaults to 10.
            mean_ratio (float, optional): Holes will be from sphere following a
                gaussian proba of mean radius * mean_ratio. Defaults to 0.5.
            std_ratio (float, optional): Holes will be from sphere following a gaussia
                proba of std radius * mean_std. Defaults to 0.25.

        Returns:
            torch.Tensor: Randomly filtered point cloud, of shape (N', 3), where N' < N.
        """
        pointcloud_noisy, true_pointcloud, info = data
        self.update_info(info)
        hole_centers, mean_scaled, std_scaled = self.get_params(pointcloud_noisy)
        nb_holes = len(hole_centers)
        if nb_holes > 0:
            sigma = torch.clamp(
                mean_scaled
                + torch.randn(size=(nb_holes,), device=std_scaled.device) * std_scaled,
                min=1e-3,
            )
            cov = (sigma[:, None, None] ** 2) * torch.eye(3, device=sigma.device)
            holes_distributions = MultivariateNormal(
                hole_centers, covariance_matrix=cov
            )  # nb_holes multivariate normal distributions
            prob = torch.exp(
                holes_distributions.log_prob(pointcloud_noisy[:, None])
            ) / torch.exp(
                holes_distributions.log_prob(hole_centers)
            )  # shape (N, nb_holes)
            dices = torch.rand(prob.size(), device=prob.device)
            keep = (dices >= prob).prod(dim=1).type(torch.bool)
            return pointcloud_noisy[keep], true_pointcloud, info
        else:
            return pointcloud_noisy, true_pointcloud, info


class Voxelise(_AbstractTransform):
    def __init__(
        self,
        center=(0, 0, 0),
        img_shape=(65, 65, 65),
        resolution=30,
        bandwidth=10,
        epsilon=1e-3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ):
        super().__init__()
        self.img_shape = Parameter(img_shape)
        self.bandwidth = Parameter(bandwidth)
        self.epsilon = Parameter(epsilon)
        self.center = Parameter(center)
        self.resolution = Parameter(resolution)

        self.device = device
        self.dtype = dtype

        start = (
            torch.as_tensor(center).cpu()
            - resolution * torch.as_tensor(img_shape).cpu() / 2
        )
        end = start + resolution * torch.as_tensor(img_shape).cpu()

        target_points = torch.meshgrid(
            *[
                torch.linspace(st, e, sh, device=self.device)
                for st, e, sh in zip(start, end, img_shape)
            ],
            indexing="ij",
        )
        self.target_points = torch.stack(target_points, dim=0)

    @staticmethod
    def get_FOV(
        pointcloud: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[Tuple[float, float], ...]:
        """Get the Field Of View of a point cloud.

        Args:
            pointcloud (Union[torch.Tensor, np.ndarray]): shape (N, 3).

        Returns:
            Tuple[float]: x_min, x_max, y_min, y_max, z_min, z_max.
        """
        return tuple(
            [
                (pointcloud[:, i].min(), pointcloud[:, i].max())
                for i in range(pointcloud.shape[1])
            ]
        )

    def get_densities(self, pointcloud):
        if self.device.type == "cuda":
            return (
                1.0
                / (2.0 * torch.pi * self.bandwidth**2)
                * torch.sum(
                    torch.exp(
                        -(
                            (
                                pointcloud[:, :, None, None, None]
                                - self.target_points[None].to(pointcloud.device)
                            )
                            ** 2
                        ).sum(dim=1)
                        / (2 * self.bandwidth.to(pointcloud.device) ** 2)
                    ),
                    dim=0,
                )
            )
        else:  # use figtree
            weights = np.ones((len(pointcloud),))
            figtree_kwargs = {
                "bandwidth": self.bandwidth.cpu().numpy(),
                "epsilon": self.epsilon.cpu().numpy(),
            }
            if pointcloud.size(0) > 0:
                densities = figtree(
                    pointcloud.cpu().numpy().astype(np.float64),
                    self.target_points.cpu()
                    .numpy()
                    .astype(np.float64)
                    .transpose(1, 2, 3, 0)
                    .reshape(-1, 3),
                    weights,
                    **figtree_kwargs,
                )
                div = densities.max() - densities.min()
                if div > 0:
                    densities = (densities - densities.min()) / div
            else:
                densities = np.zeros(self.target_points.shape[1:])
            densities = torch.as_tensor(
                densities.reshape(self.target_points.shape[1:]),
                device=pointcloud.device,
                dtype=torch.float64,
            ).type(self.dtype)
            return densities

    def transform(self, data):
        pointcloud, true_pointcloud, info = data
        self.update_info(info)
        if "Translate" in "info" and info["Translate"]["applied"]:
            info["Translate"]["tvec"] -= self.center.cpu().numpy()
            info["Translate"]["tvec"] /= self.resolution.cpu().numpy()
        densities = self.get_densities(pointcloud)
        true_densities = self.get_densities(true_pointcloud)

        return densities, true_densities, info


class ApplyPSF(_AbstractAugment):
    def __init__(
        self,
        cov=(0.1, 0.1, 0.1),
        cov_coef_sampling_strategy=None,
        anisotropy_sampling_strategy=None,
        psf_path=None,
    ):
        super().__init__()
        if psf_path is not None and os.path.exists(psf_path):
            self.psf_path = psf_path
            self.psf = torch.as_tensor(load_array(psf_path), device="cpu").type(
                torch.float32
            )
        else:
            self.psf_path = None
            self.psf = None
            if (
                cov_coef_sampling_strategy is not None
                and anisotropy_sampling_strategy is not None
            ):
                cov_sampling = SamplingStrategy(
                    lambda: self.sample_cov(
                        cov_coef_sampling_strategy, anisotropy_sampling_strategy
                    )
                )
                self.cov_coef = Parameter(cov_sampling, size=(3,), min=0.1)
            else:
                self.cov_coef = Parameter(cov, min=0.1)

        """if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'"""
        self.device = "cpu"

    def sample_cov(self, cov_sampling_strategy, anisotropy_sampling_strategy):
        c = cov_sampling_strategy()
        a = max(1.0, anisotropy_sampling_strategy())
        return torch.tensor([a * c, c, c])

    def get_psf(self, dtype):
        cov = torch.eye(3, dtype=dtype, device=self.device)
        cov[[0, 1, 2], [0, 1, 2]] = self.cov_coef.type(dtype).to(self.device) ** 2
        return torch.as_tensor(
            DataGenerator.make_psf(
                tuple((cov[[0, 1, 2], [0, 1, 2]] ** 0.5).cpu().numpy()), np.float64
            ),
            device=self.device,
            dtype=dtype,
        )

    def transform(self, data):
        img, mask, info = data
        if self.psf_path is None:
            self.update_info(info)
            psf = self.get_psf(img.dtype)
        else:
            psf = self.psf
            info["ApplyPSF"] = {"applied": True, "psf_path": self.psf_path}

        info["ApplyPSF"]["psf"] = psf
        img = F.conv3d(
            img[None, None].to(self.device), psf[None, None], padding="same"
        )[0, 0].to(img.device)
        # div = img.max() - img.min()
        # if div > 0:
        #    img = (img - img.min()) / div
        return img, mask, info


class CreateMask(_AbstractTransform):
    def __init__(self):
        super().__init__()

    def transform(self, data: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        img, true_img, info = data
        im = true_img.cpu().numpy()
        mask = np.zeros_like(im).astype(int)
        grid = np.indices(mask[0].shape).reshape(2, -1)
        for i in range(mask.shape[0]):
            ind = (im[i] > 0).nonzero()
            if len(ind[0]) > 0:
                try:
                    hull = ConvexHull(np.stack(ind, axis=1))
                    grid_ = np.concatenate((grid, np.ones((1, grid.shape[1]))), axis=0)
                    inside = (hull.equations @ grid_ < -1e-9).all(axis=0)
                    inside_points = grid[:, inside]
                    mask[i, inside_points[0], inside_points[1]] = 1
                except QhullError:
                    pass

        self.update_info(info)
        mask = torch.as_tensor(mask, dtype=torch.int16, device=img.device)
        return img, mask, info


class AddRandomParticle(_AbstractAugment):
    def __init__(self, create_particle=None, distance=0, size=(64, 64, 64)):
        super().__init__()

        self.create_particle = create_particle
        self.distance = Parameter(distance, min=-min(size), max=min(size))

    def get_translation_vec(self, reference_mask, mask_to_move, direction, distance):
        D, H, W = reference_mask.size()
        mask_moved_padded = F.pad(mask_to_move, (W, W, H, H, D, D))
        ref_mask = reference_mask > 0
        upper_bound = min((D, H, W))
        lower_bound = 0
        intersection_dist = 0
        while np.abs(upper_bound - lower_bound) > 1:
            intersection_dist = (upper_bound + lower_bound) / 2
            tvec = torch.round(direction * intersection_dist).type(torch.int64)
            mask_moved = mask_moved_padded[
                D - tvec[0] : D - tvec[0] + D,
                H - tvec[1] : H - tvec[1] + H,
                W - tvec[2] : W - tvec[2] + W,
            ]
            intersect = (ref_mask * mask_moved).sum() > 0
            if intersect:
                lower_bound = intersection_dist
            else:
                upper_bound = intersection_dist
        return torch.round(direction * (intersection_dist + distance)).type(torch.int64)

    def translate(self, img, tvec):
        D, H, W = img.size()
        img_padded = F.pad(img, (W, W, H, H, D, D))
        return img_padded[
            D - tvec[0] : D - tvec[0] + D,
            H - tvec[1] : H - tvec[1] + H,
            W - tvec[2] : W - tvec[2] + W,
        ]

    def move(self, img_to_move, ref_mask, mask_to_move, direction, distance):
        tvec = self.get_translation_vec(ref_mask, mask_to_move, direction, distance)
        img_moved = self.translate(img_to_move, tvec)
        mask_moved = self.translate(mask_to_move, tvec)
        return img_moved, mask_moved

    def transform(self, data):
        img, mask, info = data
        self.update_info(info)
        img2, mask2, info2 = self.create_particle({})
        assert "AddRandomParticle" not in info2

        if not info["DropPointcloud"]["applied"]:
            # Sample params
            random_direction = torch.as_tensor(get_random_3d_vector()).to(img.device)

            img2_moved, mask2_moved = self.move(
                img2, mask, mask2, random_direction, self.distance
            )

            img += img2_moved
            label_max = mask.max()
            mask[mask2_moved > 0] += label_max + 1
            mask[mask > label_max + 1] = -1  # overlapping between 2 particles
        else:
            info["AddRandomParticle"]["applied"] = False

        return img, mask, info


class AddOutlierClusters(_AbstractAugment):
    def __init__(
        self,
        nb_clusters=0,
        radius_ranges=((20, 80), (20, 80), (25, 80)),
        nb_points_range=(20, 80),
        intensity_range=(0, 1),
    ):
        super().__init__()
        self.nb_clusters = Parameter(nb_clusters)
        self.radius_ranges = radius_ranges
        self.nb_points_range = nb_points_range
        self.intensity_range = intensity_range

    def sample_outliers_clusters(self, img):
        if self.nb_clusters > 0:
            radius = torch.stack(
                [
                    torch.randint(r[0], r[1], size=(int(self.nb_clusters),))
                    for r in self.radius_ranges
                ],
                dim=1,
            )
            cov = radius[:, None] ** 0.5 * torch.eye(3)
            center = torch.as_tensor(
                [
                    [
                        torch.randint(0, img.shape[i], size=(1,))
                        for i in range(radius.size(1))
                    ]
                    for j in range(radius.size(0))
                ]
            ).type(cov.dtype)
            low, high = self.nb_points_range
            nb_points = torch.randint(low, high, size=(int(self.nb_clusters),))
            dist = MultivariateNormal(center, cov)
            outliers_clusters = dist.sample((max(nb_points),)).permute(1, 0, 2)
            outliers_clusters = torch.round(outliers_clusters).type(torch.int64)
            outliers_clusters = [
                cluster[:n] for cluster, n in zip(outliers_clusters, nb_points)
            ]
            masks = [
                ((cluster > 0) * (cluster < torch.as_tensor(img.shape)))
                .prod(dim=1)
                .type(torch.bool)
                for cluster in outliers_clusters
            ]
            outliers_clusters = [
                cluster[mask] for cluster, mask in zip(outliers_clusters, masks)
            ]
            return outliers_clusters
        else:
            return []

    def transform(self, data):
        img, mask, info = data
        self.update_info(info)
        outliers_clusters = self.sample_outliers_clusters(img)
        low, high = self.intensity_range
        for cluster in outliers_clusters:
            intensities = (high - low) * torch.rand(
                (len(cluster),), device=img.device, dtype=img.dtype
            ) + low
            img[cluster[:, 0], cluster[:, 1], cluster[:, 2]] = intensities

        return img, mask, info


class AddGaussianNoise(_AbstractAugment):
    def __init__(
        self,
        target_snr=100,
        gaussian_noise_mean=0,
    ):
        super().__init__()
        self.target_snr = Parameter(target_snr)
        self.gaussian_noise_mean = Parameter(gaussian_noise_mean)

    def transform(self, data):
        img, mask, info = data
        self.update_info(info)

        d = img.max() - img.min()
        img = (img - img.min()) / d if d > 0 else img - img.min()
        image_average_db = 10 * torch.log10(torch.clamp(img.mean(), 0.01))
        noise_average_db = image_average_db - self.target_snr
        noise_average = 10 ** (noise_average_db / 10)
        noise = torch.normal(
            float(self.gaussian_noise_mean), float(noise_average) ** 0.5, size=img.shape
        ).to(img.device)
        img += noise
        img = torch.clamp(img, 0, 1)

        return img, mask, info


class AddPoissonNoise(_AbstractAugment):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        img, mask, info = data
        self.update_info(info)

        # Determine unique values in image & calculate the next power of two
        img = img.clamp(0, None)
        vals = torch.as_tensor(len(torch.unique(img)))
        vals = 2 ** torch.ceil(torch.log2(vals))

        # Generating noise for each unique value in image.
        img = torch.poisson(img * vals) / vals

        return img, mask, info


class LoadPointcloud(_AbstractTransform):
    def __init__(
        self, pointcloud_path="/data1/stage_Luc/inputs/sample_centriole_point_cloud.csv"
    ):
        super().__init__()
        self.pointcloud_path = pointcloud_path
        pointcloud = load_pointcloud(pointcloud_path)
        self.pointcloud = torch.as_tensor(pointcloud)[:, [2, 1, 0]]
        self.pointcloud -= self.pointcloud.mean(dim=0)

    def transform(self, data):
        info = data
        info["LoadPointcloud"] = {"applied": True}
        info["LoadPointcloud"]["pointcloud_path"] = self.pointcloud_path
        return self.pointcloud.clone(), self.pointcloud.clone(), info


class CreatePointcloud(_AbstractAugment):
    def __init__(self, a, b, c, r, L, s, angle, N, device=None, dtype=None):
        super().__init__()
        self.a = Parameter(a)
        self.b = Parameter(b)
        self.c = Parameter(c)
        self.r = Parameter(r)
        self.L = Parameter(L)
        self.s = Parameter(s)
        self.angle = Parameter(angle)
        self.N = N

        self.device = device
        self.dtype = dtype

    def create_centriole(self):
        x = np.arange(1, 14)
        pc = simu_cylinder_rev(
            get_radius(
                x, self.a.cpu().numpy(), self.b.cpu().numpy(), self.c.cpu().numpy()
            ),
            self.r.cpu().numpy(),
            self.L.cpu().numpy(),
            self.s.cpu().numpy(),
            self.N,
            self.angle.cpu().numpy(),
        )
        pc -= pc.mean(axis=0)
        return torch.as_tensor(pc, dtype=self.dtype, device=self.device)[:, [2, 1, 0]]

    def transform(self, data):
        info = data
        self.update_info(info)
        pointcloud = self.create_centriole()
        return pointcloud, pointcloud.clone(), info


class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, generator, length):
        super().__init__()
        self.generator = generator
        self.length = length

    def __getitem__(self, idx):
        im, mask, info = self.generator({})
        im = im.unsqueeze(0)
        return im, mask, info

    def __len__(self):
        return self.length


def unbatch_dict(d):
    res = []
    for k in d:
        if isinstance(d[k], dict):
            r = unbatch_dict(d[k])
            for i in range(len(r)):
                if len(res) < i + 1:
                    res.append({})
                res[i][k] = r[i]
        else:
            for i in range(len(d[k])):
                if len(res) < i + 1:
                    res.append({})
                res[i][k] = d[k][i]
    return res


def infos_to_dataframe(infos):
    b = {}
    for k in infos:
        if isinstance(infos[k], dict):
            for k2 in infos[k]:
                if len(infos[k][k2].shape) == 1:
                    b[k + "_" + k2] = infos[k][k2]
                elif len(infos[k][k2].shape) == 2:
                    for i in range(infos[k][k2].shape[1]):
                        b[k + "_" + k2 + "_" + str(i)] = infos[k][k2][:, i]
        else:
            b[k] = infos[k]

    return pd.DataFrame(b)


class HardCopyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, **kwargs):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, **kwargs)
        self.data = [
            (deepcopy(im[0]), deepcopy(mask[0]), deepcopy(unbatch_dict(info)[0]))
            for im, mask, info in loader
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


if __name__ == "__main__":
    import napari
    from torch.utils.data import DataLoader

    pc_path = "/data1/stage_Luc/inputs/sample_centriole_point_cloud.csv"
    ds = GeneratorDataset(ISIMGenerator(), 1000)
    print(ds)
    data = ds[0]

    """
    data = None
    for m in ds.generate_image[:3]:
        data = m(data)
    
    for m in ds.generate_image[3]:
        t0 = time()
        data = m(data)
        print(m, time()-t0)
    """

    data_loader = DataLoader(ds, batch_size=8, num_workers=64, pin_memory=True)

    i = 0
    while 1:
        im, mask, info = ds[0]
        im = im.cpu().numpy()[0]
        mask = mask.cpu().numpy()
        # print(info)
        # viewer = napari.Viewer(ndisplay=3)
        # viewer.add_image(im, contrast_limits=(0,1))
        # viewer.add_labels(mask, visible=False)
        # napari.run()

        mask_ = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        mask_[mask[mask.shape[0] // 2] == 1, :] = [0, 255, 0]
        mask_[mask[mask.shape[0] // 2] == 2, :] = [255, 0, 0]

        im = im[im.shape[0] // 2]
        im = (im - im.min()) / (im.max() - im.min())
        im = (im * 255).astype(np.uint8)
        im = Image.fromarray(im)
        mask = Image.fromarray(mask_)
        im.save(f"segmentation/data/generated/im{i}.png")
        mask.save(f"segmentation/data/generated/mask{i}.png")
        i += 1

    t0 = time()
    i = 0
    for im, mask, info in data_loader:
        print(i)
        im, mask = im.to("cuda"), mask.to("cuda")
        i += 1
    print(time() - t0)

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(im.cpu().numpy())
    viewer.add_labels(mask.cpu().numpy())
    napari.run()
