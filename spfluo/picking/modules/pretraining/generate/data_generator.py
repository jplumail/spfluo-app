import os
import csv
import torch
import pickle
import numpy as np
import numpy.random as R
from tqdm import tqdm
from typing import Tuple, List
from skimage.util import random_noise
from scipy.stats import truncnorm
from scipy.ndimage.filters import gaussian_filter
import tifffile
from . import functional as F
from pyfigtree import figtree
from .config import DataGenerationConfig
from .config import Outliers as OutliersConfig

import scipy.ndimage as ndii

from time import time


class DataGenerator:

    # +---------------------------------------------------------------------------------------+ #
    # |                                        INIT                                           | #
    # +---------------------------------------------------------------------------------------+ #

    def __init__(self, config: DataGenerationConfig) -> None:
        self.config = config
        self.device = self.__set_device()
        self.dtype = self.config.dtype
        self.template_pointcloud = self.__load_pointcloud_tensor()
        x_min, x_max, y_min, y_max, z_min, z_max = F.get_FOV(self.template_pointcloud)
        max_delta = max(x_max - x_min, y_max - y_min, z_max - z_min)
        self.step = max_delta / self.config.voxelisation.max_particle_dim
        self.output_image = np.zeros(self.config.target_shape)

        # Particles FOV
        image_shape = self.config.voxelisation.image_shape
        eps = 1e-12 / 2
        self.common_fov = (-(self.step*image_shape-eps)/2, (self.step*image_shape-eps)/2) * 3

    def __load_pointcloud_tensor(self) -> torch.Tensor:
        template_point_cloud = np.loadtxt(self.config.io.point_cloud_path, delimiter=',')
        pointcloud = torch.as_tensor(template_point_cloud, dtype=self.dtype).to(self.device)
        return pointcloud - pointcloud.mean(dim=0, keepdim=True)

    def __set_device(self):
        if self.config.disable_cuda:
            return torch.device('cpu')
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # +---------------------------------------------------------------------------------------+ #
    # |                                  GENERATE CENTERS                                     | #
    # +---------------------------------------------------------------------------------------+ #

    def generate_centers_one_cluster(self, remaining_particles: int, deterministic: bool):
        shape = self.config.target_shape
        radius_xy_min, radius_xy_max = self.config.voxelisation.cluster_range_xy
        radius_z_min,  radius_z_max  = self.config.voxelisation.cluster_range_z
        radius_xy = R.randint(radius_xy_min, high=radius_xy_max, size=2)
        radius_z  = R.randint(radius_z_min,  high=radius_z_max)
        margin    = radius_xy + self.config.voxelisation.max_particle_dim
        center_xy = R.randint(margin, high=np.array(shape[1:]) - margin)
        margin    = radius_z + self.config.voxelisation.max_particle_dim
        center_z  = R.randint(margin,  high=np.array(shape[0]) - margin)
        center    = (center_z, *center_xy)
        radius    = (radius_z, *radius_xy)
        if deterministic:
            nb_particles = remaining_particles
        else:
            nb_min, nb_max = self.config.voxelisation.nb_particles_per_cluster_range
            nb_max = min(nb_max, remaining_particles)
            nb_particles = R.randint(nb_min, high=nb_max + 1)
        margin = np.array(3 * (self.config.voxelisation.max_particle_dim, ))
        clip_min, clip_max = (margin - center) / radius, (shape - margin - center) / radius
        truncnorm_kwargs = {'loc': center, 'scale': radius, 'size': (nb_particles, 3)}
        centers = truncnorm.rvs(clip_min, clip_max, **truncnorm_kwargs)
        return centers.astype(np.uint16)

    """
    def generate_centers(self):
        total_nb_to_generate = self.config.voxelisation.num_particles
        remaining_number_to_generate = total_nb_to_generate
        min_nb_particles_per_cluster = self.config.voxelisation.nb_particles_per_cluster_range[0]
        all_centers = np.ndarray((0, 3), dtype=np.uint16)
        while remaining_number_to_generate > 0:
            deterministic = (remaining_number_to_generate < min_nb_particles_per_cluster)
            args = [remaining_number_to_generate, deterministic]
            centers = self.generate_centers_one_cluster(*args)
            all_centers = np.vstack((all_centers, centers))
            remaining_number_to_generate = total_nb_to_generate - all_centers.shape[0]
        return all_centers
    """
    
    def generate_centers(self) -> np.ndarray:
        shape = self.config.target_shape    
        margin = np.array(3 * (self.config.voxelisation.max_particle_dim, ))
        num_to_generate = self.config.voxelisation.num_particles
        centers = np.zeros((1, 3), dtype=np.uint16)
        while len(centers) < num_to_generate + 1:
            center = np.random.randint(margin, shape - margin, size=(1, 3))
            if np.linalg.norm(centers - center, axis=1).min() >= 1.5*self.config.voxelisation.max_particle_dim:
                centers = np.vstack((centers, center))
        return centers[1:]


    # +---------------------------------------------------------------------------------------+ #
    # |                                TILT VIEWS HANDLERS                                    | #
    # +---------------------------------------------------------------------------------------+ #

    @staticmethod
    def sample_tilt_angles(num_particles: int, tilt_strategy: int, margin: int=10) -> List:
        distributions = {}
        distributions['gaussian_top_side_only'] = [
            {"type": R.normal, "kwargs": {"loc":   0, "scale": 10}},
            {"type": R.normal, "kwargs": {"loc":  90, "scale": 10}},
            {"type": R.normal, "kwargs": {"loc": 180, "scale": 10}},
            {"type": R.normal, "kwargs": {"loc": 270, "scale": 10}},
        ]
        distributions['uniform_top_side_only'] = [
            {"type": R.uniform, "kwargs": {"low":   0 - margin, "high":   0 + margin}},
            {"type": R.uniform, "kwargs": {"low":  90 - margin, "high":  90 + margin}},
            {"type": R.uniform, "kwargs": {"low": 180 - margin, "high": 180 + margin}},
            {"type": R.uniform, "kwargs": {"low": 270 - margin, "high": 270 + margin}},
        ]
        distributions['uniform'] = [
            # top and side
            {"type": R.uniform, "kwargs": {"low":   0 - margin, "high":   0 + margin}},
            {"type": R.uniform, "kwargs": {"low":  90 - margin, "high":  90 + margin}},
            {"type": R.uniform, "kwargs": {"low": 180 - margin, "high": 180 + margin}},
            {"type": R.uniform, "kwargs": {"low": 270 - margin, "high": 270 + margin}},
            # other
            {"type": R.uniform, "kwargs": {"low":   0 + 2 * margin, "high":  90 - 2 * margin}},
            {"type": R.uniform, "kwargs": {"low":  90 + 2 * margin, "high": 180 - 2 * margin}},
            {"type": R.uniform, "kwargs": {"low": 180 + 2 * margin, "high": 270 - 2 * margin}},
        ]
        distributions['gaussian'] = [
            # top and side
            {"type": R.normal, "kwargs": {"loc":   0, "scale": margin // 2}},
            {"type": R.normal, "kwargs": {"loc":  90, "scale": margin // 2}},
            {"type": R.normal, "kwargs": {"loc": 180, "scale": margin // 2}},
            {"type": R.normal, "kwargs": {"loc": 270, "scale": margin // 2}},
            # other
            {"type": R.normal, "kwargs": {"loc":   45, "scale":  margin}},
            {"type": R.normal, "kwargs": {"loc":  135, "scale":  margin}},
            {"type": R.normal, "kwargs": {"loc":  225, "scale":  margin}},
        ]
        distributions = distributions[tilt_strategy]
        num_distributions = len(distributions)
        weights = np.ones((num_distributions))
        weights /= weights.sum()
        data = np.zeros((num_particles, num_distributions))
        for idx, distr in enumerate(distributions):
            data[:, idx] = distr["type"](size=(num_particles, ), **distr["kwargs"])
        idx_kwargs = {'size': (num_particles, ), 'p': weights}
        random_idx = R.choice(np.arange(num_distributions), **idx_kwargs)
        return data[np.arange(num_particles), random_idx]

    @staticmethod
    def get_view_from_angles(angles: Tuple[int], margin: int=10) -> int:
        angle = angles[1]
        around0   = (  0 - margin) <= angle <= (  0 + margin)
        around180 = (180 - margin) <= angle <= (180 + margin)
        around90  = ( 90 - margin) <= angle <= ( 90 + margin)
        around270 = (270 - margin) <= angle <= (270 + margin)
        if around0 or around180:
            return 0   # TOP
        if around90 or around270:
            return 1   # SIDE
        return 2       # OTHER

    def generate_angles(self, tilt_strategy: str, tilt_margin: int) -> np.ndarray:
        num_particles = self.config.voxelisation.num_particles
        tilt_angles  = self.sample_tilt_angles(num_particles, tilt_strategy, margin=tilt_margin) # TO CHANGE
        other_angles = R.uniform(low=0, high=360, size=(num_particles, 2))
        return np.vstack((other_angles[:, 0], tilt_angles, other_angles[:, 1])).T
    
    def generate_translations(self, max_translation_ratio: float) -> np.ndarray:
        num_particles = self.config.voxelisation.num_particles
        return R.uniform(low=-max_translation_ratio, high=max_translation_ratio, size=(num_particles, 3))


    # +---------------------------------------------------------------------------------------+ #
    # |                                 AUGMENT POINTCLOUD                                    | #
    # +---------------------------------------------------------------------------------------+ #

    def augment_pointcloud(self, rotation_angles: Tuple[int]=None) -> np.ndarray:
        cfg = self.config.augmentation
        # 1. Shrink
        a, b   = cfg.shrink_range
        factor = (b - a) * R.random_sample() + a
        pointcloud = F.shrink(self.template_pointcloud, factor)
        # 2. Rotate
        pointcloud = F.rotate(pointcloud, rotation_angles, cfg.rotation_proba, self.device)
        # 3. Translate
        pointcloud = F.translate(pointcloud, translation, self.device)
        # 4. Make holes
        a, b = cfg.intensity_std_ratio_range
        std_ratio = (b - a) * R.random_sample() + a
        #pointcloud = F.non_uniform_density(
        #    pointcloud,
        #    cfg.intensity_nb_holes_min,
        #    cfg.intensity_nb_holes_max,
        #    cfg.intensity_mean_ratio,
        #    std_ratio,
        #    self.device
        #)
        return pointcloud.cpu().numpy()

    # +---------------------------------------------------------------------------------------+ #
    # |                                    VOXELISATION                                       | #
    # +---------------------------------------------------------------------------------------+ #

    def image_from_pointcloud(self, pointcloud: np.ndarray, fov: Optional[Tuple[float]]=None) -> np.ndarray:
        dtype = pointcloud.dtype
        cfg = self.config.voxelisation
        particle_fov = F.get_FOV(pointcloud)
        if fov is None:
            fov = particle_fov
        else:
            # check if particle_fov inside fov
            x_min, x_max, y_min, y_max, z_min, z_max = fov
            x_min_p, x_max_p, y_min_p, y_max_p, z_min_p, z_max_p = particle_fov
            inside = (x_min < x_min_p) and (y_min < y_min_p) and (z_min < z_min_p) and (x_max > x_max_p) and (y_max > y_max_p) and (z_max > z_max_p)
            if not inside:
                raise ValueError("Particle not inside fov", fov, particle_fov)
        x_min, x_max, y_min, y_max, z_min, z_max = fov
        x_min, x_max, y_min, y_max, z_min, z_max = map(float, (x_min, x_max, y_min, y_max, z_min, z_max))
        step = float(self.step)
        x_range = int(np.ceil((x_max - x_min) / step))
        y_range = int(np.ceil((y_max - y_min) / step))
        z_range = int(np.ceil((z_max - z_min) / step))
        target_points = np.mgrid[x_min:x_max:step, y_min:y_max:step, z_min:z_max:step]
        target_points = target_points.reshape(3, -1).T
        weights = np.ones(len(pointcloud))
        figtree_kwargs = {'bandwidth': cfg.bandwidth, 'epsilon': cfg.epsilon}
        densities = figtree(pointcloud, target_points, weights, **figtree_kwargs)
        densities = densities.reshape((x_range, y_range, z_range))
        densities = np.transpose(densities, (2,1,0))
        densities = (densities - densities.min()) / (densities.max() - densities.min())
        
        # point (0,0,0) in world coords is the center (the pointcloud is centered)
        # when converted in pixel coords, we get :
        center = np.array([-z_min/step, -y_min/step, -x_min/step])
        # function x -> (x - xmin) / step converts world coords to pixel coords
        
        densities = densities.astype(dtype)
        return densities, center

    # +---------------------------------------------------------------------------------------+ #
    # |                                   AUGMENT IMAGE                                       | #
    # +---------------------------------------------------------------------------------------+ #

    @staticmethod
    def add_one_outliers_cluster(image: np.ndarray, cfg: OutliersConfig):
        radius_xy_min, radius_xy_max = cfg.radius_range_xy
        radius_z_min,  radius_z_max  = cfg.radius_range_z
        radius_x  = R.randint(radius_xy_min, high=radius_xy_max)
        radius_y  = R.randint(radius_xy_min, high=radius_xy_max)
        radius_z  = R.randint(radius_z_min,  high=radius_z_max)
        center_x  = R.randint(radius_x, high=image.shape[2] - radius_x)
        center_y  = R.randint(radius_y, high=image.shape[1] - radius_y)
        center_z  = R.randint(radius_z, high=image.shape[0] - radius_z)
        center    = (center_z, center_y, center_x)
        radius    = (radius_z, radius_y, radius_x)
        low, high = cfg.nb_points_range
        nb_points = R.randint(low, high=high)
        gaussian_kwargs = {'loc': center, 'scale': np.sqrt(radius), 'size': (nb_points, 3)}
        outliers_cluster = R.normal(**gaussian_kwargs).astype(np.uint16)
        low, high = cfg.intensity_range
        for z, y, x in outliers_cluster:
            image[z, y, x] = (high - low) * R.random_sample() + low

    def add_outliers_clusters(self, image: np.ndarray, cfg: OutliersConfig):
        low, high   = cfg.nb_clusters_range
        nb_clusters = R.randint(low, high)
        for _ in range(nb_clusters):
            try:
                self.add_one_outliers_cluster(image, cfg)
            except IndexError:
                pass

    def add_all_outliers(self):
        for one_pass_config in self.config.outliers:
            self.add_outliers_clusters(self.output_image, one_pass_config)

    def add_anisotropic_blur(self):
        sigma = self.config.sensor.anisotropic_blur_sigma
        mode  = self.config.sensor.anisotropic_blur_border_mode
        self.output_image = gaussian_filter(self.output_image, sigma=sigma, mode=mode)

    def add_gaussian_noise_with_target_snr(self):
        image_average_db = 10 * np.log10(self.output_image.mean())
        noise_average_db = image_average_db - self.config.sensor.gaussian_noise_target_snr_db
        noise_average    = 10 ** (noise_average_db / 10)
        noise = R.normal(self.config.sensor.gaussian_noise_mean,
                         np.sqrt(noise_average),
                         self.config.target_shape)
        self.output_image += noise
        pixel_values_range = (self.output_image.max() - self.output_image.min())
        self.output_image  = (self.output_image - self.output_image.min()) / pixel_values_range

    def add_poisson_noise(self):
        self.output_image = random_noise(self.output_image, "poisson")

    def add_sensor_faults(self):
        if self.config.sensor.anisotropic_blur:
            self.add_anisotropic_blur()
        if self.config.sensor.gaussian_noise:
            self.add_gaussian_noise_with_target_snr()
        if self.config.sensor.poisson_noise:
            self.add_poisson_noise()
    
    def create_psf(self):
        step = float(self.step)
        B = self.config.voxelisation.bandwidth # bandwidth in the pointcloud coordinate system
        B = int(np.ceil(B / step)) # bandwidth in the image space
        k = 10
        shape = [2*k*B+1] * 3
        psf = np.zeros(shape, dtype=float)
        psf[k*B,k*B,k*B] = 1 # dirac
        #psf = gaussian_filter(psf, sigma=B, mode='constant') # gaussian of std B

        # Anisotropic blur
        if self.config.sensor.anisotropic_blur:
            sigma = self.config.sensor.anisotropic_blur_sigma
            mode  = self.config.sensor.anisotropic_blur_border_mode
            psf = gaussian_filter(psf, sigma=sigma, mode=mode)

        # Save
        output_path = os.path.join(self.config.io.output_dir, 'psf.tif')
        psf = (psf - psf.min()) / (psf.max() - psf.min())
        psf = psf / psf.sum()
        tifffile.imwrite(output_path, (psf*255).astype(np.uint8))




    # +---------------------------------------------------------------------------------------+ #
    # |                                      WRAPPERS                                         | #
    # +---------------------------------------------------------------------------------------+ #

    
    def create_image(self, output: str=None, output_extension: str='npz') -> np.ndarray:
        # 1. Generate centers and rotation angles for each particle, and get views from angles.
        print("Generating centers...")
        centers = self.generate_centers()
        real_centers = []
        angles  = self.generate_angles(self.config.voxelisation.tilt_strategy, self.config.voxelisation.tilt_margin)
        translations = self.generate_translations(self.config.augmentation.max_translation)
        views = np.array(list(map(self.get_view_from_angles, angles)))
        # 2. Augment, voxelize and put images inside one big image
        for center, rotation_angles, translation in tqdm(zip(centers, angles, translations), total=len(centers), desc="Creating particles..."):
            particle, crop_center = self.image_from_pointcloud(self.augment_pointcloud(rotation_angles=rotation_angles, translation=translation))
            depth, height, width = particle.shape
            z, y, x = center
            z_min, y_min, x_min = z - depth // 2, y - height // 2, x - width // 2
            z_max, y_max, x_max = z_min + depth, y_min + height, x_min + width
            real_center = np.array([z_min, y_min, x_min], dtype=float) + crop_center
            real_centers.append(real_center)
            self.output_image[z_min:z_max, y_min:y_max, x_min:x_max] += particle
        real_centers = np.stack(real_centers, axis=0)
        # 3. Add outliers and sensor faults (blur + noise(s))
        self.add_all_outliers()
        self.add_sensor_faults()
        # 4. Save
        if output is not None:
            output_path = os.path.join(self.config.io.output_dir, output) + '.' + output_extension
            if output_extension == 'npz':
                np.savez(output_path, image=self.output_image)
            elif output_extension == 'tiff':
                self.output_image = self.output_image.astype(float)
                self.output_image = (self.output_image - self.output_image.min()) / (self.output_image.max() - self.output_image.min())
                tifffile.imwrite(output_path, (255*self.output_image).astype(np.uint8))
        return real_centers, views, angles, translations

    def generate_dataset(self):
        os.makedirs(self.config.io.output_dir, exist_ok=True)
        self.create_psf()
        all_centers, all_views, all_angles = [], [], []
        for i in tqdm(range(self.config.io.generated_dataset_size)):
            centers, views, angles, _ = self.create_image(output=str(i), output_extension=self.config.io.extension)
            width = int(np.ceil(np.log10(len(centers))))
            for j, (center, view, angles_) in enumerate(zip(centers, views, angles)):
                all_centers.append((f'{i}.{self.config.io.extension}', str(j).zfill(width), *center))
                all_views.append((f'{i}.{self.config.io.extension}', str(j).zfill(width), view))
                all_angles.append((f'{i}.{self.config.io.extension}', str(j).zfill(width), *angles_))
            self.output_image = np.zeros(self.config.target_shape)  # reset output image
        with open(f"{self.config.io.output_dir}/coordinates.csv", "w", newline="") as f:
            csv.writer(f).writerows(all_centers)
        with open(f"{self.config.io.output_dir}/views.csv", "w", newline="") as f:
            csv.writer(f).writerows(all_views)
        with open(f"{self.config.io.output_dir}/angles.csv", "w", newline="") as f:
            csv.writer(f).writerows(all_angles)
        with open(f"{self.config.io.output_dir}/config.pickle", "wb") as f:
            pickle.dump(self.config, f)
