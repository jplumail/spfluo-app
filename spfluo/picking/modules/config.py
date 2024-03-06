from dataclasses import dataclass, field
from typing import Tuple

# +------------------------------------------------------------------------------------------+ #
# |                                           I/O                                            | #
# +------------------------------------------------------------------------------------------+ #


@dataclass
class IO:
    """Input/Output params.

    Args:

        point_cloud_path (str): Path to the point cloud csv. This point cloud will act as a
                                template to generate augmented particles.

        output_dir (str): Generated images and masks will be save in output_dir. One npz file per
                          pair will be created with two keys, 'image, and 'mask'.

        generated_dataset_size (int): How many pairs (image, mask) to generate.
    """

    point_cloud_path: str = "/data1/stage_Luc/inputs/sample_centriole_point_cloud.csv"
    output_dir: str = "/data1/stage_Luc/tilt/"
    generated_dataset_size: int = 25


# +------------------------------------------------------------------------------------------+ #
# |                                      Voxelisation                                        | #
# +------------------------------------------------------------------------------------------+ #


@dataclass
class Voxelisation:
    """Params controling the transformation from pointcloud to voxel grid.

    Args:

        num_particles (int): How many augmented copies to generate into each pair (image, mask).

        max_particle_dim (int): After voxelisation, each generated particle will be within an
                                irregular cube for which the side is at most max_particle_dim.

        bandwidth (int): The augmented point cloud will be converted into a voxel grid (ie a 3d
                         image) by a fast gaussian transorm algorithm. This parameters controls
                         the gaussian kernel size.

        epsilon (float): The desired error of the fast gaussian transform based voxelisation.

        cluster_range_xy (Tuple[int]): After voxelisation, particles will be placed onto the big
                                       3d image in a random center. This center will be sampled
                                       from a mixture of gaussian. Each gaussian standard
                                       deviation in x and y direction will be within
                                       radius_range_xy.

        cluster_range_z (Tuple[int]): After voxelisation, particles will be placed onto the big
                                      3d image in a random center. This center will be sampled
                                      from a mixture of gaussian. Each gaussian standard
                                      deviation in z direction will be within radius_range_z.

        nb_particles_per_cluster_range (Tuple[int]): Each particles cluster (ie each gaussian
                                                     used to sample center coordinates) will have
                                                     a number of particles inside within
                                                     nb_particles_per_cluster_range.
        tilt_strategy (str): Defines the distributions mixture used to sample tilt angles. Alike
                             {law}_{special_case} where law can be one of 'gaussian' or 'uniform'
                             and special_case can be top_side_only.
    """

    num_particles: int = 150
    max_particle_dim: int = 25
    bandwidth: int = 10
    epsilon: int = 1e-3
    cluster_range_xy: Tuple[int] = (100, 480)
    cluster_range_z: Tuple[int] = (5, 35)
    nb_particles_per_cluster_range: Tuple[int] = (45, 55)
    tilt_strategy: str = "uniform"


# +------------------------------------------------------------------------------------------+ #
# |                                      Augmentation                                        | #
# +------------------------------------------------------------------------------------------+ #


@dataclass
class Augmentation:
    """Params to control each generated particle augmentation.

    Args:

        shrink_range (Tuple[int]): The template point cloud will be shrinked by a random factor
                                       between shrink range.

        intensity_nb_holes_min (int): Non uniform density will be simulated by randomly deleting
                                      some points within spheres. This parameters controls the
                                      minimum amount of such sphere for each generated particle.

        intensity_nb_holes_max (int): Non uniform density will be simulated by randomly deleting
                                      some points within spheres. This parameters controls the
                                      maximum amount of such sphere for each generated particle.

        intensity_mean_ratio (int): Non uniform density will be simulated by randomly deleting
                                    some points within spheres of radius R. The random deletion
                                    follows a gaussian law: max proba in the center of the sphere,
                                    min proba in the edge. The mean of this gaussian law is
                                    R * mean_ratio.

        intensity_std_ratio_range (int): Non uniform density will be simulated by randomly
                                         deleting some points within spheres of radius R.
                                         The random deletion follows a gaussian law: max proba in
                                         the center of the sphere, min proba in the edge.
                                         The standard deviation of this gaussian law is:
                                         R * std_ratio.
                                         The std_ratio will be sampled uniformely within the
                                         specified range.

        rotation_proba (int): Each generated particle will be randomly rotated in 3d with a
                              probability rotation_proba. If the rotation occurs, the angles in
                              the 3 directions follow a uniform distribution from -180 to 180.
    """

    shrink_range: Tuple[int] = (0.8, 1.2)
    intensity_nb_holes_min: int = 1
    intensity_nb_holes_max: int = 8
    intensity_mean_ratio: float = 0.3
    intensity_std_ratio_range: float = (0.15, 0.25)
    rotation_proba: float = 1.0


# +------------------------------------------------------------------------------------------+ #
# |                                        Outliers                                          | #
# +------------------------------------------------------------------------------------------+ #


@dataclass
class Outliers:
    """Params controling outliers generation.

    Args:

        radius_range (Tuple[int]): Each outliers cluster will have a random radius within
                                   radius_range.
                                   Two radius are defined, one for the z-axis and one for the x
                                   and y axis because of the strong anistropy of resolution.

        nb_points_range (Tuple[int]): Each cluster will contain a number of points within
                                      nb_points_range.

        nb_clusters_range (Tuple[int]): The entire image will have a number of outliers clusters
                                        within nb_clusters_range.

        intensity_range (Tuple[int]): Each outlier point inside a cluster will have an intensity
                                      in intensity_range.
    """

    radius_range_xy: Tuple[int] = (25, 200)
    radius_range_z: Tuple[int] = (20, 40)
    nb_points_range: Tuple[int] = (200, 800)
    nb_clusters_range: Tuple[int] = (100, 200)
    intensity_range: Tuple[int] = (0.5, 1)


# +------------------------------------------------------------------------------------------+ #
# |                                         Sensor                                           | #
# +------------------------------------------------------------------------------------------+ #


@dataclass
class Sensor:
    """Params to control sensor faults: blur and noises.

    Args:

        anisotropic_blur (bool): Whether or not to add anisotropic blur to image and mask. If
                                 True, at the end of the generation process, the whole image and
                                 the whole mask will be convoluted with an anisotropic gaussian
                                 kernel.

        anisotropic_blur_sigma (Tuple[int]): This parameter controls the standard deviation of
                                             the gaussian blur for each axis, ordered as
                                             (z, y, x). E.g: (10*n, n, n) will lead to a strong
                                             z-axis resolution anisotropy.

        anisotropic_blur_border_mode (str): Because the anistropy of resolution is simulated by
                                            a kernel based convolution, the resulting image may
                                            be smaller than the original one. Thus the input
                                            image will be padded to prevent that. This parameter
                                            controls the padding strategy.
                                            Can be one of 'reflect', 'constant', 'nearest',
                                            'mirror', 'wrap'.

        gaussian_noise (bool): Whether to add gaussian noise to the final image or not.

        gaussian_noise_mean (float): If gaussian_noise is True, what will be the gaussian
                                     distribution's mean. As the generated image are float64 in
                                     range [0, 1], it is 0.5 by default.

        gaussian_noise_target_snr_db (int): What will be the Signal to Noise Ratio after the
                                            gaussian noise addition.

        poisson_noise (bool): Whether or not to add Poisson noise to the image. If both gaussian
                              and poisson are True, Poisson noise will be added after the
                              gaussian noise.
    """

    anisotropic_blur: bool = True
    anisotropic_blur_sigma: Tuple[int] = (5, 1, 1)
    anisotropic_blur_border_mode: str = "constant"
    gaussian_noise: bool = True
    gaussian_noise_mean: float = 0.5
    gaussian_noise_target_snr_db: int = 15
    poisson_noise: bool = True


# +------------------------------------------------------------------------------------------+ #
# |                                         Wrapper                                          | #
# +------------------------------------------------------------------------------------------+ #


@dataclass
class DataGenerationConfig:
    """This wraps all the parameters to generate pseudo random data defined abose.

    This config class will be used to instanciate a DataGenerator class which will use
    functional.py to perform its generation task.
    The basic idea is to generate many randomly augmented point cloud copies of a given particle
    point cloud, then convert them to images, then place them randomly into the output 3D image.

    Args:

        disable_cuda (bool): Usually, the device (cpu or gpu) will be automatically detected.
                            However it is possible to force cpu usage by setting this param to
                            True (for instance for debug purpose or for old GPU for which PyTorch
                            might not work as intended).

        target_shape (Tuple[int]): The shape of the final pair (image, mask) containing the
                                   generated copies of particles.

    Note that outliers is a Tuple of instance of the Outliers config class. This allows one to
    define several instances of Outliers config class with varying parameters. All those Outliers
    config will be used one after one in the DataGenerator class, so that several types of
    outliers can be generated.
    For instance, one may want to perform a first pass to add dense and small outliers clusters
    to  mimic artifacts, and a second pass with less dense but bigger outliers clusters to mimic
    light halo.
    """

    disable_cuda: bool = True
    target_shape: Tuple[int] = (128, 1024, 1024)
    io: IO = field(default_factory=IO)
    voxelisation: Voxelisation = field(default_factory=Voxelisation)
    augmentation: Augmentation = field(default_factory=Augmentation)
    outliers: Tuple[Outliers] = field(default_factory=tuple)
    sensor: Sensor = field(default_factory=Sensor)
