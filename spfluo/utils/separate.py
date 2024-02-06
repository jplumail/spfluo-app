from sklearn.cluster import KMeans
from sklearn.svm import SVC

from spfluo.utils.array import Array, array_namespace
from spfluo.utils.rotate_symmetry_axis import find_rotation_between_two_vectors
from spfluo.utils.volume import affine_transform, tukey


def extract_particle(
    image_data: Array,
    pos: tuple[float, float, float],
    dim: tuple[float, float, float],
    scale=tuple[float, float, float],
    subpixel: bool = False,
):
    xp = array_namespace(image_data)

    def world_to_data_coord(pos):
        return pos / xp.asarray(scale)

    pos = world_to_data_coord(xp.asarray(pos))  # World coordinates to data coords
    box_size_world = xp.asarray(dim, dtype=float)
    box_size_data = xp.astype(xp.round(world_to_data_coord(box_size_world)), xp.int64)
    mat = xp.eye(4)
    mat[:3, 3] = pos
    mat[:3, 3] -= box_size_data / 2
    C = image_data.shape[0]
    particle_data = xp.empty((C,) + tuple(box_size_data), dtype=image_data.dtype)
    if not subpixel:
        top_left_corner = xp.round(mat[:3, 3]).astype(int)
        bottom_right_corner = top_left_corner + box_size_data
        xmin, ymin, zmin = top_left_corner
        xmax, ymax, zmax = bottom_right_corner
        original_shape = image_data.shape[1:]
        x_slice = slice(max(xmin, 0), min(xmax, original_shape[0]))
        y_slice = slice(max(ymin, 0), min(ymax, original_shape[1]))
        z_slice = slice(max(zmin, 0), min(zmax, original_shape[2]))
        x_overlap = slice(
            max(0, -xmin), min(box_size_data[0], original_shape[0] - xmin)
        )
        y_overlap = slice(
            max(0, -ymin), min(box_size_data[1], original_shape[1] - ymin)
        )
        z_overlap = slice(
            max(0, -zmin), min(box_size_data[2], original_shape[2] - zmin)
        )

    for c in range(C):
        if subpixel:
            particle_data[c] = affine_transform(
                image_data[c], mat, output_shape=tuple(box_size_data)
            )
        else:
            padded_array = xp.zeros(tuple(box_size_data), dtype=image_data.dtype)
            padded_array[x_overlap, y_overlap, z_overlap] = image_data[
                c, x_slice, y_slice, z_slice
            ]
            particle_data[c] = xp.asarray(padded_array, copy=True)

    return particle_data


def separate_centrioles(
    im: Array,
    output_size: tuple[int, int, int],
    threshold_percentage: float = 0.5,
    channel: int = 0,
    tukey_alpha: float = 0.1,
):
    xp = array_namespace(im)
    return separate_centrioles_coords(
        im,
        (xp.asarray(im.shape[-3:]) - 1) / 2,
        im.shape[-3:],
        output_size,
        threshold_percentage=threshold_percentage,
        channel=channel,
        tukey_alpha=tukey_alpha,
    )


def separate_centrioles_coords(
    image: Array,
    pos: tuple[float, float, float],
    dim: tuple[float, float, float],
    output_size: tuple[float, float, float],
    *,
    scale: tuple[float, float, float] = (1, 1, 1),
    threshold_percentage: float = 0.5,
    channel: int = 0,
    tukey_alpha: float = 0.1,
):
    xp = array_namespace(image)

    if image.ndim > 3:
        multichannel = True
    else:
        multichannel = False
        image = image[None]

    # extract patch from image
    patch = extract_particle(image, pos, dim, scale, subpixel=False)[channel]

    # Thresholding
    points = xp.stack(xp.nonzero(patch > xp.max(patch) * threshold_percentage), axis=-1)
    points = xp.to_device(points, "cpu")

    patch_top_left_corner = xp.asarray(pos) - xp.asarray(dim) / 2
    points = points * scale + patch_top_left_corner  # points in the image space
    # Clustering
    kmeans = KMeans(n_clusters=2, n_init="auto")
    kmeans.fit(points)

    # extract 2 patches around the centroids
    patch1 = extract_particle(
        image, kmeans.cluster_centers_[0], output_size, scale, subpixel=False
    )
    patch2 = extract_particle(
        image, kmeans.cluster_centers_[1], output_size, scale, subpixel=False
    )

    # Separate clusters
    svc = SVC(kernel="linear")
    svc.fit(points, kmeans.labels_)

    def proj(x):
        return xp.vecdot(x, xp.asarray(svc.coef_)) + xp.asarray(svc.intercept_)

    # Compute hyperplane for each patch
    s = xp.max(patch1.shape[-3:])
    size_tukey = xp.asarray([s, s, s]) * 3**0.5
    t = tukey(xp, xp.asarray(xp.round(size_tukey), dtype=xp.int64), alpha=tukey_alpha)

    # Compute hyperplane for each patch and apply tukey window
    for patch, pos_patch in zip([patch1, patch2], kmeans.cluster_centers_):
        image_coords = xp.stack(
            xp.meshgrid(*[xp.arange(s) for s in patch.shape[1:]], indexing="ij"),
            axis=-1,
        )
        patch_top_left_corner = xp.asarray(pos_patch) - xp.asarray(output_size) / 2
        image_coords = image_coords * scale + patch_top_left_corner  # to image space
        pos_patch_proj = proj(pos_patch)

        # Compute the tukey window
        v = xp.asarray(svc.coef_[0])
        d = xp.asarray(svc.intercept_) / xp.linalg.norm(v)
        v = v / xp.linalg.norm(v)
        unit = xp.asarray([0, 1, 0])
        R = find_rotation_between_two_vectors(unit, v)

        H1 = xp.eye(4)
        H1[:3, 3] = -xp.asarray(scale) * xp.round(size_tukey) / 2
        H1[[0, 1, 2], [0, 1, 2]] = xp.asarray(scale)  # pixel to world space
        H2 = xp.eye(4)  # world space
        H2[:3, :3] = R  # world space
        pos_tukey_plane = (
            R
            @ (
                -xp.sign(pos_patch_proj)
                * xp.asarray(scale)
                * unit
                * xp.round(size_tukey)
                / 2
            )
            / xp.asarray(scale)
        )  # pixel space

        center_patch = xp.asarray(pos_patch)  # world space
        patch_top_left_corner = (
            xp.asarray(center_patch) - xp.asarray(output_size) / 2
        )  # world space
        center_patch_proj = center_patch - v * (
            xp.vecdot(center_patch, v) + d
        )  # world space
        center_patch_proj_patch_space = (
            center_patch_proj - patch_top_left_corner
        ) / xp.asarray(
            scale
        )  # pixel space
        H3 = xp.eye(4)
        H3[[0, 1, 2], [0, 1, 2]] = 1 / xp.asarray(scale)
        H3[:3, 3] = center_patch_proj_patch_space - pos_tukey_plane
        H = H3 @ H2 @ H1
        t_rotated = affine_transform(t, xp.linalg.inv(H), output_shape=patch.shape[-3:])

        patch *= t_rotated

    if not multichannel:
        patch1, patch2 = patch1[0], patch2[0]

    return patch1, patch2
